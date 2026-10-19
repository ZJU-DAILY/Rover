#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import uuid
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required: python3 -m pip install numpy") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pure_pq_layout import VectorLayout


HEADER_BYTES = 256
MAGIC = b"PUREPQ01"
VERSION_MAJOR = 1
VERSION_MINOR = 1
ENDIAN_MARKER = 0x01020304
ARTIFACT_CODEBOOK = 1
ARTIFACT_CODES = 2
METRIC_SQUARED_L2 = 1
CODE_DTYPE_UINT8 = 1
CODEBOOK_DTYPE_FLOAT32 = 1
TRAINER_SERVER_NUMPY = 2
TRAINER_SERVER_FAISS = 3
SAMPLE_UNIFORM_STRIDE = 1


def code_bits_for_ks(ks: int) -> int:
    if ks == 16:
        return 4
    if ks == 256:
        return 8
    raise ValueError("Pure-PQ requires Ks=16 (4-bit) or Ks=256 (8-bit)")


def code_bytes_for(m: int, ks: int) -> int:
    if m <= 0:
        raise ValueError("M must be positive")
    return (m + 1) // 2 if code_bits_for_ks(ks) == 4 else m


def _put_u16(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", buffer, offset, value)


def _put_u32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, value)


def _put_u64(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", buffer, offset, value)


def encode_header(kind: int, metadata: dict, payload_bytes: int,
                  payload_hash: bytes) -> bytes:
    if len(payload_hash) != 32:
        raise ValueError("payload SHA-256 must be 32 bytes")
    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    _put_u16(header, 8, VERSION_MAJOR)
    _put_u16(header, 10, VERSION_MINOR)
    _put_u32(header, 12, HEADER_BYTES)
    _put_u32(header, 16, kind)
    _put_u32(header, 20, ENDIAN_MARKER)
    header[32:48] = metadata["index_uuid"]
    _put_u64(header, 48, metadata["num_vectors"])
    _put_u32(header, 56, metadata["dim"])
    _put_u32(header, 60, metadata["m"])
    _put_u32(header, 64, metadata["ks"])
    _put_u32(header, 68, metadata["subdim"])
    _put_u32(header, 72, metadata["code_bytes"])
    _put_u32(header, 76, METRIC_SQUARED_L2)
    _put_u32(header, 80, CODE_DTYPE_UINT8)
    _put_u32(header, 84, CODEBOOK_DTYPE_FLOAT32)
    _put_u32(header, 88, metadata["trainer_kind"])
    _put_u32(header, 92, metadata["sample_strategy"])
    _put_u64(header, 96, metadata["train_count"])
    _put_u32(header, 104, metadata["kmeans_iters"])
    _put_u32(header, 108, metadata["seed"])
    _put_u64(header, 112, payload_bytes)
    header[120:152] = payload_hash
    header[152:184] = metadata["dataset_fingerprint"]
    header[184:216] = metadata["layout_fingerprint"]
    return bytes(header)


def decode_header(header: bytes, expected_kind: int | None = None) -> dict:
    if len(header) != HEADER_BYTES or header[0:8] != MAGIC:
        raise ValueError("invalid Pure-PQ magic/header length")
    major, minor = struct.unpack_from("<HH", header, 8)
    header_bytes, kind, endian = struct.unpack_from("<III", header, 12)
    metric, code_dtype, codebook_dtype = struct.unpack_from("<III", header, 76)
    if major != VERSION_MAJOR or minor > VERSION_MINOR:
        raise ValueError(f"unsupported Pure-PQ version {major}.{minor}")
    if header_bytes != HEADER_BYTES or endian != ENDIAN_MARKER:
        raise ValueError("invalid Pure-PQ header/endian marker")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"artifact kind {kind}, expected {expected_kind}")
    if (metric, code_dtype, codebook_dtype) != (
            METRIC_SQUARED_L2, CODE_DTYPE_UINT8, CODEBOOK_DTYPE_FLOAT32):
        raise ValueError("unsupported metric or payload dtype")
    metadata = {
        "version_major": major,
        "version_minor": minor,
        "kind": kind,
        "index_uuid": header[32:48],
        "num_vectors": struct.unpack_from("<Q", header, 48)[0],
        "dim": struct.unpack_from("<I", header, 56)[0],
        "m": struct.unpack_from("<I", header, 60)[0],
        "ks": struct.unpack_from("<I", header, 64)[0],
        "subdim": struct.unpack_from("<I", header, 68)[0],
        "code_bytes": struct.unpack_from("<I", header, 72)[0],
        "trainer_kind": struct.unpack_from("<I", header, 88)[0],
        "sample_strategy": struct.unpack_from("<I", header, 92)[0],
        "train_count": struct.unpack_from("<Q", header, 96)[0],
        "kmeans_iters": struct.unpack_from("<I", header, 104)[0],
        "seed": struct.unpack_from("<I", header, 108)[0],
        "payload_bytes": struct.unpack_from("<Q", header, 112)[0],
        "payload_checksum": header[120:152],
        "dataset_fingerprint": header[152:184],
        "layout_fingerprint": header[184:216],
    }
    validate_metadata(metadata)
    return metadata


def validate_metadata(metadata: dict) -> None:
    if metadata["index_uuid"] == bytes(16):
        raise ValueError("zero index UUID")
    n = metadata["num_vectors"]
    dim = metadata["dim"]
    m = metadata["m"]
    ks = metadata["ks"]
    subdim = metadata["subdim"]
    if n <= 0 or dim <= 0 or m <= 0 or subdim <= 0:
        raise ValueError("invalid N/dim/M/Ks/subdim")
    expected_code_bytes = code_bytes_for(m, ks)
    if (m * subdim != dim or
            metadata["code_bytes"] != expected_code_bytes):
        raise ValueError("inconsistent Pure-PQ dimensions")
    if not 0 < metadata["train_count"] <= n:
        raise ValueError("invalid training count")
    if metadata["kmeans_iters"] <= 0:
        raise ValueError("invalid k-means iteration count")


def inspect_fvecs(path: Path) -> tuple[int, int, int]:
    file_bytes = path.stat().st_size
    if file_bytes < 4:
        raise ValueError(f"empty/truncated fvecs: {path}")
    with path.open("rb") as file:
        raw = file.read(4)
    dim = struct.unpack("<i", raw)[0]
    if dim <= 0:
        raise ValueError(f"invalid fvecs dimension {dim}")
    record_bytes = 4 + dim * 4
    if file_bytes % record_bytes != 0:
        raise ValueError("fvecs size is not a whole number of records")
    return file_bytes // record_bytes, dim, record_bytes


def open_fvecs(path: Path, num_vectors: int | None,
               expected_dim: int | None) -> tuple[np.ndarray, int, int, int]:
    available, dim, record_bytes = inspect_fvecs(path)
    if expected_dim is not None and expected_dim != dim:
        raise ValueError(f"fvecs dim {dim}, expected {expected_dim}")
    count = available if num_vectors is None else num_vectors
    if count <= 0 or count > available:
        raise ValueError(f"requested N={count}, available N={available}")
    records = np.memmap(path, dtype="<i4", mode="r", shape=(available, dim + 1))
    header_chunk = 1 << 18
    for start in range(0, count, header_chunk):
        if np.any(records[start:min(count, start + header_chunk), 0] != dim):
            raise ValueError(f"inconsistent fvecs row dimension near row {start}")
    vectors = records[:count, 1:].view("<f4")
    return vectors, count, dim, record_bytes


def hash_file_prefix(path: Path, bytes_to_hash: int) -> bytes:
    digest = hashlib.sha256()
    remaining = bytes_to_hash
    with path.open("rb") as file:
        while remaining:
            data = file.read(min(8 << 20, remaining))
            if not data:
                raise ValueError("unexpected EOF while hashing dataset")
            digest.update(data)
            remaining -= len(data)
    return digest.digest()


def uniform_positions(population: int, count: int) -> np.ndarray:
    if population <= 0 or count <= 0:
        raise ValueError("invalid uniform position count")
    if count == 1 or population == 1:
        return np.zeros(count, dtype=np.int64)
    numerator = np.arange(count, dtype=np.int64) * (population - 1)
    return numerator // (count - 1)


def uniform_sample_ids(num_vectors: int, train_count: int) -> np.ndarray:
    if train_count > num_vectors:
        raise ValueError("invalid uniform sample count")
    return uniform_positions(num_vectors, train_count)


def train_subspace(samples: np.ndarray, ks: int, iterations: int,
                   work_batch: int) -> np.ndarray:
    count, subdim = samples.shape
    init_ids = uniform_positions(count, ks)
    centroids = np.array(samples[init_ids], dtype=np.float32, copy=True)
    sums = np.empty((ks, subdim), dtype=np.float32)
    counts = np.empty(ks, dtype=np.int64)
    for _ in range(iterations):
        sums.fill(0.0)
        counts.fill(0)
        for start in range(0, count, work_batch):
            batch = np.asarray(samples[start:min(count, start + work_batch)],
                               dtype=np.float32)
            best_distance = np.full(batch.shape[0], np.inf, dtype=np.float32)
            assignments = np.zeros(batch.shape[0], dtype=np.int32)
            for centroid_id in range(ks):
                diff = batch - centroids[centroid_id]
                distance = np.sum(diff * diff, axis=1, dtype=np.float32)
                better = distance < best_distance
                best_distance[better] = distance[better]
                assignments[better] = centroid_id
            if not np.all(np.isfinite(best_distance)):
                raise ValueError("NaN/Inf encountered during PQ training")
            np.add.at(counts, assignments, 1)
            np.add.at(sums, assignments, batch)
        populated = counts > 0
        centroids[populated] = sums[populated] / counts[populated, None]
    return centroids


def train_codebook(vectors: np.ndarray, sample_ids: np.ndarray, m: int,
                   ks: int, iterations: int, memory_budget_bytes: int,
                   chunk_vectors: int) -> tuple[np.ndarray, str, int]:
    dim = vectors.shape[1]
    subdim = dim // m
    train_count = len(sample_ids)
    full_sample_bytes = train_count * dim * 4
    codebook_bytes = m * ks * subdim * 4
    scratch_floor = ks * subdim * 8 + train_count * 8
    fast_required = full_sample_bytes + codebook_bytes + scratch_floor
    mode = "fast" if fast_required <= memory_budget_bytes else "bounded"
    codebook = np.empty((m, ks, subdim), dtype=np.float32)
    per_row_work = max(32, subdim * 12)
    work_batch = max(1, min(chunk_vectors,
                            memory_budget_bytes // max(per_row_work, 1)))
    minimum = codebook_bytes + train_count * subdim * 4 + scratch_floor
    if mode == "fast":
        samples = np.array(vectors[sample_ids], dtype=np.float32, copy=True)
        for subspace in range(m):
            start = subspace * subdim
            codebook[subspace] = train_subspace(
                samples[:, start:start + subdim], ks, iterations, work_batch)
    else:
        if minimum > memory_budget_bytes:
            raise ValueError(
                f"memory budget too small: need at least {minimum} bytes")
        for subspace in range(m):
            start = subspace * subdim
            samples = np.array(vectors[sample_ids, start:start + subdim],
                               dtype=np.float32, copy=True)
            codebook[subspace] = train_subspace(
                samples, ks, iterations, work_batch)
    if not np.all(np.isfinite(codebook)):
        raise ValueError("trained codebook contains NaN/Inf")
    estimated_peak = max(fast_required if mode == "fast" else minimum,
                         codebook_bytes + work_batch * per_row_work)
    return codebook, mode, estimated_peak


def train_codebook_faiss(vectors: np.ndarray, sample_ids: np.ndarray,
                         m: int, code_bits: int, iterations: int,
                         seed: int, threads: int):
    try:
        import faiss
    except ImportError as exc:
        message = "Faiss engine requested; install faiss-cpu or use --engine numpy"
        raise ValueError(message) from exc
    samples = np.ascontiguousarray(vectors[sample_ids], dtype=np.float32)
    dim = samples.shape[1]
    faiss.omp_set_num_threads(threads)
    product_quantizer = faiss.ProductQuantizer(dim, m, code_bits)
    product_quantizer.cp.niter = iterations
    product_quantizer.cp.nredo = 1
    product_quantizer.cp.seed = seed
    product_quantizer.train(samples)
    centroid_values = faiss.vector_to_array(product_quantizer.centroids)
    expected_values = dim * (1 << code_bits)
    if centroid_values.size != expected_values:
        raise ValueError(
            "Faiss ProductQuantizer training returned incomplete centroids")
    codebook = centroid_values.reshape(
        m, 1 << code_bits, dim // m).astype(np.float32, copy=False)
    if not np.all(np.isfinite(codebook)):
        raise ValueError("Faiss codebook contains NaN/Inf")
    estimated_peak = samples.nbytes + codebook.nbytes
    return codebook, product_quantizer, estimated_peak


def encode_batch_faiss(vectors: np.ndarray, product_quantizer) -> np.ndarray:
    batch = np.ascontiguousarray(vectors, dtype=np.float32)
    codes = product_quantizer.compute_codes(batch)
    if codes.dtype != np.uint8 or codes.ndim != 2:
        raise ValueError("Faiss returned an unsupported PQ code layout")
    return codes


def encode_batch(vectors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    vector_count, dim = vectors.shape
    m, ks, subdim = codebook.shape
    if dim != m * subdim:
        raise ValueError("invalid encode dimensions")
    code_bits = code_bits_for_ks(ks)
    codes = np.zeros((vector_count, code_bytes_for(m, ks)), dtype=np.uint8)
    for subspace in range(m):
        start = subspace * subdim
        batch = np.asarray(vectors[:, start:start + subdim], dtype=np.float32)
        best_distance = np.full(vector_count, np.inf, dtype=np.float32)
        assignments = np.zeros(vector_count, dtype=np.uint8)
        for centroid_id in range(ks):
            diff = batch - codebook[subspace, centroid_id]
            distance = np.sum(diff * diff, axis=1, dtype=np.float32)
            better = distance < best_distance
            best_distance[better] = distance[better]
            assignments[better] = centroid_id
        if not np.all(np.isfinite(best_distance)):
            raise ValueError("NaN/Inf encountered during PQ encoding")
        if code_bits == 4:
            shift = 4 if subspace & 1 else 0
            codes[:, subspace // 2] |= assignments << shift
        else:
            codes[:, subspace] = assignments
    return codes


def fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_temp(temporary: Path, final: Path) -> None:
    os.replace(temporary, final)
    fsync_parent(final)


def write_binary(path: Path, kind: int, metadata: dict,
                 payload_chunks) -> tuple[int, bytes]:
    temporary = Path(str(path) + ".tmp")
    digest = hashlib.sha256()
    payload_bytes = 0
    try:
        with temporary.open("w+b") as file:
            file.write(bytes(HEADER_BYTES))
            for payload in payload_chunks:
                data = memoryview(payload).cast("B")
                file.write(data)
                digest.update(data)
                payload_bytes += len(data)
            file.seek(0)
            file.write(encode_header(kind, metadata, payload_bytes,
                                     digest.digest()))
            file.flush()
            os.fsync(file.fileno())
        publish_temp(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return payload_bytes, digest.digest()


def atomic_json(path: Path, document: dict) -> None:
    temporary = Path(str(path) + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(document, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        publish_temp(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build(args: argparse.Namespace) -> dict:
    base = args.base.resolve()
    prefix = args.out_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    finals = [Path(str(prefix) + suffix)
              for suffix in (".pqbook", ".pqcodes", ".pq.json")]
    if not args.force and any(path.exists() for path in finals):
        raise ValueError("output exists; pass --force to replace it")
    vectors, num_vectors, dim, record_bytes = open_fvecs(
        base, args.num_vectors, args.dim)
    layout = VectorLayout.load(
        args.layout_id, getattr(args, "order_map", None), num_vectors)
    if dim % args.m != 0:
        raise ValueError(f"dim={dim} is not divisible by M={args.m}")
    ks = 1 << args.code_bits
    if args.ks is not None and args.ks != ks:
        raise ValueError(
            f"--code-bits {args.code_bits} requires --ks {ks}")
    code_bytes = code_bytes_for(args.m, ks)
    train_count = min(num_vectors, args.train_max)
    sample_ids = uniform_sample_ids(num_vectors, train_count)
    memory_budget = args.memory_budget_mb * 1024 * 1024
    product_quantizer = None
    if args.engine == "faiss":
        codebook, product_quantizer, estimated_peak = train_codebook_faiss(
            vectors, sample_ids, args.m, args.code_bits,
            args.kmeans_iters, args.seed, args.threads)
        train_mode = "faiss"
        trainer_kind = TRAINER_SERVER_FAISS
    else:
        codebook, train_mode, estimated_peak = train_codebook(
            vectors, sample_ids, args.m, ks, args.kmeans_iters,
            memory_budget, args.chunk_vectors)
        trainer_kind = TRAINER_SERVER_NUMPY
    dataset_fingerprint = hash_file_prefix(base, num_vectors * record_bytes)
    layout_fingerprint = layout.fingerprint
    metadata = {
        "index_uuid": uuid.uuid4().bytes,
        "num_vectors": num_vectors,
        "dim": dim,
        "m": args.m,
        "ks": ks,
        "subdim": dim // args.m,
        "code_bytes": code_bytes,
        "trainer_kind": trainer_kind,
        "sample_strategy": SAMPLE_UNIFORM_STRIDE,
        "train_count": train_count,
        "kmeans_iters": args.kmeans_iters,
        "seed": args.seed,
        "dataset_fingerprint": dataset_fingerprint,
        "layout_fingerprint": layout_fingerprint,
    }
    validate_metadata(metadata)
    book_path, codes_path, manifest_path = finals
    book_payload = np.asarray(codebook, dtype="<f4").tobytes(order="C")
    book_bytes, book_hash = write_binary(
        book_path, ARTIFACT_CODEBOOK, metadata, [book_payload])

    def code_chunks():
        for start in range(0, num_vectors, args.chunk_vectors):
            batch = layout.gather(
                vectors, start, min(num_vectors, start + args.chunk_vectors))
            if product_quantizer is not None:
                codes = encode_batch_faiss(batch, product_quantizer)
            else:
                codes = encode_batch(batch, codebook)
            if codes.shape[1] != code_bytes:
                raise ValueError("encoder returned the wrong code byte count")
            yield codes.tobytes(order="C")

    codes_bytes, codes_hash = write_binary(
        codes_path, ARTIFACT_CODES, metadata, code_chunks())
    manifest = {
        "format": "pure-pq-v1",
        "artifact_source": f"server-{args.engine}",
        "index_uuid": metadata["index_uuid"].hex(),
        "num_vectors": num_vectors,
        "dim": dim,
        "m": args.m,
        "ks": ks,
        "subdim": dim // args.m,
        "code_bits": args.code_bits,
        "code_bytes": code_bytes,
        "compression_x": (dim * 4) / code_bytes,
        "metric": "squared_l2",
        "train_count": train_count,
        "kmeans_iters": args.kmeans_iters,
        "sample_strategy": "uniform_stride",
        "seed": args.seed,
        "dataset_fingerprint": dataset_fingerprint.hex(),
        "training_memory_mode": train_mode,
        "estimated_peak_bytes": estimated_peak,
        "book_payload_bytes": book_bytes,
        "book_payload_sha256": book_hash.hex(),
        "codes_payload_bytes": codes_bytes,
        "codes_payload_sha256": codes_hash.hex(),
    }
    manifest.update(layout.manifest_fields())
    atomic_json(manifest_path, manifest)
    validate_prefix(prefix)
    return manifest


def read_and_validate_binary(path: Path, kind: int) -> dict:
    file_bytes = path.stat().st_size
    with path.open("rb") as file:
        header = file.read(HEADER_BYTES)
        metadata = decode_header(header, kind)
        expected_size = HEADER_BYTES + metadata["payload_bytes"]
        if file_bytes != expected_size:
            raise ValueError(
                f"{path}: size {file_bytes}, expected {expected_size}")
        digest = hashlib.sha256()
        while True:
            data = file.read(8 << 20)
            if not data:
                break
            digest.update(data)
    if digest.digest() != metadata["payload_checksum"]:
        raise ValueError(f"{path}: payload checksum mismatch")
    return metadata


def shared_metadata(metadata: dict) -> dict:
    excluded = {"kind", "payload_bytes", "payload_checksum"}
    return {key: value for key, value in metadata.items() if key not in excluded}


def validate_prefix(prefix: Path) -> dict:
    book_path = Path(str(prefix) + ".pqbook")
    codes_path = Path(str(prefix) + ".pqcodes")
    book = read_and_validate_binary(book_path, ARTIFACT_CODEBOOK)
    codes = read_and_validate_binary(codes_path, ARTIFACT_CODES)
    if shared_metadata(book) != shared_metadata(codes):
        raise ValueError("pqbook and pqcodes metadata mismatch")
    expected_book = book["m"] * book["ks"] * book["subdim"] * 4
    expected_codes = book["num_vectors"] * book["code_bytes"]
    if book["payload_bytes"] != expected_book:
        raise ValueError("pqbook payload size mismatch")
    if codes["payload_bytes"] != expected_codes:
        raise ValueError("pqcodes payload size mismatch")
    return book


def printable_metadata(metadata: dict) -> dict:
    result = dict(metadata)
    result["code_bits"] = code_bits_for_ks(metadata["ks"])
    result["compression_x"] = (
        metadata["dim"] * 4 / metadata["code_bytes"])
    for field in ("index_uuid", "payload_checksum",
                  "dataset_fingerprint", "layout_fingerprint"):
        if field in result:
            result[field] = result[field].hex()
    return result


def reference_adc(prefix: Path, query_path: Path, query_index: int,
                  sample_ids: list[int]) -> list[dict]:
    metadata = validate_prefix(prefix)
    queries, query_count, query_dim, _ = open_fvecs(query_path, None,
                                                    metadata["dim"])
    if query_index < 0 or query_index >= query_count or query_dim != metadata["dim"]:
        raise ValueError("invalid query index/dimension")
    for vector_id in sample_ids:
        if vector_id < 0 or vector_id >= metadata["num_vectors"]:
            raise ValueError(f"sample ID {vector_id} is out of range")
    codebook = np.memmap(Path(str(prefix) + ".pqbook"), mode="r", dtype="<f4",
                         offset=HEADER_BYTES,
                         shape=(metadata["m"], metadata["ks"],
                                metadata["subdim"]))
    codes = np.memmap(Path(str(prefix) + ".pqcodes"), mode="r", dtype=np.uint8,
                      offset=HEADER_BYTES,
                      shape=(metadata["num_vectors"],
                             metadata["code_bytes"]))
    query = np.asarray(queries[query_index], dtype=np.float32)
    lut = np.empty((metadata["m"], metadata["ks"]), dtype=np.float32)
    for subspace in range(metadata["m"]):
        start = subspace * metadata["subdim"]
        diff = codebook[subspace] - query[start:start + metadata["subdim"]]
        lut[subspace] = np.sum(diff * diff, axis=1, dtype=np.float32)
    results = []
    for vector_id in sample_ids:
        distance = np.float32(0.0)
        for subspace in range(metadata["m"]):
            if metadata["ks"] == 16:
                packed = int(codes[vector_id, subspace // 2])
                centroid = ((packed >> 4) if subspace & 1
                            else (packed & 0x0f))
            else:
                centroid = int(codes[vector_id, subspace])
            distance = np.float32(distance + lut[subspace,
                                                   centroid])
        results.append({"vector_id": vector_id, "distance": float(distance)})
    return results


def add_common_validate_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--query-fvecs", type=Path)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--sample-ids", default="0")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Pure-PQ v1 artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--base", required=True, type=Path)
    build_parser.add_argument("--out-prefix", required=True, type=Path)
    build_parser.add_argument("--num-vectors", type=int)
    build_parser.add_argument("--dim", type=int)
    build_parser.add_argument("--m", required=True, type=int)
    build_parser.add_argument("--code-bits", type=int, choices=(4, 8),
                              default=8)
    build_parser.add_argument("--ks", type=int, choices=(16, 256),
                              help="optional consistency check for code bits")
    build_parser.add_argument("--train-max", type=int, default=100000)
    build_parser.add_argument("--kmeans-iters", type=int, default=20)
    build_parser.add_argument("--chunk-vectors", type=int, default=8192)
    build_parser.add_argument("--memory-budget-mb", type=int, default=512)
    build_parser.add_argument("--layout-id", required=True)
    build_parser.add_argument("--order-map", type=Path,
                              help="Gorder out_order[new_id]=old_id text file")
    build_parser.add_argument("--engine", choices=["numpy", "faiss"],
                              default="numpy")
    build_parser.add_argument("--threads", type=int,
                              default=max(1, os.cpu_count() or 1))
    build_parser.add_argument("--seed", type=int, default=0)
    build_parser.add_argument("--force", action="store_true")
    validate_parser = subparsers.add_parser("validate")
    add_common_validate_options(validate_parser)
    inspect_parser = subparsers.add_parser("inspect")
    add_common_validate_options(inspect_parser)
    args = parser.parse_args(argv)

    try:
        if args.command == "build":
            if (args.m <= 0 or args.train_max <= 0 or args.kmeans_iters <= 0 or
                    args.chunk_vectors <= 0 or args.memory_budget_mb <= 0 or
                    args.threads <= 0):
                raise ValueError("all build sizes/counts must be positive")
            document = build(args)
        else:
            metadata = validate_prefix(args.prefix.resolve())
            document = printable_metadata(metadata)
            if args.query_fvecs:
                ids = [int(item) for item in args.sample_ids.split(",") if item]
                document["reference_adc"] = reference_adc(
                    args.prefix.resolve(), args.query_fvecs.resolve(),
                    args.query_index, ids)
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(f"Pure-PQ {args.command} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
