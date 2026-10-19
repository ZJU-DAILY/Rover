#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Mapping

import numpy as np


L1_HEADER = struct.Struct("<IIiiiff4s")
QPARAMS_HEADER = struct.Struct("<IIiiff8s")
BIAS_HEADER = struct.Struct("<8sIIIIIIQQ16s")
BIAS_MAGIC = b"SQMDBIAS"
BIAS_VERSION = 1
BIAS_HEADER_BYTES = 64
RESIDUAL_BLOCK_BYTES = 4096
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return value


def resolve_reference(manifest_path: Path,
                      reference: Mapping[str, object]) -> Path:
    path = Path(str(reference["path"]))
    return path.resolve() if path.is_absolute() else \
        (manifest_path.parent / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(8 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference(manifest_path: Path, name: str,
                       reference: Mapping[str, object]) -> Path:
    path = resolve_reference(manifest_path, reference)
    if not path.is_file():
        raise FileNotFoundError("missing {}: {}".format(name, path))
    if int(reference.get("bytes", path.stat().st_size)) != path.stat().st_size:
        raise ValueError("{} size differs from manifest".format(name))
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("{} hash differs from manifest".format(name))
    return path


def source_fingerprint(manifest: Mapping[str, object]) -> bytes:
    digest = hashlib.sha256()
    for name in ("l1", "residual", "qparams"):
        digest.update(bytes.fromhex(str(manifest[name]["sha256"])))
    return digest.digest()[:16]


def fnv1a64(payload: bytes) -> int:
    value = FNV_OFFSET
    for byte in payload:
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def unpack_u4(packed: np.ndarray, dim: int) -> np.ndarray:
    output = np.empty((packed.shape[0], dim), dtype=np.uint8)
    output[:, 0::2] = packed[:, : (dim + 1) // 2] & np.uint8(0x0F)
    if dim > 1:
        output[:, 1::2] = packed[:, : dim // 2] >> np.uint8(4)
    return output


def generate(manifest_path: Path, output: Path, chunk_vectors: int) -> None:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if manifest.get("layout") not in ("original", "gorder"):
        raise ValueError("unsupported artifact layout")
    n = int(manifest["N"])
    dim = int(manifest["dim"])
    if n <= 0 or dim <= 0 or chunk_vectors <= 0:
        raise ValueError("N, dim and chunk_vectors must be positive")
    l1_path = validate_reference(manifest_path, "l1", manifest["l1"])
    residual_path = validate_reference(
        manifest_path, "residual", manifest["residual"])
    qparams_path = validate_reference(
        manifest_path, "qparams", manifest["qparams"])

    with l1_path.open("rb") as source:
        l1_header = L1_HEADER.unpack(source.read(L1_HEADER.size))
    with qparams_path.open("rb") as source:
        qparams_header = QPARAMS_HEADER.unpack(
            source.read(QPARAMS_HEADER.size))
    _, l1_version, l1_n, l1_dim, l1_bits, l1_scale, l1_zero, _ = l1_header
    _, qp_version, qp_n, residual_bits, qp_scale, qp_zero, _ = qparams_header
    if (l1_version != 1 or qp_version != 1 or l1_n != n or qp_n != n or
            l1_dim != dim or l1_bits != 4 or residual_bits not in (4, 8) or
            struct.pack("<f", l1_scale) != struct.pack("<f", qp_scale) or
            struct.pack("<f", l1_zero) != struct.pack("<f", qp_zero) or
            not math.isfinite(l1_scale) or l1_scale <= 0.0 or
            not math.isfinite(l1_zero)):
        raise ValueError("split bias requires matching U4/U4 or U4/U8 headers")

    l1_code_bytes = (dim + 1) // 2
    residual_code_bytes = (dim + 1) // 2 if residual_bits == 4 else dim
    vectors_per_block = RESIDUAL_BLOCK_BYTES // residual_code_bytes
    if vectors_per_block == 0:
        raise ValueError("residual vector exceeds the block size")
    blocks = (n + vectors_per_block - 1) // vectors_per_block
    if l1_path.stat().st_size != L1_HEADER.size + n * l1_code_bytes or \
            qparams_path.stat().st_size != QPARAMS_HEADER.size + n * 8 or \
            residual_path.stat().st_size != blocks * RESIDUAL_BLOCK_BYTES:
        raise ValueError("encoded artifact size does not match its headers")

    l1 = np.memmap(l1_path, mode="r", dtype=np.uint8,
                   offset=L1_HEADER.size, shape=(n, l1_code_bytes))
    qparams = np.memmap(qparams_path, mode="r", dtype="<f4",
                       offset=QPARAMS_HEADER.size, shape=(n, 2))
    residual_blocks = np.memmap(
        residual_path, mode="r", dtype=np.uint8,
        shape=(blocks, RESIDUAL_BLOCK_BYTES))
    storage = None
    if manifest.get('residual_map'):
        storage_path=validate_reference(manifest_path,'residual_map',manifest['residual_map'])
        storage=np.loadtxt(storage_path,dtype='<i4',ndmin=1)
        if not np.array_equal(np.sort(storage),np.arange(n)):
            raise ValueError('Invalid residual map')
    temporary = output.with_name("." + output.name + ".tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_offset = BIAS_HEADER_BYTES
    with temporary.open("wb") as sink:
        sink.write(b"\x00" * BIAS_HEADER_BYTES)
        for start in range(0, n, chunk_vectors):
            end = min(n, start + chunk_vectors)
            ids = np.arange(start, end, dtype=np.int64)
            slots=ids if storage is None else storage[ids]
            block_ids = slots // vectors_per_block
            within = slots % vectors_per_block
            residual_packed = np.empty((end - start, residual_code_bytes),
                                       dtype=np.uint8)
            for row in range(end - start):
                offset = int(within[row]) * residual_code_bytes
                residual_packed[row] = residual_blocks[block_ids[row],
                                                       offset:offset + residual_code_bytes]
            l1_codes = unpack_u4(np.asarray(l1[start:end]), dim).astype(np.float64)
            residual_codes = (unpack_u4(residual_packed, dim) if residual_bits == 4
                              else residual_packed).astype(np.float64)
            a = l1_codes * float(l1_scale) + float(l1_zero)
            scales = np.asarray(qparams[start:end, 0], dtype=np.float64)[:, None]
            zeros = np.asarray(qparams[start:end, 1], dtype=np.float64)[:, None]
            if (not np.isfinite(scales).all() or (scales < 0).any() or
                    not np.isfinite(zeros).all()):
                raise ValueError("invalid residual quantization parameters")
            residual = residual_codes * scales + zeros
            bias = np.sum(2.0 * a * residual + residual * residual,
                          axis=1, dtype=np.float64).astype("<f4")
            if not np.isfinite(bias).all():
                raise ValueError("non-finite correction bias generated")
            sink.write(bias.tobytes(order="C"))
        sink.flush()
        os.fsync(sink.fileno())

    payload = temporary.read_bytes()[payload_offset:]
    if len(payload) != n * 4:
        temporary.unlink(missing_ok=True)
        raise ValueError("generated bias payload has the wrong size")
    header = BIAS_HEADER.pack(
        BIAS_MAGIC, BIAS_VERSION, BIAS_HEADER_BYTES, n, dim, l1_bits, residual_bits,
        len(payload), fnv1a64(payload), source_fingerprint(manifest))
    with temporary.open("r+b") as sink:
        sink.write(header)
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temporary, output)
    print("bias artifact: {}".format(output))
    print("vectors={} dim={} bytes={} fingerprint={}".format(
        n, dim, output.stat().st_size, source_fingerprint(manifest).hex()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-vectors", type=int, default=8192)
    args = parser.parse_args()
    if args.chunk_vectors <= 0:
        parser.error("--chunk-vectors must be positive")
    generate(args.artifact_manifest, args.output.resolve(),
             args.chunk_vectors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
