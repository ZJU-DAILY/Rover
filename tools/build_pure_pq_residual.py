#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

from build_pure_pq_index import (
    HEADER_BYTES as PQ_HEADER_BYTES,
    ARTIFACT_CODEBOOK,
    decode_header as decode_pq_header,
    open_fvecs,
    publish_temp,
    validate_prefix,
)
from pure_pq_layout import VectorLayout


MAGIC = b"PQRES01\0"
VERSION_MAJOR = 1
VERSION_MINOR = 0
ENDIAN_MARKER = 0x01020304
KIND_BLOCKS = 1
KIND_QPARAMS = 2
HEADER_BYTES = 256
BLOCK_BYTES = 4096
QPARAM_BYTES = 8


def artifact_paths(prefix: Path, residual_bits: int) -> tuple[Path, Path, Path]:
    stem = f"{prefix}.pqres{residual_bits}"
    return Path(stem), Path(stem + ".qparams"), Path(stem + ".json")


def residual_code_bytes(dim: int, residual_bits: int) -> int:
    return (dim + 1) // 2 if residual_bits == 4 else dim


def encode_header(kind: int, storage_header_bytes: int, metadata: dict,
                  payload_bytes: int, payload_hash: bytes) -> bytes:
    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    struct.pack_into("<HHIII", header, 8,
                     VERSION_MAJOR, VERSION_MINOR,
                     storage_header_bytes, kind, ENDIAN_MARKER)
    struct.pack_into("<II", header, 24,
                     metadata["residual_bits"],
                     metadata["residual_code_bytes"])
    header[32:48] = metadata["index_uuid"]
    struct.pack_into("<QIIIIQ", header, 48,
                     metadata["num_vectors"], metadata["dim"],
                     metadata["block_bytes"],
                     metadata["vectors_per_block"],
                     metadata["total_blocks"], payload_bytes)
    header[80:112] = payload_hash
    header[112:144] = metadata["dataset_fingerprint"]
    header[144:176] = metadata["layout_fingerprint"]
    struct.pack_into("<I", header, 176, QPARAM_BYTES)
    return bytes(header)


def decode_header(header: bytes, expected_kind: int,
                  expected_storage_header_bytes: int) -> dict:
    if len(header) != HEADER_BYTES or header[0:8] != MAGIC:
        raise ValueError("invalid PQ residual header")
    major, minor, storage_header_bytes, kind, endian = struct.unpack_from(
        "<HHIII", header, 8)
    if (major != VERSION_MAJOR or minor > VERSION_MINOR or
            storage_header_bytes != expected_storage_header_bytes or
            kind != expected_kind or endian != ENDIAN_MARKER):
        raise ValueError("unsupported PQ residual header")
    residual_bits, code_bytes = struct.unpack_from("<II", header, 24)
    num_vectors, dim, block_bytes, vectors_per_block, total_blocks = (
        struct.unpack_from("<QIIII", header, 48))
    payload_bytes = struct.unpack_from("<Q", header, 72)[0]
    qparam_bytes = struct.unpack_from("<I", header, 176)[0]
    if residual_bits not in (4, 8) or qparam_bytes != QPARAM_BYTES:
        raise ValueError("unsupported PQ residual encoding")
    expected_code_bytes = residual_code_bytes(dim, residual_bits)
    expected_vectors_per_block = BLOCK_BYTES // expected_code_bytes
    expected_blocks = (num_vectors + expected_vectors_per_block - 1) // (
        expected_vectors_per_block)
    if (num_vectors <= 0 or dim <= 0 or block_bytes != BLOCK_BYTES or
            code_bytes != expected_code_bytes or
            vectors_per_block != expected_vectors_per_block or
            total_blocks != expected_blocks):
        raise ValueError("inconsistent PQ residual metadata")
    return {
        "index_uuid": header[32:48],
        "num_vectors": num_vectors,
        "dim": dim,
        "residual_bits": residual_bits,
        "residual_code_bytes": code_bytes,
        "block_bytes": block_bytes,
        "vectors_per_block": vectors_per_block,
        "total_blocks": total_blocks,
        "payload_bytes": payload_bytes,
        "payload_hash": header[80:112],
        "dataset_fingerprint": header[112:144],
        "layout_fingerprint": header[144:176],
    }


def shared_metadata(metadata: dict) -> dict:
    return {key: value for key, value in metadata.items()
            if key not in {"payload_bytes", "payload_hash"}}


def unpack_codes(codes: np.ndarray, subspace: int, ks: int) -> np.ndarray:
    if ks == 16:
        packed = codes[:, subspace // 2]
        if subspace & 1:
            return packed >> np.uint8(4)
        return packed & np.uint8(0x0F)
    return codes[:, subspace]


def reconstruct_batch(codes: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    count = codes.shape[0]
    m, ks, subdim = codebook.shape
    reconstruction = np.empty((count, m * subdim), dtype=np.float32)
    for subspace in range(m):
        assignments = unpack_codes(codes, subspace, ks)
        start = subspace * subdim
        reconstruction[:, start:start + subdim] = codebook[
            subspace, assignments]
    return reconstruction


def quantize_residual(residual: np.ndarray,
                      residual_bits: int) -> tuple[np.ndarray, np.ndarray]:
    levels = 15 if residual_bits == 4 else 255
    minimum = np.min(residual, axis=1).astype(np.float32)
    maximum = np.max(residual, axis=1).astype(np.float32)
    scale = np.maximum((maximum - minimum) / np.float32(levels),
                       np.float32(1.0e-12))
    quantized = np.rint((residual - minimum[:, None]) / scale[:, None])
    quantized = np.clip(quantized, 0, levels).astype(np.uint8)
    if residual_bits == 4:
        packed = np.zeros((residual.shape[0], (residual.shape[1] + 1) // 2),
                          dtype=np.uint8)
        packed[:, :residual.shape[1] // 2] = (
            quantized[:, 0:residual.shape[1] - 1:2] |
            (quantized[:, 1:residual.shape[1]:2] << np.uint8(4)))
        if residual.shape[1] & 1:
            packed[:, -1] = quantized[:, -1]
        quantized = packed
    qparams = np.empty((residual.shape[0], 2), dtype="<f4")
    qparams[:, 0] = scale
    qparams[:, 1] = minimum
    return quantized, qparams


class ResidualBlockWriter:
    def __init__(self, file, code_bytes: int, digest):
        self.file = file
        self.code_bytes = code_bytes
        self.vectors_per_block = BLOCK_BYTES // code_bytes
        self.pending = np.zeros((self.vectors_per_block, code_bytes),
                                dtype=np.uint8)
        self.pending_count = 0
        self.digest = digest
        self.blocks = 0

    def _write_pending(self) -> None:
        block = bytearray(BLOCK_BYTES)
        used = self.pending_count * self.code_bytes
        block[:used] = self.pending[:self.pending_count].tobytes(order="C")
        self.file.write(block)
        self.digest.update(block)
        self.blocks += 1
        self.pending_count = 0

    def append(self, codes: np.ndarray) -> None:
        offset = 0
        count = codes.shape[0]
        if self.pending_count:
            take = min(count, self.vectors_per_block - self.pending_count)
            self.pending[self.pending_count:self.pending_count + take] = (
                codes[:take])
            self.pending_count += take
            offset += take
            if self.pending_count == self.vectors_per_block:
                self._write_pending()
        remaining = count - offset
        full_groups = remaining // self.vectors_per_block
        if full_groups:
            end = offset + full_groups * self.vectors_per_block
            grouped = codes[offset:end].reshape(
                full_groups, self.vectors_per_block, self.code_bytes)
            padded = np.zeros((full_groups, BLOCK_BYTES), dtype=np.uint8)
            padded[:, :self.vectors_per_block * self.code_bytes] = (
                grouped.reshape(full_groups, -1))
            data = padded.tobytes(order="C")
            self.file.write(data)
            self.digest.update(data)
            self.blocks += full_groups
            offset = end
        if offset < count:
            tail = codes[offset:]
            self.pending[:tail.shape[0]] = tail
            self.pending_count = tail.shape[0]

    def finish(self) -> None:
        if self.pending_count:
            self._write_pending()


def validate(prefix: Path, residual_bits: int, verify_blocks: bool = True) -> dict:
    pq = validate_prefix(prefix)
    blocks_path, qparams_path, _ = artifact_paths(prefix, residual_bits)
    with blocks_path.open("rb") as file:
        blocks = decode_header(file.read(HEADER_BYTES), KIND_BLOCKS, BLOCK_BYTES)
        file.seek(BLOCK_BYTES)
        blocks_digest = hashlib.sha256()
        if verify_blocks:
            while True:
                data = file.read(8 << 20)
                if not data:
                    break
                blocks_digest.update(data)
    with qparams_path.open("rb") as file:
        qparams = decode_header(file.read(HEADER_BYTES), KIND_QPARAMS,
                                HEADER_BYTES)
        qparams_payload = file.read()
    if shared_metadata(blocks) != shared_metadata(qparams):
        raise ValueError("PQ residual blocks/qparams metadata mismatch")
    for key in ("index_uuid", "num_vectors", "dim",
                "dataset_fingerprint", "layout_fingerprint"):
        if blocks[key] != pq[key]:
            raise ValueError(f"PQ residual metadata mismatch: {key}")
    expected_blocks_payload = blocks["total_blocks"] * BLOCK_BYTES
    expected_qparams_payload = blocks["num_vectors"] * QPARAM_BYTES
    if (blocks["payload_bytes"] != expected_blocks_payload or
            blocks_path.stat().st_size != BLOCK_BYTES + expected_blocks_payload):
        raise ValueError("PQ residual block payload size mismatch")
    if (qparams["payload_bytes"] != expected_qparams_payload or
            len(qparams_payload) != expected_qparams_payload):
        raise ValueError("PQ residual qparams payload size mismatch")
    if verify_blocks and blocks_digest.digest() != blocks["payload_hash"]:
        raise ValueError("PQ residual block checksum mismatch")
    if hashlib.sha256(qparams_payload).digest() != qparams["payload_hash"]:
        raise ValueError("PQ residual qparams checksum mismatch")
    return blocks


def build(args: argparse.Namespace) -> dict:
    prefix = args.prefix.resolve()
    base = args.base.resolve()
    pq = validate_prefix(prefix)
    vectors, count, dim, _ = open_fvecs(base, args.num_vectors, pq["dim"])
    if count != pq["num_vectors"] or dim != pq["dim"]:
        raise ValueError("raw dataset and PQ index dimensions differ")
    pq_manifest_path = Path(str(prefix) + ".pq.json")
    with pq_manifest_path.open("r", encoding="utf-8") as file:
        pq_manifest = json.load(file)
    layout = VectorLayout.load(
        pq_manifest.get("layout_id", ""),
        getattr(args, "order_map", None), count)
    layout.verify_manifest(pq_manifest)
    if layout.fingerprint != pq["layout_fingerprint"]:
        raise ValueError("PQ header and order-map layout fingerprints differ")
    blocks_path, qparams_path, manifest_path = artifact_paths(
        prefix, args.residual_bits)
    finals = (blocks_path, qparams_path, manifest_path)
    if not args.force and any(path.exists() for path in finals):
        raise ValueError("residual output exists; pass --force to replace it")
    blocks_temporary = Path(str(blocks_path) + ".tmp")
    qparams_temporary = Path(str(qparams_path) + ".tmp")
    manifest_temporary = Path(str(manifest_path) + ".tmp")
    codebook = np.memmap(Path(str(prefix) + ".pqbook"), mode="r",
                         dtype="<f4", offset=PQ_HEADER_BYTES,
                         shape=(pq["m"], pq["ks"], pq["subdim"]))
    codes = np.memmap(Path(str(prefix) + ".pqcodes"), mode="r",
                      dtype=np.uint8, offset=PQ_HEADER_BYTES,
                      shape=(count, pq["code_bytes"]))
    code_bytes = residual_code_bytes(dim, args.residual_bits)
    vectors_per_block = BLOCK_BYTES // code_bytes
    total_blocks = (count + vectors_per_block - 1) // vectors_per_block
    metadata = {
        "index_uuid": pq["index_uuid"],
        "num_vectors": count,
        "dim": dim,
        "residual_bits": args.residual_bits,
        "residual_code_bytes": code_bytes,
        "block_bytes": BLOCK_BYTES,
        "vectors_per_block": vectors_per_block,
        "total_blocks": total_blocks,
        "dataset_fingerprint": pq["dataset_fingerprint"],
        "layout_fingerprint": pq["layout_fingerprint"],
    }
    blocks_digest = hashlib.sha256()
    qparams_digest = hashlib.sha256()
    try:
        with blocks_temporary.open("w+b") as blocks_file, \
                qparams_temporary.open("w+b") as qparams_file:
            blocks_file.write(bytes(BLOCK_BYTES))
            qparams_file.write(bytes(HEADER_BYTES))
            block_writer = ResidualBlockWriter(
                blocks_file, code_bytes, blocks_digest)
            for start in range(0, count, args.chunk_vectors):
                end = min(count, start + args.chunk_vectors)
                raw = layout.gather(vectors, start, end)
                chunk_codes = np.asarray(codes[start:end], dtype=np.uint8)
                reconstruction = reconstruct_batch(chunk_codes, codebook)
                quantized, qparams = quantize_residual(
                    raw - reconstruction, args.residual_bits)
                block_writer.append(quantized)
                qparams_data = qparams.tobytes(order="C")
                qparams_file.write(qparams_data)
                qparams_digest.update(qparams_data)
            block_writer.finish()
            if block_writer.blocks != total_blocks:
                raise ValueError("residual block count mismatch")
            blocks_payload_bytes = total_blocks * BLOCK_BYTES
            qparams_payload_bytes = count * QPARAM_BYTES
            blocks_file.seek(0)
            blocks_file.write(encode_header(
                KIND_BLOCKS, BLOCK_BYTES, metadata,
                blocks_payload_bytes, blocks_digest.digest()))
            qparams_file.seek(0)
            qparams_file.write(encode_header(
                KIND_QPARAMS, HEADER_BYTES, metadata,
                qparams_payload_bytes, qparams_digest.digest()))
            blocks_file.flush()
            qparams_file.flush()
            os.fsync(blocks_file.fileno())
            os.fsync(qparams_file.fileno())
        manifest = {
            "format": "pure-pq-residual-v1",
            "index_uuid": pq["index_uuid"].hex(),
            "num_vectors": count,
            "dim": dim,
            "residual_bits": args.residual_bits,
            "residual_code_bytes": code_bytes,
            "block_bytes": BLOCK_BYTES,
            "vectors_per_block": vectors_per_block,
            "total_blocks": total_blocks,
            "blocks_payload_bytes": total_blocks * BLOCK_BYTES,
            "blocks_payload_sha256": blocks_digest.hexdigest(),
            "qparams_payload_bytes": count * QPARAM_BYTES,
            "qparams_payload_sha256": qparams_digest.hexdigest(),
            "dataset_fingerprint": pq["dataset_fingerprint"].hex(),
            "layout_fingerprint": pq["layout_fingerprint"].hex(),
            "layout_id": layout.layout_id,
            "order_map_direction": pq_manifest["order_map_direction"],
            "order_map_fingerprint": pq_manifest["order_map_fingerprint"],
        }
        with manifest_temporary.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        publish_temp(blocks_temporary, blocks_path)
        publish_temp(qparams_temporary, qparams_path)
        publish_temp(manifest_temporary, manifest_path)
    except Exception:
        blocks_temporary.unlink(missing_ok=True)
        qparams_temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
        raise
    validate(prefix, args.residual_bits)
    return manifest


def printable(metadata: dict) -> dict:
    result = dict(metadata)
    for key in ("index_uuid", "payload_hash",
                "dataset_fingerprint", "layout_fingerprint"):
        if key in result:
            result[key] = result[key].hex()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build PQ-reconstruction residual UFS artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--base", required=True, type=Path)
    build_parser.add_argument("--prefix", required=True, type=Path)
    build_parser.add_argument("--residual-bits", type=int, choices=(4, 8),
                              default=4)
    build_parser.add_argument("--num-vectors", type=int)
    build_parser.add_argument("--chunk-vectors", type=int, default=4096)
    build_parser.add_argument("--order-map", type=Path,
                              help="same out_order[new_id]=old_id used by PQ")
    build_parser.add_argument("--force", action="store_true")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--prefix", required=True, type=Path)
    validate_parser.add_argument("--residual-bits", type=int, choices=(4, 8),
                                 default=4)
    validate_parser.add_argument("--skip-block-checksum", action="store_true")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--prefix", required=True, type=Path)
    inspect_parser.add_argument("--residual-bits", type=int, choices=(4, 8),
                                default=4)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            if args.chunk_vectors <= 0:
                raise ValueError("--chunk-vectors must be positive")
            document = build(args)
        else:
            document = printable(validate(
                args.prefix.resolve(), args.residual_bits,
                not getattr(args, "skip_block_checksum", False)))
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(f"Pure-PQ residual {args.command} failed: {exc}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
