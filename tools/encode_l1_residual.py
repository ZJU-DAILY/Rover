#!/usr/bin/env python3

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

L1_CACHE_MAGIC = 0x314C4D53
L1_CACHE_VERSION = 1
L1_HEADER_FMT = '<IIiiiff4s'
L1_HEADER_SIZE = struct.calcsize(L1_HEADER_FMT)

QPARAMS_CACHE_MAGIC = 0x51504D53
QPARAMS_CACHE_VERSION = 1
QPARAMS_HEADER_FMT = '<IIiiff8s'
QPARAMS_HEADER_SIZE = struct.calcsize(QPARAMS_HEADER_FMT)

RESIDUAL_BLOCK_SIZE = 4096

_F32_HALF = np.float32(0.5)
_F32_EPS = np.float32(1e-12)

assert L1_HEADER_SIZE == 32, L1_HEADER_SIZE
assert QPARAMS_HEADER_SIZE == 32, QPARAMS_HEADER_SIZE


def read_fvecs_header(path: Path) -> "tuple[int, int]":
    path = Path(path)
    with open(path, 'rb') as f:
        dim = struct.unpack('<i', f.read(4))[0]
    if dim <= 0:
        raise ValueError(f"Invalid dim={dim} in {path}")

    file_size = path.stat().st_size
    row_bytes = 4 + dim * 4
    if file_size % row_bytes != 0:
        raise ValueError(
            f"{path}: file size {file_size} not aligned to row_bytes={row_bytes}"
        )
    return file_size // row_bytes, dim


def iter_fvecs_chunks(path: Path, dim: int, chunk_vectors: int, limit=None):
    dt = np.dtype([('_dim', '<i4'), ('vec', '<f4', (dim,))])
    total = 0
    with open(path, 'rb') as f:
        while True:
            want = chunk_vectors
            if limit is not None:
                remaining = limit - total
                if remaining <= 0:
                    break
                want = min(want, remaining)
            arr = np.fromfile(f, dtype=dt, count=want)
            if len(arr) == 0:
                break
            total += len(arr)
            yield np.ascontiguousarray(arr['vec'])
            if len(arr) < want:
                break


def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (np.sign(x).astype(np.float32)
            * np.floor(np.abs(x) + _F32_HALF).astype(np.float32)).astype(np.float32)


def pack_nibbles(q: np.ndarray, bits: int, dim: int) -> np.ndarray:
    if bits == 8:
        return q.astype(np.uint8)
    assert bits == 4
    rows, d = q.shape
    assert d == dim
    if dim % 2 != 0:
        pad = np.zeros((rows, 1), dtype=q.dtype)
        q = np.concatenate([q, pad], axis=1)
    low = q[:, 0::2].astype(np.uint8)
    high = q[:, 1::2].astype(np.uint8)
    return (low | (high << np.uint8(4))).astype(np.uint8)


def quantize_l1_chunk(chunk: np.ndarray, l1_scale: np.float32,
                      l1_zero_point: np.float32, l1_bits: int):
    levels = np.float32((1 << l1_bits) - 1)
    inv_scale = np.float32(1.0) / l1_scale
    q = round_half_away_from_zero((chunk - l1_zero_point) * inv_scale)
    q = np.clip(q, 0, levels).astype(np.uint8)
    deq = (q.astype(np.float32) * l1_scale + l1_zero_point).astype(np.float32)
    packed = pack_nibbles(q, l1_bits, chunk.shape[1])
    return packed, deq


def quantize_residual_chunk(residual: np.ndarray, residual_bits: int):
    rmin = residual.min(axis=1)
    rmax = residual.max(axis=1)
    r_range = np.maximum(rmax - rmin, _F32_EPS).astype(np.float32)
    levels = np.float32((1 << residual_bits) - 1)
    scale = (r_range / levels).astype(np.float32)
    zero_point = rmin.astype(np.float32)
    inv_scale = (np.float32(1.0) / scale).astype(np.float32)
    q = round_half_away_from_zero(
        (residual - zero_point[:, None]) * inv_scale[:, None])
    q = np.clip(q, 0, levels).astype(np.uint8)
    packed = pack_nibbles(q, residual_bits, residual.shape[1])
    return packed, scale, zero_point


class ResidualStreamWriter:
    def __init__(self, path: Path, residual_code_bytes: int,
                block_size: int = RESIDUAL_BLOCK_SIZE):
        self.f = open(path, 'wb')
        self.residual_code_bytes = residual_code_bytes
        self.block_size = block_size
        self.vectors_per_block = block_size // residual_code_bytes
        self.full_block_used_bytes = self.vectors_per_block * residual_code_bytes
        self.pending = bytearray()

    def write_chunk(self, codes_2d: np.ndarray) -> None:
        self.pending += codes_2d.tobytes()
        used = self.full_block_used_bytes
        while len(self.pending) >= used:
            block = bytes(self.pending[:used]) + b'\x00' * (self.block_size - used)
            self.f.write(block)
            del self.pending[:used]

    def finish(self) -> None:
        if self.pending:
            block = bytes(self.pending) + b'\x00' * (self.block_size - len(self.pending))
            self.f.write(block)
            self.pending = bytearray()
        self.f.flush()
        import os
        os.fsync(self.f.fileno())
        self.f.close()


def encode(base_fvecs: Path, l1_out: Path, residual_out: Path, qparams_out: Path,
          l1_bits: int, residual_bits: int, chunk_vectors: int,
          num_vectors_override=None) -> None:
    n_total, dim = read_fvecs_header(base_fvecs)
    n = n_total if num_vectors_override is None else min(n_total, num_vectors_override)
    l1_code_bytes = (dim + 1) // 2 if l1_bits == 4 else dim
    residual_code_bytes = (dim + 1) // 2 if residual_bits == 4 else dim

    print(f"  Encoding {base_fvecs.name}: {n:,}/{n_total:,} vectors x dim={dim}, "
          f"l1_bits={l1_bits} residual_bits={residual_bits}", flush=True)

    gmin = np.float32(np.inf)
    gmax = np.float32(-np.inf)
    for chunk in iter_fvecs_chunks(base_fvecs, dim, chunk_vectors, limit=n):
        gmin = np.minimum(gmin, chunk.min())
        gmax = np.maximum(gmax, chunk.max())

    l1_range = np.maximum(gmax - gmin, _F32_EPS).astype(np.float32)
    l1_levels = np.float32((1 << l1_bits) - 1)
    l1_scale = (l1_range / l1_levels).astype(np.float32)
    l1_zero_point = np.float32(gmin)
    print(f"  [P8.3] L1 scale={float(l1_scale):.6f} zero={float(l1_zero_point):.4f}",
          flush=True)

    l1_out.parent.mkdir(parents=True, exist_ok=True)
    residual_out.parent.mkdir(parents=True, exist_ok=True)
    qparams_out.parent.mkdir(parents=True, exist_ok=True)

    l1_f = open(l1_out, 'wb')
    l1_f.write(struct.pack(L1_HEADER_FMT, L1_CACHE_MAGIC, L1_CACHE_VERSION,
                           n, dim, l1_bits, float(l1_scale), float(l1_zero_point),
                           b'\x00' * 4))

    qp_f = open(qparams_out, 'wb')
    qp_f.write(struct.pack(QPARAMS_HEADER_FMT, QPARAMS_CACHE_MAGIC,
                           QPARAMS_CACHE_VERSION, n, residual_bits,
                           float(l1_scale), float(l1_zero_point), b'\x00' * 8))
    qp_dtype = np.dtype([('scale', '<f4'), ('zero_point', '<f4')])

    res_writer = ResidualStreamWriter(residual_out, residual_code_bytes)

    processed = 0
    next_progress = chunk_vectors * 32
    for chunk in iter_fvecs_chunks(base_fvecs, dim, chunk_vectors, limit=n):
        l1_codes, deq = quantize_l1_chunk(chunk, l1_scale, l1_zero_point, l1_bits)
        residual = (chunk - deq).astype(np.float32)
        res_codes, r_scale, r_zero_point = quantize_residual_chunk(residual, residual_bits)

        l1_f.write(l1_codes.tobytes())
        res_writer.write_chunk(res_codes)

        qp_chunk = np.empty(chunk.shape[0], dtype=qp_dtype)
        qp_chunk['scale'] = r_scale
        qp_chunk['zero_point'] = r_zero_point
        qp_f.write(qp_chunk.tobytes())

        processed += chunk.shape[0]
        if processed >= next_progress or processed == n:
            print(f"  [P8.3] encoded {processed:,}/{n:,}", flush=True)
            next_progress += chunk_vectors * 32

    res_writer.finish()
    l1_f.flush()
    import os
    os.fsync(l1_f.fileno())
    l1_f.close()
    qp_f.flush()
    os.fsync(qp_f.fileno())
    qp_f.close()

    print(f"  Wrote {l1_out} ({l1_out.stat().st_size / 1e6:.1f} MB), "
          f"{residual_out} ({residual_out.stat().st_size / 1e6:.1f} MB), "
          f"{qparams_out} ({qparams_out.stat().st_size / 1e6:.1f} MB)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=(
        '\n'
        'Pre-encode a .fvecs base file into L1 (int4/int8) codes + int4/int8 residual\n'
        'codes + qparams, on the workstation, so the on-device demo can skip encoding\n'
        'entirely (Phase 8.3, RETRIEVAL_DEMO_ENCODE_SOURCE=1).\n'
        '\n'
        "Reproduces retrieval_rpc_demo_support.c's demo_quantize_l1_vector() and\n"
        'demo_quantize_residual() bit-for-bit (same float32 arithmetic order,\n'
        'including the reciprocal-then-multiply pattern and round-half-away-from-zero\n'
        "rounding used by C's roundf() -- NOT numpy's default round-half-to-even).\n"
        'Output is byte-identical to what the C streaming encoder\n'
        '(retrieval_demo_encode_dataset_streaming_to_disk) produces for the same\n'
        'input, given RESIDUAL_REORDER_SCHEME == 0.\n'
        '\n'
        'Output files:\n'
        '  L1 codes file (--l1-out), new format (Phase 8.3):\n'
        "    Header (32 bytes): uint32 magic=0x314C4D53 ('L1MS'), uint32 version=1,\n"
        '      int32 num_vectors, int32 dim, int32 l1_bits, float32 l1_scale,\n'
        '      float32 l1_zero_point, 4 reserved zero bytes.\n'
        '    Body: num_vectors * l1_code_bytes bytes, flat, no block padding.\n'
        '      l1_code_bytes = (dim+1)//2 if l1_bits==4 else dim.\n'
        "      4-bit packing: byte b = code[2b] | (code[2b+1] << 4); the last byte's\n"
        '      high nibble is 0 when dim is odd (no partner for the last index).\n'
        '\n'
        '  Residual UFS block file (--residual-out), unchanged Phase 8.2 format:\n'
        '    RESIDUAL_BLOCK_SIZE=4096-byte blocks, vectors_per_block =\n'
        '      4096 // residual_code_bytes, vectors placed sequentially in id order,\n'
        '      final (and any partially-filled) block zero-padded past the last\n'
        "      vector's bytes.\n"
        '\n'
        '  qparams cache file (--qparams-out), unchanged Phase 8.2 format:\n'
        "    Header (32 bytes): uint32 magic=0x51504D53 ('QPMS'), uint32 version=1,\n"
        '      int32 num_vectors, int32 residual_bits, float32 l1_scale,\n'
        '      float32 l1_zero_point, 8 reserved zero bytes.\n'
        '    Body: num_vectors * 8 bytes (float32 scale, float32 zero_point) per\n'
        '      vector, in id order.\n'
        '\n'
        'Algorithm (must match demo_quantize_l1_vector / demo_quantize_residual in\n'
        'src/dsp_demo/retrieval_rpc_demo_support.c exactly):\n'
        '  Pass 1: stream all vectors once, track global (gmin, gmax) over every\n'
        '    element (matches demo_compute_range / the streaming Pass 1 scan).\n'
        '    l1_range = max(gmax - gmin, 1e-12); l1_levels = 2**l1_bits - 1\n'
        '    l1_scale = l1_range / l1_levels; l1_zero_point = gmin\n'
        "      (matches demo_set_qparams_from_range's clamp)\n"
        '  Pass 2, per vector (vectorized per chunk):\n'
        '    l1_inv_scale = 1 / l1_scale                       (computed once)\n'
        '    q_l1[i] = clip(round_half_away_from_zero((x[i] - l1_zero_point) * l1_inv_scale),\n'
        '                    0, l1_levels)\n'
        '    deq[i]  = q_l1[i] * l1_scale + l1_zero_point\n'
        '    residual[i] = x[i] - deq[i]                       (qc_compute_residual)\n'
        '    rmin, rmax = min/max over residual (per-vector, NOT global)\n'
        '    r_range = max(rmax - rmin, 1e-12); r_levels = 2**residual_bits - 1\n'
        '    r_scale = r_range / r_levels; r_zero_point = rmin\n'
        '    r_inv_scale = 1 / r_scale\n'
        '    q_res[i] = clip(round_half_away_from_zero((residual[i] - r_zero_point) * r_inv_scale),\n'
        '                     0, r_levels)\n'
        '  4-bit packing (both L1 and residual): byte b = code[2b] | (code[2b+1] << 4).\n'
        '\n'
        "All arithmetic above is done in float32, matching C's `float` throughout --\n"
        'NOT float64/Python float, which would silently diverge in the last bit and\n'
        'break the byte-identical parity requirement.\n'
    ),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-fvecs', required=True, type=Path)
    ap.add_argument('--l1-out', required=True, type=Path)
    ap.add_argument('--residual-out', required=True, type=Path)
    ap.add_argument('--qparams-out', required=True, type=Path)
    ap.add_argument('--l1-bits', type=int, choices=(4, 8), default=4)
    ap.add_argument('--residual-bits', type=int, choices=(4, 8), default=4)
    ap.add_argument('--chunk-vectors', type=int, default=16384)
    ap.add_argument('--num-vectors', type=int, default=None,
                    help='Optional override to encode only the first N vectors '
                         '(smoke testing on a subset).')
    args = ap.parse_args()

    if not args.base_fvecs.exists():
        print(f"error: {args.base_fvecs} does not exist", file=sys.stderr)
        return 1

    encode(args.base_fvecs, args.l1_out, args.residual_out, args.qparams_out,
          args.l1_bits, args.residual_bits, args.chunk_vectors, args.num_vectors)
    return 0


if __name__ == '__main__':
    sys.exit(main())
