#!/usr/bin/env python3

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LAYOUT_FINGERPRINT_DOMAIN = b"pure-pq-gorder-layout-v1\0"
ORDER_MAP_DIRECTION = "new_id_to_old_id"


@dataclass(frozen=True)
class VectorLayout:
    layout_id: str
    new_to_old: np.ndarray | None
    order_map_fingerprint: bytes | None
    fingerprint: bytes
    order_map_path: Path | None

    @classmethod
    def load(cls, layout_id: str, order_map_path: Path | None,
             num_vectors: int) -> "VectorLayout":
        if not layout_id:
            raise ValueError("layout ID must not be empty")
        if num_vectors <= 0:
            raise ValueError("vector count must be positive")
        if order_map_path is None:
            return cls(
                layout_id=layout_id,
                new_to_old=None,
                order_map_fingerprint=None,
                fingerprint=hashlib.sha256(layout_id.encode("utf-8")).digest(),
                order_map_path=None,
            )
        path = order_map_path.resolve()
        parsed = np.fromfile(path, dtype=np.int64, sep=" ")
        if parsed.size != num_vectors:
            raise ValueError(
                f"order map has {parsed.size} IDs, expected {num_vectors}")
        if np.any(parsed < 0) or np.any(parsed >= num_vectors):
            raise ValueError("order map contains an out-of-range old ID")
        new_to_old = parsed.astype("<u4", copy=True)
        del parsed
        seen = np.zeros(num_vectors, dtype=np.bool_)
        seen[new_to_old] = True
        if not np.all(seen):
            raise ValueError("order map is not a permutation")
        order_map_fingerprint = hashlib.sha256(new_to_old).digest()
        digest = hashlib.sha256()
        digest.update(LAYOUT_FINGERPRINT_DOMAIN)
        digest.update(layout_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(order_map_fingerprint)
        return cls(
            layout_id=layout_id,
            new_to_old=new_to_old,
            order_map_fingerprint=order_map_fingerprint,
            fingerprint=digest.digest(),
            order_map_path=path,
        )

    def gather(self, vectors: np.ndarray, start: int, end: int) -> np.ndarray:
        if start < 0 or end < start or end > vectors.shape[0]:
            raise ValueError("invalid layout batch range")
        rows = (vectors[start:end] if self.new_to_old is None else
                vectors[self.new_to_old[start:end]])
        return np.asarray(rows, dtype=np.float32)

    def manifest_fields(self) -> dict:
        return {
            "layout_id": self.layout_id,
            "layout_fingerprint": self.fingerprint.hex(),
            "order_map_direction": (
                ORDER_MAP_DIRECTION if self.new_to_old is not None else None),
            "order_map_fingerprint": (
                self.order_map_fingerprint.hex()
                if self.order_map_fingerprint is not None else None),
            "order_map_path": (
                str(self.order_map_path)
                if self.order_map_path is not None else None),
        }

    def verify_manifest(self, manifest: dict) -> None:
        expected = self.manifest_fields()
        for key in ("layout_id", "layout_fingerprint",
                    "order_map_direction", "order_map_fingerprint"):
            if manifest.get(key) != expected[key]:
                raise ValueError(
                    f"PQ manifest {key}={manifest.get(key)!r}, "
                    f"expected {expected[key]!r}")
