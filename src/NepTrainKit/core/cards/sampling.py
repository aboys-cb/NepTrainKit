"""Shared deterministic sampling helpers for Make Dataset operations."""

from __future__ import annotations

import hashlib

import numpy as np


def structure_seed_id(structure) -> int:
    """Return a stable content-derived id for per-structure random sampling."""
    digest = hashlib.blake2b(digest_size=8)
    for values in (
        np.asarray(structure.numbers, dtype=np.int64),
        np.asarray(structure.positions, dtype=np.float64),
        np.asarray(structure.cell.array, dtype=np.float64),
        np.asarray(structure.pbc, dtype=np.uint8),
    ):
        digest.update(np.ascontiguousarray(values).tobytes())
    if "group" in structure.arrays:
        digest.update(
            "\0".join(str(value) for value in structure.arrays["group"]).encode("utf-8")
        )
    digest.update(str(structure.info.get("Config_type", "") or "").encode("utf-8"))
    return int.from_bytes(digest.digest(), "big", signed=False)


def derived_structure_seed(seed: int, structure) -> int:
    """Derive a NumPy-compatible seed from a base seed and structure content."""
    return int(
        np.random.SeedSequence(
            [int(seed), structure_seed_id(structure)]
        ).generate_state(1, dtype=np.uint32)[0]
    )
