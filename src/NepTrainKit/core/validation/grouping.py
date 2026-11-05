"""Explicit frame-level grouping for independent EOS scans."""

from __future__ import annotations

import math

import numpy as np


def _scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (str, int, float, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def eos_group_fields(frames) -> list[str]:
    """Offer metadata fields, without inferring that any field defines a scan."""
    excluded = {"energy", "free_energy", "energy_original", "virial", "virials", "stress"}
    return sorted({
        key for frame in frames for key, value in frame.info.items()
        if isinstance(key, str) and not key.startswith("_") and key not in excluded
        and _scalar(value) is not None
    })


def eos_frame_groups(frames, field: str) -> dict[str, list[int]]:
    """Partition every frame; missing or ambiguous values block the operation."""
    groups: dict[str, list[int]] = {}
    value_type = None
    for index, frame in enumerate(frames):
        value = _scalar(frame.info.get(field))
        if value is None:
            raise ValueError(f"Frame {index + 1}: grouping field '{field}' is missing or not a non-empty scalar.")
        if value_type is not None and type(value) is not value_type:
            raise ValueError(f"Grouping field '{field}' has inconsistent value types.")
        value_type = type(value)
        groups.setdefault(str(value), []).append(index)
    if not groups:
        raise ValueError("The file contains no structures.")
    return groups
