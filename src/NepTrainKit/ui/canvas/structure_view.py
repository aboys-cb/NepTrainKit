"""Shared camera framing helpers for structure canvases."""

from __future__ import annotations

from itertools import product

import numpy as np

from NepTrainKit.core.geometry_cache import structure_cell_array


def structure_view_state(structure) -> tuple[np.ndarray, float, int, int]:
    """Return center, distance, elevation, and azimuth for a structure."""
    positions = np.asarray(structure.positions, dtype=np.float64).reshape(-1, 3)
    bounds = positions

    cell = structure_cell_array(structure, dtype=np.float64).reshape(3, 3)
    if np.all(np.isfinite(cell)) and np.any(np.linalg.norm(cell, axis=1) > 1e-12):
        corners = np.asarray(list(product((0.0, 1.0), repeat=3))) @ cell
        bounds = np.vstack((bounds, corners)) if bounds.size else corners

    if not bounds.size or not np.all(np.isfinite(bounds)):
        return np.zeros(3, dtype=np.float64), 2.8, 30, 45

    minimum = bounds.min(axis=0)
    maximum = bounds.max(axis=0)
    center = (minimum + maximum) / 2.0
    size = maximum - minimum
    max_dimension = max(float(np.max(size)), 1.0)
    distance = max_dimension / (2.0 * np.tan(np.radians(30.0))) * 2.8
    aspect_ratio = size / max_dimension

    flat_threshold = 0.5
    if (
        aspect_ratio[2] < flat_threshold
        and aspect_ratio[0] >= flat_threshold
        and aspect_ratio[1] >= flat_threshold
    ):
        elevation, azimuth = 90, 0
    elif (
        aspect_ratio[0] < flat_threshold
        and aspect_ratio[1] >= flat_threshold
        and aspect_ratio[2] >= flat_threshold
    ) or (
        aspect_ratio[1] < flat_threshold
        and aspect_ratio[0] >= flat_threshold
        and aspect_ratio[2] >= flat_threshold
    ):
        elevation, azimuth = 0, 0
    else:
        elevation, azimuth = 30, 45

    return center, distance, elevation, azimuth
