#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Selection helpers for descriptor/embedding downsampling.

Includes a NumPy implementation of pairwise distances and a farthest-point
sampling routine that can be warm-started from an existing selection.

Examples
--------
>>> import numpy as np
>>> pts = np.random.rand(10, 2).astype(np.float32)
>>> idx = farthest_point_sampling(pts, 3)
>>> len(idx) <= 3
True
"""

import numpy as np
import numpy.typing as npt

def numpy_cdist(X: npt.NDArray[np.float32], Y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Compute pairwise Euclidean distances using broadcasting.

    Parameters
    ----------
    X : numpy.ndarray
        Array of shape ``(m, d)``.
    Y : numpy.ndarray
        Array of shape ``(n, d)``.

    Returns
    -------
    numpy.ndarray
        Distance matrix of shape ``(m, n)`` where entry ``(i, j)`` is the
        Euclidean distance between ``X[i]`` and ``Y[j]``.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.zeros((2, 3), dtype=np.float32)
    >>> Y = np.ones((3, 3), dtype=np.float32)
    >>> numpy_cdist(X, Y).shape
    (2, 3)
    """
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    squared_dist = np.sum(np.square(diff), axis=2)
    return np.sqrt(squared_dist)


def farthest_point_sampling(points, n_samples, min_dist=0.1, selected_data=None) -> list[int]:
    """Greedy FPS with optional warm-start and minimum-distance constraint.

    Parameters
    ----------
    points : numpy.ndarray
        Input point set of shape ``(N, D)``.
    n_samples : int
        Maximum number of samples to select.
    min_dist : float, default=0.1
        Minimum allowed distance to any already selected point.
    selected_data : numpy.ndarray or None, optional
        Warm-start set with shape ``(M, D)``. If provided, selection respects
        the minimum distance from this set.

    Returns
    -------
    list[int]
        Indices of selected points.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.random.rand(100, 3).astype(np.float32)
    >>> idx = farthest_point_sampling(P, 5, min_dist=0.0)
    >>> len(idx) <= 5
    True
    """
    n_points = points.shape[0]

    if isinstance(selected_data, np.ndarray) and selected_data.size == 0:
        selected_data = None

    sampled_indices: list[int] = []

    if selected_data is not None:
        distances_to_samples = numpy_cdist(points, selected_data)
        min_distances = np.min(distances_to_samples, axis=1)

    else:
        first_index = 0
        sampled_indices.append(first_index)
        min_distances = np.linalg.norm(points - points[first_index], axis=1)

    while len(sampled_indices) < n_samples:
        current_index = int(np.argmax(min_distances))
        if min_distances[current_index] < float(min_dist):
            break
        sampled_indices.append(int(current_index))
        new_point = points[current_index]
        new_distances = np.linalg.norm(points - new_point, axis=1)
        min_distances = np.minimum(min_distances, new_distances)
    return sampled_indices