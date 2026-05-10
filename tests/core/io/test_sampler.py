from __future__ import annotations

import numpy as np

from NepTrainKit.core.io.sampler import farthest_point_sampling, incremental_fps_with_r2, pca


def test_farthest_point_sampling_is_deterministic_and_unique():
    points = np.array([[0.0], [1.0], [2.0], [4.0]], dtype=np.float32)

    assert farthest_point_sampling(points, 3, min_dist=0.0) == [0, 3, 2]
    assert farthest_point_sampling(points, 10, min_dist=0.0) == [0, 3, 2, 1]


def test_farthest_point_sampling_min_dist_stops_on_identical_points():
    points = np.zeros((4, 3), dtype=np.float32)

    assert farthest_point_sampling(points, 4, min_dist=0.1) == [0]
    assert farthest_point_sampling(points, 4, min_dist=0.0) == [0, 1, 2, 3]


def test_farthest_point_sampling_respects_warm_start_distance():
    points = np.array([[0.0], [0.2], [2.0]], dtype=np.float32)
    selected = np.array([[0.0]], dtype=np.float32)

    assert farthest_point_sampling(points, 2, min_dist=0.5, selected_data=selected) == [2]


def test_incremental_fps_with_r2_returns_unique_indices():
    points = np.array([[0.0], [1.0], [2.0], [4.0]], dtype=np.float32)

    indices, r2 = incremental_fps_with_r2(points, r2_threshold=2.0, n_samples=10, min_dist=0.0)

    assert indices == [0, 3, 2, 1]
    assert len(indices) == len(set(indices))
    assert r2 > 0.0


def test_pca_rejects_non_matrix_input():
    with np.testing.assert_raises(ValueError):
        pca(np.arange(3, dtype=np.float32))
