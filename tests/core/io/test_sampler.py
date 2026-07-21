from __future__ import annotations

import numpy as np
from ase import Atoms

from types import SimpleNamespace

from NepTrainKit.core.io.sampler import (
    SparseSampler,
    allocate_sqrt_quotas,
    centered_fps,
    farthest_point_sampling,
    incremental_fps_with_r2,
    pca,
)


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


def test_sqrt_quotas_reserve_groups_and_use_largest_remainders():
    quotas = allocate_sqrt_quotas({("H",): 4, ("He",): 2}, n_samples=4)

    assert quotas == {("H",): 2, ("He",): 2}


def test_centered_fps_uses_center_or_matching_warm_start():
    points = np.asarray([[0.0], [4.0], [5.0], [6.0], [10.0]])

    assert centered_fps(points, n_samples=1, min_dist=0.0) == [2]
    assert centered_fps(
        points,
        n_samples=1,
        min_dist=0.0,
        selected_data=np.asarray([[0.0]]),
    ) == [4]


def test_sparse_sampler_element_set_strategy_covers_each_group():
    raw_points = np.asarray([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0]], dtype=np.float32)
    structures = np.asarray(
        [
            SimpleNamespace(elements=["H"]),
            SimpleNamespace(elements=["H"]),
            SimpleNamespace(elements=["H"]),
            SimpleNamespace(elements=["H"]),
            SimpleNamespace(elements=["He"]),
            SimpleNamespace(elements=["He"]),
        ],
        dtype=object,
    )
    indices = np.arange(len(structures), dtype=np.int64)
    descriptor = SimpleNamespace(
        now_data=np.column_stack((raw_points[:, 0], np.zeros(len(raw_points)))).astype(np.float32),
        group_array=SimpleNamespace(now_data=indices),
        data=SimpleNamespace(now_indices=indices),
    )
    result = SimpleNamespace(
        descriptor=descriptor,
        _descriptor_raw_all=raw_points,
        structure=SimpleNamespace(all_data=structures),
        select_index=set(),
    )

    selected, reverse = SparseSampler(result).sparse_point_selection(
        n_samples=4,
        distance=0.0,
        descriptor_source="raw",
        selection_strategy="element_set",
    )

    assert reverse is False
    assert len(selected) == 4
    assert len([index for index in selected if index < 4]) == 2
    assert len([index for index in selected if index >= 4]) == 2
    assert sum(group["selected_count"] for group in result._last_sparse_group_report.values()) == 4


def test_sparse_sampler_balanced_warm_start_matches_element_sets(monkeypatch):
    raw_points = np.asarray([[0.0], [3.0], [10.0], [100.0], [110.0]], dtype=np.float32)
    structures = [Atoms("H"), Atoms("H"), Atoms("H"), Atoms("He"), Atoms("He")]
    indices = np.arange(len(structures), dtype=np.int64)
    descriptor = SimpleNamespace(
        now_data=np.column_stack((raw_points[:, 0], np.zeros(len(raw_points)))).astype(np.float32),
        group_array=SimpleNamespace(now_data=indices),
        data=SimpleNamespace(now_indices=indices),
    )
    result = SimpleNamespace(
        descriptor=descriptor,
        _descriptor_raw_all=raw_points,
        structure=SimpleNamespace(all_data=structures),
        select_index=set(),
    )
    monkeypatch.setattr(
        "NepTrainKit.core.io.sampler.import_structures",
        lambda _path: [Atoms("H"), Atoms("He")],
    )
    monkeypatch.setattr(
        "NepTrainKit.core.io.sampler.read_nep_out_file",
        lambda *_args, **_kwargs: np.asarray([[0.0], [100.0]], dtype=np.float32),
    )

    selected, _ = SparseSampler(result).sparse_point_selection(
        n_samples=2,
        distance=0.0,
        descriptor_source="raw",
        training_path="train.xyz",
        selection_strategy="element_set",
    )

    assert selected == [2, 4]
    assert all(
        group["existing_count"] == 1
        for group in result._last_sparse_group_report.values()
    )


def test_incremental_fps_with_r2_returns_unique_indices():
    points = np.array([[0.0], [1.0], [2.0], [4.0]], dtype=np.float32)

    indices, r2 = incremental_fps_with_r2(points, r2_threshold=2.0, n_samples=10, min_dist=0.0)

    assert indices == [0, 3, 2, 1]
    assert len(indices) == len(set(indices))
    assert r2 > 0.0


def test_pca_rejects_non_matrix_input():
    with np.testing.assert_raises(ValueError):
        pca(np.arange(3, dtype=np.float32))
