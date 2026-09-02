from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

import NepTrainKit.core.io.sampler as sampler_module
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


def test_farthest_point_sampling_numpy_fallback_preserves_contract(monkeypatch):
    points = np.asarray([[0.0], [1.0], [2.0], [4.0]], dtype=np.float32)
    monkeypatch.setattr(sampler_module, "_native_sampling", None)

    assert farthest_point_sampling(points, 3, min_dist=0.0) == [0, 3, 2]


@pytest.mark.skipif(
    sampler_module._native_sampling is None,
    reason="native sampling helper has not been built",
)
def test_native_fps_matches_numpy_reference_with_warm_start():
    rng = np.random.default_rng(20260901)
    points = rng.normal(size=(257, 7)).astype(np.float32)
    selected = rng.normal(size=(3, 7)).astype(np.float32)

    native = farthest_point_sampling(
        points,
        80,
        min_dist=0.0,
        selected_data=selected,
    )
    reference = sampler_module._farthest_point_sampling_numpy(
        points,
        80,
        min_dist=0.0,
        selected_data=selected,
    )

    assert native == reference


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


def test_sparse_sampler_physics_strategy_preserves_phase_and_spin_strata(
    monkeypatch,
):
    from ase.build import bulk

    bcc = bulk("Fe", "bcc", a=2.86, cubic=True)
    bcc.arrays["spin"] = np.tile([0.0, 0.0, 2.2], (len(bcc), 1))
    fcc = bulk("Fe", "fcc", a=3.55, cubic=True)
    fcc.arrays["spin"] = np.tile([0.0, 0.0, 2.0], (len(fcc), 1))
    structures = np.empty(4, dtype=object)
    structures[:] = [bcc, bcc.copy(), fcc, fcc.copy()]
    indices = np.arange(len(structures), dtype=np.int64)
    raw_points = np.asarray([[0.0], [0.1], [10.0], [10.1]], dtype=np.float32)
    descriptor = SimpleNamespace(
        now_data=np.column_stack((raw_points[:, 0], np.zeros(len(raw_points)))).astype(
            np.float32
        ),
        group_array=SimpleNamespace(now_data=indices),
        data=SimpleNamespace(now_indices=indices),
    )
    result = SimpleNamespace(
        descriptor=descriptor,
        _descriptor_raw_all=raw_points,
        structure=SimpleNamespace(all_data=structures),
        select_index=set(),
        is_spin_model=True,
    )
    real_builder = sampler_module.build_physics_sampling_plan
    build_calls = []

    def counted_builder(*args, **kwargs):
        build_calls.append(1)
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(
        sampler_module,
        "build_physics_sampling_plan",
        counted_builder,
    )
    sampler = SparseSampler(result)

    selected, reverse = sampler.sparse_point_selection(
        n_samples=2,
        distance=0.0,
        descriptor_source="reduced",
        sampling_mode="r2",
        selection_strategy="physics",
    )
    selected_again, _ = sampler.sparse_point_selection(
        n_samples=2,
        distance=0.0,
        descriptor_source="reduced",
        sampling_mode="r2",
        selection_strategy="physics",
    )

    assert reverse is False
    assert len(selected) == 2
    assert selected_again == selected
    assert len(build_calls) == 1
    assert result._last_sparse_physics_plan.group_count == 2
    assert {key.phase for key in result._last_sparse_group_report} == {"bcc", "fcc"}
    assert all(
        group["selected_count"] == 1
        for group in result._last_sparse_group_report.values()
    )


def test_sparse_sampler_recommends_physics_count_without_selecting(monkeypatch):
    from ase.build import bulk

    bcc = bulk("Fe", "bcc", a=2.86, cubic=True)
    bcc.arrays["spin"] = np.tile([0.0, 0.0, 2.2], (len(bcc), 1))
    fcc = bulk("Fe", "fcc", a=3.55, cubic=True)
    fcc.arrays["spin"] = np.tile([0.0, 0.0, 2.0], (len(fcc), 1))
    structures = np.empty(6, dtype=object)
    structures[:] = [bcc, bcc.copy(), bcc.copy(), fcc, fcc.copy(), fcc.copy()]
    indices = np.arange(len(structures), dtype=np.int64)
    raw_points = np.asarray(
        [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]],
        dtype=np.float32,
    )
    descriptor = SimpleNamespace(
        now_data=np.column_stack((raw_points[:, 0], np.zeros(len(raw_points)))),
        group_array=SimpleNamespace(now_data=indices),
        data=SimpleNamespace(now_indices=indices),
    )
    result = SimpleNamespace(
        descriptor=descriptor,
        _descriptor_raw_all=raw_points,
        structure=SimpleNamespace(all_data=structures),
        select_index=set(),
        is_spin_model=True,
    )
    sampler = SparseSampler(result)

    recommendation = sampler.recommend_physics_sample_count()
    selected, reverse = sampler.sparse_point_selection(
        n_samples=0,
        distance=0.0,
        selection_strategy="physics",
        physics_count_mode="automatic",
    )

    assert recommendation.recommended_count > 0
    assert {group.stratum.phase for group in recommendation.groups} == {
        "bcc",
        "fcc",
    }
    assert reverse is False
    assert len(selected) == recommendation.recommended_count
    assert result._last_sparse_physics_recommendation is not None
    assert (
        result._last_sparse_physics_recommendation.recommended_count
        == recommendation.recommended_count
    )
    assert result.select_index == set()


def test_sparse_sampler_physics_warm_start_matches_full_physical_stratum(
    monkeypatch,
):
    from ase.build import bulk

    bcc = bulk("Fe", "bcc", a=2.86, cubic=True)
    bcc.arrays["spin"] = np.tile([0.0, 0.0, 2.2], (len(bcc), 1))
    fcc = bulk("Fe", "fcc", a=3.55, cubic=True)
    fcc.arrays["spin"] = np.tile([0.0, 0.0, 2.0], (len(fcc), 1))
    structures = np.empty(4, dtype=object)
    structures[:] = [bcc, bcc.copy(), fcc, fcc.copy()]
    indices = np.arange(len(structures), dtype=np.int64)
    raw_points = np.asarray([[0.0], [5.0], [100.0], [105.0]], dtype=np.float32)
    descriptor = SimpleNamespace(
        now_data=np.column_stack((raw_points[:, 0], np.zeros(len(raw_points)))).astype(
            np.float32
        ),
        group_array=SimpleNamespace(now_data=indices),
        data=SimpleNamespace(now_indices=indices),
    )
    result = SimpleNamespace(
        descriptor=descriptor,
        _descriptor_raw_all=raw_points,
        structure=SimpleNamespace(all_data=structures),
        select_index=set(),
        is_spin_model=True,
    )
    monkeypatch.setattr(
        "NepTrainKit.core.io.sampler.import_structures",
        lambda _path: [bcc.copy()],
    )
    monkeypatch.setattr(
        "NepTrainKit.core.io.sampler.read_nep_out_file",
        lambda *_args, **_kwargs: np.asarray([[0.5]], dtype=np.float32),
    )

    selected, _ = SparseSampler(result).sparse_point_selection(
        n_samples=2,
        distance=0.0,
        training_path="train.xyz",
        selection_strategy="physics",
    )

    assert selected == [2, 3]
    existing_by_phase = {
        key.phase: report["existing_count"]
        for key, report in result._last_sparse_group_report.items()
    }
    selected_by_phase = {
        key.phase: report["selected_count"]
        for key, report in result._last_sparse_group_report.items()
    }
    assert existing_by_phase == {"bcc": 1, "fcc": 0}
    assert selected_by_phase == {"bcc": 0, "fcc": 2}


def test_sparse_sampler_physics_strategy_rejects_missing_spin(monkeypatch):
    from ase.build import bulk

    structure = bulk("Fe", "bcc", a=2.86, cubic=True)
    indices = np.asarray([0], dtype=np.int64)
    descriptor = SimpleNamespace(
        now_data=np.asarray([[0.0, 0.0]], dtype=np.float32),
        group_array=SimpleNamespace(now_data=indices),
        data=SimpleNamespace(now_indices=indices),
    )
    structures = np.empty(1, dtype=object)
    structures[0] = structure
    result = SimpleNamespace(
        descriptor=descriptor,
        _descriptor_raw_all=np.asarray([[0.0]], dtype=np.float32),
        structure=SimpleNamespace(all_data=structures),
        select_index=set(),
        is_spin_model=True,
    )
    messages = []
    monkeypatch.setattr(
        "NepTrainKit.core.io.sampler.MessageManager.send_message_box",
        lambda message, _title=None: messages.append(str(message)),
    )

    selected, _ = SparseSampler(result).sparse_point_selection(
        n_samples=1,
        distance=0.0,
        selection_strategy="physics",
    )

    assert selected == []
    assert messages
    assert "canonical spin:R:3" in messages[-1]


def test_incremental_fps_with_r2_returns_unique_indices():
    points = np.array([[0.0], [1.0], [2.0], [4.0]], dtype=np.float32)

    indices, r2 = incremental_fps_with_r2(points, r2_threshold=2.0, n_samples=10, min_dist=0.0)

    assert indices == [0, 3, 2, 1]
    assert len(indices) == len(set(indices))
    assert r2 > 0.0


def test_incremental_fps_r2_measures_nearest_center_residual_coverage():
    points = np.array([[0.0], [1.0], [2.0], [4.0]], dtype=np.float32)

    indices, coverage_r2 = incremental_fps_with_r2(
        points,
        r2_threshold=0.8,
        n_samples=4,
        min_dist=0.0,
    )

    assert indices == [0, 3, 2]
    np.testing.assert_allclose(coverage_r2, 1.0 - 1.0 / 8.75)


def test_pca_rejects_non_matrix_input():
    with np.testing.assert_raises(ValueError):
        pca(np.arange(3, dtype=np.float32))
