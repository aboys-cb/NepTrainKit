from __future__ import annotations

import numpy as np
import pytest

import NepTrainKit.core.io.sampling_recommendation as recommendation_module
from NepTrainKit.core.io.sampling_features import SamplingFeatureBlocks
from NepTrainKit.core.io.sampling_plan import (
    PhysicsSamplingPlan,
    PhysicsSamplingStratum,
)
from NepTrainKit.core.io.sampling_recommendation import (
    recommend_physics_sampling,
    recommend_physics_sampling_from_blocks,
    select_physics_budget,
)


def _stratum(name: str) -> PhysicsSamplingStratum:
    return PhysicsSamplingStratum(
        element_set=("Fe",),
        phase="bcc",
        magnetic_order=name,
    )


def _plan(
    groups: tuple[tuple[PhysicsSamplingStratum, tuple[int, ...]], ...],
) -> PhysicsSamplingPlan:
    row_count = sum(len(indices) for _stratum_key, indices in groups)
    return PhysicsSamplingPlan(
        groups=groups,
        spin_model=True,
        missing_spin_indices=(),
        phase_counts=(("bcc", row_count),),
        magnetic_order_counts=tuple(
            (key.magnetic_order, len(indices)) for key, indices in groups
        ),
        source_indices=tuple(range(row_count)),
    )


def test_recommendation_gives_broad_noncollinear_stratum_more_samples():
    fm = _stratum("fm")
    noncollinear = _stratum("noncollinear")
    duplicate_points = np.zeros((8, 2), dtype=np.float32)
    broad_points = np.column_stack(
        (
            np.linspace(-1.0, 1.0, 32, dtype=np.float32),
            np.linspace(-1.0, 1.0, 32, dtype=np.float32) ** 2,
        )
    )
    descriptors = np.vstack((duplicate_points, broad_points))
    plan = _plan(
        (
            (fm, tuple(range(8))),
            (noncollinear, tuple(range(8, 40))),
        )
    )

    result = recommend_physics_sampling(descriptors, plan)
    by_order = {group.stratum.magnetic_order: group for group in result.groups}

    assert by_order["fm"].recommended_count == 1
    assert by_order["noncollinear"].recommended_count > 1
    assert result.recommended_count == sum(
        group.recommended_count for group in result.groups
    )
    assert not result.is_lower_bound


def test_recommendation_reports_compact_and_conservative_envelope():
    noncollinear = _stratum("noncollinear")
    x = np.linspace(-2.0, 2.0, 64, dtype=np.float32)
    descriptors = np.column_stack((x, x**2, x**3))
    plan = _plan(((noncollinear, tuple(range(len(descriptors)))),))

    result = recommend_physics_sampling(descriptors, plan, policy="balanced")

    assert result.compact_count <= result.recommended_count
    assert result.recommended_count <= result.conservative_count
    group = result.groups[0]
    assert group.compact_count <= group.recommended_count
    assert group.recommended_count <= group.conservative_count
    assert group.achieved_coverage >= 0.99


def test_existing_same_stratum_can_reduce_incremental_recommendation_to_zero():
    noncollinear = _stratum("noncollinear")
    descriptors = np.asarray(
        [[-1.0], [-0.5], [0.0], [0.5], [1.0]], dtype=np.float32
    )
    plan = _plan(((noncollinear, tuple(range(len(descriptors)))),))

    result = recommend_physics_sampling(
        descriptors,
        plan,
        existing_descriptors=descriptors.copy(),
        existing_plan=plan,
    )

    assert result.recommended_count == 0
    assert result.selected_indices == ()
    assert result.groups[0].existing_count == len(descriptors)
    assert result.groups[0].achieved_coverage == pytest.approx(1.0)


def test_recommendation_marks_unsaturated_safety_cap_as_lower_bound():
    noncollinear = _stratum("noncollinear")
    descriptors = np.column_stack(
        (
            np.arange(40, dtype=np.float32),
            np.arange(40, dtype=np.float32) ** 2,
        )
    )
    plan = _plan(((noncollinear, tuple(range(len(descriptors)))),))

    result = recommend_physics_sampling(
        descriptors,
        plan,
        sample_cap_per_stratum=1,
    )

    assert result.recommended_count == 1
    assert result.is_lower_bound
    assert not result.groups[0].reached_target


def test_native_and_numpy_recommendations_select_same_prefix(monkeypatch):
    noncollinear = _stratum("noncollinear")
    rng = np.random.default_rng(7)
    descriptors = rng.normal(size=(80, 6)).astype(np.float32)
    plan = _plan(((noncollinear, tuple(range(len(descriptors)))),))

    native = recommend_physics_sampling(descriptors, plan)
    monkeypatch.setattr(recommendation_module, "_native_sampling", None)
    reference = recommend_physics_sampling(descriptors, plan)

    assert native.selected_indices == reference.selected_indices
    assert native.recommended_count == reference.recommended_count
    assert native.compact_count == reference.compact_count
    assert native.conservative_count == reference.conservative_count
    assert native.is_lower_bound == reference.is_lower_bound
    assert native.groups[0].achieved_coverage == pytest.approx(
        reference.groups[0].achieved_coverage,
        rel=1.0e-6,
        abs=1.0e-7,
    )


def test_recommendation_rejects_incomplete_plan():
    fm = _stratum("fm")
    plan = _plan(((fm, (0, 1)),))

    with pytest.raises(ValueError, match="cover each descriptor row"):
        recommend_physics_sampling(
            np.zeros((3, 2), dtype=np.float32),
            plan,
        )


def test_fixed_budget_follows_layer_coverage_instead_of_equal_quotas():
    fm = _stratum("fm")
    noncollinear = _stratum("noncollinear")
    duplicate_points = np.zeros((12, 2), dtype=np.float32)
    x = np.linspace(-2.0, 2.0, 32, dtype=np.float32)
    broad_points = np.column_stack((x, x**2))
    points = np.vstack((duplicate_points, broad_points))
    plan = _plan(
        (
            (fm, tuple(range(12))),
            (noncollinear, tuple(range(12, 44))),
        )
    )

    result = select_physics_budget(
        SamplingFeatureBlocks(("descriptor_mean",), (points,)),
        plan,
        n_samples=6,
    )
    by_order = {group.stratum.magnetic_order: group for group in result.groups}

    assert by_order["fm"].selected_count == 1
    assert by_order["noncollinear"].selected_count == 5
    assert len(result.selected_indices) == 6
    assert not result.exhausted


def test_fixed_budget_protects_balanced_coverage_floor_when_budget_allows():
    fm = _stratum("fm")
    noncollinear = _stratum("noncollinear")
    duplicate_points = np.zeros((12, 2), dtype=np.float32)
    x = np.linspace(-2.0, 2.0, 32, dtype=np.float32)
    points = np.vstack((duplicate_points, np.column_stack((x, x**2))))
    plan = _plan(
        (
            (fm, tuple(range(12))),
            (noncollinear, tuple(range(12, 44))),
        )
    )

    result = select_physics_budget(
        SamplingFeatureBlocks(("descriptor_mean",), (points,)),
        plan,
        n_samples=20,
    )

    assert min(group.achieved_coverage for group in result.groups) >= 0.99


def test_layer_local_scaling_makes_equivalent_manifolds_compete_equally():
    fm = _stratum("fm")
    noncollinear = _stratum("noncollinear")
    base = np.asarray([[0.0], [1.0], [3.0], [8.0]], dtype=np.float32)
    points = np.vstack((base, base * 1.0e6))
    plan = _plan(
        (
            (fm, (0, 1, 2, 3)),
            (noncollinear, (4, 5, 6, 7)),
        )
    )

    result = select_physics_budget(
        SamplingFeatureBlocks(("descriptor_mean",), (points,)),
        plan,
        n_samples=4,
    )

    assert [group.selected_count for group in result.groups] == [2, 2]


def test_block_recommendation_uses_same_layer_local_geometry_as_budget_selection():
    fm = _stratum("fm")
    noncollinear = _stratum("noncollinear")
    duplicate_points = np.zeros((6, 1), dtype=np.float32)
    broad_points = np.arange(20, dtype=np.float32).reshape(-1, 1)
    blocks = SamplingFeatureBlocks(
        ("descriptor_mean",),
        (np.vstack((duplicate_points, broad_points)),),
    )
    plan = _plan(
        (
            (fm, tuple(range(6))),
            (noncollinear, tuple(range(6, 26))),
        )
    )

    result = recommend_physics_sampling_from_blocks(blocks, plan)
    by_order = {group.stratum.magnetic_order: group for group in result.groups}

    assert by_order["fm"].recommended_count == 1
    assert by_order["noncollinear"].recommended_count > 1


def test_native_and_numpy_fixed_budget_select_same_rows(monkeypatch):
    fm = _stratum("fm")
    noncollinear = _stratum("noncollinear")
    rng = np.random.default_rng(23)
    points = rng.normal(size=(64, 5)).astype(np.float32)
    blocks = SamplingFeatureBlocks(("descriptor_mean",), (points,))
    plan = _plan(
        (
            (fm, tuple(range(24))),
            (noncollinear, tuple(range(24, 64))),
        )
    )

    native = select_physics_budget(blocks, plan, n_samples=24)
    monkeypatch.setattr(recommendation_module, "_native_sampling", None)
    reference = select_physics_budget(blocks, plan, n_samples=24)

    assert native.selected_indices == reference.selected_indices
    assert [group.selected_count for group in native.groups] == [
        group.selected_count for group in reference.groups
    ]
