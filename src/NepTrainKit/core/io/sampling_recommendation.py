"""Shared fixed-budget and automatic multi-view coverage sampling results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .sampling_coverage import RESOLUTIONS, CoverageEvidence, select_coverage
from .sampling_features import SamplingFeatureBlocks
from .sampling_plan import PhysicsSamplingPlan, PhysicsSamplingStratum

SamplingPolicy = Literal["compact", "balanced", "conservative"]


@dataclass(frozen=True)
class StratumSamplingRecommendation:
    """Coverage-derived counts for one element-set/phase/magnetic stratum."""

    stratum: PhysicsSamplingStratum
    candidate_count: int
    existing_count: int
    compact_count: int | None
    recommended_count: int
    conservative_count: int | None
    achieved_coverage: float
    coverage_radius: float | None
    selected_indices: tuple[int, ...]
    reached_target: bool


@dataclass(frozen=True)
class PhysicsSamplingRecommendation:
    """One policy choice plus its compact/conservative uncertainty envelope."""

    policy: SamplingPolicy
    selected_indices: tuple[int, ...]
    recommended_count: int
    compact_count: int | None
    conservative_count: int | None
    groups: tuple[StratumSamplingRecommendation, ...]
    is_lower_bound: bool
    evidence: CoverageEvidence | None = None


@dataclass(frozen=True)
class StratumBudgetSelection:
    """Fixed-budget selection state for one physical stratum."""

    stratum: PhysicsSamplingStratum
    candidate_count: int
    existing_count: int
    selected_count: int
    achieved_coverage: float
    coverage_radius: float | None
    selected_indices: tuple[int, ...]


@dataclass(frozen=True)
class PhysicsBudgetSelection:
    """Representative structures selected under one global sample budget."""

    requested_count: int
    selected_indices: tuple[int, ...]
    groups: tuple[StratumBudgetSelection, ...]
    exhausted: bool
    evidence: CoverageEvidence | None = None


def _group_measure(selection, rows, resolution):
    values = np.maximum(selection.distances[list(rows), -2],
                        selection.distances[list(rows), -1])
    covered = (values <= resolution) & ~np.isin(rows, selection.evidence.uncovered_indices)
    return float(np.mean(covered)), float(np.max(values, initial=0.0))


def recommend_physics_sampling_from_blocks(
    candidate: SamplingFeatureBlocks, plan: PhysicsSamplingPlan, *,
    existing: SamplingFeatureBlocks | None = None,
    existing_plan: PhysicsSamplingPlan | None = None,
    policy: SamplingPolicy = "balanced",
    sample_cap_per_stratum: int | None = None,
    resolution: float | None = None,
) -> PhysicsSamplingRecommendation:
    """Recommend nested radius-cover subsets; no implicit per-stratum cap.

    The legacy explicit cap argument is interpreted as a total resource budget
    of cap * number of strata. Hitting it is reported as incomplete coverage.
    Counts are geometric recommendations, never training-accuracy guarantees.
    An explicit resolution stops at that radius without computing a tighter
    envelope; uncomputed envelope counts are None.
    """
    if sample_cap_per_stratum is not None and int(sample_cap_per_stratum) <= 0:
        raise ValueError("sample_cap_per_stratum must be positive")
    selection = select_coverage(
        candidate, plan, existing=existing, existing_plan=existing_plan, policy=policy,
        resolution=resolution,
        resource_limit=None if sample_cap_per_stratum is None else
        int(sample_cap_per_stratum) * plan.group_count,
    )
    warm = {} if existing_plan is None else existing_plan.group_indices()
    count = len(selection.indices)
    fallback = len(selection.trajectory) if resolution is None else None
    compact_count = selection.counts.get("compact", fallback)
    conservative_count = selection.counts.get("conservative", fallback)
    compact = set(selection.trajectory[:compact_count]) if compact_count is not None else set()
    conservative = set(selection.trajectory[:conservative_count]) if conservative_count is not None else set()
    selected = set(selection.indices)
    groups = []
    for key, rows in sorted(plan.groups):
        local = tuple(sorted(selected.intersection(rows)))
        coverage, radius = _group_measure(selection, rows, selection.evidence.resolution)
        reached = coverage == 1.0 and (bool(local) or key in warm)
        groups.append(StratumSamplingRecommendation(
            key, len(rows), len(warm.get(key, ())),
            None if compact_count is None else len(compact.intersection(rows)), len(local),
            None if conservative_count is None else len(conservative.intersection(rows)),
            coverage, radius, local, reached,
        ))
    return PhysicsSamplingRecommendation(
        policy, tuple(sorted(selected)), count, compact_count, conservative_count,
        tuple(groups), any(not g.reached_target for g in groups), selection.evidence,
    )


def select_physics_budget(
    candidate: SamplingFeatureBlocks, plan: PhysicsSamplingPlan, *, n_samples: int,
    existing: SamplingFeatureBlocks | None = None,
    existing_plan: PhysicsSamplingPlan | None = None,
    min_distance: float = 0.0, coverage_policy: SamplingPolicy = "balanced",
) -> PhysicsBudgetSelection:
    """Spend one budget on the same trajectory used by automatic coverage."""
    selection = select_coverage(
        candidate, plan, existing=existing, existing_plan=existing_plan,
        budget=n_samples, min_distance=min_distance, policy=coverage_policy,
    )
    warm = {} if existing_plan is None else existing_plan.group_indices()
    selected = set(selection.indices)
    groups = []
    for key, rows in sorted(plan.groups):
        local = tuple(sorted(selected.intersection(rows)))
        coverage, radius = _group_measure(selection, rows, RESOLUTIONS[coverage_policy])
        groups.append(StratumBudgetSelection(
            key, len(rows), len(warm.get(key, ())), len(local), coverage, radius, local,
        ))
    return PhysicsBudgetSelection(
        int(n_samples), tuple(sorted(selected)), tuple(groups),
        len(selected) < min(int(n_samples), candidate.row_count), selection.evidence,
    )


def recommend_physics_sampling(
    descriptors: np.ndarray, plan: PhysicsSamplingPlan, *,
    existing_descriptors: np.ndarray | None = None,
    existing_plan: PhysicsSamplingPlan | None = None,
    policy: SamplingPolicy = "balanced", sample_cap_per_stratum: int | None = None,
) -> PhysicsSamplingRecommendation:
    """Descriptor-only entry point with the same radius-based coverage contract."""
    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("descriptors must be a finite two-dimensional array")
    existing = None if existing_descriptors is None else SamplingFeatureBlocks(
        ("descriptor_mean",), (np.asarray(existing_descriptors, dtype=np.float64),))
    return recommend_physics_sampling_from_blocks(
        SamplingFeatureBlocks(("descriptor_mean",), (values,)), plan,
        existing=existing, existing_plan=existing_plan, policy=policy,
        sample_cap_per_stratum=sample_cap_per_stratum,
    )
