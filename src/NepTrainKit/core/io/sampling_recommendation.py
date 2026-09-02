"""Data-driven sample-count recommendations for physics-aware FPS.

The public interface deliberately returns a recommendation rather than silently
changing the existing fixed-count workflow.  It estimates descriptor coverage;
it does not claim that a training or downstream-property optimum has been found.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Literal

import numpy as np

from .sampling_features import SamplingFeatureBlocks, representative_sampling_features
from .sampling_plan import PhysicsSamplingPlan, PhysicsSamplingStratum

try:
    from NepTrainKit._native import _sampling as _native_sampling
except ImportError:  # pragma: no cover - source checkout/reference fallback
    _native_sampling = None


SamplingPolicy = Literal["compact", "balanced", "conservative"]

_POLICY_COVERAGE = {
    "compact": 0.98,
    "balanced": 0.99,
    "conservative": 0.995,
}
_DEFAULT_SAMPLE_CAP_PER_STRATUM = 2048


@dataclass(frozen=True)
class StratumSamplingRecommendation:
    """Coverage-derived counts for one element-set/phase/magnetic stratum."""

    stratum: PhysicsSamplingStratum
    candidate_count: int
    existing_count: int
    compact_count: int
    recommended_count: int
    conservative_count: int
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
    compact_count: int
    conservative_count: int
    groups: tuple[StratumSamplingRecommendation, ...]
    is_lower_bound: bool


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


@dataclass(frozen=True)
class _CoverageTrace:
    indices: tuple[int, ...]
    coverage: tuple[float, ...]
    radii: tuple[float, ...]
    initial_coverage: float


def _coverage_value(
    nearest_squared: np.ndarray,
    selected_mask: np.ndarray,
    total_variance: float,
) -> tuple[float, float]:
    remaining = nearest_squared[~selected_mask]
    if remaining.size:
        remaining = np.maximum(remaining, 0.0)
        residual = float(np.sum(remaining, dtype=np.float64))
        radius = float(np.sqrt(np.max(remaining)))
    else:
        residual = 0.0
        radius = 0.0
    coverage = (
        1.0
        if total_variance <= np.finfo(np.float64).eps
        else float(np.clip(1.0 - residual / total_variance, 0.0, 1.0))
    )
    return coverage, radius


def _coverage_trace_numpy(
    points: np.ndarray,
    *,
    n_samples: int,
    target_coverage: float,
    selected_data: np.ndarray | None,
) -> _CoverageTrace:
    """Use the reference path when the private native module is absent."""
    values = np.ascontiguousarray(points)
    point_count = len(values)
    if point_count == 0 or n_samples <= 0:
        return _CoverageTrace((), (), (), 0.0)
    if values.ndim != 2:
        raise ValueError("points must be two dimensional")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    maximum = min(int(n_samples), point_count)
    total_variance = float(
        np.sum((values - np.mean(values, axis=0)) ** 2, dtype=np.float64)
    )
    nearest_squared = np.full(point_count, np.inf, dtype=values.dtype)
    selected_mask = np.zeros(point_count, dtype=bool)
    chosen: list[int] = []
    coverage_trace: list[float] = []
    radius_trace: list[float] = []
    initial_coverage = 0.0

    if selected_data is not None:
        warm = np.ascontiguousarray(selected_data, dtype=values.dtype)
        if warm.ndim != 2 or warm.shape[1] != values.shape[1]:
            raise ValueError("points and selected_data must have the same feature dimension")
        if len(warm) == 0:
            raise ValueError("selected_data must contain at least one row")
        for center in warm:
            delta = values - center
            nearest_squared = np.minimum(
                nearest_squared,
                np.sum(delta * delta, axis=1),
            )
        initial_coverage = _coverage_value(
            nearest_squared, selected_mask, total_variance
        )[0]
        if initial_coverage >= target_coverage:
            return _CoverageTrace((), (), (), initial_coverage)
    else:
        chosen.append(0)
        selected_mask[0] = True
        delta = values - values[0]
        nearest_squared = np.sum(delta * delta, axis=1)
        nearest_squared[0] = 0.0
        coverage, radius = _coverage_value(
            nearest_squared, selected_mask, total_variance
        )
        coverage_trace.append(coverage)
        radius_trace.append(radius)
        if coverage >= target_coverage:
            return _CoverageTrace(
                tuple(chosen), tuple(coverage_trace), tuple(radius_trace), 0.0
            )

    while len(chosen) < maximum:
        candidates = np.where(selected_mask, -np.inf, nearest_squared)
        farthest = int(np.argmax(candidates))
        farthest_squared = float(candidates[farthest])
        if not np.isfinite(farthest_squared) or farthest_squared <= 0.0:
            break
        chosen.append(farthest)
        selected_mask[farthest] = True
        nearest_squared[farthest] = 0.0
        delta = values - values[farthest]
        nearest_squared = np.minimum(
            nearest_squared,
            np.sum(delta * delta, axis=1),
        )
        nearest_squared[selected_mask] = 0.0
        coverage, radius = _coverage_value(
            nearest_squared, selected_mask, total_variance
        )
        coverage_trace.append(coverage)
        radius_trace.append(radius)
        if coverage >= target_coverage:
            break

    return _CoverageTrace(
        tuple(chosen),
        tuple(coverage_trace),
        tuple(radius_trace),
        initial_coverage,
    )


def _coverage_trace(
    points: np.ndarray,
    *,
    n_samples: int,
    target_coverage: float,
    selected_data: np.ndarray | None,
) -> _CoverageTrace:
    values = np.ascontiguousarray(points)
    warm = (
        None
        if selected_data is None
        else np.ascontiguousarray(selected_data, dtype=values.dtype)
    )
    if _native_sampling is not None and hasattr(
        _native_sampling, "farthest_point_sampling_trace"
    ):
        indices, coverage, radii, initial = (
            _native_sampling.farthest_point_sampling_trace(
                values,
                int(n_samples),
                float(target_coverage),
                warm,
            )
        )
        return _CoverageTrace(
            tuple(int(index) for index in indices),
            tuple(float(value) for value in coverage),
            tuple(float(value) for value in radii),
            float(initial),
        )
    return _coverage_trace_numpy(
        values,
        n_samples=n_samples,
        target_coverage=target_coverage,
        selected_data=warm,
    )


def _validated_groups(
    plan: PhysicsSamplingPlan,
    row_count: int,
) -> dict[PhysicsSamplingStratum, tuple[int, ...]]:
    groups = {
        stratum: tuple(int(index) for index in indices)
        for stratum, indices in plan.groups
    }
    observed = sorted(index for indices in groups.values() for index in indices)
    if observed != list(range(row_count)):
        raise ValueError("physics plan groups must cover each descriptor row exactly once")
    return groups


def _count_for_coverage(
    trace: _CoverageTrace,
    target: float,
) -> tuple[int, bool]:
    if trace.initial_coverage >= target:
        return 0, True
    for count, coverage in enumerate(trace.coverage, start=1):
        if coverage >= target:
            return count, True
    return len(trace.indices), False


def _centered_order(points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty(0, dtype=np.int64)
    center = np.mean(points, axis=0)
    center_index = int(np.argmin(np.linalg.norm(points - center, axis=1)))
    rows = np.arange(len(points), dtype=np.int64)
    return np.concatenate((rows[center_index : center_index + 1], np.delete(rows, center_index)))


def _stratum_local_features(
    candidate: SamplingFeatureBlocks,
    plan: PhysicsSamplingPlan,
    *,
    existing: SamplingFeatureBlocks | None = None,
    existing_plan: PhysicsSamplingPlan | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize feature blocks independently inside each physical stratum."""
    if candidate.row_count == 0:
        raise ValueError("candidate sampling feature blocks must not be empty")
    groups = _validated_groups(plan, candidate.row_count)
    if (existing is None) != (existing_plan is None):
        raise ValueError("existing feature blocks and existing plan must be supplied together")
    if existing is not None and existing.names != candidate.names:
        raise ValueError("candidate and existing sampling feature blocks differ")

    warm_groups: dict[PhysicsSamplingStratum, tuple[int, ...]] = {}
    if existing is not None and existing_plan is not None:
        warm_groups = _validated_groups(existing_plan, existing.row_count)

    feature_dim = sum(int(block.shape[1]) for block in candidate.values)
    candidate_values = np.zeros((candidate.row_count, feature_dim), dtype=np.float32)
    existing_values = (
        None
        if existing is None
        else np.zeros((existing.row_count, feature_dim), dtype=np.float32)
    )
    for stratum, row_tuple in groups.items():
        rows = np.asarray(row_tuple, dtype=np.int64)
        warm_tuple = warm_groups.get(stratum, ())
        warm_rows = np.asarray(warm_tuple, dtype=np.int64)
        local_candidate = candidate.take(rows)
        local_existing = (
            existing.take(warm_rows)
            if existing is not None and warm_rows.size
            else None
        )
        normalized, normalized_existing = representative_sampling_features(
            local_candidate,
            local_existing,
        )
        candidate_values[rows] = normalized
        if (
            existing_values is not None
            and normalized_existing is not None
            and warm_rows.size
        ):
            existing_values[warm_rows] = normalized_existing
    return candidate_values, existing_values


def recommend_physics_sampling_from_blocks(
    candidate: SamplingFeatureBlocks,
    plan: PhysicsSamplingPlan,
    *,
    existing: SamplingFeatureBlocks | None = None,
    existing_plan: PhysicsSamplingPlan | None = None,
    policy: SamplingPolicy = "balanced",
    sample_cap_per_stratum: int = _DEFAULT_SAMPLE_CAP_PER_STRATUM,
) -> PhysicsSamplingRecommendation:
    """Recommend counts using feature scaling fitted within each stratum."""
    values, warm_values = _stratum_local_features(
        candidate,
        plan,
        existing=existing,
        existing_plan=existing_plan,
    )
    return recommend_physics_sampling(
        values,
        plan,
        existing_descriptors=warm_values,
        existing_plan=existing_plan,
        policy=policy,
        sample_cap_per_stratum=sample_cap_per_stratum,
    )


def _warm_start_radius(points: np.ndarray, warm: np.ndarray | None) -> float | None:
    if warm is None:
        return None
    nearest_squared = np.full(len(points), np.inf, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64)
    for center in np.asarray(warm, dtype=np.float64):
        delta = values - center
        nearest_squared = np.minimum(
            nearest_squared,
            np.sum(delta * delta, axis=1),
        )
    return float(np.sqrt(np.max(np.maximum(nearest_squared, 0.0))))


def select_physics_budget(
    candidate: SamplingFeatureBlocks,
    plan: PhysicsSamplingPlan,
    *,
    n_samples: int,
    existing: SamplingFeatureBlocks | None = None,
    existing_plan: PhysicsSamplingPlan | None = None,
    min_distance: float = 0.0,
    coverage_policy: SamplingPolicy = "balanced",
) -> PhysicsBudgetSelection:
    """Spend one global budget according to residual layer coverage.

    Every previously uncovered physical stratum first receives one structure.
    A policy-selected per-stratum coverage floor is protected next.  Remaining
    slots go where the next FPS center removes the most layer-local distortion. The
    objective is nearest-center residual plus one farthest-point penalty, so
    large manifolds earn capacity without allowing a rare tail to be hidden by
    their average coverage.
    """
    requested = int(n_samples)
    if requested <= 0:
        raise ValueError("n_samples must be positive")
    if float(min_distance) < 0.0:
        raise ValueError("min_distance must be non-negative")
    if coverage_policy not in _POLICY_COVERAGE:
        raise ValueError(f"unsupported sampling coverage policy: {coverage_policy}")

    values, warm_values = _stratum_local_features(
        candidate,
        plan,
        existing=existing,
        existing_plan=existing_plan,
    )
    groups = _validated_groups(plan, len(values))
    warm_groups = (
        {}
        if existing_plan is None or warm_values is None
        else _validated_groups(existing_plan, len(warm_values))
    )

    states: list[dict[str, object]] = []
    minimum_required = 0
    for stratum in sorted(groups):
        candidate_rows = np.asarray(groups[stratum], dtype=np.int64)
        points = values[candidate_rows]
        warm_rows_tuple = warm_groups.get(stratum, ())
        warm_rows = np.asarray(warm_rows_tuple, dtype=np.int64)
        warm = (
            warm_values[warm_rows]
            if warm_values is not None and warm_rows.size
            else None
        )
        order = (
            np.arange(len(candidate_rows), dtype=np.int64)
            if warm is not None
            else _centered_order(points)
        )
        ordered_points = points[order]
        trace = _coverage_trace(
            ordered_points,
            n_samples=len(ordered_points),
            target_coverage=1.0,
            selected_data=warm,
        )
        mapped = tuple(int(candidate_rows[order[index]]) for index in trace.indices)
        initial_radius = _warm_start_radius(ordered_points, warm)
        mandatory = 1 if warm is None and mapped else 0
        minimum_required += mandatory
        states.append(
            {
                "stratum": stratum,
                "candidate_rows": candidate_rows,
                "warm_count": len(warm_rows),
                "trace": trace,
                "mapped": mapped,
                "count": mandatory,
                "initial_radius": initial_radius,
                "total_variance": float(
                    np.sum(
                        (ordered_points - np.mean(ordered_points, axis=0)) ** 2,
                        dtype=np.float64,
                    )
                ),
            }
        )

    if requested < minimum_required:
        raise ValueError(
            "sample budget is smaller than the number of uncovered physical strata "
            f"({requested} < {minimum_required})"
        )

    def can_take(state: dict[str, object]) -> bool:
        count = int(state["count"])
        mapped = state["mapped"]
        if count >= len(mapped):
            return False
        if count == 0:
            distance_to_centers = state["initial_radius"]
            return distance_to_centers is None or float(distance_to_centers) >= min_distance
        trace = state["trace"]
        distance_to_centers = trace.radii[count - 1]
        return float(distance_to_centers) >= min_distance

    def state_cost(state: dict[str, object], count: int) -> tuple[float, float]:
        trace = state["trace"]
        if count == 0:
            coverage = float(trace.initial_coverage)
            radius = float(state["initial_radius"] or 0.0)
        else:
            coverage = float(trace.coverage[count - 1])
            radius = float(trace.radii[count - 1])
        residual = max(0.0, 1.0 - coverage) * float(state["total_variance"])
        return residual + radius * radius, radius

    def marginal_gain(state: dict[str, object]) -> tuple[float, float, float]:
        count = int(state["count"])
        current_cost, current_radius = state_cost(state, count)
        next_cost, _next_radius = state_cost(state, count + 1)
        return max(0.0, current_cost - next_cost), current_cost, current_radius

    def allocation_priority(
        state: dict[str, object],
    ) -> tuple[int, float, float, float, float]:
        count = int(state["count"])
        trace = state["trace"]
        coverage = (
            float(trace.initial_coverage)
            if count == 0
            else float(trace.coverage[count - 1])
        )
        shortfall = max(0.0, _POLICY_COVERAGE[coverage_policy] - coverage)
        gain, current_cost, current_radius = marginal_gain(state)
        return int(shortfall > 0.0), shortfall, gain, current_cost, current_radius

    def heap_entry(index: int, state: dict[str, object]) -> tuple[float, ...]:
        priority = allocation_priority(state)
        return (
            *(-float(value) for value in priority),
            float(int(state["count"])),
            float(index),
        )

    queue: list[tuple[float, ...]] = []
    for state_index, state in enumerate(states):
        if can_take(state):
            heappush(queue, heap_entry(state_index, state))

    allocated = sum(int(state["count"]) for state in states)
    while allocated < requested and queue:
        entry = heappop(queue)
        state_index = int(entry[-1])
        chosen = states[state_index]
        chosen["count"] = int(chosen["count"]) + 1
        allocated += 1
        if can_take(chosen):
            heappush(queue, heap_entry(state_index, chosen))

    group_results: list[StratumBudgetSelection] = []
    selected_rows: list[int] = []
    for state in states:
        count = int(state["count"])
        trace = state["trace"]
        selected = tuple(state["mapped"][:count])
        selected_rows.extend(selected)
        achieved = (
            float(trace.initial_coverage)
            if count == 0
            else float(trace.coverage[count - 1])
        )
        radius = (
            state["initial_radius"]
            if count == 0
            else float(trace.radii[count - 1])
        )
        group_results.append(
            StratumBudgetSelection(
                stratum=state["stratum"],
                candidate_count=len(state["candidate_rows"]),
                existing_count=int(state["warm_count"]),
                selected_count=count,
                achieved_coverage=achieved,
                coverage_radius=None if radius is None else float(radius),
                selected_indices=selected,
            )
        )

    return PhysicsBudgetSelection(
        requested_count=requested,
        selected_indices=tuple(sorted(selected_rows)),
        groups=tuple(group_results),
        exhausted=len(selected_rows) < requested,
    )


def recommend_physics_sampling(
    descriptors: np.ndarray,
    plan: PhysicsSamplingPlan,
    *,
    existing_descriptors: np.ndarray | None = None,
    existing_plan: PhysicsSamplingPlan | None = None,
    policy: SamplingPolicy = "balanced",
    sample_cap_per_stratum: int = _DEFAULT_SAMPLE_CAP_PER_STRATUM,
) -> PhysicsSamplingRecommendation:
    """Recommend an incremental physics-aware sample count from FPS coverage.

    ``compact``, ``balanced``, and ``conservative`` correspond to per-stratum
    descriptor-coverage targets of 0.98, 0.99, and 0.995.  Every stratum is
    evaluated independently, so a broad noncollinear manifold can earn more
    samples than a narrow or duplicate-heavy magnetic stratum.  If the safety
    cap is reached before the requested policy target, ``is_lower_bound`` is
    true and the returned count must not be presented as a converged optimum.
    """
    if policy not in _POLICY_COVERAGE:
        raise ValueError(f"unsupported sampling recommendation policy: {policy}")
    if int(sample_cap_per_stratum) <= 0:
        raise ValueError("sample_cap_per_stratum must be positive")
    values = np.ascontiguousarray(descriptors, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("descriptors must be a finite two-dimensional array")
    groups = _validated_groups(plan, len(values))

    if (existing_descriptors is None) != (existing_plan is None):
        raise ValueError("existing_descriptors and existing_plan must be supplied together")
    warm_values = None
    warm_groups: dict[PhysicsSamplingStratum, tuple[int, ...]] = {}
    if existing_descriptors is not None and existing_plan is not None:
        warm_values = np.ascontiguousarray(existing_descriptors, dtype=np.float32)
        if (
            warm_values.ndim != 2
            or warm_values.shape[1] != values.shape[1]
            or not np.isfinite(warm_values).all()
        ):
            raise ValueError(
                "existing descriptors must be finite and match the candidate feature dimension"
            )
        warm_groups = _validated_groups(existing_plan, len(warm_values))

    recommendations: list[StratumSamplingRecommendation] = []
    selected_for_policy: list[int] = []
    compact_total = 0
    conservative_total = 0
    lower_bound = False
    policy_target = _POLICY_COVERAGE[policy]

    for stratum in sorted(groups):
        candidate_rows = np.asarray(groups[stratum], dtype=np.int64)
        candidate_points = values[candidate_rows]
        warm_rows = warm_groups.get(stratum, ())
        warm_points = (
            warm_values[np.asarray(warm_rows, dtype=np.int64)]
            if warm_values is not None and warm_rows
            else None
        )
        order = (
            np.arange(len(candidate_rows), dtype=np.int64)
            if warm_points is not None
            else _centered_order(candidate_points)
        )
        maximum = min(len(candidate_rows), int(sample_cap_per_stratum))
        trace = _coverage_trace(
            candidate_points[order],
            n_samples=maximum,
            target_coverage=_POLICY_COVERAGE["conservative"],
            selected_data=warm_points,
        )
        mapped = tuple(
            int(candidate_rows[order[index]]) for index in trace.indices
        )
        compact_count, _compact_reached = _count_for_coverage(
            trace, _POLICY_COVERAGE["compact"]
        )
        balanced_count, _balanced_reached = _count_for_coverage(
            trace, _POLICY_COVERAGE["balanced"]
        )
        conservative_count, _conservative_reached = _count_for_coverage(
            trace, _POLICY_COVERAGE["conservative"]
        )
        count_by_policy = {
            "compact": compact_count,
            "balanced": balanced_count,
            "conservative": conservative_count,
        }
        selected_count = count_by_policy[policy]
        selected = mapped[:selected_count]
        selected_for_policy.extend(selected)
        compact_total += compact_count
        conservative_total += conservative_count
        reached_policy = (
            trace.initial_coverage >= policy_target
            or (
                selected_count > 0
                and trace.coverage[selected_count - 1] >= policy_target
            )
        )
        lower_bound = lower_bound or not reached_policy
        achieved = (
            trace.initial_coverage
            if selected_count == 0
            else trace.coverage[selected_count - 1]
        )
        radius = (
            None
            if selected_count == 0
            else trace.radii[selected_count - 1]
        )
        recommendations.append(
            StratumSamplingRecommendation(
                stratum=stratum,
                candidate_count=len(candidate_rows),
                existing_count=len(warm_rows),
                compact_count=compact_count,
                recommended_count=selected_count,
                conservative_count=conservative_count,
                achieved_coverage=float(achieved),
                coverage_radius=None if radius is None else float(radius),
                selected_indices=selected,
                reached_target=reached_policy,
            )
        )

    selected_indices = tuple(sorted(selected_for_policy))
    return PhysicsSamplingRecommendation(
        policy=policy,
        selected_indices=selected_indices,
        recommended_count=len(selected_indices),
        compact_count=compact_total,
        conservative_count=conservative_total,
        groups=tuple(recommendations),
        is_lower_bound=lower_bound,
    )
