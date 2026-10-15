"""Streaming, non-compensating structure and local-environment coverage.

The metric is fitted once per element set (structure views) / central species
(atomic environments). Phase/spin labels protect representatives but neither
rescale nor prohibit sharing geometric coverage. No N-by-N matrix is stored.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .sampling_features import (
    SamplingFeatureBlocks,
    _normalize_block_sets,
    _normalize_descriptor_sets,
)
from .sampling_plan import PhysicsSamplingPlan

try:
    from NepTrainKit._native import _sampling as _native_sampling
except ImportError:
    _native_sampling = None

# RMS distances in the fixed, regularized feature metric, not explained variance.
RESOLUTIONS = {"compact": 0.35, "balanced": 0.25, "conservative": 0.18}
_ATOM_RESOLUTION = min(RESOLUTIONS.values())


@dataclass(frozen=True)
class CoverageEvidence:
    """Auditable geometric coverage; atomic distances are conservative bounds."""

    stop_reason: str
    resolution: float
    view_names: tuple[str, ...]
    max_distances: tuple[float, ...]
    p95_distances: tuple[float, ...]
    uncovered_indices: tuple[int, ...]
    atomic_available: bool


class _NumpyCoverageState:
    """Independent reference implementation of the native streaming state."""

    def __init__(self, points, ends, elements, atoms, offsets, species, count):
        self.points, self.ends, self.elements = points, ends, elements
        self.atoms, self.offsets, self.species = atoms, offsets, species
        self.count = count
        self.nearest = np.full((count, len(ends) + 1), np.inf)
        self.atomic = np.full(int(offsets[count]), np.inf)
        self.chosen = np.zeros(count, dtype=bool)
        self.atom_resolution = _ATOM_RESOLUTION

    def refine_atoms(self, resolution=0.0):
        """Re-enable exact distances before selecting below the automatic floor."""
        self.atom_resolution = resolution

    def add(self, center):
        compatible = np.flatnonzero(self.elements[:self.count] == self.elements[center])
        start = 0
        joint = np.zeros(len(compatible))
        for block, end in enumerate(self.ends):
            delta = self.points[compatible, start:end] - self.points[center, start:end]
            d2 = np.sum(delta * delta, axis=1)
            self.nearest[compatible, block] = np.minimum(self.nearest[compatible, block], d2)
            joint = np.maximum(joint, d2)
            start = end
        self.nearest[compatible, -1] = np.minimum(self.nearest[compatible, -1], joint)
        for atom in range(self.offsets[center], self.offsets[center + 1]):
            active = np.flatnonzero((self.species[:len(self.atomic)] == self.species[atom])
                                    & (self.atomic > self.atom_resolution ** 2))
            delta = self.atoms[active] - self.atoms[atom]
            d2 = np.sum(delta * delta, axis=1)
            self.atomic[active] = np.minimum(self.atomic[active], d2)
        if center < self.count:
            self.chosen[center] = True
            self.nearest[center] = 0.0
            self.atomic[self.offsets[center]:self.offsets[center + 1]] = 0.0

    def distances(self):
        atom_max = np.zeros(self.count)
        for row in range(self.count):
            values = self.atomic[self.offsets[row]:self.offsets[row + 1]]
            atom_max[row] = np.max(values, initial=0.0)
        return np.sqrt(np.column_stack((self.nearest, atom_max)))

    def scores(self):
        distances = self.distances()
        local = np.where(distances[:, -1] > self.atom_resolution, distances[:, -1], 0.0)
        scores = np.maximum(distances[:, -2], local)
        scores[self.chosen] = -1.0
        return scores


def _groups(plan, count):
    groups = dict(plan.groups)
    if sorted(i for rows in groups.values() for i in rows) != list(range(count)):
        raise ValueError("physics plan groups must cover each descriptor row exactly once")
    return groups


def _prepare(candidate, plan, existing, existing_plan):
    n = candidate.row_count
    groups = _groups(plan, n)
    if (existing is None) != (existing_plan is None):
        raise ValueError("existing feature blocks and existing plan must be supplied together")
    if not n or not candidate.values:
        raise ValueError("candidate sampling feature blocks must not be empty")
    if existing is not None and tuple(k for k in existing.names if not k.startswith("input_")) != tuple(
            k for k in candidate.names if not k.startswith("input_")):
        raise ValueError("candidate and existing sampling feature blocks differ")
    if existing is not None and any(k.startswith("input_") for k in existing.names) != any(
            k.startswith("input_") for k in candidate.names):
        raise ValueError("candidate and existing sampling feature blocks differ")
    warm_groups = {} if existing is None else _groups(existing_plan, existing.row_count)
    both = [candidate] if existing is None else [candidate, existing]
    keys = sorted({key.element_set for key in groups} | {key.element_set for key in warm_groups})
    elements = np.empty(sum(b.row_count for b in both), dtype=np.int64)
    for key, rows in groups.items():
        elements[list(rows)] = keys.index(key.element_set)
    for key, rows in warm_groups.items():
        elements[n + np.asarray(rows)] = keys.index(key.element_set)

    names = tuple(k for k in candidate.names if not k.startswith("input_")) + tuple(sorted(
        {k for b in both for k in b.names if k.startswith("input_")}))
    normalized = []
    for name in names:
        width = next(b.values[b.names.index(name)].shape[1] for b in both if name in b.names)
        raw = np.vstack([b.values[b.names.index(name)] if name in b.names else
                         np.zeros((b.row_count, width)) for b in both]).astype(np.float64)
        if raw.ndim != 2 or raw.shape[1] < 1 or len(raw) != len(elements) or not np.isfinite(raw).all():
            raise ValueError("sampling features must be finite and align with structures")
        block = np.empty_like(raw)
        for element in range(len(keys)):
            rows = np.flatnonzero(elements == element)
            normalize = _normalize_descriptor_sets if name == "descriptor_mean" else _normalize_block_sets
            local, _ = normalize(raw[rows], None)
            if name.startswith("input_"):
                # Keep the physical input coordinate axes.
                block[rows] = local
            else:
                unique = np.unique(local, axis=0)
                _, rotation = np.linalg.eigh(unique.T @ unique)
                block[rows] = local @ rotation[:, ::-1]
        normalized.append(block)
    if all(b.atom_counts and b.atom_species for b in both):
        symbols = sorted({s for b in both for s in b.atom_species})
        compositions = []
        for data in both:
            offset = 0
            for count in data.atom_counts:
                local = data.atom_species[offset:offset + count]
                compositions.append([local.count(symbol) / count for symbol in symbols])
                offset += count
        # Fractions already have physical range [0, 1]; do not amplify rare
        # composition fluctuations by fitting a variance scale.
        normalized.append(np.asarray(compositions, dtype=np.float64))
        names = (*names, "composition")
    points = np.ascontiguousarray(np.hstack(normalized), dtype=np.float64)
    ends = np.cumsum([b.shape[1] for b in normalized]).astype(np.int64)

    available = all(b.atomic_values is not None for b in both)
    if available:
        counts = np.asarray([c for b in both for c in b.atom_counts], dtype=np.int64)
        symbols = [s for b in both for s in b.atom_species]
        atoms = np.vstack([b.atomic_values for b in both]).astype(np.float64)
        if (len(counts) != len(points) or np.any(counts <= 0) or
                counts.sum() != len(atoms) or len(symbols) != len(atoms) or
                not np.isfinite(atoms).all()):
            raise ValueError("atomic descriptors, species and counts must align with structures")
        species_keys = sorted(set(symbols))
        species = np.asarray([species_keys.index(s) for s in symbols], dtype=np.int64)
        fingerprint_atoms = atoms.copy()
        for sp in range(len(species_keys)):
            rows = np.flatnonzero(species == sp)
            block, _ = _normalize_descriptor_sets(atoms[rows], None)
            # An orthogonal rotation changes no distance. Large-variance axes
            # first allow the native exact distance loop to reject pairs early.
            unique = np.unique(block, axis=0)
            _, rotation = np.linalg.eigh(unique.T @ unique)
            atoms[rows] = block @ rotation[:, ::-1]
        atoms = np.ascontiguousarray(atoms)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    else:
        atoms = np.empty((0, 0), dtype=np.float64)
        offsets = np.zeros(len(points) + 1, dtype=np.int64)
        species = np.empty(0, dtype=np.int64)

    # Content-based tie breaking is insensitive to input order. Duplicate rows
    # remain interchangeable; their original index is retained for the caller.
    digests = []
    for i in range(n):
        digest = hashlib.blake2b(points[i].tobytes(), digest_size=16)
        spread = 0.0
        if available:
            rows = fingerprint_atoms[offsets[i]:offsets[i + 1]]
            local_species = species[offsets[i]:offsets[i + 1]]
            local = np.column_stack((local_species, rows))
            local = np.unique(local, axis=0)
            digest.update(local.tobytes())
            centered = atoms[offsets[i]:offsets[i + 1]]
            spread = round(float(np.mean(np.sum(centered * centered, axis=1))), 12)
        # Equal structure centroids need a meaningful atomic tie-break: start
        # with the more central local environments, then use content identity.
        # Appending a constant input view must not displace that representative.
        digests.append((spread, digest.digest()))
    tie_order = np.asarray(sorted(range(n), key=digests.__getitem__), dtype=np.int64)
    state_type = _NumpyCoverageState
    if (_native_sampling is not None and hasattr(_native_sampling, "CoverageState")
            and getattr(_native_sampling, "coverage_api_version", 0) >= 3):
        state_type = _native_sampling.CoverageState
    state = state_type(points, ends, elements, atoms, offsets, species, n)
    return state, points, groups, warm_groups, tie_order, names, available


@dataclass(frozen=True)
class CoverageSelection:
    indices: tuple[int, ...]
    distances: np.ndarray
    evidence: CoverageEvidence
    counts: dict[str, int]
    trajectory: tuple[int, ...]


def _select_coverage(candidate: SamplingFeatureBlocks, plan: PhysicsSamplingPlan, *,
                    existing=None, existing_plan=None, budget=None, policy="balanced",
                    min_distance=0.0, resource_limit=None, resolution=None,
                    _prepared=None, _initial_indices=(), _requirements=None) -> CoverageSelection:
    """One deterministic trajectory for fixed budgets and radius stopping.

    Atomic environments within the conservative resolution are retired from
    further updates. Their stored distances are upper bounds, never optimistic
    estimates; every active environment is still compared with all new atoms
    of the same species. Whole-structure multi-view distances remain exact.
    """
    if policy not in RESOLUTIONS:
        raise ValueError(f"unsupported sampling coverage policy: {policy}")
    tolerance = RESOLUTIONS[policy] if resolution is None else float(resolution)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("coverage radius must be finite and positive")
    if budget is not None and int(budget) <= 0:
        raise ValueError("n_samples must be positive")
    if not np.isfinite(min_distance) or min_distance < 0:
        raise ValueError("min_distance must be finite and non-negative")
    if resource_limit is not None and int(resource_limit) <= 0:
        raise ValueError("resource_limit must be positive")
    prepared = _prepare(candidate, plan, existing, existing_plan) if _prepared is None else _prepared
    state, points, groups, warm_groups, tie_order, names, available = prepared
    if tolerance < _ATOM_RESOLUTION:
        state.refine_atoms(tolerance)
    n = candidate.row_count
    maximum = min(n, int(budget) if budget is not None else n,
                  int(resource_limit) if resource_limit is not None else n)
    required = [key for key in sorted(groups) if key not in warm_groups]
    if budget is not None and maximum < len(required):
        raise ValueError("sample budget is smaller than the number of uncovered physical strata "
                         f"({maximum} < {len(required)})")
    for i in range(n, len(points)):
        state.add(i)
    chosen = [int(i) for i in _initial_indices]
    if (len(chosen) > maximum or len(set(chosen)) != len(chosen)
            or any(i < 0 or i >= n for i in chosen)):
        raise ValueError("invalid initial coverage representatives")
    for row in chosen:
        state.add(row)
        if _requirements is not None:
            _requirements.add(row)
    # One representative per previously unrepresented label, chosen near the
    # unique-data centroid, then shared global geometry drives all remaining slots.
    for key in required:
        if len(chosen) == maximum:
            break
        if set(chosen).intersection(groups[key]):
            continue
        rows = tie_order[np.isin(tie_order, groups[key])]
        local = points[rows]
        center = np.mean(np.unique(local, axis=0), axis=0)
        row = int(rows[np.argmin(np.sum((local - center) ** 2, axis=1))])
        chosen.append(row)
        state.add(row)
        if _requirements is not None:
            _requirements.add(row)
    counts = {}
    snapshots = {}
    refined_atoms = False
    stop_reason = "budget" if budget is not None else "resource_limit"
    while True:
        scores = np.asarray(state.scores())
        radius = float(np.max(scores, initial=0.0))
        pending = np.zeros(n, dtype=bool) if _requirements is None else _requirements.pending()
        protected = len(chosen) >= len(required) and not np.any(pending)
        for name, preset_radius in RESOLUTIONS.items():
            if protected and radius <= preset_radius and name not in counts:
                counts[name] = len(chosen)
                if budget is None and resolution is None:
                    snapshots[name] = np.asarray(state.distances()).copy()
        stop_radius = min(RESOLUTIONS.values()) if resolution is None else tolerance
        if budget is None and protected and radius <= stop_radius:
            stop_reason = "resolution"
            break
        if len(chosen) >= maximum:
            break
        if (budget is not None and available and not refined_atoms
                and radius <= _ATOM_RESOLUTION and min_distance <= _ATOM_RESOLUTION):
            # Automatic coverage may retire already-covered atoms. A manual
            # budget or finer cutoff must still be able to distinguish them.
            state.refine_atoms()
            for center in (*range(n, len(points)), *chosen):
                state.add(center)
            refined_atoms = True
            continue
        if radius <= 1e-12 and not np.any(pending):
            stop_reason = "exhausted"
            break
        if budget is not None and radius < min_distance:
            stop_reason = "min_distance"
            break
        eligible = tie_order[pending[tie_order]] if np.any(pending) else tie_order
        row = int(eligible[np.argmax(scores[eligible])])
        chosen.append(row)
        state.add(row)
        if _requirements is not None:
            _requirements.add(row)
    trajectory = tuple(chosen)
    distances = np.asarray(state.distances())
    if budget is None and resolution is None and policy in snapshots:
        chosen = chosen[:counts[policy]]
        distances = snapshots[policy]
        stop_reason = "resolution"
    uncovered = np.flatnonzero(np.maximum(distances[:, -2], distances[:, -1]) > tolerance)
    pending = np.zeros(n, dtype=bool) if _requirements is None else _requirements.pending()
    uncovered = np.union1d(uncovered, np.flatnonzero(pending))
    # Missing category representatives cannot be certified merely by geometry.
    selected_set = set(chosen)
    for key in required:
        if not selected_set.intersection(groups[key]):
            uncovered = np.union1d(uncovered, groups[key])
    evidence = CoverageEvidence(
        stop_reason, tolerance, (*names, "joint_structure", "atomic_environment"),
        tuple(np.max(distances, axis=0)), tuple(np.percentile(distances, 95, axis=0)),
        tuple(int(i) for i in uncovered), available,
    )
    return CoverageSelection(tuple(chosen), distances, evidence, counts, trajectory)


def _without_inputs(blocks):
    """Drop model-independent input views; descriptors and atoms are kept."""
    if blocks is None:
        return None
    pairs = [(k, v) for k, v in zip(blocks.names, blocks.values) if not k.startswith("input_")]
    if not pairs:
        return blocks
    return SamplingFeatureBlocks(tuple(k for k, _ in pairs), tuple(v for _, v in pairs),
                                 blocks.atomic_values, blocks.atom_counts, blocks.atom_species)


def select_coverage(candidate: SamplingFeatureBlocks, plan: PhysicsSamplingPlan, *,
                    existing=None, existing_plan=None, budget=None, policy="balanced",
                    min_distance=0.0, resource_limit=None, resolution=None) -> CoverageSelection:
    """Select coverage using the established GUI/card sampling path.

    A fixed count with input views, without a training set or distance
    threshold, uses the input-state witness cover (``sampling_states``).
    Radius stopping and thresholds use the shared FPS trajectory. Its radius
    is measured on descriptors, structure features and atomic environments
    only: per-frame input fingerprints separate nearly every thermal frame, so
    they keep each element set's input extent as witnesses (unless a training
    set is supplied) instead of entering the stopping distance.
    """
    from .sampling_states import _RangeRequirements, select_state_coverage
    inputs = any(name.startswith("input_") for name in candidate.names)
    if (inputs and budget is not None and existing is None and existing_plan is None
            and min_distance == 0 and resource_limit is None and resolution is None):
        return select_state_coverage(candidate, plan, budget=budget, policy=policy)
    requirements = None
    if inputs and existing is None:
        # Keep each element set's input extent; any frame at an extreme counts.
        pools = {}
        for key, rows in plan.groups:
            pools.setdefault(key.element_set, []).extend(rows)
        requirements = _RangeRequirements(candidate, pools, np.zeros(candidate.row_count, dtype=bool))
    kwargs = dict(existing=_without_inputs(existing), existing_plan=existing_plan, budget=budget,
                  policy=policy, min_distance=min_distance, resource_limit=resource_limit,
                  resolution=resolution)
    return _select_coverage(_without_inputs(candidate), plan, _requirements=requirements, **kwargs)
