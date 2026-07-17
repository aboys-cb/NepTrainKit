"""Conservative candidate-phase refinement for L1_2 and Laves structures.

The routines in this module are deliberately fail-closed: they only emit a
specific phase label after geometry, chemistry, and structure-level agreement
all pass.  A negative result means "not confirmed", not proof that the phase
is absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Iterable, Sequence

import numpy as np

from . import phase_sketch as _phase_sketch


accelerated_periodic_knn_vectors = _phase_sketch.accelerated_periodic_knn_vectors


_EPS = 1.0e-12
_L12_LOCAL_SHAPE_MAX_RMS = 0.10
_LAVES_LOCAL_SHAPE_MAX_RMS = 0.095
_L12_MIN_GEOMETRY_FRACTION = 0.80
_L12_MIN_JOINT_FRACTION = 0.80
_LAVES_MIN_GEOMETRY_FRACTION = 0.85
_LAVES_MIN_JOINT_FRACTION = 0.85
_B2_CSP_THRESHOLD = 0.8


@dataclass(frozen=True)
class PhaseRefinement:
    """Structure-level result of a conservative candidate-phase check.

    ``defect_fraction`` is the fraction of unmatched local environments.  It
    is not an exact vacancy or anti-site concentration.
    """

    candidate: str
    label: str
    confirmed: bool
    geometry_match_fraction: float
    chemistry_match_fraction: float
    joint_match_fraction: float
    defect_fraction: float
    a_types: tuple[int, ...]
    b_types: tuple[int, ...]
    reason: str
    b2_fraction: float | None = None


def _as_inputs(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
    atom_types: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    box = np.ascontiguousarray(cell, dtype=np.float64).reshape(3, 3)
    periodic = np.asarray(pbc, dtype=bool).reshape(3)
    types = np.asarray(atom_types, dtype=np.int32).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) != len(types):
        raise ValueError("positions must be N x 3 and match atom_types")
    if not len(pos):
        raise ValueError("phase refinement requires at least one atom")
    return pos, box, periodic, types


def _resolve_roles(
    atom_types: np.ndarray,
    *,
    expected_a_fraction: float,
    a_types: Iterable[int] | None,
    b_types: Iterable[int] | None,
    auto_tolerance: float,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    present, counts = np.unique(atom_types, return_counts=True)
    present_set = {int(value) for value in present}
    if a_types is None and b_types is None:
        if len(present) != 2:
            return (), (), "element roles are ambiguous; provide A/B type groups"
        fractions = counts / np.sum(counts)
        a_index = int(np.argmin(np.abs(fractions - expected_a_fraction)))
        if abs(float(fractions[a_index]) - expected_a_fraction) > auto_tolerance:
            return (), (), "composition is outside the unambiguous auto-inference range"
        a = (int(present[a_index]),)
        b = (int(present[1 - a_index]),)
        return a, b, ""
    if a_types is None or b_types is None:
        raise ValueError("a_types and b_types must be supplied together")
    a = tuple(sorted({int(value) for value in a_types}))
    b = tuple(sorted({int(value) for value in b_types}))
    if not a or not b or set(a) & set(b):
        raise ValueError("A/B type groups must be non-empty and disjoint")
    if set(a) | set(b) != present_set:
        raise ValueError("A/B type groups must cover every atom type in the structure")
    return a, b, ""


def _shape_descriptor(vectors: np.ndarray, coordination: int) -> np.ndarray:
    distances = np.linalg.norm(vectors, axis=1)
    order = np.argsort(distances, kind="stable")[:coordination]
    selected = vectors[order]
    radii = distances[order]
    if len(selected) != coordination or radii[0] <= _EPS:
        raise ValueError("insufficient non-overlapping neighbors for phase refinement")
    scale = float(np.mean(radii))
    normalized = selected / scale
    pairwise = np.linalg.norm(
        normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :], axis=2
    )
    upper = pairwise[np.triu_indices(coordination, k=1)]
    return np.concatenate((np.sort(radii / scale), np.sort(upper)))


def _nearest_template_distances(
    descriptors: np.ndarray,
    templates: np.ndarray,
) -> np.ndarray:
    differences = descriptors[:, np.newaxis, :] - templates[np.newaxis, :, :]
    return np.sqrt(np.mean(differences * differences, axis=2)).min(axis=1)


def _repeat_crystal(
    cell: np.ndarray,
    fractional: np.ndarray,
    atom_types: np.ndarray,
    repeats: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shifts = np.asarray(tuple(product(*(range(value) for value in repeats))), dtype=float)
    repeated_fractional = (
        fractional[np.newaxis, :, :] + shifts[:, np.newaxis, :]
    ).reshape(-1, 3)
    positions = repeated_fractional @ cell
    repeated_cell = np.diag(np.asarray(repeats, dtype=float)) @ cell
    return positions, repeated_cell, np.tile(atom_types, len(shifts))


def _hexagonal_cell(a: float, c_over_a: float) -> np.ndarray:
    return np.asarray(
        (
            (0.5 * a, -0.5 * np.sqrt(3.0) * a, 0.0),
            (0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0),
            (0.0, 0.0, a * c_over_a),
        )
    )


def _c14_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AFLOW LL0C MgZn2 C14 reference."""
    a, c_over_a, z2, x3 = 5.223, 1.64005, 0.06286, 0.830483
    fractional = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5),
            (1 / 3, 2 / 3, z2),
            (2 / 3, 1 / 3, z2 + 0.5),
            (2 / 3, 1 / 3, -z2),
            (1 / 3, 2 / 3, 0.5 - z2),
            (x3, 2 * x3, 0.25),
            (-2 * x3, -x3, 0.25),
            (x3, -x3, 0.25),
            (-x3, -2 * x3, 0.75),
            (2 * x3, x3, 0.75),
            (-x3, x3, 0.75),
        ),
        dtype=float,
    )
    # A is the minority Z16 center; B is the majority Z12 center.
    types = np.asarray((1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1), dtype=np.int32)
    return _repeat_crystal(
        _hexagonal_cell(a, c_over_a), np.mod(fractional, 1.0), types, (3, 3, 2)
    )


def _c15_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AFLOW 8YL7 MgCu2 C15 reference."""
    a = 7.02
    cell = 0.5 * a * np.asarray(((0, 1, 1), (1, 0, 1), (1, 1, 0)), dtype=float)
    fractional = np.asarray(
        (
            (3 / 8, 3 / 8, 3 / 8),
            (5 / 8, 5 / 8, 5 / 8),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5),
            (0.0, 0.5, 0.0),
            (0.5, 0.0, 0.0),
        ),
        dtype=float,
    )
    types = np.asarray((0, 0, 1, 1, 1, 1), dtype=np.int32)
    return _repeat_crystal(cell, fractional, types, (3, 3, 3))


def _c36_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AFLOW HV5V MgNi2 C36 reference."""
    a, c_over_a = 4.824, 3.28068
    z1, z2, z3, x5 = 0.094, 0.84417, 0.12514, 0.16429
    fractional = np.asarray(
        (
            (0.0, 0.0, z1),
            (0.0, 0.0, z1 + 0.5),
            (0.0, 0.0, -z1),
            (0.0, 0.0, 0.5 - z1),
            (1 / 3, 2 / 3, z2),
            (2 / 3, 1 / 3, z2 + 0.5),
            (2 / 3, 1 / 3, -z2),
            (1 / 3, 2 / 3, 0.5 - z2),
            (1 / 3, 2 / 3, z3),
            (2 / 3, 1 / 3, z3 + 0.5),
            (2 / 3, 1 / 3, -z3),
            (1 / 3, 2 / 3, 0.5 - z3),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            (x5, 2 * x5, 0.25),
            (-2 * x5, -x5, 0.25),
            (x5, -x5, 0.25),
            (-x5, -2 * x5, 0.75),
            (2 * x5, x5, 0.75),
            (-x5, x5, 0.75),
        ),
        dtype=float,
    )
    types = np.asarray((0,) * 8 + (1,) * 16, dtype=np.int32)
    return _repeat_crystal(
        _hexagonal_cell(a, c_over_a), np.mod(fractional, 1.0), types, (3, 3, 2)
    )


def _fcc_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = 3.75
    fractional = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0))
    )
    return _repeat_crystal(
        np.eye(3) * a,
        fractional,
        np.asarray((0, 1, 1, 1), dtype=np.int32),
        (3, 3, 3),
    )


@lru_cache(maxsize=1)
def _reference_templates() -> dict[str, np.ndarray]:
    templates: dict[str, list[np.ndarray]] = {"fcc": [], "z12": [], "z16": []}
    for name, builder in (
        ("fcc", _fcc_prototype),
        ("laves", _c14_prototype),
        ("laves", _c15_prototype),
        ("laves", _c36_prototype),
    ):
        positions, cell, types = builder()
        vectors, _, valid = accelerated_periodic_knn_vectors(
            positions, cell, (True, True, True), neighbors=20
        )
        for atom in range(len(positions)):
            current = vectors[atom, valid[atom]]
            if name == "fcc":
                templates["fcc"].append(_shape_descriptor(current, 12))
            elif types[atom] == 0:
                templates["z16"].append(_shape_descriptor(current, 16))
            else:
                templates["z12"].append(_shape_descriptor(current, 12))
    unique: dict[str, np.ndarray] = {}
    for key, rows in templates.items():
        rounded = np.round(np.asarray(rows), decimals=10)
        unique[key] = np.unique(rounded, axis=0)
    return unique


def _neighbor_data(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors, indices, valid = accelerated_periodic_knn_vectors(
        positions, cell, pbc, neighbors=neighbors
    )
    return _ordered_neighbor_data(vectors, indices, valid)


def _ordered_neighbor_data(
    vectors: np.ndarray,
    indices: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered_vectors = np.zeros_like(vectors)
    ordered_indices = np.full_like(indices, -1)
    ordered_valid = np.zeros_like(valid)
    for atom in range(len(vectors)):
        current_vectors = vectors[atom, valid[atom]]
        current_indices = indices[atom, valid[atom]]
        order = np.argsort(np.linalg.norm(current_vectors, axis=1), kind="stable")
        count = len(order)
        ordered_vectors[atom, :count] = current_vectors[order]
        ordered_indices[atom, :count] = current_indices[order]
        ordered_valid[atom, :count] = True
    return ordered_vectors, ordered_indices, ordered_valid


def _normalized_csp(vectors: np.ndarray) -> float:
    if len(vectors) != 6:
        raise ValueError("the Laves B-site CSP requires six B neighbors")

    def minimum_cost(remaining: tuple[int, ...]) -> float:
        if not remaining:
            return 0.0
        first = remaining[0]
        return min(
            float(np.dot(vectors[first] + vectors[other], vectors[first] + vectors[other]))
            + minimum_cost(remaining[1:index] + remaining[index + 1 :])
            for index, other in enumerate(remaining[1:], start=1)
        )

    scale = float(np.mean(np.sum(vectors * vectors, axis=1)))
    return minimum_cost(tuple(range(6))) / max(scale, _EPS)


def refine_l12(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
    atom_types: Sequence[int],
    *,
    a_types: Iterable[int] | None = None,
    b_types: Iterable[int] | None = None,
) -> PhaseRefinement:
    """Confirm L1_2 ordering only after an FCC and AB3 local-order gate."""
    pos, box, periodic, types = _as_inputs(positions, cell, pbc, atom_types)
    a, b, role_error = _resolve_roles(
        types,
        expected_a_fraction=0.25,
        a_types=a_types,
        b_types=b_types,
        auto_tolerance=0.035,
    )
    if role_error:
        return PhaseRefinement(
            "l12", "unknown", False, 0.0, 0.0, 0.0, 1.0, (), (), role_error
        )
    vectors, indices, valid = accelerated_periodic_knn_vectors(
        pos, box, periodic, neighbors=20
    )
    templates = _reference_templates()["fcc"]
    native = _phase_sketch._native_phase
    if native is not None and hasattr(native, "l12_refinement_metrics"):
        geometry_fraction, chemistry_fraction, joint_fraction = (
            native.l12_refinement_metrics(
                vectors,
                indices,
                valid,
                types,
                np.asarray(a, dtype=np.int32),
                templates,
                _L12_LOCAL_SHAPE_MAX_RMS,
            )
        )
    else:
        vectors, indices, valid = _ordered_neighbor_data(vectors, indices, valid)
        a_mask = np.isin(types, a)
        geometry_match = np.zeros(len(pos), dtype=bool)
        chemistry_match = np.zeros(len(pos), dtype=bool)
        for atom in range(len(pos)):
            if int(np.sum(valid[atom])) < 18:
                continue
            descriptor = _shape_descriptor(vectors[atom, valid[atom]], 12)
            geometry_match[atom] = _nearest_template_distances(
                descriptor[np.newaxis, :], templates
            )[0] <= _L12_LOCAL_SHAPE_MAX_RMS
            first_shell_types = types[indices[atom, :12]]
            second_shell_types = types[indices[atom, 12:18]]
            first_shell_a = int(np.sum(np.isin(first_shell_types, a)))
            second_shell_a = int(np.sum(np.isin(second_shell_types, a)))
            chemistry_match[atom] = (
                first_shell_a == 0 and second_shell_a == 6
                if a_mask[atom]
                else first_shell_a == 4 and second_shell_a == 0
            )
        joint = geometry_match & chemistry_match
        geometry_fraction = float(np.mean(geometry_match))
        chemistry_fraction = float(np.mean(chemistry_match))
        joint_fraction = float(np.mean(joint))
    if geometry_fraction < _L12_MIN_GEOMETRY_FRACTION:
        label = "unknown"
        confirmed = False
        reason = "FCC geometry gate did not pass"
    elif chemistry_fraction >= 0.85 and joint_fraction >= _L12_MIN_JOINT_FRACTION:
        label = "l12"
        confirmed = True
        reason = "FCC geometry and AB3 sublattice order both passed"
    elif chemistry_fraction >= 0.40:
        label = "l12_partial"
        confirmed = False
        reason = "FCC geometry passed but joint L1_2 local agreement is only partial"
    else:
        label = "not_l12"
        confirmed = False
        reason = "FCC geometry passed but AB3 local order did not"
    return PhaseRefinement(
        "l12",
        label,
        confirmed,
        geometry_fraction,
        chemistry_fraction,
        joint_fraction,
        1.0 - chemistry_fraction,
        a,
        b,
        reason,
    )


def refine_laves(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
    atom_types: Sequence[int],
    *,
    a_types: Iterable[int] | None = None,
    b_types: Iterable[int] | None = None,
) -> PhaseRefinement:
    """Confirm an AB2 Laves candidate and conservatively refine C14/C15."""
    pos, box, periodic, types = _as_inputs(positions, cell, pbc, atom_types)
    a, b, role_error = _resolve_roles(
        types,
        expected_a_fraction=1.0 / 3.0,
        a_types=a_types,
        b_types=b_types,
        auto_tolerance=0.035,
    )
    if role_error:
        return PhaseRefinement(
            "laves", "unknown", False, 0.0, 0.0, 0.0, 1.0, (), (), role_error
        )
    vectors, indices, valid = accelerated_periodic_knn_vectors(
        pos, box, periodic, neighbors=20
    )
    templates = _reference_templates()
    native = _phase_sketch._native_phase
    if native is not None and hasattr(native, "laves_refinement_metrics"):
        geometry_fraction, chemistry_fraction, joint_fraction, b2_fraction = (
            native.laves_refinement_metrics(
                vectors,
                indices,
                valid,
                types,
                np.asarray(a, dtype=np.int32),
                templates["z12"],
                templates["z16"],
                _LAVES_LOCAL_SHAPE_MAX_RMS,
                _B2_CSP_THRESHOLD,
            )
        )
    else:
        vectors, indices, valid = _ordered_neighbor_data(vectors, indices, valid)
        a_mask = np.isin(types, a)
        geometry_match = np.zeros(len(pos), dtype=bool)
        chemistry_match = np.zeros(len(pos), dtype=bool)
        csp_values: list[float] = []
        for atom in range(len(pos)):
            coordination = 16 if a_mask[atom] else 12
            if int(np.sum(valid[atom])) < coordination:
                continue
            descriptor = _shape_descriptor(vectors[atom, valid[atom]], coordination)
            key = "z16" if a_mask[atom] else "z12"
            geometry_match[atom] = _nearest_template_distances(
                descriptor[np.newaxis, :], templates[key]
            )[0] <= _LAVES_LOCAL_SHAPE_MAX_RMS
            current_neighbor_types = types[indices[atom, :coordination]]
            a_neighbors = int(np.sum(np.isin(current_neighbor_types, a)))
            chemistry_match[atom] = a_neighbors == (4 if a_mask[atom] else 6)
            if not a_mask[atom] and geometry_match[atom] and chemistry_match[atom]:
                valid_indices = indices[atom, valid[atom]]
                valid_vectors = vectors[atom, valid[atom]]
                b_neighbor_mask = np.isin(types[valid_indices], b)
                b_vectors = valid_vectors[b_neighbor_mask][:6]
                if len(b_vectors) == 6:
                    csp_values.append(_normalized_csp(b_vectors))
        joint = geometry_match & chemistry_match
        geometry_fraction = float(np.mean(geometry_match))
        chemistry_fraction = float(np.mean(chemistry_match))
        joint_fraction = float(np.mean(joint))
        b2_fraction = (
            float(np.mean(np.asarray(csp_values) > _B2_CSP_THRESHOLD))
            if csp_values
            else None
        )
    if (
        geometry_fraction < _LAVES_MIN_GEOMETRY_FRACTION
        or joint_fraction < _LAVES_MIN_JOINT_FRACTION
        or b2_fraction is None
    ):
        label = "unknown"
        confirmed = False
        reason = "Z12/Z16 geometry and AB2 coordination gates did not both pass"
    elif b2_fraction <= 0.12:
        label = "c15"
        confirmed = True
        reason = "Laves gates passed and B sites are predominantly B1"
    elif b2_fraction >= 0.63:
        label = "c14"
        confirmed = True
        reason = "Laves gates passed and the B1/B2 population matches C14"
    elif 0.25 <= b2_fraction <= 0.50:
        label = "c36_like_or_mixed"
        confirmed = False
        reason = "local evidence cannot distinguish C36 stacking from a C14/C15 mixture"
    else:
        label = "laves_unresolved"
        confirmed = False
        reason = "Laves gates passed but the polytype evidence is inconsistent"
    return PhaseRefinement(
        "laves",
        label,
        confirmed,
        geometry_fraction,
        chemistry_fraction,
        joint_fraction,
        1.0 - joint_fraction,
        a,
        b,
        reason,
        b2_fraction,
    )


__all__ = ["PhaseRefinement", "refine_l12", "refine_laves"]
