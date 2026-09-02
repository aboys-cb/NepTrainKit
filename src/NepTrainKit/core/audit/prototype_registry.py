"""Conservative matching for common crystallographic prototypes.

The public interface deliberately stays small:

``match_common_prototype``
    Match one fully periodic snapshot against the built-in, falsified
    prototype catalog and fail closed when evidence is weak or ambiguous.

``reference_crystallography``
    Return display metadata for a phase label already confirmed by the phase
    inventory.

Matching is scale- and rotation-invariant.  It combines local geometric
templates with species-resolved neighbor-shell occupancies; neither
stoichiometry nor a cubic-looking cell is sufficient on its own.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations

import numpy as np

from . import phase_sketch as _phase_sketch

_EPS = 1.0e-12
_MAX_NEIGHBORS = 32
_MIN_TEMPLATE_NEIGHBORS = 14
_MAX_TEMPLATE_NEIGHBORS = 32
_DESCRIPTOR_BATCH_SIZE = 2048
_COMPOSITION_TOLERANCE = 0.035
_DEFAULT_LOCAL_SHAPE_MAX_RMS = 0.105
_MIN_GEOMETRY_FRACTION = 0.82
_MIN_CHEMISTRY_FRACTION = 0.80
_MIN_JOINT_FRACTION = 0.80
_AMBIGUOUS_JOINT_MARGIN = 0.08
_AMBIGUOUS_RMS_MARGIN = 0.018
_SHELL_SPLIT_RATIO = 1.06
_MAX_SHELL_ASSIGNMENT_ERROR_FRACTION = 0.08


@dataclass(frozen=True)
class ReferenceCrystallography:
    """Crystallographic metadata of an ideal reference prototype."""

    label: str
    pearson: str
    space_group: str
    space_group_number: int
    bravais: str


@dataclass(frozen=True)
class PrototypeMatch:
    """Observable result of conservative prototype matching."""

    label: str
    confirmed: bool
    geometry_match_fraction: float
    chemistry_match_fraction: float
    joint_match_fraction: float
    mean_shape_rms: float | None
    reason: str


@dataclass(frozen=True)
class _PrototypeDefinition:
    label: str
    role_counts: tuple[int, ...]
    builder: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]
    local_shape_max_rms: float = _DEFAULT_LOCAL_SHAPE_MAX_RMS


@dataclass(frozen=True)
class _SiteTemplate:
    role: int
    shell_sizes: tuple[int, ...]
    shell_role_counts: tuple[tuple[int, ...], ...]
    descriptor: np.ndarray

    @property
    def neighbor_count(self) -> int:
        return int(sum(self.shell_sizes))


@dataclass(frozen=True)
class _PreparedPrototype:
    definition: _PrototypeDefinition
    templates_by_role: tuple[tuple[_SiteTemplate, ...], ...]
    native_templates: _NativePrototypeTemplates


@dataclass(frozen=True)
class _NativePrototypeTemplates:
    template_roles: np.ndarray
    neighbor_counts: np.ndarray
    shell_sizes: np.ndarray
    shell_role_counts: np.ndarray
    descriptors: np.ndarray


_REFERENCE_CRYSTALLOGRAPHY = {
    "fcc": ReferenceCrystallography(
        "fcc", "cF4", "Fm-3m", 225, "Face-centered cubic Bravais lattice"
    ),
    "bcc": ReferenceCrystallography(
        "bcc", "cI2", "Im-3m", 229, "Body-centered cubic Bravais lattice"
    ),
    "hcp": ReferenceCrystallography(
        "hcp", "hP2", "P6₃/mmc", 194, "Primitive hexagonal Bravais lattice"
    ),
    "diamond": ReferenceCrystallography(
        "diamond", "cF8", "Fd-3m", 227, "Face-centered cubic Bravais lattice"
    ),
    "l10": ReferenceCrystallography(
        "l10",
        "tP2",
        "P4/mmm",
        123,
        "Primitive tetragonal Bravais lattice; FCC-derived ordering",
    ),
    "l12": ReferenceCrystallography(
        "l12",
        "cP4",
        "Pm-3m",
        221,
        "Primitive cubic Bravais lattice; FCC-derived ordering",
    ),
    "b1": ReferenceCrystallography(
        "b1", "cF8", "Fm-3m", 225, "Face-centered cubic Bravais lattice"
    ),
    "b2": ReferenceCrystallography(
        "b2",
        "cP2",
        "Pm-3m",
        221,
        "Primitive cubic Bravais lattice; BCC-derived ordering",
    ),
    "b3": ReferenceCrystallography(
        "b3", "cF8", "F-43m", 216, "Face-centered cubic Bravais lattice"
    ),
    "b4": ReferenceCrystallography(
        "b4", "hP4", "P6₃mc", 186, "Primitive hexagonal Bravais lattice"
    ),
    "fluorite": ReferenceCrystallography(
        "fluorite", "cF12", "Fm-3m", 225, "Face-centered cubic Bravais lattice"
    ),
    "nias": ReferenceCrystallography(
        "nias", "hP4", "P6₃/mmc", 194, "Primitive hexagonal Bravais lattice"
    ),
    "d03": ReferenceCrystallography(
        "d03",
        "cF16",
        "Fm-3m",
        225,
        "Face-centered cubic Bravais lattice; BCC-derived ordering",
    ),
    "l21": ReferenceCrystallography(
        "l21", "cF16", "Fm-3m", 225, "Face-centered cubic Bravais lattice"
    ),
    "c1b": ReferenceCrystallography(
        "c1b", "cF12", "F-43m", 216, "Face-centered cubic Bravais lattice"
    ),
    "d019": ReferenceCrystallography(
        "d019", "hP8", "P6₃/mmc", 194, "Primitive hexagonal Bravais lattice"
    ),
    "c14": ReferenceCrystallography(
        "c14", "hP12", "P6₃/mmc", 194, "Primitive hexagonal Bravais lattice"
    ),
    "c15": ReferenceCrystallography(
        "c15", "cF24", "Fd-3m", 227, "Face-centered cubic Bravais lattice"
    ),
}


def reference_crystallography(label: str) -> ReferenceCrystallography | None:
    """Return ideal-reference metadata for a confirmed phase label."""
    return _REFERENCE_CRYSTALLOGRAPHY.get(str(label))


def _shape_descriptor(vectors: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(vectors, axis=1)
    if not len(distances) or float(np.min(distances)) <= _EPS:
        raise ValueError("prototype matching requires non-overlapping neighbors")
    scale = float(np.mean(distances))
    normalized = vectors / scale
    pairwise = np.linalg.norm(
        normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :],
        axis=2,
    )
    upper = pairwise[np.triu_indices(len(vectors), k=1)]
    return np.concatenate((np.sort(distances / scale), np.sort(upper)))


def _batch_shape_descriptors(vectors: np.ndarray) -> np.ndarray:
    """Vectorized counterpart of ``_shape_descriptor`` for hot audit paths."""
    distances = np.linalg.norm(vectors, axis=2)
    if not vectors.shape[1] or np.any(np.min(distances, axis=1) <= _EPS):
        raise ValueError("prototype matching requires non-overlapping neighbors")
    scales = np.mean(distances, axis=1)
    normalized = vectors / scales[:, np.newaxis, np.newaxis]
    pairwise = np.linalg.norm(
        normalized[:, :, np.newaxis, :] - normalized[:, np.newaxis, :, :],
        axis=3,
    )
    upper_indices = np.triu_indices(vectors.shape[1], k=1)
    upper = pairwise[:, upper_indices[0], upper_indices[1]]
    return np.concatenate(
        (
            np.sort(distances / scales[:, np.newaxis], axis=1),
            np.sort(upper, axis=1),
        ),
        axis=1,
    )


def _shape_rms_in_batches(
    sorted_vectors: np.ndarray,
    rows: np.ndarray,
    neighbor_count: int,
    reference_descriptor: np.ndarray,
) -> np.ndarray:
    """Evaluate shape RMS without materializing an N x K x K dataset tensor."""
    rms = np.empty(len(rows), dtype=float)
    for start in range(0, len(rows), _DESCRIPTOR_BATCH_SIZE):
        stop = min(start + _DESCRIPTOR_BATCH_SIZE, len(rows))
        batch_rows = rows[start:stop]
        descriptors = _batch_shape_descriptors(
            sorted_vectors[batch_rows, :neighbor_count]
        )
        difference = descriptors - reference_descriptor[np.newaxis, :]
        rms[start:stop] = np.sqrt(np.mean(difference * difference, axis=1))
    return rms


def _fcc_translations() -> np.ndarray:
    return np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ),
        dtype=float,
    )


def _translated_sites(offset: Sequence[float]) -> np.ndarray:
    return np.mod(_fcc_translations() + np.asarray(offset, dtype=float), 1.0)


def _cubic_prototype(
    sites: Sequence[tuple[int, np.ndarray]],
    *,
    a: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractional = np.concatenate([values for _role, values in sites], axis=0)
    roles = np.concatenate(
        [
            np.full(len(values), role, dtype=np.int32)
            for role, values in sites
        ]
    )
    cell = np.eye(3, dtype=float) * a
    return fractional @ cell, cell, roles


def _hexagonal_cell(a: float, c_over_a: float) -> np.ndarray:
    return np.asarray(
        (
            (0.5 * a, -0.5 * np.sqrt(3.0) * a, 0.0),
            (0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0),
            (0.0, 0.0, a * c_over_a),
        ),
        dtype=float,
    )


def _diamond_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cubic_prototype(
        (
            (0, _fcc_translations()),
            (0, _translated_sites((0.25, 0.25, 0.25))),
        ),
        a=5.43,
    )


def _l10_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = np.diag((3.82, 3.82, 3.70))
    fractional = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ),
        dtype=float,
    )
    roles = np.asarray((0, 0, 1, 1), dtype=np.int32)
    return fractional @ cell, cell, roles


def _b1_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cubic_prototype(
        (
            (0, _fcc_translations()),
            (1, _translated_sites((0.5, 0.0, 0.0))),
        ),
        a=5.64,
    )


def _b2_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = np.eye(3, dtype=float) * 4.12
    fractional = np.asarray(((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)))
    return fractional @ cell, cell, np.asarray((0, 1), dtype=np.int32)


def _b3_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cubic_prototype(
        (
            (0, _fcc_translations()),
            (1, _translated_sites((0.25, 0.25, 0.25))),
        ),
        a=5.41,
    )


def _b4_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = _hexagonal_cell(3.82, 1.633)
    u = 3.0 / 8.0
    fractional = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0 / 3.0, 1.0 / 3.0, 0.5),
            (0.0, 0.0, u),
            (2.0 / 3.0, 1.0 / 3.0, 0.5 + u),
        ),
        dtype=float,
    )
    roles = np.asarray((0, 0, 1, 1), dtype=np.int32)
    return np.mod(fractional, 1.0) @ cell, cell, roles


def _fluorite_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cubic_prototype(
        (
            (0, _fcc_translations()),
            (1, _translated_sites((0.25, 0.25, 0.25))),
            (1, _translated_sites((0.75, 0.75, 0.75))),
        ),
        a=5.46,
    )


def _nias_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = _hexagonal_cell(3.62, 1.39)
    fractional = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5),
            (1.0 / 3.0, 2.0 / 3.0, 0.25),
            (2.0 / 3.0, 1.0 / 3.0, 0.75),
        ),
        dtype=float,
    )
    roles = np.asarray((0, 0, 1, 1), dtype=np.int32)
    return fractional @ cell, cell, roles


def _d03_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cubic_prototype(
        (
            (1, _fcc_translations()),
            (0, _translated_sites((0.5, 0.5, 0.5))),
            (0, _translated_sites((0.25, 0.25, 0.25))),
            (0, _translated_sites((0.75, 0.75, 0.75))),
        ),
        a=5.78,
    )


def _l21_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cubic_prototype(
        (
            (1, _fcc_translations()),
            (2, _translated_sites((0.5, 0.5, 0.5))),
            (0, _translated_sites((0.25, 0.25, 0.25))),
            (0, _translated_sites((0.75, 0.75, 0.75))),
        ),
        a=5.95,
    )


def _c1b_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _cubic_prototype(
        (
            (0, _fcc_translations()),
            (1, _translated_sites((0.25, 0.25, 0.25))),
            (2, _translated_sites((0.5, 0.5, 0.5))),
        ),
        a=5.90,
    )


def _d019_prototype() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = _hexagonal_cell(5.30, 0.81)
    x = 5.0 / 6.0
    fractional = np.asarray(
        (
            (x, 2 * x, 0.25),
            (-2 * x, -x, 0.25),
            (x, -x, 0.25),
            (-x, -2 * x, 0.75),
            (2 * x, x, 0.75),
            (-x, x, 0.75),
            (1.0 / 3.0, 2.0 / 3.0, 0.25),
            (2.0 / 3.0, 1.0 / 3.0, 0.75),
        ),
        dtype=float,
    )
    roles = np.asarray((0, 0, 0, 0, 0, 0, 1, 1), dtype=np.int32)
    return np.mod(fractional, 1.0) @ cell, cell, roles


_DEFINITIONS = (
    _PrototypeDefinition("diamond", (1,), _diamond_prototype, 0.095),
    _PrototypeDefinition("l10", (1, 1), _l10_prototype, 0.085),
    _PrototypeDefinition("b1", (1, 1), _b1_prototype),
    _PrototypeDefinition("b2", (1, 1), _b2_prototype),
    _PrototypeDefinition("b3", (1, 1), _b3_prototype, 0.095),
    _PrototypeDefinition("b4", (1, 1), _b4_prototype, 0.095),
    _PrototypeDefinition("fluorite", (1, 2), _fluorite_prototype),
    _PrototypeDefinition("nias", (1, 1), _nias_prototype),
    _PrototypeDefinition("d03", (3, 1), _d03_prototype),
    _PrototypeDefinition("l21", (2, 1, 1), _l21_prototype),
    _PrototypeDefinition("c1b", (1, 1, 1), _c1b_prototype),
    _PrototypeDefinition("d019", (3, 1), _d019_prototype),
)


def _distance_shell_sizes(distances: np.ndarray) -> tuple[int, ...]:
    ordered = np.sort(np.asarray(distances, dtype=float))
    shell_sizes: list[int] = []
    start = 0
    for index in range(1, len(ordered) + 1):
        at_end = index == len(ordered)
        split = (
            not at_end
            and ordered[index] / max(ordered[index - 1], _EPS)
            > _SHELL_SPLIT_RATIO
        )
        if not at_end and not split:
            continue
        shell_size = index - start
        if index > _MAX_TEMPLATE_NEIGHBORS:
            break
        shell_sizes.append(shell_size)
        start = index
        if index >= _MIN_TEMPLATE_NEIGHBORS:
            break
    if sum(shell_sizes) < _MIN_TEMPLATE_NEIGHBORS:
        raise ValueError("reference prototype has insufficient complete neighbor shells")
    return tuple(shell_sizes)


def _site_template(
    vectors: np.ndarray,
    indices: np.ndarray,
    valid: np.ndarray,
    roles: np.ndarray,
    atom: int,
    role_count: int,
) -> _SiteTemplate:
    current_vectors = vectors[atom, valid[atom]]
    current_indices = indices[atom, valid[atom]]
    order = np.argsort(np.linalg.norm(current_vectors, axis=1), kind="stable")
    current_vectors = current_vectors[order]
    current_indices = current_indices[order]
    distances = np.linalg.norm(current_vectors, axis=1)
    shell_sizes = _distance_shell_sizes(distances)
    neighbor_count = sum(shell_sizes)
    descriptor = _shape_descriptor(current_vectors[:neighbor_count])
    shell_role_counts: list[tuple[int, ...]] = []
    start = 0
    for size in shell_sizes:
        shell_roles = roles[current_indices[start : start + size]]
        shell_role_counts.append(
            tuple(int(np.sum(shell_roles == role)) for role in range(role_count))
        )
        start += size
    return _SiteTemplate(
        role=int(roles[atom]),
        shell_sizes=shell_sizes,
        shell_role_counts=tuple(shell_role_counts),
        descriptor=descriptor,
    )


def _template_key(template: _SiteTemplate) -> tuple[object, ...]:
    return (
        template.role,
        template.shell_sizes,
        template.shell_role_counts,
        tuple(np.round(template.descriptor, decimals=9)),
    )


def _pack_native_templates(
    templates_by_role: tuple[tuple[_SiteTemplate, ...], ...],
) -> _NativePrototypeTemplates:
    flattened = tuple(
        (role, template)
        for role, templates in enumerate(templates_by_role)
        for template in templates
    )
    shell_capacity = max(len(template.shell_sizes) for _role, template in flattened)
    descriptor_capacity = max(
        len(template.descriptor) for _role, template in flattened
    )
    role_count = len(templates_by_role)
    template_roles = np.empty(len(flattened), dtype=np.int32)
    neighbor_counts = np.empty(len(flattened), dtype=np.int32)
    shell_sizes = np.zeros((len(flattened), shell_capacity), dtype=np.int32)
    shell_role_counts = np.zeros(
        (len(flattened), shell_capacity, role_count),
        dtype=np.int32,
    )
    descriptors = np.zeros(
        (len(flattened), descriptor_capacity),
        dtype=np.float64,
    )
    for row, (role, template) in enumerate(flattened):
        template_roles[row] = role
        neighbor_counts[row] = template.neighbor_count
        shell_sizes[row, : len(template.shell_sizes)] = template.shell_sizes
        shell_role_counts[row, : len(template.shell_role_counts)] = (
            template.shell_role_counts
        )
        descriptors[row, : len(template.descriptor)] = template.descriptor
    return _NativePrototypeTemplates(
        template_roles=template_roles,
        neighbor_counts=neighbor_counts,
        shell_sizes=shell_sizes,
        shell_role_counts=shell_role_counts,
        descriptors=descriptors,
    )


@lru_cache(maxsize=1)
def _prepared_prototypes() -> tuple[_PreparedPrototype, ...]:
    prepared: list[_PreparedPrototype] = []
    for definition in _DEFINITIONS:
        positions, cell, roles = definition.builder()
        vectors, indices, valid = _phase_sketch.accelerated_periodic_knn_vectors(
            positions,
            cell,
            (True, True, True),
            neighbors=_MAX_NEIGHBORS,
        )
        role_count = len(definition.role_counts)
        by_role: list[list[_SiteTemplate]] = [[] for _ in range(role_count)]
        seen: set[tuple[object, ...]] = set()
        for atom in range(len(positions)):
            template = _site_template(
                vectors, indices, valid, roles, atom, role_count
            )
            key = _template_key(template)
            if key in seen:
                continue
            seen.add(key)
            by_role[template.role].append(template)
        if any(not templates for templates in by_role):
            raise RuntimeError(f"prototype {definition.label} has an empty site role")
        templates_by_role = tuple(tuple(values) for values in by_role)
        prepared.append(
            _PreparedPrototype(
                definition=definition,
                templates_by_role=templates_by_role,
                native_templates=_pack_native_templates(templates_by_role),
            )
        )
    return tuple(prepared)


def _candidate_mappings(
    atom_types: np.ndarray,
    role_counts: tuple[int, ...],
) -> tuple[dict[int, int], ...]:
    present, counts = np.unique(atom_types, return_counts=True)
    if len(present) != len(role_counts):
        return ()
    actual = counts.astype(float) / float(np.sum(counts))
    expected = np.asarray(role_counts, dtype=float)
    expected /= float(np.sum(expected))
    mappings: list[dict[int, int]] = []
    for role_order in permutations(range(len(role_counts))):
        if any(
            abs(float(actual[index]) - float(expected[role])) > _COMPOSITION_TOLERANCE
            for index, role in enumerate(role_order)
        ):
            continue
        mappings.append(
            {int(present[index]): int(role) for index, role in enumerate(role_order)}
        )
    return tuple(mappings)


def _prototype_match_from_metrics(
    prepared: _PreparedPrototype,
    geometry_fraction: float,
    chemistry_fraction: float,
    joint_fraction: float,
    mean_rms: float | None,
) -> PrototypeMatch:
    confirmed = (
        geometry_fraction >= _MIN_GEOMETRY_FRACTION
        and chemistry_fraction >= _MIN_CHEMISTRY_FRACTION
        and joint_fraction >= _MIN_JOINT_FRACTION
    )
    reason = (
        "local geometry and species-resolved shell occupancies passed"
        if confirmed
        else "geometry and chemistry gates did not both pass"
    )
    return PrototypeMatch(
        label=prepared.definition.label,
        confirmed=confirmed,
        geometry_match_fraction=geometry_fraction,
        chemistry_match_fraction=chemistry_fraction,
        joint_match_fraction=joint_fraction,
        mean_shape_rms=mean_rms,
        reason=reason,
    )


def _match_mapping_python(
    prepared: _PreparedPrototype,
    sorted_vectors: np.ndarray,
    sorted_indices: np.ndarray,
    sorted_valid: np.ndarray,
    atom_types: np.ndarray,
    mapping: dict[int, int],
) -> PrototypeMatch:
    mapped_roles = np.asarray([mapping[int(value)] for value in atom_types], dtype=np.int8)
    geometry_matches = np.zeros(len(atom_types), dtype=bool)
    chemistry_matches = np.zeros(len(atom_types), dtype=bool)
    rms_values = np.full(len(atom_types), np.inf, dtype=float)
    for role, templates in enumerate(prepared.templates_by_role):
        rows = np.flatnonzero(mapped_roles == role)
        if not len(rows):
            continue
        best_score = np.full(len(rows), -1, dtype=np.int8)
        best_geometry = np.zeros(len(rows), dtype=bool)
        best_chemistry = np.zeros(len(rows), dtype=bool)
        best_rms = np.full(len(rows), np.inf, dtype=float)
        for template in templates:
            neighbor_count = template.neighbor_count
            eligible = np.all(sorted_valid[rows, :neighbor_count], axis=1)
            current_chemistry = np.zeros(len(rows), dtype=bool)
            current_rms = np.full(len(rows), np.inf, dtype=float)
            current_geometry = np.zeros(len(rows), dtype=bool)
            eligible_indices = np.flatnonzero(eligible)
            if len(eligible_indices):
                eligible_rows = rows[eligible_indices]
                shell_errors = np.zeros(len(eligible_rows), dtype=np.int16)
                start = 0
                for size, expected_counts in zip(
                    template.shell_sizes,
                    template.shell_role_counts,
                ):
                    neighbor_roles = mapped_roles[
                        sorted_indices[eligible_rows, start : start + size]
                    ]
                    observed = np.stack(
                        [
                            np.sum(neighbor_roles == neighbor_role, axis=1)
                            for neighbor_role in range(len(expected_counts))
                        ],
                        axis=1,
                    )
                    shell_errors += (
                        np.sum(
                            np.abs(
                                observed
                                - np.asarray(expected_counts)[np.newaxis, :]
                            ),
                            axis=1,
                        )
                        // 2
                    ).astype(np.int16)
                    start += size
                allowed_errors = max(
                    1,
                    int(
                        np.floor(
                            neighbor_count
                            * _MAX_SHELL_ASSIGNMENT_ERROR_FRACTION
                        )
                    ),
                )
                chemistry_local = shell_errors <= allowed_errors
                chemistry_indices = eligible_indices[chemistry_local]
                current_chemistry[chemistry_indices] = True
                if len(chemistry_indices):
                    chemistry_rows = rows[chemistry_indices]
                    rms = _shape_rms_in_batches(
                        sorted_vectors,
                        chemistry_rows,
                        neighbor_count,
                        template.descriptor,
                    )
                    current_rms[chemistry_indices] = rms
                    current_geometry[chemistry_indices] = (
                        rms <= prepared.definition.local_shape_max_rms
                    )
            current_joint = current_geometry & current_chemistry
            current_score = (
                current_joint.astype(np.int8) * 4
                + current_chemistry.astype(np.int8) * 2
                + current_geometry.astype(np.int8)
            )
            replace = (current_score > best_score) | (
                (current_score == best_score) & (current_rms < best_rms)
            )
            best_score[replace] = current_score[replace]
            best_geometry[replace] = current_geometry[replace]
            best_chemistry[replace] = current_chemistry[replace]
            best_rms[replace] = current_rms[replace]
        geometry_matches[rows] = best_geometry
        chemistry_matches[rows] = best_chemistry
        rms_values[rows] = best_rms
    joint = geometry_matches & chemistry_matches
    geometry_fraction = float(np.mean(geometry_matches))
    chemistry_fraction = float(np.mean(chemistry_matches))
    joint_fraction = float(np.mean(joint))
    finite_rms = rms_values[np.isfinite(rms_values)]
    mean_rms = float(np.mean(finite_rms)) if len(finite_rms) else None
    return _prototype_match_from_metrics(
        prepared,
        geometry_fraction,
        chemistry_fraction,
        joint_fraction,
        mean_rms,
    )


def _match_mapping(
    prepared: _PreparedPrototype,
    sorted_vectors: np.ndarray,
    sorted_indices: np.ndarray,
    sorted_valid: np.ndarray,
    atom_types: np.ndarray,
    mapping: dict[int, int],
) -> PrototypeMatch:
    native = _phase_sketch._native_phase
    if native is None or not hasattr(
        native,
        "common_prototype_mapping_metrics",
    ):
        return _match_mapping_python(
            prepared,
            sorted_vectors,
            sorted_indices,
            sorted_valid,
            atom_types,
            mapping,
        )

    mapped_roles = np.asarray(
        [mapping[int(value)] for value in atom_types],
        dtype=np.int32,
    )
    templates = prepared.native_templates
    geometry_fraction, chemistry_fraction, joint_fraction, mean_rms = (
        native.common_prototype_mapping_metrics(
            sorted_vectors,
            sorted_indices,
            sorted_valid,
            mapped_roles,
            templates.template_roles,
            templates.neighbor_counts,
            templates.shell_sizes,
            templates.shell_role_counts,
            templates.descriptors,
            float(prepared.definition.local_shape_max_rms),
            float(_MAX_SHELL_ASSIGNMENT_ERROR_FRACTION),
        )
    )
    return _prototype_match_from_metrics(
        prepared,
        float(geometry_fraction),
        float(chemistry_fraction),
        float(joint_fraction),
        None if mean_rms is None else float(mean_rms),
    )


def _better(left: PrototypeMatch, right: PrototypeMatch) -> PrototypeMatch:
    return max(
        (left, right),
        key=lambda item: (
            item.joint_match_fraction,
            item.chemistry_match_fraction,
            item.geometry_match_fraction,
            -(float("inf") if item.mean_shape_rms is None else item.mean_shape_rms),
        ),
    )


def match_common_prototype(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
    atom_types: Sequence[int],
    *,
    candidate_labels: Sequence[str] | None = None,
    _neighbor_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> PrototypeMatch:
    """Match a fully periodic structure against common crystal prototypes.

    Unknown, partially periodic, composition-incompatible, weak, and
    near-degenerate candidates all return ``confirmed=False``.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    box = np.ascontiguousarray(cell, dtype=np.float64).reshape(3, 3)
    periodic = np.asarray(pbc, dtype=bool).reshape(3)
    types = np.asarray(atom_types, dtype=np.int32).reshape(-1)
    if pos.ndim != 2 or pos.shape != (len(types), 3) or not len(pos):
        raise ValueError("positions must be a non-empty N x 3 array matching atom_types")
    unresolved = PrototypeMatch(
        "unresolved", False, 0.0, 0.0, 0.0, None, "no supported prototype passed"
    )
    if not np.all(periodic):
        return PrototypeMatch(
            "unresolved",
            False,
            0.0,
            0.0,
            0.0,
            None,
            "bulk prototype matching requires three-dimensional periodicity",
        )

    allowed_labels = (
        None
        if candidate_labels is None
        else {str(label) for label in candidate_labels}
    )
    if allowed_labels == set():
        return unresolved
    prepared_candidates: list[tuple[_PreparedPrototype, tuple[dict[int, int], ...]]] = []
    for prepared in _prepared_prototypes():
        if (
            allowed_labels is not None
            and prepared.definition.label not in allowed_labels
        ):
            continue
        mappings = _candidate_mappings(types, prepared.definition.role_counts)
        if mappings:
            prepared_candidates.append((prepared, mappings))
    if not prepared_candidates:
        return unresolved

    if _neighbor_data is None:
        vectors, indices, valid = _phase_sketch.accelerated_periodic_knn_vectors(
            pos,
            box,
            periodic,
            neighbors=_MAX_NEIGHBORS,
        )
    else:
        vectors, indices, valid = _neighbor_data
    distances = np.where(valid, np.linalg.norm(vectors, axis=2), np.inf)
    order = np.argsort(distances, axis=1, kind="stable")
    sorted_vectors = np.take_along_axis(vectors, order[:, :, np.newaxis], axis=1)
    sorted_indices = np.take_along_axis(indices, order, axis=1)
    sorted_valid = np.take_along_axis(valid, order, axis=1)
    per_label: dict[str, PrototypeMatch] = {}
    for prepared, mappings in prepared_candidates:
        for mapping in mappings:
            result = _match_mapping(
                prepared,
                sorted_vectors,
                sorted_indices,
                sorted_valid,
                types,
                mapping,
            )
            previous = per_label.get(result.label)
            per_label[result.label] = (
                result if previous is None else _better(previous, result)
            )
    confirmed = sorted(
        (result for result in per_label.values() if result.confirmed),
        key=lambda item: (
            float("inf") if item.mean_shape_rms is None else item.mean_shape_rms,
            -item.joint_match_fraction,
            -item.chemistry_match_fraction,
            -item.geometry_match_fraction,
            item.label,
        ),
    )
    if not confirmed:
        return unresolved
    if len(confirmed) == 1:
        return confirmed[0]
    first, second = confirmed[:2]
    first_rms = float("inf") if first.mean_shape_rms is None else first.mean_shape_rms
    second_rms = float("inf") if second.mean_shape_rms is None else second.mean_shape_rms
    if second_rms - first_rms >= _AMBIGUOUS_RMS_MARGIN:
        return first
    if (
        first.joint_match_fraction - second.joint_match_fraction
        >= _AMBIGUOUS_JOINT_MARGIN
    ):
        return first
    return PrototypeMatch(
        "unresolved",
        False,
        first.geometry_match_fraction,
        first.chemistry_match_fraction,
        first.joint_match_fraction,
        first.mean_shape_rms,
        f"ambiguous between {first.label} and {second.label}",
    )


__all__ = [
    "PrototypeMatch",
    "ReferenceCrystallography",
    "match_common_prototype",
    "reference_crystallography",
]
