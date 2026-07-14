"""Deterministic technical data-quality checks for training structures."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from NepTrainKit.core.structure import Structure, atomic_numbers

from .neighbor_scan import find_short_distance_structure_rows, periodic_cell_statuses
from .result import (
    AuditBiasType,
    AuditConfidence,
    AuditDimension,
    AuditFindingKind,
    AuditSeverity,
    AuditSlice,
    AuditStatus,
    SliceMetric,
)


SHORT_DISTANCE_ANGSTROM = 0.5
GEOMETRY_ROUND_DECIMALS = 8
LABEL_RTOL = 0.0
LABEL_ATOL = {
    "energy": 1.0e-5,
    "forces": 1.0e-5,
    "virial": 1.0e-4,
}


def _pbc_flags(structure: Structure) -> tuple[np.ndarray, bool]:
    value = getattr(structure, "additional_fields", {}).get("pbc", "T T T")
    if isinstance(value, str):
        tokens = value.replace(",", " ").split()
        parsed: list[bool] = []
        for token in tokens:
            lowered = token.strip().lower()
            if lowered in {"t", "true", "1"}:
                parsed.append(True)
            elif lowered in {"f", "false", "0"}:
                parsed.append(False)
            else:
                return np.asarray([True, True, True]), False
        if len(parsed) == 1:
            parsed *= 3
        if len(parsed) != 3:
            return np.asarray([True, True, True]), False
        return np.asarray(parsed, dtype=bool), True
    try:
        array = np.asarray(value, dtype=bool).reshape(-1)
    except Exception:
        return np.asarray([True, True, True]), False
    if array.size == 1:
        array = np.repeat(array, 3)
    if array.size != 3:
        return np.asarray([True, True, True]), False
    return array.astype(bool, copy=False), True


def _known_elements(structure: Structure, atom_count: int) -> tuple[tuple[str, ...], bool]:
    try:
        elements = tuple(str(element) for element in structure.elements)
    except Exception:
        return (), False
    return elements, len(elements) == atom_count and all(element in atomic_numbers for element in elements)


def _label_arrays(
    structure: Structure,
    atom_count: int,
    *,
    skip_finite: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], bool, bool]:
    labels: dict[str, np.ndarray] = {}
    shape_ok = True
    finite = True
    skipped = skip_finite or set()

    if bool(getattr(structure, "has_energy", False)):
        try:
            energy = np.asarray(structure.energy, dtype=np.float64)
            if energy.size != 1:
                shape_ok = False
            else:
                labels["energy"] = energy.reshape(1)
                if "energy" not in skipped:
                    finite = finite and bool(np.all(np.isfinite(energy)))
        except Exception:
            shape_ok = False

    if bool(getattr(structure, "has_forces", False)):
        try:
            forces = np.asarray(structure.forces, dtype=np.float64)
            if forces.shape != (atom_count, 3):
                shape_ok = False
            else:
                labels["forces"] = forces
                if "forces" not in skipped:
                    finite = finite and bool(np.all(np.isfinite(forces)))
        except Exception:
            shape_ok = False

    if bool(getattr(structure, "has_virial", False)):
        try:
            virial = np.asarray(structure.virial, dtype=np.float64)
            if virial.shape not in {(6,), (9,), (3, 3)}:
                shape_ok = False
            else:
                labels["virial"] = virial.reshape(-1)
                if "virial" not in skipped:
                    finite = finite and bool(np.all(np.isfinite(virial)))
        except Exception:
            shape_ok = False

    return labels, shape_ok, finite


def _reference_label_status(
    result_data: object | None,
    source_indices: Sequence[int],
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Collect coverage and non-finite rows from materialized reference arrays."""
    covered: dict[str, set[int]] = defaultdict(set)
    nonfinite: dict[str, set[int]] = defaultdict(set)
    if result_data is None:
        return covered, nonfinite

    datasets: dict[str, object | None] = {}
    for label, attribute in (
        ("energy", "energy"),
        ("forces", "_force_vector_dataset"),
        ("virial", "virial"),
    ):
        try:
            datasets[label] = getattr(result_data, attribute, None)
        except Exception:
            datasets[label] = None
    expected_columns = {"energy": 1, "forces": 3, "virial": 6}
    scope = np.asarray(tuple(source_indices), dtype=np.int64)
    for label, dataset in datasets.items():
        if dataset is None:
            continue
        try:
            data = np.asarray(dataset.all_data)
            groups = np.asarray(dataset.group_array.all_data, dtype=np.int64).reshape(-1)
            reference = np.asarray(data[:, dataset.x_cols])
        except Exception:
            continue
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        if (
            reference.ndim != 2
            or reference.shape[0] != groups.size
            or reference.shape[1] != expected_columns[label]
        ):
            continue
        in_scope = np.isin(groups, scope)
        scoped_groups = groups[in_scope]
        if scoped_groups.size == 0:
            continue
        covered[label].update(int(index) for index in np.unique(scoped_groups))
        bad_rows = ~np.all(np.isfinite(reference[in_scope]), axis=1)
        if np.any(bad_rows):
            nonfinite[label].update(int(index) for index in np.unique(scoped_groups[bad_rows]))
    return covered, nonfinite


def _geometry_key(
    elements: tuple[str, ...],
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> tuple[Any, ...]:
    rounded_positions = np.round(positions, decimals=GEOMETRY_ROUND_DECIMALS)
    rounded_cell = np.round(cell, decimals=GEOMETRY_ROUND_DECIMALS)
    return (
        elements,
        tuple(bool(item) for item in pbc),
        rounded_cell.tobytes(),
        rounded_positions.tobytes(),
    )


def _conflicting_label_indices(
    group: Sequence[int],
    labels_by_index: Mapping[int, Mapping[str, np.ndarray]],
) -> set[int]:
    conflicts: set[int] = set()
    for label in ("energy", "forces", "virial"):
        available = [index for index in group if label in labels_by_index.get(index, {})]
        if len(available) < 2:
            continue
        reference_index = available[0]
        reference = labels_by_index[reference_index][label]
        if not np.all(np.isfinite(reference)):
            continue
        for index in available[1:]:
            value = labels_by_index[index][label]
            if value.shape != reference.shape or not np.all(np.isfinite(value)):
                continue
            if not np.allclose(
                value,
                reference,
                rtol=LABEL_RTOL,
                atol=LABEL_ATOL[label],
            ):
                conflicts.update((reference_index, index))
    return conflicts


def _finding_slice(
    *,
    id: str,
    title: str,
    indices: set[int],
    observed: str,
    interpretation: str,
    rule: str,
    limit: str,
    kind: AuditFindingKind = AuditFindingKind.BLOCKER,
    bias_type: AuditBiasType = AuditBiasType.RISK_CONCENTRATION,
) -> AuditSlice:
    ordered = tuple(sorted(indices))
    return AuditSlice(
        id=id,
        title=title,
        dimension_id="data_quality",
        severity=AuditSeverity.HIGH if kind == AuditFindingKind.BLOCKER else AuditSeverity.INFO,
        bias_type=bias_type,
        structure_indices=ordered,
        observed=observed,
        interpretation=interpretation,
        limit=limit,
        metrics=(SliceMetric("affected_structures", len(ordered), "structures"),),
        finding_kind=kind,
        rule=rule,
        confidence=AuditConfidence.DIRECT,
    )


def audit_data_quality(
    indexed_structures: Sequence[tuple[int, Structure]],
    *,
    result_data: object | None = None,
) -> tuple[AuditDimension, tuple[AuditSlice, ...], dict[str, object]]:
    """Check technical data contracts without inferring target coverage."""
    if not indexed_structures:
        return (
            AuditDimension(
                "data_quality",
                "Data quality",
                AuditStatus.UNAVAILABLE,
                "No structures are loaded.",
            ),
            (),
            {"blocker_count": 0, "review_count": 0},
        )

    issues: dict[str, set[int]] = defaultdict(set)
    geometry_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    labels_by_index: dict[int, Mapping[str, np.ndarray]] = {}
    geometry_source_indices: list[int] = []
    geometry_positions: list[np.ndarray] = []
    geometry_cells: list[np.ndarray] = []
    geometry_pbc: list[np.ndarray] = []
    source_indices = [source_index for source_index, _ in indexed_structures]
    label_coverage, batch_nonfinite = _reference_label_status(result_data, source_indices)

    prepared_geometry: list[tuple[int, Structure, np.ndarray, bool, np.ndarray, np.ndarray, bool]] = []
    for source_index, structure in indexed_structures:
        try:
            positions = np.asarray(structure.atomic_properties.get("pos"), dtype=np.float64)
        except Exception:
            positions = np.empty((0, 3), dtype=np.float64)
        position_shape_ok = positions.ndim == 2 and positions.shape[1:] == (3,)
        try:
            cell = np.asarray(structure.lattice, dtype=np.float64)
        except Exception:
            cell = np.empty((0, 0), dtype=np.float64)
        pbc, pbc_ok = _pbc_flags(structure)
        prepared_geometry.append(
            (source_index, structure, positions, position_shape_ok, cell, pbc, pbc_ok)
        )
    cell_status = periodic_cell_statuses(
        [item[4] for item in prepared_geometry],
        [item[5] for item in prepared_geometry],
    )

    for prepared_row, prepared in enumerate(prepared_geometry):
        source_index, structure, positions, position_shape_ok, cell, pbc, pbc_ok = prepared
        atom_count = int(positions.shape[0]) if position_shape_ok else 0
        geometry_finite = bool(
            position_shape_ok
            and np.all(np.isfinite(positions))
            and np.all(np.isfinite(cell))
        )
        cell_ok = bool(cell_status[prepared_row] & 1)

        if not geometry_finite:
            issues["nonfinite_geometry"].add(source_index)
        if atom_count <= 0:
            issues["empty_structure"].add(source_index)
        if not pbc_ok:
            issues["invalid_pbc"].add(source_index)
        if not cell_ok:
            issues["invalid_cell"].add(source_index)

        elements, elements_ok = _known_elements(structure, atom_count)
        if not elements_ok:
            issues["unknown_elements"].add(source_index)

        finite_from_batch = {
            label
            for label, covered_indices in label_coverage.items()
            if source_index in covered_indices
        }
        labels, label_shape_ok, labels_finite = _label_arrays(
            structure,
            atom_count,
            skip_finite=finite_from_batch,
        )
        labels_by_index[source_index] = labels
        if not label_shape_ok:
            issues["invalid_label_shape"].add(source_index)
        batch_has_nonfinite = any(
            label in labels and source_index in batch_nonfinite.get(label, ())
            for label in finite_from_batch
        )
        if not labels_finite or batch_has_nonfinite:
            issues["nonfinite_labels"].add(source_index)

        geometry_ok = (
            geometry_finite
            and cell_ok
            and pbc_ok
            and elements_ok
        )
        if not geometry_ok:
            continue
        geometry_source_indices.append(source_index)
        geometry_positions.append(positions)
        geometry_cells.append(cell)
        geometry_pbc.append(pbc)
        geometry_groups[_geometry_key(elements, positions, cell, pbc)].append(source_index)

    try:
        short_distance_rows = find_short_distance_structure_rows(
            geometry_positions,
            geometry_cells,
            geometry_pbc,
            SHORT_DISTANCE_ANGSTROM,
        )
        issues["short_distance"].update(
            geometry_source_indices[row] for row in short_distance_rows
        )
    except Exception:
        issues["short_distance_unavailable"].update(geometry_source_indices)

    duplicate_groups = [group for group in geometry_groups.values() if len(group) > 1]
    duplicate_indices = {index for group in duplicate_groups for index in group}
    conflict_indices: set[int] = set()
    for group in duplicate_groups:
        conflict_indices.update(_conflicting_label_indices(group, labels_by_index))

    slices: list[AuditSlice] = []
    definitions = (
        (
            "empty_structure",
            "Empty structures",
            "the structure contains no atoms",
            "A zero-atom frame cannot provide an atomic training example.",
            "Every structure in the training scope must contain at least one atom.",
            "This check does not impose a minimum cell size or composition.",
        ),
        (
            "nonfinite_geometry",
            "Non-finite geometry values",
            "position or cell arrays contain NaN/Inf or have an invalid position shape",
            "Training and neighbor calculations cannot safely consume non-finite geometry.",
            "Positions must have shape (N, 3), and positions/cell values must be finite.",
            "This check does not judge whether a finite geometry is physically meaningful.",
        ),
        (
            "invalid_pbc",
            "Invalid periodic-boundary metadata",
            "the pbc field cannot be interpreted as three boolean directions",
            "Periodic geometry operations need an unambiguous pbc definition.",
            "pbc must contain one or three boolean-like values (T/F, true/false, or 1/0).",
            "Missing pbc uses the existing NepTrainKit default and is not flagged.",
        ),
        (
            "invalid_cell",
            "Invalid periodic cell",
            "periodic lattice vectors are non-finite, zero length, or linearly dependent",
            "Minimum-image geometry is undefined for the declared periodic directions.",
            "Periodic lattice vectors must be finite, non-zero, and linearly independent.",
            "Non-periodic directions are not required to span a 3D volume.",
        ),
        (
            "unknown_elements",
            "Invalid element information",
            "element symbols are unknown or do not match the atom count",
            "A training backend cannot map these atoms to a valid element type.",
            "Every atom must have one chemical symbol recognized by NepTrainKit.",
            "This does not check whether the model itself supports the valid elements.",
        ),
        (
            "invalid_label_shape",
            "Invalid label shape",
            "energy, force, or virial labels do not match their required shape",
            "Mismatched labels can be assigned to the wrong atoms or rejected by training.",
            "Energy is scalar, forces are (N, 3), and virial/stress has 6 or 9 components.",
            "Missing labels are handled separately and are not automatically invalid.",
        ),
        (
            "nonfinite_labels",
            "Non-finite label values",
            "energy, force, or virial labels contain NaN/Inf",
            "Non-finite targets make common training losses non-finite.",
            "All present energy, force, and virial/stress labels must be finite.",
            "The check does not require every supported label type to be present.",
        ),
        (
            "short_distance",
            "Overlapping atoms",
            f"at least one atom pair is closer than {SHORT_DISTANCE_ANGSTROM:g} Å",
            "Such a distance is a conservative technical collision signal and should be inspected before training.",
            f"Flag structures with any pair distance below {SHORT_DISTANCE_ANGSTROM:g} Å using declared PBC.",
            "Some specialized collision datasets may intentionally contain very short distances; review provenance before removal.",
        ),
    )
    for key, title, fact, interpretation, rule, limit in definitions:
        indices = issues.get(key, set())
        if not indices:
            continue
        slices.append(
            _finding_slice(
                id=f"data_quality:{key}",
                title=title,
                indices=indices,
                observed=f"{len(indices)} structures fail this data contract: {fact}.",
                interpretation=interpretation,
                rule=rule,
                limit=limit,
            )
        )

    if conflict_indices:
        slices.append(
            _finding_slice(
                id="data_quality:label_conflicts",
                title="Duplicate geometries with conflicting labels",
                indices=conflict_indices,
                observed=(
                    f"{len(conflict_indices)} structures share geometry but have energy, force, "
                    "or virial labels that disagree beyond tolerance."
                ),
                interpretation=(
                    "The same input geometry maps to inconsistent training targets and needs provenance review."
                ),
                rule=(
                    f"Geometry equal after {GEOMETRY_ROUND_DECIMALS} decimal rounding; common labels compared "
                    f"with absolute tolerances {LABEL_ATOL} and rtol={LABEL_RTOL:g}."
                ),
                limit=(
                    "Legitimate repeated calculations can differ; this finding does not choose which label is correct."
                ),
            )
        )

    if duplicate_indices:
        slices.append(
            _finding_slice(
                id="data_quality:exact_duplicates",
                title="Repeated geometries",
                indices=duplicate_indices,
                observed=(
                    f"{len(duplicate_indices)} structures belong to {len(duplicate_groups)} repeated-geometry groups."
                ),
                interpretation=(
                    "Repeated geometries may overweight a configuration and deserve a provenance or weighting check."
                ),
                rule=(
                    f"Element order, PBC, cell, and Cartesian positions match after "
                    f"{GEOMETRY_ROUND_DECIMALS} decimal rounding."
                ),
                limit=(
                    "Repeated structures may be intentional for weighting or independent-label studies; this is not a delete instruction."
                ),
                kind=AuditFindingKind.REVIEW,
                bias_type=AuditBiasType.REDUNDANCY,
            )
        )

    blocker_count = sum(item.finding_kind == AuditFindingKind.BLOCKER for item in slices)
    review_count = sum(item.finding_kind == AuditFindingKind.REVIEW for item in slices)
    reason = ""
    if issues.get("short_distance_unavailable"):
        reason = (
            f"Short-distance checking was unavailable for {len(issues['short_distance_unavailable'])} structures."
        )
    return (
        AuditDimension(
            "data_quality",
            "Data quality",
            AuditStatus.PARTIAL if reason else AuditStatus.AVAILABLE,
            reason,
        ),
        tuple(slices),
        {
            "blocker_count": blocker_count,
            "review_count": review_count,
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_structure_count": len(duplicate_indices),
            "duplicate_groups": tuple(tuple(sorted(group)) for group in duplicate_groups),
            "affected_structure_count": len(
                {index for item in slices for index in item.structure_indices}
            ),
        },
    )
