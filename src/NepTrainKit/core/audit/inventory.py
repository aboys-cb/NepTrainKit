"""Dataset inventory and explicit target-comparison logic for Training Set Audit."""
from __future__ import annotations

from collections import Counter, defaultdict
from math import gcd
from typing import Sequence

from .extract import StructureAuditRecord
from .result import (
    CompositionPoint,
    CompositionTarget,
    DatasetInventory,
    TargetSupportCell,
    TargetSupportStatus,
)


def _reduced_counts(record: StructureAuditRecord, elements: tuple[str, ...]) -> tuple[int, ...]:
    counts = tuple(
        max(0, int(round(float(record.composition.get(element, 0.0)) * record.num_atoms)))
        for element in elements
    )
    divisor = 0
    for count in counts:
        divisor = gcd(divisor, count)
    if divisor <= 0:
        return counts
    return tuple(count // divisor for count in counts)


def build_dataset_inventory(records: Sequence[StructureAuditRecord]) -> DatasetInventory:
    """Aggregate exact normalized compositions and their drill-down dimensions."""
    elements = tuple(sorted({element for record in records for element in record.composition}))
    total = len(records)
    if not records or not elements:
        return DatasetInventory(total, elements, ())

    grouped: dict[tuple[int, ...], list[StructureAuditRecord]] = defaultdict(list)
    atom_counts = Counter[int]()
    config_types = Counter[str]()
    missing_config_type_count = 0
    for record in records:
        grouped[_reduced_counts(record, elements)].append(record)
        atom_counts[int(record.num_atoms)] += 1
        config_type = str(record.config_type or "").strip()
        if config_type:
            config_types[config_type] += 1
        else:
            missing_config_type_count += 1

    points: list[CompositionPoint] = []
    for reduced_counts, group in grouped.items():
        denominator = sum(reduced_counts)
        fractions = tuple(
            0.0 if denominator <= 0 else count / denominator for count in reduced_counts
        )
        point_atom_counts = Counter(int(record.num_atoms) for record in group)
        formulas = Counter(str(record.formula) for record in group)
        point_config_types = Counter(
            str(record.config_type).strip()
            for record in group
            if str(record.config_type or "").strip()
        )
        point_config_type_indices: dict[str, list[int]] = defaultdict(list)
        for record in group:
            config_type = str(record.config_type or "").strip()
            if config_type:
                point_config_type_indices[config_type].append(int(record.index))
        point_missing_config_type_count = sum(
            not str(record.config_type or "").strip() for record in group
        )
        indices = tuple(sorted(int(record.index) for record in group))
        points.append(
            CompositionPoint(
                reduced_counts=reduced_counts,
                fractions=fractions,
                structure_count=len(group),
                share=len(group) / total,
                structure_indices=indices,
                atom_counts=tuple(sorted(point_atom_counts.items())),
                formula_variants=tuple(
                    sorted(formulas.items(), key=lambda item: (-item[1], item[0]))
                ),
                config_types=tuple(
                    sorted(point_config_types.items(), key=lambda item: (-item[1], item[0]))
                ),
                config_type_indices=tuple(
                    (name, tuple(sorted(item_indices)))
                    for name, item_indices in sorted(point_config_type_indices.items())
                ),
                missing_config_type_count=point_missing_config_type_count,
            )
        )

    points.sort(key=lambda point: point.fractions)
    return DatasetInventory(
        structure_count=total,
        elements=elements,
        composition_points=tuple(points),
        atom_counts=tuple(sorted(atom_counts.items())),
        config_types=tuple(
            sorted(config_types.items(), key=lambda item: (-item[1], item[0]))
        ),
        missing_config_type_count=missing_config_type_count,
    )


def compare_composition_target(
    inventory: DatasetInventory,
    target: CompositionTarget,
    *,
    atol: float = 1.0e-8,
) -> tuple[TargetSupportCell, ...]:
    """Compare explicit target points with exact inventory counts.

    The status describes only the user-confirmed structure-count rule.  It is
    not a statement about descriptor coverage or independent information.
    """
    if target.element not in inventory.elements:
        return tuple(
            TargetSupportCell(point, TargetSupportStatus.UNJUDGEABLE, 0)
            for point in target.key_points
        )
    element_index = inventory.elements.index(target.element)
    available: dict[float, list[CompositionPoint]] = defaultdict(list)
    for point in inventory.composition_points:
        fraction = point.fractions[element_index]
        if target.minimum - atol <= fraction <= target.maximum + atol:
            available[round(fraction, 12)].append(point)
    cells: list[TargetSupportCell] = []
    requested_config_types = tuple(
        dict.fromkeys(value.strip() for value in target.config_types if value.strip())
    )
    for target_fraction in target.key_points:
        exact_key = next(
            (fraction for fraction in available if abs(fraction - target_fraction) <= atol),
            None,
        )
        if exact_key is None:
            nearest = min(
                available,
                key=lambda fraction: abs(fraction - target_fraction),
                default=None,
            )
            for config_type in requested_config_types or ("",):
                cells.append(
                    TargetSupportCell(
                        target_fraction=target_fraction,
                        status=TargetSupportStatus.NO_SAMPLE,
                        observed_count=0,
                        nearest_fraction=nearest,
                        config_type=config_type,
                    )
                )
            continue
        exact_points = available[exact_key]
        missing_config_type_count = sum(
            point.missing_config_type_count for point in exact_points
        )
        for config_type in requested_config_types or ("",):
            if config_type:
                matching_indices = tuple(
                    sorted(
                        index
                        for point in exact_points
                        for name, indices in point.config_type_indices
                        if name.casefold() == config_type.casefold()
                        for index in indices
                    )
                )
            else:
                matching_indices = tuple(
                    sorted(
                        index
                        for point in exact_points
                        for index in point.structure_indices
                    )
                )
            observed_count = len(matching_indices)
            minimum_count = target.minimum_structure_count
            if config_type and observed_count == 0:
                status = (
                    TargetSupportStatus.UNJUDGEABLE
                    if missing_config_type_count
                    else TargetSupportStatus.NO_CONFIG_TYPE
                )
            elif (
                minimum_count is not None
                and observed_count < minimum_count
                and missing_config_type_count
                and config_type
            ):
                status = TargetSupportStatus.UNJUDGEABLE
            elif minimum_count is not None and observed_count < minimum_count:
                status = TargetSupportStatus.THIN
            else:
                status = TargetSupportStatus.SUPPORTED
            cells.append(
                TargetSupportCell(
                    target_fraction=target_fraction,
                    status=status,
                    observed_count=observed_count,
                    structure_indices=matching_indices,
                    nearest_fraction=exact_key,
                    config_type=config_type,
                    missing_config_type_count=missing_config_type_count,
                )
            )
    return tuple(cells)
