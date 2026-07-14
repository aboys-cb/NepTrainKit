from __future__ import annotations

from NepTrainKit.core.audit.extract import StructureAuditRecord
from NepTrainKit.core.audit.inventory import build_dataset_inventory, compare_composition_target
from NepTrainKit.core.audit.result import CompositionTarget, TargetSupportStatus


def _record(index, *, fe, ni, atoms, formula, config_type="bulk"):
    return StructureAuditRecord(
        index=index,
        formula=formula,
        num_atoms=atoms,
        composition={"Fe": fe / atoms, "Ni": ni / atoms},
        config_type=config_type,
        energy_per_atom=None,
        max_force=None,
        virial_norm=None,
    )


def test_inventory_merges_equivalent_compositions_across_supercell_sizes():
    inventory = build_dataset_inventory(
        [
            _record(2, fe=10, ni=6, atoms=16, formula="Fe10Ni6"),
            _record(8, fe=20, ni=12, atoms=32, formula="Fe20Ni12", config_type="vacancy"),
            _record(9, fe=0, ni=16, atoms=16, formula="Ni16"),
        ]
    )

    assert inventory.elements == ("Fe", "Ni")
    assert len(inventory.composition_points) == 2
    mixed = next(point for point in inventory.composition_points if point.reduced_counts == (5, 3))
    assert mixed.fractions == (0.625, 0.375)
    assert mixed.structure_count == 2
    assert mixed.structure_indices == (2, 8)
    assert mixed.atom_counts == ((16, 1), (32, 1))
    assert mixed.config_types == (("bulk", 1), ("vacancy", 1))


def test_target_comparison_uses_only_explicit_count_rule():
    inventory = build_dataset_inventory(
        [
            _record(0, fe=16, ni=0, atoms=16, formula="Fe16"),
            _record(1, fe=14, ni=2, atoms=16, formula="Fe14Ni2"),
            _record(2, fe=14, ni=2, atoms=16, formula="Fe14Ni2"),
        ]
    )
    target = CompositionTarget(
        element="Ni",
        minimum=0.0,
        maximum=0.4,
        key_points=(0.0, 0.125, 0.25),
        minimum_structure_count=2,
    )

    cells = compare_composition_target(inventory, target)

    assert [cell.status for cell in cells] == [
        TargetSupportStatus.THIN,
        TargetSupportStatus.SUPPORTED,
        TargetSupportStatus.NO_SAMPLE,
    ]
    assert cells[2].nearest_fraction == 0.125


def test_target_comparison_aggregates_multinary_points_with_same_element_fraction():
    records = [
        StructureAuditRecord(
            index=index,
            formula=formula,
            num_atoms=16,
            composition=composition,
            config_type="bulk",
            energy_per_atom=None,
            max_force=None,
            virial_norm=None,
        )
        for index, formula, composition in (
            (0, "Co8Ni8", {"Co": 0.5, "Ni": 0.5, "V": 0.0}),
            (1, "Co4Ni12", {"Co": 0.25, "Ni": 0.75, "V": 0.0}),
            (2, "Co8Ni4V4", {"Co": 0.5, "Ni": 0.25, "V": 0.25}),
        )
    ]
    inventory = build_dataset_inventory(records)

    cells = compare_composition_target(
        inventory,
        CompositionTarget(
            element="V",
            minimum=0.0,
            maximum=0.5,
            key_points=(0.0,),
            minimum_structure_count=2,
        ),
    )

    assert cells[0].status == TargetSupportStatus.SUPPORTED
    assert cells[0].observed_count == 2
    assert cells[0].structure_indices == (0, 1)
