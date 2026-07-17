from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from NepTrainKit.core.audit import AuditFindingKind, PhaseInventory, build_training_set_audit
from NepTrainKit.core.audit.data_quality import audit_data_quality
from NepTrainKit.core.io.base import NepPlotData, StructureData
from NepTrainKit.core.structure import Structure


def _structure(
    elements=("Fe",),
    positions=((1.0, 1.0, 1.0),),
    *,
    cell=None,
    pbc="T T T",
    energy=-1.0,
    forces=None,
):
    lattice = np.eye(3) * 5.0 if cell is None else np.asarray(cell, dtype=np.float64)
    atomic_properties = {
        "species": np.asarray(elements),
        "pos": np.asarray(positions, dtype=np.float64),
    }
    properties = [
        {"name": "species", "type": "S", "count": 1},
        {"name": "pos", "type": "R", "count": 3},
    ]
    if forces is not None:
        atomic_properties["force"] = np.asarray(forces, dtype=np.float64)
        properties.append({"name": "force", "type": "R", "count": 3})
    additional_fields = {"pbc": pbc, "Config_type": "fixture"}
    if energy is not None:
        additional_fields["energy"] = energy
    return Structure(lattice, atomic_properties, properties, additional_fields)


def _run(structures):
    values = np.asarray(structures, dtype=object)
    dataset = SimpleNamespace(
        structure=SimpleNamespace(
            all_data=values,
            now_indices=np.arange(len(values), dtype=np.int32),
        ),
        nep_txt_path=None,
    )
    return build_training_set_audit(dataset, dataset_id="quality-fixture")


def test_engine_attaches_versioned_phase_inventory_and_cache_state():
    structure_data = StructureData([_structure()])
    dataset = SimpleNamespace(
        structure=structure_data,
        nep_txt_path=None,
    )
    phase_inventory = PhaseInventory(
        schema_version="phase-inventory-v2",
        method_id="adaptive-cna-ordering-v1",
        reference_bank_id="aflow-l12-laves-v1",
        analysis_strategy="all-structures-v1",
        source_structure_count=1,
        analyzed_structure_count=1,
        analyzed_atom_count=1,
        composition_points=(),
    )

    with patch(
        "NepTrainKit.core.audit.engine.build_phase_inventory",
        return_value=(phase_inventory, True),
    ):
        run = build_training_set_audit(dataset, dataset_id="phase-fixture")

    assert run.phase_inventory is phase_inventory
    assert run.overview_metrics["phase_inventory"] == {
        "available": True,
        "status": "complete",
        "cache_hit": True,
        "analyzed_structures": 1,
    }
    assert "phase_inventory" in run.overview_metrics["timings_ms"]["stages"]


def test_engine_can_defer_complete_phase_analysis_for_the_desktop_page():
    structure_data = StructureData([_structure()])
    dataset = SimpleNamespace(structure=structure_data, nep_txt_path=None)

    with patch("NepTrainKit.core.audit.engine.build_phase_inventory") as phase_mock:
        run = build_training_set_audit(
            dataset,
            dataset_id="deferred-phase-fixture",
            include_phase_inventory=False,
        )

    phase_mock.assert_not_called()
    assert run.phase_inventory is None
    assert run.overview_metrics["phase_inventory"] == {
        "available": False,
        "status": "pending",
        "cache_hit": False,
        "analyzed_structures": 0,
    }


def test_quick_check_reports_real_data_contract_failures():
    valid = _structure(positions=((0.5, 0.5, 0.5),))

    nonfinite_geometry = _structure(positions=((1.0, 1.0, 1.0),))
    nonfinite_geometry.atomic_properties["pos"][0, 0] = np.nan

    invalid_cell = _structure(
        positions=((1.5, 1.5, 1.5),),
        cell=np.zeros((3, 3)),
    )
    unknown_element = _structure(elements=("Xx",), positions=((2.0, 2.0, 2.0),))

    invalid_force_shape = _structure(
        positions=((2.5, 2.5, 2.5),),
        forces=((0.0, 0.0, 0.0),),
    )
    invalid_force_shape.atomic_properties["force"] = np.zeros((1, 2))

    nonfinite_label = _structure(positions=((3.0, 3.0, 3.0),), energy=float("nan"))
    short_distance = _structure(
        elements=("Fe", "Fe"),
        positions=((3.5, 3.5, 3.5), (3.6, 3.5, 3.5)),
    )
    duplicate_a = _structure(positions=((4.0, 4.0, 4.0),), energy=-1.0)
    duplicate_b = _structure(positions=((4.0, 4.0, 4.0),), energy=-2.0)

    run = _run(
        [
            valid,
            nonfinite_geometry,
            invalid_cell,
            unknown_element,
            invalid_force_shape,
            nonfinite_label,
            short_distance,
            duplicate_a,
            duplicate_b,
        ]
    )
    findings = {finding.id: finding for finding in run.findings}

    assert findings["data_quality:nonfinite_geometry"].structure_indices == (1,)
    assert findings["data_quality:invalid_cell"].structure_indices == (2,)
    assert findings["data_quality:unknown_elements"].structure_indices == (3,)
    assert findings["data_quality:invalid_label_shape"].structure_indices == (4,)
    assert findings["data_quality:nonfinite_labels"].structure_indices == (5,)
    assert findings["data_quality:short_distance"].structure_indices == (6,)
    assert findings["data_quality:label_conflicts"].structure_indices == (7, 8)
    assert findings["data_quality:label_conflicts"].kind is AuditFindingKind.BLOCKER
    assert findings["data_quality:exact_duplicates"].structure_indices == (7, 8)
    assert findings["data_quality:exact_duplicates"].kind is AuditFindingKind.REVIEW


def test_real_dataset_audit_reuses_structure_geometry_snapshot():
    structures = [
        _structure(elements=("Fe", "Ni"), positions=((1.0, 1.0, 1.0), (3.0, 1.0, 1.0))),
        _structure(elements=("Fe", "Ni"), positions=((1.0, 1.0, 1.0), (1.1, 1.0, 1.0))),
    ]
    structure_data = StructureData(structures)
    dataset = SimpleNamespace(
        structure=structure_data,
        nep_txt_path=None,
        data_xyz_path="",
    )

    with patch.object(
        structure_data,
        "geometry_snapshot",
        wraps=structure_data.geometry_snapshot,
    ) as snapshot_spy:
        run = build_training_set_audit(dataset, dataset_id="cached-geometry")

    short = next(finding for finding in run.findings if finding.id == "data_quality:short_distance")
    assert short.structure_indices == (1,)
    assert snapshot_spy.call_count >= 2


def test_partial_periodic_cell_is_valid_when_periodic_vectors_are_independent():
    slab_cell = np.asarray(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    run = _run([_structure(cell=slab_cell, pbc="T T F")])

    blocker_ids = {
        finding.id for finding in run.findings if finding.kind is AuditFindingKind.BLOCKER
    }
    assert "data_quality:invalid_cell" not in blocker_ids


def test_duplicate_label_roundoff_is_not_reported_as_a_conflict():
    first = _structure(
        positions=((2.0, 2.0, 2.0),),
        energy=-100.0,
        forces=((0.01, 0.02, 0.03),),
    )
    second = _structure(
        positions=((2.0, 2.0, 2.0),),
        energy=-100.0 + 1.0e-6,
        forces=((0.010004, 0.019996, 0.030004),),
    )
    first.virial = np.zeros(9)
    second.virial = np.full(9, 5.0e-5)

    run = _run([first, second])
    finding_ids = {finding.id for finding in run.findings}

    assert "data_quality:exact_duplicates" in finding_ids
    assert "data_quality:label_conflicts" not in finding_ids


def test_materialized_energy_array_checks_reference_columns_only():
    structures = [
        _structure(positions=((1.0, 1.0, 1.0),), energy=-1.0),
        _structure(positions=((2.0, 2.0, 2.0),), energy=-2.0),
    ]
    energy = NepPlotData(
        np.asarray(
            [
                [np.nan, -1.0],
                [0.0, np.nan],
            ]
        ),
        title="energy",
    )
    result_data = SimpleNamespace(energy=energy, virial=None, _force_vector_dataset=None)

    _, slices, _ = audit_data_quality(
        list(enumerate(structures)),
        result_data=result_data,
    )
    findings = {finding.id: finding for finding in slices}

    assert findings["data_quality:nonfinite_labels"].structure_indices == (1,)
