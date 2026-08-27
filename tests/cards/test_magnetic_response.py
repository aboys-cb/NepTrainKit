from __future__ import annotations

import json
import math

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.core.magnetic_response import (
    LocalMagneticResponseParams,
    MagneticResponseScanOperation,
    MagnetoelasticResponseParams,
    ResponseManifest,
    TextureMagneticResponseParams,
    audit_response_groups,
    derived_spin_tangent,
    write_response_audit,
)
from NepTrainKit.core.magnetism import prepare_magnetic_extxyz_export


def magnetic_pair(*, pbc=True) -> Atoms:
    atoms = Atoms(
        "Fe2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        cell=np.diag([4.0, 4.0, 4.0]),
        pbc=pbc,
    )
    atoms.set_initial_magnetic_moments([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    return atoms


def test_local_pair_response_is_complete_and_has_per_pair_reference():
    atoms = Atoms("Fe3", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], cell=[5, 5, 5])
    atoms.set_initial_magnetic_moments([[0, 0, 2]] * 3)
    operation = MagneticResponseScanOperation()
    output = operation.run_structure(
        atoms,
        LocalMagneticResponseParams(
            pair_left_indices="1,2",
            pair_right_indices="2,3",
            coordinate_scan_deg="-2,-1,0,1,2",
        ),
    )
    assert len(output) == 10
    groups = {}
    for frame in output:
        groups.setdefault(frame.info["response_group"], []).append(frame)
    assert len(groups) == 2
    for frames in groups.values():
        coordinates = [frame.info["response_coordinate"] for frame in frames]
        assert np.allclose(coordinates, np.deg2rad([-2, -1, 0, 1, 2]))
        assert sum(frame.info["response_branch"] == "reference" for frame in frames) == 1
        reference = next(frame for frame in frames if frame.info["response_branch"] == "reference")
        assert np.array_equal(reference.positions, atoms.positions)
        for frame in frames:
            assert np.allclose(np.linalg.norm(frame.arrays["spin"], axis=1), 2.0)
        plus = next(frame for frame in frames if np.isclose(frame.info["response_coordinate"], math.radians(2)))
        record_by_task = {record.task_id: record for record in operation.last_manifest.records}
        target = record_by_task[plus.info["response_task_id"]].target_indices
        spin = plus.arrays["spin"]
        angle = math.acos(np.dot(spin[target[0]], spin[target[1]]) / 4.0)
        assert angle == pytest.approx(math.radians(2), abs=1.0e-12)


def test_output_limit_never_leaves_partial_group():
    atoms = Atoms("Fe3", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], cell=[5, 5, 5])
    atoms.set_initial_magnetic_moments([[0, 0, 2]] * 3)
    output = MagneticResponseScanOperation().run_structure(
        atoms,
        LocalMagneticResponseParams(
            pair_left_indices="1,2", pair_right_indices="2,3", max_outputs=7,
        ),
    )
    assert len(output) == 5
    assert audit_response_groups(output)["invalid_groups"] == {}
    with pytest.raises(CardOperationError) as raised:
        MagneticResponseScanOperation().run_structure(
            atoms, LocalMagneticResponseParams(max_outputs=4)
        )
    assert raised.value.code == "magnetic_response_budget_too_small"


def test_local_rotation_axis_controls_the_actual_rodrigues_rotation():
    atoms = magnetic_pair(pbc=False)
    y_axis = MagneticResponseScanOperation().run_structure(
        atoms,
        LocalMagneticResponseParams(
            response_kind="Single-spin tilt",
            coordinate_scan_deg="-90,0,90",
            rotation_axis=(0.0, 1.0, 0.0),
        ),
    )
    z_axis = MagneticResponseScanOperation().run_structure(
        atoms,
        LocalMagneticResponseParams(
            response_kind="Single-spin tilt",
            coordinate_scan_deg="-90,0,90",
            rotation_axis=(0.0, 0.0, 1.0),
        ),
    )

    assert y_axis[-1].arrays["spin"][0] == pytest.approx([2.0, 0.0, 0.0], abs=1.0e-12)
    assert z_axis[-1].arrays["spin"][0] == pytest.approx([0.0, 0.0, 2.0], abs=1.0e-12)
    assert all(
        np.array_equal(frame.arrays["spin"][1], atoms.get_initial_magnetic_moments()[1])
        for frame in y_axis
    )


def test_local_response_reports_actionable_input_errors():
    without_moments = Atoms("Fe", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(CardOperationError) as missing:
        MagneticResponseScanOperation().run_structure(
            without_moments, LocalMagneticResponseParams()
        )
    assert missing.value.code == "magnetic_response_missing_moments"

    atoms = magnetic_pair(pbc=False)
    with pytest.raises(CardOperationError) as groups:
        MagneticResponseScanOperation().run_structure(
            atoms,
            LocalMagneticResponseParams(
                response_kind="Group pair canting", group_a="A", group_b="B"
            ),
        )
    assert groups.value.code == "local_response_no_group_pair"
    assert "Group Label" in str(groups.value)


def test_local_automatic_pair_filters_change_the_selected_response_groups():
    atoms = Atoms(
        ["Fe", "Co", "Co", "Fe"],
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        cell=[4, 4, 4],
        pbc=False,
    )
    atoms.set_initial_magnetic_moments([[0, 0, 2]] * 4)
    atoms.set_array("group", np.asarray(["A", "B", "B", "A"]))
    common = dict(
        pair_source="Auto by neighbor shell",
        pair_element_filter="Fe-Co",
        pair_group_filter="A-B",
        bond_filter_axis=(1.0, 0.0, 0.0),
        bond_filter_tolerance=1.0,
    )

    in_plane = MagneticResponseScanOperation().run_structure(
        atoms,
        LocalMagneticResponseParams(bond_filter_mode="Near plane", **common),
    )
    along_axis = MagneticResponseScanOperation().run_structure(
        atoms,
        LocalMagneticResponseParams(bond_filter_mode="Near axis", **common),
    )

    assert len({frame.info["response_group"] for frame in in_plane}) == 2
    assert len({frame.info["response_group"] for frame in along_axis}) == 2
    assert len(in_plane) == len(along_axis) == 10

    all_directions = MagneticResponseScanOperation().run_structure(
        atoms,
        LocalMagneticResponseParams(
            bond_filter_mode="Any",
            pair_source="Auto by neighbor shell",
            pair_element_filter="Fe-Co",
            pair_group_filter="A-B",
        ),
    )
    assert len(all_directions) == 20


def test_reused_operation_assigns_unique_deterministic_parents_to_duplicate_inputs():
    atoms = magnetic_pair(pbc=False)
    operation = MagneticResponseScanOperation()
    first = operation.run_structure(atoms, LocalMagneticResponseParams())
    second = operation.run_structure(atoms, LocalMagneticResponseParams())
    assert first[0].info["response_parent"] != second[0].info["response_parent"]
    assert first[0].info["response_group"] != second[0].info["response_group"]


def test_moment_scale_response_keeps_direction_and_uses_scale_coordinate():
    output = MagneticResponseScanOperation().run_structure(
        magnetic_pair(pbc=False),
        LocalMagneticResponseParams(response_kind="Moment magnitude", target_indices="1"),
    )
    assert [frame.info["response_coordinate"] for frame in output] == pytest.approx([-0.2, -0.1, 0, 0.1, 0.2])
    assert [np.linalg.norm(frame.arrays["spin"][0]) for frame in output] == pytest.approx([1.6, 1.8, 2.0, 2.2, 2.4])
    assert all(np.allclose(frame.arrays["spin"][0, :2], 0.0) for frame in output)


def test_global_anisotropy_preserves_spin_topology_geometry_and_time_reversal():
    atoms = magnetic_pair(pbc=False)
    operation = MagneticResponseScanOperation()
    output = operation.run_structure(
        atoms,
        TextureMagneticResponseParams(
            coordinate_scan="-90,0,90", include_time_reversal=True
        ),
    )
    assert len(output) == 6
    groups = {}
    for frame in output:
        groups.setdefault(frame.info["response_group"], []).append(frame)
        assert np.array_equal(frame.positions, atoms.positions)
        assert np.array_equal(frame.cell.array, atoms.cell.array)
    assert len(groups) == 2
    normal = next(frames for frames in groups.values() if frames[0].info["response_kind"] == "global_anisotropy")
    reversed_frames = next(frames for frames in groups.values() if "time_reversed" in frames[0].info["response_kind"])
    for first, second in zip(normal, reversed_frames):
        assert np.allclose(second.arrays["spin"], -first.arrays["spin"])
    for frames in groups.values():
        dots = frames[0].arrays["spin"] @ frames[0].arrays["spin"].T
        assert all(np.allclose(frame.arrays["spin"] @ frame.arrays["spin"].T, dots) for frame in frames)


def test_global_time_reversal_budget_must_fit_both_requested_groups():
    with pytest.raises(CardOperationError) as raised:
        MagneticResponseScanOperation().run_structure(
            magnetic_pair(pbc=False),
            TextureMagneticResponseParams(
                coordinate_scan="-90,0,90",
                include_time_reversal=True,
                max_outputs=5,
            ),
        )
    assert raised.value.code == "texture_response_budget_too_small"


@pytest.mark.parametrize(
    "kind,extra",
    [
        ("Bulk / Bloch", {}),
        ("Interfacial / Cycloidal", {"surface_normal": (1.0, 0.0, 0.0)}),
        ("General spiral", {"plane_normal": (1.0, 0.0, 0.0)}),
    ],
)
def test_symmetric_q_scans_record_chirality_and_spiral_contract(kind, extra):
    atoms = magnetic_pair()
    q0 = 2.0 * math.pi / 4.0
    operation = MagneticResponseScanOperation()
    output = operation.run_structure(
        atoms,
        TextureMagneticResponseParams(
            response_kind=kind,
            coordinate_scan="-2,-1,0,1,2",
            q_vector_cart=(0.0, 0.0, q0),
            **extra,
        ),
    )
    coordinates = [frame.info["response_coordinate"] for frame in output]
    assert coordinates == pytest.approx([-2*q0, -q0, 0.0, q0, 2*q0])
    record = operation.last_manifest.records[0]
    assert record.metadata["period_angstrom"] == pytest.approx(4.0)
    assert record.metadata["phase_radian"] == 0.0
    assert record.metadata["chirality"] == [-1, -1, 0, 1, 1]
    assert record.rotation_plane_normal is not None
    minus, plus = output[1].arrays["spin"], output[3].arrays["spin"]
    assert np.sign(np.cross(minus[0], minus[1]) @ np.asarray(record.rotation_plane_normal)) == -np.sign(
        np.cross(plus[0], plus[1]) @ np.asarray(record.rotation_plane_normal)
    )


def test_cell_reciprocal_q_is_exactly_commensurate_for_a_nonorthogonal_mixed_pbc_cell():
    atoms = Atoms(
        "Fe2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.5, 0.4]],
        cell=[[4.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, 0.2, 5.0]],
        pbc=[True, True, False],
    )
    atoms.set_initial_magnetic_moments([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    reciprocal_index = np.asarray([1, -1, 0])
    reciprocal = 2.0 * math.pi * np.linalg.inv(atoms.cell.array).T
    expected_q = reciprocal.T @ reciprocal_index

    operation = MagneticResponseScanOperation()
    output = operation.run_structure(
        atoms,
        TextureMagneticResponseParams(
            response_kind="Bulk / Bloch",
            coordinate_scan="-1,0,1",
            q_definition="Cell reciprocal vector",
            q_reciprocal_index=(1, -1, 0),
        ),
    )

    assert [frame.info["response_coordinate"] for frame in output] == pytest.approx(
        [-np.linalg.norm(expected_q), 0.0, np.linalg.norm(expected_q)]
    )
    record = operation.last_manifest.records[0]
    assert record.metadata["q_definition"] == "Cell reciprocal vector"
    assert record.metadata["q_reciprocal_index"] == [1, -1, 0]
    assert record.metadata["q_cartesian_1_per_angstrom"][2] == pytest.approx(expected_q)
    assert all(np.array_equal(frame.cell.array, atoms.cell.array) for frame in output)
    assert all(np.array_equal(frame.pbc, atoms.pbc) for frame in output)


@pytest.mark.parametrize("kind", ["Bulk / Bloch", "Interfacial / Cycloidal", "General spiral"])
def test_default_card_reciprocal_q_runs_on_an_ordinary_periodic_cell(kind):
    from PySide6.QtWidgets import QApplication

    from NepTrainKit.ui.views._card.i18n_utils import set_combo_value
    from NepTrainKit.ui.views._card.soc_texture_response_card import SOCTextureResponseCard

    app = QApplication.instance() or QApplication([])
    card = SOCTextureResponseCard()
    set_combo_value(card.kind_combo, kind)

    output = card.create_operation().run_structure(magnetic_pair(), card.get_params())

    assert len(output) == 5
    assert card.get_params().q_definition == "Cell reciprocal vector"
    assert output[0].info["response_branch"] == "minus"
    card.deleteLater()
    app.processEvents()


def test_zero_cell_reciprocal_index_is_rejected_with_a_structured_error():
    with pytest.raises(CardOperationError) as raised:
        MagneticResponseScanOperation().run_structure(
            magnetic_pair(),
            TextureMagneticResponseParams(
                response_kind="Bulk / Bloch",
                q_definition="Cell reciprocal vector",
                q_reciprocal_index=(0, 0, 0),
            ),
        )
    assert raised.value.code == "texture_response_zero_reciprocal_index"


def test_incommensurate_q_fails_closed_with_actionable_message():
    with pytest.raises(CardOperationError) as raised:
        MagneticResponseScanOperation().run_structure(
            magnetic_pair(),
            TextureMagneticResponseParams(
                response_kind="Bulk / Bloch", q_vector_cart=(0, 0, 0.2)
            ),
        )
    assert raised.value.code == "texture_response_incommensurate_q"
    assert "cell-reciprocal q mode" in str(raised.value)


def test_magnetoelastic_grid_preserves_probe_correspondence_and_lineage():
    atoms = magnetic_pair(pbc=False)
    operation = MagneticResponseScanOperation()
    output = operation.run_structure(
        atoms,
        MagnetoelasticResponseParams(
            structural_mode="Uniaxial strain",
            structural_scan="-0.02,0,0.02",
            spin_scan_deg="-2,0,2",
        ),
    )
    assert len(output) == 9
    assert len({frame.info["response_parent"] for frame in output}) == 1
    groups = {}
    for frame in output:
        groups.setdefault(frame.info["response_group"], []).append(frame)
    assert len(groups) == 3
    assert all(sorted(frame.info["response_coordinate"] for frame in frames) == pytest.approx(np.deg2rad([-2,0,2])) for frames in groups.values())
    tensors = [record.metadata["strain_tensor"] for record in operation.last_manifest.records[::3]]
    assert [tensor[2][2] for tensor in tensors] == pytest.approx([-0.02, 0.0, 0.02])


def test_magnetoelastic_spin_axis_controls_a_true_selected_atom_rotation():
    atoms = Atoms(
        "Fe2",
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    moments = np.asarray([[0.3, 0.4, 1.5], [-0.2, 1.1, 0.7]])
    atoms.set_initial_magnetic_moments(moments)
    common = dict(
        structural_scan="-0.01,0,0.01",
        spin_scan_deg="-10,0,10",
        target_indices="1",
    )

    about_y = MagneticResponseScanOperation().run_structure(
        atoms,
        MagnetoelasticResponseParams(rotation_axis=(0.0, 1.0, 0.0), **common),
    )
    about_z = MagneticResponseScanOperation().run_structure(
        atoms,
        MagnetoelasticResponseParams(rotation_axis=(0.0, 0.0, 1.0), **common),
    )

    theta = math.radians(10.0)
    expected_y = np.asarray(
        [
            moments[0, 0] * math.cos(theta) + moments[0, 2] * math.sin(theta),
            moments[0, 1],
            -moments[0, 0] * math.sin(theta) + moments[0, 2] * math.cos(theta),
        ]
    )
    assert about_y[2].arrays["spin"][0] == pytest.approx(expected_y)
    assert not np.allclose(about_y[2].arrays["spin"][0], about_z[2].arrays["spin"][0])
    assert np.array_equal(about_y[2].arrays["spin"][1], moments[1])
    assert about_y[2].info["_response_manifest_record"]["rotation_axis"] == pytest.approx(
        [0.0, 1.0, 0.0]
    )


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("Isotropic volume", lambda n, d, s: np.eye(3) * (1.0 + s) ** (1.0 / 3.0)),
        ("Uniaxial strain", lambda n, d, s: np.eye(3) + s * np.outer(n, n)),
        ("Biaxial strain", lambda n, d, s: np.eye(3) + s * (np.eye(3) - np.outer(n, n))),
        (
            "Symmetric shear",
            lambda n, d, s: np.eye(3) + 0.5 * s * (np.outer(n, d) + np.outer(d, n)),
        ),
    ],
)
def test_magnetoelastic_cartesian_modes_match_their_documented_deformation(mode, expected):
    atoms = Atoms(
        "Fe2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.5, 0.4]],
        cell=[[4.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.5, 0.2, 5.0]],
        pbc=[True, False, True],
    )
    atoms.set_initial_magnetic_moments([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    normal = np.asarray([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    direction = np.asarray([0.0, 0.0, 1.0])
    output = MagneticResponseScanOperation().run_structure(
        atoms,
        MagnetoelasticResponseParams(
            structural_mode=mode,
            structural_scan="-0.02,0,0.02",
            spin_scan_deg="-2,0,2",
            strain_axis=tuple(normal),
            shear_direction=tuple(direction),
        ),
    )

    final = output[-1]
    deformation = final.cell.array.T @ np.linalg.inv(atoms.cell.array.T)
    assert deformation == pytest.approx(expected(normal, direction, 0.02))
    assert np.array_equal(final.pbc, atoms.pbc)
    assert final.get_scaled_positions(wrap=False) == pytest.approx(
        atoms.get_scaled_positions(wrap=False)
    )


@pytest.mark.parametrize("axis,index", [("a", 0), ("b", 1), ("c", 2)])
def test_magnetoelastic_bain_axis_selects_a_lattice_vector_at_constant_volume(axis, index):
    atoms = magnetic_pair()
    operation = MagneticResponseScanOperation()
    output = operation.run_structure(
        atoms,
        MagnetoelasticResponseParams(
            structural_mode="Bain / tetragonal",
            structural_scan="-0.02,0,0.02",
            spin_scan_deg="-2,0,2",
            bain_axis=axis,
        ),
    )
    final = output[-1]
    factors = np.linalg.norm(final.cell.array, axis=1) / np.linalg.norm(atoms.cell.array, axis=1)
    expected = np.full(3, 1.0 / math.sqrt(1.02))
    expected[index] = 1.02
    assert factors == pytest.approx(expected)
    assert final.get_volume() == pytest.approx(atoms.get_volume())
    assert operation.last_manifest.records[-1].metadata["bain_lattice_axis"] == axis


def test_magnetoelastic_rejects_nonorthogonal_shear_directions():
    with pytest.raises(CardOperationError) as raised:
        MagneticResponseScanOperation().run_structure(
            magnetic_pair(),
            MagnetoelasticResponseParams(
                structural_mode="Symmetric shear",
                strain_axis=(0.0, 0.0, 1.0),
                shear_direction=(0.0, 1.0, 1.0),
            ),
        )
    assert raised.value.code == "magnetoelastic_nonorthogonal_shear_directions"

    with pytest.raises(CardOperationError) as bain:
        MagneticResponseScanOperation().run_structure(
            magnetic_pair(),
            MagnetoelasticResponseParams(
                structural_mode="Bain / tetragonal", bain_axis="z"
            ),
        )
    assert bain.value.code == "magnetoelastic_invalid_bain_axis"


def test_magnetoelastic_multiple_targets_rotate_together_without_multiplying_groups():
    atoms = magnetic_pair()
    one = MagneticResponseScanOperation().run_structure(
        atoms,
        MagnetoelasticResponseParams(
            structural_scan="-0.01,0,0.01",
            spin_scan_deg="-10,0,10",
            target_indices="1",
        ),
    )
    both = MagneticResponseScanOperation().run_structure(
        atoms,
        MagnetoelasticResponseParams(
            structural_scan="-0.01,0,0.01",
            spin_scan_deg="-10,0,10",
            target_indices="1,2",
        ),
    )
    assert len(one) == len(both) == 9
    assert not np.array_equal(one[2].arrays["spin"][0], atoms.get_initial_magnetic_moments()[0])
    assert np.array_equal(one[2].arrays["spin"][1], atoms.get_initial_magnetic_moments()[1])
    assert not np.array_equal(both[2].arrays["spin"][1], atoms.get_initial_magnetic_moments()[1])


@pytest.mark.parametrize("limit, expected", [(3, 3), (4, 3), (8, 6), (9, 9)])
def test_magnetoelastic_output_limit_keeps_only_complete_lattice_groups(limit, expected):
    output = MagneticResponseScanOperation().run_structure(
        magnetic_pair(),
        MagnetoelasticResponseParams(
            structural_scan="-0.01,0,0.01",
            spin_scan_deg="-2,0,2",
            max_outputs=limit,
        ),
    )
    assert len(output) == expected
    assert audit_response_groups(output)["invalid_groups"] == {}


def test_manifest_round_trip_reattaches_only_matching_task(tmp_path):
    operation = MagneticResponseScanOperation()
    output = operation.run_structure(magnetic_pair(pbc=False), LocalMagneticResponseParams())
    path = tmp_path / "manifest.json"
    operation.last_manifest.write(path)
    loaded = ResponseManifest.read(path)
    assert ResponseManifest.from_dataset(output).to_dict() == operation.last_manifest.to_dict()
    dft_output = output[0].copy()
    for key in list(dft_output.info):
        if key.startswith("response_"):
            del dft_output.info[key]
    restored = loaded.reattach(dft_output, operation.last_manifest.records[0].task_id)
    assert restored.info["response_group"] == output[0].info["response_group"]
    with pytest.raises(ValueError, match="exactly one"):
        loaded.reattach(dft_output, "another-task")
    with pytest.raises(ValueError, match="atom identity/order"):
        loaded.reattach(Atoms("Co2"), operation.last_manifest.records[0].task_id)
    payload = json.loads(path.read_text())
    assert len(payload["manifest_hash"]) == 64


def test_extxyz_round_trip_keeps_response_spin_and_mforce_without_alias_or_tangent(tmp_path):
    output = MagneticResponseScanOperation().run_structure(
        magnetic_pair(pbc=False), LocalMagneticResponseParams()
    )[0]
    output.set_array("mforce", np.ones((2, 3)))
    output.set_array("spin_tangent", np.ones((2, 3)))
    exported = prepare_magnetic_extxyz_export(output)
    path = tmp_path / "response.extxyz"
    write(path, exported, format="extxyz")
    header = path.read_text().splitlines()[1]
    assert ":spin:R:3" in header and ":mforce:R:3" in header
    assert "initial_magmoms" not in header and "spin_tangent" not in header
    restored = read(path, format="extxyz")
    assert restored.info["response_schema"] == "magnetic-response-v1"
    assert "spin" in restored.arrays and "mforce" in restored.arrays


def test_derived_tangent_matches_analytic_generator_and_is_not_persisted():
    output = MagneticResponseScanOperation().run_structure(
        magnetic_pair(pbc=False),
        TextureMagneticResponseParams(coordinate_scan="-1,0,1"),
    )
    tangent = derived_spin_tangent(output)
    assert tangent[1, 0] == pytest.approx([2.0, 0.0, 0.0], abs=2.0e-4)
    assert tangent[1, 1] == pytest.approx([3.0, 0.0, 0.0], abs=3.0e-4)
    assert all("spin_tangent" not in frame.arrays for frame in output)


def test_labelled_response_audit_writes_json_csv_png_and_even_odd_columns(tmp_path):
    output = MagneticResponseScanOperation().run_structure(
        magnetic_pair(pbc=False), LocalMagneticResponseParams()
    )
    for frame in output:
        x = float(frame.info["response_coordinate"])
        frame.info["energy"] = x * x + 0.1 * x
        frame.set_array("mforce", np.ones((len(frame), 3)))
    paths = write_response_audit(output, tmp_path)
    assert paths["json"].is_file() and paths["csv"].is_file() and paths["png"].is_file()
    header = paths["csv"].read_text().splitlines()[0]
    assert "energy_even" in header and "energy_odd" in header and "g_even" in header and "g_odd" in header


@pytest.mark.parametrize(
    "mutator,reason",
    [
        (lambda frames: frames.pop(), "unpaired plus/minus branches"),
        (lambda frames: frames[1].info.__setitem__("response_coordinate", frames[0].info["response_coordinate"]), "duplicate coordinate"),
        (lambda frames: frames[1].set_cell(np.eye(3) * 9), "mixed geometry/cell"),
        (lambda frames: frames.__setitem__(1, Atoms("Fe", positions=[[0,0,0]])), "mixed atom counts"),
    ],
)
def test_group_integrity_reports_invalid_groups(mutator, reason):
    frames = MagneticResponseScanOperation().run_structure(
        magnetic_pair(pbc=False), LocalMagneticResponseParams()
    )
    group = frames[0].info["response_group"]
    if reason == "mixed atom counts":
        replacement = frames[1]
        bad = Atoms("Fe", positions=[[0,0,0]])
        bad.info.update(replacement.info)
        bad.set_array("spin", np.array([[0.0, 0.0, 2.0]]))
        frames[1] = bad
    else:
        mutator(frames)
    assert reason in audit_response_groups(frames)["invalid_groups"][group]


def test_response_cards_round_trip():
    from PySide6.QtWidgets import QApplication
    from NepTrainKit.ui.views._card.local_magnetic_response_card import LocalMagneticResponseCard
    from NepTrainKit.ui.views._card.magnetoelastic_response_card import MagnetoelasticResponseCard
    from NepTrainKit.ui.views._card.soc_texture_response_card import SOCTextureResponseCard

    app = QApplication.instance() or QApplication([])
    for card_type in (LocalMagneticResponseCard, SOCTextureResponseCard, MagnetoelasticResponseCard):
        card = card_type()
        payload = card.to_dict()
        restored = card_type(); restored.from_dict(payload)
        assert restored.to_dict()["params"] == payload["params"]
        card.deleteLater(); restored.deleteLater()
    app.processEvents()


def test_response_cards_have_complete_chinese_catalog_labels_and_presets():
    from PySide6.QtWidgets import QApplication

    from NepTrainKit import i18n
    from NepTrainKit.core import CardManager
    from NepTrainKit.ui.views._card.local_magnetic_response_card import LocalMagneticResponseCard
    from NepTrainKit.ui.views._card.magnetoelastic_response_card import MagnetoelasticResponseCard
    from NepTrainKit.ui.views._card.soc_texture_response_card import SOCTextureResponseCard
    from NepTrainKit.ui.widgets.card_metadata import localized_card_description, localized_card_name

    app = QApplication.instance() or QApplication([])
    i18n.install_translator(app, "zh_CN")
    try:
        expected = {
            LocalMagneticResponseCard: ("局域磁响应", "响应路径", "原子对倾斜"),
            SOCTextureResponseCard: ("SOC / 纹理响应", "纹理路径", "全局各向异性"),
            MagnetoelasticResponseCard: ("磁弹响应网格", "响应网格", "各向同性体积"),
        }
        for card_type, (name, first_section, preset) in expected.items():
            metadata = CardManager.card_metadata_dict[card_type.__name__]
            card = card_type()
            combo = card.kind_combo if hasattr(card, "kind_combo") else card.mode_combo
            assert localized_card_name(metadata) == name
            assert localized_card_description(metadata) != metadata.description
            assert card.settingLayout.itemAt(0).widget().title_label.text() == first_section
            assert combo.currentText() == preset
            card.deleteLater()
    finally:
        i18n.install_translator(app, "en_US")
    app.processEvents()
