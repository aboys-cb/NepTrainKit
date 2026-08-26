from __future__ import annotations

import json
import math

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

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
    with pytest.raises(ValueError, match="smaller than one complete"):
        MagneticResponseScanOperation().run_structure(
            atoms, LocalMagneticResponseParams(max_outputs=4)
        )


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


def test_incommensurate_q_fails_closed_with_actionable_message():
    with pytest.raises(ValueError, match="not commensurate"):
        MagneticResponseScanOperation().run_structure(
            magnetic_pair(),
            TextureMagneticResponseParams(
                response_kind="Bulk / Bloch", q_vector_cart=(0, 0, 0.2)
            ),
        )


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
            MagnetoelasticResponseCard: ("磁弹响应", "响应网格", "各向同性体积"),
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
