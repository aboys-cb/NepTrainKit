"""Controlled displacements and rotations retain the supplied physical reference."""

import io
import json
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.core.cards.structure import VibrationModePerturbOperation, VibrationModePerturbParams
from NepTrainKit.core.magnetic_response import MagneticResponseScanOperation, TextureMagneticResponseParams
from NepTrainKit.core.magnetism import prepare_magnetic_extxyz_export
from NepTrainKit.ui.views._card.i18n_utils import combo_value, set_combo_value
from NepTrainKit.ui.views._card.soc_texture_response_card import SOCTextureResponseCard
from NepTrainKit.ui.views._card.vibration_perturb_card import VibrationModePerturbCard


def reference():
    atoms = Atoms(
        "Fe2O", positions=[[0, 0, 0], [1, 0, 0], [1, 1, 1]],
        cell=[[4, 0, 0], [0.8, 4, 0], [0.2, 0.6, 4]], pbc=[True, False, True],
    )
    atoms.set_initial_magnetic_moments([[0, 0, 2], [0, 0, -3], [0, 0, 0]])
    atoms.new_array("group", np.array(["A", "B", "O"]))
    atoms.new_array("vibration_mode_2", np.array([[0.2, 0, 0], [-0.2, 0, 0], [0, 0, 0]]))
    atoms.new_array("vibration_mode_7", np.array([[0, 0.3, 0], [0, -0.1, 0], [0, 0.2, 0]]))
    return atoms


def test_coefficient_scan_exact_combinations_and_zero_reference():
    atoms = reference()
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.info["energy"] = -1.0
    rows = [[-0.25, 0], [0, 0], [0.25, 0], [0.1, -0.5]]
    params = VibrationModePerturbParams(sampling_mode="Coefficient table", coefficient_table=json.dumps(rows))
    output = VibrationModePerturbOperation().run_structure(atoms, params)
    assert len(output) == len(rows)
    for index, (frame, row) in enumerate(zip(output, rows)):
        expected = atoms.positions + row[0] * atoms.arrays["vibration_mode_2"] + row[1] * atoms.arrays["vibration_mode_7"]
        np.testing.assert_allclose(frame.positions, expected, atol=1e-15)
        np.testing.assert_array_equal(frame.cell, atoms.cell)
        np.testing.assert_array_equal(frame.pbc, atoms.pbc)
        np.testing.assert_array_equal(frame.arrays["group"], atoms.arrays["group"])
        np.testing.assert_array_equal(frame.get_initial_magnetic_moments(), atoms.get_initial_magnetic_moments())
        assert f"ModeScan(row={index}," in frame.info["Config_type"]
        assert json.loads(frame.info["vibration_mode_perturb"])["coefficients"] == row
        if index != 1:
            assert "energy" not in frame.info
    np.testing.assert_array_equal(output[1].positions, atoms.positions)
    assert atoms.info["energy"] == -1.0
    assert atoms.positions[0, 0] == 0
    repeated = VibrationModePerturbOperation().run_structure(
        atoms, replace(params, amplitude=0, modes_per_sample=999, min_frequency=999, max_num=1, seed=19, use_seed=True)
    )
    for first, second in zip(output, repeated):
        np.testing.assert_array_equal(first.positions, second.positions)


@pytest.mark.parametrize("text", ["", "{}", "[]", "[[]]", "[1,2]", "[[1],[2,3]]", '[["1"]]', "[[true]]", "[[NaN]]", "[[1e999]]"])
def test_coefficient_table_rejects_invalid_values(text):
    with pytest.raises(CardOperationError, match="rectangular JSON"):
        VibrationModePerturbOperation().run_structure(
            reference(), VibrationModePerturbParams(sampling_mode="Coefficient table", coefficient_table=text)
        )


def test_coefficient_scan_rejects_missing_nonfinite_and_mismatched_modes():
    operation = VibrationModePerturbOperation()
    params = VibrationModePerturbParams(sampling_mode="Coefficient table")
    with pytest.raises(CardOperationError, match="columns"):
        operation.run_structure(reference(), params)
    with pytest.raises(CardOperationError, match="finite displacement modes"):
        operation.run_structure(Atoms("Fe"), params)
    atoms = reference()
    atoms.arrays["vibration_mode_2"][0, 0] = np.nan
    with pytest.raises(CardOperationError, match="finite displacement modes"):
        operation.run_structure(atoms, params)
    with pytest.raises(CardOperationError, match="Mode sampling"):
        operation.run_structure(reference(), replace(params, sampling_mode="unknown"))


@pytest.mark.parametrize("kind", ["Bulk / Bloch", "Interfacial / Cycloidal", "General spiral"])
@pytest.mark.parametrize("phase", [0.0, 37.0])
@pytest.mark.parametrize("noncollinear", [False, True])
def test_rotation_field_matches_independent_ase_rotations(kind, phase, noncollinear):
    atoms = reference()
    if noncollinear:
        atoms.set_initial_magnetic_moments([[1, 0.2, 2], [-1, -0.2, -2], [0, 0, 0]])
    original = atoms.get_initial_magnetic_moments()
    operation = MagneticResponseScanOperation()
    output = operation.run_structure(atoms, TextureMagneticResponseParams(
        response_kind=kind, texture_action="Rotate input moments", q_definition="Cell reciprocal vector",
        coordinate_scan="-1,0,1", phase_deg=phase, plane_normal=(1, 2, 3),
    ))
    assert len(output) == 3
    for frame, multiplier, record in zip(output, [-1, 0, 1], operation.last_manifest.records):
        q = multiplier * (2 * np.pi * np.linalg.inv(atoms.cell).T)[0]
        expected = []
        for position, moment in zip(atoms.positions, original):
            probe = Atoms("H", positions=[moment])
            probe.rotate(np.rad2deg(position @ q) + phase, record.rotation_plane_normal)
            expected.append(probe.positions[0])
        np.testing.assert_allclose(frame.arrays["spin"], expected, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(frame.arrays["spin"], axis=1), np.linalg.norm(original, axis=1), atol=1e-12)
        np.testing.assert_array_equal(frame.positions, atoms.positions)
        np.testing.assert_array_equal(frame.cell, atoms.cell)
        np.testing.assert_array_equal(frame.pbc, atoms.pbc)
        np.testing.assert_array_equal(frame.arrays["group"], atoms.arrays["group"])
        assert record.metadata["texture_action"] == "Rotate input moments"
    if phase == 0:
        np.testing.assert_array_equal(output[1].arrays["spin"], original)
        if not noncollinear:
            assert np.dot(output[1].arrays["spin"][0], output[1].arrays["spin"][1]) == -6
    np.testing.assert_array_equal(atoms.get_initial_magnetic_moments(), original)


def test_rotation_field_legacy_generation_and_periodic_validation():
    atoms = reference()
    params = TextureMagneticResponseParams(response_kind="General spiral", q_definition="Cell reciprocal vector", coordinate_scan="-1,0,1")
    legacy = MagneticResponseScanOperation().run_structure(atoms, params)
    assert np.dot(legacy[1].arrays["spin"][0], legacy[1].arrays["spin"][1]) == pytest.approx(6)
    for action in ("Generate directions", "Rotate input moments"):
        with pytest.raises(CardOperationError, match="does not close"):
            MagneticResponseScanOperation().run_structure(atoms, replace(
                params, texture_action=action, coordinate_scan="-0.5,0,0.5",
            ))
        with pytest.raises(CardOperationError):
            MagneticResponseScanOperation().run_structure(atoms, replace(params, texture_action=action, max_outputs=2))
    with pytest.raises(CardOperationError, match="Texture action"):
        MagneticResponseScanOperation().run_structure(atoms, replace(params, texture_action="invalid"))


def test_scalar_afm_and_reference_label_invalidation():
    atoms = reference()
    del atoms.arrays["initial_magmoms"]
    atoms.set_initial_magnetic_moments([2, -3, 0])
    atoms.info["energy"] = -1.0
    atoms.new_array("forces", np.ones((3, 3)))
    frames = MagneticResponseScanOperation().run_structure(atoms, TextureMagneticResponseParams(
        response_kind="General spiral", texture_action="Rotate input moments",
        q_definition="Cell reciprocal vector", coordinate_scan="-1,0,1",
    ))
    np.testing.assert_array_equal(frames[1].arrays["spin"], [[0, 0, 2], [0, 0, -3], [0, 0, 0]])
    assert all("energy" not in frame.info and "forces" not in frame.arrays for frame in frames)
    # Scalar-to-vector conversion changes shape, so even its zero frame drops labels.
    del atoms.arrays["initial_magmoms"]
    atoms.set_initial_magnetic_moments(frames[1].arrays["spin"])
    frames = MagneticResponseScanOperation().run_structure(atoms, TextureMagneticResponseParams(
        response_kind="General spiral", texture_action="Rotate input moments",
        q_definition="Cell reciprocal vector", coordinate_scan="-1,0,1",
    ))
    assert frames[1].info["energy"] == -1.0
    assert "energy" not in frames[0].info
    assert "forces" not in frames[2].arrays


@pytest.mark.parametrize("bad_q", [(float("nan"), 0, 0), (float("inf"), 0, 0), (1, 0)])
def test_rotation_field_rejects_invalid_q(bad_q):
    with pytest.raises(CardOperationError, match="three finite values"):
        MagneticResponseScanOperation().run_structure(reference(), TextureMagneticResponseParams(
            response_kind="General spiral", texture_action="Rotate input moments",
            q_vector_cart=bad_q,
        ))


def test_new_ui_modes_roundtrip_and_run_from_controls():
    atoms = reference()
    card = VibrationModePerturbCard()
    set_combo_value(card.sampling_combo, "Coefficient table")
    card.coefficient_edit.setPlainText("[[-0.2,0],[0,0],[0.2,0],[0.1,0.5]]")
    assert not card.coefficient_field.isHidden()
    assert all(section.isHidden() for section in card._random_sections)
    restored = VibrationModePerturbCard()
    restored.from_dict(json.loads(json.dumps(card.to_dict())))
    assert restored.get_params() == card.get_params()
    assert len(restored.process_structure(atoms)) == 4
    card.from_dict({"check_state": True, "amplitude": [0.1], "max_num": [2]})
    assert card.get_params().sampling_mode == "Random"
    assert card.coefficient_field.isHidden()
    assert all(not section.isHidden() for section in card._random_sections)
    texture = SOCTextureResponseCard()
    set_combo_value(texture.kind_combo, "General spiral")
    for index in range(texture.action_combo.count()):
        texture.action_combo.setCurrentIndex(index)
        params = texture.get_params()
        assert params.texture_action == combo_value(texture.action_combo)
        result = texture.create_operation().run_structure(atoms, params)
        assert len(result) == 5
        restored_texture = SOCTextureResponseCard()
        restored_texture.from_dict(json.loads(json.dumps(texture.to_dict())))
        assert restored_texture.get_params() == params
        assert texture.cone_field.isHidden() == (params.texture_action == "Rotate input moments")
    legacy = texture.to_dict()
    legacy["params"].pop("texture_action")
    texture.from_dict(legacy)
    assert texture.get_params().texture_action == "Generate directions"


def test_displacements_and_rotated_moments_survive_extxyz_handoff():
    atoms = reference()
    frames = VibrationModePerturbOperation().run_structure(atoms, VibrationModePerturbParams(
        sampling_mode="Coefficient table", coefficient_table="[[0.1,-0.2]]",
    ))
    frames = MagneticResponseScanOperation().run_structure(frames[0], TextureMagneticResponseParams(
        response_kind="General spiral", texture_action="Rotate input moments", q_definition="Cell reciprocal vector",
    ))
    export = [prepare_magnetic_extxyz_export(frame) for frame in frames]
    buffer = io.StringIO()
    write(buffer, export, format="extxyz")
    restored = read(io.StringIO(buffer.getvalue()), index=":", format="extxyz")
    for original, saved in zip(frames, restored):
        np.testing.assert_allclose(saved.positions, original.positions, atol=1e-8)
        np.testing.assert_allclose(saved.arrays["spin"], original.arrays["spin"], atol=1e-8)
        np.testing.assert_allclose(saved.cell, original.cell)
        np.testing.assert_array_equal(saved.pbc, original.pbc)
        np.testing.assert_array_equal(saved.arrays["group"], original.arrays["group"])
        assert saved.info["vibration_mode_perturb"] == original.info["vibration_mode_perturb"]


@pytest.mark.parametrize("language", ["en_US", "zh_CN"])
def test_new_controls_fit_real_inspector_after_resize(language):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    from NepTrainKit.i18n import install_translator
    from NepTrainKit.ui.pages.makedata import MakeDataWidget

    app = QApplication.instance()
    install_translator(app, language)
    widget = MakeDataWidget()
    try:
        widget.dataset = [reference()]
        widget.show()
        area = widget.workspace_card_widget
        for class_name in ("VibrationModePerturbCard", "SOCTextureResponseCard"):
            card = widget.add_card(class_name)
            area.select_card(card, expand=True)
            if class_name == "SOCTextureResponseCard":
                set_combo_value(card.kind_combo, "General spiral")
                combo = card.action_combo
            else:
                combo = card.sampling_combo
            for mode in range(combo.count()):
                combo.setCurrentIndex(mode)
                for width, height in ((1600, 900), (1151, 700), (1600, 900), (1151, 700)):
                    widget.resize(width, height)
                    for _ in range(3):
                        app.processEvents()
                    scroll = area.guidance_panel.parameter_scroll
                    assert area.guidance_panel.isVisible()
                    assert area.library_panel.isVisible()
                    assert scroll.horizontalScrollBar().maximum() == 0
                    assert combo.fontMetrics().horizontalAdvance(combo.currentText()) + 35 <= combo.width()
    finally:
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        install_translator(app, "en_US")
