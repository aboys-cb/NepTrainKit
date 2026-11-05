"""Scientific, export and live UI contracts for EOS-to-card sampling."""

import numpy as np
import pytest
from ase.io import read
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget
from test_validation_framework import FakePredictor, eos_frames
from test_validation_ui import _wait_until
from test_validation_ui import make_service as make_service
from test_validation_ui import scored as scored

from NepTrainKit.core.cards.lattice import CellStrainOperation
from NepTrainKit.core.validation.protocols import get_protocol
from NepTrainKit.core.validation.sampling import (
    EOSSamplingParams,
    clean_sampling_parent,
    reference_overlaps,
    sampling_payload,
    sampling_volumes,
    sampling_workflow,
    strain_params,
    suggest_sampling,
)
from NepTrainKit.ui.pages.makedata import MakeDataWidget
from NepTrainKit.ui.pages.makedata_workspace import MakeDataWorkspace


@pytest.fixture(autouse=True)
def retain_fluent_styles():
    # Match the application entry point before constructing menu-bearing widgets.
    from NepTrainKit.ui.widgets.style_lifetime import keep_created_styles_alive

    keep_created_styles_alive()


def source():
    return {"id": "eos-set", "object_hash": "reference-hash", "name": "PbTe / FCC_seed1"}


def labelled_frames():
    frames = eos_frames(points=9)
    for atoms in frames:
        atoms.set_array("spin", np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]))
        atoms.set_initial_magnetic_moments(atoms.arrays["spin"])
        atoms.info.update(energy=-8.0, virial=np.ones(9), dipole=np.ones(3))
        atoms.set_array("forces", np.ones((2, 3)))
        atoms.set_array("force_mag", np.ones((2, 3)))
    return frames


def test_nonorthogonal_geometry_labels_and_atomic_export(tmp_path):
    frames = labelled_frames()
    parent = frames[4]
    before = parent.copy()
    v = parent.get_volume() / len(parent)
    params = EOSSamplingParams(v * 0.921, v * 0.979, 8)
    clean = clean_sampling_parent(parent, {"set_id": "one", "model_hash": "two", "group": "PbTe_FCC_seed1"})
    output = CellStrainOperation().run_structure(clean, strain_params(clean, params))
    assert len(output) == 8
    for atoms in output:
        assert atoms.calc is None
        assert set(atoms.arrays) == {"numbers", "positions", "spin", "initial_magmoms"}
        assert set(atoms.info) == {"Config_type", "sampling_status", "sampling_source", "eos_group"}
        np.testing.assert_allclose(atoms.get_scaled_positions(), parent.get_scaled_positions())
        np.testing.assert_allclose(atoms.cell / np.cbrt(atoms.get_volume()), parent.cell / np.cbrt(parent.get_volume()))
        np.testing.assert_array_equal(atoms.arrays["spin"], parent.arrays["spin"])
        assert atoms.pbc.all()
    assert output[0].get_volume() / len(parent) == pytest.approx(params.start_volume, rel=1e-7)
    assert output[-1].get_volume() / len(parent) == pytest.approx(params.stop_volume, rel=1e-6)
    np.testing.assert_array_equal(parent.positions, before.positions)
    assert "energy" in parent.info and parent.calc is not None
    editor = MakeDataWidget()
    editor.load_sampling_draft(
        {"name": "geometry", "structures": [clean], "workflow": sampling_workflow(clean, params)}
    )
    editor.run_card()
    card = editor.workspace_card_widget.cards[0]
    assert _wait_until(lambda: card.run_outcome == "succeeded")
    path = tmp_path / "candidates.xyz"
    editor._export_file(path, [card])
    editor.close()
    saved = read(path, ":")
    assert len(saved) == 8
    for atoms in saved:
        assert atoms.calc is None and "virial" not in atoms.info
        assert "force_mag" not in atoms.arrays and "forces" not in atoms.arrays
        assert atoms.info["sampling_status"] == "needs_dft"
        assert atoms.info["sampling_source"]["model_hash"] == "two"
        assert atoms.info["eos_group"] == "PbTe_FCC_seed1"
        np.testing.assert_array_equal(atoms.arrays["spin"], parent.arrays["spin"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candidates.xyz"]


@pytest.mark.parametrize(
    "params",
    [
        EOSSamplingParams(2, 1),
        EOSSamplingParams(-1, 1),
        EOSSamplingParams(1, np.nan),
        EOSSamplingParams(1, 2, 1),
        EOSSamplingParams(1, 2, 101),
    ],
)
def test_invalid_sampling(params):
    with pytest.raises(ValueError):
        strain_params(eos_frames()[0], params)


def test_suggestion_uses_offset_free_residual_and_handles_manual_modes():
    frames = eos_frames(points=9)[::-1]
    protocol = get_protocol("eos")
    details = protocol.evaluate(FakePredictor(1, 30), frames).details
    assert suggest_sampling(frames, details)["reason"] == "offset_only"
    assert suggest_sampling(frames)["reason"] == "manual"
    details["predicted"][0] += 0.2
    suggestion = suggest_sampling(frames, details)
    assert suggestion["reason"] == "shape_error"
    assert suggestion["peak_mev"] > 100
    assert suggestion["params"].stop_volume < details["volumes"][1]
    parent = frames[suggestion["parent_index"]]
    assert parent.get_volume() / len(parent) == pytest.approx(details["volumes"][4])
    candidates = CellStrainOperation().run_structure(parent, strain_params(parent, suggestion["params"]))
    assert not any(reference_overlaps(candidates, frames))
    assert suggest_sampling(eos_frames(minimum=0.1))["reason"] == "reference_incomplete"
    details["volumes"] = details["volumes"][:-1]
    with pytest.raises(ValueError, match="does not match"):
        suggest_sampling(frames, details)


def test_reference_overlap_checks_geometry_and_magnetic_inputs():
    reference = labelled_frames()[0]
    same = clean_sampling_parent(reference, {})
    moved = same.copy()
    moved.positions[0] += 0.1
    spin = same.copy()
    spin.arrays["spin"] *= -1
    assert reference_overlaps([same, moved, spin], [reference]) == [True, False, False]


def payload():
    return sampling_payload(eos_frames(points=9), source())


def test_tabs_preserve_original_input_cards_results_and_multiple_groups():
    host = MakeDataWorkspace()
    original = host.primary_editor
    original.add_card("CellStrainCard")
    card = original.workspace_card_widget.cards[0]
    original.dataset = eos_frames()
    card.result_dataset = [original.dataset[0]]
    original._workflow_dirty = True
    first = host.open_sampling(payload())
    second_payload = payload()
    second_payload["name"] = "another group"
    second = host.open_sampling(second_payload)
    assert host.stack.count() == 3 and host.stack.currentWidget() is second
    assert first.dataset[0] is not second.dataset[0]
    QTest.mouseClick(host.tabs.items[0], Qt.MouseButton.LeftButton)
    assert host.stack.currentWidget() is original
    assert original.workspace_card_widget.cards == [card]
    assert card.result_dataset[0] is original.dataset[0] and original._workflow_dirty
    host.close()


def test_range_fields_and_card_parameters_are_bidirectionally_linked(tmp_path):
    host = MakeDataWorkspace()
    editor = host.open_sampling(payload())
    panel, card = editor._sampling_panel, editor._sampling_card
    assert card.result_dataset == []  # preview never generates structures
    panel.region.setRegion((29.123, 31.234))
    assert panel.start_volume.value() == pytest.approx(29.123, abs=1e-4)
    panel.count.setValue(7)
    expected = sampling_volumes(editor.dataset[0], card.get_params())
    assert len(expected) == 7
    np.testing.assert_allclose(expected, panel.volumes)
    for frame in (card.strain_x_frame,):
        frame.object_list[0].setValue(-4.123456)
    assert panel.start_volume.value() == pytest.approx(
        editor.dataset[0].get_volume() / len(editor.dataset[0]) * (1 - 0.04123456) ** 3, abs=1e-5
    )
    editor.run_card()
    assert _wait_until(lambda: card.run_outcome in ("succeeded", "failed"))
    assert card.run_outcome == "succeeded"
    np.testing.assert_allclose([a.get_volume() / len(a) for a in card.result_dataset], panel.volumes)
    out = tmp_path / "card.xyz"
    editor._export_file(out, [card])
    assert all(a.calc is None for a in read(out, ":"))
    panel.count.setValue(9)
    assert card.result_dataset == []
    assert editor._cards_for_export(False) == []
    assert editor._last_completed_card_index is None
    restored = card.__class__()
    restored.from_dict(card.to_dict())
    assert restored.get_params() == card.get_params()
    restored.close()
    host.close()


def test_invalid_edit_no_stale_outputs_and_reference_duplicates_rejected():
    host = MakeDataWorkspace()
    editor = host.open_sampling(payload())
    panel, card = editor._sampling_panel, editor._sampling_card
    panel.region.setRegion((29.12, 31.23))
    editor.run_card()
    assert _wait_until(lambda: card.run_outcome == "succeeded")
    panel.start_volume.setValue(100)
    assert not panel.valid and not card.result_dataset
    editor.run_card()
    assert not editor._has_running_card()
    # Exactly recreate the original linear-lattice scan: reject benchmark copies.
    frames = panel.context["references"]
    panel.start_volume.setValue(frames[0].get_volume() / len(frames[0]))
    panel.stop_volume.setValue(frames[-1].get_volume() / len(frames[-1]))
    panel.count.setValue(len(frames))
    editor.run_card()
    assert _wait_until(lambda: card.run_outcome in ("succeeded", "failed"))
    assert card.run_outcome == "failed" and card.result_dataset == []
    assert "repeat" in panel.status_label.text()
    host.close()


def test_input_replacement_and_card_removal_drop_eos_context(tmp_path):
    from ase.io import write

    host = MakeDataWorkspace()
    editor = host.open_sampling(payload())
    path = tmp_path / "input.xyz"
    write(path, eos_frames()[0], format="extxyz")
    editor.load_base_structure([path])
    assert not editor._sampling_attached and editor._sampling_panel.isHidden()
    another = host.open_sampling(payload())
    another.workspace_card_widget.clear_cards()
    assert not another._sampling_attached and another._sampling_panel.isHidden()
    host.close()


def test_card_selection_and_resize_preserve_readable_context():
    host = MakeDataWorkspace()
    editor = host.open_sampling(payload())
    host.show()
    for width in (1100, 1600, 1100):
        host.resize(width, 820)
        QTest.qWait(30)
        for control in (
            editor._sampling_panel.start_volume,
            editor._sampling_panel.stop_volume,
            editor._sampling_panel.count,
            *editor._sampling_card.strain_x_frame.object_list,
        ):
            assert control.lineEdit().width() >= control.fontMetrics().horizontalAdvance(control.text()) + 4
        assert editor._sampling_panel.width() < host.width()
    editor.add_card("PerturbCard")
    assert editor._sampling_panel.isHidden()
    editor.workspace_card_widget.select_card(editor._sampling_card)
    assert editor._sampling_panel.isVisible()
    host.close()


def _drag_region(panel, start, stop):
    viewport = panel.plot.viewport()

    def position(volume):
        bounds = panel.plot.getViewBox().viewRange()[1]
        point = panel.plot.getViewBox().mapViewToScene(QPointF(volume, bounds[0] + 0.65 * (bounds[1] - bounds[0])))
        return panel.plot.mapFromScene(point)

    a, b = position(start), position(stop)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=a)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        point = a + (b - a) * fraction
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(point),
            QPointF(viewport.mapToGlobal(point)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(viewport, event)
        QApplication.processEvents()
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=b)
    QApplication.processEvents()


def test_real_mouse_drag_moves_band_and_resizes_edge():
    host = MakeDataWorkspace()
    editor = host.open_sampling(payload())
    host.resize(1400, 900)
    host.show()
    panel = editor._sampling_panel
    panel.region.setRegion((29.0, 31.0))
    QTest.qWait(50)
    _drag_region(panel, 30.0, 30.8)
    low, high = panel.region.getRegion()
    assert low > 29.5 and high - low == pytest.approx(2.0, abs=0.03)
    _drag_region(panel, high, high + 1.0)
    new_low, new_high = panel.region.getRegion()
    assert new_low == pytest.approx(low, abs=0.03) and new_high > high + 0.5
    assert panel.start_volume.value() == pytest.approx(new_low, abs=1e-4)
    assert editor._sampling_card.result_dataset == []
    host.close()


def test_library_action_enters_third_module_without_dialog(scored):
    from NepTrainKit.main import NepTrainKitMainWindow
    from NepTrainKit.ui.views.model_library import ModelLibraryWidget

    host = QWidget()
    host.make_data_workspace = MakeDataWorkspace(host)
    host.switchTo = lambda widget: setattr(host, "selected_page", widget)
    library = ModelLibraryWidget(host, service=scored.service)
    library.samplingWorkflowRequested.connect(lambda p: NepTrainKitMainWindow.open_eos_sampling_workflow(host, p))
    library.open_scorecard(scored.suite["id"], scored.new["id"])
    p = library.open_sampling(scored.eos["id"])
    assert host.selected_page is host.make_data_workspace
    editor = host.make_data_workspace.stack.currentWidget()
    assert editor is not host.make_data_workspace.primary_editor
    assert p["structures"][0].info["sampling_source"]["model_hash"] == scored.new["content_hash"]
    assert not hasattr(library, "_sampling_dialog")
    assert editor._sampling_card.result_dataset == []
    host.close()


def test_toolbar_targets_current_draft_and_close_preserves_original(monkeypatch):
    host = MakeDataWorkspace()
    original = host.primary_editor
    editor = host.open_sampling(payload())
    calls = []
    monkeypatch.setattr(editor, "open_file", lambda: calls.append("open"))
    monkeypatch.setattr(editor, "export_file", lambda: calls.append("export"))
    host.open_file()
    host.export_file()
    assert calls == ["open", "export"]
    monkeypatch.setattr(editor, "_confirm_discard_workflow_changes", lambda: False)
    host._close(1)
    assert host.stack.count() == 2
    monkeypatch.setattr(editor, "_confirm_discard_workflow_changes", lambda: True)
    host._close(1)
    assert host.stack.count() == 1 and host.stack.currentWidget() is original
    assert host.tabs.isHidden()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def test_edit_during_worker_completion_cannot_restore_stale_output(monkeypatch):
    import threading

    host = MakeDataWorkspace()
    editor = host.open_sampling(payload())
    panel, card = editor._sampling_panel, editor._sampling_card
    panel.region.setRegion((29.1, 31.2))
    started, release = threading.Event(), threading.Event()
    operation = CellStrainOperation.run_structure

    def delayed(self, atoms, params):
        started.set()
        release.wait(3)
        return operation(self, atoms, params)

    monkeypatch.setattr(CellStrainOperation, "run_structure", delayed)
    editor.run_card()
    assert _wait_until(started.is_set)
    assert not panel.start_volume.isEnabled()
    try:
        panel.count.setValue(8)  # programmatic draft change while the worker is pending
    finally:
        release.set()
    assert _wait_until(lambda: not editor._has_running_card())
    QTest.qWait(30)
    assert not card.result_dataset and editor._cards_for_export(False) == []
    assert editor._last_completed_card_index is None
    editor.run_card()
    assert _wait_until(lambda: card.run_outcome in ("succeeded", "failed"))
    assert card.run_outcome == "succeeded" and len(card.result_dataset) == 8
