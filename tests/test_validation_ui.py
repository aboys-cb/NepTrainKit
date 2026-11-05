"""Model library scorecard, benchmark import dialog, viewer chip and settings."""

from __future__ import annotations

import time
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PySide6.QtCore import QItemSelectionModel, Qt, QTranslator
from PySide6.QtWidgets import QApplication, QWidget

from NepTrainKit.core.validation import SetTooLargeError
from NepTrainKit.core.validation import library as library_module
from NepTrainKit.ui.views.model_library import (
    VERDICT_COLORS,
    ModelLibraryWidget,
    ScorecardTableModel,
    summarize_model_status,
)
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog
from test_validation_framework import MODEL_HEADER, _import, efv_frames, eos_frames, write_xyz
from test_validation_framework import make_service as make_service, model_file as model_file


def _wait_until(predicate, timeout=10.0):
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _capture(service, tmp_path, name):
    path = tmp_path / name / "nep.txt"
    path.parent.mkdir()
    path.write_text(MODEL_HEADER + name)
    return service.capture_model(path)


@pytest.fixture
def scored(make_service, tmp_path):
    """Baseline ``old``, improved ``new`` and never-evaluated ``fresh`` on an EOS + EFV suite."""
    service = make_service()
    old, new, fresh = (_capture(service, tmp_path, name) for name in ("old", "new", "fresh"))
    make_service(**{old["content_hash"]: (1.10, 0.0), new["content_hash"]: (1.02, 0.0)})
    eos = _import(service, tmp_path, eos_frames(), "eos")
    efv = _import(service, tmp_path, efv_frames(), "efv")
    suite = service.create_suite("PbTe", [eos["id"], efv["id"]])
    for model in (old, new):
        service.run_suite(model["id"], suite["id"])
    service.set_baseline(suite["id"], old["id"], "first production model")
    return SimpleNamespace(service=service, suite=suite, old=old, new=new, fresh=fresh, eos=eos, efv=efv)


# -- scorecard model ------------------------------------------------------------

def test_scorecard_model_marks_baseline_and_colours_verdicts(scored):
    service = scored.service
    card = service.scorecard(scored.suite["id"], [scored.new["id"], scored.fresh["id"]])
    model = ScorecardTableModel()
    model.set_scorecard(card)
    assert model.columnCount() == 1 + len(card["columns"]) and model.rowCount() == 3
    assert model.headerData(0, Qt.Orientation.Horizontal) == "Model version"
    assert model.headerData(1, Qt.Orientation.Horizontal).startswith("eos\n")

    base = model.row_of_model(scored.old["id"])
    assert model.data(model.index(base, 0)).endswith("(baseline model)")
    assert model.data(model.index(base, 0), Qt.ItemDataRole.FontRole).bold()
    assert model.data(model.index(base, 1), Qt.ItemDataRole.BackgroundRole) is None

    new = model.row_of_model(scored.new["id"])
    energy = next(i + 1 for i, c in enumerate(card["columns"]) if c["metric"] == "energy_rmse")
    assert model.data(model.index(new, energy), Qt.ItemDataRole.BackgroundRole) == VERDICT_COLORS["better"]
    assert float(model.data(model.index(new, energy))) > 0
    assert model.data(model.index(new, energy), ScorecardTableModel.CellRole)["verdict"] == "better"

    fresh = model.row_of_model(scored.fresh["id"])
    assert model.data(model.index(fresh, 1)) == "not run"
    assert model.data(model.index(fresh, 1), Qt.ItemDataRole.BackgroundRole) is None
    model.set_scorecard(None)
    assert model.rowCount() == 0 and model.columnCount() == 1


def test_library_runs_all_in_background_and_overlays_details(scored, monkeypatch):
    service = scored.service
    widget = ModelLibraryWidget(service=service)
    widget.refresh(scored.suite["id"])
    assert widget.model.rowCount() == 3
    assert "first production model" in widget.baseline_label.text()
    assert (scored.fresh["id"], scored.eos["id"]) in widget.run_jobs()

    widget.scope_combo.setCurrentIndex(widget.scope_combo.findData("suite"))
    task = widget.run_all()
    assert widget.run_selected_button.text() == "停止验证"
    assert _wait_until(lambda: widget._run_task is None)
    assert task.outcome == "succeeded" and widget.run_selected_button.text() == "验证所选势"
    assert widget.model.row_of_model(scored.fresh["id"]) >= 0

    widget.model_combo.setCurrentIndex(widget.model_combo.findData(scored.new["id"]))
    widget.comparison_picker.set_selected_ids([scored.old["id"]])
    assert widget.detail.title_label.text() == "EOS · 1 groups"
    assert "Reference: DFT, PBE" in widget.detail.info_label.text()
    assert len(widget.detail.eos_plots[0].listDataItems()) == 3

    widget.toggle_star()
    assert service.get_model(scored.new["id"])["starred"]
    assert widget.star_action.toolTip() == "Unstar"
    assert widget.star_action.isChecked()

    from NepTrainKit.ui.widgets import GetStrMessageBox

    monkeypatch.setattr(GetStrMessageBox, "exec", lambda box: (box.lineEdit.setText("better forces"), 1)[1])
    widget.set_baseline()
    suite = service.get_suite(scored.suite["id"])
    assert suite["baseline_model_id"] == scored.new["id"] and suite["baseline_note"] == "better forces"
    assert widget.model.row(widget.model.row_of_model(scored.new["id"]))["is_baseline"]


def test_library_run_all_can_be_cancelled(scored):
    widget = ModelLibraryWidget(service=scored.service)
    widget.refresh(scored.suite["id"])
    total = len(widget.run_jobs())
    task = widget.run_all()
    widget._on_run_clicked()  # the Run button becomes Cancel while running
    assert _wait_until(lambda: widget._run_task is None)
    assert task.outcome == "canceled"
    runs = [scored.service.latest_run(m, s) for m, s in widget.run_jobs()]
    assert sum(run is not None for run in runs) < total


def test_open_scorecard_selects_suite_and_model(scored):
    service = scored.service
    other = service.create_suite("EOS only", [scored.eos["id"]])
    widget = ModelLibraryWidget(service=service)
    widget.refresh(other["id"])
    widget.open_scorecard(scored.suite["id"], scored.new["id"])
    assert widget.current_suite_id() == scored.suite["id"]
    assert widget.current_model_row()["model_id"] == scored.new["id"]


def test_eos_overview_has_fixed_columns_and_normalizes_each_group(make_service, tmp_path):
    import numpy as np

    service = make_service()
    sets = [_import(service, tmp_path, eos_frames(curvature=0.5 + i * 0.1), f"group-{i}") for i in range(10)]
    suite = service.create_suite("Ten EOS groups", [item["id"] for item in sets])
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(suite["id"])
    widget.resize(1180, 740)
    widget.show()
    QApplication.processEvents()
    assert widget.table.isHidden()
    assert not widget.metrics_button.isChecked()
    assert widget.detail._grid_columns == 3
    widget.metrics_button.click()
    assert not widget.table.isHidden()
    widget.metrics_button.click()
    assert widget.table.isHidden()
    assert widget.metrics_model.rowCount() == 10
    assert widget.metrics_model.columnCount() == 5
    assert len(widget.detail.eos_plots) == 10
    assert widget.detail.relative_checkbox.isChecked()
    assert widget.detail.offset_checkbox.isChecked()
    for plot, benchmark in zip(widget.detail.eos_plots, sets):
        x, y = plot.listDataItems()[0].getData()
        summary = benchmark["summary"]
        np.testing.assert_allclose(x, summary["volumes"])
        np.testing.assert_allclose(y, (np.asarray(summary["energies"]) - min(summary["energies"])) * 1000)
        assert min(y) == 0
    # Each plot has its own scale; a broad scan cannot flatten another group.
    assert widget.detail.eos_plots[0].viewRange()[1] != widget.detail.eos_plots[-1].viewRange()[1]
    widget.detail.groupFocused.emit("G4")
    assert len(widget.detail.eos_plots[0].listDataItems()) == 1
    assert widget.detail._groups[0][0] == "G4"
    widget.table.selectionModel().select(widget.metrics_model.index(5, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    assert [group[0] for group in widget.detail._groups] == ["G4", "G6"]
    assert len(widget.detail.eos_plots) == 2
    widget.all_groups_button.click()
    assert len(widget.detail.eos_plots) == 10
    widget.resize(800, 740)
    QApplication.processEvents()
    assert widget.detail._grid_columns == 2
    assert widget.detail.grid_scroll.horizontalScrollBar().maximum() == 0
    widget.deleteLater()


def test_eos_relative_zero_preserves_model_offset_and_refresh_selection(make_service, tmp_path):
    import numpy as np

    service = make_service()
    first = _import(service, tmp_path, eos_frames(), "first")
    second = _import(service, tmp_path, eos_frames(curvature=0.8), "second")
    model = _capture(service, tmp_path, "offset-model")
    make_service(**{model["content_hash"]: (1.0, 0.7)})
    suite = service.create_suite("EOS", [first["id"], second["id"]])
    service.run_suite(model["id"], suite["id"])
    service.set_baseline(suite["id"], model["id"])
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(suite["id"], model["id"])
    widget.comparison_picker.set_selected_ids([])
    widget.table.selectRow(1)
    widget.detail.offset_checkbox.setChecked(False)
    ref, pred = widget.detail.eos_plots[0].listDataItems()
    np.testing.assert_allclose(pred.getData()[1] - ref.getData()[1], 700)
    assert widget.metrics_model.data(widget.metrics_model.index(1, 1), Qt.ItemDataRole.BackgroundRole) is None  # baseline itself
    widget.detail.offset_checkbox.setChecked(True)
    ref, pred = widget.detail.eos_plots[0].listDataItems()
    np.testing.assert_allclose(pred.getData()[1], ref.getData()[1], atol=1e-12)
    widget.detail.offset_checkbox.setChecked(False)
    widget.detail.relative_checkbox.setChecked(False)
    ref, pred = widget.detail.eos_plots[0].listDataItems()
    np.testing.assert_allclose(ref.getData()[1], second["summary"]["energies"])
    np.testing.assert_allclose(pred.getData()[1] - ref.getData()[1], 0.7)
    widget.refresh()
    assert widget.comparison_picker.selected_ids() == []
    assert [i.row() for i in widget.table.selectionModel().selectedRows()] == [1]
    assert len(widget.detail.eos_plots[0].listDataItems()) == 2
    widget.deleteLater()


def _load_single_eos(dialog, path):
    dialog.load_file(path)
    dialog.group_field_combo.setCurrentIndex(dialog.group_field_combo.findData(""))


# -- import dialog ----------------------------------------------------------------

def test_import_dialog_requires_explicit_type_and_previews_reference(make_service, tmp_path):
    service = make_service()
    host = QWidget()
    dialog = BenchmarkImportDialog(host, service.suites())
    assert not dialog.yesButton.isEnabled()
    assert "Choose a validation type" in dialog.status_label.text()
    assert not dialog.browse_button.isEnabled()
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("eos"))
    assert dialog.browse_button.isEnabled()

    _load_single_eos(dialog, write_xyz(tmp_path / "few.xyz", eos_frames()[:3]))
    assert dialog.current_protocol().kind == "eos"  # never silently falls back to EFV
    assert any(issue.level == "error" for issue in dialog.issues)
    assert dialog.reference_summary is None and not dialog.preview_plot.listDataItems()

    _load_single_eos(dialog, write_xyz(tmp_path / "scan.xyz", eos_frames()))
    assert dialog.current_protocol().kind == "eos"
    assert dialog.issues == []
    assert len(dialog.preview_plot.listDataItems()) == 1
    assert "v0" in dialog.reference_summary["fit"]
    assert "Å³/atom" in dialog.preview_summary.text()
    dialog.name_edit.setText("PbTe rocksalt EOS")
    dialog.suite_combo.setText("PbTe daily validation")
    assert dialog.yesButton.isEnabled() and dialog.validate()
    assert dialog.preview_title.text() == "Reference curve"

    shifted = eos_frames(minimum=1.2)
    _load_single_eos(dialog, write_xyz(tmp_path / "edge.xyz", shifted))
    assert [issue.level for issue in dialog.issues] == ["warning"]
    assert "Fit unavailable" in dialog.preview_summary.text()
    assert len(dialog.preview_plot.listDataItems()) == 1
    assert dialog.yesButton.isEnabled()  # curve-only comparisons remain useful

    (tmp_path / "bad.xyz").write_text("not an xyz file\n")
    _load_single_eos(dialog, tmp_path / "bad.xyz")
    assert not dialog.yesButton.isEnabled() and "cannot be read" in dialog.status_label.text()
    assert dialog.reference_summary is None and not dialog.preview_plot.listDataItems()

    broken = eos_frames()
    broken[2].info["_ref_energy"] = float("nan")
    _load_single_eos(dialog, write_xyz(tmp_path / "nan.xyz", broken))
    assert dialog.current_protocol().kind == "eos"
    assert [issue.level for issue in dialog.issues] == ["error"]
    assert not dialog.yesButton.isEnabled() and dialog.reference_summary is None


def test_import_preview_invalidates_on_file_edit_and_type_change(make_service, tmp_path):
    host = QWidget()
    dialog = BenchmarkImportDialog(host, [], "PbTe")
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("eos"))
    path = write_xyz(tmp_path / "scan.xyz", eos_frames())
    _load_single_eos(dialog, path)
    dialog.notes_edit.setText("Synthetic test")
    assert dialog.validate()
    dialog.path_edit.setText(str(tmp_path / "different.xyz"))
    assert not dialog.validate() and dialog.reference_summary is None
    assert not dialog.preview_plot.listDataItems()
    _load_single_eos(dialog, path)
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("efv"))
    assert dialog.validate() and dialog.preview_plot.isHidden()
    assert "frames" in dialog.reference_summary
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("eos"))
    assert dialog.validate()
    Path(path).write_text("changed after preview")
    service = make_service()
    with pytest.raises(ValueError, match="file changed"):
        dialog.import_into(service, lambda error: True)
    assert service.sets() == [] and dialog.reference_summary is None


def test_reference_dataset_can_be_previewed_without_models(make_service, tmp_path):
    service = make_service()
    eos = _import(service, tmp_path, eos_frames(), "eos")
    efv = _import(service, tmp_path, efv_frames(), "efv")
    suite = service.create_suite("PbTe", [eos["id"], efv["id"]])
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(suite["id"])
    assert widget.model.rowCount() == 0
    widget.property_combo.setCurrentIndex(widget.property_combo.findData("eos"))
    assert widget.detail._args[0]["id"] == eos["id"]
    assert len(widget.detail.eos_plots[0].listDataItems()) == 1
    widget.property_combo.setCurrentIndex(widget.property_combo.findData("efv"))
    assert widget.detail._args[0]["id"] == efv["id"]
    assert not widget.detail.plot.listDataItems()


def test_eos_flow_chinese_labels_and_errors(make_service, tmp_path):
    app = QApplication.instance()
    translator = QTranslator(app)
    catalog = Path(__file__).resolve().parents[1] / "src/NepTrainKit/translations/neptrainkit_zh_CN.qm"
    assert translator.load(str(catalog))
    app.installTranslator(translator)
    try:
        host = QWidget()
        dialog = BenchmarkImportDialog(host, [], "PbTe 验证")
        assert dialog.titleLabel.text() == "导入参考基准集"
        assert dialog.supplementary_button.text() == "补充信息（可选）"
        assert dialog.group_field_label.text() == "分组字段"
        assert "每个构型的注释行" in dialog.group_hint.text()
        assert "config_type=B1" in dialog.group_hint.text()
        assert "所有体积点都写相同值" in dialog.group_hint.text()
        assert "单组扫描 · 使用整个文件" in dialog.group_hint.text()
        assert dialog.status_label.text() == "请先选择验证类型。"
        dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("eos"))
        instructions = dialog.requirements_label.text()
        assert "每组 EOS 至少需要 5 个不同体积点" in instructions
        assert "总能量（eV），请勿提前除以原子数" in instructions
        assert "E/N（eV/atom）和 V/N（Å³/atom）" in instructions
        _load_single_eos(dialog, write_xyz(tmp_path / "few.xyz", eos_frames()[:3]))
        assert "EOS 至少需要 5 个" in dialog.issues_label.text()
        _load_single_eos(dialog, write_xyz(tmp_path / "scan.xyz", eos_frames()))
        assert dialog.preview_title.text() == "参考曲线"
        assert "参考拟合" in dialog.preview_summary.text()
        widget = ModelLibraryWidget(service=make_service())
        assert widget.import_action.text() == "导入参考基准集"
        assert widget.import_action.toolTip() == "导入参考基准集"
        assert widget.baseline_button.accessibleName() == "设为基准势版本"
        assert "设为基准势版本" in widget.baseline_button.toolTip()
    finally:
        app.removeTranslator(translator)


def test_import_dialog_asks_before_copying_large_files(make_service, tmp_path):
    service = make_service()
    original = service.import_set
    service.import_set = lambda path, **kwargs: original(path, max_bytes=10, **kwargs)
    host = QWidget()
    dialog = BenchmarkImportDialog(host, [], "PbTe")
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("eos"))
    _load_single_eos(dialog, write_xyz(tmp_path / "scan.xyz", eos_frames()))
    dialog.notes_edit.setText("VASP PBE")

    asked = []
    assert dialog.import_into(service, lambda error: asked.append(error) or False) is None
    assert isinstance(asked[0], SetTooLargeError) and service.sets() == []

    records = dialog.import_into(service, lambda error: True)
    assert len(records) == 1
    record = records[0]
    assert record["provenance"] == {"notes": "VASP PBE"}
    suite = service.suites()[0]
    assert suite["name"] == "PbTe" and [s["id"] for s in service.suite_sets(suite["id"])] == [record["id"]]


@pytest.mark.parametrize("protocol", ["eos", "efv"])
@pytest.mark.parametrize("notes", ["", "PBE, 520 eV; same magnetic branch"])
def test_import_optional_notes_round_trip(make_service, tmp_path, protocol, notes):
    service = make_service()
    host = QWidget()
    dialog = BenchmarkImportDialog(host, [], "Reference checks")
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData(protocol))
    frames = eos_frames() if protocol == "eos" else efv_frames()
    path = write_xyz(tmp_path / "reference.xyz", frames)
    if protocol == "eos":
        _load_single_eos(dialog, path)
    else:
        dialog.load_file(path)
    assert dialog.validate() and dialog.yesButton.isEnabled()
    assert dialog.supplementary_panel.isHidden()
    if notes:
        dialog.supplementary_button.click()
        assert not dialog.supplementary_panel.isHidden()
        dialog.notes_edit.setText(f"  {notes}  ")
        dialog.supplementary_button.click()
        assert dialog.supplementary_panel.isHidden()
        dialog.supplementary_button.click()
        assert dialog.notes_edit.text().strip() == notes
    record, = dialog.import_into(service, lambda _: True)
    saved = service.get_set(record["id"])
    assert saved["provenance"] == ({"notes": notes} if notes else {})
    assert saved["source_ref"]["path"] == str(path)
    assert len(service.set_frames(saved["id"])) == len(frames)
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(service.suites()[0]["id"])
    widget.property_combo.setCurrentIndex(widget.property_combo.findData(protocol))
    info = widget.detail.info_label.text()
    assert "Reference: ?" not in info and "DFT" not in info
    if notes:
        assert f"Notes: {notes}" in info
    else:
        assert "Notes:" not in info
    if protocol == "eos":
        assert widget.detail.eos_plots[0].listDataItems()[0].name() == "Reference"
    widget.deleteLater()
    dialog.deleteLater()
    host.deleteLater()


# -- viewer chip ------------------------------------------------------------------

def test_grouped_eos_automatically_allows_warnings_and_blocks_errors(make_service, tmp_path):
    service = make_service()
    host = QWidget()
    dialog = BenchmarkImportDialog(host, [], "Multi-phase validation")
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("eos"))
    frames = []
    for label, scan in (("A", eos_frames()), ("B", eos_frames(minimum=1.2)), ("short", eos_frames()[:3])):
        for frame in scan:
            frame.info["config_type"] = label
            frame.info["all_scans"] = "mixed"
        frames.extend(scan)
    path = write_xyz(tmp_path / "mixed.xyz", frames)
    dialog.load_file(path)
    assert dialog.group_field_combo.currentData() is None
    assert not dialog.preview_plot.listDataItems()
    dialog.group_field_combo.setCurrentIndex(dialog.group_field_combo.findData("config_type"))
    assert len(dialog.eos_groups) == 3
    assert dialog.eos_groups[2]["summary"] is None
    assert not dialog.validate()
    dialog.group_table.item(2, 0).setCheckState(Qt.CheckState.Unchecked)
    assert dialog.validate() and dialog.yesButton.isEnabled()
    assert dialog.group_table.currentRow() == 0  # other valid groups need no manual review
    assert dialog.group_table.item(0, 3).text() == "Checks passed"
    assert dialog.group_table.item(1, 3).text() == "Ready · warning"
    dialog.group_table.setCurrentCell(0, 0)
    dialog.overlay_groups.setChecked(True)
    assert [curve.name() for curve in dialog.preview_plot.listDataItems()] == ["A", "B"]
    records = dialog.import_into(service, lambda _: True)
    assert len(records) == 2
    assert all(record["provenance"] == {} for record in records)
    assert records[1]["summary"]["warnings"]  # warning retained after successful import
    assert "v0" not in records[1]["summary"]["fit"]
    suite = service.suites()[0]
    assert {r["id"] for r in service.suite_sets(suite["id"])} == {r["id"] for r in records}
    assert all(len(service.set_frames(r["id"])) == 7 for r in records)
    dialog.group_field_combo.setCurrentIndex(dialog.group_field_combo.findData("all_scans"))
    assert len(dialog.eos_groups) == 1 and not dialog.validate()
    assert not dialog.preview_plot.listDataItems()
    dialog.group_field_combo.setCurrentIndex(dialog.group_field_combo.findData("config_type"))
    assert not dialog.validate()  # the invalid short group is selected again


def test_grouped_eos_missing_metadata_is_never_silently_dropped(tmp_path):
    host = QWidget()
    dialog = BenchmarkImportDialog(host, [], "EOS")
    dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("eos"))
    frames = eos_frames()
    for frame in frames[:-1]:
        frame.info["scan_id"] = "A"
    dialog.load_file(write_xyz(tmp_path / "missing.xyz", frames))
    dialog.group_field_combo.setCurrentIndex(dialog.group_field_combo.findData("scan_id"))
    assert "Frame 7" in dialog._group_error
    assert dialog.eos_groups == [] and not dialog.validate()
    assert not dialog.preview_plot.listDataItems()


@pytest.mark.parametrize("language", ["en", "zh"])
def test_import_dialog_resizes_without_squeezing_controls(tmp_path, language):
    app = QApplication.instance()
    translator = QTranslator(app)
    if language == "zh":
        translation = Path(__file__).parents[1] / "src/NepTrainKit/translations/neptrainkit_zh_CN.qm"
        assert translator.load(str(translation))
        app.installTranslator(translator)
    host = QWidget()
    host.resize(1280, 900)
    host.show()
    dialog = BenchmarkImportDialog(host, [], "EOS")
    try:
        dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData("eos"))
        frames = eos_frames()
        for frame in frames:
            frame.info["config_type"] = "B1"
        dialog.load_file(write_xyz(tmp_path / "eos.xyz", frames))
        dialog.group_field_combo.setCurrentIndex(dialog.group_field_combo.findData("config_type"))
        dialog.supplementary_button.click()
        dialog.show()
        for width, height in [(1280, 900), (1000, 700), (800, 600), (1280, 900)]:
            host.resize(width, height)
            for _ in range(3):
                app.processEvents()
            assert host.rect().contains(dialog.widget.geometry())
            for control in (
                dialog.path_edit, dialog.browse_button, dialog.protocol_combo,
                dialog.group_field_combo, dialog.name_edit,
                dialog.notes_edit, dialog.suite_combo,
            ):
                assert control.height() >= max(32, control.sizeHint().height())
            assert dialog.preview_plot.height() >= 260
            scroll = dialog.body_scroll
            assert scroll.widget().width() <= scroll.viewport().width()
            if width >= 1280:
                assert dialog.fields_panel.x() < dialog.preview_panel.x()
                assert dialog.fields_panel.y() == dialog.preview_panel.y()
            else:
                assert dialog.fields_panel.geometry().bottom() < dialog.preview_panel.y()
                assert scroll.verticalScrollBar().maximum() > 0
            # Scrolling can reach the final check result without moving the footer.
            button_pos = dialog.yesButton.mapTo(host, dialog.yesButton.rect().topLeft())
            scroll.ensureWidgetVisible(dialog.issues_label)
            app.processEvents()
            result_pos = dialog.issues_label.mapTo(scroll.viewport(), dialog.issues_label.rect().center())
            assert scroll.viewport().rect().contains(result_pos)
            assert dialog.yesButton.mapTo(host, dialog.yesButton.rect().topLeft()) == button_pos
            assert host.rect().contains(dialog.yesButton.mapTo(host, dialog.yesButton.rect().bottomRight()))
            assert dialog.group_field_combo.currentData() == "config_type"
    finally:
        dialog.hide()
        host.close()
        dialog.deleteLater()
        host.deleteLater()
        if language == "zh":
            app.removeTranslator(translator)


def test_status_summary_counts_regressions_on_the_preferred_suite(scored):
    service = scored.service
    service.set_baseline(scored.suite["id"], scored.new["id"])
    status = summarize_model_status(service, scored.old["id"])
    assert status["suite_name"] == "PbTe" and status["evaluated"] and status["regressions"] > 0
    other = service.create_suite("EOS only", [scored.eos["id"]])
    assert summarize_model_status(service, scored.old["id"], other["id"])["suite_id"] == other["id"]
    assert not summarize_model_status(service, scored.fresh["id"])["evaluated"]


def _viewer_stub(model_id, opened, imported=None):
    from qfluentwidgets import Action, DotInfoBadge, InfoBadge, TransparentDropDownToolButton

    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    def import_benchmark_set():
        if imported is not None:
            imported.append(True)

    stub = SimpleNamespace(
        validation_button=TransparentDropDownToolButton(),
        validation_status_action=Action("status"),
        validation_star_action=Action("star"),
        validation_save_action=Action("save"),
        validation_count_badge=InfoBadge.error("0"),
        validation_dot_badge=DotInfoBadge.info(),
        _validation_model_id=model_id,
        _validation_draft=None,
        _validation_library_root=library_module.resolve_library_root(),
        _validation_library_changed=False,
        _validation_status=None,
        _validation_starred=False,
        modelRegistryChanged=Mock(),
        _parent=SimpleNamespace(
            open_model_scorecard=lambda suite, model=None, **kwargs: opened.append((suite, model, kwargs["draft"]) if kwargs else (suite, model)),
            import_benchmark_set=import_benchmark_set,
        ),
        tr=lambda text: text,
        window=lambda: QWidget(),
    )
    for name in (
        "_validation_service",
        "refresh_validation_chip",
        "_show_validation_badge",
        "open_validation_scorecard",
        "import_validation_benchmark",
        "toggle_validation_star",
        "save_validation_model",
    ):
        setattr(stub, name, getattr(ShowNepWidget, name).__get__(stub))
    return stub


def _badge(stub):
    """What the footer shows next to the icon: a count, a dot, or nothing."""
    if not stub.validation_count_badge.isHidden():
        return int(stub.validation_count_badge.text())
    return "dot" if not stub.validation_dot_badge.isHidden() else None


def test_viewer_status_line_and_badge_follow_the_suite(scored, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    service, opened = scored.service, []
    # With the newer model as baseline, the old one regresses: a red count.
    service.set_baseline(scored.suite["id"], scored.new["id"])
    stub = _viewer_stub(scored.old["id"], opened)
    stub.refresh_validation_chip()
    count = summarize_model_status(service, scored.old["id"])["regressions"]
    assert count > 0 and _badge(stub) == count
    assert stub.validation_status_action.text() == f"Validation: PbTe · {count} regressions"
    assert stub.validation_button.toolTip() == stub.validation_status_action.text()
    stub.open_validation_scorecard()
    assert opened == [(scored.suite["id"], scored.old["id"])]

    stub._validation_model_id = scored.new["id"]
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: PbTe · baseline model" and _badge(stub) is None

    # No regressions against the baseline: no badge at all.
    service.set_baseline(scored.suite["id"], scored.old["id"])
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: PbTe · 0 regressions" and _badge(stub) is None

    stub._validation_model_id = scored.fresh["id"]
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: PbTe · not run" and _badge(stub) == "dot"
    for widget in (stub.validation_count_badge, stub.validation_dot_badge):
        widget.close()


def test_viewer_entry_guides_first_import(scored, tmp_path, monkeypatch):
    from NepTrainKit.core.validation import Library, ValidationService

    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    opened, imported = [], []
    # A model no suite covers: the menu says so, the dot asks for attention.
    nickel = tmp_path / "ni" / "nep.txt"
    nickel.parent.mkdir()
    nickel.write_text("nep4 1 Ni\ncutoff 8 4 74 10\n")
    stub = _viewer_stub(scored.service.capture_model(nickel)["id"], opened, imported)
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: no matching set" and _badge(stub) == "dot"
    assert "Import a reference dataset" in stub.validation_button.toolTip()
    stub.import_validation_benchmark()
    assert imported == [True]

    # No captured model: a neutral status, nothing to star.
    stub._validation_model_id = None
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: no NEP model open" and _badge(stub) is None
    assert not stub.validation_star_action.isEnabled()
    stub.open_validation_scorecard()
    assert opened == [(None, None)]

    # An empty library says it is not set up yet.
    empty = ValidationService(Library(tmp_path / "empty-lib"))
    monkeypatch.setenv(library_module.ENV_VAR, str(empty.library.root))
    path = tmp_path / "first" / "nep.txt"
    path.parent.mkdir()
    path.write_text(MODEL_HEADER)
    stub._validation_model_id = empty.capture_model(path)["id"]
    stub._validation_library_root = empty.library.root
    stub.refresh_validation_chip()
    assert stub.validation_status_action.text() == "Validation: not set up" and _badge(stub) == "dot"
    for widget in (stub.validation_count_badge, stub.validation_dot_badge):
        widget.close()


def test_fresh_model_is_selected_and_can_be_validated_alone(scored):
    service = scored.service
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(scored.suite["id"], scored.fresh["id"])
    row = widget.current_model_row()
    assert row["model_id"] == scored.fresh["id"]
    assert all(cell["status"] == "missing" for cell in row["cells"])
    assert widget.run_current_action.isEnabled()
    widget.refresh()  # Refresh/showEvent must retain the newly opened version.
    assert widget.current_model_row()["model_id"] == scored.fresh["id"]
    with patch.object(service, "evaluate", wraps=service.evaluate) as evaluate:
        widget.scope_combo.setCurrentIndex(widget.scope_combo.findData("suite"))
        widget.run_current_action.trigger()
        assert not widget.run_current_action.isEnabled()
        assert _wait_until(lambda: widget._run_task is None)
        assert {call.args for call in evaluate.call_args_list} == {
            (scored.fresh["id"], scored.eos["id"]), (scored.fresh["id"], scored.efv["id"]),
        }
    row = widget.current_model_row()
    assert row["model_id"] == scored.fresh["id"] and row["comparison_complete"]
    assert widget.run_current_action.isEnabled()


def test_unsaved_viewer_to_temporary_validation_then_save(scored, tmp_path, monkeypatch):
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget
    from NepTrainKit.ui.widgets.validation_dialogs import SaveModelDialog

    service = scored.service
    monkeypatch.setenv(library_module.ENV_VAR, str(service.library.root))
    path = tmp_path / "unsaved.txt"
    path.write_text(MODEL_HEADER + "unsaved")
    stub = _viewer_stub(None, [])
    stub.nep_result_data = SimpleNamespace(nep_txt_path=path, data_xyz_path=None)
    ShowNepWidget._associate_model_version(stub)
    original_count = len(service.models())
    assert stub._validation_model_id is None and stub.validation_save_action.isEnabled()
    assert not stub.validation_star_action.isEnabled()
    assert "unsaved" in stub.validation_status_action.text()

    widget = ModelLibraryWidget(service=service)
    stub._parent.open_model_scorecard = widget.open_scorecard
    stub.open_validation_scorecard()
    assert widget.current_model_row()["model_id"] == 0
    assert len(service.models()) == original_count  # entering the scorecard is still read-only
    assert not widget.baseline_button.isEnabled() and not widget.star_action.isEnabled()
    with patch.object(SaveModelDialog, "exec", return_value=0):
        widget.save_current()
    assert len(service.models()) == original_count  # cancelling Save leaves no project/version
    old_bytes = path.read_bytes()
    path.write_text(MODEL_HEADER + "changed while viewing")
    widget.scope_combo.setCurrentIndex(widget.scope_combo.findData("suite"))
    widget.run_current_action.trigger()
    assert _wait_until(lambda: widget._run_task is None)
    row = widget.current_model_row()
    temporary_id = row["model_id"]
    assert row["temporary"] and row["complete"] and temporary_id > 0
    assert service.get_model(temporary_id)["project_id"] is None
    assert service.model_file(temporary_id).read_bytes() == old_bytes
    stub.refresh_validation_chip()
    assert stub._validation_model_id == temporary_id
    assert stub.validation_status_action.text().startswith("Temporary")

    def accept(dialog):
        dialog.project_combo.setText("New project")
        dialog.name_edit.setText("v1")
        return 1

    with patch.object(SaveModelDialog, "exec", accept), patch("NepTrainKit.ui.views.model_library.MessageManager"):
        widget.save_current()
    saved = service.get_model(temporary_id)
    assert saved["name"] == "v1" and saved["project_id"] is not None
    assert saved["dataset_ref"] == {} and not saved["auto_captured"]
    assert widget.current_model_row()["model_id"] == temporary_id
    assert widget.current_model_row()["complete"] and widget.baseline_button.isEnabled()
    stub.refresh_validation_chip()
    assert not stub.validation_status_action.text().startswith("Temporary")


def test_save_dialog_requires_explicit_project_and_version(make_service):
    from NepTrainKit.ui.widgets.validation_dialogs import SaveModelDialog

    host = QWidget()
    dialog = SaveModelDialog(host, make_service(), "v1")
    assert not dialog.yesButton.isEnabled()
    dialog.project_combo.setText("FeRh")
    assert dialog.yesButton.isEnabled()
    dialog.name_edit.clear()
    assert not dialog.yesButton.isEnabled()


@pytest.mark.parametrize("accept", [False, True])
def test_save_dialog_can_choose_library_without_retargeting_on_cancel(make_service, model_file, tmp_path, monkeypatch, accept):
    from NepTrainKit.config import Config
    from NepTrainKit.core.validation import Library, ValidationService
    from NepTrainKit.core.validation.service import ModelDraft
    from NepTrainKit.ui.widgets.validation_dialogs import SaveModelDialog, save_model_to_project

    monkeypatch.delenv(library_module.ENV_VAR, raising=False)
    service = make_service()
    target = tmp_path / "chosen-library"
    previous = Config.get("validation", "library_path", "")
    Config.set("validation", "library_path", str(service.library.root))
    monkeypatch.setattr("NepTrainKit.ui.widgets.validation_dialogs.QFileDialog.getExistingDirectory", lambda *args: str(target))

    def choose(dialog):
        assert dialog.location_button.isEnabled()
        dialog._choose_library()
        dialog.project_combo.setText("Chosen project")
        dialog.name_edit.setText("v2")
        return int(accept)

    try:
        with patch.object(SaveModelDialog, "exec", choose):
            record = save_model_to_project(QWidget(), service, ModelDraft.from_file(model_file))
        destination = ValidationService(Library(target))
        assert service.models() == [] and service.projects() == []
        if accept:
            assert record["library_root"] == str(target.resolve())
            assert destination.get_model(record["id"])["name"] == "v2"
            assert Config.get("validation", "library_path") == str(target.resolve())
        else:
            assert record is None and destination.models() == [] and destination.projects() == []
            assert Config.get("validation", "library_path") == str(service.library.root)
    finally:
        Config.set("validation", "library_path", previous)


def test_viewer_save_can_add_a_second_project_without_reopening(model_file, make_service, monkeypatch):
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget
    from NepTrainKit.ui.widgets.validation_dialogs import SaveModelDialog

    service = make_service()
    monkeypatch.setenv(library_module.ENV_VAR, str(service.library.root))
    stub = _viewer_stub(None, [])
    stub.nep_result_data = SimpleNamespace(nep_txt_path=model_file, data_xyz_path=None)
    ShowNepWidget._associate_model_version(stub)

    def accept(dialog):
        dialog.project_combo.setText("A" if not service.projects() else "B")
        dialog.name_edit.setText("v1")
        return 1

    with patch.object(SaveModelDialog, "exec", accept), patch("NepTrainKit.ui.pages.show_nep.MessageManager"):
        stub.save_validation_model()
        first = stub._validation_model_id
        stub.save_validation_model()
    assert first != stub._validation_model_id
    assert stub.modelRegistryChanged.emit.call_args_list == [((first,),), ((stub._validation_model_id,),)]
    assert len(service.models()) == 2 and len(service.projects()) == 2
    assert service.get_model(first)["content_hash"] == service.get_model(stub._validation_model_id)["content_hash"]


def test_unsaved_model_is_visible_without_any_suites(make_service, model_file):
    from NepTrainKit.core.validation.service import ModelDraft
    from NepTrainKit.ui.widgets.validation_dialogs import SaveModelDialog

    service = make_service()
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(None, draft=ModelDraft.from_file(model_file))
    assert widget.current_model_row()["model_id"] == 0
    assert widget.save_action.isEnabled() and not widget.run_current_action.isEnabled()
    assert service.models() == [] and service.projects() == []

    def accept(dialog):
        dialog.project_combo.setText("First project")
        dialog.name_edit.setText("v1")
        return 1

    with patch.object(SaveModelDialog, "exec", accept), patch("NepTrainKit.ui.views.model_library.MessageManager"):
        widget.save_current()
    assert widget.current_model_row()["name"] == "v1"
    assert not widget.current_model_row()["temporary"]
    assert not widget.run_current_action.isEnabled()


def test_refresh_clears_stale_actions_when_library_cannot_open(scored):
    widget = ModelLibraryWidget(service=scored.service)
    widget.open_scorecard(scored.suite["id"], scored.new["id"])
    with patch.object(widget, "service", side_effect=OSError("unavailable")), patch("NepTrainKit.ui.views.model_library.MessageManager"):
        widget.refresh()
    assert widget.model.rowCount() == 0
    assert not widget.save_action.isEnabled() and not widget.run_current_action.isEnabled()


def test_partial_or_failed_suite_never_looks_complete(scored, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    service = scored.service
    stub = _viewer_stub(scored.fresh["id"], [])
    service.evaluate(scored.fresh["id"], scored.eos["id"])
    stub.refresh_validation_chip()
    assert "1/2 sets complete" in stub.validation_status_action.text()
    assert _badge(stub) == "dot"
    with patch.object(service, "predictor_factory", side_effect=ValueError("test failure")):
        service.evaluate(scored.fresh["id"], scored.efv["id"])
    stub.refresh_validation_chip()
    assert "1 failed" in stub.validation_status_action.text() and _badge(stub) == "dot"
    service.evaluate(scored.fresh["id"], scored.efv["id"])
    stub.refresh_validation_chip()
    assert "0 regressions" in stub.validation_status_action.text() and _badge(stub) is None


def test_missing_baseline_runs_never_looks_comparable(scored, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    service = scored.service
    service.set_baseline(scored.suite["id"], scored.fresh["id"])
    stub = _viewer_stub(scored.new["id"], [])
    stub.refresh_validation_chip()
    assert "baseline model comparison incomplete" in stub.validation_status_action.text()
    assert _badge(stub) == "dot"
    service.evaluate(scored.fresh["id"], scored.eos["id"])
    stub.refresh_validation_chip()
    assert "baseline model comparison incomplete" in stub.validation_status_action.text()


@pytest.mark.parametrize("action", ["refresh_validation_chip", "toggle_validation_star", "open_validation_scorecard"])
def test_library_switch_invalidates_viewer_ids(scored, tmp_path, monkeypatch, action):
    from NepTrainKit.core.validation import Library, ValidationService

    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    opened = []
    stub = _viewer_stub(scored.old["id"], opened)
    stub.refresh_validation_chip()
    other_service = ValidationService(Library(tmp_path / "other-lib"))
    other = _capture(other_service, tmp_path, "unrelated")
    assert other["id"] == scored.old["id"]
    monkeypatch.setenv(library_module.ENV_VAR, str(other_service.library.root))
    with patch("NepTrainKit.ui.pages.show_nep.MessageManager"):
        getattr(stub, action)()
    assert not other_service.get_model(other["id"])["starred"]
    assert stub._validation_model_id is None and not stub.validation_star_action.isEnabled()
    assert stub.validation_status_action.text() == "Validation: library changed"
    if action == "open_validation_scorecard":
        assert opened == [(None, None)]
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    stub.nep_result_data = SimpleNamespace(nep_txt_path=scored.old["source_path"], data_xyz_path=None)
    ShowNepWidget._associate_model_version(stub)
    assert not stub._validation_library_changed
    assert stub._validation_model_id is None
    assert stub._validation_draft.content_hash == scored.old["content_hash"]
    assert len(other_service.models()) == 1  # reopening does not save into the new library


def test_library_switch_does_not_star_an_old_scorecard_id(scored, tmp_path, monkeypatch):
    from NepTrainKit.core.validation import Library, ValidationService

    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    widget = ModelLibraryWidget()
    widget.open_scorecard(scored.suite["id"], scored.old["id"])
    other_service = ValidationService(Library(tmp_path / "other-lib"))
    other = _capture(other_service, tmp_path, "unrelated")
    monkeypatch.setenv(library_module.ENV_VAR, str(other_service.library.root))
    widget.toggle_star()
    assert not other_service.get_model(other["id"])["starred"]
    # The refreshed selector lists saved models even without a validation plan,
    # but the stale click must not act on an equal numeric id in another library.
    assert widget._library_root == other_service.library.root
    assert widget.current_model_row()["name"] == "unrelated"


def test_viewer_menu_stars_the_current_version(scored, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    stub = _viewer_stub(scored.fresh["id"], [])
    stub.refresh_validation_chip()
    assert stub.validation_star_action.isEnabled()
    assert stub.validation_star_action.text() == "Star this version"
    with patch("NepTrainKit.ui.pages.show_nep.MessageManager"):
        stub.toggle_validation_star()
        assert scored.service.get_model(scored.fresh["id"])["starred"] is True
        assert stub.validation_star_action.text() == "Unstar this version"
        stub.toggle_validation_star()
    assert scored.service.get_model(scored.fresh["id"])["starred"] is False
    for widget in (stub.validation_count_badge, stub.validation_dot_badge):
        widget.close()


@pytest.fixture
def viewer(monkeypatch, tmp_path):
    monkeypatch.setenv(library_module.ENV_VAR, str(tmp_path / "lib"))
    from NepTrainKit.config import Config
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    Config()
    widget = ShowNepWidget()
    yield widget
    widget.deleteLater()


def test_viewer_footer_has_compact_validation_entry(viewer):
    assert not viewer.validation_button.isHidden()
    actions = [action.text() for action in viewer.validation_menu.actions() if not action.isSeparator()]
    assert actions == [
        "Validation: no NEP model open",
        "Save to project…",
        "Star this version",
        "Open scorecard",
        "Import reference dataset…",
    ]
    assert not viewer.validation_status_action.isEnabled()  # a status line, not a command
    assert not viewer.validation_star_action.isEnabled()
    assert viewer.validation_count_badge.isHidden() and viewer.validation_dot_badge.isHidden()
    # Counts keep their natural width; only the left side stretches.
    layout = viewer.plot_widget_layout
    assert (layout.columnStretch(0), layout.columnStretch(1)) == (1, 0)


def test_viewer_elides_only_the_file_name(viewer):
    from PySide6.QtGui import QResizeEvent

    label = viewer.path_label
    name = "FeNiCoCrMn_high_entropy_alloy_train_v14.xyz"
    viewer._set_path_label_text("Current file: {file_name}", file_name=name)
    assert label.toolTip() == f"Current file: {name}"

    def at_width(width):
        label.resize(width, 30)
        QApplication.sendEvent(label, QResizeEvent(label.size(), label.size()))
        return label.text()

    metrics = label.fontMetrics()
    assert at_width(label.maximumWidth()) == f"Current file: {name}"
    roomy = at_width(metrics.horizontalAdvance("Current file: ") * 3)
    assert roomy.startswith("Current file: ") and "…" in roomy and roomy.endswith(".xyz")
    tight = at_width(metrics.horizontalAdvance("Current file: ") + 20)
    assert not tight.startswith("Current file") and tight  # the wording yields to the name


def test_viewer_capture_feeds_the_badge(scored, tmp_path, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(scored.service.library.root))
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    stub = _viewer_stub(None, [])
    model_path = tmp_path / "new" / "nep.txt"
    stub.nep_result_data = SimpleNamespace(nep_txt_path=model_path, data_xyz_path=None)
    ShowNepWidget._associate_model_version(stub)
    assert stub._validation_model_id == scored.new["id"]
    assert stub.validation_status_action.text().startswith("Validation: PbTe")
    for widget in (stub.validation_count_badge, stub.validation_dot_badge):
        widget.close()


# -- settings ---------------------------------------------------------------------

def test_settings_cleanup_previews_before_deleting(make_service, tmp_path, monkeypatch):
    import NepTrainKit.ui.pages.settings as settings_module

    service = make_service()
    monkeypatch.setenv(library_module.ENV_VAR, str(service.library.root))
    orphan = service.library.put_bytes(b"orphan bytes")
    widget = settings_module.SettingsWidget(None)
    widget.refresh_library_usage()
    content = widget.library_location_card.contentLabel.text()
    assert str(service.library.root) in content and "used" in content and "NEPTRAINKIT_LIBRARY" in content
    assert not widget.library_location_card.button.isEnabled()
    with patch.object(settings_module.QFileDialog, "getExistingDirectory") as chooser:
        widget.choose_library_location()
    chooser.assert_not_called()

    answers = []

    class FakeBox:
        def __init__(self, title, text, parent):
            answers.append(text)

        def exec(self):
            return len(answers) > 1

    with patch.object(settings_module, "MessageBox", FakeBox):
        widget.clean_up_library()  # declined after the dry run
        assert service.library.has_object(orphan) and "1 files" in answers[0]
        widget.clean_up_library()
    assert not service.library.has_object(orphan)


def test_settings_switches_library_and_keeps_old_files(make_service, tmp_path, monkeypatch):
    from NepTrainKit.config import Config
    from NepTrainKit.core.validation import ValidationService
    import NepTrainKit.ui.pages.settings as settings_module

    monkeypatch.delenv(library_module.ENV_VAR, raising=False)
    service = make_service()
    model = _capture(service, tmp_path, "original")
    previous = Config.get("validation", "library_path", "")
    Config.set("validation", "library_path", str(service.library.root))
    target = tmp_path / "new-library"
    target.mkdir()
    try:
        widget = settings_module.SettingsWidget(None)
        widget.refresh_library_usage()
        assert widget.library_location_card.button.isEnabled()
        with patch.object(settings_module.QFileDialog, "getExistingDirectory", return_value=str(target)), patch.object(settings_module, "MessageManager") as messages:
            widget.choose_library_location()
        assert Config.get("validation", "library_path") == str(target.resolve())
        assert ValidationService().library.root == target.resolve()
        assert ValidationService().models() == []
        assert service.model_file(model["id"]).exists()
        assert str(target.resolve()) in widget.library_location_card.contentLabel.text()
        assert "restart" not in messages.send_info_message.call_args.args[0]

        with patch.object(settings_module.QFileDialog, "getExistingDirectory", return_value=""):
            widget.choose_library_location()
        assert Config.get("validation", "library_path") == str(target.resolve())
        broken = tmp_path / "broken-library"
        broken.mkdir()
        (broken / "library.db").write_bytes(b"not a database")
        with patch.object(settings_module.QFileDialog, "getExistingDirectory", return_value=str(broken)), patch.object(settings_module, "MessageManager") as messages:
            widget.choose_library_location()
        messages.send_error_message.assert_called_once()
        assert Config.get("validation", "library_path") == str(target.resolve())
    finally:
        Config.set("validation", "library_path", previous)


# -- localisation -----------------------------------------------------------------

def test_scorecard_translates_stored_run_errors_but_keeps_unknown_diagnostics():
    from NepTrainKit import i18n

    column = {"set_name": "EOS", "label": "RMSE", "unit": "meV/atom", "better": "lower", "protocol": "eos"}
    errors = [
        ("failed", "Library object 0123456789ab is missing from /lib."),
        ("unsupported", "This protocol does not support spin models yet."),
        ("failed", "native crash 0x1f"),
    ]
    cells = [{"value": None, "status": status, "error": error, "verdict": None} for status, error in errors]
    card = {
        "columns": [dict(column, set_id=f"s{i}") for i in range(len(cells))],
        "rows": [
            {"model_id": 1, "name": "v1", "starred": False, "is_baseline": False, "regressions": 0, "cells": cells}
        ],
    }
    model = ScorecardTableModel()
    model.set_scorecard(card)
    app = QApplication.instance()

    def tooltip(column_index):
        return model.data(model.index(0, column_index), Qt.ItemDataRole.ToolTipRole)

    i18n.install_translator(app, "zh_CN")
    try:
        assert tooltip(1) == "模型库对象 0123456789ab 不在 /lib 中。"
        assert tooltip(2) == "此验证协议暂不支持自旋模型。"
        assert tooltip(3) == "native crash 0x1f"  # unknown diagnostics stay readable, not hidden
    finally:
        i18n.install_translator(app, "en_US")
    assert tooltip(1) == "Library object 0123456789ab is missing from /lib."


def test_multi_comparison_search_and_shared_legend_keep_identity(scored, tmp_path):
    service = scored.service
    second = _import(service, tmp_path, eos_frames(curvature=0.8), "second-phase")
    suite = service.create_suite("Two phases", [scored.eos["id"], second["id"]])
    service.set_baseline(suite["id"], scored.old["id"])
    for model in (scored.old, scored.new, scored.fresh):
        service.run_suite(model["id"], suite["id"])
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(suite["id"], scored.new["id"])
    picker = widget.comparison_picker
    picker.set_selected_ids([scored.old["id"], scored.fresh["id"], scored.new["id"]])
    assert picker.selected_ids() == [scored.old["id"], scored.fresh["id"]]  # current is never duplicated
    picker.search.setText("fresh")
    visible = [picker.list.item(i) for i in range(picker.list.count()) if not picker.list.item(i).isHidden()]
    assert len(visible) == 1 and "fresh" in visible[0].text()
    visible[0].setCheckState(Qt.CheckState.Unchecked)
    assert picker.selected_ids() == [scored.old["id"]]
    visible[0].setCheckState(Qt.CheckState.Checked)
    picker.search.clear()
    from PySide6.QtTest import QTest
    picker.open_popup()
    QTest.keyClick(picker.search, Qt.Key.Key_Escape)
    assert picker.popup.isHidden()
    assert all(len(plot.listDataItems()) == 4 for plot in widget.detail.eos_plots)
    keys = [None, scored.new["id"], scored.old["id"], scored.fresh["id"]]
    assert list(widget.detail.legend_buttons) == keys
    fresh_color = widget.detail._curves[scored.fresh["id"]][0].opts["pen"].color()
    widget.detail.legend_buttons[scored.fresh["id"]].click()
    assert all(not curve.isVisible() for curve in widget.detail._curves[scored.fresh["id"]])
    widget.refresh()
    assert picker.selected_ids() == [scored.old["id"], scored.fresh["id"]]
    assert not widget.detail.legend_buttons[scored.fresh["id"]].isChecked()
    widget.detail.comparisonRemoved.emit(scored.old["id"])
    assert picker.selected_ids() == [scored.fresh["id"]]
    assert widget.detail._curves[scored.fresh["id"]][0].opts["pen"].color() == fresh_color
    assert service.get_suite(suite["id"])["baseline_model_id"] == scored.old["id"]
    widget.detail.groupFocused.emit("G2")
    assert not widget.detail._curves[scored.fresh["id"]][0].isVisible()
    widget.detail.legend_buttons[scored.fresh["id"]].click()
    assert widget.detail._curves[scored.fresh["id"]][0].isVisible()
    widget.model_combo.setCurrentIndex(widget.model_combo.findData(scored.fresh["id"]))
    assert picker.selected_ids() == []
    assert len(widget.detail.eos_plots[0].listDataItems()) == 2
    widget.deleteLater()


def test_validate_selected_versions_includes_hidden_but_not_unselected(scored, tmp_path):
    service = scored.service
    outsider = _capture(service, tmp_path, "unselected")
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(scored.suite["id"], scored.new["id"])
    widget.comparison_picker.set_selected_ids([scored.fresh["id"]])
    assert "Not run" in widget.detail.legend_buttons[scored.fresh["id"]].text()
    widget.detail.legend_buttons[scored.fresh["id"]].click()
    widget.scope_combo.setCurrentIndex(widget.scope_combo.findData("suite"))
    with patch.object(service, "evaluate", wraps=service.evaluate) as evaluate:
        widget.run_selected_button.button.click()
        assert widget.run_selected_button.text() == "停止验证"
        assert _wait_until(lambda: widget._run_task is None)
        assert {call.args for call in evaluate.call_args_list} == {
            (model_id, set_id) for model_id in (scored.new["id"], scored.fresh["id"])
            for set_id in (scored.eos["id"], scored.efv["id"])
        }
    assert service.latest_run(outsider["id"], scored.eos["id"]) is None
    assert widget.comparison_picker.selected_ids() == [scored.fresh["id"]]
    assert not widget.detail.legend_buttons[scored.fresh["id"]].isChecked()
    assert "Not run" not in widget.detail.legend_buttons[scored.fresh["id"]].text()
    assert service.get_suite(scored.suite["id"])["baseline_model_id"] == scored.old["id"]
    assert widget.run_selected_button.isEnabled()
    widget.deleteLater()


def test_selected_validation_promotes_only_current_draft(scored):
    from NepTrainKit.core.validation.service import ModelDraft

    widget = ModelLibraryWidget(service=scored.service)
    draft = ModelDraft((MODEL_HEADER + "new draft").encode(), "/tmp/new-draft.txt")
    widget.open_scorecard(scored.suite["id"], draft=draft)
    widget.comparison_picker.set_selected_ids([scored.old["id"]])
    widget.scope_combo.setCurrentIndex(widget.scope_combo.findData("suite"))
    with patch.object(scored.service, "evaluate", wraps=scored.service.evaluate) as evaluate:
        widget.run_selected_button.button.click()
        assert _wait_until(lambda: widget._run_task is None)
        current = widget.current_model_row()
        assert current["temporary"] and current["model_id"] > 0
        assert {call.args[0] for call in evaluate.call_args_list} == {current["model_id"], scored.old["id"]}
    assert widget.comparison_picker.selected_ids() == [scored.old["id"]]
    widget.deleteLater()


def test_same_named_comparison_versions_are_independent(make_service, tmp_path):
    service = make_service()
    models = []
    for i in range(3):
        path = tmp_path / f"version-{i}.txt"
        path.write_text(MODEL_HEADER + str(i))
        models.append(service.capture_model(path, name="same"))
    eos = _import(service, tmp_path, eos_frames(), "eos")
    suite = service.create_suite("EOS", [eos["id"]])
    for model in models:
        service.run_suite(model["id"], suite["id"])
    widget = ModelLibraryWidget(service=service)
    widget.open_scorecard(suite["id"], models[0]["id"])
    widget.comparison_picker.set_selected_ids([model["id"] for model in models[1:]])
    assert len({entry[1] for entry in widget.detail._groups[0][2]}) == 3
    widget.detail.legend_buttons[models[1]["id"]].click()
    assert not widget.detail._curves[models[1]["id"]][0].isVisible()
    assert widget.detail._curves[models[2]["id"]][0].isVisible()
    widget.deleteLater()


def test_eos_focus_button_returns_to_grid_and_reset_fits_visible_curves(scored, tmp_path):
    import numpy as np
    from qfluentwidgets import TransparentToolButton

    service = scored.service
    second = _import(service, tmp_path, eos_frames(curvature=0.8), "second")
    suite = service.create_suite("Navigation", [scored.eos["id"], second["id"]])
    for model in (scored.new, scored.old):
        service.run_suite(model["id"], suite["id"])
    widget = ModelLibraryWidget(service=service)
    widget.resize(1180, 800)
    widget.open_scorecard(suite["id"], scored.new["id"])
    widget.comparison_picker.set_selected_ids([scored.old["id"]])
    widget.show()
    QApplication.processEvents()
    widget.detail.legend_buttons[scored.old["id"]].click()
    first, other = widget.detail.eos_plots
    other.setRange(xRange=(0, 1), yRange=(1000, 2000), padding=0)
    first.setRange(xRange=(0, 1), yRange=(1000, 2000), padding=0)
    widget.detail.eos_cards[0].findChild(TransparentToolButton, "resetEosView").click()
    QApplication.processEvents()
    x_range, y_range = first.viewRange()
    visible = [curve for curve in first.listDataItems() if curve.isVisible()]
    for curve in visible:
        x, y = curve.getData()
        assert x_range[0] <= min(x) <= max(x) <= x_range[1]
        assert y_range[0] <= min(y) <= max(y) <= y_range[1]
    assert x_range[0] > 1 and y_range[1] < 1000
    np.testing.assert_allclose(other.viewRange(), [[0, 1], [1000, 2000]])
    assert not widget.detail._curves[scored.old["id"]][0].isVisible()
    focus = widget.detail.eos_cards[0].findChild(TransparentToolButton, "toggleEosGroupFocus")
    focus.click()
    assert len(widget.detail.eos_cards) == 1
    assert widget.all_groups_button.text() == "Back to all groups"
    restore = widget.detail.eos_cards[0].findChild(TransparentToolButton, "toggleEosGroupFocus")
    assert restore.toolTip() == "Back to all groups"
    restore.click()
    assert len(widget.detail.eos_cards) == 2
    assert not widget.all_groups_button.isEnabled()
    assert widget.comparison_picker.selected_ids() == [scored.old["id"]]
    assert not widget.detail.legend_buttons[scored.old["id"]].isChecked()
    single = service.create_suite("Single group", [second["id"]])
    widget.open_scorecard(single["id"], scored.new["id"])
    assert widget.detail.eos_cards[0].findChild(TransparentToolButton, "toggleEosGroupFocus").isHidden()
    widget.deleteLater()
