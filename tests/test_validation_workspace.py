"""Suite topics and lifecycle preserve immutable references and historical evaluations."""

import sqlite3
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from NepTrainKit.core.validation import Library, ValidationService
from NepTrainKit.core.dataset.database import Database, SCHEMA_VERSION
from test_validation_framework import _import, eos_frames, efv_frames, make_service as make_service
from test_validation_ui import scored as scored
from test_accuracy_comparison_plot import curve_mode as curve_mode


def test_topics_and_aliases_are_suite_local_and_do_not_change_results(scored):
    service = scored.service
    other = service.create_suite("other", [scored.eos["id"]])
    group = service.create_group(scored.suite["id"], "Volume")
    old = service.latest_run(scored.new["id"], scored.eos["id"])
    service.assign_group(scored.suite["id"], [scored.eos["id"]], group["id"])
    service.rename_suite_item(scored.suite["id"], scored.eos["id"], "Volume response")
    item = next(r for r in service.suite_items(scored.suite["id"]) if r["id"] == scored.eos["id"])
    assert item["group_name"] == "Volume" and item["name"] == "Volume response"
    assert service.suite_items(other["id"])[0]["name"] == scored.eos["name"]
    assert service.suite_items(other["id"])[0]["group_id"] is None
    assert service.latest_run(scored.new["id"], scored.eos["id"])["id"] == old["id"]
    with pytest.raises(ValueError):
        service.assign_group(other["id"], [scored.eos["id"]], group["id"])
    service.remove_group(scored.suite["id"], group["id"])
    assert service.suite_items(scored.suite["id"])[0]["group_id"] is None


def test_remove_archive_and_restore_keep_snapshots_and_history(scored):
    service = scored.service
    other = service.create_suite("shared", [scored.eos["id"]])
    with pytest.raises(ValueError, match="other suites"):
        service.archive_reference(scored.suite["id"], [scored.eos["id"]])
    assert any(r["id"] == scored.eos["id"] for r in service.suite_sets(scored.suite["id"]))
    service.remove_from_suite(other["id"], [scored.eos["id"]])
    service.archive_reference(scored.suite["id"], [scored.eos["id"]])
    assert service.get_set(scored.eos["id"])["archived"]
    assert service.latest_run(scored.new["id"], scored.eos["id"])["status"] == "succeeded"
    assert service.set_frames(scored.eos["id"])
    service.restore_reference(scored.eos["id"])
    service.add_to_suite(other["id"], [scored.eos["id"]])
    assert service.suite_sets(other["id"]) and not service.get_set(scored.eos["id"])["archived"]


def test_v2_migration_adds_membership_metadata_without_losing_ids(scored):
    path = scored.service.library.db_path
    scored.service.db.engine.dispose()
    with sqlite3.connect(path) as conn:
        members = conn.execute("SELECT suite_id,set_id,position FROM benchmark_suite_sets").fetchall()
        runs = conn.execute("SELECT id,model_id,set_id FROM benchmark_runs").fetchall()
        conn.execute("DROP TABLE benchmark_suite_sets")
        conn.execute("DROP TABLE benchmark_suite_groups")
        conn.execute(
            "CREATE TABLE benchmark_suite_sets(suite_id INTEGER NOT NULL, set_id VARCHAR(64) NOT NULL, position INTEGER NOT NULL, PRIMARY KEY(suite_id,set_id))"
        )
        conn.executemany("INSERT INTO benchmark_suite_sets VALUES(?,?,?)", members)
        conn.execute("PRAGMA user_version=2")
    upgraded = Database(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert {"group_id", "display_name"} <= {r[1] for r in conn.execute("PRAGMA table_info(benchmark_suite_sets)")}
        assert conn.execute("SELECT suite_id,set_id,position FROM benchmark_suite_sets").fetchall() == members
        assert conn.execute("SELECT id,model_id,set_id FROM benchmark_runs").fetchall() == runs
        assert conn.execute("SELECT DISTINCT group_id,display_name FROM benchmark_suite_sets").fetchall() == [
            (None, "")
        ]
    upgraded.engine.dispose()


def test_workspace_separates_navigation_checks_and_model_selection(scored):
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    service = scored.service
    group = service.create_group(scored.suite["id"], "Volume")
    service.assign_group(scored.suite["id"], [scored.eos["id"]], group["id"])
    w = ValidationWorkspace(service=service)
    w.resize(1180, 780)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.show()
    QApplication.processEvents()
    assert w.pages.currentWidget() is w.result_page
    assert w.checked_ids == set() and w.selected_benchmarks() == []
    volume = w.topic_picker.findData(group["id"])
    assert w.topic_chips.isVisible()
    w.topic_flow.itemAt(volume).widget().click()
    assert w.topic_picker.currentData() == group["id"]
    assert w.pages.currentWidget() is w.result_page and len(w.visible_items()) == 1
    w.check_visible()
    assert w.checked_ids == {scored.eos["id"]}
    w.sidebar_search.setText("no matching item")
    assert w.empty_items.isVisible() and w.checked_ids == {scored.eos["id"]}
    w.sidebar_search.clear()
    assert w.sidebar_item_list.isVisible() and w._focused_reference == scored.eos["id"]
    w.topic_picker.setCurrentIndex(w.topic_picker.count() - 1)
    w.check_visible()
    assert w.checked_ids == {scored.eos["id"], scored.efv["id"]}
    before = set(w.checked_ids)
    w.topic_picker.setCurrentIndex(volume)
    w.open_item(scored.eos["id"])
    assert w.checked_ids == before and len(w.detail.eos_plots) == 1
    assert w.pages.currentWidget() is w.result_page
    w.comparison_picker.set_selected_ids([scored.old["id"]])
    assert len(w.run_jobs(model_ids=w.selected_model_ids())) == 4
    w.clear_checks()
    assert not w.run_selected_button.isEnabled()
    service.create_group(scored.suite["id"], "Long topic name " * 8)
    w.refresh()
    assert w.topic_picker.isVisible() and not w.topic_chips.isVisible()
    w.close()
    w.deleteLater()


def test_workspace_run_freezes_scope_and_does_not_force_navigation(scored):
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace
    from test_validation_ui import _wait_until

    w = ValidationWorkspace(service=scored.service)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.checked_ids = {scored.eos["id"]}
    w._update_run_scope()
    with patch.object(scored.service, "evaluate", wraps=scored.service.evaluate) as evaluate:
        w.run_selected()
        w.checked_ids = {scored.efv["id"]}
        w.show_page("items")
        assert _wait_until(lambda: w._run_task is None)
        assert {c.args[1] for c in evaluate.call_args_list} == {scored.eos["id"]}
        assert len(evaluate.call_args_list) == len(w.selected_model_ids())
        assert w.pages.currentWidget() is w.matrix_card
        assert w.checked_ids == {scored.efv["id"]}
    w.close()
    w.deleteLater()


def test_package_preserves_topic_and_suite_local_name(scored, tmp_path):
    from NepTrainKit.core.validation.package import build_package, import_package

    service = scored.service
    topic = service.create_group(scored.suite["id"], "Volume")
    service.assign_group(scored.suite["id"], [scored.eos["id"]], topic["id"])
    service.rename_suite_item(scored.suite["id"], scored.eos["id"], "My volume scan")
    target = ValidationService(Library(tmp_path / "other"))
    suite = import_package(target, build_package(service, scored.suite["id"]), name="shared")
    item = next(i for i in target.suite_items(suite["id"]) if i["group_name"] == "Volume")
    assert item["name"] == "My volume scan"


def test_workspace_overview_does_not_load_curves_and_primary_switch_keeps_targets(scored):
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    w = ValidationWorkspace(service=scored.service)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    with patch.object(scored.service, "run_details", side_effect=AssertionError("Overview must not load curve files")):
        w.show_page("items")
    before = set(w.selected_model_ids())
    w.model_combo.setCurrentIndex(w.model_combo.findData(scored.old["id"]))
    assert set(w.selected_model_ids()) == before
    assert w.checked_ids == set()
    w.close()
    w.deleteLater()


def test_workspace_diagnostic_editor_is_not_hidden_behind_overview(scored):
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    w = ValidationWorkspace(service=scored.service)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.show_diagnostic_editor()
    assert w.pages.currentWidget() is w.result_page
    assert not w.editor_scroll.isHidden()
    w.close()
    w.deleteLater()


def test_add_existing_is_atomic_and_restores_archived_history(scored):
    service = scored.service
    suite = service.create_suite("target")
    service.archive_reference(scored.suite["id"], [scored.eos["id"]])
    with pytest.raises(ValueError):
        service.add_existing_references(suite["id"], [scored.eos["id"], "missing"])
    assert service.get_set(scored.eos["id"])["archived"]
    assert service.suite_sets(suite["id"]) == []
    service.add_existing_references(suite["id"], [scored.eos["id"]])
    assert not service.get_set(scored.eos["id"])["archived"]
    assert service.latest_run(scored.new["id"], scored.eos["id"])["status"] == "succeeded"


def test_replacing_analysis_changes_only_one_suite_and_keeps_old_run(scored, tmp_path):
    service = scored.service
    replacement = _import(service, tmp_path, eos_frames(curvature=0.9), "new")
    other = service.create_suite("keep original", [scored.eos["id"]])
    group = service.create_group(scored.suite["id"], "Volume")
    service.assign_group(scored.suite["id"], [scored.eos["id"]], group["id"])
    before = service.latest_run(scored.new["id"], scored.eos["id"])
    service.replace_suite_item(scored.suite["id"], scored.eos["id"], replacement["id"])
    assert service.suite_items(scored.suite["id"])[0]["id"] == replacement["id"]
    assert service.suite_items(scored.suite["id"])[0]["group_id"] == group["id"]
    assert service.suite_sets(other["id"])[0]["id"] == scored.eos["id"]
    assert service.latest_run(scored.new["id"], scored.eos["id"])["id"] == before["id"]
    assert service.latest_run(scored.new["id"], replacement["id"]) is None


def test_workspace_edit_saves_new_version_and_preserves_other_suite(scored, monkeypatch):
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace, BenchmarkImportDialog

    service = scored.service
    other = service.create_suite("original analysis", [scored.eos["id"]])
    topic = service.create_group(scored.suite["id"], "Volume")
    service.assign_group(scored.suite["id"], [scored.eos["id"]], topic["id"])
    old_run = service.latest_run(scored.new["id"], scored.eos["id"])
    w = ValidationWorkspace(service=service)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.item_table.selectRow(0)

    def accept(dialog):
        dialog.name_edit.setText("Revised analysis")
        dialog.notes_edit.setText("reviewed settings")
        return True

    monkeypatch.setattr(BenchmarkImportDialog, "exec", accept)
    w.edit_item()
    replacement = service.suite_items(scored.suite["id"])[0]
    assert replacement["id"] != scored.eos["id"]
    assert replacement["name"] == "Revised analysis" and replacement["group_id"] == topic["id"]
    assert replacement["provenance"]["derived_from_set"] == scored.eos["id"]
    assert replacement["provenance"]["notes"] == "reviewed settings"
    assert replacement["provenance"]["original_source_path"] == scored.eos["source_ref"]["path"]
    assert replacement["object_hash"] == scored.eos["object_hash"]
    assert service.suite_sets(other["id"])[0]["id"] == scored.eos["id"]
    assert service.latest_run(scored.new["id"], scored.eos["id"])["id"] == old_run["id"]
    assert service.latest_run(scored.new["id"], replacement["id"]) is None
    w.close()
    w.deleteLater()


def test_legacy_accuracy_edit_creates_regression_version_without_changing_history(scored, monkeypatch):
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace, BenchmarkImportDialog
    from NepTrainKit.ui.widgets.validation_presentation import file_import_choices

    assert {kind for kind, _ in file_import_choices()} == {"eos", "response"}
    service = scored.service
    old = scored.efv
    other = service.create_suite("legacy accuracy", [old["id"]])
    old_run = service.latest_run(scored.new["id"], old["id"])
    w = ValidationWorkspace(service=service)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.open_item(old["id"])
    assert w.detail._args[0]["id"] == old["id"]
    w.item_table.selectRow(next(i for i, item in enumerate(w._items) if item["id"] == old["id"]))

    def accept(dialog):
        assert dialog.protocol_combo.findData("efv") == -1
        assert dialog.current_protocol().kind == "response"
        assert dialog.current_protocol().spec.mode == "slices"
        assert dialog.preview_plot.isHidden()
        assert dialog.validate()
        return True

    monkeypatch.setattr(BenchmarkImportDialog, "exec", accept)
    w.edit_item()
    replacement = next(item for item in service.suite_items(scored.suite["id"])
                       if item["provenance"].get("derived_from_set") == old["id"])
    assert replacement["protocol"] == "response" and replacement["source_ref"]["analysis"]["mode"] == "slices"
    assert replacement["object_hash"] == old["object_hash"]
    assert service.suite_sets(other["id"])[0]["id"] == old["id"]
    assert service.get_set(old["id"]) == old
    assert service.latest_run(scored.new["id"], old["id"])["id"] == old_run["id"]
    assert service.latest_run(scored.new["id"], replacement["id"]) is None
    w.close()
    w.deleteLater()


def test_topic_navigation_focuses_visible_item_and_keeps_checks(scored):
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    service = scored.service
    topic = service.create_group(scored.suite["id"], "Volume")
    service.assign_group(scored.suite["id"], [scored.eos["id"]], topic["id"])
    w = ValidationWorkspace(service=service)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.show_page("items")
    w.topic_picker.setCurrentIndex(w.topic_picker.findData(topic["id"]))
    w.item_table.selectRow(0)
    w.check_visible()
    assert w._management_ids() == [scored.eos["id"]]
    w.topic_picker.setCurrentIndex(w.topic_picker.count() - 1)
    assert w.pages.currentWidget() is w.matrix_card
    assert not w._management_ids()
    assert w.checked_ids == {scored.eos["id"]}
    assert not any(a.isEnabled() for a in w.topic_actions)
    w.close()
    w.deleteLater()


def test_append_after_removals_preserves_item_order(scored, tmp_path):
    service = scored.service
    third = _import(service, tmp_path, eos_frames(curvature=0.9), "third")
    fourth = _import(service, tmp_path, eos_frames(curvature=0.8), "fourth")
    suite_id = scored.suite["id"]
    service.add_to_suite(suite_id, [third["id"]])
    service.remove_from_suite(suite_id, [scored.efv["id"]])
    service.add_to_suite(suite_id, [fourth["id"]])
    with sqlite3.connect(service.library.db_path) as conn:
        positions = conn.execute(
            "SELECT position FROM benchmark_suite_sets WHERE suite_id=? ORDER BY position", (suite_id,)
        ).fetchall()
    assert positions == [(0,), (2,), (3,)]


def test_folder_batch_creates_topics_without_predictions(scored, tmp_path):
    from ase.io import write
    from PySide6.QtWidgets import QWidget
    from NepTrainKit.ui.widgets.reference_batch import ReferenceBatchDialog
    from test_validation_ui import _wait_until

    paths = []
    for topic, frames in [("EOS", eos_frames(curvature=0.9)), ("Accuracy", efv_frames())]:
        folder = tmp_path / topic
        folder.mkdir()
        path = folder / "frames.xyz"
        write(path, frames, format="extxyz")
        paths.append(path)
    host = QWidget()
    with patch.object(scored.service, "evaluate", side_effect=AssertionError("Import must not predict")):
        dialog = ReferenceBatchDialog(host, scored.service, paths, suite_name="folder suite", group_by_directory=True)
        assert _wait_until(lambda: dialog.task is None)
        imported = dialog.import_into()
        assert len(imported) == 2
    suite = next(s for s in scored.service.suites() if s["name"] == "folder suite")
    assert {i["group_name"] for i in scored.service.suite_items(suite["id"])} == {"EOS", "Accuracy"}
    dialog.accept()
    host.close()


def test_clicking_compare_and_run_cells_has_independent_effects(scored):
    from PySide6.QtTest import QTest
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    w = ValidationWorkspace(service=scored.service)
    w.resize(1320, 850)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.show()
    QApplication.processEvents()
    navigator = w.sidebar_item_list
    assert navigator.isVisible() and w.detail_scroll.isVisible()
    for row, column in ((0, 0), (1, 1)):
        cell = navigator.item(row, column)
        QTest.mouseClick(navigator.viewport(), Qt.MouseButton.LeftButton,
                         pos=navigator.visualItemRect(cell).center())
    assert w.checked_ids == {scored.efv["id"]}
    assert w.detail._args[0]["id"] == scored.eos["id"]
    navigator.setCurrentCell(1, 0)
    assert w.detail._args[0]["id"] == scored.efv["id"]
    w.resize(800, 650)
    QApplication.processEvents()
    assert navigator.isVisible() and w.detail.plot_pane.isVisible()
    assert not w.matrix_card.isVisible() and not w.detail.inspector_panel.isVisible()
    assert not w.metrics_button.isVisible()
    assert list(w.page_widgets) == ["result", "items", "gallery"]
    w.show_page("items")
    assert w.item_table.isVisible() and not w.detail_scroll.isVisible()
    assert w.item_table.item(1, w.check_column).checkState() == Qt.CheckState.Checked
    cell = w.item_table.item(0, 0)
    QTest.mouseClick(w.item_table.viewport(), Qt.MouseButton.LeftButton, pos=w.item_table.visualItemRect(cell).center())
    assert w.item_table.isVisible()
    w.resize(1360, 820)
    QApplication.processEvents()
    QTest.mouseClick(w.view_picker.items["gallery"], Qt.MouseButton.LeftButton)
    from test_validation_ui import _wait_until
    assert _wait_until(lambda: w.gallery.loaded == {0, 1})
    assert not w.item_table.isVisible() and not w.detail_scroll.isVisible()
    QTest.mouseClick(w.gallery.viewport(), Qt.MouseButton.LeftButton,
                     pos=w.gallery.visualItemRect(w.gallery.item(0)).center())
    assert w.detail_scroll.isVisible() and w.detail._args[0]["id"] == scored.eos["id"]
    assert w.checked_ids == {scored.efv["id"]}
    w.close()
    w.deleteLater()


def test_double_click_runs_only_missing_models_for_one_item(scored):
    from PySide6.QtTest import QTest
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace
    from test_validation_ui import _wait_until

    w = ValidationWorkspace(service=scored.service)
    w.open_scorecard(scored.suite["id"], scored.fresh["id"])
    w.show()
    QApplication.processEvents()
    nav = w.sidebar_item_list
    nav.item(1, 1).setCheckState(Qt.CheckState.Checked)
    pos = nav.visualItemRect(nav.item(0, 0)).center()
    with patch.object(scored.service, "evaluate", wraps=scored.service.evaluate) as evaluate:
        QTest.mouseClick(nav.viewport(), Qt.MouseButton.LeftButton, pos=pos)
        QTest.mouseDClick(nav.viewport(), Qt.MouseButton.LeftButton, pos=pos)
        assert _wait_until(lambda: evaluate.called and w._run_task is None)
        evaluate.assert_called_once_with(scored.fresh["id"], scored.eos["id"])
        assert w.checked_ids == {scored.efv["id"]}
        QTest.mouseClick(nav.viewport(), Qt.MouseButton.LeftButton, pos=pos)
        QTest.mouseDClick(nav.viewport(), Qt.MouseButton.LeftButton, pos=pos)
        assert w._run_task is None and evaluate.call_count == 1
    w.close()
    w.deleteLater()


def test_eos_residual_and_source_group_switch_preserve_scope(scored, tmp_path, monkeypatch):
    import numpy as np
    from qfluentwidgets import ComboBox
    from test_validation_framework import write_xyz
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    # The hover probe below maps a QPointF through PyQtGraph's scene.
    monkeypatch.setattr("NepTrainKit.ui.canvas.curve_factory._normalize_canvas_type", lambda _: "pyqtgraph")
    groups = [eos_frames(), eos_frames(curvature=0.8)]
    for frames, label in zip(groups, ("first", "second")):
        for frame in frames:
            frame.info["scan_id"] = label
    path = write_xyz(tmp_path / "both.xyz", groups[0] + groups[1])
    records = scored.service.import_eos_groups(path, name="scan", provenance={}, group_field="scan_id",
                                              group_names=["first", "second"])
    suite = scored.service.create_suite("two branches", [r["id"] for r in records] + [scored.eos["id"]])
    scored.service.run_suite(scored.new["id"], suite["id"])
    w = ValidationWorkspace(service=scored.service)
    w.open_scorecard(suite["id"], scored.new["id"])
    w.show()
    w.sidebar_item_list.item(0, 1).setCheckState(Qt.CheckState.Checked)
    from qfluentwidgets import TransparentToolButton
    first, second = (r["id"] for r in records)
    panel = w.detail
    assert len(panel.eos_plots) == 1
    for width, height in ((1050, 700), (1360, 820)):
        w.resize(width, height)
        QApplication.processEvents()
        assert panel.eos_cards[0].size() == panel.grid_scroll.viewport().size()
        assert panel.grid_scroll.verticalScrollBar().maximum() == 0
    w.add_eos_groups.click()
    picker = w.groups.dialog
    assert picker.isVisible() and not picker.isModal()
    assert w.item_table.item(0, 0).text() == w.sidebar_item_list.item(0, 0).text()
    assert not w.item_table.item(0, 0).icon().isNull()
    assert set(picker.buttons) == {first, second}
    picker.search.setText("SECOND")
    assert picker.row_widgets[0].isHidden() and not picker.row_widgets[1].isHidden()
    picker.buttons[second]["compare"].click()
    assert len(panel.eos_plots) == 2 and w.eos_grid_mode.isVisible()
    from NepTrainKit.ui.widgets.validation_model_picker import GroupMarkerButton
    from PySide6.QtGui import QColor
    assert w.groups.markers[first] != w.groups.markers[second]
    marker = picker.marker_buttons[second]
    marker.menu().aboutToShow.emit()
    next(action for action in marker.menu().actions() if action.data() == "d").trigger()
    assert panel.eos_cards[1].findChild(GroupMarkerButton).symbol == "d"
    assert {curve.opts["symbol"] for curve in panel.eos_plots[1].listDataItems()} == {"d"}
    color = QColor("#893acd")
    w._set_model_color(scored.new["id"], color)
    curve = panel.eos_plots[1].listDataItems()[-1]
    assert curve.opts["pen"].color() == color and curve.opts["symbol"] == "d"
    separate = [c for plot in panel.eos_plots for c in plot.listDataItems()]
    expected = [(c.xData.copy(), c.yData.copy()) for c in separate]
    from PySide6.QtTest import QTest
    picker.done_button.click()
    w.activateWindow()
    assert not w.eos_layout.isChecked()
    QTest.mouseClick(w.eos_layout, Qt.MouseButton.LeftButton)
    assert w.eos_layout.isChecked()
    QApplication.processEvents()
    merged, = panel.eos_plots
    assert panel.eos_cards[0].size() == panel.grid_scroll.viewport().size()
    assert len(merged.listDataItems()) == len(expected) == 4
    for curve, (x, y) in zip(merged.listDataItems(), expected):
        np.testing.assert_array_equal(curve.xData, x)
        np.testing.assert_array_equal(curve.yData, y)
    from PySide6.QtCore import QEvent, QPointF
    curve = merged.listDataItems()[1]
    volume, energy = curve.xData[2], curve.yData[2]
    point = merged.mapFromScene(merged.getViewBox().mapViewToScene(QPointF(volume, energy)))
    from PySide6.QtGui import QMouseEvent
    viewport = merged.viewport()
    QApplication.sendEvent(viewport, QMouseEvent(QEvent.Type.MouseMove, QPointF(point),
                           QPointF(viewport.mapToGlobal(point)), Qt.MouseButton.NoButton,
                           Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
    hover = merged.point_hover
    assert hover.isVisible() and f"{volume:.4f}" in hover.text() and f"{energy:.6g}" in hover.text()
    assert "●" in hover.text() and "◆" in hover.text() and "#893acd" in hover.text()
    assert all(label not in hover.text() for label in ("first", "second", "Reference", "Inbox", "new"))
    QApplication.sendEvent(merged.viewport(), QEvent(QEvent.Type.Leave))
    assert not hover.isVisible()
    w.groups.set_marker(second, "s")
    assert [c.opts["symbol"] for c in merged.listDataItems()] == [w.groups.markers[first]] * 2 + ["s"] * 2
    w.groups.set_marker(second, "d")
    panel.set_series_visible(scored.new["id"], False)
    assert [c.isVisible() for c in merged.listDataItems()] == [True, False, True, False]
    panel.set_series_visible(scored.new["id"], True)
    QTest.mouseClick(w.eos_grid_mode, Qt.MouseButton.LeftButton)
    residuals = panel.eos_plots[0].listDataItems()
    assert len(residuals) == 2
    residual_plot = panel.eos_plots[0]
    assert all(c.opts["pen"].isCosmetic() for c in residuals)
    panel.set_series_visible(scored.new["id"], False)
    QApplication.processEvents()
    assert not residual_plot.residual_placeholder.isHidden() and residual_plot.isHidden()
    panel.set_series_visible(scored.new["id"], True)
    assert not residual_plot.isHidden() and residual_plot.residual_placeholder.isHidden()
    for curve, (_, _, entries) in zip(residuals, panel._groups):
        details = entries[0][3]
        delta = np.subtract(details["predicted"], details["reference"])
        np.testing.assert_allclose(curve.yData, (delta - delta.mean()) * 1000)
    for width, height in ((1050, 700), (1360, 820)):
        w.resize(width, height)
        QApplication.processEvents()
        card, = panel.eos_cards
        assert card.size() == panel.grid_scroll.viewport().size()
        assert card.window() == w and w.sidebar_item_list.isVisible()
        assert panel.grid_scroll.verticalScrollBar().maximum() == 0
    assert w.checked_ids == {first} and w._focused_reference == first
    QTest.mouseClick(w.eos_grid_mode, Qt.MouseButton.LeftButton)
    QTest.mouseClick(w.eos_layout, Qt.MouseButton.LeftButton)
    assert not w.eos_layout.isChecked() and len(panel.eos_plots) == 2
    w.add_eos_groups.click()
    picker.buttons[second]["current"].click()
    assert w._focused_reference == second and picker.buttons[first]["compare"].isChecked()
    picker.done_button.click()
    panel.eos_cards[0].findChild(TransparentToolButton, "toggleEosGroupFocus").click()
    assert len(panel.eos_plots) == 1 and w._focused_reference == first
    panel.eos_cards[0].findChild(TransparentToolButton, "toggleEosGroupFocus").click()
    assert len(panel.eos_plots) == 2
    panel.eos_cards[1].findChild(TransparentToolButton, "removeEosGroup").click()
    assert len(panel.eos_plots) == 1 and w.checked_ids == {first}
    w.open_item(scored.eos["id"])
    assert not w.add_eos_groups.isVisible() and len(panel.eos_plots) == 1
    w.open_item(second)
    QTest.mouseClick(w.eos_grid_mode, Qt.MouseButton.LeftButton)
    details = panel._args[1][0][3]
    raw = np.subtract(details["predicted"], details["reference"])
    np.testing.assert_allclose(panel.eos_plots[0].listDataItems()[0].yData, (raw - raw.mean()) * 1000)
    panel.offset_checkbox.click()
    np.testing.assert_allclose(panel.eos_plots[0].listDataItems()[0].yData, raw * 1000)
    assert {curve.opts["symbol"] for curve in panel.eos_plots[0].listDataItems()} == {"d"}
    w._reload_workspace()
    assert w.groups.markers[second] == "d"
    w.show_page("items")
    assert w.table_metric.isVisible() and w.table_metric.count() == 3
    w.table_metric.setCurrentIndex(w.table_metric.findData("v0_error"))
    assert "%" in w.item_table.item(0, 5).text()
    assert w.item_table.item(1, 2).text() == "V₀ error"
    w._compare_cell(1, 0)
    assert w._focused_reference == second and w.checked_ids == {first}
    w.close()
    w.deleteLater()
    from PySide6.QtCore import QEvent
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_item_context_menu_targets_clicked_card_and_preserves_matrix_multiselection(scored):
    from PySide6.QtCore import QItemSelectionModel, QTimer
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    w = ValidationWorkspace(service=scored.service)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.show()
    QApplication.processEvents()
    w.sidebar_item_list.item(0, 1).setCheckState(Qt.CheckState.Checked)
    matrix = w.item_table

    def select_both():
        for row in (0, 1):
            matrix.selectionModel().select(matrix.model().index(row, 0),
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    menus = []
    def close_menu():
        menu = QApplication.activePopupWidget()
        menus.append(menu)
        menu.close()

    def open_menu(table, row):
        cell = table.item(row, 0)
        QTimer.singleShot(0, close_menu)
        table.customContextMenuRequested.emit(table.visualItemRect(cell).center())
        return menus[-1]

    select_both()
    menu = open_menu(w.sidebar_item_list, 1)
    assert w._management_ids() == [scored.efv["id"]]
    assert w.manage_actions["Rename in this suite…"] in menu.actions()
    assert w.manage_actions["Rename in this suite…"].isEnabled()
    assert w.checked_ids == {scored.eos["id"]}
    w.show_page("items")
    select_both()
    menu = open_menu(matrix, 0)
    assert set(w._management_ids()) == {scored.eos["id"], scored.efv["id"]}
    assert not w.manage_actions["Rename in this suite…"].isEnabled()
    assert w.manage_actions["Move to topic…"].isEnabled()
    next(a for a in menu.actions() if a.text() == w.tr("Clear checks")).trigger()
    assert not w.checked_ids
    w.close()
    w.deleteLater()


def test_comparison_cards_control_curves_independently_of_run_scope(scored):
    from PySide6.QtTest import QTest
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    w = ValidationWorkspace(service=scored.service)
    w.resize(800, 650)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.show()
    QApplication.processEvents()
    viewport = w.card_scroll.viewport()

    def click_visible(button):
        QApplication.processEvents()
        center = button.rect().center()
        assert button.isVisible() and viewport.rect().contains(button.mapTo(viewport, center))
        QTest.mouseClick(button, Qt.MouseButton.LeftButton, pos=center)

    new, old, fresh = [model["id"] for model in (scored.new, scored.old, scored.fresh)]
    w.check_visible()
    w.comparison_picker.set_selected_ids([fresh])
    assert list(w.cards) == [new, old, fresh]
    assert set(w.selected_model_ids()) == {new, fresh}
    assert w.controls_panel.geometry().topLeft() == w.rect().topLeft()
    assert w.controls_panel.width() == w.width()
    assert w.cards[old].pin.accessibleName() == "Baseline model"
    w.cards[old].pin.click()
    assert scored.service.get_suite(scored.suite["id"])["baseline_model_id"] == old
    assert w.cards[fresh].metrics.text() == "No recorded RMSE"
    w.cards[old].eye.click()
    assert not w.detail._curves[old][0].isVisible()
    assert w.cards[old].graphicsEffect().opacity() == 0.5
    w.open_item(scored.efv["id"])
    assert not w.cards[old].eye.isChecked() and not w.detail._curves[old][0].isVisible()
    w.cards[old].eye.click()
    assert w.cards[old].eye.isChecked()
    assert w.cards[old].graphicsEffect() is None
    click_visible(w.cards[fresh].remove)
    assert list(w.cards) == [new, old]
    assert w.selected_model_ids() == [new] and len(w.checked_ids) == 2
    w.card_scroll.ensureWidgetVisible(w.cards[new].pin)
    curve = w.detail._curves[new][0]
    click_visible(w.cards[new].pin)
    assert scored.service.get_suite(scored.suite["id"])["baseline_model_id"] == new
    assert w.current_model_row() is None
    assert w.detail._curves[new][0] is curve
    assert w.detail.multi_model_table.horizontalHeaderItem(1).text() != "Current model"
    assert w.role_dialog is None and list(w.cards) == [new, old]
    w.card_scroll.ensureWidgetVisible(w.cards[old].remove)
    click_visible(w.cards[old].remove)
    QApplication.processEvents()
    assert list(w.cards) == [new] and viewport.width() > 0
    assert w.cards[new].isVisible() and w.cards[new].width() > 0
    assert old not in w.detail._curves and w.detail._curves[new][0].isVisible()
    assert w._focused_reference == scored.efv["id"] and len(w.checked_ids) == 2
    w.open_scorecard(scored.suite["id"], new)
    assert w.current_model_row() is None and w.selected_model_ids() == [new]
    w.edit_models.click()
    buttons = w.role_dialog.buttons
    assert buttons[new]["baseline"].isChecked() and not buttons[new]["current"].isChecked()
    buttons[new]["current"].click()
    assert w.model_combo.currentData() == new
    assert scored.service.get_suite(scored.suite["id"])["baseline_model_id"] is None
    w.role_dialog.buttons[new]["hidden"].click()
    w.refresh()
    assert not w.cards and not w.selected_model_ids()
    w.role_dialog.buttons[old]["compare"].click()
    assert list(w.cards) == [old] and w.current_model_row() is None
    w.close()
    w.deleteLater()


def test_model_picker_search_matches_project_or_version_without_changing_selection(scored):
    from NepTrainKit.core.validation.service import ModelDraft
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace
    from test_validation_framework import MODEL_HEADER

    added = scored.service.save_model(ModelDraft((MODEL_HEADER + "variant").encode(), "variant.txt"),
                                      project_name="BFO candidates", name="spin-final")
    key = added["id"]
    w = ValidationWorkspace(service=scored.service)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.show()
    w.edit_models.click()
    picker = w.role_dialog
    original = list(w.cards)
    for text in (" bFo ", "IN-FIN"):
        picker.search.setText(text)
        assert [k for k, buttons in picker.buttons.items() if buttons["compare"].isVisible()] == [key]
        assert list(w.cards) == original
    picker.buttons[key]["compare"].click()
    assert list(w.cards) == original + [key] and picker.search.text() == "IN-FIN"
    picker.search.setText("no such version")
    assert picker.empty_label.isVisible() and key in w.cards
    picker.search.clear()
    assert not picker.empty_label.isVisible() and all(b["compare"].isVisible() for b in picker.buttons.values())
    picker.done_button.click()
    w.close()
    w.deleteLater()


def test_model_color_buttons_update_live_curves_and_remember_the_version(scored):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    w = ValidationWorkspace(service=scored.service)
    key, other = scored.new["id"], scored.old["id"]
    w.open_scorecard(scored.suite["id"], key)
    w.show()
    w.open_item(scored.eos["id"])
    curve = w.detail._curves[key][0]
    other_color = w.detail._series_pen(other).color()
    buttons = w.cards[key]
    w.edit_models.click()
    picker = w.role_dialog
    with patch.object(QColorDialog, "getColor", return_value=QColor("#be5577")):
        picker.color_buttons[key].click()
    assert curve.opts["pen"].color() == buttons.color_button.color == QColor("#be5577")
    picker.done_button.click()
    with patch.object(QColorDialog, "getColor", return_value=QColor("#117755")):
        buttons.color_button.click()
    assert curve is w.detail._curves[key][0] and curve.opts["pen"].color() == QColor("#117755")
    assert w.detail._series_pen(other).color() == other_color
    w.open_item(scored.efv["id"])
    assert w.detail._curves[key][0].opts["symbolBrush"] == QColor("#117755")
    w.edit_models.click()
    with patch.object(QColorDialog, "getColor", return_value=QColor()):
        picker.color_buttons[key].click()
    assert picker.color_buttons[key].color == QColor("#117755")
    w.close()
    w.deleteLater()
    reopened = ValidationWorkspace(service=scored.service)
    reopened.open_scorecard(scored.suite["id"], key)
    assert reopened.cards[key].color_button.color == QColor("#117755")
    reopened.close()
    reopened.deleteLater()


def test_model_strip_wraps_after_four_complete_cards(scored, tmp_path):
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace
    from test_validation_ui import _capture, _wait_until

    extra = [_capture(scored.service, tmp_path, name) for name in ("candidate-four", "candidate-five")]
    w = ValidationWorkspace(service=scored.service)
    w.resize(1360, 700)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.comparison_picker.set_selected_ids([r["id"] for r in (scored.old, scored.fresh, *extra)])
    w.show()
    QApplication.processEvents()
    reserved = w.width() - w.card_scroll.width()
    cards = list(w.cards.values())
    readable = max(card.minimumSizeHint().width() + 10 * card.name.fontMetrics().averageCharWidth()
                   for card in cards)
    gap = w.card_scroll.flow.horizontalSpacing()
    for strip_width, columns in ((1500, 4), (970, 3)):
        # System fonts change the space required by the role and action buttons.
        strip_width = max(strip_width, columns * readable + (columns - 1) * gap)
        w.resize(reserved + strip_width, 700)
        assert _wait_until(lambda: [c.y() == 0 for c in cards] == [True] * columns + [False] * (5 - columns))
        assert cards[columns].x() == cards[0].x()
        assert len({c.width() for c in cards}) == 1
        assert all(c.geometry().right() < w.card_scroll.viewport().width() for c in cards)
        assert w.card_scroll.horizontalScrollBar().maximum() == 0
    w.close()
    w.deleteLater()


def test_pinning_a_comparison_keeps_cards_columns_and_plot_in_place(scored):
    from PySide6.QtCore import QPoint
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    w = ValidationWorkspace(service=scored.service)
    w.resize(800, 650)
    current, old, new = [r["id"] for r in (scored.new, scored.old, scored.fresh)]
    w.open_scorecard(scored.suite["id"], current)
    w.show()
    w.open_item(scored.eos["id"])
    w.comparison_picker.set_selected_ids([new])
    QApplication.processEvents()
    cards = w.cards.copy()
    readable = max(card.minimumSizeHint().width() + 10 * card.name.fontMetrics().averageCharWidth()
                   for card in cards.values())
    reserved = w.width() - w.card_scroll.width()
    w.resize(max(1700, reserved + 3 * readable + 2 * w.card_scroll.flow.horizontalSpacing()), 650)
    from test_validation_ui import _wait_until
    assert _wait_until(lambda: len({card.y() for card in cards.values()}) == 1)
    one_row = w.controls_panel.height()
    assert "evaluated" in w.suite_combo.toolTip()
    w.resize(550, 650)
    assert _wait_until(lambda: w.controls_panel.height() > one_row)
    card_height = max(card.height() for card in cards.values())
    assert one_row < w.controls_panel.height() <= one_row + card_height + 4 + w.card_scroll.horizontalScrollBar().sizeHint().height()
    assert w.card_scroll.verticalScrollBar().maximum() > 0
    w.card_scroll.ensureWidgetVisible(cards[new].pin)
    viewport = w.card_scroll.viewport()

    def pin_positions():
        return [card.pin.mapTo(viewport, QPoint(0, 0)) for card in w.cards.values()]

    positions = pin_positions()
    plot = w.detail.eos_plots[0]
    w.cards[new].pin.click()
    QApplication.processEvents()
    assert scored.service.get_suite(scored.suite["id"])["baseline_model_id"] == new
    assert list(w.cards) == [current, old, new] and w.cards == cards
    assert pin_positions() == positions
    assert w.cards[new].pin.accessibleName() == "Baseline model" and w.cards[old].remove.isVisible()
    assert w.detail.eos_plots[0] is plot and [entry[0] for entry in w.detail._args[1]] == list(cards)
    w.edit_models.click()
    w.role_dialog.buttons[old]["current"].click()
    header = w.item_table.horizontalHeader()
    assert [header.logicalIndex(i) for i in (3, 4, 5)] == [7, 5, 8]
    assert list(w.cards) == [current, old, new] and w.cards == cards
    w.close()
    w.deleteLater()


def test_role_picker_updates_cards_baseline_and_temporary_models(scored, tmp_path):
    from PySide6.QtCore import QMimeData, QPointF, QUrl
    from PySide6.QtGui import QDropEvent
    from test_validation_framework import MODEL_HEADER
    from test_validation_ui import _wait_until
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    w = ValidationWorkspace(service=scored.service)
    w.resize(950, 700)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.show()
    w.check_visible()
    checked = w.checked_ids.copy()
    current, baseline, fresh = [r["id"] for r in (scored.new, scored.old, scored.fresh)]
    with patch.object(scored.service, "evaluate", side_effect=AssertionError("Selection must not predict")):
        w.edit_models.click()
        dlg = w.role_dialog
        assert dlg.isWindow() and dlg.isVisible() and not dlg.isModal()
        assert QApplication.activeModalWidget() is None and w.isEnabled()
        assert dlg.buttons[current]["current"].isChecked()
        assert dlg.buttons[baseline]["baseline"].isChecked() and dlg.buttons[fresh]["hidden"].isChecked()
        dlg.buttons[current]["compare"].click()
        assert w.current_model_row() is None and dlg.buttons[current]["compare"].isChecked()
        w.cards[current].eye.click()
        dlg.buttons[current]["compare"].click()
        assert not w.detail.series_visible(current)
        w.refresh()
        assert w.current_model_row() is None and current in w.cards
        dlg.buttons[current]["hidden"].click()
        assert current not in w.cards
        dlg.buttons[current]["current"].click()
        dlg.buttons[current]["hidden"].click()
        assert w.current_model_row() is None and current not in w.cards
        dlg.buttons[current]["current"].click()
        dlg.buttons[fresh]["baseline"].click()
        assert scored.service.get_suite(scored.suite["id"])["baseline_model_id"] == fresh
        assert dlg.buttons[baseline]["compare"].isChecked()
        assert all(sum(b.isChecked() for b in buttons.values()) == 1 for buttons in dlg.buttons.values())
        dlg.buttons[baseline]["current"].click()
        assert w.current_model_row()["model_id"] == baseline
        assert dlg.buttons[current]["compare"].isChecked()
        dlg.buttons[fresh]["hidden"].click()
        assert scored.service.get_suite(scored.suite["id"])["baseline_model_id"] is None
        assert fresh not in w.cards
        dlg.buttons[fresh]["compare"].click()
        assert fresh in w.cards and dlg.buttons[fresh]["compare"].isChecked()
        path = tmp_path / "candidate.txt"
        path.write_text(MODEL_HEADER + "temporary")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(path))])
        event = QDropEvent(QPointF(0, 0), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        dlg.dropEvent(event)
        assert event.isAccepted()
        tmp_model = scored.service.find_model_by_file(path)
        assert tmp_model["project_id"] is None
        assert dlg.buttons[tmp_model["id"]]["compare"].isChecked()
        assert not dlg.buttons[tmp_model["id"]]["baseline"].isEnabled()
        dlg.done_button.click()
        assert _wait_until(lambda: not dlg.isVisible())
        w.edit_models.click()
        assert dlg.buttons[tmp_model["id"]]["compare"].isChecked()
        assert w.checked_ids == checked
    dlg.close()
    w.close()
    w.deleteLater()


def test_draft_can_run_as_comparison_without_replacing_current_model(scored):
    from NepTrainKit.core.validation.service import ModelDraft
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace
    from test_validation_ui import _wait_until

    w = ValidationWorkspace(service=scored.service)
    current = scored.new["id"]
    draft = ModelDraft(scored.service.model_file(current).read_bytes() + b"\nnew version", "test.txt")
    w.open_scorecard(scored.suite["id"], draft=draft)
    w.show()
    w.edit_models.click()
    w.role_dialog.buttons[current]["current"].click()
    assert w.role_dialog.buttons[0]["compare"].isChecked()
    assert not w.role_dialog.buttons[0]["baseline"].isEnabled()
    w.role_dialog.done_button.click()
    assert _wait_until(lambda: not w.role_dialog.isVisible())
    w.checked_ids = {scored.eos["id"]}
    assert w.run_selected() is not None
    assert _wait_until(lambda: w._run_task is None)
    captured = scored.service.find_model_by_hash(draft.content_hash)
    assert captured and captured["project_id"] is None
    assert w.model_combo.currentData() == current
    assert captured["id"] in w.comparison_picker.selected_ids()
    w.close()
    w.deleteLater()

def test_reference_only_eos_explains_empty_residual(scored):
    from NepTrainKit.ui.views.model_library import RunDetailPanel

    panel = RunDetailPanel()
    panel.set_adaptive_layout(True)
    panel.show_runs(scored.eos, [])
    panel._change_eos_view(1)
    plot, = panel.eos_plots
    assert plot.isHidden() and not plot.residual_placeholder.isHidden()
    assert "No paired prediction" in plot.residual_placeholder.text()
    panel._change_eos_view(0)
    assert len(panel.eos_plots[0].listDataItems()) == 1
    panel.close()
    panel.deleteLater()


def test_magnetic_equation_of_state_badge_uses_the_scan_axes():
    from NepTrainKit.ui.widgets.validation_presentation import presentation_for

    view = presentation_for("response")
    for x, other, quantity, expected in (
        ("moment_ratio", "", "energy", "mEOS"),
        ("volume_ratio", "moment_ratio", "energy", "mEOS"),
        ("angle_deg", "", "energy", "Resp"),
        ("moment_ratio", "", "pressure", "Resp"),
    ):
        recipe = dict(mode="scan", x_field=x, second_axis=other, quantity=quantity)
        assert view.item_badge(dict(protocol="response", summary={"analysis": recipe})) == expected


def test_mae_uses_angle_curves_and_restores_old_references_without_prediction(scored, tmp_path):
    from test_reference_response import mae_frames
    from test_validation_framework import _label
    from NepTrainKit.core.validation.protocols.response import suggest_response_settings
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    frames = mae_frames()
    for a in frames:
        a.set_chemical_symbols(["Pb", "Te"])
        _label(a, a.info.pop("energy"))
    item = _import(scored.service, tmp_path, frames, "MAE", protocol="response",
                   analysis=suggest_response_settings(frames))
    suite = scored.service.create_suite("angles", [item["id"]])
    scored.service.run_suite(scored.new["id"], suite["id"])
    w = ValidationWorkspace(service=scored.service)
    w.resize(1200, 820)
    w.open_scorecard(suite["id"], scored.new["id"])
    w.show()
    w.open_item(item["id"])
    QApplication.processEvents()
    assert len(w.groups.candidates) == 3
    assert not w._plot_switch_available(1) and not w._plot_switch_available(2)
    assert w.chart_tools.isVisible() and not w.eos_layout.isVisible()
    assert len(w.detail.eos_plots) == 1
    w.add_eos_groups.click()
    for key, _, _ in w.groups.candidates[1:]:
        w.groups.dialog.buttons[key]["compare"].click()
    w.groups.dialog.close()
    QApplication.processEvents()
    assert w.eos_layout.isVisible()
    w.eos_layout.click()
    QApplication.processEvents()
    plot, = w.detail.eos_plots
    assert plot.axis_titles[0] == "Rotation angle θ (°)"
    assert "E(p)" in plot.axis_titles[1]
    assert len(plot.listDataItems()) == 6
    ref = plot.listDataItems()[0]
    assert list(ref.xData) == [-45., 0., 15., 45., 90.]
    assert list(ref.yData) == pytest.approx([20., 0., 1., 15., 5.])
    w.eos_grid_mode.click()
    QApplication.processEvents()
    assert len(w.detail.eos_plots[0].listDataItems()) == 3
    assert "Residual" in w.detail.eos_plots[0].axis_titles[1]
    legacy = dict(item, summary=dict(item["summary"]))
    legacy["summary"].pop("plot_series")
    with patch.object(scored.service, "evaluate", side_effect=AssertionError("unexpected prediction")):
        assert scored.service.response_plot_series(legacy) == item["summary"]["plot_series"]
    w.close()
    w.deleteLater()


def test_magnetic_scan_draws_measured_slices_and_saved_residuals(scored, tmp_path):
    from PySide6.QtTest import QTest
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    frames = eos_frames(points=4)
    for frame, volume, moment in zip(frames, (0.9, 1.1, 0.9, 1.1), (0.8, 0.8, 1.2, 1.2)):
        frame.info.update(volume_ratio=volume, moment_ratio=moment)
    record = _import(scored.service, tmp_path, frames, "magnetic scan", protocol="response",
                     analysis=dict(mode="scan", x_field="volume_ratio", second_axis="moment_ratio",
                                   normalization="total", energy_unit="μeV"))
    suite = scored.service.create_suite("magnetic", [record["id"]])
    scored.service.run_suite(scored.new["id"], suite["id"])
    w = ValidationWorkspace(service=scored.service)
    w.open_scorecard(suite["id"], scored.new["id"])
    w.show()
    w.open_item(record["id"])
    QTest.qWait(30)
    assert all(w._plot_switch_available(index) for index in range(4))
    w.plot_switches[2].defaultAction().trigger()  # inspect unshifted saved energies
    w.plot_switches[1].defaultAction().trigger()
    details = w.detail._groups[0][2][0][3]
    run = w.detail._groups[0][2][0][2]
    # Legacy records have only relative curves. The same absolute data must be available without evaluating.
    assert scored.service.absolute_scan_energies(record) == details["absolute_reference"]
    assert scored.service.absolute_scan_energies(record, run) == details["absolute_predicted"]
    assert scored.service.absolute_scan_energies(record, dict(run, set_id="another scan")) is None
    indices = [i for i, row in enumerate(details["rows"]) if row["second_axis"] == 0.8]
    ref, model = w.detail.eos_plots[0].listDataItems()
    assert list(model.yData) == [details["absolute_predicted"][i] for i in indices]
    assert list(ref.yData) == [details["absolute_reference"][i] for i in indices]
    for key in (None, scored.new["id"]):
        samples, minimum = w.detail._curves[key][:2]
        assert minimum.property("annotation") == "sampled_minimum"
        index = min(range(len(samples.yData)), key=lambda i: samples.yData[i])
        assert list(minimum.xData) == [samples.xData[index]]
        assert list(minimum.yData) == [samples.yData[index]]
    w.plot_switches[1].defaultAction().trigger()
    assert not w.eos_layout.isHidden() and len(w.detail.eos_plots) == 2
    w.eos_layout.click()
    QTest.qWait(30)
    w.eos_grid_mode.click()
    plot, = w.detail.eos_plots
    assert plot.point_hover.unit == "μeV/cell" and "V/V₀" in plot.point_hover.x_label
    details = w.detail._groups[0][2][0][3]
    curves = plot.listDataItems()
    assert len(curves) == 2
    assert not any(c.property("auxiliary") for items in w.detail._curves.values() for c in items)
    for curve, moment in zip(curves, (0.8, 1.2)):
        indices = [i for i, row in enumerate(details["rows"]) if row["second_axis"] == moment]
        assert list(curve.xData) == [0.9, 1.1]
        assert list(curve.yData) == [details["errors"][i] for i in indices]
    w.plot_switches[2].defaultAction().trigger()
    assert all(abs(sum(c.yData)) < 1e-6 for c in w.detail.eos_plots[0].listDataItems())
    # Branch roles and markers work on saved response rows without selecting other validation items.
    from NepTrainKit.ui.widgets.validation_model_picker import GroupMarkerButton
    from qfluentwidgets import TransparentToolButton
    first, second = [key for key, _, _ in w.groups.candidates]
    assert w.add_eos_groups.isVisible()
    w.add_eos_groups.click()
    picker = w.groups.dialog
    picker.buttons[second]["current"].click()
    assert w.groups.current == second and w.groups.comparisons == {first}
    picker.buttons[first]["hidden"].click()
    assert len(w.detail.eos_plots[0].listDataItems()) == 1
    assert w.detail.eos_cards[0].property("groupId") == second
    w.groups.set_marker(second, "d")
    assert w.detail.eos_cards[0].findChild(GroupMarkerButton).symbol == "d"
    assert w.detail.eos_plots[0].listDataItems()[0].opts["symbol"] == "d"
    picker.buttons[first]["compare"].click()
    w.eos_layout.click()
    w.detail.eos_cards[0].findChild(TransparentToolButton, "toggleEosGroupFocus").click()
    assert w.groups.focused and len(w.detail.eos_cards) == 1
    w.detail.eos_cards[0].findChild(TransparentToolButton, "toggleEosGroupFocus").click()
    assert len(w.detail.eos_cards) == 2 and not w.checked_ids
    w.close()
    w.deleteLater()


def test_curve_values_align_measured_points_and_keep_distinct_model_ids(monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtGui import QColor, QPen
    from PySide6.QtWidgets import QDialog, QTableWidget
    import pyqtgraph as pg
    from NepTrainKit.ui.views.model_library import RunDetailPanel
    from NepTrainKit.ui.widgets.validation_values import show_curve_values

    panel = RunDetailPanel()
    panel._groups = [("A", {"id": "A", "name": "First"}, []), ("B", {"id": "B", "name": "Second"}, [])]
    curves = []
    def add(group, model, xs, ys, visible=True):
        curve = pg.PlotDataItem(xs, ys, name="reference" if model is None else "same name", pen=QPen(QColor("blue")))
        curve.setProperty("groupId", group)
        panel._register_curve(model, curve)
        curve.setVisible(visible)
        curves.append(curve)

    add("A", None, [1, 1, 2], [5, 6, 7])
    add("A", 1, [1, 2], [4, 8])
    add("A", 2, [1, 1, 2], [9, 8, 7])
    add("B", 2, [1], [10])
    add("B", 3, [1], [100], False)
    def inspect(dialog):
        table = dialog.findChild(QTableWidget)
        assert (table.rowCount(), table.columnCount()) == (4, 5)
        assert [table.item(1, i).text() for i in range(5)] == ["First", "1", "6", "—", "8"]
        assert [table.item(3, i).text() for i in range(5)] == ["Second", "1", "—", "—", "10"]
        assert table.horizontalHeaderItem(3).icon().isNull() is False
        return 0
    monkeypatch.setattr(QDialog, "exec", inspect)
    show_curve_values(panel, SimpleNamespace(axis_titles=("X", "ΔE (meV/atom)"), listDataItems=lambda: curves))
    panel.close()
    panel.deleteLater()


def test_validation_tab_navigation_reuses_plots_and_invalidates_changed_data(scored):
    from PySide6.QtCore import QEvent
    from PySide6.QtTest import QTest
    from shiboken6 import isValid
    from test_validation_ui import _wait_until
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace

    w = ValidationWorkspace(service=scored.service)
    with patch.object(scored.service, "run_details", wraps=scored.service.run_details) as reads:
        w.refresh(scored.suite["id"], scored.new["id"])
        w.resize(1200, 800)
        w.show()
        QApplication.processEvents()
        assert w.pages.currentWidget() is w.matrix_card and not w.detail.eos_plots
        w.show_page("result")
        w.show_page("items")
        QTest.qWait(40)
        reads.assert_not_called()

        w.open_item(scored.eos["id"])
        chart, = w.detail.eos_plots
        w.show_page("gallery")
        assert _wait_until(lambda: w.gallery.loaded == {0, 1})
        icons = [w.gallery.item(i).icon().cacheKey() for i in range(2)]
        reads.reset_mock()
        w.show_page("items")
        w.show_page("gallery")
        QTest.qWait(40)
        assert [w.gallery.item(i).icon().cacheKey() for i in range(2)] == icons
        w.show_page("result")
        QTest.qWait(40)
        assert w.detail.eos_plots == [chart]
        reads.assert_not_called()

        w.show_page("items")
        old_checkbox = w.sidebar_item_list.item(0, 1)
        w.comparison_picker.set_selected_ids([scored.fresh["id"]])
        assert not isValid(old_checkbox)
        assert not w.gallery.loaded
        reads.assert_not_called()
        w.show_page("result")
        assert _wait_until(lambda: w.detail.eos_plots != [chart])
        assert w.detail._args[0]["id"] == scored.eos["id"]
        assert {entry[0] for entry in w.detail._args[1]} == {scored.new["id"], scored.old["id"], scored.fresh["id"]}
        w.show_page("gallery")
        assert _wait_until(lambda: w.gallery.loaded == {0, 1})
        old_icon = w.gallery.item(0).icon().cacheKey()
        QApplication.sendEvent(w.gallery, QEvent(QEvent.Type.DevicePixelRatioChange))
        assert not w.gallery.loaded
        assert _wait_until(lambda: w.gallery.loaded == {0, 1})
        assert w.gallery.item(0).icon().cacheKey() != old_icon
    w.close()
    w.deleteLater()


def test_gallery_discrete_ticks_use_indices_without_changing_curve_labels(scored, tmp_path, curve_mode):
    from ase.io import write
    from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace
    from test_reference_response import four_spec, four_states
    from test_validation_ui import _wait_until

    frames = four_states('polar_oxygen_new_with_long_name') + four_states('polar_oxygen_other_with_long_name')
    source = tmp_path / 'coupled.xyz'
    write(source, frames, format='extxyz')
    record = scored.service.import_set(source, name='JD / coupled modes', protocol='response', provenance={},
                                       analysis=four_spec().to_dict())
    suite = scored.service.create_suite('discrete gallery', [record['id']])
    w = ValidationWorkspace(service=scored.service)
    w.resize(1100, 760)
    w.open_scorecard(suite['id'], scored.new['id'])
    w.show(); w.open_item(record['id'])
    original = w.detail.eos_plots[0]
    categories = original.category_labels.copy()
    assert all('geometry=' in name for name in categories.values())
    w.show_page('gallery')
    assert _wait_until(lambda: w.gallery.loaded == {0})
    chart = w._thumbnail_renderer.eos_plots[0]
    ticks = chart.x_ticks.categories if hasattr(chart, 'x_ticks') else chart.getAxis('bottom').categories
    assert ticks == [(x, str(x)) for x in categories]
    assert chart.category_labels == categories and original.category_labels == categories
    item = w.gallery.item(0)
    assert 'State / combination' not in item.text() and '→' not in item.text()
    assert 'ΔE' in item.text() and 'Combination index' in item.toolTip()
    w.gallery._open(item)
    assert w.detail.eos_plots[0].category_labels == categories
    w.close(); w.deleteLater()
