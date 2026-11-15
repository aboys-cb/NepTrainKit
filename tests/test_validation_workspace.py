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
    assert w.pages.currentWidget() is w.detail_scroll
    assert w.checked_ids == set() and w.selected_benchmarks() == []
    volume = w.topic_picker.findData(group["id"])
    w.topic_picker.setCurrentIndex(volume)
    assert w.pages.currentWidget() is w.detail_scroll and len(w.visible_items()) == 1
    w.check_visible()
    assert w.checked_ids == {scored.eos["id"]}
    w.topic_picker.setCurrentIndex(w.topic_picker.count() - 1)
    w.check_visible()
    assert w.checked_ids == {scored.eos["id"], scored.efv["id"]}
    before = set(w.checked_ids)
    w.topic_picker.setCurrentIndex(volume)
    w.open_item(scored.eos["id"])
    assert w.checked_ids == before and len(w.detail.eos_plots) == 1
    assert w.pages.currentWidget() is w.detail_scroll
    w.comparison_picker.set_selected_ids([scored.old["id"]])
    assert len(w.run_jobs(model_ids=w.selected_model_ids())) == 4
    w.clear_checks()
    assert not w.run_selected_button.isEnabled()
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
        w.show_page("overview")
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
    assert w.pages.currentWidget() is w.detail_scroll
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


def test_topic_navigation_clears_management_selection_but_keeps_checks(scored):
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
    w.pane_button.click()
    assert w.detail.inspector_panel.isVisible() and not w.detail.plot_pane.isVisible()
    w.show_page("items")
    assert w.item_table.isVisible()
    assert w.item_table.item(1, w.check_column).checkState() == Qt.CheckState.Checked
    w.resize(1360, 860)
    QApplication.processEvents()
    assert w.item_table.isVisible()
    w.show_page("result")
    assert w.detail.plot_pane.isVisible() and w.detail.inspector_panel.isVisible()
    assert w.checked_ids == {scored.efv["id"]}
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
    from test_validation_ui import _capture

    extra = [_capture(scored.service, tmp_path, name) for name in ("candidate-four", "candidate-five")]
    w = ValidationWorkspace(service=scored.service)
    w.resize(1360, 700)
    w.open_scorecard(scored.suite["id"], scored.new["id"])
    w.comparison_picker.set_selected_ids([r["id"] for r in (scored.old, scored.fresh, *extra)])
    w.show()
    QApplication.processEvents()
    reserved = w.width() - w.card_scroll.width()
    for strip_width, columns in ((1500, 4), (970, 3)):
        w.resize(reserved + strip_width, 700)
        QApplication.processEvents()
        cards = list(w.cards.values())
        assert [c.y() == 0 for c in cards] == [True] * columns + [False] * (5 - columns)
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
    w.resize(1700, 650)
    QApplication.processEvents()
    cards = w.cards.copy()
    assert len({card.y() for card in cards.values()}) == 1
    one_row = w.controls_panel.height()
    assert "evaluated" in w.suite_combo.toolTip()
    w.resize(550, 650)
    QApplication.processEvents()
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
    assert [w.overview.horizontalHeaderItem(i).text().split(" · ")[0] for i in (2, 3, 4)] == [
        scored.new["name"], scored.old["name"], scored.fresh["name"]]
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
