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
    assert w.pages.currentWidget() is w.overview
    assert w.checked_ids == set() and w.selected_benchmarks() == []
    root = w.topic_tree.topLevelItem(0)
    volume = root.child(0)
    w._browse_topic(volume)
    assert w.pages.currentWidget() is w.item_table and len(w.visible_items()) == 1
    w.check_visible()
    assert w.checked_ids == {scored.eos["id"]}
    w._browse_topic(root.child(1))
    w.check_visible()
    assert w.checked_ids == {scored.eos["id"], scored.efv["id"]}
    before = set(w.checked_ids)
    w.open_item(scored.eos["id"])
    assert w.checked_ids == before and len(w.detail.eos_plots) == 1
    assert w.pages.currentWidget() is w.page_widgets["details"]
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
        assert w.pages.currentWidget() is w.item_table
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
    with patch.object(scored.service, "run_details", side_effect=AssertionError("Overview must not load curve files")):
        w.open_scorecard(scored.suite["id"], scored.new["id"])
    assert not w.detail.eos_plots
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
    assert w.pages.currentWidget() is w.page_widgets["details"]
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
    root = w.topic_tree.topLevelItem(0)
    w.topic_tree.setCurrentItem(root.child(0))
    w.item_table.selectRow(0)
    w.check_visible()
    assert w._management_ids() == [scored.eos["id"]]
    w.topic_tree.setCurrentItem(root.child(1))
    assert w.pages.currentWidget() is w.item_table
    assert not w._management_ids()
    assert w.checked_ids == {scored.eos["id"]}
    assert not any(a.isEnabled() for a in w.topic_tree.actions())
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
