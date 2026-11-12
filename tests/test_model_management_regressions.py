"""Model identity, registry edits, shared history and bounded scorecard queries."""

import datetime as dt
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from PySide6.QtWidgets import QWidget
from qfluentwidgets import ComboBox
from sqlalchemy import event, select
from test_validation_framework import MODEL_HEADER, _import, efv_frames
from test_validation_framework import make_service as make_service
from test_validation_framework import model_file as model_file

from NepTrainKit.core.dataset.models import BenchmarkRun, ModelVersion
from NepTrainKit.core.dataset.services import ModelService, ProjectService
from NepTrainKit.core.validation.library import sha256_file
from NepTrainKit.core.validation.service import ModelDraft
from NepTrainKit.ui.pages import show_nep
from NepTrainKit.ui.views import dataset_widget
from NepTrainKit.ui.widgets.dialog import ModelInfoMessageBox


@pytest.mark.parametrize("reason", ["unchanged", "weights", "dataset", "same_model", "missing"])
def test_viewer_cache_requires_current_dataset_and_weights(tmp_path, reason):
    path = tmp_path / "nep.txt"
    path.write_text(MODEL_HEADER + "old")
    data_path = tmp_path / "train.xyz"
    data_path.touch()
    cached = SimpleNamespace(nep_txt_path=path, data_xyz_path=data_path,
                             _loaded_model_hash=sha256_file(path), select_index=[])
    current = SimpleNamespace(nep_txt_path=tmp_path / "nep_other.txt", data_xyz_path=data_path,
                              _loaded_model_hash="other", select_index=[1])
    if reason in {"weights", "same_model"}:
        path.write_text(MODEL_HEADER + "updated")
    if reason == "same_model":
        current = cached
    if reason == "dataset":
        current.data_xyz_path = tmp_path / "test.xyz"
        current.data_xyz_path.touch()
    if reason == "missing":
        path.unlink()
    viewer = SimpleNamespace(nep_result_data=current, _available_nep_files=[path],
        _nep_result_cache={path.resolve(): cached}, stop_loading=Mock(), set_dataset=Mock(),
        _restore_selection=Mock(), _reload_with_nep_file=Mock())
    show_nep.ShowNepWidget._on_nep_model_changed(viewer, 0)
    if reason == "unchanged":
        assert viewer.nep_result_data is cached
        viewer.set_dataset.assert_called_once()
        viewer._restore_selection.assert_called_once_with([1])
        viewer._reload_with_nep_file.assert_not_called()
    else:
        viewer._reload_with_nep_file.assert_called_once_with(str(current.data_xyz_path), path)
        viewer.set_dataset.assert_not_called()
        assert not viewer._nep_result_cache


def test_explicit_dataset_load_clears_model_cache_even_in_same_folder(tmp_path, monkeypatch):
    path = tmp_path / "test.xyz"
    path.touch()
    viewer = SimpleNamespace(_nep_result_cache={tmp_path / "nep.txt": object()},
        _invalidate_background_jobs=Mock(), _refresh_export_actions=Mock(), tr=lambda text: text)
    def cancel(_path):
        raise show_nep.DumpImportCancelled()
    monkeypatch.setattr(show_nep, "load_result_data", cancel)
    with patch.object(show_nep.MessageManager, "send_info_message"):
        show_nep.ShowNepWidget.check_nep_result(viewer, str(path))
    assert viewer._nep_result_cache == {}


@pytest.mark.parametrize("source", ["changed", "deleted"])
def test_viewer_never_associates_old_predictions_with_new_model_bytes(make_service, model_file, monkeypatch, source):
    service = make_service()
    old = service.capture_model(model_file)
    monkeypatch.setenv("NEPTRAINKIT_LIBRARY", str(service.library.root))
    if source == "changed":
        model_file.write_text(MODEL_HEADER + "retrained")
        service.capture_model(model_file)
    else:
        model_file.unlink()
    viewer = SimpleNamespace(nep_result_data=SimpleNamespace(
        nep_txt_path=model_file, _loaded_model_hash=old["content_hash"]), refresh_validation_chip=Mock())
    show_nep.ShowNepWidget._associate_model_version(viewer)
    assert viewer._validation_model_id == old["id"]
    assert viewer._validation_draft is None


@pytest.mark.parametrize("parent", ["self", "child", "grandchild", "missing"])
def test_model_rejects_parent_cycles_atomically(make_service, model_file, parent):
    service = make_service()
    root = service.capture_model(model_file)
    api = ModelService(service.db)
    child = api.add_version(root["project_id"], "child", "NEP", str(model_file), parent_id=root["id"])
    grandchild = api.add_version(root["project_id"], "grandchild", "NEP", str(model_file), parent_id=child.id)
    parent_id = {"self": root["id"], "child": child.id, "grandchild": grandchild.id, "missing": 999}[parent]
    with pytest.raises(ValueError):
        api.modify_model(root["id"], parent_id=parent_id, name="must not change", tags=["must not persist"])
    assert service.get_model(root["id"])["name"] == root["name"]
    tree = api.get_models_by_project_id(root["project_id"])
    assert [row.model_id for row in tree] == [root["id"]]
    assert tree[0].children[0].children[0].model_id == grandchild.id
    api.modify_model(grandchild.id, parent_id=root["id"])
    api.modify_model(child.id, parent_id=None)
    assert len(api.get_models_by_project_id(root["project_id"])) == 2


def test_parent_picker_excludes_model_and_descendants(make_service, model_file):
    service = make_service()
    root = service.capture_model(model_file)
    api = ModelService(service.db)
    child = api.add_version(root["project_id"], "child", "NEP", str(model_file), parent_id=root["id"])
    other = api.add_version(root["project_id"], "other", "NEP", str(model_file))
    host = QWidget()
    widget = dataset_widget.ModelItemWidget(host)
    widget.model_service = api
    widget.load_models_by_project(ProjectService(service.db).get_project_by_id(root["project_id"]))
    widget._view.setCurrentIndex(widget._model.index(0, 0))
    box = MagicMock()
    box.parent_combox = ComboBox()
    box.exec_.return_value = False
    with patch.object(dataset_widget, "ModelInfoMessageBox", return_value=box):
        widget.create_model(modify=True)
    assert [box.parent_combox.itemData(i) for i in range(box.parent_combox.count())] == [None, other.id]
    assert child.id in widget.model_item_dict
    host.close()


@pytest.mark.parametrize("parent_id,model_id", [(999, None), (999, 999)])
def test_model_creation_rejects_missing_or_self_parent(make_service, model_file, parent_id, model_id):
    service = make_service()
    original = service.capture_model(model_file)
    with pytest.raises(ValueError):
        ModelService(service.db).add_version(original["project_id"], "invalid", "NEP", str(model_file),
                                             parent_id=parent_id, id=model_id, tags=["rollback"])
    assert [row["id"] for row in service.models()] == [original["id"]]


def test_edit_from_ancestor_project_preserves_model_owner(make_service, model_file):
    service = make_service()
    projects = ProjectService(service.db)
    parent = projects.create_project("parent")
    child = projects.create_project("child", parent_id=parent.id)
    original = service.capture_model(model_file, project_id=child.id)
    ModelService(service.db).modify_model(original["id"], data_size=1234)
    host = QWidget()
    widget = dataset_widget.ModelItemWidget(host)
    widget.model_service = ModelService(service.db)
    widget.load_models_by_project(projects.get_project_by_id(parent.id))
    widget._view.setCurrentIndex(widget._model.index(0, 0))
    box = MagicMock()
    box.exec_.return_value = True
    box.get_dict.return_value = dict(name="renamed", model_type="NEP", model_path=str(model_file),
                                    notes="updated", tags=[], parent_id=None, data_size=None)
    with patch.object(dataset_widget, "ModelInfoMessageBox", return_value=box), \
         patch.object(dataset_widget.MessageManager, "send_success_message"):
        widget.create_model(modify=True)
    saved = service.get_model(original["id"])
    assert saved["project_id"] == child.id and saved["notes"] == "updated"
    assert saved["data_size"] == 1234
    host.close()


def test_missing_training_data_keeps_manual_errors_and_unknown_values(tmp_path):
    (tmp_path / "nep.txt").write_text(MODEL_HEADER)
    host = QWidget()
    box = ModelInfoMessageBox(host)
    box.train_path_edit.setText(str(tmp_path))
    assert all(box.get_dict()[key] is None for key in ("energy", "force", "virial", "data_size"))
    for control, value in ((box.energy_spinBox, "1.2"), (box.force_spinBox, "3.4"), (box.virial_spinBox, "5.6")):
        control.setText(value)
    with patch("NepTrainKit.ui.widgets.dialog.MessageManager.send_message_box"):
        box.check_path()
    assert [box.get_dict()[key] for key in ("energy", "force", "virial")] == [1.2, 3.4, 5.6]
    host.close()


@pytest.mark.parametrize("all_tags,any_tags,none_tags,expected", [
    (["Fe"], ["spin"], [], True),
    (["Fe", "spin"], ["spin", "Co"], [], True),
    (["Fe"], ["Co"], [], False),
    (["Fe", "Co"], ["spin"], [], False),
    (["Fe"], ["spin"], ["spin"], False),
    (["fe", "FE"], ["SPIN"], ["Co"], True),
])
def test_combined_tag_filters_are_independent(make_service, model_file, all_tags, any_tags, none_tags, expected):
    service = make_service()
    model = service.capture_model(model_file)
    api = ModelService(service.db)
    api.modify_model(model["id"], tags=["Fe", "spin"])
    found = api.search_models_advanced(tags_all=all_tags, tags_any=any_tags, tags_none=none_tags)
    assert [row.model_id for row in found] == ([model["id"]] if expected else [])


@pytest.mark.parametrize("action", ["gc", "model", "project"])
def test_deletion_preserves_shared_baseline_history(make_service, model_file, tmp_path, action):
    service = make_service()
    original = service.capture_model(model_file)
    benchmark = _import(service, tmp_path, efv_frames(), "shared")
    run = service.evaluate(original["id"], benchmark["id"])
    saved = service.save_model(ModelDraft.from_file(model_file), project_name="manual", name="v1")
    suite = service.create_suite("protected baseline", [benchmark["id"]])
    service.set_baseline(suite["id"], saved["id"])
    with service.db.session() as session:
        session.get(ModelVersion, original["id"]).last_seen_at = dt.datetime.utcnow() - dt.timedelta(days=90)
        session.commit()
    before = service.run_details(run)
    if action == "gc":
        preview = service.collect_garbage()
        assert preview["models"] == 1 and preview["results"] == 0
        assert service.latest_run(saved["id"], benchmark["id"])["model_id"] == original["id"]
        assert service.collect_garbage(dry_run=False) == {**preview, "dry_run": False}
    elif action == "model":
        ModelService(service.db).remove_model(original["id"])
    else:
        ProjectService(service.db).remove_project(original["project_id"])
    retained = service.latest_run(saved["id"], benchmark["id"])
    assert retained["id"] == run["id"] and retained["model_id"] == saved["id"]
    assert service.run_details(retained) == before
    assert service.get_suite(suite["id"])["baseline_model_id"] == saved["id"]
    assert service.library.open_object(saved["content_hash"]).is_file()


def test_deleting_model_subtree_never_transfers_runs_to_a_deleted_child(make_service, model_file, tmp_path):
    service = make_service()
    root = service.capture_model(model_file)
    api = ModelService(service.db)
    child = api.add_version(root["project_id"], "child", "NEP", str(model_file), parent_id=root["id"])
    service.snapshot_registered_models()
    survivor = service.save_model(ModelDraft.from_file(model_file), project_name="survivor", name="kept")
    benchmark = _import(service, tmp_path, efv_frames(), "subtree")
    first = service.evaluate(root["id"], benchmark["id"])
    second = service.evaluate(child.id, benchmark["id"], force=True)
    api.remove_model(root["id"])
    with service.db.session() as session:
        rows = list(session.scalars(select(BenchmarkRun)))
    assert {row.id for row in rows} == {first["id"], second["id"]}
    assert {row.model_id for row in rows} == {survivor["id"]}


def test_failed_delete_rolls_back_shared_run_transfer(make_service, model_file, tmp_path, monkeypatch):
    service = make_service()
    old = service.capture_model(model_file)
    benchmark = _import(service, tmp_path, efv_frames(), "rollback")
    run = service.evaluate(old["id"], benchmark["id"])
    new = service.save_model(ModelDraft.from_file(model_file), project_name="saved", name="kept")
    def fail_commit(_session):
        raise OSError("commit failed")
    with monkeypatch.context() as scoped:
        scoped.setattr(service.db.Session.class_, "commit", fail_commit)
        with pytest.raises(OSError):
            ModelService(service.db).remove_model(old["id"])
    assert service.get_model(old["id"])["id"] == old["id"]
    assert service.latest_run(new["id"], benchmark["id"])["model_id"] == old["id"]
    assert service.run_details(run)


def test_scorecard_batches_queries_and_matches_individual_history(make_service, model_file, tmp_path):
    service = make_service()
    models = []
    for i in range(20):
        model_file.write_text(MODEL_HEADER + str(i))
        models.append(service.capture_model(model_file))
    shared = service.save_model(ModelDraft.from_file(model_file), project_name="copy", name="shared")
    benchmarks = [_import(service, tmp_path, efv_frames(count=i + 1), str(i)) for i in range(3)]
    suite = service.create_suite("batch", [b["id"] for b in benchmarks])
    service.set_baseline(suite["id"], models[0]["id"])
    service.evaluate(models[0]["id"], benchmarks[0]["id"])
    service.evaluate(models[-1]["id"], benchmarks[0]["id"])
    with patch.object(service, "predictor_factory", side_effect=RuntimeError("failed prediction")):
        service.evaluate(models[1]["id"], benchmarks[1]["id"])
    queries = []
    def counted(*args):
        queries.append(args[2])
    event.listen(service.db.engine, "before_cursor_execute", counted)
    try:
        card = service.scorecard(suite["id"])
    finally:
        event.remove(service.db.engine, "before_cursor_execute", counted)
    assert len(queries) < 20
    assert shared["id"] in {row["model_id"] for row in card["rows"]}
    for row in card["rows"]:
        for column, cell in zip(card["columns"], row["cells"]):
            run = service.latest_run(row["model_id"], column["set_id"], protocol_version=column["protocol_version"])
            assert cell["status"] == (run["status"] if run else "missing")
            assert cell["run_id"] == (run["id"] if run else None)
            value = run["metrics"].get(column["metric"]) if run and run["status"] == "succeeded" else None
            assert cell["value"] == value
