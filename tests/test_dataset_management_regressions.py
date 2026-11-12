"""Ownership, transactional imports, library publication and registry refresh."""

import datetime as dt
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import Mock, patch

import pytest
from sqlalchemy import select

from test_validation_framework import efv_frames, eos_frames, write_xyz, _import
from test_validation_framework import make_service as make_service, model_file as model_file
from NepTrainKit.core.dataset.catalog import import_dataset_catalog, load_dataset_catalog
from NepTrainKit.core.dataset.models import BenchmarkRun, ModelVersion, Project, Tag
from NepTrainKit.core.dataset.services import ModelService, ProjectService
from NepTrainKit.core.validation import Library, ValidationService
from NepTrainKit.ui.pages import data_manager
from NepTrainKit.ui.views import dataset_widget
from NepTrainKit.ui.widgets.dialog import ModelInfoMessageBox
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog


def test_project_delete_keeps_external_lineage_and_shared_history(make_service, model_file, tmp_path):
    service = make_service()
    projects, models = ProjectService(service.db), ModelService(service.db)
    a, b, c = (projects.create_project(name) for name in ("A", "B", "C"))
    nested = projects.create_project("A/nested", parent_id=a.id)
    owned = service.capture_model(model_file, project_id=a.id)
    local = models.add_version(project_id=nested.id, name="local", model_type="NEP", model_path="/local", parent_id=owned["id"])
    external = service.capture_model(model_file, project_id=b.id, auto=False)
    with service.db.session() as session:
        session.get(ModelVersion, external["id"]).parent_id = owned["id"]
        session.commit()
    other = models.add_version(project_id=b.id, name="other", model_type="NEP", model_path="/other", parent_id=local.id)
    descendant = models.add_version(project_id=c.id, name="descendant", model_type="NEP", model_path="/descendant", parent_id=external["id"])
    # A version may re-enter the deleted project after an external ancestor.
    reentered = models.add_version(project_id=a.id, name="reentered", model_type="NEP", model_path="/reentered", parent_id=external["id"])
    reference = _import(service, tmp_path, efv_frames(), "reference", protocol="efv")
    suite = service.create_suite("baseline survives", [reference["id"]])
    service.set_baseline(suite["id"], external["id"])
    with service.db.session() as session:
        session.add(BenchmarkRun(id="shared-run", model_id=owned["id"], set_id=reference["id"], protocol_version=1,
                                 status="succeeded", metrics={"energy_rmse": 0.}))
        session.commit()
    projects.remove_project(a.id)
    with service.db.session() as session:
        assert all(session.get(ModelVersion, key) is None for key in (owned["id"], local.id, reentered.id))
        assert session.get(ModelVersion, external["id"]).parent_id is None
        assert session.get(ModelVersion, other.id).parent_id is None
        assert session.get(ModelVersion, descendant.id).parent_id == external["id"]
        assert session.get(BenchmarkRun, "shared-run").model_id == external["id"]
    assert projects.get_project_by_id(a.id) is None
    assert projects.get_project_by_id(nested.id) is None
    assert projects.get_project_by_id(b.id).model_num == 2
    assert projects.get_project_by_id(c.id).model_num == 1
    assert service.get_suite(suite["id"])["baseline_model_id"] == external["id"]


def test_suite_membership_is_deduplicated_and_rolled_back_atomically(make_service, tmp_path):
    service = make_service()
    first = _import(service, tmp_path, efv_frames(), "first", protocol="efv")
    second = _import(service, tmp_path, efv_frames(2), "second", protocol="efv")
    for name in ("new", "new again"):
        with pytest.raises(ValueError, match="Unknown benchmark set"):
            service.create_suite(name, [first["id"], "missing"])
    assert service.suites() == []
    suite = service.create_suite("new", iter([first["id"], first["id"]]))
    service.add_to_suite(suite["id"], [first["id"], second["id"], second["id"]])
    assert [item["id"] for item in service.suite_sets(suite["id"])] == [first["id"], second["id"]]
    third = _import(service, tmp_path, efv_frames(3), "third", protocol="efv")
    with pytest.raises(ValueError, match="Unknown benchmark set"):
        service.add_to_suite(suite["id"], [third["id"], "missing"])
    assert [item["id"] for item in service.suite_sets(suite["id"])] == [first["id"], second["id"]]


@pytest.mark.parametrize("preserve_ids", [False, True])
def test_catalog_failure_rolls_back_project_models_and_tags_then_retries(make_service, monkeypatch, preserve_ids):
    service = make_service()
    projects, models = ProjectService(service.db), ModelService(service.db)
    previous = projects.create_project("previous")
    models.add_version(project_id=previous.id, name="keep", model_type="NEP", model_path="/keep", tags=["keep"], id=1000000)
    catalog = load_dataset_catalog()
    original = models.add_version
    calls = []
    def fail_second(**kwargs):
        created = original(**kwargs)
        calls.append(created.id)
        if len(calls) == 2:
            raise OSError("injected after flush")
        return created
    with monkeypatch.context() as scoped:
        scoped.setattr(models, "add_version", fail_second)
        with pytest.raises(OSError, match="after flush"):
            import_dataset_catalog(catalog, projects, models, preserve_catalog_ids=preserve_ids)
    with service.db.session() as session:
        assert [row.name for row in session.scalars(select(Project))] == ["previous"]
        assert [row.name for row in session.scalars(select(ModelVersion))] == ["keep"]
        assert [row.name for row in session.scalars(select(Tag))] == ["keep"]
    result = import_dataset_catalog(catalog, projects, models, preserve_catalog_ids=preserve_ids)
    assert result.imported_models == len(catalog.models)
    assert projects.get_project_by_id(previous.id).model_num == 1


@pytest.mark.parametrize("protocol,frames", [("efv", efv_frames), ("eos", eos_frames)])
@pytest.mark.parametrize("change", [False, True])
def test_large_copy_confirmation_keeps_the_reviewed_bytes(make_service, tmp_path, monkeypatch, protocol, frames, change):
    service = make_service()
    path = write_xyz(tmp_path / "preview.xyz", frames())
    reviewed = path.read_bytes()
    from types import SimpleNamespace
    dialog = SimpleNamespace(validation_errors=lambda: [], _loaded_bytes=reviewed,
        values=lambda: dict(path=str(path), name="reviewed", provenance={}, protocol=protocol, suite="plan"),
        group_field_combo=SimpleNamespace(currentData=lambda: None), tr=lambda text: text, _invalidate_file=Mock())
    original = service.import_set
    monkeypatch.setattr(service, "import_set", lambda path, **kwargs: original(path, max_bytes=1, **kwargs))
    def confirm(_error):
        if change:
            write_xyz(path, frames()[:2])
        return True
    if change:
        with pytest.raises(ValueError, match="file changed"):
            BenchmarkImportDialog.import_into(dialog, service, confirm)
        assert service.sets() == service.suites() == []
    else:
        imported = BenchmarkImportDialog.import_into(dialog, service, confirm)
        assert len(service.set_frames(imported[0]["id"])) == len(frames())
        assert len(service.suite_sets(service.suites()[0]["id"])) == 1


def _collect_while_publishing(service, operation, method):
    collector = ValidationService(Library(service.library.root))
    published, release, started, done = (threading.Event() for _ in range(4))
    results, errors = [], []
    original = getattr(service.library, method)
    def pause(*args, **kwargs):
        result = original(*args, **kwargs)
        published.set()
        assert release.wait(10)
        return result
    def publish():
        try:
            results.append(operation())
        except BaseException as error:
            errors.append(error)
    def collect():
        started.set()
        try:
            collector.collect_garbage(dry_run=False)
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()
    with patch.object(service.library, method, pause):
        writer = threading.Thread(target=publish)
        cleanup = threading.Thread(target=collect)
        writer.start()
        try:
            assert published.wait(5)
            cleanup.start()
            assert started.wait(2)
            assert not done.wait(.1), "GC must wait for publication and its DB reference"
        finally:
            release.set()
            writer.join(10)
            if cleanup.ident is not None:
                cleanup.join(10)
    assert not writer.is_alive() and not cleanup.is_alive()
    assert not errors
    return results[0]


@pytest.mark.parametrize("kind", ["reference", "model", "result"])
def test_gc_waits_for_each_publication_and_registration(make_service, model_file, tmp_path, kind):
    service = make_service()
    if kind == "reference":
        path = write_xyz(tmp_path / "reference.xyz", efv_frames())
        record = _collect_while_publishing(service,
            lambda: service.import_set(path, name="reference", provenance={}, protocol="efv"), "put_bytes")
        assert len(service.set_frames(record["id"])) == 4
    elif kind == "model":
        record = _collect_while_publishing(service, lambda: service.capture_model(model_file, auto=False), "put_bytes")
        assert service.model_file(record["id"]).read_bytes() == model_file.read_bytes()
    else:
        model = service.capture_model(model_file, auto=False)
        reference = _import(service, tmp_path, efv_frames(), "reference", protocol="efv")
        record = _collect_while_publishing(service, lambda: service.evaluate(model["id"], reference["id"]), "write_result")
        assert record["status"] == "succeeded"
        assert service.run_details(record)["details"]


def test_cross_process_import_is_not_collected(make_service, tmp_path):
    service = make_service()
    path = write_xyz(tmp_path / "reference.xyz", efv_frames())
    marker = tmp_path / "published"
    script = '''
import runpy, sys
runpy.run_path('tests/conftest.py')
from pathlib import Path
from NepTrainKit.core.validation import Library, ValidationService
service = ValidationService(Library(sys.argv[1]))
original = service.library.put_bytes
def pause(data):
    digest = original(data)
    Path(sys.argv[3]).touch()
    assert sys.stdin.readline().strip() == 'continue'
    return digest
service.library.put_bytes = pause
record = service.import_set(sys.argv[2], name='cross-process', provenance={}, protocol='efv')
print(record['id'], flush=True)
'''
    process = subprocess.Popen([sys.executable, "-c", script, str(service.library.root), str(path), str(marker)],
        cwd=Path(__file__).resolve().parents[1], text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    done, started = threading.Event(), threading.Event()
    errors = []
    def collect():
        started.set()
        try:
            service.collect_garbage(dry_run=False)
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()
    cleanup = threading.Thread(target=collect)
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists()
        cleanup.start()
        assert started.wait(2)
        assert not done.wait(.1)
        output, error = process.communicate("continue\n", timeout=15)
        assert process.returncode == 0, error
        cleanup.join(10)
        assert not cleanup.is_alive() and not errors
        record = service.get_set(output.strip().splitlines()[-1])
        assert len(service.set_frames(record["id"])) == 4
    finally:
        if process.poll() is None:
            process.communicate("continue\n", timeout=15)
        if cleanup.ident is not None:
            cleanup.join(10)


def test_gc_deletion_failure_does_not_restore_pruned_model_rows(make_service, model_file, monkeypatch):
    service = make_service()
    model = service.capture_model(model_file)
    with service.db.session() as session:
        session.get(ModelVersion, model["id"]).last_seen_at = dt.datetime.utcnow() - dt.timedelta(days=40)
        session.commit()
    original = service.library.delete_object
    def delete_then_fail(digest):
        original(digest)
        raise OSError("injected cleanup failure")
    monkeypatch.setattr(service.library, "delete_object", delete_then_fail)
    with pytest.raises(OSError, match="cleanup failure"):
        service.collect_garbage(dry_run=False)
    with pytest.raises(ValueError, match="Unknown model"):
        service.get_model(model["id"])
    monkeypatch.setattr(service.library, "delete_object", original)
    assert service.collect_garbage(dry_run=False)["objects"] == 0


def test_gc_rechecks_publications_between_pruning_and_file_deletion(make_service, tmp_path, monkeypatch):
    service = make_service()
    publisher = ValidationService(Library(service.library.root))
    path = write_xyz(tmp_path / "new-reference.xyz", efv_frames())
    service.library.put_bytes(path.read_bytes())
    original = service.db.write_session
    calls, records = [], []
    @contextmanager
    def interleaved():
        calls.append(1)
        with original() as session:
            yield session
        if len(calls) == 1:
            records.append(publisher.import_set(path, name="new", provenance={}, protocol="efv"))
    monkeypatch.setattr(service.db, "write_session", interleaved)
    assert service.collect_garbage(dry_run=False)["objects"] == 0
    assert len(service.set_frames(records[0]["id"])) == 4


def test_result_write_failure_is_recorded_and_releases_transaction(make_service, model_file, tmp_path):
    service = make_service()
    model = service.capture_model(model_file, auto=False)
    reference = _import(service, tmp_path, efv_frames(), "reference", protocol="efv")
    with patch.object(service.library, "write_result", side_effect=OSError("injected disk full")):
        failed = service.evaluate(model["id"], reference["id"])
    assert failed["status"] == "failed" and "disk full" in failed["error"]
    assert failed["result_path"] == ""
    retried = service.evaluate(model["id"], reference["id"])
    assert retried["status"] == "succeeded"
    assert service.run_details(retried)["details"]


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("NEPTRAINKIT_LIBRARY", str(tmp_path / "ui-library"))
    with patch.object(data_manager, "seed_nep_data_git"):
        page = data_manager.DataManagerWidget()
    page.library_widget.refresh = Mock()
    yield page
    page.close()
    page.deleteLater()


def test_moved_project_reloads_the_selected_scope(registry):
    projects, models = registry.project_service, registry.model_service
    parent, other = projects.create_project("Parent"), projects.create_project("Other")
    child = projects.create_project("Child", parent_id=parent.id)
    version = models.add_version(project_id=child.id, name="version", model_type="NEP", model_path="/unavailable")
    registry.project_widget.load_all_projects()
    registry.data_item_widget.load_models_by_project(projects.get_project_by_id(parent.id))
    assert version.id in registry.data_item_widget.model_item_dict
    projects.modify_project(child.id, "Child", parent_id=other.id)
    registry.project_widget.projectsChanged.emit()
    assert registry.data_item_widget.model_item_dict == {}
    assert registry.data_item_widget.project.model_num == 0
    registry.data_item_widget.load_models_by_project(projects.get_project_by_id(other.id))
    assert version.id in registry.data_item_widget.model_item_dict


@pytest.mark.parametrize("change", ["rename", "delete"])
def test_tag_manager_refresh_prevents_stale_tags_on_note_save(registry, change):
    project = registry.project_service.create_project("Project")
    version = registry.model_service.add_version(project_id=project.id, name="version", model_type="NEP", model_path="/unavailable", tags=["old"])
    registry.project_widget.load_all_projects()
    view = registry.data_item_widget
    view.load_models_by_project(registry.project_service.get_project_by_id(project.id))
    tag = registry.tag_service.get_tags(name="old")[0]
    dialog = Mock()
    if change == "rename":
        dialog.exec_.side_effect = lambda: registry.tag_service.update_tag(tag.tag_id, name="new")
    else:
        dialog.exec_.side_effect = lambda: registry.tag_service.remove_tag(tag.tag_id)
    with patch.object(dataset_widget, "TagManageDialog", return_value=dialog):
        view.manage_tags()
    expected = ["new"] if change == "rename" else []
    assert [item.name for item in view.model_item_dict[version.id].tags] == expected
    view._view.setCurrentIndex(view._model.index(0, 0))
    class NoteEditor(ModelInfoMessageBox):
        def exec_(self):
            self.model_note_edit.setPlainText("only notes edited")
            return True
    with patch.object(dataset_widget, "ModelInfoMessageBox", NoteEditor):
        view.create_model(modify=True)
    current = registry.model_service.get_models_by_project_id(project.id)[0]
    assert current.notes == "only notes edited"
    assert [item.name for item in current.tags] == expected
    assert not registry.tag_service.get_tags(name="old")
