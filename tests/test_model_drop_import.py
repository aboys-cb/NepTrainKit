"""Model drops identify weights independently of training folders and outputs."""

import pytest
import time
from PySide6.QtCore import QCoreApplication, QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QWidget

from NepTrainKit.core.validation import Library, ValidationService
from NepTrainKit.core.validation.service import find_model_files
from NepTrainKit.ui.pages import data_manager
from NepTrainKit.ui.widgets.validation_dialogs import ImportModelDialog


MODEL = "nep4 2 Fe O\ncutoff 8 4 74 10\n"


def write_model(path, content=MODEL):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def mime_for(*paths):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


def wait_for_outputs(dialog):
    deadline = time.monotonic() + 5
    while dialog._output_task is not None and time.monotonic() < deadline:
        QApplication.instance().processEvents()
        time.sleep(0.001)
    assert dialog._output_task is None


def test_find_models_prefers_valid_nep_then_detects_other_txt_headers(tmp_path):
    first = write_model(tmp_path / "candidate.TXT")
    write_model(tmp_path / "notes.txt", "This is a training note, not a model.")
    write_model(tmp_path / "settings.txt", "nep4 0\n")
    write_model(tmp_path / "nested" / "ignored.txt")
    assert find_model_files(tmp_path) == [first]
    assert find_model_files(first) == [first]
    assert find_model_files(tmp_path / "missing.txt") == []
    assert find_model_files(tmp_path / "notes.txt") == []
    standard = write_model(tmp_path / "nep.txt")
    assert find_model_files(tmp_path) == [standard]
    standard.write_text("invalid model")
    assert find_model_files(tmp_path) == [first]
    second = write_model(tmp_path / "second.txt", "nep4_spin 1 Fe\n")
    assert find_model_files(tmp_path) == [first, second]
    assert data_manager._dropped_models(mime_for(tmp_path, first)) == [first, second]


@pytest.fixture
def model_page(tmp_path, monkeypatch):
    library = Library(tmp_path / "library")
    monkeypatch.setenv("NEPTRAINKIT_LIBRARY", str(library.root))
    monkeypatch.setattr(data_manager, "seed_nep_data_git", lambda *args: None)
    monkeypatch.setattr(data_manager.MessageManager, "send_success_message", lambda *args: None)
    page = data_manager.DataManagerWidget()
    yield page, ValidationService(library)
    page.close()
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("folder_drop", [False, True])
@pytest.mark.parametrize("selected_project", [False, True])
def test_valid_drop_is_prefilled_and_one_confirmation_saves_snapshot(
    model_page, tmp_path, monkeypatch, folder_drop, selected_project,
):
    page, service = model_page
    path = write_model(tmp_path / "training-run" / "nep.txt")
    # Malformed nearby outputs must not become imported model accuracy.
    (path.parent / "train.xyz").write_text("unrelated data")
    for name in ("energy_train.out", "force_train.out", "virial_train.out"):
        (path.parent / name).write_text("0 0 0")
    if selected_project:
        project = page.project_service.create_project("Selected project")
        page.data_item_widget.load_models_by_project(page.project_service.get_project_by_id(project.id))
    mime = mime_for(path.parent if folder_drop else path)
    drag = QDragEnterEvent(QPoint(1, 1), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    page.dragEnterEvent(drag)
    assert drag.isAccepted()
    assert service.models() == []

    def confirm(dialog):
        wait_for_outputs(dialog)
        assert dialog.draft.source_path == str(path)
        assert dialog.name_edit.text() == "training-run"
        assert "nep4" in dialog.model_details.text() and "Fe, O" in dialog.model_details.text()
        assert dialog.yesButton.isEnabled()
        if selected_project:
            assert dialog.project_combo.text() == "Selected project"
        else:
            assert dialog.project_combo.text() == "Inbox"
        assert not service.models()
        # Save the reviewed bytes even if the external training job overwrites nep.txt.
        path.write_text(MODEL + "new weights")
        return 1

    monkeypatch.setattr(ImportModelDialog, "exec", confirm)
    event = QDropEvent(QPointF(1, 1), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    page.dropEvent(event)
    assert event.isAccepted()
    saved, = service.models()
    assert saved["name"] == "training-run" and saved["dataset_ref"] == {}
    assert all(saved[key] is None for key in ("energy", "force", "virial", "data_size"))
    path.unlink()
    assert service.model_file(saved["id"]).read_text() == MODEL
    assert page.data_item_widget.project.project_id == saved["project_id"]


def test_cancelling_valid_drop_does_not_publish_model_or_create_default_project(model_page, tmp_path, monkeypatch):
    page, service = model_page
    path = write_model(tmp_path / "weights.txt")
    monkeypatch.setattr(ImportModelDialog, "exec", lambda self: 0)
    mime = mime_for(path)
    event = QDropEvent(QPointF(1, 1), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    page.dropEvent(event)
    assert event.isAccepted()
    assert service.models() == [] and service.projects() == []
    assert list(service.library.iter_objects()) == []


def test_multiple_candidates_require_selection_and_update_model_information(tmp_path):
    service = ValidationService(Library(tmp_path / "library"))
    files = [write_model(tmp_path / "a.txt"), write_model(tmp_path / "b.txt", "nep4_spin 1 Fe\n")]
    host = QWidget()
    dialog = ImportModelDialog(host, service, files)
    assert not dialog.yesButton.isEnabled()
    assert dialog.draft is None
    dialog.model_combo.setCurrentIndex(1)
    assert dialog.yesButton.isEnabled()
    assert dialog.name_edit.text() == "b"
    assert "nep4_spin" in dialog.model_details.text()
    assert dialog.draft.source_path == str(files[1])
    dialog.project_combo.setText("")
    assert not dialog.yesButton.isEnabled()
    dialog.project_combo.setText("New project")
    dialog.model_combo.setCurrentIndex(0)
    assert dialog.name_edit.text() == "a"
    files[1].unlink()
    dialog.model_combo.setCurrentIndex(1)
    assert not dialog.yesButton.isEnabled()
    host.close()


def test_invalid_or_remote_drop_is_rejected(model_page, tmp_path, monkeypatch):
    page, _ = model_page
    called = []
    monkeypatch.setattr(ImportModelDialog, "exec", lambda self: called.append(True))
    for mime in (mime_for(write_model(tmp_path / "notes.txt", "not a model")), QMimeData()):
        if not mime.hasUrls():
            mime.setUrls([QUrl("https://example.com/nep.txt")])
        drag = QDragEnterEvent(QPoint(1, 1), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        page.dragEnterEvent(drag)
        assert not drag.isAccepted()
        drop = QDropEvent(QPointF(1, 1), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        page.dropEvent(drop)
        assert not drop.isAccepted()
    assert not called


@pytest.mark.parametrize("language", ["en_US", "zh_CN"])
def test_import_dialog_is_translated_and_fits_default_window(tmp_path, language):
    from NepTrainKit.i18n import install_translator

    app = QApplication.instance()
    install_translator(app, language)
    host = QWidget()
    host.resize(900, 700)
    host.show()
    dialog = None
    try:
        file = write_model(tmp_path / "模型文件" / "nep.txt")
        dialog = ImportModelDialog(host, ValidationService(Library(tmp_path / "library")), [file])
        dialog.show()
        for _ in range(3):
            app.processEvents()
        assert dialog.yesButton.isEnabled()
        assert dialog.widget.width() <= host.width()
        assert dialog.widget.height() <= host.height()
        assert dialog.name_edit.text() == "模型文件"
        if language == "zh_CN":
            assert dialog.yesButton.text() == "导入"
            assert dialog.title_label.text() == "导入 NEP 模型"
            assert dialog.location_label.text().startswith("模型库：")
            assert "类型：nep4" in dialog.model_details.text()
        else:
            assert dialog.yesButton.text() == "Import"
            assert dialog.location_label.text().startswith("Library:")
    finally:
        if dialog is not None:
            dialog.close()
        host.close()
        install_translator(app, "en_US")


def test_existing_outputs_prefill_grouped_rmse_and_save_historical_provenance(tmp_path):
    import numpy as np
    service = ValidationService(Library(tmp_path / 'library'))
    path = write_model(tmp_path / 'run' / 'nep.txt')
    for group, scale in [('train', 1), ('test', 2)]:
        for key, columns, error in [('energy', 1, .001), ('force', 3, .002), ('virial', 6, .003), ('mforce', 3, .004)]:
            np.savetxt(path.parent / f'{key}_{group}.out', np.full((2, columns * 2), [error * scale] * columns + [0] * columns))
    host = QWidget(); host.resize(900, 700); host.show()
    dialog = ImportModelDialog(host, service, [path])
    dialog.show()
    wait_for_outputs(dialog)
    QApplication.processEvents()
    assert dialog.widget.height() <= host.height()
    assert dialog.output_combo.currentData() == 'train'
    assert [float(field.text()) for field in dialog.output_fields.values()] == pytest.approx([1, 2, 3, 4])
    assert 'unverified' in dialog.output_status.text()
    dialog.output_combo.setCurrentIndex(1)
    assert [float(field.text()) for field in dialog.output_fields.values()] == pytest.approx([2, 4, 6, 8])
    saved = dialog.save_model(dialog.draft)
    assert [saved[key] for key in ('energy', 'force', 'virial')] == pytest.approx([2, 4, 6])
    record = service.imported_accuracy(saved['id'])
    assert record['status'] == 'unverified' and record['group'] == 'test'
    assert record['metrics']['mforce_rmse'] == pytest.approx(8)
    assert record['counts']['mforce']['components'] == 6
    assert record['model_sha256'] == saved['content_hash']
    assert len(record['outputs']) == 4
    from sqlalchemy import select
    from NepTrainKit.core.dataset.models import BenchmarkRun
    with service.db.session() as session:
        assert session.scalars(select(BenchmarkRun)).all() == []
    dialog.close(); host.close()


def test_output_scan_cancels_on_close_and_coalesces_model_changes(tmp_path, monkeypatch):
    import threading
    from NepTrainKit.ui.widgets import validation_dialogs
    started, release = threading.Event(), threading.Event()
    calls = []
    def scan(groups, digest, stopped):
        calls.append(digest); started.set(); release.wait(5)
        return {} if stopped() else {'train': {'status': 'unverified', 'metrics': {'energy_rmse': 7}}}
    monkeypatch.setattr(validation_dialogs, 'inspect_output_groups', scan)
    a = write_model(tmp_path / 'a.txt'); b = write_model(tmp_path / 'b.txt', MODEL + 'b')
    (tmp_path / 'energy_train.out').write_text('0 0')
    service = ValidationService(Library(tmp_path / 'library'))
    host = QWidget(); dialog = ImportModelDialog(host, service, [a, b])
    dialog.model_combo.setCurrentIndex(0)
    assert started.wait(2)
    dialog.model_combo.setCurrentIndex(1)
    dialog.model_combo.setCurrentIndex(0)
    assert not dialog.yesButton.isEnabled()
    dialog.model_combo.setCurrentIndex(1)
    release.set(); wait_for_outputs(dialog)
    assert len(calls) == 2 and calls[-1] == dialog.draft.content_hash
    assert dialog.output_fields['energy'].text() == '7'
    release.clear(); started.clear()
    dialog.model_combo.setCurrentIndex(0)
    assert started.wait(2)
    dialog.reject(); release.set(); wait_for_outputs(dialog)
    assert not service.models()
    host.close()


def test_outputs_from_known_different_model_are_not_saved_as_model_rmse(tmp_path):
    import json
    service = ValidationService(Library(tmp_path / 'library'))
    path = write_model(tmp_path / 'nep.txt')
    (tmp_path / 'energy_train.out').write_text('.001 0\n')
    (tmp_path / 'prediction.meta.json').write_text(json.dumps({
        'schema_version': 1, 'predictions': {'train.xyz': {'model': {'sha256': 'different'}}}}))
    host = QWidget(); dialog = ImportModelDialog(host, service, [path])
    wait_for_outputs(dialog)
    assert dialog.output_fields['energy'].text() == '1'
    assert 'weights only' in dialog.output_status.text()
    saved = dialog.save_model(dialog.draft)
    assert saved['energy'] is None and not service.imported_accuracy(saved['id'])
    dialog.close(); host.close()
