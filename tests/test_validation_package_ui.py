from PySide6.QtWidgets import QWidget
from test_validation_package import package_source as package_source
from test_validation_ui import scored as scored, make_service as make_service
from test_validation_toolbar import toolbar as toolbar
from NepTrainKit.core.validation.package import export_package
from NepTrainKit.ui.widgets.validation_package_dialogs import ExportValidationPackageDialog, ImportValidationPackageDialog
from NepTrainKit.ui.widgets.style_lifetime import keep_created_styles_alive


def test_preview_choices_and_invalid_replacement(package_source, tmp_path):
    keep_created_styles_alive()
    service, suite, model = package_source
    parent = QWidget()
    parent.resize(1100, 850)
    export = ExportValidationPackageDialog(parent, service, suite['id'], [model['id']])
    assert export.yesButton.isEnabled() and export.preview.manifest['models'] == []
    export.include_models.setChecked(True)
    assert len(export.preview.manifest['models']) == 1
    assert export.preview.manifest['suite']['baseline'] == model['content_hash']
    path = tmp_path / 'package.ntkpack'
    export_package(export.preview, path)
    dialog = ImportValidationPackageDialog(parent, service)
    assert not dialog.yesButton.isEnabled()
    before = service.suites()
    dialog.load_path(path)
    assert dialog.table.rowCount() == 1
    assert dialog.name_edit.text() == 'HEA (2)'
    assert not dialog.yesButton.isEnabled()  # an explicit destination project is required
    dialog.project_combo.setText('Shared')
    assert dialog.yesButton.isEnabled()
    dialog.name_edit.setText('HEA')
    assert dialog.destination.text() == 'Will create: HEA (2)'
    assert service.suites() == before
    bad = tmp_path / 'bad.ntkpack'
    bad.write_text('bad archive')
    dialog.load_path(bad)
    assert dialog.preview is None and dialog.table.rowCount() == 0
    assert not dialog.yesButton.isEnabled() and not dialog.project_combo.isEnabled()
    assert service.suites() == before
    dialog.close(); export.close(); parent.close()


def test_suite_only_package_and_unsaved_draft(package_source):
    keep_created_styles_alive()
    service, suite, model = package_source
    service.set_baseline(suite['id'], None)
    parent = QWidget()
    dialog = ExportValidationPackageDialog(parent, service, suite['id'], [0])
    assert dialog.yesButton.isEnabled() and not dialog.include_models.isEnabled()
    dialog.close(); parent.close()


def test_menu_export_available_without_a_model(toolbar, monkeypatch):
    library, host = toolbar.library, toolbar.host
    calls = []
    library.model_combo.clear()
    monkeypatch.setattr(library, 'current_model_row', lambda: None)
    library._on_selection_changed()
    assert not library.save_action.isEnabled()
    assert library.export_package_action.isEnabled()
    assert host.save_dir_button.isEnabled() and not host.save_dir_button.button.isEnabled()
    monkeypatch.setattr(library, 'export_validation_package', lambda: calls.append('export'))
    monkeypatch.setattr(library, 'import_validation_package', lambda: calls.append('import'))
    library.export_package_action.trigger()
    library.import_package_action.trigger()
    assert calls == ['export', 'import']


def test_page_handlers_roundtrip(toolbar, tmp_path, monkeypatch):
    from NepTrainKit.ui.views import model_library
    library, service = toolbar.library, toolbar.scored.service
    suite_before = service.get_suite(toolbar.scored.suite['id'])
    target = tmp_path / 'page.ntkpack'
    def export_accept(dialog):
        dialog.include_models.setChecked(True)
        return True
    monkeypatch.setattr(ExportValidationPackageDialog, 'exec', export_accept)
    monkeypatch.setattr(model_library.QFileDialog, 'getSaveFileName', lambda *_: (str(target), ''))
    library.export_validation_package()
    assert target.exists()
    def import_accept(dialog):
        dialog.load_path(target)
        dialog.project_combo.setText('Imported')
        assert dialog.yesButton.isEnabled()
        return True
    monkeypatch.setattr(ImportValidationPackageDialog, 'exec', import_accept)
    library.import_validation_package()
    imported = service.get_suite(library.current_suite_id())
    assert imported['id'] != suite_before['id']
    assert imported['baseline_model_id'] is not None
    assert service.get_suite(suite_before['id']) == suite_before
