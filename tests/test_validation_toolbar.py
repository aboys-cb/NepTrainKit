"""Validation actions follow the visible page and current model in the top menus."""

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import RoundMenu, SplitToolButton
from test_validation_ui import _wait_until
from test_validation_ui import make_service as make_service
from test_validation_ui import scored as scored

from NepTrainKit.main import NepTrainKitMainWindow
from NepTrainKit.ui.pages import data_manager as manager_module
from NepTrainKit.ui.views.model_library import ModelLibraryWidget
from NepTrainKit.ui.widgets.style_lifetime import keep_created_styles_alive


class ToolbarHost(QWidget):
    open_file_dialog = NepTrainKitMainWindow.open_file_dialog
    export_file_dialog = NepTrainKitMainWindow.export_file_dialog
    _refresh_page_actions = NepTrainKitMainWindow._refresh_page_actions

    def __init__(self):
        """Exercise main-window dispatch without native frameless-window setup."""
        super().__init__()
        self.load_menu = RoundMenu(parent=self)
        self.save_menu = RoundMenu(parent=self)
        self.open_dir_button = SplitToolButton(self)
        self.save_dir_button = SplitToolButton(self)
        self.open_dir_button.clicked.connect(self.open_file_dialog)
        self.save_dir_button.clicked.connect(self.export_file_dialog)
        self.stackedWidget = QStackedWidget(self)
        self.stackedWidget.currentChanged.connect(self._refresh_page_actions)
        layout = QVBoxLayout(self)
        layout.addWidget(self.open_dir_button)
        layout.addWidget(self.save_dir_button)
        layout.addWidget(self.stackedWidget)


@pytest.fixture
def toolbar(scored, monkeypatch):
    keep_created_styles_alive()
    monkeypatch.setattr(manager_module, "get_library", lambda: scored.service.library)
    monkeypatch.setattr(manager_module, "seed_nep_data_git", lambda *_: None)
    monkeypatch.setattr(manager_module, "ModelLibraryWidget", lambda parent: ModelLibraryWidget(parent, scored.service))
    host = ToolbarHost()
    page = manager_module.DataManagerWidget(host)
    page.toolbarActionsChanged.connect(host._refresh_page_actions)
    host.stackedWidget.addWidget(page)
    other = QWidget()
    host.stackedWidget.addWidget(other)
    host.resize(1180, 850)
    host.show()
    page.open_scorecard(scored.suite["id"], scored.new["id"])
    yield SimpleNamespace(host=host, page=page, library=page.library_widget, other=other, scored=scored)
    host.close()


def test_top_actions_dispatch_and_detach_on_page_and_tab_changes(toolbar, monkeypatch):
    host, page, library = toolbar.host, toolbar.page, toolbar.library
    calls = []
    monkeypatch.setattr(library, "import_benchmark", lambda: calls.append("import"))
    monkeypatch.setattr(library, "save_current", lambda: calls.append("save"))
    toolbar.other.open_file = lambda: calls.append("other-open")
    toolbar.other.export_file = lambda: calls.append("other-save")
    assert not hasattr(library, "import_button") and not hasattr(library, "save_button")
    assert not hasattr(library, "star_button")
    assert host.load_menu.actions() == [library.diagnostic_action, library.edit_diagnostic_action, library.import_action, library.import_package_action]
    assert host.save_menu.actions() == [library.save_action, library.star_action, library.save_plan_action, library.export_package_action]
    library.diagnostic_action.trigger()
    assert library.diagnostic_editor.isVisible()
    library.diagnostic_editor.hide()
    assert library.controls_panel.isVisible()
    host.open_dir_button.button.click()
    host.save_dir_button.button.click()
    library.import_action.trigger()
    library.save_action.trigger()
    assert calls == ["import", "save", "import", "save"]
    assert host.open_dir_button.toolTip() == "Import reference dataset"
    assert host.save_dir_button.toolTip() == "Save to project…"

    for _ in range(3):
        host.stackedWidget.setCurrentWidget(toolbar.other)
        assert host.load_menu.actions() == [] and host.save_menu.actions() == []
        host.open_dir_button.button.click()
        host.save_dir_button.button.click()
        assert calls[-2:] == ["other-open", "other-save"]
        assert host.save_dir_button.toolTip() == "Save data from this page"
        host.stackedWidget.setCurrentWidget(page)
        assert host.save_menu.actions() == [library.save_action, library.star_action, library.save_plan_action, library.export_package_action]
        page.show_projects()
        assert host.load_menu.actions() == [] and host.save_menu.actions() == []
        assert not host.open_dir_button.isEnabled() and not host.save_dir_button.isEnabled()
        page.open_scorecard(toolbar.scored.suite["id"], toolbar.scored.new["id"])
        assert host.load_menu.actions() == [library.diagnostic_action, library.edit_diagnostic_action, library.import_action, library.import_package_action]
        assert host.save_dir_button.isEnabled()


def test_top_favourite_and_save_state_follow_model_and_validation(toolbar, monkeypatch):
    host, library, scored = toolbar.host, toolbar.library, toolbar.scored
    assert not library.star_action.isChecked()
    library.star_action.trigger()
    assert scored.service.get_model(scored.new["id"])["starred"]
    assert library.star_action.isChecked() and library.star_action.text() == "Unstar"
    library.model_combo.setCurrentIndex(library.model_combo.findData(scored.old["id"]))
    assert not library.star_action.isChecked() and library.star_action.text() == "Star"
    library.star_action.trigger()
    assert scored.service.get_model(scored.old["id"])["starred"]

    calls = []
    monkeypatch.setattr(library, "save_current", lambda: calls.append("save"))
    task = library.run_all()
    assert task is not None
    assert not host.save_dir_button.isEnabled() and not library.save_action.isEnabled()
    host.export_file_dialog()
    assert calls == []
    host.stackedWidget.setCurrentWidget(toolbar.other)
    assert _wait_until(lambda: library._run_task is None)
    assert not host.save_dir_button.isEnabled() and host.save_menu.actions() == []
    host.stackedWidget.setCurrentWidget(toolbar.page)
    assert host.save_dir_button.isEnabled()
    host.save_dir_button.button.click()
    assert calls == ["save"]

    from NepTrainKit.core.validation.service import ModelDraft
    draft = ModelDraft(scored.service.model_file(scored.old["id"]).read_bytes() + b"\nunsaved", "unsaved.txt")
    toolbar.page.open_scorecard(scored.suite["id"], draft=draft)
    assert host.save_dir_button.isEnabled() and not library.star_action.isEnabled()

    from unittest.mock import patch
    with patch.object(library, "service", side_effect=OSError("unavailable")), patch(
        "NepTrainKit.ui.views.model_library.MessageManager"
    ):
        library.refresh()
    assert not host.save_dir_button.isEnabled() and not library.star_action.isEnabled()
    host.export_file_dialog()
    assert calls == ["save"]


def _project_index(widget, project_id, parent=None):
    from PySide6.QtCore import QModelIndex
    parent = QModelIndex() if parent is None else parent
    for row in range(widget._model.rowCount(parent)):
        index = widget._model.index(row, 0, parent)
        if index.internalPointer().data(1) == project_id:
            return index
        child = _project_index(widget, project_id, index)
        if child.isValid():
            return child
    return QModelIndex()


def test_project_model_mutations_update_counts_and_validation_without_losing_selection(toolbar, tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from NepTrainKit.ui.views import dataset_widget as model_module
    from test_validation_framework import MODEL_HEADER
    page, library = toolbar.page, toolbar.library
    parent = page.project_service.create_project('family')
    project = page.project_service.create_project('child', parent_id=parent.id)
    page.show_projects()
    tree, models = page.project_widget, page.data_item_widget
    parent_index = _project_index(tree, parent.id)
    selected = _project_index(tree, project.id)
    tree._view.setExpanded(parent_index, True)
    tree._view.setCurrentIndex(selected)
    tree.item_clicked(selected)
    current = library.model_combo.currentData()
    comparisons = library.comparison_picker.selected_ids()
    path = tmp_path / 'registered' / 'nep.txt'
    path.parent.mkdir()
    path.write_text(MODEL_HEADER + 'registered model')
    payload = dict(name='registered', model_type='NEP', model_path=str(path.parent), parent_id=None)
    box = MagicMock()
    box.exec_.return_value = True
    box.get_dict.side_effect = lambda: dict(payload)
    monkeypatch.setattr(model_module, 'ModelInfoMessageBox', lambda *_: box)
    models.create_model()
    model_id = next(iter(models.model_item_dict))
    assert tree.project_item_dict[project.id].model_num == 1
    assert tree.project_item_dict[parent.id].model_num == 1
    assert _project_index(tree, project.id).internalPointer().data(2) == 1
    assert tree._view.currentIndex().internalPointer().data(1) == project.id
    assert tree._view.isExpanded(_project_index(tree, parent.id))
    assert library.model_combo.findData(model_id) >= 0
    assert library.model_combo.currentData() == current
    assert library.comparison_picker.selected_ids() == comparisons
    assert library.service().model_file(model_id).read_bytes() == path.read_bytes()

    models._view.setCurrentIndex(models._model.index(0, 0))
    payload['name'] = 'renamed model'
    models.create_model(modify=True)
    assert 'renamed model' in library.model_combo.itemText(library.model_combo.findData(model_id))
    assert tree.project_item_dict[project.id].model_num == 1

    models._view.setCurrentIndex(models._model.index(0, 0))
    confirmation = MagicMock()
    confirmation.result.return_value = 1
    monkeypatch.setattr(model_module, 'MessageBox', lambda *_: confirmation)
    models.remove_model()
    assert library.model_combo.findData(model_id) == -1
    assert tree.project_item_dict[project.id].model_num == 0
    assert tree.project_item_dict[parent.id].model_num == 0
    assert model_id not in models.model_item_dict
    assert models.project.project_id == project.id
    assert library.model_combo.currentData() == current


def test_registered_models_are_listed_without_a_validation_plan(make_service, tmp_path):
    from NepTrainKit.core.dataset.services import ModelService, ProjectService
    from test_validation_framework import MODEL_HEADER
    service = make_service()
    project = ProjectService(service.db).create_project('saved')
    models = ModelService(service.db)
    path = tmp_path / 'standalone.txt'
    path.write_text(MODEL_HEADER + 'standalone')
    first = models.add_version(project_id=project.id, name='first', model_type='NEP', model_path=path)
    widget = ModelLibraryWidget(service=service)
    widget.refresh()
    assert widget.model_combo.findData(first.id) >= 0
    second = models.add_version(project_id=project.id, name='second', model_type='NEP', model_path=path)
    widget.refresh()
    assert widget.model_combo.findData(second.id) >= 0
    assert widget.model_combo.currentData() == first.id
    assert service.get_model(first.id)['content_hash'] == service.get_model(second.id)['content_hash']
    assert len(service.models()) == 2  # Reuse object bytes, not another registry record.
    path.unlink()
    widget.refresh()
    assert widget.model_combo.findData(first.id) >= 0  # Snapshots survive source removal.
    widget.close()


def test_registry_sync_does_not_replace_snapshots_or_duplicate_records(make_service, tmp_path):
    from NepTrainKit.core.dataset.services import ModelService, ProjectService
    from test_validation_framework import MODEL_HEADER
    service = make_service()
    project = ProjectService(service.db).create_project('registered')
    registry = ModelService(service.db)
    path = tmp_path / 'nep.txt'
    original = (MODEL_HEADER + 'first snapshot').encode()
    path.write_bytes(original)
    saved = registry.add_version(project_id=project.id, name='named version', model_type='NEP', model_path=path, notes='keep notes', energy=2.3, tags=['keep tag'])
    missing = registry.add_version(project_id=project.id, name='missing', model_type='NEP', model_path=tmp_path / 'missing.txt')
    remote = registry.add_version(project_id=project.id, name='remote', model_type='NEP', model_path='https://example.com/nep.txt')
    broken = tmp_path / 'broken.txt'
    broken.write_text('not a potential')
    invalid = registry.add_version(project_id=project.id, name='invalid', model_type='NEP', model_path=broken)
    errors = service.snapshot_registered_models()
    assert set(errors) == {missing.id, remote.id, invalid.id}
    assert [row['id'] for row in service.models()] == [saved.id]
    captured = service.get_model(saved.id)
    assert captured['name'] == 'named version' and captured['notes'] == 'keep notes'
    assert captured['energy'] == 2.3 and captured['project_id'] == project.id
    assert registry.get_models_by_project_id(project.id)[0].tags[0].name == 'keep tag'
    path.write_text(MODEL_HEADER + 'changed at source')
    service.snapshot_registered_models()
    assert service.model_file(saved.id).read_bytes() == original
    assert service.get_model(saved.id)['content_hash'] == captured['content_hash']
    assert ProjectService(service.db).get_project_by_id(project.id).model_num == 4


def test_deleting_selected_project_clears_stale_models_and_actions(toolbar):
    page, library = toolbar.page, toolbar.library
    project_id = toolbar.scored.new['project_id']
    page.show_projects()
    index = _project_index(page.project_widget, project_id)
    page.project_widget._view.setCurrentIndex(index)
    page.project_widget.item_clicked(index)
    page.project_service.remove_project(project_id)
    page.project_widget.projectsChanged.emit()
    assert page.data_item_widget.project is None
    assert not page.data_item_widget.new_model_button.isEnabled()
    assert page.data_item_widget._model.rowCount() == 0
    assert not page.data_item_widget.model_item_dict
    assert project_id not in page.project_widget.project_item_dict
    assert library.model_combo.findData(toolbar.scored.new['id']) == -1


def test_navigation_reuses_scorecard_and_plot_without_evaluation(toolbar):
    from unittest.mock import patch
    from PySide6.QtWidgets import QApplication
    host, page, library = toolbar.host, toolbar.page, toolbar.library
    service = toolbar.scored.service
    plot = library.detail.eos_plots[0]
    plot.setRange(xRange=(1, 2), yRange=(3, 4), padding=0)
    QApplication.processEvents()
    with patch.object(service, 'evaluate') as evaluate, patch.object(service, 'predictor_factory') as predictor, \
         patch.object(service, 'scorecard', wraps=service.scorecard) as scorecard, \
         patch.object(service, 'snapshot_registered_models', wraps=service.snapshot_registered_models) as snapshots, \
         patch.object(library, '_update_detail', wraps=library._update_detail) as render:
        for _ in range(3):
            host.stackedWidget.setCurrentWidget(toolbar.other)
            host.stackedWidget.setCurrentWidget(page)
            page.show_projects()
            page.tabs.setCurrentItem('validation')
            page.stack.setCurrentWidget(library)
            QApplication.processEvents()
        evaluate.assert_not_called()
        predictor.assert_not_called()
        scorecard.assert_not_called()
        snapshots.assert_not_called()
        render.assert_not_called()
        assert library.detail.eos_plots[0] is plot
        assert plot.viewRange() == [[1, 2], [3, 4]]
        # Manual refresh remains available and rebuilds the detail only once.
        library.refresh_button.click()
        assert scorecard.call_count == snapshots.call_count == render.call_count == 1
        evaluate.assert_not_called()
        predictor.assert_not_called()


def test_navigation_does_not_close_diagnostic_editor(toolbar):
    from unittest.mock import patch
    from PySide6.QtWidgets import QApplication
    library, host = toolbar.library, toolbar.host
    library.show_diagnostic_editor()
    library.diagnostic_editor.name.setText('Keep editing')
    with patch.object(library, '_update_detail', wraps=library._update_detail) as render:
        host.stackedWidget.setCurrentWidget(toolbar.other)
        QApplication.processEvents()
        host.stackedWidget.setCurrentWidget(toolbar.page)
        QApplication.processEvents()
        assert library.diagnostic_editor.isVisible()
        assert library.editor_scroll.isVisible() and library.controls_panel.isHidden()
        assert library.diagnostic_editor.name.text() == 'Keep editing'
        render.assert_not_called()
        library.diagnostic_editor.hide()  # Explicit return still restores the workbench.
        assert library.controls_panel.isVisible() and library.editor_scroll.isHidden()
        assert render.call_count == 1


def test_first_load_and_explicit_viewer_handoff_read_results_once(scored):
    from unittest.mock import patch
    from PySide6.QtWidgets import QApplication
    from NepTrainKit.core.validation.service import ModelDraft
    service = scored.service
    for handoff in (False, True):
        widget = ModelLibraryWidget(service=service)
        with patch.object(service, 'evaluate') as evaluate, patch.object(service, 'predictor_factory') as predictor, \
             patch.object(service, 'scorecard', wraps=service.scorecard) as scorecard, \
             patch.object(widget, '_update_detail', wraps=widget._update_detail) as render:
            if handoff:
                draft = ModelDraft(service.model_file(scored.new['id']).read_bytes() + b'\nnew', '/test/nep.txt')
                widget.open_scorecard(scored.suite['id'], draft=draft)
            widget.show()
            QApplication.processEvents()
            assert scorecard.call_count == render.call_count == 1
            if handoff:
                assert widget.current_model_row()['model_id'] == 0
            assert widget.current_suite_id() == scored.suite['id']
            evaluate.assert_not_called()
            predictor.assert_not_called()
        widget.close()


def test_navigation_loads_changed_library_without_reusing_old_rows(toolbar, tmp_path):
    from NepTrainKit.core.validation import Library, ValidationService
    from unittest.mock import patch
    from PySide6.QtWidgets import QApplication
    library, host = toolbar.library, toolbar.host
    host.stackedWidget.setCurrentWidget(toolbar.other)
    service = ValidationService(Library(tmp_path / 'new-library'))
    library._service = service
    with patch.object(service, 'evaluate') as evaluate:
        host.stackedWidget.setCurrentWidget(toolbar.page)
        QApplication.processEvents()
        assert library._library_root == service.library.root
        assert library.current_suite_id() is None and library.current_model_row() is None
        evaluate.assert_not_called()


def test_plain_entry_defers_result_files_and_charts_until_requested(scored):
    from unittest.mock import patch
    from PySide6.QtWidgets import QApplication
    service = scored.service
    widget = ModelLibraryWidget(service=service)
    with patch.object(service, 'run_details', wraps=service.run_details) as details, \
         patch.object(service, 'evaluate') as evaluate, patch.object(service, 'predictor_factory') as predictor, \
         patch.object(widget.detail, 'show_groups', wraps=widget.detail.show_groups) as plot:
        widget.show()
        QApplication.processEvents()
        assert widget.current_suite_id() == scored.suite['id']
        assert widget.curves_placeholder.isVisible() and widget.splitter.isHidden()
        assert not widget.curves_button.isChecked() and not widget.detail.eos_plots
        widget.refresh_button.click()
        widget.model_combo.setCurrentIndex(widget.model_combo.findData(scored.new['id']))
        details.assert_not_called()
        plot.assert_not_called()
        widget.curves_button.click()
        assert details.call_count > 0 and plot.call_count == 1
        assert widget.detail.eos_plots and widget.curves_placeholder.isHidden()
        assert widget.splitter.isVisible()
        widget.curves_button.click()
        details.reset_mock();plot.reset_mock()
        widget.model_combo.setCurrentIndex(widget.model_combo.findData(scored.old['id']))
        widget.hide();widget.show()
        QApplication.processEvents()
        assert widget.curves_placeholder.isVisible()
        details.assert_not_called()
        plot.assert_not_called()
        evaluate.assert_not_called()
        predictor.assert_not_called()
    widget.close()


def test_finished_validation_does_not_refresh_during_shutdown(toolbar):
    from unittest.mock import Mock, patch
    library, host = toolbar.library, toolbar.host
    host._thread_shutdown = SimpleNamespace(closing=True)
    task = Mock()
    library._run_task = task
    with patch.object(library, 'refresh') as refresh, patch.object(library, '_update_detail') as render:
        library._on_run_finished()
        assert library._run_task is None
        task.deleteLater.assert_called_once()
        refresh.assert_not_called()
        render.assert_not_called()
