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
