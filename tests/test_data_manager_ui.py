#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PySide6.QtCore import QTranslator, Qt
from PySide6.QtWidgets import QApplication

import NepTrainKit.ui.pages.data_manager as data_manager_module
import NepTrainKit.ui.views.project_view as project_view_module
from NepTrainKit.ui.views.dataset_widget import ModelItemWidget
from NepTrainKit.ui.views.project_view import ProjectWidget


class _Url:
    def __init__(self, path: str):
        self._path = path

    def isLocalFile(self):
        return True

    def toLocalFile(self):
        return self._path


class _MimeData:
    def __init__(self, urls):
        self._urls = urls

    def hasUrls(self):
        return bool(self._urls)

    def urls(self):
        return self._urls


class _DropEvent:
    def __init__(self, urls):
        self._mime_data = _MimeData(urls)
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime_data

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


class TestDataManagerUi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    def test_project_and_model_actions_are_visible_without_context_menu(self):
        with patch.object(project_view_module.QTimer, "singleShot"):
            project_widget = ProjectWidget()
        model_widget = ModelItemWidget()

        self.assertEqual(project_widget.new_project_button.text(), "New project")
        self.assertFalse(project_widget.modify_project_button.isEnabled())
        self.assertFalse(project_widget.delete_project_button.isEnabled())
        self.assertEqual(model_widget.new_model_button.text(), "New model")
        self.assertEqual(
            model_widget._model.headerData(4, Qt.Orientation.Horizontal),
            "F(meV/Å)",
        )
        self.assertEqual(model_widget.search_button.text(), "Search")
        self.assertFalse(model_widget.new_model_button.isEnabled())
        self.assertFalse(model_widget.open_folder_button.isEnabled())

    def test_selecting_project_enables_new_model_action(self):
        model_widget = ModelItemWidget()
        model_widget.model_service = SimpleNamespace(
            get_models_by_project_id=MagicMock(return_value=[])
        )
        project = SimpleNamespace(project_id=7)

        model_widget.load_models_by_project(project)

        self.assertTrue(model_widget.new_model_button.isEnabled())

    def test_dropped_path_prefills_existing_model_editor(self):
        model_widget = ModelItemWidget()
        model_widget.project = SimpleNamespace(project_id=7)
        model_widget.model_item_dict = {}
        with patch(
            "NepTrainKit.ui.views.dataset_widget.ModelInfoMessageBox"
        ) as box_class:
            box = box_class.return_value
            box.exec_.return_value = False

            model_widget.create_model(initial_path="/tmp/training-run")

        box.train_path_edit.setText.assert_called_once_with("/tmp/training-run")

    def test_chinese_catalog_translates_visible_data_manager_actions(self):
        catalog = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "NepTrainKit"
            / "translations"
            / "neptrainkit_zh_CN.qm"
        )
        translator = QTranslator(self._app)
        self.assertTrue(translator.load(str(catalog)))
        self._app.installTranslator(translator)
        try:
            with patch.object(project_view_module.QTimer, "singleShot"):
                project_widget = ProjectWidget()
            model_widget = ModelItemWidget()
            self.assertEqual(project_widget.new_project_button.text(), "新建项目")
            self.assertEqual(model_widget.new_model_button.text(), "新建模型")
        finally:
            self._app.removeTranslator(translator)

    @patch.object(data_manager_module.os.path, "isdir", return_value=True)
    def test_dropped_model_folder_opens_prefilled_model_editor(self, _isdir):
        folder = str(Path("/tmp/training-run"))
        model_widget = SimpleNamespace(
            project=SimpleNamespace(project_id=1),
            create_model=MagicMock(),
        )
        page = SimpleNamespace(
            data_item_widget=model_widget,
            tr=lambda text: text,
        )
        event = _DropEvent([_Url(folder)])

        data_manager_module.DataManagerWidget.dropEvent(page, event)

        model_widget.create_model.assert_called_once_with(initial_path=folder)
        self.assertTrue(event.accepted)
        self.assertFalse(event.ignored)

    @patch.object(data_manager_module.MessageManager, "send_info_message")
    @patch.object(data_manager_module.os.path, "isdir", return_value=True)
    def test_drop_without_project_explains_required_selection(
        self, _isdir, info_message
    ):
        page = SimpleNamespace(
            data_item_widget=SimpleNamespace(create_model=MagicMock()),
            tr=lambda text: text,
        )
        event = _DropEvent([_Url("/tmp/training-run")])

        data_manager_module.DataManagerWidget.dropEvent(page, event)

        info_message.assert_called_once_with(
            "Select a project before dropping a model folder"
        )
        self.assertTrue(event.ignored)


if __name__ == "__main__":
    unittest.main()
