#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/8/31 08:49
# @Author  : 兵
# @email    : 1747193328@qq.com


from PySide6.QtCore import QObject, QTimer, Qt, QPoint, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QWidget, QVBoxLayout
from qfluentwidgets import RoundMenu, Action, TreeView, TableView, MessageBox
from NepTrainKit.core import MessageManager

from NepTrainKit.core.dataset.database import Database
from NepTrainKit.core.dataset.services import ModelService, ProjectService
from NepTrainKit.custom_widget import IdNameTableModel, TreeModel, TreeItem, ProjectInfoMessageBox


class ProjectWidget(QWidget):
    project_item_dict={}
    projectChangedSignal=Signal(int)
    def __init__(self, parent=None):
        super(ProjectWidget, self).__init__(parent)
        self._parent=parent
        self._db = Database()
        self.model_service = ModelService(self._db)

        self.project_service = ProjectService(self._db)
        # self.project_service.create_project("测试","222")
        #
        # self.project_service.create_project("测试1","222")
        self._view = TreeView()
        self._view.clicked.connect(self.item_clicked)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self._view.setIndentation(10)
        self._view.header().setDefaultSectionSize(5)


        self._view.header().setStretchLastSection(True)
        self._model=TreeModel()

        self._view.setModel(self._model)
        self._model.setHeader(["Project Name","ID",""])
        self._model.count_column=2


        self._view.setColumnHidden(1,True)
        self._view.setColumnWidth(0,110)

        self.create_menu()
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0,0,0,0)
        self._layout.addWidget(self._view)
        QTimer.singleShot(2, self.load_all_projects)
    def item_clicked(self,index):
        item = index.internalPointer()

        self.projectChangedSignal.emit(item.data(1))
    def create_menu(self):
        self._menu_pos = QPoint()
        self.menu = RoundMenu(parent=self)

        create_action=Action("Create Project",self.menu)
        create_action.triggered.connect(lambda :self.create_project(modify=False))
        self.menu.addAction(create_action)
        modify_action=Action("Modify Project",self.menu)
        modify_action.triggered.connect(lambda :self.create_project(modify=True))
        self.menu.addAction(modify_action)
        delete_action = Action("Delete Project", self.menu)
        delete_action.triggered.connect(self.remove_project)
        self.menu.addAction(delete_action)

        self._view.customContextMenuRequested.connect(self.show_menu)
    def show_menu(self,pos):
        self._menu_pos=pos
        self.menu.exec_(self.mapToGlobal(pos))
    def create_project(self,modify=False):
        box = ProjectInfoMessageBox(self._parent)
        index=self._view.indexAt(self._menu_pos)
        if index.row()!=-1:
            item=index.internalPointer()
            box.parent_combox.addItem(item.data(0))
            parent_id=item.data(1)
        else:
            box.parent_combox.addItem("Top Project")
            parent_id=None
        self_id:int=None
        box.setWindowTitle(f"Project Info")
        if modify and parent_id is not None:
            project=self.project_service.get_project_by_id(parent_id)
            self_id = project.id
            box.project_name.setText(project.name)
            box.project_note.setText(project.description)
            #不允许修改parent_id
            if project.parent_id is not None:
                parent_item=self.project_item_dict[project.parent_id]
                box.parent_combox.addItem(parent_item.data(0))
                box.parent_combox.setCurrentText(parent_item.data(0))
            else:
                box.parent_combox.addItem("Top Project")

                box.parent_combox.setCurrentText("Top Project")

            box.parent_combox.setDisabled(True)

            parent_id=project.parent_id
        if not box.exec_():
            return
        name=box.project_name.text().strip()
        note=box.project_note.toPlainText().strip()
        if modify:
            self.project_service.modify_project(self_id,
                                                name=name,description=note)
            self.load_all_projects()

            MessageManager.send_success_message("Project modification successful")
            return

        project = self.project_service.create_project(
            name=name,
            description=note,
            parent_id=parent_id,

        )
        if project is None:
            MessageManager.send_error_message("Failed to create project")
        else:
            MessageManager.send_success_message("Project created successfully")
            self.load_all_projects()


    def remove_project(self):
        box = ProjectInfoMessageBox(self._parent)

        index = self._view.indexAt(self._menu_pos)

        if index.row() == -1:
            return

        item = index.internalPointer()
        box.parent_combox.addItem(item.data(0))
        parent_id = item.data(1)
        box = MessageBox("Ask",
                         "Do you want to delete this item?\nIf you delete it, all items under it will be deleted!",
                         self)
        box.exec_()
        if box.result() == 0:
            return

        self.project_service.remove_project(project_id=parent_id)

        MessageManager.send_success_message("Project deleted successfully")
        self.load_all_projects()
    def load_all_projects(self):
        self._model.clear()
        all_project = self.project_service.search_projects()
        self._model.beginResetModel()
        for project in all_project:
            # print(project.children)
            if project.parent_id is not None:
                parent=self.project_item_dict[project.parent_id]
            else:
                parent=self._model.rootItem
            child = TreeItem((project.name, project.id) )
            parent.appendChild(child)
            self.project_item_dict[project.id] = child

        self._model.endResetModel()
