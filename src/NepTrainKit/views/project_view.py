#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/8/31 08:49
# @Author  : 兵
# @email    : 1747193328@qq.com


from PySide6.QtCore import QObject, QTimer, Qt, QPoint, Signal
from PySide6.QtGui import QCursor, QIcon
from PySide6.QtWidgets import QWidget, QVBoxLayout,QGroupBox
from qfluentwidgets import RoundMenu, Action, TreeView, TableView, MessageBox, FluentIcon, SearchLineEdit

from NepTrainKit.core import MessageManager
from NepTrainKit.core.dataset import DatasetManager

from NepTrainKit.core.dataset.database import Database
from NepTrainKit.core.dataset.services import ModelService, ProjectService, ProjectItem
from NepTrainKit.custom_widget import IdNameTableModel, TreeModel, TreeItem, ProjectInfoMessageBox, TagManageDialog
from NepTrainKit.views import KitToolBarBase


class ProjectWidget(QWidget,DatasetManager):
    project_item_dict={}
    projectChangedSignal=Signal(ProjectItem)
    def __init__(self, parent=None):
        super(ProjectWidget, self).__init__(parent)
        self._parent=parent


        self._view = TreeView()
        self._view.clicked.connect(self.item_clicked)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # self._view.setIndentation(10)
        self._view.header().setDefaultSectionSize(5)
        self._view.header().setStretchLastSection(True)
        self._model=TreeModel()

        self._view.setModel(self._model)
        self._model.setHeader(["(ID) Project Name","ID",""])
        # self._model.count_column=2


        self._view.setColumnHidden(1,True)
        self._view.setColumnWidth(0,150)

        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0,0,0,0)
        self._layout.addWidget(self._view)
        self.create_menu()

        QTimer.singleShot(1, self.load)


    def item_clicked(self,index):
        item = index.internalPointer()
        project=self.project_item_dict[item.data(1)]
        self.projectChangedSignal.emit(project)
    def create_menu(self):
        self._menu_pos = QPoint()
        self.menu = RoundMenu(parent=self)

        create_action = Action( "New",self.menu)
        create_action.triggered.connect(lambda :self.create_project(modify=False))
        self.menu.addAction(create_action)
        modify_action = Action( "Modify",self.menu)
        modify_action.triggered.connect(lambda :self.create_project(modify=True))
        self.menu.addAction(modify_action)
        delete_action = Action( "Delete", self.menu)
        delete_action.triggered.connect(self.remove_project)
        self.menu.addAction(delete_action)


        self._view.customContextMenuRequested.connect(self.show_menu)
    def show_menu(self,pos):
        self._menu_pos=pos
        self.menu.exec_(self.mapToGlobal(pos))
    def create_project(self,modify=False):
        box = ProjectInfoMessageBox(self._parent)
        index=self._view.indexAt(self._menu_pos)
        box.parent_combox.addItem("Top Project",userData=None)
        for  project in self.project_item_dict.values():
            box.parent_combox.addItem(project.name, userData=project.project_id)

        if index.row()!=-1:
            item=index.internalPointer()
            project_id=item.data(1)
            project=self.project_item_dict[project_id]

            box.parent_combox.setCurrentText(project.name)
        else:
            box.parent_combox.setCurrentText("Top Project")
            if modify:
                return
            project_id=None
        box.setWindowTitle(f"Project Info")
        if modify:
            current_project = self.project_item_dict[project_id]
            box.project_name.setText(current_project.name)
            box.project_note.setText(current_project.notes)


            if current_project.parent_id is not None:
                parent_project = self.project_item_dict[current_project.parent_id]
                box.parent_combox.setCurrentText(parent_project.name)
            else:
                box.parent_combox.setCurrentText("Top Project")

        if not box.exec_():
            return
        name=box.project_name.text().strip()
        note=box.project_note.toPlainText().strip()
        project_id=box.parent_combox.currentData()
        if modify:
            self.project_service.modify_project(current_project.project_id,
                                                name=name,notes=note,parent_id=project_id)
            self.load_all_projects()

            MessageManager.send_success_message("Project modification successful")
            return

        project = self.project_service.create_project(
            name=name,
            notes=note,
            parent_id=project_id,

        )
        if project is None:
            MessageManager.send_error_message("Failed to create project")
        else:
            MessageManager.send_success_message("Project created successfully")
            self.load_all_projects()


    def remove_project(self):


        index = self._view.indexAt(self._menu_pos)

        if index.row() == -1:
            return

        item = index.internalPointer()

        project_id = item.data(1)
        box = MessageBox("Ask",
                         "Do you want to delete this item?\nIf you delete it, all items under it will be deleted!",
                         self._parent)
        box.exec_()
        if box.result() == 0:
            return

        self.project_service.remove_project(project_id=project_id)

        MessageManager.send_success_message("Project deleted successfully")
        self.load_all_projects()

    def load(self):
        self.load_all_projects()


    def _build_tree(self,project,parent:TreeItem):
        child = TreeItem((f"({project.project_id}){project.name}", project.project_id,project.model_num))
        child.icon=FluentIcon.FOLDER.icon()

        self.project_item_dict[project.project_id] = project
        parent.appendChild(child)
        for item in project.children:
            self._build_tree(item,child)
        return child



    def load_all_projects(self):
        self._model.clear()
        all_project = self.project_service.search_projects(parent_id=None)
        self._model.beginResetModel()
        for project in all_project:

            self._build_tree(project,self._model.rootItem)


        self._model.endResetModel()
