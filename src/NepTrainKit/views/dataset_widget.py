import os

from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex, Signal, QPoint, QUrl
from PySide6.QtGui import QCursor, QColor, QIcon, QDesktopServices, QShortcut, QKeySequence
from PySide6.QtWidgets import QWidget, QVBoxLayout
from qfluentwidgets import TreeItemDelegate, TreeView, RoundMenu, Action, MessageBox

from NepTrainKit.core import MessageManager
from NepTrainKit.core.dataset import DatasetManager
from NepTrainKit.core.dataset.services import ProjectItem, ModelItem
from NepTrainKit.core.types import ModelTypeIcon
from NepTrainKit.custom_widget import TreeModel, TreeItem,TagDelegate
from NepTrainKit.custom_widget.dialog import ModelInfoMessageBox, AdvancedModelSearchDialog, TagManageDialog


# from NepTrainKit.custom_widget import DatasetItemModel







class ModelItemWidget(QWidget,DatasetManager):
    model_item_dict={}
    projectChangedSignal=Signal(int)
    def __init__(self, parent=None):
        super(ModelItemWidget, self).__init__(parent)
        self._parent=parent
        self.project:ProjectItem
        # self.project_service.create_project("测试","222")
        #
        # self.project_service.create_project("测试1","222")
        self._view = TreeView()


        self._view.clicked.connect(self.item_clicked)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # self._view.setIndentation(10)
        self._view.header().setDefaultSectionSize(50)



        self._view.header().setStretchLastSection(True)
        self._model=TreeModel()

        self._view.setModel(self._model)
        self._model.setHeader(["ID","Name","Size","E(meV/atom)","F(meV/Å)","V(meV/atom)", "Tags","Create At"])
        self._view.setItemDelegateForColumn(6, TagDelegate(self._model))
        width=[90,200,60,90,80,90,200,30]
        for col,w in enumerate(width):
            self._view.setColumnWidth(col,w)


        # self._view.setColumnHidden(1,True)
        # self._view.setColumnWidth(0,110)

        self.create_menu()
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0,0,0,0)
        self._layout.addWidget(self._view)

        self.search_shortcut = QShortcut(QKeySequence("Ctrl+F"),self,context=Qt.ShortcutContext.WindowShortcut)
        self.search_shortcut.activated.connect(self.show_search_dialog)

    def item_clicked(self,index):
        item = index.internalPointer()

        self.projectChangedSignal.emit(item.data(1))
    def create_menu(self):
        self._menu_pos = QPoint()
        self.menu = RoundMenu(parent=self)

        create_action=Action("New",self.menu)
        create_action.triggered.connect(lambda :self.create_model(modify=False))
        self.menu.addAction(create_action)
        modify_action=Action("Modify",self.menu)
        modify_action.triggered.connect(lambda :self.create_model(modify=True))
        self.menu.addAction(modify_action)

        open_action = Action("Open Folder", self.menu)
        open_action.triggered.connect(self.open_folder)
        self.menu.addAction(open_action)

        delete_action = Action("Delete", self.menu)
        delete_action.triggered.connect(self.remove_model)
        self.menu.addAction(delete_action)
        tag_action = Action("Manage Tags", self.menu)
        tag_action.triggered.connect(self.manage_tags)
        self.menu.addAction(tag_action)
        self._view.customContextMenuRequested.connect(self.show_menu)
    def show_menu(self,pos):
        self._menu_pos=pos
        self.menu.exec_(self.mapToGlobal(pos))
    def manage_tags(self):
        dlg = TagManageDialog(self.tag_service, self._parent)
        dlg.exec_()
        # if dlg.tag_changed:
        #     self.

    def _build_tree(self,model,parent:TreeItem):
        child = TreeItem((model.model_id,
                              model.name,
                              model.data_size,
                              model.energy,
                              model.force ,
                              model.virial ,
                          [{"name":tag.name,"color":tag.color} for tag in model.tags],

                              model.created_at.strftime("%Y-%m-%d %H:%M:%S"),))

        child.icon = QIcon(ModelTypeIcon.NEP)

        self.model_item_dict[model.model_id] = model
        parent.appendChild(child)
        for item in model.children:
            self._build_tree(item,child)
        return child
    def load_models_by_project(self, project):
        self._model.clear()
        self.project=project
        models=self.model_service.get_models_by_project_id(project.project_id)
        self.add_models_to_table(models)
    def add_models_to_table(self,models:list[ModelItem]):
        self._model.beginResetModel()
        for model in models:
            self._build_tree(model,self._model.rootItem)
            continue
        self._model.endResetModel()
    def create_model(self,modify=False):
        box = ModelInfoMessageBox(self._parent)
        index=self._view.indexAt(self._menu_pos)
        box.parent_combox.addItem("Top Model",userData=None)
        for  model in self.model_item_dict.values():
            box.parent_combox.addItem(f"{model.model_id}-{model.name}", userData=model.model_id)

        if index.row()!=-1:
            item=index.internalPointer()
            box.parent_combox.setCurrentText(f"{item.data(0)}-{item.data(1)}")
            model_id=item.data(0)
        else:
            box.parent_combox.setCurrentText("Top Model")
            if modify:
                return
            model_id=None

        if modify:
            current_model = self.model_item_dict[model_id]
            box.model_name_edit.setText(current_model.name)
            box.model_note_edit.setText(current_model.notes)
            box.train_path_edit.setText(current_model.model_path)
            box.model_type_combox.setText(current_model.model_type)
            box.energy_spinBox.setText(str(current_model.energy))
            box.force_spinBox.setText(str(current_model.force))
            box.virial_spinBox.setText(str(current_model.virial))
            for tag in current_model.tags:
                box.add_tag(tag.name)


            if current_model.parent_id is not None:
                parent_model = self.model_item_dict[current_model.parent_id]
                box.parent_combox.setCurrentText(f"{parent_model.model_id}-{parent_model.name}")
            else:
                box.parent_combox.setCurrentText("Top Model")

        if not box.exec_():
            return
        data = box.get_dict()
        data["project_id"]=self.project.project_id

        if modify:
            self.model_service.modify_model(current_model.model_id,
                                                **data)
            self.load_models_by_project(self.project)

            MessageManager.send_success_message("Model modification successful")
            return

        project = self.model_service.add_version(
            **data

        )
        if project is None:
            MessageManager.send_error_message("Failed to create model")
        else:
            MessageManager.send_success_message("Model created successfully")
            self.load_models_by_project(self.project)
    def remove_model(self):
        index = self._view.indexAt(self._menu_pos)

        if index.row() == -1:
            return

        item = index.internalPointer()

        model_id = item.data(0)
        box = MessageBox("Ask",
                         "Do you want to delete this item?\nIf you delete it, all items under it will be deleted!",
                         self._parent)
        box.exec_()
        if box.result() == 0:
            return

        self.model_service.remove_model(model_id=model_id)

        MessageManager.send_success_message("Model deleted successfully")
        self.load_models_by_project(self.project)

    def open_folder(self):
        index = self._view.indexAt(self._menu_pos)

        if index.row() == -1:
            return

        item = index.internalPointer()
        model_id = item.data(0)
        model=self.model_item_dict[model_id]
        path=model.model_path
        if path.startswith("http"):
            QDesktopServices.openUrl(QUrl(path))
        else:
            if os.path.exists(path):
                QDesktopServices.openUrl(QUrl("file:///"+path))
    def on_search(self,params):
        models= self.model_service.search_models_advanced(**params)
        self._model.clear()
        self.add_models_to_table(models)

    def show_search_dialog(self):

        box = AdvancedModelSearchDialog(self._parent)
        box.searchRequested.connect(self.on_search)
        box.show()