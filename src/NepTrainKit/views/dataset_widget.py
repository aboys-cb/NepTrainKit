from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex, Signal, QPoint
from PySide6.QtGui import QCursor, QColor, QIcon
from PySide6.QtWidgets import QWidget, QVBoxLayout
from qfluentwidgets import TreeItemDelegate, TreeView, RoundMenu, Action
from NepTrainKit.core.dataset import DatasetManager
from NepTrainKit.core.types import ModelTypeIcon
from NepTrainKit.custom_widget import TreeModel, TreeItem,TagDelegate


# from NepTrainKit.custom_widget import DatasetItemModel







class ModelItemWidget(QWidget,DatasetManager):
    project_item_dict={}
    projectChangedSignal=Signal(int)
    def __init__(self, parent=None):
        super(ModelItemWidget, self).__init__(parent)
        self._parent=parent

        # self.project_service.create_project("测试","222")
        #
        # self.project_service.create_project("测试1","222")
        self._view = TreeView()


        self._view.clicked.connect(self.item_clicked)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self._view.setIndentation(10)
        self._view.header().setDefaultSectionSize(90)


        self._view.header().setStretchLastSection(True)
        self._model=TreeModel()

        self._view.setModel(self._model)
        self._model.setHeader(["ID","Name","Size","E(meV/atom)","F(meV/Å)","V(meV/atom)", "Tags","Create At"])
        self._view.setItemDelegateForColumn(6, TagDelegate(self._model))


        # self._view.setColumnHidden(1,True)
        # self._view.setColumnWidth(0,110)

        self.create_menu()
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0,0,0,0)
        self._layout.addWidget(self._view)




    def item_clicked(self,index):
        item = index.internalPointer()

        self.projectChangedSignal.emit(item.data(1))
    def create_menu(self):
        self._menu_pos = QPoint()
        self.menu = RoundMenu(parent=self)
        return
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
    def _build_tree(self,model,parent:TreeItem):
        child = TreeItem((model.model_id,
                              model.name,
                              model.data_size,
                              model.energy,
                              model.force ,
                              model.virial ,
                          [tag.name for tag in model.tags],

                              model.created_at.strftime("%Y-%m-%d %H:%M:%S"),))
        print(child.itemData)
        child.icon = QIcon(ModelTypeIcon.NEP)

        # self.project_item_dict[project.project_id] = project
        parent.appendChild(child)
        for item in model.children:
            self._build_tree(item,child)
        return child
    def load_models(self,project_id):
        self._model.clear()

        models=self.model_service.get_models_by_project_id(project_id)
        print(models)

        self._model.beginResetModel()
        for model in models:
            self._build_tree(model,self._model.rootItem)
            continue
            #"ID","Name","Size","Energy","Force","Virial", "Tags","Create At"
            parent = self._model.rootItem
            child = TreeItem((model.id,
                              model.name,
                              model.data_size,
                              model.energy,
                              model.force ,
                              model.virial ,
                              "",

                              model.created_at.strftime("%Y-%m-%d %H:%M:%S"),))
            parent.appendChild(child)
            rechild = TreeItem((model.id,
                              model.name,
                              model.data_size,
                              model.energy,
                              model.force ,
                              model.virial ,
                              "",

                              model.created_at.strftime("%Y-%m-%d %H:%M:%S"),))
            child.appendChild(rechild)

        self._model.endResetModel()
