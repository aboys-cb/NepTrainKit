from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex, Signal
from PySide6.QtGui import QCursor, QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout
from qfluentwidgets import TreeItemDelegate, TreeView, RoundMenu, Action
from NepTrainKit.core.dataset import DatasetManager
# from NepTrainKit.custom_widget import DatasetItemModel


class ModelItemTableWidget(TreeView):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.manager = DatasetManager()
        # self.setModel()
        # self._model=DatasetItemModel()
        # self.setModel(self._model)
        # self._model.setHorizontalHeaderLabels(["ID","Type","Name","Create At"])
        self.create_menu()
    def create_menu(self):
        self.menu = RoundMenu(parent=self)
        self.menu.addAction(Action("test",self))
        self.customContextMenuRequested.connect(self.show_menu)



    def show_menu(self,pos):
        self.menu.exec_( self.mapToGlobal(pos) )

    def load_item(self):
        all_models = self.manager.get_models()
        for item in all_models:
            # item.
            pass

