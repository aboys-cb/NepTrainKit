import uuid

from PySide6.QtCore import QObject
from .database import DataBaseEngine

class DatasetManager(QObject):
    def __init__(self,parent=None):
        super().__init__(parent)

        self._parent = parent

        # database and services
        # self._db = Database()
        # self.dataset_service = DatasetService(self._db)
        # self.model_service = ModelService(self._db)
        # self.lineage_service = LineageService(self._db)

    def generate_dataset_id(self):

        return uuid.uuid4().hex