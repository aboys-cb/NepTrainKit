import uuid

from PySide6.QtCore import QObject
from .database import Database
from .services import ModelService,ProjectService

class DatasetManager(QObject):
    def __init__(self,parent=None):
        super().__init__(parent)

        self._parent = parent

        # database and services
        self._db = Database()
        self.model_service = ModelService(self._db)

        self.project_service = ProjectService(self._db)
    def get_models(self,**kwargs):
        return self.model_service.search_models(**kwargs)

