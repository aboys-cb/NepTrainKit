import uuid

from PySide6.QtCore import QObject
from .database import DataBaseEngine

class DatasetManager(QObject):
    def __init__(self,parent=None):
        super().__init__(parent)

        self._parent = parent
        self.database=DataBaseEngine()

    def generate_dataset_id(self):

        return uuid.uuid4().hex