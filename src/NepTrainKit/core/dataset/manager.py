import random
import traceback
import uuid
from pathlib import Path

from PySide6.QtCore import QObject
from .database import Database
from .services import ModelService,ProjectService

class DatasetManager:
    _db:Database
    _model_service:ModelService
    _project_service:ProjectService
    def __init__(self,*args,**kwargs):
        # super().__init__(parent)

        pass

    @property
    def db(self):
        return self._db
    @db.setter
    def db(self,db):
        self._db=db
    @property
    def model_service(self):
        return self._model_service
    @model_service.setter
    def model_service(self,service):
        self._model_service=service

    @property
    def project_service(self):
        return self._project_service

    @project_service.setter
    def project_service(self, service):
        self._project_service = service

    def gen_test(self):
        try:
            project = self.project_service.create_project(f"Test{random.random()}","测试1")
            project = self.project_service.create_project(f"Test{random.random()}","测试1",parent_id=project.id)

            self.model_service.add_version_from_path(
                name="第1代",
                project_id=project.id,
            model_type="NEP",
            path=Path(r"D:\Desktop\nep-new"),


            notes  = "",
            parent_id  = None,


            )
            model2 = self.model_service.add_version_from_path(
                name="第2代",
                project_id=project.id,
                model_type="NEP",
                path=Path(r"D:\Desktop\nep-new"),
                notes  = "",
                    tags=["NEP", "GPUMDs" ],

                parent_id  = None,
            )
            model = self.model_service.add_version_from_path(
            name="第3代",
            project_id=project.id,
            model_type="NEP",
            path=Path(r"D:\Desktop\nep-new"),
            notes  = "",
                tags=["NEP","GPUMD","nep","Cs"],

            parent_id  = model2.id,
            )
            model = self.model_service.add_version_from_path(
                name="第4代",
                project_id=project.id,
                model_type="NEP",
                path=Path(r"D:\Desktop\nep-new"),
                notes="",
                parent_id=model2.id,
                tags=["NEP"]
            )

        except Exception as e:
            print(traceback.format_exc())