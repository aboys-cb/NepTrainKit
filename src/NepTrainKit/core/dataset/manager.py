"""High-level dataset services facade used by the UI layer.

Provides a thin wrapper that exposes database-backed services for projects,
models, and tags. The concrete logic lives in ``services.py``.

Examples
--------
>>> # Used by the GUI to wire services
>>> dm = DatasetManager()
>>> isinstance(dm, DatasetManager)
True
"""

from .database import Database
from .services import ModelService,ProjectService,TagService

class DatasetManager:
    """Container for dataset-related service singletons."""
    _db:Database
    _model_service:ModelService
    _project_service:ProjectService
    _tag_service:TagService
    def __init__(self,*args,**kwargs):
        # super().__init__(parent)

        pass

    @property
    def db(self):
        """Return the underlying :class:`Database` instance."""
        return self._db
    @db.setter
    def db(self,db):
        self._db=db
    @property
    def model_service(self):
        """Return the :class:`ModelService` for model CRUD/search."""
        return self._model_service
    @model_service.setter
    def model_service(self,service):
        self._model_service=service

    @property
    def project_service(self):
        """Return the :class:`ProjectService` for project CRUD/search."""
        return self._project_service

    @project_service.setter
    def project_service(self, service):
        self._project_service = service

    @property
    def tag_service(self):
        """Return the :class:`TagService` for tag CRUD/search."""
        return self._tag_service

    @tag_service.setter
    def tag_service(self, service):
        self._tag_service = service
