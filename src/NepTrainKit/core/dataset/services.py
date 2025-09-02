"""Service layer for registering datasets and models."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
from decimal import Decimal
from pathlib import Path

from typing import Iterable
from dataclasses import dataclass

from loguru import logger

from NepTrainKit import utils

from .database import Database
from .models import (

    Event,
    Project,
    ModelVersion,
    StorageRef,
)
def query_set_to_dict(func):
    def wrapper(*args, **kwargs):
        objs = func(*args, **kwargs)
        if not isinstance(objs, (list, tuple)):
            objs = [objs]
        result=[]
        for obj in objs:
            obj_dict = {}
            for column in obj.__table__.columns.keys():
                val = getattr(obj, column)
                if isinstance(val, Decimal):
                    val = float(val)
                if isinstance(val, datetime.datetime):
                    val = val.strftime("%Y/%m/%d %H:%M:%S")
                elif isinstance(val, datetime.date):
                    val = val.strftime("%Y/%m/%d")

                obj_dict[column] = val
            result.append(obj_dict)

        return result
    return wrapper

def _hash_file(path: str) -> str:
    """Return the SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:  # pragma: no cover - simple
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
def _stat_path(p: Path) -> tuple[int | None, datetime.datetime | None, bool]:
    try:
        st = p.stat()
        return st.st_size, datetime.datetime.fromtimestamp(st.st_mtime), True
    except FileNotFoundError:
        return None, None, False

def _create_storage(session, path: str, scheme: str = "file") -> StorageRef:
    p = Path(path).expanduser().resolve()
    size, mtime, ok = _stat_path(p)

    if scheme == "file":
        uri =  str(p)
        content_hash = utils.sha256_file(p)
    else:
        # cas 策略
        content_hash = utils.sha256_file(p)
        #TODO: 如果拷贝文件 放在那里
        cas_dir = settings.cas_root.expanduser().resolve()
        cas_dir.mkdir(parents=True, exist_ok=True)
        dst = cas_dir / content_hash
        if not dst.exists():
            shutil.copy2(p, dst)
        uri = content_hash
    storage = StorageRef(
        scheme=f"{scheme}://",
        uri=uri,
        size=size,
        content_hash=content_hash,
        mtime=mtime,
        exist_flag=True
    )
    session.add(storage)
    session.flush()
    return storage
@dataclass
class ProjectService:
    """
    项目的service
    """
    db: Database
    # @query_set_to_dict
    def search_projects(self,**kwargs) -> list[Project]:
        with self.db.session() as session:
            return (
                session.query(Project)
                .filter_by(**kwargs)
                .order_by(Project.created_at)
                .all()
            )
    def get_project_by_id(self, project_id: int) -> Project|None:
        projects = self.search_projects(id=project_id)
        if len(projects) == 0:
            return None
        return projects[0]

    def create_project(self,  name: str, description: str = "", parent_id: int = None   ) -> Project|None:
        try:
            with self.db.session() as session:
                family = Project(name=name, description=description,parent_id=parent_id )
                session.add(family)
                session.commit()
                return family
        except Exception as e:
            logger.error(e)
            return None
    def modify_project(self, project_id: int, name: str, description: str = "", parent_id: int = None ) -> Project|None:


        with self.db.session() as session:

            session.query(Project).filter_by(id=project_id).update(
                {"name":name,"description":description}
            )

            session.commit()


    def remove_project(self, project_id: int) -> Project|None:
        with self.db.session() as session:
            session.query(Project).filter_by(id=project_id).delete()


            session.commit()


@dataclass
class ModelService:
    db: Database
    def get_models_by_project_id(self, project_id: int) -> list[ModelVersion]:
        return self.search_models(project_id=project_id)

    def search_models(self, **kwargs) -> list[ModelVersion]:
        with self.db.session() as session:
            return (
                session.query(ModelVersion)
                .filter_by(**kwargs)
                .order_by(ModelVersion.created_at)
                .all()
            )


    def add_version(self,
                    project_id: int,
                    model_type: str,
                    model_file: Path|str | None = None,
                    data_file: Path|str | None = None,
                    params: dict | None = None,
                    note: str = "",
                    parent_id: int | None = None,
                    ) -> ModelVersion:
        with self.db.session() as session:
            project = session.get(Project, project_id)
            if project is None:
                raise ValueError(f"Project {project} not found")
            data_sr =   None
            model_sr =   None
            if model_file:
                model_sr  = _create_storage(session,model_file )
            if data_file:
                data_sr  = _create_storage(session,data_file)

            if parent_id is None:
                parent_id = project.active_model_version_id

            version = ModelVersion(
                project_id=project_id,
                model_type=model_type,
                model_storage_ref_id = (model_sr.id  if model_sr else None),

                data_storage_ref_id = (data_sr.id if data_sr else None),

                params_json=params,
                note=note,
                parent_id=parent_id,
            )
            session.add(version)
            session.flush()
            project.active_model_version_id = version.id
            # event = Event(
            #     entity_type="data_version",
            #     entity_id=version.id,
            #     action="register",
            #     payload_json=json.dumps({"path": path}),
            # )
            # session.add(event)
            session.commit()
            return version


class LineageService:
    """Query lineage chains for data and model versions."""

    def __init__(self, db: Database):
        self.db = db

    def _trace(self, cls, start_id: int):
        with self.db.session() as session:
            node = session.get(cls, start_id)
            chain = []
            while node is not None:
                chain.append(node)
                if node.parent_id is None:
                    break
                node = session.get(cls, node.parent_id)
            return chain


    def model_lineage(self, model_version_id: int):
        return self._trace(ModelVersion, model_version_id)
