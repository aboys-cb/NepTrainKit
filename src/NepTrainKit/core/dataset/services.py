"""Service layer for registering datasets and models."""
from __future__ import annotations

import datetime
import hashlib
import shutil
from decimal import Decimal
from pathlib import Path

from typing import Iterable
from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy import select, func, distinct
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import aliased

from NepTrainKit.utils import sha256_file
from NepTrainKit.paths import get_user_config_path, ensure_directory

from .database import Database
from .models import (

    Tag,ModelVersionTag,
    Project,
    ModelVersion,
    StorageRef,
)
from NepTrainKit.core.utils import read_nep_in, read_nep_out_file, get_rmse, get_xyz_nframe


@dataclass
class ProjectItem:
    """Hierarchical project node used in dataset navigation."""
    project_id: int
    name:str
    parent_id:int
    model_num:int
    notes:str
    children:list[ProjectItem]=field( default_factory=list)


@dataclass
class TagItem:
    """Lightweight tag representation returned to callers."""
    name:str
    tag_id:int
    color:str
    notes:str

@dataclass
class ModelItem:
    """Aggregated metadata describing a stored NEP model."""
    model_id: int
    model_type:str
    name:str
    data_size:int
    energy:float
    force:float
    virial:virial
    parent_id:int
    project_id:int
    model_path:str

    notes:str
    created_at:datetime.datetime
    tags:list[TagItem]=field( default_factory=list)

    children:list[ModelItem]=field( default_factory=list)


@dataclass
class TagService:
    """Service class for tag CRUD operations."""
    db: Database

    def create_tag(self, name: str, color: str = "#D1E9FF", notes: str = "") -> TagItem | None:
        """Create a new tag. Returns TagItem or None if name exists."""
        n = name.strip()
        if not n:
            return None
        with self.db.session() as session:
            tag = Tag(name=n, color=color, notes=notes)
            session.add(tag)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return None
            return TagItem(name=tag.name, tag_id=tag.id, color=tag.color, notes=tag.notes)

    def update_tag(self, tag_id: int, **kwargs) -> None:
        """Update tag fields by id."""
        if not kwargs:
            return
        with self.db.session() as session:
            session.query(Tag).filter_by(id=tag_id).update(kwargs)
            session.commit()

    def remove_tag(self, tag_id: int) -> None:
        """Remove tag by id."""
        with self.db.session() as session:
            session.query(Tag).filter_by(id=tag_id).delete()
            session.commit()

    def get_tags(self, **kwargs) -> list[TagItem]:
        """List tags matching filters."""
        with self.db.session() as session:
            tags = (
                session.query(Tag)
                .filter_by(**kwargs)
                .order_by(Tag.created_at)
                .all()
            )
            return [TagItem(name=t.name, tag_id=t.id, color=t.color, notes=t.notes) for t in tags]

def query_set_to_dict(func):
    """Decorator that converts SQLAlchemy ORM objects to plain dicts."""
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
    """Return size, modification time, and existence flag for a path."""
    try:
        st = p.stat()
        return st.st_size, datetime.datetime.fromtimestamp(st.st_mtime), True
    except FileNotFoundError:
        return None, None, False

def _create_storage(session, path: str, scheme: str = "file") -> StorageRef:
    """Persist a storage reference for the given filesystem path."""
    p = Path(path).expanduser().resolve()
    size, mtime, ok = _stat_path(p)

    if scheme == "file":
        uri =  str(p)
        content_hash = sha256_file(p)
    else:
        # Persist content-addressable storage copy under the user cache directory.
        content_hash = sha256_file(p)
        cas_dir = ensure_directory(get_user_config_path() / 'cas')
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
    """Manage creation and traversal of project hierarchies."""
    db: Database

    def _build_tree(self, node: Project) -> ProjectItem:
        """Recursively convert ORM project records into ProjectItem dataclasses."""
        item = ProjectItem(
            project_id=node.id,
            name=node.name,
            notes=node.notes,
            parent_id=node.parent_id,
            model_num=node.model_num,
        )
        for child in node.children:
            child_item = self._build_tree(child)
            item.model_num += child_item.model_num
            item.children.append(child_item)
        return item

    # @query_set_to_dict
    def search_projects(self,**kwargs) -> list[ProjectItem]:
        """Return ProjectItem nodes matching the provided ORM filters."""
        with self.db.session() as session:
            projects= (session.query(Project)
                        .filter_by(**kwargs)
                        .order_by(Project.created_at)
                        .all())


            return [self._build_tree(p) for p in projects]


    def get_project_by_id(self, project_id: int) -> ProjectItem|None:
        """Return a single project by id or ``None`` when not found."""
        projects = self.search_projects(id=project_id)
        if len(projects) == 0:
            return None
        return projects[0]

    def create_project(self,  name: str, notes: str = "", parent_id: int = None   ) -> Project|None:
        """Create a project record and return the ORM instance."""
        try:
            with self.db.session() as session:
                family = Project(name=name, notes=notes,parent_id=parent_id )
                session.add(family)
                session.commit()
                return family
        except Exception as e:
            logger.error(e)
            return None
    def modify_project(self, project_id: int, name: str, notes: str = "", parent_id: int = None ) -> Project|None:
        """Update project attributes and persist the changes."""


        with self.db.session() as session:

            session.query(Project).filter_by(id=project_id).update(
                {"name":name,"notes":notes,"parent_id":parent_id}
            )

            session.commit()


    def remove_project(self, project_id: int) -> Project|None:
        """Delete a project by id."""
        with self.db.session() as session:
            session.query(Project).filter_by(id=project_id).delete()


            session.commit()


@dataclass
class ModelService:
    """Query and mutate model metadata with advanced filters."""
    db: Database

    def _project_ids_with_descendants_subq(self, session, root_ids: list[int]):
        """Return a subquery selecting project ids including all descendants."""
        P = aliased(Project)

        # Support both single and multiple root project ids
        roots = select(Project.id).where(Project.id.in_(root_ids)).cte("roots")

        # Recursive CTE to collect descendant projects
        desc = select(roots.c.id.label("id")).cte("desc", recursive=True)
        desc = desc.union_all(
            select(P.id).where(P.parent_id == desc.c.id)
        )
        # Return the descendant ids subquery for use in IN clauses
        return select(desc.c.id)
    def search_models_advanced(
            self,
            *,
            project_id: int | list[int] | None = None,
            include_descendants: bool = True,
            parent_id: int | None = None,
            name_contains: str | None = None,
            notes_contains: str | None = None,
            model_type: str | None = None,
            tags_all: Iterable[str] | None = None,
            tags_any: Iterable[str] | None = None,
            tags_none: Iterable[str] | None = None,
            order_by_created_asc: bool = True,
            limit: int | None = None,
            offset: int | None = None,
    ) -> list[ModelItem]:
        """Advanced filtering interface for model listings."""
        names_all = self._normalize_names(tags_all) if tags_all else []
        names_any = self._normalize_names(tags_any) if tags_any else []
        names_none = self._normalize_names(tags_none) if tags_none else []

        with self.db.session() as session:
            mv = ModelVersion
            tg = Tag
            mvt = ModelVersionTag

            q = session.query(mv)

            # Filter by project roots and optionally include descendants
            if project_id is not None:
                if isinstance(project_id, int):
                    roots = [project_id]
                else:
                    roots = list(project_id)
                if include_descendants:
                    subq = self._project_ids_with_descendants_subq(session, roots)
                    q = q.filter(mv.project_id.in_(subq))
                else:
                    q = q.filter(mv.project_id.in_(roots))

            # Optionally filter by parent model id
            if parent_id is not None:
                if parent_id<0:
                    q = q.filter(mv.parent_id == None)

                else:

                    q = q.filter(mv.parent_id == parent_id)
            if model_type:
                q = q.filter(mv.model_type == model_type)
            if name_contains:
                q = q.filter(mv.name.like(f"%{name_contains}%"))
            if notes_contains:
                q = q.filter(mv.notes.like(f"%{notes_contains}%"))

            # Tag filter: match any of the provided tags
            if names_any:
                q = q.join(mv.tags).filter(tg.name.in_(names_any)).distinct()

            # Tag filter: require all provided tags
            if names_all:
                q = (q.join(mv.tags)
                     .filter(tg.name.in_(names_all))
                     .group_by(mv.id)
                     .having(func.count(distinct(tg.id)) == len(names_all)))

            # Tag filter: exclude models containing forbidden tags
            if names_none:
                sub_exists = (
                    session.query(mvt.model_version_id)
                    .join(tg, tg.id == mvt.tag_id)
                    .filter(mvt.model_version_id == mv.id,
                            tg.name.in_(names_none))
                    .exists()
                )
                q = q.filter(~sub_exists)

            q = q.order_by(mv.created_at.asc() if order_by_created_asc else mv.created_at.desc())
            if offset: q = q.offset(offset)
            if limit:  q = q.limit(limit)

            models = q.all()
            return [self._build_tree(m) for m in models]

    # --- Tag helpers ---
    @staticmethod
    def _normalize_names(names: Iterable[str]) -> list[str]:
        """Canonicalise tag names by trimming, deduplicating, and normalising case."""
        # Trim whitespace, normalise case, and remove duplicates while preserving order
        seen = set()
        result = []
        for n in names or []:
            if not n:
                continue
            s = n.strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(s)
        return result

    # --- Tag helpers ---
    def _get_or_create_tags(self, names: Iterable[str]) -> list["Tag"]:
        """Fetch existing tags or create new ones for the provided names."""
        names = self._normalize_names(names)
        if not names:
            return []
        with self.db.session() as session:

            # Query existing rows; Tag.name uses a NOCASE unique constraint.
            existing = session.query(Tag).where(Tag.name.in_(names)).all()
            have_lower = {t.name.lower(): t for t in existing}

            # Create any tags that do not already exist.
            to_create = [Tag(name=n) for n in names if n.lower() not in have_lower]
            if to_create:
                session.add_all(to_create)

                session.flush()


        return list(have_lower.values())
    def _get_or_create_tags_simple(self, names: Iterable[str], session) -> list["Tag"]:
        """Simplified helper that ensures tags exist within the current session."""
        names = self._normalize_names(names)
        if not names:
            return []
        # Step 1: query all existing tags


        existing = session.query(Tag).where(Tag.name.in_(names)).all()

        have_lower = {t.name.lower(): t for t in existing}
        # Step 2: create any missing tags within the same session
        to_create = [Tag(name=n) for n in names if n.lower() not in have_lower]
        if to_create:
            session.add_all(to_create)
            session.flush()  # obtain generated primary keys
            session.commit()
        return existing + to_create
    def _build_tree(self, node: ModelVersion) -> ModelItem:
        """Recursively map ORM model hierarchy into ``ModelItem`` objects."""

        item = ModelItem(
            model_id=node.id,
            name=node.name,
            notes=node.notes,
            parent_id=node.parent_id,
            project_id=node.project_id,
            model_type=node.model_type,
            data_size=node.data_size,
            energy=node.energy,
            force=node.force,
            virial=node.virial,
            model_path=node.model_path,
            # train_params=node.train_params,
            # calc_params=node.calc_params,
            created_at=node.created_at,


        )
        for tag in node.tags:
            item.tags.append(TagItem(name=tag.name, tag_id=tag.id,notes=tag.notes,color=tag.color))
        # Recursive CTE to collect descendant projects
        for child in node.children:
            item.children.append(self._build_tree(child))
        return item


    def get_models_by_project_id(self, project_id: int) -> list[ModelItem]:
        return self.search_models_advanced(project_id=project_id,parent_id=-1)


    def add_version_from_path(self,
                              model_type:str,
                              name:str,
                              path: Path,
                              project_id: int,
                              notes: str = "",
                              tags: list[str] | None = None,

                              parent_id: int | None = None,) -> ModelVersion:
        if model_type.upper()=="NEP":
            model_file=path.joinpath("nep.txt")
            data_file=path.joinpath("train.xyz")
            data_size=get_xyz_nframe(data_file)
            train_params=read_nep_in(path.joinpath("nep.in"))
            energy_array=read_nep_out_file(path.joinpath("energy_train.out"))
            energy = get_rmse(energy_array[:,0],energy_array[:,1])*1000
            force_array=read_nep_out_file(path.joinpath("force_train.out"))
            force = get_rmse(force_array[:,:3],force_array[:,3:])*1000
            virial_array=read_nep_out_file(path.joinpath("virial_train.out"))
            virial = get_rmse(virial_array[:,:6],virial_array[:,6:])*1000
            calc_params={}



            return self.add_version(
                project_id=project_id,
                name=name,
                model_type=model_type.upper(),
                # model_file=model_file,
                # data_file=data_file,
                model_path=path,
                data_size=data_size,
                energy=energy,
                force=force,
                virial=virial,
                # train_params=train_params,
                # calc_params=calc_params,
                notes=notes,
                tags=tags,
                parent_id=parent_id
            )


    def add_version(self,
                    project_id: int,
                    name:str,
                    model_type: str,
                    # model_file: Path|str | None = None,
                    # data_file: Path|str | None = None,
                    model_path: Path|str | None = None,
                    data_size: int | None = None,
                    # train_params: dict | None = None,
                    # calc_params: dict | None = None,
                    energy:float|None = None,
                    force:float|None = None,
                    virial:float|None = None,
                    tags:list[str] | None = None,
                    notes: str = "",
                    parent_id: int | None = None,
                    id:int|None = None
                    ) -> ModelVersion:

        with self.db.session() as session:
            tag_objs = self._get_or_create_tags_simple(tags or [],session)

            project = session.get(Project, project_id)
            if project is None:
                raise ValueError(f"Project {project} not found")
            # data_sr =   None
            # model_sr =   None
            # if model_file:
            #     model_sr  = _create_storage(session,model_file )
            # if data_file:
            #     data_sr  = _create_storage(session,data_file)

            # if parent_id is None:
            #     parent_id = project.active_model_version_id
            if isinstance(model_path, Path):
                model_path=model_path.as_posix()
            version = ModelVersion(
                id=id,
                project_id=project_id,
                name=name,
                model_type=model_type,
                model_path=model_path,
                # model_storage_ref_id = (model_sr.id  if model_sr else None),

                # data_storage_ref_id = (data_sr.id if data_sr else None),
                data_size=data_size,
                # train_params=train_params,

                # calc_params=calc_params,
                energy=energy,
                force=force,
                virial=virial if virial<10000 else 0,
                notes=notes,
                parent_id=parent_id,
                tags=set(tag_objs)
            )
            session.add(version)
            session.flush()


            # project.active_model_version_id = version.id
            # event = Event(
            #     entity_type="data_version",
            #     entity_id=version.id,
            #     action="register",
            #     payload_json=json.dumps({"path": path}),
            # )
            # session.add(event)
            session.commit()
            return version
    def modify_model(self,model_id:int,**kwargs):
        with self.db.session() as session:
            mv = session.get(ModelVersion, model_id)  # Load the model version
            if not mv:
                raise ValueError(f"ModelVersion {model_id} not found")
            tag_objs = self._get_or_create_tags_simple(kwargs.pop("tags", []) or [],session)

            # Update basic scalar fields
            for key, value in kwargs.items():
                setattr(mv, key, value)

            # Update relationship fields
            if tag_objs:
                mv.tags = set(tag_objs)
            session.commit()
    def remove_model(self, model_id: int) :
        with self.db.session() as session:
            session.query(ModelVersion).filter_by(id=model_id).delete()
            session.commit()

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
