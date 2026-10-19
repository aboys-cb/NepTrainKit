"""SQLAlchemy ORM models for projects, models, tags, and events.
Notes
-----
Business logic is implemented in the service layer. These models focus on
structure and relationships.
Examples
--------
>>> # Tables are created by Database() in database.py
"""
from __future__ import annotations
import datetime as dt
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, column_property, backref
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, JSON, Boolean, Float, select, func, CTE, Index
class Base(DeclarativeBase):
    pass
class StorageRef(Base):
    __tablename__ = "storage_refs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scheme: Mapped[str] = mapped_column(String(10), nullable=False)  # file:// | cas://
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    size: Mapped[int | None] = mapped_column(Integer)
    mtime: Mapped[dt.datetime | None] = mapped_column(DateTime)
    exist_flag: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
class Tag(Base):
    __tablename__ = "tags"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(
        String(64, collation="NOCASE"), nullable=False, unique=True, index=True
    )
    color: Mapped[str] = mapped_column(String(9), default="#D1E9FF")
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    models: Mapped[set["ModelVersion"]] = relationship(
        "ModelVersion",
        secondary="model_version_tags",
        back_populates="tags",
        collection_class=set,
        lazy="selectin",
    )
class ModelVersionTag(Base):
    __tablename__ = "model_version_tags"
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    __table_args__ = (
        Index("ix_mvt_tag_id", "tag_id"),
    )
class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        index=True
    )
    model_type: Mapped[str] = mapped_column(String(20))  # NEP | DeepMD | ...
    # Training metrics are unknown for models captured from the viewer.
    data_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    energy: Mapped[float | None] = mapped_column(Float, nullable=True)
    force: Mapped[float | None] = mapped_column(Float, nullable=True)
    virial: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        index=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    # Validation library: immutable snapshot (library object) of the model file.
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    model_kind: Mapped[str] = mapped_column(String(40), default="")  # nep4, nep4_zbl, nep4_spin ...
    elements: Mapped[list] = mapped_column(JSON, default=list)
    source_path: Mapped[str] = mapped_column(Text, default="")
    dataset_ref: Mapped[dict] = mapped_column(JSON, default=dict)  # FileRef of the training data
    starred: Mapped[bool] = mapped_column(Boolean, default=False)
    auto_captured: Mapped[bool] = mapped_column(Boolean, default=False)
    last_seen_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    project: Mapped["Project"] = relationship(
        back_populates="model_versions",
        foreign_keys=[project_id],
        passive_deletes=True,
    )
    parent: Mapped["ModelVersion | None"] = relationship(
        remote_side=[id],
        backref=backref(
            "children",
            cascade="all, delete-orphan",
            passive_deletes=True
        ),
        single_parent=True,
        passive_deletes=True
    )
    tags: Mapped[set["Tag"]] = relationship(
        "Tag",
        secondary="model_version_tags",
        back_populates="models",
        collection_class=set,
        lazy="selectin",
        passive_deletes=True
    )
class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    # active_model_version_id: Mapped[int | None] = mapped_column(
    #     ForeignKey("model_versions.id", ondelete="SET NULL"),
    #     nullable=True
    # )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        index=True
    )
    parent: Mapped["Project | None"] = relationship(
        remote_side=[id],
        backref=backref(
            "children",
            cascade="all, delete-orphan",
            passive_deletes=True
        ),
        single_parent=True,
        passive_deletes=True
    )
    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="project",
        foreign_keys="ModelVersion.project_id",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    # active_model_version: Mapped["ModelVersion | None"] = relationship(
    #     "ModelVersion",
    #     foreign_keys=[active_model_version_id],
    #     uselist=False,
    #     post_update=True,
    #     passive_deletes=True
    # )
    model_num: Mapped[int] = column_property(
        select(func.count(ModelVersion.id))
        .where(ModelVersion.project_id == id)
        .scalar_subquery()
    )
class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(50))
    entity_id: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(50))
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    actor: Mapped[str] = mapped_column(String(80), default="system")


class BenchmarkSet(Base):
    """A sealed reference data set evaluated by one validation protocol."""

    __tablename__ = "benchmark_sets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    protocol: Mapped[str] = mapped_column(String(40), index=True)
    protocol_version: Mapped[int] = mapped_column(Integer)
    object_hash: Mapped[str] = mapped_column(String(64))  # library object holding the frames
    source_ref: Mapped[dict] = mapped_column(JSON, default=dict)  # FileRef of the imported file
    elements: Mapped[list] = mapped_column(JSON, default=list)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class BenchmarkSuite(Base):
    """A named group of benchmark sets that is evaluated together."""

    __tablename__ = "benchmark_suites"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    baseline_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    baseline_note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class BenchmarkSuiteSet(Base):
    __tablename__ = "benchmark_suite_sets"
    suite_id: Mapped[int] = mapped_column(
        ForeignKey("benchmark_suites.id", ondelete="CASCADE"), primary_key=True
    )
    set_id: Mapped[str] = mapped_column(
        ForeignKey("benchmark_sets.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(Integer, default=0)


class BenchmarkRun(Base):
    """One evaluation of a model snapshot on a benchmark set; failures are kept."""

    __tablename__ = "benchmark_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), index=True
    )
    set_id: Mapped[str] = mapped_column(
        ForeignKey("benchmark_sets.id", ondelete="CASCADE"), index=True
    )
    protocol_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))  # succeeded | failed
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    result_path: Mapped[str] = mapped_column(Text, default="")  # relative to the library root
    error: Mapped[str] = mapped_column(Text, default="")
    environment: Mapped[dict] = mapped_column(JSON, default=dict)
    elapsed_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
