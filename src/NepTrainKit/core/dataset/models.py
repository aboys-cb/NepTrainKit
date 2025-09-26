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
    data_size: Mapped[int] = mapped_column(Integer)
    energy: Mapped[float] = mapped_column(Float)
    force: Mapped[float] = mapped_column(Float)
    virial: Mapped[float] = mapped_column(Float)
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        index=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
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
