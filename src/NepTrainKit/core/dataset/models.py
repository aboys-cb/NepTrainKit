from __future__ import annotations
import datetime as dt
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, JSON, Boolean


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")

    # 允许为空；指向“当前激活”的模型版本
    active_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )

    # 关键：指定 foreign_keys = ModelVersion.project_id
    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="project",
        foreign_keys="ModelVersion.project_id",
        cascade="all, delete-orphan",
    )

    # 单独定义一个关系指向“激活版本”，避免歧义；uselist=False 保证一对一语义
    active_model_version: Mapped["ModelVersion | None"] = relationship(
        "ModelVersion",
        foreign_keys=[active_model_version_id],
        uselist=False,
        post_update=True,  # 避免循环依赖的 UPDATE 次序问题
    )


class StorageRef(Base):
    __tablename__ = "storage_refs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scheme: Mapped[str] = mapped_column(String(10), nullable=False)  # file:// | cas://
    uri: Mapped[str] = mapped_column(Text, nullable=False)           # 路径或CAS相对路径
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    size: Mapped[int | None] = mapped_column(Integer)
    mtime: Mapped[dt.datetime | None] = mapped_column(DateTime)
    exist_flag: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # 关键：这是到 Project 的“所属”外键
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)

    model_type: Mapped[str] = mapped_column(String(20))  # NEP | DeepMD | ...
    model_storage_ref_id: Mapped[int] = mapped_column(ForeignKey("storage_refs.id"))


    data_storage_ref_id: Mapped[int | None] = mapped_column(ForeignKey("storage_refs.id"))


    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    note: Mapped[str] = mapped_column(Text, default="")

    # 自引用（版本继承/分支）
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    # 关键：在子表端也显式指定 foreign_keys = [project_id]
    project: Mapped["Project"] = relationship(
        back_populates="model_versions",
        foreign_keys=[project_id],
    )

    parent: Mapped["ModelVersion | None"] = relationship(
        remote_side=[id], backref="children"
    )

    model_storage: Mapped["StorageRef"] = relationship(
        foreign_keys=[model_storage_ref_id]
    )
    data_storage: Mapped["StorageRef | None"] = relationship(
        foreign_keys=[data_storage_ref_id]
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
