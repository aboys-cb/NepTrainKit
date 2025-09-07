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
    uri: Mapped[str] = mapped_column(Text, nullable=False)           # 路径或CAS相对路径
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    size: Mapped[int | None] = mapped_column(Integer)
    mtime: Mapped[dt.datetime | None] = mapped_column(DateTime)
    exist_flag: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # SQLite 下用 NOCASE 保证 name 唯一且大小写不敏感
    name: Mapped[str] = mapped_column(
        String(64, collation="NOCASE"), nullable=False, unique=True, index=True
    )
    color: Mapped[str] = mapped_column(String(9), default="#D1E9FF")  # 可选：HEX 颜色
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    # 反向关系
    models: Mapped[set["ModelVersion"]] = relationship(
        "ModelVersion",
        secondary="model_version_tags",
        back_populates="tags",
        collection_class=set,      # 防止重复插入
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

    # 额外索引：提高按 tag 查模型的性能
    __table_args__ = (
        Index("ix_mvt_tag_id", "tag_id"),
    )
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))

    # 归属项目 → 删项目时，模型级联删除
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

    # 自引用（版本继承/分支）→ 删父版本时，子版本级联删除
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"),
        index=True
    )

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    # 关系：归属项目
    project: Mapped["Project"] = relationship(
        back_populates="model_versions",
        foreign_keys=[project_id],
        passive_deletes=True,      # 交给 DB ondelete 处理
    )

    # 关系：父/子版本树
    parent: Mapped["ModelVersion | None"] = relationship(
        remote_side=[id],
        backref=backref(
            "children",
            cascade="all, delete-orphan",
            passive_deletes=True
        ),
        single_parent=True,        # delete-orphan 需要
        passive_deletes=True
    )

    # 多对多标签（这里不动）
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

    # 当前激活模型 → 被删时置空，避免 FK 失败
    # active_model_version_id: Mapped[int | None] = mapped_column(
    #     ForeignKey("model_versions.id", ondelete="SET NULL"),
    #     nullable=True
    # )

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )

    # 自引用项目树 → 删父项目时子项目级联删除
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        index=True
    )

    # 项目树关系
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

    # 项目下的所有模型 → 删项目时“模型”也删（双保险：ORM + DB）
    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="project",
        foreign_keys="ModelVersion.project_id",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

    # 如需：激活版本关系（可保持注释，或启用）
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
