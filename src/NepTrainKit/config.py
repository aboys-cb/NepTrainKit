import os
import platform
import shutil
from typing import Any
from pathlib import Path

from sqlalchemy import (
    create_engine, MetaData, Table, Column, String, select, update, inspect
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from NepTrainKit import module_path
from NepTrainKit.paths import get_user_config_path



class Config:

    _instance = None
    init_flag = False

    def __new__(cls, *args):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        if Config.init_flag:
            return
        Config.init_flag = True
        self.connect_db()

    def connect_db(self):
        user_config_path = get_user_config_path()

        db_file = user_config_path / "config.sqlite"
        if not db_file.exists():
            user_config_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(module_path / "Config/config.sqlite", db_file)

        # Initialize SQLAlchemy engine for SQLite
        url = f"sqlite:///{db_file.as_posix()}"
        # check_same_thread=False to be safe with GUI contexts
        self.engine: Engine = create_engine(url, future=True)

        # Ensure the table exists (reflect if present, otherwise create)
        self._metadata = MetaData()
        inspector = inspect(self.engine)
        if inspector.has_table('config'):
            self._config_table = Table('config', self._metadata, autoload_with=self.engine)
        else:
            self._config_table = Table(
                'config', self._metadata,
                Column('section', String, primary_key=True),
                Column('option', String, primary_key=True),
                Column('value', String)
            )
            self._metadata.create_all(self.engine)

    @classmethod
    def get_path(cls, section="setting", option="last_path") -> Path:
        """Return the last-used path as a :class:`Path` instance."""
        raw = cls.get(section, option)
        if raw:
            candidate = Path(raw)
            if candidate.exists():
                return candidate
        return Path("./")

    @classmethod
    def has_option(cls,section, option) ->bool:
        if cls.get(section,option) is not None:
            return True
        return False

    @classmethod
    def getboolean(cls, section, option, fallback=None)->bool|None:
        raw = cls.get(section, option, fallback)
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return fallback
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in {"1", "true", "yes", "on"}:
                return True
            if s in {"0", "false", "no", "off"}:
                return False
        return fallback

    @classmethod
    def getint(cls, section, option, fallback=None) ->int|None:
        raw = cls.get(section, option, fallback)
        if isinstance(raw, int):
            return raw
        if raw is None:
            return fallback
        try:
            return int(str(raw).strip())
        except Exception:
            return fallback
    @classmethod
    def getfloat(cls,section,option,fallback=None)->float|None:
        raw = cls.get(section, option, fallback)
        if isinstance(raw, float):
            return raw
        if isinstance(raw, int):
            return float(raw)
        if raw is None:
            return fallback
        try:
            return float(str(raw).strip())
        except Exception:
            return fallback
    @classmethod
    def get(cls,section,option,fallback=None)->Any:
        try:
            cfg = cls._instance
            table = cfg._config_table
            with cfg.engine.begin() as conn:
                stmt = select(table.c.value).where(
                    table.c.section == section,
                    table.c.option == option
                ).limit(1)
                result = conn.execute(stmt).scalar_one_or_none()
            if result is None:
                if fallback is not None:
                    cls.set(section, option, fallback)
                return fallback
            return result
        except SQLAlchemyError:
            # Fallback behavior in case of unexpected DB errors
            return fallback

    @classmethod
    def set(cls,section,option,value):
        if option == "theme":
            cls.theme = value
        cfg = cls._instance
        table = cfg._config_table
        val_str = str(value)
        with cfg.engine.begin() as conn:
            # Try update first; if no row affected, insert
            upd = (
                update(table)
                .where(table.c.section == section, table.c.option == option)
                .values(value=val_str)
            )
            res = conn.execute(upd)
            if res.rowcount == 0:
                ins = table.insert().values(section=section, option=option, value=val_str)
                conn.execute(ins)

    @classmethod
    def update_section(cls,old,new):
        cfg = cls._instance
        table = cfg._config_table
        with cfg.engine.begin() as conn:
            stmt = update(table).where(table.c.section == old).values(section=new)
            conn.execute(stmt)
Config()


