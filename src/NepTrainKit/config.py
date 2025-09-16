import os
import platform
import shutil
from typing import Any

from sqlalchemy import (
    create_engine, MetaData, Table, Column, String, select, update, inspect
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from NepTrainKit import module_path, get_user_config_path


class Config:
    """
使用数据库保存软件配置
    """
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

        db_file = os.path.join(user_config_path, "config.sqlite")
        if not os.path.exists(db_file):
            if not os.path.exists(user_config_path):
                os.makedirs(user_config_path)
            shutil.copy(os.path.join(module_path, 'Config/config.sqlite'), db_file)

        # Initialize SQLAlchemy engine for SQLite
        url = f"sqlite:///{db_file}"
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
    def get_path(cls,section="setting", option="last_path")->str:
        """
        获取上一次文件交互的路径
        :param section:
        :param option:
        :return:
        """
        path = cls.get(section, option)
        if path:
            if os.path.exists(path):
                return path
        return "./"

    @classmethod
    def has_option(cls,section, option) ->bool:
        if cls.get(section,option) is not None:
            return True
        return False

    @classmethod
    def getboolean(cls, section, option, fallback=None)->bool|None:
        v = cls.get(section, option,fallback)
        try:
            v = eval(v)
        except:
            v = None
        if v is None:
            return fallback
        return v

    @classmethod
    def getint(cls, section, option, fallback=None) ->int|None:
        v = cls.get(section, option,fallback)

        try:
            v = int(v)
        except:

            v = None
        if v is None:
            return fallback

        return v
    @classmethod
    def getfloat(cls,section,option,fallback=None)->float|None:
        v=    cls.get(section,option,fallback)

        try:
            v=float(v)
        except:

            v=None
        if v is None:
            return fallback
        return v
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
