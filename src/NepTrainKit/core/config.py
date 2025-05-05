import os
import platform
import shutil
from PySide6.QtSql import QSqlDatabase, QSqlQuery
from NepTrainKit import module_path,get_user_config_path


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
        self.db = QSqlDatabase.addDatabase("QSQLITE","config")

        user_config_path = get_user_config_path()

        if not os.path.exists(f"{user_config_path}/config.sqlite"):
            if not os.path.exists(user_config_path):
                os.makedirs(user_config_path)

            shutil.copy(os.path.join(module_path,'Config/config.sqlite'),f"{user_config_path}/config.sqlite")

        self.db.setDatabaseName(f"{user_config_path}/config.sqlite")

        self.db.open()

    @classmethod
    def get_path(self,section="setting", option="last_path"):
        """
        获取上一次文件交互的路径
        :param section:
        :param option:
        :return:
        """
        path = self.get(section, option)
        if path:
            if os.path.exists(path):
                return path
        return "./"

    @classmethod
    def has_option(self,section, option):
        if self.get(section,option) is not None:
            return True
        return False

    @classmethod
    def getboolean(self, section, option, fallback=None):
        v = self.get(section, option,fallback)
        try:
            v = eval(v)
        except:
            v = None
        if v is None:
            return fallback
        return v

    @classmethod
    def getint(self, section, option, fallback=None):
        v = self.get(section, option)

        try:
            v = int(v)
        except:

            v = None
        if v is None:
            return fallback

        return v
    @classmethod
    def getfloat(self,section,option,fallback=None):
        v=    self.get(section,option)

        try:
            v=float(v)
        except:

            v=None
        if v is None:
            return fallback
        return v
    @classmethod
    def get(self,section,option,fallback=None):
        query = QSqlQuery(self._instance.db )
        result=query.exec(f"""SELECT value FROM "config" where config.option='{option}' and config.section='{section}';""")

        query.next()
        first= query.value(0)
        if first  is None:
            return fallback
        return first

    @classmethod
    def set(self,section,option,value):
        if option=="theme":
            self.theme=value
        query = QSqlQuery(self._instance.db)
        result=query.exec(f"""INSERT OR REPLACE INTO  "main"."config"("section", "option", "value") VALUES ('{section}', '{option}', '{value}')""")

    @classmethod
    def update_section(self,old,new):
        query = QSqlQuery(self._instance.db)
        result=query.exec(f"""UPDATE  "main"."config" set   section='{new}' where section='{old}'""")

