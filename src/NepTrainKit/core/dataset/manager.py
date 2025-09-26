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
import random
import traceback
import uuid
from pathlib import Path

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

    def gen_test(self):
        try:
            project = self.project_service.create_project(f"Test{random.random()}","测试1")
            project = self.project_service.create_project(f"Test{random.random()}","测试1",parent_id=project.id)

            self.model_service.add_version_from_path(
                name="第1代",
                project_id=project.id,
            model_type="NEP",
            path=Path(r"D:\Desktop\nep-new"),


            notes  = "",
            parent_id  = None,


            )
            model2 = self.model_service.add_version_from_path(
                name="第2代",
                project_id=project.id,
                model_type="NEP",
                path=Path(r"D:\Desktop\nep-new"),
                notes  = "",
                    tags=["NEP", "GPUMDs" ],

                parent_id  = None,
            )
            model = self.model_service.add_version_from_path(
            name="第3代",
            project_id=project.id,
            model_type="NEP",
            path=Path(r"D:\Desktop\nep-new"),
            notes  = "",
                tags=["NEP","GPUMD","nep","Cs"],

            parent_id  = model2.id,
            )
            model = self.model_service.add_version_from_path(
                name="第4代",
                project_id=project.id,
                model_type="NEP",
                path=Path(r"D:\Desktop\nep-new"),
                notes="",
                parent_id=model2.id,
                tags=["NEP"]
            )

        except Exception as e:
            print(traceback.format_exc())

    def gen_nep_data_git(self):
        if not self.db.first:
            return
        project = self.project_service.create_project("NEP-Data","NEP官方的公开数据集,地址：https://gitlab.com/brucefan1983/nep-data")

        models=[
    {
        "name": "2021_Fan_PbTe_demo",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2021_Fan_PbTe_demo",
        "data_size": 325,
        "energy": 0.39,
        "force": 36.02,
        "virial": 446.38,
        "tags": [],
        "notes": "",
        "id": 0,
        "parent_id": None
    },
    {
        "name": "2021_Fan_Silicene",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2021_Fan_Silicene",
        "data_size": 914,
        "energy": 1.36,
        "force": 51.24,
        "virial": 10.76,
        "tags": [],
        "notes": "",
        "id": 1,
        "parent_id": None
    },
    {
        "name": "2022_Fan_C_GAP2017",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2022_Fan_C_GAP2017",
        "data_size": 4080,
        "energy": 40.95,
        "force": 684.65,
        "virial": 778.29,
        "tags": [],
        "notes": "",
        "id": 2,
        "parent_id": None
    },
    {
        "name": "2022_Fan_Si_GAP2018",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2022_Fan_Si_GAP2018",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 3,
        "parent_id": None
    },
    {
        "name": "2023_Dong_C60thermal",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Dong_C60thermal",
        "data_size": 152,
        "energy": 3.75,
        "force": 189.09,
        "virial": 15.44,
        "tags": [],
        "notes": "",
        "id": 4,
        "parent_id": None
    },
    {
        "name": "2023_Liang_SiO2",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Liang_SiO2",
        "data_size": 2348,
        "energy": 2.39,
        "force": 177.95,
        "virial": 24.24,
        "tags": [],
        "notes": "",
        "id": 5,
        "parent_id": None
    },
    {
        "name": "2023_Shi_CsPbX(X=Cl,Br,I)",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Shi_CsPbX(X=Cl,Br,I)",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 6,
        "parent_id": None
    },
    {
        "name": "2023_Xu_liquid_water",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Xu_liquid_water",
        "data_size": 1388,
        "energy": 0.94,
        "force": 75.93,
        "virial": 5.16,
        "tags": [],
        "notes": "",
        "id": 7,
        "parent_id": None
    },
    {
        "name": "2023_Ying_bilayer_graphene",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 8,
        "parent_id": None
    },
    {
        "name": "2023_Ying_C60mechanical",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_C60mechanical",
        "data_size": 240,
        "energy": 4.83,
        "force": 206.24,
        "virial": 19.06,
        "tags": [],
        "notes": "",
        "id": 9,
        "parent_id": None
    },
    {
        "name": "2023_Ying_MOFs",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_MOFs",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 10,
        "parent_id": None
    },
    {
        "name": "2023_Ying_Phosphorene",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_Phosphorene",
        "data_size": 2139,
        "energy": 5.0,
        "force": 125.52,
        "virial": 66.64,
        "tags": [],
        "notes": "",
        "id": 11,
        "parent_id": None
    },
    {
        "name": "2023_Zhang_HfO2",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Zhang_HfO2",
        "data_size": 3876,
        "energy": 4.62,
        "force": 199.36,
        "virial": 26.03,
        "tags": [],
        "notes": "",
        "id": 12,
        "parent_id": None
    },
    {
        "name": "2023_Zhao_PdCuNiP",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Zhao_PdCuNiP",
        "data_size": 9615,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 13,
        "parent_id": None
    },
    {
        "name": "2024_Dong_Si",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Dong_Si",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 14,
        "parent_id": None
    },
    {
        "name": "2024_Fan_C_GAP2020",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Fan_C_GAP2020",
        "data_size": 6088,
        "energy": 45.35,
        "force": 598.68,
        "virial": 631129864.14,
        "tags": [],
        "notes": "",
        "id": 15,
        "parent_id": None
    },
    {
        "name": "2024_Wang_C",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wang_C",
        "data_size": 6738,
        "energy": 36.17,
        "force": 642.06,
        "virial": 146.04,
        "tags": [],
        "notes": "",
        "id": 16,
        "parent_id": None
    },
    {
        "name": "2024_Wang_Ga2O3",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wang_Ga2O3",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 17,
        "parent_id": None
    },
    {
        "name": "2024_Wu_C_Si_GaAs_PbTe",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wu_C_Si_GaAs_PbTe",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 18,
        "parent_id": None
    },
    {
        "name": "2024_Wu_MoSe2-WSe2",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wu_MoSe2-WSe2",
        "data_size": 150,
        "energy": 0.15,
        "force": 23.21,
        "virial": 3.32,
        "tags": [],
        "notes": "",
        "id": 19,
        "parent_id": None
    },
    {
        "name": "2024_Ying_LiH",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Ying_LiH",
        "data_size": 447,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 20,
        "parent_id": None
    },
    {
        "name": "2025_Li_ICOF-10n-M",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2025_Li_ICOF-10n-M",
        "data_size": 1392,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 21,
        "parent_id": None
    },
    {
        "name": "2025_Wang_diamond-cBN",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2025_Wang_diamond-cBN",
        "data_size": 713,
        "energy": 6.95,
        "force": 135.33,
        "virial": 20.58,
        "tags": [],
        "notes": "",
        "id": 22,
        "parent_id": None
    },
    {
        "name": "3b",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2022_Fan_Si_GAP2018/3b",
        "data_size": 2474,
        "energy": 8.48,
        "force": 104.02,
        "virial": 472784168.18,
        "tags": [],
        "notes": "",
        "id": 23,
        "parent_id": 3
    },
    {
        "name": "3b4b",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2022_Fan_Si_GAP2018/3b4b",
        "data_size": 2474,
        "energy": 7.54,
        "force": 100.92,
        "virial": 472784171.67,
        "tags": [],
        "notes": "",
        "id": 24,
        "parent_id": 3
    },
    {
        "name": "3b4b5b",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2022_Fan_Si_GAP2018/3b4b5b",
        "data_size": 2474,
        "energy": 7.08,
        "force": 99.32,
        "virial": 472784179.49,
        "tags": [],
        "notes": "",
        "id": 25,
        "parent_id": 3
    },
    {
        "name": "CsPbBr",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Shi_CsPbX(X=Cl,Br,I)/CsPbBr",
        "data_size": 1200,
        "energy": 0.93,
        "force": 56.44,
        "virial": 12.69,
        "tags": [],
        "notes": "",
        "id": 26,
        "parent_id": 6
    },
    {
        "name": "CsPbCl",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Shi_CsPbX(X=Cl,Br,I)/CsPbCl",
        "data_size": 1200,
        "energy": 0.92,
        "force": 62.91,
        "virial": 12.7,
        "tags": [],
        "notes": "",
        "id": 27,
        "parent_id": 6
    },
    {
        "name": "CsPbI",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Shi_CsPbX(X=Cl,Br,I)/CsPbI",
        "data_size": 1200,
        "energy": 1.14,
        "force": 52.49,
        "virial": 12.24,
        "tags": [],
        "notes": "",
        "id": 28,
        "parent_id": 6
    },
    {
        "name": "PBE",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene/PBE",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 29,
        "parent_id": 8
    },
    {
        "name": "PBE_D3",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene/PBE_D3",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 30,
        "parent_id": 8
    },
    {
        "name": "HKUST-1",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_MOFs/HKUST-1",
        "data_size": 774,
        "energy": 0.56,
        "force": 54.56,
        "virial": 4.01,
        "tags": [],
        "notes": "",
        "id": 31,
        "parent_id": 10
    },
    {
        "name": "MOF-5",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_MOFs/MOF-5",
        "data_size": 774,
        "energy": 0.62,
        "force": 56.94,
        "virial": 4.26,
        "tags": [],
        "notes": "",
        "id": 32,
        "parent_id": 10
    },
    {
        "name": "ZIF-8",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_MOFs/ZIF-8",
        "data_size": 774,
        "energy": 0.56,
        "force": 53.32,
        "virial": 4.04,
        "tags": [],
        "notes": "",
        "id": 33,
        "parent_id": 10
    },
    {
        "name": "NEP-iteration-1",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Dong_Si/NEP-iteration-1",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 34,
        "parent_id": 14
    },
    {
        "name": "NEP-iteration-2",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Dong_Si/NEP-iteration-2",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 35,
        "parent_id": 14
    },
    {
        "name": "beta",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wang_Ga2O3/beta",
        "data_size": 1000,
        "energy": 0.33,
        "force": 37.2,
        "virial": 4.51,
        "tags": [],
        "notes": "",
        "id": 36,
        "parent_id": 17
    },
    {
        "name": "kappa",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wang_Ga2O3/kappa",
        "data_size": 999,
        "energy": 0.35,
        "force": 33.3,
        "virial": 4.6,
        "tags": [],
        "notes": "",
        "id": 37,
        "parent_id": 17
    },
    {
        "name": "GaAs",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wu_C_Si_GaAs_PbTe/GaAs",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 38,
        "parent_id": 18
    },
    {
        "name": "PbTe",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wu_C_Si_GaAs_PbTe/PbTe",
        "data_size": 124,
        "energy": 0.69,
        "force": 41.96,
        "virial": 5.79,
        "tags": [],
        "notes": "",
        "id": 39,
        "parent_id": 18
    },
    {
        "name": "Si",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wu_C_Si_GaAs_PbTe/Si",
        "data_size": 970,
        "energy": 0.48,
        "force": 25.85,
        "virial": 13.22,
        "tags": [],
        "notes": "",
        "id": 40,
        "parent_id": 18
    },
    {
        "name": "3.5_3.5",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene/PBE/3.5_3.5",
        "data_size": 3481,
        "energy": 3.7,
        "force": 53.2,
        "virial": 26.95,
        "tags": [],
        "notes": "",
        "id": 41,
        "parent_id": 29
    },
    {
        "name": "3_3",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene/PBE/3_3",
        "data_size": 3481,
        "energy": 8.16,
        "force": 68.23,
        "virial": 48.9,
        "tags": [],
        "notes": "",
        "id": 42,
        "parent_id": 29
    },
    {
        "name": "4.5_4.5",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene/PBE/4.5_4.5",
        "data_size": 3481,
        "energy": 1.12,
        "force": 46.71,
        "virial": 17.32,
        "tags": [],
        "notes": "",
        "id": 43,
        "parent_id": 29
    },
    {
        "name": "4_4",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene/PBE/4_4",
        "data_size": 3481,
        "energy": 1.21,
        "force": 48.84,
        "virial": 17.45,
        "tags": [],
        "notes": "",
        "id": 44,
        "parent_id": 29
    },
    {
        "name": "5_5",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene/PBE/5_5",
        "data_size": 3481,
        "energy": 1.13,
        "force": 49.18,
        "virial": 20.59,
        "tags": [],
        "notes": "",
        "id": 45,
        "parent_id": 29
    },
    {
        "name": "10_4.5",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene/PBE_D3/10_4.5",
        "data_size": 3481,
        "energy": 1.29,
        "force": 49.6,
        "virial": 19.93,
        "tags": [],
        "notes": "",
        "id": 46,
        "parent_id": 30
    },
    {
        "name": "6_4.5",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene/PBE_D3/6_4.5",
        "data_size": 3481,
        "energy": 1.28,
        "force": 47.45,
        "virial": 19.53,
        "tags": [],
        "notes": "",
        "id": 47,
        "parent_id": 30
    },
    {
        "name": "8_4.5",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2023_Ying_bilayer_graphene/PBE_D3/8_4.5",
        "data_size": 3481,
        "energy": 1.25,
        "force": 48.08,
        "virial": 18.19,
        "tags": [],
        "notes": "",
        "id": 48,
        "parent_id": 30
    },
    {
        "name": "predict-1",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Dong_Si/NEP-iteration-1/predict-1",
        "data_size": 100,
        "energy": 1.23,
        "force": 41.61,
        "virial": 8.53,
        "tags": [],
        "notes": "",
        "id": 49,
        "parent_id": 34
    },
    {
        "name": "train-1",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Dong_Si/NEP-iteration-1/train-1",
        "data_size": 150,
        "energy": 1.03,
        "force": 54.57,
        "virial": 21.79,
        "tags": [],
        "notes": "",
        "id": 50,
        "parent_id": 34
    },
    {
        "name": "predict-2",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Dong_Si/NEP-iteration-2/predict-2",
        "data_size": 100,
        "energy": 0.62,
        "force": 17.42,
        "virial": 7.36,
        "tags": [],
        "notes": "",
        "id": 51,
        "parent_id": 35
    },
    {
        "name": "train-2",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Dong_Si/NEP-iteration-2/train-2",
        "data_size": 350,
        "energy": 0.82,
        "force": 40.77,
        "virial": 14.58,
        "tags": [],
        "notes": "",
        "id": 52,
        "parent_id": 35
    },
    {
        "name": "01-NEP",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wu_C_Si_GaAs_PbTe/GaAs/01-NEP",
        "data_size": 0,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 53,
        "parent_id": 38
    },
    {
        "name": "data",
        "model_type": "NEP",
        "model_path": "https://gitlab.com/brucefan1983/nep-data/-/tree/main/2024_Wu_C_Si_GaAs_PbTe/GaAs/01-NEP/data",
        "data_size": 207,
        "energy": 0.0,
        "force": 0.0,
        "virial": 0.0,
        "tags": [],
        "notes": "",
        "id": 54,
        "parent_id": 53
    }
]
        for model in models:
            self.model_service.add_version(
                project_id=project.id,
                **model
            )
