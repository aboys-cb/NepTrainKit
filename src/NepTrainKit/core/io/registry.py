#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Protocol
from loguru import logger
from . import deepmd, nep
from .utils import get_nep_type
from .importers import import_structures, write_extxyz,   is_parseable
from NepTrainKit import get_user_config_path


class ResultDataProtocol(Protocol):
    load_flag: bool
    @classmethod
    def from_path(cls, path: str, *args, **kwargs):
        ...


class ResultLoader(Protocol):
    name: str
    def matches(self, path: str) -> bool: ...
    def load(self, path: str): ...


_RESULT_LOADERS: list[ResultLoader] = []


def register_result_loader(loader: ResultLoader):
    _RESULT_LOADERS.append(loader)
    return loader
def matches_result_loader(path)->bool:
    for loader in _RESULT_LOADERS:
        try:
            if loader.matches(path):
                return True
        except:
            pass
    return False
def load_result_data(path: str):
    for loader in _RESULT_LOADERS:
        try:
            if loader.matches(path):
                return loader.load(path)
        except Exception:
            # Fail soft per loader
            logger.debug(f"{loader.name}failed to load {path}")
            continue

    # Fallback: try format importers to convert to EXTXYZ and load as NEP dataset
    # try:
    #     structures = import_structures(path)
    #     if structures:
    #         base = os.path.basename(path.rstrip("/\\"))
    #         base_noext = os.path.splitext(base)[0]
    #         target_dir = os.path.join(get_user_config_path(), "imported")
    #         os.makedirs(target_dir, exist_ok=True)
    #         target_xyz = os.path.join(target_dir, f"{base_noext}.xyz")
    #         write_extxyz(target_xyz, structures)
    #         return nep.NepTrainResultData.from_path(target_xyz)
    # except Exception:
    #     pass
    return None


class DeepmdFolderLoader:
    name = "deepmd_folder"
    def matches(self, path: str) -> bool:
        return os.path.isdir(path) and deepmd.is_deepmd_path(path)
    def load(self, path: str):
        return deepmd.DeepmdResultData.from_path(path)



class NepModelTypeLoader:
    def __init__(self, name: str, model_types: set[int], factory: ResultDataProtocol):
        self.name = name
        self._types = set(model_types)
        self._factory = factory
        self.model_type=None

    def matches(self, path: str) -> bool:
        if os.path.isdir(path):
            return False
        dir_path = os.path.dirname(path)
        self.model_type = get_nep_type(os.path.join(dir_path, "nep.txt"))
        return self.model_type in self._types and path.endswith(".xyz")

    def load(self, path: str):
        # Pass through model_type for NepTrainResultData to keep behavior parity
        # model_type = get_nep_type(os.path.join(os.path.dirname(path), "nep.txt"))
        if getattr(self._factory, "__name__", "") == getattr(nep.NepTrainResultData, "__name__", ""):
            return self._factory.from_path(path, model_type=self.model_type)
        return self._factory.from_path(path)



class OtherLoader:
    def matches(self,path: str) -> bool:
        return is_parseable(path)
    def load(self, path: str):
        # Defer heavy parsing to the worker thread via ResultData.load_structures
        try:
            return nep.NepTrainResultData.from_path(path)
        except Exception:
            logger.debug(f"OtherLoader failed to prepare loader for {path}")
            return None


# Register defaults immediately

register_result_loader(DeepmdFolderLoader())

register_result_loader(NepModelTypeLoader("nep_train", {0, 3}, nep.NepTrainResultData))
register_result_loader(NepModelTypeLoader("nep_dipole", {1}, nep.NepDipoleResultData))
register_result_loader(NepModelTypeLoader("nep_polar", {2}, nep.NepPolarizabilityResultData))
register_result_loader(OtherLoader())
