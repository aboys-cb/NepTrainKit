#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from typing import Callable, Optional, Protocol
from loguru import logger
import importlib
from .utils import get_nep_type
from NepTrainKit import get_user_config_path, utils


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
        if not os.path.isdir(path):
            return False
        try:
            mod = importlib.import_module('.deepmd', __package__)
            return mod.is_deepmd_path(path)
        except Exception:
            return False
    def load(self, path: str):
        mod = importlib.import_module('.deepmd', __package__)
        return mod.DeepmdResultData.from_path(path)



class NepModelTypeLoader:
    def __init__(self, name: str, model_types: set[int], factory_path: str):
        self.name = name
        self._types = set(model_types)
        self._factory_path = factory_path
        self._factory = None
        self.model_type=None

    def matches(self, path: str) -> bool:
        if os.path.isdir(path):
            return False
        dir_path = os.path.dirname(path)
        self.model_type = get_nep_type(os.path.join(dir_path, "nep.txt"))
        return self.model_type in self._types and path.endswith(".xyz")

    def load(self, path: str):
        # Resolve factory lazily
        if self._factory is None:
            module_name, cls_name = self._factory_path.split(':', 1)
            mod = importlib.import_module(module_name)
            self._factory = getattr(mod, cls_name)
        # Pass through model_type for NepTrainResultData to keep behavior parity
        if self._factory_path.endswith(':NepTrainResultData'):
            return self._factory.from_path(path, model_type=self.model_type)
        return self._factory.from_path(path)



class OtherLoader:
    def matches(self,path: str) -> bool:
        try:
            imp_mod = importlib.import_module('.importers', __package__)
            return imp_mod.is_parseable(path)
        except Exception:
            return False

    def load(self, path: str):
        # Defer heavy parsing to the worker thread via ResultData.load_structures

        nep_mod = importlib.import_module('.nep', __package__)
        inst = nep_mod.NepTrainResultData.from_path(path)
        # If this is a LAMMPS dump, optionally prompt for element map

        imp_mod = importlib.import_module('.importers', __package__)
        lmp_imp = getattr(imp_mod, 'LammpsDumpImporter', None)
        if lmp_imp is not None and lmp_imp().matches(path):
            # Prompt user for an optional element list mapping type 1..N

            from PySide6.QtWidgets import QInputDialog
            prompt = "Please enter a list of elements (corresponding to type 1..N), separated by commas or spaces. \nFor example: Si O or Si,O"
            text, ok = QInputDialog.getText(None, "Element Mapping", prompt)

            if ok and text:
                raw = [t.strip() for t in str(text).replace(',', ' ').split() if t.strip()]
                if raw:
                    element_map = {i + 1: raw[i] for i in range(len(raw))}
                    # Attach to importer options so worker thread uses it
                    setattr(inst, '_import_options', {**getattr(inst, '_import_options', {}), 'element_map': element_map})
            else:
                return None

        return inst



# Register defaults immediately

register_result_loader(DeepmdFolderLoader())

register_result_loader(NepModelTypeLoader("nep_train", {0, 3}, 'NepTrainKit.core.io.nep:NepTrainResultData'))
register_result_loader(NepModelTypeLoader("nep_dipole", {1}, 'NepTrainKit.core.io.nep:NepDipoleResultData'))
register_result_loader(NepModelTypeLoader("nep_polar", {2}, 'NepTrainKit.core.io.nep:NepPolarizabilityResultData'))
register_result_loader(OtherLoader())
