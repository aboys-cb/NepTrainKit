#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 17:21
# @email    : 1747193328@qq.com
"""Aggregate top-level UI pages for the NEP Toolkit application."""
from importlib import import_module

__all__ = [
    "MakeDataWidget",
    "SettingsWidget",
    "ShowNepWidget",
    "DataManagerWidget",
    "TrainingSetAuditWidget",
]

_PAGE_MODULES = {
    "MakeDataWidget": ".makedata",
    "SettingsWidget": ".settings",
    "ShowNepWidget": ".show_nep",
    "DataManagerWidget": ".data_manager",
    "TrainingSetAuditWidget": ".training_set_audit",
}


def __getattr__(name: str):
    module_name = _PAGE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
