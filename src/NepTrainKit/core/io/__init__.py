#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Result-data loaders and importer registry for NEP workflows.

Lazy exports are used to reduce import time and avoid cyclic imports. Use
``from NepTrainKit.core.io import load_result_data`` to discover loaders.

Examples
--------
>>> from NepTrainKit.core.io import get_nep_type
>>> isinstance(get_nep_type, type(get_nep_type))
True
"""
# Lazy exports for core.io to reduce import time and avoid cycles.
from __future__ import annotations

from typing import Any

__all__ = [
    # base
    'ResultData',
    # nep
    'NepTrainResultData', 'NepPolarizabilityResultData', 'NepDipoleResultData',
    # deepmd
    'DeepmdResultData', 'is_deepmd_path',
    # utils
    'get_nep_type',
    # registry helpers
    'load_result_data', 'register_result_loader', 'matches_result_loader',
]

from .base import ResultData
from .deepmd import DeepmdResultData, is_deepmd_path
from .nep import NepTrainResultData, NepPolarizabilityResultData, NepDipoleResultData
from .registry import load_result_data, register_result_loader, matches_result_loader
from .utils import get_nep_type
