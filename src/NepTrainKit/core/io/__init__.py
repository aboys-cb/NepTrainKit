#!/usr/bin/env python 
# -*- coding: utf-8 -*-
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


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute loading
    if name == 'ResultData':
        from .base import ResultData as _T
        return _T
    if name in ('NepTrainResultData', 'NepPolarizabilityResultData', 'NepDipoleResultData'):
        from . import nep as _nep
        return getattr(_nep, name)
    if name in ('DeepmdResultData', 'is_deepmd_path'):
        from . import deepmd as _dp
        return getattr(_dp, name)
    if name == 'get_nep_type':
        from .utils import get_nep_type as _gt
        return _gt
    if name in ('load_result_data', 'register_result_loader', 'matches_result_loader'):
        from . import registry as _reg
        return getattr(_reg, name)
    raise AttributeError(name)
