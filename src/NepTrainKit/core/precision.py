#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Helpers for user-configurable numeric storage precision."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from NepTrainKit.config import Config
from NepTrainKit.core.types import DataPrecision, parse_data_precision


def get_storage_precision() -> DataPrecision:
    """Return the configured numeric storage precision."""

    raw = Config.get("nep", "data_precision", DataPrecision.FLOAT32)
    return parse_data_precision(raw, fallback=DataPrecision.FLOAT32)


def get_storage_float_dtype() -> np.dtype[Any]:
    """Return the numpy dtype used for persisted floating-point arrays."""

    precision = get_storage_precision()
    if precision == DataPrecision.FLOAT64:
        return np.dtype(np.float64)
    return np.dtype(np.float32)


def get_storage_float_type() -> Any:
    """Return the numpy scalar type used for persisted floating-point arrays."""

    return get_storage_float_dtype().type


def as_storage_float_array(values: Any, *, copy: bool = False) -> npt.NDArray[Any]:
    """Convert ``values`` into the configured storage float dtype."""

    array = np.asarray(values, dtype=get_storage_float_dtype())
    if copy:
        return array.copy()
    return array
