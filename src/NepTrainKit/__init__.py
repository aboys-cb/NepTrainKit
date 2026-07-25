#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 12:52
# @Author  :
# @email    : 1747193328@qq.com

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

try:
    # Nuitka injects this name into compiled modules.
    if __nuitka_binary_dir is not None:  # type: ignore[name-defined]
        is_nuitka_compiled = True
    else:
        is_nuitka_compiled = False
except NameError:
    is_nuitka_compiled = False

from NepTrainKit.runtime_package import (
    HEALTH_CHECK_FLAG,
    MANAGED_RUNTIME_SPEC,
    activate_runtime_package,
    default_runtime_root,
)

managed_runtime_root = default_runtime_root(compiled=is_nuitka_compiled)
_is_runtime_health_check = HEALTH_CHECK_FLAG in sys.argv[1:]
try:
    managed_runtime_activation = (
        None
        if _is_runtime_health_check
        else activate_runtime_package(
            MANAGED_RUNTIME_SPEC,
            managed_runtime_root,
        )
    )
    managed_runtime_error = ""
except Exception as exc:  # noqa: BLE001 - fall back to the installed dependency
    managed_runtime_activation = None
    managed_runtime_error = f"{type(exc).__name__}: {exc}"

from NepTrainKit.logging_config import DEFAULT_LOG_LEVEL, initialize_logging


def _try_import_src_rc() -> None:
    try:
        from . import src_rc  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "PySide6":
            raise


if not _is_runtime_health_check:
    _try_import_src_rc()

if is_nuitka_compiled:
    module_path = Path("./").resolve()
else:
    module_path = Path(__file__).resolve().parent

if not _is_runtime_health_check:
    initialize_logging(
        DEFAULT_LOG_LEVEL,
        file_sink="./Log/{time:%Y-%m}.log" if is_nuitka_compiled else None,
    )
