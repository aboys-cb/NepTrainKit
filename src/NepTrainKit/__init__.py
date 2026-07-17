#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 12:52
# @Author  :
# @email    : 1747193328@qq.com

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from NepTrainKit.logging_config import DEFAULT_LOG_LEVEL, initialize_logging


def _try_import_src_rc() -> None:
    try:
        from . import src_rc  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "PySide6":
            raise


_try_import_src_rc()

try:
    # Actual if statement not needed, but keeps code inspectors more happy
    if __nuitka_binary_dir is not None: # type: ignore  
        is_nuitka_compiled = True
    else:
        is_nuitka_compiled = False
except NameError:
    is_nuitka_compiled = False

if is_nuitka_compiled:
    module_path = Path("./").resolve()
else:
    module_path = Path(__file__).resolve().parent

initialize_logging(
    DEFAULT_LOG_LEVEL,
    file_sink="./Log/{time:%Y-%m}.log" if is_nuitka_compiled else None,
)
