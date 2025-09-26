#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 12:52
# @Author  :
# @email    : 1747193328@qq.com

import sys
from pathlib import Path
from loguru import logger
sys.path.append(str(Path(__file__).resolve().parent))
from NepTrainKit import src_rc

try:
    # Actual if statement not needed, but keeps code inspectors more happy
    if __nuitka_binary_dir is not None: # type: ignore  
        is_nuitka_compiled = True
    else:
        is_nuitka_compiled = False
except NameError:
    is_nuitka_compiled = False

if is_nuitka_compiled:
    logger.add("./Log/{time:%Y-%m}.log",
               level="DEBUG",
                )
    module_path = Path("./").resolve()
else:
    module_path = Path(__file__).resolve().parent

