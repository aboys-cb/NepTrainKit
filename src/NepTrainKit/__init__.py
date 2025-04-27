#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/11/28 12:52
# @Author  :
# @email    : 1747193328@qq.com

import os
import sys
from loguru import logger


try:
    # Actual if statement not needed, but keeps code inspectors more happy
    if __nuitka_binary_dir is not None:
        is_nuitka_compiled = True
except NameError:
    is_nuitka_compiled = False



if is_nuitka_compiled:


    logger.add("./Log/{time:%Y-%m}.log",
               level="DEBUG",
                )
    module_path="./"
else:

    module_path = os.path.dirname(__file__)

