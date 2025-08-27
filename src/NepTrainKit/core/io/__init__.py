#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/10/18 15:24
# @Author  : 兵
# @email    : 1747193328@qq.com
from .base import ResultData
from .nep import NepTrainResultData,NepPolarizabilityResultData,NepDipoleResultData
from .deepmd import DeepmdResultData,is_deepmd_path
from .utils import get_nep_type



