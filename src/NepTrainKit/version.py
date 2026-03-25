#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 13:33
# @Author  : Bing
# @email   : 1747193328@qq.com

import sys
from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
    try:
        from NepTrainKit._version import version as generated_version
    except Exception:
        generated_version = None
    if generated_version:
        return generated_version

    try:
        return version("NepTrainKit")
    except PackageNotFoundError:
        pass

    try:
        from setuptools_scm import get_version
    except Exception:
        return "0+unknown"

    try:
        return get_version(
            root="../..",
            relative_to=__file__,
            tag_regex=r"^v(?P<version>[0-9.]+(?:b[0-9]+)?)$",
            local_scheme="no-local-version",
        )
    except Exception:
        return "0+unknown"


__version__ = _resolve_version()

OWNER = "aboys-cb"
REPO = "NepTrainKit"
DOCS_BASE_URL = "https://neptrainkit.readthedocs.io/en/latest/"
HELP_URL = "https://neptrainkit.readthedocs.io/en/latest/index.html"
FEEDBACK_URL = f"https://github.com/{OWNER}/{REPO}/issues"
RELEASES_URL = f"https://github.com/{OWNER}/{REPO}/releases"
RELEASES_API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/latest"
RELEASES_LIST_API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/releases"
YEAR = 2024
AUTHOR = "ChengBing Chen"

if sys.platform == "win32":
    UPDATE_FILE = "update.zip"
    UPDATE_EXE = "update.exe"
    NepTrainKit_EXE = "NepTrainKit.exe"
else:
    UPDATE_FILE = "update.tar.gz"  # pyright: ignore
    UPDATE_EXE = "update.bin"  # pyright: ignore
    NepTrainKit_EXE = "NepTrainKit.bin"  # pyright: ignore
