#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/10/17 13:14
# @Author  : 鍏?
# @email    : 1747193328@qq.com
import functools
import re
import time
from typing import Any
from loguru import logger
import hashlib
from pathlib import Path
from NepTrainKit.paths import (
    PathLike,
    as_path,
    ensure_directory,
    ensure_suffix,
    iter_files,
    check_path_type,
    get_user_config_path,
)
 

 

 

def timeit(func):
    """Decorator that logs execution time via Loguru.

    Parameters
    ----------
    func : Callable
        Function to wrap.

    Examples
    --------
    >>> @timeit
    ... def demo():
    ...     pass
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

 


def call_path_dialog(*args, **kwargs):
    """Adapter to ``NepTrainKit.ui.dialogs.call_path_dialog`` to avoid GUI imports at module import time."""
    from NepTrainKit.ui.dialogs import call_path_dialog as _call
    return _call(*args, **kwargs)


def sha256_file(path: str | Path, chunk: int = 8 * 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
def unzip():
    """Adapter to ``NepTrainKit.ui.updater.unzip`` to avoid GUI imports at module import time."""
    from NepTrainKit.ui.updater import unzip as _unzip
    return _unzip()

class LoadingThread:
    """Proxy class that instantiates the real Qt thread on demand."""
    def __new__(cls, *args, **kwargs):
        from NepTrainKit.ui.threads import LoadingThread as _LT
        return _LT(*args, **kwargs)




class DataProcessingThread:
    def __new__(cls, *args, **kwargs):
        from NepTrainKit.ui.threads import DataProcessingThread as _T
        return _T(*args, **kwargs)


class FilterProcessingThread:
    def __new__(cls, *args, **kwargs):
        from NepTrainKit.ui.threads import FilterProcessingThread as _T
        return _T(*args, **kwargs)


def parse_index_string(s: str, total: int) -> list[int]:
    """Parse an index expression into a list of indices.

    Parameters
    ----------
    s : str
        Index expression like ``"1:10"``, ``":100"`` or ``"::3"``.
        Multiple expressions can be separated by comma or whitespace.
    total : int
        Maximum length of the dataset for bounds checking.

    Returns
    -------
    list[int]
        Sorted list of unique indices within ``range(total)``.
    """
    indices: list[int] = []
    tokens = [t for t in re.split(r"[,\s]+", s.strip()) if t]
    for token in tokens:
        if ":" in token:
            parts = token.split(":")
            if len(parts) > 3:
                continue
            start = int(parts[0]) if parts[0] else None
            end = int(parts[1]) if len(parts) > 1 and parts[1] else None
            step = int(parts[2]) if len(parts) == 3 and parts[2] else None
            slc = slice(start, end, step)
            indices.extend(range(*slc.indices(total)))
        else:
            try:
                idx = int(token)
            except ValueError:
                continue
            if idx < 0:
                idx += total
            if 0 <= idx < total:
                indices.append(idx)
    return sorted(set(indices))


 
