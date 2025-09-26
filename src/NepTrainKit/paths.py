#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Iterator, Sequence


PathLike = str | Path


def as_path(value: PathLike | None, *, base: Path | None = None) -> Path:
    """Return ``value`` as a :class:`Path`, optionally relative to ``base``.

    - If ``value`` is ``None``, return ``base`` as a Path (or raise if base is None).
    - If ``value`` is relative and ``base`` is provided, join with ``base``.
    """
    if value is None:
        if base is None:
            raise ValueError('value and base cannot both be None')
        return Path(base)
    path_value = Path(value)
    if base and not path_value.is_absolute():
        return Path(base) / path_value
    return path_value


def ensure_directory(path: PathLike, *, exist_ok: bool = True) -> Path:
    """Create ``path`` if needed and return it as a :class:`Path`."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=exist_ok)
    return directory


def ensure_suffix(path: PathLike, suffix: str) -> Path:
    """Return ``path`` with the provided ``suffix`` appended if missing."""
    p = Path(path)
    if not suffix.startswith('.'):
        suffix = f'.{suffix}'
    return p.with_suffix(suffix) if p.suffix != suffix else p


def iter_files(root: PathLike, patterns: Sequence[str] | None = None) -> Iterator[Path]:
    """Yield files under ``root`` optionally filtered by glob ``patterns``."""
    base = Path(root)
    if not base.exists():
        return iter(())
    if not patterns:
        return base.rglob('*')

    def generator() -> Iterator[Path]:
        for pattern in patterns:
            yield from base.rglob(pattern)

    return generator()


def check_path_type(path: str | Path) -> str:
    """Return ``folder`` or ``file`` even if ``path`` does not exist."""
    p = Path(path)
    if p.is_dir():
        return "folder"
    if p.is_file():
        return "file"
    return "file" if p.suffix else "folder"


def get_user_config_path() -> Path:
    """Return the user-specific configuration path as a ``Path`` instance."""
    if platform.system() == 'Windows':
        local = os.getenv('LOCALAPPDATA')
        if not local:
            user_profile = os.getenv('USERPROFILE')
            local_path = as_path(user_profile) / 'AppData' / 'Local' if user_profile else Path.home() / 'AppData' / 'Local'
        else:
            local_path = as_path(local)
        return ensure_directory(local_path / 'NepTrainKit')

    return ensure_directory(Path.home() / '.config' / 'NepTrainKit')


__all__ = [
    'PathLike',
    'as_path',
    'ensure_directory',
    'ensure_suffix',
    'iter_files',
    'check_path_type',
    'get_user_config_path',
]

