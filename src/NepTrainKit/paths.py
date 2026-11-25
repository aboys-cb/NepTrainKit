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


def get_user_config_path(*, create: bool = True) -> Path:
    """Return the user-specific configuration path as a ``Path`` instance.

    ``create=False`` is reserved for startup error reporting, where attempting
    to create the directory again would hide the original failure.
    """
    if platform.system() == 'Windows':
        local = os.getenv('LOCALAPPDATA')
        if not local:
            user_profile = os.getenv('USERPROFILE')
            local_path = as_path(user_profile) / 'AppData' / 'Local' if user_profile else Path.home() / 'AppData' / 'Local'
        else:
            local_path = as_path(local)
        config_path = local_path / 'NepTrainKit'
    else:
        config_path = Path.home() / '.config' / 'NepTrainKit'

    return ensure_directory(config_path) if create else config_path


def get_bundled_nep89_path() -> Path:
    """Return the bundled ``nep89`` model path shipped with the application."""
    from NepTrainKit import module_path

    return module_path / "Config" / "nep89.txt"


#: Portable stand-in for the bundled model, written into exported JSON.
#:
#: A workflow that uses the bundled model should keep working on another machine
#: or another install location, so its configuration stores this token instead
#: of an absolute path. User-chosen files keep their own path: only the bundled
#: asset can be resolved everywhere.
BUNDLED_NEP89_TOKEN = "bundled:nep89"


def portable_model_path(path) -> str:
    """Return ``path`` as a portable token when it is the bundled model.

    Parameters
    ----------
    path : PathLike or str
        Model path as currently displayed in the interface.

    Returns
    -------
    str
        ``BUNDLED_NEP89_TOKEN`` for the bundled model, otherwise the original
        path so user selections stay explicit.
    """
    text = str(path or "")
    if not text.strip():
        return text
    try:
        resolved = Path(text).expanduser().resolve()
        bundled = get_bundled_nep89_path().resolve()
    except OSError:
        return text
    return BUNDLED_NEP89_TOKEN if resolved == bundled else text


def resolve_model_path(value) -> Path:
    """Resolve a serialized model path, expanding the bundled-model token.

    Absolute paths from older exports are returned unchanged, so previously
    saved workflows keep loading.

    Parameters
    ----------
    value : PathLike or str
        Value from card parameters or an exported JSON file.

    Returns
    -------
    Path
        A filesystem path suitable for opening.
    """
    text = str(value or "").strip()
    if text == BUNDLED_NEP89_TOKEN:
        return get_bundled_nep89_path()
    return Path(text).expanduser()


__all__ = [
    'PathLike',
    'as_path',
    'ensure_directory',
    'ensure_suffix',
    'iter_files',
    'check_path_type',
    'get_user_config_path',
    'get_bundled_nep89_path',
    'BUNDLED_NEP89_TOKEN',
    'portable_model_path',
    'resolve_model_path',
]
