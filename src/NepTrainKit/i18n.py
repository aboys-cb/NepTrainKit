"""Qt translation helpers for NepTrainKit."""

from __future__ import annotations

import locale
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Final
from urllib.parse import urljoin

from PySide6.QtCore import QLocale, QTranslator
from PySide6.QtWidgets import QApplication
from loguru import logger

from NepTrainKit import module_path
from NepTrainKit.config import Config
from NepTrainKit.version import DOCS_ROOT_URL

SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = ("auto", "en_US", "zh_CN")
LANGUAGE_LABELS: Final[dict[str, str]] = {
    "auto": "Auto",
    "en_US": "English",
    "zh_CN": "中文",
}
TRANSLATION_BASENAME: Final[str] = "neptrainkit"

_translator: QTranslator | None = None
_installed_language = "en_US"
_APPLE_LANGUAGE_RE: Final[re.Pattern[str]] = re.compile(r'"([^"]+)"|([A-Za-z]{2,3}(?:[-_][A-Za-z0-9]+)*)')


def normalize_language(value: object | None) -> str:
    """Return a supported language config value, falling back to ``auto``."""
    text = str(value or "auto").strip()
    if text in SUPPORTED_LANGUAGES:
        return text
    return "auto"


def _language_from_locale_name(value: object | None) -> str | None:
    text = str(value or "").strip().replace("-", "_").lower()
    if not text or text in {"c", "posix", "c.utf_8", "c.utf-8"}:
        return None
    if text.startswith("zh"):
        return "zh_CN"
    if text.startswith("en"):
        return "en_US"
    return None


def _split_locale_candidates(value: object | None) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[:;,]", text) if part.strip()]


def _environment_locale_candidates() -> list[str]:
    candidates: list[str] = []
    for key in ("LANGUAGE", "LC_MESSAGES", "LANG", "LC_ALL"):
        candidates.extend(_split_locale_candidates(os.environ.get(key)))
    return candidates


def _macos_locale_candidates() -> list[str]:
    if sys.platform != "darwin":
        return []
    candidates: list[str] = []
    try:
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleLanguages"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return candidates
    for quoted, bare in _APPLE_LANGUAGE_RE.findall(result.stdout):
        value = quoted or bare
        if value and value not in {"AppleLanguages"}:
            candidates.append(value)
    return candidates


def _windows_locale_candidates() -> list[str]:
    if sys.platform != "win32":
        return []
    candidates: list[str] = []
    try:
        import ctypes

        langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
        name = locale.windows_locale.get(langid)
        if name:
            candidates.append(name)

        buffer = ctypes.create_unicode_buffer(85)
        if ctypes.windll.kernel32.GetUserDefaultLocaleName(buffer, len(buffer)):
            candidates.append(buffer.value)
    except (AttributeError, OSError, ValueError):
        pass
    return candidates


def _qt_locale_candidates() -> list[str]:
    locale_obj = QLocale.system()
    return [*locale_obj.uiLanguages(), locale_obj.name()]


def _system_locale_candidates() -> list[str]:
    if sys.platform == "darwin":
        platform_candidates = _macos_locale_candidates()
    elif sys.platform == "win32":
        platform_candidates = _windows_locale_candidates()
    else:
        platform_candidates = _environment_locale_candidates()
    return [*platform_candidates, *_qt_locale_candidates(), *_environment_locale_candidates()]


def resolve_language(value: object | None = None, locale_name: str | None = None) -> str:
    """Resolve a configured language value to an actual runtime language."""
    language = normalize_language(value)
    if language != "auto":
        return language

    candidates = [locale_name] if locale_name is not None else _system_locale_candidates()
    for candidate in candidates:
        resolved = _language_from_locale_name(candidate)
        if resolved is not None:
            return resolved
    return "en_US"


def translation_path(language: str) -> Path | None:
    """Return the packaged ``.qm`` path for ``language`` when one is needed."""
    resolved = resolve_language(language, "en_US") if language == "auto" else normalize_language(language)
    if resolved != "zh_CN":
        return None
    return module_path / "translations" / f"{TRANSLATION_BASENAME}_{resolved}.qm"


def install_translator(app: QApplication, language: object | None = None) -> str:
    """Install the configured Qt translator and return the resolved language."""
    global _translator, _installed_language

    configured = normalize_language(language if language is not None else Config.get("ui", "language", "auto"))
    resolved = resolve_language(configured)
    _installed_language = resolved

    if _translator is not None:
        app.removeTranslator(_translator)
        _translator = None

    path = translation_path(resolved)
    if path is None:
        return resolved

    translator = QTranslator(app)
    if not path.exists():
        logger.warning("Translation file not found: {}", path)
        return resolved

    if not translator.load(str(path)):
        logger.warning("Failed to load translation file: {}", path)
        return resolved

    app.installTranslator(translator)
    _translator = translator
    return resolved


def current_language() -> str:
    """Return the most recently resolved runtime language."""
    return _installed_language


def documentation_language(language: object | None = None) -> str:
    """Return the Read the Docs language matching the running application."""
    resolved = current_language() if language is None else resolve_language(language)
    return "zh_CN" if resolved == "zh_CN" else "en"


def localized_docs_base_url(language: object | None = None) -> str:
    """Return the latest-documentation root for the requested UI language."""
    return f"{DOCS_ROOT_URL}{documentation_language(language)}/latest/"


def localized_docs_url(value: object = "", language: object | None = None) -> str:
    """Localize an official documentation path or URL for the current UI."""
    text = str(value or "").strip()
    base_url = localized_docs_base_url(language)
    if not text:
        return base_url

    official_prefixes = (
        f"{DOCS_ROOT_URL}en/latest/",
        f"{DOCS_ROOT_URL}zh_CN/latest/",
    )
    for prefix in official_prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    else:
        if text.startswith(("http://", "https://")):
            return text
    return urljoin(base_url, text.lstrip("/"))
