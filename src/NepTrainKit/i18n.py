"""Qt translation helpers for NepTrainKit."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from PySide6.QtCore import QLocale, QTranslator
from PySide6.QtWidgets import QApplication
from loguru import logger

from NepTrainKit.config import Config

SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = ("auto", "en_US", "zh_CN")
LANGUAGE_LABELS: Final[dict[str, str]] = {
    "auto": "Auto",
    "en_US": "English",
    "zh_CN": "中文",
}
TRANSLATION_BASENAME: Final[str] = "neptrainkit"

_translator: QTranslator | None = None
_installed_language = "en_US"


def normalize_language(value: object | None) -> str:
    """Return a supported language config value, falling back to ``auto``."""
    text = str(value or "auto").strip()
    if text in SUPPORTED_LANGUAGES:
        return text
    return "auto"


def resolve_language(value: object | None = None, locale_name: str | None = None) -> str:
    """Resolve a configured language value to an actual runtime language."""
    language = normalize_language(value)
    if language != "auto":
        return language

    locale = locale_name if locale_name is not None else QLocale.system().name()
    if str(locale).lower().startswith("zh"):
        return "zh_CN"
    return "en_US"


def translation_path(language: str) -> Path | None:
    """Return the packaged ``.qm`` path for ``language`` when one is needed."""
    resolved = resolve_language(language, "en_US") if language == "auto" else normalize_language(language)
    if resolved != "zh_CN":
        return None
    return Path(__file__).resolve().parent / "translations" / f"{TRANSLATION_BASENAME}_{resolved}.qm"


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
