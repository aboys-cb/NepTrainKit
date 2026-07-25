#!/usr/bin/env python
"""Update Qt translation sources and compiled catalogs."""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "NepTrainKit"
TS = SRC / "translations" / "neptrainkit_zh_CN.ts"

TRANSLATION_HELPERS = {"_tr"}
TRANSLATION_METHODS = {"tr"}
QT_TRANSLATE_CALLS = {
    ("QCoreApplication", "translate"),
    ("QApplication", "translate"),
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _attribute_chain_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _has_translation_marker(path: Path) -> bool:
    try:
        tree = ast.parse(_read_text(path), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in TRANSLATION_HELPERS:
            return True
        if isinstance(node.func, ast.Attribute):
            owner = _attribute_chain_name(node.func.value)
            if node.func.attr in TRANSLATION_METHODS:
                return True
            if owner and (owner, node.func.attr) in QT_TRANSLATE_CALLS:
                return True
    return False


def _helper_contexts() -> dict[str, str]:
    contexts: dict[str, str] = {}
    pattern = re.compile(r'QCoreApplication\.translate\(\s*["\']([^"\']+)["\']\s*,\s*text\s*\)')
    for path in sorted(SRC.rglob("*.py")):
        text = _read_text(path)
        if "def _tr" not in text:
            continue
        match = pattern.search(text)
        if not match:
            continue
        contexts[path.relative_to(SRC).as_posix()] = match.group(1)
    return contexts


def _location_context(filename: str, helper_contexts: dict[str, str]) -> str | None:
    location = Path(filename).as_posix()
    matches = {
        context
        for suffix, context in helper_contexts.items()
        if location == suffix or location.endswith(f"/{suffix}")
    }
    if len(matches) == 1:
        return matches.pop()
    return None


def _source_files() -> list[str]:
    patterns = ("*.py", "*.ui", "*.qml")
    files: list[str] = []
    for pattern in patterns:
        for path in sorted(SRC.rglob(pattern)):
            if path.suffix != ".py":
                files.append(str(path))
                continue
            if _has_translation_marker(path):
                files.append(str(path))
    return files


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    raise SystemExit(f"{name} not found. Install PySide6 tools in the active environment.")


def _find_or_create_context(root: ET.Element, name: str) -> ET.Element:
    for context in root.findall("context"):
        if context.findtext("name") == name:
            return context
    context = ET.Element("context")
    name_elem = ET.SubElement(context, "name")
    name_elem.text = name
    root.append(context)
    return context


OBSOLETE_TRANSLATION_TYPES = {"vanished", "obsolete"}


def _prune_obsolete_duplicates(root: ET.Element) -> None:
    for context in root.findall("context"):
        active_sources = {
            (message.findtext("source") or "").strip()
            for message in context.findall("message")
            if (message.find("translation") is None or message.find("translation").get("type") not in OBSOLETE_TRANSLATION_TYPES)
        }
        for message in list(context.findall("message")):
            translation = message.find("translation")
            if translation is None or translation.get("type") not in OBSOLETE_TRANSLATION_TYPES:
                continue
            source = (message.findtext("source") or "").strip()
            if source and source in active_sources:
                context.remove(message)


def _finalize_populated_translations(root: ET.Element) -> None:
    for translation in root.findall(".//translation[@type='unfinished']"):
        if (translation.text or "").strip():
            translation.attrib.pop("type", None)


def _normalize_helper_contexts(ts_path: Path) -> None:
    tree = ET.parse(ts_path)
    root = tree.getroot()
    helper_contexts = _helper_contexts()

    for context in list(root.findall("context")):
        if (context.findtext("name") or "").strip():
            continue

        for message in list(context.findall("message")):
            locations = message.findall("location")
            if not locations:
                continue

            location_matches = [
                _location_context(location.get("filename", ""), helper_contexts)
                for location in locations
            ]
            if any(match is None for match in location_matches):
                continue
            matched_contexts = set(location_matches)
            if len(matched_contexts) != 1:
                continue

            target = _find_or_create_context(root, matched_contexts.pop())
            context.remove(message)
            target.append(message)

        if not context.findall("message"):
            root.remove(context)

    _prune_obsolete_duplicates(root)
    _finalize_populated_translations(root)
    tree.write(ts_path, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Update NepTrainKit Qt translation files.")
    parser.add_argument("--no-lupdate", action="store_true", help="Skip updating the .ts source catalog.")
    parser.add_argument("--no-lrelease", action="store_true", help="Skip compiling the .qm runtime catalog.")
    args = parser.parse_args(argv)

    TS.parent.mkdir(parents=True, exist_ok=True)

    if not args.no_lupdate:
        sources = _source_files()
        if not sources:
            raise SystemExit(f"No translation source files found under {SRC}")
        subprocess.run(
            [_tool("pyside6-lupdate"), "-tr-function-alias", "tr+=_tr", *sources, "-ts", str(TS)],
            check=True,
        )
        _normalize_helper_contexts(TS)

    if not args.no_lrelease:
        subprocess.run([_tool("pyside6-lrelease"), str(TS)], check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
