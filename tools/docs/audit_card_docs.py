"""Card-doc integrity audit.

Kept checks (catch real bugs):
  * every card source has a doc page, and every doc page has a card source
  * serialized_keys in the card-schema comment match what to_dict() writes
  * every Params dataclass field has a dedicated parameter heading
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
CARD_DIR = ROOT / "src" / "NepTrainKit" / "ui" / "views" / "_card"
DOC_DIR = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "cards"
CORE_CARDS_DIR = ROOT / "src" / "NepTrainKit" / "core" / "cards"
INDEX_DOC = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "index.md"
RECIPES_DOC = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "recipes.md"

SCHEMA_RE = re.compile(r"<!--\s*card-schema:\s*(\{.*\})\s*-->")
PARAM_HEADING_RE = re.compile(r"^\s{0,3}#{3,4}\s+.+?（([A-Za-z_][A-Za-z0-9_]*)）\s*$", re.MULTILINE)


@dataclass
class CardCode:
    source_file: str
    card_name: str
    class_name: str
    keys: list[str]


@dataclass
class CardDoc:
    path: Path
    source_file: str
    card_name: str
    keys: list[str]
    text: str


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def parse_code_cards() -> dict[str, CardCode]:
    cards: dict[str, CardCode] = {}
    for path in sorted(CARD_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8-sig")
        module = ast.parse(text)
        card_name = ""
        class_name = ""
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            class_name = node.name
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "card_name" and isinstance(stmt.value, ast.Constant):
                            card_name = str(stmt.value.value)
            if not card_name:
                continue
            # grab to_dict keys
            keys = _collect_todict_keys(node)
            break
        if not card_name:
            continue
        rel = path.relative_to(ROOT).as_posix()
        cards[rel] = CardCode(rel, card_name, class_name, keys)
    return cards


def _collect_todict_keys(node: ast.ClassDef) -> list[str]:
    keys: list[str] = []
    for stmt in node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "to_dict":
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                            if target.value.id in {"data", "data_dict"} and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                                if target.slice.value not in keys:
                                    keys.append(target.slice.value)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                    if isinstance(sub.func.value, ast.Name) and sub.func.value.id in {"data", "data_dict"} and sub.func.attr == "update":
                        if sub.args and isinstance(sub.args[0], ast.Dict):
                            for k in sub.args[0].keys:
                                if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value not in keys:
                                    keys.append(k.value)
            break
    return keys


def parse_doc_pages() -> list[CardDoc]:
    pages: list[CardDoc] = []
    for path in sorted(DOC_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = SCHEMA_RE.search(text)
        if not match:
            raise SystemExit(f"{path}: missing card-schema metadata comment")
        data = json.loads(match.group(1))
        pages.append(CardDoc(path, str(data["source_file"]), str(data["card_name"]), list(data["serialized_keys"]), text))
    return pages


def extract_params_fields(source_file: str) -> list[str] | None:
    """Extract field names from the Params dataclass in core/cards/."""
    ui_path = CARD_DIR / Path(source_file).name
    if not ui_path.exists():
        return None
    ui_text = ui_path.read_text(encoding="utf-8-sig")
    # find which operation class is used:  return FooOperation()
    op_match = re.search(r"def create_operation.*\n\s+return\s+(\w+)\(\)", ui_text)
    if not op_match:
        return None
    op_name = op_match.group(1)  # e.g., CellStrainOperation
    params_name = op_name.replace("Operation", "Params")  # e.g., CellStrainParams

    for core_path in sorted(CORE_CARDS_DIR.glob("*.py")):
        if core_path.name.startswith("_"):
            continue
        core_text = core_path.read_text(encoding="utf-8-sig")
        core_module = ast.parse(core_text)
        for node in core_module.body:
            if not isinstance(node, ast.ClassDef) or node.name != params_name:
                continue
            fields: list[str] = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append(stmt.target.id)
            return fields if fields else None
    return None


def extract_parameter_section(text: str) -> str | None:
    """Return the 参数说明 section body, excluding the next H2 section."""
    match = re.search(r"^## 参数说明\s*$", text, re.MULTILINE)
    if not match:
        return None
    next_h2 = re.search(r"^##\s+", text[match.end():], re.MULTILINE)
    end = match.end() + next_h2.start() if next_h2 else len(text)
    return text[match.end():end]


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit() -> list[str]:
    errors: list[str] = []
    code_cards = parse_code_cards()
    doc_pages = parse_doc_pages()

    doc_by_source: dict[str, CardDoc] = {}
    for doc in doc_pages:
        doc_by_source[doc.source_file] = doc
        expected_name = Path(doc.source_file).stem.replace("_", "-") + ".md"
        if doc.path.name != expected_name:
            errors.append(f"{doc.path}: expected filename `{expected_name}`")

    # ---- every card ↔ every doc ----
    for src, code in code_cards.items():
        if src not in doc_by_source:
            errors.append(f"Missing doc for {src} ({code.card_name})")

    for src, doc in doc_by_source.items():
        if src not in code_cards:
            errors.append(f"Doc references unknown source: {src}")
            continue
        code = code_cards[src]

        # ---- serialized keys must match ----
        code_set = set(code.keys)
        doc_set = set(doc.keys)
        missing_in_doc = sorted(code_set - doc_set)
        extra_in_doc = sorted(doc_set - code_set)
        if missing_in_doc:
            errors.append(f"{doc.path}: keys in code but not in schema: {missing_in_doc}")
        if extra_in_doc:
            errors.append(f"{doc.path}: keys in schema but not in code: {extra_in_doc}")

        # ---- params-only docs must document every Params field as a heading ----
        if doc.keys == ["params"]:
            params_fields = extract_params_fields(src)
            param_section = extract_parameter_section(doc.text)
            if param_section is None:
                errors.append(f"{doc.path}: missing `## 参数说明` section")
            elif params_fields:
                documented = set(PARAM_HEADING_RE.findall(param_section))
                for key in params_fields:
                    if key not in documented:
                        errors.append(f"{doc.path}: missing parameter heading for `{key}`")

    # ---- index integrity ----
    index_text = INDEX_DOC.read_text(encoding="utf-8")
    for required in ["Super Cell", "Lattice Strain", "Magnetic Order", "FPS Filter"]:
        if required not in index_text:
            errors.append(f"{INDEX_DOC}: missing mention of `{required}`")

    # ---- recipes integrity ----
    recipes_text = RECIPES_DOC.read_text(encoding="utf-8")
    for required in ["高熵合金", "富缺陷表面", "磁性数据"]:
        if required not in recipes_text:
            errors.append(f"{RECIPES_DOC}: missing recipe `{required}`")

    return errors


def main() -> int:
    errors = audit()
    if errors:
        print("Card docs audit FAILED:")
        for err in errors:
            print(f"- {err}")
        return 1
    code_count = len([p for p in CARD_DIR.glob("*.py") if p.name != "__init__.py"])
    doc_count = len(list(DOC_DIR.glob("*.md")))
    print("Card docs audit PASSED")
    print(f"- code cards: {code_count}")
    print(f"- doc pages:  {doc_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
