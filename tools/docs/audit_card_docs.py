"""Card-doc integrity audit.

Kept checks (catch real bugs):
  * every card source has a doc page, and every doc page has a card source
  * serialized_keys in the card-schema comment match what to_dict() writes
  * every Params dataclass field has a dedicated parameter heading
  * every card explains its operation principle and contains inspectable math
  * every discoverable card appears exactly once in the user-facing category tree
  * compatibility-only cards stay out of the user-facing category tree
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
DOC_SOURCE_DIR = ROOT / "docs" / "source"
CARD_DIR = ROOT / "src" / "NepTrainKit" / "ui" / "views" / "_card"
DOC_DIR = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "cards"
CATEGORY_DIR = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "categories"
CORE_CARDS_DIR = ROOT / "src" / "NepTrainKit" / "core" / "cards"
INDEX_DOC = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "index.md"
RECIPES_DOC = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "recipes.md"
ROOT_INDEX = ROOT / "docs" / "source" / "index.rst"
MODULE_INDEX = ROOT / "docs" / "source" / "module" / "index.rst"
WORKFLOW_INDEX = ROOT / "docs" / "source" / "workflows" / "index.rst"
REFERENCE_INDEX = ROOT / "docs" / "source" / "reference" / "index.rst"
GLOSSARY_DOC = ROOT / "docs" / "source" / "reference" / "glossary.md"
TROUBLESHOOTING_DOC = ROOT / "docs" / "source" / "reference" / "troubleshooting.md"
NEP_DISPLAY_EXAMPLE = ROOT / "docs" / "source" / "example" / "NEP-display.md"
FORMATS_DOC = ROOT / "docs" / "source" / "formats.md"
SPHINX_CONF = ROOT / "docs" / "source" / "conf.py"
PARTIAL_NAV_JS = ROOT / "docs" / "source" / "_static" / "js" / "partial-navigation.js"

SCHEMA_RE = re.compile(r"<!--\s*card-schema:\s*(\{.*\})\s*-->")
PARAM_HEADING_RE = re.compile(
    r"^\s{0,3}#{3,4}\s+(.+?)（([A-Za-z_][A-Za-z0-9_]*)）\s*$",
    re.MULTILINE,
)
INLINE_PARAM_RE = re.compile(r"^\s*\*\*`[^`]+`\*\*（[A-Za-z_][A-Za-z0-9_]*(?:\s*/\s*[A-Za-z_][A-Za-z0-9_]*)*）", re.MULTILINE)
BANNED_PARAM_TEXT = [
    "以 UI 下拉项为准",
    "根据这张卡要补的训练集缺口设置；调整后重点检查输出数量、几何合理性和 `Config_type` 标签是否符合预期。",
    "模式会改变结构生成语义，不是单纯的输出数量开关。",
    "选择这张卡的主操作模式。",
    "对应的生成或过滤行为。",
    "指定参与生成、替换或扰动的元素集合。",
    "选择操作沿哪个空间轴或磁矩参考轴定义。",
]
TYPE_PREFIX = "类型："
CONDITION_PREFIX = "生效条件："
PRINCIPLE_HEADING_RE = re.compile(
    r"^## (?:原理与公式|工作原理|先理解“副本平移量”|旧算法实际做什么)\s*$",
    re.MULTILINE,
)
CARD_SECTION_REQUIREMENTS = {
    "功能说明": re.compile(r"^## (?:功能说明|这张卡做什么)\s*$", re.MULTILINE),
    "操作示例": re.compile(r"^## (?:操作示例|快速使用|常用工作流)\s*$", re.MULTILINE),
    "参数说明": re.compile(r"^## 参数说明\s*$", re.MULTILINE),
    "常见问题": re.compile(r"^## 常见问题\s*$", re.MULTILINE),
}
MATH_RE = re.compile(
    r"(?:\$\$|(?<!\$)\$(?!\$)[^$\n]+\$(?!\$))",
    re.DOTALL,
)
INLINE_MATH_DELIMITER_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)")
DISPLAY_MATH_DELIMITER_RE = re.compile(r"(?<!\\)\$\$(?!\$)")


@dataclass
class CardCode:
    source_file: str
    card_name: str
    class_name: str
    keys: list[str]
    discoverable: bool


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
        discoverable = True
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            class_name = node.name
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "card_name" and isinstance(stmt.value, ast.Constant):
                            card_name = str(stmt.value.value)
                        if isinstance(target, ast.Name) and target.id == "discoverable" and isinstance(stmt.value, ast.Constant):
                            discoverable = bool(stmt.value.value)
            if not card_name:
                continue
            # grab to_dict keys
            keys = _collect_todict_keys(node)
            break
        if not card_name:
            continue
        rel = path.relative_to(ROOT).as_posix()
        cards[rel] = CardCode(rel, card_name, class_name, keys, discoverable)
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


def parse_category_card_refs() -> dict[str, list[Path]]:
    """Return card-doc filenames and the category pages that reference them."""
    refs: dict[str, list[Path]] = {}
    pattern = re.compile(r"^\s*\.\./cards/([a-z0-9-]+)(?:\.md)?\s*$", re.MULTILINE)
    for path in sorted(CATEGORY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for slug in pattern.findall(text):
            refs.setdefault(f"{slug}.md", []).append(path)
    return refs


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


def extract_parameter_blocks(section: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current_key: str | None = None
    for line in section.splitlines():
        match = PARAM_HEADING_RE.match(line)
        if match:
            current_key = match.group(2)
            blocks[current_key] = []
            continue
        if line.startswith("### ") and not match:
            current_key = None
            continue
        if current_key is not None:
            blocks[current_key].append(line)
    return blocks


def find_orphan_h4_parameter_headings(section: str) -> list[str]:
    """Return H4 parameter headings that appear before any H3 group in 参数说明."""
    seen_group = False
    orphan_headings: list[str] = []
    for line in section.splitlines():
        if line.startswith("### ") and not line.startswith("#### "):
            seen_group = True
            continue
        if line.startswith("#### ") and PARAM_HEADING_RE.match(line) and not seen_group:
            orphan_headings.append(line.strip())
    return orphan_headings


def find_malformed_display_math(text: str) -> list[str]:
    """Find ``$$`` blocks that MyST may merge with later inline math."""
    lines = text.splitlines()
    markers = [index for index, line in enumerate(lines) if line.strip() == "$$"]
    errors: list[str] = []
    if len(markers) % 2:
        errors.append("unpaired `$$` delimiter")
        return errors

    for opening, closing in zip(markers[::2], markers[1::2]):
        if opening == 0 or lines[opening - 1].strip():
            errors.append(f"line {opening + 1}: opening `$$` needs a blank line before it")
        if closing + 1 >= len(lines) or lines[closing + 1].strip():
            errors.append(f"line {closing + 1}: closing `$$` needs a blank line after it")
    return errors


def find_malformed_markdown_math(text: str) -> list[str]:
    """Check math delimiters outside fenced and inline code."""
    clean_lines: list[str] = []
    errors: list[str] = []
    in_fence = False
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        clean_line = re.sub(r"`[^`]*`", "", line)
        clean_lines.append(clean_line)
        if len(INLINE_MATH_DELIMITER_RE.findall(clean_line)) % 2:
            errors.append(f"line {line_number}: unpaired inline `$` delimiter")
        if re.match(r"^\s*\\[\[\]]\s*$", clean_line):
            errors.append(f"line {line_number}: raw `\\[`/`\\]` delimiter")

    if len(DISPLAY_MATH_DELIMITER_RE.findall("\n".join(clean_lines))) % 2:
        errors.append("unpaired `$$` delimiter")
    errors.extend(find_malformed_display_math("\n".join(clean_lines)))
    return errors


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

        title_match = re.search(r"^#\s+(.+)$", doc.text, re.MULTILINE)
        if not title_match or not re.search(r"[\u4e00-\u9fff]", title_match.group(1)):
            errors.append(f"{doc.path}: page title must use the Chinese UI card name")
        if re.search(r"^`Group`:\s*`", doc.text, re.MULTILINE):
            errors.append(f"{doc.path}: developer Group/Class metadata is visible to users")
        if not PRINCIPLE_HEADING_RE.search(doc.text):
            errors.append(f"{doc.path}: missing a user-facing operation-principle section")
        if not MATH_RE.search(doc.text):
            errors.append(f"{doc.path}: operation principle must contain an inspectable formula or rule")
        if code.discoverable:
            for section_name, pattern in CARD_SECTION_REQUIREMENTS.items():
                if not pattern.search(doc.text):
                    errors.append(f"{doc.path}: discoverable card is missing `## {section_name}`")
        if re.search(r"^\\[\[\]]\s*$", doc.text, re.MULTILINE):
            errors.append(f"{doc.path}: use MyST `$`/`$$` math delimiters, not raw `\\[`/`\\]`")
        malformed_math = find_malformed_display_math(doc.text)
        if malformed_math:
            errors.append(f"{doc.path}: malformed display math: {malformed_math[:3]}")

        # ---- serialized keys must match ----
        code_set = set(code.keys)
        doc_set = set(doc.keys)
        missing_in_doc = sorted(code_set - doc_set)
        extra_in_doc = sorted(doc_set - code_set)
        if missing_in_doc:
            errors.append(f"{doc.path}: keys in code but not in schema: {missing_in_doc}")
        if extra_in_doc:
            errors.append(f"{doc.path}: keys in schema but not in code: {extra_in_doc}")

        for banned in BANNED_PARAM_TEXT:
            if banned in doc.text:
                errors.append(f"{doc.path}: placeholder parameter text remains: `{banned}`")

        param_section = extract_parameter_section(doc.text)
        if param_section is not None:
            orphan_h4 = find_orphan_h4_parameter_headings(param_section)
            if orphan_h4:
                errors.append(f"{doc.path}: H4 parameter headings appear before any H3 group: {orphan_h4[:3]}")

        # ---- params-only docs must document every Params field as a heading ----
        if doc.keys == ["params"]:
            params_fields = extract_params_fields(src)
            if param_section is None:
                errors.append(f"{doc.path}: missing `## 参数说明` section")
            elif params_fields:
                heading_matches = PARAM_HEADING_RE.findall(param_section)
                documented = {key for _label, key in heading_matches}
                for label, key in heading_matches:
                    if not re.search(r"[\u4e00-\u9fff]", label):
                        errors.append(
                            f"{doc.path}: parameter `{key}` must show its Chinese UI label before the serialized key"
                        )
                for key in params_fields:
                    if key not in documented:
                        errors.append(f"{doc.path}: missing parameter heading for `{key}`")
                if INLINE_PARAM_RE.search(param_section):
                    errors.append(f"{doc.path}: old inline parameter entry remains in `## 参数说明`")
                blocks = extract_parameter_blocks(param_section)
                for key in params_fields:
                    body = blocks.get(key, [])
                    specific_lines = [
                        line.strip()
                        for line in body
                        if line.strip()
                        and not line.strip().startswith(TYPE_PREFIX)
                        and not line.strip().startswith(CONDITION_PREFIX)
                    ]
                    if not specific_lines:
                        errors.append(f"{doc.path}: parameter `{key}` has no concrete explanation beyond type/default")

    # ---- index integrity ----
    index_text = INDEX_DOC.read_text(encoding="utf-8")
    required_card_links = {
        "扩胞": "cards/super-cell-card.md",
        "晶格应变": "cards/cell-strain-card.md",
        "磁序": "cards/magnetic-order-card.md",
        "代表性采样": "cards/fps-filter-card.md",
    }
    for label, target in required_card_links.items():
        if target not in index_text:
            errors.append(f"{INDEX_DOC}: missing `{label}` card link `{target}`")

    # ---- user-facing category tree integrity ----
    category_refs = parse_category_card_refs()
    docs_by_name = {doc.path.name: doc for doc in doc_pages}
    for name, paths in category_refs.items():
        if name not in docs_by_name:
            errors.append(f"Category tree references unknown card doc `{name}`")
        if len(paths) > 1:
            joined = ", ".join(str(path) for path in paths)
            errors.append(f"Card doc `{name}` appears in multiple categories: {joined}")

    for code in code_cards.values():
        doc = doc_by_source.get(code.source_file)
        if doc is None:
            continue
        category_paths = category_refs.get(doc.path.name, [])
        if code.discoverable and len(category_paths) != 1:
            errors.append(
                f"{doc.path}: discoverable card must appear exactly once in category tree "
                f"(found {len(category_paths)})"
            )
        if not code.discoverable and category_paths:
            errors.append(f"{doc.path}: compatibility-only card must not appear in category tree")

    # ---- recipes integrity ----
    recipes_text = RECIPES_DOC.read_text(encoding="utf-8")
    for required in ["高熵合金", "富缺陷表面", "磁性数据"]:
        if required not in recipes_text:
            errors.append(f"{RECIPES_DOC}: missing recipe `{required}`")

    # ---- global information architecture integrity ----
    root_index_text = ROOT_INDEX.read_text(encoding="utf-8")
    root_markers = [
        ":caption: 开始使用",
        ":caption: 端到端工作流",
        ":caption: 功能指南",
        ":caption: 操作指南",
        ":caption: 参考资料",
        ":caption: 开发者",
    ]
    marker_positions = [root_index_text.find(marker) for marker in root_markers]
    if any(position < 0 for position in marker_positions):
        errors.append(f"{ROOT_INDEX}: missing one or more top-level documentation sections")
    elif marker_positions != sorted(marker_positions):
        errors.append(f"{ROOT_INDEX}: top-level documentation sections are out of user-reading order")

    feature_markers = [
        "<module/NEP-dataset-display>",
        "<module/training-set-assessment>",
        "<module/make-dataset>",
        "<module/data-management>",
        "<module/settings>",
    ]
    feature_positions = [root_index_text.find(marker) for marker in feature_markers]
    if any(position < 0 for position in feature_positions):
        errors.append(f"{ROOT_INDEX}: feature guide tree does not cover all application pages")
    elif feature_positions != sorted(feature_positions):
        errors.append(f"{ROOT_INDEX}: feature guide order no longer matches the application navigation")

    if "stacking-fault-card" in root_index_text:
        errors.append(f"{ROOT_INDEX}: compatibility-only card leaked into the global navigation")

    # ---- reader entry points added by the documentation redesign ----
    reference_index_text = REFERENCE_INDEX.read_text(encoding="utf-8")
    for marker in ("glossary", "troubleshooting"):
        if marker not in reference_index_text:
            errors.append(f"{REFERENCE_INDEX}: missing `{marker}` entry")

    glossary_text = GLOSSARY_DOC.read_text(encoding="utf-8")
    for term in ("Config_type", "FPS", "GSFE", "virial", "spin:R:3"):
        if term not in glossary_text:
            errors.append(f"{GLOSSARY_DOC}: missing core term `{term}`")

    troubleshooting_text = TROUBLESHOOTING_DOC.read_text(encoding="utf-8")
    for symptom in ("CUDA", "导入失败", "没有输出", "x、y", "energy_original"):
        if symptom not in troubleshooting_text:
            errors.append(f"{TROUBLESHOOTING_DOC}: missing troubleshooting symptom `{symptom}`")

    display_example_text = NEP_DISPLAY_EXAMPLE.read_text(encoding="utf-8")
    for step in range(1, 8):
        if f"## {step}." not in display_example_text:
            errors.append(f"{NEP_DISPLAY_EXAMPLE}: missing explained workflow step {step}")

    workflow_index_text = WORKFLOW_INDEX.read_text(encoding="utf-8")
    for workflow in ("clean-candidate-structures", "review-training-results", "manage-iterations"):
        if workflow not in workflow_index_text:
            errors.append(f"{WORKFLOW_INDEX}: missing workflow `{workflow}`")

    module_index_text = MODULE_INDEX.read_text(encoding="utf-8")
    for child_page in ("nep-display-open-data", "training-audit-overview", "make-dataset-cards/index"):
        if child_page not in module_index_text:
            errors.append(f"{MODULE_INDEX}: missing common feature entry `{child_page}`")

    formats_text = FORMATS_DOC.read_text(encoding="utf-8")
    if "## 30 秒快速判断" not in formats_text:
        errors.append(f"{FORMATS_DOC}: missing first-screen format decision guide")

    # ---- math syntax across the complete Markdown documentation ----
    for path in sorted(DOC_SOURCE_DIR.rglob("*.md")):
        math_errors = find_malformed_markdown_math(path.read_text(encoding="utf-8-sig"))
        if math_errors:
            errors.append(f"{path}: malformed math delimiters: {math_errors[:3]}")

    conf_text = SPHINX_CONF.read_text(encoding="utf-8-sig")
    if "'collapse_navigation': True" not in conf_text:
        errors.append(f"{SPHINX_CONF}: deep navigation branches must stay collapsed until the user opens them")
    if "'titles_only': True" not in conf_text:
        errors.append(f"{SPHINX_CONF}: global sidebar must not mix page-local headings into the document tree")
    if "'js/partial-navigation.js'" not in conf_text:
        errors.append(f"{SPHINX_CONF}: partial document navigation script is not enabled")
    if not PARTIAL_NAV_JS.exists():
        errors.append(f"{PARTIAL_NAV_JS}: partial document navigation script is missing")
    else:
        partial_nav_text = PARTIAL_NAV_JS.read_text(encoding="utf-8")
        if "typesetPromise([content])" not in partial_nav_text:
            errors.append(f"{PARTIAL_NAV_JS}: swapped content must be re-typeset with MathJax")
        if '!raw || raw.startsWith("#") || raw.startsWith("javascript:")' in partial_nav_text:
            errors.append(f"{PARTIAL_NAV_JS}: current-page sidebar links must be normalized before a page swap")
        if "currentBranch.innerHTML" in partial_nav_text:
            errors.append(f"{PARTIAL_NAV_JS}: do not replace the whole active sidebar branch on every navigation")

    return errors


def main() -> int:
    errors = audit()
    if errors:
        print("Card docs audit FAILED:")
        for err in errors:
            print(f"- {err}")
        return 1
    code_count = len(parse_code_cards())
    doc_count = len(list(DOC_DIR.glob("*.md")))
    print("Card docs audit PASSED")
    print(f"- code cards: {code_count}")
    print(f"- doc pages:  {doc_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
