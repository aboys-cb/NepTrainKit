from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from NepTrainKit.core import CardManager
from NepTrainKit.ui.views import _card as _registered_cards  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
CARD_DIR = ROOT / "src" / "NepTrainKit" / "ui" / "views" / "_card"
DOC_DIR = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "cards"
INDEX_DOC = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "index.md"
RECIPES_DOC = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "recipes.md"
WRITING_GUIDE = ROOT / "docs" / "source" / "module" / "make-dataset-cards" / "writing-guide.md"

SCHEMA_RE = re.compile(r"<!--\s*card-schema:\s*(\{.*\})\s*-->")
REQUIRED_H2 = [
    "## 功能说明",
    "## 操作示例",
    "## 适用场景与不适用场景",
    "## 输入前提",
    "## 参数说明（完整）",
    "## 推荐预设（可直接复制 JSON）",
    "## 推荐组合",
    "## 常见问题与排查",
    "## 输出标签 / 元数据变更",
    "## 可复现性说明",
]
EXAMPLE_LABELS = ["场景：", "**输入：**", "**目标：**", "**参数设置：**", "**输出：**", "**怎么验证结果合理：**"]
BANNED_PHRASES = [
    "当前卡片 -> 目标变换卡",
    "最小可运行示例：先将 **保守预设（Safe）** 应用到单帧结构",
    "上调幅度前先抽查最短键长、异常角度和晶胞条件数。",
]
EXTRA_INPUT_DOC_RULES = {
    "src/NepTrainKit/ui/views/_card/vibration_perturb_card.py": [
        "### 额外输入模板",
        "mode_1_x",
        "frequency_1",
    ],
}
INDEX_REQUIRED = [
    "## 按目标选卡",
    "## 易混卡片对比",
    "Random Slab` vs `Vacancy Defect Generation",
    "Random Doping` vs `Composition Sweep` vs `Random Occupancy",
    "Atomic Perturb` vs `Vib Mode Perturb",
    "Set Magnetic Moments` vs `Magnetic Order` vs `Magmom Rotation",
]
RECIPE_TITLES = ["## 高熵合金", "## 富缺陷表面", "## 磁性数据", "## 有机构象"]
RECIPE_BLOCKS = ["### 目标说明", "### 输入假设", "### 卡片顺序", "### 每步 JSON 配置", "### 最终数据集特征", "### 常见失败点"]
SPECIAL_LINKAGE_KEYS = {
    "super_cell_type",
    "super_scale_radio_button",
    "super_scale_condition",
    "super_cell_radio_button",
    "super_cell_condition",
    "max_atoms_radio_button",
    "num_radio_button",
    "concentration_radio_button",
    "concentration_condition",
    "afm_mode",
    "afm_kvec",
    "afm_group_a",
    "afm_group_b",
    "afm_zero_unknown",
    "gen_pm",
    "pm_count",
    "pm_direction",
    "pm_cone_angle",
    "source",
    "manual",
    "mode",
    "method",
    "doping_type",
    "budget_mode",
    "card_list",
    "filter_card",
}


@dataclass
class CardCode:
    source_file: str
    card_name: str
    keys: list[str]


@dataclass
class CardDoc:
    path: Path
    source_file: str
    card_name: str
    keys: list[str]
    text: str


def ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


DEFAULT_CACHE: dict[str, dict[str, object]] = {}


def runtime_defaults(source_file: str, card_name: str, class_name: str) -> dict[str, object]:
    cache_key = f"{source_file}::{class_name}"
    if cache_key in DEFAULT_CACHE:
        return DEFAULT_CACHE[cache_key]
    ensure_app()
    source_path = ROOT / source_file
    spec = importlib.util.spec_from_file_location(f"_audit_{source_path.stem}", source_path)
    if spec is None or spec.loader is None:
        DEFAULT_CACHE[cache_key] = {}
        return {}
    module = importlib.util.module_from_spec(spec)
    original_register = CardManager.register_card
    CardManager.register_card = classmethod(lambda _cls, card_cls: card_cls)
    try:
        spec.loader.exec_module(module)
    finally:
        CardManager.register_card = original_register
    cls = None
    for obj in module.__dict__.values():
        if inspect.isclass(obj) and getattr(obj, "card_name", None) == card_name:
            cls = obj
            break
    if cls is None and class_name and hasattr(module, class_name):
        cls = getattr(module, class_name)
    if cls is None:
        DEFAULT_CACHE[cache_key] = {}
        return {}
    data = dict(cls(None).to_dict())
    DEFAULT_CACHE[cache_key] = data
    return data


def parse_default_value(raw: str) -> object:
    txt = raw.strip()
    if txt.startswith("`") and txt.endswith("`"):
        txt = txt[1:-1].strip()
    if txt == "":
        return ""
    low = txt.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none"}:
        return None
    try:
        return json.loads(txt)
    except Exception:
        pass
    try:
        return ast.literal_eval(txt)
    except Exception:
        return txt


def parse_control_defaults(control_text: str) -> dict[str, tuple[str, object]]:
    rows: dict[str, tuple[str, object]] = {}
    matches = list(re.finditer(r"^###\s+`(?P<key>[^`]+)`\s+\([^)]+\)\s*$", control_text, flags=re.M))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(control_text)
        chunk = control_text[start:end]
        default_match = re.search(r"^- 默认值 \(Default\):\s*(.+)$", chunk, flags=re.M)
        if default_match:
            raw = default_match.group(1).strip()
            rows[match.group("key")] = (raw, parse_default_value(raw))
    return rows


def parse_code_cards() -> dict[str, CardCode]:
    cards: dict[str, CardCode] = {}
    for path in sorted(CARD_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8-sig")
        module = ast.parse(text)
        card_name = ""
        keys: list[str] = []
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "card_name" and isinstance(stmt.value, ast.Constant):
                            card_name = str(stmt.value.value)
            if not card_name:
                continue
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
        rel = path.relative_to(ROOT).as_posix()
        cards[rel] = CardCode(rel, card_name, keys)
    return cards


def parse_doc_pages() -> list[CardDoc]:
    pages: list[CardDoc] = []
    for path in sorted(DOC_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = SCHEMA_RE.search(text)
        if not match:
            raise RuntimeError(f"{path} missing card-schema metadata comment")
        data = json.loads(match.group(1))
        pages.append(CardDoc(path, str(data["source_file"]), str(data["card_name"]), list(data["serialized_keys"]), text))
    return pages


def expected_doc_filename(source_file: str) -> str:
    return f"{Path(source_file).stem.replace('_', '-')}.md"


def expected_online_doc_path(source_file: str) -> str:
    stem = Path(source_file).stem.replace("_", "-")
    return f"module/make-dataset-cards/cards/{stem}.html"


def section_slice(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    rest = text[start + len(heading):]
    nxt = re.search(r"^##\s+", rest, flags=re.M)
    return rest[:nxt.start()] if nxt else rest


def defaults_equivalent(key: str, doc_value: object, code_value: object) -> bool:
    if key == "nep_path":
        if isinstance(doc_value, str) and isinstance(code_value, str):
            normalized_doc = doc_value.replace("\\", "/")
            normalized_code = code_value.replace("\\", "/")
            if normalized_doc == "src/NepTrainKit/Config/nep89.txt" and normalized_code.endswith("/src/NepTrainKit/Config/nep89.txt"):
                return True
    if isinstance(code_value, tuple):
        code_value = list(code_value)
    if isinstance(doc_value, tuple):
        doc_value = list(doc_value)
    return doc_value == code_value


def should_have_physical_hint(chunk: str) -> bool:
    type_match = re.search(r"- 类型/范围 \(Type/Range\):\s*(.+)$", chunk, flags=re.M)
    typ = type_match.group(1).lower() if type_match else ""
    return any(token in typ for token in ("int", "float", "list[", "enum("))


def should_have_decision_hint(key: str, chunk: str) -> bool:
    type_match = re.search(r"- 类型/范围 \(Type/Range\):\s*(.+)$", chunk, flags=re.M)
    typ = type_match.group(1).lower() if type_match else ""
    return "bool" in typ or "string" in typ or "dict" in typ or key.endswith("_path")


def should_have_linkage(key: str) -> bool:
    return key in SPECIAL_LINKAGE_KEYS or key.endswith("_radio_button")


def audit() -> list[str]:
    errors: list[str] = []
    code_cards = parse_code_cards()
    doc_pages = parse_doc_pages()

    doc_by_source: dict[str, CardDoc] = {}
    for doc in doc_pages:
        doc_by_source[doc.source_file] = doc
        expected_name = expected_doc_filename(doc.source_file)
        if doc.path.name != expected_name:
            errors.append(f"{doc.path}: expected filename `{expected_name}`")

    for src, code in code_cards.items():
        if src not in doc_by_source:
            errors.append(f"Missing card doc for {src} ({code.card_name})")

    for src, doc in doc_by_source.items():
        if src not in code_cards:
            errors.append(f"Doc references unknown source_file: {src}")
            continue
        code = code_cards[src]
        if set(code.keys) != set(doc.keys):
            missing = sorted(set(code.keys) - set(doc.keys))
            extra = sorted(set(doc.keys) - set(code.keys))
            if missing:
                errors.append(f"{doc.path}: missing serialized keys in metadata: {missing}")
            if extra:
                errors.append(f"{doc.path}: extra serialized keys in metadata: {extra}")
        for h in REQUIRED_H2:
            if h not in doc.text:
                errors.append(f"{doc.path}: missing heading `{h}`")
        for phrase in BANNED_PHRASES:
            if phrase in doc.text:
                errors.append(f"{doc.path}: contains banned phrase `{phrase}`")

        example = section_slice(doc.text, "## 操作示例")
        for label in EXAMPLE_LABELS:
            if label not in example:
                errors.append(f"{doc.path}: 操作示例 missing `{label}`")

        combos = section_slice(doc.text, "## 推荐组合")
        if len(re.findall(r"^- ", combos, flags=re.M)) < 2:
            errors.append(f"{doc.path}: 推荐组合 must contain at least 2 bullets")

        faq = section_slice(doc.text, "## 常见问题与排查")
        if len(re.findall(r"^- ", faq, flags=re.M)) < 3:
            errors.append(f"{doc.path}: 常见问题与排查 must contain at least 3 bullets")

        control = section_slice(doc.text, "## 参数说明（完整）")
        blocks = list(re.finditer(r"^###\s+`(?P<key>[^`]+)`\s+\([^)]+\)\s*$", control, flags=re.M))
        block_texts: dict[str, str] = {}
        for i, match in enumerate(blocks):
            start = match.end()
            end = blocks[i + 1].start() if i + 1 < len(blocks) else len(control)
            block_texts[match.group("key")] = control[start:end]
        for key in code.keys:
            if key not in block_texts:
                errors.append(f"{doc.path}: key `{key}` missing parameter block")
                continue
            block = block_texts[key]
            for field in ["UI Label", "字段映射 (Field mapping)", "控件标签 (Caption)", "控件解释 (Widget)", "类型/范围 (Type/Range)", "默认值 (Default)", "含义 (Meaning)", "对输出规模/物理性的影响"]:
                if field not in block:
                    errors.append(f"{doc.path}: key `{key}` missing field `{field}`")
            if should_have_physical_hint(block) and "物理直觉 / 典型值" not in block:
                errors.append(f"{doc.path}: key `{key}` missing `物理直觉 / 典型值`")
            if should_have_decision_hint(key, block) and "怎么判断该开还是该关" not in block:
                errors.append(f"{doc.path}: key `{key}` missing `怎么判断该开还是该关`")
            if should_have_linkage(key) and "参数联动 / 生效条件" not in block:
                errors.append(f"{doc.path}: key `{key}` missing `参数联动 / 生效条件`")

        defaults = runtime_defaults(src, code.card_name, re.search(r"`Class`:\s*`([^`]+)`", doc.text).group(1))
        table_defaults = parse_control_defaults(control)
        for key in code.keys:
            if key not in table_defaults:
                errors.append(f"{doc.path}: missing Default cell for key `{key}`")
                continue
            raw_default, doc_default = table_defaults[key]
            if "(empty)" in raw_default:
                errors.append(f"{doc.path}: key `{key}` uses banned placeholder `(empty)`")
                continue
            if key not in defaults:
                errors.append(f"{doc.path}: runtime defaults missing key `{key}`")
                continue
            if not defaults_equivalent(key, doc_default, defaults[key]):
                errors.append(f"{doc.path}: key `{key}` default mismatch (doc={doc_default!r}, code={defaults[key]!r})")

        expected_online = expected_online_doc_path(src)
        if doc.path.stem + ".html" != Path(expected_online).name:
            errors.append(f"{doc.path}: doc filename no longer matches expected online doc path `{expected_online}`")
        for needle in EXTRA_INPUT_DOC_RULES.get(src, []):
            if needle not in doc.text:
                errors.append(f"{doc.path}: missing extra-input guidance `{needle}`")

    guide_text = WRITING_GUIDE.read_text(encoding="utf-8")
    for needle in ["操作示例模板", "额外输入模板", "参数说明写法", "常见问题与排查", "禁止内容"]:
        if needle not in guide_text:
            errors.append(f"{WRITING_GUIDE}: missing `{needle}`")

    index_text = INDEX_DOC.read_text(encoding="utf-8")
    for needle in INDEX_REQUIRED:
        if needle not in index_text:
            errors.append(f"{INDEX_DOC}: missing `{needle}`")

    recipes_text = RECIPES_DOC.read_text(encoding="utf-8")
    for title in RECIPE_TITLES:
        if title not in recipes_text:
            errors.append(f"{RECIPES_DOC}: missing recipe title `{title}`")
    for title in RECIPE_TITLES:
        start = recipes_text.find(title)
        if start < 0:
            continue
        rest = recipes_text[start + len(title):]
        nxt = re.search(r"^##\s+", rest, flags=re.M)
        block = rest[:nxt.start()] if nxt else rest
        for needle in RECIPE_BLOCKS:
            if needle not in block:
                errors.append(f"{RECIPES_DOC}: recipe `{title}` missing `{needle}`")
        if "```json" not in block:
            errors.append(f"{RECIPES_DOC}: recipe `{title}` missing JSON snippet")
        if "每步预期输出：" not in block:
            errors.append(f"{RECIPES_DOC}: recipe `{title}` missing per-step expected output")

    return errors


def main() -> int:
    errors = audit()
    if errors:
        print("Make Dataset card docs audit FAILED:")
        for err in errors:
            print(f"- {err}")
        return 1
    code_count = len([p for p in CARD_DIR.glob("*.py") if p.name != "__init__.py"])
    doc_count = len(list(DOC_DIR.glob("*.md")))
    print("Make Dataset card docs audit PASSED")
    print(f"- code cards: {code_count}")
    print(f"- doc pages: {doc_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
