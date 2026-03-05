"""Audit Make Dataset card documentation coverage and schema alignment.

Checks:
1) Every card file under ui/views/_card has exactly one doc page.
2) Doc metadata `serialized_keys` matches code `to_dict` keys exactly.
3) Required section template exists on every card page.
4) Preset section has exactly 3 subsections (supports Chinese/English labels).
5) Recommended combinations section has at least 2 bullets.
6) Minimal/high-throughput example sentences are present.
7) Banned template phrases are not present.
8) When-to-use section contains add/avoid signals.
9) Control section enforces per-parameter recommendation structure.
10) Rule cards include schema subsections.
"""

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

SCHEMA_RE = re.compile(r"<!--\s*card-schema:\s*(\{.*\})\s*-->")

REQUIRED_H2 = [
    "## 功能说明",
    "## 适用场景与不适用场景",
    "## 输入前提",
    "## 参数说明（完整）",
    "## 推荐预设（可直接复制 JSON）",
    "## 推荐组合",
    "## 常见问题与排查",
    "## 输出标签 / 元数据变更",
    "## 可复现性说明",
]

BANNED_PHRASES = [
    "Card parameter persisted to JSON and restored on load.",
    "Defines sampling bounds and step size for generated variants.",
    "Use when: you need this transformation/filter explicitly in your pipeline and want its parameters persisted in card JSON.",
]

RULE_SCHEMA_REQUIRED = {
    "src/NepTrainKit/ui/views/_card/random_doping_card.py": "规则输入 Schema",
    "src/NepTrainKit/ui/views/_card/random_vacancy_card.py": "规则输入 Schema",
    "src/NepTrainKit/ui/views/_card/conditional_replace_card.py": "替换输入 Schema",
}

CARD_GROUP_SOURCE = "src/NepTrainKit/ui/views/_card/card_group.py"
GROUP_LABEL_SOURCE = "src/NepTrainKit/ui/views/_card/group_label_card.py"


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


def runtime_defaults(
    source_file: str,
    card_name: str,
    class_name: str,
) -> dict[str, object]:
    cache_key = f"{source_file}::{class_name}"
    if cache_key in DEFAULT_CACHE:
        return DEFAULT_CACHE[cache_key]
    ensure_app()
    source_path = ROOT / source_file
    if not source_path.exists():
        DEFAULT_CACHE[cache_key] = {}
        return {}
    spec = importlib.util.spec_from_file_location(
        f"_audit_card_{source_path.stem}_{abs(hash(source_file))}",
        source_path,
    )
    if spec is None or spec.loader is None:
        DEFAULT_CACHE[cache_key] = {}
        return {}
    mod = importlib.util.module_from_spec(spec)
    original_register = CardManager.register_card
    CardManager.register_card = classmethod(lambda _cls, card_cls: card_cls)
    try:
        spec.loader.exec_module(mod)
    finally:
        CardManager.register_card = original_register
    cls = None
    for obj in mod.__dict__.values():
        if inspect.isclass(obj) and hasattr(obj, "card_name"):
            if getattr(obj, "card_name", None) == card_name:
                cls = obj
                break
    if cls is None and class_name and hasattr(mod, class_name):
        cls = getattr(mod, class_name)
    if cls is None:
        DEFAULT_CACHE[cache_key] = {}
        return {}
    card = cls(None)
    data = dict(card.to_dict())
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
    # Legacy table format.
    lines = [ln for ln in control_text.splitlines() if ln.strip().startswith("|")]
    if lines:
        for ln in lines[2:]:
            cols = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cols) < 4:
                continue
            key_match = re.search(r"`([^`]+)`", cols[1])
            if not key_match:
                continue
            key = key_match.group(1)
            raw_default = cols[3]
            rows[key] = (raw_default, parse_default_value(raw_default))
        if rows:
            return rows

    # New list format.
    blocks = list(
        re.finditer(
            r"^###\s+`(?P<key>[^`]+)`\s+\((?P<ui>[^)]+)\)\s*$",
            control_text,
            flags=re.MULTILINE,
        )
    )
    for i, m in enumerate(blocks):
        key = m.group("key").strip()
        start = m.end()
        end = blocks[i + 1].start() if i + 1 < len(blocks) else len(control_text)
        chunk = control_text[start:end]
        default_match = re.search(r"^- 默认值 \(Default\):\s*(.+)$", chunk, flags=re.MULTILINE)
        if not default_match:
            continue
        raw_default = default_match.group(1).strip()
        rows[key] = (raw_default, parse_default_value(raw_default))
    return rows


ENUM_LIKE_KEYS: set[str] = {
    "mode",
    "engine_type",
    "method",
    "budget_mode",
    "source",
    "distribution",
    "doping_type",
    "afm_mode",
    "pm_direction",
    "axis",
    "kvec",
    "lattice",
    "super_cell_type",
    "preset_index",
    "apply_mode",
    "rotation_mode",
    "plane",
    "slab_mode",
    "pm_mode",
}

PATH_LIKE_KEYS: set[str] = {
    "nep_path",
    "model_path",
    "path",
}

NOTE_STYLE_KEYS_BY_CARD: dict[str, set[str]] = {
    "group-label-card.md": {"mode", "kvec"},
}


def recommendation_style(card_file: str, key: str, typ: str) -> str:
    key_l = key.strip().lower()
    typ_l = typ.strip().lower()
    if key_l in NOTE_STYLE_KEYS_BY_CARD.get(card_file, set()):
        return "note"
    if key_l in PATH_LIKE_KEYS or key_l.endswith("_path") or "path" in key_l:
        return "note"
    if "bool" in typ_l:
        return "binary"
    if key_l in ENUM_LIKE_KEYS:
        return "tiered"
    if "string" in typ_l:
        return "note"
    if any(token in typ_l for token in ("int", "float", "list[")):
        return "tiered"
    return "note"


def is_numeric_tiered_type(key: str, typ: str) -> bool:
    if key.strip().lower() in ENUM_LIKE_KEYS:
        return False
    typ_l = typ.lower()
    if "enum(" in typ_l or "bool" in typ_l or "string" in typ_l:
        return False
    return any(token in typ_l for token in ("int", "float", "list["))


def parse_control_blocks(control_text: str) -> dict[str, tuple[str, str]]:
    blocks: dict[str, tuple[str, str]] = {}
    matches = list(
        re.finditer(
            r"^###\s+`(?P<key>[^`]+)`\s+\((?P<ui>[^)]+)\)\s*$",
            control_text,
            flags=re.MULTILINE,
        )
    )
    for i, m in enumerate(matches):
        key = m.group("key").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(control_text)
        chunk = control_text[start:end]
        typ_match = re.search(r"^- 类型/范围 \(Type/Range\):\s*(.+)$", chunk, flags=re.MULTILINE)
        typ = typ_match.group(1).strip() if typ_match else "unknown"
        blocks[key] = (typ, chunk)
    return blocks


def defaults_equivalent(key: str, doc_value: object, code_value: object) -> bool:
    # Environment-derived path: allow semantic placeholder values.
    if key == "nep_path":
        if isinstance(code_value, str) and code_value:
            if doc_value in {"", None, "__AUTO_PATH__", "AUTO"}:
                return True
            if isinstance(doc_value, str):
                return True
        return doc_value == code_value

    # Empty values should be explicit as "" or null.
    if key in {"target", "replacements", "condition", "elements", "species", "params", "rules", "manual", "group_filter"}:
        if doc_value == "(empty)":
            return False

    # Keep string enum strict (e.g., "111" vs 111).
    if isinstance(code_value, str) != isinstance(doc_value, str):
        return False

    if isinstance(code_value, tuple):
        code_value = list(code_value)
    if isinstance(doc_value, tuple):
        doc_value = list(doc_value)

    return doc_value == code_value


def parse_code_cards() -> dict[str, CardCode]:
    cards: dict[str, CardCode] = {}
    for path in sorted(CARD_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8-sig")
        mod = ast.parse(text)
        card_name = ""
        keys: list[str] = []

        for node in mod.body:
            if not isinstance(node, ast.ClassDef):
                continue
            has_card_name = False
            for b in node.body:
                if isinstance(b, ast.Assign):
                    for t in b.targets:
                        if (
                            isinstance(t, ast.Name)
                            and t.id == "card_name"
                            and isinstance(b.value, ast.Constant)
                            and isinstance(b.value.value, str)
                        ):
                            card_name = b.value.value
                            has_card_name = True
            if not has_card_name:
                continue

            for b in node.body:
                if isinstance(b, ast.FunctionDef) and b.name == "to_dict":
                    for n in ast.walk(b):
                        if isinstance(n, ast.Assign):
                            for t in n.targets:
                                if (
                                    isinstance(t, ast.Subscript)
                                    and isinstance(t.value, ast.Name)
                                    and t.value.id in {"data", "data_dict"}
                                    and isinstance(t.slice, ast.Constant)
                                    and isinstance(t.slice.value, str)
                                ):
                                    if t.slice.value not in keys:
                                        keys.append(t.slice.value)
                        if (
                            isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute)
                            and isinstance(n.func.value, ast.Name)
                            and n.func.value.id in {"data", "data_dict"}
                            and n.func.attr == "update"
                            and n.args
                            and isinstance(n.args[0], ast.Dict)
                        ):
                            for k in n.args[0].keys:
                                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                    if k.value not in keys:
                                        keys.append(k.value)
            break

        rel = path.relative_to(ROOT).as_posix()
        cards[rel] = CardCode(source_file=rel, card_name=card_name, keys=keys)
    return cards


def parse_doc_pages() -> list[CardDoc]:
    pages: list[CardDoc] = []
    for path in sorted(DOC_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        m = SCHEMA_RE.search(text)
        if not m:
            raise RuntimeError(f"{path} missing card-schema metadata comment")
        data = json.loads(m.group(1))
        pages.append(
            CardDoc(
                path=path,
                source_file=str(data["source_file"]),
                card_name=str(data["card_name"]),
                keys=list(data["serialized_keys"]),
                text=text,
            )
        )
    return pages


def section_slice(text: str, h2: str) -> str:
    start = text.find(h2)
    if start < 0:
        return ""
    rest = text[start + len(h2) :]
    nxt = re.search(r"^##\s+", rest, flags=re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def normalize_preset_subheading(value: str) -> str:
    label = value.strip()
    label = re.sub(r"\s+", "", label)
    label = label.replace("（", "(").replace("）", ")")
    base = re.sub(r"\(.*?\)", "", label).strip().lower()
    alias_map = {
        "safe": "safe",
        "保守": "safe",
        "balanced": "balanced",
        "平衡": "balanced",
        "aggressive/exploration": "aggressive_exploration",
        "激进/探索": "aggressive_exploration",
        "探索/激进": "aggressive_exploration",
        "激进探索": "aggressive_exploration",
    }
    return alias_map.get(base, base)


def audit() -> list[str]:
    errors: list[str] = []
    code_cards = parse_code_cards()
    doc_pages = parse_doc_pages()

    doc_by_source: dict[str, CardDoc] = {}
    for doc in doc_pages:
        if doc.source_file in doc_by_source:
            errors.append(f"Duplicate docs for source_file: {doc.source_file}")
        doc_by_source[doc.source_file] = doc

    # Coverage: every card has docs
    for src, code in code_cards.items():
        if src not in doc_by_source:
            errors.append(f"Missing card doc for {src} ({code.card_name})")

    # No stale docs
    for src in doc_by_source:
        if src not in code_cards:
            errors.append(f"Doc references unknown source_file: {src}")

    # Per-page checks
    for src, code in code_cards.items():
        doc = doc_by_source.get(src)
        if not doc:
            continue

        code_keys = code.keys
        doc_keys = doc.keys
        if set(code_keys) != set(doc_keys):
            missing = sorted(set(code_keys) - set(doc_keys))
            extra = sorted(set(doc_keys) - set(code_keys))
            if missing:
                errors.append(f"{doc.path}: missing serialized keys in metadata: {missing}")
            if extra:
                errors.append(f"{doc.path}: extra serialized keys in metadata: {extra}")

        # Required headings
        for h in REQUIRED_H2:
            if h not in doc.text:
                errors.append(f"{doc.path}: missing heading `{h}`")

        for phrase in BANNED_PHRASES:
            if phrase in doc.text:
                errors.append(f"{doc.path}: contains banned template phrase: `{phrase}`")

        # Keys must appear in control table (by backticked key)
        control_section = section_slice(doc.text, "## 参数说明（完整）")
        for k in code_keys:
            if f"`{k}`" not in control_section:
                errors.append(f"{doc.path}: key `{k}` not found in 参数说明 section")

        # Control section recommendation style checks per parameter block.
        control_blocks = parse_control_blocks(control_section)
        for k in code_keys:
            if k not in control_blocks:
                continue
            typ, chunk = control_blocks[k]
            has_range = "- 推荐范围 (Recommended range):" in chunk
            has_note = bool(re.search(r"^- 配置建议 \(Practical note\):", chunk, flags=re.MULTILINE))
            if has_range == has_note:
                errors.append(
                    f"{doc.path}: key `{k}` must contain exactly one recommendation structure "
                    f"(Recommended range or Practical note)"
                )
                continue

            expected_style = recommendation_style(doc.path.name, k, typ)
            if expected_style == "tiered" and not has_range:
                errors.append(f"{doc.path}: key `{k}` should use tiered Recommended range")
            if expected_style in {"binary", "note"} and not has_note:
                errors.append(f"{doc.path}: key `{k}` should use Practical note")
            if expected_style == "binary":
                if "开启：" not in chunk or "关闭：" not in chunk:
                    errors.append(f"{doc.path}: key `{k}` bool Practical note must include 开启/关闭")
            if expected_style == "tiered" and is_numeric_tiered_type(k, typ):
                range_lines = re.findall(r"^\s*-\s*(?:保守|平衡|探索)：(.+)$", chunk, flags=re.MULTILINE)
                if range_lines and any(not re.search(r"\d", line) for line in range_lines):
                    errors.append(f"{doc.path}: key `{k}` numeric Recommended range should include concrete numbers")
            if k == "nep_path":
                if has_range:
                    errors.append(f"{doc.path}: key `nep_path` must not use tiered Recommended range")
                if "src/NepTrainKit/Config/nep89.txt" not in chunk:
                    errors.append(f"{doc.path}: key `nep_path` should show project-relative default path")

        if src == CARD_GROUP_SOURCE:
            filter_block = control_blocks.get("filter_card")
            if not filter_block:
                errors.append(f"{doc.path}: card-group must document `filter_card` control block")
            else:
                _, filter_chunk = filter_block
                if "不作为下游卡片输入" not in filter_chunk:
                    errors.append(
                        f"{doc.path}: key `filter_card` should state current behavior "
                        f"(not used as downstream card input)"
                    )

        # Presets: exactly 3 required subheadings
        presets = section_slice(doc.text, "## 推荐预设（可直接复制 JSON）")
        subheads = re.findall(r"^###\s+(.+)$", presets, flags=re.MULTILINE)
        if src == GROUP_LABEL_SOURCE:
            if len(subheads) != 3:
                errors.append(f"{doc.path}: group-label presets should contain exactly 3 template subheadings")
        else:
            normalized = [normalize_preset_subheading(h) for h in subheads]
            expected_norm = ["safe", "balanced", "aggressive_exploration"]
            if normalized != expected_norm:
                errors.append(
                    f"{doc.path}: preset subheadings must map to Safe/Balanced/Aggressive/Exploration, got {subheads}"
                )

        # Minimal/high-throughput examples should be highlighted near top.
        function_desc = section_slice(doc.text, "## 功能说明")
        if "最小可运行示例" not in function_desc:
            errors.append(f"{doc.path}: missing `最小可运行示例` sentence in 功能说明 section")
        if "高通量示例" not in function_desc:
            errors.append(f"{doc.path}: missing `高通量示例` sentence in 功能说明 section")
        if ":::{tip}" not in function_desc:
            errors.append(f"{doc.path}: missing `::{{tip}}` wrapper for high-throughput example in 功能说明 section")
        if "最小可运行示例" in presets or "高通量示例" in presets:
            errors.append(f"{doc.path}: presets section should focus on JSON only (remove minimal/high-throughput examples)")
        if src == CARD_GROUP_SOURCE:
            if "两张互不依赖的分支卡片" not in function_desc:
                errors.append(f"{doc.path}: card-group 功能说明 should include container-specific minimal example")
            if "组外串接清洗/采样链路" not in function_desc:
                errors.append(f"{doc.path}: card-group 功能说明 should include container-specific high-throughput example")
            if "先将 **保守预设（Safe）** 应用到单帧结构" in function_desc:
                errors.append(f"{doc.path}: card-group should not use generic single-frame template sentence")
            if "建议先导出 xyz" in function_desc and "nep89" in function_desc.lower():
                errors.append(f"{doc.path}: card-group should not use global FPS/nep89 template sentence")

        # Recommended combinations >= 2 bullets
        combos = section_slice(doc.text, "## 推荐组合")
        bullets = re.findall(r"^- ", combos, flags=re.MULTILINE)
        if len(bullets) < 2:
            errors.append(f"{doc.path}: recommended combinations must contain at least 2 bullets")
        if src == CARD_GROUP_SOURCE and "并行扩展多样性" in combos:
            errors.append(f"{doc.path}: card-group combos should avoid generic diversity template wording")

        # When section should contain add/avoid signals
        when_section = section_slice(doc.text, "## 适用场景与不适用场景")
        if not re.search(r"Add-it trigger|建议添加条件", when_section):
            errors.append(f"{doc.path}: when-to-use section missing add-it signal")
        if not re.search(r"Avoid trigger|不建议添加条件", when_section):
            errors.append(f"{doc.path}: when-to-use section missing avoid signal")
        if src == CARD_GROUP_SOURCE:
            if not re.search(r"共享同一输入|同一输入", when_section):
                errors.append(f"{doc.path}: card-group when-to-use must describe shared-input semantics")
            if re.search(r"多个项目.{0,6}反复使用", when_section):
                errors.append(f"{doc.path}: card-group add-it trigger should not be 'multiple projects reuse'")

        # Default column must align with runtime defaults.
        class_match = re.search(r"`Class`:\s*`([^`]+)`", doc.text)
        class_name = class_match.group(1) if class_match else ""
        defaults = (
            runtime_defaults(src, code.card_name, class_name)
            if class_name
            else {}
        )
        if not defaults:
            errors.append(f"{doc.path}: failed to load runtime defaults for class `{class_name}`")
        else:
            table_defaults = parse_control_defaults(control_section)
            for k in code_keys:
                if k not in table_defaults:
                    errors.append(f"{doc.path}: missing Default cell for key `{k}`")
                    continue
                raw_default, doc_default = table_defaults[k]
                if "(empty)" in raw_default:
                    errors.append(f"{doc.path}: key `{k}` uses banned placeholder `(empty)`; use \"\" or null")
                    continue
                if k not in defaults:
                    errors.append(f"{doc.path}: runtime defaults missing key `{k}`")
                    continue
                if not defaults_equivalent(k, doc_default, defaults[k]):
                    errors.append(
                        f"{doc.path}: key `{k}` default mismatch (doc={doc_default!r}, code={defaults[k]!r})"
                    )
                if k == "nep_path":
                    if not isinstance(doc_default, str):
                        errors.append(f"{doc.path}: key `nep_path` default must be a string path")
                    else:
                        normalized = doc_default.replace("\\", "/")
                        if normalized != "src/NepTrainKit/Config/nep89.txt":
                            errors.append(
                                f"{doc.path}: key `nep_path` default should be "
                                f"`src/NepTrainKit/Config/nep89.txt`, got {doc_default!r}"
                            )

        # Rule cards require schema subsections
        needed_schema = RULE_SCHEMA_REQUIRED.get(src)
        if needed_schema and needed_schema not in control_section:
            errors.append(f"{doc.path}: missing required schema subsection `{needed_schema}`")

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
    sys.exit(main())
