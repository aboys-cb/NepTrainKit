"""Keep Chinese card-document labels aligned with the translated desktop UI."""

from __future__ import annotations

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CARD_DOCS = ROOT / "docs/source/module/make-dataset-cards/cards"
CATALOG_CODE = ROOT / "src/NepTrainKit/ui/widgets/card_metadata.py"
TRANSLATIONS = ROOT / "src/NepTrainKit/translations/neptrainkit_zh_CN.ts"

PARAM_HEADING_RE = re.compile(
    r"^(#{3,4})\s+(.+?)（([A-Za-z_][A-Za-z0-9_]*)）\s*$",
    re.MULTILINE,
)
SCHEMA_RE = re.compile(r'"source_file":\s*"([^"]+)"')

MANUAL_LABELS = {
    "A Range": "晶格常数 a 范围",
    "Alpha Range": "α 角范围",
    "Amplitude": "振动幅度",
    "Angle List": "倾斜角列表",
    "Angle Step Range": "角度步长范围",
    "Auto Box": "自动设置晶胞",
    "Auto Supercell": "自动扩胞",
    "Axes": "轴",
    "Axis": "轴",
    "BO C Const": "键级 C 常数",
    "BO Threshold": "键级阈值",
    "Backend": "计算后端",
    "Behavior Type": "扩胞方式",
    "Beta Range": "β 角范围",
    "Bond Detect Factor": "成键识别系数",
    "Bond Filter Axis": "键筛选轴",
    "Bond Filter Mode": "键筛选方式",
    "Bond Filter Tolerance": "键筛选容差",
    "Bond Keep Max Enable": "启用最大保留键长",
    "Bond Keep Max Factor": "最大保留键长系数",
    "Bond Keep Min Factor": "最小保留键长系数",
    "Box Padding": "晶胞留白",
    "CA Range": "c/a 范围",
    "c/a": "晶格轴比 c/a",
    "Chunk Max Atoms": "单批最大原子数",
    "Constant Moment": "固定磁矩",
    "Correlation Kernel": "相关核函数",
    "Default Moment": "默认磁矩",
    "Disturb Magnitude": "扰动磁矩大小",
    "Doping Type": "掺杂方式",
    "Element Scalings": "元素缩放",
    "Element Scaling": "元素缩放",
    "Engine Type": "生成方式",
    "Exclude Near Zero": "排除近零频率",
    "Existing Dataset Path": "已有数据集路径",
    "Fixed Axis Flags": "固定轴选择",
    "Fixed Axis Scale": "固定轴倍率",
    "Flex Gaussian Sigma": "柔性采样高斯宽度",
    "Flex Max Torsions": "柔性采样最大扭转键数",
    "Flex Solvent": "启用柔性溶剂",
    "Flex Torsion Range": "柔性扭转角范围",
    "Gamma Range": "γ 角范围",
    "H Range": "Miller 指数 h 范围",
    "Include Reference": "包含参考构型",
    "K Range": "Miller 指数 k 范围",
    "L Range": "Miller 指数 l 范围",
    "Lattice Range": "晶格常数范围",
    "Lift Scalar": "标量磁矩转为矢量",
    "Local Cutoff": "局部截断距离",
    "Local Subtree": "仅旋转局部子树",
    "Magnitude Factor": "磁矩缩放系数",
    "Magmom Map": "元素磁矩表",
    "Max Angle": "最大旋转角",
    "Max Atoms For Full": "完整协方差最大原子数",
    "Max Attempts": "最大尝试次数",
    "Max Attempts Per Atom": "每个原子最大尝试次数",
    "Max Attempts Per Solvent": "每个溶剂分子最大尝试次数",
    "Max Num": "最大输出数量",
    "Max Retries": "最大重试次数",
    "Max Structures": "最大结构数",
    "Max Volume Per Atom": "最大单原子体积",
    "Min Box": "最小晶胞尺寸",
    "Min Volume Per Atom": "最小单原子体积",
    "Mult Bond Factor": "多重键系数",
    "Mz": "z 方向磁矩分量",
    "Nonbond Min Factor": "非键原子最小距离系数",
    "Nonpbc Box Size": "非周期晶胞尺寸",
    "Num Structures": "生成结构数",
    "Only Commensurate Periods": "仅生成公度周期",
    "PBC Mode": "周期边界模式",
    "Pair Element Filter": "原子对元素筛选",
    "Pair Group Filter": "原子对分组筛选",
    "Pair Min Distances": "分元素对最小距离",
    "Pair Shell": "近邻壳层",
    "Pair Shell Tolerance": "近邻壳层容差",
    "Plane Normal": "平面法向",
    "Reference Direction": "参考方向",
    "Rep": "重复倍数",
    "Samples Per Fraction": "每个无序比例的样本数",
    "Scale BY Frequency": "按频率缩放",
    "Seed": "随机种子",
    "Sequence Mode": "序列方式",
    "Spiral Parameter Mode": "螺旋参数方式",
    "Symmetric": "对称剪切",
    "Target Cell": "目标晶胞长度",
    "Target Elements": "目标元素",
    "Target Indices": "目标原子索引",
    "Target Mode": "目标选择方式",
    "Use Element Dirs": "使用元素方向",
    "Use Element Scaling": "启用缩放",
    "Enable Scaling": "启用缩放",
    "Use Num": "按数量设置",
    "Vacuum Range": "真空层范围",
    "Volume Scale Range": "体积缩放范围",
    "X Range": "X 轴范围",
    "XY Range": "XY 剪切范围",
    "XZ Range": "XZ 剪切范围",
    "Y Range": "Y 轴范围",
    "YZ Range": "YZ 剪切范围",
    "Z Range": "Z 轴范围",
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _translations() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    root = ET.parse(TRANSLATIONS).getroot()
    normalized: dict[str, str] = {}
    exact: dict[str, str] = {}
    catalog: dict[str, str] = {}
    for context in root.findall("context"):
        context_name = context.findtext("name")
        for message in context.findall("message"):
            source = message.findtext("source")
            translation = message.findtext("translation")
            node = message.find("translation")
            if (
                not source
                or not translation
                or node is None
                or node.get("type") in {"vanished", "obsolete", "unfinished"}
            ):
                continue
            normalized.setdefault(_normalize(source), translation)
            exact.setdefault(source, translation)
            if context_name == "CardCatalog":
                catalog[source] = translation
    return normalized, exact, catalog


def _catalog_sources() -> dict[str, str]:
    module = ast.parse(CATALOG_CODE.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "names_and_descriptions" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        result: dict[str, str] = {}
        for key_node, value_node in zip(node.value.keys, node.value.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(value_node, ast.Tuple):
                continue
            name_call = value_node.elts[0]
            if (
                isinstance(name_call, ast.Call)
                and len(name_call.args) >= 2
                and isinstance(name_call.args[1], ast.Constant)
            ):
                result[str(key_node.value)] = str(name_call.args[1].value)
        return result
    raise RuntimeError("Could not parse card catalog")


def _card_class(source_file: str) -> str:
    module = ast.parse((ROOT / source_file).read_text(encoding="utf-8-sig"))
    return next(node.name for node in module.body if isinstance(node, ast.ClassDef))


def localize() -> None:
    normalized, exact_translations, catalog_translations = _translations()
    catalog_sources = _catalog_sources()
    group_translations = {
        "Alloy": "合金与组分",
        "Container": "工作流",
        "Defect": "缺陷",
        "Filter": "筛选与采样",
        "Lattice": "晶格",
        "Magnetism": "磁性",
        "Organic": "分子与溶剂",
        "Perturbation": "扰动",
        "Structure": "结构",
        "Surface": "表面",
    }

    for path in sorted(CARD_DOCS.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        schema = SCHEMA_RE.search(text)
        if not schema:
            continue
        class_name = _card_class(schema.group(1))
        english_name = catalog_sources.get(class_name)
        if english_name:
            chinese_name = catalog_translations.get(english_name, english_name)
            title = chinese_name if chinese_name == english_name else f"{chinese_name}（{english_name}）"
            text = re.sub(r"^# .+$", f"# {title}", text, count=1, flags=re.MULTILINE)

        metadata = re.search(r"^`Group`: `([^`]+)` \| `Class`: `[^`]+`[ \t]*$", text, re.MULTILINE)
        if metadata:
            group = metadata.group(1)
            label = f"**分类：** {group_translations.get(group, group)}"
            text = text[: metadata.start()] + label + text[metadata.end() :]

        def replace_heading(match: re.Match[str]) -> str:
            marks, label, key = match.groups()
            if re.search(r"[\u4e00-\u9fff]", label):
                return match.group(0)
            translated = MANUAL_LABELS.get(label) or normalized.get(_normalize(label))
            if not translated:
                raise RuntimeError(f"{path.name}: missing Chinese label for {label!r}")
            return f"{marks} {translated.rstrip('：:')}（{key}）"

        text = PARAM_HEADING_RE.sub(replace_heading, text)

        # Prose and examples should use the same Chinese labels as the card UI.
        # Keep serialized keys in parentheses so saved JSON remains searchable.
        parameter_labels = {
            key: label.rstrip("：:")
            for _marks, label, key in PARAM_HEADING_RE.findall(text)
        }
        aliases_by_key: dict[str, set[str]] = {key: {key} for key in parameter_labels}
        for key, chinese_label in parameter_labels.items():
            aliases_by_key[key].update(
                source.rstrip("：:")
                for source, translation in exact_translations.items()
                if translation.rstrip("：:") == chinese_label
            )
            aliases_by_key[key].update(
                source
                for source, translation in MANUAL_LABELS.items()
                if translation.rstrip("：:") == chinese_label
            )

        fenced_parts = re.split(r"(^```.*?^```\s*$)", text, flags=re.MULTILINE | re.DOTALL)
        for index in range(0, len(fenced_parts), 2):
            prose = fenced_parts[index]
            for key, aliases in aliases_by_key.items():
                replacement = f"`{parameter_labels[key]}`（`{key}`）"
                for alias in sorted(aliases, key=len, reverse=True):
                    prose = re.sub(
                        rf"(?<!（){re.escape(f'`{alias}`')}(?!）)",
                        replacement,
                        prose,
                    )
            fenced_parts[index] = prose
        text = "".join(fenced_parts)
        text = re.sub(r"^(\*\*分类：\*\* .+)\n(?!\n)", r"\1\n\n", text, flags=re.MULTILINE)
        text = re.sub(r"） (?=[\u4e00-\u9fff])", "）", text)

        text = re.sub(
            r"^(#{3,4}\s+.+?（[A-Za-z_][A-Za-z0-9_]*）)\n(?!\n)",
            r"\1\n\n",
            text,
            flags=re.MULTILINE,
        )
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    localize()
