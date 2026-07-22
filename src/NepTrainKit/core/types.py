#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Common types and UI styling helpers used across the core package.

This module defines enums describing backends and modes, plus lightweight
Qt-graphics helpers for pens/brushes initialised from the config.

Examples
--------
>>> from NepTrainKit.core.types import NepBackend
>>> NepBackend.AUTO.value
'auto'
"""
import re
import sys
from dataclasses import dataclass, field
from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen, QIcon
from NepTrainKit.config import Config

if sys.version_info >= (3, 11):
    from enum import StrEnum          # 3.11+
else:
    from enum import Enum
    class StrEnum(str, Enum):         # Fallback for Python 3.10-
        pass

def mkPen(*args, **kwargs):
    """Construct a ``QPen`` from flexible arguments.

    Parameters
    ----------
    color : Any, optional
        Any value accepted by :class:`QColor`, e.g. name, hex string, RGB.
    width : float, default=1
        Line width in device-independent pixels.
    style : Qt.PenStyle, optional
        Dash/line style.
    dash : Sequence[float], optional
        Custom dash pattern.
    cosmetic : bool, default=True
        If ``True``, the pen width is independent of view transforms.

    Returns
    -------
    QPen
        A configured pen instance. For widths > 4.0 the cap style is set to
        ``RoundCap`` to avoid visual artifacts for many short segments.

    Examples
    --------
    >>> isinstance(mkPen('#f00', width=2), QPen)  # doctest: +SKIP
    True
    """
    color = kwargs.get('color', None)
    width = kwargs.get('width', 1)
    style = kwargs.get('style', None)
    dash = kwargs.get('dash', None)
    cosmetic = kwargs.get('cosmetic', True)
    hsv = kwargs.get('hsv', None)

    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, dict):
            return mkPen(**arg)
        if isinstance(arg, QPen):
            return QPen(arg)  # return a copy of this pen
        elif arg is None:
            style = Qt.PenStyle.NoPen
        else:
            color = arg
    if len(args) > 1:
        color = args

    color = QColor(color)

    pen = QPen(QBrush(color), width)
    pen.setCosmetic(cosmetic)
    if style is not None:
        pen.setStyle(style)
    if dash is not None:
        pen.setDashPattern(dash)

    if width > 4.0:
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

    return pen

class ForcesMode(StrEnum):
    """How to visualise forces in plots."""
    Raw = "Raw"
    Norm = "Norm"


def parse_forces_mode(value, fallback: ForcesMode = ForcesMode.Raw) -> ForcesMode:
    """Parse a config value into :class:`ForcesMode`.

    Accepts both canonical values (``"Raw"``, ``"Norm"``) and enum-like
    strings such as ``"ForcesMode.Norm"`` produced by ``str(ForcesMode.Norm)``.
    """
    if isinstance(value, ForcesMode):
        return value

    text = str(value or "").strip()
    if not text:
        return fallback

    try:
        return ForcesMode(text)
    except Exception:
        pass

    if "." in text:
        name = text.split(".")[-1].strip()
        if name:
            try:
                return ForcesMode[name]
            except Exception:
                pass

    match = re.search(r"ForcesMode\.([A-Za-z_]+)", text)
    if match:
        try:
            return ForcesMode[match.group(1)]
        except Exception:
            pass

    lower = text.lower()
    for mode in ForcesMode:
        if lower in {mode.value.lower(), mode.name.lower()}:
            return mode

    return fallback

class CanvasMode(StrEnum):
    """Preferred canvas backend for visualisation."""
    AUTO = "auto"
    VISPY = "vispy"
    PYQTGRAPH = "pyqtgraph"

class SearchType(StrEnum):
    """Structure search attribute family."""
    TAG = "Config_type"
    FORMULA = "formula"
    ELEMENTS = "elements"
    EXPRESSION = "expression"


class FilterLogic(StrEnum):
    """How enabled structure-filter conditions are combined."""

    ALL = "all"
    ANY = "any"


class FilterField(StrEnum):
    """Supported fields in the composite structure filter."""

    CONFIG_TYPE = "config_type"
    FORMULA = "formula"
    ELEMENT_REQUIRED = "element_required"
    ELEMENT_EXCLUDED = "element_excluded"
    ELEMENT_ALLOWED = "element_allowed"
    EXPRESSION = "expression"


class TextMatchMode(StrEnum):
    """Text comparison used by Config type and Formula conditions."""

    CONTAINS = "contains"
    EXACT = "exact"
    PREFIX = "prefix"
    SUFFIX = "suffix"
    REGEX = "regex"


@dataclass(frozen=True)
class StructureFilterCondition:
    """One independently editable structure-filter condition."""

    condition_id: str
    field: FilterField
    enabled: bool = True
    text_values: tuple[str, ...] = ()
    match_mode: TextMatchMode | None = None
    case_sensitive: bool = False

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "field": self.field.value,
            "enabled": self.enabled,
            "text_values": list(self.text_values),
            "match_mode": self.match_mode.value if self.match_mode is not None else None,
            "case_sensitive": self.case_sensitive,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructureFilterCondition":
        mode = data.get("match_mode")
        return cls(
            condition_id=str(data.get("condition_id", "")),
            field=FilterField(data.get("field", FilterField.CONFIG_TYPE.value)),
            enabled=bool(data.get("enabled", True)),
            text_values=tuple(str(value) for value in data.get("text_values", ())),
            match_mode=TextMatchMode(mode) if mode else None,
            case_sensitive=bool(data.get("case_sensitive", False)),
        )


@dataclass(frozen=True)
class StructureFilterSpec:
    """Typed composite query used as the single source of filter state."""

    conditions: tuple[StructureFilterCondition, ...] = ()
    logic: FilterLogic = FilterLogic.ALL

    def enabled_conditions(self) -> tuple[StructureFilterCondition, ...]:
        return tuple(condition for condition in self.conditions if condition.enabled)

    def is_empty(self) -> bool:
        return not any(
            condition.enabled and any(str(value).strip() for value in condition.text_values)
            for condition in self.conditions
        )

    def to_dict(self) -> dict:
        return {
            "logic": self.logic.value,
            "conditions": [condition.to_dict() for condition in self.conditions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructureFilterSpec":
        return cls(
            conditions=tuple(
                StructureFilterCondition.from_dict(condition)
                for condition in data.get("conditions", ())
            ),
            logic=FilterLogic(data.get("logic", FilterLogic.ALL.value)),
        )


@dataclass(frozen=True)
class StructureFilterResult:
    """Cached result produced by evaluating one structure-filter snapshot."""

    indices: tuple[int, ...]
    active_count: int
    dataset_version: int | None
    elapsed_ms: float
    spec: StructureFilterSpec


@dataclass
class FilterCondition:
    """A single filter condition (substring to match)."""

    text: str = ""
    negate: bool = False

    def to_dict(self) -> dict:
        return {"text": self.text, "negate": self.negate}

    @classmethod
    def from_dict(cls, d: dict) -> "FilterCondition":
        return cls(text=str(d.get("text", "")), negate=bool(d.get("negate", False)))


@dataclass
class FilterGroup:
    """A group of conditions combined with AND/OR logic. Groups are AND'd together."""

    conditions: list[FilterCondition] = field(default_factory=list)
    mode: str = "or"

    def is_empty(self) -> bool:
        return not self.conditions

    def to_dict(self) -> dict:
        return {
            "conditions": [c.to_dict() for c in self.conditions],
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FilterGroup":
        return cls(
            conditions=[FilterCondition.from_dict(c) for c in d.get("conditions", [])],
            mode=str(d.get("mode", "or")),
        )


@dataclass
class TagFilterSpec:
    """Structured filter spec for tag/formula search.

    Multiple groups are AND'd together. Within each group, conditions
    use the group's ``mode`` (AND/OR). Each condition can be negated.
    """

    groups: list[FilterGroup] = field(default_factory=list)

    def is_empty(self) -> bool:
        return all(g.is_empty() for g in self.groups)

    def to_expression(self) -> str:
        parts: list[str] = []
        for group in self.groups:
            if group.is_empty():
                continue
            cond_strs: list[str] = []
            negate_group = False
            for cond in group.conditions:
                if cond.negate:
                    negate_group = True
                cond_strs.append(cond.text)
            if not cond_strs:
                continue
            if len(cond_strs) == 1:
                inner = cond_strs[0]
            else:
                inner = " | ".join(cond_strs)
            if negate_group:
                inner = f"!({inner})" if len(cond_strs) > 1 else f"!{inner}"
            parts.append(inner)
        return ", ".join(parts) if parts else ""

    def to_dict(self) -> dict:
        return {"groups": [g.to_dict() for g in self.groups]}

    @classmethod
    def from_dict(cls, d: dict) -> "TagFilterSpec":
        groups_data = d.get("groups")
        if groups_data:
            return cls(groups=[FilterGroup.from_dict(g) for g in groups_data])
        # backward compatible: migrate old include/exclude/mode format
        include = list(d.get("include", []))
        exclude = list(d.get("exclude", []))
        mode = str(d.get("mode", "or"))
        groups = []
        if include:
            groups.append(FilterGroup(
                conditions=[FilterCondition(text=t) for t in include],
                mode=mode,
            ))
        if exclude:
            groups.append(FilterGroup(
                conditions=[FilterCondition(text=t, negate=True) for t in exclude],
                mode="or",
            ))
        return cls(groups=groups)


class FieldValueShape(StrEnum):
    """Shape category for numeric fields used by distribution analysis."""
    SCALAR = "scalar"
    VECTOR3 = "vector3"
    VECTORN = "vectorn"
    TENSOR = "tensor"


class FieldDomain(StrEnum):
    """Data domain for a field."""
    STRUCTURE = "structure"
    ATOM = "atom"


class DistributionGroupMode(StrEnum):
    """Grouping mode for distribution analysis."""
    FORMULA = "formula"
    ELEMENT = "element"
    VALUE_VIEW = "value_view"
    CUSTOM = "custom"


class DistributionValueView(StrEnum):
    """Value source to visualise in distribution analysis."""
    REFERENCE = "reference"
    PREDICTION = "prediction"
    ERROR = "error"


class DistributionScope(StrEnum):
    """Structure scope for distribution analysis."""
    ACTIVE = "active"
    SELECTED = "selected"


class DistributionSelectMode(StrEnum):
    """Selection merge policy when applying a picked histogram bin."""
    REPLACE = "replace"
    ADD = "add"
    INTERSECT = "intersect"


class DistributionCurveStyle(StrEnum):
    """Curve overlay mode for histogram plots in distribution analysis."""
    NONE = "none"
    KDE = "kde"
    NORMAL = "normal"


class NepBackend(StrEnum):
    """NEP calculator backend preference."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class DataPrecision(StrEnum):
    """Storage precision preference for imported numeric dataset values."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"


def parse_data_precision(value, fallback: DataPrecision = DataPrecision.FLOAT32) -> DataPrecision:
    """Parse a config value into :class:`DataPrecision`."""

    if isinstance(value, DataPrecision):
        return value

    text = str(value or "").strip()
    if not text:
        return fallback

    try:
        return DataPrecision(text)
    except Exception:
        pass

    if "." in text:
        name = text.split(".")[-1].strip()
        if name:
            try:
                return DataPrecision[name]
            except Exception:
                pass

    lower = text.lower()
    for precision in DataPrecision:
        if lower in {precision.value.lower(), precision.name.lower()}:
            return precision

    return fallback

class Base:
    """Mixin providing a ``get`` helper that falls back to ``Default``."""
    @classmethod
    def get(cls, name):
        if hasattr(cls, name):
            return getattr(cls, name)
        else:
            return getattr(cls, "Default")

def _get_color(section: str, option: str, default_hex: str) -> QColor:
    """Read a color from config with a safe fallback to ``default_hex``."""
    val = Config.get(section, option, default_hex)
    try:
        c = QColor(val)
        if c.isValid():
            return c
        return QColor(default_hex)
    except Exception:
        return QColor(default_hex)

class Pens(Base):
    """Convenience accessors for pens configured via the config file."""
    @classmethod
    def update_from_config(cls):
        edge = _get_color("plot", "marker_edge_color", "#07519C")
        current = _get_color("plot", "current_color", "#FF0000")
        line = _get_color("plot", "line_color", "#FF0000")
        training_overlay = _get_color("plot", "training_overlay_edge_color", "#505050")
        loaded_overlay = _get_color("plot", "loaded_overlay_edge_color", "#1450B4")

        cls.Default = mkPen(color=edge, width=0.8)
        cls.Energy = cls.Default
        cls.Force = cls.Default
        cls.Virial = cls.Default
        cls.Stress = cls.Default
        cls.Descriptor = cls.Default
        cls.TrainingOverlay = mkPen(color=training_overlay, width=0.8)
        cls.LoadedOverlay = mkPen(color=loaded_overlay, width=0.8)
        cls.Current = mkPen(color=current, width=1)
        cls.Line = mkPen(color=line, width=2)

    def __getattr__(self, item):
        return getattr(self.Default, item)

class Brushes(Base):
    """Convenience accessors for brushes configured via the config file."""
    @classmethod
    def update_from_config(cls):
        face = _get_color("plot", "marker_face_color", "#FFFFFF")
        alpha = Config.getint("plot", "marker_face_alpha", 0) or 0
        face.setAlpha(int(max(0, min(255, alpha))))

        show = _get_color("plot", "show_color", "#00FF00")
        selected = _get_color("plot", "selected_color", "#FF0000")
        current = _get_color("plot", "current_color", "#FF0000")
        reject = _get_color("plot", "reject_color", "#FF8C00")
        training_overlay = _get_color("plot", "training_overlay_color", "#A0A0A0")
        loaded_overlay = _get_color("plot", "loaded_overlay_color", "#1E78D7")

        cls.BlueBrush = QBrush(QColor(0, 0, 255))
        cls.YellowBrush = QBrush(QColor(255, 255, 0))
        cls.Default = QBrush(face)
        cls.Energy = cls.Default
        cls.Force = cls.Default
        cls.Virial = cls.Default
        cls.Stress = cls.Default
        cls.Descriptor = cls.Default
        cls.TrainingOverlay = QBrush(training_overlay)
        cls.LoadedOverlay = QBrush(loaded_overlay)
        cls.Show = QBrush(show)
        cls.Selected = QBrush(selected)
        cls.Current = QBrush(current)
        cls.Reject = QBrush(reject)

    def __getattr__(self, item):
        return getattr(self.Default, item)

class ModelTypeIcon(Base):
    """Static resource paths used for model type icons."""
    NEP=':/images/src/images/gpumd_new.png'

# Initialize pens/brushes on import
Pens.update_from_config()
Brushes.update_from_config()
