"""Persistent user presets for composite structure-filter conditions."""

from __future__ import annotations

import json
import uuid

from NepTrainKit.config import Config
from NepTrainKit.core.types import (
    FilterField,
    FilterLogic,
    StructureFilterCondition,
    StructureFilterSpec,
    TextMatchMode,
)


STRUCTURE_FILTER_PRESET_SECTION = "structure_filter_preset"
STRUCTURE_FILTER_PRESET_VERSION = 1
_MAX_PRESET_NAME_LENGTH = 80


def _normalise_name(name: str) -> str:
    value = str(name or "").strip()
    if not value:
        raise ValueError("Preset name cannot be empty.")
    if len(value) > _MAX_PRESET_NAME_LENGTH:
        raise ValueError(f"Preset name cannot exceed {_MAX_PRESET_NAME_LENGTH} characters.")
    return value


def _spec_payload(spec: StructureFilterSpec) -> dict:
    if spec.is_empty():
        raise ValueError("Add at least one enabled filter condition before saving.")
    conditions = []
    for condition in spec.conditions:
        values = [str(value).strip() for value in condition.text_values if str(value).strip()]
        if not values:
            raise ValueError("Complete or remove empty filter conditions before saving.")
        conditions.append(
            {
                "field": condition.field.value,
                "enabled": bool(condition.enabled),
                "text_values": values,
                "match_mode": condition.match_mode.value if condition.match_mode is not None else None,
                "case_sensitive": bool(condition.case_sensitive),
            }
        )
    return {
        "version": STRUCTURE_FILTER_PRESET_VERSION,
        "logic": spec.logic.value,
        "conditions": conditions,
    }


def _spec_from_payload(data: dict) -> StructureFilterSpec:
    version = data.get("version")
    if type(version) is not int or version != STRUCTURE_FILTER_PRESET_VERSION:
        raise ValueError("Unsupported structure-filter preset version.")
    raw_conditions = data.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("The structure-filter preset has no conditions.")
    conditions = []
    for raw in raw_conditions:
        if not isinstance(raw, dict):
            raise ValueError("Invalid structure-filter preset condition.")
        field = FilterField(raw["field"])
        enabled = raw.get("enabled")
        values = raw.get("text_values")
        case_sensitive = raw.get("case_sensitive")
        if not isinstance(enabled, bool) or not isinstance(case_sensitive, bool):
            raise ValueError("Invalid structure-filter preset flags.")
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError("Invalid structure-filter preset values.")
        mode = raw.get("match_mode")
        if mode is not None and field not in {FilterField.CONFIG_TYPE, FilterField.FORMULA}:
            raise ValueError("Invalid match mode for structure-filter field.")
        conditions.append(
            StructureFilterCondition(
                condition_id=str(uuid.uuid4()),
                field=field,
                enabled=enabled,
                text_values=tuple(value.strip() for value in values),
                match_mode=TextMatchMode(mode) if mode else None,
                case_sensitive=case_sensitive,
            )
        )
    spec = StructureFilterSpec(
        conditions=tuple(conditions),
        logic=FilterLogic(data.get("logic", FilterLogic.ALL.value)),
    )
    _spec_payload(spec)
    return spec


def list_structure_filter_preset_names() -> list[str]:
    """Return saved preset names in a predictable, case-insensitive order."""
    return sorted(Config.list_options(STRUCTURE_FILTER_PRESET_SECTION), key=str.casefold)


def structure_filter_preset_exists(name: str) -> bool:
    """Return whether a preset with this exact display name exists."""
    try:
        value = _normalise_name(name)
    except ValueError:
        return False
    return value in Config.list_options(STRUCTURE_FILTER_PRESET_SECTION)


def save_structure_filter_preset(name: str, spec: StructureFilterSpec) -> None:
    """Save only semantic filter conditions; transient row identifiers are omitted."""
    value = _normalise_name(name)
    payload = _spec_payload(spec)
    Config.set(
        STRUCTURE_FILTER_PRESET_SECTION,
        value,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def load_structure_filter_preset(name: str) -> StructureFilterSpec | None:
    """Load a preset with fresh row identifiers, or return ``None`` if invalid."""
    try:
        value = _normalise_name(name)
        raw = Config.get(STRUCTURE_FILTER_PRESET_SECTION, value)
        if not raw:
            return None
        data = json.loads(str(raw))
        if not isinstance(data, dict):
            return None
        return _spec_from_payload(data)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def rename_structure_filter_preset(old_name: str, new_name: str) -> bool:
    """Rename a preset without changing its serialized conditions."""
    old_value = _normalise_name(old_name)
    new_value = _normalise_name(new_name)
    raw = Config.get(STRUCTURE_FILTER_PRESET_SECTION, old_value)
    if not raw:
        return False
    if old_value == new_value:
        return True
    Config.set(STRUCTURE_FILTER_PRESET_SECTION, new_value, raw)
    return Config.delete(STRUCTURE_FILTER_PRESET_SECTION, old_value) > 0


def delete_structure_filter_preset(name: str) -> bool:
    """Delete one saved preset."""
    try:
        value = _normalise_name(name)
    except ValueError:
        return False
    return Config.delete(STRUCTURE_FILTER_PRESET_SECTION, value) > 0
