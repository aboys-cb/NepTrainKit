"""Typed composite filtering for active structures."""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from NepTrainKit.core.structure import atomic_numbers
from NepTrainKit.core.types import (
    FilterField,
    FilterLogic,
    SearchType,
    StructureFilterCondition,
    StructureFilterResult,
    StructureFilterSpec,
    TextMatchMode,
)


class StructureFilterValidationError(ValueError):
    """A filter error that can be mapped back to one editor row."""

    def __init__(self, code: str, message: str, condition_id: str | None = None):
        super().__init__(message)
        self.code = str(code)
        self.condition_id = condition_id
        self.message = str(message)


class StructureFilterEngine:
    """Evaluate :class:`StructureFilterSpec` without mutating selection state."""

    @staticmethod
    def dataset_version(result_data: Any) -> int | None:
        try:
            return int(result_data.structure.data.version)
        except Exception:
            return None

    @staticmethod
    def _values(condition: StructureFilterCondition) -> tuple[str, ...]:
        values = tuple(str(value).strip() for value in condition.text_values if str(value).strip())
        if not values:
            raise StructureFilterValidationError(
                "empty_condition",
                "The filter condition has no value.",
                condition.condition_id,
            )
        return values

    @staticmethod
    def _text_mask(
        source: list[str],
        condition: StructureFilterCondition,
        default_mode: TextMatchMode,
    ) -> np.ndarray:
        values = StructureFilterEngine._values(condition)
        mode = condition.match_mode or default_mode
        if mode not in set(TextMatchMode):
            raise StructureFilterValidationError(
                "unsupported_operator",
                f"Unsupported text match mode: {mode}",
                condition.condition_id,
            )

        haystacks = source if condition.case_sensitive else [value.casefold() for value in source]
        needles = values if condition.case_sensitive else tuple(value.casefold() for value in values)
        mask = np.zeros(len(source), dtype=bool)

        if mode == TextMatchMode.REGEX:
            flags = 0 if condition.case_sensitive else re.IGNORECASE
            patterns = []
            for value in values:
                try:
                    patterns.append(re.compile(value, flags))
                except re.error as exc:
                    raise StructureFilterValidationError(
                        "invalid_regex",
                        f"Invalid regular expression: {exc}",
                        condition.condition_id,
                    ) from exc
            for pattern in patterns:
                mask |= np.fromiter((bool(pattern.search(value)) for value in source), dtype=bool, count=len(source))
            return mask

        for needle in needles:
            if mode == TextMatchMode.CONTAINS:
                mask |= np.fromiter((needle in value for value in haystacks), dtype=bool, count=len(source))
            elif mode == TextMatchMode.EXACT:
                mask |= np.fromiter((needle == value for value in haystacks), dtype=bool, count=len(source))
            elif mode == TextMatchMode.PREFIX:
                mask |= np.fromiter((value.startswith(needle) for value in haystacks), dtype=bool, count=len(source))
            elif mode == TextMatchMode.SUFFIX:
                mask |= np.fromiter((value.endswith(needle) for value in haystacks), dtype=bool, count=len(source))
        return mask

    @staticmethod
    def _normalise_elements(structure_data: Any, condition: StructureFilterCondition) -> set[str]:
        values = StructureFilterEngine._values(condition)
        result: set[str] = set()
        for value in values:
            symbol = structure_data._normalise_element_symbol(value)
            if symbol not in atomic_numbers:
                raise StructureFilterValidationError(
                    "unknown_element",
                    f"Unknown element symbol: {value}",
                    condition.condition_id,
                )
            result.add(symbol)
        return result

    @staticmethod
    def _element_mask(result_data: Any, condition: StructureFilterCondition, active_count: int) -> np.ndarray:
        structure_data = result_data.structure
        elements = StructureFilterEngine._normalise_elements(structure_data, condition)
        count_cache = structure_data.get_element_count_cache()
        mask = np.ones(active_count, dtype=bool)

        if condition.field == FilterField.ELEMENT_REQUIRED:
            for element in elements:
                values = count_cache.get(element)
                mask &= values > 0 if values is not None else False
            return mask

        if condition.field == FilterField.ELEMENT_EXCLUDED:
            for element in elements:
                values = count_cache.get(element)
                if values is not None:
                    mask &= values == 0
            return mask

        if condition.field == FilterField.ELEMENT_ALLOWED:
            outside = np.zeros(active_count, dtype=bool)
            for element, values in count_cache.items():
                if element not in elements:
                    outside |= values > 0
            return ~outside

        raise StructureFilterValidationError(
            "unsupported_operator",
            f"Unsupported element field: {condition.field}",
            condition.condition_id,
        )

    @staticmethod
    def _condition_mask(
        result_data: Any,
        condition: StructureFilterCondition,
        active_indices: np.ndarray,
    ) -> np.ndarray:
        structures = result_data.structure.now_data
        active_count = int(active_indices.size)

        if condition.field == FilterField.CONFIG_TYPE:
            values = [str(getattr(structure, "tag", "") or "") for structure in structures]
            return StructureFilterEngine._text_mask(values, condition, TextMatchMode.CONTAINS)

        if condition.field == FilterField.FORMULA:
            values = [str(getattr(structure, "formula", "") or "") for structure in structures]
            return StructureFilterEngine._text_mask(values, condition, TextMatchMode.EXACT)

        if condition.field in {
            FilterField.ELEMENT_REQUIRED,
            FilterField.ELEMENT_EXCLUDED,
            FilterField.ELEMENT_ALLOWED,
        }:
            return StructureFilterEngine._element_mask(result_data, condition, active_count)

        if condition.field == FilterField.EXPRESSION:
            expression = StructureFilterEngine._values(condition)[0]
            try:
                matched = {int(index) for index in result_data.search_config(expression, SearchType.EXPRESSION)}
            except Exception as exc:
                raise StructureFilterValidationError(
                    "invalid_expression",
                    str(exc) or "Invalid expression.",
                    condition.condition_id,
                ) from exc
            return np.fromiter(
                (int(index) in matched for index in active_indices),
                dtype=bool,
                count=active_count,
            )

        raise StructureFilterValidationError(
            "unsupported_operator",
            f"Unsupported filter field: {condition.field}",
            condition.condition_id,
        )

    @staticmethod
    def evaluate(result_data: Any, spec: StructureFilterSpec) -> StructureFilterResult:
        """Return one immutable result for the current active dataset."""
        started = time.perf_counter()
        enabled = tuple(condition for condition in spec.conditions if condition.enabled)
        if not enabled:
            raise StructureFilterValidationError("empty_condition", "Add at least one enabled filter condition.")

        try:
            active_indices = np.asarray(result_data.structure.now_indices, dtype=np.int64).reshape(-1)
        except Exception as exc:
            raise StructureFilterValidationError("invalid_dataset", "No active structure dataset is available.") from exc

        if spec.logic == FilterLogic.ALL:
            combined = np.ones(active_indices.size, dtype=bool)
        elif spec.logic == FilterLogic.ANY:
            combined = np.zeros(active_indices.size, dtype=bool)
        else:
            raise StructureFilterValidationError("unsupported_operator", f"Unsupported filter logic: {spec.logic}")

        for condition in enabled:
            current = StructureFilterEngine._condition_mask(result_data, condition, active_indices)
            if current.shape != combined.shape:
                raise StructureFilterValidationError(
                    "invalid_dataset",
                    "Filter result does not align with active structures.",
                    condition.condition_id,
                )
            if spec.logic == FilterLogic.ALL:
                combined &= current
            else:
                combined |= current

        indices = tuple(int(value) for value in active_indices[combined].tolist())
        return StructureFilterResult(
            indices=indices,
            active_count=int(active_indices.size),
            dataset_version=StructureFilterEngine.dataset_version(result_data),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            spec=spec,
        )
