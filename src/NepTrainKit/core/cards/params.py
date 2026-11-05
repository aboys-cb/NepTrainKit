"""Rebuild frozen card parameters from exported JSON without Qt.

Card configuration JSON stores ``params`` as plain JSON: tuples became lists,
integers may arrive as floats, and nested rule dictionaries keep their shape.
The UI reconstructs each dataclass by hand in ``from_dict``; headless callers
need the same result without importing widgets.

The mapping is derived from the dataclass type annotations instead of one
hand-written adapter per card. Every registered parameter dataclass in the
application round-trips through :func:`params_from_dict` unchanged.
"""

from __future__ import annotations

import collections.abc
import types
import typing
from dataclasses import fields, is_dataclass
from typing import Any

from .errors import CardOperationError

__all__ = ["UnsupportedCardFormatError", "params_from_dict"]

_SEQUENCE_ORIGINS = (
    collections.abc.Sequence,
    collections.abc.MutableSequence,
    tuple,
)
_MAPPING_ORIGINS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)


class UnsupportedCardFormatError(CardOperationError):
    """Raised when a card payload uses a format only the UI can migrate."""

    def __init__(self, card: str, detail: str) -> None:
        super().__init__(
            "card-json-unsupported-format",
            "Card {card} cannot be read from this JSON: {detail}. "
            "Open it once in the Make Dataset workbench and save it again.",
            card=card,
            detail=detail,
        )


def params_from_dict(params_type: type, payload: Any) -> Any:
    """Build a parameter dataclass from its JSON form.

    Parameters
    ----------
    params_type : type
        Frozen dataclass describing the parameters.
    payload : Any
        JSON-compatible mapping as written by the card exporter.

    Returns
    -------
    Any
        An instance of ``params_type``.

    Raises
    ------
    UnsupportedCardFormatError
        If the payload is not a mapping, carries unknown keys, or the
        dataclass cannot be constructed from it.
    """
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise UnsupportedCardFormatError(
            params_type.__name__,
            f"expected a parameter object but found {type(payload).__name__}",
        )
    if not is_dataclass(params_type):
        raise TypeError(f"{params_type.__name__} is not a dataclass")

    hints = _type_hints(params_type)
    accepted = {field.name for field in fields(params_type)}
    unknown = sorted(set(payload) - accepted)
    if unknown:
        raise UnsupportedCardFormatError(
            params_type.__name__,
            "parameter keys " + ", ".join(repr(key) for key in unknown) + " are not recognised",
        )

    values = {
        name: _coerce(value, hints[name]) if name in hints else value
        for name, value in payload.items()
    }
    try:
        return params_type(**values)
    except (TypeError, ValueError) as exc:
        raise UnsupportedCardFormatError(
            params_type.__name__,
            str(exc),
        ) from exc


def _type_hints(params_type: type) -> dict[str, Any]:
    """Return resolved annotations, tolerating forward references."""
    try:
        return typing.get_type_hints(params_type)
    except Exception:  # noqa: BLE001 - annotations are optional metadata
        return {}


def _coerce(value: Any, hint: Any) -> Any:
    """Convert one JSON value to the annotated Python type."""
    if hint is None or hint is Any:
        return value

    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin is list:
        if not isinstance(value, (list, tuple)):
            return value
        return [_coerce(item, args[0]) for item in value] if args else list(value)

    if origin in _SEQUENCE_ORIGINS:
        if not isinstance(value, (list, tuple)):
            return value
        if not args:
            return tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(item, args[0]) for item in value)
        if _is_homogeneous_sequence(hint):
            # ``Sequence[int]`` declares one item type for a variable-length run.
            return tuple(_coerce(item, args[0]) for item in value)
        return tuple(
            _coerce(item, item_hint)
            for item, item_hint in zip(value, args)
        )

    if origin in _MAPPING_ORIGINS:
        if not isinstance(value, dict):
            return value
        if not args:
            return dict(value)
        key_hint, value_hint = args
        return {
            _coerce(key, key_hint): _coerce(item, value_hint)
            for key, item in value.items()
        }

    if _is_union(hint):
        candidates = [item for item in args if item is not type(None)]
        # Prefer a candidate whose container shape matches the incoming JSON so
        # ``list[float] | tuple[float, float, float]`` keeps its list form.
        matching = [
            item
            for item in candidates
            if _shape_of(item) == _shape_of_value(value)
        ]
        for candidate in matching or candidates:
            try:
                return _coerce(value, candidate)
            except (TypeError, ValueError):
                continue
        return value

    if isinstance(hint, type) and hint is bool:
        return bool(value)

    return value


def _is_homogeneous_sequence(hint: Any) -> bool:
    """Report whether the annotation is a ``Sequence`` rather than a fixed tuple."""
    origin = typing.get_origin(hint)
    return origin in (collections.abc.Sequence, collections.abc.MutableSequence)


def _is_union(hint: Any) -> bool:
    """Report whether the annotation is ``Union[...]`` or PEP 604 ``X | Y``."""
    return typing.get_origin(hint) in (typing.Union, types.UnionType)


def _shape_of(hint: Any) -> str:
    """Classify a container annotation as ``list``, ``tuple``, ``dict``, or ``scalar``."""
    origin = typing.get_origin(hint)
    if origin is list:
        return "list"
    if origin is tuple or origin in (collections.abc.Sequence, collections.abc.MutableSequence):
        return "tuple"
    if origin in _MAPPING_ORIGINS:
        return "dict"
    if hint is list:
        return "list"
    if hint is tuple:
        return "tuple"
    if hint is dict:
        return "dict"
    return "scalar"


def _shape_of_value(value: Any) -> str:
    """Classify a JSON value the same way :func:`_shape_of` classifies hints."""
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, dict):
        return "dict"
    return "scalar"
