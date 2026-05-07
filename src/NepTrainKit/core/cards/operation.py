"""Base operation contracts for Make Dataset cards."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any


class StructureOperation(ABC):
    """Operation that transforms one structure into zero or more structures."""

    @abstractmethod
    def run_structure(self, structure, params: Any) -> list:
        """Transform a single structure."""


class DatasetOperation(ABC):
    """Operation that transforms or filters a whole dataset at once."""

    @abstractmethod
    def run_dataset(self, dataset, params: Any) -> list:
        """Transform a complete dataset."""


class GeneratorOperation(ABC):
    """Operation that generates structures without an input dataset."""

    @abstractmethod
    def generate(self, params: Any) -> list:
        """Generate a dataset."""


def params_to_dict(params: Any) -> dict[str, Any]:
    """Serialize operation parameters to a plain dictionary."""
    if params is None:
        return {}
    if is_dataclass(params):
        return asdict(params)
    if isinstance(params, dict):
        return dict(params)
    raise TypeError(f"Unsupported params type: {type(params).__name__}")
