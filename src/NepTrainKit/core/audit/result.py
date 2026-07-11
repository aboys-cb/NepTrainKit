"""Data containers for Training Set Audit results."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AuditSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AuditStatus(str, Enum):
    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class AuditBiasType(str, Enum):
    IMBALANCE = "imbalance"
    SPARSITY = "sparsity"
    REDUNDANCY = "redundancy"
    RISK_CONCENTRATION = "risk_concentration"
    INFORMATIONAL = "informational"


@dataclass(frozen=True)
class SliceMetric:
    name: str
    value: Any
    unit: str = ""
    baseline: Any | None = None
    direction: str = ""


@dataclass(frozen=True)
class AuditSlice:
    id: str
    title: str
    dimension_id: str
    severity: AuditSeverity
    bias_type: AuditBiasType
    structure_indices: tuple[int, ...]
    observed: str
    interpretation: str
    limit: str
    metrics: tuple[SliceMetric, ...] = ()


@dataclass(frozen=True)
class AuditDimension:
    id: str
    title: str
    status: AuditStatus
    reason: str = ""
    plots: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class AuditResult:
    dataset_id: str
    generated_at: str
    inputs: dict[str, Any]
    dimensions: tuple[AuditDimension, ...] = ()
    slices: tuple[AuditSlice, ...] = ()
    overview_metrics: dict[str, Any] = field(default_factory=dict)
