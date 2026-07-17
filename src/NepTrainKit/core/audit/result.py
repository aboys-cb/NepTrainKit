"""UI-independent data containers for training-set assessment runs."""
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


class AuditScopeKind(str, Enum):
    ALL = "all"
    ACTIVE = "active"
    SELECTED = "selected"
    CUSTOM = "custom"


class AuditFindingKind(str, Enum):
    BLOCKER = "blocker"
    REVIEW = "review"
    EVIDENCE = "evidence"
    UNAVAILABLE = "unavailable"


class AuditConfidence(str, Enum):
    DIRECT = "direct"
    DERIVED = "derived"
    HEURISTIC = "heuristic"


class AuditFindingState(str, Enum):
    OPEN = "open"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    RESOLVED = "resolved"


class AuditTargetRelevance(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TargetSupportStatus(str, Enum):
    SUPPORTED = "supported"
    THIN = "thin"
    NO_SAMPLE = "no_sample"
    UNJUDGEABLE = "unjudgeable"


@dataclass(frozen=True)
class AuditScope:
    kind: AuditScopeKind
    indices: tuple[int, ...]
    source_count: int

    @property
    def count(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class AuditFingerprints:
    dataset: str = ""
    scope: str = ""
    model: str = ""
    target: str = ""


@dataclass(frozen=True)
class AuditAction:
    id: str
    label: str


@dataclass(frozen=True)
class AuditContext:
    dataset: Any
    dataset_id: str = "current"
    scope_kind: AuditScopeKind = AuditScopeKind.ACTIVE
    indices: tuple[int, ...] = ()
    ruleset_version: str = "quick-check-v1"
    include_phase_inventory: bool = True


@dataclass(frozen=True)
class CompositionPoint:
    """One exact normalized composition shared across compatible supercells."""

    reduced_counts: tuple[int, ...]
    fractions: tuple[float, ...]
    structure_count: int
    share: float
    structure_indices: tuple[int, ...]
    atom_counts: tuple[tuple[int, int], ...] = ()
    formula_variants: tuple[tuple[str, int], ...] = ()
    config_types: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class DatasetInventory:
    """Compact, UI-independent inventory for one audited structure scope."""

    structure_count: int
    elements: tuple[str, ...]
    composition_points: tuple[CompositionPoint, ...]
    atom_counts: tuple[tuple[int, int], ...] = ()
    config_types: tuple[tuple[str, int], ...] = ()
    missing_config_type_count: int = 0


@dataclass(frozen=True)
class StructurePhaseEvidence:
    """One analyzed structure and the evidence used for phase drill-down."""

    source_index: int
    atom_count: int
    phase_label: str
    confidence_state: str
    local_phase_fractions: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class CompositionPhaseEvidence:
    """Complete phase evidence for one exact normalized composition."""

    reduced_counts: tuple[int, ...]
    source_structure_count: int
    analyzed_structure_count: int
    analyzed_atom_count: int
    local_phase_fractions: tuple[tuple[str, float], ...]
    structure_phase_fractions: tuple[tuple[str, float], ...]
    confidence_counts: tuple[tuple[str, int], ...]
    confirmed_candidates: tuple[tuple[str, int], ...] = ()
    structures: tuple[StructurePhaseEvidence, ...] = ()


@dataclass(frozen=True)
class PhaseInventory:
    """Versioned, complete structural-phase evidence for one audit scope."""

    schema_version: str
    method_id: str
    reference_bank_id: str
    analysis_strategy: str
    source_structure_count: int
    analyzed_structure_count: int
    analyzed_atom_count: int
    composition_points: tuple[CompositionPhaseEvidence, ...]


@dataclass(frozen=True)
class PhaseEvidenceSummary:
    """Complete local phase evidence for selected composition points."""

    source_structure_count: int
    analyzed_structure_count: int
    analyzed_atom_count: int
    local_phase_fractions: tuple[tuple[str, float], ...]
    confidence_counts: tuple[tuple[str, int], ...]
    confirmed_candidates: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class CompositionTarget:
    """Small explicit target used for honest composition-count comparisons."""

    element: str
    minimum: float
    maximum: float
    key_points: tuple[float, ...] = ()
    minimum_structure_count: int | None = None
    config_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetSupportCell:
    target_fraction: float
    status: TargetSupportStatus
    observed_count: int
    structure_indices: tuple[int, ...] = ()
    nearest_fraction: float | None = None


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
    finding_kind: AuditFindingKind | None = None
    rule: str = ""
    confidence: AuditConfidence = AuditConfidence.HEURISTIC


# ``AuditSlice`` is retained as a compatibility name for the raw evidence
# emitted by individual checks. New callers should use ``AuditEvidence``.
AuditEvidence = AuditSlice


@dataclass(frozen=True)
class AuditFinding:
    id: str
    title: str
    dimension_id: str
    kind: AuditFindingKind
    signal_type: AuditBiasType
    structure_indices: tuple[int, ...]
    conclusion: str
    observed: str
    rule: str
    limit: str
    actions: tuple[AuditAction, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    plot_id: str = ""
    target_relevance: AuditTargetRelevance = AuditTargetRelevance.UNKNOWN
    confidence: AuditConfidence = AuditConfidence.DERIVED
    state: AuditFindingState = AuditFindingState.OPEN

    @property
    def interpretation(self) -> str:
        """Compatibility wording used by the existing evidence panel."""
        return self.conclusion


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
    scope: AuditScope | None = None
    fingerprints: AuditFingerprints = field(default_factory=AuditFingerprints)
    ruleset_version: str = ""
    findings: tuple[AuditFinding, ...] = ()
    inventory: DatasetInventory | None = None
    phase_inventory: PhaseInventory | None = None

    @property
    def evidence(self) -> tuple[AuditEvidence, ...]:
        return self.slices


# New code uses the product term ``AuditRun``. Keep ``AuditResult`` available
# while callers and saved tests migrate through the same interface.
AuditRun = AuditResult
