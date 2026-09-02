"""Physics-aware grouping and quota planning for representative sampling.

The module keeps the sampling interface small: callers provide structures, the
model kind, and a total budget. Element set, structural phase, magnetic
order, and hierarchical quota details stay inside the implementation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from math import gcd
from typing import Any

from NepTrainKit.core.audit.context import build_fingerprints, resolve_audit_scope
from NepTrainKit.core.audit.evidence_cache import TrainingSetEvidenceCache
from NepTrainKit.core.audit.magnetic_inventory import (
    MAGNETIC_ANALYSIS_STRATEGY,
    MAGNETIC_METHOD_ID,
    MAGNETIC_SCHEMA_VERSION,
    analyze_structure_magnetism,
    magnetic_partition_label,
)
from NepTrainKit.core.audit.phase_inventory import (
    PHASE_ANALYSIS_STRATEGY,
    PHASE_METHOD_ID,
    PHASE_REFERENCE_BANK_ID,
    PHASE_SCHEMA_VERSION,
    analyze_structure_phase,
    phase_partition_label,
)
from NepTrainKit.core.audit.result import (
    AuditScopeKind,
    MagneticInventory,
    PhaseInventory,
)

NOT_APPLICABLE_MAGNETIC_ORDER = "not_applicable"
MISSING_SPIN_ORDER = "no_spin"
PHYSICS_CLASSIFICATION_PARALLEL_THRESHOLD = 512
PHYSICS_CLASSIFICATION_WORKERS = 2
PHYSICS_CLASSIFICATION_BLOCKS_PER_WORKER = 8
PHYSICS_SAMPLING_SCHEMA_VERSION = "physics-sampling-v2"
LEGACY_PHYSICS_SAMPLING_SCHEMA_VERSION = "physics-sampling-v1"


@dataclass(frozen=True, order=True)
class PhysicsSamplingStratum:
    """One element-set, phase, and optional magnetic-order stratum."""

    element_set: tuple[str, ...]
    phase: str
    magnetic_order: str = NOT_APPLICABLE_MAGNETIC_ORDER


@dataclass(frozen=True)
class PhysicsSamplingPlan:
    """Immutable group plan and coverage inventory for one candidate pool."""

    groups: tuple[tuple[PhysicsSamplingStratum, tuple[int, ...]], ...]
    spin_model: bool
    missing_spin_indices: tuple[int, ...]
    phase_counts: tuple[tuple[str, int], ...]
    magnetic_order_counts: tuple[tuple[str, int], ...]
    source_indices: tuple[int, ...] = ()

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def element_set_count(self) -> int:
        return len({key.element_set for key, _indices in self.groups})

    def group_indices(self) -> dict[PhysicsSamplingStratum, list[int]]:
        return {key: list(indices) for key, indices in self.groups}

    def project(self, source_indices: Sequence[int]) -> PhysicsSamplingPlan:
        """Project a full-scope plan onto requested source rows."""
        requested = tuple(int(index) for index in source_indices)
        if not self.source_indices:
            if requested == tuple(range(len(requested))):
                return self
            raise ValueError("A physics sampling plan has no source-index mapping.")
        stratum_by_source: dict[int, PhysicsSamplingStratum] = {}
        for stratum, local_indices in self.groups:
            for local_index in local_indices:
                stratum_by_source[self.source_indices[local_index]] = stratum
        missing_sources = {
            self.source_indices[local_index]
            for local_index in self.missing_spin_indices
        }
        missing = [index for index in requested if index not in stratum_by_source]
        if missing:
            raise ValueError(
                "The cached physical partitions do not cover source structure "
                f"{missing[0] + 1}."
            )
        classifications = tuple(
            _StructureClassification(
                index=local_index,
                element_set=stratum_by_source[source_index].element_set,
                phase=stratum_by_source[source_index].phase,
                magnetic_order=stratum_by_source[source_index].magnetic_order,
                missing_spin=source_index in missing_sources,
            )
            for local_index, source_index in enumerate(requested)
        )
        return _plan_from_classifications(
            classifications,
            spin_model=self.spin_model,
            source_indices=requested,
        )


@dataclass(frozen=True)
class _StructureClassification:
    index: int
    element_set: tuple[str, ...]
    phase: str
    magnetic_order: str
    missing_spin: bool


def structure_symbols(structure: Any) -> tuple[str, ...]:
    """Return ordered chemical symbols for ASE or NepTrainKit structures."""
    getter = getattr(structure, "get_chemical_symbols", None)
    if callable(getter):
        symbols = getter()
    else:
        symbols = getattr(structure, "elements", ())
    return tuple(str(symbol) for symbol in symbols)


def reduced_composition_key(structure: Any) -> tuple[tuple[str, int], ...]:
    """Return exact integer stoichiometry reduced across compatible cells."""
    counts = Counter(structure_symbols(structure))
    if not counts:
        return ()
    divisor = 0
    for count in counts.values():
        divisor = gcd(divisor, int(count))
    divisor = max(1, divisor)
    return tuple(
        (element, int(count) // divisor)
        for element, count in sorted(counts.items())
    )


def element_set_key(structure: Any) -> tuple[str, ...]:
    """Return the stable set of chemical elements present in a structure."""
    return tuple(sorted(set(structure_symbols(structure))))


def _classify_structure(
    item: tuple[int, Any],
    *,
    spin_model: bool,
) -> _StructureClassification:
    index, structure = item
    elements = element_set_key(structure)
    if not elements:
        raise ValueError(f"Structure {index + 1} contains no chemical symbols.")
    phase = phase_partition_label(
        analyze_structure_phase(structure, source_index=index)
    )

    magnetic_order = NOT_APPLICABLE_MAGNETIC_ORDER
    missing_spin = False
    if spin_model:
        magnetic_evidence = analyze_structure_magnetism(
            structure,
            source_index=index,
        )
        if magnetic_evidence is None:
            magnetic_order = MISSING_SPIN_ORDER
            missing_spin = True
        else:
            magnetic_order = magnetic_partition_label(magnetic_evidence)
    return _StructureClassification(
        index=index,
        element_set=elements,
        phase=phase,
        magnetic_order=magnetic_order,
        missing_spin=missing_spin,
    )


def _classify_block(
    block: Sequence[tuple[int, Any]],
    *,
    spin_model: bool,
) -> tuple[_StructureClassification, ...]:
    return tuple(
        _classify_structure(item, spin_model=spin_model)
        for item in block
    )


def _classify_structures(
    structures: Sequence[Any],
    *,
    spin_model: bool,
) -> tuple[_StructureClassification, ...]:
    indexed = tuple(enumerate(structures))
    if len(indexed) < PHYSICS_CLASSIFICATION_PARALLEL_THRESHOLD:
        return _classify_block(indexed, spin_model=spin_model)

    block_count = min(
        len(indexed),
        PHYSICS_CLASSIFICATION_WORKERS
        * PHYSICS_CLASSIFICATION_BLOCKS_PER_WORKER,
    )
    block_size = (len(indexed) + block_count - 1) // block_count
    blocks = tuple(
        indexed[start : start + block_size]
        for start in range(0, len(indexed), block_size)
    )
    classify = partial(_classify_block, spin_model=spin_model)
    with ThreadPoolExecutor(max_workers=PHYSICS_CLASSIFICATION_WORKERS) as executor:
        classified_blocks = executor.map(classify, blocks)
        return tuple(
            classification
            for block in classified_blocks
            for classification in block
        )


def _plan_from_classifications(
    classifications: Sequence[_StructureClassification],
    *,
    spin_model: bool,
    source_indices: Sequence[int],
) -> PhysicsSamplingPlan:
    grouped: dict[PhysicsSamplingStratum, list[int]] = defaultdict(list)
    missing_spin: list[int] = []
    phases: Counter[str] = Counter()
    magnetic_orders: Counter[str] = Counter()

    for classification in classifications:
        phases[classification.phase] += 1
        if spin_model:
            if classification.missing_spin:
                missing_spin.append(classification.index)
            magnetic_orders[classification.magnetic_order] += 1
        grouped[
            PhysicsSamplingStratum(
                element_set=classification.element_set,
                phase=classification.phase,
                magnetic_order=classification.magnetic_order,
            )
        ].append(classification.index)

    return PhysicsSamplingPlan(
        groups=tuple(
            (key, tuple(indices))
            for key, indices in sorted(grouped.items())
        ),
        spin_model=bool(spin_model),
        missing_spin_indices=tuple(missing_spin),
        phase_counts=tuple(sorted(phases.items())),
        magnetic_order_counts=tuple(sorted(magnetic_orders.items())),
        source_indices=tuple(int(index) for index in source_indices),
    )


def build_physics_sampling_plan(
    structures: Sequence[Any],
    *,
    spin_model: bool,
    source_indices: Sequence[int] | None = None,
) -> PhysicsSamplingPlan:
    """Classify structures into element-set/phase/spin strata.

    Structural phase is always retained as a coverage axis. Magnetic order is
    added only for a detected spin model, using the same conservative magnetic
    classifier as Training Set Audit. Invalid or absent canonical ``spin``
    arrays are reported explicitly instead of silently falling back to lattice
    descriptors.
    """
    classifications = _classify_structures(
        structures,
        spin_model=spin_model,
    )
    sources = (
        tuple(range(len(structures)))
        if source_indices is None
        else tuple(int(index) for index in source_indices)
    )
    if len(sources) != len(structures):
        raise ValueError("source_indices must align with the structure sequence.")
    return _plan_from_classifications(
        classifications,
        spin_model=spin_model,
        source_indices=sources,
    )


def build_physics_sampling_plan_from_evidence(
    structures: Sequence[Any],
    source_indices: Sequence[int],
    *,
    spin_model: bool,
    phase_inventory: PhaseInventory,
    magnetic_inventory: MagneticInventory | None = None,
) -> PhysicsSamplingPlan:
    """Build physical strata from previously persisted audit evidence."""
    sources = tuple(int(index) for index in source_indices)
    if len(sources) != len(structures):
        raise ValueError("source_indices must align with the structure sequence.")
    phase_by_source = {
        int(evidence.source_index): evidence
        for point in phase_inventory.composition_points
        for evidence in point.structures
    }
    magnetic_by_source = (
        {
            int(evidence.source_index): evidence
            for point in magnetic_inventory.composition_points
            for evidence in point.structures
        }
        if magnetic_inventory is not None
        else {}
    )
    if spin_model and magnetic_inventory is None:
        raise ValueError("Spin-aware physical partitions require magnetic evidence.")

    classifications: list[_StructureClassification] = []
    for local_index, (source_index, structure) in enumerate(
        zip(sources, structures)
    ):
        phase_evidence = phase_by_source.get(source_index)
        if phase_evidence is None:
            raise ValueError(
                "The phase cache does not cover source structure "
                f"{source_index + 1}."
            )
        elements = element_set_key(structure)
        if not elements:
            raise ValueError(
                f"Structure {source_index + 1} contains no chemical symbols."
            )
        magnetic_order = NOT_APPLICABLE_MAGNETIC_ORDER
        missing_spin = False
        if spin_model:
            magnetic_evidence = magnetic_by_source.get(source_index)
            if magnetic_evidence is None:
                magnetic_order = MISSING_SPIN_ORDER
                missing_spin = True
            else:
                magnetic_order = magnetic_partition_label(magnetic_evidence)
        classifications.append(
            _StructureClassification(
                index=local_index,
                element_set=elements,
                phase=phase_partition_label(phase_evidence),
                magnetic_order=magnetic_order,
                missing_spin=missing_spin,
            )
        )
    return _plan_from_classifications(
        classifications,
        spin_model=spin_model,
        source_indices=sources,
    )


def _sampling_cache_identity(
    *,
    spin_model: bool,
    source_structure_count: int,
) -> dict[str, Any]:
    return {
        "sampling_schema_version": PHYSICS_SAMPLING_SCHEMA_VERSION,
        "phase_schema_version": PHASE_SCHEMA_VERSION,
        "phase_method_id": PHASE_METHOD_ID,
        "phase_reference_bank_id": PHASE_REFERENCE_BANK_ID,
        "phase_analysis_strategy": PHASE_ANALYSIS_STRATEGY,
        "magnetic_schema_version": MAGNETIC_SCHEMA_VERSION if spin_model else "",
        "magnetic_method_id": MAGNETIC_METHOD_ID if spin_model else "",
        "magnetic_analysis_strategy": (
            MAGNETIC_ANALYSIS_STRATEGY if spin_model else ""
        ),
        "spin_model": bool(spin_model),
        "source_structure_count": int(source_structure_count),
    }


def _sampling_partition_records(
    plan: PhysicsSamplingPlan,
) -> tuple[dict[str, Any], ...]:
    if len(plan.source_indices) != sum(len(indices) for _key, indices in plan.groups):
        raise ValueError("A physics sampling plan has an incomplete source mapping.")
    missing = set(plan.missing_spin_indices)
    records: list[dict[str, Any] | None] = [None] * len(plan.source_indices)
    for stratum, local_indices in plan.groups:
        for local_index in local_indices:
            records[local_index] = {
                "source_index": int(plan.source_indices[local_index]),
                "element_set": list(stratum.element_set),
                "phase": str(stratum.phase),
                "magnetic_order": str(stratum.magnetic_order),
                "missing_spin": local_index in missing,
            }
    if any(record is None for record in records):
        raise ValueError("A physics sampling plan has an incomplete partition map.")
    return tuple(record for record in records if record is not None)


def _plan_from_sampling_partition_records(
    records: Sequence[Mapping[str, Any]],
    source_indices: Sequence[int],
    *,
    spin_model: bool,
) -> PhysicsSamplingPlan:
    by_source: dict[int, Mapping[str, Any]] = {}
    for record in records:
        source_index = int(record["source_index"])
        if source_index in by_source:
            raise ValueError("The phase-sampling cache contains duplicate rows.")
        by_source[source_index] = record
    classifications: list[_StructureClassification] = []
    sources = tuple(int(index) for index in source_indices)
    for local_index, source_index in enumerate(sources):
        record = by_source.get(source_index)
        if record is None:
            raise ValueError(
                "The phase-sampling cache does not cover source structure "
                f"{source_index + 1}."
            )
        raw_elements = record.get("element_set")
        if raw_elements is None:
            raw_elements = [pair[0] for pair in record["composition"]]
        elements = tuple(sorted({str(element) for element in raw_elements}))
        if not elements:
            raise ValueError("A cached physical partition has no element set.")
        magnetic_order = (
            str(record["magnetic_order"])
            if spin_model
            else NOT_APPLICABLE_MAGNETIC_ORDER
        )
        classifications.append(
            _StructureClassification(
                index=local_index,
                element_set=elements,
                phase=str(record["phase"]),
                magnetic_order=magnetic_order,
                missing_spin=(
                    bool(record.get("missing_spin", False))
                    if spin_model
                    else False
                ),
            )
        )
    if len(by_source) != len(sources):
        raise ValueError("The phase-sampling cache source scope is inconsistent.")
    return _plan_from_classifications(
        classifications,
        spin_model=spin_model,
        source_indices=sources,
    )


def _phase_inventory_covers(
    inventory: PhaseInventory,
    source_indices: Sequence[int],
) -> bool:
    expected = {int(index) for index in source_indices}
    observed = {
        int(evidence.source_index)
        for point in inventory.composition_points
        for evidence in point.structures
    }
    return (
        inventory.source_structure_count == len(expected)
        and inventory.analyzed_structure_count == len(expected)
        and observed == expected
    )


def _magnetic_inventory_covers(
    inventory: MagneticInventory,
    source_indices: Sequence[int],
) -> bool:
    expected = {int(index) for index in source_indices}
    observed = {
        int(evidence.source_index)
        for point in inventory.composition_points
        for evidence in point.structures
    }
    return (
        inventory.source_structure_count == len(expected)
        and inventory.analyzed_structure_count + inventory.missing_spin_count
        == len(expected)
        and observed.issubset(expected)
        and len(observed) == inventory.analyzed_structure_count
    )


def _persistent_sampling_cache(
    result_data: Any,
    scope: Any,
) -> TrainingSetEvidenceCache | None:
    descriptor_path = getattr(result_data, "descriptor_path", None)
    dataset_path = getattr(result_data, "data_xyz_path", None)
    if descriptor_path is None or dataset_path is None:
        return None
    cache_enabled = getattr(result_data, "cache_outputs_enabled", None)
    if callable(cache_enabled) and not cache_enabled():
        return None
    fingerprints = build_fingerprints(result_data, scope)
    return TrainingSetEvidenceCache.from_fingerprints(
        result_data,
        dataset_fingerprint=fingerprints.dataset,
        scope_fingerprint=fingerprints.scope,
    )


def build_result_physics_sampling_plan(
    result_data: Any,
    source_indices: Sequence[int],
    *,
    spin_model: bool,
) -> PhysicsSamplingPlan | None:
    """Build, persist, and project physical partitions for one ResultData scope.

    A compact phase-sampling cache is preferred.  Existing phase and magnetic
    evidence caches are accepted as a migration source, so datasets analyzed by
    older NepTrainKit builds do not need to repeat the expensive phase pass.
    """
    structure_data = getattr(result_data, "structure", None)
    all_structures = getattr(structure_data, "all_data", None)
    if structure_data is None or all_structures is None:
        return None
    scope, indexed_structures = resolve_audit_scope(
        result_data,
        AuditScopeKind.ACTIVE,
    )
    active_sources = tuple(int(index) for index, _structure in indexed_structures)
    requested_sources = tuple(int(index) for index in source_indices)
    if not set(requested_sources).issubset(active_sources):
        return None
    version = getattr(getattr(structure_data, "data", None), "version", None)
    memory_key = (bool(spin_model), active_sources, version)
    memory_cache = getattr(
        result_data,
        "_physics_sampling_full_plan_cache",
        None,
    )
    if (
        isinstance(memory_cache, tuple)
        and len(memory_cache) == 2
        and memory_cache[0] == memory_key
        and isinstance(memory_cache[1], PhysicsSamplingPlan)
    ):
        return memory_cache[1].project(requested_sources)

    full_structures = tuple(structure for _index, structure in indexed_structures)
    persistent = _persistent_sampling_cache(result_data, scope)
    identity = _sampling_cache_identity(
        spin_model=spin_model,
        source_structure_count=len(active_sources),
    )
    full_plan = None
    loaded_partition_cache = False
    if persistent is not None:
        records = persistent.load_sampling_partitions(identity=identity)
        loaded_current_schema = records is not None
        if records is None:
            legacy_identity = {
                **identity,
                "sampling_schema_version": LEGACY_PHYSICS_SAMPLING_SCHEMA_VERSION,
            }
            records = persistent.load_sampling_partitions(identity=legacy_identity)
        if records is not None:
            try:
                full_plan = _plan_from_sampling_partition_records(
                    records,
                    active_sources,
                    spin_model=spin_model,
                )
                loaded_partition_cache = loaded_current_schema
            except (KeyError, TypeError, ValueError):
                full_plan = None

    if full_plan is None and persistent is not None:
        phase_inventory = persistent.load_phase(
            schema_version=PHASE_SCHEMA_VERSION,
            method_id=PHASE_METHOD_ID,
            reference_bank_id=PHASE_REFERENCE_BANK_ID,
            analysis_strategy=PHASE_ANALYSIS_STRATEGY,
        )
        magnetic_inventory = (
            persistent.load_magnetic(
                schema_version=MAGNETIC_SCHEMA_VERSION,
                method_id=MAGNETIC_METHOD_ID,
                analysis_strategy=MAGNETIC_ANALYSIS_STRATEGY,
            )
            if spin_model
            else None
        )
        if (
            phase_inventory is not None
            and _phase_inventory_covers(phase_inventory, active_sources)
            and (
                not spin_model
                or (
                    magnetic_inventory is not None
                    and _magnetic_inventory_covers(
                        magnetic_inventory,
                        active_sources,
                    )
                )
            )
        ):
            try:
                full_plan = build_physics_sampling_plan_from_evidence(
                    full_structures,
                    active_sources,
                    spin_model=spin_model,
                    phase_inventory=phase_inventory,
                    magnetic_inventory=magnetic_inventory,
                )
            except ValueError:
                full_plan = None

    if full_plan is None:
        full_plan = build_physics_sampling_plan(
            full_structures,
            spin_model=spin_model,
            source_indices=active_sources,
        )

    if persistent is not None and not loaded_partition_cache:
        persistent.save_sampling_partitions(
            _sampling_partition_records(full_plan),
            identity=identity,
        )
    result_data._physics_sampling_full_plan_cache = (memory_key, full_plan)
    return full_plan.project(requested_sources)


def _allocate_even(
    capacities: dict[Hashable, int],
    budget: int,
) -> dict[Hashable, int]:
    """Water-fill a budget evenly while respecting finite capacities."""
    remaining_capacity = {
        key: max(0, int(value)) for key, value in capacities.items()
    }
    allocations = {key: 0 for key in remaining_capacity}
    remaining = min(max(0, int(budget)), sum(remaining_capacity.values()))
    while remaining > 0:
        active = [
            key for key, capacity in remaining_capacity.items() if capacity > 0
        ]
        if not active:
            break
        active.sort(key=lambda key: (-remaining_capacity[key], repr(key)))
        if remaining < len(active):
            for key in active[:remaining]:
                allocations[key] += 1
                remaining_capacity[key] -= 1
            remaining = 0
            break
        share = max(1, remaining // len(active))
        granted = 0
        for key in active:
            amount = min(share, remaining_capacity[key], remaining - granted)
            allocations[key] += amount
            remaining_capacity[key] -= amount
            granted += amount
            if granted >= remaining:
                break
        if granted <= 0:
            break
        remaining -= granted
    return allocations


def allocate_physics_quotas(
    group_sizes: dict[PhysicsSamplingStratum, int],
    n_samples: int,
) -> dict[PhysicsSamplingStratum, int]:
    """Allocate one slot per physical stratum, then balance hierarchically.

    Additional capacity is distributed in the order element set -> phase ->
    magnetic order. This prevents a populous chemical system or phase from
    consuming the global budget while still preserving every observed stratum.
    """
    sizes = {key: int(size) for key, size in group_sizes.items() if int(size) > 0}
    if not sizes or int(n_samples) <= 0:
        return {}
    budget = min(int(n_samples), sum(sizes.values()))
    if budget < len(sizes):
        raise ValueError(
            f"Target count {budget} is smaller than the {len(sizes)} "
            "element-set/phase/magnetic-order strata. Increase the target count "
            "to preserve every observed physical stratum."
        )

    quotas = {key: 1 for key in sizes}
    residual = {key: sizes[key] - 1 for key in sizes}
    remaining = budget - len(sizes)

    by_element_set: dict[
        tuple[str, ...],
        list[PhysicsSamplingStratum],
    ] = defaultdict(list)
    for key in sizes:
        by_element_set[key.element_set].append(key)
    element_set_extras = _allocate_even(
        {
            element_set: sum(residual[key] for key in keys)
            for element_set, keys in by_element_set.items()
        },
        remaining,
    )

    for element_set in sorted(by_element_set):
        keys = by_element_set[element_set]
        by_phase: dict[str, list[PhysicsSamplingStratum]] = defaultdict(list)
        for key in keys:
            by_phase[key.phase].append(key)
        phase_extras = _allocate_even(
            {
                phase: sum(residual[key] for key in phase_keys)
                for phase, phase_keys in by_phase.items()
            },
            element_set_extras.get(element_set, 0),
        )
        for phase in sorted(by_phase):
            phase_keys = by_phase[phase]
            magnetic_extras = _allocate_even(
                {key: residual[key] for key in phase_keys},
                phase_extras.get(phase, 0),
            )
            for key, extra in magnetic_extras.items():
                quotas[key] += extra

    return quotas
