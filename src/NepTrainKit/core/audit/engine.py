"""Training Set Audit orchestration."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from math import isfinite
from time import perf_counter
from typing import Any

from loguru import logger

from .composition import audit_composition
from .config_types import audit_config_types
from .context import build_fingerprints, resolve_audit_scope
from .data_quality import audit_data_quality
from .evidence_cache import TrainingSetEvidenceCache
from .extract import records_from_indexed_structures
from .findings import build_findings
from .label_ranges import audit_label_ranges
from .local_chemistry import audit_local_chemistry
from .magnetic_inventory import (
    MAGNETIC_ANALYSIS_STRATEGY,
    MAGNETIC_METHOD_ID,
    MAGNETIC_SCHEMA_VERSION,
    build_magnetic_inventory,
)
from .inventory import build_dataset_inventory
from .nep_cutoff import NepCutoffProfile, parse_nep_cutoff
from .pair_contacts import PairContactCollector
from .phase_inventory import (
    PHASE_ANALYSIS_STRATEGY,
    PHASE_METHOD_ID,
    PHASE_REFERENCE_BANK_ID,
    PHASE_SCHEMA_VERSION,
    build_phase_inventory,
)
from .result import AuditContext, AuditDimension, AuditResult, AuditRun, AuditStatus


def _restrict_profile_to_records(
    profile: NepCutoffProfile,
    records: list[Any],
) -> NepCutoffProfile:
    """Keep only model elements that can occur in the current audit scope."""
    present = {element for record in records for element in record.composition}
    unknown = sorted(present.difference(profile.elements))
    if unknown:
        raise ValueError(
            "The active data contains elements not declared in the NEP model: "
            + ", ".join(unknown)
        )
    selected = tuple(
        index for index, element in enumerate(profile.elements) if element in present
    )
    if not selected or len(selected) == len(profile.elements):
        return profile
    return NepCutoffProfile(
        elements=tuple(profile.elements[index] for index in selected),
        radial_cutoffs=tuple(profile.radial_cutoffs[index] for index in selected),
        angular_cutoffs=tuple(profile.angular_cutoffs[index] for index in selected),
    )


def _collect_structure_evidence(
    structure, geometry, inventory, *, include_phase=True, include_magnetic=True,
    cache=None, progress=None, timings_ms=None,
):
    count = inventory.structure_count
    can_analyze = geometry is not None and count > 0

    def collect(kind, enabled, build, **version):
        if not (can_analyze and enabled):
            return None, False

        def report(completed, _total):
            progress(completed + (count if kind == "magnetic" else 0), 2 * count)

        started = perf_counter()
        try:
            if cache is not None:
                saved = getattr(cache, f"load_{kind}")(**version)
                if saved is not None:
                    if progress is not None:
                        report(count, count)
                    return saved, True
            evidence, hit = build(report if progress is not None else None)
            if cache is not None and evidence is not None:
                getattr(cache, f"save_{kind}")(evidence)
            return evidence, hit
        except (RuntimeError, ValueError) as error:
            logger.warning("Training Set Audit {} evidence unavailable: {}", kind, error)
            return None, False
        finally:
            if timings_ms is not None:
                timings_ms[f"{kind}_inventory"] = (perf_counter() - started) * 1000.0

    phase, phase_hit = collect(
        "phase", include_phase,
        lambda report: build_phase_inventory(geometry, inventory, cache_owner=structure, progress=report),
        schema_version=PHASE_SCHEMA_VERSION,
        method_id=PHASE_METHOD_ID,
        reference_bank_id=PHASE_REFERENCE_BANK_ID,
        analysis_strategy=PHASE_ANALYSIS_STRATEGY,
    )

    def build_magnetic(report):
        if not hasattr(geometry, "source_indices"):
            return None, False
        return build_magnetic_inventory(
            geometry, inventory, getattr(structure, "all_data", ()), cache_owner=structure, progress=report,
        )

    magnetic, magnetic_hit = collect(
        "magnetic", include_magnetic, build_magnetic,
        schema_version=MAGNETIC_SCHEMA_VERSION,
        method_id=MAGNETIC_METHOD_ID,
        analysis_strategy=MAGNETIC_ANALYSIS_STRATEGY,
    )

    def summary(evidence, hit, enabled, *, spin=False):
        analyzed = evidence.analyzed_structure_count if evidence is not None else 0
        available = evidence is not None and (not spin or analyzed > 0)
        if evidence is not None:
            status = "complete" if available else "no-spin"
        else:
            status = "pending" if can_analyze and not enabled else "unavailable"
        info = dict(available=available, status=status, cache_hit=hit, analyzed_structures=analyzed)
        if spin:
            info["missing_spin_structures"] = evidence.missing_spin_count if evidence is not None else count
        return info

    return phase, magnetic, {
        "phase_inventory": summary(phase, phase_hit, include_phase),
        "magnetic_inventory": summary(magnetic, magnetic_hit, include_magnetic, spin=True),
    }


def complete_structure_evidence(result_data: Any, result: AuditResult, *, progress=None) -> AuditResult:
    """Attach deferred phase and magnetic evidence with persistent cache reuse."""
    if result.inventory is None:
        return result
    structure = result_data.structure
    geometry = structure.geometry_snapshot(result.scope.indices if result.scope is not None else None)
    phase, magnetic, overview = _collect_structure_evidence(
        structure, geometry, result.inventory,
        cache=TrainingSetEvidenceCache.from_result_data(result_data, result), progress=progress,
    )
    return replace(
        result, phase_inventory=phase, magnetic_inventory=magnetic,
        overview_metrics={**result.overview_metrics, **overview},
    )


def build_audit(context: AuditContext) -> AuditRun:
    """Build one immutable assessment run through the public core seam."""
    audit_started = perf_counter()
    timings_ms: dict[str, float] = {}
    result_data = context.dataset
    stage_started = perf_counter()
    scope, indexed_structures = resolve_audit_scope(
        result_data,
        context.scope_kind,
        context.indices,
    )
    timings_ms["scope_resolution"] = (perf_counter() - stage_started) * 1000.0

    stage_started = perf_counter()
    records = records_from_indexed_structures(indexed_structures)
    timings_ms["record_extraction"] = (perf_counter() - stage_started) * 1000.0

    stage_started = perf_counter()
    inventory = build_dataset_inventory(records)
    timings_ms["inventory"] = (perf_counter() - stage_started) * 1000.0

    geometry = None
    geometry_snapshot = getattr(getattr(result_data, "structure", None), "geometry_snapshot", None)
    stage_started = perf_counter()
    if callable(geometry_snapshot):
        try:
            geometry = geometry_snapshot(scope.indices)
        except (IndexError, ValueError):
            geometry = None
    timings_ms["geometry_snapshot"] = (perf_counter() - stage_started) * 1000.0
    phase_inventory, magnetic_inventory, evidence_overview = _collect_structure_evidence(
        getattr(result_data, "structure", None), geometry, inventory,
        include_phase=context.include_phase_inventory, include_magnetic=context.include_magnetic_inventory,
        timings_ms=timings_ms,
    )
    dimensions = []
    slices = []
    overview: dict[str, object] = {"structures": len(records), **evidence_overview}

    stage_started = perf_counter()
    quality_dimension, quality_slices, quality_overview = audit_data_quality(
        indexed_structures,
        result_data=result_data,
    )
    timings_ms["data_quality"] = (perf_counter() - stage_started) * 1000.0
    dimensions.append(quality_dimension)
    slices.extend(quality_slices)
    overview[quality_dimension.id] = quality_overview

    for dimension_id, run_dimension in (
        ("composition", audit_composition),
        ("config_types", audit_config_types),
        ("label_ranges", audit_label_ranges),
    ):
        stage_started = perf_counter()
        dimension, dimension_slices, dimension_overview = run_dimension(records)
        timings_ms[dimension_id] = (perf_counter() - stage_started) * 1000.0
        dimensions.append(dimension)
        slices.extend(dimension_slices)
        overview[dimension.id] = dimension_overview

    local_overview: dict[str, object] = {
        "available_scopes": (),
        "center_element_count": 0,
        "sparse_bin_count": 0,
    }
    pair_overview: dict[str, object] = {
        "pair_count": 0,
        "co_occurring_pair_count": 0,
        "zero_contact_pair_count": 0,
    }
    declared_model_elements: tuple[str, ...] = ()
    analyzed_model_elements: tuple[str, ...] = ()
    absent_model_elements: tuple[str, ...] = ()
    model_path = getattr(result_data, "nep_txt_path", None)
    if model_path is None or str(model_path).strip() == "":
        local_dimension = AuditDimension(
            "local_chemistry",
            "Local chemistry",
            AuditStatus.UNAVAILABLE,
            "No active NEP model file is available.",
        )
        local_slices = ()
    else:
        try:
            stage_started = perf_counter()
            declared_profile = parse_nep_cutoff(model_path)
            timings_ms["model_cutoff_parse"] = (perf_counter() - stage_started) * 1000.0
            stage_started = perf_counter()
            profile = _restrict_profile_to_records(declared_profile, records)
            timings_ms["model_element_filter"] = (perf_counter() - stage_started) * 1000.0
            declared_model_elements = declared_profile.elements
            analyzed_model_elements = profile.elements
            analyzed_element_set = set(analyzed_model_elements)
            absent_model_elements = tuple(
                element
                for element in declared_model_elements
                if element not in analyzed_element_set
            )
            pair_collector = PairContactCollector(profile)
            stage_started = perf_counter()
            local_dimension, local_slices, local_overview = audit_local_chemistry(
                indexed_structures,
                profile,
                pair_contact_collector=pair_collector,
                geometry=geometry,
            )
            timings_ms["local_chemistry"] = (perf_counter() - stage_started) * 1000.0
            stage_started = perf_counter()
            pair_dimension, pair_slices, pair_overview = pair_collector.finalize()
            timings_ms["pair_contacts_finalize"] = (perf_counter() - stage_started) * 1000.0
            local_overview = dict(local_overview)
            local_overview.update(
                {
                    "declared_model_elements": declared_model_elements,
                    "analyzed_model_elements": analyzed_model_elements,
                    "absent_model_elements": absent_model_elements,
                }
            )
        except ValueError as exc:
            local_dimension = AuditDimension(
                "local_chemistry",
                "Local chemistry",
                AuditStatus.UNAVAILABLE,
                str(exc) or "The local chemistry input is invalid.",
            )
            local_slices = ()
            pair_dimension = AuditDimension("pair_contacts", "Pair contacts", AuditStatus.UNAVAILABLE, str(exc))
            pair_slices = ()
        except OSError:
            local_dimension = AuditDimension(
                "local_chemistry",
                "Local chemistry",
                AuditStatus.UNAVAILABLE,
                "The active NEP model file could not be read.",
            )
            local_slices = ()
            pair_dimension = AuditDimension("pair_contacts", "Pair contacts", AuditStatus.UNAVAILABLE, "The active NEP model file could not be read.")
            pair_slices = ()
        except Exception:
            local_dimension = AuditDimension(
                "local_chemistry",
                "Local chemistry",
                AuditStatus.UNAVAILABLE,
                "Local chemistry could not be audited from the active data.",
            )
            local_slices = ()
            pair_dimension = AuditDimension("pair_contacts", "Pair contacts", AuditStatus.UNAVAILABLE, "Pair contacts could not be audited from the active data.")
            pair_slices = ()
    if model_path is None or str(model_path).strip() == "":
        pair_dimension = AuditDimension(
            "pair_contacts",
            "Pair contacts",
            AuditStatus.UNAVAILABLE,
            "No active NEP model file is available.",
        )
        pair_slices = ()
    dimensions.append(local_dimension)
    slices.extend(local_slices)
    overview[local_dimension.id] = local_overview
    dimensions.append(pair_dimension)
    slices.extend(pair_slices)
    overview[pair_dimension.id] = pair_overview

    counts = Counter(audit_slice.severity for audit_slice in slices)
    overview.update(
        {
            "finding_count": len(slices),
            "severity_counts": {severity.value: count for severity, count in counts.items()},
            "label_counts": {
                "energy": sum(record.energy_per_atom is not None and isfinite(record.energy_per_atom) for record in records),
                "force": sum(record.max_force is not None and isfinite(record.max_force) for record in records),
                "virial": sum(record.virial_norm is not None and isfinite(record.virial_norm) for record in records),
            },
        }
    )

    source_path = getattr(result_data, "data_xyz_path", "")
    stage_started = perf_counter()
    fingerprints = build_fingerprints(result_data, scope)
    timings_ms["fingerprints"] = (perf_counter() - stage_started) * 1000.0

    stage_started = perf_counter()
    audit_run = AuditResult(
        dataset_id=context.dataset_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs={
            "structure_count": len(records),
            "source_structure_count": scope.source_count,
            "scope": scope.kind.value,
            "source_path": str(source_path) if source_path else "",
            "ruleset_version": context.ruleset_version,
        },
        dimensions=tuple(dimensions),
        slices=tuple(slices),
        overview_metrics=overview,
        scope=scope,
        fingerprints=fingerprints,
        ruleset_version=context.ruleset_version,
        inventory=inventory,
        phase_inventory=phase_inventory,
        magnetic_inventory=magnetic_inventory,
    )
    timings_ms["result_assembly"] = (perf_counter() - stage_started) * 1000.0

    stage_started = perf_counter()
    result = replace(audit_run, findings=build_findings(audit_run))
    timings_ms["findings"] = (perf_counter() - stage_started) * 1000.0
    timings_ms["total"] = (perf_counter() - audit_started) * 1000.0

    preparation_keys = ("scope_resolution", "record_extraction", "inventory", "geometry_snapshot")
    finalization_keys = ("fingerprints", "result_assembly", "findings")
    timing_summary = {
        "total": round(timings_ms["total"], 3),
        "preparation": round(sum(timings_ms.get(key, 0.0) for key in preparation_keys), 3),
        "finalization": round(sum(timings_ms.get(key, 0.0) for key in finalization_keys), 3),
        "stages": {key: round(value, 3) for key, value in timings_ms.items() if key != "total"},
    }
    overview_with_timings = dict(overview)
    overview_with_timings["timings_ms"] = timing_summary
    result = replace(result, overview_metrics=overview_with_timings)
    logger.debug(
        "Training Set Audit timing: total={total:.1f} ms | {stages}",
        total=timings_ms["total"],
        stages=" | ".join(
            f"{key}={value:.1f} ms"
            for key, value in sorted(
                ((key, value) for key, value in timings_ms.items() if key != "total"),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    )
    for section_name in ("data_quality", "local_chemistry"):
        section = overview.get(section_name, {})
        section_timing = section.get("timings_ms", {}) if isinstance(section, dict) else {}
        section_stages = section_timing.get("stages", {}) if isinstance(section_timing, dict) else {}
        if isinstance(section_stages, dict) and section_stages:
            logger.debug(
                "Training Set Audit {section} timing: {stages}",
                section=section_name,
                stages=" | ".join(
                    f"{key}={float(value):.1f} ms"
                    for key, value in sorted(
                        section_stages.items(),
                        key=lambda item: float(item[1]),
                        reverse=True,
                    )
                ),
            )
    return result


def build_training_set_audit(
    result_data: Any,
    *,
    dataset_id: str = "current",
    include_phase_inventory: bool = True,
    include_magnetic_inventory: bool = True,
) -> AuditRun:
    """Compatibility adapter for callers that audit the active dataset."""
    return build_audit(
        AuditContext(
            dataset=result_data,
            dataset_id=dataset_id,
            include_phase_inventory=include_phase_inventory,
            include_magnetic_inventory=include_magnetic_inventory,
        )
    )
