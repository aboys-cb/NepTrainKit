"""Build the canonical user-facing findings from raw audit evidence."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .result import (
    AuditAction,
    AuditBiasType,
    AuditConfidence,
    AuditFinding,
    AuditFindingKind,
    AuditResult,
    AuditSlice,
)


def _metric_value(evidence: AuditSlice, name: str) -> Any | None:
    for metric in evidence.metrics:
        if metric.name == name:
            return metric.value
    return None


def _plot_by_id(result: AuditResult, plot_id: str) -> Mapping[str, Any] | None:
    for dimension in result.dimensions:
        for plot in dimension.plots:
            if str(plot.get("id", "")) == plot_id:
                return plot
    return None


def _unique_indices(items: Sequence[AuditSlice]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(index)
                for evidence in items
                for index in evidence.structure_indices
            }
        )
    )


def _compact_bin_labels(labels: list[str]) -> str:
    if not labels:
        return "No populated low-frequency range"
    if len(labels) <= 4:
        return ", ".join(labels)
    return f"{labels[0]} to {labels[-1]} ({len(labels)} ranges)"


def _structure_count(result: AuditResult) -> int:
    return int(
        result.overview_metrics.get(
            "structures", result.inputs.get("structure_count", 0)
        )
        or 0
    )


def _actions(indices: tuple[int, ...]) -> tuple[AuditAction, ...]:
    if not indices:
        return ()
    return (
        AuditAction(
            id="show_structures",
            label=f"Show {len(indices)} structures in Dataset Display",
        ),
    )


def _label_range_finding(evidence: AuditSlice) -> AuditFinding:
    count = len(evidence.structure_indices)
    labeled = int(_metric_value(evidence, "labeled_count") or 0)
    threshold = _metric_value(evidence, "threshold")
    indices = tuple(evidence.structure_indices)

    if evidence.id == "label_ranges:force_high_tail":
        title = "Maximum-force review group (top 10%)"
        plot_id = "label_ranges:max_force"
        observed = (
            f"{count} structures have maximum force above {float(threshold):.4g} eV/Å "
            f"within {labeled} labeled structures."
            if isinstance(threshold, (int, float))
            else f"{count} structures are in the highest 10% of maximum force values."
        )
        conclusion = (
            "These structures often carry disproportionate training pressure and are "
            "worth checking for difficult environments or bad geometries."
        )
        rule = "Rank-based review set: maximum force at or above the 90th percentile."
        limit = (
            "High force can be physically intended. This is a review group, not a delete recommendation."
        )
    elif evidence.id == "label_ranges:energy_high_tail":
        title = "Energy-per-atom review group (top 5%)"
        plot_id = "label_ranges:energy_per_atom"
        observed = (
            f"{count} structures have energy per atom above {float(threshold):.6g} eV/atom "
            f"within {labeled} labeled structures."
            if isinstance(threshold, (int, float))
            else f"{count} structures are in the highest 5% of energy-per-atom values."
        )
        conclusion = (
            "This group may contain strained, defective, hot, or otherwise unusual structures."
        )
        rule = "Rank-based review set: energy per atom at or above the 95th percentile."
        limit = (
            "Absolute energy per atom may not be comparable across compositions. "
            "This ranking is not an anomaly verdict."
        )
    else:
        return _fallback_finding(evidence)

    return AuditFinding(
        id=evidence.id,
        title=title,
        dimension_id=evidence.dimension_id,
        kind=AuditFindingKind.REVIEW,
        signal_type=AuditBiasType.RISK_CONCENTRATION,
        structure_indices=indices,
        conclusion=conclusion,
        observed=observed,
        rule=rule,
        limit=limit,
        actions=_actions(indices),
        evidence_ids=(evidence.id,),
        plot_id=plot_id,
        confidence=AuditConfidence.DIRECT,
    )


def _composition_finding(
    result: AuditResult, plot_id: str, evidence_items: list[AuditSlice]
) -> AuditFinding:
    element = plot_id.split(":", 1)[1] if ":" in plot_id else ""
    indices = _unique_indices(evidence_items)
    total = _structure_count(result)
    fraction = 0.0 if total <= 0 else len(indices) / total
    plot = _plot_by_id(result, plot_id) or {}
    series_items = plot.get("series", ())
    series = series_items[0] if series_items else {}
    labels = list(series.get("bin_labels", ()))
    highlighted = list(series.get("highlighted_bins", ()))
    selected_labels = [labels[index] for index in highlighted if 0 <= index < len(labels)]
    ranges = _compact_bin_labels(selected_labels)
    return AuditFinding(
        id=plot_id,
        title=f"{element} composition has {len(evidence_items)} low-frequency ranges",
        dimension_id="composition",
        kind=AuditFindingKind.EVIDENCE,
        signal_type=AuditBiasType.SPARSITY,
        structure_indices=indices,
        observed=(
            f"These composition ranges contain {len(indices)} of {total} structures "
            f"({fraction:.1%}): {ranges}."
        ),
        conclusion=(
            "They are less common than other composition regions inside the current dataset."
        ),
        rule="Relative-frequency evidence inside the current audit scope.",
        limit=(
            "Relative sparsity matters only when the range belongs to the intended model scope."
        ),
        actions=_actions(indices),
        evidence_ids=tuple(item.id for item in evidence_items),
        plot_id=plot_id,
        confidence=AuditConfidence.DERIVED,
    )


def _local_metric_label(plot: Mapping[str, Any]) -> str:
    metric = str(plot.get("metric", "") or plot.get("id", "").rsplit(":", 1)[-1])
    if metric == "neighbor_count":
        return "Neighbor count"
    if metric.startswith("neighbor_fraction_"):
        return f"{metric.removeprefix('neighbor_fraction_')} neighbor fraction"
    return str(plot.get("title") or plot.get("id") or "Distribution")


def _local_chemistry_finding(
    result: AuditResult, group_id: str, evidence_items: list[AuditSlice]
) -> AuditFinding:
    plot_ids = list(dict.fromkeys(item.id.rsplit(":", 1)[0] for item in evidence_items))
    plots = [_plot_by_id(result, plot_id) or {} for plot_id in plot_ids]
    first_plot = plots[0] if plots else {}
    scope = str(first_plot.get("scope", ""))
    center = str(first_plot.get("center_element", ""))
    scope_text = "angular-neighbor environment" if scope == "angular" else "radial-neighbor environment"
    indices = _unique_indices(evidence_items)
    evidence_lines: list[str] = []
    for plot in plots:
        series_items = plot.get("series", ())
        series = series_items[0] if series_items else {}
        labels = list(series.get("bin_labels", ()))
        counts = list(series.get("counts", ()))
        highlighted = list(series.get("highlighted_bins", ()))
        selected_labels = [labels[index] for index in highlighted if 0 <= index < len(labels)]
        thin_count = sum(
            int(counts[index]) for index in highlighted if 0 <= index < len(counts)
        )
        environment_count = int(plot.get("environment_count", 0) or 0)
        fraction = 0.0 if environment_count <= 0 else thin_count / environment_count
        evidence_lines.append(
            f"{_local_metric_label(plot)}: {_compact_bin_labels(selected_labels)}; "
            f"{thin_count} of {environment_count} environments ({fraction:.1%})."
        )
    representative_plot_id = next(
        (plot_id for plot_id in plot_ids if plot_id.endswith(":neighbor_count")),
        plot_ids[0] if plot_ids else "",
    )
    observed = "\n".join(evidence_lines)
    if observed:
        observed += "\n"
    observed += f"These ranges occur in {len(indices)} structures."
    return AuditFinding(
        id=group_id,
        title=f"{center} {scope_text} has {len(plots)} low-frequency signals",
        dimension_id="local_chemistry",
        kind=AuditFindingKind.EVIDENCE,
        signal_type=AuditBiasType.SPARSITY,
        structure_indices=indices,
        observed=observed,
        conclusion=(
            "These environments are less common than other comparable environments in this dataset."
        ),
        rule="Relative-frequency evidence inside comparable NEP-cutoff environments.",
        limit=(
            "Relative sparsity is not a model error and is actionable only for environments relevant to use."
        ),
        actions=_actions(indices),
        evidence_ids=tuple(item.id for item in evidence_items),
        plot_id=representative_plot_id,
        confidence=AuditConfidence.DERIVED,
    )


def _pair_contact_finding(evidence: AuditSlice) -> AuditFinding:
    parts = evidence.id.split(":")
    scope = parts[1] if len(parts) > 1 else ""
    scope_text = "Angular neighbors" if scope == "angular" else "Radial neighbors"
    pair = "-".join(parts[2:4]) if len(parts) >= 4 else evidence.title
    contacts = int(_metric_value(evidence, "contact_edges") or 0)
    contact_structures = int(_metric_value(evidence, "contact_structures") or 0)
    co_occurring = int(_metric_value(evidence, "co_occurring_structures") or 0)
    indices = tuple(evidence.structure_indices)
    return AuditFinding(
        id=evidence.id,
        title=f"{pair} contact support ({scope_text})",
        dimension_id="pair_contacts",
        kind=AuditFindingKind.EVIDENCE,
        signal_type=AuditBiasType.INFORMATIONAL,
        structure_indices=indices,
        observed=(
            f"{contacts} directed cutoff contacts occur in {contact_structures} of "
            f"{co_occurring} co-occurring structures."
        ),
        conclusion=(
            "This describes pair support in the current data; it is not a sampling recommendation."
        ),
        rule="Observed element-pair contacts inside the active NEP cutoff.",
        limit=(
            "Raw contact counts depend on structure size and composition, so compare them cautiously."
        ),
        actions=_actions(indices),
        evidence_ids=(evidence.id,),
        plot_id=f"pair_contacts:{scope}",
        confidence=AuditConfidence.DIRECT,
    )


def _fallback_finding(evidence: AuditSlice) -> AuditFinding:
    kind = evidence.finding_kind or (
        AuditFindingKind.REVIEW
        if evidence.bias_type == AuditBiasType.RISK_CONCENTRATION
        else AuditFindingKind.EVIDENCE
    )
    indices = tuple(evidence.structure_indices)
    return AuditFinding(
        id=evidence.id,
        title=evidence.title,
        dimension_id=evidence.dimension_id,
        kind=kind,
        signal_type=evidence.bias_type,
        structure_indices=indices,
        observed=evidence.observed,
        conclusion=evidence.interpretation,
        rule=evidence.rule or "Rule recorded by the originating audit check.",
        limit=evidence.limit,
        actions=_actions(indices),
        evidence_ids=(evidence.id,),
        confidence=evidence.confidence,
    )


def build_findings(result: AuditResult) -> tuple[AuditFinding, ...]:
    """Consolidate raw check evidence into stable user-facing findings."""
    grouped: dict[str, list[AuditSlice]] = {}
    for evidence in result.slices:
        if evidence.dimension_id == "local_chemistry":
            group_id = ":".join(evidence.id.split(":")[:3])
        elif evidence.dimension_id == "composition":
            group_id = ":".join(evidence.id.split(":", 2)[:2])
        else:
            group_id = evidence.id
        grouped.setdefault(group_id, []).append(evidence)

    findings: list[AuditFinding] = []
    for group_id, evidence_items in grouped.items():
        first = evidence_items[0]
        if first.dimension_id == "label_ranges":
            findings.append(_label_range_finding(first))
        elif first.dimension_id == "composition" and _plot_by_id(result, group_id) is not None:
            findings.append(_composition_finding(result, group_id, evidence_items))
        elif first.dimension_id == "local_chemistry" and any(
            _plot_by_id(result, item.id.rsplit(":", 1)[0]) is not None
            for item in evidence_items
        ):
            findings.append(_local_chemistry_finding(result, group_id, evidence_items))
        elif first.dimension_id == "pair_contacts":
            findings.append(_pair_contact_finding(first))
        else:
            findings.append(_fallback_finding(first))

    priority = {
        AuditFindingKind.BLOCKER: 0,
        AuditFindingKind.REVIEW: 1,
        AuditFindingKind.EVIDENCE: 2,
        AuditFindingKind.UNAVAILABLE: 3,
    }
    findings.sort(key=lambda finding: (priority[finding.kind], finding.id))
    return tuple(findings)


def canonical_findings(result: AuditResult) -> tuple[AuditFinding, ...]:
    """Return stored canonical findings or build them for legacy fixtures."""
    return result.findings or build_findings(result)
