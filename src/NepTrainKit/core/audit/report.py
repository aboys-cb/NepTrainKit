"""Static HTML report export for Training Set Audit."""
from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Iterable

from .findings import canonical_findings
from .magnetic_inventory import (
    MAGNETIC_PARTITION_LABELS,
    magnetic_partition_label,
    summarize_magnetic_inventory,
)
from .phase_inventory import PHASE_PARTITION_LABELS, phase_partition_label
from .result import AuditFinding, AuditResult


DISCLAIMER = (
    "Findings describe this dataset only. They are not sampling instructions or global coverage claims."
)

_PHASE_ORDER = PHASE_PARTITION_LABELS
_PHASE_NAMES = {
    "fcc": "FCC",
    "bcc": "BCC",
    "hcp": "HCP",
    "diamond": "Diamond (A4)",
    "l10": "L1₀",
    "l12": "L1₂",
    "b1": "B1 (rock-salt)",
    "b2": "B2 (CsCl)",
    "b3": "B3 (zinc blende)",
    "b4": "B4 (wurtzite)",
    "fluorite": "C1 (fluorite)",
    "nias": "B8₁ (NiAs)",
    "d03": "D0₃",
    "l21": "L2₁ (full-Heusler)",
    "c1b": "C1ᵦ (half-Heusler)",
    "d019": "D0₁₉",
    "c14": "C14 Laves",
    "c15": "C15 Laves",
    "mixed": "Mixed local structure",
    "unresolved": "Unresolved",
}
_MAGNETIC_ORDER = MAGNETIC_PARTITION_LABELS
_MAGNETIC_NAMES = {
    "fm": "FM",
    "afm": "AFM",
    "afm_layered": "Layered AFM (up/down)",
    "afm_double_layered": "Double-layer AFM (up/up/down/down)",
    "ferrimagnetic": "FiM",
    "pm_like": "PM-like (spin-disordered)",
    "noncollinear": "Other noncollinear",
    "unresolved": "Unresolved magnetic type",
    "low_moment": "Low / zero moment",
    "no_spin": "No valid spin field",
}

_ELEMENT_ORDER_NAMES = {
    "aligned": "Aligned (FM-like)",
    "compensated": "Compensated (AFM-like)",
    "modulated": "Modulated / spiral-like",
    "noncollinear": "Noncollinear",
    "collinear_mixed": "Mixed collinear",
    "disordered": "Disordered-like",
    "low_moment": "Low / zero moment",
    "insufficient": "Insufficient local evidence",
}

_COUPLING_NAMES = {
    "parallel": "Parallel",
    "antiparallel": "Antiparallel",
    "mixed": "Mixed",
}


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _render_kv_list(items: Iterable[tuple[str, object]], empty_text: str) -> str:
    pairs = [(key, value) for key, value in items]
    if not pairs:
        return f"<p class=\"muted\">{escape(empty_text)}</p>"
    rows = []
    for key, value in pairs:
        rows.append(
            "<div class=\"kv-row\">"
            f"<span>{escape(str(key))}</span>"
            f"<strong>{escape(_format_value(value))}</strong>"
            "</div>"
        )
    return "\n".join(rows)


def _render_finding(finding: AuditFinding) -> str:
    actions = "".join(
        f'<li>{escape(action.label)}</li>' for action in finding.actions
    )
    return (
        '<section class="finding">'
        f"<h2>{escape(finding.title)}</h2>"
        '<div class="pill-row">'
        f'<span class="pill kind-{escape(finding.kind.value)}">{escape(finding.kind.value)}</span>'
        f'<span class="pill">{escape(finding.signal_type.value)}</span>'
        f'<span class="pill">{len(finding.structure_indices)} structures</span>'
        f'<span class="pill">confidence: {escape(finding.confidence.value)}</span>'
        "</div>"
        f"<p><strong>Observed</strong>: {escape(finding.observed)}</p>"
        f"<p><strong>Conclusion / Interpretation</strong>: {escape(finding.conclusion)}</p>"
        f"<p><strong>Rule</strong>: {escape(finding.rule)}</p>"
        f"<p><strong>Limit</strong>: {escape(finding.limit)}</p>"
        + (
            '<div class="finding-action"><strong>Next action</strong><ul>'
            f"{actions}</ul></div>"
            if actions
            else ""
        )
        + "</section>"
    )


def _render_priority_finding(finding: AuditFinding, rank: int) -> str:
    action = (
        finding.actions[0].label
        if finding.actions
        else "Inspect the affected structures and confirm whether they are intentional."
    )
    return (
        '<article class="priority-item">'
        f'<span class="priority-rank">{rank}</span>'
        '<div class="priority-copy">'
        f'<div class="priority-heading"><h3>{escape(finding.title)}</h3>'
        f'<span class="pill kind-{escape(finding.kind.value)}">{escape(finding.kind.value)}</span></div>'
        f'<p>{escape(finding.observed)}</p>'
        f'<p class="next-action"><strong>Next:</strong> {escape(action)}</p>'
        '</div></article>'
    )


def _finding_group(
    title: str,
    description: str,
    findings: tuple[AuditFinding, ...],
    *,
    open_by_default: bool,
) -> str:
    if not findings:
        return ""
    open_attribute = " open" if open_by_default else ""
    return (
        f'<details class="finding-group"{open_attribute}>'
        f'<summary><span>{escape(title)}</span><strong>{len(findings)}</strong></summary>'
        f'<p class="group-description">{escape(description)}</p>'
        + "".join(_render_finding(item) for item in findings)
        + "</details>"
    )


def _render_inventory(result: AuditResult) -> str:
    inventory = result.inventory
    if inventory is None:
        return '<p class="muted">No exact composition inventory was recorded.</p>'
    rows = []
    phase_inventory = result.phase_inventory
    phase_by_composition = (
        {
            point.reduced_counts: point
            for point in phase_inventory.composition_points
        }
        if phase_inventory is not None
        else {}
    )
    magnetic_inventory = result.magnetic_inventory
    magnetic_by_composition = (
        {point.reduced_counts: point for point in magnetic_inventory.composition_points}
        if magnetic_inventory is not None else {}
    )
    for point in sorted(
        inventory.composition_points,
        key=lambda item: item.structure_count,
        reverse=True,
    ):
        composition = " · ".join(
            f"{element} {fraction:.2%}"
            for element, fraction in zip(inventory.elements, point.fractions)
        )
        atom_counts = ", ".join(
            f"{count} atoms × {structures}" for count, structures in point.atom_counts
        ) or "—"
        phase_point = phase_by_composition.get(point.reduced_counts)
        if phase_point is None or not phase_point.structure_phase_fractions:
            phase_text = "—"
        else:
            phase_label, phase_fraction = phase_point.structure_phase_fractions[0]
            phase_text = (
                f"{_PHASE_NAMES.get(phase_label, phase_label)} {phase_fraction:.0%} "
                f"({phase_point.analyzed_structure_count}/{point.structure_count} analyzed)"
            )
        magnetic_point = magnetic_by_composition.get(point.reduced_counts)
        if magnetic_point is None or not magnetic_point.order_fractions:
            magnetic_text = "No spin:R:3"
        else:
            magnetic_label, magnetic_fraction = magnetic_point.order_fractions[0]
            magnetic_text = (
                f"{magnetic_label} {magnetic_fraction:.0%} "
                f"({magnetic_point.analyzed_structure_count}/{point.structure_count} analyzed)"
            )
        rows.append(
            "<tr>"
            f"<td>{escape(composition)}</td>"
            f"<td>{point.structure_count:,}</td>"
            f"<td>{point.share:.2%}</td>"
            f"<td>{escape(phase_text)}</td>"
            f"<td>{escape(magnetic_text)}</td>"
            f"<td>{escape(atom_counts)}</td>"
            "</tr>"
        )
    return (
        f'<p><strong>{inventory.structure_count:,}</strong> structures · '
        f'<strong>{len(inventory.composition_points)}</strong> exact composition points · '
        f'{escape(" · ".join(inventory.elements))}</p>'
        '<div class="table-wrap"><table><thead><tr>'
        '<th>Exact composition</th><th>Structures</th><th>Share</th><th>Main structural phase</th><th>Magnetic order</th><th>Atom counts</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        + (
            '<p class="muted">Phase evidence includes every audited structure and classifies local geometry; '
            'it does not predict thermodynamic stability. '
            f'Method: {escape(phase_inventory.method_id)}; reference bank: '
            f'{escape(phase_inventory.reference_bank_id)}.</p>'
            if phase_inventory is not None
            else ""
        )
        + (
            '<p class="muted">Magnetic labels use only per-atom spin:R:3 and classify snapshot patterns. '
            'They do not establish thermodynamic FM/AFM/PM stability; mforce and force_mag are excluded. '
            f'Method: {escape(magnetic_inventory.method_id)}.</p>'
            if magnetic_inventory is not None else ""
        )
    )


def _render_phase_composition_maps(result: AuditResult) -> str:
    inventory = result.inventory
    phase_inventory = result.phase_inventory
    if inventory is None or phase_inventory is None:
        return ""
    phase_by_composition = {
        point.reduced_counts: point
        for point in phase_inventory.composition_points
    }
    sections = []
    for element_index, element in enumerate(inventory.elements):
        concentration_counts: dict[float, Counter[str]] = defaultdict(Counter)
        for point in inventory.composition_points:
            phase_point = phase_by_composition.get(point.reduced_counts)
            if phase_point is None:
                continue
            concentration = round(point.fractions[element_index], 12)
            if phase_point.structures:
                concentration_counts[concentration].update(
                    phase_partition_label(structure)
                    for structure in phase_point.structures
                )
            else:
                concentration_counts[concentration].update(
                    {
                        label: round(
                            fraction * phase_point.analyzed_structure_count
                        )
                        for label, fraction in phase_point.structure_phase_fractions
                    }
                )
        rows = []
        for concentration, counts in sorted(concentration_counts.items()):
            total = sum(counts.values())
            if total <= 0:
                continue
            segments = []
            for phase in _PHASE_ORDER:
                count = counts.get(phase, 0)
                if count <= 0:
                    continue
                segments.append(
                    f'<span class="phase-segment phase-{phase}" '
                    f'style="width:{100.0 * count / total:.6f}%" '
                    f'title="{escape(_PHASE_NAMES.get(phase, phase))}: '
                    f'{count:,} / {total:,}"></span>'
                )
            rows.append(
                '<div class="phase-map-row">'
                f'<span class="phase-concentration">{concentration:.2%}</span>'
                f'<span class="phase-track">{"".join(segments)}</span>'
                f'<strong>{total:,}</strong>'
                '</div>'
            )
        if rows:
            sections.append(
                f'<details class="phase-map"{" open" if not sections else ""}>'
                f'<summary>{escape(element)} concentration</summary>'
                f'{"".join(rows)}'
                '</details>'
            )
    if not sections:
        return ""
    legend = "".join(
        f'<span><i class="phase-{phase}"></i>'
        f'{escape(_PHASE_NAMES.get(phase, phase))}</span>'
        for phase in _PHASE_ORDER
        if any(
            phase == phase_partition_label(structure)
            for point in phase_inventory.composition_points
            for structure in point.structures
        )
        or any(
            phase == label and fraction > 0
            for point in phase_inventory.composition_points
            for label, fraction in point.structure_phase_fractions
        )
    )
    return (
        '<div class="phase-map-heading"><strong>Phase labels by composition</strong>'
        '<span>Bar width = all structures at that concentration</span></div>'
        f'<div class="phase-legend">{legend}</div>'
        f'{"".join(sections)}'
    )


def _render_magnetic_composition_maps(result: AuditResult) -> str:
    inventory = result.inventory
    magnetic_inventory = result.magnetic_inventory
    if inventory is None or magnetic_inventory is None or magnetic_inventory.analyzed_structure_count <= 0:
        return ""
    magnetic_by_composition = {
        point.reduced_counts: point for point in magnetic_inventory.composition_points
    }
    sections = []
    for element_index, element in enumerate(inventory.elements):
        concentration_counts: dict[float, Counter[str]] = defaultdict(Counter)
        for point in inventory.composition_points:
            magnetic_point = magnetic_by_composition.get(point.reduced_counts)
            if magnetic_point is None:
                continue
            concentration = round(point.fractions[element_index], 12)
            concentration_counts[concentration].update(
                magnetic_partition_label(structure)
                for structure in magnetic_point.structures
            )
            concentration_counts[concentration]["no_spin"] += (
                magnetic_point.missing_spin_count
            )
        rows = []
        for concentration, counts in sorted(concentration_counts.items()):
            total = sum(counts.values())
            if total <= 0:
                continue
            segments = "".join(
                f'<span class="phase-segment magnetic-{label}" '
                f'style="width:{100.0 * count / total:.6f}%" '
                f'title="{escape(_MAGNETIC_NAMES.get(label, label))}: '
                f'{count:,} / {total:,}"></span>'
                for label in _MAGNETIC_ORDER
                for count in (counts.get(label, 0),)
                if count > 0
            )
            rows.append(
                '<div class="phase-map-row">'
                f'<span class="phase-concentration">{concentration:.2%}</span>'
                f'<span class="phase-track">{segments}</span>'
                f'<strong>{total:,}</strong></div>'
            )
        if rows:
            sections.append(
                f'<details class="phase-map"{" open" if not sections else ""}>'
                f'<summary>{escape(element)} concentration</summary>{"".join(rows)}</details>'
            )
    if not sections:
        return ""
    present = {
        magnetic_partition_label(structure)
        for point in magnetic_inventory.composition_points
        for structure in point.structures
    }
    if magnetic_inventory.missing_spin_count:
        present.add("no_spin")
    legend = "".join(
        f'<span><i class="magnetic-{label}"></i>'
        f'{escape(_MAGNETIC_NAMES.get(label, label))}</span>'
        for label in _MAGNETIC_ORDER if label in present
    )
    return (
        '<div class="phase-map-heading"><strong>Magnetic-pattern labels by composition</strong>'
        '<span>Only structures carrying spin:R:3</span></div>'
        f'<div class="phase-legend">{legend}</div>{"".join(sections)}'
    )


def _render_magnetic_cross_evidence(result: AuditResult) -> str:
    """Render structure-phase and element-local magnetic evidence together."""
    magnetic_inventory = result.magnetic_inventory
    if magnetic_inventory is None or magnetic_inventory.analyzed_structure_count <= 0:
        return ""
    magnetic_structures = tuple(
        structure
        for point in magnetic_inventory.composition_points
        for structure in point.structures
    )
    sections: list[str] = []
    if result.phase_inventory is not None and magnetic_structures:
        phase_by_index = {
            structure.source_index: phase_partition_label(structure)
            for point in result.phase_inventory.composition_points
            for structure in point.structures
        }
        joint = Counter(
            (phase_by_index[structure.source_index], magnetic_partition_label(structure))
            for structure in magnetic_structures
            if structure.source_index in phase_by_index
        )
        if joint:
            rows = "".join(
                "<tr>"
                f"<td>{escape(phase.upper())}</td>"
                f"<td>{escape(_MAGNETIC_NAMES.get(order, order))}</td>"
                f"<td>{count:,}</td>"
                "</tr>"
                for (phase, order), count in sorted(
                    joint.items(), key=lambda item: (-item[1], item[0])
                )
            )
            sections.append(
                '<div class="phase-map-heading"><strong>Magnetic order inside each structural phase</strong>'
                '<span>Matched by structure index</span></div>'
                '<div class="table-wrap"><table><thead><tr>'
                '<th>Structural phase</th><th>Magnetic pattern</th><th>Structures</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>'
            )

    summary = summarize_magnetic_inventory(magnetic_inventory)
    if summary is not None and summary.element_summaries:
        rows = "".join(
            "<tr>"
            f"<td><strong>{escape(item.element)}</strong></td>"
            f"<td>{escape(_ELEMENT_ORDER_NAMES.get(item.order_fractions[0][0], item.order_fractions[0][0]))} "
            f"{item.order_fractions[0][1]:.0%}</td>"
            f"<td>{item.structure_count:,}</td>"
            f"<td>{item.mean_moment:.3f}</td>"
            f"<td>{item.mean_net_moment_ratio:.3f}</td>"
            f"<td>{item.mean_intra_element_correlation:+.3f}</td>"
            "</tr>"
            for item in summary.element_summaries
            if item.order_fractions
        )
        if rows:
            sections.append(
                '<div class="phase-map-heading"><strong>Element-local spin patterns</strong>'
                '<span>One spin sublattice per element</span></div>'
                '<div class="table-wrap"><table><thead><tr>'
                '<th>Element</th><th>Dominant local pattern</th><th>Structures</th>'
                '<th>Mean moment</th><th>Net ratio</th><th>Same-element correlation</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>'
            )
    if summary is not None and summary.element_pair_summaries:
        rows = "".join(
            "<tr>"
            f"<td><strong>{escape(item.element_a)}–{escape(item.element_b)}</strong></td>"
            f"<td>{escape(_COUPLING_NAMES.get(item.coupling_fractions[0][0], item.coupling_fractions[0][0]))} "
            f"{item.coupling_fractions[0][1]:.0%}</td>"
            f"<td>{item.structure_count:,}</td>"
            f"<td>{item.mean_correlation:+.3f}</td>"
            "</tr>"
            for item in summary.element_pair_summaries
            if item.coupling_fractions
        )
        if rows:
            sections.append(
                '<div class="phase-map-heading"><strong>Neighboring element-pair spin coupling</strong>'
                '<span>Directional correlation, not a chemical-bond label</span></div>'
                '<div class="table-wrap"><table><thead><tr>'
                '<th>Element pair</th><th>Dominant coupling</th><th>Structures</th><th>Mean correlation</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>'
            )
    if not sections:
        return ""
    return "".join(sections) + (
        '<p class="muted">Element-local labels describe spin-sublattice patterns in the saved snapshots. '
        'They are evidence such as FM-like alignment or AFM-like compensation, not independent '
        'thermodynamic phase assignments.</p>'
    )


def render_audit_report_html(result: AuditResult) -> str:
    overview_rows = _render_kv_list(result.overview_metrics.items(), "No overview metrics were recorded.")
    inputs_rows = _render_kv_list(result.inputs.items(), "No inputs were recorded.")
    fingerprint_rows = _render_kv_list(
        (
            (key, value)
            for key, value in (
                ("dataset", result.fingerprints.dataset),
                ("scope", result.fingerprints.scope),
                ("model", result.fingerprints.model),
                ("target", result.fingerprints.target),
                ("ruleset", result.ruleset_version),
            )
            if value
        ),
        "No fingerprints were recorded.",
    )
    dimensions = result.dimensions
    dimension_rows = []
    for dimension in dimensions:
        status_line = dimension.status.value
        if dimension.reason:
            status_line = f"{status_line} - {dimension.reason}"
        dimension_rows.append(
            "<div class=\"dimension-row\">"
            f"<strong>{escape(dimension.title)}</strong>"
            f"<span>{escape(status_line)}</span>"
            "</div>"
        )
    if not dimension_rows:
        dimension_rows.append('<p class="muted">No dimensions were recorded.</p>')

    findings = canonical_findings(result)
    blockers = tuple(item for item in findings if item.kind.value == "blocker")
    reviews = tuple(item for item in findings if item.kind.value == "review")
    evidence = tuple(item for item in findings if item.kind.value == "evidence")
    unavailable = tuple(item for item in findings if item.kind.value == "unavailable")

    if blockers:
        status_class = "status-blocker"
        status_label = "Action required before training"
        status_copy = (
            f"Resolve {len(blockers)} blocking finding"
            f"{'s' if len(blockers) != 1 else ''} first, then review the remaining evidence."
        )
    elif reviews:
        status_class = "status-review"
        status_label = "Review recommended before training"
        status_copy = (
            f"No blockers were found. Check {len(reviews)} review group"
            f"{'s' if len(reviews) != 1 else ''} before deciding whether the dataset is ready."
        )
    else:
        status_class = "status-clear"
        status_label = "No blocking issues found"
        status_copy = (
            "The deterministic checks did not produce a blocker or review group. "
            "This is not a guarantee of global coverage."
        )

    structure_count = (
        result.inventory.structure_count
        if result.inventory is not None
        else int(result.overview_metrics.get("structures", result.inputs.get("structure_count", 0)) or 0)
    )
    composition_count = (
        len(result.inventory.composition_points) if result.inventory is not None else 0
    )
    phase_summary = (
        f"{result.phase_inventory.analyzed_structure_count:,} / {result.phase_inventory.source_structure_count:,}"
        if result.phase_inventory is not None
        else "Not run"
    )
    if result.magnetic_inventory is None:
        magnetic_summary = "Not run"
    elif result.magnetic_inventory.analyzed_structure_count:
        magnetic_summary = (
            f"{result.magnetic_inventory.analyzed_structure_count:,} / "
            f"{result.magnetic_inventory.source_structure_count:,}"
        )
    else:
        magnetic_summary = "No spin data"

    priorities = (blockers + reviews)[:3]
    if priorities:
        priority_html = "".join(
            _render_priority_finding(item, rank)
            for rank, item in enumerate(priorities, start=1)
        )
    else:
        priority_html = (
            '<div class="empty-state"><strong>No immediate action was generated.</strong>'
            '<span>Use the dataset map and detailed evidence below for context.</span></div>'
        )

    finding_groups = "".join(
        (
            _finding_group(
                "Required action",
                "These findings can invalidate or materially weaken the intended training set.",
                blockers,
                open_by_default=True,
            ),
            _finding_group(
                "Review next",
                "These groups deserve inspection, but are not automatic deletion recommendations.",
                reviews,
                open_by_default=False,
            ),
            _finding_group(
                "Supporting evidence",
                "Deterministic dataset observations that provide context rather than a pass/fail verdict.",
                evidence,
                open_by_default=False,
            ),
            _finding_group(
                "Unavailable checks",
                "Checks that could not be evaluated from the fields present in this dataset.",
                unavailable,
                open_by_default=False,
            ),
        )
    ) or '<p class="empty">No audit findings were generated.</p>'

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "  <title>Training Set Audit Report</title>\n"
        "  <style>\n"
        "    :root { color-scheme: light; }\n"
        "    * { box-sizing: border-box; }\n"
        "    body { margin: 0; padding: 36px 24px 64px; background: #f4f7f7; color: #16252a; font: 16px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; }\n"
        "    .report { max-width: 1080px; margin: 0 auto; }\n"
        "    h1 { margin: 0 0 8px; font-size: clamp(28px, 4vw, 38px); line-height: 1.12; letter-spacing: -.025em; }\n"
        "    h2 { margin: 0 0 10px; font-size: 21px; }\n"
        "    h3 { margin: 0; font-size: 17px; line-height: 1.35; }\n"
        "    .subtitle { margin: 0; color: #5c6c72; }\n"
        "    .report-header { display: flex; justify-content: space-between; gap: 24px; align-items: end; margin-bottom: 24px; }\n"
        "    .report-meta { color: #5c6c72; font-size: 14px; text-align: right; }\n"
        "    .card, .finding, .finding-group { background: #fff; border: 1px solid #d6e0e2; border-radius: 14px; box-shadow: 0 4px 18px rgba(29, 54, 60, .045); }\n"
        "    .card { padding: 22px; margin-bottom: 18px; }\n"
        "    .section-heading { margin-bottom: 14px; }\n"
        "    .section-heading p { margin: 4px 0 0; color: #64767c; }\n"
        "    .section-title { margin: 0 0 8px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: #5f7379; }\n"
        "    .status-banner { display: grid; grid-template-columns: auto 1fr; gap: 14px; align-items: start; padding: 20px 22px; margin-bottom: 16px; border: 1px solid; border-radius: 14px; }\n"
        "    .status-banner::before { content: \"\"; width: 12px; height: 12px; margin-top: 7px; border-radius: 50%; background: currentColor; }\n"
        "    .status-banner strong { display: block; font-size: 22px; line-height: 1.3; color: #16252a; }\n"
        "    .status-banner p { margin: 4px 0 0; color: #42565d; }\n"
        "    .status-blocker { color: #b42318; background: #fff2f0; border-color: #f0b5ae; }\n"
        "    .status-review { color: #a15c00; background: #fff8e7; border-color: #ecd391; }\n"
        "    .status-clear { color: #13795b; background: #edf9f4; border-color: #a8d9c8; }\n"
        "    .metric-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-bottom: 18px; }\n"
        "    .metric { min-height: 98px; padding: 14px; background: #fff; border: 1px solid #d6e0e2; border-radius: 12px; }\n"
        "    .metric span { display: block; min-height: 38px; color: #64767c; font-size: 13px; line-height: 1.35; }\n"
        "    .metric strong { display: block; margin-top: 7px; font-size: 22px; line-height: 1.2; }\n"
        "    .priority-list { display: grid; gap: 0; }\n"
        "    .priority-item { display: grid; grid-template-columns: 34px 1fr; gap: 14px; padding: 17px 0; border-top: 1px solid #e6edef; }\n"
        "    .priority-item:first-child { padding-top: 2px; border-top: 0; }\n"
        "    .priority-rank { display: grid; place-items: center; width: 30px; height: 30px; border-radius: 9px; background: #173f48; color: #fff; font-weight: 700; }\n"
        "    .priority-heading { display: flex; justify-content: space-between; gap: 12px; align-items: start; }\n"
        "    .priority-copy p { margin: 7px 0 0; color: #50636a; }\n"
        "    .priority-copy .next-action { color: #21383f; }\n"
        "    .empty-state { display: flex; flex-direction: column; gap: 4px; padding: 18px; background: #f3f8f7; border-radius: 10px; color: #50636a; }\n"
        "    .empty-state strong { color: #183a34; }\n"
        "    .kv-row, .metric-row, .dimension-row { display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid #edf1f4; }\n"
        "    .kv-row:last-child, .metric-row:last-child, .dimension-row:last-child { border-bottom: 0; padding-bottom: 0; }\n"
        "    .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }\n"
        "    .pill { display: inline-flex; align-items: center; border: 1px solid #d8dee4; border-radius: 999px; padding: 3px 9px; font-size: 12px; color: #334155; background: #f8fafb; }\n"
        "    .kind-blocker { background: #fee2e2; border-color: #fca5a5; color: #991b1b; }\n"
        "    .kind-review { background: #fef3c7; border-color: #fcd34d; color: #92400e; }\n"
        "    .kind-evidence { background: #e0f2fe; border-color: #7dd3fc; color: #075985; }\n"
        "    .muted { color: #64748b; }\n"
        "    .empty { color: #52606d; font-style: italic; }\n"
        "    .note { margin: 14px 0 0; padding: 11px 13px; border-left: 3px solid #2b6874; background: #eef6f7; color: #174c58; border-radius: 7px; font-size: 14px; }\n"
        "    .finding-groups { display: grid; gap: 12px; margin-bottom: 18px; }\n"
        "    .finding-group { padding: 0 20px; }\n"
        "    .finding-group > summary, .dataset-details > summary, .technical-details > summary { cursor: pointer; list-style: none; display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 18px 0; font-size: 18px; font-weight: 700; }\n"
        "    .finding-group > summary::-webkit-details-marker, .dataset-details > summary::-webkit-details-marker, .technical-details > summary::-webkit-details-marker { display: none; }\n"
        "    .finding-group > summary strong { display: grid; place-items: center; min-width: 30px; height: 26px; border-radius: 999px; background: #edf3f4; color: #3f555c; font-size: 13px; }\n"
        "    .group-description { margin: -8px 0 16px; color: #64767c; }\n"
        "    .finding { margin-bottom: 14px; padding: 18px; box-shadow: none; }\n"
        "    .finding-action { margin-top: 14px; padding: 12px 14px; background: #f1f7f7; border-radius: 8px; }\n"
        "    .finding-action ul { margin: 5px 0 0; padding-left: 20px; }\n"
        "    .dataset-details, .technical-details { padding: 0 22px 20px; }\n"
        "    .details-body { padding-top: 2px; border-top: 1px solid #e6edef; }\n"
        "    .technical-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; padding-top: 18px; }\n"
        "    .technical-block { min-width: 0; }\n"
        "    .table-wrap { overflow-x: auto; }\n"
        "    .phase-map-heading { display: flex; justify-content: space-between; gap: 12px; margin: 18px 0 8px; }\n"
        "    .phase-map-heading span { color: #64748b; font-size: 12px; }\n"
        "    .phase-legend { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; font-size: 12px; color: #52606d; }\n"
        "    .phase-legend span { display: inline-flex; align-items: center; gap: 5px; }\n"
        "    .phase-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }\n"
        "    .phase-map { margin-top: 8px; border-top: 1px solid #edf1f4; padding-top: 8px; }\n"
        "    .phase-map summary { cursor: pointer; font-weight: 600; margin-bottom: 8px; }\n"
        "    .phase-map-row { display: grid; grid-template-columns: 72px minmax(180px, 1fr) 64px; align-items: center; gap: 10px; min-height: 24px; font-size: 12px; }\n"
        "    .phase-concentration { text-align: right; color: #52606d; }\n"
        "    .phase-track { display: flex; height: 12px; overflow: hidden; background: #edf1f4; border-radius: 3px; }\n"
        "    .phase-segment { display: block; min-width: 1px; }\n"
        "    .phase-fcc { background: #159a9c; } .phase-bcc { background: #3b6fb6; } .phase-hcp { background: #e8871e; }\n"
        "    .phase-diamond { background: #546e7a; } .phase-l10 { background: #9c6ade; } .phase-l12 { background: #775da6; }\n"
        "    .phase-b1 { background: #00897b; } .phase-b2 { background: #4169a1; } .phase-b3 { background: #26a69a; }\n"
        "    .phase-b4 { background: #d9822b; } .phase-fluorite { background: #00acc1; } .phase-nias { background: #8d6e63; }\n"
        "    .phase-d03 { background: #5c6bc0; } .phase-l21 { background: #ab47bc; } .phase-c1b { background: #ec407a; }\n"
        "    .phase-d019 { background: #ff7043; } .phase-c14 { background: #2e8b57; } .phase-c15 { background: #b44c6c; }\n"
        "    .phase-mixed { background: #c08a3e; } .phase-unresolved { background: #89969a; }\n"
        "    .magnetic-fm { background:#d1495b; } .magnetic-afm { background:#3b6fb6; } .magnetic-ferrimagnetic { background:#a15c9b; }\n"
        "    .magnetic-afm_layered { background:#5b8cd0; } .magnetic-afm_double_layered { background:#244b7e; }\n"
        "    .magnetic-pm_like { background:#8a6a16; } .magnetic-noncollinear { background:#159a9c; } .magnetic-unresolved { background:#89969a; }\n"
        "    .magnetic-low_moment { background:#c8d0d2; } .magnetic-no_spin { background:#e2e8ea; }\n"
        "    table { width: 100%; border-collapse: collapse; }\n"
        "    th, td { padding: 8px; text-align: left; border-bottom: 1px solid #edf1f4; }\n"
        "    th { color: #52606d; font-size: 12px; }\n"
        "    @media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }\n"
        "    @media (max-width: 640px) { body { padding: 20px 12px 40px; } .report-header { display: block; } .report-meta { margin-top: 10px; text-align: left; } .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .technical-grid { grid-template-columns: 1fr; } .priority-heading { display: block; } .priority-heading .pill { margin-top: 8px; } .phase-map-heading { display: block; } }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        '  <main class="report">\n'
        '    <header class="report-header">\n'
        '      <div><h1>Training Set Audit Report</h1>\n'
        '      <p class="subtitle">Decision summary first; evidence and provenance remain available below.</p></div>\n'
        f'      <div class="report-meta"><strong>{escape(result.dataset_id)}</strong><br>{escape(result.generated_at)}</div>\n'
        '    </header>\n'
        f'    <section class="status-banner {status_class}"><div><strong>{escape(status_label)}</strong>'
        f'<p>{escape(status_copy)}</p></div></section>\n'
        '    <section class="metric-grid" aria-label="Audit summary">\n'
        f'      <div class="metric"><span>Structures</span><strong>{structure_count:,}</strong></div>\n'
        f'      <div class="metric"><span>Exact compositions</span><strong>{composition_count:,}</strong></div>\n'
        f'      <div class="metric"><span>Blocking findings</span><strong>{len(blockers)}</strong></div>\n'
        f'      <div class="metric"><span>Review groups</span><strong>{len(reviews)}</strong></div>\n'
        f'      <div class="metric"><span>Phase analyzed</span><strong>{escape(phase_summary)}</strong></div>\n'
        f'      <div class="metric"><span>Spin analyzed</span><strong>{escape(magnetic_summary)}</strong></div>\n'
        '    </section>\n'
        '    <section class="card">\n'
        '      <div class="section-heading"><h2>Start here</h2><p>Highest-priority checks and the next concrete action.</p></div>\n'
        f'      <div class="priority-list">{priority_html}</div>\n'
        f'      <p class="note">{escape(DISCLAIMER)}</p>\n'
        '    </section>\n'
        f'    <section class="finding-groups" aria-label="Detailed findings">{finding_groups}</section>\n'
        '    <details class="card dataset-details">\n'
        '      <summary><span>Dataset inventory and phase / magnetic maps</span><span class="muted">Open evidence</span></summary>\n'
        '      <div class="details-body"><div class="section-title">Dataset inventory</div>\n'
        f"{_render_inventory(result)}\n"
        f"{_render_phase_composition_maps(result)}\n"
        f"{_render_magnetic_composition_maps(result)}\n"
        f"{_render_magnetic_cross_evidence(result)}</div>\n"
        "    </details>\n"
        '    <details class="card technical-details">\n'
        '      <summary><span>Technical provenance</span><span class="muted">Inputs, checks and fingerprints</span></summary>\n'
        '      <div class="details-body technical-grid">\n'
        f'        <div class="technical-block"><div class="section-title">Inputs</div>{inputs_rows}</div>\n'
        f'        <div class="technical-block"><div class="section-title">Dimensions</div>{"".join(dimension_rows)}</div>\n'
        f'        <div class="technical-block"><div class="section-title">Run identity</div>{fingerprint_rows}</div>\n'
        f'        <div class="technical-block"><div class="section-title">Overview metrics</div>{overview_rows}</div>\n'
        "      </div>\n"
        "    </details>\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def write_audit_report_html(result: AuditResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_audit_report_html(result), encoding="utf-8")
