"""Static HTML report export for Training Set Audit."""
from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Iterable

from .findings import canonical_findings
from .result import AuditFinding, AuditResult


DISCLAIMER = (
    "Findings describe this dataset only. They are not sampling instructions or global coverage claims."
)

_PHASE_ORDER = ("fcc", "bcc", "hcp", "l12", "c14", "c15", "unresolved")


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
        "</section>"
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
                f"{phase_label.upper()} {phase_fraction:.0%} "
                f"({phase_point.analyzed_structure_count}/{point.structure_count} analyzed)"
            )
        rows.append(
            "<tr>"
            f"<td>{escape(composition)}</td>"
            f"<td>{point.structure_count:,}</td>"
            f"<td>{point.share:.2%}</td>"
            f"<td>{escape(phase_text)}</td>"
            f"<td>{escape(atom_counts)}</td>"
            "</tr>"
        )
    return (
        f'<p><strong>{inventory.structure_count:,}</strong> structures · '
        f'<strong>{len(inventory.composition_points)}</strong> exact composition points · '
        f'{escape(" · ".join(inventory.elements))}</p>'
        '<div class="table-wrap"><table><thead><tr>'
        '<th>Exact composition</th><th>Structures</th><th>Share</th><th>Main local phase</th><th>Atom counts</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        + (
            '<p class="muted">Phase evidence includes every audited structure and classifies local geometry; '
            'it does not predict thermodynamic stability. '
            f'Method: {escape(phase_inventory.method_id)}; reference bank: '
            f'{escape(phase_inventory.reference_bank_id)}.</p>'
            if phase_inventory is not None
            else ""
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
                    structure.phase_label for structure in phase_point.structures
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
                    f'title="{escape(phase.upper())}: {count:,} / {total:,}"></span>'
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
        f'<span><i class="phase-{phase}"></i>{escape(phase.upper())}</span>'
        for phase in _PHASE_ORDER
        if any(
            phase == structure.phase_label
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
    if findings:
        finding_html = "\n".join(_render_finding(item) for item in findings)
    else:
        finding_html = '<p class="empty">No audit findings were generated.</p>'

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "  <title>Training Set Audit Report</title>\n"
        "  <style>\n"
        "    :root { color-scheme: light; }\n"
        "    body { margin: 0; padding: 32px; background: #f6f7f8; color: #182026; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; }\n"
        "    .report { max-width: 1100px; margin: 0 auto; }\n"
        "    h1 { margin: 0 0 12px; font-size: 30px; line-height: 1.1; }\n"
        "    h2 { margin: 0 0 10px; font-size: 19px; }\n"
        "    .subtitle { margin: 0 0 20px; color: #52606d; }\n"
        "    .card, .finding { background: #fff; border: 1px solid #d8dee4; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03); }\n"
        "    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }\n"
        "    .section-title { margin: 0 0 10px; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #6b7280; }\n"
        "    .kv-row, .metric-row, .dimension-row { display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid #edf1f4; }\n"
        "    .kv-row:last-child, .metric-row:last-child, .dimension-row:last-child { border-bottom: 0; padding-bottom: 0; }\n"
        "    .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }\n"
        "    .pill { display: inline-flex; align-items: center; border: 1px solid #d8dee4; border-radius: 999px; padding: 3px 9px; font-size: 12px; color: #334155; background: #f8fafb; }\n"
        "    .kind-blocker { background: #fee2e2; border-color: #fca5a5; color: #991b1b; }\n"
        "    .kind-review { background: #fef3c7; border-color: #fcd34d; color: #92400e; }\n"
        "    .kind-evidence { background: #e0f2fe; border-color: #7dd3fc; color: #075985; }\n"
        "    .muted { color: #64748b; }\n"
        "    .empty { color: #52606d; font-style: italic; }\n"
        "    .note { margin-top: 8px; padding: 10px 12px; border-left: 3px solid #205a69; background: #eef6f7; color: #174c58; border-radius: 6px; }\n"
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
        "    .phase-l12 { background: #775da6; } .phase-c14 { background: #2e8b57; } .phase-c15 { background: #b44c6c; } .phase-unresolved { background: #89969a; }\n"
        "    table { width: 100%; border-collapse: collapse; }\n"
        "    th, td { padding: 8px; text-align: left; border-bottom: 1px solid #edf1f4; }\n"
        "    th { color: #52606d; font-size: 12px; }\n"
        "    @media (max-width: 900px) { body { padding: 18px; } .grid { grid-template-columns: 1fr; } }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        '  <main class="report">\n'
        "    <h1>Training Set Audit Report</h1>\n"
        '    <p class="subtitle">Static export of the current audit result.</p>\n'
        '    <section class="card">\n'
        '      <div class="grid">\n'
        '        <div>\n'
        '          <div class="section-title">Dataset</div>\n'
        f'          <div>{escape(result.dataset_id)}</div>\n'
        "        </div>\n"
        "        <div>\n"
        '          <div class="section-title">Generated</div>\n'
        f'          <div>{escape(result.generated_at)}</div>\n'
        "        </div>\n"
        "        <div>\n"
        '          <div class="section-title">Inputs</div>\n'
        f'          <div>{escape(str(result.inputs.get("structure_count", "n/a")))}</div>\n'
        "        </div>\n"
        "      </div>\n"
        f'      <p class="note">{escape(DISCLAIMER)}</p>\n'
        "    </section>\n"
        '    <section class="card">\n'
        '      <div class="section-title">Dataset inventory</div>\n'
        f"{_render_inventory(result)}\n"
        f"{_render_phase_composition_maps(result)}\n"
        "    </section>\n"
        '    <section class="card">\n'
        '      <div class="section-title">Inputs</div>\n'
        f"{inputs_rows}\n"
        "    </section>\n"
        '    <section class="card">\n'
        '      <div class="section-title">Dimensions</div>\n'
        f"{''.join(dimension_rows)}\n"
        "    </section>\n"
        '    <section class="card">\n'
        '      <div class="section-title">Run identity</div>\n'
        f"{fingerprint_rows}\n"
        "    </section>\n"
        '    <section class="card">\n'
        '      <div class="section-title">Overview metrics</div>\n'
        f"{overview_rows}\n"
        "    </section>\n"
        '    <section class="card">\n'
        '      <div class="section-title">Findings</div>\n'
        f"{finding_html}\n"
        "    </section>\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def write_audit_report_html(result: AuditResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_audit_report_html(result), encoding="utf-8")
