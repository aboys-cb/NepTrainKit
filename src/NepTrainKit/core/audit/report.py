"""Static HTML report export for Training Set Audit."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

from .findings import canonical_findings
from .result import AuditFinding, AuditResult


DISCLAIMER = (
    "Findings describe this dataset only. They are not sampling instructions or global coverage claims."
)


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
        rows.append(
            "<tr>"
            f"<td>{escape(composition)}</td>"
            f"<td>{point.structure_count:,}</td>"
            f"<td>{point.share:.2%}</td>"
            f"<td>{escape(atom_counts)}</td>"
            "</tr>"
        )
    return (
        f'<p><strong>{inventory.structure_count:,}</strong> structures · '
        f'<strong>{len(inventory.composition_points)}</strong> exact composition points · '
        f'{escape(" · ".join(inventory.elements))}</p>'
        '<div class="table-wrap"><table><thead><tr>'
        '<th>Exact composition</th><th>Structures</th><th>Share</th><th>Atom counts</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
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
