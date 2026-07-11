"""Static HTML report export for Training Set Audit."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

from .result import AuditResult, AuditSlice


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


def _render_slice_metrics(audit_slice: AuditSlice) -> str:
    if not audit_slice.metrics:
        return '<p class="muted">No slice metrics were recorded.</p>'
    rows = []
    for metric in audit_slice.metrics:
        value = _format_value(metric.value)
        if metric.unit:
            value = f"{value} {metric.unit}"
        meta_bits = []
        if metric.baseline is not None:
            meta_bits.append(f"baseline {_format_value(metric.baseline)}")
        if metric.direction:
            meta_bits.append(metric.direction)
        meta = f" <span class=\"muted\">({escape(' · '.join(meta_bits))})</span>" if meta_bits else ""
        rows.append(
            "<div class=\"metric-row\">"
            f"<span>{escape(metric.name)}</span>"
            f"<strong>{escape(value)}</strong>"
            f"{meta}"
            "</div>"
        )
    return "\n".join(rows)


def _render_slice(audit_slice: AuditSlice) -> str:
    return (
        '<section class="slice">'
        f"<h2>{escape(audit_slice.title)}</h2>"
        '<div class="pill-row">'
        f'<span class="pill severity-{escape(audit_slice.severity.value)}">{escape(audit_slice.severity.value)}</span>'
        f'<span class="pill">{escape(audit_slice.bias_type.value)}</span>'
        f'<span class="pill">{len(audit_slice.structure_indices)} structures</span>'
        "</div>"
        f"<p><strong>Observed</strong>: {escape(audit_slice.observed)}</p>"
        f"<p><strong>Interpretation</strong>: {escape(audit_slice.interpretation)}</p>"
        f"<p><strong>Limit</strong>: {escape(audit_slice.limit)}</p>"
        '<div class="metrics">'
        f"{_render_slice_metrics(audit_slice)}"
        "</div>"
        "</section>"
    )


def render_audit_report_html(result: AuditResult) -> str:
    overview_rows = _render_kv_list(result.overview_metrics.items(), "No overview metrics were recorded.")
    inputs_rows = _render_kv_list(result.inputs.items(), "No inputs were recorded.")
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

    if result.slices:
        slice_html = "\n".join(_render_slice(item) for item in result.slices)
    else:
        slice_html = '<p class="empty">No audit findings were generated.</p>'

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
        "    .card, .slice { background: #fff; border: 1px solid #d8dee4; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03); }\n"
        "    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }\n"
        "    .section-title { margin: 0 0 10px; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #6b7280; }\n"
        "    .kv-row, .metric-row, .dimension-row { display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid #edf1f4; }\n"
        "    .kv-row:last-child, .metric-row:last-child, .dimension-row:last-child { border-bottom: 0; padding-bottom: 0; }\n"
        "    .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }\n"
        "    .pill { display: inline-flex; align-items: center; border: 1px solid #d8dee4; border-radius: 999px; padding: 3px 9px; font-size: 12px; color: #334155; background: #f8fafb; }\n"
        "    .severity-high { background: #fee2e2; border-color: #fca5a5; color: #991b1b; }\n"
        "    .severity-medium { background: #fef3c7; border-color: #fcd34d; color: #92400e; }\n"
        "    .severity-low { background: #dcfce7; border-color: #86efac; color: #166534; }\n"
        "    .severity-info { background: #e0f2fe; border-color: #7dd3fc; color: #075985; }\n"
        "    .muted { color: #64748b; }\n"
        "    .empty { color: #52606d; font-style: italic; }\n"
        "    .note { margin-top: 8px; padding: 10px 12px; border-left: 3px solid #205a69; background: #eef6f7; color: #174c58; border-radius: 6px; }\n"
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
        '      <div class="section-title">Inputs</div>\n'
        f"{inputs_rows}\n"
        "    </section>\n"
        '    <section class="card">\n'
        '      <div class="section-title">Dimensions</div>\n'
        f"{''.join(dimension_rows)}\n"
        "    </section>\n"
        '    <section class="card">\n'
        '      <div class="section-title">Overview metrics</div>\n'
        f"{overview_rows}\n"
        "    </section>\n"
        '    <section class="card">\n'
        '      <div class="section-title">Findings</div>\n'
        f"{slice_html}\n"
        "    </section>\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def write_audit_report_html(result: AuditResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_audit_report_html(result), encoding="utf-8")
