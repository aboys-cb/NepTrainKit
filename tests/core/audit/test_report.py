from pathlib import Path

from NepTrainKit.core.audit.report import render_audit_report_html, write_audit_report_html
from NepTrainKit.core.audit.result import (
    AuditBiasType,
    AuditResult,
    AuditSeverity,
    AuditSlice,
    SliceMetric,
)


def test_render_audit_report_includes_required_disclaimer():
    result = AuditResult(
        dataset_id="train.xyz",
        generated_at="2026-07-09T00:00:00+00:00",
        inputs={"structure_count": 10},
        slices=(
            AuditSlice(
                id="composition:Fe=95-100%",
                title="Sparse composition bin: Fe=95-100%",
                dimension_id="composition",
                severity=AuditSeverity.MEDIUM,
                bias_type=AuditBiasType.SPARSITY,
                structure_indices=(1, 2),
                observed="Fe-rich bin contains 2 of 10 structures.",
                interpretation="This bin is thin relative to the current dataset.",
                limit="This is not a global coverage claim.",
                metrics=(SliceMetric("dataset_fraction", 0.2),),
            ),
        ),
    )

    html = render_audit_report_html(result)

    assert "Training Set Audit Report" in html
    assert "Sparse composition bin" in html
    assert "Observed" in html
    assert "Interpretation" in html
    assert "Limit" in html
    assert "Findings describe this dataset only" in html


def test_write_audit_report_html_creates_parent_directory(tmp_path: Path):
    result = AuditResult(
        dataset_id="train.xyz",
        generated_at="2026-07-09T00:00:00+00:00",
        inputs={"structure_count": 0},
    )
    target = tmp_path / "nested" / "audit.html"

    write_audit_report_html(result, target)

    assert target.is_file()
    assert "Training Set Audit Report" in target.read_text(encoding="utf-8")
