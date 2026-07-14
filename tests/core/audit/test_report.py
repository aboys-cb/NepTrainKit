from pathlib import Path

from NepTrainKit.core.audit.report import render_audit_report_html, write_audit_report_html
from NepTrainKit.core.audit.result import (
    AuditBiasType,
    AuditDimension,
    AuditResult,
    AuditSeverity,
    AuditSlice,
    AuditStatus,
    CompositionPoint,
    DatasetInventory,
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
    assert "kind-evidence" in html
    assert "severity-medium" not in html


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


def test_report_includes_exact_dataset_inventory():
    result = AuditResult(
        dataset_id="train.xyz",
        generated_at="now",
        inputs={"structure_count": 3},
        inventory=DatasetInventory(
            structure_count=3,
            elements=("Fe", "Ni"),
            composition_points=(
                CompositionPoint(
                    reduced_counts=(5, 3),
                    fractions=(0.625, 0.375),
                    structure_count=3,
                    share=1.0,
                    structure_indices=(0, 1, 2),
                    atom_counts=((16, 2), (32, 1)),
                ),
            ),
        ),
    )

    html = render_audit_report_html(result)

    assert "Dataset inventory" in html
    assert "Fe 62.50% · Ni 37.50%" in html
    assert "16 atoms × 2, 32 atoms × 1" in html


def test_report_uses_the_same_consolidated_findings_as_the_gui_contract():
    plot = {
        "kind": "histogram",
        "id": "composition:Fe",
        "title": "Fe concentration distribution",
        "series": (
            {
                "counts": (1, 1),
                "bin_edges": (0.0, 0.5, 1.0),
                "bin_labels": ("0-50%", "50-100%"),
                "highlighted_bins": (0, 1),
                "structure_indices": ((2,), (7,)),
            },
        ),
    }
    slices = tuple(
        AuditSlice(
            id=f"composition:Fe:{label}",
            title=f"Sparse Fe bin {label}",
            dimension_id="composition",
            severity=AuditSeverity.MEDIUM,
            bias_type=AuditBiasType.SPARSITY,
            structure_indices=(index,),
            observed="Observed.",
            interpretation="Interpretation.",
            limit="Limit.",
        )
        for label, index in (("0-50%", 2), ("50-100%", 7))
    )
    result = AuditResult(
        dataset_id="train.xyz",
        generated_at="now",
        inputs={"structure_count": 2},
        dimensions=(
            AuditDimension(
                "composition",
                "Composition",
                AuditStatus.AVAILABLE,
                plots=(plot,),
            ),
        ),
        slices=slices,
        overview_metrics={"structures": 2},
    )

    html = render_audit_report_html(result)

    assert html.count('<section class="finding">') == 1
    assert "Fe composition has 2 low-frequency ranges" in html
    assert "2 structures" in html
