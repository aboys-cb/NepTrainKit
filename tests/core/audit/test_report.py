from pathlib import Path

from NepTrainKit.core.audit.report import render_audit_report_html, write_audit_report_html
from NepTrainKit.core.audit.result import (
    AuditAction,
    AuditBiasType,
    AuditDimension,
    AuditFinding,
    AuditFindingKind,
    AuditResult,
    AuditSeverity,
    AuditSlice,
    AuditStatus,
    CompositionPhaseEvidence,
    CompositionMagneticEvidence,
    CompositionPoint,
    DatasetInventory,
    ElementMagneticEvidence,
    ElementPairMagneticEvidence,
    PhaseInventory,
    MagneticInventory,
    StructureMagneticEvidence,
    StructurePhaseEvidence,
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
        phase_inventory=PhaseInventory(
            schema_version="phase-inventory-v2",
            method_id="adaptive-cna-ordering-v1",
            reference_bank_id="aflow-l12-laves-v1",
            analysis_strategy="all-structures-v1",
            source_structure_count=3,
            analyzed_structure_count=3,
            analyzed_atom_count=64,
            composition_points=(
                CompositionPhaseEvidence(
                    reduced_counts=(5, 3),
                    source_structure_count=3,
                    analyzed_structure_count=3,
                    analyzed_atom_count=64,
                    local_phase_fractions=(("fcc", 0.75), ("hcp", 0.0), ("bcc", 0.0), ("unresolved", 0.25)),
                    structure_phase_fractions=(("fcc", 2 / 3), ("unresolved", 1 / 3)),
                    confidence_counts=(("strong", 2), ("unresolved", 1)),
                ),
            ),
        ),
    )

    html = render_audit_report_html(result)

    assert "Dataset inventory" in html
    assert "Fe 62.50% · Ni 37.50%" in html
    assert "16 atoms × 2, 32 atoms × 1" in html
    assert "FCC 67% (3/3 analyzed)" in html
    assert "adaptive-cna-ordering-v1" in html
    assert "does not predict thermodynamic stability" in html
    assert "Phase labels by composition" in html
    assert "Fe concentration" in html
    assert "62.50%" in html
    assert "phase-fcc" in html


def test_report_includes_magnetic_order_map_and_boundary():
    structure = StructureMagneticEvidence(
        source_index=0, atom_count=16, spin_atom_count=16,
        order_label="fm", confidence_state="strong", mean_moment=2.1,
        moment_std=0.0, net_moment_ratio=1.0, collinearity=1.0,
        coplanarity=1.0, neighbor_correlation=1.0,
        neighbor_abs_correlation=1.0, parallel_fraction=1.0,
        antiparallel_fraction=0.0, q_peak_strength=0.0, q_vector=(0, 0, 0),
        element_evidence=(ElementMagneticEvidence(
            element="Fe", atom_count=16, spin_atom_count=16,
            order_label="aligned", mean_moment=2.1, net_moment_ratio=1.0,
            collinearity=1.0, intra_element_correlation=1.0,
            intra_element_pair_count=192, q_peak_strength=0.0,
            q_vector=(0, 0, 0),
        ),),
        element_pair_evidence=(ElementPairMagneticEvidence(
            element_a="Fe", element_b="Ni", pair_count=24,
            correlation=-0.8, coupling_label="antiparallel",
        ),),
    )
    result = AuditResult(
        dataset_id="spin.xyz", generated_at="now", inputs={"structure_count": 1},
        inventory=DatasetInventory(
            structure_count=1, elements=("Fe",),
            composition_points=(CompositionPoint(
                reduced_counts=(1,), fractions=(1.0,), structure_count=1,
                share=1.0, structure_indices=(0,),
            ),),
        ),
        magnetic_inventory=MagneticInventory(
            schema_version="magnetic-inventory-v1",
            method_id="spin-order-sf-neighbor-v1",
            analysis_strategy="all-spin-structures-v1",
            source_structure_count=1, analyzed_structure_count=1,
            missing_spin_count=0,
            composition_points=(CompositionMagneticEvidence(
                reduced_counts=(1,), source_structure_count=1,
                analyzed_structure_count=1, missing_spin_count=0,
                order_fractions=(("fm", 1.0),), confidence_counts=(("strong", 1),),
                mean_net_moment_ratio=1.0, mean_collinearity=1.0,
                mean_q_peak_strength=0.0, structures=(structure,),
            ),),
        ),
        phase_inventory=PhaseInventory(
            schema_version="phase-inventory-v2", method_id="adaptive-cna-ordering-v1",
            reference_bank_id="aflow-l12-laves-v1", analysis_strategy="all-structures-v1",
            source_structure_count=1, analyzed_structure_count=1, analyzed_atom_count=16,
            composition_points=(CompositionPhaseEvidence(
                reduced_counts=(1,), source_structure_count=1,
                analyzed_structure_count=1, analyzed_atom_count=16,
                local_phase_fractions=(("fcc", 1.0),),
                structure_phase_fractions=(("fcc", 1.0),),
                confidence_counts=(("strong", 1),),
                structures=(
                    StructurePhaseEvidence(
                        source_index=0, atom_count=16, phase_label="fcc",
                        confidence_state="strong", local_phase_fractions=(("fcc", 1.0),),
                    ),
                ),
            ),),
        ),
    )

    html = render_audit_report_html(result)

    assert "Magnetic-pattern labels by composition" in html
    assert "FM" in html
    assert "spin:R:3" in html
    assert "mforce and force_mag are excluded" in html
    assert "magnetic-fm" in html
    assert "Magnetic order inside each structural phase" in html
    assert "Element-local spin patterns" in html
    assert "Aligned (FM-like)" in html
    assert "Neighboring element-pair spin coupling" in html
    assert "Antiparallel" in html


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


def test_report_leads_with_decision_and_collapses_dense_evidence():
    result = AuditResult(
        dataset_id="train.xyz",
        generated_at="now",
        inputs={"structure_count": 12},
        findings=(
            AuditFinding(
                id="labels:missing",
                title="Missing force labels",
                dimension_id="labels",
                kind=AuditFindingKind.BLOCKER,
                signal_type=AuditBiasType.INFORMATIONAL,
                structure_indices=(2, 5),
                conclusion="The affected structures cannot supervise forces.",
                observed="2 structures do not contain force labels.",
                rule="All structures selected for force training require force labels.",
                limit="This check does not assess label accuracy.",
                actions=(AuditAction("show_structures", "Show 2 structures in Dataset Display"),),
            ),
            AuditFinding(
                id="force:tail",
                title="High-force review group",
                dimension_id="labels",
                kind=AuditFindingKind.REVIEW,
                signal_type=AuditBiasType.RISK_CONCENTRATION,
                structure_indices=(7,),
                conclusion="Inspect this structure before training.",
                observed="1 structure is in the high-force tail.",
                rule="Top 10% by maximum force.",
                limit="High force can be intentional.",
            ),
        ),
    )

    html = render_audit_report_html(result)

    assert "Action required before training" in html
    assert "Blocking findings</span><strong>1" in html
    assert "Review groups</span><strong>1" in html
    assert "<strong>Next:</strong> Show 2 structures in Dataset Display" in html
    assert html.index("Start here") < html.index("Dataset inventory")
    assert '<details class="finding-group" open><summary><span>Required action' in html
    assert '<details class="finding-group"><summary><span>Review next' in html
    assert '<details class="card dataset-details">' in html
    assert '<details class="card technical-details">' in html
