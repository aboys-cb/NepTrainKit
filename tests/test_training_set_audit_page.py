#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import unittest
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QTranslator, Qt
from PySide6.QtWidgets import QApplication, QBoxLayout, QFrame, QHeaderView, QLabel
from qfluentwidgets import ComboBox, ListWidget, PrimaryPushButton, PushButton, TableWidget

from NepTrainKit.core.audit.result import (
    AuditBiasType,
    AuditDimension,
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
)
from NepTrainKit.ui import pages as ui_pages
from NepTrainKit.ui.pages.training_set_audit import TrainingSetAuditWidget
from NepTrainKit.ui.widgets.audit_chart import AuditChartWidget


class TestTrainingSetAuditWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    @staticmethod
    def _histogram(plot_id, title):
        return {
            "kind": "histogram",
            "id": plot_id,
            "title": title,
            "x_label": "Value",
            "y_label": "Structures",
            "series": (
                {
                    "counts": (1, 2),
                    "bin_edges": (0.0, 0.5, 1.0),
                    "bin_labels": ("0-50%", "50-100%"),
                    "highlighted_bins": (0,),
                    "structure_indices": ((0,), (1, 2)),
                },
            ),
        }

    def _dashboard_result(self):
        return AuditResult(
            dataset_id="train.xyz",
            generated_at="2026-07-10T08:30:00+00:00",
            inputs={"structure_count": 3},
            dimensions=(
                AuditDimension(
                    "composition",
                    "Composition",
                    AuditStatus.AVAILABLE,
                    plots=(
                        self._histogram("composition:Fe", "Fe concentration"),
                        self._histogram("composition:O", "O concentration"),
                    ),
                ),
                AuditDimension(
                    "label_ranges",
                    "Label ranges",
                    AuditStatus.PARTIAL,
                    "Available on labeled subsets only: force (2/3).",
                    plots=(self._histogram("label_ranges:max_force", "Maximum force"),),
                ),
            ),
            slices=(
                AuditSlice(
                    id="composition:Fe:sparse",
                    title="Sparse Fe composition bin",
                    dimension_id="composition",
                    severity=AuditSeverity.HIGH,
                    bias_type=AuditBiasType.SPARSITY,
                    structure_indices=(0,),
                    observed="One structure is in this Fe bin.",
                    interpretation="The composition region is thin.",
                    limit="Review against the intended model scope.",
                ),
                AuditSlice(
                    id="label_ranges:force_high_tail",
                    title="High force tail",
                    dimension_id="label_ranges",
                    severity=AuditSeverity.MEDIUM,
                    bias_type=AuditBiasType.RISK_CONCENTRATION,
                    structure_indices=(1, 2),
                    observed="Two structures are in the force tail.",
                    interpretation="Inspect these structures.",
                    limit="High force is not automatically wrong.",
                ),
            ),
            overview_metrics={
                "structures": 3,
                "finding_count": 2,
                "severity_counts": {"high": 1, "medium": 1},
                "composition": {"sparse_bin_count": 1},
                "label_ranges": {
                    "energy_labeled_count": 3,
                    "force_labeled_count": 2,
                    "virial_labeled_count": 0,
                    "label_total_count": 3,
                },
                "label_counts": {"energy": 3, "force": 2, "virial": 0},
            },
            inventory=DatasetInventory(
                structure_count=3,
                elements=("Fe", "Ni"),
                composition_points=(
                    CompositionPoint(
                        reduced_counts=(1, 0),
                        fractions=(1.0, 0.0),
                        structure_count=1,
                        share=1 / 3,
                        structure_indices=(0,),
                        atom_counts=((16, 1),),
                    ),
                    CompositionPoint(
                        reduced_counts=(1, 1),
                        fractions=(0.5, 0.5),
                        structure_count=2,
                        share=2 / 3,
                        structure_indices=(1, 2),
                        atom_counts=((16, 2),),
                    ),
                ),
                atom_counts=((16, 3),),
            ),
        )

    @staticmethod
    def _magnetic_structure(index: int, label: str, net: float, q_peak: float):
        return StructureMagneticEvidence(
            source_index=index,
            atom_count=16,
            spin_atom_count=16,
            order_label=label,
            confidence_state="strong",
            mean_moment=1.5,
            moment_std=0.1,
            net_moment_ratio=net,
            collinearity=1.0 if label in {"fm", "afm"} else 0.4,
            coplanarity=1.0,
            neighbor_correlation=1.0 if label == "fm" else -0.5,
            neighbor_abs_correlation=1.0,
            parallel_fraction=1.0 if label == "fm" else 0.0,
            antiparallel_fraction=0.0 if label == "fm" else 0.5,
            q_peak_strength=q_peak,
            q_vector=(1, 0, 0) if q_peak else (0, 0, 0),
        )

    def test_magnetic_order_map_evidence_and_filter_are_available_together(self):
        base = self._dashboard_result()
        magnetic = MagneticInventory(
            schema_version="magnetic-inventory-v1",
            method_id="spin-order-sf-neighbor-v1",
            analysis_strategy="all-spin-structures-v1",
            source_structure_count=3,
            analyzed_structure_count=3,
            missing_spin_count=0,
            composition_points=(
                CompositionMagneticEvidence(
                    reduced_counts=(1, 0), source_structure_count=1,
                    analyzed_structure_count=1, missing_spin_count=0,
                    order_fractions=(("fm", 1.0),),
                    confidence_counts=(("strong", 1),),
                    mean_net_moment_ratio=1.0, mean_collinearity=1.0,
                    mean_q_peak_strength=0.0,
                    structures=(self._magnetic_structure(0, "fm", 1.0, 0.0),),
                ),
                CompositionMagneticEvidence(
                    reduced_counts=(1, 1), source_structure_count=2,
                    analyzed_structure_count=2, missing_spin_count=0,
                    order_fractions=(("afm", 0.5), ("pm_like", 0.5)),
                    confidence_counts=(("strong", 2),),
                    mean_net_moment_ratio=0.05, mean_collinearity=0.7,
                    mean_q_peak_strength=0.55,
                    structures=(
                        self._magnetic_structure(1, "afm", 0.0, 1.0),
                        self._magnetic_structure(2, "pm_like", 0.1, 0.1),
                    ),
                ),
            ),
        )
        result = replace(
            base,
            magnetic_inventory=magnetic,
            overview_metrics={
                **base.overview_metrics,
                "magnetic_inventory": {"available": True, "status": "complete"},
            },
        )
        widget = TrainingSetAuditWidget()
        selected = []
        widget.selectStructuresSignal.connect(selected.append)

        widget.set_result(result)

        self.assertEqual(widget.composition_chart.plot_id, "inventory:composition:Ni")
        self.assertTrue(widget.composition_phase_summary_label.isHidden())
        self.assertEqual(widget.analyze_structure_evidence_button.text(), "Analyze remaining")
        magnetic_view = widget.composition_view_selector.findData("magnetic")
        self.assertGreaterEqual(magnetic_view, 0)
        widget.composition_view_selector.setCurrentIndex(magnetic_view)
        self.assertEqual(
            widget.composition_chart.plot_id,
            "inventory:composition-magnetism:Ni",
        )
        self.assertEqual(widget.composition_chart._plot["counts"], (1.0, 2.0))
        self.assertIn("Bar height is the sample count", widget.composition_phase_summary_label.text())
        magnetic_row = next(
            row for row in range(widget.dimension_list.count())
            if widget.dimension_list.item(row).data(Qt.ItemDataRole.UserRole)
            == "magnetic_evidence"
        )
        widget.dimension_list.setCurrentRow(magnetic_row)
        self.assertEqual(widget.plot_selector.count(), 1)
        self.assertEqual(
            widget._active_plots[0]["id"], "magnetic_evidence:overall"
        )
        self.assertEqual(
            widget._active_plots[0]["kind"], "category_share_stacks"
        )
        afm_index = widget.composition_magnetic_selector.findData("afm")
        widget.composition_magnetic_selector.setCurrentIndex(afm_index)
        widget._emit_composition_structures()
        self.assertEqual(selected[-1], [1])

    def test_magnetic_evidence_crosses_structure_phase_and_element_local_patterns(self):
        base = self._dashboard_result()
        magnetic_structures = (
            replace(
                self._magnetic_structure(0, "fm", 1.0, 0.0),
                element_evidence=(ElementMagneticEvidence(
                    element="Fe", atom_count=16, spin_atom_count=16,
                    order_label="aligned", mean_moment=2.1,
                    net_moment_ratio=1.0, collinearity=1.0,
                    intra_element_correlation=1.0, intra_element_pair_count=192,
                    q_peak_strength=0.0, q_vector=(0, 0, 0),
                ),),
            ),
            replace(
                self._magnetic_structure(1, "afm", 0.0, 1.0),
                order_subtype="double_layered",
                element_evidence=(
                    ElementMagneticEvidence(
                        element="Fe", atom_count=8, spin_atom_count=8,
                        order_label="compensated", mean_moment=2.0,
                        net_moment_ratio=0.0, collinearity=1.0,
                        intra_element_correlation=-1.0, intra_element_pair_count=48,
                        q_peak_strength=1.0, q_vector=(1, 0, 0),
                    ),
                    ElementMagneticEvidence(
                        element="Ni", atom_count=8, spin_atom_count=8,
                        order_label="aligned", mean_moment=0.6,
                        net_moment_ratio=1.0, collinearity=1.0,
                        intra_element_correlation=1.0, intra_element_pair_count=48,
                        q_peak_strength=0.0, q_vector=(0, 0, 0),
                    ),
                ),
                element_pair_evidence=(ElementPairMagneticEvidence(
                    element_a="Fe", element_b="Ni", pair_count=96,
                    correlation=-1.0, coupling_label="antiparallel",
                ),),
            ),
            replace(
                self._magnetic_structure(2, "pm_like", 0.1, 0.1),
                element_evidence=(
                    ElementMagneticEvidence(
                        element="Fe", atom_count=8, spin_atom_count=8,
                        order_label="compensated", mean_moment=2.0,
                        net_moment_ratio=0.1, collinearity=1.0,
                        intra_element_correlation=-0.8, intra_element_pair_count=48,
                        q_peak_strength=0.8, q_vector=(1, 0, 0),
                    ),
                    ElementMagneticEvidence(
                        element="Ni", atom_count=8, spin_atom_count=8,
                        order_label="aligned", mean_moment=0.6,
                        net_moment_ratio=0.9, collinearity=1.0,
                        intra_element_correlation=0.9, intra_element_pair_count=48,
                        q_peak_strength=0.1, q_vector=(1, 0, 0),
                    ),
                ),
                element_pair_evidence=(ElementPairMagneticEvidence(
                    element_a="Fe", element_b="Ni", pair_count=96,
                    correlation=0.0, coupling_label="mixed",
                ),),
            ),
        )
        magnetic = MagneticInventory(
            schema_version="magnetic-inventory-v3",
            method_id="spin-order-layer-afm-v3",
            analysis_strategy="all-spin-structures-v1",
            source_structure_count=3, analyzed_structure_count=2,
            missing_spin_count=1,
            composition_points=(
                CompositionMagneticEvidence(
                    reduced_counts=(1, 0), source_structure_count=1,
                    analyzed_structure_count=1, missing_spin_count=0,
                    order_fractions=(("fm", 1.0),), confidence_counts=(("strong", 1),),
                    mean_net_moment_ratio=1.0, mean_collinearity=1.0,
                    mean_q_peak_strength=0.0, structures=(magnetic_structures[0],),
                ),
                CompositionMagneticEvidence(
                    reduced_counts=(1, 1), source_structure_count=2,
                    analyzed_structure_count=1, missing_spin_count=1,
                    order_fractions=(("afm", 1.0),),
                    confidence_counts=(("strong", 1),),
                    mean_net_moment_ratio=0.0, mean_collinearity=1.0,
                    mean_q_peak_strength=1.0, structures=(magnetic_structures[1],),
                ),
            ),
        )
        phase = PhaseInventory(
            schema_version="phase-inventory-v2", method_id="adaptive-cna-ordering-v1",
            reference_bank_id="aflow-l12-laves-v1", analysis_strategy="all-structures-v1",
            source_structure_count=3, analyzed_structure_count=3, analyzed_atom_count=48,
            composition_points=(
                CompositionPhaseEvidence(
                    reduced_counts=(1, 0), source_structure_count=1,
                    analyzed_structure_count=1, analyzed_atom_count=16,
                    local_phase_fractions=(("fcc", 1.0),),
                    structure_phase_fractions=(("fcc", 1.0),),
                    confidence_counts=(("strong", 1),),
                    structures=(StructurePhaseEvidence(
                        source_index=0, atom_count=16, phase_label="fcc",
                        confidence_state="strong", local_phase_fractions=(("fcc", 1.0),),
                    ),),
                ),
                CompositionPhaseEvidence(
                    reduced_counts=(1, 1), source_structure_count=2,
                    analyzed_structure_count=2, analyzed_atom_count=32,
                    local_phase_fractions=(("fcc", 0.5), ("bcc", 0.5)),
                    structure_phase_fractions=(("fcc", 0.5), ("bcc", 0.5)),
                    confidence_counts=(("strong", 2),),
                    structures=(
                        StructurePhaseEvidence(
                            source_index=1, atom_count=16, phase_label="fcc",
                            confidence_state="strong", local_phase_fractions=(("fcc", 1.0),),
                        ),
                        StructurePhaseEvidence(
                            source_index=2, atom_count=16, phase_label="bcc",
                            confidence_state="strong", local_phase_fractions=(("bcc", 1.0),),
                        ),
                    ),
                ),
            ),
        )
        widget = TrainingSetAuditWidget()
        widget.set_result(replace(base, magnetic_inventory=magnetic, phase_inventory=phase))

        magnetic_row = next(
            row for row in range(widget.dimension_list.count())
            if widget.dimension_list.item(row).data(Qt.ItemDataRole.UserRole)
            == "magnetic_evidence"
        )
        widget.dimension_list.setCurrentRow(magnetic_row)

        plot_ids = {plot["id"] for plot in widget._active_plots}
        self.assertEqual(
            plot_ids,
            {
                "magnetic_evidence:phase_to_order",
                "magnetic_evidence:order_to_phase",
                "magnetic_evidence:overall",
            },
        )
        widget._render_phase_summary()
        self.assertIn("Element-local spin patterns", widget.composition_phase_summary_label.text())
        self.assertIn("Fe", widget.composition_phase_summary_label.text())
        self.assertIn("Ni", widget.composition_phase_summary_label.text())
        phase_to_order = next(
            plot for plot in widget._active_plots
            if plot["id"] == "magnetic_evidence:phase_to_order"
        )
        self.assertEqual(phase_to_order["kind"], "category_share_stacks")
        double_layered = next(
            series
            for series in phase_to_order["series"]
            if series["id"] == "afm_double_layered"
        )
        self.assertEqual(double_layered["structure_indices"][0], (1,))
        no_spin = next(
            series
            for series in phase_to_order["series"]
            if series["id"] == "no_spin"
        )
        self.assertEqual(no_spin["structure_indices"][1], (2,))
        order_to_phase = next(
            plot for plot in widget._active_plots
            if plot["id"] == "magnetic_evidence:order_to_phase"
        )
        no_spin_row = order_to_phase["row_ids"].index("no_spin")
        bcc = next(
            series for series in order_to_phase["series"] if series["id"] == "bcc"
        )
        self.assertEqual(bcc["structure_indices"][no_spin_row], (2,))

    @staticmethod
    def _local_chemistry_plot(scope, center_element, metric_kind):
        scope_title = {
            "angular": "Angular core",
            "radial": "Radial context",
        }[scope]
        if metric_kind == "neighbor_count":
            metric_id = "neighbor_count"
            metric_label = "neighbor count"
        else:
            metric_id = f"neighbor_fraction_{center_element}"
            metric_label = f"{center_element} neighbor fraction"
        return {
            "kind": "histogram",
            "id": f"local_chemistry:{scope}:{center_element}:{metric_id}",
            "scope": scope,
            "center_element": center_element,
            "metric_kind": metric_kind,
            "title": f"{scope_title}: {center_element} {metric_label}",
            "x_label": metric_label.capitalize(),
            "y_label": "Local environments",
            "series": (
                {
                    "id": metric_id,
                    "label": metric_label,
                    "counts": (1, 2),
                    "bin_edges": (0.0, 0.5, 1.0),
                    "structure_indices": ((4,), (11,)),
                },
            ),
        }

    def _local_chemistry_result(self):
        plots = tuple(
            self._local_chemistry_plot(scope, center, metric_kind)
            for scope in ("angular", "radial")
            for center in ("Fe", "Ni")
            for metric_kind in ("neighbor_count", "neighbor_fraction")
        )
        return AuditResult(
            dataset_id="local.xyz",
            generated_at="2026-07-10T08:30:00+00:00",
            inputs={"structure_count": 2},
            dimensions=(
                AuditDimension(
                    "local_chemistry",
                    "Local chemistry",
                    AuditStatus.AVAILABLE,
                    plots=plots,
                ),
            ),
            overview_metrics={
                "structures": 2,
                "finding_count": 0,
                "local_chemistry": {
                    "available_scopes": ("angular", "radial"),
                    "center_element_count": 2,
                    "sparse_bin_count": 0,
                },
            },
        )

    def test_local_chemistry_selectors_filter_scope_and_preserve_center_element(self):
        widget = TrainingSetAuditWidget()
        widget.resize(1100, 760)
        widget.show()
        widget.set_result(self._local_chemistry_result())
        widget.dimension_list.setCurrentRow(2)
        widget.page_tabs.setCurrentIndex(1)
        widget.data_map_tabs.setCurrentIndex(1)
        self._app.processEvents()

        self.assertTrue(widget.local_scope_selector.isVisible())
        self.assertTrue(widget.local_center_label.isVisible())
        self.assertTrue(widget.local_center_selector.isVisible())
        self.assertEqual(widget.local_scope_selector.currentData(), "angular")
        self.assertEqual(widget.local_scope_selector.count(), 2)
        self.assertEqual(widget.local_center_selector.count(), 2)
        self.assertEqual(widget.local_center_selector.currentData(), "Fe")
        self.assertEqual(widget.plot_selector.count(), 2)
        self.assertTrue(widget.chart_widget.plot_id.startswith("local_chemistry:angular:Fe:"))
        self.assertEqual(
            widget.analysis_status_label.text(),
            "Scope from the active NEP cutoffs · Angular neighbors · center Fe. "
            "Orange marks low-frequency ranges inside the current data.",
        )

        widget.local_center_selector.setCurrentIndex(1)
        widget.local_scope_selector.setCurrentIndex(1)

        self.assertEqual(widget.local_center_selector.currentData(), "Ni")
        self.assertEqual(widget.plot_selector.count(), 2)
        self.assertTrue(widget.chart_widget.plot_id.startswith("local_chemistry:radial:Ni:"))
        self.assertIn("Radial neighbors", widget.analysis_status_label.text())

        widget.dimension_list.setCurrentRow(0)
        self.assertTrue(widget.local_scope_selector.isHidden())
        self.assertTrue(widget.local_center_selector.isHidden())
        widget.close()

    def test_model_panel_distinguishes_compute_filtering_from_absent_element_evidence(self):
        result = self._local_chemistry_result()
        local_overview = {
            **result.overview_metrics["local_chemistry"],
            "declared_model_elements": ("H", "Fe", "Ni", "O"),
            "analyzed_model_elements": ("Fe", "Ni"),
            "absent_model_elements": ("H", "O"),
        }
        result = replace(
            result,
            overview_metrics={
                **result.overview_metrics,
                "local_chemistry": local_overview,
            },
        )
        widget = TrainingSetAuditWidget()

        widget.set_result(result)

        self.assertIn("declares 4 elements", widget.model_empty_label.text())
        self.assertIn("dataset contains 2: Fe · Ni", widget.model_empty_label.text())
        self.assertIn("other 2 model elements are absent", widget.model_empty_label.text())
        self.assertEqual(widget.model_empty_label.toolTip(), "Absent model elements: H, O")
        widget.close()

    def test_local_chemistry_unavailable_hides_selectors_and_shows_parser_reason(self):
        widget = TrainingSetAuditWidget()
        reason = "NEP cutoff line does not match the declared element count."
        result = AuditResult(
            dataset_id="local.xyz",
            generated_at="now",
            inputs={"structure_count": 2},
            dimensions=(
                AuditDimension(
                    "local_chemistry",
                    "Local chemistry",
                    AuditStatus.UNAVAILABLE,
                    reason,
                ),
            ),
            overview_metrics={"structures": 2, "finding_count": 0},
        )

        widget.set_result(result)
        widget.dimension_list.setCurrentRow(2)

        self.assertTrue(widget.local_scope_selector.isHidden())
        self.assertTrue(widget.local_center_label.isHidden())
        self.assertTrue(widget.local_center_selector.isHidden())
        self.assertTrue(widget.plot_selector.isHidden())
        self.assertEqual(widget.analysis_status_label.text(), reason)
        self.assertFalse(widget.chart_widget.has_data)

    def test_local_chemistry_accepts_task_1_metric_payload_field(self):
        result = self._local_chemistry_result()
        task_1_plots = []
        for plot in result.dimensions[0].plots:
            task_1_plot = dict(plot)
            task_1_plot.pop("metric_kind")
            task_1_plot["metric"] = str(task_1_plot["id"]).rsplit(":", 1)[1]
            task_1_plots.append(task_1_plot)
        result = AuditResult(
            dataset_id=result.dataset_id,
            generated_at=result.generated_at,
            inputs=result.inputs,
            dimensions=(
                AuditDimension(
                    "local_chemistry",
                    "Local chemistry",
                    AuditStatus.AVAILABLE,
                    plots=tuple(task_1_plots),
                ),
            ),
            overview_metrics=result.overview_metrics,
        )
        widget = TrainingSetAuditWidget()

        widget.set_result(result)
        widget.dimension_list.setCurrentRow(2)

        self.assertEqual(widget.plot_selector.itemText(0), "Neighbor count")
        self.assertEqual(widget.plot_selector.itemText(1), "Fe neighbor fraction")
        self.assertTrue(widget.chart_widget.plot_id.startswith("local_chemistry:angular:Fe:"))

    def test_dashboard_exposes_approved_layout_contract(self):
        widget = TrainingSetAuditWidget()

        widget.set_result(self._dashboard_result())

        self.assertEqual(widget.dimension_list.count(), 4)
        self.assertEqual(widget.metric_structure_value.text(), "3")
        self.assertEqual(widget.generated_at_label.text(), "Generated 2026-07-10 08:30 UTC")
        self.assertIsInstance(widget.rerun_button, PrimaryPushButton)
        self.assertNotIsInstance(widget.export_report_button, PrimaryPushButton)
        self.assertIsInstance(widget.dimension_list, ListWidget)
        self.assertIsInstance(widget.local_scope_selector, ComboBox)
        self.assertIsInstance(widget.local_center_selector, ComboBox)
        self.assertIsInstance(widget.plot_selector, ComboBox)
        self.assertIsInstance(widget.slice_table, TableWidget)
        self.assertEqual(widget.metric_findings_label.text(), "Exact composition points")
        self.assertEqual(widget.metric_findings_value.text(), "2")
        self.assertEqual(widget.metric_dimension_label.text(), "Elements")
        self.assertEqual(widget.metric_dimension_value.text(), "Fe · Ni")
        self.assertEqual(widget.metric_context_label.text(), "Label availability")
        self.assertEqual(widget.metric_context_value.text(), "E 100% · F 67% · V 0%")
        self.assertEqual(widget.fact_total_atoms_value.text(), "48")
        self.assertEqual(widget.fact_atom_range_value.text(), "16")
        self.assertEqual(widget.fact_atom_center_value.text(), "16.0 / 16.0")
        self.assertIs(widget.fact_total_atoms_value.parentWidget().parentWidget(), widget.metric_band)
        self.assertFalse(hasattr(widget, "fact_config_value"))
        self.assertIsNone(widget.findChild(QLabel, "inventorySummary"))
        self.assertEqual(widget.dimension_list.item(0).text(), "Overview\n1 topic")
        self.assertEqual(
            widget.dimension_list.item(1).text(),
            "Phases and local structure\nNot calculated",
        )
        self.assertEqual(widget.dimension_list.item(2).text(), "Composition balance\nCalculated · 1 topic")
        self.assertEqual(widget.dimension_list.item(3).text(), "Labels and extremes\nPartial data · 1 topic")
        self.assertEqual(widget.label_availability_title.text(), "Label availability")
        self.assertEqual(
            widget.label_availability_value.text(),
            "Energy 3/3\nForce 2/3\nVirial 0/3",
        )
        self.assertEqual(widget.page_tabs.count(), 4)
        self.assertEqual(widget.page_tabs.tabText(0), "Overview")
        self.assertEqual(widget.page_tabs.tabText(1), "Data map")
        self.assertEqual(widget.page_tabs.tabText(2), "Review queue")
        self.assertEqual(widget.page_tabs.tabText(3), "Target & model")
        self.assertIs(widget.page_tabs.widget(0), widget.summary_tab)
        self.assertFalse(hasattr(widget, "summary_scroll"))
        self.assertEqual(
            widget.page_tabs.tabBar().elideMode(),
            Qt.TextElideMode.ElideNone,
        )
        self.assertEqual(widget.data_map_tabs.count(), 3)
        self.assertEqual(widget.composition_table.rowCount(), 2)
        self.assertEqual(widget.composition_chart.plot_id, "inventory:composition:Ni")
        self.assertIs(widget.audit_header.parentWidget(), widget.summary_tab)
        self.assertEqual(widget.composition_splitter.orientation(), Qt.Orientation.Horizontal)
        self.assertEqual(widget.composition_splitter.count(), 2)
        self.assertFalse(widget.composition_table.isColumnHidden(0))
        self.assertFalse(widget.composition_table.isColumnHidden(3))
        self.assertFalse(widget.composition_table.isColumnHidden(4))
        self.assertEqual(widget.cooccurrence_table.rowCount(), 2)
        self.assertEqual(widget.cooccurrence_table.columnCount(), 2)
        self.assertEqual(
            widget.cooccurrence_table.delegate.lightCheckedColor.alpha(),
            0,
        )
        self.assertEqual(
            widget.pair_coverage_label.text(),
            "1/1 pairs co-occur · 1/1 have exact binary structures",
        )
        self.assertIn(
            "co-occurring structures",
            widget.cooccurrence_table.item(0, 1).toolTip(),
        )
        self.assertIn(
            "Select an upper-triangle element pair",
            widget.cooccurrence_table.item(1, 0).toolTip(),
        )
        self.assertEqual(
            widget.cooccurrence_table.item(1, 0).background().color().alpha(),
            0,
        )

        widget._selected_overview_elements = ("NotPresent",)
        widget._selected_overview_mode = "element"
        widget._populate_overview_element_sets()

        self.assertEqual(widget.element_sets_table.rowCount(), 0)
        self.assertGreater(widget.element_sets_table.columnWidth(2), 0)
        widget._clear_overview_element_filter()
        self.assertEqual(widget.cooccurrence_table.item(1, 1).text(), "—")
        self.assertEqual(widget.element_sets_table.rowCount(), 2)
        self.assertEqual(widget.order_summary_values["1"].text(), "33.3% · 1")
        self.assertEqual(widget.order_summary_values["2"].text(), "66.7% · 2")
        self.assertIsNone(widget.findChild(QFrame, "auditSummaryPanel"))
        self.assertIsNone(widget.findChild(QFrame, "auditInventoryPanel"))
        self.assertIsNone(widget.findChild(QFrame, "auditNextActionsPanel"))
        self.assertIsNone(widget.findChild(QFrame, "auditDatasetFactsPanel"))
        self.assertEqual(widget.chart_widget.plot_id, "composition:Fe")
        self.assertEqual(widget.slice_table.rowCount(), 1)
        self.assertEqual(widget.audit_header.objectName(), "auditHeader")
        self.assertEqual(widget.dimension_rail.objectName(), "auditDimensionRail")
        self.assertEqual(widget.metric_band.objectName(), "auditMetricBand")
        self.assertEqual(widget.analysis_panel.objectName(), "auditAnalysisPanel")
        self.assertEqual(widget.findings_panel.objectName(), "auditFindingsPanel")
        self.assertEqual(widget.cooccurrence_panel.objectName(), "auditCooccurrencePanel")
        self.assertEqual(widget.element_sets_panel.objectName(), "auditElementSetsPanel")

    def test_overview_dataset_facts_use_existing_inventory_without_another_scan(self):
        result = self._dashboard_result()
        inventory = replace(
            result.inventory,
            structure_count=4,
            atom_counts=((16, 1), (32, 2), (64, 1)),
            config_types=(("bulk", 3),),
            missing_config_type_count=1,
        )
        widget = TrainingSetAuditWidget()

        widget.set_result(replace(result, inventory=inventory))

        self.assertEqual(widget.fact_total_atoms_value.text(), "144")
        self.assertEqual(widget.fact_atom_range_value.text(), "16–64")
        self.assertEqual(widget.fact_atom_center_value.text(), "36.0 / 32.0")
        self.assertFalse(hasattr(widget, "fact_config_value"))

    def test_overview_matrix_reveals_elements_related_to_selected_pair(self):
        result = self._dashboard_result()
        inventory = DatasetInventory(
            structure_count=15,
            elements=("Al", "Co", "Fe", "Ni"),
            composition_points=(
                CompositionPoint(
                    reduced_counts=(0, 0, 1, 0),
                    fractions=(0.0, 0.0, 1.0, 0.0),
                    structure_count=1,
                    share=1 / 15,
                    structure_indices=(0,),
                ),
                CompositionPoint(
                    reduced_counts=(0, 0, 1, 1),
                    fractions=(0.0, 0.0, 0.5, 0.5),
                    structure_count=2,
                    share=2 / 15,
                    structure_indices=(1, 2),
                ),
                CompositionPoint(
                    reduced_counts=(1, 0, 1, 1),
                    fractions=(1 / 3, 0.0, 1 / 3, 1 / 3),
                    structure_count=3,
                    share=3 / 15,
                    structure_indices=(3, 4, 5),
                ),
                CompositionPoint(
                    reduced_counts=(1, 1, 1, 1),
                    fractions=(0.25, 0.25, 0.25, 0.25),
                    structure_count=4,
                    share=4 / 15,
                    structure_indices=(6, 7, 8, 9),
                ),
                CompositionPoint(
                    reduced_counts=(1, 0, 0, 1),
                    fractions=(0.5, 0.0, 0.0, 0.5),
                    structure_count=5,
                    share=5 / 15,
                    structure_indices=(10, 11, 12, 13, 14),
                ),
            ),
            atom_counts=((16, 15),),
        )
        widget = TrainingSetAuditWidget()

        widget.set_result(replace(result, inventory=inventory))

        self.assertEqual(
            widget.pair_coverage_label.text(),
            "6/6 pairs co-occur · 2/6 have exact binary structures",
        )
        self.assertIn(
            "9 co-occurring structures",
            widget.cooccurrence_table.item(2, 3).toolTip(),
        )
        self.assertIn(
            "Select an upper-triangle element pair",
            widget.cooccurrence_table.item(3, 2).toolTip(),
        )
        self.assertEqual(
            widget.cooccurrence_table.item(3, 2).background().color().alpha(),
            0,
        )
        jet_colors = {
            widget._overview_heat_color(value, 100).name()
            for value in (1, 5, 20, 50, 100)
        }
        self.assertGreaterEqual(len(jet_colors), 4)
        self.assertEqual(widget._overview_heat_color(100, 100).name(), "#800000")
        self.assertEqual(widget.order_summary_values["1"].text(), "6.7% · 1")
        self.assertEqual(widget.order_summary_values["2"].text(), "46.7% · 7")
        self.assertEqual(widget.order_summary_values["3"].text(), "20.0% · 3")
        self.assertEqual(widget.order_summary_values["4+"].text(), "26.7% · 4")

        widget.cooccurrence_table.cellClicked.emit(2, 3)

        self.assertEqual(widget.element_sets_table.rowCount(), 3)
        self.assertIn("9 co-occurring structures", widget.element_sets_summary_label.text())
        self.assertIn("2 exact binary structures", widget.element_sets_summary_label.text())
        self.assertEqual(
            widget.matrix_selection_label.text(),
            "Basis: Fe + Ni · related-element view",
        )
        self.assertEqual(widget.cooccurrence_table.item(2, 3).text(), "✓")
        self.assertTrue(widget.cooccurrence_table.item(2, 3).isSelected())
        self.assertFalse(widget.cooccurrence_table.delegate.selectedRows)
        self.assertEqual(
            widget.cooccurrence_table.item(0, 3).background().color().alpha(),
            0,
        )
        self.assertIn(
            "7 structures",
            widget.cooccurrence_table.item(0, 0).toolTip(),
        )
        self.assertIn(
            "4 structures",
            widget.cooccurrence_table.item(1, 1).toolTip(),
        )
        self.assertIn(
            "4 structures",
            widget.cooccurrence_table.item(1, 0).toolTip(),
        )
        self.assertFalse(widget.clear_element_filter_button.isHidden())

        widget.cooccurrence_table.cellClicked.emit(1, 0)

        self.assertEqual(widget.element_sets_table.rowCount(), 1)
        self.assertEqual(
            widget.matrix_selection_label.text(),
            "Selected: Al + Co + Fe + Ni · based on Fe + Ni",
        )
        self.assertEqual(widget.cooccurrence_table.item(1, 0).text(), "✓")
        self.assertIn("4 structures", widget.element_sets_summary_label.text())

        widget.cooccurrence_table.cellClicked.emit(1, 0)

        self.assertEqual(widget.element_sets_table.rowCount(), 5)
        self.assertEqual(
            widget.matrix_selection_label.text(),
            "Filter: none · Click a cell",
        )
        self.assertEqual(
            widget.cooccurrence_table.item(1, 0).background().color().alpha(),
            0,
        )

    def test_dashboard_exposes_backend_and_render_timings_without_cluttering_panels(self):
        result = self._dashboard_result()
        result = replace(
            result,
            overview_metrics={
                **result.overview_metrics,
                "timings_ms": {
                    "total": 1234.5,
                    "preparation": 400.0,
                    "finalization": 100.0,
                    "stages": {
                        "record_extraction": 300.0,
                        "data_quality": 500.0,
                    },
                },
                "data_quality": {
                    "timings_ms": {
                        "total": 500.0,
                        "stages": {
                            "structure_contracts": 420.0,
                            "minimum_distance_scan": 16.0,
                        },
                    },
                },
            },
        )
        widget = TrainingSetAuditWidget()

        widget.set_result(result)

        self.assertIn("Audit 1.23 s", widget.generated_at_label.text())
        self.assertIn("Backend total: 1234.5 ms", widget.generated_at_label.toolTip())
        self.assertIn("data_quality: 500.0 ms", widget.generated_at_label.toolTip())
        self.assertIn("structure_contracts: 420.0 ms", widget.generated_at_label.toolTip())
        self.assertGreaterEqual(widget._last_render_timings_ms["total"], 0.0)

    def test_dimension_selection_updates_plots_findings_and_unavailable_reason(self):
        widget = TrainingSetAuditWidget()
        widget.set_result(self._dashboard_result())

        widget.dimension_list.setCurrentRow(2)
        self.assertEqual(widget.plot_selector.count(), 2)
        self.assertEqual(widget.chart_widget.plot_id, "composition:Fe")
        self.assertEqual(widget.slice_table.rowCount(), 1)
        self.assertEqual(len(widget._all_topics), 2)

        widget.plot_selector.setCurrentIndex(1)
        self.assertEqual(widget.chart_widget.plot_id, "composition:O")

        widget.dimension_list.setCurrentRow(3)
        self.assertFalse(widget.chart_widget.isHidden())

    def test_composition_map_shows_complete_phase_evidence_and_drilldown(self):
        result = replace(
            self._dashboard_result(),
            phase_inventory=PhaseInventory(
                schema_version="phase-inventory-v2",
                method_id="adaptive-cna-ordering-v1",
                reference_bank_id="aflow-l12-laves-v1",
                analysis_strategy="all-structures-v1",
                source_structure_count=3,
                analyzed_structure_count=3,
                analyzed_atom_count=48,
                composition_points=(
                    CompositionPhaseEvidence(
                        reduced_counts=(1, 0),
                        source_structure_count=1,
                        analyzed_structure_count=1,
                        analyzed_atom_count=16,
                        local_phase_fractions=(("fcc", 0.875), ("hcp", 0.0), ("bcc", 0.0), ("unresolved", 0.125)),
                        structure_phase_fractions=(("fcc", 1.0),),
                        confidence_counts=(("strong", 1),),
                        structures=(
                            StructurePhaseEvidence(
                                source_index=0,
                                atom_count=16,
                                phase_label="fcc",
                                confidence_state="strong",
                                local_phase_fractions=(("fcc", 0.875), ("hcp", 0.0), ("bcc", 0.0), ("unresolved", 0.125)),
                            ),
                        ),
                    ),
                    CompositionPhaseEvidence(
                        reduced_counts=(1, 1),
                        source_structure_count=2,
                        analyzed_structure_count=2,
                        analyzed_atom_count=32,
                        local_phase_fractions=(("fcc", 0.125), ("hcp", 0.0), ("bcc", 0.3125), ("unresolved", 0.5625)),
                        structure_phase_fractions=(("bcc", 0.5), ("unresolved", 0.5)),
                        confidence_counts=(("mixed", 1), ("unresolved", 1)),
                        structures=(
                            StructurePhaseEvidence(
                                source_index=1,
                                atom_count=16,
                                phase_label="bcc",
                                confidence_state="mixed",
                                local_phase_fractions=(("fcc", 0.25), ("hcp", 0.0), ("bcc", 0.625), ("unresolved", 0.125)),
                            ),
                            StructurePhaseEvidence(
                                source_index=2,
                                atom_count=16,
                                phase_label="unresolved",
                                confidence_state="unresolved",
                                local_phase_fractions=(("fcc", 0.0), ("hcp", 0.0), ("bcc", 0.0), ("unresolved", 1.0)),
                            ),
                        ),
                    ),
                ),
            ),
        )
        widget = TrainingSetAuditWidget()

        widget.set_result(result)

        self.assertTrue(widget.composition_phase_summary_label.isHidden())
        self.assertEqual(widget.composition_chart._plot["kind"], "composition_stems")
        self.assertEqual(widget.composition_table.columnCount(), 5)
        self.assertIn("atoms", widget.composition_table.item(0, 3).text())
        self.assertEqual(widget.analyze_structure_evidence_button.text(), "Analyze remaining")
        widget.finish_phase_analysis(result)
        self.assertEqual(widget.composition_view_selector.currentData(), "count")
        widget.composition_evidence_button.click()
        widget.finish_phase_analysis(result)
        self.assertEqual(widget.composition_view_selector.currentData(), "structural")
        self.assertEqual(
            widget.composition_chart.plot_id,
            "inventory:composition-phase:Ni",
        )
        self.assertEqual(widget.composition_chart._plot["counts"], (1.0, 2.0))
        self.assertIn("Bar height is the sample count", widget.composition_phase_summary_label.text())

        selected = []
        widget.selectStructuresSignal.connect(selected.append)
        widget.composition_phase_selector.setCurrentIndex(1)
        widget.composition_show_button.click()

        self.assertEqual(selected, [[1]])
        self.assertIn("Mixed local structure", widget.composition_show_button.text())

        phase_row = next(
            row
            for row in range(widget.dimension_list.count())
            if widget.dimension_list.item(row).data(Qt.ItemDataRole.UserRole)
            == "phase_evidence"
        )
        widget.dimension_list.setCurrentRow(phase_row)
        self.assertEqual(widget.plot_selector.count(), 2)
        self.assertEqual(widget.chart_widget.plot_id, "phase_evidence:structure_labels")
        self.assertEqual(
            widget.chart_widget._plot["bar_ids"],
            ("fcc", "mixed", "unresolved"),
        )
        widget.plot_selector.setCurrentIndex(1)
        self.assertEqual(
            widget.chart_widget._plot["bar_ids"],
            ("strong", "mixed", "unresolved"),
        )

    def test_pending_phase_analysis_reports_full_progress_without_sampling_claims(self):
        base = self._dashboard_result()
        pending = replace(
            base,
            overview_metrics={
                **base.overview_metrics,
                "phase_inventory": {
                    "available": False,
                    "status": "pending",
                    "analyzed_structures": 0,
                },
            },
        )
        widget = TrainingSetAuditWidget()

        widget.set_result(pending)
        widget.start_phase_analysis(3)
        widget.update_phase_analysis_progress(2, 3)

        self.assertFalse(widget.composition_phase_progress.isHidden())
        self.assertFalse(widget.composition_map_progress.isHidden())
        self.assertEqual(widget.composition_phase_progress.value(), 67)
        self.assertEqual(widget.composition_map_progress.value(), 67)
        self.assertIn("2/3 structures", widget.analysis_status_label.text())
        self.assertNotIn("sampled", widget.analysis_status_label.text())
        self.assertEqual(widget.composition_chart.plot_id, "inventory:composition:Ni")
        self.assertIn("Calculating all structures", widget.dimension_list.item(1).text())

    def test_composition_target_distinguishes_supported_thin_and_missing_points(self):
        widget = TrainingSetAuditWidget()
        widget.set_result(self._dashboard_result())
        self.assertEqual(widget.target_table.rowCount(), 0)
        self.assertIn("No target has been set", widget.target_result_summary_label.text())
        widget.target_points_edit.setText("0, 25, 50")
        widget.target_quantity_rule_check.setChecked(True)
        widget.target_minimum_count_spin.setValue(2)

        widget.apply_target_button.click()

        self.assertEqual(widget.target_table.rowCount(), 3)
        self.assertEqual(widget.target_table.item(0, 2).text(), "Below your quantity rule")
        self.assertEqual(widget.target_table.item(1, 2).text(), "No exact composition sample")
        self.assertEqual(widget.target_table.item(2, 2).text(), "Meets your quantity rule")
        self.assertEqual(widget.target_chart.plot_id, "inventory:composition:Ni")

    def test_blank_target_key_points_uses_existing_fraction_points_in_range(self):
        widget = TrainingSetAuditWidget()
        widget.set_result(self._dashboard_result())
        widget.target_points_edit.clear()

        widget.apply_target_button.click()

        self.assertEqual(widget.target_table.rowCount(), 2)
        self.assertIn("No key points were entered", widget.target_result_summary_label.text())
        self.assertEqual(widget.target_table.item(0, 0).text(), "0.00%")
        self.assertEqual(widget.target_table.item(1, 0).text(), "50.00%")

    def test_target_structure_family_distinguishes_missing_metadata(self):
        widget = TrainingSetAuditWidget()
        result = replace(
            self._dashboard_result(),
            inventory=DatasetInventory(
                structure_count=3,
                elements=("Fe", "Ni"),
                composition_points=(
                    CompositionPoint(
                        (1, 1),
                        (0.5, 0.5),
                        3,
                        1.0,
                        (0, 1, 2),
                        config_types=(("bulk", 2),),
                        config_type_indices=(("bulk", (0, 1)),),
                        missing_config_type_count=1,
                    ),
                ),
                config_types=(("bulk", 2),),
                missing_config_type_count=1,
            ),
        )
        widget.set_result(result)
        widget.target_points_edit.setText("50")
        widget.target_config_types_edit.setText("bulk, vacancy")

        widget.apply_target_button.click()

        self.assertEqual(widget.target_table.rowCount(), 2)
        self.assertEqual(widget.target_table.item(0, 2).text(), "Exact samples available")
        self.assertEqual(
            widget.target_table.item(1, 2).text(),
            "Metadata incomplete; cannot fully evaluate",
        )

    def test_multinary_chart_aggregates_exact_compositions_at_same_element_fraction(self):
        widget = TrainingSetAuditWidget()
        result = AuditResult(
            dataset_id="ternary.xyz",
            generated_at="now",
            inputs={"structure_count": 9},
            overview_metrics={"structures": 9},
            inventory=DatasetInventory(
                structure_count=9,
                elements=("Co", "Ni", "V"),
                composition_points=(
                    CompositionPoint((1, 1, 0), (0.5, 0.5, 0.0), 3, 3 / 9, (0, 1, 2)),
                    CompositionPoint((1, 3, 0), (0.25, 0.75, 0.0), 2, 2 / 9, (3, 4)),
                    CompositionPoint((2, 1, 1), (0.5, 0.25, 0.25), 4, 4 / 9, (5, 6, 7, 8)),
                ),
            ),
        )

        widget.set_result(result)

        self.assertEqual(widget.composition_chart._plot["x_values"], (0.0, 0.25))
        self.assertEqual(widget.composition_chart._plot["counts"], (5.0, 4.0))
        self.assertEqual(widget.composition_table.rowCount(), 3)
        self.assertEqual(widget.target_table.rowCount(), 0)
        self.assertIn("one-element projection", widget.composition_map_hint.text())

    def test_composition_table_hands_exact_point_indices_to_dataset_display(self):
        widget = TrainingSetAuditWidget()
        received = []
        widget.selectStructuresSignal.connect(received.append)
        widget.set_result(self._dashboard_result())

        widget.composition_table.selectRow(0)
        widget.composition_show_button.click()

        self.assertEqual(received, [[1, 2]])

    def test_composition_map_keeps_counts_default_and_requests_optional_evidence(self):
        widget = TrainingSetAuditWidget()
        requests = []
        widget.requestStructureEvidenceSignal.connect(lambda: requests.append(True))

        widget.set_result(self._dashboard_result())

        self.assertEqual(widget.composition_view_selector.count(), 1)
        self.assertEqual(widget.composition_view_selector.currentData(), "count")
        self.assertFalse(widget.composition_evidence_button.isHidden())
        widget.composition_evidence_button.click()
        self.assertEqual(requests, [True])

    def test_overview_element_set_has_direct_structure_action(self):
        widget = TrainingSetAuditWidget()
        received = []
        widget.selectStructuresSignal.connect(received.append)
        widget.set_result(self._dashboard_result())

        self.assertEqual(widget.element_sets_table.item(0, 0).text(), "Fe–Ni")
        self.assertEqual(widget.element_sets_table.item(0, 1).text(), "2")
        self.assertEqual(widget.element_sets_table.item(0, 2).text(), "66.67%")
        widget.element_sets_table.selectRow(0)
        self.assertEqual(widget.view_element_set_button.text(), "View 2 structures")
        widget.view_element_set_button.click()

        self.assertEqual(received, [[1, 2]])

    def test_related_sparse_bins_are_grouped_into_one_topic(self):
        widget = TrainingSetAuditWidget()
        slices = tuple(
            AuditSlice(
                id=f"composition:Fe:{label}",
                title=f"Sparse Fe bin {label}",
                dimension_id="composition",
                severity=AuditSeverity.HIGH,
                bias_type=AuditBiasType.SPARSITY,
                structure_indices=indices,
                observed="Observed.",
                interpretation="Interpretation.",
                limit="Limit.",
            )
            for label, indices in (("0-5%", (0,)), ("5-20%", (1,)))
        )
        widget.set_result(
            AuditResult(
                dataset_id="grouped.xyz",
                generated_at="now",
                inputs={"structure_count": 10},
                dimensions=(
                    AuditDimension(
                        "composition",
                        "Composition",
                        AuditStatus.AVAILABLE,
                        plots=(self._histogram("composition:Fe", "Fe concentration"),),
                    ),
                ),
                slices=slices,
                overview_metrics={"structures": 10},
            )
        )

        self.assertEqual(widget.slice_table.rowCount(), 0)
        self.assertEqual(len(widget._all_topics), 1)
        self.assertEqual(len(widget._all_topics[0].source_slices), 2)

    def test_set_result_resets_to_summary_and_first_topic(self):
        widget = TrainingSetAuditWidget()
        first_result = self._dashboard_result()
        second_result = AuditResult(
            dataset_id="second.xyz",
            generated_at=first_result.generated_at,
            inputs=first_result.inputs,
            dimensions=first_result.dimensions,
            slices=first_result.slices,
            overview_metrics=first_result.overview_metrics,
        )
        widget.set_result(first_result)
        widget.slice_table.selectRow(1)
        widget.page_tabs.setCurrentIndex(1)

        widget.set_result(second_result)

        self.assertEqual(widget.page_tabs.currentIndex(), 0)
        self.assertEqual(widget.slice_table.currentRow(), 0)
        self.assertEqual(widget.selected_topic_label.text(), widget._topics[0].title)
        self.assertEqual(widget.slice_table.rowCount(), 1)

    def test_review_queue_expands_duplicate_groups_and_namespaces_states(self):
        duplicate_slice = AuditSlice(
            id="data_quality:exact_duplicates",
            title="Repeated geometries",
            dimension_id="data_quality",
            severity=AuditSeverity.HIGH,
            bias_type=AuditBiasType.REDUNDANCY,
            finding_kind=AuditFindingKind.REVIEW,
            structure_indices=(0, 1, 2, 3),
            observed="Four structures belong to two repeated-geometry groups.",
            interpretation="Review provenance.",
            limit="Repeated structures may be intentional.",
        )

        def result(dataset_id):
            return AuditResult(
                dataset_id=dataset_id,
                generated_at="now",
                inputs={"structure_count": 4},
                slices=(duplicate_slice,),
                overview_metrics={
                    "structures": 4,
                    "data_quality": {
                        "duplicate_group_count": 2,
                        "duplicate_groups": ((0, 1), (2, 3)),
                    },
                },
            )

        widget = TrainingSetAuditWidget()
        widget.set_result(result("first.xyz"))

        self.assertEqual(widget.slice_table.rowCount(), 2)
        self.assertEqual(widget._topics[0].structure_indices, (0, 1))
        self.assertEqual(widget._topics[1].structure_indices, (2, 3))
        self.assertEqual(
            [
                widget.review_state_selector.itemData(index)
                for index in range(widget.review_state_selector.count())
            ],
            ["pending", "keep", "isolate", "recalculate"],
        )
        widget.review_state_selector.setCurrentIndex(1)
        widget.apply_review_state_button.click()
        self.assertIn("Intentionally retained", widget.slice_table.item(0, 4).text())

        widget.set_result(result("second.xyz"))
        self.assertEqual(widget.slice_table.item(0, 4).text(), "Pending")

    def test_topic_categories_use_approved_foreground_colors(self):
        widget = TrainingSetAuditWidget()
        biases = (
            AuditBiasType.RISK_CONCENTRATION,
            AuditBiasType.SPARSITY,
            AuditBiasType.INFORMATIONAL,
        )
        slices = tuple(
            AuditSlice(
                id=f"custom:{bias.value}",
                title=f"{bias.value.title()} topic",
                dimension_id="custom",
                severity=AuditSeverity.INFO,
                bias_type=bias,
                structure_indices=(0,),
                observed="Observed.",
                interpretation="Interpretation.",
                limit="Limit.",
            )
            for bias in biases
        )
        widget.set_result(
            AuditResult(
                dataset_id="colors.xyz",
                generated_at="now",
                inputs={"structure_count": 1},
                slices=slices,
            )
        )

        self.assertEqual(
            widget.slice_table.item(0, 0).foreground().color().name(), "#a14d16"
        )
        self.assertEqual(widget.slice_table.rowCount(), 1)
        self.assertEqual(
            {topic.category for topic in widget._all_topics},
            {"review", "thin", "info"},
        )

    def test_findings_columns_adapt_and_restore_without_horizontal_scrolling(self):
        widget = TrainingSetAuditWidget()
        widget.set_result(self._dashboard_result())
        widget.page_tabs.setCurrentIndex(2)
        widget.resize(1280, 820)
        widget.show()
        self._app.processEvents()

        self.assertFalse(widget.slice_table.isColumnHidden(2))
        self.assertFalse(widget.slice_table.isColumnHidden(3))

        widget.resize(960, 680)
        self._app.processEvents()

        self.assertFalse(widget.slice_table.isColumnHidden(2))
        self.assertFalse(widget.slice_table.isColumnHidden(3))
        self.assertGreaterEqual(widget.slice_table.columnWidth(1), 240)
        self.assertEqual(
            widget.slice_table.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertFalse(widget.slice_table.horizontalScrollBar().isVisible())

        widget.resize(760, 680)
        self._app.processEvents()

        self.assertFalse(widget.slice_table.isColumnHidden(2))
        self.assertTrue(widget.slice_table.isColumnHidden(3))
        self.assertGreaterEqual(widget.slice_table.columnWidth(1), 240)
        self.assertFalse(widget.slice_table.horizontalScrollBar().isVisible())

        widget.resize(1280, 820)
        self._app.processEvents()

        self.assertFalse(widget.slice_table.isColumnHidden(2))
        self.assertFalse(widget.slice_table.isColumnHidden(3))
        widget.close()

    def test_overview_matrix_fills_available_space_without_page_scroll(self):
        widget = TrainingSetAuditWidget()
        counts = (100_000, 50_000, 12_000, 8_000, 3_331)
        fractions = (
            (0.20, 0.20, 0.20, 0.20, 0.20),
            (0.50, 0.25, 0.25, 0.00, 0.00),
            (1.00, 0.00, 0.00, 0.00, 0.00),
            (0.00, 1.00, 0.00, 0.00, 0.00),
            (0.00, 0.00, 1.00, 0.00, 0.00),
        )
        inventory = DatasetInventory(
            structure_count=173_331,
            elements=("Fe", "Co", "Ni", "Ta", "Al"),
            composition_points=tuple(
                CompositionPoint(
                    reduced_counts=tuple(round(value * 100) for value in point),
                    fractions=point,
                    structure_count=count,
                    share=count / 173_331,
                    structure_indices=(index,),
                )
                for index, (count, point) in enumerate(zip(counts, fractions))
            ),
            atom_counts=((64, 173_331),),
        )
        widget.set_result(replace(self._dashboard_result(), inventory=inventory))
        widget.resize(999, 650)
        widget.show()
        self._app.processEvents()

        self.assertEqual(
            widget.overview_columns.direction(),
            QBoxLayout.Direction.LeftToRight,
        )
        self.assertEqual(widget.cooccurrence_table.rowCount(), 5)
        self.assertEqual(widget.element_sets_table.rowCount(), 5)
        self.assertEqual(
            widget.element_sets_table.horizontalHeader().sectionResizeMode(1),
            QHeaderView.ResizeMode.Fixed,
        )
        self.assertEqual(
            widget.element_sets_table.horizontalHeader().sectionResizeMode(2),
            QHeaderView.ResizeMode.Fixed,
        )
        share_widths = [
            widget.element_sets_table.fontMetrics().horizontalAdvance(
                widget.element_sets_table.horizontalHeaderItem(2).text()
            ),
            *(
                widget.element_sets_table.fontMetrics().horizontalAdvance(
                    widget.element_sets_table.item(row, 2).text()
                )
                for row in range(widget.element_sets_table.rowCount())
            ),
        ]
        self.assertGreaterEqual(
            widget.element_sets_table.columnWidth(2),
            max(share_widths) + 40,
        )
        self.assertFalse(hasattr(widget, "summary_scroll"))
        self.assertFalse(
            widget.element_sets_table.scrollDelagate.vScrollBar._isForceHidden
        )
        matrix_width = sum(
            widget.cooccurrence_table.columnWidth(column)
            for column in range(widget.cooccurrence_table.columnCount())
        )
        matrix_height = sum(
            widget.cooccurrence_table.rowHeight(row)
            for row in range(widget.cooccurrence_table.rowCount())
        )
        self.assertLessEqual(
            abs(matrix_width - widget.cooccurrence_table.viewport().width()),
            widget.cooccurrence_table.columnCount(),
        )
        self.assertLessEqual(
            abs(matrix_height - widget.cooccurrence_table.viewport().height()),
            widget.cooccurrence_table.rowCount(),
        )

        widget.page_tabs.setCurrentIndex(1)
        widget.data_map_tabs.setCurrentIndex(1)
        self._app.processEvents()
        evidence_button = widget.analyze_structure_evidence_button
        self.assertGreaterEqual(evidence_button.width(), evidence_button.sizeHint().width())
        self.assertLessEqual(
            evidence_button.geometry().right(),
            widget.dimension_rail.contentsRect().right(),
        )

        widget.resize(1100, 760)
        self._app.processEvents()
        self.assertEqual(
            widget.overview_columns.direction(),
            QBoxLayout.Direction.LeftToRight,
        )
        widget.close()

    def test_selection_updates_evidence_and_selected_count_without_auto_handoff(self):
        widget = TrainingSetAuditWidget()
        received = []
        widget.selectStructuresSignal.connect(received.append)
        widget.set_result(self._dashboard_result())

        widget.slice_table.selectRow(0)

        self.assertIn("highest 10%", widget.observed_label.toPlainText())
        self.assertEqual(
            widget.send_button.text(),
            "Show 2 structures in Dataset Display",
        )
        widget.chart_widget.selectedGroupSignal.emit([0, 1, 2])
        self.assertEqual(widget.chart_selection_label.text(), "Chart selection: 3 structures")
        self.assertTrue(widget.chart_send_button.isEnabled())
        self.assertEqual(received, [])

    def test_review_topic_opens_its_related_distribution(self):
        widget = TrainingSetAuditWidget()
        widget.set_result(self._dashboard_result())
        widget.slice_table.selectRow(0)

        widget.view_distribution_button.click()

        self.assertEqual(widget.page_tabs.currentIndex(), 1)
        self.assertEqual(widget._selected_dimension_id(), "label_ranges")
        self.assertEqual(widget.chart_widget.plot_id, "label_ranges:max_force")

    def test_empty_result_keeps_dimensions_and_shows_exact_reasons(self):
        widget = TrainingSetAuditWidget()
        result = AuditResult(
            dataset_id="a-very-long-dataset-name-that-must-not-break-the-header.xyz",
            generated_at="now",
            inputs={"structure_count": 0},
            dimensions=(
                AuditDimension("composition", "Composition", AuditStatus.UNAVAILABLE, "No structures are loaded."),
                AuditDimension("label_ranges", "Label ranges", AuditStatus.UNAVAILABLE, "No labels are available."),
            ),
            slices=(),
            overview_metrics={"structures": 0, "finding_count": 0, "severity_counts": {}},
        )

        widget.set_result(result)
        widget.dimension_list.setCurrentRow(3)

        self.assertEqual(widget.dimension_list.count(), 4)
        self.assertEqual(widget.metric_structure_value.text(), "0")
        self.assertEqual(widget.slice_table.rowCount(), 0)
        self.assertEqual(widget.analysis_status_label.text(), "No labels are available.")
        self.assertIn("No review topic", widget.findings_empty_label.text())

    def test_no_result_starts_quiet_and_set_result_restores_dashboard(self):
        widget = TrainingSetAuditWidget()

        self.assertFalse(widget.no_dataset_state.isHidden())
        self.assertFalse(widget.no_dataset_action_button.isHidden())
        self.assertIn("NEP Dataset Display", widget.no_dataset_hint.text())
        self.assertTrue(widget.audit_header.isHidden())
        self.assertTrue(widget.dashboard_body.isHidden())
        self.assertEqual(widget.dimension_list.count(), 0)
        self.assertEqual(widget.slice_table.rowCount(), 0)
        self.assertFalse(widget.chart_widget.has_data)
        self.assertFalse(widget.send_button.isEnabled())

        widget.set_result(self._dashboard_result())

        self.assertTrue(widget.no_dataset_state.isHidden())
        self.assertTrue(widget.no_dataset_panel.isHidden())
        self.assertFalse(widget.audit_header.isHidden())
        self.assertFalse(widget.dashboard_body.isHidden())
        self.assertEqual(widget.dimension_list.count(), 4)
        self.assertFalse(widget.chart_widget.isHidden())

    def test_empty_state_reserves_full_wrapped_hint_height(self):
        widget = TrainingSetAuditWidget()
        widget.resize(600, 260)
        widget.show()
        self._app.processEvents()

        hint = widget.no_dataset_hint
        self.assertGreaterEqual(widget.no_dataset_panel.width(), 430)
        self.assertGreaterEqual(
            widget.no_dataset_state.height(),
            widget.no_dataset_state.sizeHint().height(),
        )
        self.assertGreater(hint.width(), 0)
        self.assertGreaterEqual(hint.height(), hint.heightForWidth(hint.width()))

    def test_empty_state_remeasures_hint_after_late_font_polish(self):
        widget = TrainingSetAuditWidget()
        widget.no_dataset_hint.setStyleSheet("font-size: 24px;")
        widget.resize(1151, 651)
        widget.show()
        self._app.processEvents()
        self._app.processEvents()

        hint = widget.no_dataset_hint
        self.assertGreaterEqual(
            hint.minimumHeight(),
            hint.heightForWidth(hint.width()),
        )

    def test_empty_state_open_action_requests_dataset_picker(self):
        widget = TrainingSetAuditWidget()
        requests = []
        widget.requestDatasetOpenSignal.connect(lambda: requests.append(True))

        widget.no_dataset_action_button.click()

        self.assertEqual(requests, [True])

    def test_loading_state_hides_open_action(self):
        widget = TrainingSetAuditWidget()

        widget.set_loading("train.xyz")

        self.assertIn("train.xyz", widget.no_dataset_state.text())
        self.assertTrue(widget.no_dataset_action_button.isHidden())
        self.assertIn("wait", widget.no_dataset_hint.text().lower())

    def test_display_text_uses_widget_translation_before_ui_construction(self):
        class TranslationProbeWidget(TrainingSetAuditWidget):
            def tr(self, source_text, disambiguation=None, n=-1):
                del disambiguation, n
                return f"translated::{source_text}"

        widget = TranslationProbeWidget()
        result = AuditResult(
            dataset_id="train.xyz",
            generated_at="now",
            inputs={"structure_count": 1},
            slices=(
                AuditSlice(
                    id="label_ranges:force_high_tail",
                    title="High force tail",
                    dimension_id="label_ranges",
                    severity=AuditSeverity.HIGH,
                    bias_type=AuditBiasType.RISK_CONCENTRATION,
                    structure_indices=(0,),
                    observed="Observed value remains a domain value.",
                    interpretation="Interpretation remains a domain value.",
                    limit="Limit remains a domain value.",
                ),
            ),
        )

        widget.set_result(result)

        self.assertEqual(widget.header_label.text(), "translated::Training Set Audit")
        self.assertEqual(widget.slice_table.horizontalHeaderItem(0).text(), "translated::Type")
        self.assertEqual(widget.slice_table.item(0, 0).text(), "translated::Review")
        self.assertEqual(
            widget.slice_table.item(0, 1).text(),
            "translated::Maximum-force review group (top 10%)",
        )
        self.assertTrue(widget.observed_label.toPlainText().startswith("translated::1 structures"))

    def test_shipped_chinese_catalog_translates_page_owned_audit_text(self):
        catalog = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "NepTrainKit"
            / "translations"
            / "neptrainkit_zh_CN.qm"
        )
        translator = QTranslator(self._app)
        self.assertTrue(translator.load(str(catalog)))
        self._app.installTranslator(translator)
        try:
            widget = TrainingSetAuditWidget()
            self.assertEqual(widget.no_dataset_state.text(), "未加载数据集")
            self.assertEqual(widget.no_dataset_action_button.text(), "打开数据集")
            self.assertIn("NEP 数据集查看", widget.no_dataset_hint.text())
            self.assertEqual(widget.header_label.text(), "训练集评估")
            self.assertEqual(widget.slice_table.horizontalHeaderItem(0).text(), "类型")
            self.assertEqual(widget.dimension_rail.findChild(QLabel, "panelTitle").text(), "检查项目")
            self.assertEqual(widget.rerun_button.text(), "重新检查")
            self.assertEqual(widget.export_report_button.text(), "导出 HTML 报告")
            self.assertEqual(widget.fact_total_atoms_label.text(), "原子总数")
            self.assertEqual(widget.fact_atom_range_label.text(), "每结构原子数")
            self.assertFalse(hasattr(widget, "fact_config_label"))
            self.assertEqual(
                widget.cooccurrence_hint.text(),
                "上三角：全局元素对共现 · 对角线：元素出现率 · "
                "选择上三角元素对后，下方显示关联的第三、第四元素",
            )
            self.assertEqual(
                widget.matrix_selection_label.text(),
                "未筛选 · 单击单元格",
            )
            self.assertEqual(
                widget.element_sets_table.horizontalHeaderItem(0).text(),
                "元素集合",
            )
            self.assertEqual(
                widget.heat_legend_bar.accessibleName(),
                "相对数量色标",
            )
            self.assertIn(
                "相对数量",
                [label.text() for label in widget.cooccurrence_panel.findChildren(QLabel)],
            )
            self.assertEqual(
                widget.target_quantity_rule_check.text(), "启用最低支持量规则"
            )
            self.assertEqual(
                widget.analyze_structure_evidence_button.text(), "分析证据"
            )
            self.assertEqual(
                QCoreApplication.translate(
                    "TrainingSetAuditWidget",
                    "Analyzing local phases: {completed:,}/{total:,} structures. The chart will update automatically.",
                ),
                "正在分析局域相：{completed:,}/{total:,} 个结构。图表将在完成后自动更新。",
            )
            self.assertEqual(
                QCoreApplication.translate(
                    "TrainingSetAuditWidget",
                    "Structure-level phase labels",
                ),
                "结构级相别",
            )
            energy_plot = {
                "kind": "histogram",
                "id": "label_ranges:energy_per_atom",
                "title": "Energy per atom distribution",
                "x_label": "Energy per atom",
                "y_label": "Structures",
                "series": (
                    {
                        "id": "energy_per_atom",
                        "label": "Energy per atom",
                        "counts": (1, 1),
                        "bin_edges": (-2.0, -1.0, 0.0),
                        "structure_indices": ((0,), (1,)),
                    },
                ),
            }
            result = AuditResult(
                dataset_id="train.xyz",
                generated_at="2026-07-10T08:30:00+00:00",
                inputs={"structure_count": 2},
                dimensions=(
                    AuditDimension(
                        "label_ranges",
                        "Label ranges",
                        AuditStatus.AVAILABLE,
                        plots=(energy_plot,),
                    ),
                ),
                overview_metrics={
                    "structures": 2,
                    "finding_count": 0,
                    "severity_counts": {},
                    "label_counts": {"energy": 2, "force": 0, "virial": 0},
                },
            )
            widget.set_result(result)
            widget.dimension_list.setCurrentRow(2)
            self.assertEqual(widget.chart_widget._plot["title"], "单原子能量分布")
            self.assertEqual(widget.chart_widget._plot["x_label"], "单原子能量")
            self.assertEqual(widget.chart_widget._plot["y_label"], "结构数")
            self.assertEqual(
                widget._active_plots[0]["series"][0]["label"], "单原子能量"
            )
            self.assertEqual(energy_plot["title"], "Energy per atom distribution")
            self.assertEqual(energy_plot["x_label"], "Energy per atom")

            chart = AuditChartWidget()
            self.assertEqual(chart.empty_state_text, "没有可用的数值分布")
        finally:
            self._app.removeTranslator(translator)

    def test_local_chemistry_shipped_chinese_catalog_translates_runtime_controls(self):
        catalog = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "NepTrainKit"
            / "translations"
            / "neptrainkit_zh_CN.qm"
        )
        translator = QTranslator(self._app)
        self.assertTrue(translator.load(str(catalog)))
        self._app.installTranslator(translator)
        try:
            widget = TrainingSetAuditWidget()
            widget.set_result(self._local_chemistry_result())
            widget.dimension_list.setCurrentRow(2)

            self.assertTrue(widget.dimension_list.item(2).text().startswith("局域环境支持\n"))
            self.assertEqual(widget.local_scope_selector.itemText(0), "角向邻居")
            self.assertEqual(widget.local_scope_selector.itemText(1), "径向邻居")
            self.assertEqual(widget.local_center_label.text(), "中心元素")
            self.assertEqual(widget.plot_selector.itemText(0), "邻居数")
            self.assertEqual(widget.plot_selector.itemText(1), "Fe 邻居比例")
            self.assertEqual(
                widget.analysis_status_label.text(),
                "范围来自当前 NEP 截断半径 · 角向邻居 · 中心元素 Fe。"
                "橙色表示当前数据中样本较少的区间。",
            )
            self.assertEqual(widget.chart_widget._plot["title"], "角向邻居：Fe 邻居数")
            self.assertEqual(widget.chart_widget._plot["y_label"], "局域环境数")
        finally:
            self._app.removeTranslator(translator)

    def test_available_plot_remains_visible_when_there_are_no_findings(self):
        widget = TrainingSetAuditWidget()
        result = AuditResult(
            dataset_id="quiet.xyz",
            generated_at="now",
            inputs={"structure_count": 3},
            dimensions=(
                AuditDimension(
                    "composition",
                    "Composition",
                    AuditStatus.AVAILABLE,
                    plots=(self._histogram("composition:Fe", "Fe concentration"),),
                ),
            ),
            slices=(),
            overview_metrics={"structures": 3, "finding_count": 0},
        )

        widget.set_result(result)
        widget.dimension_list.setCurrentRow(2)

        self.assertEqual(widget.chart_widget.plot_id, "composition:Fe")
        self.assertFalse(widget.plot_selector.isVisibleTo(widget))
        self.assertEqual(widget.slice_table.rowCount(), 0)
        self.assertIn("No review topic", widget.findings_empty_label.text())

    def test_partial_reason_and_long_finding_title_remain_available(self):
        widget = TrainingSetAuditWidget()
        result = self._dashboard_result()
        long_title = "High force tail from an intentionally long scientific dataset slice name"
        long_slice = AuditSlice(
            id="label_ranges:long",
            title=long_title,
            dimension_id="label_ranges",
            severity=AuditSeverity.LOW,
            bias_type=AuditBiasType.INFORMATIONAL,
            structure_indices=(0,),
            observed="Observed evidence remains readable.",
            interpretation="Interpretation remains readable.",
            limit="Limit remains readable.",
        )
        result = AuditResult(
            dataset_id=result.dataset_id,
            generated_at=result.generated_at,
            inputs=result.inputs,
            dimensions=result.dimensions,
            slices=result.slices + (long_slice,),
            overview_metrics={**result.overview_metrics, "finding_count": 3},
        )

        widget.set_result(result)
        widget.dimension_list.setCurrentRow(3)

        self.assertEqual(
            widget.analysis_status_label.text(),
            "Available on labeled subsets only: force (2/3).",
        )
        self.assertIn(long_title, {topic.title for topic in widget._all_topics})
        self.assertNotIn(
            long_title,
            {
                widget.slice_table.item(row, 1).text()
                for row in range(widget.slice_table.rowCount())
            },
        )

    def test_set_result_populates_slice_table_and_evidence(self):
        widget = TrainingSetAuditWidget()
        result = AuditResult(
            dataset_id="train.xyz",
            generated_at="now",
            inputs={"structure_count": 2},
            slices=(
                AuditSlice(
                    id="label_ranges:force_high_tail",
                    title="High force tail",
                    dimension_id="label_ranges",
                    severity=AuditSeverity.HIGH,
                    bias_type=AuditBiasType.RISK_CONCENTRATION,
                    structure_indices=(0, 1),
                    observed="Two structures are in the force tail.",
                    interpretation="Inspect these structures.",
                    limit="High force is not automatically wrong.",
                ),
            ),
        )

        widget.set_result(result)

        self.assertEqual(widget.slice_table.rowCount(), 1)
        self.assertIn("Maximum-force review group", widget.slice_table.item(0, 1).text())
        widget.slice_table.selectRow(0)
        self.assertIn("highest 10%", widget.observed_label.toPlainText())
        self.assertIn("review group", widget.limit_label.toPlainText())

    def test_send_button_emits_selected_structure_indices(self):
        widget = TrainingSetAuditWidget()
        result = AuditResult(
            dataset_id="train.xyz",
            generated_at="now",
            inputs={"structure_count": 3},
            slices=(
                AuditSlice(
                    id="label_ranges:force_high_tail",
                    title="High force tail",
                    dimension_id="label_ranges",
                    severity=AuditSeverity.HIGH,
                    bias_type=AuditBiasType.RISK_CONCENTRATION,
                    structure_indices=(1, 2),
                    observed="Two structures are in the force tail.",
                    interpretation="Inspect these structures.",
                    limit="High force is not automatically wrong.",
                ),
            ),
        )
        received = []
        widget.selectStructuresSignal.connect(received.append)

        widget.set_result(result)
        widget.slice_table.selectRow(0)
        widget.send_button.click()

        self.assertEqual(received, [[1, 2]])

    def test_detail_dimension_selection_does_not_hide_summary_topics(self):
        widget = TrainingSetAuditWidget()
        result = AuditResult(
            dataset_id="train.xyz",
            generated_at="now",
            inputs={"structure_count": 3},
            slices=(
                AuditSlice(
                    id="label_ranges:force_high_tail",
                    title="High force tail",
                    dimension_id="label_ranges",
                    severity=AuditSeverity.HIGH,
                    bias_type=AuditBiasType.RISK_CONCENTRATION,
                    structure_indices=(1, 2),
                    observed="Two structures are in the force tail.",
                    interpretation="Inspect these structures.",
                    limit="High force is not automatically wrong.",
                ),
                AuditSlice(
                    id="composition:Fe:sparse",
                    title="Sparse Fe composition bin",
                    dimension_id="composition",
                    severity=AuditSeverity.MEDIUM,
                    bias_type=AuditBiasType.SPARSITY,
                    structure_indices=(0, 1, 2),
                    observed="Three structures are in this Fe bin.",
                    interpretation="This composition bin is represented by few structures.",
                    limit="May match the study target.",
                ),
            ),
        )

        widget.set_result(result)
        widget.dimension_list.setCurrentRow(2)

        self.assertEqual(widget.slice_table.rowCount(), 1)
        self.assertEqual(len(widget._all_topics), 2)
        self.assertEqual(widget.slice_table.item(0, 0).text(), "Review")

    def test_dimension_filter_defaults_to_all_dimensions(self):
        widget = TrainingSetAuditWidget()
        result = AuditResult(
            dataset_id="train.xyz",
            generated_at="now",
            inputs={"structure_count": 3},
            dimensions=(
                AuditDimension("composition", "Composition", AuditStatus.AVAILABLE),
                AuditDimension("label_ranges", "Label ranges", AuditStatus.AVAILABLE),
            ),
            slices=(
                AuditSlice(
                    id="label_ranges:force_high_tail",
                    title="High force tail",
                    dimension_id="label_ranges",
                    severity=AuditSeverity.HIGH,
                    bias_type=AuditBiasType.RISK_CONCENTRATION,
                    structure_indices=(1, 2),
                    observed="Two structures are in the force tail.",
                    interpretation="Inspect these structures.",
                    limit="High force is not automatically wrong.",
                ),
            ),
        )

        widget.set_result(result)

        self.assertEqual(widget.dimension_list.item(0).text(), "Overview\n1 topic")
        self.assertEqual(widget.slice_table.rowCount(), 1)
        self.assertEqual(widget.slice_table.item(0, 0).text(), "Review")

    def test_data_blocker_is_shown_before_review_signals(self):
        widget = TrainingSetAuditWidget()
        result = AuditResult(
            dataset_id="train.xyz",
            generated_at="now",
            inputs={"structure_count": 3},
            dimensions=(
                AuditDimension("data_quality", "Data quality", AuditStatus.AVAILABLE),
            ),
            slices=(
                AuditSlice(
                    id="data_quality:nonfinite_labels",
                    title="Non-finite label values",
                    dimension_id="data_quality",
                    severity=AuditSeverity.HIGH,
                    bias_type=AuditBiasType.RISK_CONCENTRATION,
                    structure_indices=(2,),
                    observed="One structure contains a non-finite force.",
                    interpretation="Training loss would become non-finite.",
                    limit="Missing labels are not included.",
                    finding_kind=AuditFindingKind.BLOCKER,
                    rule="Present labels must be finite.",
                ),
            ),
            overview_metrics={"structures": 3},
        )

        widget.set_result(result)

        self.assertEqual(widget.slice_table.item(0, 0).text(), "Data blocker")
        self.assertFalse(hasattr(widget, "summary_conclusion_label"))
        self.assertIsNone(widget.findChild(QFrame, "auditSummaryPanel"))
        self.assertEqual(widget.metric_structure_value.text(), "3")

    def test_rerun_button_emits_signal(self):
        widget = TrainingSetAuditWidget()
        received = []
        widget.rerunAuditSignal.connect(lambda: received.append("rerun"))

        widget.rerun_button.click()

        self.assertEqual(received, ["rerun"])

    def test_export_report_writes_html(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        widget = TrainingSetAuditWidget()
        result = AuditResult(
            dataset_id="train.xyz",
            generated_at="now",
            inputs={"structure_count": 0},
            slices=(),
        )
        widget.set_result(result)

        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "audit.html"
            widget.export_report(target)
            html = target.read_text(encoding="utf-8")

        self.assertIn("Training Set Audit Report", html)
        self.assertIn("Findings describe this dataset only", html)


class TestTrainingSetAuditPageExports(unittest.TestCase):
    def test_pages_module_resolves_training_set_audit_widget(self):
        self.assertIs(ui_pages.TrainingSetAuditWidget, TrainingSetAuditWidget)

    def test_pages_module_all_contains_main_page_exports(self):
        expected = {
            "MakeDataWidget",
            "SettingsWidget",
            "ShowNepWidget",
            "DataManagerWidget",
            "TrainingSetAuditWidget",
        }

        self.assertTrue(expected.issubset(set(ui_pages.__all__)))

    def test_pages_module_can_be_imported_without_optional_page_dependencies(self):
        module = importlib.import_module("NepTrainKit.ui.pages")

        self.assertIs(module, ui_pages)


if __name__ == "__main__":
    unittest.main()
