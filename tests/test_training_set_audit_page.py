#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import unittest
from pathlib import Path

from PySide6.QtCore import QTranslator, Qt
from PySide6.QtWidgets import QApplication, QLabel
from qfluentwidgets import ComboBox, ListWidget, PrimaryPushButton, TableWidget

from NepTrainKit.core.audit.result import (
    AuditBiasType,
    AuditDimension,
    AuditResult,
    AuditSeverity,
    AuditSlice,
    AuditStatus,
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
        )

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
        widget.dimension_list.setCurrentRow(1)
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
            "Active NEP model · Angular core · effective pair cutoff is the mean of center and "
            "neighbor cutoffs · center Fe",
        )

        widget.local_center_selector.setCurrentIndex(1)
        widget.local_scope_selector.setCurrentIndex(1)

        self.assertEqual(widget.local_center_selector.currentData(), "Ni")
        self.assertEqual(widget.plot_selector.count(), 2)
        self.assertTrue(widget.chart_widget.plot_id.startswith("local_chemistry:radial:Ni:"))
        self.assertIn("Radial context", widget.analysis_status_label.text())

        widget.dimension_list.setCurrentRow(0)
        self.assertTrue(widget.local_scope_selector.isHidden())
        self.assertTrue(widget.local_center_selector.isHidden())
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
        widget.dimension_list.setCurrentRow(1)

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
        widget.dimension_list.setCurrentRow(1)

        self.assertEqual(widget.plot_selector.itemText(0), "Neighbor count")
        self.assertEqual(widget.plot_selector.itemText(1), "Fe neighbor fraction")
        self.assertTrue(widget.chart_widget.plot_id.startswith("local_chemistry:angular:Fe:"))

    def test_dashboard_exposes_approved_layout_contract(self):
        widget = TrainingSetAuditWidget()

        widget.set_result(self._dashboard_result())

        self.assertEqual(widget.dimension_list.count(), 3)
        self.assertEqual(widget.metric_structure_value.text(), "3")
        self.assertEqual(widget.generated_at_label.text(), "Generated 2026-07-10 08:30 UTC")
        self.assertIsInstance(widget.rerun_button, PrimaryPushButton)
        self.assertNotIsInstance(widget.export_report_button, PrimaryPushButton)
        self.assertIsInstance(widget.dimension_list, ListWidget)
        self.assertIsInstance(widget.local_scope_selector, ComboBox)
        self.assertIsInstance(widget.local_center_selector, ComboBox)
        self.assertIsInstance(widget.plot_selector, ComboBox)
        self.assertIsInstance(widget.slice_table, TableWidget)
        self.assertEqual(widget.metric_findings_label.text(), "Flagged slices (H/M/L)")
        self.assertEqual(widget.metric_findings_value.text(), "2 · 1/1/0")
        self.assertEqual(widget.metric_dimension_label.text(), "Available label metrics")
        self.assertEqual(widget.metric_dimension_value.text(), "2")
        self.assertEqual(widget.metric_context_label.text(), "Label completeness")
        self.assertEqual(widget.metric_context_value.text(), "5 / 9")
        self.assertEqual(widget.dimension_list.item(0).text(), "Overview\n2 findings")
        self.assertEqual(widget.dimension_list.item(1).text(), "Composition\nAvailable · 1 finding")
        self.assertEqual(widget.dimension_list.item(2).text(), "Label ranges\nPartial · 1 finding")
        self.assertEqual(widget.label_availability_title.text(), "Label availability")
        self.assertEqual(
            widget.label_availability_value.text(),
            "Energy 3/3\nForce 2/3\nVirial 0/3",
        )
        self.assertEqual(widget.analysis_tabs.count(), 2)
        self.assertEqual(widget.chart_widget.plot_id, "composition:Fe")
        self.assertEqual(widget.slice_table.rowCount(), 2)
        self.assertEqual(widget.audit_header.objectName(), "auditHeader")
        self.assertEqual(widget.dimension_rail.objectName(), "auditDimensionRail")
        self.assertEqual(widget.metric_band.objectName(), "auditMetricBand")
        self.assertEqual(widget.analysis_panel.objectName(), "auditAnalysisPanel")
        self.assertEqual(widget.findings_panel.objectName(), "auditFindingsPanel")

    def test_dimension_selection_updates_plots_findings_and_unavailable_reason(self):
        widget = TrainingSetAuditWidget()
        widget.set_result(self._dashboard_result())

        widget.dimension_list.setCurrentRow(1)
        self.assertEqual(widget.metric_dimension_label.text(), "Sparse bins")
        self.assertEqual(widget.metric_dimension_value.text(), "1")
        self.assertEqual(widget.metric_context_label.text(), "Dimension status")
        self.assertEqual(widget.metric_context_value.text(), "Available")
        self.assertEqual(widget.plot_selector.count(), 2)
        self.assertTrue(widget.plot_selector.isVisibleTo(widget))
        self.assertEqual(widget.chart_widget.plot_id, "composition:Fe")
        self.assertEqual(widget.slice_table.rowCount(), 1)

        widget.plot_selector.setCurrentIndex(1)
        self.assertEqual(widget.chart_widget.plot_id, "composition:O")

        widget.dimension_list.setCurrentRow(2)
        self.assertEqual(widget.metric_dimension_label.text(), "Available label metrics")
        self.assertEqual(widget.metric_dimension_value.text(), "2")
        self.assertEqual(widget.metric_context_label.text(), "Label completeness")
        self.assertEqual(widget.metric_context_value.text(), "5 / 9")
        self.assertFalse(widget.chart_widget.isHidden())

    def test_severity_filter_composes_with_dimension_filter(self):
        widget = TrainingSetAuditWidget()
        widget.set_result(self._dashboard_result())

        widget.severity_buttons[AuditSeverity.HIGH].click()
        self.assertEqual(widget.slice_table.rowCount(), 1)
        self.assertEqual(widget.slice_table.item(0, 0).text(), "High")

        widget.dimension_list.setCurrentRow(2)
        self.assertEqual(widget.slice_table.rowCount(), 0)

        widget.severity_buttons[AuditSeverity.MEDIUM].click()
        self.assertEqual(widget.slice_table.rowCount(), 1)
        self.assertEqual(widget.slice_table.item(0, 2).text(), "Label ranges")

    def test_set_result_resets_severity_filter_and_active_style_each_time(self):
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
        widget.show()
        self._app.processEvents()

        def rendered_background(button):
            image = button.grab().toImage()
            return image.pixelColor(image.width() - 8, image.height() // 2).name()

        widget.set_result(first_result)
        medium_button = widget.severity_buttons[AuditSeverity.MEDIUM]
        medium_button.click()
        self._app.processEvents()
        self.assertTrue(medium_button.isChecked())
        self.assertTrue(medium_button.property("severityFilterActive"))
        self.assertIn("#087f78", medium_button.styleSheet())
        self.assertEqual(rendered_background(medium_button), "#087f78")
        self.assertEqual(rendered_background(widget.severity_all_button), "#ffffff")

        widget.set_result(second_result)
        self._app.processEvents()

        self.assertTrue(widget.severity_all_button.isChecked())
        self.assertTrue(widget.severity_all_button.property("severityFilterActive"))
        self.assertIn("#087f78", widget.severity_all_button.styleSheet())
        self.assertEqual(rendered_background(widget.severity_all_button), "#087f78")
        self.assertEqual(rendered_background(medium_button), "#ffffff")
        self.assertFalse(medium_button.isChecked())
        self.assertFalse(medium_button.property("severityFilterActive"))
        self.assertEqual(widget.slice_table.rowCount(), 2)
        widget.close()

    def test_finding_severity_uses_approved_foreground_colors(self):
        widget = TrainingSetAuditWidget()
        slices = tuple(
            AuditSlice(
                id=f"composition:{severity.value}",
                title=f"{severity.value.title()} finding",
                dimension_id="composition",
                severity=severity,
                bias_type=AuditBiasType.INFORMATIONAL,
                structure_indices=(0,),
                observed="Observed.",
                interpretation="Interpretation.",
                limit="Limit.",
            )
            for severity in (
                AuditSeverity.HIGH,
                AuditSeverity.MEDIUM,
                AuditSeverity.LOW,
            )
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
            widget.slice_table.item(0, 0).foreground().color().name(), "#c94932"
        )
        self.assertEqual(
            widget.slice_table.item(1, 0).foreground().color().name(), "#d08a17"
        )
        self.assertEqual(
            widget.slice_table.item(2, 0).foreground().color().name(), "#89979b"
        )

    def test_findings_columns_adapt_and_restore_without_horizontal_scrolling(self):
        widget = TrainingSetAuditWidget()
        widget.set_result(self._dashboard_result())
        widget.resize(1280, 820)
        widget.show()
        self._app.processEvents()

        self.assertFalse(widget.slice_table.isColumnHidden(2))
        self.assertFalse(widget.slice_table.isColumnHidden(4))

        widget.resize(960, 680)
        self._app.processEvents()

        self.assertFalse(widget.slice_table.isColumnHidden(2))
        self.assertTrue(widget.slice_table.isColumnHidden(4))
        self.assertGreaterEqual(widget.slice_table.columnWidth(1), 240)
        self.assertEqual(
            widget.slice_table.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertFalse(widget.slice_table.horizontalScrollBar().isVisible())

        widget.resize(760, 680)
        self._app.processEvents()

        self.assertTrue(widget.slice_table.isColumnHidden(2))
        self.assertTrue(widget.slice_table.isColumnHidden(4))
        self.assertGreaterEqual(widget.slice_table.columnWidth(1), 240)
        self.assertFalse(widget.slice_table.horizontalScrollBar().isVisible())

        widget.resize(1280, 820)
        self._app.processEvents()

        self.assertFalse(widget.slice_table.isColumnHidden(2))
        self.assertFalse(widget.slice_table.isColumnHidden(4))
        widget.close()

    def test_selection_updates_evidence_and_selected_count_without_auto_handoff(self):
        widget = TrainingSetAuditWidget()
        received = []
        widget.selectStructuresSignal.connect(received.append)
        widget.set_result(self._dashboard_result())

        widget.slice_table.selectRow(1)

        self.assertIn("Two structures", widget.observed_label.toPlainText())
        self.assertEqual(
            widget.send_button.text(),
            "Show 2 structures in Dataset Display",
        )
        widget.chart_widget.selectedGroupSignal.emit([0, 1, 2])
        self.assertEqual(widget.chart_selection_label.text(), "Chart selection: 3 structures")
        self.assertEqual(received, [])

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
        widget.dimension_list.setCurrentRow(2)

        self.assertEqual(widget.dimension_list.count(), 3)
        self.assertEqual(widget.metric_structure_value.text(), "0")
        self.assertEqual(widget.slice_table.rowCount(), 0)
        self.assertEqual(widget.analysis_status_label.text(), "No labels are available.")
        self.assertIn("No findings", widget.findings_empty_label.text())

    def test_no_result_starts_quiet_and_set_result_restores_dashboard(self):
        widget = TrainingSetAuditWidget()

        self.assertFalse(widget.no_dataset_state.isHidden())
        self.assertTrue(widget.audit_header.isHidden())
        self.assertTrue(widget.dashboard_body.isHidden())
        self.assertEqual(widget.dimension_list.count(), 0)
        self.assertEqual(widget.slice_table.rowCount(), 0)
        self.assertFalse(widget.chart_widget.has_data)
        self.assertFalse(widget.send_button.isEnabled())

        widget.set_result(self._dashboard_result())

        self.assertTrue(widget.no_dataset_state.isHidden())
        self.assertFalse(widget.audit_header.isHidden())
        self.assertFalse(widget.dashboard_body.isHidden())
        self.assertEqual(widget.dimension_list.count(), 3)
        self.assertFalse(widget.chart_widget.isHidden())

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
        self.assertEqual(widget.slice_table.horizontalHeaderItem(0).text(), "translated::Severity")
        self.assertEqual(widget.slice_table.item(0, 2).text(), "translated::Label ranges")
        self.assertEqual(widget.slice_table.item(0, 4).text(), "translated::Risk concentration")
        self.assertEqual(widget.observed_label.toPlainText(), "Observed value remains a domain value.")

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
            self.assertEqual(widget.header_label.text(), "训练集审计")
            self.assertEqual(widget.slice_table.horizontalHeaderItem(0).text(), "严重程度")
            self.assertEqual(widget.dimension_rail.findChild(QLabel, "panelTitle").text(), "审计维度")
            self.assertEqual(widget.rerun_button.text(), "重新运行审计")
            self.assertEqual(widget.export_report_button.text(), "导出 HTML 报告")

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
            widget.dimension_list.setCurrentRow(1)
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
            widget.dimension_list.setCurrentRow(1)

            self.assertTrue(widget.dimension_list.item(1).text().startswith("局域化学\n"))
            self.assertEqual(widget.local_scope_selector.itemText(0), "角向核心")
            self.assertEqual(widget.local_scope_selector.itemText(1), "径向环境")
            self.assertEqual(widget.local_center_label.text(), "中心元素")
            self.assertEqual(widget.plot_selector.itemText(0), "邻居数")
            self.assertEqual(widget.plot_selector.itemText(1), "Fe 邻居比例")
            self.assertEqual(
                widget.analysis_status_label.text(),
                "当前 NEP 模型 · 角向核心 · 元素对有效截断半径取中心元素与邻居元素"
                "截断半径的平均值 · 中心元素 Fe",
            )
            self.assertEqual(widget.chart_widget._plot["title"], "角向核心：Fe 邻居数")
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
        widget.dimension_list.setCurrentRow(1)

        self.assertEqual(widget.chart_widget.plot_id, "composition:Fe")
        self.assertFalse(widget.plot_selector.isVisibleTo(widget))
        self.assertEqual(widget.slice_table.rowCount(), 0)
        self.assertIn("No findings", widget.findings_empty_label.text())

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
        widget.dimension_list.setCurrentRow(2)

        self.assertEqual(
            widget.analysis_status_label.text(),
            "Available on labeled subsets only: force (2/3).",
        )
        self.assertEqual(widget.slice_table.item(1, 1).text(), long_title)
        self.assertEqual(widget.slice_table.item(1, 1).toolTip(), long_title)

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
        self.assertIn("High force tail", widget.slice_table.item(0, 1).text())
        widget.slice_table.selectRow(0)
        self.assertIn("Two structures", widget.observed_label.toPlainText())
        self.assertIn("High force", widget.limit_label.toPlainText())

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

    def test_dimension_filter_limits_visible_slices(self):
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
        widget.dimension_list.setCurrentRow(1)

        self.assertEqual(widget.slice_table.rowCount(), 1)
        self.assertEqual(widget.slice_table.item(0, 2).text(), "Label ranges")

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

        self.assertEqual(widget.dimension_list.item(0).text(), "Overview\n1 finding")
        self.assertEqual(widget.slice_table.rowCount(), 1)
        self.assertEqual(widget.slice_table.item(0, 2).text(), "Label ranges")

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
