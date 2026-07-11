#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scientific dashboard for inspecting Training Set Audit results."""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidgetItem,
    QSizePolicy,
    QTabWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    ComboBox,
    FluentIcon,
    ListWidget,
    PrimaryPushButton,
    PushButton,
    TableWidget,
)

from NepTrainKit.core import MessageManager
from NepTrainKit.core.audit.report import write_audit_report_html
from NepTrainKit.core.audit.result import (
    AuditBiasType,
    AuditDimension,
    AuditResult,
    AuditSeverity,
    AuditSlice,
    AuditStatus,
)
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.widgets.audit_chart import AuditChartWidget


_OVERVIEW = "__audit_overview__"
_SIGNAL_COLUMN_MIN_WIDTH = 1100
_DIMENSION_COLUMN_MIN_WIDTH = 840
_DIMENSION_COLUMN_WIDTH = 132
_SIGNAL_COLUMN_WIDTH = 142
_SEVERITY_COLORS = {
    AuditSeverity.HIGH: (QColor("#c94932"), QColor("#fbeeea")),
    AuditSeverity.MEDIUM: (QColor("#d08a17"), QColor("#fff5df")),
    AuditSeverity.LOW: (QColor("#89979b"), QColor("#eef2f3")),
    AuditSeverity.INFO: (QColor("#89979b"), QColor("#eef2f3")),
}
_ACTIVE_SEVERITY_FILTER_STYLE = """
    color: #ffffff;
    background-color: #087f78;
    border: 1px solid #087f78;
    border-radius: 4px;
    padding: 2px 9px;
"""
_INACTIVE_SEVERITY_FILTER_STYLE = """
    color: #526267;
    background-color: #ffffff;
    border: 1px solid #c9d4d6;
    border-radius: 4px;
    padding: 2px 9px;
"""


class TrainingSetAuditWidget(QWidget):
    """Render audit plots, findings, and evidence for the active dataset."""

    selectStructuresSignal = Signal(list)
    rerunAuditSignal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TrainingSetAuditWidget")
        self._result: AuditResult | None = None
        self._all_slices: list[AuditSlice] = []
        self._visible_slices: list[AuditSlice] = []
        self._dimensions: dict[str, AuditDimension] = {}
        self._active_plots: list[dict[str, Any]] = []
        self._local_chemistry_plots: list[dict[str, Any]] = []
        self._severity_filter: AuditSeverity | None = None
        self._build_ui()
        self._set_empty_state()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 16)
        root.setSpacing(10)

        self.no_dataset_state = QLabel(self.tr("No dataset loaded"), self)
        self.no_dataset_state.setObjectName("auditNoDatasetState")
        self.no_dataset_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_dataset_state.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self.no_dataset_state, stretch=1)

        self.audit_header = QFrame(self)
        self.audit_header.setObjectName("auditHeader")
        header_layout = QHBoxLayout(self.audit_header)
        header_layout.setContentsMargins(14, 10, 12, 10)
        header_layout.setSpacing(12)

        header_text = QVBoxLayout()
        header_text.setContentsMargins(0, 0, 0, 0)
        header_text.setSpacing(2)
        self.header_label = QLabel(self.tr("Training Set Audit"), self.audit_header)
        self.header_label.setObjectName("auditTitle")
        self.dataset_label = QLabel(self.tr("No dataset loaded"), self.audit_header)
        self.dataset_label.setObjectName("auditDataset")
        self.dataset_label.setWordWrap(True)
        self.dataset_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_text.addWidget(self.header_label)
        header_text.addWidget(self.dataset_label)
        self.generated_at_label = QLabel("", self.audit_header)
        self.generated_at_label.setObjectName("auditGeneratedAt")
        header_text.addWidget(self.generated_at_label)
        header_layout.addLayout(header_text, stretch=1)

        self.rerun_button = PrimaryPushButton(
            FluentIcon.SYNC,
            self.tr("Re-run audit"),
            self.audit_header,
        )
        self.rerun_button.clicked.connect(self.rerunAuditSignal.emit)
        self.export_report_button = PushButton(
            FluentIcon.SAVE,
            self.tr("Export HTML report"),
            self.audit_header,
        )
        self.export_report_button.clicked.connect(self._choose_and_export_report)
        header_layout.addWidget(self.rerun_button)
        header_layout.addWidget(self.export_report_button)
        root.addWidget(self.audit_header)

        self.dashboard_body = QWidget(self)
        self.dashboard_body.setObjectName("auditDashboardBody")
        body = QHBoxLayout(self.dashboard_body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)

        self.dimension_rail = QFrame(self)
        self.dimension_rail.setObjectName("auditDimensionRail")
        self.dimension_rail.setFixedWidth(192)
        rail_layout = QVBoxLayout(self.dimension_rail)
        rail_layout.setContentsMargins(10, 12, 10, 10)
        rail_layout.setSpacing(8)
        rail_title = QLabel(self.tr("Audit dimensions"), self.dimension_rail)
        rail_title.setObjectName("panelTitle")
        self.dimension_list = ListWidget(self.dimension_rail)
        self.dimension_list.setObjectName("auditDimensionList")
        self.dimension_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.dimension_list.currentRowChanged.connect(self._apply_dimension_filter)
        rail_layout.addWidget(rail_title)
        rail_layout.addWidget(self.dimension_list, stretch=1)
        self.label_availability_title = QLabel(
            self.tr("Label availability"), self.dimension_rail
        )
        self.label_availability_title.setObjectName("railMetaTitle")
        self.label_availability_value = QLabel("", self.dimension_rail)
        self.label_availability_value.setObjectName("railMetaValue")
        rail_layout.addWidget(self.label_availability_title)
        rail_layout.addWidget(self.label_availability_value)
        body.addWidget(self.dimension_rail)

        workspace = QVBoxLayout()
        workspace.setContentsMargins(0, 0, 0, 0)
        workspace.setSpacing(10)

        self.metric_band = QFrame(self)
        self.metric_band.setObjectName("auditMetricBand")
        metric_layout = QHBoxLayout(self.metric_band)
        metric_layout.setContentsMargins(14, 8, 14, 8)
        metric_layout.setSpacing(0)
        self.metric_structure_value, self.metric_structure_label = self._add_metric(
            metric_layout, self.tr("Structures"), "0"
        )
        self.metric_findings_value, self.metric_findings_label = self._add_metric(
            metric_layout, self.tr("Flagged slices (H/M/L)"), "0 · 0/0/0"
        )
        self.metric_dimension_value, self.metric_dimension_label = self._add_metric(
            metric_layout, self.tr("Available label metrics"), "0"
        )
        self.metric_context_value, self.metric_context_label = self._add_metric(
            metric_layout, self.tr("Label completeness"), "0 / 0", last=True
        )
        workspace.addWidget(self.metric_band)

        self.analysis_panel = QFrame(self)
        self.analysis_panel.setObjectName("auditAnalysisPanel")
        analysis_layout = QVBoxLayout(self.analysis_panel)
        analysis_layout.setContentsMargins(12, 10, 12, 10)
        analysis_layout.setSpacing(7)
        analysis_title = QLabel(self.tr("Analysis"), self.analysis_panel)
        analysis_title.setObjectName("panelTitle")
        analysis_layout.addWidget(analysis_title)

        self.analysis_tabs = QTabWidget(self.analysis_panel)
        self.analysis_tabs.setObjectName("auditAnalysisTabs")
        self.analysis_tabs.setDocumentMode(True)
        analysis_layout.addWidget(self.analysis_tabs)

        chart_tab = QWidget(self.analysis_tabs)
        chart_layout = QVBoxLayout(chart_tab)
        chart_layout.setContentsMargins(4, 8, 4, 4)
        chart_layout.setSpacing(6)
        chart_controls = QHBoxLayout()
        chart_controls.setContentsMargins(0, 0, 0, 0)
        self.local_scope_selector = ComboBox(chart_tab)
        self.local_scope_selector.setObjectName("auditLocalScopeSelector")
        self.local_scope_selector.setMinimumWidth(116)
        self.local_scope_selector.setMaximumWidth(148)
        self.local_scope_selector.currentIndexChanged.connect(
            self._on_local_scope_changed
        )
        self.local_center_label = QLabel(self.tr("Center element"), chart_tab)
        self.local_center_label.setObjectName("auditLocalCenterLabel")
        self.local_center_selector = ComboBox(chart_tab)
        self.local_center_selector.setObjectName("auditLocalCenterSelector")
        self.local_center_selector.setMinimumWidth(72)
        self.local_center_selector.setMaximumWidth(104)
        self.local_center_selector.currentIndexChanged.connect(
            self._on_local_center_changed
        )
        self.plot_selector_label = QLabel(self.tr("Distribution"), chart_tab)
        self.plot_selector = ComboBox(chart_tab)
        self.plot_selector.setObjectName("auditPlotSelector")
        self.plot_selector.setMinimumWidth(190)
        self.plot_selector.setMaximumWidth(320)
        self.plot_selector.currentIndexChanged.connect(self._show_selected_plot)
        self.analysis_status_label = QLabel("", chart_tab)
        self.analysis_status_label.setObjectName("auditAnalysisStatus")
        self.analysis_status_label.setWordWrap(True)
        chart_controls.addWidget(self.local_scope_selector)
        chart_controls.addWidget(self.local_center_label)
        chart_controls.addWidget(self.local_center_selector)
        chart_controls.addWidget(self.plot_selector_label)
        chart_controls.addWidget(self.plot_selector)
        chart_controls.addWidget(self.analysis_status_label, stretch=1)
        chart_layout.addLayout(chart_controls)
        self.chart_widget = AuditChartWidget(chart_tab)
        self.chart_widget.setObjectName("auditChart")
        self.chart_widget.selectedGroupSignal.connect(self._on_chart_group_selected)
        chart_layout.addWidget(self.chart_widget, stretch=1)
        self.chart_selection_label = QLabel("", chart_tab)
        self.chart_selection_label.setObjectName("auditChartSelection")
        chart_layout.addWidget(self.chart_selection_label)
        self.analysis_tabs.addTab(chart_tab, self.tr("Chart"))

        evidence_tab = QWidget(self.analysis_tabs)
        evidence_layout = QGridLayout(evidence_tab)
        evidence_layout.setContentsMargins(4, 8, 4, 4)
        evidence_layout.setHorizontalSpacing(10)
        evidence_layout.setVerticalSpacing(5)
        self.observed_label = self._add_evidence_block(
            evidence_layout, 0, self.tr("Observed")
        )
        self.interpretation_label = self._add_evidence_block(
            evidence_layout, 1, self.tr("Interpretation")
        )
        self.limit_label = self._add_evidence_block(
            evidence_layout, 2, self.tr("Limit")
        )
        self.send_button = PrimaryPushButton(
            self.tr("Show 0 structures in Dataset Display"),
            evidence_tab,
        )
        self.send_button.clicked.connect(self._emit_selected_structures)
        evidence_layout.addWidget(
            self.send_button,
            2,
            0,
            1,
            3,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        self.analysis_tabs.addTab(evidence_tab, self.tr("Evidence"))
        workspace.addWidget(self.analysis_panel, stretch=1)

        self.findings_panel = QFrame(self)
        self.findings_panel.setObjectName("auditFindingsPanel")
        findings_layout = QVBoxLayout(self.findings_panel)
        findings_layout.setContentsMargins(12, 10, 12, 10)
        findings_layout.setSpacing(7)
        findings_header = QHBoxLayout()
        findings_title = QLabel(self.tr("Findings"), self.findings_panel)
        findings_title.setObjectName("panelTitle")
        findings_header.addWidget(findings_title)
        findings_header.addStretch(1)
        filter_label = QLabel(self.tr("Severity"), self.findings_panel)
        filter_label.setObjectName("filterLabel")
        findings_header.addWidget(filter_label)
        self._severity_button_group = QButtonGroup(self)
        self._severity_button_group.setExclusive(True)
        self._severity_filter_buttons: dict[AuditSeverity | None, PushButton] = {}
        self.severity_buttons: dict[AuditSeverity, PushButton] = {}
        for severity, text in (
            (None, self.tr("All")),
            (AuditSeverity.HIGH, self.tr("High")),
            (AuditSeverity.MEDIUM, self.tr("Medium")),
            (AuditSeverity.LOW, self.tr("Low")),
        ):
            button = PushButton(text, self.findings_panel)
            button.setObjectName("severityFilterButton")
            button.setCheckable(True)
            button.setFixedHeight(28)
            button.clicked.connect(
                lambda checked=False, value=severity: self._set_severity_filter(value)
            )
            self._severity_button_group.addButton(button)
            self._severity_filter_buttons[severity] = button
            findings_header.addWidget(button)
            if severity is None:
                self.severity_all_button = button
            else:
                self.severity_buttons[severity] = button
        self._sync_severity_filter_buttons()
        findings_layout.addLayout(findings_header)

        self.findings_empty_label = QLabel("", self.findings_panel)
        self.findings_empty_label.setObjectName("auditFindingsEmpty")
        findings_layout.addWidget(self.findings_empty_label)

        self.slice_table = TableWidget(self.findings_panel)
        self.slice_table.setObjectName("auditFindingsTable")
        self.slice_table.setColumnCount(5)
        self.slice_table.setHorizontalHeaderLabels(
            [
                self.tr("Severity"),
                self.tr("Finding"),
                self.tr("Dimension"),
                self.tr("Structures"),
                self.tr("Signal"),
            ]
        )
        self.slice_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.slice_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.slice_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.slice_table.setShowGrid(False)
        self.slice_table.setAlternatingRowColors(True)
        self.slice_table.verticalHeader().setVisible(False)
        self.slice_table.verticalHeader().setDefaultSectionSize(30)
        header = self.slice_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        self.slice_table.setColumnWidth(2, _DIMENSION_COLUMN_WIDTH)
        self.slice_table.setColumnWidth(4, _SIGNAL_COLUMN_WIDTH)
        self.slice_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.slice_table.setCheckedColor("#087f78", "#087f78")
        self.slice_table.itemSelectionChanged.connect(self._on_slice_selection_changed)
        findings_layout.addWidget(self.slice_table, stretch=1)
        workspace.addWidget(self.findings_panel, stretch=1)

        body.addLayout(workspace, stretch=1)
        root.addWidget(self.dashboard_body, stretch=1)
        self._apply_stylesheet()
        self._update_responsive_columns(self.width())

    def _add_metric(
        self,
        layout: QHBoxLayout,
        label_text: str,
        value_text: str,
        *,
        last: bool = False,
    ) -> tuple[QLabel, QLabel]:
        cell = QWidget(self.metric_band)
        cell.setObjectName("auditMetricCell")
        cell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        cell_layout = QVBoxLayout(cell)
        cell_layout.setContentsMargins(10, 0, 10, 0)
        cell_layout.setSpacing(1)
        value = QLabel(value_text, cell)
        value.setObjectName("metricValue")
        label = QLabel(label_text, cell)
        label.setObjectName("metricLabel")
        cell_layout.addWidget(value)
        cell_layout.addWidget(label)
        layout.addWidget(cell, stretch=1)
        if not last:
            divider = QFrame(self.metric_band)
            divider.setObjectName("metricDivider")
            divider.setFrameShape(QFrame.Shape.VLine)
            layout.addWidget(divider)
        return value, label

    def _add_evidence_block(
        self,
        layout: QGridLayout,
        column: int,
        title: str,
    ) -> QTextEdit:
        heading = QLabel(title, self.analysis_tabs)
        heading.setObjectName("evidenceHeading")
        text = QTextEdit(self.analysis_tabs)
        text.setObjectName("evidenceText")
        text.setReadOnly(True)
        text.setAcceptRichText(False)
        text.setMinimumHeight(74)
        text.setMaximumHeight(96)
        layout.addWidget(heading, 0, column)
        layout.addWidget(text, 1, column)
        return text

    def _set_empty_state(self) -> None:
        self._result = None
        self._all_slices = []
        self._visible_slices = []
        self._dimensions = {}
        self._active_plots = []
        self._local_chemistry_plots = []
        self.dimension_list.clear()
        self._set_local_chemistry_controls_visible(False)
        self.local_scope_selector.clear()
        self.local_center_selector.clear()
        self.plot_selector.clear()
        self.chart_widget.clear()
        self.label_availability_value.clear()
        self._populate_slice_table()
        self.no_dataset_state.show()
        self.audit_header.hide()
        self.dashboard_body.hide()

    def set_loading(self, dataset_id: str) -> None:
        """Show a quiet progress state while the audit runs off the UI thread."""
        self._set_empty_state()
        self.no_dataset_state.setText(
            self.tr("Analyzing {dataset}...").format(dataset=dataset_id)
        )

    def set_result(self, result: AuditResult) -> None:
        self._result = result
        self._all_slices = list(result.slices)
        self._dimensions = {dimension.id: dimension for dimension in result.dimensions}
        self._severity_filter = None
        self._sync_severity_filter_buttons()
        self.no_dataset_state.hide()
        self.audit_header.show()
        self.dashboard_body.show()
        structure_count = result.overview_metrics.get(
            "structures", result.inputs.get("structure_count", 0)
        )
        self.dataset_label.setText(
            self.tr("{dataset} · {count} structures").format(
                dataset=result.dataset_id,
                count=structure_count,
            )
        )
        self.generated_at_label.setText(self._generated_at_text(result.generated_at))
        self._update_label_availability()

        self.dimension_list.blockSignals(True)
        self.dimension_list.clear()
        overview_count = len(self._all_slices)
        overview_title = self.tr("Overview")
        overview = QListWidgetItem(
            f"{overview_title}\n{self._finding_count_text(overview_count)}"
        )
        overview.setData(Qt.ItemDataRole.UserRole, _OVERVIEW)
        self.dimension_list.addItem(overview)
        if result.dimensions:
            for dimension in result.dimensions:
                status = self._status_text(dimension.status)
                finding_count = sum(
                    audit_slice.dimension_id == dimension.id
                    for audit_slice in self._all_slices
                )
                item = QListWidgetItem(
                    f"{self._dimension_title(dimension.id)}\n"
                    f"{status} · {self._finding_count_text(finding_count)}"
                )
                item.setData(Qt.ItemDataRole.UserRole, dimension.id)
                item.setToolTip(
                    self._localized_dimension_reason(dimension)
                    or self._dimension_title(dimension.id)
                )
                self.dimension_list.addItem(item)
        else:
            seen: set[str] = set()
            for audit_slice in self._all_slices:
                if audit_slice.dimension_id in seen:
                    continue
                seen.add(audit_slice.dimension_id)
                finding_count = sum(
                    item.dimension_id == audit_slice.dimension_id
                    for item in self._all_slices
                )
                item = QListWidgetItem(
                    f"{self._dimension_title(audit_slice.dimension_id)}\n"
                    f"{self._finding_count_text(finding_count)}"
                )
                item.setData(Qt.ItemDataRole.UserRole, audit_slice.dimension_id)
                self.dimension_list.addItem(item)
        self.dimension_list.blockSignals(False)
        self.dimension_list.setCurrentRow(0)

    def _generated_at_text(self, generated_at: str) -> str:
        compact = generated_at
        try:
            parsed = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                parsed = parsed.astimezone(timezone.utc)
                compact = parsed.strftime("%Y-%m-%d %H:%M UTC")
            else:
                compact = parsed.strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            pass
        return self.tr("Generated {timestamp}").format(timestamp=compact)

    def _finding_count_text(self, count: int) -> str:
        if count == 1:
            return self.tr("{count} finding").format(count=count)
        return self.tr("{count} findings").format(count=count)

    def _overview_label_counts(self) -> dict[str, int]:
        if self._result is None:
            return {"energy": 0, "force": 0, "virial": 0}
        metrics = self._result.overview_metrics
        label_counts = metrics.get("label_counts", {})
        if not isinstance(label_counts, Mapping):
            label_counts = {}
        label_overview = metrics.get("label_ranges", {})
        if not isinstance(label_overview, Mapping):
            label_overview = {}
        return {
            label: int(
                label_counts.get(
                    label,
                    label_overview.get(f"{label}_labeled_count", 0),
                )
                or 0
            )
            for label in ("energy", "force", "virial")
        }

    def _update_label_availability(self) -> None:
        if self._result is None:
            self.label_availability_value.clear()
            return
        counts = self._overview_label_counts()
        total = int(
            self._result.overview_metrics.get(
                "structures", self._result.inputs.get("structure_count", 0)
            )
            or 0
        )
        self.label_availability_value.setText(
            "\n".join(
                (
                    self.tr("Energy {count}/{total}").format(
                        count=counts["energy"], total=total
                    ),
                    self.tr("Force {count}/{total}").format(
                        count=counts["force"], total=total
                    ),
                    self.tr("Virial {count}/{total}").format(
                        count=counts["virial"], total=total
                    ),
                )
            )
        )

    def _selected_dimension_id(self) -> str:
        item = self.dimension_list.currentItem()
        if item is None:
            return _OVERVIEW
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, str) else _OVERVIEW

    def _apply_dimension_filter(self, row: int) -> None:
        del row
        dimension_id = self._selected_dimension_id()
        self._update_metrics(dimension_id)
        self._update_analysis(dimension_id)
        self._apply_finding_filters()

    def _set_severity_filter(self, severity: AuditSeverity | None) -> None:
        self._severity_filter = severity
        self._sync_severity_filter_buttons()
        self._apply_finding_filters()

    def _sync_severity_filter_buttons(self) -> None:
        for severity, button in self._severity_filter_buttons.items():
            is_active = severity == self._severity_filter
            button.setChecked(is_active)
            button.setProperty("severityFilterActive", is_active)
            button.setStyleSheet(
                _ACTIVE_SEVERITY_FILTER_STYLE
                if is_active
                else _INACTIVE_SEVERITY_FILTER_STYLE
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "slice_table"):
            self._update_responsive_columns(event.size().width())

    def _update_responsive_columns(self, width: int) -> None:
        show_signal = width >= _SIGNAL_COLUMN_MIN_WIDTH
        show_dimension = width >= _DIMENSION_COLUMN_MIN_WIDTH
        self.slice_table.setColumnHidden(4, not show_signal)
        self.slice_table.setColumnHidden(2, not show_dimension)
        if show_dimension:
            self.slice_table.setColumnWidth(2, _DIMENSION_COLUMN_WIDTH)
        if show_signal:
            self.slice_table.setColumnWidth(4, _SIGNAL_COLUMN_WIDTH)

    def _apply_finding_filters(self) -> None:
        dimension_id = self._selected_dimension_id()
        self._visible_slices = [
            audit_slice
            for audit_slice in self._all_slices
            if (dimension_id == _OVERVIEW or audit_slice.dimension_id == dimension_id)
            and (
                self._severity_filter is None
                or audit_slice.severity == self._severity_filter
            )
        ]
        self._populate_slice_table()

    def _update_metrics(self, dimension_id: str) -> None:
        if self._result is None:
            structures = 0
            dimension_slices: list[AuditSlice] = []
        else:
            structures = self._result.overview_metrics.get(
                "structures", self._result.inputs.get("structure_count", 0)
            )
            dimension_slices = (
                list(self._all_slices)
                if dimension_id == _OVERVIEW
                else [
                    audit_slice
                    for audit_slice in self._all_slices
                    if audit_slice.dimension_id == dimension_id
                ]
            )

        findings = len(dimension_slices)
        high = sum(item.severity == AuditSeverity.HIGH for item in dimension_slices)
        medium = sum(
            item.severity == AuditSeverity.MEDIUM for item in dimension_slices
        )
        low = sum(item.severity == AuditSeverity.LOW for item in dimension_slices)
        self.metric_structure_value.setText(str(structures))
        self.metric_findings_value.setText(
            f"{findings} · {high}/{medium}/{low}"
        )

        counts = self._overview_label_counts()
        available_label_metrics = sum(value > 0 for value in counts.values())
        label_total = int(structures or 0) * 3
        label_complete = sum(counts.values())
        if dimension_id in {_OVERVIEW, "label_ranges"}:
            self.metric_dimension_label.setText(self.tr("Available label metrics"))
            self.metric_dimension_value.setText(str(available_label_metrics))
            self.metric_context_label.setText(self.tr("Label completeness"))
            self.metric_context_value.setText(f"{label_complete} / {label_total}")
            return

        metrics = self._result.overview_metrics if self._result is not None else {}
        dimension_metrics = metrics.get(dimension_id, {})
        if not isinstance(dimension_metrics, Mapping):
            dimension_metrics = {}
        if dimension_id == "composition":
            self.metric_dimension_label.setText(self.tr("Sparse bins"))
            self.metric_dimension_value.setText(
                str(dimension_metrics.get("sparse_bin_count", findings))
            )
        else:
            self.metric_dimension_label.setText(self.tr("Flagged slices"))
            self.metric_dimension_value.setText(str(findings))

        dimension = self._dimensions.get(dimension_id)
        self.metric_context_label.setText(self.tr("Dimension status"))
        self.metric_context_value.setText(
            self._status_text(dimension.status)
            if dimension is not None
            else self.tr("Available")
        )

    def _update_analysis(self, dimension_id: str) -> None:
        self.plot_selector.blockSignals(True)
        self.plot_selector.clear()
        self._local_chemistry_plots = []
        self._set_local_chemistry_controls_visible(False)
        is_local_chemistry = dimension_id == "local_chemistry"
        if self._result is None:
            self._active_plots = []
            status_text = self.tr("No audit result is loaded.")
        elif dimension_id == _OVERVIEW:
            self._active_plots = [
                self._localized_plot(plot)
                for dimension in self._result.dimensions
                if dimension.status != AuditStatus.UNAVAILABLE
                for plot in dimension.plots
            ]
            status_text = ""
            if not self._active_plots:
                status_text = self.tr("No numeric distribution available.")
        else:
            dimension = self._dimensions.get(dimension_id)
            if dimension is not None and dimension.status == AuditStatus.UNAVAILABLE:
                self._active_plots = []
            else:
                self._active_plots = (
                    [self._localized_plot(plot) for plot in dimension.plots]
                    if dimension is not None
                    else []
                )
            if is_local_chemistry and self._active_plots:
                self._local_chemistry_plots = self._active_plots
                self._configure_local_chemistry_selectors()
                self._active_plots = self._selected_local_chemistry_plots()
                status_text = self._local_chemistry_status_text()
            elif dimension is not None and dimension.reason:
                status_text = self._localized_dimension_reason(dimension)
            elif not self._active_plots:
                status_text = self.tr("No numeric distribution available.")
            else:
                status_text = ""

        for plot in self._active_plots:
            self.plot_selector.addItem(self._plot_selector_text(plot))
        self.plot_selector.setCurrentIndex(0 if self._active_plots else -1)
        self.plot_selector.blockSignals(False)
        has_plot = bool(self._active_plots)
        show_selector = has_plot and len(self._active_plots) > 1
        self.plot_selector_label.setVisible(show_selector)
        self.plot_selector.setVisible(show_selector)
        self.analysis_status_label.setText(status_text)
        self.analysis_status_label.setVisible(bool(status_text))
        self.chart_widget.setVisible(has_plot)
        self.chart_selection_label.setVisible(has_plot)
        self.chart_selection_label.clear()
        self._show_selected_plot()

    def _set_local_chemistry_controls_visible(self, visible: bool) -> None:
        self.local_scope_selector.setVisible(visible)
        self.local_center_label.setVisible(visible)
        self.local_center_selector.setVisible(visible)

    def _configure_local_chemistry_selectors(self) -> None:
        available_scopes = [
            scope
            for scope in ("angular", "radial")
            if any(plot.get("scope") == scope for plot in self._local_chemistry_plots)
        ]
        self.local_scope_selector.blockSignals(True)
        self.local_scope_selector.clear()
        for scope in available_scopes:
            label = (
                self.tr("Angular core")
                if scope == "angular"
                else self.tr("Radial context")
            )
            self.local_scope_selector.addItem(label, userData=scope)
        default_index = self.local_scope_selector.findData("angular")
        self.local_scope_selector.setCurrentIndex(
            default_index if default_index >= 0 else (0 if available_scopes else -1)
        )
        self.local_scope_selector.blockSignals(False)
        self._populate_local_center_selector()
        self._set_local_chemistry_controls_visible(bool(available_scopes))

    def _populate_local_center_selector(self, preferred: str | None = None) -> None:
        scope = self.local_scope_selector.currentData()
        centers: list[str] = []
        for plot in self._local_chemistry_plots:
            center = plot.get("center_element")
            if (
                plot.get("scope") == scope
                and isinstance(center, str)
                and center not in centers
            ):
                centers.append(center)

        self.local_center_selector.blockSignals(True)
        self.local_center_selector.clear()
        for center in centers:
            self.local_center_selector.addItem(center, userData=center)
        preferred_index = self.local_center_selector.findData(preferred)
        self.local_center_selector.setCurrentIndex(
            preferred_index if preferred_index >= 0 else (0 if centers else -1)
        )
        self.local_center_selector.blockSignals(False)

    def _selected_local_chemistry_plots(self) -> list[dict[str, Any]]:
        scope = self.local_scope_selector.currentData()
        center = self.local_center_selector.currentData()
        return [
            plot
            for plot in self._local_chemistry_plots
            if plot.get("scope") == scope and plot.get("center_element") == center
        ]

    def _on_local_scope_changed(self, index: int) -> None:
        del index
        preferred_center = self.local_center_selector.currentData()
        self._populate_local_center_selector(
            preferred_center if isinstance(preferred_center, str) else None
        )
        self._refresh_local_chemistry_analysis()

    def _on_local_center_changed(self, index: int) -> None:
        del index
        self._refresh_local_chemistry_analysis()

    def _refresh_local_chemistry_analysis(self) -> None:
        if self._selected_dimension_id() != "local_chemistry":
            return
        self._active_plots = self._selected_local_chemistry_plots()
        self.plot_selector.blockSignals(True)
        self.plot_selector.clear()
        for plot in self._active_plots:
            self.plot_selector.addItem(self._plot_selector_text(plot))
        self.plot_selector.setCurrentIndex(0 if self._active_plots else -1)
        self.plot_selector.blockSignals(False)
        has_plot = bool(self._active_plots)
        show_selector = has_plot and len(self._active_plots) > 1
        self.plot_selector_label.setVisible(show_selector)
        self.plot_selector.setVisible(show_selector)
        status_text = self._local_chemistry_status_text()
        self.analysis_status_label.setText(status_text)
        self.analysis_status_label.setVisible(bool(status_text))
        self.chart_widget.setVisible(has_plot)
        self.chart_selection_label.setVisible(has_plot)
        self.chart_selection_label.clear()
        self._show_selected_plot()

    def _local_chemistry_status_text(self) -> str:
        scope = self.local_scope_selector.currentText()
        center = self.local_center_selector.currentData()
        if not scope or not isinstance(center, str):
            return self.tr("No numeric distribution available.")
        return self.tr(
            "Active NEP model · {scope} · effective pair cutoff is the mean of center and "
            "neighbor cutoffs · center {element}"
        ).format(scope=scope, element=center)

    def _local_metric_kind(self, plot: Mapping[str, Any]) -> str:
        return str(plot.get("metric_kind", plot.get("metric", "")))

    def _local_neighbor_element(self, plot: Mapping[str, Any]) -> str:
        value = plot.get("neighbor_element")
        if isinstance(value, str) and value:
            return value
        metric = str(plot.get("metric", ""))
        if metric.startswith("neighbor_fraction_"):
            return metric.removeprefix("neighbor_fraction_")
        plot_id = str(plot.get("id", ""))
        marker = ":neighbor_fraction_"
        if marker in plot_id:
            return plot_id.rsplit(marker, 1)[1]
        center = plot.get("center_element")
        return center if isinstance(center, str) else ""

    def _local_metric_label(self, plot: Mapping[str, Any]) -> str:
        metric_kind = self._local_metric_kind(plot)
        if metric_kind == "neighbor_count":
            return self.tr("Neighbor count")
        if metric_kind == "neighbor_fraction" or metric_kind.startswith(
            "neighbor_fraction_"
        ):
            return self.tr("{element} neighbor fraction").format(
                element=self._local_neighbor_element(plot)
            )
        return str(plot.get("title") or plot.get("id") or self.tr("Distribution"))

    def _plot_selector_text(self, plot: Mapping[str, Any]) -> str:
        if str(plot.get("id", "")).startswith("local_chemistry:"):
            return self._local_metric_label(plot)
        return str(plot.get("title") or plot.get("id") or self.tr("Distribution"))

    def _localized_dimension_reason(self, dimension: AuditDimension) -> str:
        reason = dimension.reason
        if reason == "No structures are loaded.":
            return self.tr("No structures are loaded.")
        if reason == "No element information found.":
            return self.tr("No element information found.")
        if reason == "No energy, force, or virial labels found.":
            return self.tr("No energy, force, or virial labels found.")

        prefix = "Available on labeled subsets only: "
        if reason.startswith(prefix) and reason.endswith("."):
            labels = reason[len(prefix) : -1]
            labels = re.sub(r"\benergy\b", self.tr("energy"), labels)
            labels = re.sub(r"\bforce\b", self.tr("force"), labels)
            labels = re.sub(r"\bvirial\b", self.tr("virial"), labels)
            return self.tr(
                "Available on labeled subsets only: {labels}."
            ).format(labels=labels)
        return reason

    def _localized_plot_text(self, text: str, plot_id: str) -> str:
        if plot_id.startswith("composition:"):
            element = plot_id.split(":", 1)[1]
            if text == f"{element} concentration distribution":
                return self.tr("{element} concentration distribution").format(
                    element=element
                )

        translations = {
            "Energy per atom distribution": self.tr(
                "Energy per atom distribution"
            ),
            "Maximum force distribution": self.tr("Maximum force distribution"),
            "Virial norm distribution": self.tr("Virial norm distribution"),
            "Atomic fraction": self.tr("Atomic fraction"),
            "Structures": self.tr("Structures"),
            "Energy per atom": self.tr("Energy per atom"),
            "Maximum force": self.tr("Maximum force"),
            "Virial norm": self.tr("Virial norm"),
        }
        return translations.get(text, text)

    def _localized_plot(self, plot: Mapping[str, Any]) -> dict[str, Any]:
        localized = dict(plot)
        plot_id = str(plot.get("id", ""))
        is_local_chemistry = plot_id.startswith("local_chemistry:")
        if is_local_chemistry:
            scope = (
                self.tr("Angular core")
                if plot.get("scope") == "angular"
                else self.tr("Radial context")
            )
            center = str(plot.get("center_element", ""))
            metric = self._local_metric_label(plot)
            localized["title"] = self.tr("{scope}: {center} {metric}").format(
                scope=scope,
                center=center,
                metric=metric,
            )
            localized["x_label"] = metric
            localized["y_label"] = self.tr("Local environments")
        else:
            for key in ("title", "x_label", "y_label"):
                value = plot.get(key)
                if isinstance(value, str):
                    localized[key] = self._localized_plot_text(value, plot_id)

        series_items = plot.get("series")
        if isinstance(series_items, (tuple, list)):
            localized_series = []
            for series in series_items:
                if not isinstance(series, Mapping):
                    localized_series.append(series)
                    continue
                localized_item = dict(series)
                label = series.get("label")
                if is_local_chemistry:
                    localized_item["label"] = self._local_metric_label(plot)
                elif isinstance(label, str):
                    localized_item["label"] = self._localized_plot_text(
                        label, plot_id
                    )
                localized_series.append(localized_item)
            localized["series"] = tuple(localized_series)
        return localized

    def _show_selected_plot(self, index: int = -1) -> None:
        if index < 0:
            index = self.plot_selector.currentIndex()
        if 0 <= index < len(self._active_plots):
            self.chart_widget.set_plot(self._active_plots[index])
        else:
            self.chart_widget.clear()
        self.chart_selection_label.clear()

    def _on_chart_group_selected(self, structure_indices: list[int]) -> None:
        count = len(structure_indices)
        self.chart_selection_label.setText(
            self.tr("Chart selection: {count} structures").format(count=count)
        )

    def _populate_slice_table(self) -> None:
        self.slice_table.clearContents()
        self.slice_table.setRowCount(len(self._visible_slices))
        for row, audit_slice in enumerate(self._visible_slices):
            severity_item = QTableWidgetItem(self._severity_text(audit_slice.severity))
            severity_item.setData(Qt.ItemDataRole.UserRole, audit_slice)
            foreground, background = _SEVERITY_COLORS[audit_slice.severity]
            severity_item.setForeground(foreground)
            severity_item.setBackground(background)
            self.slice_table.setItem(row, 0, severity_item)
            title_item = QTableWidgetItem(audit_slice.title)
            title_item.setToolTip(audit_slice.title)
            self.slice_table.setItem(row, 1, title_item)
            dimension_item = QTableWidgetItem(
                self._dimension_title(audit_slice.dimension_id)
            )
            dimension_item.setToolTip(dimension_item.text())
            self.slice_table.setItem(row, 2, dimension_item)
            count_item = QTableWidgetItem(str(len(audit_slice.structure_indices)))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.slice_table.setItem(row, 3, count_item)
            signal_item = QTableWidgetItem(
                self._bias_type_text(audit_slice.bias_type)
            )
            signal_item.setToolTip(signal_item.text())
            self.slice_table.setItem(row, 4, signal_item)

        has_findings = bool(self._visible_slices)
        self.findings_empty_label.setText(
            "" if has_findings else self.tr("No findings match the current filters.")
        )
        self.findings_empty_label.setVisible(not has_findings)
        self.slice_table.setVisible(has_findings)
        if has_findings:
            self.slice_table.selectRow(0)
        else:
            self._clear_evidence()

    def _selected_slice(self) -> AuditSlice | None:
        row = self.slice_table.currentRow()
        item = self.slice_table.item(row, 0) if row >= 0 else None
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, AuditSlice) else None

    def _clear_evidence(self) -> None:
        self.observed_label.clear()
        self.interpretation_label.clear()
        self.limit_label.clear()
        self.send_button.setText(
            self.tr("Show {count} structures in Dataset Display").format(count=0)
        )
        self.send_button.setEnabled(False)

    def _on_slice_selection_changed(self) -> None:
        audit_slice = self._selected_slice()
        if audit_slice is None:
            self._clear_evidence()
            return
        self.observed_label.setPlainText(audit_slice.observed)
        self.interpretation_label.setPlainText(audit_slice.interpretation)
        self.limit_label.setPlainText(audit_slice.limit)
        count = len(audit_slice.structure_indices)
        self.send_button.setText(
            self.tr("Show {count} structures in Dataset Display").format(count=count)
        )
        self.send_button.setEnabled(True)

    def _emit_selected_structures(self) -> None:
        audit_slice = self._selected_slice()
        if audit_slice is not None:
            self.selectStructuresSignal.emit(list(audit_slice.structure_indices))

    def _dimension_title(self, dimension_id: str) -> str:
        display_names = {
            "composition": self.tr("Composition"),
            "label_ranges": self.tr("Label ranges"),
            "local_chemistry": self.tr("Local chemistry"),
            "pair_contacts": self.tr("Pair contacts"),
        }
        if dimension_id in display_names:
            return display_names[dimension_id]
        dimension = self._dimensions.get(dimension_id)
        if dimension is not None:
            return dimension.title
        return dimension_id.replace("_", " ").capitalize()

    def _bias_type_text(self, bias_type: AuditBiasType) -> str:
        return {
            AuditBiasType.IMBALANCE: self.tr("Imbalance"),
            AuditBiasType.SPARSITY: self.tr("Sparsity"),
            AuditBiasType.REDUNDANCY: self.tr("Redundancy"),
            AuditBiasType.RISK_CONCENTRATION: self.tr("Risk concentration"),
            AuditBiasType.INFORMATIONAL: self.tr("Informational"),
        }[bias_type]

    def _severity_text(self, severity: AuditSeverity) -> str:
        return {
            AuditSeverity.HIGH: self.tr("High"),
            AuditSeverity.MEDIUM: self.tr("Medium"),
            AuditSeverity.LOW: self.tr("Low"),
            AuditSeverity.INFO: self.tr("Info"),
        }[severity]

    def _status_text(self, status: AuditStatus) -> str:
        return {
            AuditStatus.AVAILABLE: self.tr("Available"),
            AuditStatus.PARTIAL: self.tr("Partial"),
            AuditStatus.UNAVAILABLE: self.tr("Unavailable"),
        }[status]

    def export_report(self, path: str | Path) -> None:
        """Export the current audit result as a static HTML report."""
        if self._result is None:
            raise ValueError("No audit result is loaded.")
        write_audit_report_html(self._result, path)

    def _choose_and_export_report(self) -> None:
        if self._result is None:
            MessageManager.send_info_message(
                self.tr("Run Training Set Audit before exporting a report.")
            )
            return
        path = call_path_dialog(
            self,
            self.tr("Export Training Set Audit report"),
            "file",
            default_filename="training_set_audit.html",
        )
        if not path:
            return
        self.export_report(path)
        MessageManager.send_info_message(
            self.tr("Training Set Audit report exported to: {path}").format(path=path)
        )

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QWidget#TrainingSetAuditWidget {
                background: #f5f7f8;
                color: #243135;
            }
            QLabel#auditNoDatasetState {
                color: #657579;
                font-size: 13px;
            }
            QFrame#auditHeader,
            QFrame#auditDimensionRail,
            QFrame#auditMetricBand,
            QFrame#auditAnalysisPanel,
            QFrame#auditFindingsPanel {
                background: #ffffff;
                border: 1px solid #d9e1e3;
                border-radius: 5px;
            }
            QLabel#auditTitle {
                color: #18272b;
                font-size: 17px;
                font-weight: 600;
            }
            QLabel#auditDataset,
            QLabel#auditGeneratedAt,
            QLabel#metricLabel,
            QLabel#filterLabel,
            QLabel#auditChartSelection {
                color: #657579;
                font-size: 11px;
            }
            QLabel#panelTitle,
            QLabel#evidenceHeading,
            QLabel#railMetaTitle {
                color: #243135;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#railMetaValue {
                color: #526267;
                font-size: 11px;
            }
            QLabel#metricValue {
                color: #087f78;
                font-size: 16px;
                font-weight: 600;
            }
            QFrame#metricDivider {
                color: #d9e1e3;
                max-width: 1px;
            }
            QListWidget#auditDimensionList {
                background: transparent;
                border: 0;
                outline: 0;
            }
            QListWidget#auditDimensionList::item {
                color: #425257;
                min-height: 38px;
                padding: 4px 8px;
                border-left: 3px solid transparent;
            }
            QListWidget#auditDimensionList::item:selected {
                color: #075f5a;
                background: #eaf5f4;
                border-left: 3px solid #087f78;
            }
            QTabWidget#auditAnalysisTabs::pane {
                border: 0;
                border-top: 1px solid #d9e1e3;
                background: #ffffff;
            }
            QTabBar::tab {
                color: #657579;
                background: transparent;
                min-width: 72px;
                padding: 6px 10px;
                border: 0;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected {
                color: #087f78;
                border-bottom: 2px solid #087f78;
            }
            QLabel#auditAnalysisStatus {
                color: #657579;
                padding-left: 6px;
            }
            QTextEdit#evidenceText {
                color: #334247;
                background: #f8fafb;
                border: 1px solid #d9e1e3;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel#auditFindingsEmpty {
                color: #657579;
                padding: 10px 4px;
            }
            """
        )
