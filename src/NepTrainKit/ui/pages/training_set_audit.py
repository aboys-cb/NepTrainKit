#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scientific dashboard for inspecting Training Set Audit results."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    AuditSlice,
    AuditStatus,
)
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.widgets.audit_chart import AuditChartWidget


_OVERVIEW = "__audit_overview__"
_DIMENSION_COLUMN_MIN_WIDTH = 840
_TOPIC_CATEGORY_COLORS = {
    "review": (QColor("#a14d16"), QColor("#fff1df")),
    "thin": (QColor("#087f78"), QColor("#e7f5f3")),
    "imbalance": (QColor("#7b5a14"), QColor("#fff6dc")),
    "redundancy": (QColor("#5c4aa3"), QColor("#f0edff")),
    "info": (QColor("#526267"), QColor("#eef2f3")),
}


@dataclass(frozen=True)
class _AuditTopic:
    """One user-facing review topic, possibly consolidated from many bins."""

    id: str
    category: str
    title: str
    dimension_id: str
    structure_indices: tuple[int, ...]
    observed: str
    interpretation: str
    limit: str
    plot_id: str = ""
    source_slices: tuple[AuditSlice, ...] = ()


class TrainingSetAuditWidget(QWidget):
    """Render audit plots, findings, and evidence for the active dataset."""

    selectStructuresSignal = Signal(list)
    rerunAuditSignal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TrainingSetAuditWidget")
        self._result: AuditResult | None = None
        self._all_slices: list[AuditSlice] = []
        self._topics: list[_AuditTopic] = []
        self._dimensions: dict[str, AuditDimension] = {}
        self._active_plots: list[dict[str, Any]] = []
        self._local_chemistry_plots: list[dict[str, Any]] = []
        self._selected_chart_indices: list[int] = []
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

        self.page_tabs = QTabWidget(self)
        self.page_tabs.setObjectName("auditPageTabs")
        self.page_tabs.setDocumentMode(True)
        self.dashboard_body = self.page_tabs

        summary_tab = QWidget(self.page_tabs)
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(0, 10, 0, 0)
        summary_layout.setSpacing(10)

        self.summary_panel = QFrame(summary_tab)
        self.summary_panel.setObjectName("auditSummaryPanel")
        summary_panel_layout = QVBoxLayout(self.summary_panel)
        summary_panel_layout.setContentsMargins(16, 12, 16, 12)
        summary_panel_layout.setSpacing(4)
        summary_kicker = QLabel(self.tr("Current conclusion"), self.summary_panel)
        summary_kicker.setObjectName("summaryKicker")
        self.summary_conclusion_label = QLabel("", self.summary_panel)
        self.summary_conclusion_label.setObjectName("summaryConclusion")
        self.summary_conclusion_label.setWordWrap(True)
        self.summary_limit_label = QLabel(
            self.tr(
                "This is a relative check inside the current dataset. It does not prove "
                "that the training set is complete or that the potential is reliable."
            ),
            self.summary_panel,
        )
        self.summary_limit_label.setObjectName("summaryLimit")
        self.summary_limit_label.setWordWrap(True)
        summary_panel_layout.addWidget(summary_kicker)
        summary_panel_layout.addWidget(self.summary_conclusion_label)
        summary_panel_layout.addWidget(self.summary_limit_label)
        summary_layout.addWidget(self.summary_panel)

        self.metric_band = QFrame(summary_tab)
        self.metric_band.setObjectName("auditMetricBand")
        metric_layout = QHBoxLayout(self.metric_band)
        metric_layout.setContentsMargins(14, 8, 14, 8)
        metric_layout.setSpacing(0)
        self.metric_structure_value, self.metric_structure_label = self._add_metric(
            metric_layout, self.tr("Review groups"), "0"
        )
        self.metric_findings_value, self.metric_findings_label = self._add_metric(
            metric_layout, self.tr("Label coverage"), "0 / 3"
        )
        self.metric_dimension_value, self.metric_dimension_label = self._add_metric(
            metric_layout, self.tr("Thin-distribution signals"), "0"
        )
        self.metric_context_value, self.metric_context_label = self._add_metric(
            metric_layout, self.tr("Model errors"), self.tr("Not evaluated"), last=True
        )
        summary_layout.addWidget(self.metric_band)

        self.findings_panel = QFrame(summary_tab)
        self.findings_panel.setObjectName("auditFindingsPanel")
        findings_layout = QVBoxLayout(self.findings_panel)
        findings_layout.setContentsMargins(12, 10, 12, 10)
        findings_layout.setSpacing(6)
        findings_title = QLabel(self.tr("Review first"), self.findings_panel)
        findings_title.setObjectName("panelTitle")
        findings_hint = QLabel(
            self.tr(
                "Related low-frequency bins are grouped into one topic instead of many alarms."
            ),
            self.findings_panel,
        )
        findings_hint.setObjectName("panelHint")
        findings_hint.setWordWrap(True)
        findings_layout.addWidget(findings_title)
        findings_layout.addWidget(findings_hint)

        self.findings_empty_label = QLabel("", self.findings_panel)
        self.findings_empty_label.setObjectName("auditFindingsEmpty")
        findings_layout.addWidget(self.findings_empty_label)

        self.slice_table = TableWidget(self.findings_panel)
        self.slice_table.setObjectName("auditFindingsTable")
        self.slice_table.setColumnCount(4)
        self.slice_table.setHorizontalHeaderLabels(
            [
                self.tr("Type"),
                self.tr("What to review"),
                self.tr("Structures"),
                self.tr("Evidence"),
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
        self.slice_table.verticalHeader().setDefaultSectionSize(32)
        header = self.slice_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.slice_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.slice_table.setCheckedColor("#087f78", "#087f78")
        self.slice_table.itemSelectionChanged.connect(self._on_slice_selection_changed)
        findings_layout.addWidget(self.slice_table, stretch=1)
        summary_layout.addWidget(self.findings_panel, stretch=2)

        self.evidence_panel = QFrame(summary_tab)
        self.evidence_panel.setObjectName("auditEvidencePanel")
        evidence_layout = QGridLayout(self.evidence_panel)
        evidence_layout.setContentsMargins(12, 10, 12, 10)
        evidence_layout.setHorizontalSpacing(10)
        evidence_layout.setVerticalSpacing(5)
        self.selected_topic_label = QLabel(
            self.tr("Select a review topic to see the evidence."), self.evidence_panel
        )
        self.selected_topic_label.setObjectName("selectedTopicTitle")
        self.selected_topic_label.setWordWrap(True)
        evidence_layout.addWidget(self.selected_topic_label, 0, 0, 1, 3)
        self.observed_label = self._add_evidence_block(
            evidence_layout, 0, self.tr("What the data says"), self.evidence_panel
        )
        self.interpretation_label = self._add_evidence_block(
            evidence_layout, 1, self.tr("Why it matters"), self.evidence_panel
        )
        self.limit_label = self._add_evidence_block(
            evidence_layout, 2, self.tr("Keep in mind"), self.evidence_panel
        )
        evidence_actions = QHBoxLayout()
        evidence_actions.addStretch(1)
        self.view_distribution_button = PushButton(
            FluentIcon.VIEW,
            self.tr("View related distribution"),
            self.evidence_panel,
        )
        self.view_distribution_button.clicked.connect(
            self._show_selected_topic_distribution
        )
        self.send_button = PrimaryPushButton(
            self.tr("Show 0 structures in Dataset Display"),
            self.evidence_panel,
        )
        self.send_button.clicked.connect(self._emit_selected_structures)
        evidence_actions.addWidget(self.view_distribution_button)
        evidence_actions.addWidget(self.send_button)
        evidence_layout.addLayout(evidence_actions, 3, 0, 1, 3)
        summary_layout.addWidget(self.evidence_panel, stretch=1)

        self.page_tabs.addTab(summary_tab, self.tr("Summary"))

        detail_tab = QWidget(self.page_tabs)
        body = QHBoxLayout(detail_tab)
        body.setContentsMargins(0, 10, 0, 0)
        body.setSpacing(10)

        self.dimension_rail = QFrame(detail_tab)
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

        self.analysis_panel = QFrame(detail_tab)
        self.analysis_panel.setObjectName("auditAnalysisPanel")
        analysis_layout = QVBoxLayout(self.analysis_panel)
        analysis_layout.setContentsMargins(12, 10, 12, 10)
        analysis_layout.setSpacing(7)
        analysis_title = QLabel(self.tr("Detailed distributions"), self.analysis_panel)
        analysis_title.setObjectName("panelTitle")
        analysis_layout.addWidget(analysis_title)
        analysis_hint = QLabel(
            self.tr(
                "Use this view to inspect raw bins, NEP-cutoff scopes, and the structures "
                "behind a selected bar."
            ),
            self.analysis_panel,
        )
        analysis_hint.setObjectName("panelHint")
        analysis_hint.setWordWrap(True)
        analysis_layout.addWidget(analysis_hint)

        self.analysis_tabs = QWidget(self.analysis_panel)
        self.analysis_tabs.setObjectName("auditAnalysisWorkspace")
        analysis_layout.addWidget(self.analysis_tabs, stretch=1)

        chart_layout = QVBoxLayout(self.analysis_tabs)
        chart_layout.setContentsMargins(4, 8, 4, 4)
        chart_layout.setSpacing(6)
        chart_controls = QHBoxLayout()
        chart_controls.setContentsMargins(0, 0, 0, 0)
        self.local_scope_selector = ComboBox(self.analysis_tabs)
        self.local_scope_selector.setObjectName("auditLocalScopeSelector")
        self.local_scope_selector.setMinimumWidth(116)
        self.local_scope_selector.setMaximumWidth(148)
        self.local_scope_selector.currentIndexChanged.connect(
            self._on_local_scope_changed
        )
        self.local_center_label = QLabel(self.tr("Center element"), self.analysis_tabs)
        self.local_center_label.setObjectName("auditLocalCenterLabel")
        self.local_center_selector = ComboBox(self.analysis_tabs)
        self.local_center_selector.setObjectName("auditLocalCenterSelector")
        self.local_center_selector.setMinimumWidth(72)
        self.local_center_selector.setMaximumWidth(104)
        self.local_center_selector.currentIndexChanged.connect(
            self._on_local_center_changed
        )
        self.plot_selector_label = QLabel(self.tr("Distribution"), self.analysis_tabs)
        self.plot_selector = ComboBox(self.analysis_tabs)
        self.plot_selector.setObjectName("auditPlotSelector")
        self.plot_selector.setMinimumWidth(190)
        self.plot_selector.setMaximumWidth(320)
        self.plot_selector.currentIndexChanged.connect(self._show_selected_plot)
        self.analysis_status_label = QLabel("", self.analysis_tabs)
        self.analysis_status_label.setObjectName("auditAnalysisStatus")
        self.analysis_status_label.setWordWrap(True)
        chart_controls.addWidget(self.local_scope_selector)
        chart_controls.addWidget(self.local_center_label)
        chart_controls.addWidget(self.local_center_selector)
        chart_controls.addWidget(self.plot_selector_label)
        chart_controls.addWidget(self.plot_selector)
        chart_controls.addWidget(self.analysis_status_label, stretch=1)
        chart_layout.addLayout(chart_controls)
        self.chart_widget = AuditChartWidget(self.analysis_tabs)
        self.chart_widget.setObjectName("auditChart")
        self.chart_widget.selectedGroupSignal.connect(self._on_chart_group_selected)
        chart_layout.addWidget(self.chart_widget, stretch=1)
        chart_action_layout = QHBoxLayout()
        self.chart_selection_label = QLabel("", self.analysis_tabs)
        self.chart_selection_label.setObjectName("auditChartSelection")
        self.chart_send_button = PrimaryPushButton(
            self.tr("Show selected structures"), self.analysis_tabs
        )
        self.chart_send_button.setEnabled(False)
        self.chart_send_button.clicked.connect(self._emit_chart_structures)
        chart_action_layout.addWidget(self.chart_selection_label, stretch=1)
        chart_action_layout.addWidget(self.chart_send_button)
        chart_layout.addLayout(chart_action_layout)
        workspace.addWidget(self.analysis_panel, stretch=1)

        body.addLayout(workspace, stretch=1)
        self.page_tabs.addTab(detail_tab, self.tr("Detailed data"))
        root.addWidget(self.page_tabs, stretch=1)
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
        parent: QWidget,
    ) -> QTextEdit:
        heading = QLabel(title, parent)
        heading.setObjectName("evidenceHeading")
        text = QTextEdit(parent)
        text.setObjectName("evidenceText")
        text.setReadOnly(True)
        text.setAcceptRichText(False)
        text.setMinimumHeight(74)
        text.setMaximumHeight(96)
        layout.addWidget(heading, 1, column)
        layout.addWidget(text, 2, column)
        return text

    def _set_empty_state(self) -> None:
        self._result = None
        self._all_slices = []
        self._topics = []
        self._dimensions = {}
        self._active_plots = []
        self._local_chemistry_plots = []
        self._selected_chart_indices = []
        self.dimension_list.clear()
        self._set_local_chemistry_controls_visible(False)
        self.local_scope_selector.clear()
        self.local_center_selector.clear()
        self.plot_selector.clear()
        self.chart_widget.clear()
        self.chart_selection_label.clear()
        self.chart_send_button.setEnabled(False)
        self.label_availability_value.clear()
        self._populate_slice_table()
        self._clear_evidence()
        self.summary_conclusion_label.clear()
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
        self._topics = self._build_topics()
        self._selected_chart_indices = []
        self.no_dataset_state.hide()
        self.audit_header.show()
        self.dashboard_body.show()
        self.page_tabs.setCurrentIndex(0)
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
        self._update_summary()
        self._populate_slice_table()

        self.dimension_list.blockSignals(True)
        self.dimension_list.clear()
        overview_title = self.tr("Overview")
        overview = QListWidgetItem(
            f"{overview_title}\n{self._topic_count_text(len(self._topics))}"
        )
        overview.setData(Qt.ItemDataRole.UserRole, _OVERVIEW)
        self.dimension_list.addItem(overview)
        if result.dimensions:
            for dimension in result.dimensions:
                status = self._status_text(dimension.status)
                topic_count = sum(
                    topic.dimension_id == dimension.id for topic in self._topics
                )
                item = QListWidgetItem(
                    f"{self._dimension_title(dimension.id)}\n"
                    f"{status} · {self._topic_count_text(topic_count)}"
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
                topic_count = sum(
                    topic.dimension_id == audit_slice.dimension_id
                    for topic in self._topics
                )
                item = QListWidgetItem(
                    f"{self._dimension_title(audit_slice.dimension_id)}\n"
                    f"{self._topic_count_text(topic_count)}"
                )
                item.setData(Qt.ItemDataRole.UserRole, audit_slice.dimension_id)
                self.dimension_list.addItem(item)
        self.dimension_list.blockSignals(False)
        self.dimension_list.setCurrentRow(0)

    def _structure_count(self) -> int:
        if self._result is None:
            return 0
        return int(
            self._result.overview_metrics.get(
                "structures", self._result.inputs.get("structure_count", 0)
            )
            or 0
        )

    @staticmethod
    def _metric_value(audit_slice: AuditSlice, name: str) -> Any | None:
        for metric in audit_slice.metrics:
            if metric.name == name:
                return metric.value
        return None

    def _plot_by_id(self, plot_id: str) -> Mapping[str, Any] | None:
        if self._result is None:
            return None
        for dimension in self._result.dimensions:
            for plot in dimension.plots:
                if str(plot.get("id", "")) == plot_id:
                    return plot
        return None

    @staticmethod
    def _unique_indices(slices: list[AuditSlice] | tuple[AuditSlice, ...]) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    int(index)
                    for audit_slice in slices
                    for index in audit_slice.structure_indices
                }
            )
        )

    def _compact_bin_labels(self, labels: list[str]) -> str:
        if not labels:
            return self.tr("No populated low-frequency range")
        if len(labels) <= 4:
            return ", ".join(labels)
        return self.tr("{first} to {last} ({count} ranges)").format(
            first=labels[0], last=labels[-1], count=len(labels)
        )

    def _build_topics(self) -> list[_AuditTopic]:
        topics: list[_AuditTopic] = []
        grouped: dict[str, list[AuditSlice]] = {}

        for audit_slice in self._all_slices:
            if audit_slice.dimension_id == "local_chemistry":
                group_id = ":".join(audit_slice.id.split(":")[:3])
            elif audit_slice.dimension_id == "composition":
                parts = audit_slice.id.split(":", 2)
                group_id = ":".join(parts[:2])
            else:
                group_id = audit_slice.id
            grouped.setdefault(group_id, []).append(audit_slice)

        for group_id, source_slices in grouped.items():
            first = source_slices[0]
            if first.dimension_id == "label_ranges":
                topics.append(self._label_range_topic(first))
            elif first.dimension_id == "composition":
                topics.append(self._composition_topic(group_id, source_slices))
            elif first.dimension_id == "local_chemistry":
                topics.append(self._local_chemistry_topic(group_id, source_slices))
            elif first.dimension_id == "pair_contacts":
                topics.append(self._pair_contact_topic(first))
            else:
                topics.append(self._fallback_topic(first))

        priority = {
            "review": 0,
            "imbalance": 1,
            "thin": 2,
            "redundancy": 3,
            "info": 4,
        }
        topics.sort(key=lambda topic: priority.get(topic.category, 9))
        return topics

    def _label_range_topic(self, audit_slice: AuditSlice) -> _AuditTopic:
        count = len(audit_slice.structure_indices)
        labeled = int(self._metric_value(audit_slice, "labeled_count") or 0)
        threshold = self._metric_value(audit_slice, "threshold")
        if audit_slice.id == "label_ranges:force_high_tail":
            title = self.tr("Maximum-force review group (top 10%)")
            plot_id = "label_ranges:max_force"
            if isinstance(threshold, (int, float)):
                observed = self.tr(
                    "{count} structures have maximum force above {threshold} eV/Å "
                    "within {labeled} labeled structures."
                ).format(
                    count=count,
                    threshold=f"{float(threshold):.4g}",
                    labeled=labeled,
                )
            else:
                observed = self.tr(
                    "{count} structures are in the highest 10% of maximum force values."
                ).format(count=count)
            interpretation = self.tr(
                "These structures often carry disproportionate training pressure and are "
                "worth checking for difficult environments or bad geometries."
            )
            limit = self.tr(
                "High force can be physically intended. This is a review group, not a delete recommendation."
            )
        elif audit_slice.id == "label_ranges:energy_high_tail":
            title = self.tr("Energy-per-atom review group (top 5%)")
            plot_id = "label_ranges:energy_per_atom"
            if isinstance(threshold, (int, float)):
                observed = self.tr(
                    "{count} structures have energy per atom above {threshold} eV/atom "
                    "within {labeled} labeled structures."
                ).format(
                    count=count,
                    threshold=f"{float(threshold):.6g}",
                    labeled=labeled,
                )
            else:
                observed = self.tr(
                    "{count} structures are in the highest 5% of energy-per-atom values."
                ).format(count=count)
            interpretation = self.tr(
                "This group may contain strained, defective, hot, or otherwise unusual structures."
            )
            limit = self.tr(
                "Absolute energy per atom may not be comparable across compositions. "
                "This ranking is not an anomaly verdict."
            )
        else:
            return self._fallback_topic(audit_slice)
        return _AuditTopic(
            id=audit_slice.id,
            category="review",
            title=title,
            dimension_id=audit_slice.dimension_id,
            structure_indices=tuple(audit_slice.structure_indices),
            observed=observed,
            interpretation=interpretation,
            limit=limit,
            plot_id=plot_id,
            source_slices=(audit_slice,),
        )

    def _composition_topic(
        self, plot_id: str, source_slices: list[AuditSlice]
    ) -> _AuditTopic:
        element = plot_id.split(":", 1)[1] if ":" in plot_id else ""
        indices = self._unique_indices(source_slices)
        total = self._structure_count()
        fraction = 0.0 if total <= 0 else len(indices) / total
        plot = self._plot_by_id(plot_id) or {}
        series_items = plot.get("series", ())
        series = series_items[0] if series_items else {}
        labels = list(series.get("bin_labels", ()))
        highlighted = list(series.get("highlighted_bins", ()))
        selected_labels = [labels[index] for index in highlighted if 0 <= index < len(labels)]
        ranges = self._compact_bin_labels(selected_labels)
        return _AuditTopic(
            id=plot_id,
            category="thin",
            title=self.tr("{element} composition has {count} low-frequency ranges").format(
                element=element, count=len(source_slices)
            ),
            dimension_id="composition",
            structure_indices=indices,
            observed=self.tr(
                "These composition ranges contain {count} of {total} structures ({fraction}): {ranges}."
            ).format(
                count=len(indices), total=total, fraction=f"{fraction:.1%}", ranges=ranges
            ),
            interpretation=self.tr(
                "They are less common than other composition regions inside the current dataset."
            ),
            limit=self.tr(
                "Relative sparsity matters only when the range belongs to the intended model scope."
            ),
            plot_id=plot_id,
            source_slices=tuple(source_slices),
        )

    def _local_chemistry_topic(
        self, group_id: str, source_slices: list[AuditSlice]
    ) -> _AuditTopic:
        plot_ids = list(dict.fromkeys(audit_slice.id.rsplit(":", 1)[0] for audit_slice in source_slices))
        plots = [self._plot_by_id(plot_id) or {} for plot_id in plot_ids]
        first_plot = plots[0] if plots else {}
        scope = str(first_plot.get("scope", ""))
        center = str(first_plot.get("center_element", ""))
        scope_text = (
            self.tr("angular-neighbor environment")
            if scope == "angular"
            else self.tr("radial-neighbor environment")
        )
        indices = self._unique_indices(source_slices)
        evidence_lines: list[str] = []
        for plot in plots:
            series_items = plot.get("series", ())
            series = series_items[0] if series_items else {}
            labels = list(series.get("bin_labels", ()))
            counts = list(series.get("counts", ()))
            highlighted = list(series.get("highlighted_bins", ()))
            selected_labels = [
                labels[index] for index in highlighted if 0 <= index < len(labels)
            ]
            thin_count = sum(
                int(counts[index])
                for index in highlighted
                if 0 <= index < len(counts)
            )
            sample_count = int(plot.get("sample_count", 0) or 0)
            fraction = 0.0 if sample_count <= 0 else thin_count / sample_count
            evidence_lines.append(
                self.tr(
                    "{metric}: {ranges}; {thin} of {total} environments ({fraction})."
                ).format(
                    metric=self._local_metric_label(plot),
                    ranges=self._compact_bin_labels(selected_labels),
                    thin=thin_count,
                    total=sample_count,
                    fraction=f"{fraction:.1%}",
                )
            )
        representative_plot_id = next(
            (
                plot_id
                for plot_id in plot_ids
                if plot_id.endswith(":neighbor_count")
            ),
            plot_ids[0] if plot_ids else "",
        )
        return _AuditTopic(
            id=group_id,
            category="thin",
            title=self.tr(
                "{center} {scope} has {count} low-frequency signals"
            ).format(
                center=center,
                scope=scope_text,
                count=len(plots),
            ),
            dimension_id="local_chemistry",
            structure_indices=indices,
            observed="\n".join(evidence_lines)
            + "\n"
            + self.tr("These ranges occur in {structures} structures.").format(
                structures=len(indices)
            ),
            interpretation=self.tr(
                "These environments are less common than other comparable environments in this dataset."
            ),
            limit=self.tr(
                "Relative sparsity is not a model error and is actionable only for environments relevant to use."
            ),
            plot_id=representative_plot_id,
            source_slices=tuple(source_slices),
        )

    def _pair_contact_topic(self, audit_slice: AuditSlice) -> _AuditTopic:
        parts = audit_slice.id.split(":")
        scope = parts[1] if len(parts) > 1 else ""
        scope_text = (
            self.tr("Angular neighbors")
            if scope == "angular"
            else self.tr("Radial neighbors")
        )
        pair = "-".join(parts[2:4]) if len(parts) >= 4 else audit_slice.title
        contacts = int(self._metric_value(audit_slice, "contact_edges") or 0)
        contact_structures = int(
            self._metric_value(audit_slice, "contact_structures") or 0
        )
        co_sampled = int(
            self._metric_value(audit_slice, "co_sampled_structures") or 0
        )
        return _AuditTopic(
            id=audit_slice.id,
            category="info",
            title=self.tr("{pair} contact support ({scope})").format(
                pair=pair, scope=scope_text
            ),
            dimension_id="pair_contacts",
            structure_indices=tuple(audit_slice.structure_indices),
            observed=self.tr(
                "{contacts} directed cutoff contacts occur in {contact_structures} of "
                "{co_sampled} co-sampled structures."
            ).format(
                contacts=contacts,
                contact_structures=contact_structures,
                co_sampled=co_sampled,
            ),
            interpretation=self.tr(
                "This describes pair support in the current data; it is not a sampling recommendation."
            ),
            limit=self.tr(
                "Raw contact counts depend on structure size and composition, so compare them cautiously."
            ),
            plot_id=f"pair_contacts:{scope}",
            source_slices=(audit_slice,),
        )

    def _fallback_topic(self, audit_slice: AuditSlice) -> _AuditTopic:
        category = {
            AuditBiasType.RISK_CONCENTRATION: "review",
            AuditBiasType.SPARSITY: "thin",
            AuditBiasType.IMBALANCE: "imbalance",
            AuditBiasType.REDUNDANCY: "redundancy",
            AuditBiasType.INFORMATIONAL: "info",
        }[audit_slice.bias_type]
        return _AuditTopic(
            id=audit_slice.id,
            category=category,
            title=audit_slice.title,
            dimension_id=audit_slice.dimension_id,
            structure_indices=tuple(audit_slice.structure_indices),
            observed=audit_slice.observed,
            interpretation=audit_slice.interpretation,
            limit=audit_slice.limit,
            source_slices=(audit_slice,),
        )

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

    def _topic_count_text(self, count: int) -> str:
        if count == 1:
            return self.tr("{count} topic").format(count=count)
        return self.tr("{count} topics").format(count=count)

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
        self._update_analysis(self._selected_dimension_id())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "slice_table"):
            self._update_responsive_columns(event.size().width())

    def _update_responsive_columns(self, width: int) -> None:
        self.slice_table.setColumnHidden(3, width < _DIMENSION_COLUMN_MIN_WIDTH)
        self.slice_table.setColumnHidden(2, width < 650)

    def _update_summary(self) -> None:
        review_topics = [topic for topic in self._topics if topic.category == "review"]
        thin_topics = [
            topic
            for topic in self._topics
            if topic.category in {"thin", "imbalance", "redundancy"}
        ]
        review_indices = {
            int(index)
            for topic in review_topics
            for index in topic.structure_indices
        }
        self.metric_structure_value.setText(
            self.tr("{groups} groups · {structures} structures").format(
                groups=len(review_topics), structures=len(review_indices)
            )
        )
        counts = self._overview_label_counts()
        total = self._structure_count()
        if total > 0:
            coverage = " · ".join(
                f"{label[0].upper()} {counts[label] / total:.0%}"
                for label in ("energy", "force", "virial")
            )
        else:
            coverage = "E 0% · F 0% · V 0%"
        self.metric_findings_value.setText(coverage)
        self.metric_dimension_value.setText(str(len(thin_topics)))
        self.metric_context_value.setText(self.tr("Not evaluated"))

        if review_topics:
            lead = self.tr(
                "Start with {groups} review groups covering {structures} unique structures."
            ).format(groups=len(review_topics), structures=len(review_indices))
        else:
            lead = self.tr("No structure group requires priority review from the current checks.")
        if thin_topics:
            distribution = self.tr(
                "The data also contains {count} grouped low-frequency distribution signals."
            ).format(count=len(thin_topics))
        else:
            distribution = self.tr("No populated low-frequency distribution signal was found.")
        complete_labels = total > 0 and all(count == total for count in counts.values())
        label_text = (
            self.tr("Energy, force, and virial labels are complete.")
            if complete_labels
            else self.tr("Some energy, force, or virial labels are missing; see Detailed data.")
        )
        self.summary_conclusion_label.setText(
            self.tr("{lead} {distribution} {labels} Model errors were not evaluated.").format(
                lead=lead, distribution=distribution, labels=label_text
            )
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
                self.tr("Angular neighbors")
                if scope == "angular"
                else self.tr("Radial neighbors")
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
            "Scope from the active NEP cutoffs · {scope} · center {element}. "
            "Orange marks low-frequency ranges inside the current data."
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

        direct_translations = {
            "No active NEP model file is available.": self.tr(
                "No active NEP model file is available."
            ),
            "The active NEP model file could not be read.": self.tr(
                "The active NEP model file could not be read."
            ),
            "Local chemistry could not be audited from the active data.": self.tr(
                "Local chemistry could not be audited from the active data."
            ),
            "Pair contacts could not be audited from the active data.": self.tr(
                "Pair contacts could not be audited from the active data."
            ),
        }
        if reason in direct_translations:
            return direct_translations[reason]

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
                self.tr("Angular neighbors")
                if plot.get("scope") == "angular"
                else self.tr("Radial neighbors")
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
        self._selected_chart_indices = []
        self.chart_selection_label.clear()
        self.chart_send_button.setEnabled(False)

    def _on_chart_group_selected(self, structure_indices: list[int]) -> None:
        self._selected_chart_indices = [int(index) for index in structure_indices]
        count = len(structure_indices)
        self.chart_selection_label.setText(
            self.tr("Chart selection: {count} structures").format(count=count)
        )
        self.chart_send_button.setText(
            self.tr("Show {count} structures").format(count=count)
        )
        self.chart_send_button.setEnabled(bool(structure_indices))

    def _emit_chart_structures(self) -> None:
        if self._selected_chart_indices:
            self.selectStructuresSignal.emit(list(self._selected_chart_indices))

    def _populate_slice_table(self) -> None:
        self.slice_table.clearContents()
        self.slice_table.setRowCount(len(self._topics))
        for row, topic in enumerate(self._topics):
            category_item = QTableWidgetItem(self._topic_category_text(topic.category))
            category_item.setData(Qt.ItemDataRole.UserRole, topic)
            foreground, background = _TOPIC_CATEGORY_COLORS[topic.category]
            category_item.setForeground(foreground)
            category_item.setBackground(background)
            self.slice_table.setItem(row, 0, category_item)
            title_item = QTableWidgetItem(topic.title)
            title_item.setToolTip(topic.title)
            self.slice_table.setItem(row, 1, title_item)
            count_item = QTableWidgetItem(str(len(topic.structure_indices)))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.slice_table.setItem(row, 2, count_item)
            evidence_item = QTableWidgetItem(topic.observed)
            evidence_item.setToolTip(topic.observed)
            self.slice_table.setItem(row, 3, evidence_item)

        has_findings = bool(self._topics)
        self.findings_empty_label.setText(
            ""
            if has_findings
            else self.tr("No review topic was generated from the current checks.")
        )
        self.findings_empty_label.setVisible(not has_findings)
        self.slice_table.setVisible(has_findings)
        if has_findings:
            self.slice_table.selectRow(0)
        else:
            self._clear_evidence()

    def _selected_topic(self) -> _AuditTopic | None:
        row = self.slice_table.currentRow()
        item = self.slice_table.item(row, 0) if row >= 0 else None
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return value if isinstance(value, _AuditTopic) else None

    def _clear_evidence(self) -> None:
        self.selected_topic_label.setText(
            self.tr("Select a review topic to see the evidence.")
        )
        self.observed_label.clear()
        self.interpretation_label.clear()
        self.limit_label.clear()
        self.send_button.setText(
            self.tr("Show {count} structures in Dataset Display").format(count=0)
        )
        self.send_button.setEnabled(False)
        self.view_distribution_button.setEnabled(False)

    def _on_slice_selection_changed(self) -> None:
        topic = self._selected_topic()
        if topic is None:
            self._clear_evidence()
            return
        self.selected_topic_label.setText(topic.title)
        self.observed_label.setPlainText(topic.observed)
        self.interpretation_label.setPlainText(topic.interpretation)
        self.limit_label.setPlainText(topic.limit)
        count = len(topic.structure_indices)
        self.send_button.setText(
            self.tr("Show {count} structures in Dataset Display").format(count=count)
        )
        self.send_button.setEnabled(bool(topic.structure_indices))
        self.view_distribution_button.setEnabled(bool(topic.plot_id))

    def _emit_selected_structures(self) -> None:
        topic = self._selected_topic()
        if topic is not None and topic.structure_indices:
            self.selectStructuresSignal.emit(list(topic.structure_indices))

    def _show_selected_topic_distribution(self) -> None:
        topic = self._selected_topic()
        if topic is None or not topic.plot_id:
            return
        target_row = 0
        for row in range(self.dimension_list.count()):
            item = self.dimension_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == topic.dimension_id:
                target_row = row
                break
        self.dimension_list.setCurrentRow(target_row)

        if topic.dimension_id == "local_chemistry":
            plot = self._plot_by_id(topic.plot_id) or {}
            scope_index = self.local_scope_selector.findData(plot.get("scope"))
            if scope_index >= 0:
                self.local_scope_selector.setCurrentIndex(scope_index)
            center_index = self.local_center_selector.findData(
                plot.get("center_element")
            )
            if center_index >= 0:
                self.local_center_selector.setCurrentIndex(center_index)

        target_plot = next(
            (
                index
                for index, plot in enumerate(self._active_plots)
                if str(plot.get("id", "")) == topic.plot_id
            ),
            -1,
        )
        if target_plot >= 0:
            self.plot_selector.setCurrentIndex(target_plot)
            self._show_selected_plot(target_plot)
        self.page_tabs.setCurrentIndex(1)

    def _dimension_title(self, dimension_id: str) -> str:
        display_names = {
            "composition": self.tr("Composition balance"),
            "label_ranges": self.tr("Labels and extremes"),
            "local_chemistry": self.tr("Local-environment support"),
            "pair_contacts": self.tr("Element-pair support"),
        }
        if dimension_id in display_names:
            return display_names[dimension_id]
        dimension = self._dimensions.get(dimension_id)
        if dimension is not None:
            return dimension.title
        return dimension_id.replace("_", " ").capitalize()

    def _topic_category_text(self, category: str) -> str:
        return {
            "review": self.tr("Review"),
            "thin": self.tr("Thin distribution"),
            "imbalance": self.tr("Imbalance"),
            "redundancy": self.tr("Possible redundancy"),
            "info": self.tr("Information"),
        }.get(category, category)

    def _status_text(self, status: AuditStatus) -> str:
        return {
            AuditStatus.AVAILABLE: self.tr("Calculated"),
            AuditStatus.PARTIAL: self.tr("Partial data"),
            AuditStatus.UNAVAILABLE: self.tr("Not calculated"),
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
            QFrame#auditFindingsPanel,
            QFrame#auditSummaryPanel,
            QFrame#auditEvidencePanel {
                background: #ffffff;
                border: 1px solid #d9e1e3;
                border-radius: 5px;
            }
            QFrame#auditSummaryPanel {
                background: #eef8f6;
                border: 1px solid #b9dcd7;
                border-left: 4px solid #087f78;
            }
            QLabel#auditTitle {
                color: #18272b;
                font-size: 17px;
                font-weight: 600;
            }
            QLabel#auditDataset,
            QLabel#auditGeneratedAt,
            QLabel#metricLabel,
            QLabel#auditChartSelection,
            QLabel#panelHint,
            QLabel#summaryLimit {
                color: #657579;
                font-size: 11px;
            }
            QLabel#panelTitle,
            QLabel#evidenceHeading,
            QLabel#railMetaTitle,
            QLabel#summaryKicker {
                color: #243135;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#summaryKicker {
                color: #087f78;
                font-size: 11px;
            }
            QLabel#summaryConclusion {
                color: #183b38;
                font-size: 14px;
                font-weight: 600;
            }
            QLabel#selectedTopicTitle {
                color: #243135;
                font-size: 13px;
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
            QTabWidget#auditPageTabs::pane {
                border: 0;
                background: transparent;
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
