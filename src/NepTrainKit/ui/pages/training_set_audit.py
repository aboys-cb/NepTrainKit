#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scientific dashboard for inspecting Training Set Audit results."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QDoubleSpinBox,
    QSizePolicy,
    QSpinBox,
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
from NepTrainKit.core.audit.findings import canonical_findings
from NepTrainKit.core.audit.inventory import compare_composition_target
from NepTrainKit.core.audit.result import (
    AuditBiasType,
    AuditDimension,
    AuditFindingKind,
    AuditResult,
    AuditSlice,
    AuditStatus,
    CompositionTarget,
    DatasetInventory,
    TargetSupportStatus,
)
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.widgets.audit_chart import AuditChartWidget
from NepTrainKit.ui.widgets.dialog import DistributionExplorerWidget


_OVERVIEW = "__audit_overview__"
_DIMENSION_COLUMN_MIN_WIDTH = 840
_TOPIC_CATEGORY_COLORS = {
    "blocker": (QColor("#a61b1b"), QColor("#fdecec")),
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
        self._selected_composition_indices: list[int] = []
        self._selected_target_indices: list[int] = []
        self._review_states: dict[str, str] = {}
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
            self.tr("Re-run checks"),
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
            metric_layout, self.tr("Structures"), "0"
        )
        self.metric_findings_value, self.metric_findings_label = self._add_metric(
            metric_layout, self.tr("Exact composition points"), "0"
        )
        self.metric_dimension_value, self.metric_dimension_label = self._add_metric(
            metric_layout, self.tr("Elements"), "—"
        )
        self.metric_context_value, self.metric_context_label = self._add_metric(
            metric_layout, self.tr("Label availability"), "E — · F — · V —", last=True
        )
        summary_layout.addWidget(self.metric_band)

        overview_columns = QHBoxLayout()
        overview_columns.setContentsMargins(0, 0, 0, 0)
        overview_columns.setSpacing(10)
        self.inventory_panel = QFrame(summary_tab)
        self.inventory_panel.setObjectName("auditInventoryPanel")
        inventory_layout = QVBoxLayout(self.inventory_panel)
        inventory_layout.setContentsMargins(14, 12, 14, 12)
        inventory_layout.setSpacing(7)
        inventory_title = QLabel(self.tr("What this dataset contains"), self.inventory_panel)
        inventory_title.setObjectName("panelTitle")
        self.inventory_summary_label = QLabel("", self.inventory_panel)
        self.inventory_summary_label.setObjectName("inventorySummary")
        self.inventory_summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.inventory_summary_label.setWordWrap(True)
        self.composition_highlights_label = QLabel("", self.inventory_panel)
        self.composition_highlights_label.setObjectName("inventoryDetails")
        self.composition_highlights_label.setTextFormat(Qt.TextFormat.RichText)
        self.composition_highlights_label.setWordWrap(True)
        inventory_layout.addWidget(inventory_title)
        inventory_layout.addWidget(self.inventory_summary_label)
        inventory_layout.addWidget(self.composition_highlights_label, stretch=1)
        self.open_data_map_button = PushButton(
            FluentIcon.VIEW, self.tr("Open composition and structure map"), self.inventory_panel
        )
        self.open_data_map_button.clicked.connect(lambda: self.page_tabs.setCurrentIndex(1))
        inventory_layout.addWidget(self.open_data_map_button, alignment=Qt.AlignmentFlag.AlignRight)
        overview_columns.addWidget(self.inventory_panel, stretch=3)

        self.next_actions_panel = QFrame(summary_tab)
        self.next_actions_panel.setObjectName("auditNextActionsPanel")
        actions_layout = QVBoxLayout(self.next_actions_panel)
        actions_layout.setContentsMargins(14, 12, 14, 12)
        actions_layout.setSpacing(7)
        actions_title = QLabel(self.tr("Recommended next steps"), self.next_actions_panel)
        actions_title.setObjectName("panelTitle")
        self.next_actions_label = QLabel("", self.next_actions_panel)
        self.next_actions_label.setObjectName("nextActionsText")
        self.next_actions_label.setWordWrap(True)
        actions_layout.addWidget(actions_title)
        actions_layout.addWidget(self.next_actions_label, stretch=1)
        action_buttons = QHBoxLayout()
        self.open_review_button = PrimaryPushButton(
            self.tr("Open review queue"), self.next_actions_panel
        )
        self.open_review_button.clicked.connect(lambda: self.page_tabs.setCurrentIndex(2))
        self.open_target_button = PushButton(
            self.tr("Set target"), self.next_actions_panel
        )
        self.open_target_button.clicked.connect(lambda: self.page_tabs.setCurrentIndex(3))
        action_buttons.addWidget(self.open_review_button)
        action_buttons.addWidget(self.open_target_button)
        actions_layout.addLayout(action_buttons)
        overview_columns.addWidget(self.next_actions_panel, stretch=2)
        summary_layout.addLayout(overview_columns, stretch=1)

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
        self.slice_table.setColumnCount(5)
        self.slice_table.setHorizontalHeaderLabels(
            [
                self.tr("Type"),
                self.tr("What to review"),
                self.tr("Structures"),
                self.tr("Evidence"),
                self.tr("State"),
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
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.slice_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.slice_table.setCheckedColor("#087f78", "#087f78")
        self.slice_table.itemSelectionChanged.connect(self._on_slice_selection_changed)
        findings_layout.addWidget(self.slice_table, stretch=1)

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
        summary_layout.addStretch(1)

        self.page_tabs.addTab(summary_tab, self.tr("Overview"))

        detail_tab = QWidget(self.page_tabs)
        detail_layout = QVBoxLayout(detail_tab)
        detail_layout.setContentsMargins(0, 10, 0, 0)
        detail_layout.setSpacing(0)
        self.data_map_tabs = QTabWidget(detail_tab)
        self.data_map_tabs.setDocumentMode(True)

        composition_page = QWidget(self.data_map_tabs)
        composition_layout = QVBoxLayout(composition_page)
        composition_layout.setContentsMargins(0, 8, 0, 0)
        composition_layout.setSpacing(10)
        composition_header = QFrame(composition_page)
        composition_header.setObjectName("auditCompositionHeader")
        composition_header_layout = QHBoxLayout(composition_header)
        composition_header_layout.setContentsMargins(14, 10, 14, 10)
        composition_text = QVBoxLayout()
        composition_title = QLabel(
            self.tr("Composition and structure map"), composition_header
        )
        composition_title.setObjectName("panelTitle")
        self.composition_map_hint = QLabel("", composition_header)
        self.composition_map_hint.setObjectName("panelHint")
        self.composition_map_hint.setWordWrap(True)
        composition_text.addWidget(composition_title)
        composition_text.addWidget(self.composition_map_hint)
        composition_header_layout.addLayout(composition_text, stretch=1)
        self.composition_element_selector = ComboBox(composition_header)
        self.composition_element_selector.setMinimumWidth(100)
        self.composition_element_selector.currentIndexChanged.connect(
            self._refresh_composition_map
        )
        self.composition_scale_selector = ComboBox(composition_header)
        self.composition_scale_selector.addItem(self.tr("Linear scale"), userData=False)
        self.composition_scale_selector.addItem(self.tr("Log scale"), userData=True)
        self.composition_scale_selector.currentIndexChanged.connect(
            self._refresh_composition_map
        )
        composition_header_layout.addWidget(self.composition_element_selector)
        composition_header_layout.addWidget(self.composition_scale_selector)
        composition_layout.addWidget(composition_header)

        self.composition_chart = AuditChartWidget(composition_page)
        self.composition_chart.setObjectName("auditCompositionChart")
        self.composition_chart.selectedGroupSignal.connect(
            self._on_composition_group_selected
        )
        composition_layout.addWidget(self.composition_chart, stretch=3)

        self.composition_table = TableWidget(composition_page)
        self.composition_table.setObjectName("auditCompositionTable")
        self.composition_table.setColumnCount(5)
        self.composition_table.setHorizontalHeaderLabels(
            [
                self.tr("Exact composition"),
                self.tr("Structures"),
                self.tr("Share"),
                self.tr("Atom counts"),
                self.tr("Configuration types"),
            ]
        )
        self.composition_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.composition_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.composition_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.composition_table.verticalHeader().setVisible(False)
        composition_table_header = self.composition_table.horizontalHeader()
        composition_table_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        composition_table_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        composition_table_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        composition_table_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        composition_table_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.composition_table.itemSelectionChanged.connect(
            self._on_composition_table_selection_changed
        )
        composition_layout.addWidget(self.composition_table, stretch=2)
        composition_actions = QHBoxLayout()
        self.composition_selection_label = QLabel("", composition_page)
        self.composition_selection_label.setObjectName("auditChartSelection")
        self.composition_show_button = PrimaryPushButton(
            self.tr("Show selected structures"), composition_page
        )
        self.composition_show_button.setEnabled(False)
        self.composition_show_button.clicked.connect(
            self._emit_composition_structures
        )
        composition_actions.addWidget(self.composition_selection_label, stretch=1)
        composition_actions.addWidget(self.composition_show_button)
        composition_layout.addLayout(composition_actions)
        self.data_map_tabs.addTab(composition_page, self.tr("Composition map"))

        advanced_page = QWidget(self.data_map_tabs)
        body = QHBoxLayout(advanced_page)
        body.setContentsMargins(0, 8, 0, 0)
        body.setSpacing(10)

        self.dimension_rail = QFrame(advanced_page)
        self.dimension_rail.setObjectName("auditDimensionRail")
        self.dimension_rail.setFixedWidth(192)
        rail_layout = QVBoxLayout(self.dimension_rail)
        rail_layout.setContentsMargins(10, 12, 10, 10)
        rail_layout.setSpacing(8)
        rail_title = QLabel(self.tr("Check areas"), self.dimension_rail)
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

        self.analysis_panel = QFrame(advanced_page)
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
        self.data_map_tabs.addTab(advanced_page, self.tr("Advanced evidence"))

        self.distribution_tab = QWidget(self.data_map_tabs)
        distribution_layout = QVBoxLayout(self.distribution_tab)
        distribution_layout.setContentsMargins(0, 10, 0, 0)
        distribution_layout.setSpacing(8)
        distribution_title = QLabel(
            self.tr("Explore numeric fields"), self.distribution_tab
        )
        distribution_title.setObjectName("panelTitle")
        distribution_hint = QLabel(
            self.tr(
                "Inspect reference, prediction, or error distributions and click a bin "
                "to select its structures."
            ),
            self.distribution_tab,
        )
        distribution_hint.setObjectName("panelHint")
        distribution_hint.setWordWrap(True)
        distribution_layout.addWidget(distribution_title)
        distribution_layout.addWidget(distribution_hint)
        self.distribution_explorer = DistributionExplorerWidget(
            self.distribution_tab,
            canvas_type=None,
        )
        distribution_layout.addWidget(self.distribution_explorer, stretch=1)
        self.data_map_tabs.addTab(
            self.distribution_tab, self.tr("Explore distributions")
        )
        detail_layout.addWidget(self.data_map_tabs, stretch=1)
        self.page_tabs.addTab(detail_tab, self.tr("Data map"))

        review_tab = QWidget(self.page_tabs)
        review_layout = QVBoxLayout(review_tab)
        review_layout.setContentsMargins(0, 10, 0, 0)
        review_layout.setSpacing(10)
        self.review_banner = QFrame(review_tab)
        self.review_banner.setObjectName("auditReviewBanner")
        review_banner_layout = QHBoxLayout(self.review_banner)
        review_banner_layout.setContentsMargins(14, 10, 14, 10)
        self.review_summary_label = QLabel("", self.review_banner)
        self.review_summary_label.setObjectName("reviewSummary")
        self.review_summary_label.setWordWrap(True)
        review_banner_layout.addWidget(self.review_summary_label, stretch=1)
        self.review_state_selector = ComboBox(self.review_banner)
        for label, value in (
            (self.tr("Pending"), "pending"),
            (self.tr("Keep"), "keep"),
            (self.tr("Exclude later"), "exclude"),
            (self.tr("Known duplicate"), "duplicate"),
        ):
            self.review_state_selector.addItem(label, userData=value)
        self.apply_review_state_button = PushButton(
            self.tr("Set state"), self.review_banner
        )
        self.apply_review_state_button.clicked.connect(self._apply_review_state)
        review_banner_layout.addWidget(self.review_state_selector)
        review_banner_layout.addWidget(self.apply_review_state_button)
        review_layout.addWidget(self.review_banner)
        review_layout.addWidget(self.findings_panel, stretch=3)
        review_layout.addWidget(self.evidence_panel, stretch=2)
        self.page_tabs.addTab(review_tab, self.tr("Review queue"))

        target_tab = QWidget(self.page_tabs)
        target_layout = QHBoxLayout(target_tab)
        target_layout.setContentsMargins(0, 10, 0, 0)
        target_layout.setSpacing(10)
        target_main = QVBoxLayout()
        target_main.setSpacing(10)
        self.target_definition_panel = QFrame(target_tab)
        self.target_definition_panel.setObjectName("auditTargetDefinitionPanel")
        target_definition_layout = QHBoxLayout(self.target_definition_panel)
        target_definition_layout.setContentsMargins(14, 10, 14, 10)
        target_form = QFormLayout()
        self.target_element_selector = ComboBox(self.target_definition_panel)
        self.target_minimum_spin = QDoubleSpinBox(self.target_definition_panel)
        self.target_minimum_spin.setRange(0.0, 100.0)
        self.target_minimum_spin.setSuffix(" %")
        self.target_maximum_spin = QDoubleSpinBox(self.target_definition_panel)
        self.target_maximum_spin.setRange(0.0, 100.0)
        self.target_maximum_spin.setValue(100.0)
        self.target_maximum_spin.setSuffix(" %")
        self.target_points_edit = QLineEdit(self.target_definition_panel)
        self.target_points_edit.setPlaceholderText(
            self.tr("Optional, e.g. 0, 12.5, 25, 37.5")
        )
        self.target_minimum_count_spin = QSpinBox(self.target_definition_panel)
        self.target_minimum_count_spin.setRange(1, 1_000_000_000)
        self.target_minimum_count_spin.setValue(1000)
        target_form.addRow(self.tr("Element"), self.target_element_selector)
        target_form.addRow(self.tr("Range"), self.target_minimum_spin)
        target_form.addRow(self.tr("to"), self.target_maximum_spin)
        target_form.addRow(self.tr("Key points (optional)"), self.target_points_edit)
        target_form.addRow(self.tr("Minimum structures / point"), self.target_minimum_count_spin)
        target_definition_layout.addLayout(target_form, stretch=1)
        target_explanation = QVBoxLayout()
        target_title = QLabel(self.tr("Define the model scope you intend to support"), self.target_definition_panel)
        target_title.setObjectName("panelTitle")
        self.target_limit_label = QLabel(
            self.tr(
                "This comparison checks whether the requested composition points have enough structures. "
                "It does not prove local-environment or physical coverage."
            ),
            self.target_definition_panel,
        )
        self.target_limit_label.setObjectName("panelHint")
        self.target_limit_label.setWordWrap(True)
        self.apply_target_button = PrimaryPushButton(
            self.tr("Compare with dataset"), self.target_definition_panel
        )
        self.apply_target_button.clicked.connect(self._apply_composition_target)
        target_explanation.addWidget(target_title)
        target_explanation.addWidget(self.target_limit_label, stretch=1)
        target_explanation.addWidget(self.apply_target_button, alignment=Qt.AlignmentFlag.AlignRight)
        target_definition_layout.addLayout(target_explanation, stretch=2)
        target_main.addWidget(self.target_definition_panel)

        self.target_result_summary_label = QLabel("", target_tab)
        self.target_result_summary_label.setObjectName("targetResultSummary")
        self.target_result_summary_label.setWordWrap(True)
        target_main.addWidget(self.target_result_summary_label)

        self.target_chart = AuditChartWidget(target_tab)
        self.target_chart.setObjectName("auditTargetChart")
        self.target_chart.setMinimumHeight(160)
        self.target_chart.selectedGroupSignal.connect(self._on_target_group_selected)
        target_main.addWidget(self.target_chart, stretch=3)
        self.target_table = TableWidget(target_tab)
        self.target_table.setColumnCount(5)
        self.target_table.setHorizontalHeaderLabels(
            [
                self.tr("Composition point"),
                self.tr("Status"),
                self.tr("Observed"),
                self.tr("Nearest point"),
                self.tr("Action"),
            ]
        )
        self.target_table.verticalHeader().setVisible(False)
        self.target_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.target_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.target_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.target_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.target_table.itemSelectionChanged.connect(self._on_target_selection_changed)
        target_main.addWidget(self.target_table, stretch=2)
        target_actions = QHBoxLayout()
        self.target_selection_label = QLabel("", target_tab)
        self.target_selection_label.setObjectName("auditChartSelection")
        self.target_show_button = PrimaryPushButton(
            self.tr("Show target structures"), target_tab
        )
        self.target_show_button.setEnabled(False)
        self.target_show_button.clicked.connect(self._emit_target_structures)
        target_actions.addWidget(self.target_selection_label, stretch=1)
        target_actions.addWidget(self.target_show_button)
        target_main.addLayout(target_actions)
        target_layout.addLayout(target_main, stretch=3)

        self.model_panel = QFrame(target_tab)
        self.model_panel.setObjectName("auditModelPanel")
        model_layout = QVBoxLayout(self.model_panel)
        model_layout.setContentsMargins(14, 12, 14, 12)
        model_title = QLabel(self.tr("Model comparison"), self.model_panel)
        model_title.setObjectName("panelTitle")
        self.model_empty_label = QLabel(
            self.tr(
                "No prediction or error result is attached. Open Show NEP and calculate results, "
                "then return here to compare model errors by composition and review group."
            ),
            self.model_panel,
        )
        self.model_empty_label.setObjectName("panelHint")
        self.model_empty_label.setWordWrap(True)
        model_layout.addWidget(model_title)
        model_layout.addWidget(self.model_empty_label)
        model_layout.addStretch(1)
        target_layout.addWidget(self.model_panel, stretch=1)
        self.page_tabs.addTab(target_tab, self.tr("Target & model"))
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
        self._selected_composition_indices = []
        self._selected_target_indices = []
        self.dimension_list.clear()
        self._set_local_chemistry_controls_visible(False)
        self.local_scope_selector.clear()
        self.local_center_selector.clear()
        self.plot_selector.clear()
        self.chart_widget.clear()
        self.chart_selection_label.clear()
        self.chart_send_button.setEnabled(False)
        self.composition_element_selector.clear()
        self.target_element_selector.clear()
        self.composition_chart.clear()
        self.target_chart.clear()
        self.composition_table.setRowCount(0)
        self.target_table.setRowCount(0)
        self.target_result_summary_label.clear()
        self.composition_selection_label.clear()
        self.composition_show_button.setEnabled(False)
        self.target_selection_label.clear()
        self.target_show_button.setEnabled(False)
        self.inventory_summary_label.clear()
        self.composition_highlights_label.clear()
        self.next_actions_label.clear()
        self.review_summary_label.clear()
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

    def set_distribution_context(
        self,
        *,
        data=None,
        run_analysis_callback=None,
        apply_selection_callback=None,
    ) -> None:
        """Attach the current ResultData to the embedded distribution explorer."""
        self.distribution_explorer.set_context(
            data=data,
            run_analysis_callback=run_analysis_callback,
            apply_selection_callback=apply_selection_callback,
        )

    def show_distribution_explorer(self) -> None:
        """Switch to the unified numeric-field distribution workspace."""
        self.page_tabs.setCurrentIndex(1)
        self.data_map_tabs.setCurrentWidget(self.distribution_tab)

    def _inventory(self) -> DatasetInventory | None:
        if self._result is None:
            return None
        return self._result.inventory

    def _populate_inventory_views(self) -> None:
        inventory = self._inventory()
        self.composition_element_selector.blockSignals(True)
        self.target_element_selector.blockSignals(True)
        self.composition_element_selector.clear()
        self.target_element_selector.clear()
        if inventory is not None:
            for element in inventory.elements:
                self.composition_element_selector.addItem(element, userData=element)
                self.target_element_selector.addItem(element, userData=element)
            default_index = max(0, len(inventory.elements) - 1)
            self.composition_element_selector.setCurrentIndex(default_index)
            self.target_element_selector.setCurrentIndex(default_index)
        self.composition_element_selector.blockSignals(False)
        self.target_element_selector.blockSignals(False)
        self._refresh_composition_map()
        self._apply_composition_target()

    @staticmethod
    def _composition_formula(elements: tuple[str, ...], fractions: tuple[float, ...]) -> str:
        populated = [
            f"{element} {fraction:.2%}"
            for element, fraction in zip(elements, fractions)
            if fraction > 1.0e-10
        ]
        return " · ".join(populated) or "—"

    @staticmethod
    def _element_fraction_groups(
        inventory: DatasetInventory, element: str
    ) -> tuple[tuple[float, tuple[Any, ...]], ...]:
        element_index = inventory.elements.index(element)
        grouped: dict[float, list[Any]] = {}
        for point in inventory.composition_points:
            fraction = round(point.fractions[element_index], 12)
            grouped.setdefault(fraction, []).append(point)
        return tuple(
            (fraction, tuple(grouped[fraction])) for fraction in sorted(grouped)
        )

    def _composition_plot(
        self,
        element: str,
        *,
        target_points: tuple[float, ...] = (),
    ) -> dict[str, Any] | None:
        inventory = self._inventory()
        if inventory is None or element not in inventory.elements:
            return None
        groups = self._element_fraction_groups(inventory, element)
        return {
            "kind": "composition_stems",
            "id": f"inventory:composition:{element}",
            "title": self.tr("Exact composition support for {element}").format(
                element=element
            ),
            "x_label": self.tr("{element} atomic fraction").format(element=element),
            "y_label": self.tr("Structures"),
            "x_min": -0.01,
            "x_max": 1.0,
            "log_scale": bool(self.composition_scale_selector.currentData()),
            "target_points": target_points,
            "series": (
                {
                    "counts": tuple(
                        sum(point.structure_count for point in points)
                        for _, points in groups
                    ),
                    "x_values": tuple(fraction for fraction, _ in groups),
                    "labels": tuple(
                        self.tr(
                            "{element} {fraction:.2%} · {count} exact compositions"
                        ).format(
                            element=element,
                            fraction=fraction,
                            count=len(points),
                        )
                        for fraction, points in groups
                    ),
                    "structure_indices": tuple(
                        tuple(
                            sorted(
                                index
                                for point in points
                                for index in point.structure_indices
                            )
                        )
                        for _, points in groups
                    ),
                },
            ),
        }

    def _refresh_composition_map(self, index: int = -1) -> None:
        del index
        inventory = self._inventory()
        element = self.composition_element_selector.currentData()
        if inventory is None or not isinstance(element, str):
            self.composition_map_hint.setText(
                self.tr("No exact composition inventory is available.")
            )
            self.composition_chart.clear()
            self.composition_table.setRowCount(0)
            return
        self.composition_map_hint.setText(
            self.tr(
                "{points} exact normalized composition points across {structures:,} structures. "
                "Supercells with the same atomic fractions are merged."
            ).format(
                points=len(inventory.composition_points),
                structures=inventory.structure_count,
            )
        )
        self.composition_chart.set_plot(self._composition_plot(element))
        self._populate_composition_table(element)

    def _populate_composition_table(self, element: str) -> None:
        inventory = self._inventory()
        if inventory is None or element not in inventory.elements:
            self.composition_table.setRowCount(0)
            return
        element_index = inventory.elements.index(element)
        points = sorted(
            inventory.composition_points,
            key=lambda point: (-point.structure_count, point.fractions[element_index]),
        )
        self.composition_table.clearContents()
        self.composition_table.setRowCount(len(points))
        for row, point in enumerate(points):
            formula = self._composition_formula(inventory.elements, point.fractions)
            formula_item = QTableWidgetItem(formula)
            formula_item.setData(Qt.ItemDataRole.UserRole, point.structure_indices)
            self.composition_table.setItem(row, 0, formula_item)
            count_item = QTableWidgetItem(f"{point.structure_count:,}")
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.composition_table.setItem(row, 1, count_item)
            share_item = QTableWidgetItem(f"{point.share:.2%}")
            share_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.composition_table.setItem(row, 2, share_item)
            atom_counts = ", ".join(
                f"{count} atoms × {structures:,}"
                for count, structures in point.atom_counts
            ) or "—"
            self.composition_table.setItem(row, 3, QTableWidgetItem(atom_counts))
            config_types = ", ".join(
                f"{name} × {count:,}" for name, count in point.config_types[:4]
            ) or self.tr("Not labeled")
            self.composition_table.setItem(row, 4, QTableWidgetItem(config_types))
        if points:
            self.composition_table.selectRow(0)

    def _on_composition_table_selection_changed(self) -> None:
        row = self.composition_table.currentRow()
        item = self.composition_table.item(row, 0) if row >= 0 else None
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else ()
        self._set_composition_selection(value if isinstance(value, tuple) else ())

    def _on_composition_group_selected(self, structure_indices: list[int]) -> None:
        self._set_composition_selection(tuple(int(index) for index in structure_indices))

    def _set_composition_selection(self, structure_indices: tuple[int, ...]) -> None:
        self._selected_composition_indices = list(structure_indices)
        count = len(structure_indices)
        self.composition_selection_label.setText(
            self.tr("Selected composition point: {count:,} structures").format(
                count=count
            )
        )
        self.composition_show_button.setText(
            self.tr("Show {count:,} structures").format(count=count)
        )
        self.composition_show_button.setEnabled(bool(structure_indices))

    def _emit_composition_structures(self) -> None:
        if self._selected_composition_indices:
            self.selectStructuresSignal.emit(list(self._selected_composition_indices))

    def _on_target_group_selected(self, structure_indices: list[int]) -> None:
        self._selected_target_indices = [int(index) for index in structure_indices]
        count = len(self._selected_target_indices)
        self.target_selection_label.setText(
            self.tr("Selected chart point: {count:,} structures").format(count=count)
        )
        self.target_show_button.setEnabled(bool(self._selected_target_indices))

    def _emit_target_structures(self) -> None:
        if self._selected_target_indices:
            self.selectStructuresSignal.emit(list(self._selected_target_indices))

    def _apply_review_state(self) -> None:
        topic = self._selected_topic()
        state = self.review_state_selector.currentData()
        if topic is None or not isinstance(state, str):
            return
        self._review_states[topic.id] = state
        row = self.slice_table.currentRow()
        if row >= 0:
            self.slice_table.setItem(
                row, 4, QTableWidgetItem(self.review_state_selector.currentText())
            )
        self._update_review_summary()

    def _update_review_summary(self) -> None:
        blocker_count = sum(topic.category == "blocker" for topic in self._topics)
        review_count = sum(topic.category == "review" for topic in self._topics)
        affected = len(
            {
                index
                for topic in self._topics
                if topic.category in {"blocker", "review"}
                for index in topic.structure_indices
            }
        )
        decided = sum(topic.id in self._review_states for topic in self._topics)
        data_quality = (
            self._result.overview_metrics.get("data_quality", {})
            if self._result is not None
            else {}
        )
        duplicate_groups = (
            int(data_quality.get("duplicate_group_count", 0) or 0)
            if isinstance(data_quality, Mapping)
            else 0
        )
        summary = self.tr(
                "{total} review topics · {blockers} blockers · {reviews} review groups · "
                "{affected:,} affected structures · {decided} states recorded in this session"
            ).format(
                total=len(self._topics),
                blockers=blocker_count,
                reviews=review_count,
                affected=affected,
                decided=decided,
            )
        if duplicate_groups:
            summary += self.tr(" · {groups} repeated-geometry groups").format(
                groups=duplicate_groups
            )
        self.review_summary_label.setText(summary)

    def _parse_target_points(self) -> tuple[float, ...]:
        values: list[float] = []
        for token in re.split(r"[,，;；\s]+", self.target_points_edit.text().strip()):
            if not token:
                continue
            try:
                value = float(token) / 100.0
            except ValueError:
                continue
            if 0.0 <= value <= 1.0 and value not in values:
                values.append(value)
        return tuple(sorted(values))

    def _apply_composition_target(self) -> None:
        inventory = self._inventory()
        element = self.target_element_selector.currentData()
        if inventory is None or not isinstance(element, str):
            self.target_table.setRowCount(0)
            self.target_chart.clear()
            self.target_result_summary_label.setText(
                self.tr("Load a dataset before comparing a target.")
            )
            return
        minimum = self.target_minimum_spin.value() / 100.0
        maximum = self.target_maximum_spin.value() / 100.0
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        explicit_points = self._parse_target_points()
        if explicit_points:
            points = explicit_points
            chart_target_points = explicit_points
            mode_summary = self.tr(
                "Comparing {count} explicit {element} composition points."
            ).format(count=len(points), element=element)
        else:
            points = tuple(
                fraction
                for fraction, _ in self._element_fraction_groups(inventory, element)
                if minimum - 1.0e-8 <= fraction <= maximum + 1.0e-8
            )
            chart_target_points = tuple(
                dict.fromkeys((minimum, maximum))
            )
            mode_summary = self.tr(
                "No key points were entered. Showing {count} existing {element} fraction points "
                "inside the selected range; this can reveal thin existing points, but not missing "
                "points between them."
            ).format(count=len(points), element=element)
        self.target_chart.set_plot(
            self._composition_plot(element, target_points=chart_target_points)
        )
        if not points:
            self.target_table.clearContents()
            self.target_table.setRowCount(1)
            self.target_table.setItem(
                0,
                0,
                QTableWidgetItem(f"{minimum:.2%}–{maximum:.2%}"),
            )
            self.target_table.setItem(
                0, 1, QTableWidgetItem(self.tr("No sample in range"))
            )
            self.target_table.setItem(0, 2, QTableWidgetItem("0"))
            self.target_table.setItem(0, 3, QTableWidgetItem("—"))
            self.target_table.setItem(
                0, 4, QTableWidgetItem(self.tr("Plan sampling"))
            )
            self.target_result_summary_label.setText(mode_summary)
            self._selected_target_indices = []
            self.target_selection_label.clear()
            self.target_show_button.setEnabled(False)
            return
        target = CompositionTarget(
            element=element,
            minimum=minimum,
            maximum=maximum,
            key_points=points,
            minimum_structure_count=self.target_minimum_count_spin.value(),
        )
        cells = compare_composition_target(inventory, target)
        status_counts = {
            status: sum(cell.status == status for cell in cells)
            for status in TargetSupportStatus
        }
        self.target_result_summary_label.setText(
            mode_summary
            + " "
            + self.tr(
                "Quantity rule met: {supported} · thin: {thin} · no exact sample: {missing}."
            ).format(
                supported=status_counts[TargetSupportStatus.SUPPORTED],
                thin=status_counts[TargetSupportStatus.THIN],
                missing=status_counts[TargetSupportStatus.NO_SAMPLE],
            )
        )
        self.target_table.clearContents()
        self.target_table.setRowCount(len(cells))
        status_text = {
            TargetSupportStatus.SUPPORTED: self.tr("Quantity rule met"),
            TargetSupportStatus.THIN: self.tr("Thin"),
            TargetSupportStatus.NO_SAMPLE: self.tr("No exact sample"),
            TargetSupportStatus.UNJUDGEABLE: self.tr("Cannot evaluate"),
        }
        for row, cell in enumerate(cells):
            self.target_table.setItem(row, 0, QTableWidgetItem(f"{cell.target_fraction:.2%}"))
            self.target_table.setItem(row, 1, QTableWidgetItem(status_text[cell.status]))
            self.target_table.setItem(row, 2, QTableWidgetItem(f"{cell.observed_count:,}"))
            nearest = "—" if cell.nearest_fraction is None else f"{cell.nearest_fraction:.2%}"
            self.target_table.setItem(row, 3, QTableWidgetItem(nearest))
            action = {
                TargetSupportStatus.SUPPORTED: self.tr("Review structures"),
                TargetSupportStatus.THIN: self.tr("Add structures"),
                TargetSupportStatus.NO_SAMPLE: self.tr("Plan sampling"),
                TargetSupportStatus.UNJUDGEABLE: self.tr("Check target"),
            }[cell.status]
            action_item = QTableWidgetItem(action)
            action_item.setData(Qt.ItemDataRole.UserRole, cell.structure_indices)
            self.target_table.setItem(row, 4, action_item)
        if cells:
            self.target_table.selectRow(0)

    def _on_target_selection_changed(self) -> None:
        row = self.target_table.currentRow()
        item = self.target_table.item(row, 4) if row >= 0 else None
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else ()
        indices = value if isinstance(value, tuple) else ()
        self._selected_target_indices = list(indices)
        self.target_selection_label.setText(
            self.tr("Selected target point: {count:,} structures").format(
                count=len(indices)
            )
        )
        self.target_show_button.setEnabled(bool(indices))

    def set_result(self, result: AuditResult) -> None:
        self._result = result
        self._all_slices = list(result.slices)
        self._dimensions = {dimension.id: dimension for dimension in result.dimensions}
        self._topics = self._build_topics()
        self._selected_chart_indices = []
        self._selected_composition_indices = []
        self.no_dataset_state.hide()
        self.audit_header.show()
        self.dashboard_body.show()
        self.page_tabs.setCurrentIndex(0)
        structure_count = result.overview_metrics.get(
            "structures", result.inputs.get("structure_count", 0)
        )
        if result.scope is not None:
            self.dataset_label.setText(
                self.tr("{dataset} · {scope} scope · {count}/{total} structures").format(
                    dataset=result.dataset_id,
                    scope=result.scope.kind.value,
                    count=result.scope.count,
                    total=result.scope.source_count,
                )
            )
        else:
            self.dataset_label.setText(
                self.tr("{dataset} · {count} structures").format(
                    dataset=result.dataset_id,
                    count=structure_count,
                )
            )
        run_meta = [self._generated_at_text(result.generated_at)]
        if result.ruleset_version:
            run_meta.append(
                self.tr("Rules {version}").format(version=result.ruleset_version)
            )
        if result.fingerprints.dataset:
            run_meta.append(
                self.tr("Data {fingerprint}").format(
                    fingerprint=result.fingerprints.dataset[:10]
                )
            )
        if result.fingerprints.model:
            run_meta.append(
                self.tr("Model {fingerprint}").format(
                    fingerprint=result.fingerprints.model[:10]
                )
            )
        self.generated_at_label.setText(" · ".join(run_meta))
        self._update_label_availability()
        self._update_summary()
        self._populate_slice_table()
        self._populate_inventory_views()
        self._update_review_summary()

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
        if self._result is None:
            return []
        topics: list[_AuditTopic] = []
        evidence_by_id = {audit_slice.id: audit_slice for audit_slice in self._all_slices}

        for finding in canonical_findings(self._result):
            source_slices = [
                evidence_by_id[evidence_id]
                for evidence_id in finding.evidence_ids
                if evidence_id in evidence_by_id
            ]
            first = source_slices[0] if source_slices else None
            if first is None:
                topics.append(self._topic_from_finding(finding))
                continue
            if first.dimension_id == "label_ranges":
                topics.append(self._label_range_topic(first))
            elif first.dimension_id == "composition":
                topics.append(self._composition_topic(finding.id, source_slices))
            elif first.dimension_id == "local_chemistry":
                topics.append(self._local_chemistry_topic(finding.id, source_slices))
            elif first.dimension_id == "pair_contacts":
                topics.append(self._pair_contact_topic(first))
            else:
                topics.append(self._topic_from_finding(finding, tuple(source_slices)))

        priority = {
            "blocker": 0,
            "review": 1,
            "imbalance": 2,
            "thin": 3,
            "redundancy": 4,
            "info": 5,
        }
        topics.sort(key=lambda topic: priority.get(topic.category, 9))
        return topics

    def _topic_from_finding(self, finding, source_slices: tuple[AuditSlice, ...] = ()) -> _AuditTopic:
        category = {
            AuditFindingKind.BLOCKER: "blocker",
            AuditFindingKind.REVIEW: "review",
        }.get(finding.kind, "info")
        if finding.kind == AuditFindingKind.EVIDENCE:
            category = {
                AuditBiasType.SPARSITY: "thin",
                AuditBiasType.IMBALANCE: "imbalance",
                AuditBiasType.REDUNDANCY: "redundancy",
            }.get(finding.signal_type, "info")
        title, observed, conclusion, limit = self._localized_core_finding(finding)
        return _AuditTopic(
            id=finding.id,
            category=category,
            title=title,
            dimension_id=finding.dimension_id,
            structure_indices=finding.structure_indices,
            observed=observed,
            interpretation=conclusion,
            limit=limit,
            plot_id=finding.plot_id,
            source_slices=source_slices,
        )

    def _localized_core_finding(self, finding) -> tuple[str, str, str, str]:
        if finding.dimension_id != "data_quality":
            return finding.title, finding.observed, finding.conclusion, finding.limit

        count = len(finding.structure_indices)
        data_quality = (
            self._result.overview_metrics.get("data_quality", {})
            if self._result is not None
            else {}
        )
        duplicate_group_count = (
            int(data_quality.get("duplicate_group_count", 0) or 0)
            if isinstance(data_quality, Mapping)
            else 0
        )
        translations = {
            "data_quality:empty_structure": (
                self.tr("Empty structures"),
                self.tr("{count} structures contain no atoms.").format(count=count),
                self.tr("A zero-atom frame cannot provide an atomic training example."),
                self.tr("This check does not impose a minimum cell size or composition."),
            ),
            "data_quality:nonfinite_geometry": (
                self.tr("Non-finite geometry values"),
                self.tr(
                    "{count} structures have an invalid position shape or NaN/Inf positions or cell values."
                ).format(count=count),
                self.tr("Training and neighbor calculations cannot safely consume these geometries."),
                self.tr("This check does not judge whether a finite geometry is physically meaningful."),
            ),
            "data_quality:invalid_pbc": (
                self.tr("Invalid periodic-boundary metadata"),
                self.tr("{count} structures do not provide three readable PBC directions.").format(count=count),
                self.tr("Periodic geometry operations need an unambiguous PBC definition."),
                self.tr("Missing PBC uses the existing NepTrainKit default and is not flagged."),
            ),
            "data_quality:invalid_cell": (
                self.tr("Invalid periodic cell"),
                self.tr("{count} structures have invalid lattice vectors for their periodic directions.").format(count=count),
                self.tr("Minimum-image geometry is undefined for the declared periodic directions."),
                self.tr("Non-periodic directions are not required to span a three-dimensional volume."),
            ),
            "data_quality:unknown_elements": (
                self.tr("Invalid element information"),
                self.tr("{count} structures contain unknown element symbols or a symbol-count mismatch.").format(count=count),
                self.tr("A training backend cannot map these atoms to a valid element type."),
                self.tr("This does not check whether the attached model supports every valid element."),
            ),
            "data_quality:invalid_label_shape": (
                self.tr("Invalid label shape"),
                self.tr("{count} structures have energy, force, or virial labels with an invalid shape.").format(count=count),
                self.tr("Mismatched labels can be assigned to the wrong atoms or rejected by training."),
                self.tr("Missing labels are handled separately and are not automatically invalid."),
            ),
            "data_quality:nonfinite_labels": (
                self.tr("Non-finite label values"),
                self.tr("{count} structures contain NaN/Inf energy, force, or virial labels.").format(count=count),
                self.tr("Non-finite targets make common training losses non-finite."),
                self.tr("The check does not require every supported label type to be present."),
            ),
            "data_quality:short_distance": (
                self.tr("Overlapping atoms"),
                self.tr("{count} structures contain an atom pair closer than 0.5 Å.").format(count=count),
                self.tr("This is a conservative collision signal and should be checked before training."),
                self.tr("Specialized collision datasets may intentionally contain very short distances."),
            ),
            "data_quality:label_conflicts": (
                self.tr("Duplicate geometries with conflicting labels"),
                self.tr("{count} structures share geometry but disagree in common training labels.").format(count=count),
                self.tr("The same input geometry maps to inconsistent targets and needs provenance review."),
                self.tr("The check cannot decide which repeated calculation is correct."),
            ),
            "data_quality:exact_duplicates": (
                self.tr("Repeated geometries"),
                (
                    self.tr(
                        "{count} structures belong to {groups} repeated-geometry groups."
                    ).format(count=count, groups=duplicate_group_count)
                    if duplicate_group_count
                    else self.tr(
                        "{count} structures repeat geometry already present in the current scope."
                    ).format(count=count)
                ),
                self.tr("Repeated geometries may unintentionally overweight one configuration."),
                self.tr("They may be intentional for weighting or independent-label studies; do not delete automatically."),
            ),
        }
        return translations.get(
            finding.id,
            (finding.title, finding.observed, finding.conclusion, finding.limit),
        )

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
        blocker_topics = [topic for topic in self._topics if topic.category == "blocker"]
        review_topics = [topic for topic in self._topics if topic.category == "review"]
        thin_topics = [
            topic
            for topic in self._topics
            if topic.category in {"thin", "imbalance", "redundancy"}
        ]
        attention_topics = blocker_topics + review_topics
        blocker_indices = {
            int(index)
            for topic in blocker_topics
            for index in topic.structure_indices
        }
        review_indices = {
            int(index)
            for topic in attention_topics
            for index in topic.structure_indices
        }
        total = self._structure_count()
        inventory = self._inventory()
        self.metric_structure_value.setText(f"{total:,}")
        self.metric_findings_value.setText(
            str(len(inventory.composition_points)) if inventory is not None else "—"
        )
        self.metric_dimension_value.setText(
            " · ".join(inventory.elements) if inventory is not None else "—"
        )
        counts = self._overview_label_counts()
        if total > 0:
            coverage = " · ".join(
                f"{label[0].upper()} {counts[label] / total:.0%}"
                for label in ("energy", "force", "virial")
            )
        else:
            coverage = "E 0% · F 0% · V 0%"
        self.metric_context_value.setText(coverage)

        if blocker_topics:
            lead = self.tr(
                "Resolve {groups} data blockers affecting {structures} unique structures before training."
            ).format(groups=len(blocker_topics), structures=len(blocker_indices))
        elif review_topics:
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

        if inventory is None:
            self.inventory_summary_label.setText(
                self.tr("No exact composition inventory is available.")
            )
            self.composition_highlights_label.clear()
        else:
            self.inventory_summary_label.setText(
                "<div style='color:#425257'>"
                f"<span style='font-size:16px; font-weight:600; color:#183b38'>"
                f"{inventory.structure_count:,}</span> {escape(self.tr('structures'))}"
                "&nbsp;&nbsp;·&nbsp;&nbsp;"
                f"<span style='font-size:16px; font-weight:600; color:#183b38'>"
                f"{len(inventory.composition_points)}</span> "
                f"{escape(self.tr('exact composition points'))}"
                "&nbsp;&nbsp;·&nbsp;&nbsp;"
                f"<span style='font-weight:600'>{escape(self.tr('Atom counts'))}</span> "
                f"{escape(', '.join(f'{count} × {structures:,}' for count, structures in inventory.atom_counts) or '—')}"
                "</div>"
            )
            top_points = sorted(
                inventory.composition_points,
                key=lambda point: point.structure_count,
                reverse=True,
            )[:3]
            top_share = sum(point.share for point in top_points)
            pure_points = [
                point
                for point in inventory.composition_points
                if max(point.fractions, default=0.0) >= 1.0 - 1e-8
            ]
            rows = "".join(
                "<tr>"
                f"<td style='padding:3px 12px 3px 0'>{escape(self._composition_formula(inventory.elements, point.fractions))}</td>"
                f"<td align='right' style='padding:3px 8px'><b>{point.structure_count:,}</b></td>"
                f"<td align='right' style='padding:3px 0; color:#657579'>{point.share:.2%}</td>"
                "</tr>"
                for point in top_points
            )
            pure_summary = ""
            if pure_points:
                pure_items = []
                for point in pure_points:
                    element_index = max(
                        range(len(point.fractions)),
                        key=point.fractions.__getitem__,
                    )
                    pure_items.append(
                        self.tr("Pure {element} {count:,}").format(
                            element=inventory.elements[element_index],
                            count=point.structure_count,
                        )
                    )
                pure_summary = (
                    "<div style='margin-top:8px; padding-top:6px; color:#425257'>"
                    f"<b>{escape(self.tr('Pure-element endpoints'))}</b>&nbsp;&nbsp;"
                    f"{escape('   ·   '.join(pure_items))}</div>"
                )
            concentration = escape(
                self.tr(
                    "Top {count} composition points contain {share:.1%} of structures."
                ).format(count=len(top_points), share=top_share)
            )
            self.composition_highlights_label.setText(
                "<div style='margin-top:8px'>"
                f"<div style='font-weight:600; color:#243135'>{escape(self.tr('Main composition points'))}</div>"
                f"<div style='color:#657579; margin-bottom:4px'>{concentration}</div>"
                f"<table cellspacing='0' width='100%'>{rows}</table>"
                f"{pure_summary}</div>"
            )

        data_quality = (
            self._result.overview_metrics.get("data_quality", {})
            if self._result is not None
            else {}
        )
        duplicate_groups = (
            int(data_quality.get("duplicate_group_count", 0) or 0)
            if isinstance(data_quality, Mapping)
            else 0
        )
        next_steps: list[str] = []
        if blocker_topics:
            next_steps.append(
                self.tr("1. Resolve {count} blocker topics before training.").format(
                    count=len(blocker_topics)
                )
            )
        elif review_topics:
            if duplicate_groups:
                next_steps.append(
                    self.tr(
                        "1. Review {groups} repeated-geometry groups before deciding whether to keep or exclude them."
                    ).format(groups=duplicate_groups)
                )
            else:
                next_steps.append(
                    self.tr("1. Review {count} priority topics.").format(
                        count=len(review_topics)
                    )
                )
        else:
            next_steps.append(self.tr("1. No priority data-quality review is pending."))
        next_steps.append(
            self.tr("2. Inspect composition concentration and pure-element endpoints.")
        )
        next_steps.append(
            self.tr("3. Define the intended target range to expose missing or thin points.")
        )
        self.next_actions_label.setText("\n".join(next_steps))
        self.open_review_button.setEnabled(bool(self._topics))

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
            state = self._review_states.get(topic.id, "pending")
            state_labels = {
                "pending": self.tr("Pending"),
                "keep": self.tr("Keep"),
                "exclude": self.tr("Exclude later"),
                "duplicate": self.tr("Known duplicate"),
            }
            self.slice_table.setItem(
                row, 4, QTableWidgetItem(state_labels.get(state, state))
            )

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
        state_index = self.review_state_selector.findData(
            self._review_states.get(topic.id, "pending")
        )
        if state_index >= 0:
            self.review_state_selector.setCurrentIndex(state_index)

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
        self.data_map_tabs.setCurrentIndex(1)

    def _dimension_title(self, dimension_id: str) -> str:
        display_names = {
            "data_quality": self.tr("Data quality"),
            "composition": self.tr("Composition balance"),
            "config_types": self.tr("Configuration types"),
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
            "blocker": self.tr("Data blocker"),
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
            default_filename="training_set_check.html",
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
            QFrame#auditEvidencePanel,
            QFrame#auditInventoryPanel,
            QFrame#auditNextActionsPanel,
            QFrame#auditCompositionHeader,
            QFrame#auditReviewBanner,
            QFrame#auditTargetDefinitionPanel,
            QFrame#auditModelPanel {
                background: #ffffff;
                border: 1px solid #d9e1e3;
                border-radius: 5px;
            }
            QFrame#auditSummaryPanel {
                background: #eef8f6;
                border: 1px solid #b9dcd7;
                border-left: 4px solid #087f78;
            }
            QFrame#auditReviewBanner {
                background: #eef8f6;
                border-color: #b9dcd7;
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
            QLabel#inventorySummary,
            QLabel#reviewSummary {
                color: #183b38;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#inventoryDetails,
            QLabel#nextActionsText {
                color: #425257;
                font-size: 12px;
            }
            QLabel#targetResultSummary {
                color: #315b57;
                background: #eef8f6;
                border: 1px solid #b9dcd7;
                border-radius: 4px;
                padding: 7px 10px;
                font-size: 12px;
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
