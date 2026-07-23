#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scientific dashboard for inspecting Training Set Audit results."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from html import escape
from itertools import combinations
from math import log1p
from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QBoxLayout,
    QCheckBox,
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
    QSplitter,
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
    ProgressBar,
    PushButton,
    TableWidget,
    TableItemDelegate,
    ToolButton,
)

from NepTrainKit.core import MessageManager
from NepTrainKit.core.audit.report import write_audit_report_html
from NepTrainKit.core.audit.findings import canonical_findings
from NepTrainKit.core.audit.inventory import compare_composition_target
from NepTrainKit.core.audit.magnetic_inventory import (
    MAGNETIC_PARTITION_LABELS,
    magnetic_partition_label,
    summarize_magnetic_inventory,
)
from NepTrainKit.core.audit.phase_inventory import (
    PHASE_PARTITION_LABELS,
    phase_partition_label,
    summarize_phase_inventory,
)
from NepTrainKit.core.audit.result import (
    AuditBiasType,
    AuditDimension,
    AuditFindingKind,
    AuditResult,
    AuditSlice,
    AuditStatus,
    CompositionTarget,
    DatasetInventory,
    PhaseInventory,
    TargetSupportStatus,
)
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.widgets.audit_chart import AuditChartWidget
from NepTrainKit.ui.widgets.dialog import DistributionExplorerWidget


_OVERVIEW = "__audit_overview__"
_DIMENSION_COLUMN_MIN_WIDTH = 840
_OVERVIEW_JET_COLORS = (
    "#000080",
    "#0000ff",
    "#007fff",
    "#00dfff",
    "#40ff80",
    "#dfff20",
    "#ffbf00",
    "#ff4000",
    "#800000",
)
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


@dataclass(frozen=True)
class _ElementSetSummary:
    """One exact set of present elements, aggregated across stoichiometries."""

    elements: tuple[str, ...]
    structure_count: int
    structure_indices: tuple[int, ...]


class _MatrixItemDelegate(TableItemDelegate):
    """Keep cell selection without the row-oriented Fluent indicator."""

    def setSelectedRows(self, indexes) -> None:
        del indexes
        self.selectedRows.clear()


class TrainingSetAuditWidget(QWidget):
    """Render audit plots, findings, and evidence for the active dataset."""

    selectStructuresSignal = Signal(list)
    rerunAuditSignal = Signal()
    requestDatasetOpenSignal = Signal()
    requestStructureEvidenceSignal = Signal()
    detachRequestedSignal = Signal()
    phaseAnalysisProgressSignal = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TrainingSetAuditWidget")
        self._result: AuditResult | None = None
        self._all_slices: list[AuditSlice] = []
        self._all_topics: list[_AuditTopic] = []
        self._topics: list[_AuditTopic] = []
        self._dimensions: dict[str, AuditDimension] = {}
        self._active_plots: list[dict[str, Any]] = []
        self._local_chemistry_plots: list[dict[str, Any]] = []
        self._selected_chart_indices: list[int] = []
        self._selected_composition_indices: list[int] = []
        self._selected_phase_indices: list[int] = []
        self._selected_magnetic_indices: list[int] = []
        self._selected_composition_key: tuple[int, ...] | None = None
        self._selected_target_indices: list[int] = []
        self._review_states: dict[tuple[str, str], str] = {}
        self._target_configured = False
        self._target_dataset_fingerprint = ""
        self._overview_element_sets: tuple[_ElementSetSummary, ...] = ()
        self._overview_elements: tuple[str, ...] = ()
        self._overview_structure_count = 0
        self._overview_element_counts: dict[str, int] = {}
        self._overview_pair_counts: dict[tuple[str, str], int] = {}
        self._overview_exact_pair_counts: dict[tuple[str, str], int] = {}
        self._overview_pure_counts: dict[str, int] = {}
        self._overview_basis_elements: tuple[str, ...] = ()
        self._selected_overview_elements: tuple[str, ...] = ()
        self._selected_overview_cell: tuple[int, int] | None = None
        self._selected_overview_mode = ""
        self._selected_overview_indices: list[int] = []
        self._requested_composition_view = ""
        self._build_ui()
        self.phaseAnalysisProgressSignal.connect(
            self.update_phase_analysis_progress
        )
        self._set_empty_state()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 16)
        root.setSpacing(10)

        self.no_dataset_panel = QFrame(self)
        self.no_dataset_panel.setObjectName("auditNoDatasetPanel")
        no_dataset_layout = QVBoxLayout(self.no_dataset_panel)
        no_dataset_layout.setContentsMargins(28, 28, 28, 28)
        no_dataset_layout.setSpacing(10)
        no_dataset_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.no_dataset_state = QLabel(
            self.tr("No dataset loaded"), self.no_dataset_panel
        )
        self.no_dataset_state.setObjectName("auditNoDatasetState")
        self.no_dataset_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_dataset_state.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        self.no_dataset_hint = QLabel(
            self.tr(
                "Open a structure or result file in NEP Dataset Display before running checks."
            ),
            self.no_dataset_panel,
        )
        self.no_dataset_hint.setObjectName("auditNoDatasetHint")
        self.no_dataset_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_dataset_hint.setWordWrap(True)
        self.no_dataset_hint.setMinimumWidth(360)
        self.no_dataset_hint.setMaximumWidth(480)
        self.no_dataset_hint.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )
        self.no_dataset_action_button = PrimaryPushButton(
            self.tr("Open dataset"), self.no_dataset_panel
        )
        self.no_dataset_action_button.setAccessibleName(self.tr("Open dataset"))
        self.no_dataset_action_button.clicked.connect(self.requestDatasetOpenSignal)
        no_dataset_layout.addWidget(self.no_dataset_state)
        no_dataset_layout.addWidget(self.no_dataset_hint)
        no_dataset_layout.addWidget(
            self.no_dataset_action_button,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        self.no_dataset_panel.setMinimumWidth(430)
        self.no_dataset_panel.setMaximumWidth(560)
        root.addWidget(
            self.no_dataset_panel,
            stretch=1,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

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
        self.page_tabs = QTabWidget(self)
        self.page_tabs.setObjectName("auditPageTabs")
        self.page_tabs.setDocumentMode(True)
        self.page_tabs.tabBar().setExpanding(False)
        self.page_tabs.tabBar().setElideMode(Qt.TextElideMode.ElideNone)
        self.page_tabs.tabBar().setUsesScrollButtons(True)
        self.detach_button = ToolButton(FluentIcon.FULL_SCREEN, self.page_tabs)
        self.detach_button.setToolTip(self.tr("Open in separate window"))
        self.detach_button.setAccessibleName(
            self.tr("Open Training Set Check in a separate window")
        )
        self.detach_button.clicked.connect(self.detachRequestedSignal)
        self.page_tabs.setCornerWidget(
            self.detach_button,
            Qt.Corner.TopRightCorner,
        )
        self.dashboard_body = self.page_tabs

        summary_tab = QWidget(self.page_tabs)
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(0, 10, 0, 0)
        summary_layout.setSpacing(10)
        self.summary_tab = summary_tab
        summary_layout.addWidget(self.audit_header)

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
            metric_layout, self.tr("Label availability"), "E — · F — · V —"
        )
        self.fact_total_atoms_value, self.fact_total_atoms_label = self._add_metric(
            metric_layout,
            self.tr("Total atoms"),
            "—",
        )
        self.fact_atom_range_value, self.fact_atom_range_label = self._add_metric(
            metric_layout,
            self.tr("Atoms per structure"),
            "—",
        )
        self.fact_atom_center_value, self.fact_atom_center_label = self._add_metric(
            metric_layout,
            self.tr("Mean / median atoms"),
            "—",
            last=True,
        )
        summary_layout.addWidget(self.metric_band)

        self.overview_columns = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.overview_columns.setContentsMargins(0, 0, 0, 0)
        self.overview_columns.setSpacing(10)

        self.cooccurrence_panel = QFrame(summary_tab)
        self.cooccurrence_panel.setObjectName("auditCooccurrencePanel")
        cooccurrence_layout = QVBoxLayout(self.cooccurrence_panel)
        cooccurrence_layout.setContentsMargins(14, 12, 14, 12)
        cooccurrence_layout.setSpacing(7)
        cooccurrence_header = QHBoxLayout()
        cooccurrence_text = QVBoxLayout()
        cooccurrence_title = QLabel(
            self.tr("Element co-occurrence map"), self.cooccurrence_panel
        )
        cooccurrence_title.setObjectName("panelTitle")
        self.cooccurrence_hint = QLabel(
            self.tr(
                "Upper triangle: global pair co-occurrence · Diagonal: element presence · "
                "Select an upper pair to reveal related third and fourth elements below"
            ),
            self.cooccurrence_panel,
        )
        self.cooccurrence_hint.setObjectName("panelHint")
        self.cooccurrence_hint.setWordWrap(True)
        cooccurrence_text.addWidget(cooccurrence_title)
        cooccurrence_text.addWidget(self.cooccurrence_hint)
        cooccurrence_header.addLayout(cooccurrence_text, stretch=1)
        self.pair_coverage_label = QLabel("—", self.cooccurrence_panel)
        self.pair_coverage_label.setObjectName("coverageBadge")
        self.pair_coverage_label.setWordWrap(True)
        self.pair_coverage_label.setMaximumWidth(300)
        self.pair_coverage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cooccurrence_header.addWidget(
            self.pair_coverage_label,
            alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
        )
        cooccurrence_layout.addLayout(cooccurrence_header)
        self.order_summary_layout = QHBoxLayout()
        self.order_summary_layout.setContentsMargins(0, 2, 0, 2)
        self.order_summary_layout.setSpacing(6)
        self.order_summary_values: dict[str, QLabel] = {}
        for key, label in (
            ("1", self.tr("Unary")),
            ("2", self.tr("Binary")),
            ("3", self.tr("Ternary")),
            ("4+", self.tr("Quaternary+")),
        ):
            card = QFrame(self.cooccurrence_panel)
            card.setObjectName("overviewOrderCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(9, 5, 9, 5)
            card_layout.setSpacing(0)
            value = QLabel("—", card)
            value.setObjectName("overviewOrderValue")
            caption = QLabel(label, card)
            caption.setObjectName("overviewOrderLabel")
            card_layout.addWidget(value)
            card_layout.addWidget(caption)
            self.order_summary_values[key] = value
            self.order_summary_layout.addWidget(card, stretch=1)
        cooccurrence_layout.addLayout(self.order_summary_layout)

        heat_legend = QHBoxLayout()
        heat_legend.setContentsMargins(0, 0, 0, 0)
        heat_legend.setSpacing(3)
        heat_legend.addStretch(1)
        heat_legend.addWidget(
            QLabel(self.tr("Relative count"), self.cooccurrence_panel)
        )
        heat_legend.addWidget(QLabel(self.tr("Low"), self.cooccurrence_panel))
        self.heat_legend_bar = QFrame(self.cooccurrence_panel)
        self.heat_legend_bar.setObjectName("overviewHeatLegend")
        self.heat_legend_bar.setAccessibleName(self.tr("Relative count color scale"))
        heat_legend_bar_layout = QHBoxLayout(self.heat_legend_bar)
        heat_legend_bar_layout.setContentsMargins(0, 0, 0, 0)
        heat_legend_bar_layout.setSpacing(0)
        for color in _OVERVIEW_JET_COLORS:
            swatch = QFrame(self.heat_legend_bar)
            swatch.setFixedSize(20, 9)
            swatch.setStyleSheet(f"background: {color}; border: 0;")
            heat_legend_bar_layout.addWidget(swatch)
        heat_legend.addWidget(self.heat_legend_bar)
        heat_legend.addWidget(QLabel(self.tr("High"), self.cooccurrence_panel))
        heat_legend.addStretch(1)
        cooccurrence_layout.addLayout(heat_legend)

        self.matrix_selection_label = QLabel(
            self.tr("Filter: none · Click a cell"),
            self.cooccurrence_panel,
        )
        self.matrix_selection_label.setObjectName("matrixSelectionStatus")
        self.matrix_selection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.matrix_selection_label.setWordWrap(True)
        cooccurrence_layout.addWidget(self.matrix_selection_label)

        self.cooccurrence_table = TableWidget(self.cooccurrence_panel)
        self.cooccurrence_table.setObjectName("elementCooccurrenceTable")
        default_matrix_delegate = self.cooccurrence_table.delegate
        matrix_delegate = _MatrixItemDelegate(
            self.cooccurrence_table
        )
        self.cooccurrence_table.delegate = matrix_delegate
        self.cooccurrence_table.setItemDelegate(matrix_delegate)
        default_matrix_delegate.deleteLater()
        self.cooccurrence_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.cooccurrence_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.cooccurrence_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.cooccurrence_table.setShowGrid(True)
        self.cooccurrence_table.setAlternatingRowColors(False)
        self.cooccurrence_table.setCheckedColor(
            QColor(Qt.GlobalColor.transparent),
            QColor(Qt.GlobalColor.transparent),
        )
        self.cooccurrence_table.setMinimumHeight(260)
        self.cooccurrence_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.cooccurrence_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.cooccurrence_table.verticalHeader().setDefaultSectionSize(26)
        self.cooccurrence_table.horizontalHeader().setDefaultSectionSize(26)
        self.cooccurrence_table.horizontalHeader().setMinimumSectionSize(18)
        self.cooccurrence_table.verticalHeader().setMinimumSectionSize(18)
        self.cooccurrence_table.cellClicked.connect(
            self._on_overview_matrix_cell_clicked
        )
        self.cooccurrence_table.cellActivated.connect(
            self._on_overview_matrix_cell_clicked
        )
        cooccurrence_layout.addWidget(self.cooccurrence_table, stretch=1)
        self.overview_columns.addWidget(self.cooccurrence_panel, stretch=3)

        self.element_sets_panel = QFrame(summary_tab)
        self.element_sets_panel.setObjectName("auditElementSetsPanel")
        element_sets_layout = QVBoxLayout(self.element_sets_panel)
        element_sets_layout.setContentsMargins(14, 12, 14, 12)
        element_sets_layout.setSpacing(7)
        element_sets_title = QLabel(
            self.tr("Main element sets"), self.element_sets_panel
        )
        element_sets_title.setObjectName("panelTitle")
        self.element_sets_summary_label = QLabel("", self.element_sets_panel)
        self.element_sets_summary_label.setObjectName("panelHint")
        self.element_sets_summary_label.setWordWrap(True)
        self.clear_element_filter_button = PushButton(
            self.tr("Show all element sets"), self.element_sets_panel
        )
        self.clear_element_filter_button.clicked.connect(
            self._clear_overview_element_filter
        )
        self.clear_element_filter_button.hide()
        element_sets_layout.addWidget(element_sets_title)
        element_sets_layout.addWidget(self.element_sets_summary_label)
        element_sets_layout.addWidget(
            self.clear_element_filter_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        self.element_sets_table = TableWidget(self.element_sets_panel)
        self.element_sets_table.setObjectName("overviewElementSetsTable")
        self.element_sets_table.setColumnCount(3)
        self.element_sets_table.setHorizontalHeaderLabels(
            [self.tr("Element set"), self.tr("Structures"), self.tr("Share")]
        )
        self.element_sets_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.element_sets_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.element_sets_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.element_sets_table.setShowGrid(False)
        self.element_sets_table.setAlternatingRowColors(True)
        self.element_sets_table.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.element_sets_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.element_sets_table.verticalHeader().setVisible(False)
        self.element_sets_table.verticalHeader().setDefaultSectionSize(30)
        element_sets_header = self.element_sets_table.horizontalHeader()
        element_sets_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        element_sets_header.setSectionResizeMode(
            1, QHeaderView.ResizeMode.Fixed
        )
        element_sets_header.setSectionResizeMode(
            2, QHeaderView.ResizeMode.Fixed
        )
        self.element_sets_table.itemSelectionChanged.connect(
            self._on_overview_element_set_selected
        )
        element_sets_layout.addWidget(self.element_sets_table, stretch=1)
        self.view_element_set_button = PrimaryPushButton(
            self.tr("View selected structures"), self.element_sets_panel
        )
        self.view_element_set_button.setEnabled(False)
        self.view_element_set_button.clicked.connect(
            lambda: self.selectStructuresSignal.emit(
                list(self._selected_overview_indices)
            )
        )
        element_sets_layout.addWidget(
            self.view_element_set_button,
            alignment=Qt.AlignmentFlag.AlignRight,
        )
        self.overview_columns.addWidget(self.element_sets_panel, stretch=2)
        summary_layout.addLayout(self.overview_columns, stretch=1)

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
            self.tr("View distribution"),
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
        self.composition_element_selector.setFixedWidth(86)
        self.composition_element_selector.currentIndexChanged.connect(
            self._refresh_composition_map
        )
        composition_header_layout.addWidget(self.composition_element_selector)
        self.composition_view_selector = ComboBox(composition_header)
        self.composition_view_selector.setFixedWidth(170)
        self.composition_view_selector.currentIndexChanged.connect(
            self._refresh_composition_map
        )
        composition_header_layout.addWidget(self.composition_view_selector)
        self.composition_evidence_button = PushButton(
            self.tr("Analyze phases and magnetic order"), composition_header
        )
        self.composition_evidence_button.clicked.connect(
            self._request_composition_structure_evidence
        )
        composition_header_layout.addWidget(self.composition_evidence_button)
        composition_layout.addWidget(composition_header)

        self.composition_splitter = QSplitter(
            Qt.Orientation.Horizontal, composition_page
        )
        self.composition_splitter.setObjectName("auditCompositionSplitter")
        self.composition_splitter.setChildrenCollapsible(False)
        self.composition_splitter.setHandleWidth(8)

        composition_visual_panel = QWidget(self.composition_splitter)
        composition_visual_layout = QVBoxLayout(composition_visual_panel)
        composition_visual_layout.setContentsMargins(0, 0, 0, 0)
        composition_visual_layout.setSpacing(8)

        self.composition_phase_summary_label = QLabel("", composition_visual_panel)
        self.composition_phase_summary_label.setObjectName("inventoryDetails")
        self.composition_phase_summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.composition_phase_summary_label.setWordWrap(True)
        self.composition_phase_summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        composition_visual_layout.addWidget(self.composition_phase_summary_label)
        self.composition_map_progress = ProgressBar(composition_visual_panel)
        self.composition_map_progress.setRange(0, 100)
        self.composition_map_progress.hide()
        composition_visual_layout.addWidget(self.composition_map_progress)
        self.composition_phase_progress = ProgressBar(composition_visual_panel)
        self.composition_phase_progress.setRange(0, 100)
        self.composition_phase_progress.hide()
        composition_visual_layout.addWidget(self.composition_phase_progress)

        self.composition_chart = AuditChartWidget(composition_visual_panel)
        self.composition_chart.setObjectName("auditCompositionChart")
        self.composition_chart.selectedGroupSignal.connect(
            self._on_composition_group_selected
        )
        composition_visual_layout.addWidget(self.composition_chart, stretch=1)

        composition_table_panel = QWidget(self.composition_splitter)
        composition_table_layout = QVBoxLayout(composition_table_panel)
        composition_table_layout.setContentsMargins(0, 0, 0, 0)
        composition_table_layout.setSpacing(6)
        composition_table_title = QLabel(
            self.tr("Exact composition groups"), composition_table_panel
        )
        composition_table_title.setObjectName("panelTitle")
        composition_table_layout.addWidget(composition_table_title)

        self.composition_table = TableWidget(composition_table_panel)
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
        composition_table_layout.addWidget(self.composition_table, stretch=1)
        composition_actions = QHBoxLayout()
        self.composition_selection_label = QLabel("", composition_table_panel)
        self.composition_selection_label.setObjectName("auditChartSelection")
        self.composition_selection_label.setWordWrap(True)
        self.composition_phase_selector = ComboBox(composition_table_panel)
        self.composition_phase_selector.setMinimumWidth(150)
        self.composition_phase_selector.currentIndexChanged.connect(
            self._refresh_phase_drilldown
        )
        self.composition_magnetic_selector = ComboBox(composition_table_panel)
        self.composition_magnetic_selector.setMinimumWidth(170)
        self.composition_magnetic_selector.currentIndexChanged.connect(
            self._refresh_phase_drilldown
        )
        self.composition_phase_selector.hide()
        self.composition_magnetic_selector.hide()
        self.composition_show_button = PrimaryPushButton(
            self.tr("Show selected structures"), composition_table_panel
        )
        self.composition_show_button.setEnabled(False)
        self.composition_show_button.clicked.connect(
            self._emit_composition_structures
        )
        composition_table_layout.addWidget(self.composition_selection_label)
        composition_actions.addStretch(1)
        composition_actions.addWidget(self.composition_phase_selector)
        composition_actions.addWidget(self.composition_magnetic_selector)
        composition_actions.addWidget(self.composition_show_button)
        composition_table_layout.addLayout(composition_actions)

        self.composition_splitter.addWidget(composition_visual_panel)
        self.composition_splitter.addWidget(composition_table_panel)
        self.composition_splitter.setStretchFactor(0, 9)
        self.composition_splitter.setStretchFactor(1, 11)
        self.composition_splitter.setSizes([520, 640])
        composition_layout.addWidget(self.composition_splitter, stretch=1)
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
        self.analyze_structure_evidence_button = PushButton(
            self.tr("Analyze evidence"), self.dimension_rail
        )
        self.analyze_structure_evidence_button.clicked.connect(
            self.requestStructureEvidenceSignal.emit
        )
        rail_layout.addWidget(self.analyze_structure_evidence_button)
        rail_layout.addWidget(self.composition_phase_progress)
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
        self.target_config_types_edit = QLineEdit(self.target_definition_panel)
        self.target_config_types_edit.setPlaceholderText(
            self.tr("Optional exact config_type values, e.g. bulk, vacancy")
        )
        self.target_quantity_rule_check = QCheckBox(
            self.tr("Use a minimum support rule"), self.target_definition_panel
        )
        self.target_minimum_count_spin = QSpinBox(self.target_definition_panel)
        self.target_minimum_count_spin.setRange(1, 1_000_000_000)
        self.target_minimum_count_spin.setValue(1000)
        self.target_minimum_count_spin.setEnabled(False)
        self.target_quantity_rule_check.toggled.connect(
            self.target_minimum_count_spin.setEnabled
        )
        target_form.addRow(self.tr("Element"), self.target_element_selector)
        target_form.addRow(self.tr("Range"), self.target_minimum_spin)
        target_form.addRow(self.tr("to"), self.target_maximum_spin)
        target_form.addRow(self.tr("Key points (optional)"), self.target_points_edit)
        target_form.addRow(self.tr("Structure families"), self.target_config_types_edit)
        target_form.addRow(self.target_quantity_rule_check)
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
        self.target_table.setColumnCount(6)
        self.target_table.setHorizontalHeaderLabels(
            [
                self.tr("Composition point"),
                self.tr("Structure family"),
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

    def set_detached_state(self, detached: bool) -> None:
        """Keep the compact header action honest about the current host."""
        if detached:
            self.detach_button.setIcon(FluentIcon.BACK_TO_WINDOW)
            text = self.tr("Return to main window")
        else:
            self.detach_button.setIcon(FluentIcon.FULL_SCREEN)
            text = self.tr("Open in separate window")
        self.detach_button.setToolTip(text)
        self.detach_button.setAccessibleName(text)

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
        cell.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        cell_layout = QVBoxLayout(cell)
        cell_layout.setContentsMargins(10, 0, 10, 0)
        cell_layout.setSpacing(1)
        value = QLabel(value_text, cell)
        value.setObjectName("metricValue")
        value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value.setMinimumWidth(0)
        value.setWordWrap(True)
        value.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        label = QLabel(label_text, cell)
        label.setObjectName("metricLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
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
        self._all_topics = []
        self._topics = []
        self._dimensions = {}
        self._active_plots = []
        self._local_chemistry_plots = []
        self._selected_chart_indices = []
        self._selected_composition_indices = []
        self._selected_phase_indices = []
        self._selected_magnetic_indices = []
        self._selected_composition_key = None
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
        self.composition_view_selector.clear()
        self.composition_evidence_button.hide()
        self.target_element_selector.clear()
        self.composition_chart.clear()
        self.composition_phase_summary_label.clear()
        self.composition_phase_summary_label.hide()
        self.composition_phase_progress.hide()
        self.target_chart.clear()
        self.composition_table.setRowCount(0)
        self.target_table.setRowCount(0)
        self._target_configured = False
        self._target_dataset_fingerprint = ""
        self._requested_composition_view = ""
        self.target_result_summary_label.setText(
            self.tr(
                "No target has been set. The dataset inventory remains valid; define a range or key points before comparing support."
            )
        )
        self.composition_selection_label.clear()
        self.composition_phase_selector.clear()
        self.composition_phase_selector.setEnabled(False)
        self.composition_magnetic_selector.clear()
        self.composition_magnetic_selector.setEnabled(False)
        self.composition_show_button.setEnabled(False)
        self.target_selection_label.clear()
        self.target_show_button.setEnabled(False)
        self._overview_element_sets = ()
        self._overview_elements = ()
        self._overview_structure_count = 0
        self._overview_element_counts = {}
        self._overview_pair_counts = {}
        self._overview_exact_pair_counts = {}
        self._overview_pure_counts = {}
        self._overview_basis_elements = ()
        self._selected_overview_elements = ()
        self._selected_overview_cell = None
        self._selected_overview_mode = ""
        self._selected_overview_indices = []
        self.cooccurrence_table.setRowCount(0)
        self.cooccurrence_table.setColumnCount(0)
        self.element_sets_table.setRowCount(0)
        self.element_sets_summary_label.clear()
        self.pair_coverage_label.setText("—")
        self.clear_element_filter_button.hide()
        self.view_element_set_button.setEnabled(False)
        self._update_overview_selection_status()
        for value in self.order_summary_values.values():
            value.setText("—")
        self.review_summary_label.clear()
        self.label_availability_value.clear()
        self._populate_slice_table()
        self._clear_evidence()
        self.no_dataset_state.setText(self.tr("No dataset loaded"))
        self.no_dataset_hint.setText(
            self.tr(
                "Open a structure or result file in NEP Dataset Display before running checks."
            )
        )
        self._reserve_no_dataset_hint_height()
        self.no_dataset_action_button.show()
        self.no_dataset_panel.show()
        self.no_dataset_state.show()
        self.audit_header.hide()
        self.dashboard_body.hide()

    def set_loading(self, dataset_id: str) -> None:
        """Show a quiet progress state while the audit runs off the UI thread."""
        self._set_empty_state()
        self.no_dataset_state.setText(
            self.tr("Analyzing {dataset}...").format(dataset=dataset_id)
        )
        self.no_dataset_hint.setText(self.tr("Please wait while the checks run."))
        self._reserve_no_dataset_hint_height()
        self.no_dataset_action_button.hide()

    def _reserve_no_dataset_hint_height(self) -> None:
        """Keep every wrapped hint line visible at the active font scale."""
        self.no_dataset_hint.ensurePolished()
        self.no_dataset_state.setMinimumHeight(
            self.no_dataset_state.sizeHint().height()
        )
        hint_width = max(
            1,
            self.no_dataset_hint.width(),
            self.no_dataset_hint.minimumWidth(),
        )
        required_height = self.no_dataset_hint.heightForWidth(hint_width)
        if required_height > 0:
            self.no_dataset_hint.setMinimumHeight(required_height)
            self.no_dataset_hint.updateGeometry()
            self.no_dataset_panel.updateGeometry()

    def showEvent(self, event) -> None:
        """Re-measure the empty-state hint after the active style is polished."""
        super().showEvent(event)
        if self.no_dataset_panel.isVisible():
            QTimer.singleShot(0, self._reserve_no_dataset_hint_height)
        if hasattr(self, "cooccurrence_table"):
            QTimer.singleShot(0, self._resize_overview_matrix)

    def open_file(self) -> None:
        """Ask the main window to open a dataset for this page."""
        self.requestDatasetOpenSignal.emit()

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

    @staticmethod
    def _dataset_state_key(result: AuditResult | None) -> str:
        if result is None:
            return ""
        return result.fingerprints.dataset or result.dataset_id

    def _populate_inventory_views(self) -> None:
        inventory = self._inventory()
        preferred_composition_element = self.composition_element_selector.currentData()
        preferred_target_element = self.target_element_selector.currentData()
        self.composition_element_selector.blockSignals(True)
        self.target_element_selector.blockSignals(True)
        self.composition_element_selector.clear()
        self.target_element_selector.clear()
        if inventory is not None:
            for element in inventory.elements:
                self.composition_element_selector.addItem(element, userData=element)
                self.target_element_selector.addItem(element, userData=element)
            default_index = max(0, len(inventory.elements) - 1)
            composition_index = self.composition_element_selector.findData(
                preferred_composition_element
            )
            target_index = self.target_element_selector.findData(
                preferred_target_element
            )
            self.composition_element_selector.setCurrentIndex(
                composition_index if composition_index >= 0 else default_index
            )
            self.target_element_selector.setCurrentIndex(
                target_index if target_index >= 0 else default_index
            )
        self.composition_element_selector.blockSignals(False)
        self.target_element_selector.blockSignals(False)
        phase_ready, magnetic_ready, _ = self._structure_evidence_state()
        self.composition_phase_selector.setVisible(phase_ready)
        self.composition_magnetic_selector.setVisible(magnetic_ready)
        self._populate_composition_view_selector()
        self._refresh_composition_map()
        if self._target_configured:
            self._apply_composition_target()
        else:
            self.target_table.setRowCount(0)
            self.target_chart.clear()
            self.target_result_summary_label.setText(
                self.tr(
                    "No target has been set. The dataset inventory remains valid; define a range or key points before comparing support."
                )
            )

    def _structure_evidence_state(self) -> tuple[bool, bool, bool]:
        if self._result is None:
            return False, False, False
        phase_ready = self._result.phase_inventory is not None
        magnetic_ready = self._result.magnetic_inventory is not None
        magnetic_meta = self._result.overview_metrics.get("magnetic_inventory", {})
        no_spin = (
            isinstance(magnetic_meta, Mapping)
            and magnetic_meta.get("status") == "no-spin"
        )
        return phase_ready, magnetic_ready, no_spin

    def _populate_composition_view_selector(self) -> None:
        current_mode = self.composition_view_selector.currentData() or "count"
        phase_ready, magnetic_ready, no_spin = self._structure_evidence_state()
        self.composition_view_selector.blockSignals(True)
        self.composition_view_selector.clear()
        self.composition_view_selector.addItem(
            self.tr("Sample counts"), userData="count"
        )
        if phase_ready:
            self.composition_view_selector.addItem(
                self.tr("Counts colored by structural phase"), userData="structural"
            )
        if magnetic_ready:
            self.composition_view_selector.addItem(
                self.tr("Counts colored by magnetic order"), userData="magnetic"
            )
        selected_index = self.composition_view_selector.findData(current_mode)
        self.composition_view_selector.setCurrentIndex(max(0, selected_index))
        self.composition_view_selector.blockSignals(False)

        evidence_complete = phase_ready and (magnetic_ready or no_spin)
        evidence_partial = phase_ready or magnetic_ready
        self.composition_evidence_button.setVisible(not evidence_complete)
        self._set_fitted_button_text(
            self.composition_evidence_button,
            self.tr("Analyze remaining evidence")
            if evidence_partial
            else self.tr("Analyze phases and magnetic order"),
        )
        self.analyze_structure_evidence_button.setEnabled(not evidence_complete)
        self._set_fitted_button_text(
            self.analyze_structure_evidence_button,
            self.tr("Evidence available")
            if evidence_complete
            else (
                self.tr("Analyze remaining")
                if evidence_partial
                else self.tr("Analyze evidence")
            ),
        )

    def _request_composition_structure_evidence(self) -> None:
        self._requested_composition_view = "structural"
        self.requestStructureEvidenceSignal.emit()

    @staticmethod
    def _set_fitted_button_text(button: PushButton, text: str) -> None:
        """Update dynamic button text without retaining a stale narrow geometry."""
        button.setText(text)
        button.ensurePolished()
        button.setMinimumWidth(button.sizeHint().width())
        button.updateGeometry()

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
        mode = self.composition_view_selector.currentData() or "count"
        labels_by_index: dict[int, str] = {}
        if mode == "structural" and self._result is not None:
            phase_inventory = self._result.phase_inventory
            if phase_inventory is not None:
                labels_by_index = {
                    structure.source_index: phase_partition_label(structure)
                    for point in phase_inventory.composition_points
                    for structure in point.structures
                }
            ordered_labels = PHASE_PARTITION_LABELS
            display_name = self._phase_display_name
            title = self.tr(
                "Sample counts by {element} concentration, colored by structural phase"
            ).format(element=element)
            plot_id = f"inventory:composition-phase:{element}"
        elif mode == "magnetic" and self._result is not None:
            magnetic_inventory = self._result.magnetic_inventory
            if magnetic_inventory is not None:
                labels_by_index = {
                    structure.source_index: magnetic_partition_label(structure)
                    for point in magnetic_inventory.composition_points
                    for structure in point.structures
                }
            ordered_labels = MAGNETIC_PARTITION_LABELS
            display_name = self._magnetic_display_name
            title = self.tr(
                "Sample counts by {element} concentration, colored by magnetic order"
            ).format(element=element)
            plot_id = f"inventory:composition-magnetism:{element}"
        else:
            ordered_labels = ()
            display_name = str
            title = ""
            plot_id = ""
        if ordered_labels and labels_by_index:
            stacked_series = []
            for label in ordered_labels:
                index_groups = tuple(
                    tuple(
                        sorted(
                            index
                            for point in points
                            for index in point.structure_indices
                            if (
                                labels_by_index.get(index) == label
                                or (label == "no_spin" and index not in labels_by_index)
                            )
                        )
                    )
                    for _, points in groups
                )
                counts = tuple(len(indices) for indices in index_groups)
                if any(counts):
                    stacked_series.append(
                        {
                            "id": label,
                            "label": display_name(label),
                            "counts": counts,
                            "structure_indices": index_groups,
                        }
                    )
            return {
                "kind": "composition_phase_stacks",
                "id": plot_id,
                "title": title,
                "x_label": self.tr("{element} atomic fraction").format(
                    element=element
                ),
                "y_label": self.tr("Structures"),
                "x_min": -0.01,
                "x_max": 1.0,
                "target_points": target_points,
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
                "series": tuple(stacked_series),
            }
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
            "log_scale": False,
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
            self.composition_phase_summary_label.clear()
            self.composition_phase_summary_label.hide()
            self.composition_table.setRowCount(0)
            return
        hint = self.tr(
                "{points} exact normalized composition points across {structures:,} structures. "
                "Supercells with the same atomic fractions are merged."
            ).format(
                points=len(inventory.composition_points),
                structures=inventory.structure_count,
            )
        if len(inventory.elements) > 2:
            hint += " " + self.tr(
                "This chart is a one-element projection; one plotted concentration may contain multiple exact multicomponent compositions."
            )
        mode = self.composition_view_selector.currentData() or "count"
        if mode == "structural":
            self.composition_phase_summary_label.setText(
                self.tr(
                    "Bar height is the sample count. Colored segments show the snapshot structural-phase shares within that concentration; click a segment to open those structures."
                )
            )
            self.composition_phase_summary_label.show()
        elif mode == "magnetic":
            self.composition_phase_summary_label.setText(
                self.tr(
                    "Bar height is the sample count. Colored segments show the snapshot magnetic-order shares within that concentration; structures without a valid spin field remain a separate group. Click a segment to open those structures."
                )
            )
            self.composition_phase_summary_label.show()
        else:
            self.composition_phase_summary_label.hide()
        self.composition_map_hint.setText(hint)
        self.composition_chart.set_plot(self._composition_plot(element))
        self._populate_composition_table(element)

    def _phase_display_name(self, label: str) -> str:
        return {
            "fcc": "FCC",
            "hcp": "HCP",
            "bcc": "BCC",
            "l12": "L1₂",
            "c14": "C14 Laves",
            "c15": "C15 Laves",
            "mixed": self.tr("Mixed local structure"),
            "unresolved": self.tr("Unresolved"),
        }.get(label, label)

    def _magnetic_display_name(self, label: str) -> str:
        return {
            "fm": self.tr("FM"),
            "afm": self.tr("AFM"),
            "afm_layered": self.tr("Layered AFM (↑↓)"),
            "afm_double_layered": self.tr("Double-layer AFM (↑↑↓↓)"),
            "ferrimagnetic": self.tr("FiM"),
            "pm_like": self.tr("PM-like (spin-disordered)"),
            "noncollinear": self.tr("Other noncollinear"),
            "unresolved": self.tr("Unresolved magnetic type"),
            "low_moment": self.tr("Low / zero moment"),
            "no_spin": self.tr("No valid spin field"),
        }.get(label, label)

    def _element_order_display_name(self, label: str) -> str:
        return {
            "aligned": self.tr("Aligned (FM-like)"),
            "compensated": self.tr("Compensated (AFM-like)"),
            "modulated": self.tr("Modulated / spiral-like"),
            "noncollinear": self.tr("Noncollinear"),
            "collinear_mixed": self.tr("Mixed collinear"),
            "disordered": self.tr("Disordered-like"),
            "low_moment": self.tr("Low / zero moment"),
            "insufficient": self.tr("Insufficient local evidence"),
        }.get(label, label)

    def _coupling_display_name(self, label: str) -> str:
        return {
            "parallel": self.tr("Parallel coupling"),
            "antiparallel": self.tr("Antiparallel coupling"),
            "mixed": self.tr("Mixed coupling"),
        }.get(label, label)

    def _phase_point_for_reduced_counts(self, reduced_counts: tuple[int, ...]):
        if self._result is None or self._result.phase_inventory is None:
            return None
        return next(
            (
                point
                for point in self._result.phase_inventory.composition_points
                if point.reduced_counts == reduced_counts
            ),
            None,
        )

    def _magnetic_point_for_reduced_counts(self, reduced_counts: tuple[int, ...]):
        if self._result is None or self._result.magnetic_inventory is None:
            return None
        return next(
            (
                point
                for point in self._result.magnetic_inventory.composition_points
                if point.reduced_counts == reduced_counts
            ),
            None,
        )

    def _render_phase_summary(
        self,
        structure_indices: tuple[int, ...] = (),
    ) -> None:
        if self._result is None:
            self.composition_phase_summary_label.hide()
            return
        inventory = self._inventory()
        selected = set(structure_indices)
        allowed_keys = None
        if selected and inventory is not None:
            allowed_keys = {
                point.reduced_counts
                for point in inventory.composition_points
                if selected.intersection(point.structure_indices)
            }
        scope_text = (
            self.tr("Selected composition group")
            if selected
            else self.tr("Current audited scope")
        )
        sections: list[str] = []
        phase_inventory = self._result.phase_inventory
        if phase_inventory is not None:
            summary = summarize_phase_inventory(phase_inventory, allowed_keys)
            if summary is not None:
                local_fractions = dict(summary.local_phase_fractions)
                confidence_totals = dict(summary.confidence_counts)
                confirmed_totals = dict(summary.confirmed_candidates)
                phase_text = " &nbsp;·&nbsp; ".join(
                    f"<b>{escape(self._phase_display_name(label))} "
                    f"{local_fractions[label]:.1%}</b>"
                    for label in ("fcc", "hcp", "bcc", "unresolved")
                )
                confirmed_text = ""
                if confirmed_totals:
                    confirmed_text = self.tr(" Confirmed ordering: {values}.").format(
                        values=", ".join(
                            f"{escape(self._phase_display_name(label))} × {count:,}"
                            for label, count in sorted(confirmed_totals.items())
                        )
                    )
                sections.append(self.tr(
                    "<b>{scope}: structural order</b> &nbsp; {phases}<br>"
                    "Analyzed all {analyzed:,} structures; {strong:,} have strong evidence.{confirmed} "
                    "This classifies local structure; it does not predict thermodynamic stability."
                ).format(
                    scope=escape(scope_text), phases=phase_text,
                    analyzed=summary.analyzed_structure_count,
                    strong=confidence_totals.get("strong", 0), confirmed=confirmed_text,
                ))
        else:
            phase_meta = self._result.overview_metrics.get("phase_inventory", {})
            sections.append(
                self.tr("Structural-order analysis is running for every structure.")
                if isinstance(phase_meta, Mapping) and phase_meta.get("status") == "pending"
                else self.tr("Structural-order evidence is unavailable.")
            )

        magnetic_inventory = self._result.magnetic_inventory
        if magnetic_inventory is not None:
            magnetic = summarize_magnetic_inventory(magnetic_inventory, allowed_keys)
            if magnetic is not None:
                order_text = " &nbsp;·&nbsp; ".join(
                    f"<b>{escape(self._magnetic_display_name(label))} {fraction:.1%}</b>"
                    for label, fraction in magnetic.order_fractions[:5]
                )
                sections.append(self.tr(
                    "<b>{scope}: magnetic order</b> &nbsp; {orders}<br>"
                    "Analyzed {analyzed:,} spin structures; {missing:,} lack a valid spin:R:3 field. "
                    "Pattern evidence: net moment ratio {net:.2f}, collinearity {col:.2f}, q-peak {q:.2f}. "
                    "This is a snapshot-pattern classification, not a thermodynamic FM/AFM/PM claim."
                ).format(
                    scope=escape(scope_text), orders=order_text,
                    analyzed=magnetic.analyzed_structure_count,
                    missing=magnetic.missing_spin_count,
                    net=magnetic.mean_net_moment_ratio,
                    col=magnetic.mean_collinearity,
                    q=magnetic.mean_q_peak_strength,
                ))
                element_rows = "".join(
                    "<tr>"
                    f"<td style='padding:2px 12px 2px 0'><b>{escape(item.element)}</b></td>"
                    f"<td style='padding:2px 12px 2px 0'>{escape(self._element_order_display_name(item.order_fractions[0][0]))} {item.order_fractions[0][1]:.0%}</td>"
                    f"<td style='padding:2px 12px 2px 0'>{escape(self.tr('moment'))} {item.mean_moment:.2f}</td>"
                    f"<td style='padding:2px 12px 2px 0'>{escape(self.tr('net'))} {item.mean_net_moment_ratio:.2f}</td>"
                    f"<td style='padding:2px 0'>{escape(self.tr('same-element correlation'))} {item.mean_intra_element_correlation:+.2f}</td>"
                    "</tr>"
                    for item in magnetic.element_summaries
                    if item.order_fractions
                )
                if element_rows:
                    sections.append(
                        self.tr(
                            "<b>Element-local spin patterns</b><br>"
                            "{rows}"
                            "These labels describe each element's spin sublattice inside the selected structures."
                        ).format(rows=f"<table cellspacing='0'>{element_rows}</table>")
                    )
                pair_text = " &nbsp;·&nbsp; ".join(
                    f"<b>{escape(item.element_a)}–{escape(item.element_b)}</b> "
                    f"{escape(self._coupling_display_name(item.coupling_fractions[0][0]))} "
                    f"{item.coupling_fractions[0][1]:.0%} ({item.mean_correlation:+.2f})"
                    for item in magnetic.element_pair_summaries
                    if item.coupling_fractions
                )
                if pair_text:
                    sections.append(self.tr(
                        "<b>Element-pair coupling</b><br>{pairs}<br>"
                        "Correlation compares neighboring spin directions; it is not a chemical-bond label."
                    ).format(pairs=pair_text))
            else:
                sections.append(self.tr(
                    "<b>Magnetic order</b><br>No valid per-atom spin:R:3 field is available in this scope. "
                    "mforce and force_mag are force labels and are not used as spin states."
                ))
        else:
            magnetic_meta = self._result.overview_metrics.get("magnetic_inventory", {})
            if isinstance(magnetic_meta, Mapping) and magnetic_meta.get("status") == "pending":
                sections.append(self.tr("Magnetic-order analysis is running for every spin structure."))
        self.composition_phase_summary_label.setText(
            "<div>" + "<hr style='border:0;border-top:1px solid #d9e3e3;margin:7px 0'>".join(sections) + "</div>"
        )
        self.composition_phase_summary_label.show()

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
            formula_item.setData(
                Qt.ItemDataRole.UserRole + 1,
                point.reduced_counts,
            )
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
            all_config_types = ", ".join(
                f"{name} × {count:,}" for name, count in point.config_types
            )
            visible_config_types = ", ".join(
                f"{name} × {count:,}" for name, count in point.config_types[:4]
            )
            if len(point.config_types) > 4:
                visible_config_types += self.tr(" · +{count} more").format(
                    count=len(point.config_types) - 4
                )
            config_item = QTableWidgetItem(
                visible_config_types or self.tr("Not labeled")
            )
            config_item.setToolTip(all_config_types)
            self.composition_table.setItem(row, 4, config_item)
            formula_item.setToolTip(
                f"{self.tr('Atom counts')}: {atom_counts}\n"
                f"{self.tr('Configuration types')}: "
                f"{all_config_types or self.tr('Not labeled')}"
            )
        if points:
            self.composition_table.selectRow(0)

    def _on_composition_table_selection_changed(self) -> None:
        row = self.composition_table.currentRow()
        item = self.composition_table.item(row, 0) if row >= 0 else None
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else ()
        reduced_counts = (
            item.data(Qt.ItemDataRole.UserRole + 1) if item is not None else None
        )
        self._selected_composition_key = (
            tuple(int(value) for value in reduced_counts)
            if isinstance(reduced_counts, tuple)
            else None
        )
        self._set_composition_selection(value if isinstance(value, tuple) else ())

    def _on_composition_group_selected(self, structure_indices: list[int]) -> None:
        self._selected_composition_key = None
        self._set_composition_selection(tuple(int(index) for index in structure_indices))

    def _set_composition_selection(self, structure_indices: tuple[int, ...]) -> None:
        self._selected_composition_indices = list(structure_indices)
        count = len(structure_indices)
        self.composition_selection_label.setText(
            self.tr("Selected composition point: {count:,} structures").format(
                count=count
            )
        )
        self._set_fitted_button_text(
            self.composition_show_button,
            self.tr("Show {count:,} structures").format(count=count),
        )
        self.composition_show_button.setEnabled(bool(structure_indices))
        self._refresh_phase_drilldown()

    def _selected_phase_evidence(self):
        if self._result is None or self._result.phase_inventory is None:
            return ()
        selected = set(self._selected_composition_indices)
        return tuple(
            structure
            for point in self._result.phase_inventory.composition_points
            for structure in point.structures
            if structure.source_index in selected
        )

    def _selected_magnetic_evidence(self):
        if self._result is None or self._result.magnetic_inventory is None:
            return ()
        selected = set(self._selected_composition_indices)
        return tuple(
            structure
            for point in self._result.magnetic_inventory.composition_points
            for structure in point.structures
            if structure.source_index in selected
        )

    def _refresh_phase_drilldown(self, index: int = -1) -> None:
        del index
        phase_previous = self.composition_phase_selector.currentData()
        magnetic_previous = self.composition_magnetic_selector.currentData()
        phase_evidence = self._selected_phase_evidence()
        magnetic_evidence = self._selected_magnetic_evidence()

        def populate(selector, evidence, value_getter, display, previous, all_text):
            counts: dict[str, int] = {}
            for structure in evidence:
                label = value_getter(structure)
                counts[label] = counts.get(label, 0) + 1
            selector.blockSignals(True)
            selector.clear()
            selector.addItem(all_text, userData="")
            selected_index = 0
            for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                selector.addItem(
                    self.tr("{label} · {count} structures").format(
                        label=display(label), count=count
                    ),
                    userData=label,
                )
                if label == previous:
                    selected_index = selector.count() - 1
            selector.setCurrentIndex(selected_index)
            selector.setEnabled(bool(evidence))
            selector.blockSignals(False)

        populate(
            self.composition_phase_selector, phase_evidence,
            phase_partition_label,
            self._phase_display_name, phase_previous, self.tr("All structural phases")
        )
        populate(
            self.composition_magnetic_selector, magnetic_evidence,
            magnetic_partition_label,
            self._magnetic_display_name, magnetic_previous, self.tr("All magnetic orders")
        )
        selected_phase = self.composition_phase_selector.currentData() or ""
        selected_magnetic = self.composition_magnetic_selector.currentData() or ""
        selected_indices = set(self._selected_composition_indices)
        if selected_phase:
            selected_indices.intersection_update(
                structure.source_index
                for structure in phase_evidence
                if phase_partition_label(structure) == selected_phase
            )
        if selected_magnetic:
            selected_indices.intersection_update(
                structure.source_index
                for structure in magnetic_evidence
                if magnetic_partition_label(structure) == selected_magnetic
            )
        self._selected_phase_indices = sorted(selected_indices)
        self._selected_magnetic_indices = sorted(selected_indices)
        if selected_phase and not selected_magnetic:
            button_text = self.tr("Show {count:,} {phase} structures").format(
                count=len(selected_indices),
                phase=self._phase_display_name(selected_phase),
            )
        else:
            button_text = self.tr("Show {count:,} matching structures").format(
                count=len(selected_indices)
            )
        self._set_fitted_button_text(self.composition_show_button, button_text)
        self.composition_show_button.setEnabled(bool(selected_indices))

    def start_phase_analysis(self, total: int) -> None:
        del total
        self.analyze_structure_evidence_button.setEnabled(False)
        self._set_fitted_button_text(
            self.analyze_structure_evidence_button,
            self.tr("Analyzing evidence..."),
        )
        self.composition_evidence_button.show()
        self.composition_evidence_button.setEnabled(False)
        self._set_fitted_button_text(
            self.composition_evidence_button,
            self.tr("Analyzing phases and magnetic order..."),
        )
        self.export_report_button.setEnabled(False)
        self.export_report_button.setToolTip(
            self.tr("Wait for complete structural and magnetic-order analysis before exporting the report.")
        )
        self.composition_phase_progress.setValue(0)
        self.composition_phase_progress.show()
        self.composition_map_progress.setValue(0)
        self.composition_map_progress.show()
        self.analysis_status_label.setText(
            self.tr("Structural and magnetic snapshot evidence is being analyzed on demand.")
        )
        self.composition_phase_summary_label.setText(
            self.tr("Structural and magnetic snapshot evidence is being analyzed on demand.")
        )
        self.composition_phase_summary_label.show()

    def update_phase_analysis_progress(self, completed: int, total: int) -> None:
        total = max(1, int(total))
        completed = min(total, max(0, int(completed)))
        progress = round(100 * completed / total)
        self.composition_phase_progress.setValue(progress)
        self.composition_map_progress.setValue(progress)
        if total == self._structure_count():
            progress_text = self.tr(
                "Analyzing local phases: {completed:,}/{total:,} structures. "
                "The chart will update automatically."
            ).format(completed=completed, total=total)
        else:
            progress_text = self.tr(
                "Analyzing structural and magnetic order: {completed:,}/{total:,} checks "
                "across {structures:,} structures. The chart will update automatically."
            ).format(
                completed=completed,
                total=total,
                structures=self._structure_count(),
            )
        self.analysis_status_label.setText(progress_text)
        self.composition_phase_summary_label.setText(progress_text)
        self.composition_phase_summary_label.show()

    def finish_phase_analysis(self, result: AuditResult) -> None:
        """Apply complete background phase evidence without resetting navigation."""
        selected_key = self._selected_composition_key
        selected_dimension = self._selected_dimension_id()
        self._result = result
        self.export_report_button.setEnabled(True)
        self.export_report_button.setToolTip("")
        self.composition_phase_progress.hide()
        self.composition_map_progress.hide()
        self.analysis_status_label.setText(
            self.tr("Structural and magnetic snapshot evidence is available.")
        )
        self._update_summary()
        self._populate_inventory_views()
        self.composition_evidence_button.setEnabled(True)
        if self._requested_composition_view:
            requested_index = self.composition_view_selector.findData(
                self._requested_composition_view
            )
            if requested_index >= 0:
                self.composition_view_selector.setCurrentIndex(requested_index)
            self._requested_composition_view = ""
        self._sync_phase_dimension_item()
        self._sync_magnetic_dimension_item()
        if selected_key is not None:
            for row in range(self.composition_table.rowCount()):
                item = self.composition_table.item(row, 0)
                if item is not None and item.data(
                    Qt.ItemDataRole.UserRole + 1
                ) == selected_key:
                    self.composition_table.selectRow(row)
                    break
        if selected_dimension == "phase_evidence":
            self._update_analysis("phase_evidence")
        elif selected_dimension == "magnetic_evidence":
            self._update_analysis("magnetic_evidence")

    def fail_phase_analysis(self, message: str) -> None:
        if self._result is not None:
            overview = dict(self._result.overview_metrics)
            phase_meta = dict(overview.get("phase_inventory", {}))
            phase_meta.update({"available": False, "status": "unavailable"})
            overview["phase_inventory"] = phase_meta
            magnetic_meta = dict(overview.get("magnetic_inventory", {}))
            magnetic_meta.update({"available": False, "status": "unavailable"})
            overview["magnetic_inventory"] = magnetic_meta
            self._result = replace(self._result, overview_metrics=overview)
        self.composition_phase_progress.hide()
        self.composition_map_progress.hide()
        self.analyze_structure_evidence_button.setEnabled(True)
        self._set_fitted_button_text(
            self.analyze_structure_evidence_button,
            self.tr("Retry evidence"),
        )
        self.composition_evidence_button.show()
        self.composition_evidence_button.setEnabled(True)
        self._set_fitted_button_text(
            self.composition_evidence_button,
            self.tr("Retry phases and magnetic order"),
        )
        self.export_report_button.setEnabled(True)
        self.export_report_button.setToolTip("")
        self.analysis_status_label.setText(
            self.tr("Structural or magnetic-order analysis failed: {message}").format(message=message)
        )
        self.composition_phase_summary_label.setText(
            self.tr("Structural or magnetic-order analysis failed: {message}").format(message=message)
        )
        self.composition_phase_summary_label.show()
        self._requested_composition_view = ""
        self._update_summary()
        self._sync_phase_dimension_item()
        self._sync_magnetic_dimension_item()

    def _emit_composition_structures(self) -> None:
        if self._selected_phase_indices:
            self.selectStructuresSignal.emit(list(self._selected_phase_indices))

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
        self._review_states[self._review_state_key(topic)] = state
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
        decided = sum(
            self._review_states.get(self._review_state_key(topic), "pending") != "pending"
            for topic in self._topics
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

    def _parse_target_config_types(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                token.strip()
                for token in re.split(r"[,，;；]", self.target_config_types_edit.text())
                if token.strip()
            )
        )

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
        self._target_configured = True
        self._target_dataset_fingerprint = self._dataset_state_key(self._result)
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
                "inside the selected range. This is an inventory view and cannot reveal missing "
                "points between existing samples."
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
                0, 2, QTableWidgetItem(self.tr("No sample in range"))
            )
            self.target_table.setItem(0, 1, QTableWidgetItem(self.tr("All structures")))
            self.target_table.setItem(0, 3, QTableWidgetItem("0"))
            self.target_table.setItem(0, 4, QTableWidgetItem("—"))
            self.target_table.setItem(
                0, 5, QTableWidgetItem(self.tr("Review nearby compositions"))
            )
            self.target_result_summary_label.setText(mode_summary)
            self._selected_target_indices = []
            self.target_selection_label.clear()
            self.target_show_button.setEnabled(False)
            return
        config_types = self._parse_target_config_types()
        minimum_structure_count = (
            self.target_minimum_count_spin.value()
            if self.target_quantity_rule_check.isChecked()
            else None
        )
        target = CompositionTarget(
            element=element,
            minimum=minimum,
            maximum=maximum,
            key_points=points,
            minimum_structure_count=minimum_structure_count,
            config_types=config_types,
        )
        cells = compare_composition_target(inventory, target)
        status_counts = {
            status: sum(cell.status == status for cell in cells)
            for status in TargetSupportStatus
        }
        if minimum_structure_count is None:
            rule_summary = self.tr(
                "Minimum support rule is off. Exact samples: {supported} · no matching sample: {missing} · metadata incomplete: {unknown}."
            ).format(
                supported=status_counts[TargetSupportStatus.SUPPORTED],
                missing=status_counts[TargetSupportStatus.NO_SAMPLE]
                + status_counts[TargetSupportStatus.NO_CONFIG_TYPE],
                unknown=status_counts[TargetSupportStatus.UNJUDGEABLE],
            )
        else:
            rule_summary = self.tr(
                "Using your rule of at least {minimum:,} structures per point: met {supported} · below rule {thin} · no matching sample {missing} · cannot fully evaluate {unknown}."
            ).format(
                minimum=minimum_structure_count,
                supported=status_counts[TargetSupportStatus.SUPPORTED],
                thin=status_counts[TargetSupportStatus.THIN],
                missing=status_counts[TargetSupportStatus.NO_SAMPLE]
                + status_counts[TargetSupportStatus.NO_CONFIG_TYPE],
                unknown=status_counts[TargetSupportStatus.UNJUDGEABLE],
            )
        self.target_result_summary_label.setText(mode_summary + " " + rule_summary)
        self.target_table.clearContents()
        self.target_table.setRowCount(len(cells))
        status_text = {
            TargetSupportStatus.SUPPORTED: (
                self.tr("Meets your quantity rule")
                if minimum_structure_count is not None
                else self.tr("Exact samples available")
            ),
            TargetSupportStatus.THIN: self.tr("Below your quantity rule"),
            TargetSupportStatus.NO_SAMPLE: self.tr("No exact composition sample"),
            TargetSupportStatus.NO_CONFIG_TYPE: self.tr("No matching structure family"),
            TargetSupportStatus.UNJUDGEABLE: self.tr(
                "Metadata incomplete; cannot fully evaluate"
            ),
        }
        for row, cell in enumerate(cells):
            self.target_table.setItem(row, 0, QTableWidgetItem(f"{cell.target_fraction:.2%}"))
            self.target_table.setItem(
                row, 1, QTableWidgetItem(cell.config_type or self.tr("All structures"))
            )
            status_item = QTableWidgetItem(status_text[cell.status])
            if cell.missing_config_type_count:
                status_item.setToolTip(
                    self.tr(
                        "{count:,} structures at this composition have no usable config_type."
                    ).format(count=cell.missing_config_type_count)
                )
            self.target_table.setItem(row, 2, status_item)
            self.target_table.setItem(row, 3, QTableWidgetItem(f"{cell.observed_count:,}"))
            nearest = "—" if cell.nearest_fraction is None else f"{cell.nearest_fraction:.2%}"
            self.target_table.setItem(row, 4, QTableWidgetItem(nearest))
            action = {
                TargetSupportStatus.SUPPORTED: self.tr("View structures"),
                TargetSupportStatus.THIN: self.tr("Review sources before deciding"),
                TargetSupportStatus.NO_SAMPLE: self.tr("Review nearby compositions"),
                TargetSupportStatus.NO_CONFIG_TYPE: self.tr(
                    "Review structure-family plan"
                ),
                TargetSupportStatus.UNJUDGEABLE: self.tr("Inspect missing metadata"),
            }[cell.status]
            action_item = QTableWidgetItem(action)
            action_item.setData(Qt.ItemDataRole.UserRole, cell.structure_indices)
            self.target_table.setItem(row, 5, action_item)
        if cells:
            self.target_table.selectRow(0)
        self._update_summary()

    def _on_target_selection_changed(self) -> None:
        row = self.target_table.currentRow()
        item = self.target_table.item(row, 5) if row >= 0 else None
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else ()
        indices = value if isinstance(value, tuple) else ()
        self._selected_target_indices = list(indices)
        self.target_selection_label.setText(
            self.tr("Selected target point: {count:,} structures").format(
                count=len(indices)
            )
        )
        self.target_show_button.setEnabled(bool(indices))

    def _update_model_scope_summary(self) -> None:
        local_overview = (
            self._result.overview_metrics.get("local_chemistry", {})
            if self._result is not None
            else {}
        )
        declared = tuple(
            str(element)
            for element in (
                local_overview.get("declared_model_elements", ())
                if isinstance(local_overview, Mapping)
                else ()
            )
        )
        analyzed = tuple(
            str(element)
            for element in (
                local_overview.get("analyzed_model_elements", ())
                if isinstance(local_overview, Mapping)
                else ()
            )
        )
        absent = tuple(
            str(element)
            for element in (
                local_overview.get("absent_model_elements", ())
                if isinstance(local_overview, Mapping)
                else ()
            )
        )
        if not declared:
            self.model_empty_label.setText(
                self.tr(
                    "No independent model evidence is attached. There are no reference and prediction values "
                    "that have passed structure mapping and unit checks. Show NEP predictions on the current "
                    "training data may be used for error browsing, but are not automatically treated as "
                    "independent model validation evidence."
                )
            )
            self.model_empty_label.setToolTip("")
            return
        present_text = " · ".join(analyzed) or "—"
        if absent:
            self.model_empty_label.setText(
                self.tr(
                    "The model declares {declared} elements; this dataset contains {present}: {elements}. "
                    "The other {absent} model elements are absent, so their compositions and local environments "
                    "cannot be audited here. Neighbor analysis computes only present elements. This is informational "
                    "and may be intentional for a subsystem or universal model. Independent reference and prediction "
                    "evidence has not been attached."
                ).format(
                    declared=len(declared),
                    present=len(analyzed),
                    elements=present_text,
                    absent=len(absent),
                )
            )
            self.model_empty_label.setToolTip(
                self.tr("Absent model elements: {elements}").format(
                    elements=", ".join(absent)
                )
            )
            return
        self.model_empty_label.setText(
            self.tr(
                "All {count} model-declared elements occur in this dataset: {elements}. "
                "Independent reference and prediction evidence has not been attached."
            ).format(count=len(declared), elements=present_text)
        )
        self.model_empty_label.setToolTip("")

    def set_result(self, result: AuditResult) -> None:
        render_started = perf_counter()
        render_timings_ms: dict[str, float] = {}
        stage_started = perf_counter()
        incoming_state_key = self._dataset_state_key(result)
        if (
            self._target_dataset_fingerprint
            and self._target_dataset_fingerprint != incoming_state_key
        ):
            self._target_configured = False
            self.target_table.setRowCount(0)
            self.target_chart.clear()
        self._target_dataset_fingerprint = incoming_state_key
        self._result = result
        self._all_slices = list(result.slices)
        self._dimensions = {dimension.id: dimension for dimension in result.dimensions}
        self._all_topics = self._build_topics()
        self._topics = self._build_review_topics(self._all_topics)
        render_timings_ms["topic_prepare"] = (perf_counter() - stage_started) * 1000.0
        self._selected_chart_indices = []
        self._selected_composition_indices = []
        self.no_dataset_panel.hide()
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
        backend_timing = result.overview_metrics.get("timings_ms", {})
        if isinstance(backend_timing, Mapping):
            backend_total = float(backend_timing.get("total", 0.0) or 0.0)
            if backend_total > 0.0:
                run_meta.append(
                    self.tr("Audit {seconds} s").format(seconds=f"{backend_total / 1000.0:.2f}")
                )
        self.generated_at_label.setText(" · ".join(run_meta))

        stage_started = perf_counter()
        self._update_label_availability()
        self._update_summary()
        self._populate_slice_table()
        self._populate_inventory_views()
        self._update_model_scope_summary()
        self._update_review_summary()
        render_timings_ms["dashboard_widgets"] = (perf_counter() - stage_started) * 1000.0

        stage_started = perf_counter()
        self.dimension_list.blockSignals(True)
        self.dimension_list.clear()
        overview_title = self.tr("Overview")
        overview = QListWidgetItem(
            f"{overview_title}\n{self._topic_count_text(len(self._topics))}"
        )
        overview.setData(Qt.ItemDataRole.UserRole, _OVERVIEW)
        self.dimension_list.addItem(overview)
        self._sync_phase_dimension_item()
        self._sync_magnetic_dimension_item()
        if result.dimensions:
            for dimension in result.dimensions:
                status = self._status_text(dimension.status)
                topic_count = sum(
                    topic.dimension_id == dimension.id for topic in self._all_topics
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
                    for topic in self._all_topics
                )
                item = QListWidgetItem(
                    f"{self._dimension_title(audit_slice.dimension_id)}\n"
                    f"{self._topic_count_text(topic_count)}"
                )
                item.setData(Qt.ItemDataRole.UserRole, audit_slice.dimension_id)
                self.dimension_list.addItem(item)
        self.dimension_list.blockSignals(False)
        self.dimension_list.setCurrentRow(0)
        render_timings_ms["dimension_list"] = (perf_counter() - stage_started) * 1000.0
        render_timings_ms["total"] = (perf_counter() - render_started) * 1000.0
        self._last_render_timings_ms = {
            key: round(value, 3) for key, value in render_timings_ms.items()
        }

        timing_lines: list[str] = []
        if isinstance(backend_timing, Mapping):
            timing_lines.append(
                self.tr("Backend total: {milliseconds} ms").format(
                    milliseconds=f"{float(backend_timing.get('total', 0.0) or 0.0):.1f}"
                )
            )
            stages = backend_timing.get("stages", {})
            if isinstance(stages, Mapping):
                timing_lines.extend(
                    f"{name}: {float(milliseconds):.1f} ms"
                    for name, milliseconds in sorted(
                        stages.items(), key=lambda item: float(item[1]), reverse=True
                    )
                )
        for section_name in ("data_quality", "local_chemistry"):
            section = result.overview_metrics.get(section_name, {})
            section_timing = section.get("timings_ms", {}) if isinstance(section, Mapping) else {}
            section_stages = (
                section_timing.get("stages", {})
                if isinstance(section_timing, Mapping)
                else {}
            )
            if isinstance(section_stages, Mapping) and section_stages:
                timing_lines.append(f"[{section_name}]")
                timing_lines.extend(
                    f"  {name}: {float(milliseconds):.1f} ms"
                    for name, milliseconds in sorted(
                        section_stages.items(),
                        key=lambda item: float(item[1]),
                        reverse=True,
                    )
                )
        timing_lines.append(
            self.tr("UI render: {milliseconds} ms").format(
                milliseconds=f"{render_timings_ms['total']:.1f}"
            )
        )
        self.generated_at_label.setToolTip("\n".join(timing_lines))
        logger.debug(
            "Training Set Audit UI timing: total={total:.1f} ms | {stages}",
            total=render_timings_ms["total"],
            stages=" | ".join(
                f"{key}={value:.1f} ms"
                for key, value in sorted(
                    (
                        (key, value)
                        for key, value in render_timings_ms.items()
                        if key != "total"
                    ),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ),
        )

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

    def _build_review_topics(
        self, topics: list[_AuditTopic]
    ) -> list[_AuditTopic]:
        """Return only actionable work, expanding repeated geometry per group."""
        actionable = [
            topic for topic in topics if topic.category in {"blocker", "review"}
        ]
        duplicate_topic = next(
            (
                topic
                for topic in actionable
                if topic.id == "data_quality:exact_duplicates"
            ),
            None,
        )
        if duplicate_topic is None or self._result is None:
            return actionable
        data_quality = self._result.overview_metrics.get("data_quality", {})
        duplicate_groups = (
            data_quality.get("duplicate_groups", ())
            if isinstance(data_quality, Mapping)
            else ()
        )
        expanded: list[_AuditTopic] = []
        for topic in actionable:
            if topic is not duplicate_topic:
                expanded.append(topic)
                continue
            for group_number, group in enumerate(duplicate_groups, start=1):
                indices = tuple(sorted(int(index) for index in group))
                expanded.append(
                    replace(
                        topic,
                        id=f"{topic.id}:group:{group_number}",
                        title=self.tr("Repeated geometry group {group}").format(
                            group=group_number
                        ),
                        structure_indices=indices,
                        observed=self.tr(
                            "This repeated-geometry group contains {count} structures."
                        ).format(count=len(indices)),
                    )
                )
        return expanded

    def _review_state_key(self, topic: _AuditTopic) -> tuple[str, str]:
        return self._dataset_state_key(self._result), topic.id

    def _review_state_options(
        self, topic: _AuditTopic
    ) -> tuple[tuple[str, str], ...]:
        if topic.id.startswith("data_quality:exact_duplicates:group:"):
            return (
                (self.tr("Pending"), "pending"),
                (self.tr("Intentionally retained"), "keep"),
                (self.tr("Isolation candidate"), "isolate"),
                (self.tr("Recalculation needed"), "recalculate"),
            )
        if topic.category == "blocker":
            return (
                (self.tr("Unresolved"), "pending"),
                (self.tr("Resolved and rechecked"), "resolved"),
            )
        if topic.id == "data_quality:label_conflicts":
            return (
                (self.tr("Pending"), "pending"),
                (self.tr("Trusted source identified"), "trusted_source"),
                (self.tr("Recalculation needed"), "recalculate"),
            )
        return (
            (self.tr("Pending"), "pending"),
            (self.tr("Physically reasonable"), "keep"),
            (self.tr("Inspect geometry"), "inspect_geometry"),
        )

    def _sync_review_state_selector(self, topic: _AuditTopic) -> None:
        current_state = self._review_states.get(
            self._review_state_key(topic), "pending"
        )
        self.review_state_selector.blockSignals(True)
        self.review_state_selector.clear()
        for label, value in self._review_state_options(topic):
            self.review_state_selector.addItem(label, userData=value)
        index = self.review_state_selector.findData(current_state)
        self.review_state_selector.setCurrentIndex(max(0, index))
        self.review_state_selector.blockSignals(False)

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
            environment_count = int(plot.get("environment_count", 0) or 0)
            fraction = 0.0 if environment_count <= 0 else thin_count / environment_count
            evidence_lines.append(
                self.tr(
                    "{metric}: {ranges}; {thin} of {total} environments ({fraction})."
                ).format(
                    metric=self._local_metric_label(plot),
                    ranges=self._compact_bin_labels(selected_labels),
                    thin=thin_count,
                    total=environment_count,
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
        co_occurring = int(
            self._metric_value(audit_slice, "co_occurring_structures") or 0
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
                "{co_occurring} co-occurring structures."
            ).format(
                contacts=contacts,
                contact_structures=contact_structures,
                co_occurring=co_occurring,
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
        if hasattr(self, "cooccurrence_table"):
            QTimer.singleShot(0, self._resize_overview_matrix)

    def _update_responsive_columns(self, width: int) -> None:
        self.slice_table.setColumnHidden(3, width < _DIMENSION_COLUMN_MIN_WIDTH)
        self.slice_table.setColumnHidden(2, width < 650)

    def _resize_overview_matrix(self) -> None:
        """Fill the matrix viewport while retaining usable cells for large systems."""
        table = self.cooccurrence_table
        count = table.columnCount()
        if count <= 0 or table.rowCount() != count:
            return

        available_width = max(1, table.viewport().width())
        available_height = max(1, table.viewport().height())
        minimum_cell = 18
        fits_width = available_width >= count * minimum_cell
        fits_height = available_height >= count * minimum_cell
        table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            if fits_width
            else Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            if fits_height
            else Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        column_base, column_extra = divmod(available_width, count)
        row_base, row_extra = divmod(available_height, count)
        column_base = max(minimum_cell, column_base)
        row_base = max(minimum_cell, row_base)
        table.setUpdatesEnabled(False)
        for index in range(count):
            table.setColumnWidth(
                index,
                column_base + (1 if fits_width and index < column_extra else 0),
            )
            table.setRowHeight(
                index,
                row_base + (1 if fits_height and index < row_extra else 0),
            )
        table.setUpdatesEnabled(True)

    def _update_dataset_facts(self, inventory: DatasetInventory | None) -> None:
        if inventory is None:
            for label in (
                self.fact_total_atoms_value,
                self.fact_atom_range_value,
                self.fact_atom_center_value,
            ):
                label.setText("—")
            return

        atom_counts = sorted(
            (int(atom_count), int(structure_count))
            for atom_count, structure_count in inventory.atom_counts
            if int(structure_count) > 0
        )
        structure_count = sum(count for _, count in atom_counts)
        total_atoms = sum(atom_count * count for atom_count, count in atom_counts)
        self.fact_total_atoms_value.setText(f"{total_atoms:,}")
        if not atom_counts or structure_count <= 0:
            self.fact_atom_range_value.setText("—")
            self.fact_atom_center_value.setText("—")
        else:
            minimum = atom_counts[0][0]
            maximum = atom_counts[-1][0]
            self.fact_atom_range_value.setText(
                f"{minimum:,}" if minimum == maximum else f"{minimum:,}–{maximum:,}"
            )
            mean = total_atoms / structure_count
            middle_positions = ((structure_count - 1) // 2, structure_count // 2)
            middle_values: list[int] = []
            cumulative = 0
            for atom_count, count in atom_counts:
                next_cumulative = cumulative + count
                for position in middle_positions[len(middle_values):]:
                    if position < next_cumulative:
                        middle_values.append(atom_count)
                cumulative = next_cumulative
                if len(middle_values) == 2:
                    break
            median = sum(middle_values) / len(middle_values)
            self.fact_atom_center_value.setText(f"{mean:.1f} / {median:.1f}")

    def _update_summary(self) -> None:
        total = self._structure_count()
        inventory = self._inventory()
        self.metric_structure_value.setText(f"{total:,}")
        self.metric_findings_value.setText(
            str(len(inventory.composition_points)) if inventory is not None else "—"
        )
        self.metric_dimension_value.setText(
            " · ".join(inventory.elements) if inventory is not None else "—"
        )
        self.metric_dimension_value.setToolTip(
            " · ".join(inventory.elements) if inventory is not None else ""
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
        self.metric_context_value.setToolTip(coverage)
        self._update_dataset_facts(inventory)
        self._update_element_overview(inventory)

    @staticmethod
    def _pair_key(first: str, second: str) -> tuple[str, str]:
        return tuple(sorted((first, second)))

    @staticmethod
    def _overview_heat_color(value: int, maximum: int) -> QColor:
        if value <= 0 or maximum <= 0:
            return QColor("#f8fafc")
        strength = log1p(value) / log1p(maximum)
        palette_index = min(
            len(_OVERVIEW_JET_COLORS) - 1,
            round(strength * (len(_OVERVIEW_JET_COLORS) - 1)),
        )
        return QColor(_OVERVIEW_JET_COLORS[palette_index])

    @staticmethod
    def _element_set_summaries(
        inventory: DatasetInventory,
    ) -> tuple[_ElementSetSummary, ...]:
        grouped_counts: dict[tuple[str, ...], int] = {}
        grouped_indices: dict[tuple[str, ...], set[int]] = {}
        for point in inventory.composition_points:
            element_set = tuple(
                element
                for element, count in zip(inventory.elements, point.reduced_counts)
                if int(count) > 0
            )
            if not element_set:
                continue
            grouped_counts[element_set] = (
                grouped_counts.get(element_set, 0) + int(point.structure_count)
            )
            grouped_indices.setdefault(element_set, set()).update(
                int(index) for index in point.structure_indices
            )
        return tuple(
            sorted(
                (
                    _ElementSetSummary(
                        elements=elements,
                        structure_count=grouped_counts[elements],
                        structure_indices=tuple(sorted(grouped_indices[elements])),
                    )
                    for elements in grouped_counts
                ),
                key=lambda item: (-item.structure_count, item.elements),
            )
        )

    def _update_element_overview(
        self, inventory: DatasetInventory | None
    ) -> None:
        if inventory is None:
            self._overview_element_sets = ()
            self._overview_elements = ()
            self._overview_structure_count = 0
            self._overview_element_counts = {}
            self._overview_pair_counts = {}
            self._overview_exact_pair_counts = {}
            self._overview_pure_counts = {}
            self._overview_basis_elements = ()
            self._selected_overview_elements = ()
            self._selected_overview_cell = None
            self._selected_overview_mode = ""
            self.cooccurrence_table.setRowCount(0)
            self.cooccurrence_table.setColumnCount(0)
            self.element_sets_table.setRowCount(0)
            self.element_sets_summary_label.setText(
                self.tr("No exact element-set inventory is available.")
            )
            self.pair_coverage_label.setText("—")
            self._update_overview_selection_status()
            for value in self.order_summary_values.values():
                value.setText("—")
            return

        self._overview_element_sets = self._element_set_summaries(inventory)
        self._overview_elements = tuple(inventory.elements)
        self._overview_structure_count = int(inventory.structure_count)
        self._overview_basis_elements = ()
        self._selected_overview_elements = ()
        self._selected_overview_cell = None
        self._selected_overview_mode = ""
        self._selected_overview_indices = []
        self._update_overview_selection_status()
        self._overview_element_counts = {
            element: sum(
                item.structure_count
                for item in self._overview_element_sets
                if element in item.elements
            )
            for element in self._overview_elements
        }
        self._overview_pure_counts = {
            item.elements[0]: item.structure_count
            for item in self._overview_element_sets
            if len(item.elements) == 1
        }
        pair_counts: dict[tuple[str, str], int] = {}
        exact_pair_counts: dict[tuple[str, str], int] = {}
        for row, first in enumerate(self._overview_elements):
            for second in self._overview_elements[row + 1 :]:
                key = self._pair_key(first, second)
                pair_counts[key] = sum(
                    item.structure_count
                    for item in self._overview_element_sets
                    if first in item.elements and second in item.elements
                )
                exact_pair_counts[key] = sum(
                    item.structure_count
                    for item in self._overview_element_sets
                    if len(item.elements) == 2
                    and set(item.elements) == set(key)
                )
        self._overview_pair_counts = pair_counts
        self._overview_exact_pair_counts = exact_pair_counts

        order_counts = {"1": 0, "2": 0, "3": 0, "4+": 0}
        for item in self._overview_element_sets:
            key = "4+" if len(item.elements) >= 4 else str(len(item.elements))
            order_counts[key] += item.structure_count
        for key, value in self.order_summary_values.items():
            count = order_counts[key]
            share = (
                0.0
                if self._overview_structure_count <= 0
                else count / self._overview_structure_count
            )
            value.setText(
                self.tr("{share:.1%} · {count:,}").format(
                    share=share,
                    count=count,
                )
            )

        possible_pairs = len(pair_counts)
        covered_pairs = sum(count > 0 for count in pair_counts.values())
        exact_pairs = sum(count > 0 for count in exact_pair_counts.values())
        self.pair_coverage_label.setText(
            self.tr(
                "{covered}/{possible} pairs co-occur · "
                "{exact}/{possible} have exact binary structures"
            ).format(
                covered=covered_pairs,
                exact=exact_pairs,
                possible=possible_pairs,
            )
        )
        self._populate_overview_matrix()
        self._populate_overview_element_sets()

    def _populate_overview_matrix(self) -> None:
        elements = self._overview_elements
        table = self.cooccurrence_table
        table.blockSignals(True)
        table.clear()
        table.clearSelection()
        table.setRowCount(len(elements))
        table.setColumnCount(len(elements))
        table.setHorizontalHeaderLabels(elements)
        table.setVerticalHeaderLabels(elements)
        max_element_count = max(self._overview_element_counts.values(), default=1)
        max_pair_count = max(self._overview_pair_counts.values(), default=1)
        selected = set(self._selected_overview_elements)
        basis = set(self._overview_basis_elements)
        basis_count = (
            self._overview_pair_counts.get(
                self._pair_key(*self._overview_basis_elements),
                0,
            )
            if len(self._overview_basis_elements) == 2
            else 0
        )
        related_element_counts: dict[str, int] = {}
        related_pair_counts: dict[tuple[str, str], int] = {}
        if basis:
            for summary in self._overview_element_sets:
                if not basis.issubset(summary.elements):
                    continue
                extras = sorted(set(summary.elements).difference(basis))
                for element in extras:
                    related_element_counts[element] = (
                        related_element_counts.get(element, 0)
                        + summary.structure_count
                    )
                for first, second in combinations(extras, 2):
                    pair = self._pair_key(first, second)
                    related_pair_counts[pair] = (
                        related_pair_counts.get(pair, 0)
                        + summary.structure_count
                    )
        max_related_element_count = max(related_element_counts.values(), default=1)
        max_related_pair_count = max(related_pair_counts.values(), default=1)

        for index, element in enumerate(elements):
            horizontal = table.horizontalHeaderItem(index)
            vertical = table.verticalHeaderItem(index)
            if element in selected:
                horizontal.setForeground(QColor("#c2410c"))
                vertical.setForeground(QColor("#c2410c"))
            horizontal.setToolTip(element)
            vertical.setToolTip(element)

        for row, first in enumerate(elements):
            for column, second in enumerate(elements):
                diagonal = row == column
                upper = column > row
                pair = self._pair_key(first, second)
                visible = True
                if diagonal:
                    if basis:
                        if first in basis:
                            value = 0
                            maximum = 1
                            cell_elements = ()
                            tooltip = self.tr(
                                "{element} is part of the selected basis {basis}."
                            ).format(
                                element=first,
                                basis=" + ".join(self._overview_basis_elements),
                            )
                        else:
                            value = related_element_counts.get(first, 0)
                            maximum = max_related_element_count
                            cell_elements = tuple(sorted((*basis, first)))
                            tooltip = self.tr(
                                "{basis} + {element}: {count:,} structures "
                                "({share:.1%} of the selected pair)."
                            ).format(
                                basis=" + ".join(self._overview_basis_elements),
                                element=first,
                                count=value,
                                share=(
                                    0.0
                                    if basis_count <= 0
                                    else value / basis_count
                                ),
                            )
                    else:
                        value = self._overview_element_counts.get(first, 0)
                        maximum = max_element_count
                        pure = self._overview_pure_counts.get(first, 0)
                        tooltip = self.tr(
                            "{element}: {count:,} related structures ({share:.1%}); "
                            "pure-element structures: {pure:,}."
                        ).format(
                            element=first,
                            count=value,
                            share=(
                                0.0
                                if self._overview_structure_count <= 0
                                else value / self._overview_structure_count
                            ),
                            pure=pure,
                        )
                        cell_elements = (first,)
                elif upper:
                    value = self._overview_pair_counts.get(pair, 0)
                    maximum = max_pair_count
                    tooltip = self.tr(
                        "{first} + {second}: {count:,} co-occurring structures "
                        "across binary and higher-order sets."
                    ).format(first=first, second=second, count=value)
                    cell_elements = pair
                    if basis and set(pair) != basis:
                        visible = False
                    elif (
                        not basis
                        and self._selected_overview_mode == "element"
                        and not selected.intersection(pair)
                    ):
                        visible = False
                else:
                    if basis and not basis.intersection(pair):
                        value = related_pair_counts.get(pair, 0)
                        maximum = max_related_pair_count
                        cell_elements = tuple(sorted((*basis, *pair)))
                        tooltip = self.tr(
                            "{basis} + {first} + {second}: {count:,} structures "
                            "({share:.1%} of the selected pair)."
                        ).format(
                            basis=" + ".join(self._overview_basis_elements),
                            first=first,
                            second=second,
                            count=value,
                            share=(
                                0.0
                                if basis_count <= 0
                                else value / basis_count
                            ),
                        )
                    else:
                        value = 0
                        maximum = 1
                        cell_elements = ()
                        tooltip = (
                            self.tr(
                                "Select an upper-triangle element pair to show "
                                "its related third and fourth elements."
                            )
                            if not basis
                            else self.tr(
                                "This cell repeats a selected basis element."
                            )
                        )

                if value <= 0 or not visible:
                    cell_elements = ()
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setData(Qt.ItemDataRole.UserRole, cell_elements)
                item.setToolTip(tooltip)
                item.setBackground(
                    self._overview_heat_color(value, maximum)
                    if cell_elements
                    else QColor(Qt.GlobalColor.transparent)
                )
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | (
                        Qt.ItemFlag.ItemIsSelectable
                        if cell_elements and value > 0
                        else Qt.ItemFlag.NoItemFlags
                    )
                )
                if (
                    diagonal
                    and not basis
                    and self._overview_pure_counts.get(first, 0) <= 0
                ):
                    item.setText("—")
                    item.setForeground(QColor("#64748b"))
                table.setItem(row, column, item)

        for index in range(len(elements)):
            table.setColumnWidth(index, 26)
            table.setRowHeight(index, 26)
        if self._selected_overview_cell is not None:
            selected_row, selected_column = self._selected_overview_cell
            if (
                0 <= selected_row < len(elements)
                and 0 <= selected_column < len(elements)
            ):
                table.setCurrentCell(selected_row, selected_column)
                selected_item = table.item(selected_row, selected_column)
                if selected_item is not None:
                    selected_item.setText("✓")
                    background = selected_item.background().color()
                    luminance = (
                        0.2126 * background.red()
                        + 0.7152 * background.green()
                        + 0.0722 * background.blue()
                    )
                    selected_item.setForeground(
                        QColor("#ffffff" if luminance < 145 else "#111827")
                    )
                    font = selected_item.font()
                    font.setBold(True)
                    selected_item.setFont(font)
                    selected_item.setSelected(True)
        table.blockSignals(False)
        QTimer.singleShot(0, self._resize_overview_matrix)

    def _on_overview_matrix_cell_clicked(self, row: int, column: int) -> None:
        item = self.cooccurrence_table.item(row, column)
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else ()
        elements = tuple(value) if isinstance(value, tuple) else ()
        if not elements:
            return
        if self._selected_overview_cell == (row, column):
            self._clear_overview_element_filter()
            return
        self._selected_overview_elements = elements
        self._selected_overview_cell = (row, column)
        if column > row:
            self._overview_basis_elements = elements
            self._selected_overview_mode = "cooccurrence"
        elif self._overview_basis_elements:
            self._selected_overview_mode = "conditional"
        else:
            self._selected_overview_mode = "element"
        self._selected_overview_indices = []
        self.view_element_set_button.setEnabled(False)
        self.clear_element_filter_button.show()
        self._update_overview_selection_status()
        self._populate_overview_matrix()
        self._populate_overview_element_sets()

    def _clear_overview_element_filter(self) -> None:
        self._overview_basis_elements = ()
        self._selected_overview_elements = ()
        self._selected_overview_cell = None
        self._selected_overview_mode = ""
        self._selected_overview_indices = []
        self.clear_element_filter_button.hide()
        self.view_element_set_button.setEnabled(False)
        self._update_overview_selection_status()
        self._populate_overview_matrix()
        self._populate_overview_element_sets()

    def _update_overview_selection_status(self) -> None:
        elements = " + ".join(self._selected_overview_elements)
        if self._selected_overview_mode == "conditional":
            text = self.tr(
                "Selected: {elements} · based on {basis}"
            ).format(
                elements=elements,
                basis=" + ".join(self._overview_basis_elements),
            )
        elif self._selected_overview_mode == "cooccurrence":
            text = self.tr(
                "Basis: {elements} · related-element view"
            ).format(elements=elements)
        elif self._selected_overview_mode == "element":
            text = self.tr(
                "Selected element: {elements} · element presence"
            ).format(
                elements=elements,
            )
        else:
            text = self.tr("Filter: none · Click a cell")
        self.matrix_selection_label.setText(text)
        self.matrix_selection_label.setProperty(
            "active", bool(elements and self._selected_overview_mode)
        )
        self.matrix_selection_label.style().unpolish(
            self.matrix_selection_label
        )
        self.matrix_selection_label.style().polish(
            self.matrix_selection_label
        )

    def _populate_overview_element_sets(self) -> None:
        selected = set(self._selected_overview_elements)
        items = [
            item
            for item in self._overview_element_sets
            if selected.issubset(item.elements)
        ]
        total = sum(item.structure_count for item in items)
        if not selected:
            self.element_sets_summary_label.setText(
                self.tr(
                    "{count} exact element sets, sorted by structure count."
                ).format(count=len(items))
            )
        elif self._selected_overview_mode == "element":
            element = self._selected_overview_elements[0]
            self.element_sets_summary_label.setText(
                self.tr(
                    "{count} sets · {structures:,} related structures · "
                    "{pure:,} pure-element structures"
                ).format(
                    count=len(items),
                    structures=total,
                    pure=self._overview_pure_counts.get(element, 0),
                )
            )
        elif self._selected_overview_mode == "conditional":
            self.element_sets_summary_label.setText(
                self.tr(
                    "{count} sets · {structures:,} structures · based on {basis}"
                ).format(
                    count=len(items),
                    structures=total,
                    basis=" + ".join(self._overview_basis_elements),
                )
            )
        else:
            pair = self._pair_key(*self._selected_overview_elements)
            self.element_sets_summary_label.setText(
                self.tr(
                    "{count} sets · {structures:,} co-occurring structures · "
                    "{binary:,} exact binary structures"
                ).format(
                    count=len(items),
                    structures=total,
                    binary=self._overview_exact_pair_counts.get(pair, 0),
                )
            )

        table = self.element_sets_table
        table.blockSignals(True)
        table.setUpdatesEnabled(False)
        table.setRowCount(len(items))
        count_texts: list[str] = []
        share_texts: list[str] = []
        for row, item in enumerate(items):
            formula = "–".join(item.elements)
            formula_item = QTableWidgetItem(formula)
            formula_item.setData(
                Qt.ItemDataRole.UserRole, item.structure_indices
            )
            formula_item.setToolTip(formula)
            count_text = f"{item.structure_count:,}"
            count_texts.append(count_text)
            count_item = QTableWidgetItem(count_text)
            share = (
                0.0
                if self._overview_structure_count <= 0
                else item.structure_count / self._overview_structure_count
            )
            share_text = f"{share:.2%}"
            share_texts.append(share_text)
            share_item = QTableWidgetItem(share_text)
            count_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            share_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            table.setItem(row, 0, formula_item)
            table.setItem(row, 1, count_item)
            table.setItem(row, 2, share_item)
        metrics = table.fontMetrics()
        count_header = table.horizontalHeaderItem(1).text()
        share_header = table.horizontalHeaderItem(2).text()
        table.setColumnWidth(
            1,
            max(
                [
                    metrics.horizontalAdvance(count_header),
                    *(metrics.horizontalAdvance(text) for text in count_texts),
                ]
            )
            + 24,
        )
        table.setColumnWidth(
            2,
            max(
                [
                    metrics.horizontalAdvance(share_header),
                    *(metrics.horizontalAdvance(text) for text in share_texts),
                ]
            )
            + 40,
        )
        table.clearSelection()
        table.setUpdatesEnabled(True)
        table.blockSignals(False)
        self._selected_overview_indices = []
        self.view_element_set_button.setEnabled(False)

    def _on_overview_element_set_selected(self) -> None:
        row = self.element_sets_table.currentRow()
        item = self.element_sets_table.item(row, 0) if row >= 0 else None
        indices = (
            item.data(Qt.ItemDataRole.UserRole)
            if item is not None
            else ()
        )
        self._selected_overview_indices = (
            [int(index) for index in indices]
            if isinstance(indices, (tuple, list))
            else []
        )
        count = len(self._selected_overview_indices)
        self.view_element_set_button.setEnabled(count > 0)
        self.view_element_set_button.setText(
            self.tr("View {count:,} structures").format(count=count)
            if count
            else self.tr("View selected structures")
        )

    def _update_analysis(self, dimension_id: str) -> None:
        self.plot_selector.blockSignals(True)
        self.plot_selector.clear()
        self._local_chemistry_plots = []
        self._set_local_chemistry_controls_visible(False)
        is_local_chemistry = dimension_id == "local_chemistry"
        is_phase_evidence = dimension_id == "phase_evidence"
        is_magnetic_evidence = dimension_id == "magnetic_evidence"
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
        elif is_phase_evidence:
            self._active_plots = self._phase_evidence_plots()
            status_text = self._phase_evidence_status_text()
        elif is_magnetic_evidence:
            self._active_plots = self._magnetic_evidence_plots()
            status_text = self._magnetic_evidence_status_text()
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
        if plot_id.startswith("pair_contacts:"):
            scope_key = plot_id.split(":", 1)[1]
            scope = (
                self.tr("Angular neighbors")
                if scope_key == "angular"
                else self.tr("Radial neighbors")
            )
            if text == f"{scope_key.title()} element-pair contact edges":
                return self.tr("{scope}: element-pair contact edges").format(
                    scope=scope
                )
            if text == scope_key:
                return scope

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
            "Directed NEP-cutoff contact edges": self.tr(
                "Directed NEP-cutoff contact edges"
            ),
            "Element pair": self.tr("Element pair"),
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
        self._set_fitted_button_text(
            self.chart_send_button,
            self.tr("Show {count} structures").format(count=count),
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
            state = self._review_states.get(
                self._review_state_key(topic), "pending"
            )
            state_labels = dict(
                (value, label) for label, value in self._review_state_options(topic)
            )
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
        self._set_fitted_button_text(
            self.send_button,
            self.tr("Show {count} structures in Dataset Display").format(count=0),
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
        self._set_fitted_button_text(
            self.send_button,
            self.tr("Show {count} structures in Dataset Display").format(count=count),
        )
        self.send_button.setEnabled(bool(topic.structure_indices))
        self.view_distribution_button.setEnabled(bool(topic.plot_id))
        self._sync_review_state_selector(topic)

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
            "phase_evidence": self.tr("Phases and local structure"),
            "magnetic_evidence": self.tr("Magnetic order"),
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

    def _phase_evidence_status_text(self) -> str:
        if self._result is None:
            return self.tr("No audit result is loaded.")
        phase_inventory = self._result.phase_inventory
        if phase_inventory is None:
            phase_meta = self._result.overview_metrics.get("phase_inventory", {})
            if isinstance(phase_meta, Mapping) and phase_meta.get("status") == "pending":
                return self.tr(
                    "Analyzing every structure in the audited scope. Results will appear automatically."
                )
            return self.tr("Phase evidence is unavailable for the current data.")
        return self.tr(
            "All {count:,} structures were analyzed. Structure labels summarize local geometry; "
            "they do not establish thermodynamic phase stability."
        ).format(count=phase_inventory.analyzed_structure_count)

    def _phase_evidence_plots(self) -> list[dict[str, Any]]:
        if self._result is None or self._result.phase_inventory is None:
            return []
        structures = tuple(
            structure
            for point in self._result.phase_inventory.composition_points
            for structure in point.structures
        )
        phase_order = PHASE_PARTITION_LABELS
        phase_groups = tuple(
            tuple(
                structure.source_index
                for structure in structures
                if phase_partition_label(structure) == label
            )
            for label in phase_order
        )
        phase_labels = tuple(
            label for label, indices in zip(phase_order, phase_groups) if indices
        )
        phase_groups = tuple(indices for indices in phase_groups if indices)
        confidence_order = ("strong", "mixed", "unresolved")
        confidence_groups = tuple(
            tuple(
                structure.source_index
                for structure in structures
                if structure.confidence_state == state
            )
            for state in confidence_order
        )
        confidence_labels = {
            "strong": self.tr("Strong evidence"),
            "mixed": self.tr("Mixed local structure"),
            "unresolved": self.tr("Unresolved"),
        }
        return [
            {
                "kind": "categorical_bars",
                "id": "phase_evidence:structure_labels",
                "title": self.tr("Structure-level phase labels"),
                "x_label": self.tr("Structures"),
                "y_label": self.tr("Phase label"),
                "series": (
                    {
                        "labels": tuple(
                            self._phase_display_name(label) for label in phase_labels
                        ),
                        "bar_ids": phase_labels,
                        "counts": tuple(len(indices) for indices in phase_groups),
                        "structure_indices": phase_groups,
                    },
                ),
            },
            {
                "kind": "categorical_bars",
                "id": "phase_evidence:confidence",
                "title": self.tr("Phase-evidence confidence"),
                "x_label": self.tr("Structures"),
                "y_label": self.tr("Evidence state"),
                "series": (
                    {
                        "labels": tuple(
                            confidence_labels[state] for state in confidence_order
                        ),
                        "bar_ids": confidence_order,
                        "counts": tuple(
                            len(indices) for indices in confidence_groups
                        ),
                        "structure_indices": confidence_groups,
                    },
                ),
            },
        ]

    def _magnetic_evidence_status_text(self) -> str:
        if self._result is None:
            return self.tr("No audit result is loaded.")
        inventory = self._result.magnetic_inventory
        if inventory is None:
            meta = self._result.overview_metrics.get("magnetic_inventory", {})
            if isinstance(meta, Mapping) and meta.get("status") == "pending":
                return self.tr("Analyzing every structure carrying spin:R:3.")
            return self.tr("Magnetic-order evidence is unavailable.")
        if inventory.analyzed_structure_count <= 0:
            return self.tr(
                "No valid per-atom spin:R:3 field was found. mforce and force_mag are not spin states."
            )
        return self.tr(
            "All {count:,} structures carrying spin:R:3 were analyzed; {missing:,} structures lack spin. "
            "Labels describe snapshot patterns, not thermodynamic magnetic stability."
        ).format(
            count=inventory.analyzed_structure_count,
            missing=inventory.missing_spin_count,
        )

    def _magnetic_evidence_plots(self) -> list[dict[str, Any]]:
        if (
            self._result is None
            or self._result.inventory is None
            or self._result.magnetic_inventory is None
        ):
            return []
        magnetic_structures = tuple(
            structure
            for point in self._result.magnetic_inventory.composition_points
            for structure in point.structures
        )
        magnetic_by_index = {
            structure.source_index: magnetic_partition_label(structure)
            for structure in magnetic_structures
        }
        all_indices = tuple(
            sorted(
                {
                    index
                    for point in self._result.inventory.composition_points
                    for index in point.structure_indices
                }
            )
        )
        magnetic_by_index.update(
            (index, "no_spin") for index in all_indices if index not in magnetic_by_index
        )

        def ordered_present(values, preferred):
            present = set(values)
            return tuple(label for label in preferred if label in present) + tuple(
                sorted(present.difference(preferred))
            )

        def partition_plot(
            *,
            plot_id: str,
            title: str,
            row_by_index: Mapping[int, str],
            segment_by_index: Mapping[int, str],
            row_order: tuple[str, ...],
            segment_order: tuple[str, ...],
            row_display,
            segment_display,
            y_label: str,
        ) -> dict[str, Any] | None:
            row_ids = ordered_present(row_by_index.values(), row_order)
            segment_ids = ordered_present(segment_by_index.values(), segment_order)
            if not row_ids or not segment_ids:
                return None
            series = []
            for segment in segment_ids:
                groups = tuple(
                    tuple(
                        index
                        for index in all_indices
                        if row_by_index.get(index) == row
                        and segment_by_index.get(index) == segment
                    )
                    for row in row_ids
                )
                if any(groups):
                    series.append(
                        {
                            "id": segment,
                            "label": segment_display(segment),
                            "counts": tuple(len(group) for group in groups),
                            "structure_indices": groups,
                        }
                    )
            if not series:
                return None
            return {
                "kind": "category_share_stacks",
                "id": plot_id,
                "title": title,
                "x_label": self.tr("Share of structure frames"),
                "y_label": y_label,
                "row_ids": row_ids,
                "row_labels": tuple(row_display(row) for row in row_ids),
                "series": tuple(series),
            }

        plots: list[dict[str, Any]] = []
        phase_inventory = self._result.phase_inventory
        if phase_inventory is not None:
            phase_by_index = {
                structure.source_index: phase_partition_label(structure)
                for point in phase_inventory.composition_points
                for structure in point.structures
            }
            phase_by_index.update(
                (index, "unresolved")
                for index in all_indices
                if index not in phase_by_index
            )
            phase_order = PHASE_PARTITION_LABELS
            phase_to_magnetic = partition_plot(
                plot_id="magnetic_evidence:phase_to_order",
                title=self.tr("Magnetic types inside each structural phase"),
                row_by_index=phase_by_index,
                segment_by_index=magnetic_by_index,
                row_order=phase_order,
                segment_order=MAGNETIC_PARTITION_LABELS,
                row_display=self._phase_display_name,
                segment_display=self._magnetic_display_name,
                y_label=self.tr("Structural phase"),
            )
            if phase_to_magnetic is not None:
                plots.append(phase_to_magnetic)
            magnetic_to_phase = partition_plot(
                plot_id="magnetic_evidence:order_to_phase",
                title=self.tr("Structural phases inside each magnetic type"),
                row_by_index=magnetic_by_index,
                segment_by_index=phase_by_index,
                row_order=MAGNETIC_PARTITION_LABELS,
                segment_order=phase_order,
                row_display=self._magnetic_display_name,
                segment_display=self._phase_display_name,
                y_label=self.tr("Magnetic type"),
            )
            if magnetic_to_phase is not None:
                plots.append(magnetic_to_phase)

        overall = partition_plot(
            plot_id="magnetic_evidence:overall",
            title=self.tr("Magnetic-type shares in the audited dataset"),
            row_by_index={index: "all" for index in all_indices},
            segment_by_index=magnetic_by_index,
            row_order=("all",),
            segment_order=MAGNETIC_PARTITION_LABELS,
            row_display=lambda _label: self.tr("All structures"),
            segment_display=self._magnetic_display_name,
            y_label=self.tr("Audited scope"),
        )
        if overall is not None:
            plots.append(overall)
        return plots

    def _sync_phase_dimension_item(self) -> None:
        if not hasattr(self, "dimension_list") or self._result is None:
            return
        item = next(
            (
                self.dimension_list.item(row)
                for row in range(self.dimension_list.count())
                if self.dimension_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "phase_evidence"
            ),
            None,
        )
        if item is None:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, "phase_evidence")
            self.dimension_list.insertItem(1, item)
        phase_inventory = self._result.phase_inventory
        phase_meta = self._result.overview_metrics.get("phase_inventory", {})
        status = phase_meta.get("status") if isinstance(phase_meta, Mapping) else None
        if phase_inventory is not None:
            detail = self.tr("Calculated · {count:,} structures").format(
                count=phase_inventory.analyzed_structure_count
            )
        elif status == "pending":
            detail = self.tr("Calculating all structures")
        else:
            detail = self.tr("Not calculated")
        item.setText(f"{self._dimension_title('phase_evidence')}\n{detail}")
        item.setToolTip(self._phase_evidence_status_text())

    def _sync_magnetic_dimension_item(self) -> None:
        if not hasattr(self, "dimension_list") or self._result is None:
            return
        if (
            self._result.magnetic_inventory is None
            and "magnetic_inventory" not in self._result.overview_metrics
        ):
            return
        item = next(
            (
                self.dimension_list.item(row)
                for row in range(self.dimension_list.count())
                if self.dimension_list.item(row).data(Qt.ItemDataRole.UserRole)
                == "magnetic_evidence"
            ),
            None,
        )
        if item is None:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, "magnetic_evidence")
            self.dimension_list.insertItem(2, item)
        inventory = self._result.magnetic_inventory
        meta = self._result.overview_metrics.get("magnetic_inventory", {})
        status = meta.get("status") if isinstance(meta, Mapping) else None
        if inventory is not None and inventory.analyzed_structure_count > 0:
            detail = self.tr("Calculated · {count:,} spin structures").format(
                count=inventory.analyzed_structure_count
            )
        elif status == "pending":
            detail = self.tr("Calculating all spin structures")
        elif status == "no-spin" or inventory is not None:
            detail = self.tr("No spin:R:3 data")
        else:
            detail = self.tr("Not calculated")
        item.setText(f"{self._dimension_title('magnetic_evidence')}\n{detail}")
        item.setToolTip(self._magnetic_evidence_status_text())

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
                color: #243135;
                font-size: 18px;
                font-weight: 600;
            }
            QFrame#auditNoDatasetPanel {
                min-width: 430px;
                max-width: 560px;
                background: #ffffff;
                border: 1px solid #dfe6e8;
                border-radius: 10px;
            }
            QLabel#auditNoDatasetHint {
                color: #657579;
                font-size: 13px;
            }
            QFrame#auditHeader,
            QFrame#auditDimensionRail,
            QFrame#auditMetricBand,
            QFrame#auditAnalysisPanel,
            QFrame#auditFindingsPanel,
            QFrame#auditEvidencePanel,
            QFrame#auditCooccurrencePanel,
            QFrame#auditElementSetsPanel,
            QFrame#auditCompositionHeader,
            QFrame#auditReviewBanner,
            QFrame#auditTargetDefinitionPanel,
            QFrame#auditModelPanel {
                background: #ffffff;
                border: 1px solid #d9e1e3;
                border-radius: 5px;
            }
            QFrame#auditReviewBanner {
                background: #eef8f6;
                border-color: #b9dcd7;
            }
            QFrame#overviewOrderCard {
                background: #f7f9fa;
                border: 1px solid #e1e7e9;
                border-radius: 4px;
            }
            QFrame#overviewHeatLegend {
                background: transparent;
                border: 1px solid #cbd5e1;
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
            QLabel#panelHint {
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
            QLabel#reviewSummary {
                color: #183b38;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#coverageBadge {
                color: #4338ca;
                background: #eef2ff;
                border: 1px solid #c7d2fe;
                border-radius: 9px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: 600;
            }
            QLabel#overviewOrderValue {
                color: #243135;
                font-size: 13px;
                font-weight: 600;
            }
            QLabel#overviewOrderLabel {
                color: #657579;
                font-size: 10px;
            }
            QLabel#matrixSelectionStatus {
                color: #526267;
                background: #f8fafc;
                border: 1px solid #dbe3e6;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QLabel#matrixSelectionStatus[active="true"] {
                color: #9a3412;
                background: #fff7ed;
                border-color: #fb923c;
                font-weight: 600;
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
                font-size: 14px;
                font-weight: 600;
            }
            QFrame#metricDivider {
                color: #d9e1e3;
                max-width: 1px;
            }
            QTableWidget#elementCooccurrenceTable {
                background: #ffffff;
                alternate-background-color: #ffffff;
                gridline-color: #ffffff;
                border: 1px solid #e1e7e9;
                border-radius: 4px;
                outline: 0;
            }
            QTableWidget#elementCooccurrenceTable::item {
                padding: 0;
                border: 1px solid #ffffff;
            }
            QTableWidget#elementCooccurrenceTable::item:selected {
                border: 3px solid #f97316;
            }
            QTableWidget#overviewElementSetsTable {
                background: #ffffff;
                alternate-background-color: #f7f9fa;
                border: 1px solid #e1e7e9;
                border-radius: 4px;
                outline: 0;
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
            QSplitter#auditCompositionSplitter::handle {
                background: #e5ebed;
                margin: 6px 3px;
                border-radius: 1px;
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
