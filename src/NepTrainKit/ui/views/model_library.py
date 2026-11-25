"""Model library: scorecard of captured models against benchmark suites."""

from __future__ import annotations


from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QMenu,
    QHBoxLayout,
    QHeaderView,
    QScrollArea,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    FlowLayout,
    FluentIcon,
    MessageBox,
    PrimarySplitPushButton,
    PushButton,
    SegmentedWidget,
    TableView,
    TableWidget,
    TransparentPushButton,
    TransparentToolButton,
)

from NepTrainKit import palette
from NepTrainKit.ui.widgets.validation_model_picker import GroupMarkerButton
from NepTrainKit.ui.widgets.validation_presentation import presentation_for, uses_accuracy_cards, uses_curve_chart
from NepTrainKit.ui.widgets.eos_plot import EOS_REFERENCE_WIDTH, update_residual_placeholder
from NepTrainKit.config import Config
from NepTrainKit.core import MessageManager
from NepTrainKit.core.validation import SetRejectedError, SetTooLargeError, ValidationService
from NepTrainKit.core.validation.protocols import get_protocol, protocol_for_set
from NepTrainKit.core.validation.service import ModelDraft, parse_model_header
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.threads import BackgroundTask
from NepTrainKit.ui.widgets import GetStrMessageBox
from NepTrainKit.ui.widgets.comparison_picker import ComparisonPicker
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog, save_model_to_project

CONFIG_SUITE = "suite"
VERDICT_COLORS = {
    "better": QColor(47, 158, 68, 70),
    "worse": QColor(224, 49, 49, 80),
    "same": QColor(134, 142, 150, 40),
}
SERIES_COLORS = (
    "#2F6FA8",
    "#F76707",
    "#2F9E44",
    "#AE3EC9",
    "#E03131",
    "#1098AD",
    "#A66B00",
    "#D63384",
    "#5F3DC4",
    "#607D8B",
)


def format_metric(value: float | None, better: str) -> str:
    if value is None:
        return ""
    return f"{value:+.2f}" if better == "closer-to-zero" else f"{value:.3g}"


class ScorecardTableModel(QAbstractTableModel):
    """Rows are models, columns are the headline metrics of each set in a suite."""

    CellRole = Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: list[dict] = []
        self._rows: list[dict] = []

    def set_scorecard(self, card: dict | None) -> None:
        self.beginResetModel()
        self._columns = list((card or {}).get("columns", []))
        self._rows = list((card or {}).get("rows", []))
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else 1 + len(self._columns)

    def row(self, row: int) -> dict | None:
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def column(self, column: int) -> dict | None:
        return self._columns[column - 1] if 1 <= column <= len(self._columns) else None

    def row_of_model(self, model_id: int) -> int:
        return next((i for i, row in enumerate(self._rows) if row["model_id"] == model_id), -1)

    def first_column_of_set(self, set_id: str) -> int:
        return next((i + 1 for i, column in enumerate(self._columns) if column["set_id"] == set_id), -1)

    def metric_label(self, column):
        labels = {
            "shape_rmse": self.tr("Curve RMSE (offset removed)"),
            "max_shape_error": self.tr("Max curve error"),
            "raw_rmse": self.tr("Raw RMSE"),
            "v0_error": self.tr("V₀ error"),
            "b0_error": self.tr("B₀ error"),
            "energy_rmse": self.tr("Energy RMSE"),
            "force_rmse": self.tr("Force RMSE"),
            "mforce_rmse": self.tr("Magnetic force RMSE"),
            "virial_rmse": self.tr("Virial RMSE"),
            "response_rmse": self.tr("Response RMSE"),
            "max_response_error": self.tr("Max response error"),
        }
        return labels.get(column.get("metric"), column["label"])

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        column = self.column(section)
        if role == Qt.ItemDataRole.DisplayRole:
            if column is None:
                return self.tr("Model version")
            unit = f" ({column['unit']})" if column["unit"] else ""
            label = self.metric_label(column)
            return f"{column['set_name']}\n{label}{unit}"
        if role == Qt.ItemDataRole.ToolTipRole and column is not None:
            return self.tr("{set} · {protocol}").format(set=column["set_name"], protocol=column["protocol"])
        return None

    def model_name(self, row, *, include_baseline=True):
        name = f"★ {row['name']}" if row["starred"] else row["name"]
        if row["model_id"] == 0:
            name = self.tr("{name} (unsaved)").format(name=name)
        elif row.get("temporary"):
            name = self.tr("{name} (temporary)").format(name=name)
        elif row.get("project_name"):
            name = f"{row['project_name']} / {name}"
        return self.tr("{name} (baseline model)").format(name=name) if include_baseline and row["is_baseline"] else name

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        row = self.row(index.row()) if index.isValid() else None
        if row is None:
            return None
        column = self.column(index.column())
        if column is None:
            if role == Qt.ItemDataRole.DisplayRole:
                return self.model_name(row)
            if role == Qt.ItemDataRole.FontRole and row["is_baseline"]:
                font = QFont()
                font.setBold(True)
                return font
            if role == Qt.ItemDataRole.ToolTipRole:
                if not row.get("complete", False):
                    return self.tr("Validation incomplete")
                if row["is_baseline"]:
                    return self.tr("Baseline model: other versions are compared with this model")
                if not row.get("comparison_complete", False):
                    return self.tr("Baseline model comparison incomplete")
                return self.tr("{count} regressions").format(count=row["regressions"])
            return None
        cell = row["cells"][index.column() - 1]
        if role == self.CellRole:
            return cell
        if role == Qt.ItemDataRole.DisplayRole:
            if cell["status"] == "succeeded":
                return format_metric(cell["value"], column["better"]) or self.tr("n/a")
            return {
                "missing": self.tr("not run"),
                "failed": self.tr("failed"),
                "unsupported": self.tr("unsupported"),
            }.get(cell["status"], cell["status"])
        if role == Qt.ItemDataRole.BackgroundRole:
            return VERDICT_COLORS.get(cell.get("verdict"))
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            if cell["error"]:
                # Stored diagnostics keep their raw text when no template matches.
                return translate_runtime_message(cell["error"])
            return {
                "better": self.tr("Better than the baseline model"),
                "worse": self.tr("Worse than the baseline model"),
                "same": self.tr("Within tolerance of the baseline model"),
            }.get(cell.get("verdict"))
        return None


class GroupMetricsTableModel(QAbstractTableModel):
    """One reference group per row, reusing scorecard values and verdicts."""

    def __init__(self, scorecard, parent=None):
        super().__init__(parent)
        self.scorecard = scorecard
        self.benchmarks = []
        self.model_row = -1
        self.metric_columns = []
        self.kind = "eos"
        self._metric_keys = ("shape_rmse", "v0_error", "b0_error")

    def set_groups(self, benchmarks, model_row, kind):
        self.beginResetModel()
        self.benchmarks = list(benchmarks)
        self.model_row = model_row
        self.kind = kind
        keys = ("energy_span", "check_failures") if kind == "diagnostic" else ("energy_rmse", "force_rmse")
        if presentation_for(kind).choose_table_metric:
            keys = tuple(metric.key for metric in get_protocol(kind).headline_metrics())
        if kind == "efv" and any(
            item.get("summary", {}).get("label_counts", {}).get("mforce", {}).get("components", 0)
            for item in self.benchmarks
        ):
            keys += ("mforce_rmse",)
        if kind == "response" and self.benchmarks:
            keys = tuple(metric.key for metric in protocol_for_set(self.benchmarks[0]).headline_metrics())
        self._metric_keys = keys
        self.metric_columns = [
            [
                next(
                    (
                        i + 1
                        for i, col in enumerate(self.scorecard._columns)
                        if col["set_id"] == benchmark["id"] and col["metric"] == key
                    ),
                    -1,
                )
                for key in keys
            ]
            for benchmark in self.benchmarks
        ]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.benchmarks)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._metric_keys) + 2

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal or role != Qt.ItemDataRole.DisplayRole:
            return None
        metrics = (
            [self.tr("Curve RMSE\n(meV/atom)"), self.tr("V₀ error (%)"), self.tr("B₀ error (%)")]
            if self.kind == "eos"
            else [self.tr("Energy RMSE\n(meV/atom)"), self.tr("Force RMSE\n(meV/Å)")]
        )
        if "mforce_rmse" in self._metric_keys:
            metrics.append(self.tr("Magnetic force RMSE\n(meV/μB)"))
        if self.kind == "response":
            protocol = protocol_for_set(self.benchmarks[0]) if self.benchmarks else get_protocol("response")
            return [
                self.tr("Reference group"),
                *[self.tr(metric.label) + "\n(" + metric.unit + ")" for metric in protocol.headline_metrics()],
                self.tr("Status"),
            ][section]
        if self.kind == "diagnostic":
            return ["诊断实验", "能量跨度 (meV/atom)", "检查异常数", "运行状态"][section]
        return [self.tr("Reference group"), *metrics, self.tr("Status")][section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.benchmarks):
            return None
        benchmark = self.benchmarks[index.row()]
        if index.column() == 0:
            if role == Qt.ItemDataRole.DisplayRole:
                name = presentation_for(benchmark["protocol"]).item_title(benchmark)
                return f"G{index.row() + 1} · {name}"
            if role == Qt.ItemDataRole.ToolTipRole:
                return benchmark["name"]
            return None
        columns = self.metric_columns[index.row()]
        if index.column() < self.columnCount() - 1:
            column = columns[index.column() - 1]
            if self.model_row < 0 or column < 0:
                return "—" if role == Qt.ItemDataRole.DisplayRole else None
            source_index = self.scorecard.index(self.model_row, column)
            cell = self.scorecard.data(source_index, ScorecardTableModel.CellRole)
            if role == Qt.ItemDataRole.DisplayRole and cell and cell["status"] != "succeeded":
                return "—"
            return self.scorecard.data(source_index, role)
        cell = (
            self.scorecard.data(self.scorecard.index(self.model_row, columns[0]), ScorecardTableModel.CellRole)
            if self.model_row >= 0 and columns[0] >= 0
            else None
        )
        warnings = benchmark.get("summary", {}).get("warnings", [])
        if role == Qt.ItemDataRole.ToolTipRole:
            return "\n".join(
                [translate_runtime_message(w) for w in warnings]
                + ([translate_runtime_message(cell["error"])] if cell and cell.get("error") else [])
            )
        if role == Qt.ItemDataRole.DisplayRole:
            status = {
                "missing": self.tr("Not run"),
                "succeeded": self.tr("Completed"),
                "failed": self.tr("Failed"),
                "unsupported": self.tr("Unsupported"),
            }.get((cell or {}).get("status"), self.tr("Reference only"))
            return status + (" · " + self.tr("Fit warning") if warnings else "")
        return None


def summarize_model_status(service, model_id: int, preferred_suite_id: int | None = None) -> dict | None:
    """Pick the suite to report for one model and count its regressions.

    The preferred suite wins when it holds sets the model covers; otherwise the
    first such suite by name. ``None`` means no suite applies to this model.
    """
    covered = {item["id"] for item in service.matching_sets(model_id)}
    if not covered:
        return None
    suites = service.suites()
    suites.sort(key=lambda suite: suite["id"] != preferred_suite_id)
    for suite in suites:
        if not covered & {item["id"] for item in service.suite_sets(suite["id"])}:
            continue
        card = service.scorecard(suite["id"], [model_id])
        row = next(item for item in card["rows"] if item["model_id"] == model_id)
        return {
            "suite_id": suite["id"],
            "suite_name": suite["name"],
            "model_id": model_id,
            "is_baseline": row["is_baseline"],
            "has_baseline": suite["baseline_model_id"] is not None,
            "evaluated": any(cell["status"] == "succeeded" for cell in row["cells"]),
            "regressions": row["regressions"],
            **{
                key: row[key]
                for key in ("total_sets", "completed_sets", "failed_sets", "complete", "comparison_complete")
            },
        }
    return None


class RunDetailPanel(QWidget):
    """Protocol details of one set, overlaying every selected model."""

    seriesVisibilityChanged = Signal(object, bool)
    groupFocused = Signal(str)
    groupRemoved = Signal(str)
    markerRequested = Signal(str, str)
    allGroupsRequested = Signal()
    comparisonRemoved = Signal(int)
    samplingRequested = Signal(str)
    baselineRequested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = None
        self._args = None
        self._groups = None
        self._total_groups = 0
        self._series_styles = {}
        self.model_colors = {}
        self.baseline_id = None
        self.response_kind = "response"
        self.response_view_changed = self._redraw
        self._hidden_series = set()
        self._curves = {}
        self._redrawing = False
        self._redraw_pending = False
        self._eos_overlay = False
        self._eos_residual = False
        self.load_scan_energies = None
        self.magnetic_selection = None
        self.magnetic_can_focus = False
        self.eos_current_id = None
        self.group_markers = {}
        self._adaptive_layout = False
        self._inspector_expanded = True
        self.legend_buttons = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        # Attach before populating: the validation workspace later moves these widgets.
        layout.addLayout(header)
        self.header_layout = header
        from NepTrainKit.ui.widgets.accuracy_comparison_plot import accuracy_controls

        self.accuracy_tools = accuracy_controls(self)
        header.addWidget(self.accuracy_tools)
        self.exchange_picker = SegmentedWidget(self)
        def choose_response(kind):
            self.response_kind = kind
            self.response_view_changed()
        for key, label in (("response", "Energy response"), ("projections", "Bilinear projection"),
                           ("symmetric", "Symmetric projection"), ("d_projection", "Chiral D projection")):
            self.exchange_picker.addItem(key, self.tr(label), lambda checked=False, k=key: choose_response(k))
        self.exchange_picker.setCurrentItem("response")
        self.exchange_picker.hide()
        header.addWidget(self.exchange_picker)
        header.setSpacing(16)
        self.title_label = BodyLabel(self)
        self.offset_checkbox = CheckBox(self.tr("Remove constant offset"), self)
        self.offset_checkbox.setChecked(True)
        self.offset_checkbox.setToolTip(self.tr("Shift each model curve by its mean error, as the shape metrics do"))
        self.offset_checkbox.stateChanged.connect(lambda _state: self._redraw())
        self.relative_checkbox = CheckBox(self.tr("Relative energy"), self)
        self.relative_checkbox.setChecked(True)
        self.relative_checkbox.setToolTip(
            self.tr("Show energies relative to the group's reference state.")
        )
        self.relative_checkbox.stateChanged.connect(lambda _state: self._redraw())
        header.addWidget(self.title_label, 1)
        self.diagnostic_mode = ComboBox(self)
        self.diagnostic_mode.addItem(self.tr("Energy response"), userData="relative_energy")
        self.diagnostic_mode.addItem(self.tr("Energy derivatives"), userData="derivative")
        self.diagnostic_mode.addItem(self.tr("Derivative residual"), userData="derivative_residual")
        self.diagnostic_mode.setToolTip(
            self.tr("Switches every diagnostic chart at once; the derivative view uses total energy.")
        )
        self._diagnostic_charts = []
        from NepTrainKit.ui.widgets.diagnostic_plot import set_response_metric

        self.diagnostic_mode.currentIndexChanged.connect(lambda *_: set_response_metric(self))
        header.addWidget(self.diagnostic_mode)
        self.wheel_zoom_checkbox = CheckBox(self.tr("Wheel zoom"), self)
        self.wheel_zoom_checkbox.setToolTip(
            self.tr("Off: scroll the chart list. On: use the wheel to zoom the chart under the pointer.")
        )
        header.addWidget(self.wheel_zoom_checkbox)
        header.addWidget(self.relative_checkbox)
        header.addWidget(self.offset_checkbox)
        self.info_label = CaptionLabel(self)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        layout.addWidget(self.info_label)
        self.legend_panel = QWidget(self)
        self.legend_flow = FlowLayout(self.legend_panel, needAni=False)
        self.legend_flow.setContentsMargins(0, 0, 0, 2)
        self.legend_flow.setHorizontalSpacing(8)
        self.legend_flow.setVerticalSpacing(4)
        layout.addWidget(self.legend_panel)

        # Horizontal splitter: Left = Plot Canvas / Multi-Group Grid; Right = Side-by-Side Quantitative Inspector
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.content_splitter.setChildrenCollapsible(False)

        self.plot_pane = QWidget(self.content_splitter)
        plot_pane_layout = QVBoxLayout(self.plot_pane)
        plot_pane_layout.setContentsMargins(0, 0, 0, 0)
        plot_pane_layout.setSpacing(0)

        self.grid_scroll = QScrollArea(self.plot_pane)
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 0, 8, 0)
        self.grid_layout.setSpacing(14)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.grid_scroll.setWidget(self.grid_container)
        self.grid_container.setAutoFillBackground(False)
        self.grid_scroll.viewport().setAutoFillBackground(False)
        plot_pane_layout.addWidget(self.grid_scroll, 1)

        # Right-hand Side-by-Side Multi-Model Quantitative Comparison & Verdict Inspector
        self.inspector_panel = QFrame(self.content_splitter)
        self.inspector_panel.setObjectName("multiModelInspector")
        self.inspector_panel.setStyleSheet(
            f"#multiModelInspector {{ background: {palette.SURFACE}; border: 1px solid {palette.BORDER}; border-radius: 10px; }}"
        )
        self.inspector_panel.setMinimumWidth(310)
        inspector_layout = QVBoxLayout(self.inspector_panel)
        inspector_layout.setContentsMargins(12, 10, 12, 10)
        inspector_layout.setSpacing(8)

        inspector_header = QHBoxLayout()
        self.inspector_title = BodyLabel(self.tr("Physical metrics"), self.inspector_panel)
        inspector_font = self.inspector_title.font()
        inspector_font.setBold(True)
        self.inspector_title.setFont(inspector_font)
        inspector_header.addWidget(self.inspector_title, 1)
        self.inspector_badge = CaptionLabel("", self.inspector_panel)
        self.inspector_badge.setStyleSheet(
            f"color: {palette.ACCENT}; background: {palette.ACCENT_SOFT}; padding: 2px 8px; border-radius: 5px; font-weight: 600;"
        )
        inspector_header.addWidget(self.inspector_badge)
        inspector_layout.addLayout(inspector_header)

        self.multi_model_table = TableWidget(self.inspector_panel)
        self.multi_model_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.multi_model_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.multi_model_table.verticalHeader().hide()
        self.multi_model_table.verticalHeader().setDefaultSectionSize(30)
        self.multi_model_table.setMinimumHeight(140)
        inspector_layout.addWidget(self.multi_model_table, 1)

        self.verdict_card = QFrame(self.inspector_panel)
        self.verdict_card.setObjectName("verdictCard")
        self.verdict_card.setStyleSheet(
            f"#verdictCard {{ background: {palette.ACCENT_SOFT}; border: 1px solid {palette.BORDER}; border-radius: 8px; }}"
        )
        verdict_layout = QVBoxLayout(self.verdict_card)
        verdict_layout.setContentsMargins(10, 8, 10, 8)
        verdict_layout.setSpacing(4)
        self.verdict_heading = BodyLabel("分层诊断与比较结论", self.verdict_card)
        v_font = self.verdict_heading.font()
        v_font.setBold(True)
        self.verdict_heading.setFont(v_font)
        verdict_layout.addWidget(self.verdict_heading)
        self.verdict_summary_label = CaptionLabel(self.verdict_card)
        self.verdict_summary_label.setWordWrap(True)
        self.verdict_summary_label.setStyleSheet(f"color: {palette.TEXT_PRIMARY};")
        verdict_layout.addWidget(self.verdict_summary_label)

        verdict_actions = QHBoxLayout()
        verdict_actions.setSpacing(6)
        self.inspector_sample_button = PushButton(
            QIcon(":/images/src/images/make.svg"), "送入卡片工作流补采样", self.verdict_card
        )
        self.inspector_sample_button.setToolTip(self.tr("Create sampling workflow"))
        self.inspector_sample_button.clicked.connect(self._emit_active_sampling)
        verdict_actions.addWidget(self.inspector_sample_button)
        verdict_actions.addStretch(1)
        verdict_layout.addLayout(verdict_actions)
        inspector_layout.addWidget(self.verdict_card)

        self.content_splitter.addWidget(self.plot_pane)
        self.content_splitter.addWidget(self.inspector_panel)
        self.content_splitter.setStretchFactor(0, 7)
        self.content_splitter.setStretchFactor(1, 5)
        self.content_splitter.setSizes([620, 420])
        self.inspector_panel.hide()
        layout.addWidget(self.content_splitter, 1)

        self.eos_cards = []
        self.eos_plots = []
        self._grid_columns = 0
        self.plot_pane.installEventFilter(self)
        self.show_runs(None, [])

    @property
    def _pg(self):
        import NepTrainKit.ui.canvas.pyqtgraph  # shared style for the legacy plots
        import pyqtgraph
        return pyqtgraph

    def set_adaptive_layout(self, enabled: bool) -> None:
        """Enable full-width responsive plot cards and the side-by-side multi-model inspector."""
        self._adaptive_layout = bool(enabled)
        self.set_inspector_expanded(self._inspector_expanded)
        self._layout_cards()

    def set_inspector_expanded(self, expanded: bool) -> None:
        self._inspector_expanded = expanded
        has_item = bool(self._groups or (self._args and self._args[0]))
        self.inspector_panel.setVisible(self._adaptive_layout and expanded and has_item)

    def _emit_active_sampling(self) -> None:
        benchmark = (
            self._groups[0][1]
            if self._groups
            else (self._args[0] if self._args and self._args[0] else None)
        )
        if benchmark and benchmark.get("id"):
            self.samplingRequested.emit(benchmark["id"])

    def show_runs(self, benchmark: dict | None, entries: list[tuple[int, str, dict | None, dict]]) -> None:
        """``entries`` are ``(model id, label, run or None, run details)`` in overlay order."""
        self._args = (benchmark, entries)
        self._total_groups = 1 if benchmark else 0
        self._groups = (
            [("G1", benchmark, entries)]
            if benchmark and presentation_for(benchmark["protocol"]).draw_groups
            else None
        )
        self._redraw()

    def show_groups(self, groups, *, total_groups=None):
        """EOS curves across selected reference groups; entries are already per atom."""
        if not groups:
            self.show_runs(None, [])
            return
        self._groups = groups
        self._total_groups = len(groups) if total_groups is None else total_groups
        self._args = (groups[0][1], groups[0][2]) if groups else (None, [])
        self._redraw()

    def _redraw(self) -> None:
        # Native GL creation can process selection events while cards are still being built.
        if self._redrawing:
            self._redraw_pending = True
            return
        self._redrawing = True
        try:
            self._redraw_contents()
        finally:
            self._redrawing = False
        if self._redraw_pending:
            self._redraw_pending = False
            QTimer.singleShot(0, self, self._redraw)

    def _redraw_contents(self) -> None:
        groups = self._groups if self._groups is not None else [("", *self._args)] if self._args[0] else []
        self._diagnostic_charts = []
        diagnostic = bool(groups and groups[0][1]["protocol"] == "diagnostic")
        self.accuracy_tools.setVisible(bool(groups and uses_accuracy_cards(groups[0][1])))
        exchange = groups[0][1].get("summary", {}).get("exchange", {}) if groups else {}
        self.exchange_picker.setVisible(bool(exchange.get("projections")))
        for key in ("symmetric", "d_projection"):
            self.exchange_picker.items[key].setEnabled(bool(exchange.get("symmetric_chiral")))
        if self.response_kind != "response" and not (
                exchange.get("projections") if self.response_kind == "projections" else exchange.get("symmetric_chiral")):
            self.response_kind = "response"
        self.exchange_picker.setCurrentItem(self.response_kind)
        self.diagnostic_mode.setVisible(diagnostic)
        self.wheel_zoom_checkbox.setVisible(bool(self._groups))
        self._rebuild_legend(groups)
        self.relative_checkbox.setVisible(self._groups is not None)
        self.grid_scroll.setVisible(self._groups is not None)
        if self.plot is None and self._groups is None and self._args[0]:
            self.plot = self._pg.PlotWidget(self.plot_pane)
            self.plot.showGrid(x=True, y=True, alpha=0.2)
            self.legend = self.plot.addLegend(offset=(-10, 10))
            self.plot.setMinimumHeight(240)
            self.plot_pane.layout().insertWidget(0, self.plot, 1)
        if self.plot is not None:
            self.plot.setVisible(self._groups is None and bool(self._args[0]))
        self._update_inspector(groups)
        if self._groups is not None:
            presentation_for(self._groups[0][1]["protocol"]).draw_groups(self)
            return
        benchmark, entries = self._args
        if self.plot is not None:
            self.plot.clear()
            self.legend.clear()
        self.offset_checkbox.setVisible(bool(benchmark) and benchmark["protocol"] == "eos")
        if not benchmark:
            self.title_label.setText(self.tr("Click a cell to see its details"))
            self.info_label.setText("")
            return
        protocol = protocol_for_set(benchmark)
        provenance = benchmark.get("provenance") or {}
        source = provenance.get("source_kind")
        source = {"Demo": self.tr("Demo"), "Other": self.tr("Other")}.get(source, source)
        self.title_label.setText(self.tr("Reference dataset: {name}").format(name=benchmark["name"]))
        notes = []
        if source:
            notes.append(
                self.tr("Reference: {source}, {conditions}").format(
                    source=source, conditions=provenance.get("conditions", "-")
                )
            )
        elif provenance.get("conditions"):
            notes.append(self.tr("Notes: {notes}").format(notes=provenance["conditions"]))
        if provenance.get("notes"):
            notes.append(self.tr("Notes: {notes}").format(notes=provenance["notes"]))
        notes.append(
            self.tr("Fixed reference geometries")
            if protocol.mode == "fixed-geometry"
            else self.tr("Mode: {mode}").format(mode=protocol.mode)
        )
        summary = benchmark.get("summary") or {}
        if protocol.kind == "eos" and summary.get("volumes"):
            notes.append(
                self.tr("{count} volumes · {low:.3f}–{high:.3f} Å³/atom").format(
                    count=len(summary["volumes"]),
                    low=summary["volumes"][0],
                    high=summary["volumes"][-1],
                )
            )
            fit = summary.get("fit") or {}
            if "v0" in fit:
                notes.append(
                    self.tr("Reference fit: V₀ = {v0:.3f} Å³/atom · B₀ = {b0:.2f} GPa").format(
                        v0=fit["v0"], b0=fit["b0_gpa"]
                    )
                )
            else:
                notes.append(self.tr("Reference fit unavailable: E–V curve comparison only."))
        for _key, name, run, _details in entries:
            if run is None:
                notes.append(self.tr("{name}: not run").format(name=name))
            elif run["status"] != "succeeded":
                status = {"failed": self.tr("failed"), "unsupported": self.tr("unsupported")}.get(
                    run["status"], run["status"]
                )
                notes.append(
                    self.tr("{name}: {status} {error}")
                    .format(name=name, status=status, error=translate_runtime_message(run["error"]))
                    .rstrip()
                )
            elif _details.get("label_counts"):
                counts = _details["label_counts"]
                notes.append(
                    self.tr("RMSE E/F/V/MF: {values}").format(
                        values=" / ".join(
                            "—"
                            if run["metrics"].get(key + "_rmse") is None
                            else f"{run['metrics'][key + '_rmse']:.6g}"
                            for key in ("energy", "force", "virial", "mforce")
                        )
                    )
                )
                notes.append(
                    self.tr(
                        "{name}: valid labels — E {energy} structures; F {force} atoms; "
                        "V {virial} structures; MF {mforce} atoms."
                    ).format(
                        name=name,
                        energy=counts["energy"]["structures"],
                        force=counts["force"]["atoms"],
                        virial=counts["virial"]["structures"],
                        mforce=counts["mforce"]["atoms"],
                    )
                )
                notes.append(
                    self.tr("Compared components E/F/V/MF: {counts}").format(
                        counts=" / ".join(
                            str(counts[key]["evaluated_components"])
                            for key in ("energy", "force", "virial", "mforce")
                        )
                    )
                )
                scope = _details.get("evaluation_scope", {})
                source = _details.get("prediction_source", {}).get("kind")
                notes.append(
                    self.tr("Model {model} · data {data} · {frames} frames / {atoms} atoms · {source}").format(
                        model=_details.get("model_sha256", "")[:12],
                        data=_details.get("dataset_sha256", "")[:12],
                        frames=len(scope.get("frame_indices", [])),
                        atoms=sum(scope.get("atom_counts", [])),
                        source=self.tr("Verified cache")
                        if source == "verified_cache"
                        else self.tr("Saved predictions")
                        if source == "prediction_cache"
                        else self.tr("Recalculated"),
                    )
                )
                notes.extend(translate_runtime_message(warning) for warning in _details.get("warnings", []))
        self.info_label.setText("\n".join(notes))
        self.info_label.setToolTip(
            "\n".join(
                self.tr("{name}\nModel SHA-256: {model}\nDataset SHA-256: {data}").format(
                    name=name, model=details.get("model_sha256", ""), data=details.get("dataset_sha256", "")
                )
                for _, name, run, details in entries
                if run and details.get("model_sha256")
            )
        )
        renderer = {"efv": self._plot_efv}.get(protocol.kind)
        if renderer is not None:
            renderer(
                benchmark,
                [
                    (key, name, details)
                    for key, name, run, details in entries
                    if run and run["status"] == "succeeded" and details
                ],
            )

    def _update_inspector(self, groups) -> None:
        """Populate the side-by-side multi-model metrics table and 3-tier diagnostic verdict card."""
        if not self._adaptive_layout or not groups:
            self.inspector_panel.hide()
            return
        self.inspector_panel.setVisible(self._inspector_expanded)
        tag, benchmark, entries = groups[0]
        protocol_kind = benchmark.get("protocol", "eos")
        summary = benchmark.get("summary") or {}
        short_name = benchmark.get("name", "").rsplit(" / ", 1)[-1]
        self.inspector_badge.setText(f"{tag or protocol_kind.upper()} · {summary.get('formula') or short_name}")
        self.inspector_sample_button.setVisible(protocol_kind == "eos")

        protocol = protocol_for_set(benchmark)
        headers = [self.tr("Metric")]
        for key, name, _, _ in entries:
            # Keep units in the row and compact role labels above the numbers.
            headers.append(self.tr("Current model") if key == self._current_key else name.split(" / ")[-1].split(" (")[0].split("（")[0].strip())
        self.multi_model_table.setColumnCount(len(headers))
        self.multi_model_table.setHorizontalHeaderLabels(headers)
        for column, (_, name, _, _) in enumerate(entries, 1):
            self.multi_model_table.horizontalHeaderItem(column).setToolTip(name)
        header_view = self.multi_model_table.horizontalHeader()
        header_view.setStretchLastSection(False)
        header_view.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header_view.setMinimumSectionSize(54)
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.multi_model_table.setColumnWidth(0, 170)
        rows_data = []
        for metric in protocol.metrics:
            values = []
            for _, _, run, _ in entries:
                value = (run.get("metrics") or {}).get(metric.key) if run and run["status"] == "succeeded" else None
                values.append(format_metric(value, metric.better) if value is not None else "—")
            if metric.headline or any(v != "—" for v in values):
                label = self.tr(metric.label) + (f" ({metric.unit})" if metric.unit else "")
                rows_data.append((label, values))
        if protocol_kind == "eos":
            for key, label in (("v0", "V₀ (Å³/atom)"), ("b0_gpa", "B₀ (GPa)")):
                fits = [(details or {}).get("model_fit") or {} for _, _, _, details in entries]
                ref = (summary.get("fit") or {}).get(key)
                if ref is not None:
                    label += f" · {self.tr('Reference')} {ref:.3g}"
                rows_data.append((label, [format(fit[key], ".4g") if key in fit else "—" for fit in fits]))

        self.multi_model_table.setRowCount(len(rows_data))
        for r_idx, (metric_label, values) in enumerate(rows_data):
            label_item = QTableWidgetItem(metric_label)
            label_item.setToolTip(metric_label)
            self.multi_model_table.setItem(r_idx, 0, label_item)
            for c_idx, val_text in enumerate(values, 1):
                item = QTableWidgetItem(str(val_text))
                item.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
                self.multi_model_table.setItem(r_idx, c_idx, item)

        warnings = summary.get("warnings") or []
        self.verdict_card.setVisible(bool(warnings) or protocol_kind == "eos")
        self.verdict_heading.hide()
        self.verdict_summary_label.setVisible(bool(warnings))
        self.verdict_summary_label.setText("\n".join(map(translate_runtime_message, warnings)))

    def reset_series(self):
        self._series_styles.clear()
        self.model_colors.clear()
        self._hidden_series.clear()

    def _series_pen(self, key):
        eos = bool(self._adaptive_layout and self._groups and uses_curve_chart(self._groups[0][1]))
        if key is None:
            pen = QPen(QColor(palette.PLOT_AXIS), EOS_REFERENCE_WIDTH if eos else 1.5, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            return pen
        if key not in self._series_styles:
            index = len(self._series_styles)
            styles = (
                Qt.PenStyle.SolidLine,
                Qt.PenStyle.DashDotLine,
                Qt.PenStyle.DashLine,
                Qt.PenStyle.DotLine,
            )
            self._series_styles[key] = (
                SERIES_COLORS[index % len(SERIES_COLORS)],
                styles[index % len(styles)],
            )
        color, style = self._series_styles[key]
        width = 2.8 if key == self._current_key else 1.8
        pen = QPen(QColor(self.model_colors.get(key, color)), width + (0.9 if eos else 0), style)
        pen.setCosmetic(True)
        return pen

    def _rebuild_legend(self, groups):
        while self.legend_flow.count():
            widget = self.legend_flow.takeAt(0)
            widget.hide()
            widget.deleteLater()
        self._curves = {}
        self.legend_buttons = {}
        entries = groups[0][2] if groups else []
        diagnostic = bool(groups and groups[0][1]["protocol"] == "diagnostic")
        if not self._adaptive_layout:
            self._current_key = entries[0][0] if entries else None
        keys = ([] if groups and groups[0][1]["protocol"] == "diagnostic" else [None]) + [
            entry[0] for entry in entries
        ]
        if self._adaptive_layout:
            residual = self._eos_residual and groups and uses_curve_chart(groups[0][1])
            keys = [key for key in keys if key is None and not residual]
        else:
            self._hidden_series.intersection_update(keys)
        self.legend_panel.setVisible(bool(groups and keys))
        for key in keys if groups else []:
            if key is None:
                label = self.tr("Reference")
            else:
                label = next(entry[1] for entry in entries if entry[0] == key)
                if key == self._current_key:
                    label += " · " + self.tr("Current model")
                matches = [entry for _tag, _benchmark, items in groups for entry in items if entry[0] == key]
                ready = sum(
                    bool(run and run["status"] == "succeeded" and details)
                    for _key, _name, run, details in matches
                )
                if matches and all(run is None for _key, _name, run, _details in matches):
                    label += " · " + self.tr("Not run")
                elif ready < len(matches):
                    label += " · " + self.tr("{ready}/{total} curves ready").format(
                        ready=ready, total=len(matches)
                    )
            holder = QWidget(self.legend_panel)
            row = QHBoxLayout(holder)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            swatch = QPixmap(32, 16)
            swatch.fill(Qt.GlobalColor.transparent)
            painter = QPainter(swatch)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(self._series_pen(key))
            painter.drawLine(1, 8, 31, 8)
            if key is None:
                painter.setBrush(QColor(palette.PLOT_AXIS))
                painter.drawEllipse(13, 5, 6, 6)
            painter.end()
            button = PushButton(
                QIcon(swatch),
                self.fontMetrics().elidedText(label, Qt.TextElideMode.ElideMiddle, 280),
                holder,
            )
            button.setIconSize(QSize(32, 16))
            button.setStyleSheet(
                button.styleSheet() + "\nPushButton[hasIcon=true] { padding: 5px 12px 6px 52px; }"
            )
            if diagnostic:
                button.setStyleSheet(
                    button.styleSheet()
                    + f"\nPushButton {{border:0;background:transparent;color:{palette.TEXT_PRIMARY};border-radius:6px;}} PushButton:hover {{background:{palette.ACCENT_SOFT};}}"
                )
            button.setCheckable(True)
            button.setChecked(key not in self._hidden_series)
            font = button.font()
            font.setStrikeOut(key in self._hidden_series)
            button.setFont(font)
            button.setToolTip(label + "\n" + self.tr("Click to hide or show this curve in every chart."))
            button.toggled.connect(lambda checked, series=key: self.set_series_visible(series, checked))
            row.addWidget(button)
            if key is not None and key != self._current_key and not self._adaptive_layout:
                pin_btn = TransparentToolButton(
                    FluentIcon.PIN.colored(palette.PLOT_CURRENT, palette.PLOT_CURRENT), holder
                )
                pin_btn.setToolTip(self.tr("Set baseline model"))
                pin_btn.setFixedSize(26, 26)
                pin_btn.clicked.connect(lambda _checked=False, series=key: self.baselineRequested.emit(series))
                row.addWidget(pin_btn)
                remove = TransparentToolButton(
                    FluentIcon.CLOSE.colored(
                        palette.TEXT_SECONDARY if diagnostic else palette.PLOT_REJECT,
                        palette.TEXT_SECONDARY if diagnostic else palette.PLOT_REJECT,
                    ),
                    holder,
                )
                remove.setToolTip(self.tr("Remove comparison model"))
                remove.setAccessibleName(self.tr("Remove comparison model"))
                remove.clicked.connect(lambda _checked=False, series=key: self.comparisonRemoved.emit(series))
                row.addWidget(remove)
            self.legend_flow.addWidget(holder)
            self.legend_buttons[key] = button

    def _register_curve(self, key, curve):
        curve.setProperty("modelId", key)
        self._curves.setdefault(key, []).append(curve)
        curve.setVisible(key not in self._hidden_series and curve.property("viewVisible") is not False)

    def series_visible(self, key):
        return key not in self._hidden_series

    def set_series_color(self, key, color):
        self.model_colors[key] = color.name()
        if self._groups and uses_accuracy_cards(self._groups[0][1]):
            self._redraw()
            return
        for curve in self._curves.get(key, ()):
            if curve.opts["pen"] is not None:
                pen = QPen(curve.opts["pen"]) if curve.property("auxiliary") or curve.property("fixedStyle") else self._series_pen(key)
                pen.setColor(color)
                curve.setPen(pen)
            if curve.opts["symbol"] is not None and hasattr(curve, "setSymbolBrush"):
                curve.setSymbolBrush(color)
                if curve.opts["symbolPen"] is not None:
                    curve.setSymbolPen(color)

    def set_current_series(self, key):
        previous = self._current_key
        if previous == key:
            return
        self._current_key = key
        for model in (previous, key):
            if model is not None:
                for curve in self._curves.get(model, ()):
                    if curve.opts["pen"] is not None and not (curve.property("auxiliary") or curve.property("fixedStyle")):
                        curve.setPen(self._series_pen(model))
        if self._args[0]:
            self._update_inspector(self._groups or [("", *self._args)])

    def set_series_visible(self, key, visible):
        button = self.legend_buttons.get(key)
        if button is not None:
            button.blockSignals(True)
            button.setChecked(visible)
            button.blockSignals(False)
            font = button.font()
            font.setStrikeOut(not visible)
            button.setFont(font)
        if visible:
            self._hidden_series.discard(key)
        else:
            self._hidden_series.add(key)
        if self._groups and uses_accuracy_cards(self._groups[0][1]):
            self._redraw()
            self.seriesVisibilityChanged.emit(key, visible)
            return
        for curve in self._curves.get(key, []):
            curve.setVisible(visible and curve.property("viewVisible") is not False)
        if self._diagnostic_charts:
            from NepTrainKit.ui.widgets.diagnostic_plot import set_response_metric
            set_response_metric(self)
        for plot in self.eos_plots:
            update_residual_placeholder(plot)
            self._reset_plot_view(plot)
        self.seriesVisibilityChanged.emit(key, visible)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Native macOS checkbox size hints can undercount translated text.
        for checkbox in (self.relative_checkbox, self.offset_checkbox, self.wheel_zoom_checkbox):
            checkbox.setMinimumWidth(checkbox.fontMetrics().horizontalAdvance(checkbox.text()) + 40)
        self._layout_cards()

    def _configure_subplot(self, plot):
        plot.viewport().installEventFilter(self)
        plot.getViewBox().state["wheelScaleFactor"] = -0.03

    def eventFilter(self, obj, event):
        if obj is self.plot_pane and event.type() == QEvent.Type.Resize and self._adaptive_layout:
            self._layout_cards()
        if event.type() == QEvent.Type.Wheel and not self.wheel_zoom_checkbox.isChecked():
            # A graphics view consumes wheel events before the surrounding scroll area.
            pixels = event.pixelDelta()
            if not pixels.isNull():
                self.grid_scroll.verticalScrollBar().setValue(
                    self.grid_scroll.verticalScrollBar().value() - pixels.y()
                )
                self.grid_scroll.horizontalScrollBar().setValue(
                    self.grid_scroll.horizontalScrollBar().value() - pixels.x()
                )
                event.accept()
            else:
                QApplication.sendEvent(self.grid_scroll.viewport(), event)
            return True
        return super().eventFilter(obj, event)

    def _layout_cards(self):
        protocol = self._groups[0][1]["protocol"] if self._groups else None
        adaptive_eos = self._adaptive_layout and self._groups and (
            uses_curve_chart(self._groups[0][1]) or uses_accuracy_cards(self._groups[0][1]) or protocol == "diagnostic")
        single_eos = adaptive_eos and len(self.eos_cards) == 1
        limit = 2 if adaptive_eos or protocol == "diagnostic" else 3
        available_width = self.plot_pane.width() if self._adaptive_layout and self.plot_pane.width() > 200 else self.width()
        columns = min(limit, max(1, (available_width - 30) // 360))
        if self._adaptive_layout:
            columns = min(columns, max(1, len(self.eos_cards)))
            width = max(320, (available_width - 24 - 14 * (columns - 1)) // columns)
            for card in self.eos_cards:
                if not adaptive_eos:
                    card.setFixedSize(min(520, width), 274)
                elif not single_eos:
                    card.setMinimumHeight(308)
        elif limit == 2:
            columns = min(columns, max(1, len(self.eos_cards)))
        else:
            width = min(460, max(1, (self.width() - 30 - 14 * (columns - 1)) // columns))
            for card in self.eos_cards:
                card.setFixedWidth(width)
        if columns == self._grid_columns:
            return
        self._grid_columns = columns
        while self.grid_layout.count():
            self.grid_layout.takeAt(0)
        self.grid_layout.setContentsMargins(0, 0, 0 if adaptive_eos else 8, 0)
        self.grid_layout.setSpacing(4 if adaptive_eos else 14)
        fill_row = adaptive_eos and len(self.eos_cards) <= columns
        alignment = Qt.AlignmentFlag(0) if fill_row else Qt.AlignmentFlag.AlignTop
        self.grid_layout.setAlignment(alignment)
        if self._adaptive_layout and len(self.eos_cards) == 1 and not single_eos:
            alignment |= Qt.AlignmentFlag.AlignHCenter
        elif not self._adaptive_layout and limit != 2:
            alignment |= Qt.AlignmentFlag.AlignLeft
        for index, card in enumerate(self.eos_cards):
            self.grid_layout.addWidget(
                card,
                index // columns,
                index % columns,
                alignment,
            )
        for column in range(3):
            self.grid_layout.setColumnStretch(column, 1 if column < columns else 0)

    def set_group_marker(self, key, symbol):
        self.group_markers[key] = symbol
        for card, plot in zip(self.eos_cards, self.eos_plots):
            for marker in card.findChildren(GroupMarkerButton):
                if marker.property("groupId") == key:
                    marker.set_marker(symbol)
            for curve in plot.listDataItems():
                if curve.property("groupId") == key:
                    curve.setSymbol(symbol)

    def set_eos_overlay(self, enabled):
        self._eos_overlay = bool(enabled)
        self._redraw()

    def _change_eos_view(self, index):
        self._eos_residual = index == 1
        self._redraw()

    @staticmethod
    def _reset_plot_view(plot):
        if hasattr(plot, "fit"):
            plot.fit()
            return
        view = plot.getViewBox()
        view.enableAutoRange(x=True, y=True)
        view.updateAutoRange()

    def _plot_efv(self, _benchmark: dict, runs: list[tuple[int, str, dict]]) -> None:
        pg = self._pg
        self.plot.setLabel("bottom", self.tr("Reference energy (eV/atom)"))
        self.plot.setLabel("left", self.tr("Predicted energy (eV/atom)"))
        bounds = []
        for key, name, details in runs:
            energy = details.get("energy") or {}
            reference, predicted = energy.get("reference", []), energy.get("predicted", [])
            bounds += reference + predicted
            color = self._series_pen(key).color()
            curve = self.plot.plot(
                reference,
                predicted,
                pen=None,
                symbol="o",
                symbolBrush=color,
                symbolPen=color,
                symbolSize=6,
            )
            self._register_curve(key, curve)
        if bounds:
            low, high = min(bounds), max(bounds)
            self._register_curve(
                None,
                self.plot.plot(
                    [low, high],
                    [low, high],
                    pen=pg.mkPen(palette.PLOT_LINE, style=Qt.PenStyle.DashLine),
                    antialias=True,
                ),
            )


class ModelLibraryWidget(QWidget):
    """Suite selector, scorecard table and per-set details for captured models."""

    scorecardChanged = Signal()
    toolbarContextChanged = Signal()
    samplingWorkflowRequested = Signal(object)

    def __init__(self, parent=None, service: ValidationService | None = None):
        super().__init__(parent)
        self._service = service
        self._run_task: BackgroundTask | None = None
        self._run_total = 0
        self._run_records = []
        self._failed_jobs = []
        self._failed_context = None
        self._library_root = None
        self._loaded = False
        self._displayed_suite_id = None
        self._draft: ModelDraft | None = None

        self.controls_panel = QFrame(self)
        self.controls_panel.setObjectName("validationControls")
        self.controls_panel.setStyleSheet(
            f"#validationControls {{ background: {palette.SURFACE}; border: 1px solid {palette.BORDER}; border-radius: 10px; }}"
        )
        controls = QVBoxLayout(self.controls_panel)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setSpacing(8)
        self.suite_combo = ComboBox(self)
        self.suite_combo.setMinimumWidth(150)
        self.suite_combo.currentIndexChanged.connect(self._on_suite_changed)
        self.refresh_button = TransparentToolButton(
            FluentIcon.SYNC.colored(palette.ACCENT, palette.ACCENT), self
        )
        self.refresh_button.setToolTip(self.tr("Refresh"))
        self.refresh_button.setAccessibleName(self.tr("Refresh"))
        self.refresh_button.clicked.connect(lambda: self.refresh())
        self.diagnostic_action = Action(FluentIcon.ADD.icon(color=palette.ACCENT), "新建无参考诊断…", self)
        self.edit_diagnostic_action = Action(
            FluentIcon.EDIT.icon(color=palette.ACCENT), "编辑所选诊断实验的副本…", self
        )
        self.edit_diagnostic_action.triggered.connect(self.edit_diagnostic_plan)
        self.save_plan_action = Action(FluentIcon.SAVE.icon(color=palette.ACCENT), "保存诊断方案…", self)
        self.diagnostic_action.triggered.connect(self.show_diagnostic_editor)
        self.save_plan_action.triggered.connect(self.save_diagnostic_plan)
        self.import_action = Action(
            QIcon(":/images/src/images/open.svg"), self.tr("Import reference dataset"), self
        )
        self.import_action.setToolTip(self.tr("Import reference dataset"))
        self.import_files_action = Action(
            FluentIcon.FOLDER, self.tr("Import multiple reference files…"), self
        )
        self.import_folder_action = Action(FluentIcon.FOLDER, self.tr("Import reference folder…"), self)
        self.import_files_action.triggered.connect(lambda: self.import_reference_batch())
        self.import_folder_action.triggered.connect(lambda: self.import_reference_batch(folder=True))
        self.import_package_action = Action(
            FluentIcon.DOWNLOAD.icon(color=palette.ACCENT),
            self.tr("Import validation sharing package…"),
            self,
        )
        self.export_package_action = Action(
            FluentIcon.SHARE.icon(color=palette.ACCENT),
            self.tr("Export validation sharing package…"),
            self,
        )
        self.import_package_action.triggered.connect(lambda: self.import_validation_package())
        self.export_package_action.triggered.connect(lambda: self.export_validation_package())
        self.star_action = Action(FluentIcon.HEART.icon(color=palette.PLOT_SELECTED), self.tr("Star"), self)
        self.star_action.setCheckable(True)
        self.save_action = Action(QIcon(":/images/src/images/save.svg"), "保存势版本到项目…", self)
        self.save_action.setToolTip(
            self.tr("Saves the model version to the project; the diagnostic plan is not saved")
        )
        self.baseline_button = TransparentToolButton(
            FluentIcon.PIN.colored(palette.PLOT_CURRENT, palette.PLOT_CURRENT), self
        )
        self.baseline_button.setToolTip(
            self.tr("Set baseline model")
            + "\n"
            + self.tr("Set a saved version to compare model improvements and regressions.")
        )
        self.baseline_button.setAccessibleName(self.tr("Set baseline model"))
        self.run_selected_button = PrimarySplitPushButton(FluentIcon.PLAY, "验证所选势", self)
        self.run_selected_button.setToolTip(
            self.tr("Runs the current model and the ticked comparison models; hiding a curve does not untick it.")
        )
        self.run_selected_button.clicked.connect(self._on_run_clicked)
        self.run_menu = QMenu(self)
        self.run_current_action = Action(FluentIcon.PLAY, "仅验证当前势", self)
        self.run_all_action = Action(FluentIcon.PLAY, "验证全库兼容势（含其他项目）", self)
        self.run_current_action.triggered.connect(self.run_current)
        self.run_all_action.triggered.connect(self.run_all)
        self.run_menu.addAction(self.run_current_action)
        self.run_menu.addSeparator()
        self.run_menu.addAction(self.run_all_action)
        self.recalculate_action = Action(FluentIcon.SYNC, self.tr("Recalculate selected models"), self)
        self.recalculate_action.triggered.connect(
            lambda: self._run_validation(self.selected_model_ids(), force=True)
        )
        self.output_cache_action = Action(
            FluentIcon.FOLDER, self.tr("Import .out cache for accuracy datasets…"), self
        )
        self.output_cache_action.setToolTip(
            self.tr(
                "Only verified outputs are reused. Missing or mismatched provenance triggers a fresh evaluation; "
                "source files are kept."
            )
        )
        self.output_cache_action.triggered.connect(self.import_output_cache)
        self.run_menu.addSeparator()
        self.run_menu.addAction(self.recalculate_action)
        self.run_menu.addAction(self.output_cache_action)
        self.run_selected_button.setFlyout(self.run_menu)
        self.scope_combo = ComboBox(self)
        for label, scope in (
            ("当前性质 · 全部组", "property"),
            ("当前性质 · 选中组", "selection"),
            ("整个方案 · 所有性质", "suite"),
        ):
            self.scope_combo.addItem(label, userData=scope)
        self.scope_combo.setAccessibleName("验证范围")
        self.scope_combo.setToolTip(
            self.tr(
                "This scope applies to the selected, current, and library-wide runs alike. Focus a group by enlarging its chart or by selecting rows in the detail table."
            )
        )
        self.scope_combo.currentIndexChanged.connect(self._update_run_scope)
        self.scope_hint = CaptionLabel(self)
        self.scope_hint.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        self.model_combo = ComboBox(self)
        self.model_combo.setMinimumWidth(130)
        self.comparison_picker = ComparisonPicker(self)
        self.comparison_picker.setMinimumWidth(130)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.comparison_picker.selectionChanged.connect(self._on_selection_changed)
        self.import_action.triggered.connect(lambda: self.import_benchmark())
        self.star_action.triggered.connect(lambda: self.toggle_star())
        self.save_action.triggered.connect(lambda: self.save_current())
        self.baseline_button.clicked.connect(self.set_baseline)

        self.baseline_label = CaptionLabel(self)
        self.baseline_label.setWordWrap(True)
        self.baseline_label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        self.tolerance_label = CaptionLabel(self)
        self.tolerance_label.setWordWrap(True)
        self.tolerance_label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        self.tolerance_label.hide()
        self.curves_button = PushButton(FluentIcon.VIEW, "查看曲线", self)
        self.curves_button.setCheckable(True)
        self.curves_button.toggled.connect(self._toggle_curves)

        self.property_combo = ComboBox(self)
        self.property_combo.currentIndexChanged.connect(lambda *_: self._load_groups())
        self.property_combo.setAccessibleName("验证性质")
        self._layout_model_controls(controls)
        self.all_groups_button = TransparentPushButton(self.tr("Show all groups"), self)
        self.all_groups_button.clicked.connect(self._select_all_groups)
        self.all_groups_button.setToolTip(
            self.tr("Each group has independent axes. Focus a chart or select rows in the error details."),
        )
        self._benchmarks = []

        self.model = ScorecardTableModel(self)
        self.table = TableView(self)
        self.metrics_model = GroupMetricsTableModel(self.model, self)
        self.table.setModel(self.metrics_model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.selectionModel().selectionChanged.connect(self._on_group_selection_changed)
        self.table.hide()

        self.detail = RunDetailPanel(self)
        self.detail.load_scan_energies = lambda item, run: self.service().absolute_scan_energies(item, run)
        self.detail.header_layout.insertWidget(2, self.all_groups_button)
        self.detail.comparisonRemoved.connect(self.comparison_picker.remove_id)
        self.detail.baselineRequested.connect(self.set_baseline_for_model)
        self.detail.allGroupsRequested.connect(self._select_all_groups)
        self.detail.groupFocused.connect(self._focus_plot_group)
        self.detail.samplingRequested.connect(self.open_sampling)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.controls_row = QHBoxLayout()
        self.controls_row.setContentsMargins(12, 10, 12, 4)
        self.controls_row.addWidget(self.controls_panel)
        layout.addLayout(self.controls_row)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(16, 2, 16, 4)
        self.run_summary = CaptionLabel(self)
        self.run_summary.setWordWrap(True)
        self.run_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.retry_button = TransparentPushButton(FluentIcon.SYNC, "重试失败项", self)
        self.retry_button.clicked.connect(self.retry_failed)
        self.retry_button.hide()
        status_row.addWidget(self.run_summary, 1)
        status_row.addWidget(self.retry_button)
        layout.addLayout(status_row)
        self.result_panel = QFrame(self)
        result_layout = QVBoxLayout(self.result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(0)
        from NepTrainKit.ui.widgets.diagnostic_editor import DiagnosticEditor

        self.diagnostic_editor = DiagnosticEditor(self)
        self.diagnostic_editor.prepared.connect(self._prepare_diagnostic_plan)
        self.diagnostic_editor.readinessChanged.connect(self._sync_save_context)
        self.editor_scroll = QScrollArea(self)
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.editor_scroll.setWidget(self.diagnostic_editor)
        self.editor_scroll.hide()
        self.diagnostic_editor.hide()
        self.diagnostic_editor.installEventFilter(self)
        result_layout.addWidget(self.editor_scroll, 1)
        result_layout.addWidget(self.detail, 1)
        self.curves_placeholder = QWidget(self)
        placeholder = QVBoxLayout(self.curves_placeholder)
        placeholder.addStretch()
        self.overview_label = BodyLabel("选择方案与势版本", self.curves_placeholder)
        self.overview_label.setWordWrap(True)
        self.overview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.addWidget(self.overview_label)
        hint = CaptionLabel(
            "选择方案和势版本后，点击“查看曲线”读取已有结果；验证计算需手动运行。",
            self.curves_placeholder,
        )
        hint.setWordWrap(True)
        self.curves_hint = hint
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.addWidget(hint)
        placeholder.addStretch()
        result_layout.addWidget(self.curves_placeholder, 1)
        layout.addWidget(self.result_panel, 1)
        self._sync_curve_visibility()

    def _layout_model_controls(self, controls):
        fields = QHBoxLayout()
        fields.setSpacing(14)
        controls.addLayout(fields)
        for title, combo, button, stretch in (
            (self.tr("Validation plan"), self.suite_combo, self.refresh_button, 4),
            (self.tr("Current model"), self.model_combo, self.baseline_button, 5),
            (self.tr("Compare with"), self.comparison_picker, None, 3),
        ):
            field = QHBoxLayout()
            field.setSpacing(6)
            fields.addLayout(field, stretch)
            caption = CaptionLabel(title, self)
            caption.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")
            field.addWidget(caption)
            field.addWidget(combo, 1)
            if button is not None:
                button.setFixedSize(28, 28)
                field.addWidget(button)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        controls.addLayout(actions)
        for widget, stretch in (
            (self.baseline_label, 1), (self.property_combo, 0), (self.tolerance_label, 1),
            (self.scope_combo, 0), (self.scope_hint, 0), (self.curves_button, 0),
            (self.run_selected_button, 0),
        ):
            actions.addWidget(widget, stretch)

    def _sync_curve_visibility(self):
        expanded = self.curves_button.isChecked()
        editing = not self.editor_scroll.isHidden()
        self.curves_button.setText("收起曲线" if expanded else "查看曲线")
        self.detail.setVisible(expanded and not editing)
        self.curves_placeholder.setVisible(not expanded and not editing)

    def _toggle_curves(self, expanded):
        self._sync_curve_visibility()
        if expanded:
            self._update_detail()

    def _expand_curves(self):
        self.curves_button.blockSignals(True)
        self.curves_button.setChecked(True)
        self.curves_button.blockSignals(False)
        self._sync_curve_visibility()

    # -- data -----------------------------------------------------------------
    def service(self) -> ValidationService:
        if self._service is not None:
            return self._service
        from NepTrainKit.core.validation.library import get_library

        library = get_library()
        cached = getattr(self, "_session_service", None)
        if cached is None or cached.library.root != library.root:
            self._session_service = ValidationService(library=library)
        return self._session_service

    def current_suite_id(self) -> int | None:
        return self.suite_combo.currentData()

    def showEvent(self, event):
        super().showEvent(event)
        if self._run_task is None:
            # Navigation reuses the existing view; it is not a data-change event.
            try:
                needs_load = not self._loaded or self._library_root != self.service().library.root
            except Exception:
                needs_load = True  # Let refresh report an unavailable library.
            if needs_load:
                self.refresh()

    def refresh(self, suite_id: int | None = None, model_id: int | None = None) -> None:
        if self._run_task is not None:
            return
        self._loaded = False
        try:
            service = self.service()
            if self._library_root is not None and self._library_root != service.library.root:
                suite_id, model_id = None, None
                self._draft = None
                self._displayed_suite_id = None
                self.comparison_picker.set_options([], [])
                self.detail.reset_series()
                self.suite_combo.blockSignals(True)
                self.suite_combo.clear()
                self.suite_combo.blockSignals(False)
                self.model.set_scorecard(None)
            elif model_id is None:
                current = self.current_model_row()
                model_id = current["model_id"] if current else None
            self._library_root = service.library.root
            if suite_id is None:
                suite_id = self.current_suite_id()
                if suite_id is None:
                    suite_id = Config.getint("validation", CONFIG_SUITE, 0) or None
            suites = service.suites()
            self._unavailable_models = service.snapshot_registered_models()
        except Exception as error:  # noqa: BLE001 - a broken library must not break the page
            MessageManager.send_error_message(
                self.tr("Cannot open the model library: {error}").format(error=error)
            )
            self.suite_combo.blockSignals(True)
            self.suite_combo.clear()
            self.suite_combo.blockSignals(False)
            self._benchmarks = []
            for combo in (self.model_combo, self.property_combo):
                combo.blockSignals(True)
                combo.clear()
                combo.blockSignals(False)
            self.comparison_picker.set_options([], [])
            self.detail.reset_series()
            self.metrics_model.set_groups([], -1, "eos")
            self.model.set_scorecard(None)
            self._on_selection_changed()
            return
        self.suite_combo.blockSignals(True)
        self.suite_combo.clear()
        for suite in suites:
            self.suite_combo.addItem(suite["name"], userData=suite["id"])
        ids = [suite["id"] for suite in suites]
        if ids:
            self.suite_combo.setCurrentIndex(ids.index(suite_id) if suite_id in ids else 0)
        self.suite_combo.blockSignals(False)
        self._load_scorecard(model_id)
        self._loaded = True

    def _on_suite_changed(self, _index: int) -> None:
        suite_id = self.current_suite_id()
        if suite_id is not None and suite_id > 0:
            Config.set("validation", CONFIG_SUITE, suite_id)
        self._load_scorecard()

    def _load_scorecard(self, model_id: int | None = None, *, comparison_ids=None, update_detail=True) -> None:
        suite_id = self.current_suite_id()
        service = self.service()
        self._benchmarks = service.suite_sets(suite_id) if suite_id is not None else []
        previous_kind = self.property_combo.currentData()
        kinds = list(dict.fromkeys(item["protocol"] for item in self._benchmarks))
        kinds.sort(key=lambda kind: kind != "eos")
        self.property_combo.blockSignals(True)
        self.property_combo.clear()
        for kind in kinds:
            self.property_combo.addItem(
                {
                    "eos": "EOS",
                    "diagnostic": "无参考诊断",
                    "response": self.tr("Reference response analysis"),
                }.get(kind, self.tr("Energy / force / virial")),
                userData=kind,
            )
        previous_index = self.property_combo.findData(previous_kind)
        if previous_index >= 0:
            self.property_combo.setCurrentIndex(previous_index)
        self.property_combo.blockSignals(False)
        if self._draft is not None:
            existing = service.find_model_by_hash(self._draft.content_hash)
            if existing:
                model_id = existing["id"]
                self._draft = None
        card = service.scorecard(suite_id, draft=self._draft)
        if (
            model_id
            and all(row["model_id"] != model_id for row in card["rows"])
            and any(model["id"] == model_id for model in service.models())
        ):
            card = service.scorecard(
                suite_id, [row["model_id"] for row in card["rows"]] + [model_id], draft=self._draft
            )
        self.model.set_scorecard(card)
        suite = (card or {}).get("suite") or {}
        baseline = next((row for row in (card or {}).get("rows", []) if row["is_baseline"]), None)
        tolerance = (card or {}).get("tolerance")
        if suite_id is None:
            self.baseline_label.setText(
                self.tr("Open the menu above to start a reference-free diagnostic, or to import reference data.")
            )
        elif baseline is None:
            self.baseline_label.setText(self.tr("Baseline model: not selected"))
        else:
            text = self.tr("Baseline model: {name}").format(name=baseline["name"])
            if suite.get("baseline_note"):
                text += f" — {suite['baseline_note']}"
            self.baseline_label.setText(text)
        # Say what "worse" means; otherwise the red cells state a verdict whose
        # threshold the user has to guess.
        self.tolerance_label.setText(
            self.tr("Worse = beyond {percent}% of the baseline, or beyond a metric's own tolerance").format(
                percent=f"{tolerance * 100:.0f}"
            )
            if isinstance(tolerance, (int, float))
            else ""
        )
        self.tolerance_label.setVisible(bool(self.tolerance_label.text()) and baseline is not None)
        current_id = self._scorecard_current_id(model_id, comparison_ids)
        if comparison_ids is None:
            comparison_ids = (self.comparison_picker.selected_ids() if self._displayed_suite_id == suite_id
                              else [suite.get("baseline_model_id")])
        if model_id is not None and current_id is None and baseline and model_id == baseline["model_id"]:
            comparison_ids = [*comparison_ids, model_id]
        self._displayed_suite_id = suite_id
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for index, row in enumerate(self.model._rows):
            label = self._model_label(index)
            self.model_combo.addItem(label, userData=row["model_id"])
        if not self.model._rows:
            self.model_combo.addItem(self._empty_model_placeholder(), userData=None)
        self.model_combo.setEnabled(bool(self.model._rows))
        self.model_combo.setToolTip(self._model_combo_hint())
        self.model_combo.setCurrentIndex(self.model_combo.findData(current_id))
        self.model_combo.blockSignals(False)
        self._update_comparison_options(comparison_ids)
        self._load_groups(update_detail=False)
        self._on_selection_changed(update_detail=update_detail)
        self.scorecardChanged.emit()

    def _scorecard_current_id(self, requested, comparison_ids):
        if self.model.row_of_model(requested) >= 0:
            return requested
        return self.model._rows[0]["model_id"] if self.model._rows else None

    def open_scorecard(
        self,
        suite_id: int | None,
        model_id: int | None = None,
        *,
        draft: ModelDraft | None = None,
        show_curves: bool = True,
    ) -> None:
        """Show ``suite_id`` (``None``: the last used suite) with ``model_id`` selected."""
        if self._run_task is not None:
            MessageManager.send_info_message("验证正在运行，请完成或停止后再切换方案与势版本。")
            return
        if self._library_root is not None and self._library_root != self.service().library.root:
            self.refresh()
        self._draft = draft
        if draft is not None:
            model_id = 0
            if suite_id is None:
                elements = set(parse_model_header(draft.data)[1])
                service = self.service()
                suite_id = next(
                    (
                        suite["id"]
                        for suite in service.suites()
                        if any(set(item["elements"]) <= elements for item in service.suite_sets(suite["id"]))
                    ),
                    None,
                )
        # Explicit viewer/history handoffs request a preview; plain navigation does not.
        if show_curves:
            self._expand_curves()
        else:
            self.curves_button.blockSignals(True)
            self.curves_button.setChecked(False)
            self.curves_button.blockSignals(False)
            self._sync_curve_visibility()
        self.refresh(suite_id, model_id)

    # -- selection and details --------------------------------------------------
    def current_model_row(self) -> dict | None:
        index = self.model.row_of_model(self.model_combo.currentData())
        return self.model.row(index)

    def _model_label(self, index, *, include_baseline=True):
        row = self.model.row(index)
        label = self.model.model_name(row, include_baseline=include_baseline)
        if (
            sum(
                item["name"] == row["name"] and item.get("project_name") == row.get("project_name")
                for item in self.model._rows
            )
            > 1
        ):
            label += f" · #{row['model_id']}"
        return label

    def _empty_model_placeholder(self) -> str:
        hidden = self._hidden_model_count()
        if hidden:
            return self.tr("No compatible model · {count} hidden").format(count=hidden)
        return (
            "选择势版本后运行"
            if any(b["protocol"] == "diagnostic" for b in self._benchmarks)
            else self.tr("Reference preview only")
        )

    def _model_combo_hint(self) -> str:
        hidden = self._hidden_model_count()
        if hidden:
            elements = "、".join("/".join(item["elements"]) for item in self._benchmarks if item["elements"])
            return self.tr(
                "{count} model(s) were left out because their elements do not cover this plan ({elements}). "
                "Switch plan, or validate them against a plan covering their elements."
            ).format(count=hidden, elements=elements)
        if not self.model._rows:
            return self.tr("No model has been captured yet. Open a NEP model to associate it.")
        return self.tr("The model whose metrics this page shows.")

    def _hidden_model_count(self) -> int:
        try:
            service = self.service()
            rows = {row["model_id"] for row in self.model._rows}
            wanted = [set(item["elements"]) for item in self._benchmarks]
            return sum(
                1
                for model in service.models()
                if model["id"] not in rows
                and wanted
                and not any(need <= set(model["elements"] or ()) for need in wanted)
            )
        except Exception:
            return 0  # A count is decoration; never break the page over it.

    def _update_comparison_options(self, selected_ids=None):
        if selected_ids is None:
            selected_ids = self.comparison_picker.selected_ids()
        options = []
        for index, row in enumerate(self.model._rows):
            if row["model_id"] == self.model_combo.currentData():
                continue
            cells = list(
                {
                    column["set_id"]: cell for column, cell in zip(self.model._columns, row.get("cells", []))
                }.values()
            )
            ready = sum(cell["status"] == "succeeded" for cell in cells)
            status = (
                self.tr("Not run")
                if not cells or all(cell["status"] == "missing" for cell in cells)
                else self.tr("{ready}/{total} validations completed").format(ready=ready, total=len(cells))
            )
            label = self._model_label(index)
            options.append((row["model_id"], label, status))
        self.comparison_picker.set_options(options, selected_ids)

    def selected_model_ids(self):
        current = self.current_model_row()
        return list(
            dict.fromkeys(([current["model_id"]] if current else []) + self.comparison_picker.selected_ids())
        )

    def comparison_curve_ids(self):
        return self.selected_model_ids()

    def _on_model_changed(self, *_args):
        self._update_comparison_options()
        self._load_groups(update_detail=False)
        self._on_selection_changed()

    def _load_groups(self, *, update_detail=True):
        kind = self.property_combo.currentData() or "eos"
        self.property_combo.setVisible(self.property_combo.count() > 1)
        selected = {self.metrics_model.benchmarks[i.row()]["id"] for i in self.table.selectionModel().selectedRows()}
        benchmarks = [item for item in self._benchmarks if item["protocol"] == kind]
        selection = self.table.selectionModel()
        selection.blockSignals(True)
        self.metrics_model.set_groups(
            benchmarks, self.model.row_of_model(self.model_combo.currentData()), kind
        )
        if kind in ("eos", "diagnostic", "response"):
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            if selected and selected < {b["id"] for b in benchmarks}:
                from PySide6.QtCore import QItemSelectionModel

                for i, benchmark in enumerate(benchmarks):
                    if benchmark["id"] in selected:
                        selection.select(
                            self.metrics_model.index(i, 0),
                            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                        )
            else:
                self.table.selectAll()
        else:
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            if benchmarks:
                self.table.selectRow(next((i for i, b in enumerate(benchmarks) if b["id"] in selected), 0))
        selection.blockSignals(False)
        self.all_groups_button.setVisible(kind in ("eos", "diagnostic", "response"))
        self._update_run_scope()
        if update_detail:
            self._update_detail()

    def _on_group_selection_changed(self, *_args):
        self._update_run_scope()
        self._update_detail()

    def _select_all_groups(self):
        self.table.selectAll()
        self._update_detail()

    def _on_selection_changed(self, *, update_detail=True) -> None:
        row = self.current_model_row()
        has_suite = self.current_suite_id() is not None
        saved = row is not None and not row.get("temporary")
        diagnostic = any(b["protocol"] == "diagnostic" for b in self._benchmarks)
        editing = not self.editor_scroll.isHidden()
        self.edit_diagnostic_action.setEnabled(diagnostic and self._run_task is None and not editing)
        self._sync_save_context()
        self.diagnostic_action.setEnabled(self._run_task is None and not editing)
        self.import_action.setEnabled(self._run_task is None and not editing)
        self.import_files_action.setEnabled(self._run_task is None and not editing)
        self.import_folder_action.setEnabled(self._run_task is None and not editing)
        self.import_package_action.setEnabled(self._run_task is None and not editing)
        self.curves_button.setEnabled(bool(self._benchmarks))
        self.export_package_action.setEnabled(
            has_suite and not diagnostic and bool(self.metrics_model.benchmarks) and self._run_task is None
        )
        self.export_package_action.setToolTip(
            "无参考诊断分享将在下一阶段支持；当前可保存本地方案。"
            if diagnostic
            else self.export_package_action.text()
        )
        self.save_action.setEnabled(row is not None and self._run_task is None and not editing)
        self.star_action.setEnabled(saved and self._run_task is None and not editing)
        self.star_action.setText(self.tr("Unstar") if row and row["starred"] else self.tr("Star"))
        self.star_action.setToolTip(self.star_action.text())
        self.star_action.setChecked(bool(row and row["starred"]))
        self.baseline_button.setEnabled(saved and has_suite and self._run_task is None)
        self._update_run_scope()
        if update_detail:
            self._update_detail()

    def _focus_plot_group(self, tag):
        self.table.selectRow(int(tag[1:]) - 1)

    def _update_detail(self, *, group_rows=None, group_total=None) -> None:
        if not self.curves_button.isChecked():
            return  # Do not read result files or instantiate charts for an overview.
        self.detail.baseline_id = next((row["model_id"] for row in self.model._rows if row["is_baseline"]), None)
        benchmarks = self.metrics_model.benchmarks
        if not benchmarks:
            self.detail.show_runs(None, [])
            return
        selected = group_rows if group_rows is not None else sorted(index.row() for index in self.table.selectionModel().selectedRows())
        focused = len(selected) < len(benchmarks)
        self.all_groups_button.setEnabled(focused)
        self.all_groups_button.setVisible(focused or self.metrics_model.kind == "eos")
        self.all_groups_button.setText(self.tr("Back to all groups") if focused else self.tr("Show all groups"))
        self.all_groups_button.setIcon(
            FluentIcon.RETURN.colored(palette.PLOT_LOADED_OVERLAY, palette.PLOT_LOADED_OVERLAY)
            if focused
            else None
        )
        rows = []
        for model_id in self.comparison_curve_ids():
            row = self.model.row(self.model.row_of_model(model_id))
            if row and row not in rows:
                rows.append(row)
        groups = [(f"G{index + 1}", benchmarks[index], self._curve_entries(benchmarks[index], rows))
                  for index in selected]
        if not groups:
            self.detail.show_runs(None, [])
        elif presentation_for(self.metrics_model.kind).draw_groups:
            self.detail.show_groups(groups, total_groups=len(benchmarks) if group_total is None else group_total)
        elif groups:
            self.detail.show_runs(groups[0][1], groups[0][2])
        else:
            self.detail.show_runs(None, [])

    def _curve_entries(self, benchmark, rows):
        service = self.service()
        entries = []
        for row in rows:
            column = self.model.first_column_of_set(benchmark["id"])
            cell = row["cells"][column - 1] if column > 0 else None
            run = (
                service.latest_run(
                    row["model_id"], benchmark["id"], protocol_version=protocol_for_set(benchmark).version
                )
                if row["model_id"]
                else None
            )
            if cell and cell["status"] == "missing":
                run = None
            elif run is not None and cell and run["status"] != cell["status"]:
                run = dict(run, status=cell["status"], error=cell["error"])
            details = {}
            if cell and cell.get("previous_version"):
                details["previous_version"] = cell["previous_version"]
            if run is not None and run["status"] == "succeeded":
                try:
                    result = service.run_details(run)
                    details = dict(result.get("details", {}), warnings=result.get("warnings", []))
                    if uses_accuracy_cards(benchmark):
                        details = service.accuracy_view_data(benchmark, run, details)
                except OSError:
                    pass
            entries.append(
                (
                    row["model_id"],
                    self._model_label(self.model.row_of_model(row["model_id"]), include_baseline=not self.detail._adaptive_layout),
                    run,
                    details,
                )
            )
        return entries

    def eventFilter(self, watched, event):
        from PySide6.QtCore import QEvent

        if (
            watched is getattr(self, "diagnostic_editor", None)
            and event.type() == QEvent.Type.Hide
            and watched.isHidden()
        ):
            self.editor_scroll.hide()
            self._sync_curve_visibility()
            self.controls_panel.setEnabled(True)
            self.controls_panel.show()
            self.run_summary.show()
            self._on_selection_changed()
        return super().eventFilter(watched, event)

    def show_diagnostic_editor(self):
        if self._run_task is not None:
            return
        self.controls_panel.setEnabled(False)
        self.controls_panel.hide()
        self.save_plan_action.setEnabled(False)
        self.edit_diagnostic_action.setEnabled(False)
        self.export_package_action.setEnabled(False)
        self.detail.hide()
        self.curves_placeholder.hide()
        self.editor_scroll.show()
        self.diagnostic_editor.show()
        self._on_selection_changed()
        self.run_summary.hide()
        self.retry_button.hide()

    def _sync_save_context(self, *_args):
        if not hasattr(self, "editor_scroll"):
            return
        editing = not self.editor_scroll.isHidden()
        temporary = self.current_suite_id() is not None and self.current_suite_id() < 0
        ready = self.diagnostic_editor.use_button.isEnabled() if editing else temporary
        self.save_plan_action.setEnabled(ready and self._run_task is None)
        self.save_plan_action.setToolTip(
            "保存当前编辑的诊断方案（结构与参数）" if editing else "保存临时诊断方案及已有结果"
        )
        self.toolbarContextChanged.emit()

    def edit_diagnostic_plan(self):
        from NepTrainKit.core.validation.protocols.diagnostic import experiment

        if self._run_task is not None:
            return
        try:
            selected = [
                self.metrics_model.benchmarks[index.row()]
                for index in self.table.selectionModel().selectedRows()
            ]
            experiments = [
                experiment(self.service().set_frames(item["id"]))
                for item in selected
                if item["protocol"] == "diagnostic"
            ]
            if not experiments:
                raise ValueError("请先选择或放大一项无参考诊断实验。")
            self.diagnostic_editor.load_experiments(self.suite_combo.currentText(), experiments)
            self.show_diagnostic_editor()
        except ValueError as exc:
            MessageManager.send_error_message(str(exc))

    def set_diagnostic_seeds(self, seeds):
        self.diagnostic_editor.set_seeds(seeds)

    def _prepare_diagnostic_plan(self, name, experiments, temporary=False):
        try:
            suite = self.service().create_diagnostic_plan(name, experiments, temporary=temporary)
            if not temporary:
                Config.set("validation", CONFIG_SUITE, suite["id"])
            self.diagnostic_editor.hide()
            self._expand_curves()
            self.refresh(suite["id"])
            if not temporary:
                MessageManager.send_success_message("诊断方案已保存，重启后可继续使用；运行结果将自动保存。")
        except Exception as exc:
            self.diagnostic_editor.preview.setText(str(exc))
            MessageManager.send_error_message(str(exc))

    def save_diagnostic_plan(self):
        if not self.editor_scroll.isHidden():
            if self._run_task is None and self.diagnostic_editor.use_button.isEnabled():
                self.diagnostic_editor.prepare()
            return
        if self._run_task is not None or self.current_suite_id() is None or self.current_suite_id() >= 0:
            return
        box = GetStrMessageBox(self.window(), "保存为新的诊断方案；不会覆盖已有方案或基准势。")
        box.lineEdit.setText(self.suite_combo.currentText().removesuffix(" · 临时"))
        if not box.exec():
            return
        try:
            suite = self.service().save_diagnostic_plan(self.current_suite_id(), box.lineEdit.text())
            Config.set("validation", CONFIG_SUITE, suite["id"])
            self.refresh(suite["id"])
            MessageManager.send_success_message("诊断方案及已有运行结果已保存。")
        except Exception as exc:
            MessageManager.send_error_message(str(exc))

    # -- actions --------------------------------------------------------------
    def open_sampling(self, set_id):
        from NepTrainKit.core.validation.sampling import magnetic_sampling_payload, sampling_payload
        from NepTrainKit.ui.widgets.reference_response_plot import is_magnetic_eos, magnetic_branches

        service = self._view_service()
        if service is None:
            return None
        try:
            group_id = set_id
            set_id = group_id.partition("/")[0]
            benchmark = service.get_set(set_id)
            if benchmark["protocol"] != "eos" and not is_magnetic_eos(benchmark):
                return None
            row = self.current_model_row()
            model = service.get_model(row["model_id"]) if row and row["model_id"] else None
            run = service.latest_run(model["id"], set_id, protocol_version=protocol_for_set(benchmark).version) if model else None
            if not run or run["status"] != "succeeded":
                run = None
            frames = service.set_frames(set_id)
            if is_magnetic_eos(benchmark):
                branches = magnetic_branches(benchmark)
                if group_id == set_id:
                    group_id = self.detail.eos_current_id or branches[0][0]
                branch = next((b for b in branches if b[0] == group_id), None)
                if branch is None:
                    raise ValueError(f"Unknown scan branch: {group_id}")
                payload = magnetic_sampling_payload(frames, benchmark, branch, model, run)
            else:
                details = service.run_details(run).get("details", {}) if run else None
                payload = sampling_payload(frames, benchmark, details, model, run)
            self.samplingWorkflowRequested.emit(payload)
            return payload
        except Exception as error:
            MessageManager.send_error_message(str(error))
            return None

    def save_current(self) -> None:
        service = self._view_service()
        row = self.current_model_row()
        if service is None or row is None:
            return
        try:
            if row["model_id"] == 0:
                draft = self._draft
            else:
                model = service.get_model(row["model_id"])
                draft = ModelDraft(service.model_file(model["id"]).read_bytes(), model["source_path"])
            if draft is None:
                return
            saved = save_model_to_project(self.window(), service, draft, name=row["name"])
            if saved is not None:
                self._draft = None
                self.open_scorecard(None, saved["id"])
                MessageManager.send_success_message(self.tr("Saved to project"))
        except Exception as error:
            MessageManager.send_error_message(str(error))

    def _view_service(self):
        service = self.service()
        if self._library_root != service.library.root:
            self.refresh()
            return None
        return service

    def toggle_star(self) -> None:
        service = self._view_service()
        if service is None:
            return
        row = self.current_model_row()
        if row is None:
            return
        starred = not row["starred"]
        service.star_model(row["model_id"], starred)
        row["starred"] = starred
        index = self.model.index(self.model.row_of_model(row["model_id"]), 0)
        self.model.dataChanged.emit(index, index)
        self.star_action.setChecked(starred)
        self.star_action.setText(self.tr("Unstar") if starred else self.tr("Star"))
        self.star_action.setToolTip(self.star_action.text())
        self.scorecardChanged.emit()

    def set_baseline(self) -> None:
        row = self.current_model_row()
        if row is not None:
            self.set_baseline_for_model(row["model_id"])

    def set_baseline_for_model(self, model_id: int) -> None:
        service = self._view_service()
        if service is None:
            return
        suite_id = self.current_suite_id()
        if not model_id or suite_id is None:
            return
        box = GetStrMessageBox(
            self.window(),
            self.tr(
                "Note for the baseline model (optional). This changes the comparison version only; reference datasets and original model files stay unchanged."
            ),
        )
        box.yesButton.setText(self.tr("Set baseline model"))
        if not box.exec():
            return
        current = self.current_model_row()
        keep_id = current["model_id"] if current else model_id
        service.set_baseline(suite_id, model_id, box.lineEdit.text())
        self.refresh(suite_id, keep_id)

    def _confirm_large_copy(self, error: SetTooLargeError) -> bool:
        box = MessageBox(
            self.tr("Large benchmark file"),
            self.tr(
                "The file is {size:.1f} MB, above the {limit:.0f} MB limit. Copy it into the library anyway?"
            ).format(size=error.size / 1e6, limit=error.limit / 1e6),
            self.window(),
        )
        return bool(box.exec())

    def export_validation_package(self):
        from NepTrainKit.core.validation.package import export_package
        from NepTrainKit.ui.widgets.validation_package_dialogs import ExportValidationPackageDialog

        suite_id = self.current_suite_id()
        if suite_id is None:
            return
        try:
            dialog = ExportValidationPackageDialog(
                self.window(), self._view_service(), suite_id, self.selected_model_ids()
            )
            if not dialog.exec():
                return
            path, _ = QFileDialog.getSaveFileName(
                self,
                self.tr("Export validation sharing package…"),
                "validation.ntkpack",
                "NepTrainKit (*.ntkpack)",
            )
            if path:
                if not path.lower().endswith(".ntkpack"):
                    path += ".ntkpack"
                export_package(dialog.preview, path)
                MessageManager.send_info_message(self.tr("Validation sharing package exported."))
        except (ValueError, OSError) as error:
            MessageManager.send_error_message(str(error))

    def import_validation_package(self):
        from NepTrainKit.core.validation.package import import_package
        from NepTrainKit.ui.widgets.validation_package_dialogs import ImportValidationPackageDialog

        try:
            service = self._view_service()
            dialog = ImportValidationPackageDialog(self.window(), service)
            if not dialog.exec():
                return
            suite = import_package(
                service,
                dialog.preview,
                name=dialog.name_edit.text(),
                project_name=dialog.project_combo.text(),
            )
            self.open_scorecard(suite["id"], suite["baseline_model_id"])
            self.scorecardChanged.emit()
            MessageManager.send_info_message(self.tr("Sharing package imported as a new validation suite."))
        except (ValueError, OSError) as error:
            MessageManager.send_error_message(str(error))

    def import_reference_batch(self, *, folder=False):
        from pathlib import Path
        from NepTrainKit.ui.widgets.reference_batch import ReferenceBatchDialog

        if folder:
            directory = QFileDialog.getExistingDirectory(self, self.tr("Select reference dataset folder"))
            paths = (
                sorted(
                    p
                    for p in Path(directory).rglob("*")
                    if p.suffix.lower() in (".xyz", ".extxyz") and p.is_file()
                )
                if directory
                else []
            )
        else:
            paths, _ = QFileDialog.getOpenFileNames(
                self, self.tr("Select reference datasets"), "", "EXTXYZ (*.xyz *.extxyz)"
            )
        if not paths:
            return None
        dialog = ReferenceBatchDialog(
            self, self.service(), paths, suite_name=self.suite_combo.currentText(), group_by_directory=folder
        )
        if not dialog.exec():
            return None
        try:
            try:
                records = dialog.import_into()
            except SetTooLargeError as error:
                if not self._confirm_large_copy(error):
                    return None
                records = dialog.import_into(allow_large=True)
        except (OSError, ValueError, SetRejectedError) as error:
            MessageManager.send_error_message(str(error))
            return None
        suite = next(s for s in self.service().suites() if s["name"] == dialog.suite.text().strip())
        self.refresh(suite["id"])
        return records

    def import_benchmark(self) -> list[dict] | None:
        """Import one or more reference datasets and return their records."""
        service = self.service()
        dialog = BenchmarkImportDialog(
            self.window(),
            [s for s in service.suites() if s["id"] > 0],
            self.suite_combo.currentText() if (self.current_suite_id() or 0) > 0 else "",
        )
        if not dialog.exec():
            return None
        try:
            records = dialog.import_into(service, self._confirm_large_copy)
        except (SetRejectedError, ValueError, OSError) as error:
            MessageManager.send_error_message(self.tr("Import failed: {error}").format(error=error))
            return None
        if not records:
            return None
        MessageManager.send_success_message(
            self.tr("Imported {count} reference datasets").format(count=len(records))
        )
        suite_name = dialog.values()["suite"]
        suite = next((item for item in service.suites() if item["name"] == suite_name), None)
        self.refresh(suite["id"] if suite else None)
        self.property_combo.setCurrentIndex(self.property_combo.findData(records[0]["protocol"]))
        return records

    def selected_benchmarks(self):
        scope = self.scope_combo.currentData()
        if scope == "suite":
            return list(self._benchmarks)
        benchmarks = list(self.metrics_model.benchmarks)
        if scope == "selection":
            indexes = {index.row() for index in self.table.selectionModel().selectedRows()}
            return [item for i, item in enumerate(benchmarks) if i in indexes]
        return benchmarks

    def _update_run_scope(self, *_args):
        if not hasattr(self, "metrics_model") or not hasattr(self, "overview_label"):
            return
        benchmarks = self.selected_benchmarks()
        ids = self.selected_model_ids()
        all_jobs = self.run_jobs()
        jobs = [job for job in all_jobs if job[0] in ids]
        self.scope_hint.setText(
            self.tr("{models} models · {items} items · {runs} validations").format(
                models=len(ids), items=len(benchmarks), runs=len(jobs)
            )
        )
        self.scope_hint.setToolTip(
            self.tr(
                "Current model plus the ticked comparison models; only element-compatible jobs are created. The scope does not change when curves are hidden."
            )
        )
        idle = self._run_task is None
        self.run_selected_button.setEnabled(bool(jobs) if idle else True)
        self.run_current_action.setEnabled(
            idle and any(job[0] == self.model_combo.currentData() for job in all_jobs)
        )
        self.run_all_action.setEnabled(idle and bool(all_jobs))
        self.recalculate_action.setEnabled(idle and bool(jobs))
        accuracy_ids = {item["id"] for item in benchmarks if item["protocol"] == "efv"}
        self.output_cache_action.setEnabled(idle and any(set_id in accuracy_ids for _, set_id in jobs))
        self.run_all_action.setText(
            self.tr("Validate every compatible model in the library ({runs} runs, other projects included)").format(
                runs=len(all_jobs)
            )
        )
        self.scope_combo.setEnabled(idle)
        self.refresh_button.setEnabled(idle)
        self.model_combo.setEnabled(idle and bool(self.model._rows))
        self.comparison_picker.setEnabled(idle)
        self.property_combo.setEnabled(idle)
        self.table.setEnabled(idle)
        self.retry_button.setVisible(
            idle
            and bool(self._failed_jobs)
            and self._failed_context == (self._library_root, self.current_suite_id())
            and self.editor_scroll.isHidden()
        )
        counts = dict(succeeded=0, failed=0, unsupported=0, missing=0)
        allowed = set(jobs)
        seen = set()
        for row in self.model._rows:
            for col, cell in zip(self.model._columns, row["cells"]):
                key = (row["model_id"], col["set_id"])
                if key in allowed and key not in seen:
                    seen.add(key)
                    status = cell["status"]
                    counts[status if status in counts else "missing"] += 1
        self.overview_label.setText(
            self.tr(
                "Selected scope · {jobs} validations\nDone {done}    Pending {pending}    Failed {failed}    Unsupported {unsupported}"
            ).format(
                jobs=len(jobs),
                done=counts["succeeded"],
                pending=counts["missing"],
                failed=counts["failed"],
                unsupported=counts["unsupported"],
            )
            if self._benchmarks
            else self.tr("Open the menu above to start a diagnostic plan, or to import reference data.")
        )

    def run_jobs(self, model_id=None, *, model_ids=None, benchmarks=None):
        """One explicit experiment scope, shared by every model-selection action."""
        if self.current_suite_id() is None:
            return []
        if model_ids is None and model_id is not None:
            model_ids = [model_id]
        rows = self.service().models()
        if self._draft is not None:
            rows.append({"id": 0, "elements": parse_model_header(self._draft.data)[1]})
        if benchmarks is None:
            benchmarks = self.selected_benchmarks()
        return [
            (model["id"], benchmark["id"])
            for model in rows
            if model_ids is None or model["id"] in model_ids
            for benchmark in benchmarks
            if set(benchmark["elements"]) <= set(model["elements"] or ())
        ]

    def _on_run_clicked(self):
        if self._run_task is not None:
            self._run_task.stop_work()
            self.run_selected_button.setText(self.tr("Stopping…"))
            self.run_selected_button.setEnabled(False)
            return
        self.run_selected()

    def run_all(self):
        return self._run_validation()

    def run_current(self):
        row = self.current_model_row()
        return self._run_validation([row["model_id"]]) if row else None

    def run_selected(self):
        return self._run_validation(self.selected_model_ids())

    def retry_failed(self):
        if self._failed_context == (self._library_root, self.current_suite_id()):
            return self._run_validation(jobs=list(self._failed_jobs))

    def import_output_cache(self):
        """Import verified outputs for the selected models and accuracy datasets."""
        accuracy_ids = {item["id"] for item in self.selected_benchmarks() if item["protocol"] == "efv"}
        jobs = [job for job in self.run_jobs(model_ids=self.selected_model_ids()) if job[1] in accuracy_ids]
        if not jobs:
            return None
        folder = QFileDialog.getExistingDirectory(self, self.tr("Select official .out cache folder"))
        if folder:
            return self._run_validation(jobs=jobs, cache_directory=folder)
        return None

    def _run_validation(self, model_ids=None, *, jobs=None, force=False, cache_directory=None):
        if self._run_task is not None or self.current_suite_id() is None:
            return None
        service = self._view_service()
        if service is None:
            return None
        # Freeze the visible scope before capture/refresh can change selections.
        jobs = list(jobs if jobs is not None else self.run_jobs(model_ids=model_ids))
        if not jobs:
            MessageManager.send_info_message(
                self.tr("Nothing in this scope can run. Pick a model version and an experiment with compatible elements.")
            )
            return None
        if self._draft is not None and any(model_id == 0 for model_id, _ in jobs):
            try:
                record = service.capture_model(self._draft, temporary=True)
            except Exception as error:
                MessageManager.send_error_message(str(error))
                return None
            self._draft = None
            jobs = [(record["id"] if model_id == 0 else model_id, set_id) for model_id, set_id in jobs]
            primary = self.model_combo.currentData()
            comparisons = [record["id"] if key == 0 else key for key in self.comparison_picker.selected_ids()]
            self._load_scorecard(record["id"] if primary == 0 else primary, comparison_ids=comparisons)
        self._run_records = []
        self._run_names = {
            job: f"{service.get_model(job[0])['name']} · {service.get_set(job[1])['name']}" for job in jobs
        }
        self._failed_jobs = []
        self._failed_context = (self._library_root, self.current_suite_id())
        self._run_total = len(jobs)
        self._run_label = (
            f"{self.suite_combo.currentText()} · {len({m for m, _ in jobs})} 个势 × {len({s for _, s in jobs})} 项实验"
        )

        def evaluate_all():
            options = {}
            if force:
                options["force"] = True
            if cache_directory is not None:
                options["cache_directory"] = cache_directory
            for model_id, set_id in jobs:
                if task.isInterruptionRequested():
                    break
                run = service.evaluate(model_id, set_id, **options)
                self._run_records.append(((model_id, set_id), run))
                yield run

        # Completion of a worker is not a successful scientific evaluation.
        task = BackgroundTask(self, show_tip=False, title=self.tr("Running validation"))
        task.progressSignal.connect(self._on_run_progress)
        task.finished.connect(self._on_run_finished)
        self._run_task = task
        self._on_selection_changed(update_detail=False)
        self.diagnostic_editor.setEnabled(False)
        self.suite_combo.setEnabled(False)
        self.run_selected_button.setText(self.tr("Stop validation"))
        self.run_selected_button.setIcon(FluentIcon.PAUSE)
        self.run_selected_button.dropButton.setEnabled(False)
        self.run_summary.setText(
            self.tr("Validating · {label} · 0 / {total}").format(label=self._run_label, total=len(jobs))
        )
        task.start_work(evaluate_all)
        return task

    def _on_run_progress(self, index):
        self.run_summary.setText(
            self.tr("Validating · {label} · {done} / {total}").format(
                label=self._run_label, done=index + 1, total=self._run_total
            )
        )

    def _on_run_finished(self):
        task, self._run_task = self._run_task, None
        if getattr(getattr(self.window(), "_thread_shutdown", None), "closing", False):
            if task is not None:
                task.deleteLater()
            return
        counts = {
            status: sum(run["status"] == status for _, run in self._run_records)
            for status in ("succeeded", "failed", "unsupported")
        }
        self._failed_jobs = [job for job, run in self._run_records if run["status"] == "failed"]
        heading = "已停止" if task and task.outcome == "canceled" else "运行结束"
        summary = (
            f"{heading} · {self._run_label} · 完成 {len(self._run_records)}/{self._run_total} · "
            f"计算成功 {counts['succeeded']} / 失败 {counts['failed']} / 不支持 {counts['unsupported']}"
        )
        errors = [
            f"{self._run_names[job]}: {translate_runtime_message(run['error'])}"
            for job, run in self._run_records
            if run["status"] != "succeeded"
        ]
        if task and task.outcome == "failed":
            errors.append(task.error_message)
        self.run_summary.setText(summary + ("\n" + errors[0] if errors else " · 计算成功不代表诊断通过"))
        self.run_summary.setToolTip("\n".join(errors))
        self.diagnostic_editor.setEnabled(True)
        self.suite_combo.setEnabled(True)
        self.run_selected_button.setText(self.tr("Validate selected models"))
        self.run_selected_button.setIcon(FluentIcon.PLAY)
        self.run_selected_button.dropButton.setEnabled(True)
        current = self.current_model_row()
        self._expand_curves()
        self.refresh(model_id=current["model_id"] if current else None)
        if task is not None:
            task.deleteLater()
