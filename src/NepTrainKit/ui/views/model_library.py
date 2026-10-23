"""Model library: scorecard of captured models against benchmark suites."""

from __future__ import annotations

import re

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QAbstractItemView, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QScrollArea, QSplitter, QVBoxLayout, QWidget
from qfluentwidgets import (
    Action,
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    FluentIcon,
    FlowLayout,
    MessageBox,
    PrimaryPushButton,
    PushButton,
    TableView,
    ToolButton,
    TransparentToolButton,
    TransparentPushButton,
)

from NepTrainKit import palette
from NepTrainKit.config import Config
from NepTrainKit.core import MessageManager
from NepTrainKit.core.validation import SetRejectedError, SetTooLargeError, ValidationService
from NepTrainKit.core.validation.service import ModelDraft, parse_model_header
from NepTrainKit.core.validation.protocols import get_protocol
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
SERIES_COLORS = ("#2F6FA8", "#F76707", "#2F9E44", "#AE3EC9", "#E03131", "#1098AD", "#A66B00", "#D63384", "#5F3DC4", "#607D8B")


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

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        column = self.column(section)
        if role == Qt.ItemDataRole.DisplayRole:
            if column is None:
                return self.tr("Model version")
            unit = f" ({column['unit']})" if column["unit"] else ""
            labels = {
                "shape_rmse": self.tr("Curve RMSE (offset removed)"),
                "max_shape_error": self.tr("Max curve error"),
                "raw_rmse": self.tr("Raw RMSE"),
                "v0_error": self.tr("V₀ error"),
                "b0_error": self.tr("B₀ error"),
                "energy_rmse": self.tr("Energy RMSE"),
                "force_rmse": self.tr("Force RMSE"),
                "virial_rmse": self.tr("Virial RMSE"),
            }
            label = labels.get(column.get("metric"), column["label"])
            return f"{column['set_name']}\n{label}{unit}"
        if role == Qt.ItemDataRole.ToolTipRole and column is not None:
            return self.tr("{set} · {protocol}").format(set=column["set_name"], protocol=column["protocol"])
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        row = self.row(index.row()) if index.isValid() else None
        if row is None:
            return None
        column = self.column(index.column())
        if column is None:
            if role == Qt.ItemDataRole.DisplayRole:
                name = f"★ {row['name']}" if row["starred"] else row["name"]
                if row["model_id"] == 0:
                    name = self.tr("{name} (unsaved)").format(name=name)
                elif row.get("temporary"):
                    name = self.tr("{name} (temporary)").format(name=name)
                elif row.get("project_name"):
                    name = f"{row['project_name']} / {name}"
                return self.tr("{name} (baseline model)").format(name=name) if row["is_baseline"] else name
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

    def set_groups(self, benchmarks, model_row, kind):
        self.beginResetModel()
        self.benchmarks = list(benchmarks)
        self.model_row = model_row
        self.kind = kind
        keys = {"eos": ("shape_rmse", "v0_error", "b0_error"), "diagnostic": ("energy_span", "check_failures")}.get(kind, ("energy_rmse", "force_rmse"))
        self.metric_columns = [
            [next((i + 1 for i, col in enumerate(self.scorecard._columns)
                   if col["set_id"] == benchmark["id"] and col["metric"] == key), -1) for key in keys]
            for benchmark in self.benchmarks
        ]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.benchmarks)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else (5 if self.kind == "eos" else 4)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal or role != Qt.ItemDataRole.DisplayRole:
            return None
        metrics = [self.tr("Curve RMSE\n(meV/atom)"), self.tr("V₀ error (%)"), self.tr("B₀ error (%)")] if self.kind == "eos" else [
            self.tr("Energy RMSE\n(meV/atom)"), self.tr("Force RMSE\n(meV/Å)")]
        if self.kind == "diagnostic":
            return ["诊断实验", "能量跨度 (meV/atom)", "检查异常数", "运行状态"][section]
        return [self.tr("Reference group"), *metrics, self.tr("Status")][section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.benchmarks):
            return None
        benchmark = self.benchmarks[index.row()]
        if index.column() == 0:
            if role == Qt.ItemDataRole.DisplayRole:
                name = (benchmark.get("source_ref") or {}).get("eos_group", {}).get("value") or benchmark["name"].rsplit(" / ", 1)[-1]
                seed = re.search(r"_seed(\d+)$", name)
                if seed and benchmark.get("summary", {}).get("formula"):
                    digits = seed[1]
                    suffix = "…" + digits[-4:] if len(digits) > 6 else digits
                    name = f"{benchmark['summary']['formula']} · seed {suffix}"
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
        cell = self.scorecard.data(self.scorecard.index(self.model_row, columns[0]), ScorecardTableModel.CellRole) if self.model_row >= 0 and columns[0] >= 0 else None
        warnings = benchmark.get("summary", {}).get("warnings", [])
        if role == Qt.ItemDataRole.ToolTipRole:
            return "\n".join([translate_runtime_message(w) for w in warnings] + ([translate_runtime_message(cell["error"])] if cell and cell.get("error") else []))
        if role == Qt.ItemDataRole.DisplayRole:
            status = {"missing": self.tr("Not run"), "succeeded": self.tr("Completed"), "failed": self.tr("Failed"), "unsupported": self.tr("Unsupported")}.get((cell or {}).get("status"), self.tr("Reference only"))
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
            **{key: row[key] for key in ("total_sets", "completed_sets", "failed_sets", "complete", "comparison_complete")},
        }
    return None


class RunDetailPanel(QWidget):
    """Protocol details of one set, overlaying every selected model."""

    groupFocused = Signal(str)
    allGroupsRequested = Signal()
    comparisonRemoved = Signal(int)
    samplingRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        import NepTrainKit.ui.canvas.pyqtgraph  # noqa: F401 - shared pyqtgraph configuration
        import pyqtgraph as pg

        self._pg = pg
        self._args = None
        self._groups = None
        self._total_groups = 0
        self._series_styles = {}
        self._hidden_series = set()
        self._curves = {}
        self.legend_buttons = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        self.header_layout = header
        self.title_label = BodyLabel(self)
        self.offset_checkbox = CheckBox(self.tr("Remove constant offset"), self)
        self.offset_checkbox.setChecked(True)
        self.offset_checkbox.setToolTip(self.tr("Shift each model curve by its mean error, as the shape metrics do"))
        self.offset_checkbox.stateChanged.connect(lambda _state: self._redraw())
        self.relative_checkbox = CheckBox(self.tr("Relative energy"), self)
        self.relative_checkbox.setChecked(True)
        self.relative_checkbox.setToolTip(self.tr("Subtract each group's reference minimum from both its reference and model energies (eV/atom)."))
        self.relative_checkbox.stateChanged.connect(lambda _state: self._redraw())
        header.addWidget(self.title_label, 1)
        self.diagnostic_mode = ComboBox(self)
        self.diagnostic_mode.addItem("能量响应", userData="relative_energy")
        self.diagnostic_mode.addItem("能量导数", userData="derivative")
        self.diagnostic_mode.setToolTip("统一切换所有诊断图；导数使用总能量。")
        self._diagnostic_charts = []
        from NepTrainKit.ui.widgets.diagnostic_plot import set_response_metric
        self.diagnostic_mode.currentIndexChanged.connect(lambda *_: set_response_metric(self))
        header.addWidget(self.diagnostic_mode)
        header.addWidget(self.relative_checkbox)
        header.addWidget(self.offset_checkbox)
        layout.addLayout(header)
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
        self.plot = pg.PlotWidget(self)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        self.plot.setMinimumHeight(240)
        layout.addWidget(self.plot, 1)
        self.grid_scroll = QScrollArea(self)
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
        layout.addWidget(self.grid_scroll, 1)
        self.eos_cards = []
        self.eos_plots = []
        self._grid_columns = 0
        self.show_runs(None, [])

    def show_runs(self, benchmark: dict | None, entries: list[tuple[int, str, dict | None, dict]]) -> None:
        """``entries`` are ``(model id, label, run or None, run details)`` in overlay order."""
        self._args = (benchmark, entries)
        self._total_groups = 1 if benchmark else 0
        self._groups = [("G1", benchmark, entries)] if benchmark and benchmark["protocol"] in ("eos", "diagnostic") else None
        self._redraw()

    def show_groups(self, groups, *, total_groups=None):
        """EOS curves across selected reference groups; entries are already per atom."""
        self._groups = groups
        self._total_groups = len(groups) if total_groups is None else total_groups
        self._args = (groups[0][1], groups[0][2]) if groups else (None, [])
        self._redraw()

    def _redraw(self) -> None:
        groups = self._groups if self._groups is not None else [("", *self._args)] if self._args[0] else []
        self._diagnostic_charts = []
        diagnostic = bool(groups and groups[0][1]["protocol"] == "diagnostic")
        self.diagnostic_mode.setVisible(diagnostic)
        self._rebuild_legend(groups)
        self.relative_checkbox.setVisible(self._groups is not None)
        self.grid_scroll.setVisible(self._groups is not None)
        self.plot.setVisible(self._groups is None)
        if self._groups is not None:
            if self._groups and self._groups[0][1]["protocol"] == "diagnostic":
                from NepTrainKit.ui.widgets.diagnostic_plot import redraw_diagnostics
                redraw_diagnostics(self)
                return
            self._redraw_eos()
            return
        benchmark, entries = self._args
        self.plot.clear()
        self.legend.clear()
        self.offset_checkbox.setVisible(bool(benchmark) and benchmark["protocol"] == "eos")
        if not benchmark:
            self.title_label.setText(self.tr("Click a cell to see its details"))
            self.info_label.setText("")
            return
        protocol = get_protocol(benchmark["protocol"])
        provenance = benchmark.get("provenance") or {}
        source = provenance.get("source_kind")
        source = {"Demo": self.tr("Demo"), "Other": self.tr("Other")}.get(source, source)
        self.title_label.setText(self.tr("Reference dataset: {name}").format(name=benchmark['name']))
        notes = []
        if source:
            notes.append(self.tr("Reference: {source}, {conditions}").format(
                source=source, conditions=provenance.get("conditions", "-")
            ))
        elif provenance.get("conditions"):
            notes.append(self.tr("Notes: {notes}").format(notes=provenance["conditions"]))
        if provenance.get("notes"):
            notes.append(self.tr("Notes: {notes}").format(notes=provenance["notes"]))
        notes.append(self.tr("Fixed reference geometries") if protocol.mode == "fixed-geometry" else self.tr("Mode: {mode}").format(mode=protocol.mode))
        summary = benchmark.get("summary") or {}
        if protocol.kind == "eos" and summary.get("volumes"):
            notes.append(self.tr("{count} volumes · {low:.3f}–{high:.3f} Å³/atom").format(
                count=len(summary["volumes"]), low=summary["volumes"][0], high=summary["volumes"][-1]))
            fit = summary.get("fit") or {}
            if "v0" in fit:
                notes.append(self.tr("Reference fit: V₀ = {v0:.3f} Å³/atom · B₀ = {b0:.2f} GPa").format(v0=fit["v0"], b0=fit["b0_gpa"]))
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
        self.info_label.setText("\n".join(notes))
        renderer = {"efv": self._plot_efv}.get(protocol.kind)
        if renderer is not None:
            renderer(benchmark, [(key, name, details) for key, name, run, details in entries if run and run["status"] == "succeeded" and details])

    def reset_series(self):
        self._series_styles.clear()
        self._hidden_series.clear()

    def _series_pen(self, key):
        if key is None:
            return self._pg.mkPen(palette.PLOT_AXIS, width=1.5, style=Qt.PenStyle.DashLine)
        if key not in self._series_styles:
            index = len(self._series_styles)
            styles = (Qt.PenStyle.SolidLine, Qt.PenStyle.DashDotLine, Qt.PenStyle.DashLine, Qt.PenStyle.DotLine)
            self._series_styles[key] = (SERIES_COLORS[index % len(SERIES_COLORS)], styles[index % len(styles)])
        color, style = self._series_styles[key]
        return self._pg.mkPen(color, width=2.8 if key == self._current_key else 1.8, style=style)

    def _rebuild_legend(self, groups):
        while self.legend_flow.count():
            widget = self.legend_flow.takeAt(0)
            widget.hide()
            widget.deleteLater()
        self._curves = {}
        self.legend_buttons = {}
        entries = groups[0][2] if groups else []
        diagnostic = bool(groups and groups[0][1]["protocol"] == "diagnostic")
        self._current_key = entries[0][0] if entries else None
        keys = ([] if groups and groups[0][1]["protocol"] == "diagnostic" else [None]) + [entry[0] for entry in entries]
        self._hidden_series.intersection_update(keys)
        self.legend_panel.setVisible(bool(groups))
        for key in keys if groups else []:
            if key is None:
                label = self.tr("Reference")
            else:
                label = next(entry[1] for entry in entries if entry[0] == key)
                if key == self._current_key:
                    label += " · " + self.tr("Current model")
                matches = [entry for _tag, _benchmark, items in groups for entry in items if entry[0] == key]
                ready = sum(bool(run and run["status"] == "succeeded" and details) for _key, _name, run, details in matches)
                if matches and all(run is None for _key, _name, run, _details in matches):
                    label += " · " + self.tr("Not run")
                elif ready < len(matches):
                    label += " · " + self.tr("{ready}/{total} curves ready").format(ready=ready, total=len(matches))
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
            button = PushButton(QIcon(swatch), self.fontMetrics().elidedText(label, Qt.TextElideMode.ElideMiddle, 280), holder)
            button.setIconSize(QSize(32, 16))
            button.setStyleSheet(button.styleSheet() + "\nPushButton[hasIcon=true] { padding: 5px 12px 6px 52px; }")
            if diagnostic:
                button.setStyleSheet(button.styleSheet() + f"\nPushButton {{border:0;background:transparent;color:{palette.TEXT_PRIMARY};border-radius:6px;}} PushButton:hover {{background:{palette.ACCENT_SOFT};}}")
            button.setCheckable(True)
            button.setChecked(key not in self._hidden_series)
            font = button.font()
            font.setStrikeOut(key in self._hidden_series)
            button.setFont(font)
            button.setToolTip(label + "\n" + self.tr("Click to hide or show this curve in every chart."))
            button.toggled.connect(lambda checked, series=key: self._set_series_visible(series, checked))
            row.addWidget(button)
            if key is not None and key != self._current_key:
                remove = TransparentToolButton(FluentIcon.CLOSE.colored(palette.TEXT_SECONDARY if diagnostic else palette.PLOT_REJECT, palette.TEXT_SECONDARY if diagnostic else palette.PLOT_REJECT), holder)
                remove.setToolTip(self.tr("Remove comparison model"))
                remove.setAccessibleName(self.tr("Remove comparison model"))
                remove.clicked.connect(lambda _checked=False, series=key: self.comparisonRemoved.emit(series))
                row.addWidget(remove)
            self.legend_flow.addWidget(holder)
            self.legend_buttons[key] = button

    def _register_curve(self, key, curve):
        self._curves.setdefault(key, []).append(curve)
        curve.setVisible(key not in self._hidden_series)

    def _set_series_visible(self, key, visible):
        button = self.legend_buttons[key]
        font = button.font()
        font.setStrikeOut(not visible)
        button.setFont(font)
        if visible:
            self._hidden_series.discard(key)
        else:
            self._hidden_series.add(key)
        for curve in self._curves.get(key, []):
            curve.setVisible(visible)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_cards()

    def _layout_cards(self):
        limit = 2 if self._groups and self._groups[0][1]["protocol"] == "diagnostic" else 3
        columns = min(limit, max(1, (self.width() - 30) // 360), max(1, len(self.eos_cards)))
        if columns == self._grid_columns:
            return
        self._grid_columns = columns
        while self.grid_layout.count():
            self.grid_layout.takeAt(0)
        for index, card in enumerate(self.eos_cards):
            self.grid_layout.addWidget(card, index // columns, index % columns)
        for column in range(3):
            self.grid_layout.setColumnStretch(column, 1 if column < columns else 0)

    def _redraw_eos(self):
        for card in self.eos_cards:
            self.grid_layout.removeWidget(card)
            card.hide()
            card.deleteLater()
        self.eos_cards = []
        self.eos_plots = []
        self._grid_columns = 0
        self.offset_checkbox.setVisible(bool(self._groups))
        self.title_label.setText(self.tr("EOS · {count} groups").format(count=len(self._groups)))
        relative = self.relative_checkbox.isChecked()
        notes = [self.tr("Independent axes · Energy per atom")]
        if relative:
            notes.append(self.tr("Reference minimum = 0"))
        if self.offset_checkbox.isChecked():
            notes.append(self.tr("Constant offset removed; shape comparison"))
        if len(self._groups) == 1:
            benchmark = self._groups[0][1]
            provenance = benchmark.get("provenance") or {}
            if provenance.get("source_kind"):
                notes.append(self.tr("Reference: {source}, {conditions}").format(source=provenance["source_kind"], conditions=provenance.get("conditions", "-")))
            if provenance.get("notes") or provenance.get("conditions"):
                notes.append(self.tr("Notes: {notes}").format(notes=provenance.get("notes") or provenance["conditions"]))
        self.info_label.setText(" · ".join(notes))
        for tag, benchmark, entries in self._groups:
            summary = benchmark.get("summary") or {}
            if not summary.get("volumes"):
                continue
            card = QFrame(self.grid_container)
            card.setObjectName("eosGroupCard")
            card.setStyleSheet(f"#eosGroupCard {{ background: {palette.SURFACE}; border: 1px solid {palette.BORDER}; border-radius: 10px; }}")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(4)
            heading = QHBoxLayout()
            name = benchmark["name"].rsplit(" / ", 1)[-1]
            formula = summary.get("formula")
            badge = CaptionLabel(tag, card)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setFixedSize(30, 26)
            badge.setStyleSheet(f"color: {palette.ACCENT}; background: {palette.ACCENT_SOFT}; border-radius: 5px; font-weight: 600;")
            heading.addWidget(badge)
            heading.setSpacing(8)
            title = BodyLabel(formula or name, card)
            title_font = title.font()
            title_font.setBold(True)
            title.setFont(title_font)
            title.setWordWrap(True)
            title.setToolTip(benchmark["name"])
            heading.addWidget(title, 1)
            reset = TransparentToolButton(FluentIcon.SYNC.colored(palette.ACCENT, palette.ACCENT), card)
            reset.setObjectName("resetEosView")
            reset.setToolTip(self.tr("Reset view: fit all visible curves"))
            reset.setAccessibleName(self.tr("Reset view"))
            heading.addWidget(reset)
            focused = len(self._groups) == 1 and self._total_groups > 1
            focus = TransparentToolButton((FluentIcon.ZOOM_OUT if focused else FluentIcon.ZOOM_IN).colored(
                palette.PLOT_LOADED_OVERLAY, palette.PLOT_LOADED_OVERLAY), card)
            focus.setObjectName("toggleEosGroupFocus")
            focus.setToolTip(self.tr("Back to all groups") if focused else self.tr("Enlarge this group"))
            focus.setAccessibleName(focus.toolTip())
            if focused:
                focus.clicked.connect(lambda _checked=False: self.allGroupsRequested.emit())
            else:
                focus.clicked.connect(lambda _checked=False, group=tag: self.groupFocused.emit(group))
            focus.setVisible(self._total_groups > 1)
            heading.addWidget(focus)
            sample = TransparentToolButton(QIcon(':/images/src/images/make.svg'), card)
            sample.setObjectName("eosSamplingButton")
            sample.setToolTip(self.tr("Create sampling workflow"))
            sample.setAccessibleName(sample.toolTip())
            sample.clicked.connect(lambda _checked=False, set_id=benchmark["id"]: self.samplingRequested.emit(set_id))
            heading.addWidget(sample)
            card_layout.addLayout(heading)
            status = [self.tr("{count} volume points").format(count=len(summary["volumes"]))]
            if formula and name.startswith(formula + "_"):
                branch = name[len(formula) + 1:]
                branch = re.sub(r"seed(\d{7,})", lambda match: "seed…" + match.group(1)[-4:], branch)
                status.insert(0, branch.replace("_", " "))
            warnings = list(summary.get("warnings", []))
            if warnings:
                status.append(self.tr("Fit warning"))
            for _key, name, run, details in entries:
                if not run:
                    status.append(self.tr("{name}: not run").format(name=name))
                elif run["status"] != "succeeded":
                    status.append(self.tr("{name}: unavailable").format(name=name))
                    warnings.append(run.get("error") or run["status"])
                elif not details:
                    status.append(self.tr("{name}: unavailable").format(name=name))
            caption = CaptionLabel(" · ".join(status), card)
            caption.setWordWrap(True)
            caption.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")
            caption.setToolTip(benchmark["name"] + "\n" + "\n".join(translate_runtime_message(w) for w in warnings))
            card_layout.addWidget(caption)
            plot = self._pg.PlotWidget(card)
            reset.clicked.connect(lambda _checked=False, target=plot: self._reset_plot_view(target))
            plot.showGrid(x=True, y=True, alpha=0.12)
            for axis_name in ("left", "bottom"):
                axis = plot.getAxis(axis_name)
                axis.setPen(self._pg.mkPen(palette.BORDER))
                axis.setTextPen(self._pg.mkPen(palette.PLOT_AXIS))
                axis.setStyle(maxTickLevel=0, maxTextLevel=0)
            plot.setLabel("bottom", self.tr("Volume (Å³/atom)"))
            plot.setLabel("left", self.tr("Relative energy (meV/atom)") if relative else self.tr("Energy (eV/atom)"))
            plot.setMinimumHeight(190)
            plot.setMinimumWidth(0)
            zero = min(summary["energies"]) if relative else 0.0
            scale = 1000.0 if relative else 1.0
            reference = [(value - zero) * scale for value in summary["energies"]]
            curve = plot.plot(summary["volumes"], reference,
                      pen=self._pg.mkPen(palette.PLOT_AXIS, width=1.5, style=Qt.PenStyle.DashLine),
                      symbol="o", symbolBrush=palette.PLOT_AXIS, symbolPen=palette.PLOT_AXIS,
                      symbolSize=5, name=self.tr("Reference"), antialias=True)
            self._register_curve(None, curve)
            for key, name, run, details in entries:
                if not run or run["status"] != "succeeded" or not details:
                    continue
                offset = details.get("offset_ev_atom", 0.0) if self.offset_checkbox.isChecked() else 0.0
                energies = [(value - zero - offset) * scale for value in details.get("predicted", [])]
                curve = plot.plot(details.get("volumes", []), energies,
                                  pen=self._series_pen(key), name=name, antialias=True)
                self._register_curve(key, curve)
            card_layout.addWidget(plot, 1)
            card.setMinimumHeight(290 if len(self._groups) > 1 else 390)
            self.eos_cards.append(card)
            self.eos_plots.append(plot)
        self._layout_cards()

    @staticmethod
    def _reset_plot_view(plot):
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
            curve = self.plot.plot(reference, predicted, pen=None, symbol="o", symbolBrush=color, symbolPen=color,
                                   symbolSize=6)
            self._register_curve(key, curve)
        if bounds:
            low, high = min(bounds), max(bounds)
            self._register_curve(None, self.plot.plot([low, high], [low, high], pen=pg.mkPen(palette.PLOT_LINE, style=Qt.PenStyle.DashLine)))


class ModelLibraryWidget(QWidget):
    """Suite selector, scorecard table and per-set details for captured models."""

    scorecardChanged = Signal()
    samplingWorkflowRequested = Signal(object)

    def __init__(self, parent=None, service: ValidationService | None = None):
        super().__init__(parent)
        self._service = service
        self._run_task: BackgroundTask | None = None
        self._run_total = 0
        self._library_root = None
        self._loaded = False
        self._displayed_suite_id = None
        self._draft: ModelDraft | None = None

        self.controls_panel = QFrame(self)
        self.controls_panel.setObjectName("validationControls")
        self.controls_panel.setStyleSheet(f"#validationControls {{ background: {palette.SURFACE}; border: 1px solid {palette.BORDER}; border-radius: 10px; }}")
        controls = QVBoxLayout(self.controls_panel)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setSpacing(8)
        fields = QHBoxLayout()
        fields.setSpacing(14)
        self.suite_combo = ComboBox(self)
        self.suite_combo.setMinimumWidth(150)
        self.suite_combo.currentIndexChanged.connect(self._on_suite_changed)
        self.refresh_button = TransparentToolButton(FluentIcon.SYNC.colored(palette.ACCENT, palette.ACCENT), self)
        self.refresh_button.setToolTip(self.tr("Refresh"))
        self.refresh_button.setAccessibleName(self.tr("Refresh"))
        self.refresh_button.clicked.connect(lambda: self.refresh())
        self.diagnostic_action = Action(FluentIcon.ADD.icon(color=palette.ACCENT), "新建无参考诊断…", self)
        self.edit_diagnostic_action = Action(FluentIcon.EDIT.icon(color=palette.ACCENT), "编辑所选诊断实验的副本…", self)
        self.edit_diagnostic_action.triggered.connect(self.edit_diagnostic_plan)
        self.save_plan_action = Action(FluentIcon.SAVE.icon(color=palette.ACCENT), "保存诊断方案…", self)
        self.diagnostic_action.triggered.connect(self.show_diagnostic_editor)
        self.save_plan_action.triggered.connect(self.save_diagnostic_plan)
        self.import_action = Action(QIcon(':/images/src/images/open.svg'), self.tr("Import reference dataset"), self)
        self.import_action.setToolTip(self.tr("Import reference dataset"))
        self.import_package_action = Action(FluentIcon.DOWNLOAD.icon(color=palette.ACCENT), self.tr("Import validation sharing package…"), self)
        self.export_package_action = Action(FluentIcon.SHARE.icon(color=palette.ACCENT), self.tr("Export validation sharing package…"), self)
        self.import_package_action.triggered.connect(lambda: self.import_validation_package())
        self.export_package_action.triggered.connect(lambda: self.export_validation_package())
        self.star_action = Action(FluentIcon.HEART.icon(color=palette.PLOT_SELECTED), self.tr("Star"), self)
        self.star_action.setCheckable(True)
        self.save_action = Action(QIcon(':/images/src/images/save.svg'), self.tr("Save to project…"), self)
        self.save_action.setToolTip(self.tr("Save to project…"))
        self.baseline_button = TransparentToolButton(FluentIcon.PIN.colored(palette.PLOT_CURRENT, palette.PLOT_CURRENT), self)
        self.baseline_button.setToolTip(self.tr("Set baseline model") + "\n" + self.tr("Set a saved version to compare model improvements and regressions."))
        self.baseline_button.setAccessibleName(self.tr("Set baseline model"))
        self.run_current_button = PrimaryPushButton(FluentIcon.PLAY, self.tr("Validate current version"), self)
        self.run_button = PushButton(QIcon(':/images/src/images/run.svg'), self.tr("Run all"), self)
        self.run_selected_button = PushButton(QIcon(':/images/src/images/run.svg'), self.tr("Validate selected versions"), self)
        self.run_selected_button.setToolTip(self.tr("Validate the current model and selected comparison models, including hidden curves."))
        self.run_selected_button.clicked.connect(self.run_selected)
        self.model_combo = ComboBox(self)
        self.model_combo.setMinimumWidth(130)
        self.comparison_picker = ComparisonPicker(self)
        self.comparison_picker.setMinimumWidth(130)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.comparison_picker.selectionChanged.connect(self._on_selection_changed)
        for title, combo, buttons, stretch in (
            (self.tr("Validation plan"), self.suite_combo, (self.refresh_button,), 4),
            (self.tr("Current model"), self.model_combo, (self.baseline_button,), 5),
            (self.tr("Compare with"), self.comparison_picker, (), 3),
        ):
            field = QHBoxLayout()
            field.setSpacing(6)
            caption = CaptionLabel(title, self)
            caption.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")
            field.addWidget(caption)
            field.addWidget(combo, 1)
            for button in buttons:
                button.setFixedSize(28, 28)
                field.addWidget(button)
            fields.addLayout(field, stretch)
        controls.addLayout(fields)
        self.import_action.triggered.connect(lambda: self.import_benchmark())
        self.star_action.triggered.connect(lambda: self.toggle_star())
        self.save_action.triggered.connect(lambda: self.save_current())
        self.baseline_button.clicked.connect(self.set_baseline)
        self.run_button.clicked.connect(self._on_run_clicked)
        self.run_current_button.clicked.connect(self.run_current)

        self.baseline_label = CaptionLabel(self)
        self.baseline_label.setWordWrap(True)
        self.baseline_label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(self.baseline_label, 1)
        actions.addWidget(self.run_button)
        actions.addWidget(self.run_selected_button)
        actions.addWidget(self.run_current_button)
        controls.addLayout(actions)
        self.property_combo = ComboBox(self)
        self.property_combo.currentIndexChanged.connect(lambda *_: self._load_groups())
        self.group_hint = CaptionLabel(self.tr("Each group has independent axes. Enlarge a chart or select rows in the error details to focus."), self)
        self.group_hint.hide()
        self.all_groups_button = TransparentPushButton(self.tr("Show all groups"), self)
        self.all_groups_button.clicked.connect(self._select_all_groups)
        self.all_groups_button.setToolTip(self.group_hint.text())
        self._benchmarks = []

        self.model = ScorecardTableModel(self)
        self.table = TableView(self)
        self.metrics_model = GroupMetricsTableModel(self.model, self)
        self.table.setModel(self.metrics_model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 420)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setMinimumHeight(130)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self._update_detail())
        self.empty_hint = CaptionLabel(self.table.viewport())
        self.empty_hint.setWordWrap(True)
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        hint_layout = QVBoxLayout(self.table.viewport())
        hint_layout.addWidget(self.empty_hint, 0, Qt.AlignmentFlag.AlignCenter)

        self.detail = RunDetailPanel(self)
        self.detail.header_layout.insertWidget(0, self.property_combo)
        self.detail.header_layout.insertWidget(2, self.all_groups_button)
        self.detail.comparisonRemoved.connect(self.comparison_picker.remove_id)
        self.detail.allGroupsRequested.connect(self._select_all_groups)
        self.detail.groupFocused.connect(lambda tag: self.table.selectRow(int(tag[1:]) - 1))
        self.detail.samplingRequested.connect(self.open_sampling)
        self.metrics_button = PushButton(FluentIcon.CHEVRON_RIGHT.colored(palette.PLOT_LOADED_OVERLAY, palette.PLOT_LOADED_OVERLAY), self.tr("Error details"), self)
        self.metrics_button.setCheckable(True)
        self.metrics_button.toggled.connect(self._toggle_metrics)
        self.table.hide()
        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.splitter.addWidget(self.detail)
        self.splitter.addWidget(self.table)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([440, 180])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        panel_row = QHBoxLayout()
        panel_row.setContentsMargins(12, 10, 12, 4)
        panel_row.addWidget(self.controls_panel)
        layout.addLayout(panel_row)
        legend = CaptionLabel(
            self.tr("Metrics are for the current model. Green / red compare against the baseline model; choosing a comparison curve does not change the baseline."),
            self,
        )
        legend.setContentsMargins(10, 0, 10, 4)
        legend.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        self.table.setToolTip(legend.text() + "\n" + self.group_hint.text())
        legend.hide()
        from NepTrainKit.ui.widgets.diagnostic_editor import DiagnosticEditor
        self.diagnostic_editor = DiagnosticEditor(self)
        self.diagnostic_editor.prepared.connect(self._prepare_diagnostic_plan)
        self.editor_scroll = QScrollArea(self)
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.editor_scroll.setWidget(self.diagnostic_editor)
        self.editor_scroll.hide()
        self.diagnostic_editor.hide()
        self.diagnostic_editor.installEventFilter(self)
        layout.addWidget(self.editor_scroll, 1)
        layout.addWidget(self.splitter, 1)
        metrics_footer = QHBoxLayout()
        metrics_footer.setContentsMargins(12, 4, 12, 8)
        metrics_footer.addWidget(self.metrics_button)
        metrics_footer.addStretch(1)
        layout.addLayout(metrics_footer)

    def _toggle_metrics(self, expanded):
        self.table.setVisible(expanded)
        self.metrics_button.setIcon((FluentIcon.CHEVRON_DOWN_MED if expanded else FluentIcon.CHEVRON_RIGHT).colored(
            palette.PLOT_LOADED_OVERLAY, palette.PLOT_LOADED_OVERLAY))
        if expanded:
            self.splitter.setSizes([max(260, self.splitter.height() - 180), 180])

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
            MessageManager.send_error_message(self.tr("Cannot open the model library: {error}").format(error=error))
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

    def _load_scorecard(self, model_id: int | None = None) -> None:
        suite_id = self.current_suite_id()
        service = self.service()
        self._benchmarks = service.suite_sets(suite_id) if suite_id is not None else []
        previous_kind = self.property_combo.currentData()
        kinds = list(dict.fromkeys(item["protocol"] for item in self._benchmarks))
        kinds.sort(key=lambda kind: kind != "eos")
        self.property_combo.blockSignals(True)
        self.property_combo.clear()
        for kind in kinds:
            self.property_combo.addItem({"eos": "EOS", "diagnostic": "无参考诊断"}.get(kind, self.tr("Energy / force / virial")), userData=kind)
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
        if model_id and all(row["model_id"] != model_id for row in card["rows"]) and any(
            model["id"] == model_id for model in service.models()
        ):
            card = service.scorecard(suite_id, [row["model_id"] for row in card["rows"]] + [model_id], draft=self._draft)
        self.model.set_scorecard(card)
        if suite_id is None:
            self.empty_hint.setText("从顶部打开菜单新建无参考诊断，或导入参考基准集。")
        elif not card["columns"]:
            self.empty_hint.setText(self.tr("This validation plan has no reference datasets yet."))
        elif not card["rows"]:
            self.empty_hint.setText(self.tr("Open a compatible NEP model to validate this plan. Reference data can already be previewed below."))
        self.empty_hint.setVisible(not self._benchmarks)
        suite = (card or {}).get("suite") or {}
        baseline = next((row for row in (card or {}).get("rows", []) if row["is_baseline"]), None)
        if suite_id is None:
            self.baseline_label.setText("从顶部打开菜单新建无参考诊断，或导入参考基准集。")
        elif baseline is None:
            self.baseline_label.setText(self.tr("Baseline model: not selected"))
        else:
            text = self.tr("Baseline model: {name}").format(name=baseline["name"])
            if suite.get("baseline_note"):
                text += f" — {suite['baseline_note']}"
            self.baseline_label.setText(text)
        comparison_ids = self.comparison_picker.selected_ids() if self._displayed_suite_id == suite_id else [suite.get("baseline_model_id")]
        self._displayed_suite_id = suite_id
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for index, row in enumerate(self.model._rows):
            label = self._model_label(index)
            self.model_combo.addItem(label, userData=row["model_id"])
        if not self.model._rows:
            self.model_combo.addItem("选择势版本后运行" if any(b["protocol"] == "diagnostic" for b in self._benchmarks) else self.tr("Reference preview only"), userData=None)
        self.model_combo.setEnabled(bool(self.model._rows))
        selected = self.model_combo.findData(model_id)
        if selected >= 0:
            self.model_combo.setCurrentIndex(selected)
        self.model_combo.blockSignals(False)
        self._update_comparison_options(comparison_ids)
        self._load_groups(update_detail=False)
        self._on_selection_changed()
        self.scorecardChanged.emit()

    def open_scorecard(self, suite_id: int | None, model_id: int | None = None, *, draft: ModelDraft | None = None) -> None:
        """Show ``suite_id`` (``None``: the last used suite) with ``model_id`` selected."""
        if self._library_root is not None and self._library_root != self.service().library.root:
            self.refresh()
        self._draft = draft
        if draft is not None:
            model_id = 0
            if suite_id is None:
                elements = set(parse_model_header(draft.data)[1])
                service = self.service()
                suite_id = next((suite["id"] for suite in service.suites() if any(
                    set(item["elements"]) <= elements for item in service.suite_sets(suite["id"])
                )), None)
        self.refresh(suite_id, model_id)

    # -- selection and details --------------------------------------------------
    def current_model_row(self) -> dict | None:
        index = self.model.row_of_model(self.model_combo.currentData())
        return self.model.row(index)

    def _model_label(self, index):
        row = self.model.row(index)
        label = self.model.data(self.model.index(index, 0))
        if sum(item["name"] == row["name"] and item.get("project_name") == row.get("project_name") for item in self.model._rows) > 1:
            label += f" · #{row['model_id']}"
        return label

    def _update_comparison_options(self, selected_ids=None):
        if selected_ids is None:
            selected_ids = self.comparison_picker.selected_ids()
        options = []
        for index, row in enumerate(self.model._rows):
            if row["model_id"] in (0, self.model_combo.currentData()):
                continue
            cells = list({column["set_id"]: cell for column, cell in zip(self.model._columns, row.get("cells", []))}.values())
            ready = sum(cell["status"] == "succeeded" for cell in cells)
            status = self.tr("Not run") if not cells or all(cell["status"] == "missing" for cell in cells) else self.tr("{ready}/{total} validations completed").format(ready=ready, total=len(cells))
            label = self._model_label(index)
            options.append((row["model_id"], label, status))
        self.comparison_picker.set_options(options, selected_ids)

    def selected_model_ids(self):
        current = self.current_model_row()
        return list(dict.fromkeys(([current["model_id"]] if current else []) + self.comparison_picker.selected_ids()))

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
        self.metrics_model.set_groups(benchmarks, self.model.row_of_model(self.model_combo.currentData()), kind)
        if kind in ("eos", "diagnostic"):
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            if selected and selected < {b["id"] for b in benchmarks}:
                from PySide6.QtCore import QItemSelectionModel
                for i, benchmark in enumerate(benchmarks):
                    if benchmark["id"] in selected:
                        selection.select(self.metrics_model.index(i, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
            else:
                self.table.selectAll()
        else:
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            if benchmarks:
                self.table.selectRow(next((i for i, b in enumerate(benchmarks) if b["id"] in selected), 0))
        selection.blockSignals(False)
        self.all_groups_button.setVisible(kind in ("eos", "diagnostic"))
        self.metrics_button.setText("响应与运行详情" if kind == "diagnostic" else self.tr("Error details"))
        if update_detail:
            self._update_detail()

    def _select_all_groups(self):
        self.table.selectAll()
        self._update_detail()

    def _on_selection_changed(self) -> None:
        row = self.current_model_row()
        has_suite = self.current_suite_id() is not None
        saved = row is not None and not row.get("temporary")
        diagnostic = any(b["protocol"] == "diagnostic" for b in self._benchmarks)
        temporary_plan = has_suite and self.current_suite_id() < 0
        editing = not self.editor_scroll.isHidden()
        self.edit_diagnostic_action.setEnabled(diagnostic and self._run_task is None and not editing)
        self.save_plan_action.setEnabled(temporary_plan and self._run_task is None and not editing)
        self.diagnostic_action.setEnabled(self._run_task is None)
        self.export_package_action.setEnabled(has_suite and not diagnostic and bool(self.metrics_model.benchmarks) and self._run_task is None)
        self.export_package_action.setToolTip("无参考诊断分享将在下一阶段支持；当前可保存本地方案。" if diagnostic else self.export_package_action.text())
        self.save_action.setEnabled(row is not None and self._run_task is None)
        self.star_action.setEnabled(saved)
        self.star_action.setText(self.tr("Unstar") if row and row["starred"] else self.tr("Star"))
        self.star_action.setToolTip(self.star_action.text())
        self.star_action.setChecked(bool(row and row["starred"]))
        self.baseline_button.setEnabled(saved and has_suite)
        self.run_current_button.setEnabled(row is not None and has_suite and self._run_task is None)
        self.run_selected_button.setEnabled(bool(self.selected_model_ids()) and has_suite and self._run_task is None)
        if self._run_task is None:
            self.run_button.setEnabled(has_suite)
        self._update_detail()

    def _update_detail(self) -> None:
        benchmarks = self.metrics_model.benchmarks
        if not benchmarks:
            self.detail.show_runs(None, [])
            return
        selected = sorted(index.row() for index in self.table.selectionModel().selectedRows())
        focused = len(selected) < len(benchmarks)
        self.all_groups_button.setEnabled(focused)
        self.all_groups_button.setVisible(focused or self.metrics_model.kind == "eos")
        self.all_groups_button.setText(self.tr("Back to all groups") if focused else self.tr("Show all groups"))
        self.all_groups_button.setIcon(FluentIcon.RETURN.colored(
            palette.PLOT_LOADED_OVERLAY, palette.PLOT_LOADED_OVERLAY) if focused else None)
        rows = []
        for model_id in self.selected_model_ids():
            row = self.model.row(self.model.row_of_model(model_id))
            if row and row not in rows:
                rows.append(row)
        service = self.service()
        groups = []
        for index in selected:
            benchmark = benchmarks[index]
            entries = []
            for row in rows:
                column = self.model.first_column_of_set(benchmark["id"])
                cell = row["cells"][column - 1] if column > 0 else None
                run = service.latest_run(row["model_id"], benchmark["id"], protocol_version=get_protocol(benchmark["protocol"]).version) if row["model_id"] else None
                if cell and cell["status"] == "missing":
                    run = None
                elif run is not None and cell and run["status"] != cell["status"]:
                    run = dict(run, status=cell["status"], error=cell["error"])
                details = {}
                if run is not None and run["status"] == "succeeded":
                    try:
                        details = service.run_details(run).get("details", {})
                    except OSError:
                        pass
                entries.append((row["model_id"], self._model_label(self.model.row_of_model(row["model_id"])), run, details))
            groups.append((f"G{index + 1}", benchmark, entries))
        if not groups:
            self.detail.show_runs(None, [])
        elif self.metrics_model.kind in ("eos", "diagnostic"):
            self.detail.show_groups(groups, total_groups=len(benchmarks))
        elif groups:
            self.detail.show_runs(groups[0][1], groups[0][2])
        else:
            self.detail.show_runs(None, [])

    def eventFilter(self, watched, event):
        from PySide6.QtCore import QEvent
        if (watched is getattr(self, "diagnostic_editor", None)
                and event.type() == QEvent.Type.Hide and watched.isHidden()):
            self.editor_scroll.hide()
            self.splitter.show()
            self.metrics_button.show()
            self.controls_panel.setEnabled(True)
            self.controls_panel.show()
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
        self.splitter.hide()
        self.metrics_button.hide()
        self.editor_scroll.show()
        self.diagnostic_editor.show()

    def edit_diagnostic_plan(self):
        from NepTrainKit.core.validation.protocols.diagnostic import experiment
        if self._run_task is not None:
            return
        try:
            selected = [self.metrics_model.benchmarks[index.row()] for index in self.table.selectionModel().selectedRows()]
            experiments = [experiment(self.service().set_frames(item["id"])) for item in selected if item["protocol"] == "diagnostic"]
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
            self.refresh(suite["id"])
            if not temporary:
                MessageManager.send_success_message("诊断方案已保存，重启后可继续使用；运行结果将自动保存。")
        except Exception as exc:
            self.diagnostic_editor.preview.setText(str(exc))
            MessageManager.send_error_message(str(exc))

    def save_diagnostic_plan(self):
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
        from NepTrainKit.core.validation.sampling import sampling_payload
        service = self._view_service()
        if service is None:
            return None
        try:
            benchmark = service.get_set(set_id)
            if benchmark["protocol"] != "eos":
                return None
            row = self.current_model_row()
            model = service.get_model(row["model_id"]) if row and row["model_id"] else None
            run = service.latest_run(model["id"], set_id, protocol_version=get_protocol("eos").version) if model else None
            details = service.run_details(run).get("details", {}) if run and run["status"] == "succeeded" else None
            payload = sampling_payload(service.set_frames(set_id), benchmark, details=details,
                                       model=model, run=run if details else None)
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
        service.star_model(row["model_id"], not row["starred"])
        self.refresh(model_id=row["model_id"])

    def set_baseline(self) -> None:
        service = self._view_service()
        if service is None:
            return
        row = self.current_model_row()
        suite_id = self.current_suite_id()
        if row is None or suite_id is None:
            return
        box = GetStrMessageBox(self.window(), self.tr("Note for the baseline model (optional). This changes the comparison version only; reference datasets and original model files stay unchanged."))
        box.yesButton.setText(self.tr("Set baseline model"))
        if not box.exec():
            return
        service.set_baseline(suite_id, row["model_id"], box.lineEdit.text())
        self.refresh(suite_id, row["model_id"])

    def _confirm_large_copy(self, error: SetTooLargeError) -> bool:
        box = MessageBox(
            self.tr("Large benchmark file"),
            self.tr("The file is {size:.1f} MB, above the {limit:.0f} MB limit. Copy it into the library anyway?").format(
                size=error.size / 1e6, limit=error.limit / 1e6
            ),
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
            dialog = ExportValidationPackageDialog(self.window(), self._view_service(), suite_id, self.selected_model_ids())
            if not dialog.exec():
                return
            path, _ = QFileDialog.getSaveFileName(self, self.tr("Export validation sharing package…"), "validation.ntkpack", "NepTrainKit (*.ntkpack)")
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
            suite = import_package(service, dialog.preview, name=dialog.name_edit.text(), project_name=dialog.project_combo.text())
            self.open_scorecard(suite["id"], suite["baseline_model_id"])
            self.scorecardChanged.emit()
            MessageManager.send_info_message(self.tr("Sharing package imported as a new validation suite."))
        except (ValueError, OSError) as error:
            MessageManager.send_error_message(str(error))

    def import_benchmark(self) -> list[dict] | None:
        """Import one or more reference datasets and return their records."""
        service = self.service()
        dialog = BenchmarkImportDialog(self.window(), [s for s in service.suites() if s["id"] > 0], self.suite_combo.currentText() if (self.current_suite_id() or 0) > 0 else "")
        if not dialog.exec():
            return None
        try:
            records = dialog.import_into(service, self._confirm_large_copy)
        except (SetRejectedError, ValueError, OSError) as error:
            MessageManager.send_error_message(self.tr("Import failed: {error}").format(error=error))
            return None
        if not records:
            return None
        MessageManager.send_success_message(self.tr("Imported {count} reference datasets").format(count=len(records)))
        suite_name = dialog.values()["suite"]
        suite = next((item for item in service.suites() if item["name"] == suite_name), None)
        self.refresh(suite["id"] if suite else None)
        self.property_combo.setCurrentIndex(self.property_combo.findData(records[0]["protocol"]))
        return records

    def run_jobs(self, model_id: int | None = None) -> list[tuple[int, str]]:
        """Compatible suite jobs for one version, or all captured versions."""
        suite_id = self.current_suite_id()
        if suite_id is None:
            return []
        service = self.service()
        sets = service.suite_sets(suite_id)
        return [
            (model["id"], benchmark["id"])
            for model in service.models()
            if model_id is None or model["id"] == model_id
            for benchmark in sets
            if set(benchmark["elements"]) <= set(model["elements"] or ())
        ]

    def _on_run_clicked(self) -> None:
        if self._run_task is not None:
            self._run_task.stop_work()
            self.run_button.setEnabled(False)
            return
        self.run_all()

    def run_all(self) -> BackgroundTask | None:
        return self._run_validation()

    def run_current(self) -> BackgroundTask | None:
        row = self.current_model_row()
        return self._run_validation([row["model_id"]]) if row else None

    def run_selected(self) -> BackgroundTask | None:
        return self._run_validation(self.selected_model_ids())

    def _run_validation(self, model_ids: list[int] | None = None) -> BackgroundTask | None:
        if self._run_task is not None or self.current_suite_id() is None:
            return None
        service = self._view_service()
        if service is None:
            return None
        if self._draft is not None and (model_ids is None or 0 in model_ids):
            try:
                record = service.capture_model(self._draft, temporary=True)
            except Exception as error:
                MessageManager.send_error_message(str(error))
                return None
            self._draft = None
            self.refresh(model_id=record["id"])
            if model_ids is not None:
                model_ids = list(dict.fromkeys(record["id"] if key == 0 else key for key in model_ids))
        jobs = [job for job in self.run_jobs() if model_ids is None or job[0] in model_ids]
        if not jobs:
            MessageManager.send_info_message(
                self.tr("No captured model covers the elements of this suite. "
                        "Open a model in NEP Dataset Display to capture it.")
            )
            return None

        def evaluate_all():
            for model_id, set_id in jobs:
                yield service.evaluate(model_id, set_id)

        self._run_total = len(jobs)
        task = BackgroundTask(self, show_tip=True, title=self.tr("Running validation"))
        task.progressSignal.connect(self._on_run_progress)
        task.finished.connect(self._on_run_finished)
        self._run_task = task
        self.run_current_button.setEnabled(False)
        self.run_selected_button.setEnabled(False)
        self.export_package_action.setEnabled(False)
        self.save_action.setEnabled(False)
        self.save_plan_action.setEnabled(False)
        self.diagnostic_action.setEnabled(False)
        self.edit_diagnostic_action.setEnabled(False)
        self.diagnostic_editor.setEnabled(False)
        self.suite_combo.setEnabled(False)
        self.run_button.setText(self.tr("Cancel"))
        self.run_button.setIcon(QIcon(':/images/src/images/stop.svg'))
        task.start_work(evaluate_all)
        return task

    def _on_run_progress(self, index: int) -> None:
        tip = getattr(self._run_task, "tip", None)
        if tip is not None:
            tip.setContent(self.tr("{done} / {total} evaluations").format(done=index + 1, total=self._run_total))

    def _on_run_finished(self) -> None:
        task, self._run_task = self._run_task, None
        self.diagnostic_editor.setEnabled(True)
        self.suite_combo.setEnabled(True)
        self.run_button.setText(self.tr("Run all"))
        self.run_button.setIcon(QIcon(':/images/src/images/run.svg'))
        self.run_button.setEnabled(True)
        if task is not None and task.outcome == "failed":
            MessageManager.send_error_message(self.tr("Validation failed: {error}").format(error=task.error_message))
        current = self.current_model_row()
        self.refresh(model_id=current["model_id"] if current else None)
        if task is not None:
            task.deleteLater()
