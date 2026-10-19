"""Model library: scorecard of captured models against benchmark suites."""

from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QHeaderView, QSplitter, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    FluentIcon,
    MessageBox,
    PrimaryPushButton,
    PushButton,
    TableView,
    ToolButton,
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
from NepTrainKit.ui.widgets.validation_dialogs import BenchmarkImportDialog, save_model_to_project

CONFIG_SUITE = "suite"
VERDICT_COLORS = {
    "better": QColor(47, 158, 68, 70),
    "worse": QColor(224, 49, 49, 80),
    "same": QColor(134, 142, 150, 40),
}
SERIES_COLORS = ("#2F6FA8", "#F76707", "#2F9E44", "#AE3EC9", "#E03131", "#1098AD")


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

    def __init__(self, parent=None):
        super().__init__(parent)
        import NepTrainKit.ui.canvas.pyqtgraph  # noqa: F401 - shared pyqtgraph configuration
        import pyqtgraph as pg

        self._pg = pg
        self._args = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        header = QHBoxLayout()
        self.title_label = BodyLabel(self)
        self.offset_checkbox = CheckBox(self.tr("Remove constant offset"), self)
        self.offset_checkbox.setChecked(True)
        self.offset_checkbox.setToolTip(self.tr("Shift each model curve by its mean error, as the shape metrics do"))
        self.offset_checkbox.stateChanged.connect(lambda _state: self._redraw())
        header.addWidget(self.title_label, 1)
        header.addWidget(self.offset_checkbox)
        layout.addLayout(header)
        self.info_label = CaptionLabel(self)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        layout.addWidget(self.info_label)
        self.plot = pg.PlotWidget(self)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        layout.addWidget(self.plot, 1)
        self.show_runs(None, [])

    def show_runs(self, benchmark: dict | None, entries: list[tuple[str, dict | None, dict]]) -> None:
        """``entries`` are ``(model name, run or None, run details)`` in overlay order."""
        self._args = (benchmark, entries)
        self._redraw()

    def _redraw(self) -> None:
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
        for name, run, _details in entries:
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
        renderer = {"eos": self._plot_eos, "efv": self._plot_efv}.get(protocol.kind)
        if renderer is not None:
            renderer(benchmark, [(name, details) for name, run, details in entries if details])

    def _plot_eos(self, benchmark: dict, curves: list[tuple[str, dict]]) -> None:
        pg = self._pg
        summary = benchmark.get("summary") or {}
        self.plot.setLabel("bottom", self.tr("Volume (Å³/atom)"))
        self.plot.setLabel("left", self.tr("Energy (eV/atom)"))
        if summary.get("volumes"):
            self.plot.plot(
                summary["volumes"], summary["energies"], pen=pg.mkPen("k", width=2),
                symbol="o", symbolBrush="k", symbolSize=7,
                name=self.tr("Reference curve"),
            )
        aligned = self.offset_checkbox.isChecked()
        for index, (name, details) in enumerate(curves):
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            offset = details.get("offset_ev_atom", 0.0) if aligned else 0.0
            energies = [value - offset for value in details.get("predicted", [])]
            self.plot.plot(
                details.get("volumes", []), energies, pen=pg.mkPen(color, width=2),
                symbol="t", symbolBrush=color, symbolPen=color, symbolSize=7, name=name,
            )

    def _plot_efv(self, _benchmark: dict, runs: list[tuple[str, dict]]) -> None:
        pg = self._pg
        self.plot.setLabel("bottom", self.tr("Reference energy (eV/atom)"))
        self.plot.setLabel("left", self.tr("Predicted energy (eV/atom)"))
        bounds = []
        for index, (name, details) in enumerate(runs):
            energy = details.get("energy") or {}
            reference, predicted = energy.get("reference", []), energy.get("predicted", [])
            bounds += reference + predicted
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            self.plot.plot(reference, predicted, pen=None, symbol="o", symbolBrush=color, symbolPen=color,
                           symbolSize=6, name=name)
        if bounds:
            low, high = min(bounds), max(bounds)
            self.plot.plot([low, high], [low, high], pen=pg.mkPen(palette.PLOT_LINE, style=Qt.PenStyle.DashLine))


class ModelLibraryWidget(QWidget):
    """Suite selector, scorecard table and per-set details for captured models."""

    scorecardChanged = Signal()

    def __init__(self, parent=None, service: ValidationService | None = None):
        super().__init__(parent)
        self._service = service
        self._run_task: BackgroundTask | None = None
        self._run_total = 0
        self._library_root = None
        self._draft: ModelDraft | None = None

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 8, 10, 4)
        toolbar.setSpacing(6)
        toolbar.addWidget(BodyLabel(self.tr("Validation plan"), self))
        self.suite_combo = ComboBox(self)
        self.suite_combo.setMinimumWidth(200)
        self.suite_combo.currentIndexChanged.connect(self._on_suite_changed)
        toolbar.addWidget(self.suite_combo)
        self.refresh_button = ToolButton(FluentIcon.SYNC, self)
        self.refresh_button.setToolTip(self.tr("Refresh"))
        self.refresh_button.clicked.connect(lambda: self.refresh())
        toolbar.addWidget(self.refresh_button)
        toolbar.addStretch(1)
        self.import_button = PushButton(FluentIcon.ADD, self.tr("Import reference dataset"), self)
        self.star_button = PushButton(FluentIcon.HEART, self.tr("Star"), self)
        self.save_button = ToolButton(FluentIcon.SAVE, self)
        self.save_button.setToolTip(self.tr("Save to project…"))
        self.save_button.setAccessibleName(self.tr("Save to project…"))
        self.baseline_button = PushButton(FluentIcon.PIN, self.tr("Set baseline model"), self)
        self.run_current_button = PrimaryPushButton(FluentIcon.PLAY, self.tr("Validate current version"), self)
        self.run_button = PushButton(FluentIcon.PLAY, self.tr("Run all"), self)
        toolbar.addWidget(self.import_button)
        model_actions = QHBoxLayout()
        model_actions.setContentsMargins(10, 4, 10, 4)
        model_actions.addWidget(BodyLabel(self.tr("Model versions"), self))
        model_actions.addStretch(1)
        for button in (self.save_button, self.star_button, self.baseline_button, self.run_current_button, self.run_button):
            model_actions.addWidget(button)
        self.import_button.clicked.connect(self.import_benchmark)
        self.star_button.clicked.connect(self.toggle_star)
        self.save_button.clicked.connect(self.save_current)
        self.baseline_button.clicked.connect(self.set_baseline)
        self.run_button.clicked.connect(self._on_run_clicked)
        self.run_current_button.clicked.connect(self.run_current)

        self.baseline_label = CaptionLabel(self)
        self.baseline_label.setWordWrap(True)
        self.baseline_label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        reference_row = QHBoxLayout()
        reference_row.setContentsMargins(10, 4, 10, 4)
        reference_row.addWidget(BodyLabel(self.tr("Reference dataset"), self))
        self.reference_combo = ComboBox(self)
        self.reference_combo.setMinimumWidth(240)
        self.reference_combo.currentIndexChanged.connect(self._on_reference_changed)
        reference_row.addWidget(self.reference_combo, 1)
        reference_hint = CaptionLabel(self.tr("Fixed reference data · reused across model versions"), self)
        reference_row.addWidget(reference_hint)

        self.model = ScorecardTableModel(self)
        self.table = TableView(self)
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self._on_selection_changed())
        self.table.selectionModel().currentChanged.connect(lambda *_: self._on_selection_changed())
        self.empty_hint = CaptionLabel(self.table.viewport())
        self.empty_hint.setWordWrap(True)
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        hint_layout = QVBoxLayout(self.table.viewport())
        hint_layout.addWidget(self.empty_hint, 0, Qt.AlignmentFlag.AlignCenter)

        self.detail = RunDetailPanel(self)
        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.splitter.addWidget(self.table)
        self.splitter.addWidget(self.detail)
        self.splitter.setSizes([320, 360])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(toolbar)
        layout.addLayout(model_actions)
        legend = CaptionLabel(
            self.tr("Green: better than the baseline model · Red: worse · Grey: within tolerance. "
                    "Ctrl+click cells to overlay several models."),
            self,
        )
        legend.setContentsMargins(10, 0, 10, 4)
        legend.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        layout.addWidget(legend)
        layout.addWidget(self.baseline_label)
        layout.addLayout(reference_row)
        layout.addWidget(self.splitter, 1)
        self.baseline_label.setContentsMargins(10, 0, 10, 4)

    # -- data -----------------------------------------------------------------
    def service(self) -> ValidationService:
        # A new service follows a changed library location without a restart.
        return self._service or ValidationService()

    def current_suite_id(self) -> int | None:
        return self.suite_combo.currentData()

    def showEvent(self, event):
        super().showEvent(event)
        if self._run_task is None:
            self.refresh()

    def refresh(self, suite_id: int | None = None, model_id: int | None = None) -> None:
        try:
            service = self.service()
            if self._library_root is not None and self._library_root != service.library.root:
                suite_id, model_id = None, None
                self._draft = None
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
        except Exception as error:  # noqa: BLE001 - a broken library must not break the page
            MessageManager.send_error_message(self.tr("Cannot open the model library: {error}").format(error=error))
            self.suite_combo.blockSignals(True)
            self.suite_combo.clear()
            self.suite_combo.blockSignals(False)
            self.reference_combo.clear()
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

    def _on_suite_changed(self, _index: int) -> None:
        suite_id = self.current_suite_id()
        if suite_id is not None:
            Config.set("validation", CONFIG_SUITE, suite_id)
        self._load_scorecard()

    def _load_scorecard(self, model_id: int | None = None) -> None:
        suite_id = self.current_suite_id()
        service = self.service()
        previous_set = self.reference_combo.currentData()
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        for benchmark in service.suite_sets(suite_id) if suite_id is not None else []:
            kind = "EOS" if benchmark["protocol"] == "eos" else self.tr("Energy / force / virial")
            self.reference_combo.addItem(f"{benchmark['name']} · {kind}", userData=benchmark["id"])
        previous_index = self.reference_combo.findData(previous_set)
        if previous_index >= 0:
            self.reference_combo.setCurrentIndex(previous_index)
        self.reference_combo.setEnabled(self.reference_combo.count() > 0)
        self.reference_combo.blockSignals(False)
        if self._draft is not None:
            existing = service.find_model_by_hash(self._draft.content_hash)
            if existing:
                model_id = existing["id"]
                self._draft = None
        if suite_id is None:
            card = service.scorecard(None, [model_id] if model_id else [], draft=self._draft) if model_id or self._draft else None
        else:
            card = service.scorecard(suite_id, draft=self._draft)
        if card is not None and model_id and all(row["model_id"] != model_id for row in card["rows"]):
            card = service.scorecard(suite_id, [row["model_id"] for row in card["rows"]] + [model_id], draft=self._draft)
        self.model.set_scorecard(card)
        if card is None:
            self.empty_hint.setText(self.tr("Import a reference dataset into a validation plan to start validating models."))
        elif not card["columns"]:
            self.empty_hint.setText(self.tr("This validation plan has no reference datasets yet."))
        elif not card["rows"]:
            self.empty_hint.setText(self.tr("Open a compatible NEP model to validate this plan. Reference data can already be previewed below."))
        self.empty_hint.setVisible(self.model.rowCount() == 0)
        suite = (card or {}).get("suite") or {}
        baseline = next((row for row in (card or {}).get("rows", []) if row["is_baseline"]), None)
        if suite_id is None:
            self.baseline_label.setText(self.tr("Import a reference dataset into a validation plan to start validating models."))
        elif baseline is None:
            self.baseline_label.setText(self.tr("Baseline model: not selected. Set a saved version to compare model improvements and regressions."))
        else:
            text = self.tr("Baseline model: {name}").format(name=baseline["name"])
            if suite.get("baseline_note"):
                text += f" — {suite['baseline_note']}"
            self.baseline_label.setText(text)
        row = self.model.row_of_model(model_id) if model_id is not None else -1
        if row >= 0:
            self.table.setCurrentIndex(self.model.index(row, 1 if self.model.columnCount() > 1 else 0))
        self._on_selection_changed()
        self.scorecardChanged.emit()

    def open_scorecard(self, suite_id: int | None, model_id: int | None = None, *, draft: ModelDraft | None = None) -> None:
        """Show ``suite_id`` (``None``: the last used suite) with ``model_id`` selected."""
        if self._library_root != self.service().library.root:
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
    def selected_rows(self) -> list[int]:
        rows = {index.row() for index in self.table.selectionModel().selectedIndexes()}
        current = self.table.currentIndex()
        if current.isValid():
            rows.add(current.row())
        return sorted(rows)

    def current_model_row(self) -> dict | None:
        current = self.table.currentIndex()
        return self.model.row(current.row()) if current.isValid() else None

    def _on_selection_changed(self) -> None:
        row = self.current_model_row()
        has_suite = self.current_suite_id() is not None
        saved = row is not None and not row.get("temporary")
        self.save_button.setEnabled(row is not None and self._run_task is None)
        self.star_button.setEnabled(saved)
        self.star_button.setText(self.tr("Unstar") if row and row["starred"] else self.tr("Star"))
        self.baseline_button.setEnabled(saved and has_suite)
        self.run_current_button.setEnabled(row is not None and has_suite and self._run_task is None)
        if self._run_task is None:
            self.run_button.setEnabled(has_suite)
        self._update_detail()

    def _update_detail(self) -> None:
        current = self.table.currentIndex()
        column = self.model.column(current.column()) if current.isValid() else None
        if column is None and current.isValid():
            column = self.model.column(1)
        if column is None:
            set_id = self.reference_combo.currentData()
            self.detail.show_runs(self.service().get_set(set_id) if set_id else None, [])
            return
        self.reference_combo.blockSignals(True)
        self.reference_combo.setCurrentIndex(self.reference_combo.findData(column["set_id"]))
        self.reference_combo.blockSignals(False)
        service = self.service()
        entries = []
        for row_index in self.selected_rows():
            row = self.model.row(row_index)
            cell = row["cells"][self.model.first_column_of_set(column["set_id"]) - 1]
            run = service.latest_run(row["model_id"], column["set_id"], protocol_version=column["protocol_version"])
            if run is not None and run["status"] != cell["status"]:
                run = dict(run, status=cell["status"], error=cell["error"])
            details = {}
            if run is not None and run["status"] == "succeeded":
                try:
                    details = service.run_details(run).get("details", {})
                except OSError:
                    details = {}
            if cell["status"] == "missing":
                run = None
            entries.append((row["name"], run, details))
        self.detail.show_runs(service.get_set(column["set_id"]), entries)

    def _on_reference_changed(self, _index: int) -> None:
        set_id = self.reference_combo.currentData()
        if set_id is None:
            self.detail.show_runs(None, [])
            return
        column = self.model.first_column_of_set(set_id)
        current = self.table.currentIndex()
        if current.isValid() and column >= 0:
            self.table.setCurrentIndex(self.model.index(current.row(), column))
            self._update_detail()
        else:
            self.detail.show_runs(self.service().get_set(set_id), [])

    # -- actions --------------------------------------------------------------
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

    def import_benchmark(self) -> list[dict] | None:
        """Import one or more reference datasets and return their records."""
        service = self.service()
        dialog = BenchmarkImportDialog(self.window(), service.suites(), self.suite_combo.currentText())
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
        self.reference_combo.setCurrentIndex(self.reference_combo.findData(records[0]["id"]))
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
        return self._run_validation(row["model_id"]) if row else None

    def _run_validation(self, model_id: int | None = None) -> BackgroundTask | None:
        if self._run_task is not None or self.current_suite_id() is None:
            return None
        service = self._view_service()
        if service is None:
            return None
        if self._draft is not None and model_id in (None, 0):
            try:
                record = service.capture_model(self._draft, temporary=True)
            except Exception as error:
                MessageManager.send_error_message(str(error))
                return None
            self._draft = None
            self.refresh(model_id=record["id"])
            if model_id == 0:
                model_id = record["id"]
        jobs = self.run_jobs(model_id)
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
        self.save_button.setEnabled(False)
        self.run_button.setText(self.tr("Cancel"))
        self.run_button.setIcon(FluentIcon.CLOSE)
        task.start_work(evaluate_all)
        return task

    def _on_run_progress(self, index: int) -> None:
        tip = getattr(self._run_task, "tip", None)
        if tip is not None:
            tip.setContent(self.tr("{done} / {total} evaluations").format(done=index + 1, total=self._run_total))

    def _on_run_finished(self) -> None:
        task, self._run_task = self._run_task, None
        self.run_button.setText(self.tr("Run all"))
        self.run_button.setIcon(FluentIcon.PLAY)
        self.run_button.setEnabled(True)
        if task is not None and task.outcome == "failed":
            MessageManager.send_error_message(self.tr("Validation failed: {error}").format(error=task.error_message))
        current = self.current_model_row()
        self.refresh(model_id=current["model_id"] if current else None)
        if task is not None:
            task.deleteLater()
