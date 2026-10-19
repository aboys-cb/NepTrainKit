"""Dialogs for the model validation library."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
from html import escape

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QAbstractItemView, QBoxLayout, QFileDialog, QFormLayout, QHeaderView, QHBoxLayout, QLayout, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    EditableComboBox,
    FluentIcon,
    LineEdit,
    MessageBoxBase,
    PushButton,
    ScrollArea,
    SubtitleLabel,
    TableWidget,
    TransparentPushButton,
)

from NepTrainKit import palette
from NepTrainKit.config import Config
from NepTrainKit.core.validation import Library, SetTooLargeError, ValidationService
from NepTrainKit.core.validation.library import ENV_VAR, CONFIG_LIBRARY_PATH, CONFIG_SECTION, sha256_bytes
from NepTrainKit.core.validation.grouping import eos_frame_groups, eos_group_fields
from NepTrainKit.core.validation.protocols import Issue, get_protocol, read_frames
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.messages import translate_runtime_message


class SaveModelDialog(MessageBoxBase):
    """Choose an existing/new project and a version name before writing a snapshot."""

    def __init__(self, parent, service, name: str):
        super().__init__(parent)
        self.service = service
        self.viewLayout.addWidget(SubtitleLabel(self.tr("Save model to project"), self))
        self.location_label = BodyLabel(self.tr("Library: {path}").format(path=service.library.root), self)
        self.location_label.setWordWrap(True)
        location_row = QHBoxLayout()
        location_row.addWidget(self.location_label, 1)
        self.location_button = PushButton(self.tr("Change…"), self)
        self.location_button.clicked.connect(self._choose_library)
        if os.environ.get(ENV_VAR):
            self.location_button.setEnabled(False)
            self.location_button.setToolTip(self.tr("Library location is controlled by NEPTRAINKIT_LIBRARY"))
        location_row.addWidget(self.location_button)
        self.viewLayout.addLayout(location_row)
        self.project_combo = EditableComboBox(self)
        self.project_combo.setPlaceholderText(self.tr("Choose a project or type a new name"))
        projects = service.projects()
        for project in projects:
            self.project_combo.addItem(project["name"])
        previous = Config.get("validation", "last_save_project", "")
        self.project_combo.setText(previous if any(p["name"] == previous for p in projects) else "")
        self.name_edit = LineEdit(self)
        self.name_edit.setMaxLength(80)
        self.name_edit.setText(name)
        form = QFormLayout()
        form.addRow(BodyLabel(self.tr("Project"), self), self.project_combo)
        form.addRow(BodyLabel(self.tr("Version name"), self), self.name_edit)
        self.viewLayout.addLayout(form)
        self.yesButton.setText(self.tr("Save"))
        self.cancelButton.setText(self.tr("Cancel"))
        self.project_combo.textChanged.connect(self._validate)
        self.name_edit.textChanged.connect(self._validate)
        self._validate()
        self.widget.setMinimumWidth(480)

    def _choose_library(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr("Choose model library"), str(self.service.library.root))
        if not folder:
            return
        try:
            service = ValidationService(Library(folder))
            projects = service.projects()
        except Exception as error:
            from NepTrainKit.core import MessageManager
            MessageManager.send_error_message(str(error))
            return
        self.service = service
        self.location_label.setText(self.tr("Library: {path}").format(path=service.library.root))
        self.project_combo.clear()
        for project in projects:
            self.project_combo.addItem(project["name"])
        self.project_combo.setText("")
        self._validate()

    def _validate(self, *_args):
        project = self.project_combo.text().strip()
        self.yesButton.setEnabled(bool(project and len(project) <= 200 and self.name_edit.text().strip()))


def save_model_to_project(parent, service, draft, *, name=None):
    dialog = SaveModelDialog(parent, service, name or draft.name)
    if not dialog.exec():
        return None
    record = dialog.service.save_model(draft, project_name=dialog.project_combo.text(), name=dialog.name_edit.text())
    if dialog.service.library.root != service.library.root:
        Config.set(CONFIG_SECTION, CONFIG_LIBRARY_PATH, str(dialog.service.library.root))
    Config.set("validation", "last_save_project", dialog.project_combo.text().strip())
    record["library_root"] = str(dialog.service.library.root)
    return record


class BenchmarkImportDialog(MessageBoxBase):
    """Choose a property, preview its reference data and optionally add notes."""

    def __init__(self, parent=None, suites: list[dict] | tuple = (), suite_name: str = ""):
        super().__init__(parent)
        self.frames = None
        self.issues: list[Issue] = []
        self._read_error = ""
        self._loaded_bytes = None
        self.reference_summary = None
        self.eos_groups = []
        self._group_error = ""

        self.titleLabel = SubtitleLabel(self.tr("Import reference dataset"), self)
        self.viewLayout.addWidget(self.titleLabel)
        self.body_scroll = ScrollArea(self.widget)
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.body_scroll.viewport().setStyleSheet("background: transparent;")
        body = QWidget()
        body.setObjectName("benchmarkImportBody")
        body.setStyleSheet("#benchmarkImportBody { background: transparent; }")
        columns = QBoxLayout(QBoxLayout.Direction.LeftToRight, body)
        columns.setContentsMargins(0, 0, 12, 0)
        columns.setSpacing(24)
        columns.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self._columns = columns
        self.body_scroll.setWidget(body)
        self.viewLayout.addWidget(self.body_scroll, 1)
        self.fields_panel = QWidget(body)
        self.preview_panel = QWidget(body)
        fields = QVBoxLayout(self.fields_panel)
        fields.setContentsMargins(0, 0, 0, 0)
        fields.setSpacing(12)
        preview = QVBoxLayout(self.preview_panel)
        preview.setContentsMargins(0, 0, 0, 0)
        preview.setSpacing(6)
        self._preview_layout = preview
        columns.addWidget(self.fields_panel, 1)
        columns.addWidget(self.preview_panel, 1)

        fields.addWidget(BodyLabel(self.tr("1 · Choose validation type"), self))
        self.protocol_combo = ComboBox(self)
        self.protocol_combo.addItem(self.tr("Select validation type…"), userData=None)
        self.protocol_combo.addItem(self.tr("EOS · Energy–volume curve"), userData="eos")
        self.protocol_combo.addItem(self.tr("Energy / force / virial errors"), userData="efv")
        fields.addWidget(self.protocol_combo)
        self.requirements_label = CaptionLabel(self)
        self.requirements_label.setWordWrap(True)
        fields.addWidget(self.requirements_label)
        unavailable = CaptionLabel(self.tr("Phonons, elastic constants and spin properties are not supported yet."), self)
        unavailable.setWordWrap(True)
        fields.addWidget(unavailable)
        fields.addWidget(BodyLabel(self.tr("2 · Load reference structures"), self))

        file_row = QWidget(self)
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.path_edit = LineEdit(file_row)
        self.path_edit.setPlaceholderText(self.tr("Labelled EXTXYZ file"))
        self.path_edit.editingFinished.connect(lambda: self.load_file(self.path_edit.text()))
        self.browse_button = PushButton(self.tr("Browse..."), file_row)
        self.browse_button.clicked.connect(self._browse)
        file_layout.addWidget(self.path_edit, 1)
        file_layout.addWidget(self.browse_button)

        self.name_edit = LineEdit(self)
        self.suite_combo = EditableComboBox(self)
        self.suite_combo.setPlaceholderText(self.tr("Choose a validation plan or type a new name"))
        for suite in suites:
            self.suite_combo.addItem(suite["name"])
        self.suite_combo.setText(suite_name)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.addRow(BodyLabel(self.tr("File"), self), file_row)
        self.group_field_combo = ComboBox(self)
        self.group_field_label = BodyLabel(self.tr("Group by field"), self)
        form.addRow(self.group_field_label, self.group_field_combo)
        self.group_hint = CaptionLabel(self.tr(
            "Fields are read from each structure's EXTXYZ comment line (its second line), e.g. config_type or eos_group.\n"
            "Example: config_type=B1 means field config_type, group B1. Use the same value for all volume points in that scan; use config_type=B3 for another scan.\n"
            "Every structure must have the chosen field. Each group must be one structure / magnetic branch. If the file contains only one scan, choose Single scan · whole file."
        ), self)
        self.group_hint.setWordWrap(True)
        form.addRow(self.group_hint)
        form.addRow(BodyLabel(self.tr("Name"), self), self.name_edit)
        fields.addLayout(form)
        fields.addWidget(BodyLabel(self.tr("3 · Add to a fixed validation plan"), self))
        fields.addWidget(self.suite_combo)
        plan_hint = CaptionLabel(self.tr("Future model versions reuse this reference dataset. Importing does not change the baseline model."), self)
        plan_hint.setWordWrap(True)
        fields.addWidget(plan_hint)
        self.supplementary_button = TransparentPushButton(
            FluentIcon.CARE_RIGHT_SOLID, self.tr("Supplementary information (optional)"), self
        )
        self.supplementary_button.setCheckable(True)
        fields.addWidget(self.supplementary_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.supplementary_panel = QWidget(self)
        supplementary_layout = QVBoxLayout(self.supplementary_panel)
        supplementary_layout.setContentsMargins(0, 0, 0, 0)
        supplementary_layout.addWidget(BodyLabel(self.tr("Notes (optional)"), self))
        self.notes_edit = LineEdit(self)
        self.notes_edit.setPlaceholderText(self.tr("e.g. calculation settings or a description of this dataset"))
        supplementary_layout.addWidget(self.notes_edit)
        fields.addWidget(self.supplementary_panel)
        self.supplementary_panel.hide()
        self.supplementary_button.toggled.connect(self._toggle_supplementary)
        fields.addStretch(1)

        self.preview_title = BodyLabel(self.tr("Reference curve preview"), self)
        preview.addWidget(self.preview_title)
        self.group_table = TableWidget(self)
        self.group_table.setColumnCount(4)
        self.group_table.setHorizontalHeaderLabels([
            self.tr("Import / group"), self.tr("Frames"), self.tr("Volume (Å³/atom)"), self.tr("Status"),
        ])
        self.group_table.verticalHeader().hide()
        self.group_table.verticalHeader().setDefaultSectionSize(30)
        self.group_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.group_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.group_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.group_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.group_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.group_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.group_table.setColumnWidth(2, 136)
        self.group_table.setFixedHeight(138)
        preview.addWidget(self.group_table)
        self.overlay_groups = CheckBox(self.tr("Overlay checked groups (no energy shifts)"), self)
        preview.addWidget(self.overlay_groups)
        self.preview_summary = CaptionLabel(self)
        self.preview_summary.setWordWrap(True)
        preview.addWidget(self.preview_summary)
        import NepTrainKit.ui.canvas.pyqtgraph  # noqa: F401 - shared plot configuration
        import pyqtgraph as pg
        self.preview_plot = pg.PlotWidget(self)
        self.preview_legend = self.preview_plot.addLegend(offset=(-10, 10))
        self.preview_plot.setMinimumHeight(260)
        self.preview_plot.showGrid(x=True, y=True, alpha=0.2)
        preview.addWidget(self.preview_plot, 1)

        self.issues_label = CaptionLabel(self)
        self.issues_label.setWordWrap(True)
        preview.addWidget(self.issues_label)
        preview.addStretch(0)
        self.status_label = CaptionLabel(self)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        self.viewLayout.addWidget(self.status_label)

        self.name_edit.textChanged.connect(self._validate)

        self.yesButton.setText(self.tr("Save reference dataset"))
        self.cancelButton.setText(self.tr("Cancel"))
        for control in (
            self.protocol_combo, self.path_edit, self.browse_button,
            self.group_field_combo, self.name_edit, self.notes_edit,
            self.suite_combo, self.overlay_groups,
        ):
            control.setMinimumHeight(max(32, control.sizeHint().height()))
        self.protocol_combo.currentIndexChanged.connect(self._protocol_changed)
        self.path_edit.textChanged.connect(self._invalidate_file)
        self.suite_combo.textChanged.connect(self._validate)
        self.group_field_combo.currentIndexChanged.connect(self._check_protocol)
        self.group_table.currentCellChanged.connect(self._update_preview)
        self.group_table.itemChanged.connect(self._include_group)
        self.overlay_groups.stateChanged.connect(self._update_preview)
        self._populate_group_fields()
        self._protocol_changed()
        parent.installEventFilter(self)
        self._fit_to_parent()

    def _toggle_supplementary(self, expanded: bool):
        self.supplementary_panel.setVisible(expanded)
        self.supplementary_button.setIcon(FluentIcon.CARE_DOWN_SOLID if expanded else FluentIcon.CARE_RIGHT_SOLID)

    def _fit_to_parent(self):
        """Scroll the body instead of squeezing fields into the host window."""
        parent = self.parentWidget()
        self.resize(parent.size())
        width = min(1120, max(0, parent.width() - 32))
        height = min(840, max(0, parent.height() - 32))
        self._columns.setDirection(
            QBoxLayout.Direction.LeftToRight if width >= 1040
            else QBoxLayout.Direction.TopToBottom
        )
        self.widget.setFixedSize(width, height)

    def eventFilter(self, obj, event):
        if obj is self.parentWidget() and event.type() == QEvent.Type.Resize and hasattr(self, "body_scroll"):
            self._fit_to_parent()
        return super().eventFilter(obj, event)

    def showEvent(self, event):
        self._fit_to_parent()
        super().showEvent(event)

    # -- file and protocol ----------------------------------------------------
    def _browse(self) -> None:
        path = call_path_dialog(
            self, self.tr("Select benchmark set"), "select", file_filter="EXTXYZ (*.xyz *.extxyz);;All files (*)"
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str | Path) -> None:
        path = str(path).strip()
        self.path_edit.blockSignals(True)
        self.path_edit.setText(path)
        self.path_edit.blockSignals(False)
        self._invalidate_file()
        if path:
            try:
                data = Path(path).expanduser().read_bytes()
                self.frames = read_frames(data)
                self._loaded_bytes = data
            except Exception as error:  # noqa: BLE001 - any parse failure blocks the import
                self._read_error = str(error) or type(error).__name__
            else:
                if not self.name_edit.text().strip():
                    self.name_edit.setText(Path(path).stem)
        self._populate_group_fields()
        self._check_protocol()

    def current_protocol(self):
        kind = self.protocol_combo.currentData()
        return get_protocol(kind) if kind else None

    def _invalidate_file(self, *_args):
        self.frames, self._loaded_bytes, self._read_error = None, None, ""
        self._populate_group_fields()
        self._check_protocol()

    def _populate_group_fields(self):
        self.group_field_combo.blockSignals(True)
        self.group_field_combo.clear()
        self.group_field_combo.addItem(self.tr("Choose grouping…"), userData=None)
        self.group_field_combo.addItem(self.tr("Single scan · whole file"), userData="")
        for field in eos_group_fields(self.frames or []):
            self.group_field_combo.addItem(field, userData=field)
        self.group_field_combo.blockSignals(False)

    def _protocol_changed(self, *_args):
        kind = self.protocol_combo.currentData()
        self.path_edit.setEnabled(bool(kind))
        self.browse_button.setEnabled(bool(kind))
        self.group_field_label.setVisible(kind == "eos")
        self.group_field_combo.setVisible(kind == "eos")
        self.group_hint.setVisible(kind == "eos")
        if kind == "eos":
            text = self.tr(
                'Each EOS group needs at least 5 distinct volume points from one phase / magnetic branch.\n'
                "Input: each structure's total energy in eV. Do not divide by the atom count beforehand.\n"
                'Comparison: E/N (eV/atom) and V/N (Å³/atom), normalized automatically.\n'
                'Fixed periodic reference structures; ordinary NEP models only.'
            )
        elif kind == "efv":
            text = self.tr("Labelled EXTXYZ structures with total energies; force and virial labels are optional. This compares prediction errors, not derived material properties.")
        else:
            text = self.tr("Choose what to validate first. The file will not select a validation type for you.")
        self.requirements_label.setText(text)
        self._check_protocol()

    def _check_protocol(self, *_args) -> None:
        protocol = self.current_protocol()
        self.eos_groups, self._group_error = [], ""
        self.issues = []
        self.group_table.blockSignals(True)
        self.group_table.setRowCount(0)
        if protocol is not None and self.frames is not None:
            if protocol.kind == "eos":
                field = self.group_field_combo.currentData()
                if field is not None:
                    try:
                        groups = eos_frame_groups(self.frames, field) if field else {"": list(range(len(self.frames)))}
                    except ValueError as error:
                        self._group_error = str(error)
                    else:
                        for key, indices in groups.items():
                            frames = [self.frames[i] for i in indices]
                            issues = protocol.check(frames)
                            summary = None if any(i.level == "error" for i in issues) else protocol.summarize_reference(frames)
                            self.eos_groups.append(dict(key=key, indices=indices, issues=issues,
                                                        summary=summary, include=True))
                        self.group_table.setRowCount(len(self.eos_groups))
                        for row, group in enumerate(self.eos_groups):
                            name = QTableWidgetItem(group["key"] or self.tr("Whole file"))
                            name.setFlags(name.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                            name.setCheckState(Qt.CheckState.Checked)
                            self.group_table.setItem(row, 0, name)
                            self.group_table.setItem(row, 1, QTableWidgetItem(str(len(group["indices"]))))
                            summary = group["summary"]
                            volume = "—" if summary is None else f'{summary["volumes"][0]:.2f}–{summary["volumes"][-1]:.2f}'
                            self.group_table.setItem(row, 2, QTableWidgetItem(volume))
                            self._set_group_status(row)
                        if self.eos_groups:
                            self.group_table.setCurrentCell(0, 0)
            else:
                self.issues = protocol.check(self.frames)
        self.group_table.blockSignals(False)
        multiple = protocol is not None and protocol.kind == "eos" and bool(self.group_field_combo.currentData())
        self.group_table.setVisible(multiple)
        self.overlay_groups.setVisible(multiple)
        self.overlay_groups.setChecked(False)
        self._update_preview()
        self._validate()

    def _set_group_status(self, row):
        group = self.eos_groups[row]
        if not group["include"]:
            text = self.tr("Excluded")
        elif group["summary"] is None:
            text = self.tr("Error")
        else:
            text = self.tr("Ready · warning") if group["issues"] else self.tr("Checks passed")
        self.group_table.setItem(row, 3, QTableWidgetItem(text))

    def _include_group(self, item):
        if item.column() != 0:
            return
        group = self.eos_groups[item.row()]
        group["include"] = item.checkState() == Qt.CheckState.Checked
        self._set_group_status(item.row())
        self._update_preview()
        self._validate()

    def _show_issues(self):
        lines = []
        for issue in self.issues:
            color = palette.PLOT_REJECT if issue.level == "error" else palette.PLOT_SELECTED
            label = self.tr("Error") if issue.level == "error" else self.tr("Warning")
            lines.append(f'<span style="color:{color};">{label}:</span> {escape(translate_runtime_message(issue.message))}')
        if self._group_error:
            lines.append(escape(translate_runtime_message(self._group_error)))
        elif self.reference_summary is not None and not self.issues:
            lines.append(self.tr("Reference data checks passed."))
        self.issues_label.setText("<br>".join(lines))

    def _update_preview(self, *_args):
        self.preview_plot.clear()
        self.preview_legend.clear()
        self.reference_summary = None
        protocol = self.current_protocol()
        is_eos = protocol is not None and protocol.kind == "eos"
        self.preview_plot.setVisible(is_eos)
        self._preview_layout.setStretch(self._preview_layout.count() - 1, 0 if is_eos else 1)
        self.preview_summary.setText(self.tr("Load a reference file to inspect its data before saving."))
        self.preview_title.setText(self.tr("Reference curve") if is_eos else self.tr("Reference data preview"))
        group = None
        if is_eos:
            row = self.group_table.currentRow()
            group = self.eos_groups[row] if 0 <= row < len(self.eos_groups) else None
            self.issues = group["issues"] if group else []
            self.reference_summary = group["summary"] if group else None
        elif protocol is not None and self.frames is not None and not any(i.level == "error" for i in self.issues):
            self.reference_summary = protocol.summarize_reference(self.frames)
        self._show_issues()
        summary = self.reference_summary
        if summary is None:
            return
        if protocol.kind == "eos":
            import pyqtgraph as pg
            volumes = summary["volumes"]
            self.preview_plot.setLabel("bottom", self.tr("Volume (Å³/atom)"))
            self.preview_plot.setLabel("left", self.tr("Energy (eV/atom)"))
            plotted = [g for g in self.eos_groups if g["include"] and g["summary"]] if self.overlay_groups.isChecked() else [group]
            for index, entry in enumerate(plotted):
                curve = entry["summary"]
                color = (palette.PLOT_MARKER_EDGE, palette.PLOT_SELECTED, palette.PLOT_CURRENT)[index % 3]
                self.preview_plot.plot(curve["volumes"], curve["energies"], pen=pg.mkPen(color, width=2),
                                       symbol="o", symbolBrush=color, symbolSize=7,
                                       name=entry["key"] or self.tr("Whole file"))
            text = self.tr("{formula} · {count} structures\nVolume: {low:.3f}–{high:.3f} Å³/atom").format(
                formula=summary["formula"], count=len(group["indices"]), low=volumes[0], high=volumes[-1])
            if group["key"]:
                text = self.tr("Group: {name}").format(name=group["key"]) + "\n" + text
            fit = summary["fit"]
            if "v0" in fit:
                text += "\n" + self.tr("Reference fit: V₀ = {v0:.3f} Å³/atom · B₀ = {b0:.2f} GPa").format(v0=fit["v0"], b0=fit["b0_gpa"])
            else:
                text += "\n" + self.tr("Fit unavailable: compare the E–V curve only; V₀/B₀ will not be compared.")
        else:
            text = self.tr("{count} structures · {atoms} atoms\nEnergy labels available. Forces: {forces}; virials: {virials}.").format(
                count=summary["frames"], atoms=summary["atoms"],
                forces=self.tr("Yes") if summary["has_forces"] else self.tr("No"),
                virials=self.tr("Yes") if summary["has_virials"] else self.tr("No"))
        self.preview_summary.setText(text)

    # -- validation -----------------------------------------------------------
    def validation_errors(self) -> list[str]:
        errors = []
        if self.current_protocol() is None:
            errors.append(self.tr("Choose a validation type first."))
        if not self.path_edit.text().strip():
            errors.append(self.tr("Choose a reference file."))
        elif self._read_error:
            errors.append(self.tr("The file cannot be read as EXTXYZ: {error}").format(error=self._read_error))
        elif self.frames is None:
            errors.append(self.tr("Load the file to refresh the reference preview."))
        elif self.protocol_combo.currentData() != "eos" and any(issue.level == "error" for issue in self.issues):
            errors.append(self.tr("Fix the errors above before importing."))
        if not self.name_edit.text().strip():
            errors.append(self.tr("Enter a name for the set."))
        if not self.suite_combo.text().strip():
            errors.append(self.tr("Choose or name a validation plan."))
        if self.protocol_combo.currentData() == "eos":
            if self.group_field_combo.currentData() is None:
                errors.append(self.tr("Choose a grouping field, or explicitly use the whole file as one scan."))
            elif self._group_error:
                errors.append(translate_runtime_message(self._group_error))
            else:
                selected = [group for group in self.eos_groups if group["include"]]
                if not selected:
                    errors.append(self.tr("Select at least one EOS group to import."))
                elif any(group["summary"] is None for group in selected):
                    errors.append(self.tr("Some checked groups have errors. Fix them or explicitly uncheck them."))
        return errors

    def _validate(self, *_args) -> None:
        errors = self.validation_errors()
        self.yesButton.setEnabled(not errors)
        status = errors[0] if errors else ""
        if not errors and self.protocol_combo.currentData() == "eos":
            status = self.tr("Ready to import {count} EOS groups; {excluded} excluded.").format(
                count=sum(group["include"] for group in self.eos_groups),
                excluded=sum(not group["include"] for group in self.eos_groups),
            )
        self.status_label.setText(status)

    def validate(self) -> bool:
        return not self.validation_errors()

    def values(self) -> dict:
        protocol = self.current_protocol()
        return {
            "path": self.path_edit.text().strip(),
            "name": self.name_edit.text().strip(),
            "protocol": protocol.kind if protocol is not None else None,
            "provenance": {"notes": self.notes_edit.text().strip()} if self.notes_edit.text().strip() else {},
            "suite": self.suite_combo.text().strip(),
        }

    def import_into(self, service, confirm_large: Callable[[SetTooLargeError], bool]) -> list[dict] | None:
        """Seal the set; ``confirm_large`` decides whether an oversized file is copied anyway.

        Returns the set records, or ``None`` when the user declined a large copy.
        Rejections propagate as ``SetRejectedError`` / ``ValueError``.
        """
        errors = self.validation_errors()
        if errors:
            raise ValueError(errors[0])
        values = self.values()
        if Path(values["path"]).expanduser().read_bytes() != self._loaded_bytes:
            self._invalidate_file()
            raise ValueError(self.tr("The reference file changed. Load it again and review the new curve."))
        kwargs = dict(name=values["name"], provenance=values["provenance"], protocol=values["protocol"])
        grouped = values["protocol"] == "eos" and bool(self.group_field_combo.currentData())
        if grouped:
            kwargs.pop("protocol")
            kwargs.update(
                group_field=self.group_field_combo.currentData(),
                group_names=[group["key"] for group in self.eos_groups if group["include"]],
                expected_hash=sha256_bytes(self._loaded_bytes),
            )
        importer = service.import_eos_groups if grouped else service.import_set
        try:
            records = importer(values["path"], **kwargs)
        except SetTooLargeError as error:
            if not confirm_large(error):
                return None
            records = importer(values["path"], allow_large=True, **kwargs)
        if not grouped:
            records = [records]
        if values["suite"]:
            suite = next((item for item in service.suites() if item["name"] == values["suite"]), None)
            if suite is None:
                service.create_suite(values["suite"], [record["id"] for record in records])
            else:
                service.add_to_suite(suite["id"], [record["id"] for record in records])
        return records
