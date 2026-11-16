"""Dialogs for the model validation library."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
from html import escape

from PySide6.QtCore import QEvent, QSize, Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QAbstractItemView, QBoxLayout, QComboBox, QDialog, QFrame, QSizePolicy, QStackedWidget, QFileDialog, QFormLayout, QHeaderView, QHBoxLayout, QLayout, QListView, QStyledItemDelegate, QTableWidgetItem, QVBoxLayout, QWidget
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
    PrimaryPushButton,
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
from NepTrainKit.core.validation.service import INBOX_PROJECT, ModelDraft, parse_model_header
from NepTrainKit.core.prediction_cache import find_output_groups, inspect_output_groups
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.threads import BackgroundTask
from NepTrainKit.ui.widgets.validation_group_comparison import remember_import_markers
from NepTrainKit.ui.widgets.validation_model_picker import GroupMarkerButton, default_group_marker, group_marker_icon
from NepTrainKit.ui.widgets.validation_presentation import file_import_choices, presentation_for


class _ProjectComboBox(QComboBox):
    """Use a Qt-owned editable popup with the project library's folder styling."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("importProject")
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setMinimumHeight(32)
        self.setView(QListView(self))
        self.setIconSize(QSize(16, 16))
        self.view().setIconSize(self.iconSize())
        self.setItemDelegate(QStyledItemDelegate(self.view()))
        self.view().setMouseTracking(True)
        self.view().setUniformItemSizes(True)
        self.setStyleSheet(f"""
            QComboBox#importProject {{background: white; color: {palette.TEXT_PRIMARY};
                border: 1px solid {palette.BORDER_STRONG}; border-radius: 6px; padding: 1px 28px 1px 8px;}}
            QComboBox#importProject:hover {{border-color: {palette.ACCENT_BORDER};}}
            QComboBox#importProject:focus, QComboBox#importProject:on {{border-color: {palette.ACCENT};}}
            QComboBox#importProject QLineEdit {{background: transparent; border: none; padding: 0;}}
            QComboBox#importProject::drop-down {{subcontrol-origin: border; subcontrol-position: right center;
                width: 28px; border: none;}}
            QComboBox#importProject::down-arrow {{image: url({FluentIcon.ARROW_DOWN.path()}); width: 12px; height: 12px;}}
            QListView {{background: white; color: {palette.TEXT_PRIMARY}; border: 1px solid {palette.BORDER_STRONG};
                border-radius: 7px; padding: 4px; outline: none;}}
            QListView::item {{height: 31px; padding: 0 8px; border: none; border-radius: 4px;}}
            QListView::item:hover {{background: {palette.SURFACE_SUBTLE};}}
            QListView::item:selected {{background: {palette.ACCENT_SOFT}; color: {palette.ACCENT_PRESSED};}}
        """)

    def load_projects(self, projects):
        self.clear()
        folder = FluentIcon.FOLDER.icon(color=palette.ACCENT_PRESSED)
        for project in projects:
            self.addItem(folder, project["name"])
        self.setCurrentIndex(-1)

    def select_name(self, name):
        self.setCurrentIndex(self.findText(name))
        self.setEditText(name)


class SaveModelDialog(MessageBoxBase):
    """Choose an existing/new project and a version name before writing a snapshot."""

    def __init__(self, parent, service, name: str):
        super().__init__(parent)
        self.service = service
        self._initial_library_root = service.library.root
        self.title_label = SubtitleLabel(self.tr("Save model to project"), self)
        self.viewLayout.addWidget(self.title_label)
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
        self.project_combo = _ProjectComboBox(self)
        self.project_combo.setAccessibleName(self.tr("Project"))
        self.project_combo.lineEdit().setPlaceholderText(self.tr("Choose a project or type a new name"))
        self.project_combo.load_projects(service.projects())
        previous = Config.get("validation", "last_save_project", "")
        self.project_combo.setCurrentIndex(self.project_combo.findText(previous))
        self.name_edit = LineEdit(self)
        self.name_edit.setMaxLength(80)
        self.name_edit.setText(name)
        form = QFormLayout()
        form.addRow(BodyLabel(self.tr("Project"), self), self.project_combo)
        form.addRow(BodyLabel(self.tr("Version name"), self), self.name_edit)
        self.viewLayout.addLayout(form)
        self.yesButton.setText(self.tr("Save"))
        self.cancelButton.setText(self.tr("Cancel"))
        self.project_combo.currentTextChanged.connect(self._validate)
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
        self.project_combo.load_projects(projects)
        self._validate()

    def _validate(self, *_args):
        project = self.project_combo.currentText().strip()
        self.yesButton.setEnabled(bool(project and len(project) <= 200 and self.name_edit.text().strip()))

    def save_model(self, draft, *, accuracy=None):
        """Save the reviewed bytes and persist the selected destination."""
        record = self.service.save_model(draft, project_name=self.project_combo.currentText(),
                                         name=self.name_edit.text(), accuracy=accuracy)
        if self.service.library.root != self._initial_library_root:
            Config.set(CONFIG_SECTION, CONFIG_LIBRARY_PATH, str(self.service.library.root))
        Config.set("validation", "last_save_project", self.project_combo.currentText().strip())
        record["library_root"] = str(self.service.library.root)
        return record


class ImportModelDialog(SaveModelDialog):
    """Review recognised model files in one prefilled import dialog."""

    def __init__(self, parent, service, paths, *, project_name=""):
        self.draft = None
        self._output_task = None
        self._output_busy = False
        self._output_generation = 0
        self._pending_output_scan = None
        self._output_summaries = {}
        super().__init__(parent, service, "")
        self.title_label.setText(self.tr("Import NEP model"))
        self.yesButton.setText(self.tr("Import"))
        self.project_combo.select_name(project_name or self.project_combo.currentText() or INBOX_PROJECT)
        self.model_combo = ComboBox(self)
        self.model_combo.setAccessibleName(self.tr("Model file"))
        for path in paths:
            self.model_combo.addItem(Path(path).name, userData=str(path))
        self.model_combo.setVisible(len(paths) > 1)
        if len(paths) > 1:
            self.model_combo.setCurrentIndex(-1)
        self.model_details = BodyLabel(self)
        self.model_details.setWordWrap(True)
        self.viewLayout.insertWidget(1, self.model_combo)
        self.viewLayout.insertWidget(2, self.model_details)
        self.output_box = QWidget(self)
        output_layout = QVBoxLayout(self.output_box)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(BodyLabel(self.tr("RMSE from existing outputs"), self.output_box))
        self.output_combo = ComboBox(self.output_box)
        output_layout.addWidget(self.output_combo)
        self.output_fields = {}
        form = QFormLayout()
        for key, label in (("energy", self.tr("Energy (meV/atom)")), ("force", self.tr("Force (meV/Å)")),
                           ("virial", self.tr("Virial (meV/atom)")), ("mforce", self.tr("Magnetic force (meV/μB)"))):
            field = LineEdit(self.output_box)
            field.setReadOnly(True)
            self.output_fields[key] = field
            form.addRow(BodyLabel(label, self.output_box), field)
        output_layout.addLayout(form)
        self.output_status = CaptionLabel(self.output_box)
        self.output_status.setWordWrap(True)
        self.output_status.setMaximumHeight(self.output_status.fontMetrics().lineSpacing() * 3)
        output_layout.addWidget(self.output_status)
        self.viewLayout.addWidget(self.output_box)
        self.output_combo.currentIndexChanged.connect(self._show_output_summary)
        self.model_combo.currentIndexChanged.connect(self._select_model)
        self._select_model()

    def _select_model(self, *_args):
        self.draft = None
        if self.model_combo.currentIndex() < 0:
            self.model_details.setText(self.tr("Choose one of the recognised model files."))
            self._request_output_scan()
            self._validate()
            return
        try:
            draft = ModelDraft.from_file(self.model_combo.currentData())
            kind, elements = parse_model_header(draft.data)
        except (OSError, ValueError, TypeError) as error:
            self.model_details.setText(self.tr("Cannot read model: {error}").format(error=error))
        else:
            self.draft = draft
            self.name_edit.setText(draft.name)
            self.model_details.setText(self.tr("File: {path}\nType: {kind} · Elements: {elements}").format(
                path=draft.source_path, kind=kind, elements=", ".join(elements),
            ))
        self._request_output_scan()
        self._validate()

    def _request_output_scan(self):
        self._output_generation += 1
        self._output_summaries = {}
        self.output_combo.clear()
        for field in self.output_fields.values():
            field.setText("—")
        try:
            groups = find_output_groups(Path(self.draft.source_path).parent) if self.draft else {}
        except OSError:
            groups = {}
        self.output_box.setVisible(bool(groups))
        self._output_busy = bool(groups)
        self._pending_output_scan = (self._output_generation, groups, self.draft.content_hash) if groups else None
        if self._output_task is not None:
            self._output_task.stop_work()
        elif groups:
            self._start_output_scan()

    def _start_output_scan(self):
        generation, groups, digest = self._pending_output_scan
        self._pending_output_scan = None
        self.output_status.setText(self.tr("Reading existing outputs in the background; no model prediction is run."))
        task = BackgroundTask(QApplication.instance(), show_tip=False)
        task.scan_generation = generation
        self._output_task = task
        task.succeeded.connect(self._apply_output_scan)
        task.failed.connect(self._output_scan_failed)
        task.finished.connect(self._output_scan_finished)
        task.finished.connect(task.deleteLater)
        self.destroyed.connect(task.stop_work)
        task.start_work(inspect_output_groups, groups, digest, task.isInterruptionRequested)

    @Slot(object)
    def _apply_output_scan(self, summaries):
        if self.sender().scan_generation != self._output_generation:
            return
        self._output_busy = False
        self._output_summaries = summaries
        self.output_combo.blockSignals(True)
        for name in sorted(summaries, key=lambda name: (name != "train", name != "test", name)):
            label = self.tr("Training outputs") if name == "train" else self.tr("Test outputs") if name == "test" else name
            self.output_combo.addItem(label, userData=name)
        self.output_combo.blockSignals(False)
        self._show_output_summary()
        self._validate()

    @Slot(str)
    def _output_scan_failed(self, error):
        if self.sender().scan_generation == self._output_generation:
            self._output_busy = False
            self.output_status.setText(self.tr("Cannot read existing outputs: {error}").format(error=error))
            self._validate()

    @Slot()
    def _output_scan_finished(self):
        self._output_task = None
        if self._pending_output_scan is not None:
            self._start_output_scan()

    def _show_output_summary(self, *_args):
        summary = self._output_summaries.get(self.output_combo.currentData(), {})
        for key, field in self.output_fields.items():
            value = summary.get("metrics", {}).get(key + "_rmse")
            field.setText("—" if value is None else f"{value:.6g}")
        if not summary:
            return
        if summary["status"] == "verified":
            text = self.tr("Model, dataset and output hashes verified. Existing outputs were read without prediction.")
        elif summary["status"] == "mismatch":
            text = self.tr("Outputs do not match this model or dataset. Import will save weights only.")
        else:
            text = self.tr("Historical output precision: its association with this model is unverified; this is not a validation record.")
        reasons = [summary.get("reason", ""), *summary.get("errors", {}).values()]
        text = "\n".join([text, *[translate_runtime_message(reason) for reason in reasons if reason]])
        self.output_status.setText(text)
        self.output_status.setToolTip(text)

    def save_model(self, draft):
        summary = self._output_summaries.get(self.output_combo.currentData(), {})
        if summary.get("status") in {"verified", "unverified"} and any(
            value is not None for value in summary.get("metrics", {}).values()
        ):
            return super().save_model(draft, accuracy=summary)
        return super().save_model(draft)

    def done(self, result):
        self._pending_output_scan = None
        if self._output_task is not None:
            self._output_task.stop_work()
        super().done(result)

    def _validate(self, *_args):
        super()._validate()
        self.yesButton.setEnabled(self.yesButton.isEnabled() and self.draft is not None and not self._output_busy)


def save_model_to_project(parent, service, draft, *, name=None):
    dialog = SaveModelDialog(parent, service, name or draft.name)
    if not dialog.exec():
        return None
    return dialog.save_model(draft)


class BenchmarkImportDialog(QDialog):
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

        self.setWindowTitle(self.tr("Import reference dataset"))
        self.setObjectName("benchmarkImportDialog")
        self.setStyleSheet(
            f"#benchmarkImportDialog, #benchmarkImportBody {{ background: {palette.SURFACE_SUBTLE}; }}"
            f"QFrame#importCard {{ background: white; border: 1px solid {palette.BORDER_STRONG}; border-radius: 9px; }}"
            f"QFrame#importFooter {{ background: white; border-top: 1px solid {palette.BORDER}; }}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.body_scroll = ScrollArea(self)
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("benchmarkImportBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 14, 14, 12)
        body_layout.setSpacing(10)
        body_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.body_scroll.setWidget(body)
        root.addWidget(self.body_scroll, 1)

        def card():
            frame = QFrame(body)
            frame.setObjectName("importCard")
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(8)
            return frame, layout

        source_panel, source_layout = card()
        body_layout.addWidget(source_panel)
        self._source_row = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self._source_row.setSpacing(12)
        source_layout.addLayout(self._source_row)
        self.fields_panel, fields = card()
        self.preview_panel, preview = card()
        self._preview_layout = preview
        self._columns = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self._columns.setSpacing(10)
        self._columns.addWidget(self.fields_panel, 2)
        self._columns.addWidget(self.preview_panel, 3)
        body_layout.addLayout(self._columns, 1)
        type_layout = QVBoxLayout()
        type_layout.setSpacing(4)
        type_layout.addWidget(BodyLabel(self.tr("Validation type"), self))
        self._source_row.addLayout(type_layout, 2)
        self.protocol_combo = ComboBox(self)
        self.protocol_combo.addItem(self.tr("Select validation type…"), userData=None)
        for kind, label in file_import_choices():
            self.protocol_combo.addItem(self.tr(label), userData=kind)
        self.protocol_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        type_layout.addWidget(self.protocol_combo)
        self.requirements_label = CaptionLabel(self)
        self.requirements_label.setWordWrap(True)

        file_row = QWidget(self)
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.path_edit = LineEdit(file_row)
        self.path_edit.setPlaceholderText(self.tr("Labelled EXTXYZ file"))
        self.file_picker = QStackedWidget(file_row)
        self.file_picker.addWidget(self.path_edit)
        self.file_name = PushButton("", file_row)
        self.file_name.setAutoDefault(False)
        self.file_name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.file_name.setStyleSheet(f"PushButton {{ text-align: left; padding: 0 10px; background: {palette.SURFACE}; color: {palette.TEXT_PRIMARY}; border: 1px solid {palette.BORDER}; border-radius: 5px; }}")
        self.file_name.clicked.connect(self._edit_path)
        self.file_picker.addWidget(self.file_name)
        self.path_edit.editingFinished.connect(self._finish_path_edit)
        self.browse_button = PushButton(self.tr("Browse..."), file_row)
        self.browse_button.clicked.connect(self._browse)
        file_layout.addWidget(self.file_picker, 1)
        file_layout.addWidget(self.browse_button)

        self.name_edit = LineEdit(self)
        self.suite_combo = EditableComboBox(self)
        self.suite_combo.setPlaceholderText(self.tr("Choose a validation plan or type a new name"))
        for suite in suites:
            self.suite_combo.addItem(suite["name"])
        self.suite_combo.setText(suite_name)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        source_file = QVBoxLayout()
        source_file.setSpacing(4)
        source_file.addWidget(BodyLabel(self.tr("Reference file"), self))
        source_file.addWidget(file_row)
        self._source_row.addLayout(source_file, 3)
        fields.addWidget(BodyLabel(self.tr("Import settings"), self))
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.addRow(self.tr("Suite"), self.suite_combo)
        form.addRow(self.tr("Name"), self.name_edit)
        self.group_field_combo = ComboBox(self)
        self.group_field_label = BodyLabel(self.tr("Group by field"), self)
        form.addRow(self.group_field_label, self.group_field_combo)
        self.group_hint = CaptionLabel(self.tr(
            "Fields are read from each structure's EXTXYZ comment line (its second line), e.g. config_type or eos_group.\n"
            "Example: config_type=B1 means field config_type, group B1. Use the same value for all volume points in that scan; use config_type=B3 for another scan.\n"
            "Every structure must have the chosen field. Each group must be one structure / magnetic branch. If the file contains only one scan, choose Single scan · whole file."
        ), self)
        self.group_hint.setWordWrap(True)
        self.help_button = TransparentPushButton(FluentIcon.INFO, self.tr("Data requirements"), self)
        self.help_button.setCheckable(True)
        self.help_button.setAutoDefault(False)
        self._source_row.addWidget(self.help_button, 0, Qt.AlignmentFlag.AlignBottom)
        self.help_panel = QWidget(source_panel)
        help_layout = QVBoxLayout(self.help_panel)
        help_layout.setContentsMargins(0, 4, 0, 0)
        help_layout.addWidget(self.requirements_label)
        help_layout.addWidget(self.group_hint)
        self.help_panel.hide()
        self.help_button.toggled.connect(self.help_panel.setVisible)
        source_layout.addWidget(self.help_panel)
        fields.addLayout(form)
        from .reference_analysis import ReferenceAnalysisEditor
        self.analysis_editor = ReferenceAnalysisEditor(self)
        fields.addWidget(self.analysis_editor)
        self.analysis_editor.hide()
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

        self.preview_title = BodyLabel(self.tr("Reference curve preview"), self)
        preview_header = QHBoxLayout()
        self.preview_title.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {palette.TEXT_PRIMARY};")
        preview_header.addWidget(self.preview_title, 1)
        self.preview_marker = GroupMarkerButton("o", self)
        self.preview_marker.markerChanged.connect(self._change_preview_marker)
        preview_header.addWidget(self.preview_marker)
        preview.addLayout(preview_header)
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
        self.group_table.setColumnHidden(2, True)
        self.group_table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.group_table.setMinimumHeight(148)
        fields.addWidget(self.group_table, 1)
        self.overlay_groups = CheckBox(self.tr("Overlay checked groups (no energy shifts)"), self)
        fields.addWidget(self.overlay_groups)
        fields.addStretch(0)
        self.preview_summary = CaptionLabel(self)
        self.preview_summary.setWordWrap(True)
        self.preview_summary.setStyleSheet(f"font-size: 13px; color: {palette.TEXT_SECONDARY};")
        self.preview_plot = None
        self.preview_layout = preview
        preview.addWidget(self.preview_summary)

        self.issues_label = CaptionLabel(self)
        self.issues_label.setWordWrap(True)
        preview.addWidget(self.issues_label)
        footer = QFrame(self)
        footer.setObjectName("importFooter")
        actions = QHBoxLayout(footer)
        actions.setContentsMargins(16, 9, 16, 9)
        actions.setSpacing(8)
        root.addWidget(footer)
        self.cancelButton = PushButton(self.tr("Cancel"), footer)
        self.cancelButton.setAutoDefault(False)
        self.cancelButton.clicked.connect(self.reject)
        self.yesButton = PrimaryPushButton(self.tr("Import"), footer)
        self.yesButton.setDefault(True)
        self.yesButton.clicked.connect(self.accept)
        self.status_label = CaptionLabel(footer)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        actions.addWidget(self.status_label, 1)
        actions.addWidget(self.cancelButton)
        actions.addWidget(self.yesButton)
        self.yesButton.setMinimumWidth(92)
        self.name_edit.textChanged.connect(self._validate)
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
        self.analysis_editor.changed.connect(self._check_protocol)
        self.group_table.currentCellChanged.connect(self._update_preview)
        self.group_table.itemChanged.connect(self._include_group)
        self.overlay_groups.stateChanged.connect(self._update_preview)
        self._populate_group_fields()
        self._protocol_changed()
        screen = self.screen().availableGeometry()
        width = min(1100, (parent.width() if parent and parent.isVisible() else screen.width()) - 32, screen.width() - 32)
        self.resize(max(320, width), min(720, screen.height() - 72))

    def _toggle_supplementary(self, expanded: bool):
        self.supplementary_panel.setVisible(expanded)
        self.supplementary_button.setIcon(FluentIcon.CARE_DOWN_SOLID if expanded else FluentIcon.CARE_RIGHT_SOLID)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        narrow = self.width() < 880
        self._columns.setDirection(QBoxLayout.Direction.TopToBottom if narrow else QBoxLayout.Direction.LeftToRight)
        self._source_row.setDirection(QBoxLayout.Direction.TopToBottom if self.width() < 680 else QBoxLayout.Direction.LeftToRight)
        self.group_table.setMaximumHeight(190 if narrow else 16777215)

    def accept(self):
        if self.validate():
            super().accept()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and self.preview_plot is not None and obj is self.preview_plot.viewport():
            pixels = event.pixelDelta()
            if pixels.isNull():
                QApplication.sendEvent(self.body_scroll.viewport(), event)
            else:
                bar = self.body_scroll.verticalScrollBar(); bar.setValue(bar.value() - pixels.y())
            return True
        return super().eventFilter(obj, event)

    # -- file and protocol ----------------------------------------------------
    def _edit_path(self):
        self.file_picker.setCurrentWidget(self.path_edit)
        self.path_edit.setFocus()
        self.path_edit.selectAll()

    def _finish_path_edit(self):
        if self._loaded_bytes is None:
            self.load_file(self.path_edit.text())
        else:
            self.file_picker.setCurrentWidget(self.file_name)

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
                self.file_name.setText(Path(path).name)
                self.file_name.setToolTip(path)
                self.file_picker.setCurrentWidget(self.file_name)
            except Exception as error:  # noqa: BLE001 - any parse failure blocks the import
                self._read_error = str(error) or type(error).__name__
            else:
                if not self.name_edit.text().strip():
                    self.name_edit.setText(Path(path).stem)
        self._populate_group_fields()
        self.analysis_editor.set_frames(self.frames)
        self._check_protocol()

    def current_protocol(self):
        kind = self.protocol_combo.currentData()
        try:
            return get_protocol(kind, analysis=self.analysis_editor.settings() if kind == "response" else None) if kind else None
        except (ValueError, TypeError):
            return None

    def _invalidate_file(self, *_args):
        self.frames, self._loaded_bytes, self._read_error = None, None, ""
        self.file_picker.setCurrentWidget(self.path_edit)
        self._populate_group_fields()
        self._check_protocol()

    def _populate_group_fields(self):
        self.group_field_combo.blockSignals(True)
        self.group_field_combo.clear()
        self.group_field_combo.addItem(self.tr("Choose grouping…"), userData=None)
        self.group_field_combo.addItem(self.tr("Single scan · whole file"), userData="")
        for field in eos_group_fields(self.frames or []):
            self.group_field_combo.addItem(field, userData=field)
        explicit = self.group_field_combo.findData(presentation_for("eos").source_group)
        if explicit >= 0:
            self.group_field_combo.setCurrentIndex(explicit)
        self.group_field_combo.blockSignals(False)

    def _protocol_changed(self, *_args):
        kind = self.protocol_combo.currentData()
        self.path_edit.setEnabled(bool(kind))
        self.browse_button.setEnabled(bool(kind))
        self.group_table.setColumnHidden(2, kind == "eos")
        self.group_field_label.setVisible(kind == "eos")
        self.group_field_combo.setVisible(kind == "eos")
        self.group_hint.setVisible(kind == "eos")
        self.analysis_editor.setVisible(kind == "response")
        if kind == "eos":
            self.group_table.setHorizontalHeaderLabels([self.tr("Import / group"), self.tr("Frames"), self.tr("Volume (Å³/atom)"), self.tr("Status")])
            text = self.tr(
                'Each EOS group needs at least 5 distinct volume points from one phase / magnetic branch.\n'
                "Input: each structure's total energy in eV. Do not divide by the atom count beforehand.\n"
                'Comparison: E/N (eV/atom) and V/N (Å³/atom), normalized automatically.\n'
                'Fixed periodic reference structures; keep input magnetic order and moment magnitudes fixed.'
            )
        elif kind == "efv":
            text = self.tr(
                "Choose a dataset independently of the model. Energy, force, virial and magnetic-force labels "
                "are counted separately; at least one labelled quantity is required."
            )
        elif kind == "response":
            text = self.tr("Compare fixed reference structures using metadata groups, scan coordinates and signed state combinations. Review matching and exclusions before saving.")
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
                        for index, (key, indices) in enumerate(groups.items()):
                            frames = [self.frames[i] for i in indices]
                            issues = protocol.check(frames)
                            summary = None if any(i.level == "error" for i in issues) else protocol.summarize_reference(frames)
                            self.eos_groups.append(dict(key=key, indices=indices, issues=issues,
                                                        summary=summary, include=True, marker=default_group_marker(index)))
                        self.group_table.setRowCount(len(self.eos_groups))
                        for row, group in enumerate(self.eos_groups):
                            title, _ = presentation_for("eos").group_title(group["key"], (group["summary"] or {}).get("formula"))
                            name = QTableWidgetItem(title.replace(" · ", "\n", 1) or self.tr("Whole file"))
                            self.group_table.setRowHeight(row, 44 if "\n" in name.text() else 32)
                            name.setIcon(group_marker_icon(group["marker"]))
                            name.setToolTip(group["key"])
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
            text, color = self.tr("Excluded"), palette.TEXT_MUTED
        elif group["summary"] is None:
            text, color = self.tr("Error"), palette.DANGER
        elif group["issues"]:
            text, color = self.tr("Ready · warning"), palette.WARNING_STRONG
        else:
            text, color = self.tr("Checks passed"), palette.SUCCESS
        status = QTableWidgetItem(text)
        status.setForeground(QColor(color))
        self.group_table.setItem(row, 3, status)

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

    def _change_preview_marker(self, symbol):
        row = self.group_table.currentRow()
        if 0 <= row < len(self.eos_groups):
            group = self.eos_groups[row]
            group["marker"] = symbol
            group["marker_edited"] = True
            self.group_table.blockSignals(True)
            self.group_table.item(row, 0).setIcon(group_marker_icon(symbol))
            self.group_table.blockSignals(False)
            self._update_preview()

    def _update_preview(self, *_args):
        self.reference_summary = None
        protocol = self.current_protocol()
        is_eos = protocol is not None and protocol.kind == "eos"
        is_response = protocol is not None and protocol.kind == "response"
        if self.preview_plot is None or self._response_canvas != is_response:
            if self.preview_plot is not None:
                self.preview_layout.removeWidget(self.preview_plot)
                self.preview_plot.hide()
                self.preview_plot.deleteLater()
            self._response_canvas = is_response
            if is_response:
                import NepTrainKit.ui.canvas.pyqtgraph
                import pyqtgraph as pg
                self.preview_plot = pg.PlotWidget(self)
                self.preview_plot.addLegend(offset=(-10, 10))
                self.preview_plot.showGrid(x=True, y=True, alpha=.2)
            else:
                from NepTrainKit.ui.canvas.curve_factory import create_curve_plot
                self.preview_plot = create_curve_plot(self)
            self.preview_plot.setMinimumHeight(260)
            self.preview_plot.viewport().installEventFilter(self)
            self.preview_layout.insertWidget(self.preview_layout.indexOf(self.preview_summary), self.preview_plot, 1)
        if hasattr(self.preview_plot, "point_hover"):
            self.preview_plot.point_hover.hide()
        self.preview_plot.clear()
        self.preview_plot.setMaximumHeight(320 if is_response else 16777215)
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
        self.preview_marker.setVisible(bool(is_eos and group and group["summary"]))
        if group:
            self.preview_marker.set_marker(group["marker"])
        self._show_issues()
        summary = self.reference_summary
        self.preview_plot.setVisible(summary is not None and (is_eos or is_response))
        if summary is None:
            return
        preview = presentation_for(protocol.kind).draw_import_preview
        if preview:
            text = preview(self, summary, group)
        elif protocol.kind == "response":
            import pyqtgraph as pg
            self.preview_title.setText(self.tr("Reference response preview"))
            self.preview_plot.setLabel("bottom", summary["x_label"] if protocol.spec.mode == "scan" else self.tr("State / combination"))
            self.preview_plot.setLabel("left", summary["unit"])
            self.group_table.blockSignals(True)
            self.group_table.setHorizontalHeaderLabels([self.tr("Reference group"), self.tr("Frames"), self.tr("Response points"), self.tr("Status")])
            self.group_table.setRowCount(len(summary['series']) + len(summary['excluded']))
            for row, group in enumerate(summary['series'] + summary['excluded']):
                values = [group.get('name', group.get('group')), len(group.get('frame_indices', group.get('indices', []))),
                          len(group.get('indices', [])) if 'name' in group else '—', group.get('reason', self.tr('Checks passed'))]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value)); item.setToolTip(str(value)); self.group_table.setItem(row, column, item)
            self.group_table.blockSignals(False); self.group_table.show()
            combined = protocol.spec.mode in ('four_state', 'cross', 'combination')
            groups = [dict(name=self.tr('Reference'), indices=list(range(len(summary['rows']))))] if combined else summary['series']
            for index, series in enumerate(groups):
                rows = [summary['rows'][i] for i in series['indices']]
                selected = [i for i in series['indices'] if summary['reference'][i] is not None]
                x = [summary['rows'][i]['x'] for i in selected]
                if combined or not all(isinstance(value, (int, float)) for value in x):
                    x = list(range(1, len(selected)+1))
                self.preview_plot.plot(x, [summary['reference'][i] for i in selected],
                                       pen=None if combined or protocol.spec.second_axis else pg.mkPen((index, max(1, len(groups))), width=2),
                                       symbol="o", symbolSize=6, name=series['name'] if len(groups) <= 4 else None)
            roles = {'unknown': self.tr('Not declared'), 'training': self.tr('Training regression'),
                     'development': self.tr('Development validation'), 'validation': self.tr('Validation (user declared)')}
            text = self.tr("{groups} complete groups · {points} response points · {excluded} excluded groups\nRole: {role} · {unit}").format(
                groups=len(summary["series"]), points=len(summary["rows"]), excluded=len(summary["excluded"]),
                role=roles[protocol.spec.role], unit=summary["unit"])
        else:
            counts = summary["label_counts"]
            text = self.tr(
                "{count} structures · {atoms} atoms\nValid labels: E {energy} structures; F {forces} atoms; "
                "V {virials} structures; MF {mforces} atoms."
            ).format(
                count=summary["frames"], atoms=summary["atoms"],
                energy=counts["energy"]["structures"], forces=counts["force"]["atoms"],
                virials=counts["virial"]["structures"], mforces=counts["mforce"]["atoms"])
        self.preview_summary.setText(text)

    # -- validation -----------------------------------------------------------
    def validation_errors(self) -> list[str]:
        errors = []
        if self.current_protocol() is None:
            if self.protocol_combo.currentData() == "response":
                try:
                    get_protocol("response", analysis=self.analysis_editor.settings())
                except (ValueError, TypeError) as error:
                    errors.append(translate_runtime_message(str(error)))
            else:
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
            "provenance": dict(getattr(self, "provenance_extra", {}), **({"notes": self.notes_edit.text().strip()} if self.notes_edit.text().strip() else {})),
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
        kwargs = dict(name=values["name"], provenance=values["provenance"], protocol=values["protocol"],
                      expected_hash=sha256_bytes(self._loaded_bytes))
        if values["protocol"] == "response":
            kwargs["analysis"] = self.analysis_editor.settings()
        grouped = values["protocol"] == "eos" and bool(self.group_field_combo.currentData())
        if grouped:
            kwargs.pop("protocol")
            kwargs.update(
                group_field=self.group_field_combo.currentData(),
                group_names=[group["key"] for group in self.eos_groups if group["include"]],
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
        if values["protocol"] == "eos":
            groups = {g["key"]: g for g in self.eos_groups if g["include"]}
            assignments = []
            for record in records:
                group = groups[(record["source_ref"].get("eos_group") or {}).get("value", "")]
                assignments.append((record["id"], group["marker"], group.get("marker_edited", False)))
            remember_import_markers(service.library.root, assignments)
        return records
