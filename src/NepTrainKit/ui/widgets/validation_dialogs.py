"""Dialogs for the model validation library."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QFormLayout, QHBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    ComboBox,
    EditableComboBox,
    LineEdit,
    MessageBoxBase,
    PushButton,
    SubtitleLabel,
)

from NepTrainKit import palette
from NepTrainKit.config import Config
from NepTrainKit.core.validation import Library, SetTooLargeError, ValidationService
from NepTrainKit.core.validation.library import ENV_VAR, CONFIG_LIBRARY_PATH, CONFIG_SECTION
from NepTrainKit.core.validation.protocols import Issue, detect_protocol, read_frames
from NepTrainKit.core.validation.service import SOURCE_KINDS
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
    """Pick an EXTXYZ file, confirm the detected protocol and declare its provenance.

    Labels are never assumed to be DFT, so the source kind and the calculation
    conditions are required before the set can be sealed.
    """

    def __init__(self, parent=None, suites: list[dict] | tuple = (), suite_name: str = ""):
        super().__init__(parent)
        self.frames = None
        self.issues: list[Issue] = []
        self._read_error = ""
        self._ranked = []

        self.titleLabel = SubtitleLabel(self.tr("Import benchmark set"), self)
        self.viewLayout.addWidget(self.titleLabel)

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
        self.protocol_combo = ComboBox(self)
        self.protocol_combo.currentIndexChanged.connect(self._check_protocol)
        self.source_combo = ComboBox(self)
        self.source_combo.addItem(self.tr("Select..."), userData=None)
        source_labels = {"DFT": "DFT", "Other": self.tr("Other"), "Demo": self.tr("Demo")}
        for kind in SOURCE_KINDS:
            self.source_combo.addItem(source_labels.get(kind, kind), userData=kind)
        self.conditions_edit = LineEdit(self)
        self.conditions_edit.setPlaceholderText(self.tr("e.g. VASP PBE, 520 eV cutoff, k-spacing 0.2"))
        self.suite_combo = EditableComboBox(self)
        self.suite_combo.setPlaceholderText(self.tr("Optional: add to a suite"))
        for suite in suites:
            self.suite_combo.addItem(suite["name"])
        self.suite_combo.setText(suite_name)

        form = QFormLayout()
        form.addRow(BodyLabel(self.tr("File"), self), file_row)
        form.addRow(BodyLabel(self.tr("Name"), self), self.name_edit)
        form.addRow(BodyLabel(self.tr("Protocol"), self), self.protocol_combo)
        form.addRow(BodyLabel(self.tr("Reference source"), self), self.source_combo)
        form.addRow(BodyLabel(self.tr("Conditions"), self), self.conditions_edit)
        form.addRow(BodyLabel(self.tr("Suite"), self), self.suite_combo)
        self.viewLayout.addLayout(form)

        self.issues_label = CaptionLabel(self)
        self.issues_label.setWordWrap(True)
        self.viewLayout.addWidget(self.issues_label)
        self.status_label = CaptionLabel(self)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")
        self.viewLayout.addWidget(self.status_label)

        for widget in (self.name_edit, self.conditions_edit):
            widget.textChanged.connect(self._validate)
        self.source_combo.currentIndexChanged.connect(self._validate)

        self.yesButton.setText(self.tr("Import"))
        self.widget.setMinimumWidth(520)
        self._validate()

    # -- file and protocol ----------------------------------------------------
    def _browse(self) -> None:
        path = call_path_dialog(
            self, self.tr("Select benchmark set"), "select", file_filter="EXTXYZ (*.xyz *.extxyz);;All files (*)"
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str | Path) -> None:
        path = str(path).strip()
        self.path_edit.setText(path)
        self.frames, self._read_error, self._ranked = None, "", []
        self.protocol_combo.blockSignals(True)
        self.protocol_combo.clear()
        self.protocol_combo.blockSignals(False)
        if path:
            try:
                self.frames = read_frames(Path(path).expanduser().read_bytes())
            except Exception as error:  # noqa: BLE001 - any parse failure blocks the import
                self._read_error = str(error) or type(error).__name__
            else:
                self._ranked = detect_protocol(self.frames)
                if not self.name_edit.text().strip():
                    self.name_edit.setText(Path(path).stem)
        self.protocol_combo.blockSignals(True)
        for confidence, protocol in self._ranked:
            self.protocol_combo.addItem(
                self.tr("{title} (match {confidence:.0%})").format(title=protocol.title, confidence=confidence),
                userData=protocol.kind,
            )
        self.protocol_combo.blockSignals(False)
        self._check_protocol()

    def current_protocol(self):
        index = self.protocol_combo.currentIndex()
        if 0 <= index < len(self._ranked):
            return self._ranked[index][1]
        return None

    def _check_protocol(self, *_args) -> None:
        protocol = self.current_protocol()
        self.issues = protocol.check(self.frames) if protocol is not None and self.frames is not None else []
        lines = []
        for issue in self.issues:
            color = palette.PLOT_REJECT if issue.level == "error" else palette.PLOT_SELECTED
            label = self.tr("Error") if issue.level == "error" else self.tr("Warning")
            lines.append(f'<span style="color:{color};">{label}:</span> {translate_runtime_message(issue.message)}')
        if protocol is not None and not self.issues:
            lines.append(self.tr("No issues found for {title}.").format(title=protocol.title))
        self.issues_label.setText("<br>".join(lines))
        self._validate()

    # -- validation -----------------------------------------------------------
    def validation_errors(self) -> list[str]:
        errors = []
        if not self.path_edit.text().strip():
            errors.append(self.tr("Choose a benchmark file."))
        elif self._read_error:
            errors.append(self.tr("The file cannot be read as EXTXYZ: {error}").format(error=self._read_error))
        elif self.current_protocol() is None:
            errors.append(self.tr("No validation protocol recognises this file."))
        elif any(issue.level == "error" for issue in self.issues):
            errors.append(self.tr("Fix the errors above before importing."))
        if not self.name_edit.text().strip():
            errors.append(self.tr("Enter a name for the set."))
        if self.source_combo.currentData() is None:
            errors.append(self.tr("Declare where the reference labels come from."))
        if not self.conditions_edit.text().strip():
            errors.append(self.tr("Describe the reference calculation conditions."))
        return errors

    def _validate(self, *_args) -> None:
        errors = self.validation_errors()
        self.yesButton.setEnabled(not errors)
        self.status_label.setText(errors[0] if errors else "")

    def validate(self) -> bool:
        return not self.validation_errors()

    def values(self) -> dict:
        protocol = self.current_protocol()
        return {
            "path": self.path_edit.text().strip(),
            "name": self.name_edit.text().strip(),
            "protocol": protocol.kind if protocol is not None else None,
            "provenance": {
                "source_kind": self.source_combo.currentData(),
                "conditions": self.conditions_edit.text().strip(),
            },
            "suite": self.suite_combo.text().strip(),
        }

    def import_into(self, service, confirm_large: Callable[[SetTooLargeError], bool]) -> dict | None:
        """Seal the set; ``confirm_large`` decides whether an oversized file is copied anyway.

        Returns the set record, or ``None`` when the user declined a large copy.
        Rejections propagate as ``SetRejectedError`` / ``ValueError``.
        """
        values = self.values()
        kwargs = dict(name=values["name"], provenance=values["provenance"], protocol=values["protocol"])
        try:
            record = service.import_set(values["path"], **kwargs)
        except SetTooLargeError as error:
            if not confirm_large(error):
                return None
            record = service.import_set(values["path"], allow_large=True, **kwargs)
        if values["suite"]:
            suite = next((item for item in service.suites() if item["name"] == values["suite"]), None)
            if suite is None:
                service.create_suite(values["suite"], [record["id"]])
            else:
                service.add_to_suite(suite["id"], [record["id"]])
        return record
