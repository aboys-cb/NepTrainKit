"""Preview and exchange portable validation suites without mutating on cancel."""
from PySide6.QtWidgets import QFileDialog, QFormLayout, QHeaderView, QHBoxLayout, QTableWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, CheckBox, EditableComboBox, LineEdit, MessageBoxBase, PushButton, ScrollArea, SubtitleLabel, TableWidget

from NepTrainKit.core.validation.package import available_suite_name, build_package, inspect_package


class ValidationPackageDialog(MessageBoxBase):
    def __init__(self, parent):
        super().__init__(parent)
        self.preview = None
        self.title = SubtitleLabel(self)
        self.viewLayout.addWidget(self.title)
        self.description = BodyLabel(self)
        self.description.setWordWrap(True)
        self.viewLayout.addWidget(self.description)
        self.table = TableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels([self.tr("Reference dataset"), self.tr("Type"), self.tr("Structures")])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(TableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setFixedHeight(max(100, min(220, parent.height() - 500)))
        self.viewLayout.addWidget(self.table)
        self.summary = BodyLabel(self)
        self.summary.setWordWrap(True)
        self.viewLayout.addWidget(self.summary)
        self.status = CaptionLabel(self)
        self.status.setWordWrap(True)
        self.cancelButton.setText(self.tr("Cancel"))
        self.widget.setMinimumWidth(700)

    def show_preview(self, preview):
        self.preview = preview
        self.table.setRowCount(len(preview.references))
        for i, (row, checked) in enumerate(zip(preview.manifest["references"], preview.references)):
            for j, value in enumerate((row["name"], row["protocol"].upper(), checked["frame_count"])):
                item = QTableWidgetItem(str(value))
                item.setToolTip(str(value))
                self.table.setItem(i, j, item)
        models = preview.manifest["models"]
        model_names = ", ".join(m["name"] for m in models)
        self.summary.setToolTip(model_names)
        baseline = next((m["name"] for m in models if m["object"] == preview.manifest["suite"].get("baseline")), self.tr("Not included"))
        self.summary.setText(self.tr("{sets} datasets · {frames} structures · {size:.1f} MB\nModels: {models}\nBaseline model: {baseline}").format(
            sets=len(preview.references), frames=sum(r["frame_count"] for r in preview.references),
            size=len(preview.data)/1e6, models=(model_names if len(model_names) <= 100 else model_names[:97] + "…") or self.tr("Not included"), baseline=baseline))
        self.status.setText(self.tr("Checks passed. Results will be calculated locally after import.") if not preview.warnings else
                            self.tr("Checks passed with {count} fitting warnings; import is allowed.").format(count=len(preview.warnings)))
        self.status.setToolTip("\n".join(preview.warnings))


class ExportValidationPackageDialog(ValidationPackageDialog):
    def __init__(self, parent, service, suite_id, model_ids=()):
        super().__init__(parent)
        self.service, self.suite_id = service, suite_id
        suite = service.get_suite(suite_id)
        self.title.setText(self.tr("Export validation sharing package"))
        self.description.setText(self.tr("Share “{name}” with its reference structures and DFT labels. Choose whether to include model files.").format(name=suite["name"]))
        self.include_models = CheckBox(self.tr("Include model files"), self)
        self.viewLayout.addWidget(self.include_models)
        self.model_checks = []
        self.model_area = ScrollArea(self)
        self.model_area.setWidgetResizable(True)
        self.model_area.setFixedHeight(100)
        self.model_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        model_host = QWidget(self.model_area)
        model_layout = QVBoxLayout(model_host)
        model_layout.setContentsMargins(8, 0, 8, 0)
        self.model_area.setWidget(model_host)
        self.viewLayout.addWidget(self.model_area)
        ids = list(dict.fromkeys([*model_ids, *([suite["baseline_model_id"]] if suite["baseline_model_id"] else [])]))
        for model_id in ids:
            if model_id is None or model_id < 1:
                continue
            model = service.get_model(model_id)
            if not model["content_hash"]:
                continue
            label = model["name"]
            if model_id == suite["baseline_model_id"]:
                label += self.tr(" · baseline")
            check = CheckBox(label, self)
            check.setChecked(True)
            check.setVisible(False)
            check.toggled.connect(self._update_preview)
            self.model_checks.append((model_id, check))
            model_layout.addWidget(check)
        model_layout.addStretch(1)
        self.include_models.setEnabled(bool(self.model_checks))
        self.include_models.toggled.connect(self._update_preview)
        self.viewLayout.addWidget(self.status)
        self.yesButton.setText(self.tr("Export…"))
        self._update_preview()

    def _update_preview(self, *_):
        include = self.include_models.isChecked()
        self.model_area.setVisible(include)
        for _, check in self.model_checks:
            check.setVisible(include)
        ids = [model_id for model_id, check in self.model_checks if include and check.isChecked()]
        try:
            self.show_preview(build_package(self.service, self.suite_id, ids))
        except (ValueError, OSError) as error:
            self.preview = None
            self.status.setText(str(error))
        self.yesButton.setEnabled(self.preview is not None)


class ImportValidationPackageDialog(ValidationPackageDialog):
    def __init__(self, parent, service):
        super().__init__(parent)
        self.service = service
        self.title.setText(self.tr("Import validation sharing package"))
        self.description.setText(self.tr("Preview the package, then create a new suite. Existing suites and baseline models remain unchanged."))
        row = QHBoxLayout()
        self.path_edit = LineEdit(self)
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(self.tr("Choose a .ntkpack file"))
        self.browse_button = PushButton(self.tr("Browse…"), self)
        self.browse_button.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(self.browse_button)
        self.viewLayout.insertLayout(2, row)
        self.name_edit = LineEdit(self)
        self.name_edit.setMaxLength(200)
        self.project_combo = EditableComboBox(self)
        self.project_combo.setPlaceholderText(self.tr("Choose a project or type a new name"))
        for project in service.projects():
            self.project_combo.addItem(project["name"])
        self.project_combo.setText("")
        self.project_combo.setEnabled(False)
        form = QFormLayout()
        form.addRow(self.tr("New suite name"), self.name_edit)
        form.addRow(self.tr("Model project"), self.project_combo)
        self.viewLayout.addLayout(form)
        self.destination = CaptionLabel(self)
        self.destination.setWordWrap(True)
        self.viewLayout.addWidget(self.destination)
        self.viewLayout.addWidget(self.status)
        self.yesButton.setText(self.tr("Import as new suite"))
        self.name_edit.textChanged.connect(self._validate)
        self.project_combo.textChanged.connect(self._validate)
        self._validate()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Import validation sharing package"), "", "NepTrainKit (*.ntkpack)")
        if path:
            self.load_path(path)

    def load_path(self, path):
        self.preview = None
        self.table.setRowCount(0)
        self.summary.clear()
        self.name_edit.clear()
        self.project_combo.setEnabled(False)
        self.path_edit.setText(str(path))
        try:
            preview = inspect_package(path)
            self.show_preview(preview)
            self.name_edit.setText(available_suite_name(self.service, preview.manifest["suite"]["name"]))
            self.project_combo.setEnabled(bool(preview.manifest["models"]))
        except (ValueError, OSError) as error:
            self.status.setText(str(error))
        self._validate()

    def _validate(self, *_):
        name = self.name_edit.text().strip()
        project = self.project_combo.text().strip()
        need_project = self.preview is not None and bool(self.preview.manifest["models"])
        valid = self.preview is not None and bool(name) and (not need_project or bool(project) and len(project) <= 200)
        self.yesButton.setEnabled(valid)
        self.destination.setText(self.tr("Will create: {name}").format(name=available_suite_name(self.service, name)) if name else "")
