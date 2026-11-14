"""Review multiple independent reference datasets without running model predictions."""

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QTableWidgetItem, QVBoxLayout
from qfluentwidgets import CaptionLabel, CheckBox, EditableComboBox, PushButton, TableWidget

from NepTrainKit.core.validation.library import sha256_bytes
from NepTrainKit.core.validation.protocols import get_protocol, read_frames
from NepTrainKit.core.validation.protocols.response import suggest_response_settings
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.threads import BackgroundTask


def inspect_references(paths, stopped=lambda: False):
    from NepTrainKit.core.magnetic_response import structure_fingerprint

    result = []
    for source in paths:
        if stopped():
            break
        path = Path(source)
        row = dict(path=str(path), name=path.parent.name + " / " + path.stem, protocol="response", provenance={})
        try:
            data = path.read_bytes()
            frames = read_frames(data)
            settings = suggest_response_settings(frames)
            protocol = get_protocol("response", analysis=settings)
            # A real volume scan may use the dedicated EOS fit. Other scans keep their own axis.
            eos = get_protocol("eos")
            if (
                settings["mode"] == "scan"
                and settings["x_field"] == "volume_ratio"
                and not settings["second_axis"]
                and not any(i.level == "error" for i in eos.check(frames))
            ):
                protocol = eos
            issues = protocol.check(frames)
            row.update(
                expected_hash=sha256_bytes(data),
                protocol=protocol.kind,
                analysis=settings if protocol.kind == "response" else None,
                frames=len(frames),
            )
            errors = [i.message for i in issues if i.level == "error"]
            row["error"] = "\n".join(errors)
            if not errors:
                row["summary"] = protocol.summarize_reference(frames)
            fingerprints, geometries = [], []
            for frame in frames:
                if stopped():
                    return result
                fingerprints.append(structure_fingerprint(frame))
                geometry = frame.copy()
                for key in ("spin", "spins", "initial_magmoms", "magmom", "magmoms"):
                    geometry.arrays.pop(key, None)
                geometries.append(structure_fingerprint(geometry))
            row["_inputs"], row["_geometries"] = fingerprints, geometries
        except (OSError, ValueError, TypeError, KeyError) as error:
            row["error"] = str(error)
        result.append(row)
    for row in result:
        overlaps = []
        for other in result:
            if other is row:
                continue
            inputs, geometries = set(other.get("_inputs", [])), set(other.get("_geometries", []))
            exact = sum(value in inputs for value in row.get("_inputs", []))
            geometry = sum(value in geometries for value in row.get("_geometries", []))
            if exact or geometry:
                overlaps.append(
                    dict(
                        dataset=other["name"],
                        dataset_sha256=other.get("expected_hash"),
                        exact_inputs=exact,
                        geometries=geometry,
                    )
                )
        row["provenance"]["batch_overlap"] = dict(
            method="ordered atoms; geometry/spins rounded to 12 decimals", datasets=overlaps
        )
    for row in result:
        row.pop("_inputs", None)
        row.pop("_geometries", None)
    return result


class ReferenceBatchDialog(QDialog):
    def __init__(self, parent, service, paths, *, suite_name="", group_by_directory=False):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Import reference datasets"))
        self.resize(1080, 680)
        self.service = service
        self.rows = []
        self.task = None
        layout = QVBoxLayout(self)
        self.suite = EditableComboBox(self)
        self.suite.addItems([s["name"] for s in service.suites()])
        self.suite.setText(suite_name)
        self.suite.setPlaceholderText(self.tr("Choose or name a validation plan"))
        layout.addWidget(self.suite)
        self.table = TableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            [
                self.tr("Import / file"),
                self.tr("Frames"),
                self.tr("Analysis"),
                self.tr("Groups"),
                self.tr("Checks / dataset role"),
            ]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 330)
        self.table.setSelectionBehavior(TableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(TableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        controls = QHBoxLayout()
        self.edit = PushButton(self.tr("Review selected dataset…"), self)
        self.edit.clicked.connect(self._edit)
        self.edit.setEnabled(False)
        controls.addWidget(self.edit)
        controls.addStretch()
        layout.addLayout(controls)
        self.directory_groups = CheckBox(self.tr("Use source folders as topics"), self)
        self.directory_groups.setChecked(group_by_directory)
        layout.addWidget(self.directory_groups)
        self.status = CaptionLabel(
            self.tr("Inspecting reference datasets in the background; no predictions are run."), self
        )
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(self.tr("Import"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(self.tr("Cancel"))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.suite.textChanged.connect(self._validate)
        self.table.itemChanged.connect(self._validate)
        self._validate()
        task = BackgroundTask(QApplication.instance(), show_tip=False)
        self.task = task
        task.succeeded.connect(self._loaded)
        task.failed.connect(self.status.setText)
        task.finished.connect(self._finished)
        task.finished.connect(task.deleteLater)
        self.destroyed.connect(task.stop_work)
        task.start_work(
            inspect_references, sorted({str(Path(p).resolve()) for p in paths}), task.isInterruptionRequested
        )

    @Slot(object)
    def _loaded(self, rows):
        self.rows = rows
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self._paint_row(i, row)
        self.table.blockSignals(False)
        self.status.setText(
            self.tr("Review analysis mappings and dataset roles. Invalid files are not selected automatically.")
        )
        self.edit.setEnabled(bool(rows))

    def _paint_row(self, index, row):
        item = QTableWidgetItem(row["name"])
        item.setToolTip(row["path"])
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked if row.get("error") else Qt.CheckState.Checked)
        self.table.setItem(index, 0, item)
        recipe = row.get("analysis") or {}
        summary = row.get("summary") or {}
        modes = {
            key: self.tr(label)
            for key, label in [
                ("scan", "Reference scan"),
                ("difference", "State energy differences"),
                ("four_state", "Four-state response"),
                ("cross", "Cross response"),
                ("combination", "Signed state combination"),
                ("slices", "Accuracy by group"),
            ]
        }
        roles = {
            key: self.tr(label)
            for key, label in [
                ("unknown", "Not declared"),
                ("training", "Training regression"),
                ("development", "Development validation"),
                ("validation", "Validation (user declared)"),
            ]
        }
        for column, value in enumerate(
            (
                row.get("frames", "—"),
                modes.get(recipe.get("mode"), row["protocol"]),
                len(summary.get("series", [])) or 1,
                row.get("error") or roles.get(recipe.get("role", "unknown")),
            ),
            1,
        ):
            cell = QTableWidgetItem(translate_runtime_message(str(value)))
            overlaps = row.get("provenance", {}).get("batch_overlap", {}).get("datasets", [])
            cell.setToolTip(
                cell.text()
                + "\n"
                + self.tr("Overlap within this batch (same ordered inputs / geometry):")
                + "\n"
                + "\n".join(f"{o['dataset']}: {o['exact_inputs']} / {o['geometries']}" for o in overlaps)
            )
            if column == 4 and overlaps and not row.get("error"):
                cell.setText(cell.text() + " · " + self.tr("Overlap found"))
            self.table.setItem(index, column, cell)

    @Slot()
    def _finished(self):
        self.task = None
        self._validate()

    def _validate(self, *_):
        selected = [
            row for i, row in enumerate(self.rows) if self.table.item(i, 0).checkState() == Qt.CheckState.Checked
        ]
        ready = (
            self.task is None
            and bool(self.suite.text().strip())
            and bool(selected)
            and not any(row.get("error") for row in selected)
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ready)

    def _edit(self):
        from .validation_dialogs import BenchmarkImportDialog

        index = self.table.currentRow()
        if not 0 <= index < len(self.rows):
            return
        row = self.rows[index]
        dialog = BenchmarkImportDialog(self, suites=self.service.suites(), suite_name=self.suite.text())
        dialog.protocol_combo.setCurrentIndex(dialog.protocol_combo.findData(row["protocol"]))
        dialog.load_file(row["path"])
        dialog.name_edit.setText(row["name"])
        if row.get("analysis"):
            dialog.analysis_editor.set_settings(row["analysis"])
        elif row["protocol"] == "eos":
            dialog.group_field_combo.setCurrentIndex(dialog.group_field_combo.findData(""))
        dialog.yesButton.setText(self.tr("Use these settings"))
        if dialog.exec():
            values = dialog.values()
            if values["protocol"] == "eos" and dialog.group_field_combo.currentData():
                self.status.setText(self.tr("For multiple EOS groups, use the single-file importer."))
                return
            row.update(
                name=values["name"],
                protocol=values["protocol"],
                provenance=values["provenance"],
                analysis=dialog.analysis_editor.settings() if values["protocol"] == "response" else None,
                expected_hash=sha256_bytes(dialog._loaded_bytes),
                summary=dialog.reference_summary,
                error="",
            )
            self._paint_row(index, row)
            self._validate()

    def import_into(self, *, allow_large=False):
        selected = [
            row for i, row in enumerate(self.rows) if self.table.item(i, 0).checkState() == Qt.CheckState.Checked
        ]
        records = self.service.import_reference_batch([dict(row, allow_large=allow_large) for row in selected])
        name = self.suite.text().strip()
        suite = next((s for s in self.service.suites() if s["name"] == name), None)
        if suite:
            self.service.add_to_suite(suite["id"], [r["id"] for r in records])
        else:
            self.service.create_suite(name, [r["id"] for r in records])
        if self.directory_groups.isChecked():
            target = next(item for item in self.service.suites() if item['name'] == name)
            for source, record in zip(selected, records):
                group = self.service.create_group(target['id'], Path(source['path']).parent.name)
                self.service.assign_group(target['id'], [record['id']], group['id'])
        return records

    def done(self, result):
        if self.task is not None:
            self.task.stop_work()
        super().done(result)
