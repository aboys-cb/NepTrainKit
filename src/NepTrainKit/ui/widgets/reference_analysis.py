"""A small recipe editor for fixed-structure reference analyses."""

import csv
import json

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    EditableComboBox,
    LineEdit,
    PushButton,
    TableWidget,
)

from NepTrainKit.core.validation.grouping import eos_group_fields
from NepTrainKit.core.validation.protocols.response import ResponseSpec


def _names(text):
    return tuple(x.strip() for x in text.split(",") if x.strip())


class ReferenceAnalysisEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames = []
        self.frame_metadata = {}
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.preset = ComboBox(self)
        for label, mode in [
            ("Reference scan", "scan"),
            ("State energy differences", "difference"),
            ("Four-state response", "four_state"),
            ("Cross response", "cross"),
            ("Signed state combination", "combination"),
            ("Accuracy by group", "slices"),
        ]:
            self.preset.addItem(self.tr(label), userData=mode)
        form.addRow(BodyLabel(self.tr("Analysis"), self), self.preset)
        self.group_fields = LineEdit(self)
        self.group_fields.setPlaceholderText(self.tr("Grouping fields, separated by commas; empty means one group"))
        form.addRow(BodyLabel(self.tr("Group fields"), self), self.group_fields)
        self.x_field = EditableComboBox(self)
        self.x_field.setText("volume_per_atom")
        self.second_axis = EditableComboBox(self)
        self.state_fields = LineEdit(self)
        self.state_fields.setPlaceholderText(self.tr("State fields, separated by commas"))
        self.reference = EditableComboBox(self)
        self.quantity = ComboBox(self)
        for quantity in (
            "energy",
            "pressure",
            "stress_xx",
            "stress_yy",
            "stress_zz",
            "stress_xy",
            "stress_yz",
            "stress_zx",
        ):
            self.quantity.addItem(
                self.tr("Energy")
                if quantity == "energy"
                else self.tr("Pressure")
                if quantity == "pressure"
                else self.tr("Stress") + " " + quantity.removeprefix("stress_"),
                userData=quantity,
            )
        self.reference_field = EditableComboBox(self)
        self.reference_field.setPlaceholderText(self.tr("Empty: use energy or virial/stress labels"))
        self.reference_scale = LineEdit(self)
        self.reference_scale.setText("1")
        self.normalization = ComboBox(self)
        for label, value in [("Per atom", "atom"), ("Per cell", "total"), ("Per selected element", "element")]:
            self.normalization.addItem(self.tr(label), userData=value)
        self.element = LineEdit(self)
        self.element.setPlaceholderText(self.tr("Element symbol"))
        self.energy_unit = ComboBox(self)
        self.energy_unit.addItems(["meV", "μeV"])
        self.role = ComboBox(self)
        for label, value in [
            ("Not declared", "unknown"),
            ("Training regression", "training"),
            ("Development validation", "development"),
            ("Validation (user declared)", "validation"),
        ]:
            self.role.addItem(self.tr(label), userData=value)
        self._rows = {}
        for label, widget in [
            ("Scan axis", self.x_field),
            ("Second axis (optional)", self.second_axis),
            ("State fields", self.state_fields),
            ("Reference state", self.reference),
            ("Quantity", self.quantity),
            ("Reference field (optional)", self.reference_field),
            ("Reference conversion factor", self.reference_scale),
            ("Normalization", self.normalization),
            ("Element", self.element),
            ("Energy unit", self.energy_unit),
            ("Dataset role", self.role),
        ]:
            title = BodyLabel(self.tr(label), self)
            form.addRow(title, widget)
            self._rows[widget] = title
        self.relative = CheckBox(self.tr("Subtract the reference-curve minimum from both curves"), self)
        self.skip = CheckBox(self.tr("Explicitly exclude incomplete or invalid groups"), self)
        form.addRow(self.relative)
        form.addRow(self.skip)
        self.exchange = CheckBox(self.tr("Extract bilinear unit-spin exchange projections"), self)
        self.exchange.setToolTip(
            self.tr(
                "Assumption: H = sum e_i^T J_ij e_j with unit spins and a positive sign. Reports measured projections only; local-bond interpretation requires the correct periodic multiplicity."
            )
        )
        form.addRow(self.exchange)
        self.pair_fields = LineEdit(self)
        self.pair_fields.setText("site_i, site_j")
        self.index_base = ComboBox(self)
        self.index_base.addItems(["0", "1"])
        self.multiplicity = LineEdit(self)
        self.multiplicity.setText("1")
        for label, widget in [
            ("Ordered atom-pair fields", self.pair_fields),
            ("Atom index base", self.index_base),
            ("Interaction multiplicity", self.multiplicity),
        ]:
            title = BodyLabel(self.tr(label), self)
            form.addRow(title, widget)
            self._rows[widget] = title
        self.exchange.stateChanged.connect(self._update_visibility)
        self.exchange.stateChanged.connect(lambda *_: self.changed.emit())
        self.pair_fields.textChanged.connect(lambda *_: self.changed.emit())
        self.multiplicity.textChanged.connect(lambda *_: self.changed.emit())
        self.index_base.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.terms_box = QWidget(self)
        layout = QVBoxLayout(self.terms_box)
        layout.setContentsMargins(0, 0, 0, 0)
        self.terms = TableWidget(self.terms_box)
        self.terms.setColumnCount(2)
        self.terms.setHorizontalHeaderLabels([self.tr("State values (comma separated)"), self.tr("Coefficient")])
        self.terms.horizontalHeader().setStretchLastSection(True)
        self.terms.setColumnWidth(0, 210)
        self.terms.setFixedHeight(150)
        layout.addWidget(self.terms)
        buttons = QHBoxLayout()
        self.add_term = PushButton(self.tr("Add term"), self)
        self.remove_term = PushButton(self.tr("Remove term"), self)
        buttons.addWidget(self.add_term)
        buttons.addWidget(self.remove_term)
        buttons.addStretch()
        layout.addLayout(buttons)
        form.addRow(self.terms_box)
        self.metadata_button = PushButton(self.tr("Import grouping metadata CSV…"), self)
        self.metadata_button.setToolTip(
            self.tr(
                "Join by frame_index (zero based) or a unique frame_id. Other columns define grouping or scan metadata, not calculation labels."
            )
        )
        self.metadata_button.clicked.connect(self._load_metadata)
        form.addRow(self.metadata_button)
        self.metadata_status = CaptionLabel(self)
        self.metadata_status.setWordWrap(True)
        form.addRow(self.metadata_status)
        self.hint = CaptionLabel(
            self.tr(
                "Four-state response reports a signed energy combination, not an assumed J/D Hamiltonian parameter. Cross response uses binary states 0/1. Stress and pressure are in GPa; reference conversion is explicit."
            ),
            self,
        )
        self.hint.setWordWrap(True)
        form.addRow(self.hint)
        self.add_term.clicked.connect(lambda: self.terms.insertRow(self.terms.rowCount()))
        self.remove_term.clicked.connect(lambda: self.terms.removeRow(self.terms.currentRow()))
        self.add_term.clicked.connect(lambda: self.changed.emit())
        self.remove_term.clicked.connect(lambda: self.changed.emit())
        self.terms.itemChanged.connect(lambda *_: self.changed.emit())
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.normalization.currentIndexChanged.connect(self._update_visibility)
        self.quantity.currentIndexChanged.connect(self._update_visibility)
        self.state_fields.textChanged.connect(self._reference_options)
        for widget in (
            self.group_fields,
            self.x_field,
            self.second_axis,
            self.state_fields,
            self.reference,
            self.reference_field,
            self.reference_scale,
            self.element,
        ):
            widget.textChanged.connect(lambda *_: self.changed.emit())
        for widget in (self.quantity, self.normalization, self.energy_unit, self.role):
            widget.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.relative.stateChanged.connect(lambda *_: self.changed.emit())
        self.skip.stateChanged.connect(lambda *_: self.changed.emit())
        self.relative.setChecked(True)
        self._preset_changed()

    def _update_visibility(self, *_):
        mode = self.preset.currentData()
        energy = self.quantity.currentData() == "energy"
        visible = {
            self.x_field: mode == "scan",
            self.second_axis: mode == "scan",
            self.state_fields: mode in ("difference", "four_state", "cross", "combination"),
            self.reference: mode == "difference",
            self.element: energy and self.normalization.currentData() == "element",
            self.quantity: mode != "slices",
            self.normalization: energy and mode != "slices",
            self.energy_unit: energy and mode != "slices",
            self.reference_field: mode != "slices",
            self.reference_scale: mode != "slices",
        }
        for widget, show in visible.items():
            widget.setVisible(show)
            self._rows[widget].setVisible(show)
        self.exchange.setVisible(mode == "four_state")
        for widget in (self.pair_fields, self.index_base, self.multiplicity):
            show = mode == "four_state" and self.exchange.isChecked()
            widget.setVisible(show)
            self._rows[widget].setVisible(show)
        self.relative.setVisible(mode == "scan")
        self.terms_box.setVisible(mode in ("four_state", "cross", "combination"))
        self.add_term.setVisible(mode == "combination")
        self.remove_term.setVisible(mode == "combination")
        self.terms.setEditTriggers(
            QAbstractItemView.EditTrigger.AllEditTriggers
            if mode == "combination"
            else QAbstractItemView.EditTrigger.NoEditTriggers
        )

    def _preset_changed(self, *_):
        mode = self.preset.currentData()
        terms = (
            [([1, 1], 1), ([-1, -1], 1), ([1, -1], -1), ([-1, 1], -1)]
            if mode == "four_state"
            else [([1, 1], 1), ([0, 0], 1), ([1, 0], -1), ([0, 1], -1)]
        )
        self.terms.blockSignals(True)
        self.terms.setRowCount(4)
        for row, (states, coefficient) in enumerate(terms):
            self.terms.setItem(row, 0, QTableWidgetItem(", ".join(map(str, states))))
            self.terms.setItem(row, 1, QTableWidgetItem(str(coefficient)))
        self.terms.blockSignals(False)
        if mode in ("four_state", "cross", "combination"):
            self.normalization.setCurrentIndex(self.normalization.findData("total"))
        self._update_visibility()
        self.changed.emit()

    def _reference_options(self, *_):
        names = _names(self.state_fields.text())
        previous = self.reference.text()
        self.reference.blockSignals(True)
        self.reference.clear()
        if names:
            values = sorted({", ".join(str(a.info.get(name, "")) for name in names) for a in self.frames})
            self.reference.addItems(values)
        self.reference.setText(previous)
        self.reference.blockSignals(False)

    def set_frames(self, frames):
        self.blockSignals(True)
        self.frames = frames or []
        self.frame_metadata = {}
        self.metadata_status.setText("")
        available = eos_group_fields(self.frames)
        for widget, virtual in (
            (self.x_field, ["volume_per_atom", "volume"]),
            (self.second_axis, [""]),
            (self.reference_field, [""]),
        ):
            previous = widget.text()
            widget.blockSignals(True)
            widget.clear()
            widget.addItems(virtual + available)
            widget.setText(previous)
            widget.blockSignals(False)
        if not self.frames:
            self.blockSignals(False)
            return
        from NepTrainKit.core.validation.protocols.response import suggest_response_settings

        self.set_settings(suggest_response_settings(self.frames))
        self.blockSignals(False)
        self.changed.emit()

    def set_settings(self, settings):
        spec = ResponseSpec.from_dict(settings)
        self.blockSignals(True)
        self.preset.setCurrentIndex(self.preset.findData(spec.mode))
        self.group_fields.setText(", ".join(spec.group_fields))
        self.x_field.setText(spec.x_field)
        self.second_axis.setText(spec.second_axis)
        self.state_fields.setText(", ".join(spec.state_fields))
        self.reference.setText(spec.reference)
        self.quantity.setCurrentIndex(self.quantity.findData(spec.quantity))
        self.reference_field.setText(spec.reference_field)
        self.reference_scale.setText(str(spec.reference_scale))
        self.normalization.setCurrentIndex(self.normalization.findData(spec.normalization))
        self.element.setText(spec.element)
        self.energy_unit.setCurrentIndex(self.energy_unit.findText(spec.energy_unit))
        self.role.setCurrentIndex(self.role.findData(spec.role))
        self.relative.setChecked(spec.relative)
        self.skip.setChecked(spec.skip_incomplete)
        self.frame_metadata = spec.frame_metadata
        self.exchange.setChecked(spec.extract_exchange)
        self.pair_fields.setText(", ".join(spec.pair_fields))
        self.index_base.setCurrentIndex(spec.index_base)
        self.multiplicity.setText(str(spec.multiplicity))
        if spec.mode == "combination":
            self.terms.blockSignals(True)
            self.terms.setRowCount(len(spec.terms))
            for row, (state, coefficient) in enumerate(spec.terms):
                self.terms.setItem(row, 0, QTableWidgetItem(", ".join(map(str, state))))
                self.terms.setItem(row, 1, QTableWidgetItem(str(coefficient)))
            self.terms.blockSignals(False)
        self._reference_options()
        self._update_visibility()
        self.blockSignals(False)
        self.changed.emit()

    def _load_metadata(self):
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Import grouping metadata"), "", "CSV (*.csv)")
        if not path:
            return
        try:
            self.load_metadata(path)
        except (OSError, ValueError, TypeError) as error:
            self.metadata_status.setText(str(error))

    def load_metadata(self, path):
        mapped = {}
        ids = {}
        for i, frame in enumerate(self.frames):
            ids.setdefault(str(frame.info.get("frame_id", "")), []).append(i)
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"frame_index", "frame_id"} & set(reader.fieldnames):
                raise ValueError(self.tr("Metadata CSV needs frame_index (zero based) or frame_id."))
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(self.tr("Metadata CSV column names must be unique."))
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(self.tr("Metadata CSV rows must match the header columns."))
                if not row.get("frame_index", "").strip() and not row.get("frame_id", "").strip():
                    raise ValueError(self.tr("Every metadata row needs a frame index or ID."))
                if row.get("frame_index", "").strip():
                    index = int(row["frame_index"])
                else:
                    matches = ids.get(row.get("frame_id", ""), [])
                    if len(matches) != 1:
                        raise ValueError(self.tr("Each metadata frame_id must identify exactly one frame."))
                    index = matches[0]
                if not 0 <= index < len(self.frames) or str(index) in mapped:
                    raise ValueError(self.tr("Metadata contains duplicate or out-of-range frame indices."))
                values = {}
                for key, value in row.items():
                    if key in ("frame_index", "frame_id") or not value.strip():
                        continue
                    try:
                        values[key] = json.loads(value)
                    except ValueError:
                        values[key] = value
                mapped[str(index)] = values
        ResponseSpec.from_dict(dict(self.settings(), frame_metadata=mapped))
        self.frame_metadata = mapped
        self.metadata_status.setText(
            self.tr("Grouping metadata mapped to {count} frames. Source structures and labels are unchanged.").format(
                count=len(mapped)
            )
        )
        self.changed.emit()

    def settings(self):
        terms = []
        if self.preset.currentData() == "combination":
            for row in range(self.terms.rowCount()):
                state = self.terms.item(row, 0)
                coefficient = self.terms.item(row, 1)
                if state is None or coefficient is None:
                    raise ValueError(self.tr("Fill every combination state and coefficient."))
                values = []
                for value in state.text().split(","):
                    try:
                        values.append(json.loads(value.strip()))
                    except ValueError:
                        values.append(value.strip())
                terms.append((values, float(coefficient.text())))
        mode = self.preset.currentData()
        return ResponseSpec(
            mode=mode,
            group_fields=_names(self.group_fields.text()),
            x_field=self.x_field.text(),
            second_axis=self.second_axis.text(),
            state_fields=_names(self.state_fields.text()) if mode not in ("scan", "slices") else (),
            reference=self.reference.text(),
            terms=tuple(terms),
            normalization=self.normalization.currentData(),
            element=self.element.text().strip(),
            energy_unit=self.energy_unit.currentText(),
            quantity="energy" if mode == "slices" else self.quantity.currentData(),
            reference_field=self.reference_field.text().strip(),
            reference_scale=1.0 if mode == "slices" else float(self.reference_scale.text()),
            relative=self.relative.isChecked(),
            skip_incomplete=self.skip.isChecked(),
            role=self.role.currentData(),
            frame_metadata=self.frame_metadata,
            extract_exchange=mode == "four_state" and self.exchange.isChecked(),
            pair_fields=_names(self.pair_fields.text()),
            index_base=self.index_base.currentIndex(),
            multiplicity=float(self.multiplicity.text()) if mode == "four_state" and self.exchange.isChecked() else 1.0,
        ).to_dict()
