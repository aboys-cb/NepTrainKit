"""Card for deterministic surface-slab scans across explicit Miller planes."""

from __future__ import annotations

import itertools

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QHBoxLayout, QHeaderView, QTableWidget, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox, PushButton, SpinBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import RandomSlabOperation, RandomSlabParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import MessageManager, translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


class MillerIndexTableInput(QWidget):
    """Editable three-column h/k/l list with one Miller plane per row."""

    planesChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutating_rows = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels((self.tr("h"), self.tr("k"), self.tr("l")))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.setMinimumHeight(112)
        self.table.setMaximumHeight(136)
        self.table.setAccessibleName(self.tr("Miller plane list"))
        root.addWidget(self.table)

        button_row = QWidget(self)
        buttons = QHBoxLayout(button_row)
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        self.add_button = PushButton(self.tr("Add plane"), button_row)
        self.remove_button = PushButton(self.tr("Remove selected"), button_row)
        self.remove_button.setEnabled(False)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        root.addWidget(button_row)
        self.add_button.clicked.connect(self._add_clicked)
        self.remove_button.clicked.connect(self._remove_selected)
        self.table.itemSelectionChanged.connect(
            lambda: self.remove_button.setEnabled(bool(self.table.selectionModel().selectedRows()))
        )

    def planes(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            tuple(int(self.table.cellWidget(row, column).value()) for column in range(3))
            for row in range(self.table.rowCount())
        )

    def set_planes(self, planes) -> None:
        self._mutating_rows = True
        try:
            self.table.setRowCount(0)
            for plane in planes:
                self._append_plane(tuple(int(value) for value in plane), emit=False)
        finally:
            self._mutating_rows = False
        self.planesChanged.emit()

    def _append_plane(self, plane: tuple[int, int, int], *, emit: bool = True) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate(plane):
            editor = SpinBox(self.table)
            editor.setRange(-20, 20)
            editor.setValue(value)
            editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
            editor.setAccessibleName(
                self.tr("Miller index {axis}, row {row}").format(axis=("h", "k", "l")[column], row=row + 1)
            )
            editor.valueChanged.connect(lambda _value: self.planesChanged.emit())
            editor.editingFinished.connect(self._normalize_after_edit)
            self.table.setCellWidget(row, column, editor)
        if emit:
            self.planesChanged.emit()

    def _add_clicked(self) -> None:
        self._append_plane((0, 0, 1))
        row = self.table.rowCount() - 1
        self.table.selectRow(row)
        self.table.cellWidget(row, 0).setFocus()

    def _remove_selected(self) -> None:
        rows = sorted((index.row() for index in self.table.selectionModel().selectedRows()), reverse=True)
        self._mutating_rows = True
        try:
            for row in rows:
                self.table.removeRow(row)
        finally:
            self._mutating_rows = False
        if rows:
            self.planesChanged.emit()

    def _normalize_after_edit(self) -> None:
        if self._mutating_rows:
            return
        try:
            normalized = RandomSlabOperation.canonical_hkl_list(self.planes())
        except ValueError:
            self.planesChanged.emit()
            return
        if normalized != self.planes():
            self.set_planes(normalized)


@CardManager.register_card
class RandomSlabCard(MakeDataCard):
    """Cut deterministic surface slabs from a three-periodic bulk structure."""

    group = "Surface"
    card_name = "Surface Slab Scan"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    _PRESETS = {
        "low_index": ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
        "111": ((1, 1, 1),),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._preview_input_count = None
        self._updating_planes = False
        self.setTitle(self.tr("Surface Slab Scan"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("random_slab_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)

        self.preset_combo = ComboBox(self.setting_widget)
        self.preset_combo.addItem(self.tr("Common low-index planes (100, 110, 111)"), userData="low_index")
        self.preset_combo.addItem(self.tr("Single plane (111)"), userData="111")
        self.preset_combo.addItem(self.tr("Custom plane list"), userData="custom")
        self.plane_table = MillerIndexTableInput(self.setting_widget)
        self.plane_table.set_planes(self._PRESETS["low_index"])

        plane_section = InspectorSection(
            self.tr("Surface planes"),
            self.setting_widget,
            self.tr(
                "Miller indices refer to the reciprocal basis of the input cell. "
                "Use a conventional cell when you need conventional crystallographic labels."
            ),
        )
        plane_section.addWidget(CompactField(self.tr("Preset"), self.preset_combo, plane_section))
        plane_section.addWidget(self.plane_table)
        self.plane_note = CaptionLabel(
            self.tr(
                "One row is one plane. Proportional rows are reduced and exact duplicates are removed automatically."
            ),
            plane_section,
        )
        self.plane_note.setWordWrap(True)
        self.plane_note.setStyleSheet("color:#8a95a0;")
        plane_section.addWidget(self.plane_note)
        self.legacy_notice = CaptionLabel("", plane_section)
        self.legacy_notice.setWordWrap(True)
        self.legacy_notice.setStyleSheet("color:#c67c00;")
        self.legacy_notice.hide()
        plane_section.addWidget(self.legacy_notice)

        self.layer_frame = self._range_frame("int", 1, 100, (3, 6, 1))
        self.vacuum_frame = self._range_frame("float", 0.0, 100.0, (10.0, 10.0, 1.0), unit="Å")
        self.normal_pbc_checkbox = CheckBox(self.tr("Keep periodicity along surface normal"), self.setting_widget)
        self.normal_pbc_checkbox.setChecked(True)

        geometry_section = InspectorSection(
            self.tr("Slab geometry"),
            self.setting_widget,
            self.tr("Each selected plane is scanned over the two inclusive ranges below."),
        )
        geometry_grid = ResponsiveFormGrid(geometry_section, two_column_threshold=520)
        self.layer_field = CompactField(
            self.tr("Normal repeats"),
            self.layer_frame,
            geometry_section,
            self.tr("ASE surface repeats of the oriented bulk unit; not a guaranteed count of atomic planes."),
        )
        self.vacuum_field = CompactField(
            self.tr("Vacuum per side"),
            self.vacuum_frame,
            geometry_section,
            self.tr("The cell receives this much vacuum above and below, for twice this value in total."),
        )
        geometry_grid.add_field(self.layer_field, span=2)
        geometry_grid.add_field(self.vacuum_field, span=2)
        geometry_grid.add_field(self.normal_pbc_checkbox, span=2)
        geometry_section.addWidget(geometry_grid)

        self.safety_checkbox = CheckBox(self.tr("Show safety limits"), self.setting_widget)
        self.max_outputs_frame = self._single_integer_frame(1, 100_000, 200)
        self.atom_budget_frame = self._single_integer_frame(1, 100_000_000, 200_000)
        self.safety_section = InspectorSection(self.tr("Safety limits"), self.setting_widget)
        safety_grid = ResponsiveFormGrid(self.safety_section, two_column_threshold=520)
        safety_grid.add_field(
            CompactField(
                self.tr("Maximum outputs/input"),
                self.max_outputs_frame,
                self.safety_section,
                inline=True,
                input_max_width=150,
            ),
            span=2,
        )
        safety_grid.add_field(
            CompactField(
                self.tr("Generated atom budget/input"),
                self.atom_budget_frame,
                self.safety_section,
                inline=True,
                input_max_width=150,
            ),
            span=2,
        )
        self.safety_section.addWidget(safety_grid)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        preview_section = InspectorSection(self.tr("Exact size preview"), self.setting_widget)
        preview_section.addWidget(self.preview_label)

        self.settingLayout.addWidget(plane_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(geometry_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.safety_checkbox, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.safety_section, 3, 0, 1, 3)
        self.settingLayout.addWidget(preview_section, 4, 0, 1, 3)

        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        self.plane_table.planesChanged.connect(self._planes_changed)
        self.normal_pbc_checkbox.stateChanged.connect(self._parameters_changed)
        self.safety_checkbox.stateChanged.connect(self._parameters_changed)
        for frame in (self.layer_frame, self.vacuum_frame, self.max_outputs_frame, self.atom_budget_frame):
            for control in frame.object_list:
                control.valueChanged.connect(self._parameters_changed)
        self._parameters_changed()

    def _range_frame(self, kind, minimum, maximum, values, *, unit=""):
        frame = SpinBoxUnitInputFrame(self.setting_widget)
        frame.set_input([self.tr("to"), self.tr("step"), unit], 3, kind)
        frame.setRange(minimum, maximum)
        if kind == "float":
            frame.setDecimals(3)
            frame.setSingleStep(0.5)
        frame.set_input_value(list(values))
        return frame

    def _single_integer_frame(self, minimum, maximum, value):
        frame = SpinBoxUnitInputFrame(self.setting_widget)
        frame.set_input("", 1, "int")
        frame.setRange(minimum, maximum)
        frame.set_input_value([value])
        return frame

    @staticmethod
    def _first_structure(dataset):
        if dataset is None:
            return None
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return dataset
        try:
            return next(iter(dataset))
        except (StopIteration, TypeError):
            return None

    @staticmethod
    def _dataset_count(dataset) -> int:
        if dataset is None:
            return 0
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return 1
        try:
            return len(dataset)
        except TypeError:
            return 0

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        if self._preview_input_count is None:
            self._preview_input_count = self._dataset_count(dataset) or None
        self._refresh_preview()

    def set_preview_structure(self, structure) -> None:
        self._input_structure = structure
        self._refresh_preview()

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def _preset_changed(self, _index=None) -> None:
        if self._updating_planes:
            return
        preset = combo_value(self.preset_combo)
        if preset in self._PRESETS:
            self._updating_planes = True
            self.plane_table.set_planes(self._PRESETS[preset])
            self._updating_planes = False
        self._parameters_changed()

    def _planes_changed(self) -> None:
        if not self._updating_planes:
            self._updating_planes = True
            set_combo_value(self.preset_combo, "custom")
            self._updating_planes = False
        self._parameters_changed()

    def _parameters_changed(self, *_args) -> None:
        self.safety_section.setVisible(self.safety_checkbox.isChecked())
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if not hasattr(self, "preview_label"):
            return
        self.refresh_compact_presentation()
        if self._input_structure is None:
            try:
                planes = len(RandomSlabOperation.canonical_hkl_list(self.get_params().hkl_list))
            except ValueError as exc:
                self.preview_label.setText("⚠ " + translate_runtime_message(exc))
                return
            self.preview_label.setText(
                self.tr(
                    "Selected planes: {planes}. Load an upstream bulk structure for exact output and atom counts."
                ).format(planes=planes)
            )
            return
        try:
            plan = self.create_operation().plan(self._input_structure, self.get_params())
        except ValueError as exc:
            self.preview_label.setText(
                "⚠ " + self.tr("Preview unavailable: {error}").format(error=translate_runtime_message(exc))
            )
            return
        atom_range = (
            str(plan.min_atoms_per_output)
            if plan.min_atoms_per_output == plan.max_atoms_per_output
            else f"{plan.min_atoms_per_output}–{plan.max_atoms_per_output}"
        )
        self.preview_label.setText(
            self.tr(
                "First input: planes {planes} × repeat values {repeats} × vacuum values {vacuums} "
                "= {outputs} outputs; {atoms} atoms/output, {total} generated atoms."
            ).format(
                planes=len(plan.hkl_list),
                repeats=len(plan.repeats),
                vacuums=len(plan.vacuums),
                outputs=plan.outputs,
                atoms=atom_range,
                total=plan.generated_atoms,
            )
        )

    def create_operation(self):
        return RandomSlabOperation()

    def get_params(self) -> RandomSlabParams:
        return RandomSlabParams(
            hkl_list=self.plane_table.planes(),
            layer_range=tuple(int(value) for value in self.layer_frame.get_input_value()),
            vacuum_range=tuple(float(value) for value in self.vacuum_frame.get_input_value()),
            normal_pbc=self.normal_pbc_checkbox.isChecked(),
            max_outputs=int(self.max_outputs_frame.get_input_value()[0]),
            max_generated_atoms=int(self.atom_budget_frame.get_input_value()[0]),
        )

    def set_params(self, params: RandomSlabParams) -> None:
        try:
            planes = RandomSlabOperation.canonical_hkl_list(params.hkl_list)
        except ValueError:
            planes = tuple(tuple(int(value) for value in row) for row in params.hkl_list)
        self._updating_planes = True
        self.plane_table.set_planes(planes)
        preset = next((key for key, value in self._PRESETS.items() if value == planes), "custom")
        set_combo_value(self.preset_combo, preset)
        self._updating_planes = False
        self.layer_frame.set_input_value([int(value) for value in params.layer_range])
        self.vacuum_frame.set_input_value([float(value) for value in params.vacuum_range])
        self.normal_pbc_checkbox.setChecked(bool(params.normal_pbc))
        self.max_outputs_frame.set_input_value([int(params.max_outputs)])
        self.atom_budget_frame.set_input_value([int(params.max_generated_atoms)])
        self._parameters_changed()

    def get_summary_text(self) -> str:
        try:
            params = self.get_params()
            planes = len(RandomSlabOperation.canonical_hkl_list(params.hkl_list))
            if self._input_structure is not None:
                plan = self.create_operation().plan(self._input_structure, params)
                return self.tr("Planes {planes} · {outputs}/input · {atoms} generated atoms").format(
                    planes=planes, outputs=plan.outputs, atoms=plan.generated_atoms
                )
            return self.tr("Planes {planes} · exact scan").format(planes=planes)
        except ValueError:
            return self.tr("Check slab parameters")

    def get_guidance_text(self) -> str:
        inputs = self._preview_input_count
        if self._input_structure is None:
            return self.tr(
                "Load a three-periodic bulk structure. Miller indices are interpreted in its reciprocal basis."
            )
        try:
            plan = self.create_operation().plan(self._input_structure, self.get_params())
        except ValueError as exc:
            return translate_runtime_message(exc)
        if inputs is None or inputs <= 0:
            return self.tr("Exact outputs/input: {outputs}; first input generates {atoms} atoms.").format(
                outputs=plan.outputs, atoms=plan.generated_atoms
            )
        return self.tr(
            "Inputs {inputs} × {per_input} outputs/input = {total} outputs; first input generates {atoms} atoms."
        ).format(inputs=inputs, per_input=plan.outputs, total=inputs * plan.outputs, atoms=plan.generated_atoms)

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    @staticmethod
    def _legacy_planes(data) -> tuple[tuple[int, int, int], ...]:
        ranges = []
        for key, default in (("h_range", (0, 1, 1)), ("k_range", (0, 1, 1)), ("l_range", (1, 3, 1))):
            values = tuple(data.get(key, default))
            if len(values) != 3:
                raise ValueError(f"{key} must contain start, stop, and step")
            start, stop, step = (int(value) for value in values)
            if start > stop:
                start, stop = stop, start
            ranges.append(
                RandomSlabOperation._inclusive_integer_range(
                    (start, stop, step), code="surface-slab-legacy-range", label=key, minimum=-20
                )
            )
        planes = [hkl for hkl in itertools.product(*ranges) if hkl != (0, 0, 0)]
        return RandomSlabOperation.canonical_hkl_list(planes)

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw = dict(data_dict.get("params") or {})
        source = raw if raw else data_dict
        legacy = "hkl_list" not in source and any(key in source for key in ("h_range", "k_range", "l_range"))
        defaults = RandomSlabParams()
        planes = self._legacy_planes(source) if legacy else source.get("hkl_list", defaults.hkl_list)
        params = RandomSlabParams(
            hkl_list=planes,
            layer_range=tuple(source.get("layer_range", defaults.layer_range)),
            vacuum_range=tuple(source.get("vacuum_range", defaults.vacuum_range)),
            normal_pbc=bool(source.get("normal_pbc", defaults.normal_pbc)),
            max_outputs=source.get("max_outputs", defaults.max_outputs),
            max_generated_atoms=source.get("max_generated_atoms", defaults.max_generated_atoms),
        )
        self.set_params(params)
        if legacy:
            message = self.tr(
                "This saved card used h/k/l ranges. They were converted to the visible plane list, reduced, "
                "and deduplicated; review the list before running."
            )
            self.legacy_notice.setText("⚠ " + message)
            self.legacy_notice.show()
            MessageManager.send_warning_message(message)
