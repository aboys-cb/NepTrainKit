"""Card for local solvent-shell generation."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPlainTextEdit
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.solvation import DEFAULT_WATER_XYZ, LocalSolvationOperation, LocalSolvationParams, has_valid_cell
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import CompactField, InspectorSection, MakeDataCard, ResponsiveFormGrid, SpinBoxUnitInputFrame


@CardManager.register_card
class LocalSolvationCard(MakeDataCard):
    """Generate local solvent shells around selected atoms."""

    group = "Organic"
    card_name = "Local Solvation"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [{"name": "Chen Zherui", "role": "author", "email": "chenzherui0124@foxmail.com"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._input_count = 0
        self.setTitle(self.tr("Solvent Shell"))
        self._init_ui()

    def _spin_field(self, label, units, count, kinds, values, minimum, maximum, helper="", *, decimals=None, inline=False):
        frame = SpinBoxUnitInputFrame(self)
        frame.set_input(units, count, kinds)
        frame.setRange(minimum, maximum)
        if decimals is not None:
            frame.setDecimals(decimals)
        frame.set_input_value(values)
        frame.setAccessibleName(label)
        field = CompactField(label, frame, self.setting_widget, helper, inline=inline, input_max_width=144 if inline else None)
        return frame, field

    def _init_ui(self):
        self.setObjectName("local_solvation_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.structures_frame, self.structures_field = self._spin_field(
            self.tr("Independent outputs per input"), "", 1, "int", [1], 1, 100000,
            self.tr("Each output uses an independent placement attempt."), inline=True,
        )
        self.count_frame, self.count_field = self._spin_field(
            self.tr("Total solvent molecules per output"), "", 1, "int", [6], 1, 100000,
            self.tr("Shared across all selected centers, not repeated per center."), inline=True,
        )
        self.strict_checkbox = CheckBox(self.tr("Require the full requested count"), self.setting_widget)
        self.strict_checkbox.setChecked(True)
        self.strict_checkbox.setToolTip(self.tr("Fail instead of returning a partially filled output."))
        self.seed_checkbox = CheckBox(self.tr("Use reproducible seed"), self.setting_widget)
        self.seed_frame, self.seed_field = self._spin_field(
            self.tr("Random seed"), "", 1, "int", [0], 0, 2**31 - 1, inline=True
        )
        output_section = InspectorSection(self.tr("Output"), self.setting_widget)
        output_grid = ResponsiveFormGrid(output_section)
        output_grid.add_field(self.structures_field, span=2)
        output_grid.add_field(self.count_field, span=2)
        output_section.addWidget(output_grid)
        output_section.addWidget(self.strict_checkbox)
        output_section.addWidget(self.seed_checkbox)
        output_section.addWidget(self.seed_field)

        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.mode_combo, [
            ("auto", "Auto-select placement"),
            ("general", "Random molecular orientation"),
            ("water", "Water dipole orientation"),
            ("ion-water", "Supported ion hydration"),
            ("loose", "Loose random placement"),
            ("dense", "Dense water placement"),
        ])
        self.mode_field = CompactField(
            self.tr("Placement method"), self.mode_combo, self.setting_widget,
            self.tr("Auto uses ion hydration only for water around supported ion centers."),
        )
        self.mode_label = self.mode_field.caption

        self.center_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.center_mode_combo, [
            ("all", "All host atoms"),
            ("elements", "By element"),
            ("indices", "By 1-based atom index"),
            ("z_range", "By Cartesian z range"),
        ])
        self.center_mode_field = CompactField(self.tr("Solvation centers"), self.center_mode_combo, self.setting_widget)
        self.center_mode_label = self.center_mode_field.caption

        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setPlaceholderText(self.tr("Ca, Na, O"))
        self.elements_field = CompactField(self.tr("Center elements"), self.elements_edit, self.setting_widget, self.tr("Comma-separated element symbols."))
        self.elements_label = self.elements_field.caption
        self.indices_edit = LineEdit(self.setting_widget)
        self.indices_edit.setPlaceholderText(self.tr("1,3,5-8"))
        self.indices_field = CompactField(self.tr("Center indices"), self.indices_edit, self.setting_widget, self.tr("Use 1-based indices and ranges."))
        self.indices_label = self.indices_field.caption
        self.z_frame, self.z_field = self._spin_field(
            self.tr("Cartesian z range"), ["Å", "Å"], 2, ["float", "float"], [0.0, 0.0], -100000.0, 100000.0,
            self.tr("Select centers by their Cartesian z coordinate."), decimals=3,
        )
        self.z_label = self.z_field.caption
        self.shell_frame, self.shell_field = self._spin_field(
            self.tr("Fallback center-to-COM shell"), ["Å", "Å"], 2, ["float", "float"], [2.2, 4.5], 0.0, 1000.0,
            self.tr("Used for ordinary placement and after a supported first shell is full."), decimals=3,
        )
        self.shell_label = self.shell_field.caption
        self.placement_note = CaptionLabel("", self.setting_widget)
        self.placement_note.setWordWrap(True)
        self.placement_note.setStyleSheet("color:#8a95a0;")
        placement_section = InspectorSection(self.tr("Centers and placement"), self.setting_widget)
        placement_grid = ResponsiveFormGrid(placement_section)
        for field in (self.mode_field, self.center_mode_field, self.elements_field, self.indices_field, self.z_field, self.shell_field):
            placement_grid.add_field(field, span=2)
        placement_section.addWidget(placement_grid)
        placement_section.addWidget(self.placement_note)

        self.advanced_checkbox = CheckBox(self.tr("Show collision and solvent details"), self.setting_widget)
        self.min_distance_frame, self.min_distance_field = self._spin_field(
            self.tr("Uniform minimum distance"), "Å", 1, "float", [0.0], 0.0, 100.0,
            self.tr("0 uses element radii; a positive value overrides every pair cutoff."), decimals=3, inline=True,
        )
        self.min_distance_label = self.min_distance_field.caption
        self.collision_frame, self.collision_field = self._spin_field(
            self.tr("Element-radius collision scale"), "×", 1, "float", [0.0], 0.0, 5.0,
            self.tr("0 uses the placement-method default."), decimals=3, inline=True,
        )
        self.collision_label = self.collision_field.caption
        self.attempts_frame, self.attempts_field = self._spin_field(
            self.tr("Placement attempts per output"), "", 1, "int", [3000], 1, 10000000,
            self.tr("All requested solvent molecules share this attempt budget."), inline=True,
        )
        self.attempts_label = self.attempts_field.caption
        self.collision_section = InspectorSection(self.tr("Collision checks"), self.setting_widget)
        collision_grid = ResponsiveFormGrid(self.collision_section)
        for field in (self.min_distance_field, self.collision_field, self.attempts_field):
            collision_grid.add_field(field, span=2)
        self.collision_section.addWidget(collision_grid)

        self.auto_box_checkbox = CheckBox(self.tr("Size the nonperiodic box from the output"), self.setting_widget)
        self.box_size_frame, self.box_size_field = self._spin_field(
            self.tr("Fixed display box"), "Å", 1, "float", [100.0], 0.001, 100000.0,
            self.tr("Used only when the input has no valid cell."), decimals=3, inline=True,
        )
        self.box_size_label = self.box_size_field.caption
        self.box_frame, self.box_field = self._spin_field(
            self.tr("Auto-box padding and minimum edge"), ["Å", "Å"], 2, ["float", "float"], [8.0, 0.0], 0.0, 100000.0, decimals=3,
        )
        self.box_label = self.box_field.caption
        self.box_section = InspectorSection(self.tr("Nonperiodic display box"), self.setting_widget)
        self.box_section.addWidget(self.auto_box_checkbox)
        box_grid = ResponsiveFormGrid(self.box_section)
        box_grid.add_field(self.box_size_field, span=2)
        box_grid.add_field(self.box_field, span=2)
        self.box_section.addWidget(box_grid)

        self.edit_solvent_checkbox = CheckBox(self.tr("Edit solvent molecule (default: water)"), self.setting_widget)
        self.solvent_edit = QPlainTextEdit(self.setting_widget)
        self.solvent_edit.setPlainText(DEFAULT_WATER_XYZ)
        self.solvent_edit.setFixedHeight(92)
        self.solvent_field = CompactField(self.tr("Solvent XYZ"), self.solvent_edit, self.setting_widget, self.tr("One molecule in XYZ or extxyz text."))
        self.solvent_label = self.solvent_field.caption
        self.flex_checkbox = CheckBox(self.tr("Pre-sample flexible solvent conformers"), self.setting_widget)
        self.flex_pool_frame, self.flex_pool_field = self._spin_field(self.tr("Flexible conformer pool"), "", 1, "int", [32], 1, 10000, inline=True)
        self.flex_pool_label = self.flex_pool_field.caption
        self.flex_torsion_frame, self.flex_torsion_field = self._spin_field(
            self.tr("Flexible torsion range"), ["°", "°"], 2, ["float", "float"], [-180.0, 180.0], -360.0, 360.0, decimals=3,
        )
        self.flex_torsion_label = self.flex_torsion_field.caption
        self.flex_max_frame, self.flex_max_field = self._spin_field(
            self.tr("Torsions per conformer / coordinate noise"), ["", "Å"], 2, ["int", "float"], [5, 0.03], 0.0, 10000.0, decimals=4,
        )
        self.flex_max_label = self.flex_max_field.caption
        self.solvent_section = InspectorSection(self.tr("Solvent template"), self.setting_widget)
        self.solvent_section.addWidget(self.edit_solvent_checkbox)
        self.solvent_section.addWidget(self.solvent_field)
        self.solvent_section.addWidget(self.flex_checkbox)
        solvent_grid = ResponsiveFormGrid(self.solvent_section)
        for field in (self.flex_pool_field, self.flex_torsion_field, self.flex_max_field):
            solvent_grid.add_field(field, span=2)
        self.solvent_section.addWidget(solvent_grid)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_label.setObjectName("localSolvationPreview")
        preview_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        preview_section.addWidget(self.preview_label)

        for row, widget in enumerate((output_section, placement_section, self.advanced_checkbox, self.collision_section, self.box_section, self.solvent_section, preview_section)):
            self.settingLayout.addWidget(widget, row, 0, 1, 3)

        self.center_element_controls = (self.elements_field,)
        self.center_index_controls = (self.indices_field,)
        self.center_z_controls = (self.z_field,)
        self.flex_controls = (self.flex_pool_field, self.flex_torsion_field, self.flex_max_field)

        self.edit_solvent_checkbox.stateChanged.connect(self._update_solvent_visibility)
        self.advanced_checkbox.stateChanged.connect(self._update_advanced_visibility)
        self.center_mode_combo.currentIndexChanged.connect(self._update_center_visibility)
        self.mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self.auto_box_checkbox.stateChanged.connect(self._update_box_visibility)
        self.flex_checkbox.stateChanged.connect(self._update_flex_visibility)
        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self.solvent_edit.textChanged.connect(self._refresh_preview)
        self.elements_edit.textChanged.connect(self._refresh_preview)
        self.indices_edit.textChanged.connect(self._refresh_preview)
        self.strict_checkbox.stateChanged.connect(self._refresh_preview)
        for frame in (self.structures_frame, self.count_frame, self.z_frame, self.shell_frame, self.min_distance_frame, self.collision_frame, self.attempts_frame, self.box_size_frame, self.box_frame, self.flex_pool_frame, self.flex_torsion_frame, self.flex_max_frame, self.seed_frame):
            for control in frame.object_list:
                control.valueChanged.connect(self._on_value_changed)

        self._update_solvent_visibility()
        self._update_center_visibility()
        self._update_advanced_visibility()
        self._on_seed_changed()
        self._refresh_preview()

    def _on_value_changed(self, *_args):
        self._update_collision_linkage()
        self._refresh_preview()

    def _update_solvent_visibility(self, *_args):
        self.solvent_field.setVisible(self.edit_solvent_checkbox.isChecked())
        self._update_tab_order()

    def _update_center_visibility(self, *_args):
        mode = combo_value(self.center_mode_combo)
        self.elements_field.setVisible(mode == "elements")
        self.indices_field.setVisible(mode == "indices")
        self.z_field.setVisible(mode == "z_range")
        self._update_tab_order()
        self._refresh_preview()

    def _update_advanced_visibility(self, *_args):
        visible = self.advanced_checkbox.isChecked()
        self.collision_section.setVisible(visible)
        self.solvent_section.setVisible(visible)
        self._update_box_visibility()
        self._update_flex_visibility()
        self._update_tab_order()

    def _update_collision_linkage(self):
        uniform = float(self.min_distance_frame.get_input_value()[0]) > 0.0
        self.collision_field.setEnabled(not uniform)
        self.collision_field.set_helper_text(
            self.tr("Ignored while a uniform minimum distance is active.") if uniform else self.tr("0 uses the placement-method default.")
        )

    def _update_box_visibility(self, *_args):
        advanced = self.advanced_checkbox.isChecked()
        input_has_cell = self._input_structure is not None and has_valid_cell(self._input_structure)
        applicable = advanced and not input_has_cell
        self.box_section.setVisible(applicable)
        use_auto = applicable and self.auto_box_checkbox.isChecked()
        self.box_size_field.setVisible(applicable and not use_auto)
        self.box_field.setVisible(use_auto)
        self._update_tab_order()

    def _update_flex_visibility(self, *_args):
        visible = self.advanced_checkbox.isChecked() and self.flex_checkbox.isChecked()
        for field in self.flex_controls:
            field.setVisible(visible)
        self._update_tab_order()
        self._refresh_preview()

    def _on_seed_changed(self, *_args):
        self.seed_field.setVisible(self.seed_checkbox.isChecked())
        self._update_tab_order()

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
    def _dataset_count(dataset):
        if dataset is None:
            return 0
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return 1
        try:
            return len(dataset)
        except (TypeError, AttributeError):
            return 0

    def set_dataset(self, dataset):
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        self._input_count = self._dataset_count(dataset)
        self._update_box_visibility()
        self._refresh_preview()

    def _refresh_preview(self, *_args):
        if not hasattr(self, "preview_label"):
            return
        if self._input_structure is None:
            self.shell_field.setVisible(True)
            self.placement_note.setText(self.tr("Load an upstream structure to resolve the center selection and placement method."))
            self.preview_label.setText(self.tr("No input loaded · {outputs} output(s) per input · {count} total solvent molecule(s) per output").format(
                outputs=int(self.structures_frame.get_input_value()[0]), count=int(self.count_frame.get_input_value()[0])
            ))
            return
        try:
            summary = self.create_operation().placement_summary(self._input_structure, self.get_params())
        except (TypeError, ValueError, IndexError) as exc:
            self.shell_field.setVisible(True)
            self.placement_note.setText("")
            self.preview_label.setText("⚠ " + self.tr("Preview unavailable: {error}").format(error=translate_runtime_message(exc)))
            return

        self.shell_field.setVisible(bool(summary["fallback_needed"]))
        if summary["mode"] == "ion-water":
            ranges = ", ".join(f"{symbol} {bounds[0]:g}–{bounds[1]:g} Å" for symbol, bounds in summary["ion_oxygen_ranges"].items())
            note = self.tr("Supported first shell: {ranges}; total capacity {capacity} water molecule(s).").format(
                ranges=ranges, capacity=summary["first_shell_capacity"]
            )
            if summary["fallback_needed"]:
                note += " " + self.tr("Additional molecules use the fallback COM shell.")
        else:
            note = self.tr("The fallback shell measures center-to-solvent center of mass.")
        self.placement_note.setText(note)
        if summary["uniform_min_distance"] > 0.0:
            collision = self.tr("uniform minimum {distance} Å").format(distance=f"{summary['uniform_min_distance']:.3g}")
        else:
            collision = self.tr("element-radius scale {scale}").format(scale=f"{summary['collision_scale']:.3g}")
        mode_index = self.mode_combo.findData(summary["mode"])
        mode_text = self.mode_combo.itemText(mode_index) if mode_index >= 0 else summary["mode"]
        dataset_maximum = summary["structures"] * max(self._input_count, 1)
        self.preview_label.setText(self.tr(
            "First input: {host} host atoms · {centers} center(s) ({elements}) · {mode} · {count} total molecule(s) shared across centers · up to {outputs} output(s) for this dataset · {collision}"
        ).format(
            host=summary["host_atoms"], centers=summary["center_count"], elements=", ".join(summary["selected_elements"]),
            mode=mode_text, count=summary["solvent_count"], outputs=dataset_maximum, collision=collision,
        ))

    def _update_tab_order(self):
        if not hasattr(self, "advanced_checkbox"):
            return
        widgets = [*self.structures_frame.object_list, *self.count_frame.object_list, self.strict_checkbox, self.seed_checkbox]
        if self.seed_checkbox.isChecked():
            widgets.extend(self.seed_frame.object_list)
        widgets.extend([self.mode_combo, self.center_mode_combo])
        center_mode = combo_value(self.center_mode_combo)
        if center_mode == "elements":
            widgets.append(self.elements_edit)
        elif center_mode == "indices":
            widgets.append(self.indices_edit)
        elif center_mode == "z_range":
            widgets.extend(self.z_frame.object_list)
        if self.shell_field.isVisible():
            widgets.extend(self.shell_frame.object_list)
        widgets.append(self.advanced_checkbox)
        if self.advanced_checkbox.isChecked():
            widgets.extend(self.min_distance_frame.object_list)
            if self.collision_field.isEnabled():
                widgets.extend(self.collision_frame.object_list)
            widgets.extend(self.attempts_frame.object_list)
            if self.box_section.isVisible():
                widgets.append(self.auto_box_checkbox)
                widgets.extend(self.box_frame.object_list if self.auto_box_checkbox.isChecked() else self.box_size_frame.object_list)
            widgets.append(self.edit_solvent_checkbox)
            if self.edit_solvent_checkbox.isChecked():
                widgets.append(self.solvent_edit)
            widgets.append(self.flex_checkbox)
            if self.flex_checkbox.isChecked():
                widgets.extend(self.flex_pool_frame.object_list)
                widgets.extend(self.flex_torsion_frame.object_list)
                widgets.extend(self.flex_max_frame.object_list)
        self.tab_order_widgets = widgets

    def create_operation(self):
        return LocalSolvationOperation()

    def get_params(self):
        flex_max_values = self.flex_max_frame.get_input_value()
        box_values = self.box_frame.get_input_value()
        return LocalSolvationParams(
            solvent_xyz=self.solvent_edit.toPlainText(), structures=int(self.structures_frame.get_input_value()[0]),
            solvent_count=int(self.count_frame.get_input_value()[0]), sampling_mode=combo_value(self.mode_combo),
            center_mode=combo_value(self.center_mode_combo), center_elements=self.elements_edit.text(), center_indices=self.indices_edit.text(),
            z_range=tuple(map(float, self.z_frame.get_input_value())), shell=tuple(map(float, self.shell_frame.get_input_value())),
            min_distance=float(self.min_distance_frame.get_input_value()[0]), collision_scale=float(self.collision_frame.get_input_value()[0]),
            max_attempts=int(self.attempts_frame.get_input_value()[0]), strict_count=self.strict_checkbox.isChecked(),
            auto_box=self.auto_box_checkbox.isChecked(), box_size=float(self.box_size_frame.get_input_value()[0]),
            box_padding=float(box_values[0]), min_box=float(box_values[1]), flex_solvent=self.flex_checkbox.isChecked(),
            flex_pool=int(self.flex_pool_frame.get_input_value()[0]), flex_torsion_range=tuple(map(float, self.flex_torsion_frame.get_input_value())),
            flex_max_torsions=int(flex_max_values[0]), flex_gaussian_sigma=float(flex_max_values[1]),
            use_seed=self.seed_checkbox.isChecked(), seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params):
        self.solvent_edit.setPlainText(params.solvent_xyz)
        self.structures_frame.set_input_value([int(params.structures)])
        self.count_frame.set_input_value([int(params.solvent_count)])
        set_combo_value(self.mode_combo, params.sampling_mode)
        set_combo_value(self.center_mode_combo, params.center_mode)
        self.elements_edit.setText(params.center_elements)
        self.indices_edit.setText(params.center_indices)
        self.z_frame.set_input_value([float(value) for value in params.z_range])
        self.shell_frame.set_input_value([float(value) for value in params.shell])
        self.min_distance_frame.set_input_value([float(params.min_distance)])
        self.collision_frame.set_input_value([float(params.collision_scale)])
        self.attempts_frame.set_input_value([int(params.max_attempts)])
        self.strict_checkbox.setChecked(bool(params.strict_count))
        self.auto_box_checkbox.setChecked(bool(params.auto_box))
        self.box_size_frame.set_input_value([float(params.box_size)])
        self.box_frame.set_input_value([float(params.box_padding), float(params.min_box)])
        self.flex_checkbox.setChecked(bool(params.flex_solvent))
        self.flex_pool_frame.set_input_value([int(params.flex_pool)])
        self.flex_torsion_frame.set_input_value([float(value) for value in params.flex_torsion_range])
        self.flex_max_frame.set_input_value([int(params.flex_max_torsions), float(params.flex_gaussian_sigma)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._update_center_visibility()
        self._update_advanced_visibility()
        self._update_collision_linkage()
        self._on_seed_changed()
        self._refresh_preview()

    def get_summary_text(self):
        params = self.get_params()
        return self.tr("{outputs} output(s) · {count} total molecule(s) · {mode}").format(
            outputs=params.structures, count=params.solvent_count, mode=self.mode_combo.currentText()
        )

    def get_guidance_text(self):
        return self.tr("Check center selection, resolved placement, and the total solvent count before generating.")

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data):
        super().from_dict(data)
        raw = data.get("params")
        self.set_params(LocalSolvationParams(**raw) if raw else LocalSolvationParams())
