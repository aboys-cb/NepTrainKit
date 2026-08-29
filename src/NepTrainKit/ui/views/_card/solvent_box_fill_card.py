"""Card for filling periodic cells with solvent."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPlainTextEdit
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.solvation import DEFAULT_WATER_XYZ, SolventBoxFillOperation, SolventBoxFillParams
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import CompactField, InspectorSection, MakeDataCard, ResponsiveFormGrid, SpinBoxUnitInputFrame


@CardManager.register_card
class SolventBoxFillCard(MakeDataCard):
    """Fill an existing periodic cell with solvent molecules."""

    group = "Organic"
    card_name = "Solvent Box Fill"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [{"name": "Chen Zherui", "role": "author", "email": "chenzherui0124@foxmail.com"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._input_count = 0
        self.setTitle(self.tr("Periodic Solvent Box"))
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
        self.setObjectName("solvent_box_fill_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.input_note = CaptionLabel(
            self.tr("Requires a finite cell and at least one periodic axis. Keeps the host and fills the whole cell."),
            self.setting_widget,
        )
        self.input_note.setWordWrap(True)
        self.input_note.setStyleSheet("color:#8a95a0;")

        self.structures_frame, self.structures_field = self._spin_field(
            self.tr("Independent boxes per input"), "", 1, "int", [1], 1, 100000,
            inline=True,
        )
        self.structures_label = self.structures_field.caption
        self.count_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.count_mode_combo, [
            ("fixed", "Fixed molecule count"),
            ("density", "Nominal density from full cell"),
        ])
        set_combo_value(self.count_mode_combo, "density")
        self.count_mode_field = CompactField(
            self.tr("Target amount"), self.count_mode_combo, self.setting_widget,
            self.tr("Density mode uses the complete cell volume without subtracting the host."),
        )
        self.count_mode_label = self.count_mode_field.caption
        self.count_frame, self.count_field = self._spin_field(
            self.tr("Target solvent molecules"), "", 1, "int", [100], 1, 1000000,
            inline=True,
        )
        self.count_label = self.count_field.caption
        self.density_frame, self.density_field = self._spin_field(
            self.tr("Nominal pure-solvent density"), "g/cm³", 1, "float", [1.0], 0.0001, 1000.0,
            decimals=4, inline=True,
        )
        self.density_label = self.density_field.caption
        self.fill_packing_frame, self.fill_packing_field = self._spin_field(
            self.tr("Full-cell count factor"), "×", 1, "float", [1.0], 0.0001, 1.0,
            self.tr("Multiply the full-cell density estimate by a value in (0, 1]."), decimals=4, inline=True,
        )
        self.fill_packing_label = self.fill_packing_field.caption
        self.strict_checkbox = CheckBox(self.tr("Require the full requested count"), self.setting_widget)
        self.strict_checkbox.setChecked(True)
        self.strict_checkbox.setToolTip(self.tr("Fail instead of returning a partially filled box."))
        self.seed_checkbox = CheckBox(self.tr("Use reproducible seed"), self.setting_widget)
        self.seed_frame, self.seed_field = self._spin_field(
            self.tr("Random seed"), "", 1, "int", [0], 0, 2**31 - 1, inline=True,
        )
        output_section = InspectorSection(self.tr("Output and amount"), self.setting_widget)
        output_grid = ResponsiveFormGrid(output_section)
        for field in (self.structures_field, self.count_mode_field, self.count_field, self.density_field, self.fill_packing_field):
            output_grid.add_field(field, span=2)
        output_section.addWidget(output_grid)
        output_section.addWidget(self.strict_checkbox)
        output_section.addWidget(self.seed_checkbox)
        output_section.addWidget(self.seed_field)

        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.mode_combo, [
            ("loose", "Compact clearance (0.62×)"),
            ("general", "Standard clearance (0.70×)"),
            ("dense", "Conservative clearance (0.78×)"),
        ])
        set_combo_value(self.mode_combo, "general")
        self.mode_field = CompactField(
            self.tr("Placement clearance"), self.mode_combo, self.setting_widget,
            self.tr("Sets the default element-radius pair cutoff."),
        )
        self.mode_label = self.mode_field.caption
        placement_section = InspectorSection(
            self.tr("Placement"), self.setting_widget,
            self.tr("Molecule centers and orientations are sampled randomly throughout the cell."),
        )
        placement_grid = ResponsiveFormGrid(placement_section)
        placement_grid.add_field(self.mode_field, span=2)
        placement_section.addWidget(placement_grid)

        self.advanced_checkbox = CheckBox(self.tr("Show manual collision and solvent details"), self.setting_widget)
        self.min_distance_frame, self.min_distance_field = self._spin_field(
            self.tr("Uniform minimum distance"), "Å", 1, "float", [0.0], 0.0, 100.0,
            self.tr("0 uses element radii; a positive value overrides every pair cutoff."), decimals=3, inline=True,
        )
        self.min_distance_label = self.min_distance_field.caption
        self.collision_frame, self.collision_field = self._spin_field(
            self.tr("Manual element-radius scale"), "×", 1, "float", [0.0], 0.0, 5.0,
            self.tr("0 uses the selected placement clearance."), decimals=3, inline=True,
        )
        self.collision_label = self.collision_field.caption
        self.attempts_frame, self.attempts_field = self._spin_field(
            self.tr("Attempts per requested molecule"), "", 1, "int", [500], 1, 100000,
            self.tr("The total attempt budget is this value times the target count."), inline=True,
        )
        self.attempts_label = self.attempts_field.caption
        self.collision_section = InspectorSection(self.tr("Collision checks"), self.setting_widget)
        collision_grid = ResponsiveFormGrid(self.collision_section)
        for field in (self.min_distance_field, self.collision_field, self.attempts_field):
            collision_grid.add_field(field, span=2)
        self.collision_section.addWidget(collision_grid)

        self.edit_solvent_checkbox = CheckBox(self.tr("Edit solvent molecule (default: water)"), self.setting_widget)
        self.solvent_edit = QPlainTextEdit(self.setting_widget)
        self.solvent_edit.setPlainText(DEFAULT_WATER_XYZ)
        self.solvent_edit.setFixedHeight(92)
        self.solvent_field = CompactField(
            self.tr("Solvent XYZ"), self.solvent_edit, self.setting_widget,
            self.tr("One molecule in XYZ or extxyz text."),
        )
        self.solvent_label = self.solvent_field.caption
        self.flex_checkbox = CheckBox(self.tr("Pre-sample flexible solvent conformers"), self.setting_widget)
        self.flex_pool_frame, self.flex_pool_field = self._spin_field(
            self.tr("Flexible conformer pool"), "", 1, "int", [32], 1, 10000, inline=True,
        )
        self.flex_torsion_min_frame, self.flex_torsion_min_field = self._spin_field(
            self.tr("Torsion increment minimum"), "°", 1, "float", [-180.0], -360.0, 360.0, decimals=3, inline=True,
        )
        self.flex_torsion_max_frame, self.flex_torsion_max_field = self._spin_field(
            self.tr("Torsion increment maximum"), "°", 1, "float", [180.0], -360.0, 360.0, decimals=3, inline=True,
        )
        self.flex_max_torsions_frame, self.flex_max_torsions_field = self._spin_field(
            self.tr("Maximum torsions per conformer"), "", 1, "int", [5], 0, 10000, inline=True,
        )
        self.flex_sigma_frame, self.flex_sigma_field = self._spin_field(
            self.tr("Coordinate noise sigma"), "Å", 1, "float", [0.03], 0.0, 10000.0, decimals=4, inline=True,
        )
        self.solvent_section = InspectorSection(self.tr("Solvent template"), self.setting_widget)
        self.solvent_section.addWidget(self.edit_solvent_checkbox)
        self.solvent_section.addWidget(self.solvent_field)
        self.solvent_section.addWidget(self.flex_checkbox)
        solvent_grid = ResponsiveFormGrid(self.solvent_section)
        for field in (self.flex_pool_field, self.flex_torsion_min_field, self.flex_torsion_max_field, self.flex_max_torsions_field, self.flex_sigma_field):
            solvent_grid.add_field(field, span=2)
        self.solvent_section.addWidget(solvent_grid)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_label.setObjectName("solventBoxFillPreview")
        preview_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        preview_section.addWidget(self.preview_label)

        for row, widget in enumerate((self.input_note, output_section, placement_section, self.advanced_checkbox, self.collision_section, self.solvent_section, preview_section)):
            self.settingLayout.addWidget(widget, row, 0, 1, 3)

        self.density_controls = (self.density_field, self.fill_packing_field)
        self.fixed_count_controls = (self.count_field,)
        self.flex_controls = (
            self.flex_pool_field, self.flex_torsion_min_field, self.flex_torsion_max_field,
            self.flex_max_torsions_field, self.flex_sigma_field,
        )

        self.edit_solvent_checkbox.stateChanged.connect(self._update_solvent_visibility)
        self.count_mode_combo.currentIndexChanged.connect(self._update_count_visibility)
        self.advanced_checkbox.stateChanged.connect(self._update_advanced_visibility)
        self.flex_checkbox.stateChanged.connect(self._update_flex_visibility)
        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self.mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self.solvent_edit.textChanged.connect(self._refresh_preview)
        self.strict_checkbox.stateChanged.connect(self._refresh_preview)
        for frame in (
            self.structures_frame, self.count_frame, self.density_frame, self.fill_packing_frame,
            self.min_distance_frame, self.collision_frame, self.attempts_frame, self.flex_pool_frame,
            self.flex_torsion_min_frame, self.flex_torsion_max_frame, self.flex_max_torsions_frame,
            self.flex_sigma_frame, self.seed_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._on_value_changed)

        self._update_solvent_visibility()
        self._update_count_visibility()
        self._update_advanced_visibility()
        self._update_collision_linkage()
        self._on_seed_changed()
        self._refresh_preview()

    def _on_value_changed(self, *_args):
        self._update_collision_linkage()
        self._refresh_preview()

    def _update_solvent_visibility(self, *_args):
        visible = self.edit_solvent_checkbox.isChecked()
        self.solvent_field.setVisible(visible)
        self.solvent_edit.setVisible(visible)
        self._update_tab_order()

    def _update_count_visibility(self, *_args):
        density_mode = combo_value(self.count_mode_combo) == "density"
        for field in self.fixed_count_controls:
            field.setVisible(not density_mode)
            field.input_widget.setVisible(not density_mode)
        for field in self.density_controls:
            field.setVisible(density_mode)
            field.input_widget.setVisible(density_mode)
        self._update_tab_order()
        self._refresh_preview()

    def _update_advanced_visibility(self, *_args):
        visible = self.advanced_checkbox.isChecked()
        self.collision_section.setVisible(visible)
        self.solvent_section.setVisible(visible)
        for field in (self.min_distance_field, self.collision_field, self.attempts_field):
            field.input_widget.setVisible(visible)
        self._update_flex_visibility()
        self._update_tab_order()

    def _update_flex_visibility(self, *_args):
        visible = self.advanced_checkbox.isChecked() and self.flex_checkbox.isChecked()
        for field in self.flex_controls:
            field.setVisible(visible)
            field.input_widget.setVisible(visible)
        self._update_tab_order()
        self._refresh_preview()

    def _on_seed_changed(self, *_args):
        self.seed_field.setVisible(self.seed_checkbox.isChecked())
        self._update_tab_order()

    def _update_collision_linkage(self):
        uniform = float(self.min_distance_frame.get_input_value()[0]) > 0.0
        manual = float(self.collision_frame.get_input_value()[0]) > 0.0
        self.collision_field.setEnabled(not uniform)
        self.mode_field.setEnabled(not uniform and not manual)
        if uniform:
            helper = self.tr("Ignored while a uniform minimum distance is active.")
            self.mode_field.set_helper_text(helper)
            self.collision_field.set_helper_text(helper)
        elif manual:
            self.mode_field.set_helper_text(self.tr("Ignored while a manual element-radius scale is active."))
            self.collision_field.set_helper_text(self.tr("This value overrides the selected placement clearance."))
        else:
            self.mode_field.set_helper_text(self.tr("Sets the default element-radius pair cutoff."))
            self.collision_field.set_helper_text(self.tr("0 uses the selected placement clearance."))

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
        self._refresh_preview()

    def _refresh_preview(self, *_args):
        if not hasattr(self, "preview_label"):
            return
        outputs = int(self.structures_frame.get_input_value()[0])
        if self._input_structure is None:
            if combo_value(self.count_mode_combo) == "fixed":
                target = self.tr("target {count} molecule(s)").format(count=int(self.count_frame.get_input_value()[0]))
            else:
                target = self.tr("target count resolves after a cell is loaded")
            self.preview_label.setText(self.tr("No input loaded · {outputs} box(es) per input · {target}").format(outputs=outputs, target=target))
            return
        try:
            summary = self.create_operation().capacity_summary(self._input_structure, self.get_params())
        except (TypeError, ValueError, IndexError) as exc:
            self.preview_label.setText("⚠ " + self.tr("Preview unavailable: {error}").format(error=translate_runtime_message(exc)))
            return

        if summary["min_distance"] > 0.0:
            collision = self.tr("uniform minimum {distance} Å").format(distance=f"{summary['min_distance']:.3g}")
        else:
            collision = self.tr("element-radius scale {scale}").format(scale=f"{summary['collision_scale']:.3g}")
        dataset_outputs = summary["structures"] * max(self._input_count, 1)
        first_line = self.tr("First input: {host} host atoms · cell {volume} Å³ / PBC {axes}").format(
            host=summary["host_atoms"], volume=f"{summary['volume']:.5g}", axes=",".join(summary["pbc_axes"]),
        )
        second_line = self.tr("Target {count} {formula} molecule(s) (+{added} atoms) per box · up to {outputs} dataset output(s)").format(
            count=summary["target_count"], formula=summary["solvent_formula"], added=summary["added_atoms"], outputs=dataset_outputs,
        )
        if summary["count_mode"] == "density":
            amount_note = self.tr("Full-cell density estimate: {density} g/cm³ × {factor}; host occupancy is not subtracted.").format(
                density=f"{summary['density']:.4g}", factor=f"{summary['fill_packing']:.4g}",
            )
        else:
            amount_note = self.tr("The fixed target corresponds to {density} g/cm³ if the full cell contained only this solvent.").format(
                density=f"{summary['nominal_density']:.4g}",
            )
        strict_note = (
            self.tr("Generation fails unless every requested molecule fits.")
            if self.strict_checkbox.isChecked()
            else self.tr("A non-empty partial box may be returned if placement is exhausted.")
        )
        self.preview_label.setText("\n".join((first_line, second_line, f"{amount_note} {collision}. {strict_note}")))

    def _update_tab_order(self):
        if not hasattr(self, "advanced_checkbox"):
            return
        widgets = [*self.structures_frame.object_list, self.count_mode_combo]
        if combo_value(self.count_mode_combo) == "density":
            widgets.extend(self.density_frame.object_list)
            widgets.extend(self.fill_packing_frame.object_list)
        else:
            widgets.extend(self.count_frame.object_list)
        widgets.extend((self.strict_checkbox, self.seed_checkbox))
        if self.seed_checkbox.isChecked():
            widgets.extend(self.seed_frame.object_list)
        if self.mode_field.isEnabled():
            widgets.append(self.mode_combo)
        widgets.append(self.advanced_checkbox)
        if self.advanced_checkbox.isChecked():
            widgets.extend(self.min_distance_frame.object_list)
            if self.collision_field.isEnabled():
                widgets.extend(self.collision_frame.object_list)
            widgets.extend(self.attempts_frame.object_list)
            widgets.append(self.edit_solvent_checkbox)
            if self.edit_solvent_checkbox.isChecked():
                widgets.append(self.solvent_edit)
            widgets.append(self.flex_checkbox)
            if self.flex_checkbox.isChecked():
                for frame in (self.flex_pool_frame, self.flex_torsion_min_frame, self.flex_torsion_max_frame, self.flex_max_torsions_frame, self.flex_sigma_frame):
                    widgets.extend(frame.object_list)
        self.tab_order_widgets = widgets

    def create_operation(self):
        return SolventBoxFillOperation()

    def get_params(self) -> SolventBoxFillParams:
        return SolventBoxFillParams(
            solvent_xyz=self.solvent_edit.toPlainText(), structures=int(self.structures_frame.get_input_value()[0]),
            count_mode=combo_value(self.count_mode_combo), solvent_count=int(self.count_frame.get_input_value()[0]),
            density=float(self.density_frame.get_input_value()[0]), sampling_mode=combo_value(self.mode_combo),
            fill_packing=float(self.fill_packing_frame.get_input_value()[0]),
            min_distance=float(self.min_distance_frame.get_input_value()[0]), collision_scale=float(self.collision_frame.get_input_value()[0]),
            max_attempts_per_solvent=int(self.attempts_frame.get_input_value()[0]), strict_count=self.strict_checkbox.isChecked(),
            flex_solvent=self.flex_checkbox.isChecked(), flex_pool=int(self.flex_pool_frame.get_input_value()[0]),
            flex_torsion_range=(float(self.flex_torsion_min_frame.get_input_value()[0]), float(self.flex_torsion_max_frame.get_input_value()[0])),
            flex_max_torsions=int(self.flex_max_torsions_frame.get_input_value()[0]),
            flex_gaussian_sigma=float(self.flex_sigma_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(), seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: SolventBoxFillParams) -> None:
        self.solvent_edit.setPlainText(params.solvent_xyz)
        self.structures_frame.set_input_value([int(params.structures)])
        set_combo_value(self.count_mode_combo, params.count_mode)
        self.count_frame.set_input_value([int(params.solvent_count)])
        self.density_frame.set_input_value([float(params.density)])
        legacy_mode = str(params.sampling_mode).strip().lower()
        set_combo_value(self.mode_combo, "general" if legacy_mode in {"auto", "water"} else legacy_mode)
        self.fill_packing_frame.set_input_value([float(params.fill_packing)])
        self.min_distance_frame.set_input_value([float(params.min_distance)])
        self.collision_frame.set_input_value([float(params.collision_scale)])
        self.attempts_frame.set_input_value([int(params.max_attempts_per_solvent)])
        self.strict_checkbox.setChecked(bool(params.strict_count))
        self.flex_checkbox.setChecked(bool(params.flex_solvent))
        self.flex_pool_frame.set_input_value([int(params.flex_pool)])
        self.flex_torsion_min_frame.set_input_value([float(params.flex_torsion_range[0])])
        self.flex_torsion_max_frame.set_input_value([float(params.flex_torsion_range[1])])
        self.flex_max_torsions_frame.set_input_value([int(params.flex_max_torsions)])
        self.flex_sigma_frame.set_input_value([float(params.flex_gaussian_sigma)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._update_count_visibility()
        self._update_advanced_visibility()
        self._update_collision_linkage()
        self._on_seed_changed()
        self._refresh_preview()

    def get_summary_text(self):
        params = self.get_params()
        amount = (
            self.tr("{count} molecule(s)").format(count=params.solvent_count)
            if params.count_mode == "fixed"
            else self.tr("density {density} g/cm³").format(density=f"{params.density:g}")
        )
        return self.tr("{outputs} box(es) · {amount} · {clearance}").format(
            outputs=params.structures, amount=amount, clearance=self.mode_combo.currentText(),
        )

    def get_guidance_text(self):
        return self.tr("Check the resolved count, full-cell density assumption, collision rule, and strict-count behavior before generating.")

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data):
        super().from_dict(data)
        raw = data.get("params")
        self.set_params(SolventBoxFillParams(**raw) if raw else SolventBoxFillParams())
