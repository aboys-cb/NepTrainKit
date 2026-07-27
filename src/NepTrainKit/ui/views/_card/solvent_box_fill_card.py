"""Card for filling periodic cells with solvent."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPlainTextEdit
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.solvation import DEFAULT_WATER_XYZ, SolventBoxFillOperation, SolventBoxFillParams
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class SolventBoxFillCard(MakeDataCard):
    """Fill an existing periodic cell with solvent molecules."""

    group = "Organic"
    card_name = "Solvent Box Fill"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [
        {"name": "Chen Zherui", "role": "author", "email": "chenzherui0124@foxmail.com"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self.setTitle(self.tr("Periodic Solvent Box"))
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("solvent_box_fill_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.setColumnStretch(1, 1)
        row = 0

        self.edit_solvent_checkbox = CheckBox(
            self.tr("Edit solvent XYZ (default: water)"),
            self.setting_widget,
        )
        self.edit_solvent_checkbox.setChecked(False)
        self.settingLayout.addWidget(
            self.edit_solvent_checkbox,
            row,
            0,
            1,
            3,
        )
        row += 1

        self.solvent_label = BodyLabel(self.tr("Solvent XYZ"), self.setting_widget)
        self.solvent_label.setToolTip(self.tr("Single solvent molecule in XYZ/extxyz text"))
        self.solvent_label.installEventFilter(ToolTipFilter(self.solvent_label, 300, ToolTipPosition.TOP))
        self.solvent_edit = QPlainTextEdit(self.setting_widget)
        self.solvent_edit.setPlainText(DEFAULT_WATER_XYZ)
        self.solvent_edit.setFixedHeight(92)
        self.settingLayout.addWidget(self.solvent_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.solvent_edit, row, 1, 1, 2)
        row += 1

        self.structures_label = BodyLabel(self.tr("Independent outputs per input"), self.setting_widget)
        self.structures_label.setToolTip(self.tr("Independent filled boxes generated per input structure"))
        self.structures_label.installEventFilter(ToolTipFilter(self.structures_label, 300, ToolTipPosition.TOP))
        self.structures_frame = SpinBoxUnitInputFrame(self)
        self.structures_frame.set_input("", 1, "int")
        self.structures_frame.setRange(1, 100000)
        self.structures_frame.set_input_value([1])
        self.settingLayout.addWidget(self.structures_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.structures_frame, row, 1, 1, 2)
        row += 1

        self.count_mode_label = BodyLabel(self.tr("Target amount"), self.setting_widget)
        self.count_mode_label.setToolTip(self.tr("fixed uses solvent count; density derives the count from box volume"))
        self.count_mode_label.installEventFilter(ToolTipFilter(self.count_mode_label, 300, ToolTipPosition.TOP))
        self.count_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.count_mode_combo,
            [
                ("fixed", "Fixed molecule count"),
                ("density", "Nominal solvent density"),
            ],
        )
        self.settingLayout.addWidget(self.count_mode_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.count_mode_combo, row, 1, 1, 2)
        row += 1

        self.count_label = BodyLabel(self.tr("Target solvent molecules"), self.setting_widget)
        self.count_label.setToolTip(self.tr("Number of solvent molecules inserted when count mode is fixed"))
        self.count_label.installEventFilter(ToolTipFilter(self.count_label, 300, ToolTipPosition.TOP))
        self.count_frame = SpinBoxUnitInputFrame(self)
        self.count_frame.set_input("", 1, "int")
        self.count_frame.setRange(1, 1000000)
        self.count_frame.set_input_value([100])
        self.settingLayout.addWidget(self.count_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.count_frame, row, 1, 1, 2)
        row += 1

        self.density_label = BodyLabel(self.tr("Density"), self.setting_widget)
        self.density_label.setToolTip(self.tr("Solvent mass density in g/cm^3 when count mode is density"))
        self.density_label.installEventFilter(ToolTipFilter(self.density_label, 300, ToolTipPosition.TOP))
        self.density_frame = SpinBoxUnitInputFrame(self)
        self.density_frame.set_input("g/cm³", 1, "float")
        self.density_frame.setRange(0.0001, 1000.0)
        self.density_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.density_frame.set_input_value([1.0])
        self.settingLayout.addWidget(self.density_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.density_frame, row, 1, 1, 2)
        row += 1

        self.mode_label = BodyLabel(self.tr("Collision profile"), self.setting_widget)
        self.mode_label.setToolTip(self.tr("Controls the default element-radius collision scale; all box orientations remain random"))
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.mode_combo,
            [
                ("auto", "Auto-detect solvent type"),
                ("general", "General collision profile"),
                ("water", "Water collision profile"),
                ("loose", "Loose collision profile"),
                ("dense", "Dense collision profile"),
            ],
        )
        self.settingLayout.addWidget(self.mode_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, row, 1, 1, 2)
        row += 1

        self.fill_packing_label = BodyLabel(self.tr("Density count multiplier"), self.setting_widget)
        self.fill_packing_label.setToolTip(self.tr("Multiply the nominal pure-solvent molecule count by a value in (0, 1]"))
        self.fill_packing_label.installEventFilter(ToolTipFilter(self.fill_packing_label, 300, ToolTipPosition.TOP))
        self.fill_packing_frame = SpinBoxUnitInputFrame(self)
        self.fill_packing_frame.set_input("x", 1, "float")
        self.fill_packing_frame.setRange(0.0001, 1.0)
        self.fill_packing_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.fill_packing_frame.set_input_value([1.0])
        self.settingLayout.addWidget(self.fill_packing_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.fill_packing_frame, row, 1, 1, 2)
        row += 1

        self.advanced_checkbox = CheckBox(
            self.tr("Show collision and flexible-solvent settings"),
            self.setting_widget,
        )
        self.advanced_checkbox.setChecked(False)
        self.settingLayout.addWidget(self.advanced_checkbox, row, 0, 1, 3)
        row += 1

        self.min_distance_label = BodyLabel(self.tr("Uniform minimum-distance override"), self.setting_widget)
        self.min_distance_label.setToolTip(self.tr("A positive value replaces all element-radius collision cutoffs; 0 disables"))
        self.min_distance_label.installEventFilter(ToolTipFilter(self.min_distance_label, 300, ToolTipPosition.TOP))
        self.min_distance_frame = SpinBoxUnitInputFrame(self)
        self.min_distance_frame.set_input("Å", 1, "float")
        self.min_distance_frame.setRange(0.0, 100.0)
        self.min_distance_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_distance_frame.set_input_value([0.0])
        self.settingLayout.addWidget(self.min_distance_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.min_distance_frame, row, 1, 1, 2)
        row += 1

        self.collision_label = BodyLabel(self.tr("Element-radius collision scale"), self.setting_widget)
        self.collision_label.setToolTip(self.tr("0 uses the collision profile; ignored when a uniform minimum distance is positive"))
        self.collision_label.installEventFilter(ToolTipFilter(self.collision_label, 300, ToolTipPosition.TOP))
        self.collision_frame = SpinBoxUnitInputFrame(self)
        self.collision_frame.set_input("x", 1, "float")
        self.collision_frame.setRange(0.0, 5.0)
        self.collision_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.collision_frame.set_input_value([0.0])
        self.settingLayout.addWidget(self.collision_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.collision_frame, row, 1, 1, 2)
        row += 1

        self.attempts_label = BodyLabel(self.tr("Attempts/solvent"), self.setting_widget)
        self.attempts_label.setToolTip(self.tr("Maximum placement attempts per requested solvent molecule"))
        self.attempts_label.installEventFilter(ToolTipFilter(self.attempts_label, 300, ToolTipPosition.TOP))
        self.attempts_frame = SpinBoxUnitInputFrame(self)
        self.attempts_frame.set_input("", 1, "int")
        self.attempts_frame.setRange(1, 100000)
        self.attempts_frame.set_input_value([500])
        self.settingLayout.addWidget(self.attempts_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.attempts_frame, row, 1, 1, 2)
        row += 1

        self.strict_checkbox = CheckBox(self.tr("Strict count"), self.setting_widget)
        self.strict_checkbox.setToolTip(self.tr("Fail if the requested solvent count cannot be placed"))
        self.strict_checkbox.setChecked(True)
        self.settingLayout.addWidget(self.strict_checkbox, row, 0, 1, 3)
        row += 1

        self.flex_checkbox = CheckBox(self.tr("Pre-sample flexible solvent conformers"), self.setting_widget)
        self.flex_checkbox.setChecked(False)
        self.flex_checkbox.setToolTip(self.tr("Use the existing torsion-guard core to pre-generate solvent conformers"))
        self.settingLayout.addWidget(self.flex_checkbox, row, 0, 1, 3)
        row += 1

        self.flex_pool_label = BodyLabel(self.tr("Flex pool"), self.setting_widget)
        self.flex_pool_label.setToolTip(self.tr("Number of pre-generated solvent conformers"))
        self.flex_pool_label.installEventFilter(ToolTipFilter(self.flex_pool_label, 300, ToolTipPosition.TOP))
        self.flex_pool_frame = SpinBoxUnitInputFrame(self)
        self.flex_pool_frame.set_input("", 1, "int")
        self.flex_pool_frame.setRange(1, 10000)
        self.flex_pool_frame.set_input_value([32])
        self.settingLayout.addWidget(self.flex_pool_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.flex_pool_frame, row, 1, 1, 2)
        row += 1

        self.flex_torsion_label = BodyLabel(self.tr("Flex torsion"), self.setting_widget)
        self.flex_torsion_label.setToolTip(self.tr("Torsion angle range for flexible solvent conformers"))
        self.flex_torsion_label.installEventFilter(ToolTipFilter(self.flex_torsion_label, 300, ToolTipPosition.TOP))
        self.flex_torsion_frame = SpinBoxUnitInputFrame(self)
        self.flex_torsion_frame.set_input(["deg", "deg"], 2, ["float", "float"])
        self.flex_torsion_frame.setRange(-360.0, 360.0)
        self.flex_torsion_frame.set_input_value([-180.0, 180.0])
        self.settingLayout.addWidget(self.flex_torsion_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.flex_torsion_frame, row, 1, 1, 2)
        row += 1

        self.flex_max_label = BodyLabel(self.tr("Flex max/sigma"), self.setting_widget)
        self.flex_max_label.setToolTip(self.tr("Max torsions per conformer and Gaussian coordinate noise"))
        self.flex_max_label.installEventFilter(ToolTipFilter(self.flex_max_label, 300, ToolTipPosition.TOP))
        self.flex_max_frame = SpinBoxUnitInputFrame(self)
        self.flex_max_frame.set_input(["", "Å"], 2, ["int", "float"])
        self.flex_max_frame.setRange(0.0, 10000.0)
        self.flex_max_frame.object_list[1].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.flex_max_frame.set_input_value([5, 0.03])
        self.settingLayout.addWidget(self.flex_max_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.flex_max_frame, row, 1, 1, 2)
        row += 1

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.settingLayout.addWidget(self.seed_checkbox, row, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, row, 1, 1, 2)
        row += 1

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("solventBoxFillPreview")
        self.settingLayout.addWidget(self.preview_label, row, 0, 1, 3)

        self.density_controls = (
            self.density_label,
            self.density_frame,
            self.fill_packing_label,
            self.fill_packing_frame,
        )
        self.fixed_count_controls = (
            self.count_label,
            self.count_frame,
        )
        self.flex_controls = (
            self.flex_pool_label,
            self.flex_pool_frame,
            self.flex_torsion_label,
            self.flex_torsion_frame,
            self.flex_max_label,
            self.flex_max_frame,
        )
        self.advanced_controls = (
            self.min_distance_label,
            self.min_distance_frame,
            self.collision_label,
            self.collision_frame,
            self.attempts_label,
            self.attempts_frame,
            self.strict_checkbox,
            self.flex_checkbox,
            *self.flex_controls,
        )

        self.edit_solvent_checkbox.stateChanged.connect(
            self._update_solvent_visibility
        )
        self.count_mode_combo.currentIndexChanged.connect(
            self._update_count_visibility
        )
        self.advanced_checkbox.stateChanged.connect(
            self._update_advanced_visibility
        )
        self.flex_checkbox.stateChanged.connect(
            self._update_flex_visibility
        )
        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self.mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self.solvent_edit.textChanged.connect(self._refresh_preview)
        for frame in (
            self.structures_frame,
            self.count_frame,
            self.density_frame,
            self.fill_packing_frame,
            self.min_distance_frame,
            self.collision_frame,
            self.attempts_frame,
            self.flex_pool_frame,
            self.flex_torsion_frame,
            self.flex_max_frame,
            self.seed_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._refresh_preview)

        self._update_solvent_visibility()
        self._update_count_visibility()
        self._update_advanced_visibility()
        self._on_seed_changed()
        self._refresh_preview()

    def _update_solvent_visibility(self, *_args) -> None:
        visible = self.edit_solvent_checkbox.isChecked()
        self.solvent_label.setVisible(visible)
        self.solvent_edit.setVisible(visible)
        self._update_tab_order()

    def _update_count_visibility(self, *_args) -> None:
        density_mode = combo_value(self.count_mode_combo) == "density"
        for widget in self.fixed_count_controls:
            widget.setVisible(not density_mode)
        for widget in self.density_controls:
            widget.setVisible(density_mode)
        self._update_tab_order()
        self._refresh_preview()

    def _update_advanced_visibility(self, *_args) -> None:
        visible = self.advanced_checkbox.isChecked()
        for widget in self.advanced_controls:
            widget.setVisible(visible)
        self._update_flex_visibility()
        self._update_tab_order()

    def _update_flex_visibility(self, *_args) -> None:
        visible = (
            self.advanced_checkbox.isChecked()
            and self.flex_checkbox.isChecked()
        )
        for widget in self.flex_controls:
            widget.setVisible(visible)
        self._update_tab_order()
        self._refresh_preview()

    def _on_seed_changed(self, *_args) -> None:
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())
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

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr(
                    "Load an upstream periodic cell to preview the resolved solvent count and added atom count."
                )
            )
            return
        try:
            summary = self.create_operation().capacity_summary(
                self._input_structure,
                self.get_params(),
            )
        except (TypeError, ValueError, IndexError) as exc:
            self.preview_label.setText(
                "⚠ "
                + self.tr("Preview unavailable: {error}").format(
                    error=translate_runtime_message(exc)
                )
            )
            return

        if summary["min_distance"] > 0.0:
            collision = self.tr("uniform minimum distance {distance} Å").format(
                distance=f"{summary['min_distance']:.3g}",
            )
        else:
            collision = self.tr("element-radius scale {scale}").format(
                scale=f"{summary['collision_scale']:.3g}",
            )
        mode_index = self.mode_combo.findData(summary["mode"])
        mode_text = (
            self.mode_combo.itemText(mode_index)
            if mode_index >= 0
            else summary["mode"]
        )
        message = self.tr(
            "First input: {host} existing atoms · cell {volume} Å³ / PBC {axes} · solvent {formula} / {solvent_atoms} atoms · target {count} molecules (+{added} atoms) × {outputs} outputs · {profile} · {collision}"
        ).format(
            host=summary["host_atoms"],
            volume=f"{summary['volume']:.5g}",
            axes=",".join(summary["pbc_axes"]),
            formula=summary["solvent_formula"],
            solvent_atoms=summary["solvent_atoms"],
            count=summary["target_count"],
            added=summary["added_atoms"],
            outputs=summary["structures"],
            profile=mode_text,
            collision=collision,
        )
        if summary["count_mode"] == "density":
            message += " · " + self.tr(
                "nominal density count uses the full cell volume and does not subtract host occupancy"
            )
        self.preview_label.setText(message)

    def _update_tab_order(self) -> None:
        if not hasattr(self, "advanced_checkbox"):
            return
        widgets = [self.edit_solvent_checkbox]
        if self.edit_solvent_checkbox.isChecked():
            widgets.append(self.solvent_edit)
        widgets.extend(
            [
                *self.structures_frame.object_list,
                self.count_mode_combo,
            ]
        )
        if combo_value(self.count_mode_combo) == "density":
            widgets.extend(self.density_frame.object_list)
            widgets.extend(self.fill_packing_frame.object_list)
        else:
            widgets.extend(self.count_frame.object_list)
        widgets.extend([self.mode_combo, self.advanced_checkbox])
        if self.advanced_checkbox.isChecked():
            widgets.extend(
                [
                    *self.min_distance_frame.object_list,
                    *self.collision_frame.object_list,
                    *self.attempts_frame.object_list,
                    self.strict_checkbox,
                    self.flex_checkbox,
                ]
            )
            if self.flex_checkbox.isChecked():
                widgets.extend(
                    [
                        *self.flex_pool_frame.object_list,
                        *self.flex_torsion_frame.object_list,
                        *self.flex_max_frame.object_list,
                    ]
                )
        widgets.append(self.seed_checkbox)
        if self.seed_frame.isEnabled():
            widgets.extend(self.seed_frame.object_list)
        self.tab_order_widgets = widgets

    def create_operation(self):
        return SolventBoxFillOperation()

    def get_params(self) -> SolventBoxFillParams:
        flex_max_values = self.flex_max_frame.get_input_value()
        return SolventBoxFillParams(
            solvent_xyz=self.solvent_edit.toPlainText(),
            structures=int(self.structures_frame.get_input_value()[0]),
            count_mode=combo_value(self.count_mode_combo),
            solvent_count=int(self.count_frame.get_input_value()[0]),
            density=float(self.density_frame.get_input_value()[0]),
            sampling_mode=combo_value(self.mode_combo),
            fill_packing=float(self.fill_packing_frame.get_input_value()[0]),
            min_distance=float(self.min_distance_frame.get_input_value()[0]),
            collision_scale=float(self.collision_frame.get_input_value()[0]),
            max_attempts_per_solvent=int(self.attempts_frame.get_input_value()[0]),
            strict_count=self.strict_checkbox.isChecked(),
            flex_solvent=self.flex_checkbox.isChecked(),
            flex_pool=int(self.flex_pool_frame.get_input_value()[0]),
            flex_torsion_range=tuple(map(float, self.flex_torsion_frame.get_input_value())),
            flex_max_torsions=int(flex_max_values[0]),
            flex_gaussian_sigma=float(flex_max_values[1]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: SolventBoxFillParams) -> None:
        self.solvent_edit.setPlainText(params.solvent_xyz)
        self.structures_frame.set_input_value([int(params.structures)])
        set_combo_value(self.count_mode_combo, params.count_mode)
        self.count_frame.set_input_value([int(params.solvent_count)])
        self.density_frame.set_input_value([float(params.density)])
        set_combo_value(self.mode_combo, params.sampling_mode)
        self.fill_packing_frame.set_input_value([float(params.fill_packing)])
        self.min_distance_frame.set_input_value([float(params.min_distance)])
        self.collision_frame.set_input_value([float(params.collision_scale)])
        self.attempts_frame.set_input_value([int(params.max_attempts_per_solvent)])
        self.strict_checkbox.setChecked(bool(params.strict_count))
        self.flex_checkbox.setChecked(bool(params.flex_solvent))
        self.flex_pool_frame.set_input_value([int(params.flex_pool)])
        self.flex_torsion_frame.set_input_value([float(value) for value in params.flex_torsion_range])
        self.flex_max_frame.set_input_value([int(params.flex_max_torsions), float(params.flex_gaussian_sigma)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._update_count_visibility()
        self._update_advanced_visibility()
        self._on_seed_changed()
        self._refresh_preview()

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data):
        super().from_dict(data)
        raw = data.get("params")
        params = SolventBoxFillParams(**raw) if raw else SolventBoxFillParams()
        self.set_params(params)
