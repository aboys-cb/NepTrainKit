"""Card for local solvent-shell generation."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPlainTextEdit
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    LineEdit,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.solvation import (
    DEFAULT_WATER_XYZ,
    LocalSolvationOperation,
    LocalSolvationParams,
    has_valid_cell,
)
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class LocalSolvationCard(MakeDataCard):
    """Generate local solvent shells around selected atoms."""

    group = "Organic"
    card_name = "Local Solvation"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [
        {"name": "Chen Zherui", "role": "author", "email": "chenzherui0124@foxmail.com"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self.setTitle(self.tr("Local Solvent Shell"))
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("local_solvation_card_widget")
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
        self.structures_label.setToolTip(self.tr("Independent solvated structures generated per input structure"))
        self.structures_label.installEventFilter(ToolTipFilter(self.structures_label, 300, ToolTipPosition.TOP))
        self.structures_frame = SpinBoxUnitInputFrame(self)
        self.structures_frame.set_input("", 1, "int")
        self.structures_frame.setRange(1, 100000)
        self.structures_frame.set_input_value([1])
        self.settingLayout.addWidget(self.structures_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.structures_frame, row, 1, 1, 2)
        row += 1

        self.count_label = BodyLabel(self.tr("Solvent molecules per output"), self.setting_widget)
        self.count_label.setToolTip(self.tr("Number of solvent molecules inserted in each generated structure"))
        self.count_label.installEventFilter(ToolTipFilter(self.count_label, 300, ToolTipPosition.TOP))
        self.count_frame = SpinBoxUnitInputFrame(self)
        self.count_frame.set_input("", 1, "int")
        self.count_frame.setRange(1, 100000)
        self.count_frame.set_input_value([6])
        self.settingLayout.addWidget(self.count_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.count_frame, row, 1, 1, 2)
        row += 1

        self.mode_label = BodyLabel(self.tr("Placement profile"), self.setting_widget)
        self.mode_label.setToolTip(self.tr("Auto detects water and ion-water from the solvent and selected center elements"))
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.mode_combo,
            [
                ("auto", "Auto-detect solvent / ion"),
                ("general", "General random orientation"),
                ("water", "Water dipole orientation"),
                ("ion-water", "Ion-water first shell"),
                ("loose", "Loose collision profile"),
                ("dense", "Dense collision profile"),
            ],
        )
        self.settingLayout.addWidget(self.mode_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, row, 1, 1, 2)
        row += 1

        self.center_mode_label = BodyLabel(self.tr("Solvation centers"), self.setting_widget)
        self.center_mode_label.setToolTip(self.tr("How center atoms for local solvation are selected"))
        self.center_mode_label.installEventFilter(ToolTipFilter(self.center_mode_label, 300, ToolTipPosition.TOP))
        self.center_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.center_mode_combo,
            [
                ("all", "All host atoms"),
                ("elements", "By element"),
                ("indices", "By 1-based atom index"),
                ("z_range", "By Cartesian z range"),
            ],
        )
        self.settingLayout.addWidget(self.center_mode_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.center_mode_combo, row, 1, 1, 2)
        row += 1

        self.elements_label = BodyLabel(self.tr("Center elements"), self.setting_widget)
        self.elements_label.setToolTip(self.tr("Comma-separated element symbols used when center mode is elements"))
        self.elements_label.installEventFilter(ToolTipFilter(self.elements_label, 300, ToolTipPosition.TOP))
        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setPlaceholderText(self.tr("Ca, Na, O"))
        self.settingLayout.addWidget(self.elements_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.elements_edit, row, 1, 1, 2)
        row += 1

        self.indices_label = BodyLabel(self.tr("Center indices"), self.setting_widget)
        self.indices_label.setToolTip(self.tr("1-based atom indices/ranges used when center mode is indices"))
        self.indices_label.installEventFilter(ToolTipFilter(self.indices_label, 300, ToolTipPosition.TOP))
        self.indices_edit = LineEdit(self.setting_widget)
        self.indices_edit.setPlaceholderText(self.tr("1,3,5-8"))
        self.settingLayout.addWidget(self.indices_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.indices_edit, row, 1, 1, 2)
        row += 1

        self.z_label = BodyLabel(self.tr("Z range"), self.setting_widget)
        self.z_label.setToolTip(self.tr("Cartesian z interval used when center mode is z_range"))
        self.z_label.installEventFilter(ToolTipFilter(self.z_label, 300, ToolTipPosition.TOP))
        self.z_frame = SpinBoxUnitInputFrame(self)
        self.z_frame.set_input(["A", "A"], 2, ["float", "float"])
        self.z_frame.setRange(-100000.0, 100000.0)
        self.z_frame.set_input_value([0.0, 0.0])
        self.settingLayout.addWidget(self.z_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.z_frame, row, 1, 1, 2)
        row += 1

        self.shell_label = BodyLabel(self.tr("Fallback center-to-COM shell"), self.setting_widget)
        self.shell_label.setToolTip(self.tr("Used for general placement and after an ion-water coordination shell is filled"))
        self.shell_label.installEventFilter(ToolTipFilter(self.shell_label, 300, ToolTipPosition.TOP))
        self.shell_frame = SpinBoxUnitInputFrame(self)
        self.shell_frame.set_input(["Å", "Å"], 2, ["float", "float"])
        self.shell_frame.setRange(0.0, 1000.0)
        self.shell_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.shell_frame.object_list[1].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.shell_frame.set_input_value([2.2, 4.5])
        self.settingLayout.addWidget(self.shell_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.shell_frame, row, 1, 1, 2)
        row += 1

        self.advanced_checkbox = CheckBox(
            self.tr("Show collision, box, and flexible-solvent settings"),
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
        self.collision_label.setToolTip(self.tr("0 uses the placement profile; ignored when a uniform minimum distance is positive"))
        self.collision_label.installEventFilter(ToolTipFilter(self.collision_label, 300, ToolTipPosition.TOP))
        self.collision_frame = SpinBoxUnitInputFrame(self)
        self.collision_frame.set_input("x", 1, "float")
        self.collision_frame.setRange(0.0, 5.0)
        self.collision_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.collision_frame.set_input_value([0.0])
        self.settingLayout.addWidget(self.collision_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.collision_frame, row, 1, 1, 2)
        row += 1

        self.attempts_label = BodyLabel(self.tr("Total placement attempts per output"), self.setting_widget)
        self.attempts_label.setToolTip(self.tr("Maximum placement attempts per generated structure"))
        self.attempts_label.installEventFilter(ToolTipFilter(self.attempts_label, 300, ToolTipPosition.TOP))
        self.attempts_frame = SpinBoxUnitInputFrame(self)
        self.attempts_frame.set_input("", 1, "int")
        self.attempts_frame.setRange(1, 10000000)
        self.attempts_frame.set_input_value([3000])
        self.settingLayout.addWidget(self.attempts_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.attempts_frame, row, 1, 1, 2)
        row += 1

        self.strict_checkbox = CheckBox(self.tr("Strict count"), self.setting_widget)
        self.strict_checkbox.setToolTip(self.tr("Fail if the requested solvent count cannot be placed"))
        self.strict_checkbox.setChecked(True)
        self.settingLayout.addWidget(self.strict_checkbox, row, 0, 1, 3)
        row += 1

        self.auto_box_checkbox = CheckBox(self.tr("Auto-size box when input has no valid cell"), self.setting_widget)
        self.auto_box_checkbox.setChecked(False)
        self.settingLayout.addWidget(self.auto_box_checkbox, row, 0, 1, 3)
        row += 1

        self.box_size_label = BodyLabel(self.tr("Fixed box size"), self.setting_widget)
        self.box_size_label.setToolTip(self.tr("Non-periodic fixed output box when auto box is off"))
        self.box_size_label.installEventFilter(ToolTipFilter(self.box_size_label, 300, ToolTipPosition.TOP))
        self.box_size_frame = SpinBoxUnitInputFrame(self)
        self.box_size_frame.set_input("Å", 1, "float")
        self.box_size_frame.setRange(0.001, 100000.0)
        self.box_size_frame.set_input_value([100.0])
        self.settingLayout.addWidget(self.box_size_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.box_size_frame, row, 1, 1, 2)
        row += 1

        self.box_label = BodyLabel(self.tr("Auto box padding/min"), self.setting_widget)
        self.box_label.setToolTip(self.tr("Padding and minimum edge length used by auto box"))
        self.box_label.installEventFilter(ToolTipFilter(self.box_label, 300, ToolTipPosition.TOP))
        self.box_frame = SpinBoxUnitInputFrame(self)
        self.box_frame.set_input(["Å", "Å"], 2, ["float", "float"])
        self.box_frame.setRange(0.0, 100000.0)
        self.box_frame.set_input_value([8.0, 0.0])
        self.settingLayout.addWidget(self.box_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.box_frame, row, 1, 1, 2)
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
        self.preview_label.setObjectName("localSolvationPreview")
        self.settingLayout.addWidget(self.preview_label, row, 0, 1, 3)

        self.center_element_controls = (
            self.elements_label,
            self.elements_edit,
        )
        self.center_index_controls = (
            self.indices_label,
            self.indices_edit,
        )
        self.center_z_controls = (
            self.z_label,
            self.z_frame,
        )
        self.flex_controls = (
            self.flex_pool_label,
            self.flex_pool_frame,
            self.flex_torsion_label,
            self.flex_torsion_frame,
            self.flex_max_label,
            self.flex_max_frame,
        )
        self.box_controls = (
            self.auto_box_checkbox,
            self.box_size_label,
            self.box_size_frame,
            self.box_label,
            self.box_frame,
        )
        self.advanced_controls = (
            self.min_distance_label,
            self.min_distance_frame,
            self.collision_label,
            self.collision_frame,
            self.attempts_label,
            self.attempts_frame,
            self.strict_checkbox,
            *self.box_controls,
            self.flex_checkbox,
            *self.flex_controls,
        )

        self.edit_solvent_checkbox.stateChanged.connect(
            self._update_solvent_visibility
        )
        self.advanced_checkbox.stateChanged.connect(
            self._update_advanced_visibility
        )
        self.center_mode_combo.currentIndexChanged.connect(
            self._update_center_visibility
        )
        self.auto_box_checkbox.stateChanged.connect(
            self._update_box_visibility
        )
        self.flex_checkbox.stateChanged.connect(
            self._update_flex_visibility
        )
        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self.mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self.solvent_edit.textChanged.connect(self._refresh_preview)
        self.elements_edit.textChanged.connect(self._refresh_preview)
        self.indices_edit.textChanged.connect(self._refresh_preview)
        for frame in (
            self.structures_frame,
            self.count_frame,
            self.z_frame,
            self.shell_frame,
            self.min_distance_frame,
            self.collision_frame,
            self.attempts_frame,
            self.box_size_frame,
            self.box_frame,
            self.flex_pool_frame,
            self.flex_torsion_frame,
            self.flex_max_frame,
            self.seed_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._refresh_preview)

        self._update_solvent_visibility()
        self._update_center_visibility()
        self._update_advanced_visibility()
        self._on_seed_changed()
        self._refresh_preview()

    def _update_solvent_visibility(self, *_args) -> None:
        visible = self.edit_solvent_checkbox.isChecked()
        self.solvent_label.setVisible(visible)
        self.solvent_edit.setVisible(visible)
        self._update_tab_order()

    def _update_center_visibility(self, *_args) -> None:
        mode = combo_value(self.center_mode_combo)
        for widget in self.center_element_controls:
            widget.setVisible(mode == "elements")
        for widget in self.center_index_controls:
            widget.setVisible(mode == "indices")
        for widget in self.center_z_controls:
            widget.setVisible(mode == "z_range")
        self._update_tab_order()
        self._refresh_preview()

    def _update_advanced_visibility(self, *_args) -> None:
        visible = self.advanced_checkbox.isChecked()
        for widget in self.advanced_controls:
            widget.setVisible(visible)
        self._update_box_visibility()
        self._update_flex_visibility()
        self._update_tab_order()

    def _update_box_visibility(self, *_args) -> None:
        advanced = self.advanced_checkbox.isChecked()
        input_has_cell = (
            self._input_structure is not None
            and has_valid_cell(self._input_structure)
        )
        applicable = advanced and not input_has_cell
        self.auto_box_checkbox.setVisible(applicable)
        use_auto = applicable and self.auto_box_checkbox.isChecked()
        self.box_size_label.setVisible(applicable and not use_auto)
        self.box_size_frame.setVisible(applicable and not use_auto)
        self.box_label.setVisible(use_auto)
        self.box_frame.setVisible(use_auto)
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
        self._update_box_visibility()
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr(
                    "Load an upstream structure to preview selected centers and the resolved placement profile."
                )
            )
            return
        try:
            summary = self.create_operation().placement_summary(
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

        if summary["uniform_min_distance"] > 0.0:
            collision = self.tr("uniform minimum distance {distance} Å").format(
                distance=f"{summary['uniform_min_distance']:.3g}",
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
        elements = ", ".join(summary["selected_elements"])
        message = self.tr(
            "First input: {host} host atoms · centers {centers} ({elements}) · solvent {formula} / {solvent_atoms} atoms · resolved profile {mode} · {outputs} outputs × {count} molecules · {collision}"
        ).format(
            host=summary["host_atoms"],
            centers=summary["center_count"],
            elements=elements,
            formula=summary["solvent_formula"],
            solvent_atoms=summary["solvent_atoms"],
            mode=mode_text,
            outputs=summary["structures"],
            count=summary["solvent_count"],
            collision=collision,
        )
        if summary["ion_oxygen_ranges"] and summary["mode"] == "ion-water":
            ranges = ", ".join(
                f"{symbol} {bounds[0]:g}–{bounds[1]:g} Å"
                for symbol, bounds in summary["ion_oxygen_ranges"].items()
            )
            message += " · " + self.tr(
                "first-shell ion–O ranges: {ranges}; fallback shell applies afterwards"
            ).format(ranges=ranges)
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
                *self.count_frame.object_list,
                self.mode_combo,
                self.center_mode_combo,
            ]
        )
        mode = combo_value(self.center_mode_combo)
        if mode == "elements":
            widgets.append(self.elements_edit)
        elif mode == "indices":
            widgets.append(self.indices_edit)
        elif mode == "z_range":
            widgets.extend(self.z_frame.object_list)
        widgets.extend(self.shell_frame.object_list)
        widgets.append(self.advanced_checkbox)
        if self.advanced_checkbox.isChecked():
            widgets.extend(
                [
                    *self.min_distance_frame.object_list,
                    *self.collision_frame.object_list,
                    *self.attempts_frame.object_list,
                    self.strict_checkbox,
                ]
            )
            if self.auto_box_checkbox.isVisible():
                widgets.append(self.auto_box_checkbox)
                if self.auto_box_checkbox.isChecked():
                    widgets.extend(self.box_frame.object_list)
                else:
                    widgets.extend(self.box_size_frame.object_list)
            widgets.append(self.flex_checkbox)
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
        return LocalSolvationOperation()

    def get_params(self) -> LocalSolvationParams:
        flex_max_values = self.flex_max_frame.get_input_value()
        box_values = self.box_frame.get_input_value()
        return LocalSolvationParams(
            solvent_xyz=self.solvent_edit.toPlainText(),
            structures=int(self.structures_frame.get_input_value()[0]),
            solvent_count=int(self.count_frame.get_input_value()[0]),
            sampling_mode=combo_value(self.mode_combo),
            center_mode=combo_value(self.center_mode_combo),
            center_elements=self.elements_edit.text(),
            center_indices=self.indices_edit.text(),
            z_range=tuple(map(float, self.z_frame.get_input_value())),
            shell=tuple(map(float, self.shell_frame.get_input_value())),
            min_distance=float(self.min_distance_frame.get_input_value()[0]),
            collision_scale=float(self.collision_frame.get_input_value()[0]),
            max_attempts=int(self.attempts_frame.get_input_value()[0]),
            strict_count=self.strict_checkbox.isChecked(),
            auto_box=self.auto_box_checkbox.isChecked(),
            box_size=float(self.box_size_frame.get_input_value()[0]),
            box_padding=float(box_values[0]),
            min_box=float(box_values[1]),
            flex_solvent=self.flex_checkbox.isChecked(),
            flex_pool=int(self.flex_pool_frame.get_input_value()[0]),
            flex_torsion_range=tuple(map(float, self.flex_torsion_frame.get_input_value())),
            flex_max_torsions=int(flex_max_values[0]),
            flex_gaussian_sigma=float(flex_max_values[1]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: LocalSolvationParams) -> None:
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
        params = LocalSolvationParams(**raw) if raw else LocalSolvationParams()
        self.set_params(params)
