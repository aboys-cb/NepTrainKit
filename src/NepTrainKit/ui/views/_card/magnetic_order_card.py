"""Card for assigning collinear magnetic moments and generating FM/AFM/PM patterns."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.cards.magnetism import MagneticOrderOperation, MagneticOrderParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class MagneticOrderCard(MakeDataCard):
    """Assign initial magnetic moments and generate common collinear spin patterns."""

    group = "Magnetism"
    card_name = "Magnetic Order"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Magnetic Order Generator")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("magnetic_order_card_widget")

        self.format_label = BodyLabel("Format", self.setting_widget)
        self.format_label.setToolTip("Collinear: scalar MAGMOM; Non-collinear: vector MAGMOM (mx,my,mz)")
        self.format_label.installEventFilter(ToolTipFilter(self.format_label, 300, ToolTipPosition.TOP))
        self.format_combo = ComboBox(self.setting_widget)
        self.format_combo.addItems(["Collinear (scalar)", "Non-collinear (vector)"])
        self.format_combo.setCurrentText("Collinear (scalar)")

        self.axis_label = BodyLabel("Axis (x,y,z)", self.setting_widget)
        self.axis_label.setToolTip("Reference axis for FM/AFM (and PM cone/plane when selected)")
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setDecimals(6)
        self.axis_frame.setRange(-1.0, 1.0)
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.map_label = BodyLabel("Magmom map", self.setting_widget)
        self.map_label.setToolTip('Per-element moments, e.g. "Fe:2.2,Co:1.7,Ni:0.6,Cr:1.0" or JSON')
        self.map_label.installEventFilter(ToolTipFilter(self.map_label, 300, ToolTipPosition.TOP))
        self.map_edit = LineEdit(self.setting_widget)
        self.map_edit.setPlaceholderText("Fe:2.2,Co:1.7,Ni:0.6,Cr:1.0")

        self.use_element_dir_checkbox = CheckBox("Use element vector directions (if provided)", self.setting_widget)
        self.use_element_dir_checkbox.setChecked(False)
        self.use_element_dir_checkbox.setToolTip('If the map provides vectors (e.g. Cr:[0,0,1]), use them as directions for FM/AFM')
        self.use_element_dir_checkbox.installEventFilter(
            ToolTipFilter(self.use_element_dir_checkbox, 300, ToolTipPosition.TOP)
        )

        self.default_label = BodyLabel("Default |m|", self.setting_widget)
        self.default_label.setToolTip("Magnitude for elements not listed in Magmom map")
        self.default_label.installEventFilter(ToolTipFilter(self.default_label, 300, ToolTipPosition.TOP))
        self.default_frame = SpinBoxUnitInputFrame(self)
        self.default_frame.set_input("", 1, "float")
        self.default_frame.setDecimals(6)
        self.default_frame.setRange(0.0, 20.0)
        self.default_frame.set_input_value([0.0])

        self.apply_label = BodyLabel("Apply elements", self.setting_widget)
        self.apply_label.setToolTip("Optional: only assign moments to these elements (comma-separated). Empty=all")
        self.apply_label.installEventFilter(ToolTipFilter(self.apply_label, 300, ToolTipPosition.TOP))
        self.apply_edit = LineEdit(self.setting_widget)
        self.apply_edit.setPlaceholderText("Fe,Co,Ni,Cr")

        self.fm_checkbox = CheckBox("Generate FM", self.setting_widget)
        self.fm_checkbox.setChecked(True)
        self.afm_checkbox = CheckBox("Generate AFM", self.setting_widget)
        self.afm_checkbox.setChecked(True)

        self.afm_mode_label = BodyLabel("AFM mode", self.setting_widget)
        self.afm_mode_label.setToolTip("AFM sign assignment: k-vector layers or explicit A/B groups")
        self.afm_mode_label.installEventFilter(ToolTipFilter(self.afm_mode_label, 300, ToolTipPosition.TOP))
        self.afm_mode_combo = ComboBox(self.setting_widget)
        self.afm_mode_combo.addItems(["k-vector", "group A/B"])

        self.kvec_label = BodyLabel("AFM k-vector", self.setting_widget)
        self.kvec_label.setToolTip("AFM modulation in fractional coordinates: 100, 010, 001, 110, 111")
        self.kvec_label.installEventFilter(ToolTipFilter(self.kvec_label, 300, ToolTipPosition.TOP))
        self.kvec_combo = ComboBox(self.setting_widget)
        self.kvec_combo.addItems(["100", "010", "001", "110", "111"])
        self.kvec_combo.setCurrentText("111")

        self.group_a_label = BodyLabel("AFM group +", self.setting_widget)
        self.group_a_label.setToolTip("Group label assigned + sign (requires arrays['group'])")
        self.group_a_label.installEventFilter(ToolTipFilter(self.group_a_label, 300, ToolTipPosition.TOP))
        self.group_a_edit = LineEdit(self.setting_widget)
        self.group_a_edit.setText("A")

        self.group_b_label = BodyLabel("AFM group -", self.setting_widget)
        self.group_b_label.setToolTip("Group label assigned - sign (requires arrays['group'])")
        self.group_b_label.installEventFilter(ToolTipFilter(self.group_b_label, 300, ToolTipPosition.TOP))
        self.group_b_edit = LineEdit(self.setting_widget)
        self.group_b_edit.setText("B")

        self.zero_unknown_groups_checkbox = CheckBox("Zero unknown groups", self.setting_widget)
        self.zero_unknown_groups_checkbox.setChecked(True)
        self.zero_unknown_groups_checkbox.setToolTip("If an atom group is neither A nor B, set its moment to 0")
        self.zero_unknown_groups_checkbox.installEventFilter(
            ToolTipFilter(self.zero_unknown_groups_checkbox, 300, ToolTipPosition.TOP)
        )

        self.pm_checkbox = CheckBox("Generate PM (random signs)", self.setting_widget)
        self.pm_checkbox.setChecked(False)
        self.pm_count_label = BodyLabel("PM structures", self.setting_widget)
        self.pm_count_frame = SpinBoxUnitInputFrame(self)
        self.pm_count_frame.set_input("unit", 1, "int")
        self.pm_count_frame.setRange(1, 999999)
        self.pm_count_frame.set_input_value([10])

        self.pm_direction_label = BodyLabel("PM directions", self.setting_widget)
        self.pm_direction_label.setToolTip("Non-collinear PM direction distribution")
        self.pm_direction_label.installEventFilter(ToolTipFilter(self.pm_direction_label, 300, ToolTipPosition.TOP))
        self.pm_direction_combo = ComboBox(self.setting_widget)
        self.pm_direction_combo.addItems(["sphere", "cone", "plane", "axis"])
        self.pm_direction_combo.setCurrentText("sphere")

        self.pm_cone_label = BodyLabel("PM cone angle", self.setting_widget)
        self.pm_cone_label.setToolTip("Cone half-angle in degrees (used when PM directions=cone)")
        self.pm_cone_label.installEventFilter(ToolTipFilter(self.pm_cone_label, 300, ToolTipPosition.TOP))
        self.pm_cone_frame = SpinBoxUnitInputFrame(self)
        self.pm_cone_frame.set_input("deg", 1, "float")
        self.pm_cone_frame.setDecimals(3)
        self.pm_cone_frame.setRange(0.0, 180.0)
        self.pm_cone_frame.set_input_value([30.0])

        self.pm_balanced_checkbox = CheckBox("PM balanced", self.setting_widget)
        self.pm_balanced_checkbox.setChecked(True)
        self.pm_balanced_checkbox.setToolTip("Try to balance +/- signs to near-zero net moment")
        self.pm_balanced_checkbox.installEventFilter(ToolTipFilter(self.pm_balanced_checkbox, 300, ToolTipPosition.TOP))

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.settingLayout.addWidget(self.format_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.format_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.axis_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_frame, 1, 1, 1, 2)

        self.settingLayout.addWidget(self.map_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.use_element_dir_checkbox, 3, 0, 1, 3)

        self.settingLayout.addWidget(self.default_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.apply_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 5, 1, 1, 2)

        self.settingLayout.addWidget(self.fm_checkbox, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.afm_checkbox, 6, 1, 1, 1)
        self.settingLayout.addWidget(self.afm_mode_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.afm_mode_combo, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.kvec_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.kvec_combo, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.group_a_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.group_a_edit, 9, 1, 1, 2)
        self.settingLayout.addWidget(self.group_b_label, 10, 0, 1, 1)
        self.settingLayout.addWidget(self.group_b_edit, 10, 1, 1, 2)
        self.settingLayout.addWidget(self.zero_unknown_groups_checkbox, 11, 0, 1, 2)

        self.settingLayout.addWidget(self.pm_checkbox, 12, 0, 1, 2)
        self.settingLayout.addWidget(self.pm_count_label, 13, 0, 1, 1)
        self.settingLayout.addWidget(self.pm_count_frame, 13, 1, 1, 2)
        self.settingLayout.addWidget(self.pm_direction_label, 14, 0, 1, 1)
        self.settingLayout.addWidget(self.pm_direction_combo, 14, 1, 1, 2)
        self.settingLayout.addWidget(self.pm_cone_label, 15, 0, 1, 1)
        self.settingLayout.addWidget(self.pm_cone_frame, 15, 1, 1, 2)
        self.settingLayout.addWidget(self.pm_balanced_checkbox, 16, 0, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 17, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 17, 1, 1, 2)

        self.afm_mode_combo.currentTextChanged.connect(self._update_afm_mode_widgets)
        self.afm_checkbox.stateChanged.connect(lambda _state: self._update_afm_mode_widgets())
        self._update_afm_mode_widgets()

    def _update_afm_mode_widgets(self):
        """Show/hide AFM controls based on the selected AFM mode."""
        afm_enabled = self.afm_checkbox.isChecked()
        mode = self.afm_mode_combo.currentText()

        use_group = mode == "group A/B"

        self.kvec_label.setVisible(not use_group)
        self.kvec_combo.setVisible(not use_group)
        self.kvec_label.setEnabled(afm_enabled and (not use_group))
        self.kvec_combo.setEnabled(afm_enabled and (not use_group))

        self.group_a_label.setVisible(use_group)
        self.group_a_edit.setVisible(use_group)
        self.group_b_label.setVisible(use_group)
        self.group_b_edit.setVisible(use_group)
        self.zero_unknown_groups_checkbox.setVisible(use_group)

        self.group_a_label.setEnabled(afm_enabled and use_group)
        self.group_a_edit.setEnabled(afm_enabled and use_group)
        self.group_b_label.setEnabled(afm_enabled and use_group)
        self.group_b_edit.setEnabled(afm_enabled and use_group)
        self.zero_unknown_groups_checkbox.setEnabled(afm_enabled and use_group)

    def create_operation(self):
        return MagneticOrderOperation()

    def get_params(self) -> MagneticOrderParams:
        return MagneticOrderParams(
            format=self.format_combo.currentText(),
            axis=self.axis_frame.get_input_value(),
            magmom_map=self.map_edit.text(),
            use_element_dirs=self.use_element_dir_checkbox.isChecked(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            apply_elements=self.apply_edit.text(),
            gen_fm=self.fm_checkbox.isChecked(),
            gen_afm=self.afm_checkbox.isChecked(),
            afm_mode=self.afm_mode_combo.currentText(),
            afm_kvec=self.kvec_combo.currentText(),
            afm_group_a=self.group_a_edit.text(),
            afm_group_b=self.group_b_edit.text(),
            afm_zero_unknown=self.zero_unknown_groups_checkbox.isChecked(),
            gen_pm=self.pm_checkbox.isChecked(),
            pm_count=int(self.pm_count_frame.get_input_value()[0]),
            pm_direction=self.pm_direction_combo.currentText(),
            pm_cone_angle=float(self.pm_cone_frame.get_input_value()[0]),
            pm_balanced=self.pm_balanced_checkbox.isChecked(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: MagneticOrderParams) -> None:
        self.format_combo.setCurrentText(params.format)
        self.axis_frame.set_input_value([float(v) for v in params.axis])
        self.map_edit.setText(params.magmom_map)
        self.use_element_dir_checkbox.setChecked(bool(params.use_element_dirs))
        self.default_frame.set_input_value([float(params.default_moment)])
        self.apply_edit.setText(params.apply_elements)
        self.fm_checkbox.setChecked(bool(params.gen_fm))
        self.afm_checkbox.setChecked(bool(params.gen_afm))
        self.afm_mode_combo.setCurrentText(params.afm_mode)
        self.kvec_combo.setCurrentText(params.afm_kvec)
        self.group_a_edit.setText(params.afm_group_a)
        self.group_b_edit.setText(params.afm_group_b)
        self.zero_unknown_groups_checkbox.setChecked(bool(params.afm_zero_unknown))
        self.pm_checkbox.setChecked(bool(params.gen_pm))
        self.pm_count_frame.set_input_value([int(params.pm_count)])
        self.pm_direction_combo.setCurrentText(params.pm_direction)
        self.pm_cone_frame.set_input_value([float(params.pm_cone_angle)])
        self.pm_balanced_checkbox.setChecked(bool(params.pm_balanced))
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())
        self._update_afm_mode_widgets()

    def process_structure(self, structure):
        try:
            if self.afm_checkbox.isChecked() and self.afm_mode_combo.currentText() == "group A/B" and "group" not in structure.arrays:
                MessageManager.send_warning_message("MagneticOrder: AFM mode 'group A/B' requires arrays['group']; falling back to k-vector.")
            return self.create_operation().run_structure(structure, self.get_params())
        except Exception as exc:  # noqa: BLE001
            MessageManager.send_warning_message(f"MagneticOrder: invalid magmom map: {exc}")
            return [structure]

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = MagneticOrderParams(**raw_params)
        else:
            params = MagneticOrderParams(
                format=data_dict.get("format", "Collinear (scalar)"),
                axis=data_dict.get("axis", [0.0, 0.0, 1.0]),
                magmom_map=data_dict.get("magmom_map", ""),
                use_element_dirs=data_dict.get("use_element_dirs", False),
                default_moment=data_dict.get("default_moment", [0.0])[0],
                apply_elements=data_dict.get("apply_elements", ""),
                gen_fm=data_dict.get("gen_fm", True),
                gen_afm=data_dict.get("gen_afm", True),
                afm_mode=data_dict.get("afm_mode", "k-vector"),
                afm_kvec=data_dict.get("afm_kvec", "111"),
                afm_group_a=data_dict.get("afm_group_a", "A"),
                afm_group_b=data_dict.get("afm_group_b", "B"),
                afm_zero_unknown=data_dict.get("afm_zero_unknown", True),
                gen_pm=data_dict.get("gen_pm", False),
                pm_count=data_dict.get("pm_count", [10])[0],
                pm_direction=data_dict.get("pm_direction", "sphere"),
                pm_cone_angle=data_dict.get("pm_cone_angle", [30.0])[0],
                pm_balanced=data_dict.get("pm_balanced", True),
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
