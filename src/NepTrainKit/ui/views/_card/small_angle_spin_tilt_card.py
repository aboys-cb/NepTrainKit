"""Card for generating single-spin small-angle tilt configurations."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.magnetism import SmallAngleSpinTiltOperation, SmallAngleSpinTiltParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import KeyValueTableInput, MakeDataCard, SpinBoxUnitInputFrame
from .i18n_utils import add_translated_items, combo_value, set_combo_value


@CardManager.register_card
class SmallAngleSpinTiltCard(MakeDataCard):
    """Generate deterministic single-spin small-angle tilt configurations."""

    group = "Magnetism"
    card_name = "Small-Angle Spin Tilt"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Small-Angle Spin Tilt"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("small_angle_spin_tilt_card_widget")

        self.canting_mode_label = BodyLabel(self.tr("Canting mode"), self.setting_widget)
        self.canting_mode_label.setToolTip(self.tr("Choose global tilt, single-spin tilt, atom-pair canting, or group-pair canting"))
        self.canting_mode_label.installEventFilter(ToolTipFilter(self.canting_mode_label, 300, ToolTipPosition.TOP))
        self.canting_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.canting_mode_combo,
            ["Single-spin tilt", "Global tilt", "Atom pair canting", "Group pair canting"],
        )
        set_combo_value(self.canting_mode_combo, "Single-spin tilt")

        self.target_mode_label = BodyLabel(self.tr("Target atoms"), self.setting_widget)
        self.target_mode_label.setToolTip(self.tr("Choose which atoms receive the single-spin tilt"))
        self.target_mode_label.installEventFilter(ToolTipFilter(self.target_mode_label, 300, ToolTipPosition.TOP))
        self.target_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.target_mode_combo,
            ["First eligible atom", "All eligible atoms", "Explicit indices (1-based)"],
        )
        set_combo_value(self.target_mode_combo, "First eligible atom")

        self.target_indices_label = BodyLabel(self.tr("Atom indices"), self.setting_widget)
        self.target_indices_label.setToolTip(self.tr("Used when Target atoms = Explicit indices (1-based), for example 1,3-5"))
        self.target_indices_label.installEventFilter(
            ToolTipFilter(self.target_indices_label, 300, ToolTipPosition.TOP)
        )
        self.target_indices_edit = LineEdit(self.setting_widget)
        self.target_indices_edit.setPlaceholderText(self.tr("1,3-5"))

        self.pair_left_label = BodyLabel(self.tr("Pair left indices"), self.setting_widget)
        self.pair_left_label.setToolTip(self.tr("1-based indices for the left side of each atom pair, for example 1,3-5"))
        self.pair_left_label.installEventFilter(ToolTipFilter(self.pair_left_label, 300, ToolTipPosition.TOP))
        self.pair_left_edit = LineEdit(self.setting_widget)
        self.pair_left_edit.setPlaceholderText(self.tr("1"))

        self.pair_right_label = BodyLabel(self.tr("Pair right indices"), self.setting_widget)
        self.pair_right_label.setToolTip(self.tr("1-based indices for the right side of each atom pair, paired in order with left indices"))
        self.pair_right_label.installEventFilter(ToolTipFilter(self.pair_right_label, 300, ToolTipPosition.TOP))
        self.pair_right_edit = LineEdit(self.setting_widget)
        self.pair_right_edit.setPlaceholderText(self.tr("2"))

        self.pair_source_label = BodyLabel(self.tr("Pair source"), self.setting_widget)
        self.pair_source_label.setToolTip(self.tr("Use explicit atom indices or auto-select unique neighbor-shell pairs"))
        self.pair_source_label.installEventFilter(ToolTipFilter(self.pair_source_label, 300, ToolTipPosition.TOP))
        self.pair_source_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.pair_source_combo, ["Manual indices", "Auto by neighbor shell"])
        set_combo_value(self.pair_source_combo, "Manual indices")

        self.pair_shell_label = BodyLabel(self.tr("Neighbor shell"), self.setting_widget)
        self.pair_shell_label.setToolTip(self.tr("1 = first-neighbor shell, 2 = second-neighbor shell, etc."))
        self.pair_shell_label.installEventFilter(ToolTipFilter(self.pair_shell_label, 300, ToolTipPosition.TOP))
        self.pair_shell_frame = SpinBoxUnitInputFrame(self)
        self.pair_shell_frame.set_input("", 1, "int")
        self.pair_shell_frame.setRange(1, 20)
        self.pair_shell_frame.set_input_value([1])

        self.pair_tol_label = BodyLabel(self.tr("Shell tolerance"), self.setting_widget)
        self.pair_tol_label.setToolTip(self.tr("Distances within this tolerance belong to the same neighbor shell"))
        self.pair_tol_label.installEventFilter(ToolTipFilter(self.pair_tol_label, 300, ToolTipPosition.TOP))
        self.pair_tol_frame = SpinBoxUnitInputFrame(self)
        self.pair_tol_frame.set_input("A", 1, "float")
        self.pair_tol_frame.setRange(0.0001, 5.0)
        self.pair_tol_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.pair_tol_frame.set_input_value([0.05])

        self.pair_element_label = BodyLabel(self.tr("Pair elements"), self.setting_widget)
        self.pair_element_label.setToolTip(self.tr("Optional pair filter such as Fe-Fe,Fe-Co; empty means any element pair"))
        self.pair_element_label.installEventFilter(ToolTipFilter(self.pair_element_label, 300, ToolTipPosition.TOP))
        self.pair_element_edit = LineEdit(self.setting_widget)
        self.pair_element_edit.setPlaceholderText(self.tr("Fe-Fe,Fe-Co"))

        self.pair_group_label = BodyLabel(self.tr("Pair groups"), self.setting_widget)
        self.pair_group_label.setToolTip(self.tr("Optional group-pair filter such as A-B,A-A; requires arrays['group']"))
        self.pair_group_label.installEventFilter(ToolTipFilter(self.pair_group_label, 300, ToolTipPosition.TOP))
        self.pair_group_edit = LineEdit(self.setting_widget)
        self.pair_group_edit.setPlaceholderText(self.tr("A-B"))

        self.bond_mode_label = BodyLabel(self.tr("Bond filter"), self.setting_widget)
        self.bond_mode_label.setToolTip(self.tr("Optional bond-direction filter for auto pairs"))
        self.bond_mode_label.installEventFilter(ToolTipFilter(self.bond_mode_label, 300, ToolTipPosition.TOP))
        self.bond_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.bond_mode_combo, ["Any", "Near axis", "In plane (normal)"])
        set_combo_value(self.bond_mode_combo, "Any")

        self.bond_axis_label = BodyLabel(self.tr("Bond reference"), self.setting_widget)
        self.bond_axis_label.setToolTip(self.tr("Reference axis or plane normal used by the bond-direction filter"))
        self.bond_axis_label.installEventFilter(ToolTipFilter(self.bond_axis_label, 300, ToolTipPosition.TOP))
        self.bond_axis_frame = SpinBoxUnitInputFrame(self)
        self.bond_axis_frame.set_input("", 3, "float")
        self.bond_axis_frame.setRange(-1.0, 1.0)
        for obj in self.bond_axis_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.bond_axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.bond_tol_label = BodyLabel(self.tr("Bond angle tol"), self.setting_widget)
        self.bond_tol_label.setToolTip(self.tr("Angular tolerance in degrees for the bond-direction filter"))
        self.bond_tol_label.installEventFilter(ToolTipFilter(self.bond_tol_label, 300, ToolTipPosition.TOP))
        self.bond_tol_frame = SpinBoxUnitInputFrame(self)
        self.bond_tol_frame.set_input("deg", 1, "float")
        self.bond_tol_frame.setRange(0.1, 90.0)
        self.bond_tol_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.bond_tol_frame.set_input_value([20.0])

        self.group_a_label = BodyLabel(self.tr("Group A"), self.setting_widget)
        self.group_a_label.setToolTip(self.tr("Atoms with arrays['group']==Group A rotate by +theta/2 in group-pair mode"))
        self.group_a_label.installEventFilter(ToolTipFilter(self.group_a_label, 300, ToolTipPosition.TOP))
        self.group_a_edit = LineEdit(self.setting_widget)
        self.group_a_edit.setText("A")

        self.group_b_label = BodyLabel(self.tr("Group B"), self.setting_widget)
        self.group_b_label.setToolTip(self.tr("Atoms with arrays['group']==Group B rotate by -theta/2 in group-pair mode"))
        self.group_b_label.installEventFilter(ToolTipFilter(self.group_b_label, 300, ToolTipPosition.TOP))
        self.group_b_edit = LineEdit(self.setting_widget)
        self.group_b_edit.setText("B")

        self.angle_label = BodyLabel(self.tr("Tilt angles"), self.setting_widget)
        self.angle_label.setToolTip(self.tr("Comma-separated tilt angles in degrees, for example 1,2,5,10"))
        self.angle_label.installEventFilter(ToolTipFilter(self.angle_label, 300, ToolTipPosition.TOP))
        self.angle_edit = LineEdit(self.setting_widget)
        self.angle_edit.setPlaceholderText(self.tr("1,2,5,10"))
        self.angle_edit.setText("1,2,5,10")

        self.sign_label = BodyLabel(self.tr("Tilt signs"), self.setting_widget)
        self.sign_label.setToolTip(self.tr("Choose whether to emit +theta only, -theta only, or paired +/-theta variants"))
        self.sign_label.installEventFilter(ToolTipFilter(self.sign_label, 300, ToolTipPosition.TOP))
        self.sign_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.sign_combo, ["Positive only", "Negative only", "Both (+/- pair)"])
        set_combo_value(self.sign_combo, "Positive only")

        self.include_reference_checkbox = CheckBox(self.tr("Include reference state"), self.setting_widget)
        self.include_reference_checkbox.setChecked(True)
        self.include_reference_checkbox.setToolTip(self.tr("Emit the un-tilted reference magnetic state before tilted variants"))
        self.include_reference_checkbox.installEventFilter(
            ToolTipFilter(self.include_reference_checkbox, 300, ToolTipPosition.TOP)
        )

        self.advanced_checkbox = CheckBox(
            self.tr("Show magnetic-moment setup, axes, and output limit"), self.setting_widget
        )
        self.advanced_checkbox.setChecked(False)

        self.source_label = BodyLabel(self.tr("Magnitude source"), self.setting_widget)
        self.source_label.setToolTip(self.tr("Use existing initial magmoms or build a ferromagnetic reference from map/default"))
        self.source_label.installEventFilter(ToolTipFilter(self.source_label, 300, ToolTipPosition.TOP))
        self.source_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.source_combo, ["Existing initial magmoms", "Map/default magnitude"])
        set_combo_value(self.source_combo, "Existing initial magmoms")

        self.map_label = BodyLabel(self.tr("Magmom map"), self.setting_widget)
        self.map_label.setToolTip(self.tr('Used when source=Map/default magnitude, for example "Fe:2.2,Ni:0.6"'))
        self.map_label.installEventFilter(ToolTipFilter(self.map_label, 300, ToolTipPosition.TOP))
        self.map_edit = KeyValueTableInput(
            self.tr("Element"), self.tr("Moment magnitude"), self.setting_widget
        )

        self.default_label = BodyLabel(self.tr("Default |m|"), self.setting_widget)
        self.default_label.setToolTip(self.tr("Magnitude used for elements not listed in the magmom map"))
        self.default_label.installEventFilter(ToolTipFilter(self.default_label, 300, ToolTipPosition.TOP))
        self.default_frame = SpinBoxUnitInputFrame(self)
        self.default_frame.set_input("", 1, "float")
        self.default_frame.setRange(0.0, 20.0)
        self.default_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.default_frame.set_input_value([0.0])

        self.lift_scalar_checkbox = CheckBox(self.tr("Lift scalar magmoms to vectors"), self.setting_widget)
        self.lift_scalar_checkbox.setChecked(True)
        self.lift_scalar_checkbox.setToolTip(self.tr("If input magmoms are scalars, place them along Base axis before tilting"))
        self.lift_scalar_checkbox.installEventFilter(
            ToolTipFilter(self.lift_scalar_checkbox, 300, ToolTipPosition.TOP)
        )

        self.axis_label = BodyLabel(self.tr("Base axis"), self.setting_widget)
        self.axis_label.setToolTip(self.tr("Reference axis for lifted scalar magmoms and map/default ferromagnetic states"))
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setRange(-1.0, 1.0)
        for obj in self.axis_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.reference_label = BodyLabel(self.tr("Tilt reference"), self.setting_widget)
        self.reference_label.setToolTip(self.tr("Preferred direction that defines the tilt plane; it is orthogonalised automatically"))
        self.reference_label.installEventFilter(ToolTipFilter(self.reference_label, 300, ToolTipPosition.TOP))
        self.reference_frame = SpinBoxUnitInputFrame(self)
        self.reference_frame.set_input("", 3, "float")
        self.reference_frame.setRange(-1.0, 1.0)
        for obj in self.reference_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.reference_frame.set_input_value([1.0, 0.0, 0.0])

        self.apply_label = BodyLabel(self.tr("Apply elements"), self.setting_widget)
        self.apply_label.setToolTip(self.tr("Optional comma-separated element list; empty means all atoms are eligible targets"))
        self.apply_label.installEventFilter(ToolTipFilter(self.apply_label, 300, ToolTipPosition.TOP))
        self.apply_edit = LineEdit(self.setting_widget)
        self.apply_edit.setPlaceholderText(self.tr("Fe,Co,Ni"))

        self.max_output_label = BodyLabel(self.tr("Max outputs"), self.setting_widget)
        self.max_output_label.setToolTip(self.tr("Stop after this many generated structures"))
        self.max_output_label.installEventFilter(ToolTipFilter(self.max_output_label, 300, ToolTipPosition.TOP))
        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input("unit", 1, "int")
        self.max_output_frame.setRange(1, 999999)
        self.max_output_frame.set_input_value([100])

        self.settingLayout.addWidget(self.canting_mode_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.canting_mode_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.target_mode_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.target_mode_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.target_indices_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.target_indices_edit, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_source_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_source_combo, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_left_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_left_edit, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_right_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_right_edit, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_shell_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_shell_frame, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_tol_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_tol_frame, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_element_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_element_edit, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_group_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_group_edit, 9, 1, 1, 2)
        self.settingLayout.addWidget(self.bond_mode_label, 10, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_mode_combo, 10, 1, 1, 2)
        self.settingLayout.addWidget(self.bond_axis_label, 11, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_axis_frame, 11, 1, 1, 2)
        self.settingLayout.addWidget(self.bond_tol_label, 12, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_tol_frame, 12, 1, 1, 2)
        self.settingLayout.addWidget(self.group_a_label, 13, 0, 1, 1)
        self.settingLayout.addWidget(self.group_a_edit, 13, 1, 1, 2)
        self.settingLayout.addWidget(self.group_b_label, 14, 0, 1, 1)
        self.settingLayout.addWidget(self.group_b_edit, 14, 1, 1, 2)
        self.settingLayout.addWidget(self.angle_label, 15, 0, 1, 1)
        self.settingLayout.addWidget(self.angle_edit, 15, 1, 1, 2)
        self.settingLayout.addWidget(self.sign_label, 16, 0, 1, 1)
        self.settingLayout.addWidget(self.sign_combo, 16, 1, 1, 2)
        self.settingLayout.addWidget(self.include_reference_checkbox, 17, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_checkbox, 18, 0, 1, 3)
        self.settingLayout.addWidget(self.source_label, 19, 0, 1, 1)
        self.settingLayout.addWidget(self.source_combo, 19, 1, 1, 2)
        self.settingLayout.addWidget(self.map_label, 20, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 20, 1, 1, 2)
        self.settingLayout.addWidget(self.default_label, 21, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 21, 1, 1, 2)
        self.settingLayout.addWidget(self.lift_scalar_checkbox, 22, 0, 1, 3)
        self.settingLayout.addWidget(self.axis_label, 23, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_frame, 23, 1, 1, 2)
        self.settingLayout.addWidget(self.reference_label, 24, 0, 1, 1)
        self.settingLayout.addWidget(self.reference_frame, 24, 1, 1, 2)
        self.settingLayout.addWidget(self.apply_label, 25, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 25, 1, 1, 2)
        self.settingLayout.addWidget(self.max_output_label, 26, 0, 1, 1)
        self.settingLayout.addWidget(self.max_output_frame, 26, 1, 1, 2)

        self.canting_mode_combo.currentTextChanged.connect(self._update_canting_mode_widgets)
        self.pair_source_combo.currentTextChanged.connect(self._update_canting_mode_widgets)
        self.target_mode_combo.currentTextChanged.connect(self._update_target_widgets)
        self.source_combo.currentTextChanged.connect(self._update_magnitude_source_widgets)
        self.advanced_checkbox.toggled.connect(self._update_magnitude_source_widgets)
        self.bond_mode_combo.currentTextChanged.connect(self._update_canting_mode_widgets)
        self._update_canting_mode_widgets()
        self._update_target_widgets()
        self._update_magnitude_source_widgets()

    def _update_canting_mode_widgets(self):
        mode = combo_value(self.canting_mode_combo)
        single_mode = mode == "Single-spin tilt"
        pair_mode = mode == "Atom pair canting"
        group_mode = mode == "Group pair canting"
        auto_pair = pair_mode and combo_value(self.pair_source_combo) == "Auto by neighbor shell"
        manual_pair = pair_mode and not auto_pair

        self.target_mode_label.setVisible(single_mode)
        self.target_mode_combo.setVisible(single_mode)
        self.target_mode_label.setEnabled(single_mode)
        self.target_mode_combo.setEnabled(single_mode)

        explicit = mode == "Single-spin tilt" and combo_value(self.target_mode_combo) == "Explicit indices (1-based)"
        self.target_indices_label.setVisible(explicit)
        self.target_indices_edit.setVisible(explicit)
        self.target_indices_label.setEnabled(explicit)
        self.target_indices_edit.setEnabled(explicit)

        for widget in (self.pair_source_label, self.pair_source_combo):
            widget.setVisible(pair_mode)
            widget.setEnabled(pair_mode)

        for widget in (self.pair_left_label, self.pair_left_edit, self.pair_right_label, self.pair_right_edit):
            widget.setVisible(manual_pair)
            widget.setEnabled(manual_pair)

        for widget in (
            self.pair_shell_label,
            self.pair_shell_frame,
            self.pair_tol_label,
            self.pair_tol_frame,
            self.pair_element_label,
            self.pair_element_edit,
            self.pair_group_label,
            self.pair_group_edit,
            self.bond_mode_label,
            self.bond_mode_combo,
        ):
            widget.setVisible(auto_pair)
            widget.setEnabled(auto_pair)

        show_bond_detail = auto_pair and combo_value(self.bond_mode_combo) != "Any"
        for widget in (self.bond_axis_label, self.bond_axis_frame, self.bond_tol_label, self.bond_tol_frame):
            widget.setVisible(show_bond_detail)
            widget.setEnabled(show_bond_detail)

        for widget in (self.group_a_label, self.group_a_edit, self.group_b_label, self.group_b_edit):
            widget.setVisible(group_mode)
            widget.setEnabled(group_mode)

    def _update_target_widgets(self):
        explicit = (
            combo_value(self.canting_mode_combo) == "Single-spin tilt"
            and combo_value(self.target_mode_combo) == "Explicit indices (1-based)"
        )
        self.target_indices_label.setEnabled(explicit)
        self.target_indices_edit.setEnabled(explicit)
        self.target_indices_label.setVisible(explicit)
        self.target_indices_edit.setVisible(explicit)

    def _update_magnitude_source_widgets(self):
        show_advanced = self.advanced_checkbox.isChecked()
        for widget in (
            self.source_label,
            self.source_combo,
            self.lift_scalar_checkbox,
            self.axis_label,
            self.axis_frame,
            self.reference_label,
            self.reference_frame,
            self.apply_label,
            self.apply_edit,
            self.max_output_label,
            self.max_output_frame,
        ):
            widget.setVisible(show_advanced)
            widget.setEnabled(show_advanced)
        use_map = show_advanced and combo_value(self.source_combo) == "Map/default magnitude"
        self.map_label.setEnabled(use_map)
        self.map_edit.setEnabled(use_map)
        self.map_label.setVisible(use_map)
        self.map_edit.setVisible(use_map)
        self.default_label.setEnabled(use_map)
        self.default_frame.setEnabled(use_map)
        self.default_label.setVisible(use_map)
        self.default_frame.setVisible(use_map)

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def create_operation(self):
        return SmallAngleSpinTiltOperation()

    def get_params(self) -> SmallAngleSpinTiltParams:
        return SmallAngleSpinTiltParams(
            canting_mode=combo_value(self.canting_mode_combo),
            target_mode=combo_value(self.target_mode_combo),
            target_indices=self.target_indices_edit.text(),
            pair_left_indices=self.pair_left_edit.text(),
            pair_right_indices=self.pair_right_edit.text(),
            pair_source=combo_value(self.pair_source_combo),
            pair_shell=int(self.pair_shell_frame.get_input_value()[0]),
            pair_shell_tolerance=float(self.pair_tol_frame.get_input_value()[0]),
            pair_element_filter=self.pair_element_edit.text(),
            pair_group_filter=self.pair_group_edit.text(),
            bond_filter_mode=combo_value(self.bond_mode_combo),
            bond_filter_axis=self.bond_axis_frame.get_input_value(),
            bond_filter_tolerance=float(self.bond_tol_frame.get_input_value()[0]),
            group_a=self.group_a_edit.text(),
            group_b=self.group_b_edit.text(),
            angle_list=self.angle_edit.text(),
            tilt_signs=combo_value(self.sign_combo),
            include_reference=self.include_reference_checkbox.isChecked(),
            magnitude_source=combo_value(self.source_combo),
            magmom_map=self.map_edit.text(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            lift_scalar=self.lift_scalar_checkbox.isChecked(),
            axis=self.axis_frame.get_input_value(),
            reference_direction=self.reference_frame.get_input_value(),
            apply_elements=self.apply_edit.text(),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
        )

    def set_params(self, params: SmallAngleSpinTiltParams) -> None:
        set_combo_value(self.canting_mode_combo, params.canting_mode)
        set_combo_value(self.target_mode_combo, params.target_mode)
        self.target_indices_edit.setText(params.target_indices)
        self.pair_left_edit.setText(params.pair_left_indices)
        self.pair_right_edit.setText(params.pair_right_indices)
        set_combo_value(self.pair_source_combo, params.pair_source)
        self.pair_shell_frame.set_input_value([int(params.pair_shell)])
        self.pair_tol_frame.set_input_value([float(params.pair_shell_tolerance)])
        self.pair_element_edit.setText(params.pair_element_filter)
        self.pair_group_edit.setText(params.pair_group_filter)
        set_combo_value(self.bond_mode_combo, params.bond_filter_mode)
        self.bond_axis_frame.set_input_value([float(v) for v in params.bond_filter_axis])
        self.bond_tol_frame.set_input_value([float(params.bond_filter_tolerance)])
        self.group_a_edit.setText(params.group_a)
        self.group_b_edit.setText(params.group_b)
        self.angle_edit.setText(params.angle_list)
        set_combo_value(self.sign_combo, params.tilt_signs)
        self.include_reference_checkbox.setChecked(bool(params.include_reference))
        set_combo_value(self.source_combo, params.magnitude_source)
        self.map_edit.setText(params.magmom_map)
        self.default_frame.set_input_value([float(params.default_moment)])
        self.lift_scalar_checkbox.setChecked(bool(params.lift_scalar))
        self.axis_frame.set_input_value([float(v) for v in params.axis])
        self.reference_frame.set_input_value([float(v) for v in params.reference_direction])
        self.apply_edit.setText(params.apply_elements)
        self.max_output_frame.set_input_value([int(params.max_outputs)])
        self.advanced_checkbox.setChecked(
            params.magnitude_source != "Existing initial magmoms"
            or bool(params.magmom_map.strip())
            or float(params.default_moment) != 0.0
            or not bool(params.lift_scalar)
            or tuple(float(v) for v in params.axis) != (0.0, 0.0, 1.0)
            or tuple(float(v) for v in params.reference_direction) != (1.0, 0.0, 0.0)
            or bool(params.apply_elements.strip())
            or int(params.max_outputs) != 100
        )
        self._update_canting_mode_widgets()
        self._update_target_widgets()
        self._update_magnitude_source_widgets()

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = SmallAngleSpinTiltParams(**raw_params)
        else:
            params = SmallAngleSpinTiltParams(
                canting_mode=data_dict.get("canting_mode", "Single-spin tilt"),
                target_mode=data_dict.get("target_mode", "First eligible atom"),
                target_indices=data_dict.get("target_indices", ""),
                pair_left_indices=data_dict.get("pair_left_indices", ""),
                pair_right_indices=data_dict.get("pair_right_indices", ""),
                pair_source=data_dict.get("pair_source", "Manual indices"),
                pair_shell=data_dict.get("pair_shell", [1])[0],
                pair_shell_tolerance=data_dict.get("pair_shell_tolerance", [0.05])[0],
                pair_element_filter=data_dict.get("pair_element_filter", ""),
                pair_group_filter=data_dict.get("pair_group_filter", ""),
                bond_filter_mode=data_dict.get("bond_filter_mode", "Any"),
                bond_filter_axis=data_dict.get("bond_filter_axis", [0.0, 0.0, 1.0]),
                bond_filter_tolerance=data_dict.get("bond_filter_tolerance", [20.0])[0],
                group_a=data_dict.get("group_a", "A"),
                group_b=data_dict.get("group_b", "B"),
                angle_list=data_dict.get("angle_list", "1,2,5,10"),
                tilt_signs=data_dict.get("tilt_signs", "Positive only"),
                include_reference=data_dict.get("include_reference", True),
                magnitude_source=data_dict.get("magnitude_source", "Existing initial magmoms"),
                magmom_map=data_dict.get("magmom_map", ""),
                default_moment=data_dict.get("default_moment", [0.0])[0],
                lift_scalar=data_dict.get("lift_scalar", True),
                axis=data_dict.get("axis", [0.0, 0.0, 1.0]),
                reference_direction=data_dict.get("reference_direction", [1.0, 0.0, 0.0]),
                apply_elements=data_dict.get("apply_elements", ""),
                max_outputs=data_dict.get("max_outputs", [100])[0],
            )
        self.set_params(params)
