"""Card for building supercells using several expansion strategies."""

from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, RadioButton

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import SuperCellOperation, SuperCellParams
from NepTrainKit.core.cards.operation import params_to_dict

from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard

@CardManager.register_card
class SuperCellCard(MakeDataCard):
    """Create supercells based on fixed scale factors, target lattice lengths, or atom limits.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Lattice"
    card_name= "Super Cell"
    menu_icon=r":/images/src/images/supercell.svg"
    separator = False
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Make Supercell")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("super_cell_card_widget")
        self.behavior_type_combo=ComboBox(self.setting_widget)
        self.behavior_type_combo.addItem("Maximum")
        self.behavior_type_combo.addItem("Iteration")
        self.behavior_type_combo.addItem("Minimum")

        self.combo_label=BodyLabel("Behavior:",self.setting_widget)
        self.combo_label.setToolTip("Select supercell generation method")
        self.combo_label.installEventFilter(ToolTipFilter(self.combo_label, 300, ToolTipPosition.TOP))

        self.super_scale_radio_button = RadioButton("Super scale",self.setting_widget)
        self.super_scale_radio_button.setChecked(True)
        self.super_scale_condition_frame = SpinBoxUnitInputFrame(self)
        self.super_scale_condition_frame.set_input("",3)
        self.super_scale_condition_frame.setRange(1,999)
        self.super_scale_condition_frame.set_input_value([3,3,3])
        self.super_scale_radio_button.setToolTip("Scale factors along axes")
        self.super_scale_radio_button.installEventFilter(ToolTipFilter(self.super_scale_radio_button, 300, ToolTipPosition.TOP))

        self.super_cell_radio_button = RadioButton("Super cell",self.setting_widget)
        self.super_cell_condition_frame = SpinBoxUnitInputFrame(self)
        self.super_cell_condition_frame.set_input("Å",3)
        self.super_cell_condition_frame.setRange(1,9999)
        self.super_cell_condition_frame.set_input_value([20,20,20])

        self.super_cell_radio_button.setToolTip("Target lattice constant in Å")
        self.super_cell_radio_button.installEventFilter(ToolTipFilter(self.super_cell_radio_button, 300, ToolTipPosition.TOP))


        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit",1)
        self.max_atoms_condition_frame.setRange(1,10000)
        # self.max_atoms_condition_frame.setToolTip("Maximum allowed atoms")
        self.max_atoms_condition_frame.set_input_value([100])

        self.max_atoms_radio_button = RadioButton("Max atoms",self.setting_widget)
        self.max_atoms_radio_button.setToolTip("Limit cell size by atom count")
        self.max_atoms_radio_button.installEventFilter(ToolTipFilter(self.max_atoms_radio_button, 300, ToolTipPosition.TOP))

        self.fixed_axis_label = BodyLabel("Fixed axes:", self.setting_widget)
        self.fixed_axis_label.setToolTip("Keep selected axes at a fixed multiplier in all supercell modes")
        self.fixed_axis_label.installEventFilter(ToolTipFilter(self.fixed_axis_label, 300, ToolTipPosition.TOP))

        self.fixed_axis_frame = QFrame(self.setting_widget)
        self.fixed_axis_layout = QGridLayout(self.fixed_axis_frame)
        self.fixed_axis_layout.setContentsMargins(0, 0, 0, 0)
        self.fixed_axis_layout.setHorizontalSpacing(12)

        self.fixed_axis_a_checkbox = CheckBox("a", self.fixed_axis_frame)
        self.fixed_axis_b_checkbox = CheckBox("b", self.fixed_axis_frame)
        self.fixed_axis_c_checkbox = CheckBox("c", self.fixed_axis_frame)

        self.fixed_axis_a_checkbox.setToolTip("Lock the a-axis multiplier to the fixed scale below")
        self.fixed_axis_b_checkbox.setToolTip("Lock the b-axis multiplier to the fixed scale below")
        self.fixed_axis_c_checkbox.setToolTip("Lock the c-axis multiplier to the fixed scale below")

        self.fixed_axis_layout.addWidget(self.fixed_axis_a_checkbox, 0, 0)
        self.fixed_axis_layout.addWidget(self.fixed_axis_b_checkbox, 0, 1)
        self.fixed_axis_layout.addWidget(self.fixed_axis_c_checkbox, 0, 2)

        self.fixed_scale_label = BodyLabel("Fixed scale:", self.setting_widget)
        self.fixed_scale_label.setToolTip("Multipliers used for the locked a/b/c axes")
        self.fixed_scale_label.installEventFilter(ToolTipFilter(self.fixed_scale_label, 300, ToolTipPosition.TOP))

        self.fixed_scale_condition_frame = SpinBoxUnitInputFrame(self)
        self.fixed_scale_condition_frame.set_input("x", 3)
        self.fixed_scale_condition_frame.setRange(1, 999)
        self.fixed_scale_condition_frame.set_input_value([1, 1, 1])


        self.settingLayout.addWidget(self.combo_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.behavior_type_combo,0, 1, 1, 2)
        self.settingLayout.addWidget(self.super_scale_radio_button, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.super_scale_condition_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.super_cell_radio_button, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.super_cell_condition_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_radio_button, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.fixed_axis_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.fixed_axis_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.fixed_scale_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.fixed_scale_condition_frame, 5, 1, 1, 2)

    def _get_fixed_axis_flags(self) -> tuple[bool, bool, bool]:
        """Return the axis-lock toggles for a, b, and c."""
        return (
            self.fixed_axis_a_checkbox.isChecked(),
            self.fixed_axis_b_checkbox.isChecked(),
            self.fixed_axis_c_checkbox.isChecked(),
        )

    def _get_fixed_axis_values(self) -> tuple[int, int, int]:
        """Return the configured fixed multipliers for a, b, and c."""
        na, nb, nc = self.fixed_scale_condition_frame.get_input_value()
        return (int(na), int(nb), int(nc))

    def create_operation(self):
        """Return the UI-independent supercell operation."""
        return SuperCellOperation()

    def get_params(self) -> SuperCellParams:
        """Read supercell parameters from UI controls."""
        if self.super_cell_radio_button.isChecked():
            mode = "cell"
        elif self.max_atoms_radio_button.isChecked():
            mode = "max_atoms"
        else:
            mode = "scale"

        return SuperCellParams(
            behavior_type=int(self.behavior_type_combo.currentIndex()),
            mode=mode,
            super_scale=tuple(int(value) for value in self.super_scale_condition_frame.get_input_value()),
            target_cell=tuple(float(value) for value in self.super_cell_condition_frame.get_input_value()),
            max_atoms=int(self.max_atoms_condition_frame.get_input_value()[0]),
            fixed_axis_flags=self._get_fixed_axis_flags(),
            fixed_axis_scale=self._get_fixed_axis_values(),
        )

    def set_params(self, params: SuperCellParams) -> None:
        """Apply supercell parameters to UI controls."""
        self.behavior_type_combo.setCurrentIndex(int(params.behavior_type))
        self.super_scale_radio_button.setChecked(params.mode == "scale")
        self.super_cell_radio_button.setChecked(params.mode == "cell")
        self.max_atoms_radio_button.setChecked(params.mode == "max_atoms")
        self.super_scale_condition_frame.set_input_value(list(params.super_scale))
        self.super_cell_condition_frame.set_input_value(list(params.target_cell))
        self.max_atoms_condition_frame.set_input_value([int(params.max_atoms)])
        self.fixed_axis_a_checkbox.setChecked(bool(params.fixed_axis_flags[0]))
        self.fixed_axis_b_checkbox.setChecked(bool(params.fixed_axis_flags[1]))
        self.fixed_axis_c_checkbox.setChecked(bool(params.fixed_axis_flags[2]))
        self.fixed_scale_condition_frame.set_input_value(list(params.fixed_axis_scale))

    def process_structure(self, structure):
        """Generate supercells from UI-independent parameters."""
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.
        
        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        super().from_dict(data_dict)

        raw_params = data_dict.get("params")
        if raw_params:
            params = SuperCellParams(
                behavior_type=raw_params.get("behavior_type", 0),
                mode=raw_params.get("mode", "scale"),
                super_scale=tuple(raw_params.get("super_scale", [3, 3, 3])),
                target_cell=tuple(raw_params.get("target_cell", [20.0, 20.0, 20.0])),
                max_atoms=raw_params.get("max_atoms", 100),
                fixed_axis_flags=tuple(raw_params.get("fixed_axis_flags", [False, False, False])),
                fixed_axis_scale=tuple(raw_params.get("fixed_axis_scale", [1, 1, 1])),
            )
        else:
            if data_dict.get('super_cell_radio_button', False):
                mode = "cell"
            elif data_dict.get('max_atoms_radio_button', False):
                mode = "max_atoms"
            else:
                mode = "scale"
            params = SuperCellParams(
                behavior_type=data_dict.get('super_cell_type', 0),
                mode=mode,
                super_scale=tuple(data_dict.get('super_scale_condition', [3, 3, 3])),
                target_cell=tuple(data_dict.get('super_cell_condition', [20, 20, 20])),
                max_atoms=data_dict.get('max_atoms_condition', [100])[0],
                fixed_axis_flags=tuple(data_dict.get('fixed_axis_flags', [False, False, False])),
                fixed_axis_scale=tuple(data_dict.get('fixed_axis_scale', [1, 1, 1])),
            )
        self.set_params(params)
