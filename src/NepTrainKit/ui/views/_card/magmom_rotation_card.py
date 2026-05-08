"""Card for rotating magnetic moments of selected atoms."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CheckBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.magnetism import MagneticMomentRotationOperation, MagneticMomentRotationParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class MagneticMomentRotationCard(MakeDataCard):
    """Rotate and optionally rescale atomic magnetic moments for selected species."""

    group = "Magnetism"
    card_name = "Magmom Rotation"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Rotate Magnetic Moments")
        self.init_ui()

    def init_ui(self):
        """Build card controls for magnetic moment perturbations."""
        self.setObjectName("magmom_rotation_card_widget")

        self.elements_label = BodyLabel("Elements", self.setting_widget)
        self.elements_label.setToolTip("Comma separated element symbols; empty means all atoms")
        self.elements_label.installEventFilter(ToolTipFilter(self.elements_label, 300, ToolTipPosition.TOP))
        self.elements_input = LineEdit(self.setting_widget)
        self.elements_input.setPlaceholderText("Fe,Ni")


        self.angle_label = BodyLabel("Max rotation", self.setting_widget)
        self.angle_label.setToolTip("Upper bound for random rotation angles in degrees")
        self.angle_label.installEventFilter(ToolTipFilter(self.angle_label, 300, ToolTipPosition.TOP))
        self.angle_frame = SpinBoxUnitInputFrame(self)
        self.angle_frame.set_input("deg", 1, "float")
        self.angle_frame.setRange(-180, 180)
        self.angle_frame.object_list[0].setDecimals(2)  # pyright:ignore
        self.angle_frame.set_input_value([10.0])

        self.count_label = BodyLabel("Structures", self.setting_widget)
        self.count_label.setToolTip("Number of perturbed structures to generate")
        self.count_label.installEventFilter(ToolTipFilter(self.count_label, 300, ToolTipPosition.TOP))
        self.count_frame = SpinBoxUnitInputFrame(self)
        self.count_frame.set_input("unit", 1, "int")
        self.count_frame.setRange(1, 100)
        self.count_frame.set_input_value([5])

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip("Enable reproducible random sampling")
        self.seed_checkbox.installEventFilter(ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP))
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.lift_scalar_checkbox = CheckBox("Rotate scalar magmoms (lift to vector)", self.setting_widget)
        self.lift_scalar_checkbox.setChecked(True)
        self.lift_scalar_checkbox.setToolTip("If magmoms are scalars, treat them as vectors along Axis before rotating")
        self.lift_scalar_checkbox.installEventFilter(
            ToolTipFilter(self.lift_scalar_checkbox, 300, ToolTipPosition.TOP)
        )

        self.axis_label = BodyLabel("Axis (x,y,z)", self.setting_widget)
        self.axis_label.setToolTip("Axis used when lifting scalar magmoms to vectors")
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setRange(-1.0, 1.0)
        for obj in self.axis_frame.object_list:
            obj.setDecimals(6)  # pyright:ignore
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.magnitude_checkbox = CheckBox("Randomise magnitude", self.setting_widget)
        self.magnitude_checkbox.setChecked(True)
        self.magnitude_checkbox.setToolTip("Enable scaling of magnetic-moment magnitudes")
        self.magnitude_checkbox.installEventFilter(ToolTipFilter(self.magnitude_checkbox, 300, ToolTipPosition.TOP))
        self.magnitude_checkbox.stateChanged.connect(self._toggle_magnitude_inputs)

        self.min_factor_label = BodyLabel("magnitude scaling factor", self.setting_widget)
        self.min_factor_label.setToolTip("magnitude scaling factor")
        self.min_factor_label.installEventFilter(ToolTipFilter(self.min_factor_label, 300, ToolTipPosition.TOP))
        self.magnitude_factor_frame = SpinBoxUnitInputFrame(self)
        self.magnitude_factor_frame.set_input(["-", ""], 2, "float")
        self.magnitude_factor_frame.setRange(0, 10)
        self.magnitude_factor_frame.set_input_value([0.95,1.05])
        self.magnitude_factor_frame.object_list[0].setDecimals(3)  # pyright:ignore




        self.settingLayout.addWidget(self.elements_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.elements_input, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.angle_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.angle_frame, 1, 1, 1, 2)

        self.settingLayout.addWidget(self.lift_scalar_checkbox, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.axis_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_frame, 3, 1, 1, 2)

        self.settingLayout.addWidget(self.magnitude_checkbox, 4, 0, 1, 3)
        self.settingLayout.addWidget(self.min_factor_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.magnitude_factor_frame, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.count_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.count_frame, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 7, 1, 1, 2)

        self._toggle_magnitude_inputs(self.magnitude_checkbox.checkState())

    def _toggle_magnitude_inputs(self, state):
        enabled = bool(state)
        self.min_factor_label.setEnabled(enabled)
        self.magnitude_factor_frame.setEnabled(enabled)


    def create_operation(self):
        return MagneticMomentRotationOperation()

    def get_params(self) -> MagneticMomentRotationParams:
        return MagneticMomentRotationParams(
            elements=self.elements_input.text(),
            max_angle=float(self.angle_frame.get_input_value()[0]),
            num_structures=int(self.count_frame.get_input_value()[0]),
            lift_scalar=self.lift_scalar_checkbox.isChecked(),
            axis=self.axis_frame.get_input_value(),
            disturb_magnitude=self.magnitude_checkbox.isChecked(),
            magnitude_factor=self.magnitude_factor_frame.get_input_value(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: MagneticMomentRotationParams) -> None:
        self.elements_input.setText(params.elements)
        self.angle_frame.set_input_value([float(params.max_angle)])
        self.count_frame.set_input_value([int(params.num_structures)])
        self.lift_scalar_checkbox.setChecked(bool(params.lift_scalar))
        self.axis_frame.set_input_value([float(v) for v in params.axis])
        self.magnitude_checkbox.setChecked(bool(params.disturb_magnitude))
        self.magnitude_factor_frame.set_input_value([float(v) for v in params.magnitude_factor])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())
        self._toggle_magnitude_inputs(self.magnitude_checkbox.checkState())

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = MagneticMomentRotationParams(**raw_params)
        else:
            params = MagneticMomentRotationParams(
                elements=data_dict.get("elements", ""),
                max_angle=data_dict.get("max_angle", [10.0])[0],
                num_structures=data_dict.get("num_structures", [5])[0],
                lift_scalar=data_dict.get("lift_scalar", True),
                axis=data_dict.get("axis", [0.0, 0.0, 1.0]),
                disturb_magnitude=data_dict.get("disturb_magnitude", True),
                magnitude_factor=data_dict.get("magnitude_factor", [0.95, 1.05]),
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
