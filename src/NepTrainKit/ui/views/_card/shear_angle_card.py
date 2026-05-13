"""Card for perturbing lattice angles by configurable increments."""

from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import ShearAngleOperation, ShearAngleParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard

@CardManager.register_card
class ShearAngleCard(MakeDataCard):
    """Perturb lattice angles while preserving cell lengths.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Lattice"
    card_name = "Shear Angle Strain"
    menu_icon = r":/images/src/images/scaling.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Make Shear Angle Strain")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("shear_angle_card_widget")
        self.optional_frame = QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.optional_frame_layout.setSpacing(2)

        self.optional_label = BodyLabel("Optional", self.setting_widget)
        self.organic_checkbox = CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(False)
        self.optional_label.setToolTip("Treat organic molecules as rigid units")
        self.optional_label.installEventFilter(ToolTipFilter(self.optional_label, 300, ToolTipPosition.TOP))
        self.optional_frame_layout.addWidget(self.organic_checkbox, 0, 0, 1, 1)

        self.alpha_label = BodyLabel("Alpha:", self.setting_widget)
        self.alpha_frame = SpinBoxUnitInputFrame(self)
        self.alpha_frame.set_input(["-", "deg step:", "deg"], 3, "float")
        self.alpha_frame.setRange(-30, 30)
        self.alpha_frame.set_input_value([-2, 2, 1])
        self.alpha_label.setToolTip("Alpha angle adjustment range")
        self.alpha_label.installEventFilter(ToolTipFilter(self.alpha_label, 300, ToolTipPosition.TOP))

        self.beta_label = BodyLabel("Beta:", self.setting_widget)
        self.beta_frame = SpinBoxUnitInputFrame(self)
        self.beta_frame.set_input(["-", "deg step:", "deg"], 3, "float")
        self.beta_frame.setRange(-30, 30)
        self.beta_frame.set_input_value([-2, 2, 1])
        self.beta_label.setToolTip("Beta angle adjustment range")
        self.beta_label.installEventFilter(ToolTipFilter(self.beta_label, 300, ToolTipPosition.TOP))

        self.gamma_label = BodyLabel("Gamma:", self.setting_widget)
        self.gamma_frame = SpinBoxUnitInputFrame(self)
        self.gamma_frame.set_input(["-", "deg step:", "deg"], 3, "float")
        self.gamma_frame.setRange(-30, 30)
        self.gamma_frame.set_input_value([-2, 2, 1])
        self.gamma_label.setToolTip("Gamma angle adjustment range")
        self.gamma_label.installEventFilter(ToolTipFilter(self.gamma_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.optional_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.alpha_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.alpha_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.beta_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.beta_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.gamma_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.gamma_frame, 3, 1, 1, 2)

    def create_operation(self):
        """Return the UI-independent shear angle operation."""
        return ShearAngleOperation()

    def get_params(self) -> ShearAngleParams:
        """Read shear angle parameters from UI controls."""
        return ShearAngleParams(
            alpha_range=tuple(float(value) for value in self.alpha_frame.get_input_value()),
            beta_range=tuple(float(value) for value in self.beta_frame.get_input_value()),
            gamma_range=tuple(float(value) for value in self.gamma_frame.get_input_value()),
            identify_organic=self.organic_checkbox.isChecked(),
        )

    def set_params(self, params: ShearAngleParams) -> None:
        """Apply shear angle parameters to UI controls."""
        self.organic_checkbox.setChecked(bool(params.identify_organic))
        self.alpha_frame.set_input_value(list(params.alpha_range))
        self.beta_frame.set_input_value(list(params.beta_range))
        self.gamma_frame.set_input_value(list(params.gamma_range))

    def process_structure(self, structure):
        """Sweep cell angles from UI-independent parameters."""
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
            params = ShearAngleParams(
                alpha_range=tuple(raw_params.get("alpha_range", [-2.0, 2.0, 1.0])),
                beta_range=tuple(raw_params.get("beta_range", [-2.0, 2.0, 1.0])),
                gamma_range=tuple(raw_params.get("gamma_range", [-2.0, 2.0, 1.0])),
                identify_organic=raw_params.get("identify_organic", False),
            )
        else:
            params = ShearAngleParams(
                alpha_range=tuple(data_dict.get("alpha_range", [-2, 2, 1])),
                beta_range=tuple(data_dict.get("beta_range", [-2, 2, 1])),
                gamma_range=tuple(data_dict.get("gamma_range", [-2, 2, 1])),
                identify_organic=data_dict.get("organic", False),
            )
        self.set_params(params)

