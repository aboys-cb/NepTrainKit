"""Card for generating lattice perturbations via stochastic scaling."""

from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import CellScalingOperation, CellScalingParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard


@CardManager.register_card
class CellScalingCard(MakeDataCard):
    """Generate perturbed lattice structures using stochastic scaling factors.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget owning the card controls.
    """

    group = "Lattice"
    card_name= "Lattice Perturb"
    menu_icon=r":/images/src/images/scaling.svg"
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Make Lattice Perturb")

        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("cell_scaling_card_widget")


        self.engine_label=BodyLabel("Random engine:",self.setting_widget)
        self.engine_type_combo=ComboBox(self.setting_widget)
        self.engine_type_combo.addItem("Sobol")
        self.engine_type_combo.addItem("Uniform")
        self.engine_type_combo.setCurrentIndex(1)
        self.engine_label.setToolTip("Select random engine")
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))


        self.scaling_condition_frame = SpinBoxUnitInputFrame(self)
        self.scaling_condition_frame.set_input("",1,"float")
        self.scaling_condition_frame.setRange(0,1)
        self.scaling_condition_frame.set_input_value([0.04])

        self.scaling_radio_label=BodyLabel("Max Scaling:",self.setting_widget)
        self.scaling_radio_label.setToolTip("Maximum scaling factor")

        self.scaling_radio_label.installEventFilter(ToolTipFilter(self.scaling_radio_label, 300, ToolTipPosition.TOP))

        self.optional_frame=QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0,0,0,0)
        self.optional_frame_layout.setSpacing(2)
        self.perturb_angle_checkbox=CheckBox( self.setting_widget)
        self.perturb_angle_checkbox.setText("Perturb angle")
        self.perturb_angle_checkbox.setChecked(True)
        self.perturb_angle_checkbox.setToolTip("Also perturb lattice angles")
        self.perturb_angle_checkbox.installEventFilter(ToolTipFilter(self.perturb_angle_checkbox, 300, ToolTipPosition.TOP))


        self.optional_label=BodyLabel("Optional",self.setting_widget)
        self.organic_checkbox=CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(False)
        self.organic_checkbox.setToolTip("Treat organic molecules as rigid units")
        self.organic_checkbox.installEventFilter(ToolTipFilter(self.organic_checkbox, 300, ToolTipPosition.TOP))

        self.optional_frame_layout.addWidget(self.perturb_angle_checkbox,0,0,1,1)
        self.optional_frame_layout.addWidget(self.organic_checkbox,1,0,1,1)

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit",1,"int")
        self.num_condition_frame.setRange(1,10000)
        self.num_label=BodyLabel("Structures",self.setting_widget)
        self.num_condition_frame.set_input_value([50])
        self.num_label.setToolTip("Number of scaled structures to generate")
        self.num_label.installEventFilter(ToolTipFilter(self.num_label, 300, ToolTipPosition.TOP))

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

        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)

        self.settingLayout.addWidget(self.optional_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame, 1, 1, 1,1)

        self.settingLayout.addWidget(self.scaling_radio_label, 2, 0, 1, 1)

        self.settingLayout.addWidget(self.scaling_condition_frame, 2, 1, 1,2)
        self.settingLayout.addWidget(self.num_label, 3, 0, 1, 1)

        self.settingLayout.addWidget(self.num_condition_frame, 3, 1, 1,2)
        self.settingLayout.addWidget(self.seed_checkbox, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 4, 1, 1, 2)

    def create_operation(self):
        """Return the UI-independent lattice scaling operation."""
        return CellScalingOperation()

    def get_params(self) -> CellScalingParams:
        """Read lattice scaling parameters from UI controls."""
        return CellScalingParams(
            engine_type=int(self.engine_type_combo.currentIndex()),
            max_scaling=float(self.scaling_condition_frame.get_input_value()[0]),
            max_num=int(self.num_condition_frame.get_input_value()[0]),
            perturb_angle=self.perturb_angle_checkbox.isChecked(),
            identify_organic=self.organic_checkbox.isChecked(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: CellScalingParams) -> None:
        """Apply lattice scaling parameters to UI controls."""
        self.engine_type_combo.setCurrentIndex(int(params.engine_type))
        self.perturb_angle_checkbox.setChecked(bool(params.perturb_angle))
        self.organic_checkbox.setChecked(bool(params.identify_organic))
        self.scaling_condition_frame.set_input_value([float(params.max_scaling)])
        self.num_condition_frame.set_input_value([int(params.max_num)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])

    def process_structure(self, structure):
        """Generate lattice perturbations from UI-independent parameters."""
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
            params = CellScalingParams(
                engine_type=raw_params.get("engine_type", 1),
                max_scaling=raw_params.get("max_scaling", 0.04),
                max_num=raw_params.get("max_num", 50),
                perturb_angle=raw_params.get("perturb_angle", True),
                identify_organic=raw_params.get("identify_organic", False),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
            )
        else:
            params = CellScalingParams(
                engine_type=data_dict.get("engine_type", 1),
                max_scaling=data_dict.get("scaling_condition", [0.04])[0],
                max_num=data_dict.get("num_condition", [50])[0],
                perturb_angle=data_dict.get("perturb_angle", True),
                identify_organic=data_dict.get("organic", False),
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
