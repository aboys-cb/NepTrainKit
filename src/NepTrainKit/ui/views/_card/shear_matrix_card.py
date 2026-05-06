"""Card for applying shear matrices to lattice vectors."""

from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import ShearMatrixOperation, ShearMatrixParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard

@CardManager.register_card
class ShearMatrixCard(MakeDataCard):
    """Apply shear matrices along the principal lattice planes.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Lattice"
    card_name = "Shear Matrix Strain"
    menu_icon = r":/images/src/images/scaling.svg"

    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Make Shear Matrix Strain")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("shear_strain_card_widget")
        self.optional_frame = QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.optional_frame_layout.setSpacing(2)

        self.optional_label = BodyLabel("Optional", self.setting_widget)
        self.organic_checkbox = CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(False)
        self.symmetric_checkbox = CheckBox("Symmetric shear", self.setting_widget)
        self.symmetric_checkbox.setChecked(True)
        self.symmetric_checkbox.setToolTip("Apply shear symmetrically")
        self.symmetric_checkbox.installEventFilter(ToolTipFilter(self.symmetric_checkbox, 300, ToolTipPosition.TOP))
        self.optional_label.setToolTip("Treat organic molecules as rigid units")
        self.optional_label.installEventFilter(ToolTipFilter(self.optional_label, 300, ToolTipPosition.TOP))
        self.optional_frame_layout.addWidget(self.organic_checkbox, 0, 0, 1, 1)
        self.optional_frame_layout.addWidget(self.symmetric_checkbox, 1, 0, 1, 1)

        self.xy_label = BodyLabel("XY:", self.setting_widget)
        self.xy_frame = SpinBoxUnitInputFrame(self)
        self.xy_frame.set_input(["-", "% step:", "%"], 3, "float")
        self.xy_frame.setRange(-100, 100)
        self.xy_frame.set_input_value([-5, 5, 1])
        self.xy_label.setToolTip("XY shear strain range")
        self.xy_label.installEventFilter(ToolTipFilter(self.xy_label, 300, ToolTipPosition.TOP))

        self.yz_label = BodyLabel("YZ:", self.setting_widget)
        self.yz_frame = SpinBoxUnitInputFrame(self)
        self.yz_frame.set_input(["-", "% step:", "%"], 3, "float")
        self.yz_frame.setRange(-100, 100)
        self.yz_frame.set_input_value([-5, 5, 1])
        self.yz_label.setToolTip("YZ shear strain range")
        self.yz_label.installEventFilter(ToolTipFilter(self.yz_label, 300, ToolTipPosition.TOP))

        self.xz_label = BodyLabel("XZ:", self.setting_widget)
        self.xz_frame = SpinBoxUnitInputFrame(self)
        self.xz_frame.set_input(["-", "% step:", "%"], 3, "float")
        self.xz_frame.setRange(-100, 100)
        self.xz_frame.set_input_value([-5, 5, 1])
        self.xz_label.setToolTip("XZ shear strain range")
        self.xz_label.installEventFilter(ToolTipFilter(self.xz_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.optional_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.xy_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.xy_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.yz_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.yz_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.xz_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.xz_frame, 3, 1, 1, 2)

    def create_operation(self):
        """Return the UI-independent shear matrix operation."""
        return ShearMatrixOperation()

    def get_params(self) -> ShearMatrixParams:
        """Read shear matrix parameters from UI controls."""
        return ShearMatrixParams(
            xy_range=tuple(float(value) for value in self.xy_frame.get_input_value()),
            yz_range=tuple(float(value) for value in self.yz_frame.get_input_value()),
            xz_range=tuple(float(value) for value in self.xz_frame.get_input_value()),
            symmetric=self.symmetric_checkbox.isChecked(),
            identify_organic=self.organic_checkbox.isChecked(),
        )

    def set_params(self, params: ShearMatrixParams) -> None:
        """Apply shear matrix parameters to UI controls."""
        self.organic_checkbox.setChecked(bool(params.identify_organic))
        self.symmetric_checkbox.setChecked(bool(params.symmetric))
        self.xy_frame.set_input_value(list(params.xy_range))
        self.yz_frame.set_input_value(list(params.yz_range))
        self.xz_frame.set_input_value(list(params.xz_range))

    def process_structure(self, structure):
        """Apply shear matrices from UI-independent parameters."""
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        """Serialize the current configuration to a plain dictionary.
        
        Returns
        -------
        dict
            Dictionary that can be fed into ``from_dict`` to rebuild the state.
        """
        data_dict = super().to_dict()
        params = self.get_params()
        data_dict["params"] = params_to_dict(params)
        data_dict["organic"] = params.identify_organic
        data_dict["symmetric"] = params.symmetric
        data_dict["xy_range"] = list(params.xy_range)
        data_dict["yz_range"] = list(params.yz_range)
        data_dict["xz_range"] = list(params.xz_range)
        return data_dict

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
            params = ShearMatrixParams(
                xy_range=tuple(raw_params.get("xy_range", [-5.0, 5.0, 1.0])),
                yz_range=tuple(raw_params.get("yz_range", [-5.0, 5.0, 1.0])),
                xz_range=tuple(raw_params.get("xz_range", [-5.0, 5.0, 1.0])),
                symmetric=raw_params.get("symmetric", True),
                identify_organic=raw_params.get("identify_organic", False),
            )
        else:
            params = ShearMatrixParams(
                xy_range=tuple(data_dict.get("xy_range", [-5, 5, 1])),
                yz_range=tuple(data_dict.get("yz_range", [-5, 5, 1])),
                xz_range=tuple(data_dict.get("xz_range", [-5, 5, 1])),
                symmetric=data_dict.get("symmetric", True),
                identify_organic=data_dict.get("organic", False),
            )
        self.set_params(params)
