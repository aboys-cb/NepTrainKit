"""Card for applying axial strain variations to lattice vectors."""

from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import CellStrainOperation, CellStrainParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard
@CardManager.register_card

class CellStrainCard(MakeDataCard):
    """Produce strained lattice variants along user-selected axes and ranges.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget owning the card controls.
    """

    group = "Lattice"

    card_name= "Lattice Strain"
    menu_icon=r":/images/src/images/scaling.svg"
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Make Cell Strain")

        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("cell_strain_card_widget")


        self.engine_label=BodyLabel("Axes:",self.setting_widget)
        self.engine_type_combo=EditableComboBox(self.setting_widget)
        axes_type=["uniaxial","biaxial","triaxial","isotropic"]
        self.engine_type_combo.addItems(axes_type)
        self.engine_label.setToolTip('Pull down to select or enter a specific axis, such as X or XY')
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))

        self.optional_frame=QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0,0,0,0)
        self.optional_frame_layout.setSpacing(2)

        self.optional_label=BodyLabel("Optional",self.setting_widget)
        self.organic_checkbox=CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(False)
        self.organic_checkbox.setToolTip("Treat organic molecules as rigid units")
        self.organic_checkbox.installEventFilter(ToolTipFilter(self.organic_checkbox, 300, ToolTipPosition.TOP))


        self.optional_frame_layout.addWidget(self.organic_checkbox,0,0,1,1)

        self.strain_x_label=BodyLabel("X:",self.setting_widget)
        self.strain_x_frame = SpinBoxUnitInputFrame(self)
        self.strain_x_frame.set_input(["-","% step:","%"],3,"float")
        self.strain_x_frame.setRange(-100,100)

        self.strain_x_frame.set_input_value([-5,5,1])
        self.strain_x_label.setToolTip("X-axis strain range")
        self.strain_x_label.installEventFilter(ToolTipFilter(self.strain_x_label, 300, ToolTipPosition.TOP))

        self.strain_y_label=BodyLabel("Y:",self.setting_widget)
        self.strain_y_frame = SpinBoxUnitInputFrame(self)
        self.strain_y_frame.set_input(["-","% step:","%"],3,"float")
        self.strain_y_frame.setRange(-100,100)
        self.strain_y_frame.set_input_value([-5,5,1])
        self.strain_y_label.setToolTip("Y-axis strain range")
        self.strain_y_label.installEventFilter(ToolTipFilter(self.strain_y_label, 300, ToolTipPosition.TOP))

        self.strain_z_label=BodyLabel("Z:",self.setting_widget)
        self.strain_z_frame = SpinBoxUnitInputFrame(self)
        self.strain_z_frame.set_input(["-","% step:","%"],3,"float")
        self.strain_z_frame.setRange(-100,100)
        self.strain_z_frame.set_input_value([-5,5,1])
        self.strain_z_label.setToolTip("Z-axis strain range")
        self.strain_z_label.installEventFilter(ToolTipFilter(self.strain_z_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)
        self.settingLayout.addWidget(self.optional_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame, 1, 1, 1,1)
        self.settingLayout.addWidget(self.strain_x_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.strain_x_frame, 2, 1, 1,1)
        self.settingLayout.addWidget(self.strain_y_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.strain_y_frame, 3, 1, 1,1)
        self.settingLayout.addWidget(self.strain_z_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.strain_z_frame, 4, 1, 1,1)

    def create_operation(self):
        """Return the UI-independent strain operation."""
        return CellStrainOperation()

    def get_params(self) -> CellStrainParams:
        """Read strain parameters from the UI controls."""
        return CellStrainParams(
            axes=self.engine_type_combo.currentText(),
            x_range=tuple(float(value) for value in self.strain_x_frame.get_input_value()),
            y_range=tuple(float(value) for value in self.strain_y_frame.get_input_value()),
            z_range=tuple(float(value) for value in self.strain_z_frame.get_input_value()),
            identify_organic=self.organic_checkbox.isChecked(),
        )

    def set_params(self, params: CellStrainParams) -> None:
        """Apply strain parameters to the UI controls."""
        self.organic_checkbox.setChecked(bool(params.identify_organic))
        self.engine_type_combo.setText(params.axes)
        self.strain_x_frame.set_input_value(list(params.x_range))
        self.strain_y_frame.set_input_value(list(params.y_range))
        self.strain_z_frame.set_input_value(list(params.z_range))

    def process_structure(self, structure):
        """Generate strained lattices from UI-independent parameters."""
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
            params = CellStrainParams(
                axes=raw_params.get("axes", "uniaxial"),
                x_range=tuple(raw_params.get("x_range", [-5.0, 5.0, 1.0])),
                y_range=tuple(raw_params.get("y_range", [-5.0, 5.0, 1.0])),
                z_range=tuple(raw_params.get("z_range", [-5.0, 5.0, 1.0])),
                identify_organic=raw_params.get("identify_organic", False),
            )
        else:
            params = CellStrainParams(
                axes=data_dict.get("engine_type", "uniaxial"),
                x_range=tuple(data_dict.get("x_range", [-5.0, 5.0, 1.0])),
                y_range=tuple(data_dict.get("y_range", [-5.0, 5.0, 1.0])),
                z_range=tuple(data_dict.get("z_range", [-5.0, 5.0, 1.0])),
                identify_organic=data_dict.get("organic", False),
            )
        self.set_params(params)






