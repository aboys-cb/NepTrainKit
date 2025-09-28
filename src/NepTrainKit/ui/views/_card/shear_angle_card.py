"""Card for perturbing lattice angles by configurable increments."""

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ToolTipFilter, ToolTipPosition, CheckBox
from ase.geometry import cell_to_cellpar, cellpar_to_cell

from NepTrainKit.core import CardManager
from NepTrainKit.core.structure import get_clusters,process_organic_clusters
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

    def process_structure(self, structure):
        """Sweep the cell angles within the configured ranges while preserving lengths.
        
        Parameters
        ----------
        structure : ase.Atoms
            Structure whose lattice angles will be perturbed.
        
        Returns
        -------
        list[ase.Atoms]
            Structures containing the angle variations.
        """
        structure_list = []
        alpha = self.alpha_frame.get_input_value()
        beta = self.beta_frame.get_input_value()
        gamma = self.gamma_frame.get_input_value()
        identify_organic = self.organic_checkbox.isChecked()

        if identify_organic:
            clusters, is_organic_list = get_clusters(structure)

        alpha_range = np.arange(alpha[0], alpha[1] + 0.001, alpha[2])
        beta_range = np.arange(beta[0], beta[1] + 0.001, beta[2])
        gamma_range = np.arange(gamma[0], gamma[1] + 0.001, gamma[2])
        cell = structure.get_cell()
        cellpar = cell_to_cellpar(cell)  # [a,b,c,alpha,beta,gamma] in degrees
        lengths = cellpar[:3]
        angles0 = cellpar[3:]

        for da in alpha_range:
            for db in beta_range:
                for dg in gamma_range:
                    new_structure = structure.copy()
                    new_angles = angles0 + np.array([da, db, dg])
                    new_cellpar = [*lengths, *new_angles]
                    new_lattice = cellpar_to_cell(new_cellpar)
                    new_structure.set_cell(new_lattice, scale_atoms=True)
                    if identify_organic:
                        process_organic_clusters(structure, new_structure, clusters, is_organic_list)  # pyright:ignore
                    info_list = []
                    if abs(da) > 1e-8:
                        info_list.append(f"alpha:{da}deg")
                    if abs(db) > 1e-8:
                        info_list.append(f"beta:{db}deg")
                    if abs(dg) > 1e-8:
                        info_list.append(f"gamma:{dg}deg")
                    info_str = "|".join(info_list)
                    new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + f" ShearAngle({info_str})"
                    structure_list.append(new_structure)
        return structure_list

    def to_dict(self):
        """Serialize the current configuration to a plain dictionary.
        
        Returns
        -------
        dict
            Dictionary that can be fed into ``from_dict`` to rebuild the state.
        """
        data_dict = super().to_dict()
        data_dict["organic"] = self.organic_checkbox.isChecked()
        data_dict["alpha_range"] = self.alpha_frame.get_input_value()
        data_dict["beta_range"] = self.beta_frame.get_input_value()
        data_dict["gamma_range"] = self.gamma_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.
        
        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        super().from_dict(data_dict)
        self.organic_checkbox.setChecked(data_dict.get("organic", False))
        self.alpha_frame.set_input_value(data_dict.get("alpha_range", [-2, 2, 1]))
        self.beta_frame.set_input_value(data_dict.get("beta_range", [-2, 2, 1]))
        self.gamma_frame.set_input_value(data_dict.get("gamma_range", [-2, 2, 1]))

