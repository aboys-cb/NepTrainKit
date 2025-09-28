"""Card for applying random atomic perturbations."""

from itertools import combinations

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.structure import get_clusters
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard
from scipy.stats.qmc import Sobol

@CardManager.register_card
class PerturbCard(MakeDataCard):
    """Apply random atomic displacements within a configurable distance budget.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Perturbation"
    card_name= "Atomic Perturb"
    menu_icon=r":/images/src/images/perturb.svg"
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Make Atomic Perturb")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("perturb_card_widget")
        self.engine_label=BodyLabel("Random engine:",self.setting_widget)
        self.engine_type_combo=ComboBox(self.setting_widget)
        self.engine_type_combo.addItem("Sobol")
        self.engine_type_combo.addItem("Uniform")
        self.engine_type_combo.setCurrentIndex(1)

        self.engine_label.setToolTip("Select random engine")
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))

        self.optional_frame = QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.optional_frame_layout.setSpacing(2)

        self.optional_label=BodyLabel("Optional",self.setting_widget)
        self.organic_checkbox=CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(False)
        self.optional_label.setToolTip("Treat organic molecules as rigid units")
        self.optional_label.installEventFilter(ToolTipFilter(self.optional_label, 300, ToolTipPosition.TOP))



        self.optional_frame_layout.addWidget(self.organic_checkbox,0,0,1,1)

        self.scaling_condition_frame = SpinBoxUnitInputFrame(self)
        self.scaling_condition_frame.set_input("Å",1,"float")
        self.scaling_condition_frame.setRange(0,1)
        self.scaling_radio_label=BodyLabel("Max distance:",self.setting_widget)
        self.scaling_condition_frame.set_input_value([0.3])
        self.scaling_radio_label.setToolTip("Maximum displacement distance")
        self.scaling_radio_label.installEventFilter(ToolTipFilter(self.scaling_radio_label, 300, ToolTipPosition.TOP))

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit",1,"int")
        self.num_condition_frame.setRange(1,10000)
        self.num_condition_frame.set_input_value([50])

        self.num_label=BodyLabel("Max num:",self.setting_widget)
        self.num_label.setToolTip("Number of structures to generate")

        self.num_label.installEventFilter(ToolTipFilter(self.num_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)

        self.settingLayout.addWidget(self.optional_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame,1, 1, 1, 2)

        self.settingLayout.addWidget(self.scaling_radio_label, 2, 0, 1, 1)

        self.settingLayout.addWidget(self.scaling_condition_frame, 2, 1, 1,2)

        self.settingLayout.addWidget(self.num_label,3, 0, 1, 1)

        self.settingLayout.addWidget(self.num_condition_frame,3, 1, 1,2)



    def process_structure(self, structure):
        """Apply random atomic displacements within the configured distance bounds.
        
        Parameters
        ----------
        structure : ase.Atoms
            Structure to perturb.
        
        Returns
        -------
        list[ase.Atoms]
            Structures containing perturbed atomic positions.
        """
        structure_list = []
        engine_type = self.engine_type_combo.currentIndex()
        max_scaling = self.scaling_condition_frame.get_input_value()[0]
        max_num = int(self.num_condition_frame.get_input_value()[0])
        identify_organic = self.organic_checkbox.isChecked()

        n_atoms = len(structure)
        dim = n_atoms * 3

        if engine_type == 0:
            sobol_engine = Sobol(d=dim, scramble=True)
            perturbation_factors = (sobol_engine.random(max_num) - 0.5) * 2
        else:
            perturbation_factors = np.random.uniform(-1, 1, (max_num, dim))

        if identify_organic:
            clusters, is_organic_list = get_clusters(structure)
            organic_clusters = [cluster for cluster, is_org in zip(clusters, is_organic_list) if is_org]
            inorganic_clusters = [cluster for cluster, is_org in zip(clusters, is_organic_list) if not is_org]

        orig_positions = structure.positions

        for i in range(max_num):
            delta = perturbation_factors[i].reshape(n_atoms, 3) * max_scaling
            new_positions = orig_positions + delta

            if identify_organic:
                new_positions = orig_positions.copy()


                for cluster in organic_clusters:  # pyright:ignore
                    cluster_delta = delta[cluster[0]]
                    new_positions[cluster] += cluster_delta

                for cluster in inorganic_clusters:  # pyright:ignore
                    new_positions[cluster] += delta[cluster]

            new_structure = structure.copy()
            new_structure.set_positions(new_positions)
            new_structure.wrap()
            config_str = f" Perturb(distance={max_scaling}, {'uniform' if engine_type == 1 else 'Sobol'})"
            new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + config_str
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


        data_dict['engine_type'] = self.engine_type_combo.currentIndex()
        data_dict["organic"]=self.organic_checkbox.isChecked()
        data_dict['scaling_condition'] = self.scaling_condition_frame.get_input_value()

        data_dict['num_condition'] = self.num_condition_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.
        
        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        super().from_dict(data_dict)

        self.engine_type_combo.setCurrentIndex(data_dict['engine_type'])

        self.scaling_condition_frame.set_input_value(data_dict['scaling_condition'])

        self.num_condition_frame.set_input_value(data_dict['num_condition'])
        self.organic_checkbox.setChecked(data_dict.get("organic", False))



