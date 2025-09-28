"""Card for generating vacancy defects using stochastic sampling."""

import json
from itertools import combinations

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox, RadioButton

from NepTrainKit.core import CardManager
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame, VacancyRulesWidget
from NepTrainKit.ui.widgets import MakeDataCard
from scipy.stats.qmc import Sobol

@CardManager.register_card
class VacancyDefectCard(MakeDataCard):
    """Sample vacancy defects either by concentration or by explicit counts.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Defect"
    card_name= "Vacancy Defect Generation"
    menu_icon=r":/images/src/images/defect.svg"
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Make Vacancy Defect")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("vacancy_defect_card_widget")

        self.engine_label=BodyLabel("Random engine:",self.setting_widget)
        self.engine_label.setToolTip("Select random engine")
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))

        self.engine_type_combo=ComboBox(self.setting_widget)
        self.engine_type_combo.addItem("Sobol")
        self.engine_type_combo.addItem("Uniform")
        self.engine_type_combo.setCurrentIndex(1)

        self.num_radio_button = RadioButton("Vacancy num",self.setting_widget)
        self.num_radio_button.setChecked(True)
        self.num_radio_button.setToolTip("Set fixed number of vacancies")
        self.num_radio_button.installEventFilter(ToolTipFilter(self.num_radio_button, 300, ToolTipPosition.TOP))

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit",1)
        self.num_condition_frame.setRange(1,10000)


        self.concentration_radio_button = RadioButton("Vacancy concentration",self.setting_widget)
        self.concentration_radio_button.setToolTip("Set vacancy concentration")
        self.concentration_radio_button.installEventFilter(ToolTipFilter(self.concentration_radio_button, 300, ToolTipPosition.TOP))


        self.concentration_condition_frame = SpinBoxUnitInputFrame(self)
        self.concentration_condition_frame.set_input("",1,"float")
        self.concentration_condition_frame.setRange(0,1)


        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit",1)
        self.max_atoms_condition_frame.setRange(1,10000)

        self.max_atoms_label= BodyLabel("Max num",self.setting_widget)
        self.max_atoms_label.setToolTip("Number of structures to generate")

        self.max_atoms_label.installEventFilter(ToolTipFilter(self.max_atoms_label, 300, ToolTipPosition.TOP))

        #
        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)
        self.settingLayout.addWidget(self.num_radio_button, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.num_condition_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.concentration_radio_button, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.concentration_condition_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 3, 1, 1, 2)

    def process_structure(self,structure):
        """Create vacancy defect structures based on either fixed counts or concentrations.
        
        Parameters
        ----------
        structure : ase.Atoms
            Structure to modify.
        
        Returns
        -------
        list[ase.Atoms]
            Structures with randomly placed vacancies.
        """
        structure_list = []
        engine_type = self.engine_type_combo.currentIndex()
        concentration = self.concentration_condition_frame.get_input_value()[0]

        defect_num = int(self.num_condition_frame.get_input_value()[0])

        max_num = int(self.max_atoms_condition_frame.get_input_value()[0])

        n_atoms = len(structure)
        if self.concentration_radio_button.isChecked():
            max_defects = int(concentration * n_atoms)
        else:
            max_defects =  defect_num
        if max_defects ==n_atoms:
            max_defects=max_defects-1

        if engine_type == 0:
            sobol_engine = Sobol(d=n_atoms + 1, scramble=True)
            sobol_seq = sobol_engine.random(max_num)
        else:
            defect_counts = np.random.randint(1, max_defects + 1, max_num)

        for i in range(max_num):
            new_structure =structure.copy()

            if engine_type == 0:

                target_defects = 1 + int(sobol_seq[i, 0] * max_defects)  # pyright:ignore
                target_defects = int(min(target_defects, max_defects))
            else:
                target_defects = int(defect_counts[i])  # pyright:ignore

            if target_defects == 0:
                structure_list.append(new_structure)
                continue
            if engine_type == 0:
                position_scores = sobol_seq[i, 1:] # pyright:ignore

                sorted_indices = np.argsort(position_scores)
                defect_indices = sorted_indices[:target_defects]
            else:

                defect_indices = np.random.choice(n_atoms, target_defects, replace=False)
            mask = np.zeros(n_atoms, dtype=bool)
            mask[defect_indices] = True
            n_vacancies = np.sum(mask)
            del new_structure[mask]
            new_structure.info["Config_type"] = new_structure.info.get("Config_type","") + f" Vacancy(num={n_vacancies})"
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
        data_dict['num_condition'] = self.num_condition_frame.get_input_value()
        data_dict["num_radio_button"]=self.num_radio_button.isChecked()
        data_dict["concentration_radio_button"]=self.concentration_radio_button.isChecked()
        data_dict['concentration_condition'] = self.concentration_condition_frame.get_input_value()
        data_dict['max_atoms_condition'] = self.max_atoms_condition_frame.get_input_value()

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
        self.num_condition_frame.set_input_value(data_dict['num_condition'])
        self.concentration_condition_frame.set_input_value(data_dict['concentration_condition'])
        self.max_atoms_condition_frame.set_input_value(data_dict['max_atoms_condition'])
        self.concentration_radio_button.setChecked(data_dict['concentration_radio_button'])
        self.num_radio_button.setChecked(data_dict['num_radio_button'])

