"""Card for generating vacancy configurations according to rules."""

import json
from itertools import combinations

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame, VacancyRulesWidget
from NepTrainKit.ui.widgets import MakeDataCard
from scipy.stats.qmc import Sobol

@CardManager.register_card
class RandomVacancyCard(MakeDataCard):
    """Create vacancy structures by probabilistically removing atoms according to rules.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Defect"

    card_name = "Random Vacancy"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Random Vacancy Delete")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("random_vacancy_card_widget")

        self.rules_label = BodyLabel("Rules", self.setting_widget)
        self.rules_widget = VacancyRulesWidget(self.setting_widget)
        self.rules_label.setToolTip("vacancy rules")
        self.rules_label.installEventFilter(ToolTipFilter(self.rules_label, 300, ToolTipPosition.TOP))

        self.max_atoms_label = BodyLabel("Max structures", self.setting_widget)
        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit", 1)
        self.max_atoms_condition_frame.setRange(1, 10000)
        self.max_atoms_label.setToolTip("Number of structures to generate")
        self.max_atoms_label.installEventFilter(ToolTipFilter(self.max_atoms_label, 300, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.rules_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.rules_widget, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 1, 1, 1, 2)

    def process_structure(self, structure):
        """Create vacancy configurations by removing atoms that match the configured rules.
        
        Parameters
        ----------
        structure : ase.Atoms
            Structure to modify.
        
        Returns
        -------
        list[ase.Atoms]
            Structures with vacancies applied according to the rule set.
        """
        structure_list = []

        rules = self.rules_widget.to_rules()
        if not isinstance(rules, list) or not rules:
            return [structure]

        max_num = int(self.max_atoms_condition_frame.get_input_value()[0])
        for _ in range(max_num):
            new_structure = structure.copy()
            total_remove = 0
            for rule in rules:
                element = rule.get("element")
                count_min, count_max = rule.get("count", [0, 0])
                if not element or int(count_max) <= 0:
                    continue

                groups = rule.get("group")
                if groups and "group" in new_structure.arrays:
                    candidate_indices = [i for i, elem, g in zip(range(len(new_structure)), new_structure, new_structure.arrays["group"]) if elem.symbol == element and g in groups]
                else:
                    candidate_indices = [i for i, a in enumerate(new_structure) if a.symbol == element]

                if not candidate_indices:
                    continue

                remove_num = np.random.randint(int(count_min), int(count_max) + 1)
                remove_num = min(remove_num, len(candidate_indices))
                if remove_num <= 0:
                    continue

                idxs = np.random.choice(candidate_indices, remove_num, replace=False)
                for idx in sorted(idxs, reverse=True):
                    del new_structure[idx]
                total_remove += remove_num

            if total_remove:
                new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + f" Vacancy(num={total_remove})"

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

        data_dict['rules'] = json.dumps(self.rules_widget.to_rules(), ensure_ascii=False)
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

        rules = data_dict.get('rules', '')
        if isinstance(rules, str):
            try:
                rules = json.loads(rules)
            except Exception:
                rules = []
        self.rules_widget.from_rules(rules)
        self.max_atoms_condition_frame.set_input_value(data_dict.get('max_atoms_condition', [1]))


