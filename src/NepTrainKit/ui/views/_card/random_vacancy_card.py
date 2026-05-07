"""Card for generating vacancy configurations according to rules."""

import json

from qfluentwidgets import BodyLabel, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import RandomVacancyOperation, RandomVacancyParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame, VacancyRulesWidget
from NepTrainKit.ui.widgets import MakeDataCard

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

        self.max_atoms_label = BodyLabel("Structures", self.setting_widget)
        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit", 1)
        self.max_atoms_condition_frame.setRange(1, 10000)
        self.max_atoms_label.setToolTip("Number of vacancy structures to generate")
        self.max_atoms_label.installEventFilter(ToolTipFilter(self.max_atoms_label, 300, ToolTipPosition.TOP))

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

        self.settingLayout.addWidget(self.rules_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.rules_widget, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 2, 1, 1, 2)

    def create_operation(self):
        """Return the UI-independent random vacancy operation."""
        return RandomVacancyOperation()

    def get_params(self) -> RandomVacancyParams:
        """Read random vacancy parameters from UI controls."""
        return RandomVacancyParams(
            rules=self.rules_widget.to_rules(),
            max_structures=int(self.max_atoms_condition_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: RandomVacancyParams) -> None:
        """Apply random vacancy parameters to UI controls."""
        self.rules_widget.from_rules(params.rules)
        self.max_atoms_condition_frame.set_input_value([int(params.max_structures)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])

    def process_structure(self, structure):
        """Create vacancy configurations from UI-independent parameters.
        
        Parameters
        ----------
        structure : ase.Atoms
            Structure to modify.
        
        Returns
        -------
        list[ase.Atoms]
            Structures with vacancies applied according to the rule set.
        """
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
            params = RandomVacancyParams(
                rules=raw_params.get("rules", []),
                max_structures=raw_params.get("max_structures", 1),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
            )
        else:
            rules = data_dict.get("rules", "")
            if isinstance(rules, str):
                try:
                    rules = json.loads(rules)
                except Exception:
                    rules = []
            params = RandomVacancyParams(
                rules=rules,
                max_structures=data_dict.get("max_atoms_condition", [1])[0],
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)


