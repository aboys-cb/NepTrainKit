"""Card for stochastic site doping based on user-defined rules."""

import json

from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import RandomDopingOperation, RandomDopingParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame, DopingRulesWidget
from NepTrainKit.ui.widgets import MakeDataCard


@CardManager.register_card
class RandomDopingCard(MakeDataCard):
    """Perform random atomic substitutions according to user-specified doping rules.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Alloy"
    card_name = "Random Doping"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Random Doping Replacement")
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration."""
        self.setObjectName("random_doping_card_widget")

        self.rules_label = BodyLabel("Rules", self.setting_widget)
        self.rules_widget = DopingRulesWidget(self.setting_widget)
        self.rules_label.setToolTip("doping rules")
        self.rules_label.installEventFilter(ToolTipFilter(self.rules_label, 300, ToolTipPosition.TOP))

        self.doping_label = BodyLabel("Doping", self.setting_widget)

        self.doping_type_combo = ComboBox(self.setting_widget)
        self.doping_type_combo.addItem("Random")
        self.doping_type_combo.addItem("Exact")
        self.doping_label.setToolTip("Select doping algorithm")
        self.doping_label.installEventFilter(ToolTipFilter(self.doping_label, 300, ToolTipPosition.TOP))

        self.max_atoms_label = BodyLabel("Max structures", self.setting_widget)
        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("unit", 1)
        self.max_atoms_condition_frame.setRange(1, 999999)
        self.max_atoms_label.setToolTip("Number of structures to generate")
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
        self.settingLayout.addWidget(self.doping_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.doping_type_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 3, 1, 1, 2)

    def create_operation(self):
        """Return the UI-independent random doping operation."""
        return RandomDopingOperation()

    def get_params(self) -> RandomDopingParams:
        """Read random doping parameters from UI controls."""
        return RandomDopingParams(
            rules=self.rules_widget.to_rules(),
            doping_type=self.doping_type_combo.currentText(),
            max_structures=int(self.max_atoms_condition_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: RandomDopingParams) -> None:
        """Apply random doping parameters to UI controls."""
        self.rules_widget.from_rules(params.rules)
        self.doping_type_combo.setCurrentText(params.doping_type)
        self.max_atoms_condition_frame.set_input_value([int(params.max_structures)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])

    def process_structure(self, structure):
        """Apply stochastic dopant replacements from UI-independent parameters."""
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
        data_dict["rules"] = json.dumps(params.rules, ensure_ascii=False)
        data_dict["doping_type"] = params.doping_type
        data_dict["max_atoms_condition"] = [params.max_structures]
        data_dict["use_seed"] = params.use_seed
        data_dict["seed"] = [params.seed]
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
            params = RandomDopingParams(
                rules=raw_params.get("rules", []),
                doping_type=raw_params.get("doping_type", "Random"),
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
            params = RandomDopingParams(
                rules=rules,
                doping_type=data_dict.get("doping_type", "Exact"),
                max_structures=data_dict.get("max_atoms_condition", [1])[0],
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
