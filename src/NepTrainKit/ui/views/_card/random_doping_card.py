"""Card for stochastic site doping based on user-defined rules."""

import json
from itertools import combinations

import numpy as np
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox, EditableComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame, DopingRulesWidget
from NepTrainKit.ui.widgets import MakeDataCard
from scipy.stats.qmc import Sobol
from ase.data import atomic_masses, atomic_numbers


def sample_dopants(dopant_list, ratios, N, exact=False, rng: np.random.Generator | None = None, ratio_type="atom"):
    if rng is None:
        rng = np.random.default_rng()

    dopant_list = list(dopant_list)
    ratios = np.array(ratios, dtype=float)

    if ratio_type == "mass":
        masses = np.array([atomic_masses[atomic_numbers.get(elem, 1)] for elem in dopant_list])
        atom_ratios = ratios / masses
        atom_ratios = atom_ratios / atom_ratios.sum()
    else:
        atom_ratios = ratios / ratios.sum()

    if not exact:
        return list(rng.choice(dopant_list, size=N, p=atom_ratios, replace=True))
    else:
        counts = (atom_ratios * N).astype(int)
        diff = N - counts.sum()
        if diff != 0:
            max_idx = np.argmax(atom_ratios)
            counts[max_idx] += diff

        arr = np.repeat(dopant_list, counts)
        rng.shuffle(arr)
        return list(arr)


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

    def process_structure(self, structure):
        """Apply stochastic dopant replacements according to the configured rules.

        Parameters
        ----------
        structure : ase.Atoms
            Structure to modify.

        Returns
        -------
        list[ase.Atoms]
            Doped structures that satisfy the user-defined rules.
        """
        structure_list = []

        rules = self.rules_widget.to_rules()
        if not isinstance(rules, list) or not rules:
            return [structure]

        max_num = int(self.max_atoms_condition_frame.get_input_value()[0])
        exact = self.doping_type_combo.currentText() == "Exact"
        base_seed = int(self.seed_frame.get_input_value()[0]) if self.seed_checkbox.isChecked() else None
        rng = np.random.default_rng(base_seed)
        for _ in range(max_num):
            new_structure = structure.copy()
            total_doping = 0
            for rule in rules:
                target = rule.get("target")
                dopants = rule.get("dopants", {})
                if not target or not dopants:
                    continue

                groups = rule.get("group")

                if groups and "group" in new_structure.arrays:
                    candidate_indices = [
                        i
                        for i, elem, g in zip(range(len(new_structure)), new_structure, new_structure.arrays["group"])
                        if elem.symbol == target and g in groups
                    ]
                else:
                    candidate_indices = [i for i, a in enumerate(new_structure) if a.symbol == target]

                if not candidate_indices:
                    continue

                use_mode = rule.get("use", "atomic_percent")

                if use_mode == "atomic_percent":
                    percent_min, percent_max = rule.get("percent", [0.0, 100.0])
                    value = rng.uniform(float(percent_min), float(percent_max)) / 100.0
                    doping_num = max(1, int(len(candidate_indices) * value))
                elif use_mode == "mass_percent":
                    percent_min, percent_max = rule.get("percent", [0.0, 100.0])
                    target_mass_percent = rng.uniform(float(percent_min), float(percent_max)) / 100.0

                    target_atomic_number = atomic_numbers.get(target, 1)
                    target_mass = atomic_masses[target_atomic_number]
                    total_target_mass = len(candidate_indices) * target_mass

                    dopant_elements = list(dopants.keys())
                    if dopant_elements:
                        avg_dopant_mass = np.mean(
                            [atomic_masses[atomic_numbers.get(elem, 1)] for elem in dopant_elements]
                        )
                    else:
                        avg_dopant_mass = target_mass

                    doped_mass = total_target_mass * target_mass_percent
                    doping_num = max(1, int(doped_mass / avg_dopant_mass))

                elif use_mode == "count":
                    count_min, count_max = rule.get("count", [1, 1])
                    doping_num = int(rng.integers(int(count_min), int(count_max) + 1))
                else:
                    doping_num = len(candidate_indices)

                doping_num = min(doping_num, len(candidate_indices))

                idxs = rng.choice(candidate_indices, doping_num, replace=False)

                dopant_list = list(dopants.keys())
                ratios = np.array(list(dopants.values()), dtype=float)
                ratio_type = rule.get("ratio_type", "atom")
                sample = sample_dopants(dopant_list, ratios, doping_num, exact, rng=rng, ratio_type=ratio_type)

                for idx, elem in zip(idxs, sample):
                    new_structure[idx].symbol = elem
                total_doping += doping_num
            if total_doping:
                append_config_tag(new_structure, f"Dop(n={total_doping})")

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

        data_dict["rules"] = json.dumps(self.rules_widget.to_rules(), ensure_ascii=False)
        data_dict["doping_type"] = self.doping_type_combo.currentText()
        data_dict["max_atoms_condition"] = self.max_atoms_condition_frame.get_input_value()
        data_dict["use_seed"] = self.seed_checkbox.isChecked()
        data_dict["seed"] = self.seed_frame.get_input_value()
        return data_dict

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.

        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        super().from_dict(data_dict)

        rules = data_dict.get("rules", "")
        if isinstance(rules, str):
            try:
                rules = json.loads(rules)
            except Exception:
                rules = []
        self.rules_widget.from_rules(rules)
        self.doping_type_combo.setCurrentText(data_dict.get("doping_type", "Exact"))
        self.max_atoms_condition_frame.set_input_value(data_dict.get("max_atoms_condition", [1]))
        self.seed_checkbox.setChecked(bool(data_dict.get("use_seed", False)))
        self.seed_frame.set_input_value(data_dict.get("seed", [0]))
