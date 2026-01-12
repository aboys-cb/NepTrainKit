"""Card for assigning global alloy occupancies from a target composition."""

from __future__ import annotations

import json

import numpy as np
from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.alloy import assign_random_occupancy, parse_composition
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class RandomOccupancyCard(MakeDataCard):
    """Assign alloy elements to all (or grouped) lattice sites using a target composition."""

    group = "Alloy"
    card_name = "Random Occupancy"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Random Occupancy Assignment")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("random_occupancy_card_widget")

        self.source_label = BodyLabel("Composition", self.setting_widget)
        self.source_combo = ComboBox(self.setting_widget)
        self.source_combo.addItems(["From info (alloy_composition)", "Manual"])
        self.source_label.setToolTip("Use composition stored by Composition Sweep or provide one manually")
        self.source_label.installEventFilter(ToolTipFilter(self.source_label, 300, ToolTipPosition.TOP))

        self.manual_label = BodyLabel("Manual comp", self.setting_widget)
        self.manual_edit = LineEdit(self.setting_widget)
        self.manual_edit.setPlaceholderText('Co:0.33,Cr:0.33,Ni:0.34 or {"Co":0.33,"Cr":0.33,"Ni":0.34}')
        self.manual_label.setToolTip("Element fractions. Used when 'Manual' is selected or when info is missing.")
        self.manual_label.installEventFilter(ToolTipFilter(self.manual_label, 300, ToolTipPosition.TOP))

        self.mode_label = BodyLabel("Mode", self.setting_widget)
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItems(["Exact", "Random"])
        self.mode_label.setToolTip("Exact: integer counts match fractions; Random: multinomial sampling")
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))

        self.samples_label = BodyLabel("Samples/frame", self.setting_widget)
        self.samples_frame = SpinBoxUnitInputFrame(self)
        self.samples_frame.set_input("unit", 1, "int")
        self.samples_frame.setRange(1, 999999)
        self.samples_frame.set_input_value([1])

        self.group_label = BodyLabel("Group filter", self.setting_widget)
        self.group_edit = LineEdit(self.setting_widget)
        self.group_edit.setPlaceholderText("Optional: a,b,c")
        self.group_label.setToolTip("If the structure has arrays['group'], restrict assignment to these groups")
        self.group_label.installEventFilter(ToolTipFilter(self.group_label, 300, ToolTipPosition.TOP))

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.settingLayout.addWidget(self.source_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.source_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.manual_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.manual_edit, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.mode_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.samples_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.samples_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.group_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.group_edit, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 5, 1, 1, 2)

    def _read_composition(self, structure) -> dict[str, float]:
        if self.source_combo.currentText().startswith("From info"):
            raw = structure.info.get("alloy_composition")
            if isinstance(raw, dict):
                return {str(k): float(v) for k, v in raw.items()}
            if isinstance(raw, str) and raw.strip():
                try:
                    if raw.strip().startswith("{"):
                        data = json.loads(raw)
                        if isinstance(data, dict):
                            return {str(k): float(v) for k, v in data.items()}
                except Exception:
                    pass
                try:
                    return parse_composition(raw)
                except Exception:
                    return {}
            # fall back to manual if present
        manual = self.manual_edit.text().strip()
        if not manual:
            return {}
        try:
            return parse_composition(manual)
        except Exception as exc:  # noqa: BLE001
            MessageManager.send_warning_message(f"RandomOccupancy: invalid manual composition: {exc}")
            return {}

    def _eligible_indices(self, structure) -> np.ndarray | None:
        groups_text = self.group_edit.text().strip()
        if not groups_text:
            return None
        if "group" not in structure.arrays:
            return None
        allowed = {g.strip() for g in groups_text.split(",") if g.strip()}
        if not allowed:
            return None
        grp = structure.arrays["group"]
        return np.array([i for i, g in enumerate(grp) if str(g) in allowed], dtype=int)

    def process_structure(self, structure):
        comp = self._read_composition(structure)
        if not comp:
            MessageManager.send_warning_message("RandomOccupancy: missing composition (info or manual).")
            return [structure]

        mode = self.mode_combo.currentText()
        n_samples = int(self.samples_frame.get_input_value()[0])
        indices = self._eligible_indices(structure)

        base_seed = int(self.seed_frame.get_input_value()[0]) if self.seed_checkbox.isChecked() else None
        sweep_index = int(structure.info.get("_sweep_index", 0) or 0)

        out = []
        for sample_idx in range(max(n_samples, 1)):
            if base_seed is None:
                rng = np.random.default_rng()
                seed_note = ""
            else:
                derived_seed = int(base_seed + sweep_index * 1000003 + sample_idx)
                rng = np.random.default_rng(derived_seed)
                seed_note = f",seed={derived_seed}"

            new_atoms = assign_random_occupancy(
                structure,
                comp,
                indices=indices,
                mode=mode,
                rng=rng,
            )
            new_atoms.info["Config_type"] = new_atoms.info.get("Config_type", "") + f" Occ(mode={mode}{seed_note})"
            out.append(new_atoms)
        return out

    def to_dict(self):
        data = super().to_dict()
        data["source"] = self.source_combo.currentText()
        data["manual"] = self.manual_edit.text()
        data["mode"] = self.mode_combo.currentText()
        data["samples"] = self.samples_frame.get_input_value()
        data["group_filter"] = self.group_edit.text()
        data["use_seed"] = self.seed_checkbox.isChecked()
        data["seed"] = self.seed_frame.get_input_value()
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.source_combo.setCurrentText(data_dict.get("source", "From info (alloy_composition)"))
        self.manual_edit.setText(data_dict.get("manual", ""))
        self.mode_combo.setCurrentText(data_dict.get("mode", "Exact"))
        self.samples_frame.set_input_value(data_dict.get("samples", [1]))
        self.group_edit.setText(data_dict.get("group_filter", ""))
        self.seed_checkbox.setChecked(bool(data_dict.get("use_seed", False)))
        self.seed_frame.set_input_value(data_dict.get("seed", [0]))
