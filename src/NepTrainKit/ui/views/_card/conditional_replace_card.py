"""Card for replacing atoms based on spatial conditions in the current dataset."""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import ConditionalReplaceOperation, ConditionalReplaceParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class ConditionalReplaceCard(MakeDataCard):
    """Replace atoms in the active structures using spatial conditions and ratios."""

    group = "Alloy"
    card_name = "Conditional Replace"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Conditional Atom Replacement")
        self._build_ui()

    def _build_ui(self):
        self.target_label = BodyLabel("Target element", self.setting_widget)
        self.target_edit = LineEdit(self.setting_widget)
        self.target_edit.setPlaceholderText("e.g., O")

        self.replacements_label = BodyLabel("Replacements", self.setting_widget)
        self.replacements_edit = LineEdit(self.setting_widget)
        self.replacements_edit.setPlaceholderText("Cs:0.6,Na:0.4 or Ni")
        self.replacements_label.setToolTip("Use element:ratio pairs, comma-separated. Ratio defaults to 1.0 when omitted.")
        self.replacements_label.installEventFilter(ToolTipFilter(self.replacements_label, 300, ToolTipPosition.TOP))

        self.mode_label = BodyLabel("Mode", self.setting_widget)
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItems(["Random", "Exact ratio"])
        self.mode_label.setToolTip("Random: sample each atom by probability. Exact ratio: allocate counts by ratio then assign.")
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))

        self.condition_label = BodyLabel("Condition", self.setting_widget)
        self.condition_edit = LineEdit(self.setting_widget)
        self.condition_edit.setPlaceholderText('Use x, y, z; e.g., "z==2.658", "z>=1.0 and z<=3.0", or "all"')
        self.condition_label.setToolTip("Expression on coordinates; supports x, y, z with >, <, ==, >=, <=, and/or.")
        self.condition_label.installEventFilter(ToolTipFilter(self.condition_label, 300, ToolTipPosition.TOP))

        self.seed_label = BodyLabel("Seed", self.setting_widget)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("seed", 1)
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.setToolTip("Random seed (0 leaves it random)")
        self.seed_label.installEventFilter(ToolTipFilter(self.seed_label, 300, ToolTipPosition.TOP))

        layout: QGridLayout = self.settingLayout
        layout.addWidget(self.target_label, 0, 0, 1, 1)
        layout.addWidget(self.target_edit, 0, 1, 1, 2)
        layout.addWidget(self.replacements_label, 1, 0, 1, 1)
        layout.addWidget(self.replacements_edit, 1, 1, 1, 2)
        layout.addWidget(self.mode_label, 2, 0, 1, 1)
        layout.addWidget(self.mode_combo, 2, 1, 1, 2)
        layout.addWidget(self.condition_label, 3, 0, 1, 1)
        layout.addWidget(self.condition_edit, 3, 1, 1, 2)
        layout.addWidget(self.seed_label, 4, 0, 1, 1)
        layout.addWidget(self.seed_frame, 4, 1, 1, 2)

    def create_operation(self):
        """Return the UI-independent conditional replacement operation."""
        return ConditionalReplaceOperation()

    def get_params(self) -> ConditionalReplaceParams:
        """Read replacement parameters from UI controls."""
        return ConditionalReplaceParams(
            target=self.target_edit.text(),
            replacements=self.replacements_edit.text(),
            condition=self.condition_edit.text(),
            seed=int(self.seed_frame.get_input_value()[0]),
            mode=self.mode_combo.currentIndex(),
        )

    def set_params(self, params: ConditionalReplaceParams) -> None:
        """Apply replacement parameters to UI controls."""
        self.target_edit.setText(params.target)
        self.replacements_edit.setText(params.replacements)
        self.condition_edit.setText(params.condition)
        self.seed_frame.set_input_value([int(params.seed)])
        self.mode_combo.setCurrentIndex(int(params.mode))

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        params = self.get_params()
        data.update(
            {
                "params": params_to_dict(params),
                "target": params.target,
                "replacements": params.replacements,
                "condition": params.condition,
                "seed": [params.seed],
                "mode": params.mode,
            }
        )
        return data

    def from_dict(self, data_dict: dict[str, Any]) -> None:
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = ConditionalReplaceParams(
                target=raw_params.get("target", ""),
                replacements=raw_params.get("replacements", ""),
                condition=raw_params.get("condition", ""),
                seed=raw_params.get("seed", 0),
                mode=raw_params.get("mode", 0),
            )
        else:
            replacements = data_dict.get("replacements", "")
            if not replacements:
                new_atoms = data_dict.get("new_atoms", "")
                ratios = data_dict.get("ratios", "")
                if new_atoms and ratios:
                    atom_list = [item.strip() for item in str(new_atoms).split(",") if item.strip()]
                    ratio_list = [item.strip() for item in str(ratios).split(",") if item.strip()]
                    replacements = ",".join(f"{atom}:{ratio}" for atom, ratio in zip(atom_list, ratio_list))
            seed = data_dict.get("seed", [0])
            params = ConditionalReplaceParams(
                target=data_dict.get("target", ""),
                replacements=replacements,
                condition=data_dict.get("condition", ""),
                seed=seed[0] if isinstance(seed, (list, tuple)) else seed,
                mode=data_dict.get("mode", 0),
            )
        self.set_params(params)
