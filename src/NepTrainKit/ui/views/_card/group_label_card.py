"""Card for labeling atoms into groups (e.g., A/B sublattices) for downstream rules."""

from __future__ import annotations

import numpy as np
from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.ui.widgets import MakeDataCard


@CardManager.register_card
class GroupLabelCard(MakeDataCard):
    """Attach ``atoms.arrays['group']`` labels using common, lattice-agnostic rules.

    The default strategy labels atoms by a commensurate k-vector layering in
    fractional coordinates, producing two groups (even/odd) suitable for
    two-sublattice AFM patterns.
    """

    group = "Alloy"
    card_name = "Group Label"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Group Label (A/B Sublattice)")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("group_label_card_widget")

        self.mode_label = BodyLabel("Mode", self.setting_widget)
        self.mode_label.setToolTip("How to assign group labels")
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItems(["k-vector layers (recommended)", "fractional parity (2x rounding)"])

        self.kvec_label = BodyLabel("k-vector", self.setting_widget)
        self.kvec_label.setToolTip("Layering vector in fractional coordinates: 100, 010, 001, 110, 111")
        self.kvec_label.installEventFilter(ToolTipFilter(self.kvec_label, 300, ToolTipPosition.TOP))
        self.kvec_combo = ComboBox(self.setting_widget)
        self.kvec_combo.addItems(["100", "010", "001", "110", "111"])
        self.kvec_combo.setCurrentText("111")

        self.group_a_label = BodyLabel("Group A", self.setting_widget)
        self.group_a_label.setToolTip("Label assigned to even layer/parity")
        self.group_a_label.installEventFilter(ToolTipFilter(self.group_a_label, 300, ToolTipPosition.TOP))
        self.group_a_edit = LineEdit(self.setting_widget)
        self.group_a_edit.setText("A")

        self.group_b_label = BodyLabel("Group B", self.setting_widget)
        self.group_b_label.setToolTip("Label assigned to odd layer/parity")
        self.group_b_label.installEventFilter(ToolTipFilter(self.group_b_label, 300, ToolTipPosition.TOP))
        self.group_b_edit = LineEdit(self.setting_widget)
        self.group_b_edit.setText("B")

        self.overwrite_checkbox = CheckBox("Overwrite existing group", self.setting_widget)
        self.overwrite_checkbox.setChecked(True)

        self.settingLayout.addWidget(self.mode_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.kvec_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.kvec_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.group_a_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.group_a_edit, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.group_b_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.group_b_edit, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.overwrite_checkbox, 4, 0, 1, 2)

    @staticmethod
    def _parse_kvec(text: str) -> np.ndarray:
        text = (text or "").strip()
        if text in {"100", "010", "001", "110", "111"}:
            return np.array([int(c) for c in text], dtype=float)
        return np.array([1.0, 1.0, 1.0], dtype=float)

    def _label_by_kvec(self, atoms) -> np.ndarray:
        k = self._parse_kvec(self.kvec_combo.currentText())
        scaled = atoms.get_scaled_positions(wrap=True)
        phase = np.floor(2.0 * (scaled @ k)).astype(int)
        return (phase % 2).astype(int)

    @staticmethod
    def _label_by_parity(atoms) -> np.ndarray:
        scaled = atoms.get_scaled_positions(wrap=True)
        ints = np.rint(2.0 * scaled).astype(int)
        parity = (ints.sum(axis=1) % 2).astype(int)
        return parity

    def process_structure(self, structure):
        if (not self.overwrite_checkbox.isChecked()) and "group" in structure.arrays:
            return [structure]

        if structure.cell is None or np.linalg.det(structure.cell.array) == 0:  # pyright:ignore
            MessageManager.send_warning_message("GroupLabel: structure has no valid cell.")
            return [structure]

        mode = self.mode_combo.currentText()
        a_label = (self.group_a_edit.text() or "A").strip()
        b_label = (self.group_b_edit.text() or "B").strip()
        if not a_label:
            a_label = "A"
        if not b_label:
            b_label = "B"

        atoms = structure.copy()
        if mode.startswith("fractional parity"):
            flags = self._label_by_parity(atoms)
            tag = "par"
        else:
            flags = self._label_by_kvec(atoms)
            tag = f"k{self.kvec_combo.currentText()}"

        groups = np.where(flags == 0, a_label, b_label).astype(object)
        atoms.arrays["group"] = groups
        append_config_tag(atoms, f"Grp({tag},{a_label}/{b_label})")
        return [atoms]

    def to_dict(self):
        data = super().to_dict()
        data["mode"] = self.mode_combo.currentText()
        data["kvec"] = self.kvec_combo.currentText()
        data["group_a"] = self.group_a_edit.text()
        data["group_b"] = self.group_b_edit.text()
        data["overwrite"] = self.overwrite_checkbox.isChecked()
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.mode_combo.setCurrentText(data_dict.get("mode", "k-vector layers (recommended)"))
        self.kvec_combo.setCurrentText(data_dict.get("kvec", "111"))
        self.group_a_edit.setText(data_dict.get("group_a", "A"))
        self.group_b_edit.setText(data_dict.get("group_b", "B"))
        self.overwrite_checkbox.setChecked(bool(data_dict.get("overwrite", True)))
