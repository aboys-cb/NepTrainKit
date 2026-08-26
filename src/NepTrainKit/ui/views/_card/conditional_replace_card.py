"""Card for replacing atoms based on spatial conditions in the current dataset."""

from __future__ import annotations

from typing import Any
import re

from PySide6.QtWidgets import QGridLayout
from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import ConditionalReplaceOperation, ConditionalReplaceParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import KeyValueTableInput, MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class ConditionalReplaceCard(MakeDataCard):
    """Replace atoms in the active structures using spatial conditions and ratios."""

    group = "Alloy"
    card_name = "Conditional Replace"
    description = (
        "Select a target element by Cartesian coordinates, then replace every "
        "matching site using the specified replacement mixture."
    )
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Conditional Replace"))
        self._build_ui()

    def _build_ui(self):
        self.target_label = BodyLabel(self.tr("Target element"), self.setting_widget)
        self.target_edit = LineEdit(self.setting_widget)
        self.target_edit.setPlaceholderText(self.tr("e.g., O"))

        self.replacements_label = BodyLabel(self.tr("Replacement mixture"), self.setting_widget)
        self.replacements_edit = KeyValueTableInput(
            self.tr("Replacement element"), self.tr("Relative ratio"), self.setting_widget
        )
        self.replacements_label.setToolTip(
            self.tr(
                "Every matching site is replaced. Ratios only distribute the replacement elements; "
                "a bare element means ratio 1.0."
            )
        )
        self.replacements_label.installEventFilter(ToolTipFilter(self.replacements_label, 300, ToolTipPosition.TOP))

        self.mode_label = BodyLabel(self.tr("Element allocation"), self.setting_widget)
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItem(self.tr("Independent random assignment"), userData=0)
        self.mode_combo.addItem(self.tr("Match overall ratio"), userData=1)
        self.mode_label.setToolTip(
            self.tr(
                "Independent: draw a replacement for each matching site. "
                "Match overall ratio: allocate integer counts as closely as possible."
            )
        )
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))

        self.condition_label = BodyLabel(self.tr("Position filter (Cartesian, Å)"), self.setting_widget)
        self.condition_edit = LineEdit(self.setting_widget)
        self.condition_edit.setText("all")
        self.condition_edit.setPlaceholderText(
            self.tr('all, z>=8, or z>=8 and z<=10')
        )
        self.condition_mode_combo = ComboBox(self.setting_widget)
        self.condition_mode_combo.addItem(self.tr("All target atoms"), userData="all")
        self.condition_mode_combo.addItem(self.tr("One coordinate boundary"), userData="simple")
        self.condition_mode_combo.addItem(self.tr("Advanced expression"), userData="advanced")
        self.condition_axis_combo = ComboBox(self.setting_widget)
        for axis in ("x", "y", "z"):
            self.condition_axis_combo.addItem(axis.upper(), userData=axis)
        self.condition_operator_combo = ComboBox(self.setting_widget)
        for operator in (">=", "<=", ">", "<", "=="):
            self.condition_operator_combo.addItem(operator, userData=operator)
        self.condition_value_frame = SpinBoxUnitInputFrame(self)
        self.condition_value_frame.set_input("A", 1, "float")
        self.condition_value_frame.setRange(-1000000.0, 1000000.0)
        self.condition_value_frame.setDecimals(6)
        self.condition_value_frame.set_input_value([0.0])
        self.condition_label.setToolTip(
            self.tr(
                'Use Cartesian x, y, and z coordinates in angstrom. '
                'Enter "all" to select every atom of the target element.'
            )
        )
        self.condition_label.installEventFilter(ToolTipFilter(self.condition_label, 300, ToolTipPosition.TOP))

        self.seed_checkbox = CheckBox(self.tr("Use fixed seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip(self.tr("Enable reproducible replacement allocation"))
        self.seed_checkbox.installEventFilter(
            ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP)
        )
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(1, 2**31 - 1)
        self.seed_frame.set_input_value([1])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(
            lambda _state: self.seed_frame.setEnabled(self.seed_checkbox.isChecked())
        )

        layout: QGridLayout = self.settingLayout
        layout.addWidget(self.target_label, 0, 0, 1, 1)
        layout.addWidget(self.target_edit, 0, 1, 1, 2)
        layout.addWidget(self.replacements_label, 1, 0, 1, 1)
        layout.addWidget(self.replacements_edit, 1, 1, 1, 2)
        layout.addWidget(self.mode_label, 2, 0, 1, 1)
        layout.addWidget(self.mode_combo, 2, 1, 1, 2)
        layout.addWidget(self.condition_label, 3, 0, 1, 1)
        layout.addWidget(self.condition_mode_combo, 3, 1, 1, 2)
        layout.addWidget(self.condition_axis_combo, 4, 0, 1, 1)
        layout.addWidget(self.condition_operator_combo, 4, 1, 1, 1)
        layout.addWidget(self.condition_value_frame, 4, 2, 1, 1)
        layout.addWidget(self.condition_edit, 5, 0, 1, 3)
        layout.addWidget(self.seed_checkbox, 6, 0, 1, 1)
        layout.addWidget(self.seed_frame, 6, 1, 1, 2)

        self._syncing_condition = False
        self.condition_mode_combo.currentIndexChanged.connect(self._update_condition_widgets)
        self.condition_axis_combo.currentIndexChanged.connect(self._write_simple_condition)
        self.condition_operator_combo.currentIndexChanged.connect(self._write_simple_condition)
        self.condition_value_frame.object_list[0].valueChanged.connect(self._write_simple_condition)
        self.condition_edit.textChanged.connect(self._read_condition_text)
        self._set_condition("all")

    def _set_condition(self, condition: str) -> None:
        text = str(condition or "").strip() or "all"
        match = re.fullmatch(
            r"([xyzXYZ])\s*(>=|<=|>|<|==)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            text,
        )
        self._syncing_condition = True
        self.condition_edit.setText(text)
        if text.lower() == "all":
            self.condition_mode_combo.setCurrentIndex(self.condition_mode_combo.findData("all"))
        elif match:
            axis, operator, value = match.groups()
            self.condition_mode_combo.setCurrentIndex(self.condition_mode_combo.findData("simple"))
            self.condition_axis_combo.setCurrentIndex(self.condition_axis_combo.findData(axis.lower()))
            self.condition_operator_combo.setCurrentIndex(self.condition_operator_combo.findData(operator))
            self.condition_value_frame.set_input_value([float(value)])
        else:
            self.condition_mode_combo.setCurrentIndex(self.condition_mode_combo.findData("advanced"))
        self._syncing_condition = False
        self._update_condition_widgets()

    def _read_condition_text(self, text: str) -> None:
        if not self._syncing_condition:
            self._set_condition(text)

    def _write_simple_condition(self, *_args) -> None:
        if self._syncing_condition or self.condition_mode_combo.currentData() != "simple":
            return
        axis = self.condition_axis_combo.currentData()
        operator = self.condition_operator_combo.currentData()
        value = float(self.condition_value_frame.get_input_value()[0])
        self._syncing_condition = True
        self.condition_edit.setText(f"{axis}{operator}{value:.12g}")
        self._syncing_condition = False

    def _update_condition_widgets(self, *_args) -> None:
        mode = self.condition_mode_combo.currentData()
        simple = mode == "simple"
        for widget in (self.condition_axis_combo, self.condition_operator_combo, self.condition_value_frame):
            widget.setVisible(simple)
        self.condition_edit.setVisible(mode == "advanced")
        if self._syncing_condition:
            return
        if mode == "all":
            self._syncing_condition = True
            self.condition_edit.setText("all")
            self._syncing_condition = False
        elif simple:
            self._write_simple_condition()

    def create_operation(self):
        """Return the UI-independent conditional replacement operation."""
        return ConditionalReplaceOperation()

    def get_params(self) -> ConditionalReplaceParams:
        """Read replacement parameters from UI controls."""
        return ConditionalReplaceParams(
            target=self.target_edit.text(),
            replacements=self.replacements_edit.text(),
            condition=self.condition_edit.text().strip() or "all",
            seed=(
                int(self.seed_frame.get_input_value()[0])
                if self.seed_checkbox.isChecked()
                else 0
            ),
            mode=int(self.mode_combo.currentData()),
        )

    def set_params(self, params: ConditionalReplaceParams) -> None:
        """Apply replacement parameters to UI controls."""
        self.target_edit.setText(params.target)
        self.replacements_edit.setText(params.replacements)
        self._set_condition(params.condition)
        seed = int(params.seed)
        self.seed_checkbox.setChecked(seed != 0)
        self.seed_frame.set_input_value([seed if seed != 0 else 1])
        self.seed_frame.setEnabled(seed != 0)
        mode_index = self.mode_combo.findData(int(params.mode))
        self.mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict: dict[str, Any]) -> None:
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = ConditionalReplaceParams(
                target=raw_params.get("target", ""),
                replacements=raw_params.get("replacements", ""),
                condition=raw_params.get("condition", "all"),
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
                condition=data_dict.get("condition", "all"),
                seed=seed[0] if isinstance(seed, (list, tuple)) else seed,
                mode=data_dict.get("mode", 0),
            )
        self.set_params(params)
