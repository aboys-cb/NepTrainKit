"""Card for replacing all target atoms inside a Cartesian region."""

from __future__ import annotations

import re
from typing import Any

from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CheckBox, ComboBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import ConditionalReplaceOperation, ConditionalReplaceParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    KeyValueTableInput,
    MakeDataCard,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class ConditionalReplaceCard(MakeDataCard):
    """Replace every matching target site in a fixed Cartesian region."""

    group = "Alloy"
    card_name = "Conditional Replace"
    description = (
        "Replace every matching target atom inside a fixed Cartesian region "
        "with one or more specified elements."
    )
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._preview_input_count: int | None = None
        self.setTitle(self.tr("Conditional Replace"))
        self._build_ui()

    def _build_ui(self) -> None:
        self.setObjectName("conditional_replace_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.target_edit = LineEdit(self.setting_widget)
        self.target_edit.setPlaceholderText(self.tr("For example: O"))
        target_field = CompactField(
            self.tr("Target element"), self.target_edit, self.setting_widget
        )

        self.replacements_edit = KeyValueTableInput(
            self.tr("Element"), self.tr("Relative ratio"), self.setting_widget
        )
        self.replacements_edit.add_row("F", "1.0")
        replacements_field = CompactField(
            self.tr("Replacement mixture"),
            self.replacements_edit,
            self.setting_widget,
            self.tr(
                "All matched sites are replaced. Ratios distribute replacement "
                "elements; they are not a replacement fraction."
            ),
        )

        replacement_section = InspectorSection(
            self.tr("Replacement rule"),
            self.setting_widget,
            self.tr("Choose one existing element and the elements that will replace it."),
        )
        replacement_section.addWidget(target_field)
        replacement_section.addWidget(replacements_field)

        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItem(
            self.tr("Independent random assignment"), userData=0
        )
        self.mode_combo.addItem(self.tr("Match overall ratio"), userData=1)
        mode_field = CompactField(
            self.tr("Element allocation"), self.mode_combo, self.setting_widget
        )

        self.seed_checkbox = CheckBox(
            self.tr("Use fixed random seed"), self.setting_widget
        )
        self.seed_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(1, 2**31 - 1)
        self.seed_frame.set_input_value([1])
        self.seed_field = CompactField(
            self.tr("Random seed"),
            self.seed_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )
        self.seed_field.hide()

        allocation_section = InspectorSection(
            self.tr("Element allocation"),
            self.setting_widget,
            self.tr(
                "Random assignment allows count fluctuations; ratio matching fixes the nearest feasible integer counts."
            ),
        )
        allocation_section.addWidget(mode_field)
        allocation_section.addWidget(self.seed_checkbox)
        allocation_section.addWidget(self.seed_field)

        self.condition_mode_combo = ComboBox(self.setting_widget)
        self.condition_mode_combo.addItem(
            self.tr("All target atoms"), userData="all"
        )
        self.condition_mode_combo.addItem(
            self.tr("One Cartesian boundary"), userData="simple"
        )
        self.condition_mode_combo.addItem(
            self.tr("Advanced expression"), userData="advanced"
        )
        condition_mode_field = CompactField(
            self.tr("Position selection"),
            self.condition_mode_combo,
            self.setting_widget,
        )

        self.condition_axis_combo = ComboBox(self.setting_widget)
        for axis in ("x", "y", "z"):
            self.condition_axis_combo.addItem(axis.upper(), userData=axis)
        self.condition_axis_combo.setFixedWidth(62)
        self.condition_operator_combo = ComboBox(self.setting_widget)
        for operator in (">=", "<=", ">", "<", "=="):
            self.condition_operator_combo.addItem(operator, userData=operator)
        self.condition_operator_combo.setFixedWidth(72)
        self.condition_value_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.condition_value_frame.set_input("Å", 1, "float")
        self.condition_value_frame.setRange(-1_000_000.0, 1_000_000.0)
        self.condition_value_frame.setDecimals(6)
        self.condition_value_frame.set_input_value([0.0])
        self.condition_value_frame.setMaximumWidth(150)

        self.simple_condition_row = QWidget(self.setting_widget)
        simple_layout = QHBoxLayout(self.simple_condition_row)
        simple_layout.setContentsMargins(0, 0, 0, 0)
        simple_layout.setSpacing(7)
        simple_layout.addWidget(self.condition_axis_combo)
        simple_layout.addWidget(self.condition_operator_combo)
        simple_layout.addWidget(self.condition_value_frame, 1)
        simple_layout.addStretch(1)

        self.condition_edit = LineEdit(self.setting_widget)
        self.condition_edit.setPlaceholderText(
            self.tr("For example: z>=8 and z<=10")
        )
        self.condition_field = CompactField(
            self.tr("Cartesian expression"),
            self.condition_edit,
            self.setting_widget,
            self.tr("Use x, y, z, comparisons, and/or; equality tolerance is 0.0001 Å."),
        )

        position_section = InspectorSection(
            self.tr("Cartesian region"),
            self.setting_widget,
            self.tr(
                "Coordinates use the current fixed global x/y/z axes and origin "
                "in Å; no periodic wrapping or lattice-axis conversion is applied."
            ),
        )
        position_section.addWidget(condition_mode_field)
        position_section.addWidget(self.simple_condition_row)
        position_section.addWidget(self.condition_field)

        self.settingLayout.addWidget(replacement_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(position_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(allocation_section, 2, 0, 1, 3)

        self._syncing_condition = False
        self.condition_mode_combo.currentIndexChanged.connect(
            self._update_condition_widgets
        )
        self.condition_axis_combo.currentIndexChanged.connect(
            self._write_simple_condition
        )
        self.condition_operator_combo.currentIndexChanged.connect(
            self._write_simple_condition
        )
        self.condition_value_frame.object_list[0].valueChanged.connect(
            self._write_simple_condition
        )
        self.condition_edit.textChanged.connect(self._read_condition_text)
        self.target_edit.textChanged.connect(self.refresh_compact_presentation)
        self.replacements_edit.editingFinished.connect(
            self.refresh_compact_presentation
        )
        self.replacements_edit.add_button.clicked.connect(
            self.refresh_compact_presentation
        )
        self.replacements_edit.remove_button.clicked.connect(
            self.refresh_compact_presentation
        )
        self.mode_combo.currentIndexChanged.connect(
            self.refresh_compact_presentation
        )
        self.seed_checkbox.toggled.connect(self._update_seed_visibility)
        self.seed_frame.object_list[0].valueChanged.connect(
            self.refresh_compact_presentation
        )
        self._set_condition("all")
        self._update_seed_visibility(False)

    def _set_condition(self, condition: str) -> None:
        text = str(condition or "").strip() or "all"
        match = re.fullmatch(
            r"([xyzXYZ])\s*(>=|<=|>|<|==)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            text,
        )
        self._syncing_condition = True
        self.condition_edit.setText(text)
        if text.lower() == "all":
            self.condition_mode_combo.setCurrentIndex(
                self.condition_mode_combo.findData("all")
            )
        elif match:
            axis, operator, value = match.groups()
            self.condition_mode_combo.setCurrentIndex(
                self.condition_mode_combo.findData("simple")
            )
            self.condition_axis_combo.setCurrentIndex(
                self.condition_axis_combo.findData(axis.lower())
            )
            self.condition_operator_combo.setCurrentIndex(
                self.condition_operator_combo.findData(operator)
            )
            self.condition_value_frame.set_input_value([float(value)])
        else:
            self.condition_mode_combo.setCurrentIndex(
                self.condition_mode_combo.findData("advanced")
            )
        self._syncing_condition = False
        self._update_condition_widgets()

    def _read_condition_text(self, text: str) -> None:
        if not self._syncing_condition:
            self._set_condition(text)
            self.refresh_compact_presentation()

    def _write_simple_condition(self, *_args) -> None:
        if (
            self._syncing_condition
            or self.condition_mode_combo.currentData() != "simple"
        ):
            return
        axis = self.condition_axis_combo.currentData()
        operator = self.condition_operator_combo.currentData()
        value = float(self.condition_value_frame.get_input_value()[0])
        self._syncing_condition = True
        self.condition_edit.setText(f"{axis}{operator}{value:.12g}")
        self._syncing_condition = False
        self.refresh_compact_presentation()

    def _update_condition_widgets(self, *_args) -> None:
        mode = self.condition_mode_combo.currentData()
        self.simple_condition_row.setVisible(mode == "simple")
        self.condition_field.setVisible(mode == "advanced")
        if not self._syncing_condition:
            if mode == "all":
                self._syncing_condition = True
                self.condition_edit.setText("all")
                self._syncing_condition = False
            elif mode == "simple":
                self._write_simple_condition()
        self.refresh_compact_presentation()

    def _update_seed_visibility(self, checked: bool) -> None:
        self.seed_field.setVisible(bool(checked))
        self.refresh_compact_presentation()

    @staticmethod
    def _first_structure(dataset):
        if dataset is None:
            return None
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return dataset
        try:
            return next(iter(dataset))
        except (StopIteration, TypeError):
            return None

    @staticmethod
    def _dataset_count(dataset) -> int:
        if dataset is None:
            return 0
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return 1
        try:
            return len(dataset)
        except TypeError:
            return 0

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        self.refresh_compact_presentation()

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def set_preview_structure(self, structure) -> None:
        self._input_structure = structure
        self.refresh_compact_presentation()

    def create_operation(self):
        return ConditionalReplaceOperation()

    def get_summary_text(self) -> str:
        try:
            summary = self.create_operation().selection_summary(
                self.get_params(), self._input_structure
            )
        except ValueError:
            return self.tr("Complete the replacement rule")
        replacements = ",".join(summary["replacement_elements"])
        if "matched_sites" in summary:
            return self.tr(
                "{target} → {replacements} · {matched} matched · 1/input"
            ).format(
                target=summary["target"],
                replacements=replacements,
                matched=summary["matched_sites"],
            )
        return self.tr("{target} → {replacements} · 1 output/input").format(
            target=summary["target"], replacements=replacements
        )

    def get_guidance_text(self) -> str:
        try:
            summary = self.create_operation().selection_summary(
                self.get_params(), self._input_structure
            )
        except ValueError as exc:
            return translate_runtime_message(exc)

        parts: list[str] = []
        input_count = self._preview_input_count
        if input_count is None:
            input_count = self._dataset_count(getattr(self, "dataset", None)) or None
        elif input_count == 0:
            input_count = None
        if input_count is None:
            parts.append(self.tr("Outputs/input: 1."))
        else:
            parts.append(
                self.tr("Inputs {inputs} × 1 output/input = outputs {total}.").format(
                    inputs=input_count, total=input_count
                )
            )
        if "matched_sites" in summary:
            parts.append(
                self.tr(
                    "First input: {targets} target sites → {matched} Cartesian matches → all replaced."
                ).format(
                    targets=summary["target_sites"], matched=summary["matched_sites"]
                )
            )
            counts = summary.get("replacement_counts")
            if counts:
                count_text = ", ".join(
                    f"{element}:{count}" for element, count in counts
                )
                parts.append(
                    self.tr("Matched integer allocation: {counts}.").format(
                        counts=count_text
                    )
                )
            arrays = getattr(self._input_structure, "arrays", {})
            if any(
                name in arrays for name in ("spin", "initial_magmoms", "magmoms")
            ):
                parts.append(
                    self.tr(
                        "Existing spin and initial magnetic moments stay on their "
                        "sites; check or reset them after changing elements."
                    )
                )
        else:
            parts.append(
                self.tr(
                    "Every matched target site is replaced; ratios do not keep part of the target element."
                )
            )
        return " ".join(parts)

    def get_params(self) -> ConditionalReplaceParams:
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
        self.target_edit.setText(params.target)
        self.replacements_edit.setText(params.replacements)
        self._set_condition(params.condition)
        seed = int(params.seed)
        self.seed_checkbox.setChecked(seed != 0)
        self.seed_frame.set_input_value([seed if seed > 0 else 1])
        mode_index = self.mode_combo.findData(int(params.mode))
        self.mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        self._update_seed_visibility(seed != 0)

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
                    atom_list = [
                        item.strip()
                        for item in str(new_atoms).split(",")
                        if item.strip()
                    ]
                    ratio_list = [
                        item.strip()
                        for item in str(ratios).split(",")
                        if item.strip()
                    ]
                    replacements = ",".join(
                        f"{atom}:{ratio}"
                        for atom, ratio in zip(atom_list, ratio_list)
                    )
            seed = data_dict.get("seed", [0])
            params = ConditionalReplaceParams(
                target=data_dict.get("target", ""),
                replacements=replacements,
                condition=data_dict.get("condition", "all"),
                seed=seed[0] if isinstance(seed, (list, tuple)) else seed,
                mode=data_dict.get("mode", 0),
            )
        self.set_params(params)
