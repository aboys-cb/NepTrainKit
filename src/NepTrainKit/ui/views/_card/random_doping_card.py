"""Card for stochastic site doping based on user-defined rules."""

import json

from PySide6.QtCore import QCoreApplication
from qfluentwidgets import CaptionLabel, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import RandomDopingOperation, RandomDopingParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.widgets import (
    CompactField,
    DopingRulesWidget,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SegmentedControl,
    SpinBoxUnitInputFrame,
)


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
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self._input_structure = None
        self._preview_input_count: int | None = None
        self._preview: dict[str, object] | None = None
        self._preview_error = ""
        self.setTitle(QCoreApplication.translate("CardCatalog", "Random Doping"))
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration."""
        self.setObjectName("random_doping_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.rules_widget = DopingRulesWidget(self.setting_widget)
        self.rules_widget.add_rule()
        rules_section = InspectorSection(
            self.tr("Replacement rules"),
            self.setting_widget,
            self.tr(
                "Each rule chooses an element, an optional group scope, replacement elements, and an amount. Rules run from top to bottom."
            ),
        )
        rules_section.addWidget(self.rules_widget)
        self.rules_label = rules_section.title_label

        self.doping_type_combo = SegmentedControl(parent=self.setting_widget)
        self.doping_type_combo.addItem(
            self.tr("Sampled proportions"), userData="Random"
        )
        self.doping_type_combo.addItem(
            self.tr("Fixed proportions"), userData="Exact"
        )
        self.doping_field = CompactField(
            self.tr("Dopant allocation"),
            self.doping_type_combo,
            self.setting_widget,
            self.tr(
                "This controls how multiple dopants share the selected replacement sites; it does not change how many sites are replaced."
            ),
        )
        self.doping_label = self.doping_field.caption

        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.max_atoms_condition_frame.set_input("", 1, "int")
        self.max_atoms_condition_frame.setRange(1, 999999)
        self.max_atoms_condition_frame.set_input_value([1])
        self.outputs_field = CompactField(
            self.tr("Outputs per input"),
            self.max_atoms_condition_frame,
            self.setting_widget,
            inline=True,
            input_max_width=150,
        )
        self.max_atoms_label = self.outputs_field.caption

        generation_section = InspectorSection(
            self.tr("Output generation"), self.setting_widget
        )
        generation_grid = ResponsiveFormGrid(generation_section)
        generation_grid.add_field(self.doping_field, span=2)
        generation_grid.add_field(self.outputs_field, span=2)
        generation_section.addWidget(generation_grid)
        self.preview_label = CaptionLabel("", generation_section)
        self.preview_label.setWordWrap(True)
        generation_section.addWidget(self.preview_label)

        self.seed_checkbox = CheckBox(
            self.tr("Use fixed random seed"), self.setting_widget
        )
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_field = CompactField(
            self.tr("Random seed"),
            self.seed_frame,
            self.setting_widget,
            inline=True,
            input_max_width=150,
        )
        self.seed_field.hide()
        random_section = InspectorSection(self.tr("Randomness"), self.setting_widget)
        random_section.addWidget(self.seed_checkbox)
        random_section.addWidget(self.seed_field)

        self.settingLayout.addWidget(rules_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(generation_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(random_section, 2, 0, 1, 3)

        self.rules_widget.rulesChanged.connect(self._refresh_preview_and_presentation)
        self.doping_type_combo.currentIndexChanged.connect(
            self._refresh_preview_and_presentation
        )
        self.max_atoms_condition_frame.object_list[0].valueChanged.connect(
            self._refresh_preview_and_presentation
        )
        self.seed_checkbox.toggled.connect(self._update_seed_widgets)
        self.seed_frame.object_list[0].valueChanged.connect(
            self._refresh_preview_and_presentation
        )
        self._update_seed_widgets(False)

    def _update_seed_widgets(self, checked: bool) -> None:
        self.seed_field.setVisible(bool(checked))
        self._refresh_preview_and_presentation()

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

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        self._refresh_preview_and_presentation()

    def set_preview_structure(self, structure) -> None:
        self._input_structure = structure
        self._refresh_preview_and_presentation()

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self._refresh_preview_and_presentation()

    def _refresh_preview_and_presentation(self, *_args) -> None:
        self._preview = None
        self._preview_error = ""
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr(
                    "Load an upstream structure to preview eligible sites and integer replacement bounds."
                )
            )
            self.refresh_compact_presentation()
            return
        try:
            self._preview = self.create_operation().sampling_summary(
                self._input_structure,
                self.get_params(),
            )
        except ValueError as exc:
            self._preview_error = translate_runtime_message(exc)

        if self._preview_error:
            self.preview_label.setText(self._preview_error)
        elif self._preview is not None:
            details = []
            for rule in self._preview.get("rules", ()):
                minimum = int(rule["replacement_min"])
                maximum = int(rule["replacement_max"])
                amount = (
                    str(minimum)
                    if minimum == maximum
                    else self.tr("{minimum}–{maximum}").format(
                        minimum=minimum,
                        maximum=maximum,
                    )
                )
                details.append(
                    self.tr(
                        "Rule {index}: {target}, {eligible} eligible, replace {amount}"
                    ).format(
                        index=rule["rule_index"],
                        target=rule["target"],
                        eligible=rule["eligible_sites"],
                        amount=amount,
                    )
                )
            self.preview_label.setText("\n".join(details))
        self.refresh_compact_presentation()

    def get_summary_text(self) -> str:
        params = self.get_params()
        allocation = (
            self.tr("fixed dopant proportions")
            if params.doping_type == "Exact"
            else self.tr("sampled dopant proportions")
        )
        if self._preview_error:
            return self.tr("Parameters need attention: {error}").format(
                error=self._preview_error
            )
        if self._preview is not None:
            return self.tr("{rules} rule(s) · {allocation} · {outputs}/input").format(
                rules=len(self._preview.get("rules", ())),
                allocation=allocation,
                outputs=params.max_structures,
            )
        return self.tr("{rules} rule(s) · {allocation} · {outputs}/input").format(
            rules=len(params.rules),
            allocation=allocation,
            outputs=params.max_structures,
        )

    def get_guidance_text(self) -> str:
        if self._preview_error:
            return self._preview_error
        params = self.get_params()
        if not params.rules:
            return self.tr("Add at least one complete replacement rule.")
        if self._preview_input_count is None:
            output_text = self.tr("Outputs per input: {outputs}").format(
                outputs=params.max_structures
            )
        else:
            output_text = self.tr(
                "Inputs {inputs} × {per_input}/input = outputs {outputs}"
            ).format(
                inputs=self._preview_input_count,
                per_input=params.max_structures,
                outputs=self._preview_input_count * params.max_structures,
            )
        if self._preview is None:
            return self.tr(
                "Load an upstream structure to validate targets, groups, and replacement bounds. {outputs}"
            ).format(outputs=output_text)
        return self.tr(
            "Preview uses the first input and exact integer bounds. {outputs}"
        ).format(outputs=output_text)

    def create_operation(self):
        """Return the UI-independent random doping operation."""
        return RandomDopingOperation()

    def get_params(self) -> RandomDopingParams:
        """Read random doping parameters from UI controls."""
        return RandomDopingParams(
            rules=self.rules_widget.to_rules(),
            doping_type=str(self.doping_type_combo.currentData()),
            max_structures=int(self.max_atoms_condition_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: RandomDopingParams) -> None:
        """Apply random doping parameters to UI controls."""
        self.rules_widget.from_rules(params.rules)
        self.doping_type_combo.setCurrentIndex(
            self.doping_type_combo.findData(params.doping_type)
        )
        self.max_atoms_condition_frame.set_input_value([int(params.max_structures)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._update_seed_widgets(bool(params.use_seed))

    def process_structure(self, structure):
        """Apply stochastic dopant replacements from UI-independent parameters."""
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
