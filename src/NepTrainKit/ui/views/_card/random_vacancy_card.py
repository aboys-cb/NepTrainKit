"""Card for generating vacancy configurations according to rules."""

import json

from PySide6.QtCore import Qt
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import RandomVacancyOperation, RandomVacancyParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import (
    MakeDataCard,
    SpinBoxUnitInputFrame,
    VacancyRulesWidget,
)


@CardManager.register_card
class RandomVacancyCard(MakeDataCard):
    """Create targeted vacancy structures from element and optional group rules."""

    group = "Defect"

    card_name = "Random Vacancy"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self.setTitle(self.tr("Rule-based Vacancy"))
        self.init_ui()

    def init_ui(self):
        """Build the rule editor, output budget, seed control, and preview."""
        self.setObjectName("random_vacancy_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(5)
        self.settingLayout.setColumnStretch(1, 1)

        self.rules_label = BodyLabel(self.tr("Vacancy rules"), self.setting_widget)
        self.rules_widget = VacancyRulesWidget(self.setting_widget)
        self.rules_label.setToolTip(
            self.tr("Each rule removes one element, optionally within existing group labels")
        )
        self.rules_label.installEventFilter(ToolTipFilter(self.rules_label, 300, ToolTipPosition.TOP))

        self.max_atoms_label = BodyLabel(self.tr("Maximum outputs per input"), self.setting_widget)
        self.max_atoms_condition_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_condition_frame.set_input("", 1)
        self.max_atoms_condition_frame.setRange(1, 10000)
        self.max_atoms_condition_frame.set_input_value([1])
        self.max_atoms_label.setToolTip(
            self.tr("Duplicate vacancy placements are removed, so the actual count can be lower")
        )
        self.max_atoms_label.installEventFilter(ToolTipFilter(self.max_atoms_label, 300, ToolTipPosition.TOP))

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip(self.tr("Enable reproducible random sampling"))
        self.seed_checkbox.installEventFilter(ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP))
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_frame.setAccessibleName(self.tr("Random seed"))

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("randomVacancyPreview")

        self.settingLayout.addWidget(
            self.rules_label,
            0,
            0,
            1,
            1,
            Qt.AlignmentFlag.AlignTop,
        )
        self.settingLayout.addWidget(self.rules_widget, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_condition_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.preview_label, 3, 0, 1, 3)

        self.rules_widget.rulesChanged.connect(self._refresh_preview)
        self.rules_widget.rulesChanged.connect(self._update_tab_order)
        self.max_atoms_condition_frame.object_list[0].valueChanged.connect(
            self._refresh_preview
        )
        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self._on_seed_changed()
        self._refresh_preview()
        self._update_tab_order()

    def _on_seed_changed(self) -> None:
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())
        self._update_tab_order()

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
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if not hasattr(self, "preview_label"):
            return
        rules = self.rules_widget.to_rules()
        if not rules:
            self.preview_label.setText(
                "⚠ " + self.tr("Add an element to the vacancy rule before running.")
            )
            return
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr("Load an upstream structure to preview matched atom counts.")
            )
            return

        try:
            operation = self.create_operation()
            summary = operation.rule_match_summary(
                self._input_structure,
                rules,
            )
            requested_outputs = int(
                self.max_atoms_condition_frame.get_input_value()[0]
            )
            maximum_outputs = operation.maximum_unique_outputs(
                self._input_structure,
                rules,
                requested_outputs,
            )
        except ValueError as exc:
            self.preview_label.setText(
                "⚠ " + self.tr("Preview unavailable: {error}").format(error=str(exc))
            )
            return

        parts = []
        for item in summary:
            target = item["element"]
            if item["groups"]:
                target += "/" + ",".join(item["groups"])
            count_text = (
                str(item["count_min"])
                if item["count_min"] == item["count_max"]
                else f"{item['count_min']}–{item['count_max']}"
            )
            parts.append(
                self.tr("{target}: {matches} matches, remove {count}").format(
                    target=target,
                    matches=item["candidate_count"],
                    count=count_text,
                )
            )
        self.preview_label.setText(
            self.tr("First input preview: {rules} · up to {outputs} unique outputs").format(
                rules="; ".join(parts),
                outputs=maximum_outputs,
            )
        )

    def _update_tab_order(self) -> None:
        widgets = []
        for item in self.rules_widget.rule_items():
            widgets.extend(
                [
                    item.element_edit,
                    item.group_edit,
                    item.count_mode_combo,
                ]
            )
            active_count_frame = (
                item.fixed_count_frame
                if item.count_mode_combo.currentData() == "fixed"
                else item.count_range_frame
            )
            widgets.extend(active_count_frame.object_list)
            widgets.append(item.delete_button)
        widgets.extend(
            [
                self.rules_widget.add_button,
                self.max_atoms_condition_frame.object_list[0],
                self.seed_checkbox,
            ]
        )
        if self.seed_frame.isEnabled():
            widgets.append(self.seed_frame.object_list[0])
        self.tab_order_widgets = widgets

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
        self._on_seed_changed()
        self._refresh_preview()
        self._update_tab_order()

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
                max_structures=int(
                    self._legacy_scalar(data_dict.get("max_atoms_condition", 1), 1)
                ),
                use_seed=data_dict.get("use_seed", False),
                seed=int(self._legacy_scalar(data_dict.get("seed", 0), 0)),
            )
        self.set_params(params)

    @staticmethod
    def _legacy_scalar(value, default):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value if value is not None else default
