"""Card for integer-authoritative finite-cell alloy occupancies."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    FluentIcon,
    MessageBox,
    PlainTextEdit,
    PushButton,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import FiniteCellAlloyOccupancyOperation, FiniteCellAlloyOccupancyParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)
from NepTrainKit.ui.widgets.alloy_site_rules import AlloySiteRulesEditor


@CardManager.register_card
class FiniteCellAlloyOccupancyCard(MakeDataCard):
    """Assign allowed elements using feasible integer counts on each site set."""

    group = "Alloy"
    card_name = "Finite-Cell Alloy Occupancy"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._input_counts: dict[str, int] | None = None
        self._refreshing = False
        self._rules_are_auto_managed = True
        self._applying_auto_rules = False
        self._allow_legacy_fraction_weights = False
        self.setTitle(self.tr("Finite-Cell Alloy Occupancy"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("finite_cell_alloy_occupancy_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.setColumnStretch(1, 1)

        self.rules_editor = AlloySiteRulesEditor(self.setting_widget)
        self.rules_editor.set_replacement_confirmation(self._confirm_rule_replacement)
        self.rules_editor.changed.connect(self._on_rules_changed)
        self.rules_editor.layoutChanged.connect(self._update_tab_order)

        self.auto_match_label = CaptionLabel("", self.setting_widget)
        self.auto_match_label.setWordWrap(True)
        self.auto_match_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.auto_match_label.hide()

        self.arrangements_label = BodyLabel(
            self.tr("Arrangements per composition"),
            self.setting_widget,
        )
        self.arrangements_label.hide()
        self.arrangements_frame = SpinBoxUnitInputFrame(self)
        self.arrangements_frame.set_input("", 1, "int")
        self.arrangements_frame.setRange(1, 999999)
        self.arrangements_frame.set_input_value([1])

        self.seed_checkbox = CheckBox(self.tr("Use fixed seed"), self.setting_widget)
        self.seed_checkbox.setChecked(True)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])

        self.max_outputs_label = BodyLabel(
            self.tr("Max outputs per input"),
            self.setting_widget,
        )
        self.max_outputs_label.hide()
        self.max_outputs_frame = SpinBoxUnitInputFrame(self)
        self.max_outputs_frame.set_input("", 1, "int")
        self.max_outputs_frame.setRange(1, 999999)
        self.max_outputs_frame.set_input_value([200])

        self.estimate_label = BodyLabel("", self.setting_widget)
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.estimate_label.setObjectName("alloyEstimateLabel")

        self.advanced_button = PushButton(
            FluentIcon.CODE,
            self.tr("Advanced: view or paste JSON"),
            self.setting_widget,
        )
        self.advanced_button.setCheckable(True)
        self.advanced_button.setAccessibleName(self.tr("Advanced: view or paste JSON"))
        self.advanced_button.setFixedHeight(28)

        self.advanced_widget = QWidget(self.setting_widget)
        advanced_layout = QVBoxLayout(self.advanced_widget)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(6)
        self.advanced_json_edit = PlainTextEdit(self.advanced_widget)
        self.advanced_json_edit.setMinimumHeight(90)
        self.advanced_json_edit.setPlaceholderText(
            self.tr("Paste the existing site_rules JSON format here.")
        )
        advanced_actions = QHBoxLayout()
        self.apply_json_button = PushButton(self.tr("Apply JSON"), self.advanced_widget)
        self.apply_json_button.setFixedHeight(28)
        self.site_rules_copy_button = PushButton(
            FluentIcon.COPY,
            self.tr("Copy JSON"),
            self.advanced_widget,
        )
        self.site_rules_copy_button.setFixedHeight(28)
        advanced_actions.addWidget(self.apply_json_button)
        advanced_actions.addWidget(self.site_rules_copy_button)
        advanced_actions.addStretch(1)
        self.json_error_label = CaptionLabel("", self.advanced_widget)
        self.json_error_label.setWordWrap(True)
        self.json_error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        advanced_layout.addWidget(self.advanced_json_edit)
        advanced_layout.addLayout(advanced_actions)
        advanced_layout.addWidget(self.json_error_label)
        self.advanced_widget.hide()

        rules_section = InspectorSection(
            self.tr("Site rules"),
            self.setting_widget,
            self.tr("Choose the site partition, then edit allowed elements and integer-feasible ranges."),
        )
        rules_section.addWidget(self.rules_editor)
        rules_section.addWidget(self.auto_match_label)

        generation_section = InspectorSection(self.tr("Generation"), self.setting_widget)
        generation_grid = ResponsiveFormGrid(generation_section)
        arrangements_field = CompactField(
            self.tr("Arrangements per composition"),
            self.arrangements_frame,
            generation_section,
        )
        seed_row = QWidget(generation_section)
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(6)
        seed_layout.addWidget(self.seed_checkbox)
        seed_layout.addWidget(self.seed_frame, 1)
        seed_field = CompactField(self.tr("Reproducibility"), seed_row, generation_section)
        max_outputs_field = CompactField(
            self.tr("Max outputs per input"),
            self.max_outputs_frame,
            generation_section,
        )
        generation_grid.add_field(arrangements_field)
        generation_grid.add_field(max_outputs_field)
        generation_grid.add_field(seed_field, span=2)
        generation_section.addWidget(generation_grid)
        generation_section.addWidget(self.estimate_label)

        advanced_section = InspectorSection(self.tr("Advanced"), self.setting_widget)
        advanced_section.addWidget(self.advanced_button)
        advanced_section.addWidget(self.advanced_widget)

        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.addWidget(rules_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(generation_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(advanced_section, 2, 0, 1, 3)

        self.seed_checkbox.stateChanged.connect(self._on_seed_changed)
        self.arrangements_frame.object_list[0].valueChanged.connect(self._refresh_validation_and_estimate)
        self.max_outputs_frame.object_list[0].valueChanged.connect(self._refresh_validation_and_estimate)
        self.advanced_button.toggled.connect(self._toggle_advanced)
        self.apply_json_button.clicked.connect(self.apply_advanced_json)
        self.site_rules_copy_button.clicked.connect(self.copy_site_rules_json)
        self._on_seed_changed()
        self._sync_advanced_json()
        self._refresh_validation_and_estimate()
        self._update_tab_order()

    def _on_seed_changed(self) -> None:
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())

    def _toggle_advanced(self, visible: bool) -> None:
        if visible:
            self._sync_advanced_json()
        self.advanced_widget.setVisible(bool(visible))
        self._update_tab_order()

    def _on_rules_changed(self) -> None:
        if self._refreshing:
            return
        if not self._applying_auto_rules:
            if self._rules_are_auto_managed and self._matching_placeholder_template():
                self._auto_match_rules_to_input()
                return
            self._allow_legacy_fraction_weights = False
            self._rules_are_auto_managed = False
            self.auto_match_label.hide()
        self.json_error_label.clear()
        self._sync_advanced_json()
        self._refresh_validation_and_estimate()
        self._update_tab_order()

    def _matching_placeholder_template(self) -> bool:
        if not self._input_counts or self._input_structure is None:
            return False
        rules = self.rules_editor.to_rules()
        if set(rules) != set(self._input_counts):
            return False
        return all(
            rule.get("elements") == ["X"]
            and rule.get("mode") == "fixed_fraction"
            and rule.get("composition") == {"X": 1.0}
            for rule in rules.values()
        )

    def _confirm_rule_replacement(self) -> bool:
        if self._rules_are_auto_managed:
            return True
        box = MessageBox(
            self.tr("Replace current site rules?"),
            self.tr(
                "Changing the site partition or applying a rule template will discard "
                "your current site-set and element edits."
            ),
            self,
        )
        box.yesButton.setText(self.tr("Replace rules"))
        box.cancelButton.setText(self.tr("Keep current rules"))
        box.exec()
        return box.result() != 0

    def _site_rules_text(self) -> str:
        return json.dumps(
            self.rules_editor.to_rules(),
            sort_keys=True,
            separators=(",", ":"),
        )

    def _sync_advanced_json(self) -> None:
        if not hasattr(self, "advanced_json_edit"):
            return
        text = json.dumps(self.rules_editor.to_rules(), indent=2, sort_keys=True)
        if self.advanced_json_edit.toPlainText() != text:
            self.advanced_json_edit.setPlainText(text)

    def apply_rule_json(self, text: str) -> bool:
        """Apply site-rules JSON transactionally, preserving the current rules on failure."""
        attempted_text = str(text or "")
        previous = self.rules_editor.to_rules()
        try:
            parsed = json.loads(attempted_text)
            if not isinstance(parsed, Mapping):
                raise ValueError(self.tr("site_rules must be a non-empty JSON object."))
            self.rules_editor.from_rules(parsed)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            try:
                self.rules_editor.from_rules(previous)
            except ValueError:  # pragma: no cover - previous rules came from the editor
                pass
            self.json_error_label.setText(
                "⚠ " + self.tr("JSON was not applied: {error}").format(error=str(exc))
            )
            self.json_error_label.show()
            self.advanced_json_edit.setPlainText(attempted_text)
            return False
        self.json_error_label.clear()
        self.json_error_label.hide()
        self._sync_advanced_json()
        return True

    def apply_advanced_json(self) -> bool:
        return self.apply_rule_json(self.advanced_json_edit.toPlainText())

    def copy_site_rules_json(self) -> None:
        from PySide6.QtWidgets import QApplication

        text = self._site_rules_text()
        QApplication.clipboard().setText(text)
        self.advanced_json_edit.setPlainText(json.dumps(json.loads(text), indent=2, sort_keys=True))

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        self._input_counts = self._site_counts(self._input_structure)
        self._auto_match_rules_to_input()
        self.rules_editor.set_input_counts(self._input_counts)
        self._refresh_validation_and_estimate()

    @staticmethod
    def _rules_from_structure(structure) -> dict[str, dict[str, object]]:
        symbols = np.asarray(structure.get_chemical_symbols(), dtype=str)
        if "sublattice" not in structure.arrays:
            site_indices = {"all": np.arange(len(structure), dtype=int)}
        else:
            raw = np.asarray(structure.arrays["sublattice"], dtype=str)
            labels = sorted({str(value).strip() for value in raw})
            site_indices = {
                label: np.nonzero(raw == label)[0].astype(int)
                for label in labels
            }

        rules: dict[str, dict[str, object]] = {}
        for label, indices in site_indices.items():
            site_symbols = symbols[indices].tolist()
            elements = list(dict.fromkeys(str(symbol) for symbol in site_symbols))
            site_count = len(site_symbols)
            composition = {
                element: site_symbols.count(element) / site_count
                for element in elements
            }
            rules[label] = {
                "elements": elements,
                "mode": "fixed_fraction",
                "composition": composition,
            }
        return rules

    def _auto_match_rules_to_input(self) -> None:
        """Match untouched rules to the input partition and its current elements."""
        if (
            not self._rules_are_auto_managed
            or not self._input_counts
            or self._input_structure is None
            or any(count <= 0 for count in self._input_counts.values())
        ):
            self.auto_match_label.hide()
            return

        target_rules = self._rules_from_structure(self._input_structure)
        ordered_labels = tuple(target_rules)
        if self.rules_editor.to_rules() != target_rules:
            self._applying_auto_rules = True
            try:
                self.rules_editor.from_rules(target_rules)
            finally:
                self._applying_auto_rules = False

        self.auto_match_label.setText(
            self.tr(
                "Automatically matched site sets and current elements from input: {labels}."
            ).format(
                labels=", ".join(ordered_labels)
            )
        )
        self.auto_match_label.show()

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
    def _site_counts(structure) -> dict[str, int] | None:
        if structure is None:
            return None
        if "sublattice" not in structure.arrays:
            return {"all": len(structure)}
        raw = np.asarray(structure.arrays["sublattice"], dtype=str)
        labels = sorted({str(value).strip() for value in raw})
        return {label: int(np.count_nonzero(raw == label)) for label in labels}

    def _params_from_controls(self) -> FiniteCellAlloyOccupancyParams:
        return FiniteCellAlloyOccupancyParams(
            site_rules=self._site_rules_text(),
            arrangements_per_composition=int(self.arrangements_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
            max_outputs=int(self.max_outputs_frame.get_input_value()[0]),
        )

    def _refresh_validation_and_estimate(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        try:
            self.rules_editor.set_input_counts(self._input_counts)
            errors = self.rules_editor.validation_errors(self._input_counts)
            if self._input_structure is None:
                self.estimate_label.setText(
                    self.tr(
                        "Load an upstream structure to estimate outputs from its first structure."
                    )
                )
                return
            if errors:
                self.estimate_label.setText(
                    self.tr("Fix the highlighted site-rule errors to calculate a feasible output estimate.")
                )
                return

            try:
                estimate = self.create_operation().estimate(
                    self._input_structure,
                    self._params_from_controls(),
                )
            except ValueError as exc:
                message = str(exc)
                friendly = self._show_operation_error_near_site(message)
                self.estimate_label.setText(
                    self.tr("No feasible integer composition: {error}").format(error=friendly)
                )
                return

            count_map = dict(estimate.site_counts)
            ordered_labels = [
                editor.label_edit.text().strip()
                for editor in self.rules_editor.site_editors
                if editor.label_edit.text().strip() in count_map
            ]
            counts = ", ".join(f"{label}={count_map[label]}" for label in ordered_labels)
            theoretical = estimate.composition_count * estimate.arrangements_per_composition
            truncated = theoretical > estimate.max_outputs
            self.estimate_label.setText(
                self.tr(
                    "First input sites: {counts} · {compositions} feasible integer compositions · "
                    "{arrangements} arrangements requested per composition\n"
                    "Output upper-bound estimate: {theoretical} · Max outputs per input: {maximum} · "
                    "{truncation}"
                ).format(
                    counts=counts,
                    compositions=estimate.composition_count,
                    arrangements=estimate.arrangements_per_composition,
                    theoretical=theoretical,
                    maximum=estimate.max_outputs,
                    truncation=(
                        self.tr(
                            "Will truncate; different compositions are covered before extra arrangements."
                        )
                        if truncated
                        else self.tr("Within the output limit.")
                    ),
                )
            )
        finally:
            self._refreshing = False

    def _show_operation_error_near_site(self, message: str) -> str:
        match = re.search(r"site set ['\"]([^'\"]+)['\"]", message)
        if not match:
            self.rules_editor.set_status_errors([message])
            return message
        label = match.group(1)
        no_solution = re.search(r"no integer count solution for (\d+) sites", message)
        friendly = message
        if no_solution:
            friendly = self.tr(
                "Constraints for site set {label} have no integer count solution for {count} sites."
            ).format(label=label, count=no_solution.group(1))
        for editor in self.rules_editor.site_editors:
            if editor.label_edit.text().strip() == label:
                existing = editor.validation_errors()
                editor.set_errors(existing + [friendly])
                return friendly
        self.rules_editor.set_status_errors([friendly])
        return friendly

    def _update_tab_order(self) -> None:
        widgets = self.rules_editor.tab_widgets()
        widgets.extend(
            [
                self.arrangements_frame.object_list[0],
                self.seed_checkbox,
                self.seed_frame.object_list[0],
                self.max_outputs_frame.object_list[0],
                self.advanced_button,
            ]
        )
        if self.advanced_widget.isVisible():
            widgets.extend(
                [
                    self.advanced_json_edit,
                    self.apply_json_button,
                    self.site_rules_copy_button,
                ]
            )
        self.tab_order_widgets = [
            widget for widget in widgets if widget.isEnabled() and not widget.isHidden()
        ]
        for previous, current in zip(self.tab_order_widgets, self.tab_order_widgets[1:]):
            QWidget.setTabOrder(previous, current)

    def create_operation(self):
        return FiniteCellAlloyOccupancyOperation(
            require_normalized_fixed_fractions=not self._allow_legacy_fraction_weights
        )

    def get_params(self) -> FiniteCellAlloyOccupancyParams:
        return self._params_from_controls()

    def set_params(self, params: FiniteCellAlloyOccupancyParams) -> None:
        self._allow_legacy_fraction_weights = self.apply_rule_json(params.site_rules)
        self.arrangements_frame.set_input_value([int(params.arrangements_per_composition)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.setEnabled(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.max_outputs_frame.set_input_value([int(params.max_outputs)])
        self._refresh_validation_and_estimate()
        self._update_tab_order()

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw = dict(data_dict.get("params") or {})
        params = FiniteCellAlloyOccupancyParams(**raw) if raw else FiniteCellAlloyOccupancyParams()
        self.set_params(params)
