"""Card for random atomic packing inside an existing cell."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import RandomPackingOperation, RandomPackingParams
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    KeyValueTableInput,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class RandomPackingCard(MakeDataCard):
    """Rebuild all atom positions in each input cell under distance constraints."""

    group = "Structure"
    card_name = "Random Packing"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._preview_input_count = None
        self.setTitle(self.tr("Random Packing"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("random_packing_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.setColumnStretch(1, 1)

        self.structures_frame = self._integer_frame(1, 100000, 1)
        self.composition_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.composition_mode_combo,
            [("input", "Input composition"), ("manual", "Manual counts")],
        )
        set_combo_value(self.composition_mode_combo, "input")
        self.composition_mode_combo.setMinimumWidth(0)
        self.composition_mode_combo.setFixedHeight(28)
        self.composition_edit = KeyValueTableInput(self.tr("Element"), self.tr("Atom count"), self.setting_widget)

        self.min_distance_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.min_distance_frame.set_input("Å", 1, "float")
        self.min_distance_frame.setRange(0.01, 100.0)
        self.min_distance_frame.setDecimals(3)
        self.min_distance_frame.setSingleStep(0.05)
        self.min_distance_frame.set_input_value([1.5])

        self.strict_checkbox = CheckBox(self.tr("Require all requested outputs"), self.setting_widget)
        self.strict_checkbox.setChecked(True)
        self.budget_frame = self._integer_frame(1, 10_000_000, 10_000)
        self.advanced_checkbox = CheckBox(self.tr("Advanced packing controls"), self.setting_widget)
        self.advanced_checkbox.setChecked(False)
        self.pair_distance_edit = KeyValueTableInput(
            self.tr("Element pair"), self.tr("Minimum distance (Å)"), self.setting_widget
        )
        self.attempts_frame = self._integer_frame(1, 1_000_000, 500)

        self.seed_checkbox = CheckBox(self.tr("Use fixed seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = self._integer_frame(0, 2**31 - 1, 0)
        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        output_section = InspectorSection(
            self.tr("Outputs and composition"),
            self.setting_widget,
            self.tr("Each input supplies the cell and boundary conditions; all atom positions are rebuilt."),
        )
        output_grid = ResponsiveFormGrid(output_section, two_column_threshold=500)
        self.structures_field = CompactField(
            self.tr("Outputs per input"),
            self.structures_frame,
            output_section,
            self.tr("Exact when all outputs are required; otherwise this is the maximum."),
            inline=True,
            input_max_width=150,
        )
        self.composition_mode_field = CompactField(
            self.tr("Atom composition"), self.composition_mode_combo, output_section
        )
        self.composition_field = CompactField(
            self.tr("Manual integer counts"),
            self.composition_edit,
            output_section,
            self.tr("For example Fe:32 and O:64. The input atom list is not retained."),
        )
        output_grid.add_field(self.structures_field)
        output_grid.add_field(self.composition_mode_field)
        output_grid.add_field(self.composition_field, span=2)
        output_section.addWidget(output_grid)

        constraint_section = InspectorSection(
            self.tr("Packing constraints"),
            self.setting_widget,
            self.tr("Every accepted pair must satisfy its minimum-image distance threshold."),
        )
        constraint_grid = ResponsiveFormGrid(constraint_section, two_column_threshold=500)
        self.min_distance_field = CompactField(
            self.tr("Global minimum distance"),
            self.min_distance_frame,
            constraint_section,
            self.tr("A starting constraint, not a chemistry-specific bond-length guarantee."),
        )
        self.min_distance_frame.setMaximumWidth(220)
        self.budget_field = CompactField(
            self.tr("Generated atom budget/input"),
            self.budget_frame,
            constraint_section,
            self.tr("Outputs per input × atoms per output must not exceed this safety limit."),
        )
        self.budget_frame.setMaximumWidth(220)
        constraint_grid.add_field(self.min_distance_field, span=2)
        constraint_grid.add_field(self.budget_field, span=2)
        constraint_section.addWidget(constraint_grid)
        constraint_section.addWidget(self.strict_checkbox)
        constraint_section.addWidget(self.advanced_checkbox)

        self.advanced_section = InspectorSection(self.tr("Advanced packing controls"), self.setting_widget)
        advanced_grid = ResponsiveFormGrid(self.advanced_section)
        self.pair_distance_field = CompactField(
            self.tr("Pair-specific minimum distances"),
            self.pair_distance_edit,
            self.advanced_section,
            self.tr("Rules such as Fe-O:1.8 override the global threshold for that pair."),
        )
        self.attempts_field = CompactField(
            self.tr("Maximum attempts per atom"),
            self.attempts_frame,
            self.advanced_section,
            self.tr("Raise this only when a feasible dense packing needs more random trials."),
            inline=True,
            input_max_width=150,
        )
        advanced_grid.add_field(self.pair_distance_field, span=2)
        advanced_grid.add_field(self.attempts_field, span=2)
        self.advanced_section.addWidget(advanced_grid)

        random_section = InspectorSection(self.tr("Randomness"), self.setting_widget)
        random_section.addWidget(self.seed_checkbox)
        self.seed_field = CompactField(
            self.tr("Seed"), self.seed_frame, random_section, inline=True, input_max_width=150
        )
        random_section.addWidget(self.seed_field)
        preview_section = InspectorSection(self.tr("Exact size preview"), self.setting_widget)
        preview_section.addWidget(self.preview_label)

        for row, section in enumerate(
            (output_section, constraint_section, self.advanced_section, random_section, preview_section)
        ):
            self.settingLayout.addWidget(section, row, 0, 1, 3)

        self.composition_mode_combo.currentIndexChanged.connect(self._parameters_changed)
        self.composition_edit.editingFinished.connect(self._composition_edited)
        self.pair_distance_edit.editingFinished.connect(self._parameters_changed)
        for checkbox in (self.advanced_checkbox, self.strict_checkbox, self.seed_checkbox):
            checkbox.stateChanged.connect(self._parameters_changed)
        for frame in (
            self.structures_frame,
            self.min_distance_frame,
            self.budget_frame,
            self.attempts_frame,
            self.seed_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._parameters_changed)
        self._parameters_changed()

    def _integer_frame(self, minimum: int, maximum: int, value: int, *, unit: str = ""):
        frame = SpinBoxUnitInputFrame(self.setting_widget)
        frame.set_input(unit, 1, "int")
        frame.setRange(minimum, maximum)
        frame.set_input_value([value])
        return frame

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
        if self._preview_input_count is None:
            self._preview_input_count = self._dataset_count(dataset) or None
        self._refresh_preview()

    def set_preview_structure(self, structure) -> None:
        self._input_structure = structure
        self._refresh_preview()

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def _parameters_changed(self, *_args) -> None:
        self.composition_field.setVisible(combo_value(self.composition_mode_combo) == "manual")
        self.advanced_section.setVisible(self.advanced_checkbox.isChecked())
        fixed_seed = self.seed_checkbox.isChecked()
        self.seed_field.setVisible(fixed_seed)
        self.seed_frame.setEnabled(fixed_seed)
        self._update_tab_order()
        self._refresh_preview()

    def _composition_edited(self) -> None:
        if self.composition_edit.text().strip():
            set_combo_value(self.composition_mode_combo, "manual")
        self._parameters_changed()

    def _refresh_preview(self) -> None:
        if not hasattr(self, "preview_label"):
            return
        self.refresh_compact_presentation()
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr("Load an upstream structure to validate its cell and preview exact atom counts.")
            )
            return
        try:
            plan = self.create_operation().plan(self._input_structure, self.get_params())
        except ValueError as exc:
            self.preview_label.setText(
                "⚠ " + self.tr("Preview unavailable: {error}").format(error=translate_runtime_message(exc))
            )
            return
        contract = self.tr("exact") if self.strict_checkbox.isChecked() else self.tr("at most")
        self.preview_label.setText(
            self.tr(
                "First input: {atoms} atoms/output × {outputs} = {total} generated atoms ({contract}) · budget {budget}"
            ).format(
                atoms=plan.atoms_per_output,
                outputs=plan.structures,
                total=plan.requested_generated_atoms,
                contract=contract,
                budget=plan.max_generated_atoms,
            )
        )

    def _update_tab_order(self) -> None:
        widgets = [
            *self.structures_frame.object_list,
            self.composition_mode_combo,
            self.composition_edit,
            *self.min_distance_frame.object_list,
            *self.budget_frame.object_list,
            self.strict_checkbox,
            self.advanced_checkbox,
            self.pair_distance_edit,
            *self.attempts_frame.object_list,
            self.seed_checkbox,
            *self.seed_frame.object_list,
        ]
        self.tab_order_widgets = [widget for widget in widgets if widget.isEnabled() and not widget.isHidden()]
        for previous, current in zip(self.tab_order_widgets, self.tab_order_widgets[1:]):
            QWidget.setTabOrder(previous, current)

    def create_operation(self):
        return RandomPackingOperation()

    def get_params(self) -> RandomPackingParams:
        composition_mode = combo_value(self.composition_mode_combo)
        composition = self.composition_edit.text() if composition_mode == "manual" else ""
        return RandomPackingParams(
            structures=int(self.structures_frame.get_input_value()[0]),
            composition=composition,
            composition_mode=composition_mode,
            min_distance=float(self.min_distance_frame.get_input_value()[0]),
            pair_min_distances=self.pair_distance_edit.text(),
            max_attempts_per_atom=int(self.attempts_frame.get_input_value()[0]),
            strict_mode=self.strict_checkbox.isChecked(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
            max_generated_atoms=int(self.budget_frame.get_input_value()[0]),
        )

    def set_params(self, params: RandomPackingParams) -> None:
        self.structures_frame.set_input_value([int(params.structures)])
        self.composition_edit.setText(params.composition)
        set_combo_value(self.composition_mode_combo, params.composition_mode)
        self.min_distance_frame.set_input_value([float(params.min_distance)])
        self.pair_distance_edit.setText(params.pair_min_distances)
        self.attempts_frame.set_input_value([int(params.max_attempts_per_atom)])
        self.strict_checkbox.setChecked(bool(params.strict_mode))
        self.advanced_checkbox.setChecked(
            bool(params.pair_min_distances.strip()) or int(params.max_attempts_per_atom) != 500
        )
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.budget_frame.set_input_value([int(params.max_generated_atoms)])
        self._parameters_changed()

    def get_summary_text(self) -> str:
        params = self.get_params()
        prefix = "" if params.strict_mode else "≤"
        if self._input_structure is not None:
            try:
                plan = self.create_operation().plan(self._input_structure, params)
            except ValueError:
                return self.tr("Check packing parameters")
            return self.tr("{outputs}/input · {atoms} atoms/output").format(
                outputs=f"{prefix}{plan.structures}", atoms=plan.atoms_per_output
            )
        composition = (
            self.tr("input composition") if params.composition_mode == "input" else self.tr("manual composition")
        )
        return self.tr("{outputs}/input · {composition}").format(
            outputs=f"{prefix}{params.structures}", composition=composition
        )

    def get_guidance_text(self) -> str:
        params = self.get_params()
        inputs = self._preview_input_count
        if inputs is None or inputs <= 0:
            count_text = (
                self.tr("Outputs/input: {count}.").format(count=params.structures)
                if params.strict_mode
                else self.tr("Outputs/input: at most {count}.").format(count=params.structures)
            )
        elif params.strict_mode:
            count_text = self.tr("Inputs {inputs} × {per_input} outputs/input = outputs {total}.").format(
                inputs=inputs, per_input=params.structures, total=inputs * params.structures
            )
        else:
            count_text = self.tr(
                "Inputs {inputs} × at most {per_input} outputs/input = at most {total} outputs."
            ).format(inputs=inputs, per_input=params.structures, total=inputs * params.structures)
        if self._input_structure is None:
            return count_text + " " + self.tr("Load an upstream structure for exact atom counts.")
        try:
            plan = self.create_operation().plan(self._input_structure, params)
        except ValueError as exc:
            return count_text + " " + translate_runtime_message(exc)
        return (
            count_text
            + " "
            + self.tr(
                "First input requests {total} generated atoms; site arrays such as spin and group are not carried to outputs."
            ).format(total=plan.requested_generated_atoms)
        )

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw = dict(data_dict.get("params") or {})
        defaults = RandomPackingParams()
        values = {name: raw.get(name, getattr(defaults, name)) for name in defaults.__dataclass_fields__}
        if "composition_mode" not in raw:
            values["composition_mode"] = "manual" if str(values["composition"]).strip() else "input"
        params = RandomPackingParams(**values)
        self.set_params(params)
