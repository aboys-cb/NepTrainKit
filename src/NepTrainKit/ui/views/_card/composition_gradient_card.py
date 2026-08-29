"""Card for applying a composition gradient along a lattice coordinate."""

from __future__ import annotations

import numpy as np
from qfluentwidgets import CheckBox, ComboBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import (
    CompositionGradientOperation,
    CompositionGradientParams,
)
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import (
    add_translated_items,
    combo_value,
    set_combo_value,
)
from NepTrainKit.ui.widgets import (
    CompactField,
    CompositionPathTableInput,
    ElementLineEdit,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class CompositionGradientCard(MakeDataCard):
    """Assign species along equal-count groups ordered by lattice coordinate."""

    group = "Alloy"
    card_name = "Composition Gradient"
    description = (
        "Build a one-dimensional composition transition along lattice a, b, or c "
        "without moving atoms."
    )
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Composition Path"))
        self._input_structure = None
        self.init_ui()

    def init_ui(self):
        self.setObjectName("composition_gradient_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        # Hidden text fields retain the existing programmatic card interface.
        self.elements_edit = LineEdit(self.setting_widget)
        self.start_edit = LineEdit(self.setting_widget)
        self.end_edit = LineEdit(self.setting_widget)
        self.elements_edit.setText("Ni,Co")
        self.start_edit.setText("Ni:1,Co:0")
        self.end_edit.setText("Ni:0,Co:1")
        for widget in (self.elements_edit, self.start_edit, self.end_edit):
            widget.hide()

        self.composition_table = CompositionPathTableInput(self.setting_widget)
        self.composition_table.set_values(
            self.elements_edit.text(), self.start_edit.text(), self.end_edit.text()
        )
        composition_section = InspectorSection(
            self.tr("Gradient composition"),
            self.setting_widget,
            self.tr("Each row gives one output element and its low/high-end ratio."),
        )
        composition_section.addWidget(self.composition_table)

        self.axis_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.axis_combo,
            [("a", "Lattice a"), ("b", "Lattice b"), ("c", "Lattice c")],
        )
        set_combo_value(self.axis_combo, "a")
        self.axis_field = CompactField(
            self.tr("Gradient direction"), self.axis_combo, self.setting_widget
        )

        self.bins_frame = self._integer_frame(2, 10000, 8)
        self.bins_field = CompactField(
            self.tr("Requested equal-count groups"),
            self.bins_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )
        self.definition_section = InspectorSection(
            self.tr("Gradient definition"), self.setting_widget
        )
        definition_grid = ResponsiveFormGrid(
            self.definition_section, two_column_threshold=520
        )
        definition_grid.add_field(self.axis_field)
        definition_grid.add_field(self.bins_field)
        self.definition_section.addWidget(definition_grid)

        self.scope_combo = ComboBox(self.setting_widget)
        self.scope_combo.addItem(self.tr("All atoms"), userData="all")
        self.scope_combo.addItem(
            self.tr("Listed existing elements"), userData="listed"
        )
        self.scope_field = CompactField(
            self.tr("Eligible sites"), self.scope_combo, self.setting_widget
        )
        self.target_edit = ElementLineEdit(self.setting_widget, multiple=True)
        self.target_edit.setPlaceholderText(self.tr("For example: Ni,Co"))
        self.target_field = CompactField(
            self.tr("Existing elements to replace"),
            self.target_edit,
            self.setting_widget,
            self.tr("Elements not listed here remain unchanged."),
        )
        self.target_field.hide()
        scope_section = InspectorSection(
            self.tr("Site scope"),
            self.setting_widget,
            self.tr(
                "The selected sites are sorted by lattice fractional coordinate; atoms are not moved."
            ),
        )
        scope_section.addWidget(self.scope_field)
        scope_section.addWidget(self.target_field)

        self.samples_frame = self._integer_frame(1, 10000, 1)
        self.samples_field = CompactField(
            self.tr("Random samples per input"),
            self.samples_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )
        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = self._integer_frame(0, 2**31 - 1, 0)
        self.seed_field = CompactField(
            self.tr("Random seed"),
            self.seed_frame,
            self.setting_widget,
            inline=True,
            input_max_width=144,
        )
        self.seed_field.hide()
        sampling_section = InspectorSection(
            self.tr("Sampling"),
            self.setting_widget,
            self.tr(
                "Samples keep the same integer composition in each group and randomize site assignments."
            ),
        )
        sampling_section.addWidget(self.samples_field)
        sampling_section.addWidget(self.seed_checkbox)
        sampling_section.addWidget(self.seed_field)

        self.settingLayout.addWidget(composition_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(self.definition_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(scope_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(sampling_section, 3, 0, 1, 3)

        self.elements_edit.textChanged.connect(self._sync_legacy_composition_fields)
        self.start_edit.textChanged.connect(self._sync_legacy_composition_fields)
        self.end_edit.textChanged.connect(self._sync_legacy_composition_fields)
        self.composition_table.table.itemChanged.connect(
            self.refresh_compact_presentation
        )
        self.composition_table.add_button.clicked.connect(
            self.refresh_compact_presentation
        )
        self.composition_table.remove_button.clicked.connect(
            self.refresh_compact_presentation
        )
        self.axis_combo.currentIndexChanged.connect(self._update_direction_hint)
        self.scope_combo.currentIndexChanged.connect(self._update_scope)
        self.target_edit.textChanged.connect(self.refresh_compact_presentation)
        for frame in (self.bins_frame, self.samples_frame, self.seed_frame):
            for control in frame.object_list:
                control.valueChanged.connect(self.refresh_compact_presentation)
        self.seed_checkbox.toggled.connect(self._update_seed_visibility)

        self._update_direction_hint()
        self._update_scope()
        self._update_seed_visibility(False)

    def _integer_frame(self, minimum: int, maximum: int, value: int):
        frame = SpinBoxUnitInputFrame(self)
        frame.set_input("", 1, "int")
        frame.setRange(minimum, maximum)
        frame.set_input_value([value])
        return frame

    def _sync_legacy_composition_fields(self, *_args) -> None:
        self.composition_table.set_values(
            self.elements_edit.text(), self.start_edit.text(), self.end_edit.text()
        )

    def _update_direction_hint(self, *_args) -> None:
        direction = combo_value(self.axis_combo, "a")
        self.definition_section.description_label.setText(
            self.tr(
                "Sorts eligible sites by fractional {direction}, then splits them into groups with nearly equal atom counts. A periodic {direction} direction joins the ends and creates a second composition jump."
            ).format(direction=direction)
        )
        self.definition_section.description_label.show()
        self.refresh_compact_presentation()

    def _update_scope(self, *_args) -> None:
        self.target_field.setVisible(combo_value(self.scope_combo, "all") == "listed")
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
        """Attach one read-only imported structure for pre-run guidance."""
        self._input_structure = structure
        self.refresh_compact_presentation()

    def create_operation(self):
        return CompositionGradientOperation()

    def get_summary_text(self) -> str:
        try:
            summary = self.create_operation().sampling_summary(
                self.get_params(), self._input_structure
            )
        except ValueError:
            return self.tr("Complete the gradient definition")
        if "effective_groups" in summary:
            return self.tr(
                "lattice {axis} · {requested} requested → {effective} effective · {count}/input"
            ).format(
                axis=summary["axis"],
                requested=summary["requested_groups"],
                effective=summary["effective_groups"],
                count=summary["outputs_per_input"],
            )
        return self.tr("lattice {axis} · {groups} groups · {count}/input").format(
            axis=summary["axis"],
            groups=summary["requested_groups"],
            count=summary["outputs_per_input"],
        )

    def get_guidance_text(self) -> str:
        try:
            summary = self.create_operation().sampling_summary(
                self.get_params(), self._input_structure
            )
        except ValueError as exc:
            return translate_runtime_message(exc)

        parts: list[str] = []
        input_count = getattr(self, "_preview_input_count", None)
        if input_count is None:
            input_count = self._dataset_count(getattr(self, "dataset", None)) or None
        elif input_count == 0:
            input_count = None
        if input_count is not None:
            parts.append(
                self.tr(
                    "Inputs {inputs} × samples/input {samples} = outputs {total}."
                ).format(
                    inputs=input_count,
                    samples=summary["outputs_per_input"],
                    total=input_count * int(summary["outputs_per_input"]),
                )
            )
        else:
            parts.append(
                self.tr("Outputs/input: {samples}.").format(
                    samples=summary["outputs_per_input"]
                )
            )

        if self._input_structure is not None:
            minimum = int(summary["min_group_size"])
            maximum = int(summary["max_group_size"])
            size_text = str(minimum) if minimum == maximum else f"{minimum}–{maximum}"
            parts.append(
                self.tr(
                    "Eligible sites {sites} → effective groups {groups} → sites/group {size}."
                ).format(
                    sites=summary["candidate_sites"],
                    groups=summary["effective_groups"],
                    size=size_text,
                )
            )
            axis_idx = self.create_operation().AXIS_INDEX[str(summary["axis"])]
            pbc = np.asarray(self._input_structure.get_pbc(), dtype=bool)
            if pbc.size > axis_idx and bool(pbc[axis_idx]):
                parts.append(
                    self.tr(
                        "The periodic {axis} boundary joins the two compositions and creates a second jump."
                    ).format(axis=summary["axis"])
                )
            arrays = getattr(self._input_structure, "arrays", {})
            if "spin" in arrays or "initial_magmoms" in arrays:
                parts.append(
                    self.tr(
                        "Existing spin and initial magnetic moments are copied unchanged; reset them after changing species if needed."
                    )
                )
        if int(summary["samples"]) > 1:
            parts.append(
                self.tr("Random samples may repeat when groups contain few sites.")
            )
        return " ".join(parts)

    def get_params(self) -> CompositionGradientParams:
        elements, start_composition, end_composition = self.composition_table.values()
        target_mode = combo_value(self.scope_combo, "all")
        return CompositionGradientParams(
            elements=elements,
            start_composition=start_composition,
            end_composition=end_composition,
            axis=combo_value(self.axis_combo, "a"),
            bins=int(self.bins_frame.get_input_value()[0]),
            target_mode=target_mode,
            target_elements=self.target_edit.text() if target_mode == "listed" else "",
            samples=int(self.samples_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: CompositionGradientParams) -> None:
        self.elements_edit.setText(params.elements)
        self.start_edit.setText(params.start_composition)
        self.end_edit.setText(params.end_composition)
        self.composition_table.set_values(
            params.elements, params.start_composition, params.end_composition
        )
        legacy_axis = {"x": "a", "y": "b", "z": "c"}.get(
            str(params.axis).strip().lower(),
            str(params.axis).strip().lower(),
        )
        set_combo_value(self.axis_combo, legacy_axis)
        self.bins_frame.set_input_value([max(2, int(params.bins))])
        target_mode = str(params.target_mode or "all").strip().lower()
        if params.target_elements and target_mode == "all":
            target_mode = "listed"
        set_combo_value(self.scope_combo, target_mode)
        self.target_edit.setText(params.target_elements)
        self.samples_frame.set_input_value([max(1, int(params.samples))])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._update_direction_hint()
        self._update_scope()
        self._update_seed_visibility(bool(params.use_seed))

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            normalized = dict(raw_params)
            normalized.setdefault(
                "target_mode",
                "listed" if normalized.get("target_elements") else "all",
            )
            params = CompositionGradientParams(**normalized)
        else:
            params = CompositionGradientParams()
        self.set_params(params)
