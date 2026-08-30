"""Card for assigning global alloy occupancies from a target composition."""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import RandomOccupancyOperation, RandomOccupancyParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    KeyValueTableInput,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)
from .i18n_utils import add_translated_items, combo_value, set_combo_value


@CardManager.register_card
class RandomOccupancyCard(MakeDataCard):
    """Assign alloy elements to all (or grouped) lattice sites using a target composition."""

    group = "Alloy"
    card_name = "Random Occupancy"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._preview_input_count: int | None = None
        self._preview: dict[str, object] | None = None
        self._preview_error = ""
        self.setTitle(self.tr("Occupancy Mix"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("random_occupancy_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.source_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.source_combo, ["Auto (Comp tag)", "Manual"])
        self.source_field = CompactField(
            self.tr("Composition"),
            self.source_combo,
            self.setting_widget,
            inline=True,
            input_max_width=190,
        )
        self.source_label = self.source_field.caption

        self.manual_edit = KeyValueTableInput(
            self.tr("Element"), self.tr("Target fraction"), self.setting_widget,
            element_picker=True, new_element_value="1.0",
        )
        self.manual_field = CompactField(
            self.tr("Manual composition"),
            self.manual_edit,
            self.setting_widget,
            self.tr("Enter element weights; they are normalized before occupancy assignment."),
        )
        self.manual_label = self.manual_field.caption
        self.manual_field.hide()

        composition_section = InspectorSection(
            self.tr("Target composition"),
            self.setting_widget,
            self.tr(
                "Auto reads the last Comp(...) tag from each input. Manual uses the table below."
            ),
        )
        composition_grid = ResponsiveFormGrid(composition_section)
        composition_grid.add_field(self.source_field, span=2)
        composition_grid.add_field(self.manual_field, span=2)
        composition_section.addWidget(composition_grid)

        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.mode_combo,
            [
                ("Exact", self.tr("Fixed counts")),
                ("Random", self.tr("Sampled counts")),
            ],
        )
        self.mode_field = CompactField(
            self.tr("Count behavior"),
            self.mode_combo,
            self.setting_widget,
            inline=True,
            input_max_width=170,
        )
        self.mode_label = self.mode_field.caption

        self.samples_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.samples_frame.set_input("", 1, "int")
        self.samples_frame.setRange(1, 999999)
        self.samples_frame.set_input_value([1])
        self.samples_field = CompactField(
            self.tr("Outputs per input"),
            self.samples_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )
        self.samples_label = self.samples_field.caption

        self.group_edit = LineEdit(self.setting_widget)
        self.group_edit.setPlaceholderText(self.tr("Optional: A,B"))
        self.group_field = CompactField(
            self.tr("Occupancy groups"),
            self.group_edit,
            self.setting_widget,
            self.tr(
                "Comma-separated input group labels. Only matched sites are reassigned; all other atoms keep their elements."
            ),
        )
        self.group_label = self.group_field.caption

        assignment_section = InspectorSection(
            self.tr("Occupancy generation"),
            self.setting_widget,
            self.tr(
                "Both modes randomize element positions. Fixed counts keep one integer composition; sampled counts redraw it for each output."
            ),
        )
        assignment_grid = ResponsiveFormGrid(assignment_section, two_column_threshold=420)
        assignment_grid.add_field(self.mode_field)
        assignment_grid.add_field(self.samples_field)
        assignment_grid.add_field(self.group_field, span=2)
        assignment_section.addWidget(assignment_grid)
        self.preview_label = CaptionLabel("", assignment_section)
        self.preview_label.setWordWrap(True)
        assignment_section.addWidget(self.preview_label)

        self.seed_checkbox = CheckBox(self.tr("Use fixed random seed"), self.setting_widget)
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

        self.settingLayout.addWidget(composition_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(assignment_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(random_section, 2, 0, 1, 3)

        self.source_combo.currentIndexChanged.connect(self._update_source_widgets)
        self.mode_combo.currentIndexChanged.connect(self._refresh_preview_and_presentation)
        self.samples_frame.object_list[0].valueChanged.connect(
            self._refresh_preview_and_presentation
        )
        self.group_edit.textChanged.connect(self._refresh_preview_and_presentation)
        self.manual_edit.editingFinished.connect(self._refresh_preview_and_presentation)
        self.seed_checkbox.toggled.connect(self._update_seed_widgets)
        self.seed_frame.object_list[0].valueChanged.connect(
            self._refresh_preview_and_presentation
        )
        self._update_source_widgets()
        self._update_seed_widgets(False)

    def _update_source_widgets(self, *_args) -> None:
        manual = combo_value(self.source_combo) == "Manual"
        self.manual_field.setVisible(manual)
        self._refresh_preview_and_presentation()

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

    @staticmethod
    def _format_composition(composition) -> str:
        return ", ".join(
            f"{element} {100.0 * float(fraction):.1f}%"
            for element, fraction in dict(composition or {}).items()
        )

    @staticmethod
    def _format_counts(counts) -> str:
        return ", ".join(
            f"{element} {int(count)}"
            for element, count in dict(counts or {}).items()
        )

    def _refresh_preview_and_presentation(self, *_args) -> None:
        self._preview = None
        self._preview_error = ""
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr("Load an upstream structure to preview target composition and eligible sites.")
            )
            self.refresh_compact_presentation()
            return
        try:
            summary = self.create_operation().sampling_summary(
                self._input_structure,
                self.get_params(),
            )
            self._preview = {
                "composition": summary["target"],
                "eligible_sites": summary["eligible_count"],
                "total_sites": len(self._input_structure),
                "realized_counts": summary["fixed_counts"],
                "outputs_per_input": summary["outputs_per_input"],
                "group_filter": summary["groups"],
            }
        except ValueError as exc:
            self._preview_error = translate_runtime_message(exc)

        if self._preview_error:
            self.preview_label.setText(self._preview_error)
        elif self._preview is not None:
            composition = self._format_composition(self._preview.get("composition"))
            eligible = int(self._preview.get("eligible_sites", 0))
            total = int(self._preview.get("total_sites", 0))
            details = [
                self.tr("Eligible sites {eligible}/{total}").format(
                    eligible=eligible,
                    total=total,
                ),
                self.tr("target {composition}").format(composition=composition),
            ]
            realized = self._format_counts(self._preview.get("realized_counts"))
            if realized:
                details.append(
                    self.tr("fixed counts {counts}").format(counts=realized)
                )
            details.append(
                self.tr("{outputs} output(s)/input").format(
                    outputs=self.get_params().samples
                )
            )
            self.preview_label.setText(" · ".join(details))
        self.refresh_compact_presentation()

    def get_summary_text(self) -> str:
        params = self.get_params()
        mode = self.tr("fixed counts") if params.mode == "Exact" else self.tr("sampled counts")
        scope = (
            self.tr("groups {groups}").format(groups=params.group_filter)
            if params.group_filter.strip()
            else self.tr("all sites")
        )
        if self._preview_error:
            return self.tr("Parameters need attention: {error}").format(
                error=self._preview_error
            )
        if self._preview is not None:
            eligible = int(self._preview.get("eligible_sites", 0))
            total = int(self._preview.get("total_sites", 0))
            return self.tr(
                "{mode} · {eligible}/{total} sites · {outputs}/input"
            ).format(
                mode=mode,
                eligible=eligible,
                total=total,
                outputs=params.samples,
            )
        return self.tr("{mode} · {scope} · {outputs}/input").format(
            mode=mode,
            scope=scope,
            outputs=params.samples,
        )

    def get_guidance_text(self) -> str:
        params = self.get_params()
        if self._preview_error:
            return self._preview_error
        outputs = params.samples
        if self._preview_input_count is not None:
            output_text = self.tr("Inputs {inputs} × {per_input}/input = outputs {outputs}").format(
                inputs=self._preview_input_count,
                per_input=outputs,
                outputs=self._preview_input_count * outputs,
            )
        else:
            output_text = self.tr("Outputs per input: {outputs}").format(outputs=outputs)
        if self._preview is None:
            source_text = (
                self.tr("Auto requires a Comp(...) tag on each input.")
                if params.source == "Auto (Comp tag)"
                else self.tr("Complete the manual element table.")
            )
            return f"{source_text} {output_text}"
        composition = self._format_composition(self._preview.get("composition"))
        eligible = int(self._preview.get("eligible_sites", 0))
        total = int(self._preview.get("total_sites", 0))
        if params.mode == "Exact":
            count_text = self.tr("Realized fixed counts: {counts}.").format(
                counts=self._format_counts(self._preview.get("realized_counts"))
            )
        else:
            count_text = self.tr(
                "Element counts are sampled independently for each output."
            )
        return self.tr(
            "Target {composition}. Eligible sites {eligible}/{total}. {counts} {outputs}"
        ).format(
            composition=composition,
            eligible=eligible,
            total=total,
            counts=count_text,
            outputs=output_text,
        )

    def create_operation(self):
        """Return the UI-independent random occupancy operation."""
        return RandomOccupancyOperation()

    def get_params(self) -> RandomOccupancyParams:
        """Read random occupancy parameters from UI controls."""
        return RandomOccupancyParams(
            source=combo_value(self.source_combo),
            manual=self.manual_edit.text(),
            mode=combo_value(self.mode_combo),
            samples=int(self.samples_frame.get_input_value()[0]),
            group_filter=self.group_edit.text(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: RandomOccupancyParams) -> None:
        """Apply random occupancy parameters to UI controls."""
        set_combo_value(self.source_combo, params.source)
        self.manual_edit.setText(params.manual)
        set_combo_value(self.mode_combo, params.mode)
        self.samples_frame.set_input_value([int(params.samples)])
        self.group_edit.setText(params.group_filter)
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._update_source_widgets()
        self._update_seed_widgets(bool(params.use_seed))

    def process_structure(self, structure):
        """Assign occupancy from UI-independent parameters."""
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = RandomOccupancyParams(
                source=raw_params.get("source", "Auto (Comp tag)"),
                manual=raw_params.get("manual", ""),
                mode=raw_params.get("mode", "Exact"),
                samples=raw_params.get("samples", 1),
                group_filter=raw_params.get("group_filter", ""),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
            )
        else:
            params = RandomOccupancyParams(
                source=data_dict.get("source", "Auto (Comp tag)"),
                manual=data_dict.get("manual", ""),
                mode=data_dict.get("mode", "Exact"),
                samples=data_dict.get("samples", [1])[0],
                group_filter=data_dict.get("group_filter", ""),
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
