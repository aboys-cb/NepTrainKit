"""Card for controlled spin-disorder generation."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.magnetism import SpinDisorderOperation, SpinDisorderParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    KeyValueTableInput,
    MakeDataCard,
    NumericScanInput,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)
from .i18n_utils import add_translated_items, combo_value, set_combo_value


@CardManager.register_card
class SpinDisorderCard(MakeDataCard):
    """Generate spin states with explicit disorder fractions."""

    group = "Magnetism"
    card_name = "Moment Disorder"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Moment Disorder"))
        self._preview_input_count: int | None = None
        self.init_ui()

    def init_ui(self):
        self.setObjectName("spin_disorder_card_widget")

        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.mode_combo,
            [
                ("Flip fraction", "Flip selected moments"),
                ("Randomize fraction", "Assign random directions"),
                ("Cone disorder", "Deflect within a cone"),
            ],
        )
        set_combo_value(self.mode_combo, "Flip fraction")
        self.mode_field = CompactField(
            self.tr("How selected moments change"),
            self.mode_combo,
            self.setting_widget,
            self.tr("The fraction below decides which moments are selected; this option decides their new directions."),
        )

        self.fractions_edit = NumericScanInput(
            self.setting_widget, minimum=0.001, maximum=1.0, decimals=3
        )
        self.fractions_edit.set_range(0.1, 0.7, 0.2)
        self.fractions_edit.custom_checkbox.setText(self.tr("Specify fractions to generate"))
        self.fractions_edit.custom_edit.setPlaceholderText(self.tr("For example: 0.1,0.3,0.5,0.7"))
        self.fractions_field = CompactField(
            self.tr("Fraction of moments changed"),
            self.fractions_edit,
            self.setting_widget,
            self.tr("Set minimum, maximum, and step, or switch to a custom list. Every fraction must be greater than 0 and at most 1."),
        )

        self.samples_frame = SpinBoxUnitInputFrame(self)
        self.samples_frame.set_input("", 1, "int")
        self.samples_frame.setRange(1, 10000)
        self.samples_frame.set_input_value([1])
        self.samples_frame.setFixedWidth(132)
        self.samples_field = CompactField(
            self.tr("Samples per fraction"),
            self.samples_frame,
            self.setting_widget,
            self.tr("Independent random selections generated at each fraction."),
            inline=True,
            input_max_width=132,
        )

        self.cone_frame = SpinBoxUnitInputFrame(self)
        self.cone_frame.set_input("°", 1, "float")
        self.cone_frame.setRange(0.0, 180.0)
        self.cone_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.cone_frame.set_input_value([30.0])
        self.cone_frame.setFixedWidth(132)
        self.cone_field = CompactField(
            self.tr("Maximum cone angle"),
            self.cone_frame,
            self.setting_widget,
            self.tr("Selected moments are sampled uniformly inside this cone around their current directions."),
            inline=True,
            input_max_width=132,
        )

        disorder_section = InspectorSection(
            self.tr("Moment changes"),
            self.setting_widget,
            self.tr("For every fraction, randomly choose that share of eligible non-zero moments and generate a separate set of structures."),
        )
        disorder_section.addWidget(self.mode_field)
        disorder_section.addWidget(self.fractions_field)
        disorder_grid = ResponsiveFormGrid(disorder_section)
        disorder_grid.add_field(self.samples_field, span=2)
        disorder_grid.add_field(self.cone_field, span=2)
        disorder_section.addWidget(disorder_grid)

        self.source_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.source_combo,
            [
                ("Existing initial magmoms", "Existing moments"),
                ("Map/default magnitude", "Element map / default"),
            ],
        )
        set_combo_value(self.source_combo, "Existing initial magmoms")
        self.source_field = CompactField(
            self.tr("Moment source"), self.source_combo, self.setting_widget
        )

        self.map_edit = KeyValueTableInput(
            self.tr("Element"), self.tr("Moment magnitude"), self.setting_widget
        )
        self.map_field = CompactField(
            self.tr("Element moments"), self.map_edit, self.setting_widget
        )

        self.default_frame = SpinBoxUnitInputFrame(self)
        self.default_frame.set_input("", 1, "float")
        self.default_frame.setRange(0.0, 20.0)
        self.default_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.default_frame.set_input_value([0.0])
        self.default_frame.setFixedWidth(132)
        self.default_field = CompactField(
            self.tr("Default moment magnitude"),
            self.default_frame,
            self.setting_widget,
            self.tr("Used only for elements missing from the table."),
            inline=True,
            input_max_width=132,
        )

        self.lift_scalar_checkbox = CheckBox(self.tr("Lift scalar magmoms to vectors"), self.setting_widget)
        self.lift_scalar_checkbox.setChecked(True)

        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setRange(-1.0, 1.0)
        for obj in self.axis_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])
        self.axis_field = CompactField(
            self.tr("Scalar lift direction (Cartesian)"),
            self.axis_frame,
            self.setting_widget,
            self.tr("Used only to initialize scalar or map-based moments; existing vector directions are preserved."),
        )

        self.apply_edit = LineEdit(self.setting_widget)
        self.apply_edit.setPlaceholderText(self.tr("All non-zero moments"))
        self.apply_field = CompactField(
            self.tr("Target elements"),
            self.apply_edit,
            self.setting_widget,
            self.tr("Comma-separated symbols such as Fe,Co; empty selects all non-zero moments."),
        )

        self.advanced_checkbox = CheckBox(
            self.tr("Show moment source and target filter"), self.setting_widget
        )
        self.advanced_checkbox.setChecked(False)

        self.source_section = InspectorSection(self.tr("Moment source and targets"), self.setting_widget)
        source_grid = ResponsiveFormGrid(self.source_section)
        source_grid.add_field(self.source_field, span=2)
        source_grid.add_field(self.map_field, span=2)
        source_grid.add_field(self.default_field, span=2)
        source_grid.add_field(self.lift_scalar_checkbox, span=2)
        source_grid.add_field(self.axis_field, span=2)
        source_grid.add_field(self.apply_field, span=2)
        self.source_section.addWidget(source_grid)
        self.source_section.hide()

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_frame.setFixedWidth(132)
        seed_row = QWidget(self.setting_widget)
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(6)
        seed_layout.addWidget(self.seed_checkbox)
        seed_layout.addWidget(self.seed_frame)
        seed_layout.addStretch(1)

        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input("", 1, "int")
        self.max_output_frame.setRange(1, 999999)
        self.max_output_frame.set_input_value([100])
        self.max_output_frame.setFixedWidth(132)
        self.max_output_field = CompactField(
            self.tr("Maximum outputs per input"),
            self.max_output_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )

        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        generation_section = InspectorSection(self.tr("Generation"), self.setting_widget)
        generation_grid = ResponsiveFormGrid(generation_section)
        generation_grid.add_field(seed_row, span=2)
        generation_grid.add_field(self.max_output_field, span=2)
        generation_section.addWidget(generation_grid)
        generation_section.addWidget(self.output_preview)

        self.settingLayout.addWidget(disorder_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_checkbox, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.source_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(generation_section, 3, 0, 1, 3)

        self.mode_combo.currentTextChanged.connect(self._update_mode_widgets)
        self.source_combo.currentTextChanged.connect(self._update_source_widgets)
        self.advanced_checkbox.toggled.connect(self._update_source_widgets)
        self.seed_checkbox.toggled.connect(self.seed_frame.setEnabled)
        self.samples_frame.object_list[0].valueChanged.connect(self._update_output_preview)
        self.max_output_frame.object_list[0].valueChanged.connect(self._update_output_preview)
        for spin in self.fractions_edit.range_frame.object_list:
            spin.valueChanged.connect(self._update_output_preview)
        self.fractions_edit.custom_edit.textChanged.connect(self._update_output_preview)
        self.fractions_edit.custom_checkbox.toggled.connect(self._update_output_preview)
        self._update_mode_widgets()
        self._update_source_widgets()

    def _update_mode_widgets(self):
        show_cone = combo_value(self.mode_combo) == "Cone disorder"
        self.cone_field.setVisible(show_cone)
        self.cone_field.setEnabled(show_cone)
        self._update_output_preview()

    def _update_source_widgets(self):
        show_advanced = self.advanced_checkbox.isChecked()
        self.source_section.setVisible(show_advanced)
        self.source_section.setEnabled(show_advanced)
        use_map = show_advanced and combo_value(self.source_combo) == "Map/default magnitude"
        for widget in (self.map_field, self.default_field):
            widget.setVisible(use_map)
            widget.setEnabled(use_map)
        self.lift_scalar_checkbox.setVisible(show_advanced and not use_map)
        self.lift_scalar_checkbox.setEnabled(show_advanced and not use_map)

    def _validated_fraction_values(self) -> list[float]:
        try:
            values = self.fractions_edit.values()
        except ValueError as exc:
            raise ValueError(
                self.tr("The custom fraction list must contain finite numbers greater than 0 and at most 1.")
            ) from exc
        if any(not 0.0 < value <= 1.0 for value in values):
            raise ValueError(self.tr("Every disorder fraction must be greater than 0 and at most 1."))
        return values

    def _update_output_preview(self, *_args):
        try:
            values = self._validated_fraction_values()
            fractions = len(values)
            samples = int(self.samples_frame.get_input_value()[0])
            requested = fractions * samples
            limit = int(self.max_output_frame.get_input_value()[0])
            emitted = min(requested, limit)
            remaining = emitted
            allocation_parts = []
            for value in values:
                count = min(samples, remaining)
                remaining -= count
                allocation_parts.append(f"{value * 100:.4g}% × {count}")
            allocation = " · ".join(allocation_parts)
            if emitted == requested:
                preview = self.tr(
                    "Per input: {allocation} = {emitted} structures."
                )
            else:
                preview = self.tr(
                    "Per input: {allocation} = {emitted} structures ({requested} requested; output limit reached)."
                )
            self.output_preview.setText(
                preview.format(
                    allocation=allocation,
                    emitted=emitted,
                    requested=requested,
                )
            )
        except ValueError as exc:
            self.output_preview.setText(str(exc))

    def get_summary_text(self) -> str:
        try:
            fraction_count = len(self._validated_fraction_values())
        except ValueError:
            return self.tr("Fraction list needs attention")
        return self.tr("{mode} · {fractions} fractions · {samples} each").format(
            mode=self.mode_combo.currentText(),
            fractions=fraction_count,
            samples=int(self.samples_frame.get_input_value()[0]),
        )

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def get_guidance_text(self) -> str:
        try:
            values = self._validated_fraction_values()
        except ValueError:
            return self.tr("Enter one or more fractions between 0 and 1 before running.")
        params = self.get_params()
        changed = self.create_operation().count_for_fraction(100, values[0])
        note = self.tr(
            "At fraction {fraction}, about {changed} of every 100 eligible non-zero moments change; moment magnitudes stay fixed."
        ).format(fraction=f"{values[0]:.4g}", changed=changed)
        if self._preview_input_count:
            per_input = min(
                self.fractions_edit.count() * params.samples_per_fraction,
                params.max_outputs,
            )
            note += " " + self.tr("Planned maximum: {total} outputs.").format(
                total=self._preview_input_count * per_input
            )
        return note

    def create_operation(self):
        return SpinDisorderOperation()

    def get_params(self) -> SpinDisorderParams:
        return SpinDisorderParams(
            mode=combo_value(self.mode_combo),
            fractions=self.fractions_edit.scan_text(),
            samples_per_fraction=int(self.samples_frame.get_input_value()[0]),
            cone_angle=float(self.cone_frame.get_input_value()[0]),
            magnitude_source=combo_value(self.source_combo),
            magmom_map=self.map_edit.text(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            lift_scalar=self.lift_scalar_checkbox.isChecked(),
            axis=self.axis_frame.get_input_value(),
            apply_elements=self.apply_edit.text(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
        )

    def set_params(self, params: SpinDisorderParams) -> None:
        set_combo_value(self.mode_combo, params.mode)
        self.fractions_edit.set_scan_text(params.fractions)
        self.samples_frame.set_input_value([int(params.samples_per_fraction)])
        self.cone_frame.set_input_value([float(params.cone_angle)])
        set_combo_value(self.source_combo, params.magnitude_source)
        self.map_edit.setText(params.magmom_map)
        self.default_frame.set_input_value([float(params.default_moment)])
        self.lift_scalar_checkbox.setChecked(bool(params.lift_scalar))
        self.axis_frame.set_input_value([float(v) for v in params.axis])
        self.apply_edit.setText(params.apply_elements)
        self.advanced_checkbox.setChecked(
            params.magnitude_source != "Existing initial magmoms"
            or bool(params.apply_elements.strip())
            or tuple(float(value) for value in params.axis) != (0.0, 0.0, 1.0)
            or not bool(params.lift_scalar)
        )
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.max_output_frame.set_input_value([int(params.max_outputs)])
        self._update_mode_widgets()
        self._update_source_widgets()

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        params = SpinDisorderParams(**raw_params) if raw_params else SpinDisorderParams()
        self.set_params(params)
