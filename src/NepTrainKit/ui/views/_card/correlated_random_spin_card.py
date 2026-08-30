"""Card for spatially correlated random non-collinear spins."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.magnetism import CorrelatedRandomSpinOperation, CorrelatedRandomSpinParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    ElementLineEdit,
    InspectorSection,
    KeyValueTableInput,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class CorrelatedRandomSpinCard(MakeDataCard):
    """Generate non-collinear directions with an explicit spatial correlation length."""

    group = "Magnetism"
    card_name = "Correlated Spins"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Correlated Spins"))
        self._preview_input_count: int | None = None
        self.init_ui()

    def init_ui(self):
        self.setObjectName("correlated_random_spin_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.mode_combo,
            [
                ("Cone around reference", "Perturb around current directions"),
                ("Full random directions", "Replace with random directions"),
            ],
        )
        set_combo_value(self.mode_combo, "Cone around reference")
        self.mode_field = CompactField(
            self.tr("How all eligible moments change"),
            self.mode_combo,
            self.setting_widget,
            self.tr("Every eligible non-zero moment is processed; nearby moments receive spatially correlated directions."),
        )

        self.xi_frame = SpinBoxUnitInputFrame(self)
        self.xi_frame.set_input("Å", 1, "float")
        self.xi_frame.setRange(0.000001, 1000000.0)
        self.xi_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.xi_frame.set_input_value([3.0])
        self.xi_frame.setFixedWidth(144)
        self.xi_field = CompactField(
            self.tr("Spatial correlation length"),
            self.xi_frame,
            self.setting_widget,
            self.tr("Larger values make nearby moment directions vary together over longer distances."),
            inline=True,
            input_max_width=144,
        )

        self.cone_frame = SpinBoxUnitInputFrame(self)
        self.cone_frame.set_input("°", 1, "float")
        self.cone_frame.setRange(0.0, 180.0)
        self.cone_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.cone_frame.set_input_value([30.0])
        self.cone_frame.setFixedWidth(132)
        self.cone_field = CompactField(
            self.tr("Maximum deflection angle"),
            self.cone_frame,
            self.setting_widget,
            self.tr("Only used around current directions; 0° reproduces the reference directions."),
            inline=True,
            input_max_width=132,
        )

        direction_section = InspectorSection(
            self.tr("Spatially correlated directions"),
            self.setting_widget,
            self.tr("Unlike fraction-based disorder, this card processes every eligible moment and correlates the direction field through atomic distance."),
        )
        direction_section.addWidget(self.mode_field)
        direction_grid = ResponsiveFormGrid(direction_section)
        direction_grid.add_field(self.xi_field, span=2)
        direction_grid.add_field(self.cone_field, span=2)
        direction_section.addWidget(direction_grid)

        self.samples_frame = SpinBoxUnitInputFrame(self)
        self.samples_frame.set_input("", 1, "int")
        self.samples_frame.setRange(1, 100000)
        self.samples_frame.set_input_value([1])
        self.samples_frame.setFixedWidth(132)
        self.samples_field = CompactField(
            self.tr("Structures per input"),
            self.samples_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )

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

        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        generation_section = InspectorSection(self.tr("Generation"), self.setting_widget)
        generation_grid = ResponsiveFormGrid(generation_section)
        generation_grid.add_field(self.samples_field, span=2)
        generation_grid.add_field(seed_row, span=2)
        generation_section.addWidget(generation_grid)
        generation_section.addWidget(self.output_preview)

        self.kernel_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.kernel_combo,
            [
                ("exponential", "Exponential (longer tail)"),
                ("squared_exponential", "Squared exponential (smoother nearby)"),
            ],
        )
        set_combo_value(self.kernel_combo, "exponential")
        self.kernel_field = CompactField(
            self.tr("Correlation profile"),
            self.kernel_combo,
            self.setting_widget,
            self.tr("Exponential keeps a longer-distance tail; squared exponential is smoother at short distance and decays faster."),
        )

        self.max_atoms_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_frame.set_input("", 1, "int")
        self.max_atoms_frame.setRange(1, 1000000)
        self.max_atoms_frame.set_input_value([200])
        self.max_atoms_frame.setFixedWidth(132)
        self.max_atoms_field = CompactField(
            self.tr("Maximum eligible moments"),
            self.max_atoms_frame,
            self.setting_widget,
            self.tr("Safety limit for the exact covariance matrix; exceeding it stops with an error."),
            inline=True,
            input_max_width=132,
        )

        self.field_section = InspectorSection(self.tr("Correlation model and size guard"), self.setting_widget)
        field_grid = ResponsiveFormGrid(self.field_section)
        field_grid.add_field(self.kernel_field, span=2)
        field_grid.add_field(self.max_atoms_field, span=2)
        self.field_section.addWidget(field_grid)
        self.field_section.hide()

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
        self.source_field = CompactField(self.tr("Moment source"), self.source_combo, self.setting_widget)

        self.map_edit = KeyValueTableInput(
            self.tr("Element"), self.tr("Moment magnitude"), self.setting_widget,
            element_picker=True, new_element_value="1.0",
        )
        self.map_field = CompactField(self.tr("Element moments"), self.map_edit, self.setting_widget)

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

        self.lift_scalar_checkbox = CheckBox(
            self.tr("Lift scalar moments to vectors"), self.setting_widget
        )
        self.lift_scalar_checkbox.setChecked(True)

        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setRange(-1.0, 1.0)
        for obj in self.axis_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])
        self.axis_field = CompactField(
            self.tr("Scalar reference direction (Cartesian)"),
            self.axis_frame,
            self.setting_widget,
            self.tr("Initializes scalar or map-based moments for cone perturbation; existing vector directions are preserved."),
        )

        self.apply_edit = ElementLineEdit(self.setting_widget, multiple=True)
        self.apply_edit.setPlaceholderText(self.tr("All non-zero moments"))
        self.apply_field = CompactField(
            self.tr("Target elements"),
            self.apply_edit,
            self.setting_widget,
            self.tr("Comma-separated symbols such as Fe,Co; empty selects all non-zero moments."),
        )

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

        self.advanced_checkbox = CheckBox(
            self.tr("Advanced correlation and moment settings"), self.setting_widget
        )
        self.advanced_checkbox.setChecked(False)

        self.settingLayout.addWidget(direction_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(generation_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_checkbox, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.field_section, 3, 0, 1, 3)
        self.settingLayout.addWidget(self.source_section, 4, 0, 1, 3)

        self.mode_combo.currentIndexChanged.connect(self._update_mode_widgets)
        self.source_combo.currentIndexChanged.connect(self._update_source_widgets)
        self.advanced_checkbox.toggled.connect(self._update_source_widgets)
        self.seed_checkbox.toggled.connect(self.seed_frame.setEnabled)
        self.samples_frame.object_list[0].valueChanged.connect(self._update_output_preview)
        for frame in (self.xi_frame, self.cone_frame, self.samples_frame, self.seed_frame):
            for control in frame.object_list:
                control.valueChanged.connect(lambda _value: self.refresh_compact_presentation())
        self.mode_combo.currentIndexChanged.connect(lambda _index: self.refresh_compact_presentation())
        self._update_mode_widgets()
        self._update_source_widgets()
        self._update_output_preview()

    def _update_mode_widgets(self, *_args):
        show_cone = combo_value(self.mode_combo) == "Cone around reference"
        self.cone_field.setVisible(show_cone)
        self.cone_field.setEnabled(show_cone)
        self._update_source_widgets()

    def _update_source_widgets(self, *_args):
        show_advanced = self.advanced_checkbox.isChecked()
        self.field_section.setVisible(show_advanced)
        self.field_section.setEnabled(show_advanced)
        self.source_section.setVisible(show_advanced)
        self.source_section.setEnabled(show_advanced)
        use_map = show_advanced and combo_value(self.source_combo) == "Map/default magnitude"
        for widget in (self.map_field, self.default_field):
            widget.setVisible(use_map)
            widget.setEnabled(use_map)
        self.lift_scalar_checkbox.setVisible(show_advanced and not use_map)
        self.lift_scalar_checkbox.setEnabled(show_advanced and not use_map)
        show_axis = show_advanced and combo_value(self.mode_combo) == "Cone around reference"
        self.axis_field.setVisible(show_axis)
        self.axis_field.setEnabled(show_axis)

    def _update_output_preview(self, *_args):
        samples = int(self.samples_frame.get_input_value()[0])
        self.output_preview.setText(
            self.tr(
                "Structures per valid input: {samples}; every eligible non-zero moment receives a spatially correlated direction."
            ).format(samples=samples)
        )

    def get_summary_text(self) -> str:
        mode = self.tr("Cone") if combo_value(self.mode_combo) == "Cone around reference" else self.tr("Random")
        return self.tr("{mode} · ξ={length} Å · n={samples}").format(
            mode=mode,
            length=f"{float(self.xi_frame.get_input_value()[0]):.4g}",
            samples=int(self.samples_frame.get_input_value()[0]),
        )

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def get_guidance_text(self) -> str:
        note = self.tr(
            "Larger correlation length produces smoother direction patches. Verify it statistically across several samples; one structure is not enough to establish the correlation length."
        )
        if self._preview_input_count:
            note += " " + self.tr("Planned: {total} outputs.").format(
                total=self._preview_input_count * int(self.samples_frame.get_input_value()[0])
            )
        return note

    def create_operation(self):
        return CorrelatedRandomSpinOperation()

    def get_params(self) -> CorrelatedRandomSpinParams:
        return CorrelatedRandomSpinParams(
            mode=combo_value(self.mode_combo),
            correlation_kernel=combo_value(self.kernel_combo),
            correlation_length=float(self.xi_frame.get_input_value()[0]),
            samples=int(self.samples_frame.get_input_value()[0]),
            cone_angle=float(self.cone_frame.get_input_value()[0]),
            magnitude_source=combo_value(self.source_combo),
            magmom_map=self.map_edit.text(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            lift_scalar=self.lift_scalar_checkbox.isChecked(),
            axis=self.axis_frame.get_input_value(),
            apply_elements=self.apply_edit.text(),
            max_atoms_for_full=int(self.max_atoms_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: CorrelatedRandomSpinParams) -> None:
        set_combo_value(self.mode_combo, params.mode)
        set_combo_value(self.kernel_combo, params.correlation_kernel)
        self.xi_frame.set_input_value([float(params.correlation_length)])
        self.samples_frame.set_input_value([int(params.samples)])
        self.cone_frame.set_input_value([float(params.cone_angle)])
        set_combo_value(self.source_combo, params.magnitude_source)
        self.map_edit.setText(params.magmom_map)
        self.default_frame.set_input_value([float(params.default_moment)])
        self.lift_scalar_checkbox.setChecked(bool(params.lift_scalar))
        self.axis_frame.set_input_value([float(v) for v in params.axis])
        self.apply_edit.setText(params.apply_elements)
        self.max_atoms_frame.set_input_value([int(params.max_atoms_for_full)])
        self.advanced_checkbox.setChecked(
            params.correlation_kernel != "exponential"
            or params.magnitude_source != "Existing initial magmoms"
            or bool(params.magmom_map.strip())
            or float(params.default_moment) != 0.0
            or not bool(params.lift_scalar)
            or tuple(float(v) for v in params.axis) != (0.0, 0.0, 1.0)
            or bool(params.apply_elements.strip())
            or int(params.max_atoms_for_full) != 200
        )
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())
        self._update_mode_widgets()
        self._update_source_widgets()
        self._update_output_preview()

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        params = CorrelatedRandomSpinParams(**raw_params) if raw_params else CorrelatedRandomSpinParams()
        self.set_params(params)
