"""Card for generating symmetric folded-helix magnetic textures layer by layer."""

from __future__ import annotations

from qfluentwidgets import CaptionLabel, CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.magnetism import FoldedHelixOperation, FoldedHelixParams
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
class FoldedHelixCard(MakeDataCard):
    """Assign a deterministic triangular phase profile across detected layers."""

    group = "Magnetism"
    card_name = "Folded Helix"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Folded Helix"))
        self._preview_input_count: int | None = None
        self.init_ui()

    def init_ui(self):
        self.setObjectName("folded_helix_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.layer_axis_frame = SpinBoxUnitInputFrame(self)
        self.layer_axis_frame.set_input("", 3, "float")
        self.layer_axis_frame.setRange(-1.0, 1.0)
        for control in self.layer_axis_frame.object_list:
            control.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.layer_axis_frame.set_input_value([0.0, 0.0, 1.0])
        self.layer_axis_field = CompactField(
            self.tr("Layer direction (Cartesian)"),
            self.layer_axis_frame,
            self.setting_widget,
            self.tr("Atomic positions are projected onto this direction and grouped into layers."),
        )

        self.plane_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.plane_mode_combo,
            [
                ("Follow layer direction", "Rotate in the layer plane"),
                ("Custom plane normal", "Custom spin plane"),
            ],
        )
        set_combo_value(self.plane_mode_combo, "Follow layer direction")
        self.plane_mode_field = CompactField(
            self.tr("Spin rotation plane"),
            self.plane_mode_combo,
            self.setting_widget,
            self.tr("The common case rotates moments within each atomic layer."),
        )

        self.plane_normal_frame = SpinBoxUnitInputFrame(self)
        self.plane_normal_frame.set_input("", 3, "float")
        self.plane_normal_frame.setRange(-1.0, 1.0)
        for control in self.plane_normal_frame.object_list:
            control.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.plane_normal_frame.set_input_value([0.0, 0.0, 1.0])
        self.plane_normal_field = CompactField(
            self.tr("Custom spin-plane normal (Cartesian)"),
            self.plane_normal_frame,
            self.setting_widget,
            self.tr("Generated moments rotate in the plane perpendicular to this direction."),
        )

        self.layer_tol_frame = SpinBoxUnitInputFrame(self)
        self.layer_tol_frame.set_input("Å", 1, "float")
        self.layer_tol_frame.setRange(0.0001, 10.0)
        self.layer_tol_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.layer_tol_frame.set_input_value([0.05])
        self.layer_tol_frame.setFixedWidth(144)
        self.layer_tol_field = CompactField(
            self.tr("Layer grouping tolerance"),
            self.layer_tol_frame,
            self.setting_widget,
            self.tr("Projected coordinates within this distance share one spin direction."),
            inline=True,
            input_max_width=144,
        )

        geometry_section = InspectorSection(
            self.tr("Layer geometry"),
            self.setting_widget,
            self.tr("At least three detected layers are required to form a turn-and-return profile."),
        )
        geometry_grid = ResponsiveFormGrid(geometry_section)
        geometry_grid.add_field(self.layer_axis_field, span=2)
        geometry_grid.add_field(self.plane_mode_field, span=2)
        geometry_grid.add_field(self.plane_normal_field, span=2)
        geometry_grid.add_field(self.layer_tol_field, span=2)
        geometry_section.addWidget(geometry_grid)

        self.half_period_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.half_period_mode_combo, ["Auto from layer count", "Manual"])
        set_combo_value(self.half_period_mode_combo, "Auto from layer count")
        self.half_period_mode_field = CompactField(
            self.tr("Fold length"),
            self.half_period_mode_combo,
            self.setting_widget,
            self.tr("Automatic mode spans the detected layer stack once; manual mode repeats a chosen triangular period."),
        )

        self.half_period_frame = SpinBoxUnitInputFrame(self)
        self.half_period_frame.set_input(["–", self.tr("step"), self.tr("layers")], 3, "int")
        self.half_period_frame.setRange(1, 999999)
        self.half_period_frame.set_input_value([2, 4, 1])
        self.half_period_field = CompactField(
            self.tr("Half-period layer steps"),
            self.half_period_frame,
            self.setting_widget,
            self.tr("Range format: minimum, maximum, step. Used only for manual repeating folds."),
        )

        self.angle_step_frame = SpinBoxUnitInputFrame(self)
        self.angle_step_frame.set_input(["–", self.tr("step"), "°"], 3, "float")
        self.angle_step_frame.setRange(0.0, 360.0)
        for control in self.angle_step_frame.object_list:
            control.setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.angle_step_frame.set_input_value([15.0, 45.0, 15.0])
        self.angle_step_field = CompactField(
            self.tr("Angle change per layer"),
            self.angle_step_frame,
            self.setting_widget,
            self.tr("Range format: minimum, maximum, step. The sign reverses at each fold."),
        )

        self.sequence_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.sequence_combo,
            [
                "Clockwise then counterclockwise",
                "Counterclockwise then clockwise",
                "Both",
            ],
        )
        set_combo_value(self.sequence_combo, "Clockwise then counterclockwise")
        self.sequence_field = CompactField(
            self.tr("Fold direction"),
            self.sequence_combo,
            self.setting_widget,
            self.tr("Both always emits complete opposite-direction pairs; the output limit never splits a pair."),
        )

        pattern_section = InspectorSection(
            self.tr("Triangular phase pattern"),
            self.setting_widget,
            self.tr("Spin phase advances layer by layer, reaches a turning point, and then returns."),
        )
        pattern_grid = ResponsiveFormGrid(pattern_section)
        pattern_grid.add_field(self.half_period_mode_field, span=2)
        pattern_grid.add_field(self.half_period_field, span=2)
        pattern_grid.add_field(self.angle_step_field, span=2)
        pattern_grid.add_field(self.sequence_field, span=2)
        pattern_section.addWidget(pattern_grid)

        self.phase_frame = SpinBoxUnitInputFrame(self)
        self.phase_frame.set_input(["–", self.tr("step"), "°"], 3, "float")
        self.phase_frame.setRange(-360.0, 360.0)
        for control in self.phase_frame.object_list:
            control.setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.phase_frame.set_input_value([0.0, 0.0, 15.0])
        self.phase_field = CompactField(
            self.tr("Global phase offsets"),
            self.phase_frame,
            self.setting_widget,
            self.tr("Each phase rotates the complete folded texture within the spin plane."),
        )

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

        self.apply_edit = ElementLineEdit(self.setting_widget, multiple=True)
        self.apply_edit.setPlaceholderText(self.tr("All elements"))
        self.apply_field = CompactField(
            self.tr("Target elements"),
            self.apply_edit,
            self.setting_widget,
            self.tr("Comma-separated symbols such as Fe,Co; empty selects all elements."),
        )

        source_section = InspectorSection(self.tr("Moment source and targets"), self.setting_widget)
        source_grid = ResponsiveFormGrid(source_section)
        source_grid.add_field(self.source_field, span=2)
        source_grid.add_field(self.map_field, span=2)
        source_grid.add_field(self.default_field, span=2)
        source_grid.add_field(self.apply_field, span=2)
        source_section.addWidget(source_grid)
        self.source_section = source_section
        self.source_section.hide()

        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input("", 1, "int")
        self.max_output_frame.setRange(1, 999999)
        self.max_output_frame.set_input_value([100])
        self.max_output_frame.setFixedWidth(132)
        self.max_output_field = CompactField(
            self.tr("Maximum structures per input"),
            self.max_output_frame,
            self.setting_widget,
            self.tr("The scan stops at this budget without splitting an opposite-direction pair."),
            inline=True,
            input_max_width=132,
        )

        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        pattern_section.addWidget(self.output_preview)
        generation_section = InspectorSection(self.tr("Generation"), self.setting_widget)
        generation_section.addWidget(self.max_output_field)
        self.generation_section = generation_section
        self.generation_section.hide()

        self.advanced_checkbox = CheckBox(
            self.tr("Advanced phase, moment, and output settings"), self.setting_widget
        )
        self.advanced_checkbox.setChecked(False)

        self.settingLayout.addWidget(geometry_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(pattern_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_checkbox, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.phase_field, 3, 0, 1, 3)
        self.settingLayout.addWidget(self.source_section, 4, 0, 1, 3)
        self.settingLayout.addWidget(self.generation_section, 5, 0, 1, 3)

        self.half_period_mode_combo.currentIndexChanged.connect(self._update_half_period_mode_widgets)
        self.plane_mode_combo.currentIndexChanged.connect(self._update_plane_mode_widgets)
        self.source_combo.currentIndexChanged.connect(self._update_magnitude_source_widgets)
        self.advanced_checkbox.toggled.connect(self._update_advanced_widgets)
        for frame in (
            self.layer_axis_frame,
            self.plane_normal_frame,
            self.layer_tol_frame,
            self.half_period_frame,
            self.angle_step_frame,
            self.phase_frame,
            self.default_frame,
            self.max_output_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._parameters_changed)
        self.sequence_combo.currentIndexChanged.connect(self._parameters_changed)
        for control in self.layer_axis_frame.object_list:
            control.valueChanged.connect(self._sync_followed_plane_normal)
        self._update_plane_mode_widgets()
        self._update_half_period_mode_widgets()
        self._update_advanced_widgets()
        self._update_output_preview()

    def _parameters_changed(self, *_args):
        self._update_output_preview()
        self.refresh_compact_presentation()

    def _update_magnitude_source_widgets(self, *_args):
        use_map = self.advanced_checkbox.isChecked() and combo_value(self.source_combo) == "Map/default magnitude"
        for field in (self.map_field, self.default_field):
            field.setVisible(use_map)
            field.setEnabled(use_map)

    def _sync_followed_plane_normal(self, *_args):
        if combo_value(self.plane_mode_combo) != "Follow layer direction":
            return
        self.plane_normal_frame.set_input_value(self.layer_axis_frame.get_input_value())

    def _update_plane_mode_widgets(self, *_args):
        custom = combo_value(self.plane_mode_combo) == "Custom plane normal"
        self.plane_normal_field.setVisible(custom)
        self.plane_normal_field.setEnabled(custom)
        if not custom:
            self._sync_followed_plane_normal()
        self._parameters_changed()

    def _update_half_period_mode_widgets(self, *_args):
        manual = combo_value(self.half_period_mode_combo) == "Manual"
        self.half_period_field.setVisible(manual)
        self.half_period_field.setEnabled(manual)
        self._parameters_changed()

    def _update_advanced_widgets(self, *_args):
        visible = self.advanced_checkbox.isChecked()
        for widget in (self.phase_field, self.source_section, self.generation_section):
            widget.setVisible(visible)
            widget.setEnabled(visible)
        self._update_magnitude_source_widgets()

    def _update_output_preview(self, *_args):
        theoretical, generated = FoldedHelixOperation.output_counts(self.get_params())
        if generated < theoretical:
            text = self.tr("Per valid input: {generated} of {theoretical} planned structures fit the output limit.").format(
                generated=generated,
                theoretical=theoretical,
            )
        else:
            text = self.tr("Structures per valid input: {generated}.").format(generated=generated)
        self.output_preview.setText(text)

    def get_summary_text(self) -> str:
        _theoretical, generated = FoldedHelixOperation.output_counts(self.get_params())
        mode = self.tr("Auto") if combo_value(self.half_period_mode_combo) == "Auto from layer count" else self.tr("Manual")
        return self.tr("{mode} · {angles} · {count} outputs").format(
            mode=mode,
            angles=self.tr("{minimum}–{maximum}°/layer").format(
                minimum=f"{float(self.angle_step_frame.get_input_value()[0]):.4g}",
                maximum=f"{float(self.angle_step_frame.get_input_value()[1]):.4g}",
            ),
            count=generated,
        )

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def get_guidance_text(self) -> str:
        _theoretical, generated = FoldedHelixOperation.output_counts(self.get_params())
        text = self.tr(
            "Check that at least three layers are detected and that the phase reaches a turning layer before returning."
        )
        if self._preview_input_count:
            text += " " + self.tr("Planned maximum: {total} outputs.").format(
                total=self._preview_input_count * generated
            )
        return text

    def create_operation(self):
        return FoldedHelixOperation()

    def get_params(self) -> FoldedHelixParams:
        layer_axis = self.layer_axis_frame.get_input_value()
        plane_normal = (
            layer_axis
            if combo_value(self.plane_mode_combo) == "Follow layer direction"
            else self.plane_normal_frame.get_input_value()
        )
        return FoldedHelixParams(
            layer_axis=layer_axis,
            plane_normal=plane_normal,
            layer_tolerance=float(self.layer_tol_frame.get_input_value()[0]),
            half_period_mode=combo_value(self.half_period_mode_combo),
            half_period_layers=self.half_period_frame.get_input_value(),
            angle_step_range=self.angle_step_frame.get_input_value(),
            phase_range=self.phase_frame.get_input_value(),
            sequence_mode=combo_value(self.sequence_combo),
            magnitude_source=combo_value(self.source_combo),
            magmom_map=self.map_edit.text(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            apply_elements=self.apply_edit.text(),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
        )

    def set_params(self, params: FoldedHelixParams) -> None:
        self.layer_axis_frame.set_input_value([float(v) for v in params.layer_axis])
        self.plane_normal_frame.set_input_value([float(v) for v in params.plane_normal])
        set_combo_value(
            self.plane_mode_combo,
            "Follow layer direction"
            if self._same_direction(params.layer_axis, params.plane_normal)
            else "Custom plane normal",
        )
        self.layer_tol_frame.set_input_value([float(params.layer_tolerance)])
        set_combo_value(self.half_period_mode_combo, params.half_period_mode)
        self.half_period_frame.set_input_value([int(v) for v in params.half_period_layers])
        self.angle_step_frame.set_input_value([float(v) for v in params.angle_step_range])
        self.phase_frame.set_input_value([float(v) for v in params.phase_range])
        set_combo_value(self.sequence_combo, params.sequence_mode)
        set_combo_value(self.source_combo, params.magnitude_source)
        self.map_edit.setText(params.magmom_map)
        self.default_frame.set_input_value([float(params.default_moment)])
        self.apply_edit.setText(params.apply_elements)
        self.max_output_frame.set_input_value([int(params.max_outputs)])
        self.advanced_checkbox.setChecked(
            tuple(float(v) for v in params.phase_range) != (0.0, 0.0, 15.0)
            or params.magnitude_source != "Existing initial magmoms"
            or bool(params.magmom_map.strip())
            or float(params.default_moment) != 0.0
            or bool(params.apply_elements.strip())
            or int(params.max_outputs) != 100
        )
        self._update_half_period_mode_widgets()
        self._update_plane_mode_widgets()
        self._update_advanced_widgets()
        self._update_output_preview()

    @staticmethod
    def _same_direction(left, right) -> bool:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
        left_norm = sum(value * value for value in left_values) ** 0.5
        right_norm = sum(value * value for value in right_values) ** 0.5
        if left_norm <= 1e-12 or right_norm <= 1e-12:
            return False
        return all(
            abs(a / left_norm - b / right_norm) <= 1e-8
            for a, b in zip(left_values, right_values)
        )

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = FoldedHelixParams(**raw_params)
        else:
            params = FoldedHelixParams(
                layer_axis=data_dict.get("layer_axis", [0.0, 0.0, 1.0]),
                plane_normal=data_dict.get("plane_normal", [0.0, 0.0, 1.0]),
                layer_tolerance=data_dict.get("layer_tolerance", [0.05])[0],
                half_period_mode=data_dict.get("half_period_mode", "Auto from layer count"),
                half_period_layers=data_dict.get("half_period_layers", [2, 4, 1]),
                angle_step_range=data_dict.get("angle_step_range", [15.0, 45.0, 15.0]),
                phase_range=data_dict.get("phase_range", [0.0, 0.0, 15.0]),
                sequence_mode=data_dict.get("sequence_mode", "Clockwise then counterclockwise"),
                magnitude_source=data_dict.get("magnitude_source", "Existing initial magmoms"),
                magmom_map=data_dict.get("magmom_map", ""),
                default_moment=data_dict.get("default_moment", [0.0])[0],
                apply_elements=data_dict.get("apply_elements", ""),
                max_outputs=data_dict.get("max_outputs", [100])[0],
            )
        self.set_params(params)
