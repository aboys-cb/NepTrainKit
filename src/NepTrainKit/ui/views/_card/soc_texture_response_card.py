"""Thin UI for rigid-rotation and finite-q response groups."""

from __future__ import annotations

import math

from qfluentwidgets import CaptionLabel, CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.magnetic_response import MagneticResponseScanOperation, TextureMagneticResponseParams
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    DirectionInput,
    InspectorSection,
    MakeDataCard,
    NumericScanInput,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class SOCTextureResponseCard(MakeDataCard):
    """Build rigid anisotropy and signed finite-q texture response paths."""

    group = "Magnetism"
    card_name = "SOC / Texture Response"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("SOC / Texture Response"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("soc_texture_response_card_widget")
        self.kind_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.kind_combo,
            ["Global anisotropy", "Bulk / Bloch", "Interfacial / Cycloidal", "General spiral"],
        )
        set_combo_value(self.kind_combo, "Global anisotropy")

        self.scan_input = NumericScanInput(
            self.setting_widget, minimum=-180.0, maximum=180.0, decimals=3
        )
        self.scan_input.set_range(-2.0, 2.0, 1.0)
        self.scan_field = CompactField(
            self.tr("Rigid rotation scan (degrees)"),
            self.scan_input,
            self.setting_widget,
            self.tr("A single ordered path rotates the complete spin texture relative to the fixed lattice."),
        )

        path_section = InspectorSection(
            self.tr("Texture path"),
            self.setting_widget,
            self.tr("Choose anisotropy or a spiral geometry; the scan meaning and required directions update automatically."),
        )
        path_section.addWidget(CompactField(self.tr("Preset"), self.kind_combo, path_section))
        path_section.addWidget(self.scan_field)

        self.axis_input = DirectionInput(self.setting_widget, default=(0.0, 1.0, 0.0))
        self.axis_field = CompactField(
            self.tr("Rigid rotation axis (Cartesian)"), self.axis_input, self.setting_widget
        )
        self.time_reversal = CheckBox(self.tr("Include global time-reversal control S → −S"), self.setting_widget)

        self.q_direction = DirectionInput(self.setting_widget, default=(0.0, 0.0, 1.0))
        self.q_direction_field = CompactField(
            self.tr("Propagation direction q (Cartesian)"), self.q_direction, self.setting_widget
        )
        self.q_magnitude_frame = SpinBoxUnitInputFrame(self)
        self.q_magnitude_frame.set_input("1/A", 1, "float")
        self.q_magnitude_frame.setRange(0.0, 100.0)
        self.q_magnitude_frame.setDecimals(8)
        self.q_magnitude_frame.set_input_value([0.1])
        self.q_magnitude_field = CompactField(
            self.tr("Base |q|"),
            self.q_magnitude_frame,
            self.setting_widget,
            self.tr("The signed scan multiplies this base wave-vector magnitude."),
        )
        self.plane_input = DirectionInput(self.setting_widget, default=(0.0, 1.0, 0.0))
        self.plane_field = CompactField(
            self.tr("Rotation-plane normal (Cartesian)"), self.plane_input, self.setting_widget
        )
        self.surface_input = DirectionInput(self.setting_widget, default=(0.0, 0.0, 1.0))
        self.surface_field = CompactField(
            self.tr("Surface normal (Cartesian)"),
            self.surface_input,
            self.setting_widget,
            self.tr("For the cycloidal preset, spins rotate in the plane spanned by q and this normal."),
        )

        geometry_section = InspectorSection(self.tr("Directions"), self.setting_widget)
        geometry_grid = ResponsiveFormGrid(geometry_section, two_column_threshold=520)
        for field in (
            self.axis_field,
            self.q_direction_field,
            self.q_magnitude_field,
            self.plane_field,
            self.surface_field,
        ):
            geometry_grid.add_field(field)
        geometry_section.addWidget(geometry_grid)
        geometry_section.addWidget(self.time_reversal)

        self.advanced_checkbox = CheckBox(self.tr("Show phase, cone, closure check, and output limit"), self.setting_widget)
        self.cone_frame = SpinBoxUnitInputFrame(self)
        self.cone_frame.set_input("", 1, "float")
        self.cone_frame.setRange(-1.0, 1.0)
        self.cone_frame.setDecimals(6)
        self.cone_frame.set_input_value([0.0])
        self.cone_field = CompactField(self.tr("Cone component"), self.cone_frame, self.setting_widget)
        self.phase_frame = SpinBoxUnitInputFrame(self)
        self.phase_frame.set_input("deg", 1, "float")
        self.phase_frame.setRange(-360.0, 360.0)
        self.phase_frame.setDecimals(6)
        self.phase_frame.set_input_value([0.0])
        self.phase_field = CompactField(self.tr("Initial phase"), self.phase_frame, self.setting_widget)
        self.commensurate = CheckBox(self.tr("Require the spiral to close in the current periodic cell"), self.setting_widget)
        self.commensurate.setChecked(True)
        self.limit_frame = SpinBoxUnitInputFrame(self)
        self.limit_frame.set_input("", 1, "int")
        self.limit_frame.setRange(3, 999999)
        self.limit_frame.set_input_value([100])
        self.limit_field = CompactField(
            self.tr("Maximum structures"),
            self.limit_frame,
            self.setting_widget,
            self.tr("Only complete signed-q or rotation groups are retained."),
        )
        self.advanced_section = InspectorSection(self.tr("Advanced texture controls"), self.setting_widget)
        advanced_grid = ResponsiveFormGrid(self.advanced_section)
        advanced_grid.add_field(self.cone_field)
        advanced_grid.add_field(self.phase_field)
        advanced_grid.add_field(self.limit_field, span=2)
        self.advanced_section.addWidget(advanced_grid)
        self.advanced_section.addWidget(self.commensurate)
        self.advanced_section.hide()

        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        output_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        output_section.addWidget(self.output_preview)

        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.addWidget(path_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(geometry_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_checkbox, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_section, 3, 0, 1, 3)
        self.settingLayout.addWidget(output_section, 4, 0, 1, 3)

        self.kind_combo.currentIndexChanged.connect(self._update_widgets)
        self.time_reversal.toggled.connect(self._update_output_preview)
        self.advanced_checkbox.toggled.connect(self.advanced_section.setVisible)
        for spin in self.scan_input.range_frame.object_list:
            spin.valueChanged.connect(self._update_output_preview)
        self.scan_input.custom_edit.textChanged.connect(self._update_output_preview)
        self._update_widgets()

    def _update_widgets(self, *_args):
        kind = combo_value(self.kind_combo)
        global_scan = kind == "Global anisotropy"
        self.scan_field.set_label(
            self.tr("Rigid rotation scan (degrees)") if global_scan else self.tr("Signed q scan (multiples of base q)")
        )
        self.scan_field.set_helper_text(
            self.tr("Angles form one sortable rigid-rotation path; the lattice and relative spin topology stay fixed.")
            if global_scan
            else self.tr("Use a symmetric scan such as -2, -1, 0, +1, +2 to compare opposite chirality and even/odd q response.")
        )
        self.axis_field.setVisible(global_scan)
        self.time_reversal.setVisible(global_scan)
        self.q_direction_field.setVisible(not global_scan)
        self.q_magnitude_field.setVisible(not global_scan)
        self.plane_field.setVisible(kind == "General spiral")
        self.surface_field.setVisible(kind == "Interfacial / Cycloidal")
        self.cone_field.setVisible(not global_scan)
        self.phase_field.setVisible(not global_scan)
        self.commensurate.setVisible(not global_scan)
        self._update_output_preview()

    def _update_output_preview(self, *_args):
        try:
            count = self.scan_input.count()
            groups = 2 if combo_value(self.kind_combo) == "Global anisotropy" and self.time_reversal.isChecked() else 1
            total = count * groups
            detail = self.tr(" across the normal and time-reversed groups") if groups == 2 else ""
            self.output_preview.setText(
                self.tr("{total} structures{detail}; each group contains {count} ordered coordinates.").format(
                    total=total, detail=detail, count=count
                )
            )
        except ValueError as exc:
            self.output_preview.setText(str(exc))

    def get_summary_text(self) -> str:
        return self.tr("{preset} · {count} ordered coordinates").format(
            preset=self.kind_combo.currentText(), count=self.scan_input.count()
        )

    def get_guidance_text(self) -> str:
        return self.tr(
            "Bloch, cycloidal, and general are generation presets, not fixed Hamiltonian models. Non-commensurate periodic spirals fail closed."
        )

    def create_operation(self):
        return MagneticResponseScanOperation()

    def get_params(self):
        direction = self.q_direction.vector()
        q_magnitude = float(self.q_magnitude_frame.get_input_value()[0])
        q_vector = tuple(q_magnitude * value for value in direction)
        return TextureMagneticResponseParams(
            response_kind=combo_value(self.kind_combo),
            coordinate_scan=self.scan_input.scan_text(),
            rotation_axis=self.axis_input.vector(),
            q_vector_cart=q_vector,
            plane_normal=self.plane_input.vector(),
            surface_normal=self.surface_input.vector(),
            cone_component=float(self.cone_frame.get_input_value()[0]),
            phase_deg=float(self.phase_frame.get_input_value()[0]),
            include_time_reversal=self.time_reversal.isChecked(),
            require_commensurate=self.commensurate.isChecked(),
            max_outputs=int(self.limit_frame.get_input_value()[0]),
        )

    def set_params(self, params):
        set_combo_value(self.kind_combo, params.response_kind)
        self.scan_input.set_scan_text(params.coordinate_scan)
        self.axis_input.set_vector(params.rotation_axis)
        q_vector = tuple(float(value) for value in params.q_vector_cart)
        q_magnitude = math.sqrt(sum(value * value for value in q_vector))
        self.q_magnitude_frame.set_input_value([q_magnitude])
        self.q_direction.set_vector(q_vector if q_magnitude > 1.0e-12 else (0.0, 0.0, 1.0))
        self.plane_input.set_vector(params.plane_normal)
        self.surface_input.set_vector(params.surface_normal)
        self.cone_frame.set_input_value([params.cone_component])
        self.phase_frame.set_input_value([params.phase_deg])
        self.time_reversal.setChecked(params.include_time_reversal)
        self.commensurate.setChecked(params.require_commensurate)
        self.limit_frame.set_input_value([params.max_outputs])
        self._update_widgets()

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.set_params(TextureMagneticResponseParams(**data_dict.get("params", {})))
