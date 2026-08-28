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

        self.path_section = InspectorSection(
            self.tr("Texture path"),
            self.setting_widget,
            "",
        )
        self.path_section.addWidget(
            CompactField(self.tr("Preset"), self.kind_combo, self.path_section)
        )
        self.path_section.addWidget(self.scan_field)

        self.axis_input = DirectionInput(self.setting_widget, default=(0.0, 1.0, 0.0))
        self.axis_field = CompactField(
            self.tr("Rigid rotation axis (Cartesian)"), self.axis_input, self.setting_widget
        )
        self.time_reversal = CheckBox(self.tr("Include global time-reversal control S → −S"), self.setting_widget)

        self.q_definition_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.q_definition_combo,
            ["Cell reciprocal vector", "Cartesian vector"],
        )
        set_combo_value(self.q_definition_combo, "Cell reciprocal vector")
        self.q_definition_field = CompactField(
            self.tr("Base q definition"), self.q_definition_combo, self.setting_widget
        )
        self.q_reciprocal_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.q_reciprocal_frame.set_input("", 3, "int")
        self.q_reciprocal_frame.setRange(-100, 100)
        self.q_reciprocal_frame.set_input_value([1, 0, 0])
        self.q_reciprocal_field = CompactField(
            self.tr("Cell-reciprocal index (h, k, l)"),
            self.q_reciprocal_frame,
            self.setting_widget,
            self.tr("Integer indices derive q from the current cell and close exactly across its periodic vectors."),
        )
        self.q_direction = DirectionInput(self.setting_widget, default=(0.0, 0.0, 1.0))
        self.q_direction_field = CompactField(
            self.tr("Cartesian q direction"),
            self.q_direction,
            self.setting_widget,
            self.tr("This is a laboratory Cartesian direction, not a Miller index or lattice axis."),
        )
        self.q_magnitude_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.q_magnitude_frame.set_input(self.tr("Å⁻¹"), 1, "float")
        self.q_magnitude_frame.setRange(0.0, 100.0)
        self.q_magnitude_frame.setDecimals(8)
        self.q_magnitude_frame.set_input_value([0.1])
        self.q_magnitude_field = CompactField(
            self.tr("Base |q|"),
            self.q_magnitude_frame,
            self.setting_widget,
            self.tr("The signed scan multiplies this magnitude; periodic closure depends on the input cell."),
            inline=True,
            input_max_width=170,
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
            self.q_definition_field,
            self.q_reciprocal_field,
            self.q_direction_field,
            self.q_magnitude_field,
            self.plane_field,
            self.surface_field,
        ):
            geometry_grid.add_field(field)
        geometry_section.addWidget(geometry_grid)
        geometry_section.addWidget(self.time_reversal)

        self.advanced_checkbox = CheckBox(self.tr("Show advanced texture controls"), self.setting_widget)
        self.cone_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.cone_frame.set_input("", 1, "float")
        self.cone_frame.setRange(-1.0, 1.0)
        self.cone_frame.setDecimals(6)
        self.cone_frame.set_input_value([0.0])
        self.cone_field = CompactField(
            self.tr("Normal spin component m∥/|m|"),
            self.cone_frame,
            self.setting_widget,
            self.tr("0 gives a planar spiral; ±1 removes the rotating in-plane component."),
            inline=True,
            input_max_width=150,
        )
        self.cone_frame.setMinimumWidth(132)
        self.phase_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.phase_frame.set_input(self.tr("°"), 1, "float")
        self.phase_frame.setRange(-360.0, 360.0)
        self.phase_frame.setDecimals(6)
        self.phase_frame.set_input_value([0.0])
        self.phase_field = CompactField(
            self.tr("Initial phase φ₀"),
            self.phase_frame,
            self.setting_widget,
            self.tr("The generated texture uses φᵢ = q·rᵢ + φ₀ with current Cartesian positions."),
            inline=True,
            input_max_width=150,
        )
        self.phase_frame.setMinimumWidth(150)
        self.commensurate = CheckBox(self.tr("Require periodic closure"), self.setting_widget)
        self.commensurate.setChecked(True)
        self.commensurate_field = CompactField(
            self.tr("Periodic boundary"),
            self.commensurate,
            self.setting_widget,
            self.tr("For every periodic cell vector aᵢ, q·aᵢ/(2π) must be an integer."),
        )
        self.limit_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.limit_frame.set_input("", 1, "int")
        self.limit_frame.setRange(3, 999999)
        self.limit_frame.set_input_value([100])
        self.limit_field = CompactField(
            self.tr("Maximum structures"),
            self.limit_frame,
            self.setting_widget,
            self.tr("Only complete signed-q or rotation groups are retained."),
            inline=True,
            input_max_width=150,
        )
        self.advanced_section = InspectorSection(self.tr("Advanced texture controls"), self.setting_widget)
        advanced_grid = ResponsiveFormGrid(self.advanced_section)
        advanced_grid.add_field(self.cone_field, span=2)
        advanced_grid.add_field(self.phase_field, span=2)
        advanced_grid.add_field(self.commensurate_field, span=2)
        advanced_grid.add_field(self.limit_field, span=2)
        self.advanced_section.addWidget(advanced_grid)
        self.advanced_section.hide()

        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        output_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        output_section.addWidget(self.output_preview)

        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.addWidget(self.path_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(geometry_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_checkbox, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_section, 3, 0, 1, 3)
        self.settingLayout.addWidget(output_section, 4, 0, 1, 3)

        self.kind_combo.currentIndexChanged.connect(self._update_widgets)
        self.q_definition_combo.currentIndexChanged.connect(self._update_widgets)
        self.time_reversal.toggled.connect(self._update_output_preview)
        self.advanced_checkbox.toggled.connect(self.advanced_section.setVisible)
        for spin in self.scan_input.range_frame.object_list:
            spin.valueChanged.connect(self._update_output_preview)
        for spin in (
            *self.q_reciprocal_frame.object_list,
            *self.q_magnitude_frame.object_list,
            *self.limit_frame.object_list,
        ):
            spin.valueChanged.connect(self._update_output_preview)
        self.scan_input.custom_edit.textChanged.connect(self._update_output_preview)
        self._update_widgets()

    def _update_widgets(self, *_args):
        kind = combo_value(self.kind_combo)
        global_scan = kind == "Global anisotropy"
        reciprocal_q = combo_value(self.q_definition_combo) == "Cell reciprocal vector"
        descriptions = {
            "Global anisotropy": self.tr(
                "Rigidly rotate every input spin together; relative spin angles and the lattice stay fixed."
            ),
            "Bulk / Bloch": self.tr(
                "Regenerate a finite-q texture from the input moment magnitudes; the rotation-plane normal is q."
            ),
            "Interfacial / Cycloidal": self.tr(
                "Regenerate a finite-q texture whose spins rotate in the plane spanned by q and the surface normal."
            ),
            "General spiral": self.tr(
                "Regenerate a finite-q texture in the plane specified by its normal."
            ),
        }
        self.path_section.description_label.setText(descriptions[kind])
        self.path_section.description_label.show()
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
        self.q_definition_field.setVisible(not global_scan)
        self.q_reciprocal_field.setVisible(not global_scan and reciprocal_q)
        self.q_direction_field.setVisible(not global_scan and not reciprocal_q)
        self.q_magnitude_field.setVisible(not global_scan and not reciprocal_q)
        self.plane_field.setVisible(kind == "General spiral")
        self.surface_field.setVisible(kind == "Interfacial / Cycloidal")
        self.cone_field.setVisible(not global_scan)
        self.phase_field.setVisible(not global_scan)
        self.commensurate_field.setVisible(not global_scan)
        self._update_output_preview()

    def _update_output_preview(self, *_args):
        try:
            count = self.scan_input.count()
            groups = 2 if combo_value(self.kind_combo) == "Global anisotropy" and self.time_reversal.isChecked() else 1
            total = count * groups
            limit = int(self.limit_frame.get_input_value()[0])
            if limit < total:
                self.output_preview.setText(
                    self.tr(
                        "This path needs {total} structures, but the current limit is {limit}."
                    ).format(total=total, limit=limit)
                )
                return
            kind = combo_value(self.kind_combo)
            if groups == 2:
                detail = self.tr("two complete groups: normal and time reversed")
            elif kind == "Global anisotropy":
                detail = self.tr("one complete rigid-rotation group")
            elif combo_value(self.q_definition_combo) == "Cell reciprocal vector":
                detail = self.tr("one signed-q group; base q is derived from the input cell")
            else:
                q_magnitude = float(self.q_magnitude_frame.get_input_value()[0])
                if q_magnitude <= 1.0e-12:
                    self.output_preview.setText(
                        self.tr("The Cartesian base q vector must be non-zero.")
                    )
                    return
                period = 2.0 * math.pi / q_magnitude
                detail = self.tr("one signed-q group; base period {period:.3f} Å").format(
                    period=period
                )
            self.output_preview.setText(
                self.tr("{total} structures in {detail}.").format(total=total, detail=detail)
            )
        except ValueError as exc:
            self.output_preview.setText(str(exc))

    def get_summary_text(self) -> str:
        return self.tr("{preset} · {count} per group").format(
            preset=self.kind_combo.currentText(), count=self.scan_input.count()
        )

    def get_guidance_text(self) -> str:
        kind = combo_value(self.kind_combo)
        if kind == "Global anisotropy":
            return self.tr(
                "Compare the reference and rotated frames after an SOC-enabled calculation; all relative spin angles should stay fixed."
            )
        if combo_value(self.q_definition_combo) == "Cell reciprocal vector":
            return self.tr(
                "The integer reciprocal index closes in the current cell. The q=0 frame is a generated collinear reference, not the input spin directions."
            )
        return self.tr(
            "Check periodic closure for Cartesian q. The q=0 frame is a generated collinear reference, not the input spin directions."
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
            q_definition=combo_value(self.q_definition_combo),
            q_reciprocal_index=tuple(
                int(value) for value in self.q_reciprocal_frame.get_input_value()
            ),
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
        set_combo_value(
            self.q_definition_combo,
            params.q_definition or "Cartesian vector",
        )
        self.q_reciprocal_frame.set_input_value(
            [int(value) for value in params.q_reciprocal_index]
        )
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
        self.advanced_checkbox.setChecked(
            (
                params.response_kind != "Global anisotropy"
                and (
                    params.cone_component != 0.0
                    or params.phase_deg != 0.0
                    or not params.require_commensurate
                )
            )
            or params.max_outputs != 100
        )
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
