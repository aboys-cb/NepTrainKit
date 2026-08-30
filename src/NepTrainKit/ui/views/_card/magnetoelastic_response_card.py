"""Thin UI for linked structural and spin-response grids."""

import math

from qfluentwidgets import CaptionLabel, CheckBox, ComboBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.magnetic_response import MagneticResponseScanOperation, MagnetoelasticResponseParams
from NepTrainKit.ui.messages import MessageManager
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


def _scaled_scan_text(text: str, factor: float) -> str:
    return ",".join(f"{float(item.strip()) * factor:.12g}" for item in text.split(",") if item.strip())


def _legacy_shear_direction(axis) -> tuple[float, float, float]:
    """Reproduce the hidden shear-direction choice used by legacy workflows."""
    values = [float(value) for value in axis]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1.0e-12:
        return (0.0, 1.0, 0.0)
    x, y, z = (value / norm for value in values)
    reference = (1.0, 0.0, 0.0) if abs(x) < 0.9 else (0.0, 1.0, 0.0)
    cross = (
        y * reference[2] - z * reference[1],
        z * reference[0] - x * reference[2],
        x * reference[1] - y * reference[0],
    )
    cross_norm = math.sqrt(sum(value * value for value in cross))
    return tuple(value / cross_norm for value in cross)


@CardManager.register_card
class MagnetoelasticResponseCard(MakeDataCard):
    """Combine structural coordinates with matched complete spin probes."""

    group = "Magnetism"
    card_name = "Magnetoelastic Grid"
    menu_icon = r":/images/src/images/scaling.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Magnetoelastic"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("magnetoelastic_response_card_widget")
        self.mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.mode_combo,
            ["Isotropic volume", "Uniaxial strain", "Biaxial strain", "Symmetric shear", "Bain / tetragonal"],
        )
        set_combo_value(self.mode_combo, "Isotropic volume")

        self.struct_scan = NumericScanInput(
            self.setting_widget, minimum=-50.0, maximum=50.0, decimals=3, suffix="%"
        )
        self.struct_scan.set_range(-2.0, 2.0, 1.0)
        self.struct_scan_field = CompactField(
            self.tr("Volume change (%)"),
            self.struct_scan,
            self.setting_widget,
            self.tr("Minimum, maximum, and step. Values are converted to strain fractions internally."),
        )

        self.spin_scan = NumericScanInput(
            self.setting_widget, minimum=-180.0, maximum=180.0, decimals=3
        )
        self.spin_scan.set_range(-2.0, 2.0, 2.0)
        self.spin_scan_field = CompactField(
            self.tr("Spin rotation scan (degrees)"),
            self.spin_scan,
            self.setting_widget,
            self.tr("The same signed rotation scan is repeated at every lattice point."),
        )

        self.target_edit = LineEdit(self.setting_widget)
        self.target_edit.setText("1")
        self.target_edit.setPlaceholderText(self.tr("For example: 1 or 1,3-5"))
        self.target_field = CompactField(
            self.tr("Atoms rotated together (1-based)"),
            self.target_edit,
            self.setting_widget,
            self.tr("All listed atoms rotate together in each frame; ranges such as 3-5 are accepted."),
        )

        self.path_section = InspectorSection(
            self.tr("Response grid"),
            self.setting_widget,
            "",
        )
        self.path_section.addWidget(
            CompactField(self.tr("Lattice path"), self.mode_combo, self.path_section)
        )
        self.path_section.addWidget(self.struct_scan_field)
        self.path_section.addWidget(self.spin_scan_field)
        self.path_section.addWidget(self.target_field)

        self.rotation_input = DirectionInput(self.setting_widget, default=(0.0, 1.0, 0.0))
        self.rotation_field = CompactField(
            self.tr("Spin axis"),
            self.rotation_input,
            self.setting_widget,
            self.tr("The selected spins rotate rigidly about this laboratory Cartesian axis."),
            inline=True,
            input_max_width=176,
        )
        self.strain_input = DirectionInput(self.setting_widget, default=(0.0, 0.0, 1.0))
        self.strain_field = CompactField(
            self.tr("Loading direction"),
            self.strain_input,
            self.setting_widget,
            inline=True,
            input_max_width=176,
        )
        self.shear_direction_input = DirectionInput(
            self.setting_widget, default=(0.0, 1.0, 0.0)
        )
        self.shear_direction_field = CompactField(
            self.tr("Shear direction v"),
            self.shear_direction_input,
            self.setting_widget,
            self.tr("The two shear directions must be perpendicular."),
            inline=True,
            input_max_width=176,
        )
        self.bain_axis_combo = ComboBox(self.setting_widget)
        for axis in ("a", "b", "c"):
            self.bain_axis_combo.addItem(axis, userData=axis)
        set_combo_value(self.bain_axis_combo, "c")
        self.bain_axis_field = CompactField(
            self.tr("Tetragonal axis"),
            self.bain_axis_combo,
            self.setting_widget,
            self.tr("Choose lattice vector a, b, or c; the Bain path preserves cell volume."),
            inline=True,
            input_max_width=176,
        )
        self.direction_section = InspectorSection(
            self.tr("Directions"), self.setting_widget
        )
        direction_grid = ResponsiveFormGrid(
            self.direction_section, two_column_threshold=520
        )
        direction_grid.add_field(self.rotation_field, span=2)
        direction_grid.add_field(self.strain_field, span=2)
        direction_grid.add_field(self.shear_direction_field, span=2)
        direction_grid.add_field(self.bain_axis_field, span=2)
        self.direction_section.addWidget(direction_grid)

        self.advanced_checkbox = CheckBox(
            self.tr("Show output limit"), self.setting_widget
        )
        self.limit_frame = SpinBoxUnitInputFrame(self)
        self.limit_frame.set_input("", 1, "int")
        self.limit_frame.setRange(3, 999999)
        self.limit_frame.set_input_value([100])
        self.limit_field = CompactField(
            self.tr("Maximum structures"),
            self.limit_frame,
            self.setting_widget,
            self.tr("Only complete spin-scan groups are retained."),
            inline=True,
            input_max_width=150,
        )
        self.limit_frame.setMinimumWidth(120)
        self.advanced_section = InspectorSection(
            self.tr("Output limit"), self.setting_widget
        )
        advanced_grid = ResponsiveFormGrid(self.advanced_section)
        advanced_grid.add_field(self.limit_field, span=2)
        self.advanced_section.addWidget(advanced_grid)
        self.advanced_section.hide()

        self.legacy_notice = CaptionLabel("", self.setting_widget)
        self.legacy_notice.setWordWrap(True)
        self.legacy_notice.setStyleSheet("color:#c56a00; font-weight:600;")
        self.legacy_notice.hide()

        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        output_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        output_section.addWidget(self.output_preview)

        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.addWidget(self.path_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(self.direction_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_checkbox, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_section, 3, 0, 1, 3)
        self.settingLayout.addWidget(self.legacy_notice, 4, 0, 1, 3)
        self.settingLayout.addWidget(output_section, 5, 0, 1, 3)

        self.mode_combo.currentIndexChanged.connect(self._update_widgets)
        self.advanced_checkbox.toggled.connect(self.advanced_section.setVisible)
        for spin in (
            *self.struct_scan.range_frame.object_list,
            *self.spin_scan.range_frame.object_list,
            *self.limit_frame.object_list,
        ):
            spin.valueChanged.connect(self._update_output_preview)
        self.struct_scan.custom_edit.textChanged.connect(self._update_output_preview)
        self.spin_scan.custom_edit.textChanged.connect(self._update_output_preview)
        self._update_widgets()

    def _update_widgets(self, *_args):
        mode = combo_value(self.mode_combo)
        labels = {
            "Isotropic volume": self.tr("Volume change (%)"),
            "Uniaxial strain": self.tr("Axial strain (%)"),
            "Biaxial strain": self.tr("In-plane strain (%)"),
            "Symmetric shear": self.tr("Shear strain (%)"),
            "Bain / tetragonal": self.tr("Tetragonal strain (%)"),
        }
        descriptions = {
            "Isotropic volume": self.tr(
                "Change the total cell volume, then repeat one complete local spin-rotation scan at every volume."
            ),
            "Uniaxial strain": self.tr(
                "Strain one Cartesian loading direction while leaving its perpendicular directions unchanged."
            ),
            "Biaxial strain": self.tr(
                "Apply equal strain in the Cartesian plane perpendicular to the selected normal."
            ),
            "Symmetric shear": self.tr(
                "Apply a symmetric shear defined by two perpendicular Cartesian directions."
            ),
            "Bain / tetragonal": self.tr(
                "Change one lattice vector relative to the other two while preserving cell volume."
            ),
        }
        self.path_section.description_label.setText(descriptions[mode])
        self.path_section.description_label.show()
        self.struct_scan_field.set_label(labels.get(mode, self.tr("Structural coordinate (%)")))
        if mode == "Uniaxial strain":
            self.strain_field.set_label(self.tr("Loading direction"))
        elif mode == "Biaxial strain":
            self.strain_field.set_label(self.tr("Unstrained normal"))
        elif mode == "Symmetric shear":
            self.strain_field.set_label(self.tr("Shear direction u"))
        self.strain_field.setVisible(
            mode in {"Uniaxial strain", "Biaxial strain", "Symmetric shear"}
        )
        self.shear_direction_field.setVisible(mode == "Symmetric shear")
        self.bain_axis_field.setVisible(mode == "Bain / tetragonal")
        self._update_output_preview()

    def _update_output_preview(self, *_args):
        try:
            structural_count = self.struct_scan.count()
            spin_count = self.spin_scan.count()
            total = structural_count * spin_count
            limit = int(self.limit_frame.get_input_value()[0])
            kept_lattice_points = min(structural_count, limit // spin_count)
            kept_total = kept_lattice_points * spin_count
            if kept_lattice_points == 0:
                text = self.tr(
                    "One complete lattice point needs {spin} structures; the current limit is {limit}."
                ).format(spin=spin_count, limit=limit)
            elif kept_total < total:
                text = self.tr(
                    "{total} requested; the limit keeps {kept} complete lattice points ({actual} structures)."
                ).format(
                    total=total,
                    kept=kept_lattice_points,
                    actual=kept_total,
                )
            else:
                text = self.tr(
                    "{structural} lattice points × {spin} spin rotations = {total} structures."
                ).format(
                    structural=structural_count,
                    spin=spin_count,
                    total=total,
                )
            self.output_preview.setText(text)
        except ValueError as exc:
            self.output_preview.setText(str(exc))

    def get_summary_text(self) -> str:
        return self.tr("{mode} · {structural}×{spin} grid").format(
            mode=self.mode_combo.currentText(), structural=self.struct_scan.count(), spin=self.spin_scan.count()
        )

    def get_guidance_text(self) -> str:
        return self.tr(
            "Input structures need finite non-zero vector spins. Every listed atom rotates together about the Cartesian spin axis at each lattice point."
        )

    def create_operation(self):
        return MagneticResponseScanOperation()

    def get_params(self):
        return MagnetoelasticResponseParams(
            structural_mode=combo_value(self.mode_combo),
            structural_scan=_scaled_scan_text(self.struct_scan.scan_text(), 0.01),
            spin_scan_deg=self.spin_scan.scan_text(),
            rotation_axis=self.rotation_input.vector(),
            target_indices=self.target_edit.text(),
            strain_axis=self.strain_input.vector(),
            shear_direction=self.shear_direction_input.vector(),
            bain_axis=combo_value(self.bain_axis_combo),
            max_outputs=int(self.limit_frame.get_input_value()[0]),
        )

    def set_params(self, params):
        set_combo_value(self.mode_combo, params.structural_mode)
        self.struct_scan.set_scan_text(_scaled_scan_text(params.structural_scan, 100.0))
        self.spin_scan.set_scan_text(params.spin_scan_deg)
        self.target_edit.setText(params.target_indices)
        self.rotation_input.set_vector(params.rotation_axis)
        self.strain_input.set_vector(params.strain_axis)
        self.shear_direction_input.set_vector(params.shear_direction)
        set_combo_value(self.bain_axis_combo, params.bain_axis)
        self.limit_frame.set_input_value([params.max_outputs])
        self.advanced_checkbox.setChecked(params.max_outputs != 100)
        self._update_widgets()

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.legacy_notice.clear()
        self.legacy_notice.hide()
        raw_params = dict(data_dict.get("params", {}))
        legacy = bool(raw_params) and (
            "shear_direction" not in raw_params or "bain_axis" not in raw_params
        )
        if "shear_direction" not in raw_params:
            raw_params["shear_direction"] = _legacy_shear_direction(
                raw_params.get("strain_axis", (0.0, 0.0, 1.0))
            )
        raw_params.setdefault("bain_axis", "c")
        self.set_params(MagnetoelasticResponseParams(**raw_params))
        if legacy:
            migration_message = self.tr(
                "Legacy workflow loaded: spin probes now rotate about the saved Cartesian axis. Verify old and new response data before combining them."
            )
            self.legacy_notice.setText("⚠ " + migration_message)
            self.legacy_notice.show()
            MessageManager.send_warning_message(migration_message)
