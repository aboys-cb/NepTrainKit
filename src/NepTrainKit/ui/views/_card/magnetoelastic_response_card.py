"""Thin UI for linked structural and spin-response grids."""

from qfluentwidgets import CaptionLabel, CheckBox, ComboBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.magnetic_response import MagneticResponseScanOperation, MagnetoelasticResponseParams
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


@CardManager.register_card
class MagnetoelasticResponseCard(MakeDataCard):
    """Combine structural coordinates with matched complete spin probes."""

    group = "Magnetism"
    card_name = "Magnetoelastic Response"
    menu_icon = r":/images/src/images/scaling.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Magnetoelastic Response"))
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
            self.tr("Matched spin rotation (degrees)"),
            self.spin_scan,
            self.setting_widget,
            self.tr("The same complete reference/minus/plus spin probes are generated at every structural point."),
        )

        self.target_edit = LineEdit(self.setting_widget)
        self.target_edit.setText("1")
        self.target_edit.setPlaceholderText(self.tr("For example: 1 or 1,3-5"))
        self.target_field = CompactField(
            self.tr("Rotated atoms (1-based)"),
            self.target_edit,
            self.setting_widget,
            self.tr("Choose the atoms used for the local spin probe; ranges such as 3-5 are accepted."),
        )

        path_section = InspectorSection(
            self.tr("Response grid"),
            self.setting_widget,
            self.tr("Choose the lattice path, then define the structural and spin scans with explicit units."),
        )
        path_section.addWidget(CompactField(self.tr("Lattice path"), self.mode_combo, path_section))
        path_section.addWidget(self.struct_scan_field)
        path_section.addWidget(self.spin_scan_field)
        path_section.addWidget(self.target_field)

        self.advanced_checkbox = CheckBox(self.tr("Show directions and output limit"), self.setting_widget)
        self.rotation_input = DirectionInput(self.setting_widget, default=(0.0, 1.0, 0.0))
        self.rotation_field = CompactField(
            self.tr("Spin rotation axis (Cartesian)"), self.rotation_input, self.setting_widget
        )
        self.strain_input = DirectionInput(self.setting_widget, default=(0.0, 0.0, 1.0))
        self.strain_field = CompactField(
            self.tr("Strain axis (Cartesian)"),
            self.strain_input,
            self.setting_widget,
            self.tr("Used by directional strain paths; isotropic volume does not use this direction."),
        )
        self.limit_frame = SpinBoxUnitInputFrame(self)
        self.limit_frame.set_input("", 1, "int")
        self.limit_frame.setRange(3, 999999)
        self.limit_frame.set_input_value([100])
        self.limit_field = CompactField(
            self.tr("Maximum structures"),
            self.limit_frame,
            self.setting_widget,
            self.tr("The limit is applied only between complete response groups."),
        )
        self.advanced_section = InspectorSection(self.tr("Directions and limit"), self.setting_widget)
        advanced_grid = ResponsiveFormGrid(self.advanced_section, two_column_threshold=520)
        advanced_grid.add_field(self.rotation_field)
        advanced_grid.add_field(self.strain_field)
        advanced_grid.add_field(self.limit_field, span=2)
        self.advanced_section.addWidget(advanced_grid)
        self.advanced_section.hide()

        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        output_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        output_section.addWidget(self.output_preview)

        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(10)
        self.settingLayout.addWidget(path_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_checkbox, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(output_section, 3, 0, 1, 3)

        self.mode_combo.currentIndexChanged.connect(self._update_widgets)
        self.advanced_checkbox.toggled.connect(self.advanced_section.setVisible)
        for spin in (*self.struct_scan.range_frame.object_list, *self.spin_scan.range_frame.object_list):
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
        self.struct_scan_field.set_label(labels.get(mode, self.tr("Structural coordinate (%)")))
        self.strain_field.setVisible(mode != "Isotropic volume")
        self._update_output_preview()

    def _update_output_preview(self, *_args):
        try:
            structural_count = self.struct_scan.count()
            spin_count = self.spin_scan.count()
            total = structural_count * spin_count
            self.output_preview.setText(
                self.tr("{structural} lattice points × {spin} spin probes = {total} structures per selected target.").format(
                    structural=structural_count, spin=spin_count, total=total
                )
            )
        except ValueError as exc:
            self.output_preview.setText(str(exc))

    def get_summary_text(self) -> str:
        return self.tr("{mode} · {structural} lattice points × {spin} spin probes").format(
            mode=self.mode_combo.currentText(), structural=self.struct_scan.count(), spin=self.spin_scan.count()
        )

    def get_guidance_text(self) -> str:
        return self.tr(
            "Each lattice point receives the same complete spin scan. The structural percentage is converted to a deformation coordinate; spin angles remain in degrees in the UI."
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
            max_outputs=int(self.limit_frame.get_input_value()[0]),
        )

    def set_params(self, params):
        set_combo_value(self.mode_combo, params.structural_mode)
        self.struct_scan.set_scan_text(_scaled_scan_text(params.structural_scan, 100.0))
        self.spin_scan.set_scan_text(params.spin_scan_deg)
        self.target_edit.setText(params.target_indices)
        self.rotation_input.set_vector(params.rotation_axis)
        self.strain_input.set_vector(params.strain_axis)
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
        self.set_params(MagnetoelasticResponseParams(**data_dict.get("params", {})))
