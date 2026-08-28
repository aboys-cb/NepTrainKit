"""Card for scanning Cartesian shear-matrix components."""

from qfluentwidgets import CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import ShearMatrixOperation, ShearMatrixParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    RangeTripletInputFrame,
    ResponsiveFormGrid,
)


@CardManager.register_card
class ShearMatrixCard(MakeDataCard):
    """Scan fixed-Cartesian simple shear or symmetric strain components."""

    group = "Lattice"
    card_name = "Shear Matrix Strain"
    menu_icon = r":/images/src/images/scaling.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Shear Matrix Strain"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("shear_strain_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItem(
            self.tr("Symmetric strain tensor εij"), userData="symmetric"
        )
        self.mode_combo.addItem(self.tr("Simple shear γij"), userData="simple")
        self.mode_field = CompactField(
            self.tr("Deformation mode"), self.mode_combo, self.setting_widget
        )
        definition_section = InspectorSection(
            self.tr("Cartesian matrix definition"),
            self.setting_widget,
            self.tr("The matrix acts in fixed Cartesian x/y/z coordinates: C′ = CS."),
        )
        definition_section.addWidget(self.mode_field)

        self.xy_frame = self._range_frame()
        self.yz_frame = self._range_frame()
        self.xz_frame = self._range_frame()
        self.xy_field = CompactField("", self.xy_frame, self.setting_widget)
        self.yz_field = CompactField("", self.yz_frame, self.setting_widget)
        self.xz_field = CompactField("", self.xz_frame, self.setting_widget)

        self.ranges_section = InspectorSection(
            self.tr("Component ranges"), self.setting_widget
        )
        ranges_grid = ResponsiveFormGrid(
            self.ranges_section, two_column_threshold=520
        )
        ranges_grid.add_field(self.xy_field)
        ranges_grid.add_field(self.yz_field)
        ranges_grid.add_field(self.xz_field)
        self.ranges_section.addWidget(ranges_grid)

        self.organic_checkbox = CheckBox(
            self.tr("Keep detected molecules rigid"), self.setting_widget
        )
        self.organic_checkbox.setChecked(False)
        molecule_section = InspectorSection(
            self.tr("Molecular handling"),
            self.setting_widget,
            self.tr(
                "After affine cell deformation, restore the internal geometry of detected molecular clusters."
            ),
        )
        molecule_section.addWidget(self.organic_checkbox)

        self.settingLayout.addWidget(definition_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(self.ranges_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(molecule_section, 2, 0, 1, 3)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        for frame in (self.xy_frame, self.yz_frame, self.xz_frame):
            for control in frame.object_list:
                control.valueChanged.connect(self.refresh_compact_presentation)
        self.organic_checkbox.toggled.connect(self.refresh_compact_presentation)
        self._on_mode_changed()

    def _range_frame(self) -> RangeTripletInputFrame:
        frame = RangeTripletInputFrame(self, suffix="%")
        frame.object_list[0].setRange(-100.0, 100.0)
        frame.object_list[1].setRange(-100.0, 100.0)
        frame.object_list[2].setRange(0.001, 200.0)
        frame.set_input_value([-5.0, 5.0, 1.0])
        return frame

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

    def _on_mode_changed(self, *_args) -> None:
        if combo_value(self.mode_combo, "symmetric") == "symmetric":
            self.xy_field.set_label(self.tr("εxy tensor component"))
            self.yz_field.set_label(self.tr("εyz tensor component"))
            self.xz_field.set_label(self.tr("εxz tensor component"))
            self.ranges_section.description_label.setText(
                self.tr(
                    "Values are εij in percent; engineering shear is γij = 2εij."
                )
            )
        else:
            self.xy_field.set_label(self.tr("γxy · y ← y + γxy x"))
            self.yz_field.set_label(self.tr("γyz · z ← z + γyz y"))
            self.xz_field.set_label(self.tr("γxz · z ← z + γxz x"))
            self.ranges_section.description_label.setText(
                self.tr("Values are engineering simple shear γij in percent.")
            )
        self.ranges_section.description_label.show()
        self.refresh_compact_presentation()

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def create_operation(self):
        return ShearMatrixOperation()

    def get_summary_text(self) -> str:
        try:
            summary = self.create_operation().sampling_summary(self.get_params())
        except ValueError:
            return self.tr("Complete the three component ranges")
        mode = (
            self.tr("symmetric ε components")
            if self.get_params().symmetric
            else self.tr("simple shear γ components")
        )
        return self.tr("{mode} · {count}/input").format(
            mode=mode, count=summary["outputs_per_input"]
        )

    def get_guidance_text(self) -> str:
        try:
            summary = self.create_operation().sampling_summary(self.get_params())
        except ValueError as exc:
            return translate_runtime_message(exc)
        per_input = int(summary["outputs_per_input"])
        guidance = self.tr(
            "{xy} × {yz} × {xz} = {count} combinations/input."
        ).format(
            xy=summary["xy_points"],
            yz=summary["yz_points"],
            xz=summary["xz_points"],
            count=per_input,
        )
        input_count = getattr(self, "_preview_input_count", None)
        if input_count is None:
            input_count = self._dataset_count(getattr(self, "dataset", None)) or None
        elif input_count == 0:
            input_count = None
        if input_count is not None:
            guidance += self.tr(" {inputs} inputs → {total} outputs.").format(
                inputs=input_count, total=input_count * per_input
            )
        return guidance + " " + self.tr(
            "Fractional coordinates follow the cell; Cartesian spin and ASE initial magnetic moments remain in the input global frame."
        )

    def get_params(self) -> ShearMatrixParams:
        return ShearMatrixParams(
            xy_range=tuple(float(value) for value in self.xy_frame.get_input_value()),
            yz_range=tuple(float(value) for value in self.yz_frame.get_input_value()),
            xz_range=tuple(float(value) for value in self.xz_frame.get_input_value()),
            symmetric=combo_value(self.mode_combo, "symmetric") == "symmetric",
            identify_organic=self.organic_checkbox.isChecked(),
        )

    def set_params(self, params: ShearMatrixParams) -> None:
        set_combo_value(self.mode_combo, "symmetric" if params.symmetric else "simple")
        self.organic_checkbox.setChecked(bool(params.identify_organic))
        self.xy_frame.set_input_value(list(params.xy_range))
        self.yz_frame.set_input_value(list(params.yz_range))
        self.xz_frame.set_input_value(list(params.xz_range))
        self._on_mode_changed()

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
            params = ShearMatrixParams(
                xy_range=tuple(raw_params.get("xy_range", [-5.0, 5.0, 1.0])),
                yz_range=tuple(raw_params.get("yz_range", [-5.0, 5.0, 1.0])),
                xz_range=tuple(raw_params.get("xz_range", [-5.0, 5.0, 1.0])),
                symmetric=raw_params.get("symmetric", True),
                identify_organic=raw_params.get("identify_organic", False),
            )
        else:
            params = ShearMatrixParams(
                xy_range=tuple(data_dict.get("xy_range", [-5, 5, 1])),
                yz_range=tuple(data_dict.get("yz_range", [-5, 5, 1])),
                xz_range=tuple(data_dict.get("xz_range", [-5, 5, 1])),
                symmetric=data_dict.get("symmetric", True),
                identify_organic=data_dict.get("organic", False),
            )
        self.set_params(params)
