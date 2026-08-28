"""Card for scanning lattice-angle increments."""

from qfluentwidgets import CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import ShearAngleOperation, ShearAngleParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    RangeTripletInputFrame,
    ResponsiveFormGrid,
)


@CardManager.register_card
class ShearAngleCard(MakeDataCard):
    """Scan alpha, beta, and gamma increments at fixed lattice lengths."""

    group = "Lattice"
    card_name = "Shear Angle Strain"
    menu_icon = r":/images/src/images/scaling.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Shear Angle Strain"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("shear_angle_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.alpha_frame = self._range_frame()
        self.beta_frame = self._range_frame()
        self.gamma_frame = self._range_frame()
        self.alpha_field = CompactField(
            self.tr("Alpha increment · α = ∠(b, c)"),
            self.alpha_frame,
            self.setting_widget,
        )
        self.beta_field = CompactField(
            self.tr("Beta increment · β = ∠(a, c)"),
            self.beta_frame,
            self.setting_widget,
        )
        self.gamma_field = CompactField(
            self.tr("Gamma increment · γ = ∠(a, b)"),
            self.gamma_frame,
            self.setting_widget,
        )

        angles_section = InspectorSection(
            self.tr("Angle increments"),
            self.setting_widget,
            self.tr(
                "Values are added to the input angles in degrees; lattice-vector lengths stay fixed."
            ),
        )
        angles_grid = ResponsiveFormGrid(angles_section, two_column_threshold=520)
        angles_grid.add_field(self.alpha_field)
        angles_grid.add_field(self.beta_field)
        angles_grid.add_field(self.gamma_field)
        angles_section.addWidget(angles_grid)

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

        self.settingLayout.addWidget(angles_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(molecule_section, 1, 0, 1, 3)

        for frame in (self.alpha_frame, self.beta_frame, self.gamma_frame):
            for control in frame.object_list:
                control.valueChanged.connect(self.refresh_compact_presentation)
        self.organic_checkbox.toggled.connect(self.refresh_compact_presentation)

    def _range_frame(self) -> RangeTripletInputFrame:
        frame = RangeTripletInputFrame(self, suffix="°")
        frame.object_list[0].setRange(-30.0, 30.0)
        frame.object_list[1].setRange(-30.0, 30.0)
        frame.object_list[2].setRange(0.001, 60.0)
        frame.set_input_value([-2.0, 2.0, 1.0])
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

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def create_operation(self):
        return ShearAngleOperation()

    def get_summary_text(self) -> str:
        try:
            summary = self.create_operation().sampling_summary(self.get_params())
        except ValueError:
            return self.tr("Complete the three angle ranges")
        return self.tr("angle increments · {count}/input").format(
            count=summary["outputs_per_input"]
        )

    def get_guidance_text(self) -> str:
        try:
            summary = self.create_operation().sampling_summary(self.get_params())
        except ValueError as exc:
            return translate_runtime_message(exc)
        per_input = int(summary["outputs_per_input"])
        grid = self.tr("{alpha} × {beta} × {gamma} = {count} combinations/input.").format(
            alpha=summary["alpha_points"],
            beta=summary["beta_points"],
            gamma=summary["gamma_points"],
            count=per_input,
        )
        input_count = getattr(self, "_preview_input_count", None)
        if input_count is None:
            input_count = self._dataset_count(getattr(self, "dataset", None)) or None
        elif input_count == 0:
            input_count = None
        if input_count is not None:
            grid += self.tr(" {inputs} inputs → {total} outputs.").format(
                inputs=input_count, total=input_count * per_input
            )
        return grid + " " + self.tr(
            "Fractional coordinates follow the cell; Cartesian spin and ASE initial magnetic moments remain in the input global frame."
        )

    def get_params(self) -> ShearAngleParams:
        return ShearAngleParams(
            alpha_range=tuple(float(value) for value in self.alpha_frame.get_input_value()),
            beta_range=tuple(float(value) for value in self.beta_frame.get_input_value()),
            gamma_range=tuple(float(value) for value in self.gamma_frame.get_input_value()),
            identify_organic=self.organic_checkbox.isChecked(),
        )

    def set_params(self, params: ShearAngleParams) -> None:
        self.organic_checkbox.setChecked(bool(params.identify_organic))
        self.alpha_frame.set_input_value(list(params.alpha_range))
        self.beta_frame.set_input_value(list(params.beta_range))
        self.gamma_frame.set_input_value(list(params.gamma_range))

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
            params = ShearAngleParams(
                alpha_range=tuple(raw_params.get("alpha_range", [-2.0, 2.0, 1.0])),
                beta_range=tuple(raw_params.get("beta_range", [-2.0, 2.0, 1.0])),
                gamma_range=tuple(raw_params.get("gamma_range", [-2.0, 2.0, 1.0])),
                identify_organic=raw_params.get("identify_organic", False),
            )
        else:
            params = ShearAngleParams(
                alpha_range=tuple(data_dict.get("alpha_range", [-2, 2, 1])),
                beta_range=tuple(data_dict.get("beta_range", [-2, 2, 1])),
                gamma_range=tuple(data_dict.get("gamma_range", [-2, 2, 1])),
                identify_organic=data_dict.get("organic", False),
            )
        self.set_params(params)
