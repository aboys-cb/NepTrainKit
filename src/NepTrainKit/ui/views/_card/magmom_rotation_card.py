"""Card for sampling random local spin perturbations."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CheckBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.magnetism import (
    MagneticMomentRotationOperation,
    MagneticMomentRotationParams,
)
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class MagneticMomentRotationCard(MakeDataCard):
    """Sample random directions around an existing magnetic state."""

    group = "Magnetism"
    card_name = "Spin Perturbation"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Spin Perturbation"))
        self._preview_input_count: int | None = None
        self.init_ui()

    def init_ui(self):
        self.setObjectName("magmom_rotation_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(12)

        input_section = InspectorSection(
            self.tr("Input and targets"),
            self.setting_widget,
            self.tr(
                "Each input must contain spin:R:3 or ASE initial magnetic moments. "
                "Selected non-zero moments are sampled independently around their current directions."
            ),
        )
        self.elements_input = LineEdit(self.setting_widget)
        self.elements_input.setPlaceholderText(self.tr("All elements"))
        elements_field = CompactField(
            self.tr("Target elements"),
            self.elements_input,
            self.setting_widget,
            self.tr("Comma-separated symbols such as Fe,Ni; empty selects every element."),
        )
        input_section.addWidget(elements_field)

        self.angle_frame = SpinBoxUnitInputFrame(self)
        self.angle_frame.set_input("°", 1, "float")
        self.angle_frame.setRange(0.0, 180.0)
        self.angle_frame.setDecimals(2)
        self.angle_frame.setSingleStep(1.0)
        self.angle_frame.set_input_value([10.0])
        angle_field = CompactField(
            self.tr("Maximum perturbation angle"),
            self.angle_frame,
            self.setting_widget,
            self.tr("Sample each direction uniformly inside this angular cap around its input moment."),
        )

        self.lift_scalar_checkbox = CheckBox(
            self.tr("Lift scalar moments to vectors"), self.setting_widget
        )
        self.lift_scalar_checkbox.setChecked(True)
        lift_field = CompactField(
            self.tr("Collinear input"),
            self.lift_scalar_checkbox,
            self.setting_widget,
        )

        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setRange(-1.0, 1.0)
        self.axis_frame.setDecimals(6)
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])
        self.axis_field = CompactField(
            self.tr("Scalar lift direction (Cartesian)"),
            self.axis_frame,
            self.setting_widget,
            self.tr("Used only to turn scalar moments into vectors; it does not control perturbation directions."),
        )

        rotation_section = InspectorSection(self.tr("Direction sampling"), self.setting_widget)
        rotation_grid = ResponsiveFormGrid(rotation_section)
        rotation_grid.add_field(angle_field, span=2)
        rotation_grid.add_field(lift_field, span=2)
        rotation_grid.add_field(self.axis_field, span=2)
        rotation_section.addWidget(rotation_grid)

        self.magnitude_checkbox = CheckBox(
            self.tr("Vary magnetic-moment magnitude"), self.setting_widget
        )
        self.magnitude_checkbox.setChecked(True)
        magnitude_toggle_field = CompactField(
            self.tr("Magnitude sampling"),
            self.magnitude_checkbox,
            self.setting_widget,
        )

        self.magnitude_factor_frame = SpinBoxUnitInputFrame(self)
        self.magnitude_factor_frame.set_input(["min", "max"], 2, "float")
        self.magnitude_factor_frame.setRange(0.0, 10.0)
        self.magnitude_factor_frame.setDecimals(3)
        self.magnitude_factor_frame.setSingleStep(0.01)
        self.magnitude_factor_frame.set_input_value([0.95, 1.05])
        self.magnitude_factor_field = CompactField(
            self.tr("Magnitude scale range"),
            self.magnitude_factor_frame,
            self.setting_widget,
            self.tr("Each selected moment is multiplied by an independent factor in this interval."),
        )

        magnitude_section = InspectorSection(self.tr("Magnitude"), self.setting_widget)
        magnitude_grid = ResponsiveFormGrid(magnitude_section)
        magnitude_grid.add_field(magnitude_toggle_field, span=2)
        magnitude_grid.add_field(self.magnitude_factor_field, span=2)
        magnitude_section.addWidget(magnitude_grid)

        self.count_frame = SpinBoxUnitInputFrame(self)
        self.count_frame.set_input("", 1, "int")
        self.count_frame.setRange(1, 10000)
        self.count_frame.set_input_value([5])
        count_field = CompactField(
            self.tr("Structures per input"), self.count_frame, self.setting_widget
        )

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        seed_row = QWidget(self.setting_widget)
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(6)
        seed_layout.addWidget(self.seed_checkbox)
        seed_layout.addWidget(self.seed_frame, 1)
        seed_field = CompactField(self.tr("Reproducibility"), seed_row, self.setting_widget)

        generation_section = InspectorSection(self.tr("Generation"), self.setting_widget)
        generation_grid = ResponsiveFormGrid(generation_section)
        generation_grid.add_field(count_field)
        generation_grid.add_field(seed_field, span=2)
        generation_section.addWidget(generation_grid)

        sampling_note = BodyLabel(
            self.tr(
                "The atomic coordinates and cell are carried through unchanged; outputs differ in their spin vectors."
            ),
            generation_section,
        )
        sampling_note.setWordWrap(True)
        generation_section.addWidget(sampling_note)

        self.settingLayout.addWidget(input_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(rotation_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(magnitude_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(generation_section, 3, 0, 1, 3)

        self.lift_scalar_checkbox.toggled.connect(self._update_scalar_controls)
        self.magnitude_checkbox.toggled.connect(self._update_magnitude_controls)
        self.seed_checkbox.toggled.connect(self.seed_frame.setEnabled)
        self.elements_input.textChanged.connect(
            lambda _text: self.refresh_compact_presentation()
        )
        for frame in (
            self.angle_frame,
            self.axis_frame,
            self.magnitude_factor_frame,
            self.count_frame,
            self.seed_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(
                    lambda _value: self.refresh_compact_presentation()
                )
        for checkbox in (
            self.lift_scalar_checkbox,
            self.magnitude_checkbox,
            self.seed_checkbox,
        ):
            checkbox.toggled.connect(
                lambda _checked: self.refresh_compact_presentation()
            )
        self._update_scalar_controls(self.lift_scalar_checkbox.isChecked())
        self._update_magnitude_controls(self.magnitude_checkbox.isChecked())

    def _update_scalar_controls(self, enabled: bool) -> None:
        self.axis_field.setVisible(bool(enabled))

    def _update_magnitude_controls(self, enabled: bool) -> None:
        self.magnitude_factor_field.setVisible(bool(enabled))

    def get_summary_text(self) -> str:
        params = self.get_params()
        target = params.elements.strip() or self.tr("all moments")
        summary = self.tr("≤ {angle}° · {target} · {count}/input").format(
            angle=f"{params.max_angle:.4g}", target=target, count=params.num_structures
        )
        if params.disturb_magnitude:
            summary += self.tr(" · {minimum}–{maximum}×").format(
                minimum=f"{float(params.magnitude_factor[0]):.4g}",
                maximum=f"{float(params.magnitude_factor[1]):.4g}",
            )
        return summary

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def get_guidance_text(self) -> str:
        params = self.get_params()
        notes = []
        if self._preview_input_count:
            notes.append(
                self.tr(
                    "Planned (valid input): {inputs} × {count} = {total} outputs"
                ).format(
                    inputs=self._preview_input_count,
                    count=params.num_structures,
                    total=self._preview_input_count * params.num_structures,
                )
            )
        notes.append(
            self.tr(
                "Each selected moment is sampled independently inside its angular cap; the scalar lift direction only initializes collinear input."
            )
        )
        if params.max_angle == 0.0 and not params.disturb_magnitude:
            notes.append(
                self.tr("Increase the perturbation angle or enable magnitude sampling before running.")
            )
        return " ".join(notes)

    def create_operation(self):
        return MagneticMomentRotationOperation()

    def get_params(self) -> MagneticMomentRotationParams:
        return MagneticMomentRotationParams(
            elements=self.elements_input.text(),
            max_angle=float(self.angle_frame.get_input_value()[0]),
            num_structures=int(self.count_frame.get_input_value()[0]),
            lift_scalar=self.lift_scalar_checkbox.isChecked(),
            axis=self.axis_frame.get_input_value(),
            disturb_magnitude=self.magnitude_checkbox.isChecked(),
            magnitude_factor=self.magnitude_factor_frame.get_input_value(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: MagneticMomentRotationParams) -> None:
        self.elements_input.setText(params.elements)
        self.angle_frame.set_input_value([float(params.max_angle)])
        self.count_frame.set_input_value([int(params.num_structures)])
        self.lift_scalar_checkbox.setChecked(bool(params.lift_scalar))
        self.axis_frame.set_input_value([float(value) for value in params.axis])
        self.magnitude_checkbox.setChecked(bool(params.disturb_magnitude))
        self.magnitude_factor_frame.set_input_value(
            [float(value) for value in params.magnitude_factor]
        )
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())
        self._update_scalar_controls(self.lift_scalar_checkbox.isChecked())
        self._update_magnitude_controls(self.magnitude_checkbox.isChecked())

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
            params = MagneticMomentRotationParams(**raw_params)
        else:
            params = MagneticMomentRotationParams(
                elements=data_dict.get("elements", ""),
                max_angle=data_dict.get("max_angle", [10.0])[0],
                num_structures=data_dict.get("num_structures", [5])[0],
                lift_scalar=data_dict.get("lift_scalar", True),
                axis=data_dict.get("axis", [0.0, 0.0, 1.0]),
                disturb_magnitude=data_dict.get("disturb_magnitude", True),
                magnitude_factor=data_dict.get("magnitude_factor", [0.95, 1.05]),
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
