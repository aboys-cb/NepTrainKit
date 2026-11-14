"""Card for applying vibrational mode-informed atomic perturbations."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox, PlainTextEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import VibrationModePerturbOperation, VibrationModePerturbParams
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SegmentedControl,
    SpinBoxUnitInputFrame,
)
from NepTrainKit import palette


@CardManager.register_card
class VibrationModePerturbCard(MakeDataCard):
    """Generate correlated displacements from modes stored on each input."""

    group = "Perturbation"
    card_name = "Vib Mode Perturb"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Mode Perturb"))
        self._preview_input_count: int | None = None
        self.init_ui()

    def init_ui(self):
        """Build a responsive inspector with the input contract kept visible."""
        self.setObjectName("vibration_perturb_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        input_section = InspectorSection(
            self.tr("Required input"),
            self.setting_widget,
            self.tr(
                "Every input structure must carry recognizable vibrational-mode arrays. "
                "Frequency options require finite values, and weighting also requires non-zero values."
            ),
        )

        self.distribution_combo = SegmentedControl(
            [self.tr("Normal"), self.tr("Uniform")], self.setting_widget
        )
        self.distribution_combo.setCurrentIndex(0)
        distribution_field = CompactField(
            self.tr("Coefficient distribution"),
            self.distribution_combo,
            self.setting_widget,
            self.tr("Normal is unbounded; Uniform samples each coefficient from −1 to 1."),
        )

        self.amplitude_frame = SpinBoxUnitInputFrame(self)
        self.amplitude_frame.set_input(self.tr("× mode"), 1, "float")
        self.amplitude_frame.setDecimals(4)
        self.amplitude_frame.setSingleStep(0.01)
        self.amplitude_frame.setRange(0.0, 1.0)
        self.amplitude_frame.set_input_value([0.05])
        amplitude_field = CompactField(
            self.tr("Mode coefficient scale"),
            self.amplitude_frame,
            self.setting_widget,
            self.tr("A multiplier for the combined mode vectors, not a maximum atomic displacement."),
            inline=True,
            input_max_width=144,
        )
        self.amplitude_frame.setFixedWidth(144)

        self.modes_frame = SpinBoxUnitInputFrame(self)
        self.modes_frame.set_input("", 1, "int")
        self.modes_frame.setRange(1, 999)
        self.modes_frame.set_input_value([2])
        modes_field = CompactField(
            self.tr("Modes combined per sample"),
            self.modes_frame,
            self.setting_widget,
            self.tr("Modes are selected without replacement from those that pass the filter."),
            inline=True,
            input_max_width=132,
        )
        self.modes_frame.setFixedWidth(132)

        perturb_section = InspectorSection(self.tr("Mode sampling"), self.setting_widget)
        perturb_grid = ResponsiveFormGrid(perturb_section)
        perturb_grid.add_field(distribution_field, span=2)
        perturb_grid.add_field(amplitude_field, span=2)
        perturb_grid.add_field(modes_field, span=2)
        perturb_section.addWidget(perturb_grid)

        self.scale_checkbox = CheckBox(self.tr("Use 1/√|frequency|"), self.setting_widget)
        self.scale_checkbox.setChecked(True)
        scale_helper = CaptionLabel(
            self.tr("Reduces coefficients for higher-frequency modes using the stored frequency values."),
            self.setting_widget,
        )
        scale_helper.setWordWrap(True)
        scale_helper.setStyleSheet(f"color:{palette.TEXT_SECONDARY};")

        self.exclude_checkbox = CheckBox(self.tr("Apply cutoff"), self.setting_widget)
        self.exclude_checkbox.setChecked(True)
        self.min_freq_frame = SpinBoxUnitInputFrame(self)
        self.min_freq_frame.set_input("", 1, "float")
        self.min_freq_frame.setDecimals(3)
        self.min_freq_frame.setSingleStep(1.0)
        self.min_freq_frame.setRange(0.0, 1e5)
        self.min_freq_frame.set_input_value([10.0])
        self.min_freq_field = CompactField(
            self.tr("Absolute frequency cutoff"),
            self.min_freq_frame,
            self.setting_widget,
            self.tr("Uses the same numerical unit as the frequencies stored in the input."),
            inline=True,
            input_max_width=132,
        )
        self.min_freq_frame.setFixedWidth(132)

        frequency_section = InspectorSection(self.tr("Frequency handling"), self.setting_widget)
        frequency_section.addWidget(self.scale_checkbox)
        frequency_section.addWidget(scale_helper)
        frequency_section.addWidget(self.exclude_checkbox)
        frequency_section.addWidget(self.min_freq_field)

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("", 1, "int")
        self.num_condition_frame.setRange(1, 10000)
        self.num_condition_frame.set_input_value([32])
        num_field = CompactField(
            self.tr("Structures per input"),
            self.num_condition_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )
        self.num_condition_frame.setFixedWidth(132)

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
        self.seed_frame.setFixedWidth(132)
        seed_layout.addWidget(self.seed_frame)
        seed_layout.addStretch(1)

        output_section = InspectorSection(self.tr("Generation"), self.setting_widget)
        output_grid = ResponsiveFormGrid(output_section)
        output_grid.add_field(num_field, span=2)
        output_grid.add_field(seed_row, span=2)
        output_section.addWidget(output_grid)

        self.settingLayout.addWidget(input_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(perturb_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(frequency_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(output_section, 3, 0, 1, 3)

        self.sampling_combo = ComboBox(self.setting_widget)
        self.sampling_combo.setAccessibleName(self.tr("Mode sampling"))
        add_translated_items(self, self.sampling_combo, ["Random", "Coefficient table"])
        sampling_field = CompactField(self.tr("Mode sampling"), self.sampling_combo, input_section)
        input_section.addWidget(sampling_field)
        self.coefficient_edit = PlainTextEdit(self.setting_widget)
        self.coefficient_edit.setAccessibleName(self.tr("Coefficient table (JSON)"))
        self.coefficient_edit.setPlainText("[[-1], [0], [1]]")
        self.coefficient_edit.setFixedHeight(110)
        self.coefficient_field = CompactField(
            self.tr("Coefficient table (JSON)"), self.coefficient_edit, self.setting_widget,
            self.tr(
                "One row per output, one column per input mode in extraction order. "
                "Uses raw Cartesian displacement modes without frequency weighting, normalization or wrapping."
            ),
        )
        self.settingLayout.addWidget(self.coefficient_field, 4, 0, 1, 3)
        self._random_sections = (perturb_section, frequency_section, output_section)
        self.sampling_combo.currentIndexChanged.connect(self._update_sampling_controls)
        self.coefficient_edit.textChanged.connect(self.refresh_compact_presentation)
        self._update_sampling_controls()

        self.exclude_checkbox.toggled.connect(self._update_frequency_controls)
        self.seed_checkbox.toggled.connect(self.seed_frame.setEnabled)
        self.distribution_combo.currentIndexChanged.connect(
            lambda _index: self.refresh_compact_presentation()
        )
        for frame in (
            self.amplitude_frame,
            self.modes_frame,
            self.min_freq_frame,
            self.num_condition_frame,
            self.seed_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(lambda _value: self.refresh_compact_presentation())
        for checkbox in (self.scale_checkbox, self.exclude_checkbox, self.seed_checkbox):
            checkbox.toggled.connect(lambda _checked: self.refresh_compact_presentation())
        self._update_frequency_controls(self.exclude_checkbox.isChecked())

    def _update_sampling_controls(self, *_args) -> None:
        deterministic = combo_value(self.sampling_combo) == "Coefficient table"
        for section in self._random_sections:
            section.setVisible(not deterministic)
        self.coefficient_field.setVisible(deterministic)
        self.refresh_compact_presentation()

    def _update_frequency_controls(self, enabled: bool) -> None:
        self.min_freq_frame.setEnabled(bool(enabled))
        self.min_freq_field.setToolTip(
            self.tr("Discard modes with |frequency| below this value.")
            if enabled
            else self.tr("Disabled because frequency cutoff is not applied.")
        )

    def get_summary_text(self) -> str:
        params = self.get_params()
        if params.sampling_mode == "Coefficient table":
            try:
                rows = VibrationModePerturbOperation.parse_coefficient_table(params.coefficient_table)
                return self.tr("{count} coefficient rows · {modes} modes").format(count=len(rows), modes=rows.shape[1])
            except ValueError:
                return self.tr("Invalid coefficient table")
        distribution = self.tr("Normal") if params.distribution == 0 else self.tr("Uniform")
        parts = [
            self.tr("{distribution} · scale {scale} · {modes} modes").format(
                distribution=distribution,
                scale=f"{params.amplitude:.4g}",
                modes=params.modes_per_sample,
            ),
            self.tr("{count} per input").format(count=params.max_num),
        ]
        if params.use_seed:
            parts.append(self.tr("seed {seed}").format(seed=params.seed))
        return " · ".join(parts)

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def get_guidance_text(self) -> str:
        params = self.get_params()
        if params.sampling_mode == "Coefficient table":
            return self.tr(
                "Each row gives an exact linear combination of the supplied displacement modes. "
                "A zero row preserves the input positions; no frequency data is required."
            )
        notes = []
        if self._preview_input_count:
            notes.append(
                self.tr("{inputs} × {count} = {total} outputs").format(
                    inputs=self._preview_input_count,
                    count=params.max_num,
                    total=self._preview_input_count * params.max_num,
                )
            )
        notes.append(
            self.tr(
                "The scale multiplies the supplied mode vectors; the resulting maximum atomic displacement depends on their normalization and sampled coefficients."
            )
        )
        if params.scale_by_frequency or params.exclude_near_zero:
            notes.append(
                self.tr(
                    "Frequency values must use one consistent input unit; weighting also requires non-zero values."
                )
            )
        return " ".join(notes)

    def create_operation(self):
        return VibrationModePerturbOperation()

    def get_params(self) -> VibrationModePerturbParams:
        return VibrationModePerturbParams(
            sampling_mode=combo_value(self.sampling_combo),
            coefficient_table=self.coefficient_edit.toPlainText(),
            distribution=self.distribution_combo.currentIndex(),
            amplitude=float(self.amplitude_frame.get_input_value()[0]),
            modes_per_sample=int(self.modes_frame.get_input_value()[0]),
            min_frequency=float(self.min_freq_frame.get_input_value()[0]),
            max_num=int(self.num_condition_frame.get_input_value()[0]),
            scale_by_frequency=self.scale_checkbox.isChecked(),
            exclude_near_zero=self.exclude_checkbox.isChecked(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: VibrationModePerturbParams) -> None:
        set_combo_value(self.sampling_combo, params.sampling_mode)
        self.coefficient_edit.setPlainText(params.coefficient_table)
        self.distribution_combo.setCurrentIndex(int(params.distribution))
        self.amplitude_frame.set_input_value([float(params.amplitude)])
        self.modes_frame.set_input_value([int(params.modes_per_sample)])
        self.min_freq_frame.set_input_value([float(params.min_frequency)])
        self.num_condition_frame.set_input_value([int(params.max_num)])
        self.scale_checkbox.setChecked(bool(params.scale_by_frequency))
        self.exclude_checkbox.setChecked(bool(params.exclude_near_zero))
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self._update_frequency_controls(bool(params.exclude_near_zero))

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
            params = VibrationModePerturbParams(**raw_params)
        else:
            params = VibrationModePerturbParams(
                distribution=data_dict.get("distribution", 0),
                amplitude=data_dict.get("amplitude", [0.05])[0],
                modes_per_sample=data_dict.get("modes_per_sample", [2])[0],
                min_frequency=data_dict.get("min_frequency", [10.0])[0],
                max_num=data_dict.get("max_num", [32])[0],
                scale_by_frequency=data_dict.get("scale_by_frequency", True),
                exclude_near_zero=data_dict.get("exclude_near_zero", True),
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
