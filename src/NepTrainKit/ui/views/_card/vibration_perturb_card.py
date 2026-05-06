"""Card for applying vibrational mode-informed atomic perturbations."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import VibrationModePerturbOperation, VibrationModePerturbParams
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard


@CardManager.register_card
class VibrationModePerturbCard(MakeDataCard):
    """Generate perturbations along precomputed vibrational modes."""

    group = "Perturbation"
    card_name = "Vib Mode Perturb"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Vibrational Perturb")
        self.init_ui()

    def init_ui(self):
        """Construct UI controls for vibrational perturbation settings."""

        self.setObjectName("vibration_perturb_card_widget")

        self.distribution_label = BodyLabel("Amplitude distribution:", self.setting_widget)
        self.distribution_combo = ComboBox(self.setting_widget)
        self.distribution_combo.addItems(["Normal", "Uniform"])
        self.distribution_combo.setCurrentIndex(0)
        self.distribution_label.setToolTip("Select random distribution used for mode amplitudes")
        self.distribution_label.installEventFilter(ToolTipFilter(self.distribution_label, 300, ToolTipPosition.TOP))

        self.amplitude_label = BodyLabel("Mode amplitude:", self.setting_widget)
        self.amplitude_frame = SpinBoxUnitInputFrame(self)
        self.amplitude_frame.set_input("Å", 1, "float")
        self.amplitude_frame.setRange(0.0, 1.0)
        self.amplitude_frame.set_input_value([0.05])
        self.amplitude_label.setToolTip("Global scaling factor applied to the combined vibrational displacement")
        self.amplitude_label.installEventFilter(ToolTipFilter(self.amplitude_label, 300, ToolTipPosition.TOP))

        self.modes_label = BodyLabel("Modes per sample:", self.setting_widget)
        self.modes_frame = SpinBoxUnitInputFrame(self)
        self.modes_frame.set_input("mode", 1, "int")
        self.modes_frame.setRange(1, 999)
        self.modes_frame.set_input_value([2])
        self.modes_label.setToolTip("Number of vibrational modes combined for each generated structure")
        self.modes_label.installEventFilter(ToolTipFilter(self.modes_label, 300, ToolTipPosition.TOP))

        self.min_freq_label = BodyLabel("Min frequency:", self.setting_widget)
        self.min_freq_frame = SpinBoxUnitInputFrame(self)
        self.min_freq_frame.set_input("cm^-1", 1, "float")
        self.min_freq_frame.setRange(0.0, 1e5)
        self.min_freq_frame.set_input_value([10.0])
        self.min_freq_label.setToolTip("Discard modes whose |frequency| is below this threshold")
        self.min_freq_label.installEventFilter(ToolTipFilter(self.min_freq_label, 300, ToolTipPosition.TOP))

        self.num_label = BodyLabel("Max num:", self.setting_widget)
        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit", 1, "int")
        self.num_condition_frame.setRange(1, 10000)
        self.num_condition_frame.set_input_value([32])
        self.num_label.setToolTip("Maximum number of perturbed structures to generate")
        self.num_label.installEventFilter(ToolTipFilter(self.num_label, 300, ToolTipPosition.TOP))

        self.optional_label = BodyLabel("Options", self.setting_widget)
        self.optional_label.setToolTip("Optional controls for how vibrational amplitudes are scaled")
        self.optional_label.installEventFilter(ToolTipFilter(self.optional_label, 300, ToolTipPosition.TOP))

        self.optional_frame = QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.optional_frame_layout.setSpacing(2)

        self.scale_checkbox = CheckBox("Scale by 1/sqrt(|freq|)", self.optional_frame)
        self.scale_checkbox.setChecked(True)
        self.scale_checkbox.setToolTip("Divide sampled amplitudes by sqrt(|frequency|) to favour softer modes")

        self.exclude_checkbox = CheckBox("Drop near-zero modes", self.optional_frame)
        self.exclude_checkbox.setChecked(True)
        self.exclude_checkbox.setToolTip("Ignore translational modes below the minimum frequency threshold")

        self.optional_frame_layout.addWidget(self.scale_checkbox, 0, 0, 1, 1)
        self.optional_frame_layout.addWidget(self.exclude_checkbox, 0, 1, 1, 1)

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip("Enable reproducible random sampling")
        self.seed_checkbox.installEventFilter(ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP))
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.settingLayout.addWidget(self.distribution_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.distribution_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.amplitude_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.amplitude_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.modes_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.modes_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.min_freq_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.min_freq_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.num_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.num_condition_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.optional_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 6, 1, 1, 2)

    def create_operation(self):
        return VibrationModePerturbOperation()

    def get_params(self) -> VibrationModePerturbParams:
        return VibrationModePerturbParams(
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
        self.distribution_combo.setCurrentIndex(int(params.distribution))
        self.amplitude_frame.set_input_value([float(params.amplitude)])
        self.modes_frame.set_input_value([int(params.modes_per_sample)])
        self.min_freq_frame.set_input_value([float(params.min_frequency)])
        self.num_condition_frame.set_input_value([int(params.max_num)])
        self.scale_checkbox.setChecked(bool(params.scale_by_frequency))
        self.exclude_checkbox.setChecked(bool(params.exclude_near_zero))
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])

    def process_structure(self, structure):
        """Create perturbed structures aligned with available vibrational modes."""
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values."""
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
