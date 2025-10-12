"""Card for applying vibrational mode-informed atomic perturbations."""

from __future__ import annotations

import numpy as np
from loguru import logger
from PySide6.QtWidgets import QFrame, QGridLayout
from qfluentwidgets import BodyLabel, ComboBox, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.structure import get_vibration_modes
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

    def process_structure(self, structure):
        """Create perturbed structures aligned with available vibrational modes."""

        amplitude = float(self.amplitude_frame.get_input_value()[0])
        if amplitude <= 0.0:
            logger.warning("VibrationModePerturbCard: amplitude must be positive.")
            return []

        modes_per_sample = int(self.modes_frame.get_input_value()[0])
        if modes_per_sample <= 0:
            logger.warning("VibrationModePerturbCard: modes_per_sample must be >= 1.")
            return []

        min_frequency = float(self.min_freq_frame.get_input_value()[0]) if self.exclude_checkbox.isChecked() else 0.0

        frequencies, modes = get_vibration_modes(structure, min_frequency=min_frequency)
        if modes.size == 0:
            logger.warning("VibrationModePerturbCard: no vibrational modes found on structure.")
            return []

        max_samples = int(self.num_condition_frame.get_input_value()[0])
        distribution = self.distribution_combo.currentIndex()
        scale_by_frequency = self.scale_checkbox.isChecked()

        generated = []
        freq_for_scaling = np.abs(frequencies)
        freq_for_scaling[~np.isfinite(freq_for_scaling)] = 0.0

        replace = modes_per_sample > modes.shape[0]
        orig_positions = structure.get_positions()

        for _ in range(max_samples):
            indices = np.random.choice(modes.shape[0], size=modes_per_sample, replace=replace)
            if distribution == 0:
                coeffs = np.random.normal(loc=0.0, scale=1.0, size=modes_per_sample)
            else:
                coeffs = np.random.uniform(-1.0, 1.0, size=modes_per_sample)

            if scale_by_frequency:
                denominators = np.sqrt(np.clip(freq_for_scaling[indices], a_min=1e-12, a_max=None))
                denominators[denominators == 0.0] = 1.0
                coeffs = coeffs / denominators

            displacement = np.sum(coeffs[:, None, None] * modes[indices], axis=0)
            new_positions = orig_positions + amplitude * displacement

            new_structure = structure.copy()
            new_structure.set_positions(new_positions)
            if hasattr(new_structure, "wrap"):
                new_structure.wrap()

            config_suffix = f" VibPerturb(amp={amplitude:.3f}, modes={modes_per_sample})"
            new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + config_suffix
            generated.append(new_structure)

        return generated

    def to_dict(self):
        """Serialize the card configuration."""
        data_dict = super().to_dict()
        data_dict["distribution"] = self.distribution_combo.currentIndex()
        data_dict["amplitude"] = self.amplitude_frame.get_input_value()
        data_dict["modes_per_sample"] = self.modes_frame.get_input_value()
        data_dict["min_frequency"] = self.min_freq_frame.get_input_value()
        data_dict["max_num"] = self.num_condition_frame.get_input_value()
        data_dict["scale_by_frequency"] = self.scale_checkbox.isChecked()
        data_dict["exclude_near_zero"] = self.exclude_checkbox.isChecked()
        return data_dict

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values."""
        super().from_dict(data_dict)

        self.distribution_combo.setCurrentIndex(data_dict.get("distribution", 0))
        if "amplitude" in data_dict:
            self.amplitude_frame.set_input_value(data_dict["amplitude"])
        if "modes_per_sample" in data_dict:
            self.modes_frame.set_input_value(data_dict["modes_per_sample"])
        if "min_frequency" in data_dict:
            self.min_freq_frame.set_input_value(data_dict["min_frequency"])
        if "max_num" in data_dict:
            self.num_condition_frame.set_input_value(data_dict["max_num"])
        self.scale_checkbox.setChecked(data_dict.get("scale_by_frequency", True))
        self.exclude_checkbox.setChecked(data_dict.get("exclude_near_zero", True))
