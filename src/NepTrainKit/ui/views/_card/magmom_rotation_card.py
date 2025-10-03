"""Card for rotating magnetic moments of selected atoms."""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
from qfluentwidgets import BodyLabel, CheckBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


def _parse_elements(text: str) -> set[str]:
    """Split an element string on commas and whitespace."""
    tokens = [token.strip() for token in re.split(r"[\s,]+", text) if token.strip()]
    return set(tokens)


@CardManager.register_card
class MagneticMomentRotationCard(MakeDataCard):
    """Rotate and optionally rescale atomic magnetic moments for selected species."""

    group = "Perturbation"
    card_name = "Magmom Rotation"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Rotate Magnetic Moments")
        self.init_ui()

    def init_ui(self):
        """Build card controls for magnetic moment perturbations."""
        self.setObjectName("magmom_rotation_card_widget")

        self.elements_label = BodyLabel("Elements", self.setting_widget)
        self.elements_label.setToolTip("Comma separated element symbols; empty means all atoms")
        self.elements_label.installEventFilter(ToolTipFilter(self.elements_label, 300, ToolTipPosition.TOP))
        self.elements_input = LineEdit(self.setting_widget)
        self.elements_input.setPlaceholderText("Fe,Ni")


        self.angle_label = BodyLabel("Max rotation", self.setting_widget)
        self.angle_label.setToolTip("Upper bound for random rotation angles in degrees")
        self.angle_label.installEventFilter(ToolTipFilter(self.angle_label, 300, ToolTipPosition.TOP))
        self.angle_frame = SpinBoxUnitInputFrame(self)
        self.angle_frame.set_input("deg", 1, "float")
        self.angle_frame.setRange(-180, 180)
        self.angle_frame.object_list[0].setDecimals(2)  # pyright:ignore
        self.angle_frame.set_input_value([10.0])

        self.count_label = BodyLabel("Structures", self.setting_widget)
        self.count_label.setToolTip("Number of perturbed structures to generate")
        self.count_label.installEventFilter(ToolTipFilter(self.count_label, 300, ToolTipPosition.TOP))
        self.count_frame = SpinBoxUnitInputFrame(self)
        self.count_frame.set_input("unit", 1, "int")
        self.count_frame.setRange(1, 100)
        self.count_frame.set_input_value([5])

        self.magnitude_checkbox = CheckBox("Randomise magnitude", self.setting_widget)
        self.magnitude_checkbox.setChecked(True)
        self.magnitude_checkbox.setToolTip("Enable scaling of magnetic-moment magnitudes")
        self.magnitude_checkbox.installEventFilter(ToolTipFilter(self.magnitude_checkbox, 300, ToolTipPosition.TOP))
        self.magnitude_checkbox.stateChanged.connect(self._toggle_magnitude_inputs)

        self.min_factor_label = BodyLabel("magnitude scaling factor", self.setting_widget)
        self.min_factor_label.setToolTip("magnitude scaling factor")
        self.min_factor_label.installEventFilter(ToolTipFilter(self.min_factor_label, 300, ToolTipPosition.TOP))
        self.magnitude_factor_frame = SpinBoxUnitInputFrame(self)
        self.magnitude_factor_frame.set_input(["-", ""], 2, "float")
        self.magnitude_factor_frame.setRange(0, 10)
        self.magnitude_factor_frame.set_input_value([0.95,1.05])
        self.magnitude_factor_frame.object_list[0].setDecimals(3)  # pyright:ignore




        self.settingLayout.addWidget(self.elements_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.elements_input, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.angle_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.angle_frame, 1, 1, 1, 2)

        self.settingLayout.addWidget(self.magnitude_checkbox, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.min_factor_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.magnitude_factor_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.count_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.count_frame, 4, 1, 1, 2)

        self._toggle_magnitude_inputs(self.magnitude_checkbox.checkState())

    def _toggle_magnitude_inputs(self, state):
        enabled = bool(state)
        self.min_factor_label.setEnabled(enabled)
        self.magnitude_factor_frame.setEnabled(enabled)


    @staticmethod
    def _rotate_vector(vector: np.ndarray, angle_deg: float) -> np.ndarray:
        vec = np.asarray(vector, dtype=float)
        if not np.any(vec) or angle_deg <= 0:
            return vec.copy()

        axis = np.random.normal(size=3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / axis_norm

        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        ux, uy, uz = axis
        rotation_matrix = np.array([
            [cos_t + ux * ux * (1 - cos_t), ux * uy * (1 - cos_t) - uz * sin_t, ux * uz * (1 - cos_t) + uy * sin_t],
            [uy * ux * (1 - cos_t) + uz * sin_t, cos_t + uy * uy * (1 - cos_t), uy * uz * (1 - cos_t) - ux * sin_t],
            [uz * ux * (1 - cos_t) - uy * sin_t, uz * uy * (1 - cos_t) + ux * sin_t, cos_t + uz * uz * (1 - cos_t)],
        ])
        return rotation_matrix @ vec

    @staticmethod
    def _rescale_vector(vector: np.ndarray, target_length: float) -> np.ndarray:
        vec = np.asarray(vector, dtype=float)
        current = np.linalg.norm(vec)
        if current == 0 or target_length == 0:
            return np.zeros_like(vec)
        return vec / current * target_length

    def process_structure(self, structure):
        elements = _parse_elements(self.elements_input.text())
        max_angle = float(self.angle_frame.get_input_value()[0])
        num_structures = int(self.count_frame.get_input_value()[0])
        disturb_magnitude = self.magnitude_checkbox.isChecked()
        min_factor = float(self.magnitude_factor_frame.get_input_value()[0])
        max_factor = float(self.magnitude_factor_frame.get_input_value()[1])
        if min_factor > max_factor:
            min_factor, max_factor = max_factor, min_factor

        if num_structures <= 0:
            return [structure.copy()]

        base_magmoms = structure.get_initial_magnetic_moments()
        if base_magmoms is None or len(base_magmoms) == 0:
            return [structure.copy()]

        base_magmoms = np.array(base_magmoms, dtype=float)
        is_vector = base_magmoms.ndim == 2
        can_rotate = is_vector and max_angle > 0

        if not elements:
            symbols: Iterable[str] = structure.get_chemical_symbols()
            elements = set(symbols)

        results = []
        symbols = structure.get_chemical_symbols()

        for _ in range(num_structures):
            new_structure = structure.copy()
            moment_array = np.array(base_magmoms, copy=True)

            for idx, symbol in enumerate(symbols):
                if symbol not in elements:
                    continue

                if can_rotate:
                    angle = np.random.uniform(0.0, max_angle)
                    rotated = self._rotate_vector(base_magmoms[idx], angle)
                    if disturb_magnitude:
                        original_length = np.linalg.norm(base_magmoms[idx])
                        scale = np.random.uniform(min_factor, max_factor)
                        target_length = original_length * scale
                        rotated = self._rescale_vector(rotated, target_length)
                    moment_array[idx] = rotated
                elif disturb_magnitude:
                    scale = np.random.uniform(min_factor, max_factor)
                    moment_array[idx] = base_magmoms[idx] * scale

            new_structure.set_initial_magnetic_moments(moment_array)

            label = "MagmomRotate" if can_rotate else "MagmomScale"
            details = []
            if can_rotate:
                details.append(f"max={max_angle:.1f}deg")
            if disturb_magnitude:
                details.append(f"scale={min_factor:.2f}-{max_factor:.2f}")
            suffix = f" {label}"
            if details:
                suffix += "(" + ", ".join(details) + ")"
            new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + suffix
            results.append(new_structure)

        return results

    def to_dict(self):
        data_dict = super().to_dict()
        data_dict["elements"] = self.elements_input.text()
        data_dict["max_angle"] = self.angle_frame.get_input_value()
        data_dict["num_structures"] = self.count_frame.get_input_value()
        data_dict["disturb_magnitude"] = self.magnitude_checkbox.isChecked()
        data_dict["magnitude_factor"] = self.magnitude_factor_frame.get_input_value()

        return data_dict

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.elements_input.setText(data_dict.get("elements", ""))
        if "max_angle" in data_dict:
            self.angle_frame.set_input_value(data_dict["max_angle"])
        if "num_structures" in data_dict:
            self.count_frame.set_input_value(data_dict["num_structures"])
        self.magnitude_checkbox.setChecked(data_dict.get("disturb_magnitude", True))
        if "magnitude_factor" in data_dict:
            self.magnitude_factor_frame.set_input_value(data_dict["magnitude_factor"])

        self._toggle_magnitude_inputs(self.magnitude_checkbox.checkState())
