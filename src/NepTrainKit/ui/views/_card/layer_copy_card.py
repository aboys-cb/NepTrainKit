"""Card for copying structures into stacked layers with sinusoidal z-perturbation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PySide6.QtWidgets import QGridLayout
from qfluentwidgets import BodyLabel

from NepTrainKit.core import CardManager
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


def apply_sine_displacement(positions: np.ndarray, amplitude: float, wavelength: float) -> np.ndarray:
    """Apply a z displacement sin(x/λ) + sin(y/λ) scaled by amplitude."""
    if amplitude == 0.0:
        return positions
    pos = positions.copy()
    phase = pos[:, 0] / wavelength
    phase_y = pos[:, 1] / wavelength
    offset = amplitude * (np.sin(phase) + np.sin(phase_y))
    pos[:, 2] = pos[:, 2] + offset
    return pos


def build_layers(base_positions: np.ndarray, num_layers: int, layer_distance: float) -> list[np.ndarray]:
    """Stack copies of the positions along z."""
    num_layers = max(1, int(num_layers))
    layers = []
    for i in range(num_layers):
        shifted = base_positions.copy()
        shifted[:, 2] = shifted[:, 2] + i * layer_distance
        layers.append(shifted)
    return layers


@CardManager.register_card
class LayerCopyCard(MakeDataCard):
    """Create multiple layers of the structure with optional sinusoidal z modulation."""

    group = "Structure"
    card_name = "Layer Copy"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Layer Copy with Sinusoidal Warp")
        self._build_ui()

    def _build_ui(self):
        layout: QGridLayout = self.settingLayout

        self.layers_label = BodyLabel("Number of layers", self.setting_widget)
        self.layers_frame = SpinBoxUnitInputFrame(self)
        self.layers_frame.set_input("layers", 1, input_type="int")
        self.layers_frame.setRange(1, 999)

        self.distance_label = BodyLabel("Layer spacing (Å)", self.setting_widget)
        self.distance_frame = SpinBoxUnitInputFrame(self)
        self.distance_frame.set_input("Å", 1, input_type="float")
        self.distance_frame.setRange(-1e4, 1e4)

        self.amp_label = BodyLabel("Amplitude (Å)", self.setting_widget)
        self.amp_frame = SpinBoxUnitInputFrame(self)
        self.amp_frame.set_input("Å", 1, input_type="float")
        self.amp_frame.setRange(-1e4, 1e4)

        self.wave_label = BodyLabel("Wavelength (Å)", self.setting_widget)
        self.wave_frame = SpinBoxUnitInputFrame(self)
        self.wave_frame.set_input("Å", 1, input_type="float")
        self.wave_frame.setRange(1e-6, 1e6)

        layout.addWidget(self.layers_label, 0, 0, 1, 1)
        layout.addWidget(self.layers_frame, 0, 1, 1, 2)
        layout.addWidget(self.distance_label, 1, 0, 1, 1)
        layout.addWidget(self.distance_frame, 1, 1, 1, 2)
        layout.addWidget(self.amp_label, 2, 0, 1, 1)
        layout.addWidget(self.amp_frame, 2, 1, 1, 2)
        layout.addWidget(self.wave_label, 3, 0, 1, 1)
        layout.addWidget(self.wave_frame, 3, 1, 1, 2)

        # Defaults mirroring the provided script
        self.layers_frame.set_input_value([3])
        self.distance_frame.set_input_value([3.0])
        self.amp_frame.set_input_value([1.0])
        self.wave_frame.set_input_value([math.pi])

    def process_structure(self, structure):
        num_layers = int(self.layers_frame.get_input_value()[0])
        layer_distance = float(self.distance_frame.get_input_value()[0])
        amplitude = float(self.amp_frame.get_input_value()[0])
        wavelength = float(self.wave_frame.get_input_value()[0])

        base = structure.copy()
        positions = base.get_positions()
        warped_positions = apply_sine_displacement(positions, amplitude=amplitude, wavelength=wavelength)

        structures = []
        for idx, layer_pos in enumerate(build_layers(warped_positions, num_layers=num_layers, layer_distance=layer_distance)):
            layer_struct = base.copy()
            layer_struct.set_positions(layer_pos)
            tag = layer_struct.info.get("Config_type", "")
            layer_struct.info["Config_type"] = f"{tag} LayerCopy(idx={idx},layers={num_layers})".strip()
            structures.append(layer_struct)
        return structures

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "layers": self.layers_frame.get_input_value(),
                "distance": self.distance_frame.get_input_value(),
                "amplitude": self.amp_frame.get_input_value(),
                "wavelength": self.wave_frame.get_input_value(),
            }
        )
        return data

    def from_dict(self, data_dict: dict[str, Any]) -> None:
        super().from_dict(data_dict)
        self.layers_frame.set_input_value(data_dict.get("layers", [3]))
        self.distance_frame.set_input_value(data_dict.get("distance", [3.0]))
        self.amp_frame.set_input_value(data_dict.get("amplitude", [1.0]))
        self.wave_frame.set_input_value(data_dict.get("wavelength", [math.pi]))
