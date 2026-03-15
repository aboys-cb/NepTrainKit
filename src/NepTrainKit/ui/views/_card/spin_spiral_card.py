"""Card for generating helical and conical spin-spiral initial states."""

from __future__ import annotations

import numpy as np
from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.core.magnetism import (
    existing_moment_magnitudes,
    mapped_moment_magnitudes,
    normalize_vector,
    parse_element_set,
    parse_magmom_map_any,
    set_initial_magmoms_safe,
    spiral_unit_vectors,
)
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class SpinSpiralCard(MakeDataCard):
    """Assign non-collinear spiral magnetic moments using a 1D phase field."""

    group = "Magnetism"
    card_name = "Spin Spiral"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Spin Spiral Generator")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("spin_spiral_card_widget")

        self.axis_label = BodyLabel("Propagation axis", self.setting_widget)
        self.axis_label.setToolTip("Axis used to project atomic positions before building the spiral phase")
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setRange(-1.0, 1.0)
        for obj in self.axis_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.parameter_mode_label = BodyLabel("Spiral parameter", self.setting_widget)
        self.parameter_mode_label.setToolTip("Choose whether the spiral is defined by period L_D or by phase gradient")
        self.parameter_mode_label.installEventFilter(ToolTipFilter(self.parameter_mode_label, 300, ToolTipPosition.TOP))
        self.parameter_mode_combo = ComboBox(self.setting_widget)
        self.parameter_mode_combo.addItems(["Period (L_D)", "Angle gradient (deg/A)"])
        self.parameter_mode_combo.setCurrentText("Period (L_D)")

        self.period_label = BodyLabel("Period range", self.setting_widget)
        self.period_label.setToolTip("Spiral period L_D in angstrom: [min, max, step]")
        self.period_label.installEventFilter(ToolTipFilter(self.period_label, 300, ToolTipPosition.TOP))
        self.period_frame = SpinBoxUnitInputFrame(self)
        self.period_frame.set_input(["-", "step", "A"], 3, "float")
        self.period_frame.setRange(0.001, 1_000_000.0)
        for obj in self.period_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.period_frame.set_input_value([20.0, 40.0, 10.0])

        self.angle_gradient_label = BodyLabel("Angle gradient range", self.setting_widget)
        self.angle_gradient_label.setToolTip("Phase change per angstrom: [min, max, step]. Equivalent to 360/L_D")
        self.angle_gradient_label.installEventFilter(ToolTipFilter(self.angle_gradient_label, 300, ToolTipPosition.TOP))
        self.angle_gradient_frame = SpinBoxUnitInputFrame(self)
        self.angle_gradient_frame.set_input(["-", "step", "deg/A"], 3, "float")
        self.angle_gradient_frame.setRange(0.001, 1_000_000.0)
        for obj in self.angle_gradient_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.angle_gradient_frame.set_input_value([18.0, 18.0, 1.0])

        self.phase_label = BodyLabel("Phase range", self.setting_widget)
        self.phase_label.setToolTip("Phase offset phi0 in degrees: [min, max, step]")
        self.phase_label.installEventFilter(ToolTipFilter(self.phase_label, 300, ToolTipPosition.TOP))
        self.phase_frame = SpinBoxUnitInputFrame(self)
        self.phase_frame.set_input(["-", "step", "deg"], 3, "float")
        self.phase_frame.setRange(-360.0, 360.0)
        for obj in self.phase_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.phase_frame.set_input_value([0.0, 0.0, 15.0])

        self.mz_label = BodyLabel("Constant m_parallel", self.setting_widget)
        self.mz_label.setToolTip("Projection along the propagation axis. m_parallel=0 gives a helix")
        self.mz_label.installEventFilter(ToolTipFilter(self.mz_label, 300, ToolTipPosition.TOP))
        self.mz_frame = SpinBoxUnitInputFrame(self)
        self.mz_frame.set_input("", 1, "float")
        self.mz_frame.setRange(-1.0, 1.0)
        self.mz_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.mz_frame.set_input_value([0.0])

        self.chirality_label = BodyLabel("Chirality", self.setting_widget)
        self.chirality_label.setToolTip("CW uses -2pi*u/L_D; CCW uses +2pi*u/L_D when looking along +axis")
        self.chirality_label.installEventFilter(ToolTipFilter(self.chirality_label, 300, ToolTipPosition.TOP))
        self.chirality_combo = ComboBox(self.setting_widget)
        self.chirality_combo.addItems(["Both", "Clockwise", "Counterclockwise"])
        self.chirality_combo.setCurrentText("Both")

        self.source_label = BodyLabel("Magnitude source", self.setting_widget)
        self.source_label.setToolTip("Use existing initial magmoms or build magnitudes from the map/default below")
        self.source_label.installEventFilter(ToolTipFilter(self.source_label, 300, ToolTipPosition.TOP))
        self.source_combo = ComboBox(self.setting_widget)
        self.source_combo.addItems(["Existing initial magmoms", "Map/default magnitude"])
        self.source_combo.setCurrentText("Existing initial magmoms")

        self.map_label = BodyLabel("Magmom map", self.setting_widget)
        self.map_label.setToolTip('Used when source=Map/default magnitude, for example "Fe:2.2,Ni:0.6"')
        self.map_label.installEventFilter(ToolTipFilter(self.map_label, 300, ToolTipPosition.TOP))
        self.map_edit = LineEdit(self.setting_widget)
        self.map_edit.setPlaceholderText("Fe:2.2,Ni:0.6")

        self.default_label = BodyLabel("Default |m|", self.setting_widget)
        self.default_label.setToolTip("Magnitude used for elements not listed in the magmom map")
        self.default_label.installEventFilter(ToolTipFilter(self.default_label, 300, ToolTipPosition.TOP))
        self.default_frame = SpinBoxUnitInputFrame(self)
        self.default_frame.set_input("", 1, "float")
        self.default_frame.setRange(0.0, 20.0)
        self.default_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.default_frame.set_input_value([0.0])

        self.apply_label = BodyLabel("Apply elements", self.setting_widget)
        self.apply_label.setToolTip("Optional comma-separated element list; empty means all atoms")
        self.apply_label.installEventFilter(ToolTipFilter(self.apply_label, 300, ToolTipPosition.TOP))
        self.apply_edit = LineEdit(self.setting_widget)
        self.apply_edit.setPlaceholderText("Fe,Co,Ni")

        self.max_output_label = BodyLabel("Max outputs", self.setting_widget)
        self.max_output_label.setToolTip("Stop after this many generated structures")
        self.max_output_label.installEventFilter(ToolTipFilter(self.max_output_label, 300, ToolTipPosition.TOP))
        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input("unit", 1, "int")
        self.max_output_frame.setRange(1, 999999)
        self.max_output_frame.set_input_value([100])

        self.settingLayout.addWidget(self.axis_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.parameter_mode_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.parameter_mode_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.period_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.period_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.angle_gradient_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.angle_gradient_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.phase_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.phase_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.mz_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.mz_frame, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.chirality_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.chirality_combo, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.source_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.source_combo, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.map_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.default_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 9, 1, 1, 2)
        self.settingLayout.addWidget(self.apply_label, 10, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 10, 1, 1, 2)
        self.settingLayout.addWidget(self.max_output_label, 11, 0, 1, 1)
        self.settingLayout.addWidget(self.max_output_frame, 11, 1, 1, 2)

        self.source_combo.currentTextChanged.connect(self._update_magnitude_source_widgets)
        self.parameter_mode_combo.currentTextChanged.connect(self._update_parameter_mode_widgets)
        self._update_magnitude_source_widgets()
        self._update_parameter_mode_widgets()

    def _update_magnitude_source_widgets(self):
        use_map = self.source_combo.currentText() == "Map/default magnitude"
        self.map_label.setEnabled(use_map)
        self.map_edit.setEnabled(use_map)
        self.map_label.setVisible(use_map)
        self.map_edit.setVisible(use_map)
        self.default_label.setEnabled(use_map)
        self.default_frame.setEnabled(use_map)
        self.default_label.setVisible(use_map)
        self.default_frame.setVisible(use_map)

    def _update_parameter_mode_widgets(self):
        use_period = self.parameter_mode_combo.currentText() == "Period (L_D)"
        self.period_label.setEnabled(use_period)
        self.period_frame.setEnabled(use_period)
        self.period_label.setVisible(use_period)
        self.period_frame.setVisible(use_period)

        self.angle_gradient_label.setEnabled(not use_period)
        self.angle_gradient_frame.setEnabled(not use_period)
        self.angle_gradient_label.setVisible(not use_period)
        self.angle_gradient_frame.setVisible(not use_period)

    @staticmethod
    def _range_values(values: list[float], *, minimum: float | None = None) -> list[float]:
        start, stop, step = [float(v) for v in values]
        if minimum is not None:
            start = max(start, minimum)
            stop = max(stop, minimum)
        if stop < start:
            start, stop = stop, start
        if abs(stop - start) <= 1e-12 or step <= 0:
            return [start]
        raw = list(np.arange(start, stop + abs(step) * 0.5, abs(step), dtype=float))
        if not raw:
            raw = [start]
        ordered: list[float] = []
        for value in raw:
            rounded = float(np.round(value, 12))
            if not ordered or abs(rounded - ordered[-1]) > 1e-10:
                ordered.append(rounded)
        return ordered

    @staticmethod
    def _axis_tag(axis: np.ndarray) -> str:
        v = np.asarray(axis, dtype=float).reshape(3)
        basis = [
            (np.array([1.0, 0.0, 0.0]), "100"),
            (np.array([0.0, 1.0, 0.0]), "010"),
            (np.array([0.0, 0.0, 1.0]), "001"),
        ]
        for ref, tag in basis:
            if np.allclose(v, ref, atol=1e-8, rtol=0.0):
                return tag
            if np.allclose(v, -ref, atol=1e-8, rtol=0.0):
                return f"-{tag}"
        return f"{v[0]:.3g},{v[1]:.3g},{v[2]:.3g}"

    def _axis(self) -> np.ndarray:
        values = [float(v) for v in self.axis_frame.get_input_value()]
        return normalize_vector(np.array(values, dtype=float))

    def _period_values(self) -> list[float]:
        mode = self.parameter_mode_combo.currentText()
        if mode == "Angle gradient (deg/A)":
            gradients = self._range_values(self.angle_gradient_frame.get_input_value(), minimum=0.001)
            return [float(360.0 / g) for g in gradients if g > 0]
        return self._range_values(self.period_frame.get_input_value(), minimum=0.001)

    def _chirality_values(self) -> list[tuple[str, int]]:
        mode = self.chirality_combo.currentText()
        if mode == "Clockwise":
            return [("cw", -1)]
        if mode == "Counterclockwise":
            return [("ccw", 1)]
        return [("cw", -1), ("ccw", 1)]

    def _magnitudes(self, structure) -> np.ndarray:
        use_existing = self.source_combo.currentText() == "Existing initial magmoms"
        if use_existing:
            mags = existing_moment_magnitudes(structure)
            if mags is not None:
                return mags
            MessageManager.send_warning_message(
                "SpinSpiral: no valid initial_magmoms found; falling back to magmom map/default."
            )

        try:
            moment_map = parse_magmom_map_any(self.map_edit.text())
        except Exception as exc:  # noqa: BLE001
            MessageManager.send_warning_message(f"SpinSpiral: invalid magmom map: {exc}")
            moment_map = {}

        return mapped_moment_magnitudes(
            structure,
            moment_map,
            default_moment=float(self.default_frame.get_input_value()[0]),
            apply_elements=parse_element_set(self.apply_edit.text()),
        )

    def process_structure(self, structure):
        periods = self._period_values()
        phases = self._range_values(self.phase_frame.get_input_value())
        mz = float(np.clip(float(self.mz_frame.get_input_value()[0]), -1.0, 1.0))
        max_outputs = int(self.max_output_frame.get_input_value()[0])

        mags = self._magnitudes(structure)
        if mags.shape[0] != len(structure) or not np.any(mags > 0):
            MessageManager.send_warning_message("SpinSpiral: all magnetic-moment magnitudes are zero; returning input.")
            return [structure.copy()]

        axis = self._axis()
        positions = np.asarray(structure.get_positions(), dtype=float)
        outputs = []
        reached_limit = False

        for period in periods:
            for phase_deg in phases:
                for chirality_tag, chirality_sign in self._chirality_values():
                    atoms = structure.copy()
                    unit_vectors = spiral_unit_vectors(
                        positions,
                        axis=axis,
                        period=float(period),
                        mz=mz,
                        phase_deg=float(phase_deg),
                        chirality=chirality_sign,
                    )
                    magmoms = mags[:, None] * unit_vectors
                    set_initial_magmoms_safe(atoms, magmoms)

                    kind = "Helix" if abs(mz) <= 1e-10 else "Spiral"
                    append_config_tag(
                        atoms,
                        f"{kind}(L={float(period):.6g},ph={float(phase_deg):.6g},mz={mz:.6g},chi={chirality_tag},ax={self._axis_tag(axis)})",
                    )
                    outputs.append(atoms)
                    if len(outputs) >= max_outputs:
                        reached_limit = True
                        break
                if reached_limit:
                    break
            if reached_limit:
                break

        if reached_limit:
            MessageManager.send_warning_message("SpinSpiral: output truncated by Max outputs.")
        return outputs or [structure.copy()]

    def to_dict(self):
        data = super().to_dict()
        data["axis"] = self.axis_frame.get_input_value()
        data["spiral_parameter_mode"] = self.parameter_mode_combo.currentText()
        data["period_range"] = self.period_frame.get_input_value()
        data["angle_gradient_range"] = self.angle_gradient_frame.get_input_value()
        data["phase_range"] = self.phase_frame.get_input_value()
        data["mz"] = self.mz_frame.get_input_value()
        data["chirality"] = self.chirality_combo.currentText()
        data["magnitude_source"] = self.source_combo.currentText()
        data["magmom_map"] = self.map_edit.text()
        data["default_moment"] = self.default_frame.get_input_value()
        data["apply_elements"] = self.apply_edit.text()
        data["max_outputs"] = self.max_output_frame.get_input_value()
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.axis_frame.set_input_value(data_dict.get("axis", [0.0, 0.0, 1.0]))
        self.parameter_mode_combo.setCurrentText(data_dict.get("spiral_parameter_mode", "Period (L_D)"))
        self.period_frame.set_input_value(data_dict.get("period_range", [20.0, 40.0, 10.0]))
        self.angle_gradient_frame.set_input_value(data_dict.get("angle_gradient_range", [18.0, 18.0, 1.0]))
        self.phase_frame.set_input_value(data_dict.get("phase_range", [0.0, 0.0, 15.0]))
        self.mz_frame.set_input_value(data_dict.get("mz", [0.0]))
        self.chirality_combo.setCurrentText(data_dict.get("chirality", "Both"))
        self.source_combo.setCurrentText(data_dict.get("magnitude_source", "Existing initial magmoms"))
        self.map_edit.setText(data_dict.get("magmom_map", ""))
        self.default_frame.set_input_value(data_dict.get("default_moment", [0.0]))
        self.apply_edit.setText(data_dict.get("apply_elements", ""))
        self.max_output_frame.set_input_value(data_dict.get("max_outputs", [100]))
        self._update_magnitude_source_widgets()
        self._update_parameter_mode_widgets()
