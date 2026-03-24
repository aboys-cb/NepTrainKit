"""Card for generating helical and conical spin-spiral initial states."""

from __future__ import annotations

import numpy as np
from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

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

        self.mz_label = BodyLabel("m_parallel range", self.setting_widget)
        self.mz_label.setToolTip(
            "Normalized, dimensionless axial component m_parallel/|m|: [min, max, step], range [-1, 1]. "
            "m_parallel=0 gives a helix; nonzero values give conical spirals"
        )
        self.mz_label.installEventFilter(ToolTipFilter(self.mz_label, 300, ToolTipPosition.TOP))
        self.mz_frame = SpinBoxUnitInputFrame(self)
        self.mz_frame.set_input(["-", "step", ""], 3, "float")
        self.mz_frame.setRange(-1.0, 1.0)
        for obj in self.mz_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.mz_frame.set_input_value([0.0, 0.0, 0.1])

        self.chirality_label = BodyLabel("Chirality", self.setting_widget)
        self.chirality_label.setToolTip("CW uses -2pi*u/L_D; CCW uses +2pi*u/L_D when looking along +axis")
        self.chirality_label.installEventFilter(ToolTipFilter(self.chirality_label, 300, ToolTipPosition.TOP))
        self.chirality_combo = ComboBox(self.setting_widget)
        self.chirality_combo.addItems(["Both", "Clockwise", "Counterclockwise"])
        self.chirality_combo.setCurrentText("Both")

        self.phase_mode_label = BodyLabel("Phase mode", self.setting_widget)
        self.phase_mode_label.setToolTip(
            "Continuous by position uses each atom's projected coordinate; Layer-locked gives one shared phase per layer"
        )
        self.phase_mode_label.installEventFilter(ToolTipFilter(self.phase_mode_label, 300, ToolTipPosition.TOP))
        self.phase_mode_combo = ComboBox(self.setting_widget)
        self.phase_mode_combo.addItems(["Continuous by position", "Layer-locked"])
        self.phase_mode_combo.setCurrentText("Continuous by position")

        self.layer_tol_label = BodyLabel("Layer tolerance", self.setting_widget)
        self.layer_tol_label.setToolTip(
            "Used only in Layer-locked mode: atoms whose projected coordinates differ by <= tolerance share one phase"
        )
        self.layer_tol_label.installEventFilter(ToolTipFilter(self.layer_tol_label, 300, ToolTipPosition.TOP))
        self.layer_tol_frame = SpinBoxUnitInputFrame(self)
        self.layer_tol_frame.set_input("A", 1, "float")
        self.layer_tol_frame.setRange(0.0001, 10.0)
        self.layer_tol_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.layer_tol_frame.set_input_value([0.05])

        self.commensurate_label = BodyLabel("Period filter", self.setting_widget)
        self.commensurate_label.setToolTip(
            "Keep only periods whose phase advance over each periodic lattice vector is an integer multiple of 360 deg"
        )
        self.commensurate_label.installEventFilter(
            ToolTipFilter(self.commensurate_label, 300, ToolTipPosition.TOP)
        )
        self.commensurate_checkbox = CheckBox("Only lattice-compatible periods", self.setting_widget)
        self.commensurate_checkbox.setChecked(False)
        self.commensurate_checkbox.setToolTip(
            "Use only periods commensurate with the current cell, pbc, and propagation axis"
        )
        self.commensurate_checkbox.installEventFilter(
            ToolTipFilter(self.commensurate_checkbox, 300, ToolTipPosition.TOP)
        )

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
        self.settingLayout.addWidget(self.phase_mode_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.phase_mode_combo, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.layer_tol_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.layer_tol_frame, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.commensurate_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.commensurate_checkbox, 9, 1, 1, 2)
        self.settingLayout.addWidget(self.source_label, 10, 0, 1, 1)
        self.settingLayout.addWidget(self.source_combo, 10, 1, 1, 2)
        self.settingLayout.addWidget(self.map_label, 11, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 11, 1, 1, 2)
        self.settingLayout.addWidget(self.default_label, 12, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 12, 1, 1, 2)
        self.settingLayout.addWidget(self.apply_label, 13, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 13, 1, 1, 2)
        self.settingLayout.addWidget(self.max_output_label, 14, 0, 1, 1)
        self.settingLayout.addWidget(self.max_output_frame, 14, 1, 1, 2)

        self.source_combo.currentTextChanged.connect(self._update_magnitude_source_widgets)
        self.parameter_mode_combo.currentTextChanged.connect(self._update_parameter_mode_widgets)
        self.phase_mode_combo.currentTextChanged.connect(self._update_phase_mode_widgets)
        self._update_magnitude_source_widgets()
        self._update_parameter_mode_widgets()
        self._update_phase_mode_widgets()

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

    def _update_phase_mode_widgets(self):
        layer_locked = self.phase_mode_combo.currentText() == "Layer-locked"
        self.layer_tol_label.setEnabled(layer_locked)
        self.layer_tol_frame.setEnabled(layer_locked)
        self.layer_tol_label.setVisible(layer_locked)
        self.layer_tol_frame.setVisible(layer_locked)

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

    def _period_bounds(self) -> tuple[float, float]:
        mode = self.parameter_mode_combo.currentText()
        if mode == "Angle gradient (deg/A)":
            start, stop, _step = [float(v) for v in self.angle_gradient_frame.get_input_value()]
            start = max(start, 0.001)
            stop = max(stop, 0.001)
            if stop < start:
                start, stop = stop, start
            period_min = float(360.0 / stop)
            period_max = float(360.0 / start)
            return (min(period_min, period_max), max(period_min, period_max))

        start, stop, _step = [float(v) for v in self.period_frame.get_input_value()]
        start = max(start, 0.001)
        stop = max(stop, 0.001)
        if stop < start:
            start, stop = stop, start
        return (float(start), float(stop))

    @staticmethod
    def _coerce_scan_triplet(values, *, default_step: float) -> list[float]:
        if isinstance(values, (int, float)):
            scalar = float(values)
            return [scalar, scalar, float(default_step)]

        if not isinstance(values, (list, tuple)):
            return [0.0, 0.0, float(default_step)]

        if len(values) >= 3:
            return [float(values[0]), float(values[1]), float(values[2])]
        if len(values) == 2:
            return [float(values[0]), float(values[1]), float(default_step)]
        if len(values) == 1:
            scalar = float(values[0])
            return [scalar, scalar, float(default_step)]
        return [0.0, 0.0, float(default_step)]

    def _mz_values(self) -> list[float]:
        raw = self._coerce_scan_triplet(self.mz_frame.get_input_value(), default_step=0.1)
        values: list[float] = []
        for value in self._range_values(raw, minimum=-1.0):
            clipped = float(np.clip(value, -1.0, 1.0))
            if not values or abs(clipped - values[-1]) > 1e-10:
                values.append(clipped)
        return values or [0.0]

    @staticmethod
    def _layer_ids(positions: np.ndarray, axis: np.ndarray, tolerance: float) -> np.ndarray:
        projections = np.asarray(positions, dtype=float) @ normalize_vector(axis)
        if projections.size == 0:
            return np.zeros(0, dtype=int)

        order = np.argsort(projections, kind="stable")
        layer_ids = np.zeros(len(projections), dtype=int)
        tol = max(float(tolerance), 1e-8)

        current_layer = 0
        center = float(projections[order[0]])
        count = 1
        layer_ids[order[0]] = current_layer

        for idx in order[1:]:
            value = float(projections[idx])
            if abs(value - center) <= tol:
                count += 1
                center += (value - center) / float(count)
                layer_ids[idx] = current_layer
                continue

            current_layer += 1
            center = value
            count = 1
            layer_ids[idx] = current_layer

        return layer_ids

    @staticmethod
    def _layer_locked_positions(positions: np.ndarray, axis: np.ndarray, tolerance: float) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        if pos.size == 0:
            return np.zeros((0, 3), dtype=float)

        axis_hat = normalize_vector(axis)
        projections = pos @ axis_hat
        layer_ids = SpinSpiralCard._layer_ids(pos, axis_hat, tolerance)
        shared_projections = np.array(projections, copy=True)
        for layer_id in np.unique(layer_ids):
            mask = layer_ids == layer_id
            shared_projections[mask] = float(np.mean(projections[mask]))

        return pos + (shared_projections - projections)[:, None] * axis_hat[None, :]

    def _phase_positions(self, positions: np.ndarray, axis: np.ndarray) -> np.ndarray:
        if self.phase_mode_combo.currentText() != "Layer-locked":
            return np.asarray(positions, dtype=float)
        return self._layer_locked_positions(
            positions,
            axis,
            float(self.layer_tol_frame.get_input_value()[0]),
        )

    @staticmethod
    def _filter_commensurate_periods(
        periods: list[float],
        *,
        structure,
        axis: np.ndarray,
        tolerance: float = 1e-6,
    ) -> list[float]:
        cell = np.asarray(structure.get_cell(), dtype=float).reshape(3, 3)
        pbc = np.asarray(structure.get_pbc(), dtype=bool).reshape(3)
        periodic_vectors = [cell[idx] for idx in range(3) if pbc[idx]]
        if not periodic_vectors:
            return periods

        compatible: list[float] = []
        for period in periods:
            period_ok = True
            for vector in periodic_vectors:
                turns = float(np.dot(vector, axis) / float(period))
                if abs(turns) <= tolerance:
                    continue
                if abs(turns - round(turns)) > tolerance:
                    period_ok = False
                    break
            if period_ok:
                compatible.append(float(period))
        return compatible

    @staticmethod
    def _discover_commensurate_periods_in_bounds(
        period_bounds: tuple[float, float],
        *,
        structure,
        axis: np.ndarray,
        tolerance: float = 1e-6,
    ) -> list[float] | None:
        period_min, period_max = [float(v) for v in period_bounds]
        if period_max < period_min:
            period_min, period_max = period_max, period_min

        cell = np.asarray(structure.get_cell(), dtype=float).reshape(3, 3)
        pbc = np.asarray(structure.get_pbc(), dtype=bool).reshape(3)
        projections = [abs(float(np.dot(cell[idx], axis))) for idx in range(3) if pbc[idx]]
        projections = [proj for proj in projections if proj > tolerance]
        if not projections:
            return None

        ref_projection = max(projections)
        n_min = max(1, int(np.ceil(ref_projection / period_max - tolerance)))
        n_max = max(1, int(np.floor(ref_projection / period_min + tolerance)))

        discovered: list[float] = []
        for turns_ref in range(n_min, n_max + 1):
            period = ref_projection / float(turns_ref)
            if period < period_min - tolerance or period > period_max + tolerance:
                continue

            period_ok = True
            for projection in projections:
                turns = projection / period
                if abs(turns - round(turns)) > tolerance:
                    period_ok = False
                    break

            if not period_ok:
                continue

            rounded = float(np.round(period, 12))
            if not discovered or abs(rounded - discovered[-1]) > 1e-10:
                discovered.append(rounded)

        return discovered

    @staticmethod
    def _suggest_supercell_multipliers(
        periods: list[float],
        *,
        structure,
        axis: np.ndarray,
        tolerance: float = 1e-6,
        max_multiplier: int = 24,
    ) -> tuple[float, list[int]] | None:
        cell = np.asarray(structure.get_cell(), dtype=float).reshape(3, 3)
        pbc = np.asarray(structure.get_pbc(), dtype=bool).reshape(3)
        periodic_indices = [idx for idx in range(3) if pbc[idx]]
        if not periodic_indices:
            return None

        best: tuple[int, int, float, list[int]] | None = None
        for period in periods:
            multipliers = [1, 1, 1]
            feasible = True
            for idx in periodic_indices:
                turns = float(np.dot(cell[idx], axis) / float(period))
                if abs(turns) <= tolerance:
                    continue

                found = None
                for multiplier in range(1, max_multiplier + 1):
                    if abs(multiplier * turns - round(multiplier * turns)) <= tolerance:
                        found = multiplier
                        break
                if found is None:
                    feasible = False
                    break
                multipliers[idx] = found

            if not feasible:
                continue

            volume_factor = int(np.prod([multipliers[idx] for idx in periodic_indices], dtype=int))
            max_factor = max(multipliers[idx] for idx in periodic_indices)
            candidate = (volume_factor, max_factor, float(period), multipliers)
            if best is None or candidate < best:
                best = candidate

        if best is None:
            return None
        return best[2], best[3]

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
        original_periods = list(periods)
        period_bounds = self._period_bounds()
        phases = self._range_values(self.phase_frame.get_input_value())
        mz_values = self._mz_values()
        max_outputs = int(self.max_output_frame.get_input_value()[0])
        axis = self._axis()

        if self.commensurate_checkbox.isChecked():
            discovered_periods = self._discover_commensurate_periods_in_bounds(
                period_bounds,
                structure=structure,
                axis=axis,
            )
            filtered_periods = (
                periods
                if discovered_periods is None
                else discovered_periods
            )
            if len(filtered_periods) != len(periods) or discovered_periods is not None:
                MessageManager.send_warning_message(
                    f"SpinSpiral: kept {len(filtered_periods)}/{len(periods)} lattice-compatible periods."
                )
            periods = filtered_periods
            if not periods:
                suggestion = self._suggest_supercell_multipliers(original_periods, structure=structure, axis=axis)
                if suggestion is not None:
                    suggested_period, multipliers = suggestion
                    MessageManager.send_warning_message(
                        "SpinSpiral: no lattice-compatible periods found in the selected range; "
                        f"try supercell multiples {multipliers} for L={suggested_period:.6g}."
                    )
                MessageManager.send_warning_message(
                    "SpinSpiral: no lattice-compatible periods found in the selected range; returning input."
                )
                return [structure.copy()]

        mags = self._magnitudes(structure)
        if mags.shape[0] != len(structure) or not np.any(mags > 0):
            MessageManager.send_warning_message("SpinSpiral: all magnetic-moment magnitudes are zero; returning input.")
            return [structure.copy()]

        positions = np.asarray(structure.get_positions(), dtype=float)
        phase_positions = self._phase_positions(positions, axis)
        outputs = []
        reached_limit = False

        for period in periods:
            for phase_deg in phases:
                for mz in mz_values:
                    for chirality_tag, chirality_sign in self._chirality_values():
                        atoms = structure.copy()
                        unit_vectors = spiral_unit_vectors(
                            phase_positions,
                            axis=axis,
                            period=float(period),
                            mz=float(mz),
                            phase_deg=float(phase_deg),
                            chirality=chirality_sign,
                        )
                        magmoms = mags[:, None] * unit_vectors
                        set_initial_magmoms_safe(atoms, magmoms)

                        kind = "Helix" if abs(mz) <= 1e-10 else "Spiral"
                        extra_tag = ""
                        if self.phase_mode_combo.currentText() == "Layer-locked":
                            extra_tag = f",pm=layer,ltol={float(self.layer_tol_frame.get_input_value()[0]):.4g}"
                        append_config_tag(
                            atoms,
                            f"{kind}(L={float(period):.6g},ph={float(phase_deg):.6g},mz={float(mz):.6g},"
                            f"chi={chirality_tag},ax={self._axis_tag(axis)}{extra_tag})",
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
        data["phase_mode"] = self.phase_mode_combo.currentText()
        data["layer_tolerance"] = self.layer_tol_frame.get_input_value()
        data["only_commensurate_periods"] = self.commensurate_checkbox.isChecked()
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
        self.mz_frame.set_input_value(self._coerce_scan_triplet(data_dict.get("mz", [0.0, 0.0, 0.1]), default_step=0.1))
        self.chirality_combo.setCurrentText(data_dict.get("chirality", "Both"))
        self.phase_mode_combo.setCurrentText(data_dict.get("phase_mode", "Continuous by position"))
        self.layer_tol_frame.set_input_value(data_dict.get("layer_tolerance", [0.05]))
        self.commensurate_checkbox.setChecked(bool(data_dict.get("only_commensurate_periods", False)))
        self.source_combo.setCurrentText(data_dict.get("magnitude_source", "Existing initial magmoms"))
        self.map_edit.setText(data_dict.get("magmom_map", ""))
        self.default_frame.set_input_value(data_dict.get("default_moment", [0.0]))
        self.apply_edit.setText(data_dict.get("apply_elements", ""))
        self.max_output_frame.set_input_value(data_dict.get("max_outputs", [100]))
        self._update_magnitude_source_widgets()
        self._update_parameter_mode_widgets()
        self._update_phase_mode_widgets()
