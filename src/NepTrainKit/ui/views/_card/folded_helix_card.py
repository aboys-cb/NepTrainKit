"""Card for generating symmetric folded-helix magnetic textures layer by layer."""

from __future__ import annotations

import numpy as np
from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.core.magnetism import (
    element_mask,
    existing_moment_magnitudes,
    mapped_moment_magnitudes,
    normalize_vector,
    orthonormal_frame,
    parse_element_set,
    parse_magmom_map_any,
    set_initial_magmoms_safe,
)
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class FoldedHelixCard(MakeDataCard):
    """Assign symmetric clockwise-then-counterclockwise layered helix moments."""

    group = "Magnetism"
    card_name = "Folded Helix"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Folded Helix Generator")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("folded_helix_card_widget")

        self.layer_axis_label = BodyLabel("Layer axis", self.setting_widget)
        self.layer_axis_label.setToolTip("Axis used to project positions and group atoms into layers")
        self.layer_axis_label.installEventFilter(ToolTipFilter(self.layer_axis_label, 300, ToolTipPosition.TOP))
        self.layer_axis_frame = SpinBoxUnitInputFrame(self)
        self.layer_axis_frame.set_input("", 3, "float")
        self.layer_axis_frame.setRange(-1.0, 1.0)
        for obj in self.layer_axis_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.layer_axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.plane_normal_label = BodyLabel("Rotation-plane normal", self.setting_widget)
        self.plane_normal_label.setToolTip("Magnetic moments stay in the plane perpendicular to this normal")
        self.plane_normal_label.installEventFilter(ToolTipFilter(self.plane_normal_label, 300, ToolTipPosition.TOP))
        self.plane_normal_frame = SpinBoxUnitInputFrame(self)
        self.plane_normal_frame.set_input("", 3, "float")
        self.plane_normal_frame.setRange(-1.0, 1.0)
        for obj in self.plane_normal_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.plane_normal_frame.set_input_value([0.0, 0.0, 1.0])

        self.layer_tol_label = BodyLabel("Layer tolerance", self.setting_widget)
        self.layer_tol_label.setToolTip("Atoms whose projected coordinates differ by <= tolerance are treated as one layer")
        self.layer_tol_label.installEventFilter(ToolTipFilter(self.layer_tol_label, 300, ToolTipPosition.TOP))
        self.layer_tol_frame = SpinBoxUnitInputFrame(self)
        self.layer_tol_frame.set_input("A", 1, "float")
        self.layer_tol_frame.setRange(0.0001, 10.0)
        self.layer_tol_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.layer_tol_frame.set_input_value([0.05])

        self.half_period_mode_label = BodyLabel("Half-period mode", self.setting_widget)
        self.half_period_mode_label.setToolTip("Auto derives the folded half period from the detected layer count")
        self.half_period_mode_label.installEventFilter(
            ToolTipFilter(self.half_period_mode_label, 300, ToolTipPosition.TOP)
        )
        self.half_period_mode_combo = ComboBox(self.setting_widget)
        self.half_period_mode_combo.addItems(["Auto from layer count", "Manual"])
        self.half_period_mode_combo.setCurrentText("Auto from layer count")

        self.half_period_label = BodyLabel("Half-period layers", self.setting_widget)
        self.half_period_label.setToolTip("Used only in Manual mode: number of layer-to-layer steps from a boundary to the turning layer")
        self.half_period_label.installEventFilter(ToolTipFilter(self.half_period_label, 300, ToolTipPosition.TOP))
        self.half_period_frame = SpinBoxUnitInputFrame(self)
        self.half_period_frame.set_input(["-", "step", "layers"], 3, "int")
        self.half_period_frame.setRange(1, 999999)
        self.half_period_frame.set_input_value([2, 4, 1])

        self.angle_step_label = BodyLabel("Layer angle step", self.setting_widget)
        self.angle_step_label.setToolTip("In-plane rotation added between neighboring layers within each half period")
        self.angle_step_label.installEventFilter(ToolTipFilter(self.angle_step_label, 300, ToolTipPosition.TOP))
        self.angle_step_frame = SpinBoxUnitInputFrame(self)
        self.angle_step_frame.set_input(["-", "step", "deg"], 3, "float")
        self.angle_step_frame.setRange(0.0, 360.0)
        for obj in self.angle_step_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.angle_step_frame.set_input_value([15.0, 45.0, 15.0])

        self.phase_label = BodyLabel("Phase range", self.setting_widget)
        self.phase_label.setToolTip("Global in-plane phase offset in degrees: [min, max, step]")
        self.phase_label.installEventFilter(ToolTipFilter(self.phase_label, 300, ToolTipPosition.TOP))
        self.phase_frame = SpinBoxUnitInputFrame(self)
        self.phase_frame.set_input(["-", "step", "deg"], 3, "float")
        self.phase_frame.setRange(-360.0, 360.0)
        for obj in self.phase_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.phase_frame.set_input_value([0.0, 0.0, 15.0])

        self.sequence_label = BodyLabel("Sequence", self.setting_widget)
        self.sequence_label.setToolTip("Choose the handedness order across one folded period")
        self.sequence_label.installEventFilter(ToolTipFilter(self.sequence_label, 300, ToolTipPosition.TOP))
        self.sequence_combo = ComboBox(self.setting_widget)
        self.sequence_combo.addItems(
            [
                "Clockwise then counterclockwise",
                "Counterclockwise then clockwise",
                "Both",
            ]
        )
        self.sequence_combo.setCurrentText("Clockwise then counterclockwise")

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

        self.settingLayout.addWidget(self.layer_axis_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.layer_axis_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.plane_normal_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.plane_normal_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.layer_tol_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.layer_tol_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.half_period_mode_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.half_period_mode_combo, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.half_period_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.half_period_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.angle_step_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.angle_step_frame, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.phase_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.phase_frame, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.sequence_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.sequence_combo, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.source_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.source_combo, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.map_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 9, 1, 1, 2)
        self.settingLayout.addWidget(self.default_label, 10, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 10, 1, 1, 2)
        self.settingLayout.addWidget(self.apply_label, 11, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 11, 1, 1, 2)
        self.settingLayout.addWidget(self.max_output_label, 12, 0, 1, 1)
        self.settingLayout.addWidget(self.max_output_frame, 12, 1, 1, 2)

        self.source_combo.currentTextChanged.connect(self._update_magnitude_source_widgets)
        self.half_period_mode_combo.currentTextChanged.connect(self._update_half_period_mode_widgets)
        self._update_magnitude_source_widgets()
        self._update_half_period_mode_widgets()

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

    def _update_half_period_mode_widgets(self):
        manual = self.half_period_mode_combo.currentText() == "Manual"
        self.half_period_label.setEnabled(manual)
        self.half_period_frame.setEnabled(manual)
        self.half_period_label.setVisible(manual)
        self.half_period_frame.setVisible(manual)

    @staticmethod
    def _float_range_values(values: list[float], *, minimum: float | None = None) -> list[float]:
        start, stop, step = [float(v) for v in values]
        if minimum is not None:
            start = max(start, minimum)
            stop = max(stop, minimum)
        if stop < start:
            start, stop = stop, start
        if abs(stop - start) <= 1e-12 or step <= 0:
            return [start]
        raw = list(np.arange(start, stop + abs(step) * 0.5, abs(step), dtype=float))
        ordered: list[float] = []
        for value in raw or [start]:
            rounded = float(np.round(value, 12))
            if not ordered or abs(rounded - ordered[-1]) > 1e-10:
                ordered.append(rounded)
        return ordered

    @staticmethod
    def _int_range_values(values: list[int], *, minimum: int = 1) -> list[int]:
        start, stop, step = [int(v) for v in values]
        start = max(start, minimum)
        stop = max(stop, minimum)
        if stop < start:
            start, stop = stop, start
        step = abs(step)
        if stop == start or step <= 0:
            return [start]
        return list(range(start, stop + 1, step))

    @staticmethod
    def _vector_tag(axis: np.ndarray) -> str:
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

    def _layer_axis(self) -> np.ndarray:
        values = [float(v) for v in self.layer_axis_frame.get_input_value()]
        return normalize_vector(np.array(values, dtype=float))

    def _plane_normal(self) -> np.ndarray:
        values = [float(v) for v in self.plane_normal_frame.get_input_value()]
        return normalize_vector(np.array(values, dtype=float))

    def _sequence_values(self) -> list[tuple[str, int]]:
        mode = self.sequence_combo.currentText()
        if mode == "Counterclockwise then clockwise":
            return [("ccw-cw", 1)]
        if mode == "Both":
            return [("cw-ccw", -1), ("ccw-cw", 1)]
        return [("cw-ccw", -1)]

    def _half_period_values(self, *, layer_count: int) -> list[int]:
        if self.half_period_mode_combo.currentText() == "Manual":
            return self._int_range_values(self.half_period_frame.get_input_value(), minimum=1)
        derived = max(1, (int(layer_count) - 1) // 2)
        return [derived]

    @staticmethod
    def _auto_folded_steps(layer_ids: np.ndarray, *, layer_count: int) -> np.ndarray:
        if layer_ids.size == 0 or layer_count <= 1:
            return np.zeros_like(layer_ids, dtype=float)
        top_index = float(layer_count - 1)
        half_span = top_index / 2.0
        layer_pos = layer_ids.astype(float)
        return half_span - np.abs(layer_pos - half_span)

    def _magnitudes(self, structure) -> np.ndarray:
        apply_elements = parse_element_set(self.apply_edit.text())
        use_existing = self.source_combo.currentText() == "Existing initial magmoms"
        if use_existing:
            mags = existing_moment_magnitudes(structure)
            if mags is not None:
                mask = element_mask(structure.get_chemical_symbols(), apply_elements)
                return np.where(mask, mags, 0.0)
            MessageManager.send_warning_message(
                "FoldedHelix: no valid initial_magmoms found; falling back to magmom map/default."
            )

        try:
            moment_map = parse_magmom_map_any(self.map_edit.text())
        except Exception as exc:  # noqa: BLE001
            MessageManager.send_warning_message(f"FoldedHelix: invalid magmom map: {exc}")
            moment_map = {}

        return mapped_moment_magnitudes(
            structure,
            moment_map,
            default_moment=float(self.default_frame.get_input_value()[0]),
            apply_elements=apply_elements,
        )

    def process_structure(self, structure):
        angle_steps = self._float_range_values(self.angle_step_frame.get_input_value(), minimum=0.0)
        phases = self._float_range_values(self.phase_frame.get_input_value())
        max_outputs = int(self.max_output_frame.get_input_value()[0])

        mags = self._magnitudes(structure)
        if mags.shape[0] != len(structure) or not np.any(mags > 0):
            MessageManager.send_warning_message("FoldedHelix: all magnetic-moment magnitudes are zero; returning input.")
            return [structure.copy()]

        layer_axis = self._layer_axis()
        plane_normal = self._plane_normal()
        e1, e2, plane_hat = orthonormal_frame(plane_normal)
        layer_ids = self._layer_ids(
            np.asarray(structure.get_positions(), dtype=float),
            layer_axis,
            float(self.layer_tol_frame.get_input_value()[0]),
        )
        layer_count = int(layer_ids.max()) + 1 if layer_ids.size else 0
        half_periods = self._half_period_values(layer_count=layer_count)

        outputs = []
        reached_limit = False

        auto_mode = self.half_period_mode_combo.currentText() == "Auto from layer count"

        for half_period in half_periods:
            if auto_mode:
                folded_steps = self._auto_folded_steps(layer_ids, layer_count=layer_count)
            else:
                period_layers = max(2, 2 * int(half_period))
                local_layer = np.mod(layer_ids, period_layers)
                folded_steps = np.where(local_layer <= half_period, local_layer, period_layers - local_layer).astype(float)

            for angle_step in angle_steps:
                for phase_deg in phases:
                    for seq_tag, seq_sign in self._sequence_values():
                        atoms = structure.copy()
                        phase_rad = np.deg2rad(float(phase_deg) + seq_sign * folded_steps * float(angle_step))
                        unit_vectors = (
                            np.cos(phase_rad)[:, None] * e1[None, :]
                            + np.sin(phase_rad)[:, None] * e2[None, :]
                        )
                        set_initial_magmoms_safe(atoms, mags[:, None] * unit_vectors)
                        append_config_tag(
                            atoms,
                            (
                                f"FoldedHelix(h={half_period},da={float(angle_step):.6g},"
                                f"ph={float(phase_deg):.6g},seq={seq_tag},"
                                f"ax={self._vector_tag(layer_axis)},pn={self._vector_tag(plane_hat)})"
                            ),
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
            MessageManager.send_warning_message("FoldedHelix: output truncated by Max outputs.")
        return outputs or [structure.copy()]

    def to_dict(self):
        data = super().to_dict()
        data["layer_axis"] = self.layer_axis_frame.get_input_value()
        data["plane_normal"] = self.plane_normal_frame.get_input_value()
        data["layer_tolerance"] = self.layer_tol_frame.get_input_value()
        data["half_period_mode"] = self.half_period_mode_combo.currentText()
        data["half_period_layers"] = self.half_period_frame.get_input_value()
        data["angle_step_range"] = self.angle_step_frame.get_input_value()
        data["phase_range"] = self.phase_frame.get_input_value()
        data["sequence_mode"] = self.sequence_combo.currentText()
        data["magnitude_source"] = self.source_combo.currentText()
        data["magmom_map"] = self.map_edit.text()
        data["default_moment"] = self.default_frame.get_input_value()
        data["apply_elements"] = self.apply_edit.text()
        data["max_outputs"] = self.max_output_frame.get_input_value()
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.layer_axis_frame.set_input_value(data_dict.get("layer_axis", [0.0, 0.0, 1.0]))
        self.plane_normal_frame.set_input_value(data_dict.get("plane_normal", [0.0, 0.0, 1.0]))
        self.layer_tol_frame.set_input_value(data_dict.get("layer_tolerance", [0.05]))
        self.half_period_mode_combo.setCurrentText(data_dict.get("half_period_mode", "Auto from layer count"))
        self.half_period_frame.set_input_value(data_dict.get("half_period_layers", [2, 4, 1]))
        self.angle_step_frame.set_input_value(data_dict.get("angle_step_range", [15.0, 45.0, 15.0]))
        self.phase_frame.set_input_value(data_dict.get("phase_range", [0.0, 0.0, 15.0]))
        self.sequence_combo.setCurrentText(data_dict.get("sequence_mode", "Clockwise then counterclockwise"))
        self.source_combo.setCurrentText(data_dict.get("magnitude_source", "Existing initial magmoms"))
        self.map_edit.setText(data_dict.get("magmom_map", ""))
        self.default_frame.set_input_value(data_dict.get("default_moment", [0.0]))
        self.apply_edit.setText(data_dict.get("apply_elements", ""))
        self.max_output_frame.set_input_value(data_dict.get("max_outputs", [100]))
        self._update_magnitude_source_widgets()
        self._update_half_period_mode_widgets()
