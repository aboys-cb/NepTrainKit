"""Card for generating single-spin small-angle tilt configurations."""

from __future__ import annotations

import math
import re

import numpy as np
from ase.geometry import get_distances
from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.core.magnetism import (
    existing_moment_vectors,
    mapped_moment_vectors,
    normalize_vector,
    orthonormal_frame,
    parse_element_set,
    parse_magmom_map_any,
    set_initial_magmoms_safe,
)
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


def _parse_angle_list(text: str) -> list[float]:
    """Parse a comma-separated list of positive tilt angles in degrees."""
    values: list[float] = []
    seen: set[float] = set()
    for token in re.split(r"[\s,;]+", text or ""):
        if not token.strip():
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if value <= 0:
            continue
        rounded = float(np.round(value, 12))
        if rounded in seen:
            continue
        seen.add(rounded)
        values.append(rounded)
    return values


def _parse_atom_indices(text: str, natoms: int) -> list[int]:
    """Parse 1-based atom indices from tokens such as ``1,3-5``."""
    indices: list[int] = []
    seen: set[int] = set()
    for token in re.split(r"[\s,;]+", text or ""):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            raw_values = range(start, end + 1)
        else:
            try:
                raw_values = [int(token)]
            except ValueError:
                continue

        for raw in raw_values:
            idx = raw - 1
            if idx < 0 or idx >= natoms or idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
    return indices


def _parse_pair_filter(text: str, *, normalize_case: bool = False) -> set[tuple[str, str]]:
    """Parse pair filters such as ``Fe-Co,Fe-Fe`` into canonical tuple pairs."""
    pairs: set[tuple[str, str]] = set()
    for token in re.split(r"[\s,;]+", text or ""):
        token = token.strip()
        if not token or "-" not in token:
            continue
        left, right = [part.strip() for part in token.split("-", 1)]
        if not left or not right:
            continue
        if normalize_case:
            left = left[0].upper() + left[1:].lower()
            right = right[0].upper() + right[1:].lower()
        pair = tuple(sorted((left, right)))
        pairs.add(pair)
    return pairs


@CardManager.register_card
class SmallAngleSpinTiltCard(MakeDataCard):
    """Generate deterministic single-spin small-angle tilt configurations."""

    group = "Magnetism"
    card_name = "Small-Angle Spin Tilt"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Small-Angle Spin Tilt")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("small_angle_spin_tilt_card_widget")

        self.canting_mode_label = BodyLabel("Canting mode", self.setting_widget)
        self.canting_mode_label.setToolTip("Choose single-spin tilt, explicit atom-pair canting, or group-pair canting")
        self.canting_mode_label.installEventFilter(ToolTipFilter(self.canting_mode_label, 300, ToolTipPosition.TOP))
        self.canting_mode_combo = ComboBox(self.setting_widget)
        self.canting_mode_combo.addItems(["Single-spin tilt", "Atom pair canting", "Group pair canting"])
        self.canting_mode_combo.setCurrentText("Single-spin tilt")

        self.target_mode_label = BodyLabel("Target atoms", self.setting_widget)
        self.target_mode_label.setToolTip("Choose which atoms receive the single-spin tilt")
        self.target_mode_label.installEventFilter(ToolTipFilter(self.target_mode_label, 300, ToolTipPosition.TOP))
        self.target_mode_combo = ComboBox(self.setting_widget)
        self.target_mode_combo.addItems(["First eligible atom", "All eligible atoms", "Explicit indices (1-based)"])
        self.target_mode_combo.setCurrentText("First eligible atom")

        self.target_indices_label = BodyLabel("Atom indices", self.setting_widget)
        self.target_indices_label.setToolTip("Used when Target atoms = Explicit indices (1-based), for example 1,3-5")
        self.target_indices_label.installEventFilter(
            ToolTipFilter(self.target_indices_label, 300, ToolTipPosition.TOP)
        )
        self.target_indices_edit = LineEdit(self.setting_widget)
        self.target_indices_edit.setPlaceholderText("1,3-5")

        self.pair_left_label = BodyLabel("Pair left indices", self.setting_widget)
        self.pair_left_label.setToolTip("1-based indices for the left side of each atom pair, for example 1,3-5")
        self.pair_left_label.installEventFilter(ToolTipFilter(self.pair_left_label, 300, ToolTipPosition.TOP))
        self.pair_left_edit = LineEdit(self.setting_widget)
        self.pair_left_edit.setPlaceholderText("1")

        self.pair_right_label = BodyLabel("Pair right indices", self.setting_widget)
        self.pair_right_label.setToolTip("1-based indices for the right side of each atom pair, paired in order with left indices")
        self.pair_right_label.installEventFilter(ToolTipFilter(self.pair_right_label, 300, ToolTipPosition.TOP))
        self.pair_right_edit = LineEdit(self.setting_widget)
        self.pair_right_edit.setPlaceholderText("2")

        self.pair_source_label = BodyLabel("Pair source", self.setting_widget)
        self.pair_source_label.setToolTip("Use explicit atom indices or auto-select unique neighbor-shell pairs")
        self.pair_source_label.installEventFilter(ToolTipFilter(self.pair_source_label, 300, ToolTipPosition.TOP))
        self.pair_source_combo = ComboBox(self.setting_widget)
        self.pair_source_combo.addItems(["Manual indices", "Auto by neighbor shell"])
        self.pair_source_combo.setCurrentText("Manual indices")

        self.pair_shell_label = BodyLabel("Neighbor shell", self.setting_widget)
        self.pair_shell_label.setToolTip("1 = first-neighbor shell, 2 = second-neighbor shell, etc.")
        self.pair_shell_label.installEventFilter(ToolTipFilter(self.pair_shell_label, 300, ToolTipPosition.TOP))
        self.pair_shell_frame = SpinBoxUnitInputFrame(self)
        self.pair_shell_frame.set_input("", 1, "int")
        self.pair_shell_frame.setRange(1, 20)
        self.pair_shell_frame.set_input_value([1])

        self.pair_tol_label = BodyLabel("Shell tolerance", self.setting_widget)
        self.pair_tol_label.setToolTip("Distances within this tolerance belong to the same neighbor shell")
        self.pair_tol_label.installEventFilter(ToolTipFilter(self.pair_tol_label, 300, ToolTipPosition.TOP))
        self.pair_tol_frame = SpinBoxUnitInputFrame(self)
        self.pair_tol_frame.set_input("A", 1, "float")
        self.pair_tol_frame.setRange(0.0001, 5.0)
        self.pair_tol_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.pair_tol_frame.set_input_value([0.05])

        self.pair_element_label = BodyLabel("Pair elements", self.setting_widget)
        self.pair_element_label.setToolTip("Optional pair filter such as Fe-Fe,Fe-Co; empty means any element pair")
        self.pair_element_label.installEventFilter(ToolTipFilter(self.pair_element_label, 300, ToolTipPosition.TOP))
        self.pair_element_edit = LineEdit(self.setting_widget)
        self.pair_element_edit.setPlaceholderText("Fe-Fe,Fe-Co")

        self.pair_group_label = BodyLabel("Pair groups", self.setting_widget)
        self.pair_group_label.setToolTip("Optional group-pair filter such as A-B,A-A; requires arrays['group']")
        self.pair_group_label.installEventFilter(ToolTipFilter(self.pair_group_label, 300, ToolTipPosition.TOP))
        self.pair_group_edit = LineEdit(self.setting_widget)
        self.pair_group_edit.setPlaceholderText("A-B")

        self.bond_mode_label = BodyLabel("Bond filter", self.setting_widget)
        self.bond_mode_label.setToolTip("Optional bond-direction filter for auto pairs")
        self.bond_mode_label.installEventFilter(ToolTipFilter(self.bond_mode_label, 300, ToolTipPosition.TOP))
        self.bond_mode_combo = ComboBox(self.setting_widget)
        self.bond_mode_combo.addItems(["Any", "Near axis", "In plane (normal)"])
        self.bond_mode_combo.setCurrentText("Any")

        self.bond_axis_label = BodyLabel("Bond reference", self.setting_widget)
        self.bond_axis_label.setToolTip("Reference axis or plane normal used by the bond-direction filter")
        self.bond_axis_label.installEventFilter(ToolTipFilter(self.bond_axis_label, 300, ToolTipPosition.TOP))
        self.bond_axis_frame = SpinBoxUnitInputFrame(self)
        self.bond_axis_frame.set_input("", 3, "float")
        self.bond_axis_frame.setRange(-1.0, 1.0)
        for obj in self.bond_axis_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.bond_axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.bond_tol_label = BodyLabel("Bond angle tol", self.setting_widget)
        self.bond_tol_label.setToolTip("Angular tolerance in degrees for the bond-direction filter")
        self.bond_tol_label.installEventFilter(ToolTipFilter(self.bond_tol_label, 300, ToolTipPosition.TOP))
        self.bond_tol_frame = SpinBoxUnitInputFrame(self)
        self.bond_tol_frame.set_input("deg", 1, "float")
        self.bond_tol_frame.setRange(0.1, 90.0)
        self.bond_tol_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.bond_tol_frame.set_input_value([20.0])

        self.group_a_label = BodyLabel("Group A", self.setting_widget)
        self.group_a_label.setToolTip("Atoms with arrays['group']==Group A rotate by +theta/2 in group-pair mode")
        self.group_a_label.installEventFilter(ToolTipFilter(self.group_a_label, 300, ToolTipPosition.TOP))
        self.group_a_edit = LineEdit(self.setting_widget)
        self.group_a_edit.setText("A")

        self.group_b_label = BodyLabel("Group B", self.setting_widget)
        self.group_b_label.setToolTip("Atoms with arrays['group']==Group B rotate by -theta/2 in group-pair mode")
        self.group_b_label.installEventFilter(ToolTipFilter(self.group_b_label, 300, ToolTipPosition.TOP))
        self.group_b_edit = LineEdit(self.setting_widget)
        self.group_b_edit.setText("B")

        self.angle_label = BodyLabel("Tilt angles", self.setting_widget)
        self.angle_label.setToolTip("Comma-separated tilt angles in degrees, for example 1,2,5,10")
        self.angle_label.installEventFilter(ToolTipFilter(self.angle_label, 300, ToolTipPosition.TOP))
        self.angle_edit = LineEdit(self.setting_widget)
        self.angle_edit.setPlaceholderText("1,2,5,10")
        self.angle_edit.setText("1,2,5,10")

        self.sign_label = BodyLabel("Tilt signs", self.setting_widget)
        self.sign_label.setToolTip("Choose whether to emit +theta only, -theta only, or paired +/-theta variants")
        self.sign_label.installEventFilter(ToolTipFilter(self.sign_label, 300, ToolTipPosition.TOP))
        self.sign_combo = ComboBox(self.setting_widget)
        self.sign_combo.addItems(["Positive only", "Negative only", "Both (+/- pair)"])
        self.sign_combo.setCurrentText("Positive only")

        self.include_reference_checkbox = CheckBox("Include reference state", self.setting_widget)
        self.include_reference_checkbox.setChecked(True)
        self.include_reference_checkbox.setToolTip("Emit the un-tilted reference magnetic state before tilted variants")
        self.include_reference_checkbox.installEventFilter(
            ToolTipFilter(self.include_reference_checkbox, 300, ToolTipPosition.TOP)
        )

        self.source_label = BodyLabel("Magnitude source", self.setting_widget)
        self.source_label.setToolTip("Use existing initial magmoms or build a ferromagnetic reference from map/default")
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

        self.lift_scalar_checkbox = CheckBox("Lift scalar magmoms to vectors", self.setting_widget)
        self.lift_scalar_checkbox.setChecked(True)
        self.lift_scalar_checkbox.setToolTip("If input magmoms are scalars, place them along Base axis before tilting")
        self.lift_scalar_checkbox.installEventFilter(
            ToolTipFilter(self.lift_scalar_checkbox, 300, ToolTipPosition.TOP)
        )

        self.axis_label = BodyLabel("Base axis", self.setting_widget)
        self.axis_label.setToolTip("Reference axis for lifted scalar magmoms and map/default ferromagnetic states")
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setRange(-1.0, 1.0)
        for obj in self.axis_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.reference_label = BodyLabel("Tilt reference", self.setting_widget)
        self.reference_label.setToolTip("Preferred direction that defines the tilt plane; it is orthogonalised automatically")
        self.reference_label.installEventFilter(ToolTipFilter(self.reference_label, 300, ToolTipPosition.TOP))
        self.reference_frame = SpinBoxUnitInputFrame(self)
        self.reference_frame.set_input("", 3, "float")
        self.reference_frame.setRange(-1.0, 1.0)
        for obj in self.reference_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.reference_frame.set_input_value([1.0, 0.0, 0.0])

        self.apply_label = BodyLabel("Apply elements", self.setting_widget)
        self.apply_label.setToolTip("Optional comma-separated element list; empty means all atoms are eligible targets")
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

        self.settingLayout.addWidget(self.canting_mode_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.canting_mode_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.target_mode_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.target_mode_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.target_indices_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.target_indices_edit, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_source_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_source_combo, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_left_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_left_edit, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_right_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_right_edit, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_shell_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_shell_frame, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_tol_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_tol_frame, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_element_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_element_edit, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_group_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_group_edit, 9, 1, 1, 2)
        self.settingLayout.addWidget(self.bond_mode_label, 10, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_mode_combo, 10, 1, 1, 2)
        self.settingLayout.addWidget(self.bond_axis_label, 11, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_axis_frame, 11, 1, 1, 2)
        self.settingLayout.addWidget(self.bond_tol_label, 12, 0, 1, 1)
        self.settingLayout.addWidget(self.bond_tol_frame, 12, 1, 1, 2)
        self.settingLayout.addWidget(self.group_a_label, 13, 0, 1, 1)
        self.settingLayout.addWidget(self.group_a_edit, 13, 1, 1, 2)
        self.settingLayout.addWidget(self.group_b_label, 14, 0, 1, 1)
        self.settingLayout.addWidget(self.group_b_edit, 14, 1, 1, 2)
        self.settingLayout.addWidget(self.angle_label, 15, 0, 1, 1)
        self.settingLayout.addWidget(self.angle_edit, 15, 1, 1, 2)
        self.settingLayout.addWidget(self.sign_label, 16, 0, 1, 1)
        self.settingLayout.addWidget(self.sign_combo, 16, 1, 1, 2)
        self.settingLayout.addWidget(self.include_reference_checkbox, 17, 0, 1, 3)
        self.settingLayout.addWidget(self.source_label, 18, 0, 1, 1)
        self.settingLayout.addWidget(self.source_combo, 18, 1, 1, 2)
        self.settingLayout.addWidget(self.map_label, 19, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 19, 1, 1, 2)
        self.settingLayout.addWidget(self.default_label, 20, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 20, 1, 1, 2)
        self.settingLayout.addWidget(self.lift_scalar_checkbox, 21, 0, 1, 3)
        self.settingLayout.addWidget(self.axis_label, 22, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_frame, 22, 1, 1, 2)
        self.settingLayout.addWidget(self.reference_label, 23, 0, 1, 1)
        self.settingLayout.addWidget(self.reference_frame, 23, 1, 1, 2)
        self.settingLayout.addWidget(self.apply_label, 24, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 24, 1, 1, 2)
        self.settingLayout.addWidget(self.max_output_label, 25, 0, 1, 1)
        self.settingLayout.addWidget(self.max_output_frame, 25, 1, 1, 2)

        self.canting_mode_combo.currentTextChanged.connect(self._update_canting_mode_widgets)
        self.pair_source_combo.currentTextChanged.connect(self._update_canting_mode_widgets)
        self.target_mode_combo.currentTextChanged.connect(self._update_target_widgets)
        self.source_combo.currentTextChanged.connect(self._update_magnitude_source_widgets)
        self.bond_mode_combo.currentTextChanged.connect(self._update_canting_mode_widgets)
        self._update_canting_mode_widgets()
        self._update_target_widgets()
        self._update_magnitude_source_widgets()

    def _update_canting_mode_widgets(self):
        mode = self.canting_mode_combo.currentText()
        single_mode = mode == "Single-spin tilt"
        pair_mode = mode == "Atom pair canting"
        group_mode = mode == "Group pair canting"
        auto_pair = pair_mode and self.pair_source_combo.currentText() == "Auto by neighbor shell"
        manual_pair = pair_mode and not auto_pair

        self.target_mode_label.setVisible(single_mode)
        self.target_mode_combo.setVisible(single_mode)
        self.target_mode_label.setEnabled(single_mode)
        self.target_mode_combo.setEnabled(single_mode)

        explicit = single_mode and self.target_mode_combo.currentText() == "Explicit indices (1-based)"
        self.target_indices_label.setVisible(explicit)
        self.target_indices_edit.setVisible(explicit)
        self.target_indices_label.setEnabled(explicit)
        self.target_indices_edit.setEnabled(explicit)

        for widget in (self.pair_source_label, self.pair_source_combo):
            widget.setVisible(pair_mode)
            widget.setEnabled(pair_mode)

        for widget in (self.pair_left_label, self.pair_left_edit, self.pair_right_label, self.pair_right_edit):
            widget.setVisible(manual_pair)
            widget.setEnabled(manual_pair)

        for widget in (
            self.pair_shell_label,
            self.pair_shell_frame,
            self.pair_tol_label,
            self.pair_tol_frame,
            self.pair_element_label,
            self.pair_element_edit,
            self.pair_group_label,
            self.pair_group_edit,
            self.bond_mode_label,
            self.bond_mode_combo,
        ):
            widget.setVisible(auto_pair)
            widget.setEnabled(auto_pair)

        show_bond_detail = auto_pair and self.bond_mode_combo.currentText() != "Any"
        for widget in (self.bond_axis_label, self.bond_axis_frame, self.bond_tol_label, self.bond_tol_frame):
            widget.setVisible(show_bond_detail)
            widget.setEnabled(show_bond_detail)

        for widget in (self.group_a_label, self.group_a_edit, self.group_b_label, self.group_b_edit):
            widget.setVisible(group_mode)
            widget.setEnabled(group_mode)

    def _update_target_widgets(self):
        explicit = (
            self.canting_mode_combo.currentText() == "Single-spin tilt"
            and self.target_mode_combo.currentText() == "Explicit indices (1-based)"
        )
        self.target_indices_label.setEnabled(explicit)
        self.target_indices_edit.setEnabled(explicit)
        self.target_indices_label.setVisible(explicit)
        self.target_indices_edit.setVisible(explicit)

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

    def _axis(self) -> np.ndarray:
        values = [float(v) for v in self.axis_frame.get_input_value()]
        return normalize_vector(np.array(values, dtype=float))

    def _reference_direction(self) -> np.ndarray:
        values = [float(v) for v in self.reference_frame.get_input_value()]
        return normalize_vector(np.array(values, dtype=float), default=np.array([1.0, 0.0, 0.0], dtype=float))

    def _angles(self) -> list[float]:
        values = _parse_angle_list(self.angle_edit.text())
        if values:
            return values
        MessageManager.send_warning_message("SmallAngleSpinTilt: invalid tilt angles; using default 1,2,5,10 deg.")
        return [1.0, 2.0, 5.0, 10.0]

    def _signs(self) -> list[tuple[str, float]]:
        mode = self.sign_combo.currentText()
        if mode == "Negative only":
            return [("neg", -1.0)]
        if mode == "Both (+/- pair)":
            return [("pos", 1.0), ("neg", -1.0)]
        return [("pos", 1.0)]

    def _pair_targets(self, structure, base_moments: np.ndarray) -> list[tuple[int, int]]:
        if self.pair_source_combo.currentText() == "Auto by neighbor shell":
            return self._auto_pair_targets(structure, base_moments)
        left = _parse_atom_indices(self.pair_left_edit.text(), len(structure))
        right = _parse_atom_indices(self.pair_right_edit.text(), len(structure))
        if not left or not right:
            return []
        if len(left) != len(right):
            MessageManager.send_warning_message(
                "SmallAngleSpinTilt: pair index counts differ; truncating to the shorter side."
            )
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        norms = np.linalg.norm(base_moments, axis=1)
        for left_idx, right_idx in zip(left, right):
            if left_idx == right_idx:
                continue
            if norms[left_idx] <= 1e-10 or norms[right_idx] <= 1e-10:
                continue
            pair = (left_idx, right_idx)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
        return pairs

    def _auto_pair_targets(self, structure, base_moments: np.ndarray) -> list[tuple[int, int]]:
        norms = np.linalg.norm(base_moments, axis=1)
        apply_elements = parse_element_set(self.apply_edit.text())
        eligible = [
            idx
            for idx, (sym, mag) in enumerate(zip(structure.get_chemical_symbols(), norms))
            if mag > 1e-10 and (not apply_elements or sym in apply_elements)
        ]
        if len(eligible) < 2:
            return []

        positions = np.asarray(structure.get_positions(), dtype=float)
        vec_matrix, dist_matrix = get_distances(
            positions[eligible],
            positions[eligible],
            cell=np.asarray(structure.cell),
            pbc=np.asarray(structure.pbc, dtype=bool),
        )
        tol = float(self.pair_tol_frame.get_input_value()[0])
        shell_index = int(self.pair_shell_frame.get_input_value()[0]) - 1

        pair_distances: list[tuple[int, int, float, np.ndarray]] = []
        for row in range(len(eligible)):
            for col in range(row + 1, len(eligible)):
                dist = float(dist_matrix[row, col])
                if dist <= 1e-12:
                    continue
                pair_distances.append((eligible[row], eligible[col], dist, np.asarray(vec_matrix[row, col], dtype=float)))
        if not pair_distances:
            return []

        unique_distances = sorted(dist for _, _, dist, _ in pair_distances)
        shells: list[float] = []
        for dist in unique_distances:
            if not shells or abs(dist - shells[-1]) > tol:
                shells.append(dist)
        if shell_index < 0 or shell_index >= len(shells):
            MessageManager.send_warning_message(
                f"SmallAngleSpinTilt: neighbor shell {shell_index + 1} not found; available shells={len(shells)}."
            )
            return []

        target_distance = shells[shell_index]
        return [
            (i, j)
            for i, j, dist, bond_vector in pair_distances
            if abs(dist - target_distance) <= tol
            and self._passes_pair_filters(structure, i, j, bond_vector)
        ]

    def _passes_pair_filters(self, structure, left_idx: int, right_idx: int, bond_vector: np.ndarray) -> bool:
        element_pairs = _parse_pair_filter(self.pair_element_edit.text(), normalize_case=True)
        if element_pairs:
            syms = structure.get_chemical_symbols()
            pair = tuple(sorted((syms[left_idx], syms[right_idx])))
            if pair not in element_pairs:
                return False

        group_pairs = _parse_pair_filter(self.pair_group_edit.text(), normalize_case=False)
        if group_pairs:
            if "group" not in structure.arrays:
                return False
            groups = [str(g) for g in np.asarray(structure.arrays["group"])]
            pair = tuple(sorted((groups[left_idx], groups[right_idx])))
            if pair not in group_pairs:
                return False

        mode = self.bond_mode_combo.currentText()
        if mode == "Any":
            return True

        reference = normalize_vector(np.array(self.bond_axis_frame.get_input_value(), dtype=float))
        bond_hat = normalize_vector(np.asarray(bond_vector, dtype=float), default=reference)
        cos_angle = float(np.clip(abs(np.dot(bond_hat, reference)), 0.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_angle)))
        tolerance = float(self.bond_tol_frame.get_input_value()[0])
        if mode == "Near axis":
            return angle <= tolerance
        return abs(90.0 - angle) <= tolerance

    def _group_targets(self, structure, base_moments: np.ndarray) -> tuple[list[int], list[int]]:
        if "group" not in structure.arrays:
            return [], []
        group_values = [str(g) for g in np.asarray(structure.arrays["group"])]
        group_a = (self.group_a_edit.text() or "A").strip()
        group_b = (self.group_b_edit.text() or "B").strip()
        norms = np.linalg.norm(base_moments, axis=1)
        left = [idx for idx, (g, mag) in enumerate(zip(group_values, norms)) if g == group_a and mag > 1e-10]
        right = [idx for idx, (g, mag) in enumerate(zip(group_values, norms)) if g == group_b and mag > 1e-10]
        return left, right

    def _vector_moments(self, structure) -> np.ndarray | None:
        use_existing = self.source_combo.currentText() == "Existing initial magmoms"
        axis = self._axis()

        if use_existing:
            values = existing_moment_vectors(
                structure,
                axis=axis,
                lift_scalar=self.lift_scalar_checkbox.isChecked(),
            )
            if values is not None:
                return values
            if len(np.asarray(structure.get_initial_magnetic_moments(), dtype=float).shape) == 1 and not self.lift_scalar_checkbox.isChecked():
                MessageManager.send_warning_message(
                    "SmallAngleSpinTilt: scalar initial_magmoms require Lift scalar magmoms to vectors."
                )
                return None
            MessageManager.send_warning_message(
                "SmallAngleSpinTilt: no valid initial_magmoms found; falling back to magmom map/default."
            )

        try:
            moment_map = parse_magmom_map_any(self.map_edit.text())
        except Exception as exc:  # noqa: BLE001
            MessageManager.send_warning_message(f"SmallAngleSpinTilt: invalid magmom map: {exc}")
            moment_map = {}

        return mapped_moment_vectors(
            structure,
            moment_map,
            default_moment=float(self.default_frame.get_input_value()[0]),
            axis=axis,
            apply_elements=parse_element_set(self.apply_edit.text()),
            use_element_directions=False,
        )

    def _candidate_indices(self, structure, base_moments: np.ndarray) -> list[int]:
        norms = np.linalg.norm(base_moments, axis=1)
        apply_elements = parse_element_set(self.apply_edit.text())
        eligible = [
            idx
            for idx, (sym, mag) in enumerate(zip(structure.get_chemical_symbols(), norms))
            if mag > 1e-10 and (not apply_elements or sym in apply_elements)
        ]
        if not eligible:
            return []

        mode = self.target_mode_combo.currentText()
        if mode == "First eligible atom":
            return [eligible[0]]
        if mode == "All eligible atoms":
            return eligible

        explicit = _parse_atom_indices(self.target_indices_edit.text(), len(structure))
        explicit = [idx for idx in explicit if idx in set(eligible)]
        return explicit

    def _tilt_direction(self, base_direction: np.ndarray) -> np.ndarray:
        base_hat = normalize_vector(base_direction)
        preferred = self._reference_direction()
        preferred = preferred - float(np.dot(preferred, base_hat)) * base_hat
        if np.linalg.norm(preferred) <= 1e-10:
            e1, _, _ = orthonormal_frame(base_hat)
            return e1
        return normalize_vector(preferred)

    def _tilted_vector(self, vector: np.ndarray, angle_deg: float, *, sign: float = 1.0) -> np.ndarray:
        vec = np.asarray(vector, dtype=float).reshape(3)
        magnitude = float(np.linalg.norm(vec))
        if magnitude <= 0.0 or angle_deg <= 0.0:
            return vec.copy()

        base_hat = vec / magnitude
        tilt_hat = self._tilt_direction(base_hat)
        theta = math.radians(float(angle_deg) * float(sign))
        direction = (math.cos(theta) * base_hat) + (math.sin(theta) * tilt_hat)
        direction = normalize_vector(direction, default=base_hat)
        return magnitude * direction

    @staticmethod
    def _set_vector_magmoms(atoms, magmoms: np.ndarray):
        set_initial_magmoms_safe(atoms, magmoms)

    def _apply_pair_canting(
        self,
        base_moments: np.ndarray,
        left_indices: list[int],
        right_indices: list[int],
        angle_deg: float,
        *,
        sign: float,
    ) -> np.ndarray:
        moment_array = np.array(base_moments, copy=True)
        half_angle = float(angle_deg) * 0.5
        for idx in left_indices:
            moment_array[idx] = self._tilted_vector(moment_array[idx], half_angle, sign=sign)
        for idx in right_indices:
            moment_array[idx] = self._tilted_vector(moment_array[idx], half_angle, sign=-sign)
        return moment_array

    def process_structure(self, structure):
        base_moments = self._vector_moments(structure)
        if base_moments is None or base_moments.shape != (len(structure), 3):
            return [structure.copy()]

        if not np.any(np.linalg.norm(base_moments, axis=1) > 1e-10):
            MessageManager.send_warning_message("SmallAngleSpinTilt: all magnetic-moment magnitudes are zero; returning input.")
            return [structure.copy()]

        angles = self._angles()
        max_outputs = int(self.max_output_frame.get_input_value()[0])
        outputs = []
        reached_limit = False
        mode = self.canting_mode_combo.currentText()

        if self.include_reference_checkbox.isChecked():
            reference = structure.copy()
            self._set_vector_magmoms(reference, base_moments)
            append_config_tag(reference, "SpinTiltRef")
            outputs.append(reference)

        if mode == "Single-spin tilt":
            target_indices = self._candidate_indices(structure, base_moments)
            if not target_indices:
                MessageManager.send_warning_message("SmallAngleSpinTilt: no eligible target atoms found; returning input.")
                return outputs or [structure.copy()]
            for atom_index in target_indices:
                for angle_deg in angles:
                    for sign_tag, sign_value in self._signs():
                        tilted = structure.copy()
                        moment_array = np.array(base_moments, copy=True)
                        moment_array[atom_index] = self._tilted_vector(
                            moment_array[atom_index],
                            angle_deg,
                            sign=sign_value,
                        )
                        self._set_vector_magmoms(tilted, moment_array)
                        append_config_tag(
                            tilted,
                            f"SpinTilt(i={atom_index + 1},a={float(angle_deg):.6g},sg={sign_tag})",
                        )
                        outputs.append(tilted)
                        if len(outputs) >= max_outputs:
                            reached_limit = True
                            break
                    if reached_limit:
                        break
                if reached_limit:
                    break
        elif mode == "Atom pair canting":
            pairs = self._pair_targets(structure, base_moments)
            if not pairs:
                MessageManager.send_warning_message("SmallAngleSpinTilt: no valid atom pairs found; returning input.")
                return outputs or [structure.copy()]
            for left_idx, right_idx in pairs:
                for angle_deg in angles:
                    for sign_tag, sign_value in self._signs():
                        tilted = structure.copy()
                        moment_array = self._apply_pair_canting(
                            base_moments,
                            [left_idx],
                            [right_idx],
                            angle_deg,
                            sign=sign_value,
                        )
                        self._set_vector_magmoms(tilted, moment_array)
                        append_config_tag(
                            tilted,
                            f"SpinPair(i={left_idx + 1},j={right_idx + 1},a={float(angle_deg):.6g},sg={sign_tag})",
                        )
                        outputs.append(tilted)
                        if len(outputs) >= max_outputs:
                            reached_limit = True
                            break
                    if reached_limit:
                        break
                if reached_limit:
                    break
        else:
            left_group, right_group = self._group_targets(structure, base_moments)
            if not left_group or not right_group:
                MessageManager.send_warning_message(
                    "SmallAngleSpinTilt: group-pair mode requires arrays['group'] and non-empty Group A/B targets."
                )
                return outputs or [structure.copy()]
            group_a = (self.group_a_edit.text() or "A").strip()
            group_b = (self.group_b_edit.text() or "B").strip()
            for angle_deg in angles:
                for sign_tag, sign_value in self._signs():
                    tilted = structure.copy()
                    moment_array = self._apply_pair_canting(
                        base_moments,
                        left_group,
                        right_group,
                        angle_deg,
                        sign=sign_value,
                    )
                    self._set_vector_magmoms(tilted, moment_array)
                    append_config_tag(
                        tilted,
                        f"SpinPairG(A={group_a},B={group_b},a={float(angle_deg):.6g},sg={sign_tag})",
                    )
                    outputs.append(tilted)
                    if len(outputs) >= max_outputs:
                        reached_limit = True
                        break
                if reached_limit:
                    break

        if reached_limit:
            MessageManager.send_warning_message("SmallAngleSpinTilt: output truncated by Max outputs.")
        return outputs or [structure.copy()]

    def to_dict(self):
        data = super().to_dict()
        data["canting_mode"] = self.canting_mode_combo.currentText()
        data["target_mode"] = self.target_mode_combo.currentText()
        data["target_indices"] = self.target_indices_edit.text()
        data["pair_left_indices"] = self.pair_left_edit.text()
        data["pair_right_indices"] = self.pair_right_edit.text()
        data["pair_source"] = self.pair_source_combo.currentText()
        data["pair_shell"] = self.pair_shell_frame.get_input_value()
        data["pair_shell_tolerance"] = self.pair_tol_frame.get_input_value()
        data["pair_element_filter"] = self.pair_element_edit.text()
        data["pair_group_filter"] = self.pair_group_edit.text()
        data["bond_filter_mode"] = self.bond_mode_combo.currentText()
        data["bond_filter_axis"] = self.bond_axis_frame.get_input_value()
        data["bond_filter_tolerance"] = self.bond_tol_frame.get_input_value()
        data["group_a"] = self.group_a_edit.text()
        data["group_b"] = self.group_b_edit.text()
        data["angle_list"] = self.angle_edit.text()
        data["tilt_signs"] = self.sign_combo.currentText()
        data["include_reference"] = self.include_reference_checkbox.isChecked()
        data["magnitude_source"] = self.source_combo.currentText()
        data["magmom_map"] = self.map_edit.text()
        data["default_moment"] = self.default_frame.get_input_value()
        data["lift_scalar"] = self.lift_scalar_checkbox.isChecked()
        data["axis"] = self.axis_frame.get_input_value()
        data["reference_direction"] = self.reference_frame.get_input_value()
        data["apply_elements"] = self.apply_edit.text()
        data["max_outputs"] = self.max_output_frame.get_input_value()
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.canting_mode_combo.setCurrentText(data_dict.get("canting_mode", "Single-spin tilt"))
        self.target_mode_combo.setCurrentText(data_dict.get("target_mode", "First eligible atom"))
        self.target_indices_edit.setText(data_dict.get("target_indices", ""))
        self.pair_left_edit.setText(data_dict.get("pair_left_indices", ""))
        self.pair_right_edit.setText(data_dict.get("pair_right_indices", ""))
        self.pair_source_combo.setCurrentText(data_dict.get("pair_source", "Manual indices"))
        self.pair_shell_frame.set_input_value(data_dict.get("pair_shell", [1]))
        self.pair_tol_frame.set_input_value(data_dict.get("pair_shell_tolerance", [0.05]))
        self.pair_element_edit.setText(data_dict.get("pair_element_filter", ""))
        self.pair_group_edit.setText(data_dict.get("pair_group_filter", ""))
        self.bond_mode_combo.setCurrentText(data_dict.get("bond_filter_mode", "Any"))
        self.bond_axis_frame.set_input_value(data_dict.get("bond_filter_axis", [0.0, 0.0, 1.0]))
        self.bond_tol_frame.set_input_value(data_dict.get("bond_filter_tolerance", [20.0]))
        self.group_a_edit.setText(data_dict.get("group_a", "A"))
        self.group_b_edit.setText(data_dict.get("group_b", "B"))
        self.angle_edit.setText(data_dict.get("angle_list", "1,2,5,10"))
        self.sign_combo.setCurrentText(data_dict.get("tilt_signs", "Positive only"))
        self.include_reference_checkbox.setChecked(bool(data_dict.get("include_reference", True)))
        self.source_combo.setCurrentText(data_dict.get("magnitude_source", "Existing initial magmoms"))
        self.map_edit.setText(data_dict.get("magmom_map", ""))
        self.default_frame.set_input_value(data_dict.get("default_moment", [0.0]))
        self.lift_scalar_checkbox.setChecked(bool(data_dict.get("lift_scalar", True)))
        self.axis_frame.set_input_value(data_dict.get("axis", [0.0, 0.0, 1.0]))
        self.reference_frame.set_input_value(data_dict.get("reference_direction", [1.0, 0.0, 0.0]))
        self.apply_edit.setText(data_dict.get("apply_elements", ""))
        self.max_output_frame.set_input_value(data_dict.get("max_outputs", [100]))
        self._update_canting_mode_widgets()
        self._update_target_widgets()
        self._update_magnitude_source_widgets()
