"""Card for assigning collinear magnetic moments and generating FM/AFM/PM patterns."""

from __future__ import annotations

import json

import numpy as np
from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.magnetism import (
    kvec_signs,
    normalize_vector,
    parse_magmom_map_any,
    per_atom_magnitudes,
    random_signs,
    random_vector_moments,
)
from NepTrainKit.core.config_type import append_config_tag, sanitize_config_tag, stable_config_id
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class MagneticOrderCard(MakeDataCard):
    """Assign initial magnetic moments and generate common collinear spin patterns."""

    group = "Magnetism"
    card_name = "Magnetic Order"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Magnetic Order Generator")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("magnetic_order_card_widget")

        self.format_label = BodyLabel("Format", self.setting_widget)
        self.format_label.setToolTip("Collinear: scalar MAGMOM; Non-collinear: vector MAGMOM (mx,my,mz)")
        self.format_label.installEventFilter(ToolTipFilter(self.format_label, 300, ToolTipPosition.TOP))
        self.format_combo = ComboBox(self.setting_widget)
        self.format_combo.addItems(["Collinear (scalar)", "Non-collinear (vector)"])
        self.format_combo.setCurrentText("Collinear (scalar)")

        self.axis_label = BodyLabel("Axis (x,y,z)", self.setting_widget)
        self.axis_label.setToolTip("Reference axis for FM/AFM (and PM cone/plane when selected)")
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setDecimals(6)
        self.axis_frame.setRange(-1.0, 1.0)
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.map_label = BodyLabel("Magmom map", self.setting_widget)
        self.map_label.setToolTip('Per-element moments, e.g. "Fe:2.2,Co:1.7,Ni:0.6,Cr:1.0" or JSON')
        self.map_label.installEventFilter(ToolTipFilter(self.map_label, 300, ToolTipPosition.TOP))
        self.map_edit = LineEdit(self.setting_widget)
        self.map_edit.setPlaceholderText("Fe:2.2,Co:1.7,Ni:0.6,Cr:1.0")

        self.use_element_dir_checkbox = CheckBox("Use element vector directions (if provided)", self.setting_widget)
        self.use_element_dir_checkbox.setChecked(False)
        self.use_element_dir_checkbox.setToolTip('If the map provides vectors (e.g. Cr:[0,0,1]), use them as directions for FM/AFM')
        self.use_element_dir_checkbox.installEventFilter(
            ToolTipFilter(self.use_element_dir_checkbox, 300, ToolTipPosition.TOP)
        )

        self.default_label = BodyLabel("Default |m|", self.setting_widget)
        self.default_label.setToolTip("Magnitude for elements not listed in Magmom map")
        self.default_label.installEventFilter(ToolTipFilter(self.default_label, 300, ToolTipPosition.TOP))
        self.default_frame = SpinBoxUnitInputFrame(self)
        self.default_frame.set_input("", 1, "float")
        self.default_frame.setDecimals(6)
        self.default_frame.setRange(0.0, 20.0)
        self.default_frame.set_input_value([0.0])

        self.apply_label = BodyLabel("Apply elements", self.setting_widget)
        self.apply_label.setToolTip("Optional: only assign moments to these elements (comma-separated). Empty=all")
        self.apply_label.installEventFilter(ToolTipFilter(self.apply_label, 300, ToolTipPosition.TOP))
        self.apply_edit = LineEdit(self.setting_widget)
        self.apply_edit.setPlaceholderText("Fe,Co,Ni,Cr")

        self.fm_checkbox = CheckBox("Generate FM", self.setting_widget)
        self.fm_checkbox.setChecked(True)
        self.afm_checkbox = CheckBox("Generate AFM", self.setting_widget)
        self.afm_checkbox.setChecked(True)

        self.afm_mode_label = BodyLabel("AFM mode", self.setting_widget)
        self.afm_mode_label.setToolTip("AFM sign assignment: k-vector layers or explicit A/B groups")
        self.afm_mode_label.installEventFilter(ToolTipFilter(self.afm_mode_label, 300, ToolTipPosition.TOP))
        self.afm_mode_combo = ComboBox(self.setting_widget)
        self.afm_mode_combo.addItems(["k-vector", "group A/B"])

        self.kvec_label = BodyLabel("AFM k-vector", self.setting_widget)
        self.kvec_label.setToolTip("AFM modulation in fractional coordinates: 100, 010, 001, 110, 111")
        self.kvec_label.installEventFilter(ToolTipFilter(self.kvec_label, 300, ToolTipPosition.TOP))
        self.kvec_combo = ComboBox(self.setting_widget)
        self.kvec_combo.addItems(["100", "010", "001", "110", "111"])
        self.kvec_combo.setCurrentText("111")

        self.group_a_label = BodyLabel("AFM group +", self.setting_widget)
        self.group_a_label.setToolTip("Group label assigned + sign (requires arrays['group'])")
        self.group_a_label.installEventFilter(ToolTipFilter(self.group_a_label, 300, ToolTipPosition.TOP))
        self.group_a_edit = LineEdit(self.setting_widget)
        self.group_a_edit.setText("A")

        self.group_b_label = BodyLabel("AFM group -", self.setting_widget)
        self.group_b_label.setToolTip("Group label assigned - sign (requires arrays['group'])")
        self.group_b_label.installEventFilter(ToolTipFilter(self.group_b_label, 300, ToolTipPosition.TOP))
        self.group_b_edit = LineEdit(self.setting_widget)
        self.group_b_edit.setText("B")

        self.zero_unknown_groups_checkbox = CheckBox("Zero unknown groups", self.setting_widget)
        self.zero_unknown_groups_checkbox.setChecked(True)
        self.zero_unknown_groups_checkbox.setToolTip("If an atom group is neither A nor B, set its moment to 0")
        self.zero_unknown_groups_checkbox.installEventFilter(
            ToolTipFilter(self.zero_unknown_groups_checkbox, 300, ToolTipPosition.TOP)
        )

        self.pm_checkbox = CheckBox("Generate PM (random signs)", self.setting_widget)
        self.pm_checkbox.setChecked(False)
        self.pm_count_label = BodyLabel("PM structures", self.setting_widget)
        self.pm_count_frame = SpinBoxUnitInputFrame(self)
        self.pm_count_frame.set_input("unit", 1, "int")
        self.pm_count_frame.setRange(1, 999999)
        self.pm_count_frame.set_input_value([10])

        self.pm_direction_label = BodyLabel("PM directions", self.setting_widget)
        self.pm_direction_label.setToolTip("Non-collinear PM direction distribution")
        self.pm_direction_label.installEventFilter(ToolTipFilter(self.pm_direction_label, 300, ToolTipPosition.TOP))
        self.pm_direction_combo = ComboBox(self.setting_widget)
        self.pm_direction_combo.addItems(["sphere", "cone", "plane", "axis"])
        self.pm_direction_combo.setCurrentText("sphere")

        self.pm_cone_label = BodyLabel("PM cone angle", self.setting_widget)
        self.pm_cone_label.setToolTip("Cone half-angle in degrees (used when PM directions=cone)")
        self.pm_cone_label.installEventFilter(ToolTipFilter(self.pm_cone_label, 300, ToolTipPosition.TOP))
        self.pm_cone_frame = SpinBoxUnitInputFrame(self)
        self.pm_cone_frame.set_input("deg", 1, "float")
        self.pm_cone_frame.setDecimals(3)
        self.pm_cone_frame.setRange(0.0, 180.0)
        self.pm_cone_frame.set_input_value([30.0])

        self.pm_balanced_checkbox = CheckBox("PM balanced", self.setting_widget)
        self.pm_balanced_checkbox.setChecked(True)
        self.pm_balanced_checkbox.setToolTip("Try to balance +/- signs to near-zero net moment")
        self.pm_balanced_checkbox.installEventFilter(ToolTipFilter(self.pm_balanced_checkbox, 300, ToolTipPosition.TOP))

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.settingLayout.addWidget(self.format_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.format_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.axis_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_frame, 1, 1, 1, 2)

        self.settingLayout.addWidget(self.map_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.use_element_dir_checkbox, 3, 0, 1, 3)

        self.settingLayout.addWidget(self.default_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.apply_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 5, 1, 1, 2)

        self.settingLayout.addWidget(self.fm_checkbox, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.afm_checkbox, 6, 1, 1, 1)
        self.settingLayout.addWidget(self.afm_mode_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.afm_mode_combo, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.kvec_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.kvec_combo, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.group_a_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.group_a_edit, 9, 1, 1, 2)
        self.settingLayout.addWidget(self.group_b_label, 10, 0, 1, 1)
        self.settingLayout.addWidget(self.group_b_edit, 10, 1, 1, 2)
        self.settingLayout.addWidget(self.zero_unknown_groups_checkbox, 11, 0, 1, 2)

        self.settingLayout.addWidget(self.pm_checkbox, 12, 0, 1, 2)
        self.settingLayout.addWidget(self.pm_count_label, 13, 0, 1, 1)
        self.settingLayout.addWidget(self.pm_count_frame, 13, 1, 1, 2)
        self.settingLayout.addWidget(self.pm_direction_label, 14, 0, 1, 1)
        self.settingLayout.addWidget(self.pm_direction_combo, 14, 1, 1, 2)
        self.settingLayout.addWidget(self.pm_cone_label, 15, 0, 1, 1)
        self.settingLayout.addWidget(self.pm_cone_frame, 15, 1, 1, 2)
        self.settingLayout.addWidget(self.pm_balanced_checkbox, 16, 0, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 17, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 17, 1, 1, 2)

        self.afm_mode_combo.currentTextChanged.connect(self._update_afm_mode_widgets)
        self.afm_checkbox.stateChanged.connect(lambda _state: self._update_afm_mode_widgets())
        self._update_afm_mode_widgets()

    def _update_afm_mode_widgets(self):
        """Show/hide AFM controls based on the selected AFM mode."""
        afm_enabled = self.afm_checkbox.isChecked()
        mode = self.afm_mode_combo.currentText()

        use_group = mode == "group A/B"

        self.kvec_label.setVisible(not use_group)
        self.kvec_combo.setVisible(not use_group)
        self.kvec_label.setEnabled(afm_enabled and (not use_group))
        self.kvec_combo.setEnabled(afm_enabled and (not use_group))

        self.group_a_label.setVisible(use_group)
        self.group_a_edit.setVisible(use_group)
        self.group_b_label.setVisible(use_group)
        self.group_b_edit.setVisible(use_group)
        self.zero_unknown_groups_checkbox.setVisible(use_group)

        self.group_a_label.setEnabled(afm_enabled and use_group)
        self.group_a_edit.setEnabled(afm_enabled and use_group)
        self.group_b_label.setEnabled(afm_enabled and use_group)
        self.group_b_edit.setEnabled(afm_enabled and use_group)
        self.zero_unknown_groups_checkbox.setEnabled(afm_enabled and use_group)

    @staticmethod
    def _parse_elements(text: str) -> set[str]:
        tokens = [t.strip() for t in (text or "").replace(";", ",").split(",") if t.strip()]
        return {t[0].upper() + t[1:].lower() for t in tokens}

    @staticmethod
    def _parse_kvec(text: str) -> tuple[int, int, int]:
        text = (text or "").strip()
        if text in {"100", "010", "001", "110", "111"}:
            return tuple(int(c) for c in text)  # type: ignore[return-value]
        return (1, 1, 1)

    def _axis(self) -> np.ndarray:
        x, y, z = [float(v) for v in self.axis_frame.get_input_value()]
        return normalize_vector(np.array([x, y, z], dtype=float))

    def _moment_map(self) -> dict[str, float | np.ndarray]:
        magmom_map: dict[str, float | np.ndarray] = {}
        try:
            magmom_map = parse_magmom_map_any(self.map_edit.text())
        except Exception as exc:  # noqa: BLE001
            MessageManager.send_warning_message(f"MagneticOrder: invalid magmom map: {exc}")
        return magmom_map

    def _per_atom_mags_and_dirs(self, structure) -> tuple[np.ndarray, np.ndarray]:
        magmom_map = self._moment_map()
        default_m = float(self.default_frame.get_input_value()[0])

        axis = self._axis()
        use_elem_dir = self.use_element_dir_checkbox.isChecked()

        mags = np.zeros(len(structure), dtype=float)
        dirs = np.repeat(axis[None, :], len(structure), axis=0)
        symbols = structure.get_chemical_symbols()
        for i, sym in enumerate(symbols):
            val = magmom_map.get(sym, default_m)
            if isinstance(val, np.ndarray):
                mags[i] = float(np.linalg.norm(val))
                if use_elem_dir and mags[i] > 0:
                    dirs[i] = normalize_vector(val)
            else:
                mags[i] = abs(float(val))

        only = self._parse_elements(self.apply_edit.text())
        if only:
            mask = np.array([sym in only for sym in symbols], dtype=bool)
            mags = np.where(mask, mags, 0.0)
        return mags, dirs

    def _make_collinear(self, structure, *, signs: np.ndarray) -> np.ndarray:
        mags, _dirs = self._per_atom_mags_and_dirs(structure)
        signs = np.asarray(signs, dtype=float).reshape(-1)
        if signs.shape[0] != mags.shape[0]:
            raise ValueError("signs shape mismatch")
        return signs * mags

    def _make_noncollinear_axis(self, structure, *, signs: np.ndarray) -> np.ndarray:
        mags, dirs = self._per_atom_mags_and_dirs(structure)
        signs = np.asarray(signs, dtype=float).reshape(-1)
        if signs.shape[0] != mags.shape[0]:
            raise ValueError("signs shape mismatch")
        return (signs[:, None] * mags[:, None]) * dirs

    def _attach_metadata(self, atoms, *, order: str):  # noqa: ARG002
        # Intentionally do not store extra `atoms.info[...]` metadata.
        # Card configuration is persisted separately; Config_type tags are the user-facing hint.
        return

    @staticmethod
    def _axis_tag(axis: np.ndarray) -> str:
        """Return an EXTXYZ-friendly axis tag without quotes/spaces (e.g. 001 or 0,0,1)."""
        v = np.asarray(axis, dtype=float).reshape(3)
        # Prefer compact cardinal encoding when aligned with axes.
        basis = [
            (np.array([1.0, 0.0, 0.0]), "100"),
            (np.array([0.0, 1.0, 0.0]), "010"),
            (np.array([0.0, 0.0, 1.0]), "001"),
        ]
        for b, tag in basis:
            if np.allclose(v, b, atol=1e-8, rtol=0.0):
                return tag
            if np.allclose(v, -b, atol=1e-8, rtol=0.0):
                return f"-{tag}"
        return f"{v[0]:.6g},{v[1]:.6g},{v[2]:.6g}"

    def process_structure(self, structure):
        outputs = []

        do_fm = self.fm_checkbox.isChecked()
        do_afm = self.afm_checkbox.isChecked()
        do_pm = self.pm_checkbox.isChecked()
        if not (do_fm or do_afm or do_pm):
            return [structure]

        base_seed = int(self.seed_frame.get_input_value()[0]) if self.seed_checkbox.isChecked() else None
        cfg_id = stable_config_id(structure)
        noncollinear = self.format_combo.currentText().startswith("Non")

        if do_fm:
            signs = np.ones(len(structure), dtype=float)
            moms = self._make_noncollinear_axis(structure, signs=signs) if noncollinear else self._make_collinear(structure, signs=signs)
            atoms = structure.copy()
            if "initial_magmoms" in atoms.arrays and np.asarray(atoms.arrays["initial_magmoms"]).shape != np.asarray(moms).shape:
                del atoms.arrays["initial_magmoms"]
            atoms.set_initial_magnetic_moments(moms)
            self._attach_metadata(atoms, order="FM")
            append_config_tag(atoms, "MagFMnc" if noncollinear else "MagFM")
            outputs.append(atoms)

        if do_afm:
            if self.afm_mode_combo.currentText() == "group A/B" and "group" in structure.arrays:
                gA = (self.group_a_edit.text() or "A").strip()
                gB = (self.group_b_edit.text() or "B").strip()
                grp = np.asarray(structure.arrays["group"])
                grp = np.array([str(g) for g in grp], dtype=object)
                signs = np.zeros(len(structure), dtype=float)
                signs[grp == gA] = 1.0
                signs[grp == gB] = -1.0
                if not self.zero_unknown_groups_checkbox.isChecked():
                    signs[(grp != gA) & (grp != gB)] = 1.0
            else:
                if self.afm_mode_combo.currentText() == "group A/B" and "group" not in structure.arrays:
                    MessageManager.send_warning_message("MagneticOrder: AFM mode 'group A/B' requires arrays['group']; falling back to k-vector.")
                k = self._parse_kvec(self.kvec_combo.currentText())
                signs = kvec_signs(structure, k)
            moms = self._make_noncollinear_axis(structure, signs=signs) if noncollinear else self._make_collinear(structure, signs=signs)
            atoms = structure.copy()
            if "initial_magmoms" in atoms.arrays and np.asarray(atoms.arrays["initial_magmoms"]).shape != np.asarray(moms).shape:
                del atoms.arrays["initial_magmoms"]
            atoms.set_initial_magnetic_moments(moms)
            if self.afm_mode_combo.currentText() == "k-vector":
                k = self._parse_kvec(self.kvec_combo.currentText())
                self._attach_metadata(atoms, order=f"AFM{k[0]}{k[1]}{k[2]}")
                base = f"MagAFM{k[0]}{k[1]}{k[2]}"
                append_config_tag(atoms, base + ("nc" if noncollinear else ""))
            else:
                gA = sanitize_config_tag(self.group_a_edit.text() or "")
                gB = sanitize_config_tag(self.group_b_edit.text() or "")
                self._attach_metadata(atoms, order="AFM_group")
                append_config_tag(atoms, "MagAFMg" + ("nc" if noncollinear else ""))
            outputs.append(atoms)

        if do_pm:
            pm_n = int(self.pm_count_frame.get_input_value()[0])
            balanced = self.pm_balanced_checkbox.isChecked()
            direction_mode = self.pm_direction_combo.currentText()
            cone_angle = float(self.pm_cone_frame.get_input_value()[0])
            for i in range(max(pm_n, 1)):
                if base_seed is None:
                    rng = np.random.default_rng()
                    seed_note = ""
                else:
                    derived_seed = int(base_seed + cfg_id * 1000003 + i)
                    rng = np.random.default_rng(derived_seed)
                    seed_note = f"s{derived_seed}"

                if noncollinear:
                    mags, _dirs = self._per_atom_mags_and_dirs(structure)
                    moms = random_vector_moments(
                        mags,
                        rng=rng,
                        direction_mode=direction_mode,
                        axis=self._axis(),
                        max_angle_deg=cone_angle,
                        balanced=balanced,
                    )
                else:
                    signs = random_signs(len(structure), rng=rng, balanced=balanced)
                    moms = self._make_collinear(structure, signs=signs)
                atoms = structure.copy()
                if "initial_magmoms" in atoms.arrays and np.asarray(atoms.arrays["initial_magmoms"]).shape != np.asarray(moms).shape:
                    del atoms.arrays["initial_magmoms"]
                atoms.set_initial_magnetic_moments(moms)
                self._attach_metadata(atoms, order="PM")
                base = "MagPM" + ("nc" if noncollinear else "")
                append_config_tag(atoms, base + (f"_{seed_note}" if seed_note else ""))
                outputs.append(atoms)

        return outputs or [structure]

    def to_dict(self):
        data = super().to_dict()
        data["format"] = self.format_combo.currentText()
        data["axis"] = self.axis_frame.get_input_value()
        data["magmom_map"] = self.map_edit.text()
        data["use_element_dirs"] = self.use_element_dir_checkbox.isChecked()
        data["default_moment"] = self.default_frame.get_input_value()
        data["apply_elements"] = self.apply_edit.text()
        data["gen_fm"] = self.fm_checkbox.isChecked()
        data["gen_afm"] = self.afm_checkbox.isChecked()
        data["afm_mode"] = self.afm_mode_combo.currentText()
        data["afm_kvec"] = self.kvec_combo.currentText()
        data["afm_group_a"] = self.group_a_edit.text()
        data["afm_group_b"] = self.group_b_edit.text()
        data["afm_zero_unknown"] = self.zero_unknown_groups_checkbox.isChecked()
        data["gen_pm"] = self.pm_checkbox.isChecked()
        data["pm_count"] = self.pm_count_frame.get_input_value()
        data["pm_direction"] = self.pm_direction_combo.currentText()
        data["pm_cone_angle"] = self.pm_cone_frame.get_input_value()
        data["pm_balanced"] = self.pm_balanced_checkbox.isChecked()
        data["use_seed"] = self.seed_checkbox.isChecked()
        data["seed"] = self.seed_frame.get_input_value()
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.format_combo.setCurrentText(data_dict.get("format", "Collinear (scalar)"))
        self.axis_frame.set_input_value(data_dict.get("axis", [0.0, 0.0, 1.0]))
        self.map_edit.setText(data_dict.get("magmom_map", ""))
        self.use_element_dir_checkbox.setChecked(bool(data_dict.get("use_element_dirs", False)))
        self.default_frame.set_input_value(data_dict.get("default_moment", [0.0]))
        self.apply_edit.setText(data_dict.get("apply_elements", ""))
        self.fm_checkbox.setChecked(bool(data_dict.get("gen_fm", True)))
        self.afm_checkbox.setChecked(bool(data_dict.get("gen_afm", True)))
        self.afm_mode_combo.setCurrentText(data_dict.get("afm_mode", "k-vector"))
        self.kvec_combo.setCurrentText(data_dict.get("afm_kvec", "111"))
        self.group_a_edit.setText(data_dict.get("afm_group_a", "A"))
        self.group_b_edit.setText(data_dict.get("afm_group_b", "B"))
        self.zero_unknown_groups_checkbox.setChecked(bool(data_dict.get("afm_zero_unknown", True)))
        self.pm_checkbox.setChecked(bool(data_dict.get("gen_pm", False)))
        self.pm_count_frame.set_input_value(data_dict.get("pm_count", [10]))
        self.pm_direction_combo.setCurrentText(data_dict.get("pm_direction", "sphere"))
        self.pm_cone_frame.set_input_value(data_dict.get("pm_cone_angle", [30.0]))
        self.pm_balanced_checkbox.setChecked(bool(data_dict.get("pm_balanced", True)))
        self.seed_checkbox.setChecked(bool(data_dict.get("use_seed", False)))
        self.seed_frame.set_input_value(data_dict.get("seed", [0]))
