"""Card for setting or normalising initial magnetic moments."""

from __future__ import annotations

import numpy as np
from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.core.magnetism import (
    existing_moment_scalars,
    existing_moment_vectors,
    mapped_moment_magnitudes,
    mapped_moment_vectors,
    normalize_vector,
    parse_element_set,
    parse_magmom_map_any,
    set_initial_magmoms_safe,
)
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class SetMagneticMomentsCard(MakeDataCard):
    """Set or convert magnetic moments into a consistent scalar/vector representation."""

    group = "Magnetism"
    card_name = "Set Magnetic Moments"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Set Magnetic Moments")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("set_magnetic_moments_card_widget")

        self.source_label = BodyLabel("Source", self.setting_widget)
        self.source_label.setToolTip("Choose whether to reuse existing magmoms, element map/default values, or a constant moment")
        self.source_label.installEventFilter(ToolTipFilter(self.source_label, 300, ToolTipPosition.TOP))
        self.source_combo = ComboBox(self.setting_widget)
        self.source_combo.addItems(["Existing initial magmoms", "Map/default magnitude", "Constant magnitude"])
        self.source_combo.setCurrentText("Map/default magnitude")

        self.format_label = BodyLabel("Format", self.setting_widget)
        self.format_label.setToolTip("Collinear writes scalar MAGMOM; Non-collinear writes vector MAGMOM")
        self.format_label.installEventFilter(ToolTipFilter(self.format_label, 300, ToolTipPosition.TOP))
        self.format_combo = ComboBox(self.setting_widget)
        self.format_combo.addItems(["Collinear (scalar)", "Non-collinear (vector)"])
        self.format_combo.setCurrentText("Non-collinear (vector)")

        self.axis_label = BodyLabel("Axis (x,y,z)", self.setting_widget)
        self.axis_label.setToolTip("Reference axis used for vector output and scalar-to-vector lifting")
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setRange(-1.0, 1.0)
        for obj in self.axis_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.map_label = BodyLabel("Magmom map", self.setting_widget)
        self.map_label.setToolTip('Per-element moments, e.g. "Fe:2.2,Co:1.7" or JSON such as {"Cr":[0,0,1.0]}')
        self.map_label.installEventFilter(ToolTipFilter(self.map_label, 300, ToolTipPosition.TOP))
        self.map_edit = LineEdit(self.setting_widget)
        self.map_edit.setPlaceholderText("Fe:2.2,Co:1.7")

        self.use_element_dir_checkbox = CheckBox("Use element vector directions", self.setting_widget)
        self.use_element_dir_checkbox.setChecked(False)
        self.use_element_dir_checkbox.setToolTip("If the map provides vectors, preserve their directions in vector output")
        self.use_element_dir_checkbox.installEventFilter(
            ToolTipFilter(self.use_element_dir_checkbox, 300, ToolTipPosition.TOP)
        )

        self.default_label = BodyLabel("Default |m|", self.setting_widget)
        self.default_label.setToolTip("Magnitude for elements not listed in Magmom map")
        self.default_label.installEventFilter(ToolTipFilter(self.default_label, 300, ToolTipPosition.TOP))
        self.default_frame = SpinBoxUnitInputFrame(self)
        self.default_frame.set_input("", 1, "float")
        self.default_frame.setRange(0.0, 20.0)
        self.default_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.default_frame.set_input_value([0.0])

        self.constant_label = BodyLabel("Constant |m|", self.setting_widget)
        self.constant_label.setToolTip("Uniform magnitude used when Source = Constant magnitude")
        self.constant_label.installEventFilter(ToolTipFilter(self.constant_label, 300, ToolTipPosition.TOP))
        self.constant_frame = SpinBoxUnitInputFrame(self)
        self.constant_frame.set_input("", 1, "float")
        self.constant_frame.setRange(0.0, 20.0)
        self.constant_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.constant_frame.set_input_value([2.0])

        self.lift_scalar_checkbox = CheckBox("Lift scalar magmoms to vectors", self.setting_widget)
        self.lift_scalar_checkbox.setChecked(True)
        self.lift_scalar_checkbox.setToolTip("When Source = Existing initial magmoms, lift scalar input onto Axis for vector output")
        self.lift_scalar_checkbox.installEventFilter(
            ToolTipFilter(self.lift_scalar_checkbox, 300, ToolTipPosition.TOP)
        )

        self.apply_label = BodyLabel("Apply elements", self.setting_widget)
        self.apply_label.setToolTip("Optional comma-separated element list; empty means all atoms")
        self.apply_label.installEventFilter(ToolTipFilter(self.apply_label, 300, ToolTipPosition.TOP))
        self.apply_edit = LineEdit(self.setting_widget)
        self.apply_edit.setPlaceholderText("Fe,Co,Ni")

        self.settingLayout.addWidget(self.source_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.source_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.format_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.format_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.axis_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.map_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.use_element_dir_checkbox, 4, 0, 1, 3)
        self.settingLayout.addWidget(self.default_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.constant_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.constant_frame, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.lift_scalar_checkbox, 7, 0, 1, 3)
        self.settingLayout.addWidget(self.apply_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 8, 1, 1, 2)

        self.source_combo.currentTextChanged.connect(self._update_source_widgets)
        self.format_combo.currentTextChanged.connect(self._update_source_widgets)
        self._update_source_widgets()

    def _update_source_widgets(self):
        source = self.source_combo.currentText()
        vector_output = self.format_combo.currentText() == "Non-collinear (vector)"

        use_map = source == "Map/default magnitude"
        use_constant = source == "Constant magnitude"
        use_existing = source == "Existing initial magmoms"

        self.map_label.setEnabled(use_map)
        self.map_edit.setEnabled(use_map)
        self.map_label.setVisible(use_map)
        self.map_edit.setVisible(use_map)

        self.use_element_dir_checkbox.setEnabled(use_map and vector_output)
        self.use_element_dir_checkbox.setVisible(use_map)

        self.default_label.setEnabled(use_map)
        self.default_frame.setEnabled(use_map)
        self.default_label.setVisible(use_map)
        self.default_frame.setVisible(use_map)

        self.constant_label.setEnabled(use_constant)
        self.constant_frame.setEnabled(use_constant)
        self.constant_label.setVisible(use_constant)
        self.constant_frame.setVisible(use_constant)

        self.lift_scalar_checkbox.setEnabled(use_existing and vector_output)
        self.lift_scalar_checkbox.setVisible(use_existing)

    def _axis(self) -> np.ndarray:
        values = [float(v) for v in self.axis_frame.get_input_value()]
        return normalize_vector(np.array(values, dtype=float))

    def _apply_elements(self) -> set[str]:
        return parse_element_set(self.apply_edit.text())

    def _moment_map(self) -> dict[str, float | np.ndarray]:
        try:
            return parse_magmom_map_any(self.map_edit.text())
        except Exception as exc:  # noqa: BLE001
            MessageManager.send_warning_message(f"SetMagneticMoments: invalid magmom map: {exc}")
            return {}

    def _constant_magnitudes(self, structure) -> np.ndarray:
        magnitude = abs(float(self.constant_frame.get_input_value()[0]))
        mask = np.array(
            [sym in self._apply_elements() for sym in structure.get_chemical_symbols()],
            dtype=bool,
        ) if self._apply_elements() else np.ones(len(structure), dtype=bool)
        values = np.zeros(len(structure), dtype=float)
        values[mask] = magnitude
        return values

    def _scalar_moments(self, structure) -> np.ndarray | None:
        source = self.source_combo.currentText()
        selected = self._apply_elements()
        if source == "Existing initial magmoms":
            values = existing_moment_scalars(structure, axis=self._axis())
            if values is None:
                MessageManager.send_warning_message("SetMagneticMoments: no usable initial_magmoms found.")
                return None
            if selected:
                mask = np.array([sym in selected for sym in structure.get_chemical_symbols()], dtype=bool)
                values = np.where(mask, values, 0.0)
            return values
        if source == "Constant magnitude":
            return self._constant_magnitudes(structure)
        return mapped_moment_magnitudes(
            structure,
            self._moment_map(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            apply_elements=selected,
        )

    def _vector_moments(self, structure) -> np.ndarray | None:
        source = self.source_combo.currentText()
        selected = self._apply_elements()
        axis = self._axis()
        if source == "Existing initial magmoms":
            values = existing_moment_vectors(
                structure,
                axis=axis,
                lift_scalar=self.lift_scalar_checkbox.isChecked(),
            )
            if values is None:
                MessageManager.send_warning_message("SetMagneticMoments: no usable initial_magmoms found.")
                return None
            if selected:
                mask = np.array([sym in selected for sym in structure.get_chemical_symbols()], dtype=bool)
                values = np.where(mask[:, None], values, 0.0)
            return values
        if source == "Constant magnitude":
            return self._constant_magnitudes(structure).reshape(-1, 1) * axis.reshape(1, 3)
        return mapped_moment_vectors(
            structure,
            self._moment_map(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            axis=axis,
            apply_elements=selected,
            use_element_directions=self.use_element_dir_checkbox.isChecked(),
        )

    def process_structure(self, structure):
        vector_output = self.format_combo.currentText() == "Non-collinear (vector)"
        moments = self._vector_moments(structure) if vector_output else self._scalar_moments(structure)
        if moments is None:
            return [structure.copy()]

        atoms = structure.copy()
        set_initial_magmoms_safe(atoms, np.asarray(moments, dtype=float))
        source_map = {
            "Existing initial magmoms": "existing",
            "Map/default magnitude": "map",
            "Constant magnitude": "const",
        }
        fmt_tag = "vec" if vector_output else "sca"
        append_config_tag(atoms, f"MagSet({source_map[self.source_combo.currentText()]},{fmt_tag})")
        return [atoms]

    def to_dict(self):
        data = super().to_dict()
        data["source"] = self.source_combo.currentText()
        data["format"] = self.format_combo.currentText()
        data["axis"] = self.axis_frame.get_input_value()
        data["magmom_map"] = self.map_edit.text()
        data["use_element_dirs"] = self.use_element_dir_checkbox.isChecked()
        data["default_moment"] = self.default_frame.get_input_value()
        data["constant_moment"] = self.constant_frame.get_input_value()
        data["lift_scalar"] = self.lift_scalar_checkbox.isChecked()
        data["apply_elements"] = self.apply_edit.text()
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.source_combo.setCurrentText(data_dict.get("source", "Map/default magnitude"))
        self.format_combo.setCurrentText(data_dict.get("format", "Non-collinear (vector)"))
        self.axis_frame.set_input_value(data_dict.get("axis", [0.0, 0.0, 1.0]))
        self.map_edit.setText(data_dict.get("magmom_map", ""))
        self.use_element_dir_checkbox.setChecked(bool(data_dict.get("use_element_dirs", False)))
        self.default_frame.set_input_value(data_dict.get("default_moment", [0.0]))
        self.constant_frame.set_input_value(data_dict.get("constant_moment", [2.0]))
        self.lift_scalar_checkbox.setChecked(bool(data_dict.get("lift_scalar", True)))
        self.apply_edit.setText(data_dict.get("apply_elements", ""))
        self._update_source_widgets()
