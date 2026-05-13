"""Card for generating helical and conical spin-spiral initial states."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.cards.magnetism import (
    SpinSpiralOperation,
    SpinSpiralParams,
    coerce_scan_triplet,
    suggest_supercell_multipliers,
)
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class SpinSpiralCard(MakeDataCard):
    """Assign non-collinear spiral magnetic moments using a 1D phase field."""

    group = "Magnetism"
    card_name = "Spin Spiral"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]
    _coerce_scan_triplet = staticmethod(coerce_scan_triplet)
    _suggest_supercell_multipliers = staticmethod(suggest_supercell_multipliers)

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

    def process_structure(self, structure):
        try:
            return self.create_operation().run_structure(structure, self.get_params())
        except Exception as exc:  # noqa: BLE001
            MessageManager.send_warning_message(f"SpinSpiral: invalid magmom map: {exc}")
            return [structure.copy()]

    def create_operation(self):
        return SpinSpiralOperation()

    def get_params(self) -> SpinSpiralParams:
        return SpinSpiralParams(
            axis=self.axis_frame.get_input_value(),
            spiral_parameter_mode=self.parameter_mode_combo.currentText(),
            period_range=self.period_frame.get_input_value(),
            angle_gradient_range=self.angle_gradient_frame.get_input_value(),
            phase_range=self.phase_frame.get_input_value(),
            mz=self.mz_frame.get_input_value(),
            chirality=self.chirality_combo.currentText(),
            phase_mode=self.phase_mode_combo.currentText(),
            layer_tolerance=float(self.layer_tol_frame.get_input_value()[0]),
            only_commensurate_periods=self.commensurate_checkbox.isChecked(),
            magnitude_source=self.source_combo.currentText(),
            magmom_map=self.map_edit.text(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            apply_elements=self.apply_edit.text(),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
        )

    def set_params(self, params: SpinSpiralParams) -> None:
        self.axis_frame.set_input_value([float(v) for v in params.axis])
        self.parameter_mode_combo.setCurrentText(params.spiral_parameter_mode)
        self.period_frame.set_input_value([float(v) for v in params.period_range])
        self.angle_gradient_frame.set_input_value([float(v) for v in params.angle_gradient_range])
        self.phase_frame.set_input_value([float(v) for v in params.phase_range])
        self.mz_frame.set_input_value([float(v) for v in params.mz])
        self.chirality_combo.setCurrentText(params.chirality)
        self.phase_mode_combo.setCurrentText(params.phase_mode)
        self.layer_tol_frame.set_input_value([float(params.layer_tolerance)])
        self.commensurate_checkbox.setChecked(bool(params.only_commensurate_periods))
        self.source_combo.setCurrentText(params.magnitude_source)
        self.map_edit.setText(params.magmom_map)
        self.default_frame.set_input_value([float(params.default_moment)])
        self.apply_edit.setText(params.apply_elements)
        self.max_output_frame.set_input_value([int(params.max_outputs)])
        self._update_magnitude_source_widgets()
        self._update_parameter_mode_widgets()
        self._update_phase_mode_widgets()

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = SpinSpiralParams(**raw_params)
        else:
            params = SpinSpiralParams(
                axis=data_dict.get("axis", [0.0, 0.0, 1.0]),
                spiral_parameter_mode=data_dict.get("spiral_parameter_mode", "Period (L_D)"),
                period_range=data_dict.get("period_range", [20.0, 40.0, 10.0]),
                angle_gradient_range=data_dict.get("angle_gradient_range", [18.0, 18.0, 1.0]),
                phase_range=data_dict.get("phase_range", [0.0, 0.0, 15.0]),
                mz=self._coerce_scan_triplet(data_dict.get("mz", [0.0, 0.0, 0.1]), default_step=0.1),
                chirality=data_dict.get("chirality", "Both"),
                phase_mode=data_dict.get("phase_mode", "Continuous by position"),
                layer_tolerance=data_dict.get("layer_tolerance", [0.05])[0],
                only_commensurate_periods=data_dict.get("only_commensurate_periods", False),
                magnitude_source=data_dict.get("magnitude_source", "Existing initial magmoms"),
                magmom_map=data_dict.get("magmom_map", ""),
                default_moment=data_dict.get("default_moment", [0.0])[0],
                apply_elements=data_dict.get("apply_elements", ""),
                max_outputs=data_dict.get("max_outputs", [100])[0],
            )
        self.set_params(params)
