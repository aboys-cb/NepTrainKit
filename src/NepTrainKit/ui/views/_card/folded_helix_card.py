"""Card for generating symmetric folded-helix magnetic textures layer by layer."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.cards.magnetism import FoldedHelixOperation, FoldedHelixParams
from NepTrainKit.core.cards.operation import params_to_dict
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

    def create_operation(self):
        return FoldedHelixOperation()

    def get_params(self) -> FoldedHelixParams:
        return FoldedHelixParams(
            layer_axis=self.layer_axis_frame.get_input_value(),
            plane_normal=self.plane_normal_frame.get_input_value(),
            layer_tolerance=float(self.layer_tol_frame.get_input_value()[0]),
            half_period_mode=self.half_period_mode_combo.currentText(),
            half_period_layers=self.half_period_frame.get_input_value(),
            angle_step_range=self.angle_step_frame.get_input_value(),
            phase_range=self.phase_frame.get_input_value(),
            sequence_mode=self.sequence_combo.currentText(),
            magnitude_source=self.source_combo.currentText(),
            magmom_map=self.map_edit.text(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            apply_elements=self.apply_edit.text(),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
        )

    def set_params(self, params: FoldedHelixParams) -> None:
        self.layer_axis_frame.set_input_value([float(v) for v in params.layer_axis])
        self.plane_normal_frame.set_input_value([float(v) for v in params.plane_normal])
        self.layer_tol_frame.set_input_value([float(params.layer_tolerance)])
        self.half_period_mode_combo.setCurrentText(params.half_period_mode)
        self.half_period_frame.set_input_value([int(v) for v in params.half_period_layers])
        self.angle_step_frame.set_input_value([float(v) for v in params.angle_step_range])
        self.phase_frame.set_input_value([float(v) for v in params.phase_range])
        self.sequence_combo.setCurrentText(params.sequence_mode)
        self.source_combo.setCurrentText(params.magnitude_source)
        self.map_edit.setText(params.magmom_map)
        self.default_frame.set_input_value([float(params.default_moment)])
        self.apply_edit.setText(params.apply_elements)
        self.max_output_frame.set_input_value([int(params.max_outputs)])
        self._update_magnitude_source_widgets()
        self._update_half_period_mode_widgets()

    def process_structure(self, structure):
        try:
            return self.create_operation().run_structure(structure, self.get_params())
        except Exception as exc:  # noqa: BLE001
            MessageManager.send_warning_message(f"FoldedHelix: invalid magmom map: {exc}")
            return [structure.copy()]

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = FoldedHelixParams(**raw_params)
        else:
            params = FoldedHelixParams(
                layer_axis=data_dict.get("layer_axis", [0.0, 0.0, 1.0]),
                plane_normal=data_dict.get("plane_normal", [0.0, 0.0, 1.0]),
                layer_tolerance=data_dict.get("layer_tolerance", [0.05])[0],
                half_period_mode=data_dict.get("half_period_mode", "Auto from layer count"),
                half_period_layers=data_dict.get("half_period_layers", [2, 4, 1]),
                angle_step_range=data_dict.get("angle_step_range", [15.0, 45.0, 15.0]),
                phase_range=data_dict.get("phase_range", [0.0, 0.0, 15.0]),
                sequence_mode=data_dict.get("sequence_mode", "Clockwise then counterclockwise"),
                magnitude_source=data_dict.get("magnitude_source", "Existing initial magmoms"),
                magmom_map=data_dict.get("magmom_map", ""),
                default_moment=data_dict.get("default_moment", [0.0])[0],
                apply_elements=data_dict.get("apply_elements", ""),
                max_outputs=data_dict.get("max_outputs", [100])[0],
            )
        self.set_params(params)
