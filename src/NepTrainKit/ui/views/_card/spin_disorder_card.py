"""Card for controlled spin-disorder generation."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.magnetism import SpinDisorderOperation, SpinDisorderParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class SpinDisorderCard(MakeDataCard):
    """Generate spin states with explicit disorder fractions."""

    group = "Magnetism"
    card_name = "Spin Disorder"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Spin Disorder")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("spin_disorder_card_widget")

        self.mode_label = BodyLabel("Disorder mode", self.setting_widget)
        self.mode_label.setToolTip("Flip selected moments, randomize selected directions, or sample within a cone")
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItems(["Flip fraction", "Randomize fraction", "Cone disorder"])
        self.mode_combo.setCurrentText("Flip fraction")

        self.fractions_label = BodyLabel("Fractions", self.setting_widget)
        self.fractions_label.setToolTip("Comma-separated disorder fractions, for example 0.1,0.3,0.5,0.7")
        self.fractions_label.installEventFilter(ToolTipFilter(self.fractions_label, 300, ToolTipPosition.TOP))
        self.fractions_edit = LineEdit(self.setting_widget)
        self.fractions_edit.setText("0.1,0.3,0.5,0.7")

        self.samples_label = BodyLabel("Samples/fraction", self.setting_widget)
        self.samples_label.setToolTip("Number of independent selections emitted for each disorder fraction")
        self.samples_label.installEventFilter(ToolTipFilter(self.samples_label, 300, ToolTipPosition.TOP))
        self.samples_frame = SpinBoxUnitInputFrame(self)
        self.samples_frame.set_input("unit", 1, "int")
        self.samples_frame.setRange(1, 10000)
        self.samples_frame.set_input_value([1])

        self.cone_label = BodyLabel("Cone angle", self.setting_widget)
        self.cone_label.setToolTip("Maximum cone angle in degrees for Cone disorder")
        self.cone_label.installEventFilter(ToolTipFilter(self.cone_label, 300, ToolTipPosition.TOP))
        self.cone_frame = SpinBoxUnitInputFrame(self)
        self.cone_frame.set_input("deg", 1, "float")
        self.cone_frame.setRange(0.0, 180.0)
        self.cone_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.cone_frame.set_input_value([30.0])

        self.source_label = BodyLabel("Magnitude source", self.setting_widget)
        self.source_label.setToolTip("Use existing initial magmoms or build a reference from map/default")
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

        self.axis_label = BodyLabel("Reference axis", self.setting_widget)
        self.axis_label.setToolTip("Axis for lifted scalar magmoms and map/default reference states")
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setRange(-1.0, 1.0)
        for obj in self.axis_frame.object_list:
            obj.setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.apply_label = BodyLabel("Apply elements", self.setting_widget)
        self.apply_label.setToolTip("Optional comma-separated element list; empty means all nonzero moments")
        self.apply_label.installEventFilter(ToolTipFilter(self.apply_label, 300, ToolTipPosition.TOP))
        self.apply_edit = LineEdit(self.setting_widget)
        self.apply_edit.setPlaceholderText("Fe,Co")

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.max_output_label = BodyLabel("Max outputs", self.setting_widget)
        self.max_output_label.setToolTip("Stop after this many generated structures")
        self.max_output_label.installEventFilter(ToolTipFilter(self.max_output_label, 300, ToolTipPosition.TOP))
        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input("unit", 1, "int")
        self.max_output_frame.setRange(1, 999999)
        self.max_output_frame.set_input_value([100])

        self.settingLayout.addWidget(self.mode_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.fractions_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.fractions_edit, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.samples_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.samples_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.cone_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.cone_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.source_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.source_combo, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.map_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.default_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.lift_scalar_checkbox, 7, 0, 1, 3)
        self.settingLayout.addWidget(self.axis_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_frame, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.apply_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 9, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 10, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 10, 1, 1, 2)
        self.settingLayout.addWidget(self.max_output_label, 11, 0, 1, 1)
        self.settingLayout.addWidget(self.max_output_frame, 11, 1, 1, 2)

        self.mode_combo.currentTextChanged.connect(self._update_mode_widgets)
        self.source_combo.currentTextChanged.connect(self._update_source_widgets)
        self._update_mode_widgets()
        self._update_source_widgets()

    def _update_mode_widgets(self):
        show_cone = self.mode_combo.currentText() == "Cone disorder"
        self.cone_label.setVisible(show_cone)
        self.cone_frame.setVisible(show_cone)
        self.cone_label.setEnabled(show_cone)
        self.cone_frame.setEnabled(show_cone)

    def _update_source_widgets(self):
        use_map = self.source_combo.currentText() == "Map/default magnitude"
        for widget in (self.map_label, self.map_edit, self.default_label, self.default_frame):
            widget.setVisible(use_map)
            widget.setEnabled(use_map)

    def create_operation(self):
        return SpinDisorderOperation()

    def get_params(self) -> SpinDisorderParams:
        return SpinDisorderParams(
            mode=self.mode_combo.currentText(),
            fractions=self.fractions_edit.text(),
            samples_per_fraction=int(self.samples_frame.get_input_value()[0]),
            cone_angle=float(self.cone_frame.get_input_value()[0]),
            magnitude_source=self.source_combo.currentText(),
            magmom_map=self.map_edit.text(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            lift_scalar=self.lift_scalar_checkbox.isChecked(),
            axis=self.axis_frame.get_input_value(),
            apply_elements=self.apply_edit.text(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
        )

    def set_params(self, params: SpinDisorderParams) -> None:
        self.mode_combo.setCurrentText(params.mode)
        self.fractions_edit.setText(params.fractions)
        self.samples_frame.set_input_value([int(params.samples_per_fraction)])
        self.cone_frame.set_input_value([float(params.cone_angle)])
        self.source_combo.setCurrentText(params.magnitude_source)
        self.map_edit.setText(params.magmom_map)
        self.default_frame.set_input_value([float(params.default_moment)])
        self.lift_scalar_checkbox.setChecked(bool(params.lift_scalar))
        self.axis_frame.set_input_value([float(v) for v in params.axis])
        self.apply_edit.setText(params.apply_elements)
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.max_output_frame.set_input_value([int(params.max_outputs)])
        self._update_mode_widgets()
        self._update_source_widgets()

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        params = SpinDisorderParams(**raw_params) if raw_params else SpinDisorderParams()
        self.set_params(params)
