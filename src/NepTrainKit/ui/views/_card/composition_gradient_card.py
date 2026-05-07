"""Card for applying composition gradients along a structure axis."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.alloy import CompositionGradientOperation, CompositionGradientParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class CompositionGradientCard(MakeDataCard):
    """Assign atom types from a layerwise composition gradient."""

    group = "Alloy"
    card_name = "Composition Gradient"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Composition Gradient")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("composition_gradient_card_widget")

        self.elements_label = BodyLabel("Elements", self.setting_widget)
        self.elements_label.setToolTip("Elements participating in the gradient")
        self.elements_label.installEventFilter(ToolTipFilter(self.elements_label, 300, ToolTipPosition.TOP))
        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setText("Ni,Co")

        self.start_label = BodyLabel("Start composition", self.setting_widget)
        self.start_label.setToolTip("Composition at the low-coordinate end, e.g. Ni:1,Co:0")
        self.start_label.installEventFilter(ToolTipFilter(self.start_label, 300, ToolTipPosition.TOP))
        self.start_edit = LineEdit(self.setting_widget)
        self.start_edit.setText("Ni:1,Co:0")

        self.end_label = BodyLabel("End composition", self.setting_widget)
        self.end_label.setToolTip("Composition at the high-coordinate end, e.g. Ni:0,Co:1")
        self.end_label.installEventFilter(ToolTipFilter(self.end_label, 300, ToolTipPosition.TOP))
        self.end_edit = LineEdit(self.setting_widget)
        self.end_edit.setText("Ni:0,Co:1")

        self.axis_label = BodyLabel("Axis", self.setting_widget)
        self.axis_label.setToolTip("Gradient direction")
        self.axis_label.installEventFilter(ToolTipFilter(self.axis_label, 300, ToolTipPosition.TOP))
        self.axis_combo = ComboBox(self.setting_widget)
        self.axis_combo.addItems(["x", "y", "z"])
        self.axis_combo.setCurrentText("x")

        self.bins_label = BodyLabel("Bins", self.setting_widget)
        self.bins_label.setToolTip("Number of coordinate layers used to approximate the gradient")
        self.bins_label.installEventFilter(ToolTipFilter(self.bins_label, 300, ToolTipPosition.TOP))
        self.bins_frame = SpinBoxUnitInputFrame(self)
        self.bins_frame.set_input("unit", 1, "int")
        self.bins_frame.setRange(1, 10000)
        self.bins_frame.set_input_value([8])

        self.target_label = BodyLabel("Target elements", self.setting_widget)
        self.target_label.setToolTip("Optional existing elements eligible for replacement; empty means all atoms")
        self.target_label.installEventFilter(ToolTipFilter(self.target_label, 300, ToolTipPosition.TOP))
        self.target_edit = LineEdit(self.setting_widget)
        self.target_edit.setPlaceholderText("Ni,Co")

        self.samples_label = BodyLabel("Samples", self.setting_widget)
        self.samples_label.setToolTip("Number of random assignments emitted for the same layer compositions")
        self.samples_label.installEventFilter(ToolTipFilter(self.samples_label, 300, ToolTipPosition.TOP))
        self.samples_frame = SpinBoxUnitInputFrame(self)
        self.samples_frame.set_input("unit", 1, "int")
        self.samples_frame.setRange(1, 10000)
        self.samples_frame.set_input_value([1])

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.settingLayout.addWidget(self.elements_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.elements_edit, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.start_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.start_edit, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.end_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.end_edit, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.axis_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_combo, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.bins_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.bins_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.target_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.target_edit, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.samples_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.samples_frame, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 7, 1, 1, 2)

    def create_operation(self):
        return CompositionGradientOperation()

    def get_params(self) -> CompositionGradientParams:
        return CompositionGradientParams(
            elements=self.elements_edit.text(),
            start_composition=self.start_edit.text(),
            end_composition=self.end_edit.text(),
            axis=self.axis_combo.currentText(),
            bins=int(self.bins_frame.get_input_value()[0]),
            target_elements=self.target_edit.text(),
            samples=int(self.samples_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: CompositionGradientParams) -> None:
        self.elements_edit.setText(params.elements)
        self.start_edit.setText(params.start_composition)
        self.end_edit.setText(params.end_composition)
        self.axis_combo.setCurrentText(params.axis)
        self.bins_frame.set_input_value([int(params.bins)])
        self.target_edit.setText(params.target_elements)
        self.samples_frame.set_input_value([int(params.samples)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        params = CompositionGradientParams(**raw_params) if raw_params else CompositionGradientParams()
        self.set_params(params)
