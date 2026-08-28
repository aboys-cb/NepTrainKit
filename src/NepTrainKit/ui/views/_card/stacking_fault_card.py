# -*- coding: utf-8 -*-

from qfluentwidgets import BodyLabel, CaptionLabel, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import StackingFaultOperation, StackingFaultParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class StackingFaultCard(MakeDataCard):
    """Load and reproduce the legacy automatic-direction layer-shift operation."""

    group = "Defect"
    card_name = "Stacking Fault"
    discoverable = False
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Legacy Stacking Fault"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("stacking_fault_card_widget")
        self.warning_label = CaptionLabel(
            self.tr(
                "Compatibility card for existing workflows. It shifts one side of a projected cut along an automatically chosen Cartesian direction; use GSFE Path to specify the physical slip direction."
            ),
            self.setting_widget,
        )
        self.warning_label.setWordWrap(True)

        self.hkl_label = BodyLabel(self.tr("Approximate plane h k l"), self.setting_widget)
        self.hkl_frame = SpinBoxUnitInputFrame(self)
        self.hkl_frame.set_input("", 3, "int")
        self.hkl_frame.setRange(-10, 10)
        self.hkl_frame.set_input_value([1, 1, 1])
        self.hkl_label.setToolTip(self.tr("Enter Miller indices (h k l): Used to define the stacking fault plane."))
        self.hkl_label.installEventFilter(ToolTipFilter(self.hkl_label, 0, ToolTipPosition.TOP))

        self.step_label = BodyLabel(self.tr("Displacement"), self.setting_widget)
        self.step_frame = SpinBoxUnitInputFrame(self)
        self.step_frame.set_input(["-", "step", "Å"], 3, "float")
        self.step_frame.setDecimals(4)
        self.step_frame.setSingleStep(0.1)
        self.step_frame.setRange(-10, 10)
        self.step_frame.set_input_value([0.0, 1.0, 0.5])
        self.step_label.setToolTip(
            self.tr("Displacement start, end, and step in Å along a deterministic in-plane slip direction.")
        )
        self.step_label.installEventFilter(ToolTipFilter(self.step_label, 0, ToolTipPosition.TOP))

        self.layer_label = BodyLabel(self.tr("Legacy projected-layer rank"), self.setting_widget)
        self.layer_frame = SpinBoxUnitInputFrame(self)
        self.layer_frame.set_input("", 1, "int")
        self.layer_frame.setRange(1, 100)
        self.layer_frame.set_input_value([1])
        self.layer_label.setToolTip(
            self.tr(
                "Selects a threshold in sorted projected coordinates; rank 1 can move every atom, and out-of-range values fall back to the middle"
            )
        )
        self.layer_label.installEventFilter(ToolTipFilter(self.layer_label, 0, ToolTipPosition.TOP))

        self.settingLayout.addWidget(self.warning_label, 0, 0, 1, 3)
        self.settingLayout.addWidget(self.hkl_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.hkl_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.layer_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.layer_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.step_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.step_frame, 3, 1, 1, 2)

    def create_operation(self):
        """Return the UI-independent stacking-fault operation."""
        return StackingFaultOperation()

    def get_params(self) -> StackingFaultParams:
        """Read stacking-fault parameters from UI controls."""
        return StackingFaultParams(
            hkl=tuple(int(v) for v in self.hkl_frame.get_input_value()),
            step=tuple(float(v) for v in self.step_frame.get_input_value()),
            layers=int(self.layer_frame.get_input_value()[0]),
        )

    def set_params(self, params: StackingFaultParams) -> None:
        """Apply stacking-fault parameters to UI controls."""
        self.hkl_frame.set_input_value([int(v) for v in params.hkl])
        self.step_frame.set_input_value([float(v) for v in params.step])
        self.layer_frame.set_input_value([int(params.layers)])

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = StackingFaultParams(
                hkl=raw_params.get("hkl", [1, 1, 1]),
                step=raw_params.get("step", [0.0, 1.0, 0.5]),
                layers=raw_params.get("layers", 1),
            )
        else:
            params = StackingFaultParams(
                hkl=data_dict.get("hkl", [1, 1, 1]),
                step=data_dict.get("step", [0.0, 1.0, 0.5]),
                layers=data_dict.get("layers", [1])[0],
            )
        self.set_params(params)
