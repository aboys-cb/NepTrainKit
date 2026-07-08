"""Card for Bain/tetragonal distortion paths."""

from qfluentwidgets import BodyLabel, CheckBox, ComboBox, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import BainPathOperation, BainPathParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class BainPathCard(MakeDataCard):
    """Generate fixed-structure Bain/tetragonal distortion paths."""

    group = "Lattice"
    card_name = "Bain Path"
    menu_icon = r":/images/src/images/scaling.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Make Bain Path")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("bain_path_card_widget")

        self.axis_label = BodyLabel("c axis", self.setting_widget)
        self.axis_combo = ComboBox(self.setting_widget)
        self.axis_combo.addItems(["x", "y", "z"])
        self.axis_combo.setCurrentText("z")

        self.ca_label = BodyLabel("c/a scale", self.setting_widget)
        self.ca_frame = SpinBoxUnitInputFrame(self)
        self.ca_frame.set_input(["-", "step", ""], 3, "float")
        self.ca_frame.setDecimals(4)
        self.ca_frame.setRange(0.0001, 100.0)
        self.ca_frame.set_input_value([1.0, 1.0, 1.0])
        self.ca_label.setToolTip("Relative c/a scale r: c *= r; constant-volume modes compensate perpendicular axes.")
        self.ca_label.installEventFilter(ToolTipFilter(self.ca_label, 300, ToolTipPosition.TOP))

        self.mode_label = BodyLabel("Mode", self.setting_widget)
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItems(["constant_volume", "scale_volume", "free_c"])

        self.volume_label = BodyLabel("V scale", self.setting_widget)
        self.volume_frame = SpinBoxUnitInputFrame(self)
        self.volume_frame.set_input(["-", "step", ""], 3, "float")
        self.volume_frame.setDecimals(4)
        self.volume_frame.setRange(0.0001, 100.0)
        self.volume_frame.set_input_value([1.0, 1.0, 1.0])

        self.scale_atoms_checkbox = CheckBox("Scale atoms", self.setting_widget)
        self.scale_atoms_checkbox.setChecked(True)

        self.settingLayout.addWidget(self.axis_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.ca_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.ca_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.mode_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.volume_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.volume_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.scale_atoms_checkbox, 4, 0, 1, 2)

    def create_operation(self):
        return BainPathOperation()

    def get_params(self) -> BainPathParams:
        return BainPathParams(
            axis=self.axis_combo.currentText(),
            ca_range=tuple(float(v) for v in self.ca_frame.get_input_value()),
            mode=self.mode_combo.currentText(),
            volume_scale_range=tuple(float(v) for v in self.volume_frame.get_input_value()),
            scale_atoms=self.scale_atoms_checkbox.isChecked(),
        )

    def set_params(self, params: BainPathParams) -> None:
        self.axis_combo.setCurrentText(params.axis)
        self.ca_frame.set_input_value(list(params.ca_range))
        self.mode_combo.setCurrentText(params.mode)
        self.volume_frame.set_input_value(list(params.volume_scale_range))
        self.scale_atoms_checkbox.setChecked(bool(params.scale_atoms))

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw = data_dict.get("params", {})
        self.set_params(BainPathParams(**raw))
