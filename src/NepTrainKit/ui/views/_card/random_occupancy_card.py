"""Card for assigning global alloy occupancies from a target composition."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition, CheckBox

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.cards.alloy import RandomOccupancyOperation, RandomOccupancyParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class RandomOccupancyCard(MakeDataCard):
    """Assign alloy elements to all (or grouped) lattice sites using a target composition."""

    group = "Alloy"
    card_name = "Random Occupancy"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Random Occupancy Assignment")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("random_occupancy_card_widget")

        self.source_label = BodyLabel("Composition", self.setting_widget)
        self.source_combo = ComboBox(self.setting_widget)
        self.source_combo.addItems(["Auto (Comp tag)", "Manual"])
        self.source_label.setToolTip("Auto reads Comp(...) from Config_type")
        self.source_label.installEventFilter(ToolTipFilter(self.source_label, 300, ToolTipPosition.TOP))

        self.manual_label = BodyLabel("Manual comp", self.setting_widget)
        self.manual_edit = LineEdit(self.setting_widget)
        self.manual_edit.setPlaceholderText("Co:0.33,Cr:0.33,Ni:0.34")
        self.manual_label.setToolTip("Element fractions. Used when 'Manual' is selected or Config_type lacks Comp(...).")
        self.manual_label.installEventFilter(ToolTipFilter(self.manual_label, 300, ToolTipPosition.TOP))

        self.mode_label = BodyLabel("Mode", self.setting_widget)
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItems(["Exact", "Random"])
        self.mode_label.setToolTip("Exact: integer counts match fractions; Random: multinomial sampling")
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))

        self.samples_label = BodyLabel("Structures/input", self.setting_widget)
        self.samples_frame = SpinBoxUnitInputFrame(self)
        self.samples_frame.set_input("unit", 1, "int")
        self.samples_frame.setRange(1, 999999)
        self.samples_frame.set_input_value([1])
        self.samples_label.setToolTip("Number of occupancy samples generated from each input structure")
        self.samples_label.installEventFilter(ToolTipFilter(self.samples_label, 300, ToolTipPosition.TOP))

        self.group_label = BodyLabel("Group filter", self.setting_widget)
        self.group_edit = LineEdit(self.setting_widget)
        self.group_edit.setPlaceholderText("Optional: a,b,c")
        self.group_label.setToolTip("If the structure has arrays['group'], restrict assignment to these groups")
        self.group_label.installEventFilter(ToolTipFilter(self.group_label, 300, ToolTipPosition.TOP))

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.settingLayout.addWidget(self.source_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.source_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.manual_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.manual_edit, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.mode_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.samples_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.samples_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.group_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.group_edit, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 5, 1, 1, 2)

    def create_operation(self):
        """Return the UI-independent random occupancy operation."""
        return RandomOccupancyOperation()

    def get_params(self) -> RandomOccupancyParams:
        """Read random occupancy parameters from UI controls."""
        return RandomOccupancyParams(
            source=self.source_combo.currentText(),
            manual=self.manual_edit.text(),
            mode=self.mode_combo.currentText(),
            samples=int(self.samples_frame.get_input_value()[0]),
            group_filter=self.group_edit.text(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: RandomOccupancyParams) -> None:
        """Apply random occupancy parameters to UI controls."""
        self.source_combo.setCurrentText(params.source)
        self.manual_edit.setText(params.manual)
        self.mode_combo.setCurrentText(params.mode)
        self.samples_frame.set_input_value([int(params.samples)])
        self.group_edit.setText(params.group_filter)
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])

    def process_structure(self, structure):
        """Assign occupancy from UI-independent parameters."""
        try:
            result = self.create_operation().run_structure(structure, self.get_params())
        except Exception as exc:  # noqa: BLE001
            MessageManager.send_warning_message(f"RandomOccupancy: invalid composition: {exc}")
            return [structure]
        if len(result) == 1 and result[0] is structure:
            MessageManager.send_warning_message("RandomOccupancy: missing composition (Config_type Comp tag or manual input).")
        return result

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = RandomOccupancyParams(
                source=raw_params.get("source", "Auto (Comp tag)"),
                manual=raw_params.get("manual", ""),
                mode=raw_params.get("mode", "Exact"),
                samples=raw_params.get("samples", 1),
                group_filter=raw_params.get("group_filter", ""),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
            )
        else:
            params = RandomOccupancyParams(
                source=data_dict.get("source", "Auto (Comp tag)"),
                manual=data_dict.get("manual", ""),
                mode=data_dict.get("mode", "Exact"),
                samples=data_dict.get("samples", [1])[0],
                group_filter=data_dict.get("group_filter", ""),
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
