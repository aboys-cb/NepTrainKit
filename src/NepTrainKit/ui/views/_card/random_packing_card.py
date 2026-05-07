"""Card for random atomic packing inside an existing cell."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CheckBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import RandomPackingOperation, RandomPackingParams
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class RandomPackingCard(MakeDataCard):
    """Generate random atomic coordinates while preserving cell constraints."""

    group = "Structure"
    card_name = "Random Packing"
    menu_icon = r":/images/src/images/perturb.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Random Packing")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("random_packing_card_widget")

        self.structures_label = BodyLabel("Structures", self.setting_widget)
        self.structures_label.setToolTip("Number of random packings generated per input structure")
        self.structures_label.installEventFilter(ToolTipFilter(self.structures_label, 300, ToolTipPosition.TOP))
        self.structures_frame = SpinBoxUnitInputFrame(self)
        self.structures_frame.set_input("unit", 1, "int")
        self.structures_frame.setRange(1, 100000)
        self.structures_frame.set_input_value([1])

        self.composition_label = BodyLabel("Composition", self.setting_widget)
        self.composition_label.setToolTip("Empty keeps input atom counts; otherwise use exact counts such as Fe:32,O:64")
        self.composition_label.installEventFilter(ToolTipFilter(self.composition_label, 300, ToolTipPosition.TOP))
        self.composition_edit = LineEdit(self.setting_widget)
        self.composition_edit.setPlaceholderText("Fe:32,O:64")

        self.min_distance_label = BodyLabel("Min distance", self.setting_widget)
        self.min_distance_label.setToolTip("Global minimum interatomic distance in Angstrom")
        self.min_distance_label.installEventFilter(ToolTipFilter(self.min_distance_label, 300, ToolTipPosition.TOP))
        self.min_distance_frame = SpinBoxUnitInputFrame(self)
        self.min_distance_frame.set_input("A", 1, "float")
        self.min_distance_frame.setRange(0.01, 100.0)
        self.min_distance_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_distance_frame.set_input_value([1.5])

        self.pair_distance_label = BodyLabel("Pair distances", self.setting_widget)
        self.pair_distance_label.setToolTip("Optional pair-specific overrides, for example Fe-O:1.8,O-O:1.2")
        self.pair_distance_label.installEventFilter(ToolTipFilter(self.pair_distance_label, 300, ToolTipPosition.TOP))
        self.pair_distance_edit = LineEdit(self.setting_widget)
        self.pair_distance_edit.setPlaceholderText("Fe-O:1.8, O-O:1.2")

        self.attempts_label = BodyLabel("Attempts/atom", self.setting_widget)
        self.attempts_label.setToolTip("Maximum random placement attempts for each atom")
        self.attempts_label.installEventFilter(ToolTipFilter(self.attempts_label, 300, ToolTipPosition.TOP))
        self.attempts_frame = SpinBoxUnitInputFrame(self)
        self.attempts_frame.set_input("unit", 1, "int")
        self.attempts_frame.setRange(1, 1000000)
        self.attempts_frame.set_input_value([500])

        self.strict_checkbox = CheckBox("Strict mode", self.setting_widget)
        self.strict_checkbox.setToolTip("Fail the whole card when any requested sample cannot be packed")
        self.strict_checkbox.setChecked(True)

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _state: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.settingLayout.addWidget(self.structures_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.structures_frame, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.composition_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.composition_edit, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.min_distance_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.min_distance_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.pair_distance_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.pair_distance_edit, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.attempts_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.attempts_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.strict_checkbox, 5, 0, 1, 3)
        self.settingLayout.addWidget(self.seed_checkbox, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 6, 1, 1, 2)

    def create_operation(self):
        return RandomPackingOperation()

    def get_params(self) -> RandomPackingParams:
        return RandomPackingParams(
            structures=int(self.structures_frame.get_input_value()[0]),
            composition=self.composition_edit.text(),
            min_distance=float(self.min_distance_frame.get_input_value()[0]),
            pair_min_distances=self.pair_distance_edit.text(),
            max_attempts_per_atom=int(self.attempts_frame.get_input_value()[0]),
            strict_mode=self.strict_checkbox.isChecked(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: RandomPackingParams) -> None:
        self.structures_frame.set_input_value([int(params.structures)])
        self.composition_edit.setText(params.composition)
        self.min_distance_frame.set_input_value([float(params.min_distance)])
        self.pair_distance_edit.setText(params.pair_min_distances)
        self.attempts_frame.set_input_value([int(params.max_attempts_per_atom)])
        self.strict_checkbox.setChecked(bool(params.strict_mode))
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        params = RandomPackingParams(**raw_params) if raw_params else RandomPackingParams()
        self.set_params(params)
