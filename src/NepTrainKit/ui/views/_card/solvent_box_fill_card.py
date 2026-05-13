"""Card for filling periodic cells with solvent."""

from __future__ import annotations

from PySide6.QtWidgets import QPlainTextEdit
from qfluentwidgets import BodyLabel, CheckBox, ComboBox, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.solvation import DEFAULT_WATER_XYZ, SolventBoxFillOperation, SolventBoxFillParams
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class SolventBoxFillCard(MakeDataCard):
    """Fill an existing periodic cell with solvent molecules."""

    group = "Organic"
    card_name = "Solvent Box Fill"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [
        {"name": "Chen Zherui", "role": "author", "email": "chenzherui0124@foxmail.com"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Solvent Box Fill")
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("solvent_box_fill_card_widget")
        row = 0

        self.solvent_label = BodyLabel("Solvent XYZ", self.setting_widget)
        self.solvent_label.setToolTip("Single solvent molecule in XYZ/extxyz text")
        self.solvent_label.installEventFilter(ToolTipFilter(self.solvent_label, 300, ToolTipPosition.TOP))
        self.solvent_edit = QPlainTextEdit(self.setting_widget)
        self.solvent_edit.setPlainText(DEFAULT_WATER_XYZ)
        self.solvent_edit.setFixedHeight(92)
        self.settingLayout.addWidget(self.solvent_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.solvent_edit, row, 1, 1, 2)
        row += 1

        self.structures_label = BodyLabel("Structures", self.setting_widget)
        self.structures_label.setToolTip("Independent filled boxes generated per input structure")
        self.structures_label.installEventFilter(ToolTipFilter(self.structures_label, 300, ToolTipPosition.TOP))
        self.structures_frame = SpinBoxUnitInputFrame(self)
        self.structures_frame.set_input("unit", 1, "int")
        self.structures_frame.setRange(1, 100000)
        self.structures_frame.set_input_value([1])
        self.settingLayout.addWidget(self.structures_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.structures_frame, row, 1, 1, 2)
        row += 1

        self.count_mode_label = BodyLabel("Count mode", self.setting_widget)
        self.count_mode_label.setToolTip("fixed uses solvent count; density derives the count from box volume")
        self.count_mode_label.installEventFilter(ToolTipFilter(self.count_mode_label, 300, ToolTipPosition.TOP))
        self.count_mode_combo = ComboBox(self.setting_widget)
        self.count_mode_combo.addItems(["fixed", "density"])
        self.settingLayout.addWidget(self.count_mode_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.count_mode_combo, row, 1, 1, 2)
        row += 1

        self.count_label = BodyLabel("Solvent count", self.setting_widget)
        self.count_label.setToolTip("Number of solvent molecules inserted when count mode is fixed")
        self.count_label.installEventFilter(ToolTipFilter(self.count_label, 300, ToolTipPosition.TOP))
        self.count_frame = SpinBoxUnitInputFrame(self)
        self.count_frame.set_input("unit", 1, "int")
        self.count_frame.setRange(1, 1000000)
        self.count_frame.set_input_value([100])
        self.settingLayout.addWidget(self.count_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.count_frame, row, 1, 1, 2)
        row += 1

        self.density_label = BodyLabel("Density", self.setting_widget)
        self.density_label.setToolTip("Solvent mass density in g/cm^3 when count mode is density")
        self.density_label.installEventFilter(ToolTipFilter(self.density_label, 300, ToolTipPosition.TOP))
        self.density_frame = SpinBoxUnitInputFrame(self)
        self.density_frame.set_input("g/cm3", 1, "float")
        self.density_frame.setRange(0.0001, 1000.0)
        self.density_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.density_frame.set_input_value([1.0])
        self.settingLayout.addWidget(self.density_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.density_frame, row, 1, 1, 2)
        row += 1

        self.mode_label = BodyLabel("Sampling mode", self.setting_widget)
        self.mode_label.setToolTip("auto resolves water/general from the solvent molecule")
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItems(["auto", "general", "water", "loose", "dense"])
        self.settingLayout.addWidget(self.mode_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, row, 1, 1, 2)
        row += 1

        self.fill_packing_label = BodyLabel("Fill packing", self.setting_widget)
        self.fill_packing_label.setToolTip("Density scaling factor used when count mode is density")
        self.fill_packing_label.installEventFilter(ToolTipFilter(self.fill_packing_label, 300, ToolTipPosition.TOP))
        self.fill_packing_frame = SpinBoxUnitInputFrame(self)
        self.fill_packing_frame.set_input("x", 1, "float")
        self.fill_packing_frame.setRange(0.0001, 5.0)
        self.fill_packing_frame.object_list[0].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.fill_packing_frame.set_input_value([1.0])
        self.settingLayout.addWidget(self.fill_packing_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.fill_packing_frame, row, 1, 1, 2)
        row += 1

        self.min_distance_label = BodyLabel("Min distance", self.setting_widget)
        self.min_distance_label.setToolTip("Optional global atom-atom distance cutoff; 0 uses element radii")
        self.min_distance_label.installEventFilter(ToolTipFilter(self.min_distance_label, 300, ToolTipPosition.TOP))
        self.min_distance_frame = SpinBoxUnitInputFrame(self)
        self.min_distance_frame.set_input("A", 1, "float")
        self.min_distance_frame.setRange(0.0, 100.0)
        self.min_distance_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.min_distance_frame.set_input_value([0.0])
        self.settingLayout.addWidget(self.min_distance_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.min_distance_frame, row, 1, 1, 2)
        row += 1

        self.collision_label = BodyLabel("Collision scale", self.setting_widget)
        self.collision_label.setToolTip("0 uses the selected mode profile; positive values override radii scaling")
        self.collision_label.installEventFilter(ToolTipFilter(self.collision_label, 300, ToolTipPosition.TOP))
        self.collision_frame = SpinBoxUnitInputFrame(self)
        self.collision_frame.set_input("x", 1, "float")
        self.collision_frame.setRange(0.0, 5.0)
        self.collision_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.collision_frame.set_input_value([0.0])
        self.settingLayout.addWidget(self.collision_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.collision_frame, row, 1, 1, 2)
        row += 1

        self.attempts_label = BodyLabel("Attempts/solvent", self.setting_widget)
        self.attempts_label.setToolTip("Maximum placement attempts per requested solvent molecule")
        self.attempts_label.installEventFilter(ToolTipFilter(self.attempts_label, 300, ToolTipPosition.TOP))
        self.attempts_frame = SpinBoxUnitInputFrame(self)
        self.attempts_frame.set_input("unit", 1, "int")
        self.attempts_frame.setRange(1, 100000)
        self.attempts_frame.set_input_value([500])
        self.settingLayout.addWidget(self.attempts_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.attempts_frame, row, 1, 1, 2)
        row += 1

        self.strict_checkbox = CheckBox("Strict count", self.setting_widget)
        self.strict_checkbox.setToolTip("Fail if the requested solvent count cannot be placed")
        self.strict_checkbox.setChecked(True)
        self.settingLayout.addWidget(self.strict_checkbox, row, 0, 1, 3)
        row += 1

        self.flex_checkbox = CheckBox("Flexible solvent", self.setting_widget)
        self.flex_checkbox.setChecked(False)
        self.flex_checkbox.setToolTip("Use the existing torsion-guard core to pre-generate solvent conformers")
        self.settingLayout.addWidget(self.flex_checkbox, row, 0, 1, 3)
        row += 1

        self.flex_pool_label = BodyLabel("Flex pool", self.setting_widget)
        self.flex_pool_label.setToolTip("Number of pre-generated solvent conformers")
        self.flex_pool_label.installEventFilter(ToolTipFilter(self.flex_pool_label, 300, ToolTipPosition.TOP))
        self.flex_pool_frame = SpinBoxUnitInputFrame(self)
        self.flex_pool_frame.set_input("unit", 1, "int")
        self.flex_pool_frame.setRange(1, 10000)
        self.flex_pool_frame.set_input_value([32])
        self.settingLayout.addWidget(self.flex_pool_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.flex_pool_frame, row, 1, 1, 2)
        row += 1

        self.flex_torsion_label = BodyLabel("Flex torsion", self.setting_widget)
        self.flex_torsion_label.setToolTip("Torsion angle range for flexible solvent conformers")
        self.flex_torsion_label.installEventFilter(ToolTipFilter(self.flex_torsion_label, 300, ToolTipPosition.TOP))
        self.flex_torsion_frame = SpinBoxUnitInputFrame(self)
        self.flex_torsion_frame.set_input(["deg", "deg"], 2, ["float", "float"])
        self.flex_torsion_frame.setRange(-360.0, 360.0)
        self.flex_torsion_frame.set_input_value([-180.0, 180.0])
        self.settingLayout.addWidget(self.flex_torsion_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.flex_torsion_frame, row, 1, 1, 2)
        row += 1

        self.flex_max_label = BodyLabel("Flex max/sigma", self.setting_widget)
        self.flex_max_label.setToolTip("Max torsions per conformer and Gaussian coordinate noise")
        self.flex_max_label.installEventFilter(ToolTipFilter(self.flex_max_label, 300, ToolTipPosition.TOP))
        self.flex_max_frame = SpinBoxUnitInputFrame(self)
        self.flex_max_frame.set_input(["unit", "A"], 2, ["int", "float"])
        self.flex_max_frame.setRange(0.0, 10000.0)
        self.flex_max_frame.object_list[1].setDecimals(4)  # pyright: ignore[reportAttributeAccessIssue]
        self.flex_max_frame.set_input_value([5, 0.03])
        self.settingLayout.addWidget(self.flex_max_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.flex_max_frame, row, 1, 1, 2)
        row += 1

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _state: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))
        self.settingLayout.addWidget(self.seed_checkbox, row, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, row, 1, 1, 2)

    def create_operation(self):
        return SolventBoxFillOperation()

    def get_params(self) -> SolventBoxFillParams:
        flex_max_values = self.flex_max_frame.get_input_value()
        return SolventBoxFillParams(
            solvent_xyz=self.solvent_edit.toPlainText(),
            structures=int(self.structures_frame.get_input_value()[0]),
            count_mode=self.count_mode_combo.currentText(),
            solvent_count=int(self.count_frame.get_input_value()[0]),
            density=float(self.density_frame.get_input_value()[0]),
            sampling_mode=self.mode_combo.currentText(),
            fill_packing=float(self.fill_packing_frame.get_input_value()[0]),
            min_distance=float(self.min_distance_frame.get_input_value()[0]),
            collision_scale=float(self.collision_frame.get_input_value()[0]),
            max_attempts_per_solvent=int(self.attempts_frame.get_input_value()[0]),
            strict_count=self.strict_checkbox.isChecked(),
            flex_solvent=self.flex_checkbox.isChecked(),
            flex_pool=int(self.flex_pool_frame.get_input_value()[0]),
            flex_torsion_range=tuple(map(float, self.flex_torsion_frame.get_input_value())),
            flex_max_torsions=int(flex_max_values[0]),
            flex_gaussian_sigma=float(flex_max_values[1]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: SolventBoxFillParams) -> None:
        self.solvent_edit.setPlainText(params.solvent_xyz)
        self.structures_frame.set_input_value([int(params.structures)])
        self.count_mode_combo.setCurrentText(params.count_mode)
        self.count_frame.set_input_value([int(params.solvent_count)])
        self.density_frame.set_input_value([float(params.density)])
        self.mode_combo.setCurrentText(params.sampling_mode)
        self.fill_packing_frame.set_input_value([float(params.fill_packing)])
        self.min_distance_frame.set_input_value([float(params.min_distance)])
        self.collision_frame.set_input_value([float(params.collision_scale)])
        self.attempts_frame.set_input_value([int(params.max_attempts_per_solvent)])
        self.strict_checkbox.setChecked(bool(params.strict_count))
        self.flex_checkbox.setChecked(bool(params.flex_solvent))
        self.flex_pool_frame.set_input_value([int(params.flex_pool)])
        self.flex_torsion_frame.set_input_value([float(value) for value in params.flex_torsion_range])
        self.flex_max_frame.set_input_value([int(params.flex_max_torsions), float(params.flex_gaussian_sigma)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data):
        super().from_dict(data)
        raw = data.get("params")
        params = SolventBoxFillParams(**raw) if raw else SolventBoxFillParams()
        self.set_params(params)
