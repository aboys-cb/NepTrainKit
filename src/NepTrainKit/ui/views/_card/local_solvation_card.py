"""Card for local solvent-shell generation."""

from __future__ import annotations

from PySide6.QtWidgets import QPlainTextEdit
from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.solvation import DEFAULT_WATER_XYZ, LocalSolvationOperation, LocalSolvationParams
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class LocalSolvationCard(MakeDataCard):
    """Generate local solvent shells around selected atoms."""

    group = "Organic"
    card_name = "Local Solvation"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [
        {"name": "Chen Zherui", "role": "author", "email": "chenzherui0124@foxmail.com"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Local Solvation")
        self._init_ui()

    def _init_ui(self):
        self.setObjectName("local_solvation_card_widget")
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
        self.structures_label.setToolTip("Independent solvated structures generated per input structure")
        self.structures_label.installEventFilter(ToolTipFilter(self.structures_label, 300, ToolTipPosition.TOP))
        self.structures_frame = SpinBoxUnitInputFrame(self)
        self.structures_frame.set_input("unit", 1, "int")
        self.structures_frame.setRange(1, 100000)
        self.structures_frame.set_input_value([1])
        self.settingLayout.addWidget(self.structures_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.structures_frame, row, 1, 1, 2)
        row += 1

        self.count_label = BodyLabel("Solvent count", self.setting_widget)
        self.count_label.setToolTip("Number of solvent molecules inserted in each generated structure")
        self.count_label.installEventFilter(ToolTipFilter(self.count_label, 300, ToolTipPosition.TOP))
        self.count_frame = SpinBoxUnitInputFrame(self)
        self.count_frame.set_input("unit", 1, "int")
        self.count_frame.setRange(1, 100000)
        self.count_frame.set_input_value([30])
        self.settingLayout.addWidget(self.count_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.count_frame, row, 1, 1, 2)
        row += 1

        self.mode_label = BodyLabel("Sampling mode", self.setting_widget)
        self.mode_label.setToolTip("auto resolves water and ion-water from solvent and selected centers")
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItems(["auto", "general", "water", "ion-water", "loose", "dense"])
        self.settingLayout.addWidget(self.mode_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, row, 1, 1, 2)
        row += 1

        self.center_mode_label = BodyLabel("Center mode", self.setting_widget)
        self.center_mode_label.setToolTip("How center atoms for local solvation are selected")
        self.center_mode_label.installEventFilter(ToolTipFilter(self.center_mode_label, 300, ToolTipPosition.TOP))
        self.center_mode_combo = ComboBox(self.setting_widget)
        self.center_mode_combo.addItems(["all", "elements", "indices", "z_range"])
        self.settingLayout.addWidget(self.center_mode_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.center_mode_combo, row, 1, 1, 2)
        row += 1

        self.elements_label = BodyLabel("Center elements", self.setting_widget)
        self.elements_label.setToolTip("Comma-separated element symbols used when center mode is elements")
        self.elements_label.installEventFilter(ToolTipFilter(self.elements_label, 300, ToolTipPosition.TOP))
        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setPlaceholderText("Ca, Na, O")
        self.settingLayout.addWidget(self.elements_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.elements_edit, row, 1, 1, 2)
        row += 1

        self.indices_label = BodyLabel("Center indices", self.setting_widget)
        self.indices_label.setToolTip("1-based atom indices/ranges used when center mode is indices")
        self.indices_label.installEventFilter(ToolTipFilter(self.indices_label, 300, ToolTipPosition.TOP))
        self.indices_edit = LineEdit(self.setting_widget)
        self.indices_edit.setPlaceholderText("1,3,5-8")
        self.settingLayout.addWidget(self.indices_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.indices_edit, row, 1, 1, 2)
        row += 1

        self.z_label = BodyLabel("Z range", self.setting_widget)
        self.z_label.setToolTip("Cartesian z interval used when center mode is z_range")
        self.z_label.installEventFilter(ToolTipFilter(self.z_label, 300, ToolTipPosition.TOP))
        self.z_frame = SpinBoxUnitInputFrame(self)
        self.z_frame.set_input(["A", "A"], 2, ["float", "float"])
        self.z_frame.setRange(-100000.0, 100000.0)
        self.z_frame.set_input_value([0.0, 0.0])
        self.settingLayout.addWidget(self.z_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.z_frame, row, 1, 1, 2)
        row += 1

        self.shell_label = BodyLabel("Shell range", self.setting_widget)
        self.shell_label.setToolTip("Center-to-solvent shell range in Angstrom for local placement")
        self.shell_label.installEventFilter(ToolTipFilter(self.shell_label, 300, ToolTipPosition.TOP))
        self.shell_frame = SpinBoxUnitInputFrame(self)
        self.shell_frame.set_input(["A", "A"], 2, ["float", "float"])
        self.shell_frame.setRange(0.0, 1000.0)
        self.shell_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.shell_frame.object_list[1].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.shell_frame.set_input_value([2.2, 4.5])
        self.settingLayout.addWidget(self.shell_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.shell_frame, row, 1, 1, 2)
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

        self.attempts_label = BodyLabel("Max attempts", self.setting_widget)
        self.attempts_label.setToolTip("Maximum placement attempts per generated structure")
        self.attempts_label.installEventFilter(ToolTipFilter(self.attempts_label, 300, ToolTipPosition.TOP))
        self.attempts_frame = SpinBoxUnitInputFrame(self)
        self.attempts_frame.set_input("unit", 1, "int")
        self.attempts_frame.setRange(1, 10000000)
        self.attempts_frame.set_input_value([3000])
        self.settingLayout.addWidget(self.attempts_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.attempts_frame, row, 1, 1, 2)
        row += 1

        self.strict_checkbox = CheckBox("Strict count", self.setting_widget)
        self.strict_checkbox.setToolTip("Fail if the requested solvent count cannot be placed")
        self.strict_checkbox.setChecked(True)
        self.settingLayout.addWidget(self.strict_checkbox, row, 0, 1, 3)
        row += 1

        self.auto_box_checkbox = CheckBox("Auto box for non-periodic output", self.setting_widget)
        self.auto_box_checkbox.setChecked(False)
        self.settingLayout.addWidget(self.auto_box_checkbox, row, 0, 1, 3)
        row += 1

        self.box_size_label = BodyLabel("Fixed box size", self.setting_widget)
        self.box_size_label.setToolTip("Non-periodic fixed output box when auto box is off")
        self.box_size_label.installEventFilter(ToolTipFilter(self.box_size_label, 300, ToolTipPosition.TOP))
        self.box_size_frame = SpinBoxUnitInputFrame(self)
        self.box_size_frame.set_input("A", 1, "float")
        self.box_size_frame.setRange(0.001, 100000.0)
        self.box_size_frame.set_input_value([100.0])
        self.settingLayout.addWidget(self.box_size_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.box_size_frame, row, 1, 1, 2)
        row += 1

        self.box_label = BodyLabel("Auto box padding/min", self.setting_widget)
        self.box_label.setToolTip("Padding and minimum edge length used by auto box")
        self.box_label.installEventFilter(ToolTipFilter(self.box_label, 300, ToolTipPosition.TOP))
        self.box_frame = SpinBoxUnitInputFrame(self)
        self.box_frame.set_input(["A", "A"], 2, ["float", "float"])
        self.box_frame.setRange(0.0, 100000.0)
        self.box_frame.set_input_value([8.0, 0.0])
        self.settingLayout.addWidget(self.box_label, row, 0, 1, 1)
        self.settingLayout.addWidget(self.box_frame, row, 1, 1, 2)
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
        return LocalSolvationOperation()

    def get_params(self) -> LocalSolvationParams:
        flex_max_values = self.flex_max_frame.get_input_value()
        box_values = self.box_frame.get_input_value()
        return LocalSolvationParams(
            solvent_xyz=self.solvent_edit.toPlainText(),
            structures=int(self.structures_frame.get_input_value()[0]),
            solvent_count=int(self.count_frame.get_input_value()[0]),
            sampling_mode=self.mode_combo.currentText(),
            center_mode=self.center_mode_combo.currentText(),
            center_elements=self.elements_edit.text(),
            center_indices=self.indices_edit.text(),
            z_range=tuple(map(float, self.z_frame.get_input_value())),
            shell=tuple(map(float, self.shell_frame.get_input_value())),
            min_distance=float(self.min_distance_frame.get_input_value()[0]),
            collision_scale=float(self.collision_frame.get_input_value()[0]),
            max_attempts=int(self.attempts_frame.get_input_value()[0]),
            strict_count=self.strict_checkbox.isChecked(),
            auto_box=self.auto_box_checkbox.isChecked(),
            box_size=float(self.box_size_frame.get_input_value()[0]),
            box_padding=float(box_values[0]),
            min_box=float(box_values[1]),
            flex_solvent=self.flex_checkbox.isChecked(),
            flex_pool=int(self.flex_pool_frame.get_input_value()[0]),
            flex_torsion_range=tuple(map(float, self.flex_torsion_frame.get_input_value())),
            flex_max_torsions=int(flex_max_values[0]),
            flex_gaussian_sigma=float(flex_max_values[1]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: LocalSolvationParams) -> None:
        self.solvent_edit.setPlainText(params.solvent_xyz)
        self.structures_frame.set_input_value([int(params.structures)])
        self.count_frame.set_input_value([int(params.solvent_count)])
        self.mode_combo.setCurrentText(params.sampling_mode)
        self.center_mode_combo.setCurrentText(params.center_mode)
        self.elements_edit.setText(params.center_elements)
        self.indices_edit.setText(params.center_indices)
        self.z_frame.set_input_value([float(value) for value in params.z_range])
        self.shell_frame.set_input_value([float(value) for value in params.shell])
        self.min_distance_frame.set_input_value([float(params.min_distance)])
        self.collision_frame.set_input_value([float(params.collision_scale)])
        self.attempts_frame.set_input_value([int(params.max_attempts)])
        self.strict_checkbox.setChecked(bool(params.strict_count))
        self.auto_box_checkbox.setChecked(bool(params.auto_box))
        self.box_size_frame.set_input_value([float(params.box_size)])
        self.box_frame.set_input_value([float(params.box_padding), float(params.min_box)])
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
        params = LocalSolvationParams(**raw) if raw else LocalSolvationParams()
        self.set_params(params)
