"""Card for spatially correlated random non-collinear spins."""

from __future__ import annotations

from qfluentwidgets import BodyLabel, CheckBox, ComboBox, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.magnetism import CorrelatedRandomSpinOperation, CorrelatedRandomSpinParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class CorrelatedRandomSpinCard(MakeDataCard):
    """Generate non-collinear random spins with an explicit spatial correlation length."""

    group = "Magnetism"
    card_name = "Correlated Random Spin"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Correlated Random Spin")
        self.init_ui()

    def init_ui(self):
        self.setObjectName("correlated_random_spin_card_widget")

        self.mode_label = BodyLabel("Mode", self.setting_widget)
        self.mode_label.setToolTip("Full random directions or cone disorder around the reference magnetic moments")
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItems(["Cone around reference", "Full random directions"])
        self.mode_combo.setCurrentText("Cone around reference")

        self.kernel_label = BodyLabel("Kernel", self.setting_widget)
        self.kernel_label.setToolTip("Correlation kernel used for the exact covariance matrix")
        self.kernel_label.installEventFilter(ToolTipFilter(self.kernel_label, 300, ToolTipPosition.TOP))
        self.kernel_combo = ComboBox(self.setting_widget)
        self.kernel_combo.addItems(["exponential", "squared_exponential"])
        self.kernel_combo.setCurrentText("exponential")

        self.xi_label = BodyLabel("Correlation length", self.setting_widget)
        self.xi_label.setToolTip("Spatial correlation length xi in Angstrom")
        self.xi_label.installEventFilter(ToolTipFilter(self.xi_label, 300, ToolTipPosition.TOP))
        self.xi_frame = SpinBoxUnitInputFrame(self)
        self.xi_frame.set_input("A", 1, "float")
        self.xi_frame.setRange(0.000001, 1000000.0)
        self.xi_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.xi_frame.set_input_value([3.0])

        self.samples_label = BodyLabel("Samples", self.setting_widget)
        self.samples_label.setToolTip("Number of independent correlated spin fields generated per input structure")
        self.samples_label.installEventFilter(ToolTipFilter(self.samples_label, 300, ToolTipPosition.TOP))
        self.samples_frame = SpinBoxUnitInputFrame(self)
        self.samples_frame.set_input("unit", 1, "int")
        self.samples_frame.setRange(1, 100000)
        self.samples_frame.set_input_value([1])

        self.cone_label = BodyLabel("Cone angle", self.setting_widget)
        self.cone_label.setToolTip("Maximum cone angle in degrees for cone disorder")
        self.cone_label.installEventFilter(ToolTipFilter(self.cone_label, 300, ToolTipPosition.TOP))
        self.cone_frame = SpinBoxUnitInputFrame(self)
        self.cone_frame.set_input("deg", 1, "float")
        self.cone_frame.setRange(0.0, 180.0)
        self.cone_frame.object_list[0].setDecimals(3)  # pyright: ignore[reportAttributeAccessIssue]
        self.cone_frame.set_input_value([30.0])

        self.source_label = BodyLabel("Magnitude source", self.setting_widget)
        self.source_label.setToolTip("Use existing initial magmoms or build reference magnitudes from map/default")
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

        self.max_atoms_label = BodyLabel("Max exact atoms", self.setting_widget)
        self.max_atoms_label.setToolTip("Maximum eligible atoms allowed for exact full-covariance sampling")
        self.max_atoms_label.installEventFilter(ToolTipFilter(self.max_atoms_label, 300, ToolTipPosition.TOP))
        self.max_atoms_frame = SpinBoxUnitInputFrame(self)
        self.max_atoms_frame.set_input("unit", 1, "int")
        self.max_atoms_frame.setRange(1, 1000000)
        self.max_atoms_frame.set_input_value([200])

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _state: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.settingLayout.addWidget(self.mode_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.mode_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.kernel_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.kernel_combo, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.xi_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.xi_frame, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.samples_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.samples_frame, 3, 1, 1, 2)
        self.settingLayout.addWidget(self.cone_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.cone_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.source_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.source_combo, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.map_label, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 6, 1, 1, 2)
        self.settingLayout.addWidget(self.default_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.lift_scalar_checkbox, 8, 0, 1, 3)
        self.settingLayout.addWidget(self.axis_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_frame, 9, 1, 1, 2)
        self.settingLayout.addWidget(self.apply_label, 10, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 10, 1, 1, 2)
        self.settingLayout.addWidget(self.max_atoms_label, 11, 0, 1, 1)
        self.settingLayout.addWidget(self.max_atoms_frame, 11, 1, 1, 2)
        self.settingLayout.addWidget(self.seed_checkbox, 12, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 12, 1, 1, 2)

        self.mode_combo.currentTextChanged.connect(self._update_mode_widgets)
        self.source_combo.currentTextChanged.connect(self._update_source_widgets)
        self._update_mode_widgets()
        self._update_source_widgets()

    def _update_mode_widgets(self):
        show_cone = self.mode_combo.currentText() == "Cone around reference"
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
        return CorrelatedRandomSpinOperation()

    def get_params(self) -> CorrelatedRandomSpinParams:
        return CorrelatedRandomSpinParams(
            mode=self.mode_combo.currentText(),
            correlation_kernel=self.kernel_combo.currentText(),
            correlation_length=float(self.xi_frame.get_input_value()[0]),
            samples=int(self.samples_frame.get_input_value()[0]),
            cone_angle=float(self.cone_frame.get_input_value()[0]),
            magnitude_source=self.source_combo.currentText(),
            magmom_map=self.map_edit.text(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            lift_scalar=self.lift_scalar_checkbox.isChecked(),
            axis=self.axis_frame.get_input_value(),
            apply_elements=self.apply_edit.text(),
            max_atoms_for_full=int(self.max_atoms_frame.get_input_value()[0]),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: CorrelatedRandomSpinParams) -> None:
        self.mode_combo.setCurrentText(params.mode)
        self.kernel_combo.setCurrentText(params.correlation_kernel)
        self.xi_frame.set_input_value([float(params.correlation_length)])
        self.samples_frame.set_input_value([int(params.samples)])
        self.cone_frame.set_input_value([float(params.cone_angle)])
        self.source_combo.setCurrentText(params.magnitude_source)
        self.map_edit.setText(params.magmom_map)
        self.default_frame.set_input_value([float(params.default_moment)])
        self.lift_scalar_checkbox.setChecked(bool(params.lift_scalar))
        self.axis_frame.set_input_value([float(v) for v in params.axis])
        self.apply_edit.setText(params.apply_elements)
        self.max_atoms_frame.set_input_value([int(params.max_atoms_for_full)])
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.seed_frame.setEnabled(self.seed_checkbox.isChecked())
        self._update_mode_widgets()
        self._update_source_widgets()

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        params = CorrelatedRandomSpinParams(**raw_params) if raw_params else CorrelatedRandomSpinParams()
        self.set_params(params)
