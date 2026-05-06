"""Card for applying random atomic perturbations."""

from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QVBoxLayout, QLineEdit
from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    ToolTipFilter,
    ToolTipPosition,
    CheckBox,
    TransparentToolButton,
    FluentIcon,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import PerturbOperation, PerturbParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import SpinBoxUnitInputFrame
from NepTrainKit.ui.widgets import MakeDataCard


class ElementScalingRow(QFrame):
    """UI row for a single element-specific perturbation limit."""

    def __init__(self, parent=None, default_distance: float = 0.3):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self.element_input = QLineEdit(self)
        self.element_input.setPlaceholderText("Fe")

        self.distance_frame = SpinBoxUnitInputFrame(self)
        self.distance_frame.set_input("Å", 1, "float")
        self.distance_frame.setRange(0, 1)
        self.distance_frame.set_input_value([default_distance])

        self.delete_button = TransparentToolButton(FluentIcon.DELETE, self)
        self.delete_button.setToolTip("Remove this element override")
        self.delete_button.installEventFilter(
            ToolTipFilter(self.delete_button, 300, ToolTipPosition.TOP)
        )

        self._layout.addWidget(self.element_input)
        self._layout.addWidget(self.distance_frame)
        self._layout.addWidget(self.delete_button)

    def set_value(self, element: str, distance: float | None = None) -> None:
        """Populate the row with given element and distance."""
        if element:
            self.element_input.setText(element)
        if distance is not None:
            self.distance_frame.set_input_value([float(distance)])

    def get_value(self) -> tuple[str, float] | None:
        """Return (element, distance) if valid, otherwise None."""
        element = self.element_input.text().strip()
        if not element:
            return None
        distance = float(self.distance_frame.get_input_value()[0])
        return element, distance


@CardManager.register_card
class PerturbCard(MakeDataCard):
    """Apply random atomic displacements within a configurable distance budget.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the card controls.
    """

    group = "Perturbation"
    card_name= "Atomic Perturb"
    menu_icon=r":/images/src/images/perturb.svg"
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.
        
        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle("Make Atomic Perturb")
        self.element_rows = []
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("perturb_card_widget")
        self.engine_label=BodyLabel("Random engine:",self.setting_widget)
        self.engine_type_combo=ComboBox(self.setting_widget)
        self.engine_type_combo.addItem("Sobol")
        self.engine_type_combo.addItem("Uniform")
        self.engine_type_combo.setCurrentIndex(1)

        self.engine_label.setToolTip("Select random engine")
        self.engine_label.installEventFilter(ToolTipFilter(self.engine_label, 300, ToolTipPosition.TOP))

        self.optional_frame = QFrame(self.setting_widget)
        self.optional_frame_layout = QGridLayout(self.optional_frame)
        self.optional_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.optional_frame_layout.setSpacing(2)

        self.optional_label=BodyLabel("Optional",self.setting_widget)
        self.organic_checkbox=CheckBox("Identify organic", self.setting_widget)
        self.organic_checkbox.setChecked(False)
        self.optional_label.setToolTip("Treat organic molecules as rigid units")
        self.optional_label.installEventFilter(ToolTipFilter(self.optional_label, 300, ToolTipPosition.TOP))



        self.optional_frame_layout.addWidget(self.organic_checkbox,0,0,1,1)

        self.scaling_condition_frame = SpinBoxUnitInputFrame(self)
        self.scaling_condition_frame.set_input("Å",1,"float")
        self.scaling_condition_frame.setRange(0,1)
        self.scaling_radio_label=BodyLabel("Max distance:",self.setting_widget)
        self.scaling_condition_frame.set_input_value([0.3])
        self.scaling_radio_label.setToolTip("Maximum displacement distance")
        self.scaling_radio_label.installEventFilter(ToolTipFilter(self.scaling_radio_label, 300, ToolTipPosition.TOP))

        self.element_scaling_label = BodyLabel("Element overrides:", self.setting_widget)
        self.element_scaling_label.setToolTip("Override max distance per element; fallback to global value when empty")
        self.element_scaling_label.installEventFilter(ToolTipFilter(self.element_scaling_label, 300, ToolTipPosition.TOP))
        self.element_scaling_frame = QFrame(self.setting_widget)
        self.element_scaling_layout = QVBoxLayout(self.element_scaling_frame)
        self.element_scaling_layout.setContentsMargins(0, 0, 0, 0)
        self.element_scaling_layout.setSpacing(4)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        self.element_scaling_checkbox = CheckBox("Enable per-element", self.setting_widget)
        self.element_scaling_checkbox.setChecked(False)
        self.element_scaling_checkbox.setToolTip("Use per-element max distance instead of a single global value")
        self.element_scaling_checkbox.installEventFilter(
            ToolTipFilter(self.element_scaling_checkbox, 300, ToolTipPosition.TOP)
        )
        self.add_element_button = TransparentToolButton(FluentIcon.ADD, self.setting_widget)
        self.add_element_button.setToolTip("Add an element-specific distance")
        self.add_element_button.installEventFilter(
            ToolTipFilter(self.add_element_button, 300, ToolTipPosition.TOP)
        )
        header_layout.addWidget(self.element_scaling_checkbox)
        header_layout.addWidget(self.add_element_button)
        header_layout.addStretch(1)
        self.element_rows_frame = QFrame(self.element_scaling_frame)
        self.element_rows_layout = QVBoxLayout(self.element_rows_frame)
        self.element_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.element_rows_layout.setSpacing(4)
        self.element_scaling_layout.addLayout(header_layout)
        self.element_scaling_layout.addWidget(self.element_rows_frame)
        self.element_rows_frame.setVisible(False)

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit",1,"int")
        self.num_condition_frame.setRange(1,10000)
        self.num_condition_frame.set_input_value([50])

        self.num_label=BodyLabel("Max num:",self.setting_widget)
        self.num_label.setToolTip("Number of structures to generate")

        self.num_label.installEventFilter(ToolTipFilter(self.num_label, 300, ToolTipPosition.TOP))

        self.seed_checkbox = CheckBox("Use seed", self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip("Enable reproducible random perturbations")
        self.seed_checkbox.installEventFilter(ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP))
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))

        self.settingLayout.addWidget(self.engine_label,0, 0,1, 1)
        self.settingLayout.addWidget(self.engine_type_combo,0, 1, 1, 2)

        self.settingLayout.addWidget(self.optional_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.optional_frame,1, 1, 1, 2)

        self.settingLayout.addWidget(self.scaling_radio_label, 2, 0, 1, 1)

        self.settingLayout.addWidget(self.scaling_condition_frame, 2, 1, 1,2)

        self.settingLayout.addWidget(self.element_scaling_label, 3, 0, 1, 1)
        self.settingLayout.addWidget(self.element_scaling_frame, 3, 1, 1, 2)

        self.settingLayout.addWidget(self.num_label,4, 0, 1, 1)

        self.settingLayout.addWidget(self.num_condition_frame,4, 1, 1,2)

        self.settingLayout.addWidget(self.seed_checkbox, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 5, 1, 1, 2)

        self.add_element_button.clicked.connect(self._add_element_row)
        self.element_scaling_checkbox.toggled.connect(self._toggle_element_scaling_frame)

    def _toggle_element_scaling_frame(self, checked: bool) -> None:
        """Show or hide element override rows."""
        self.element_rows_frame.setVisible(checked)

    def _add_element_row(self, element: str | None = None, distance: float | None = None) -> ElementScalingRow:
        """Append an element override row."""
        row = ElementScalingRow(self.element_rows_frame, default_distance=self.scaling_condition_frame.get_input_value()[0])
        if element:
            row.set_value(element)
        if distance is not None:
            row.set_value(element or "", distance)
        row.delete_button.clicked.connect(lambda: self._remove_element_row(row))
        self.element_rows_layout.addWidget(row)
        self.element_rows.append(row)
        self.element_rows_frame.setVisible(self.element_scaling_checkbox.isChecked())
        return row

    def _remove_element_row(self, row: ElementScalingRow) -> None:
        """Remove a specific element row."""
        if row in self.element_rows:
            self.element_rows.remove(row)
        row.setParent(None)
        row.deleteLater()

    def _collect_element_scalings(self) -> dict[str, float]:
        """Gather valid element override values."""
        scalings: dict[str, float] = {}
        for row in self.element_rows:
            value = row.get_value()
            if value:
                element, distance = value
                scalings[element] = distance
        return scalings

    def _load_element_scalings(self, scalings: dict[str, float]) -> None:
        """Rebuild element rows from persisted data."""
        while self.element_rows_layout.count():
            item = self.element_rows_layout.takeAt(0).widget()
            if item is not None:
                item.deleteLater()
        self.element_rows.clear()
        for element, distance in (scalings or {}).items():
            self._add_element_row(element, distance)
        if self.element_rows:
            self.element_rows_frame.setVisible(self.element_scaling_checkbox.isChecked())


    def create_operation(self):
        """Return the UI-independent atomic perturbation operation."""
        return PerturbOperation()

    def get_params(self) -> PerturbParams:
        """Read atomic perturbation parameters from UI controls."""
        use_element_scaling = self.element_scaling_checkbox.isChecked()
        return PerturbParams(
            engine_type=int(self.engine_type_combo.currentIndex()),
            max_distance=float(self.scaling_condition_frame.get_input_value()[0]),
            max_num=int(self.num_condition_frame.get_input_value()[0]),
            identify_organic=self.organic_checkbox.isChecked(),
            use_element_scaling=use_element_scaling,
            element_scalings=self._collect_element_scalings() if use_element_scaling else {},
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
        )

    def set_params(self, params: PerturbParams) -> None:
        """Apply atomic perturbation parameters to UI controls."""
        self.engine_type_combo.setCurrentIndex(int(params.engine_type))
        self.scaling_condition_frame.set_input_value([float(params.max_distance)])
        self.num_condition_frame.set_input_value([int(params.max_num)])
        self.organic_checkbox.setChecked(bool(params.identify_organic))
        self.element_scaling_checkbox.setChecked(bool(params.use_element_scaling))
        self._load_element_scalings(params.element_scalings or {})
        self.element_rows_frame.setVisible(self.element_scaling_checkbox.isChecked())
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])

    def process_structure(self, structure):
        """Apply random atomic displacements from UI-independent parameters."""
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        """Restore the card configuration from serialized values.
        
        Parameters
        ----------
        data_dict : dict
            Serialized configuration previously produced by ``to_dict``.
        """
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = PerturbParams(
                engine_type=raw_params.get("engine_type", 1),
                max_distance=raw_params.get("max_distance", 0.3),
                max_num=raw_params.get("max_num", 50),
                identify_organic=raw_params.get("identify_organic", False),
                use_element_scaling=raw_params.get("use_element_scaling", False),
                element_scalings=raw_params.get("element_scalings", {}),
                use_seed=raw_params.get("use_seed", False),
                seed=raw_params.get("seed", 0),
            )
        else:
            params = PerturbParams(
                engine_type=data_dict.get("engine_type", 1),
                max_distance=data_dict.get("scaling_condition", [0.3])[0],
                max_num=data_dict.get("num_condition", [50])[0],
                identify_organic=data_dict.get("organic", False),
                use_element_scaling=data_dict.get("use_element_scaling", False),
                element_scalings=data_dict.get("element_scalings", {}),
                use_seed=data_dict.get("use_seed", False),
                seed=data_dict.get("seed", [0])[0],
            )
        self.set_params(params)
