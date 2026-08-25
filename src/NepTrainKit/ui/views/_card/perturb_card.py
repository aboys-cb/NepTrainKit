"""Card for applying random atomic perturbations."""

from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget, QLineEdit
from qfluentwidgets import (
    BodyLabel,
    ToolTipFilter,
    ToolTipPosition,
    CheckBox,
    TransparentToolButton,
    FluentIcon,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import PerturbOperation, PerturbParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    ResponsiveFormGrid,
    SegmentedControl,
    SpinBoxUnitInputFrame,
)
from NepTrainKit.ui.widgets import MakeDataCard


class ElementScalingRow(QFrame):
    """UI row for a single element-specific perturbation limit."""

    def __init__(self, parent=None, default_distance: float = 0.3):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)

        self.element_input = QLineEdit(self)
        self.element_input.setPlaceholderText(self.tr("Fe"))

        self.distance_frame = SpinBoxUnitInputFrame(self)
        self.distance_frame.set_input("Å", 1, "float")
        self.distance_frame.setDecimals(4)
        self.distance_frame.setSingleStep(0.01)
        self.distance_frame.setRange(0, 1)
        self.distance_frame.set_input_value([default_distance])

        self.delete_button = TransparentToolButton(FluentIcon.DELETE, self)
        self.delete_button.setToolTip(self.tr("Remove this element scaling"))
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
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]
    def __init__(self, parent=None):
        """Initialise the card and build its configuration widgets.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget passed to the base card constructor.
        """
        super().__init__(parent)
        self.setTitle(self.tr("Make Atomic Perturb"))
        self.element_rows = []
        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.

        Related fields are separated into sections and reflow to one column
        inside the right inspector.
        """
        self.setObjectName("perturb_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(12)

        self.engine_type_combo = SegmentedControl(
            [self.tr("Sobol"), self.tr("Uniform")], self.setting_widget
        )
        self.engine_type_combo.setCurrentIndex(1)
        engine_field = CompactField(self.tr("Random engine"), self.engine_type_combo, self.setting_widget)
        engine_field.setToolTip(
            self.tr(
                "Uniform is the general default; Sobol improves small-sample coverage for up to 7,067 atoms"
            )
        )
        engine_field.installEventFilter(ToolTipFilter(engine_field, 300, ToolTipPosition.TOP))

        self.organic_checkbox = CheckBox(self.tr("Identify organic"), self.setting_widget)
        self.organic_checkbox.setChecked(False)
        optional_field = CompactField(self.tr("Optional"), self.organic_checkbox, self.setting_widget)
        optional_field.setToolTip(self.tr("Treat organic molecules as rigid units"))
        optional_field.installEventFilter(ToolTipFilter(optional_field, 300, ToolTipPosition.TOP))

        self.scaling_condition_frame = SpinBoxUnitInputFrame(self)
        self.scaling_condition_frame.set_input("Å", 1, "float")
        self.scaling_condition_frame.setDecimals(4)
        self.scaling_condition_frame.setSingleStep(0.01)
        self.scaling_condition_frame.setRange(0, 1)
        self.scaling_condition_frame.set_input_value([0.3])
        distance_field = CompactField(self.tr("Max distance"), self.scaling_condition_frame, self.setting_widget)
        distance_field.setToolTip(self.tr("Maximum displacement distance"))
        distance_field.installEventFilter(ToolTipFilter(distance_field, 300, ToolTipPosition.TOP))

        self.element_scaling_label = BodyLabel(self.tr("Element Scaling:"), self.setting_widget)
        self.element_scaling_label.setToolTip(
            self.tr("Set maximum displacement per element; unlisted elements use Max distance")
        )
        self.element_scaling_label.installEventFilter(ToolTipFilter(self.element_scaling_label, 300, ToolTipPosition.TOP))
        self.element_scaling_checkbox = CheckBox(self.tr("Enable Scaling"), self.setting_widget)
        self.element_scaling_checkbox.setChecked(False)
        self.element_scaling_checkbox.setToolTip(
            self.tr("Enable element-specific maximum displacement")
        )
        self.element_scaling_checkbox.installEventFilter(
            ToolTipFilter(self.element_scaling_checkbox, 300, ToolTipPosition.TOP)
        )
        self.add_element_button = TransparentToolButton(FluentIcon.ADD, self.setting_widget)
        self.add_element_button.setToolTip(self.tr("Add Element Scaling"))
        self.add_element_button.installEventFilter(
            ToolTipFilter(self.add_element_button, 300, ToolTipPosition.TOP)
        )
        element_toggle_row = QWidget(self.setting_widget)
        element_toggle_layout = QHBoxLayout(element_toggle_row)
        element_toggle_layout.setContentsMargins(0, 0, 0, 0)
        element_toggle_layout.setSpacing(4)
        element_toggle_layout.addWidget(self.element_scaling_checkbox)
        element_toggle_layout.addWidget(self.add_element_button)
        element_toggle_layout.addStretch(1)
        element_field = CompactField(self.tr("Element scaling"), element_toggle_row, self.setting_widget)

        self.element_rows_frame = QFrame(self.setting_widget)
        self.element_rows_layout = QVBoxLayout(self.element_rows_frame)
        self.element_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.element_rows_layout.setSpacing(4)
        self.element_scaling_label.setVisible(False)
        self.element_rows_frame.setVisible(False)
        self.add_element_button.setEnabled(False)

        self.num_condition_frame = SpinBoxUnitInputFrame(self)
        self.num_condition_frame.set_input("unit", 1, "int")
        self.num_condition_frame.setRange(1, 10000)
        self.num_condition_frame.set_input_value([50])
        num_field = CompactField(self.tr("Structures"), self.num_condition_frame, self.setting_widget)
        num_field.setToolTip(self.tr("Number of perturbed structures to generate"))
        num_field.installEventFilter(ToolTipFilter(num_field, 300, ToolTipPosition.TOP))

        self.seed_checkbox = CheckBox(self.tr("Use seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_checkbox.setToolTip(self.tr("Enable reproducible random perturbations"))
        self.seed_checkbox.installEventFilter(ToolTipFilter(self.seed_checkbox, 300, ToolTipPosition.TOP))
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])
        self.seed_frame.setEnabled(False)
        self.seed_checkbox.stateChanged.connect(lambda _s: self.seed_frame.setEnabled(self.seed_checkbox.isChecked()))
        seed_row = QWidget(self.setting_widget)
        seed_row_layout = QHBoxLayout(seed_row)
        seed_row_layout.setContentsMargins(0, 0, 0, 0)
        seed_row_layout.setSpacing(6)
        seed_row_layout.addWidget(self.seed_checkbox)
        seed_row_layout.addWidget(self.seed_frame, 1)
        seed_field = CompactField(self.tr("Reproducibility"), seed_row, self.setting_widget)

        basics_section = InspectorSection(self.tr("Perturbation"), self.setting_widget)
        basics_grid = ResponsiveFormGrid(basics_section)
        basics_grid.add_field(engine_field, span=2)
        basics_grid.add_field(distance_field)
        basics_grid.add_field(optional_field)
        basics_section.addWidget(basics_grid)

        element_section = InspectorSection(self.tr("Element limits"), self.setting_widget)
        element_section.addWidget(element_field)
        element_section.addWidget(self.element_scaling_label)
        element_section.addWidget(self.element_rows_frame)

        output_section = InspectorSection(self.tr("Generation"), self.setting_widget)
        output_grid = ResponsiveFormGrid(output_section)
        output_grid.add_field(num_field)
        output_grid.add_field(seed_field, span=2)
        output_section.addWidget(output_grid)

        self.settingLayout.addWidget(basics_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(element_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(output_section, 2, 0, 1, 3)

        self.add_element_button.clicked.connect(self._add_element_row)
        self.element_scaling_checkbox.toggled.connect(self._toggle_element_scaling_frame)

    def _toggle_element_scaling_frame(self, checked: bool) -> None:
        """Show or hide element scaling controls."""
        self.element_scaling_label.setVisible(checked)
        self.element_rows_frame.setVisible(checked)
        self.add_element_button.setEnabled(checked)

    def _add_element_row(self, element: str | None = None, distance: float | None = None) -> ElementScalingRow:
        """Append an element scaling row."""
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
        """Gather valid element scaling values."""
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


    def get_summary_text(self) -> str:
        """Return a one-line description shown while the card is collapsed."""
        params = self.get_params()
        parts = [self.tr("max {distance} Å").format(distance=params.max_distance)]
        if params.use_element_scaling and params.element_scalings:
            parts.append(
                self.tr("{count} element overrides").format(count=len(params.element_scalings))
            )
        parts.append(self.tr("{count} structures").format(count=params.max_num))
        if params.use_seed:
            parts.append(self.tr("seed {seed}").format(seed=params.seed))
        return " · ".join(parts)

    def get_guidance_text(self) -> str:
        """Return bounded guidance without inventing a chemistry-independent optimum."""
        params = self.get_params()
        engine = self.tr("Sobol") if params.engine_type == 0 else self.tr("Uniform")
        return self.tr(
            "{engine} engine · {distance} Å is a hard displacement ceiling. "
            "Inspect shortest distances in a small output sample before scaling up."
        ).format(engine=engine, distance=f"{params.max_distance:.4g}")

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
