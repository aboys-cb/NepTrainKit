"""Card for applying axial strain variations to lattice vectors."""

from qfluentwidgets import ToolTipFilter, ToolTipPosition, CheckBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import CellStrainOperation, CellStrainParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    RangeTripletInputFrame,
    ResponsiveFormGrid,
    SegmentedControl,
)
@CardManager.register_card

class CellStrainCard(MakeDataCard):
    """Produce strained lattice variants along user-selected axes and ranges.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget owning the card controls.
    """

    group = "Lattice"

    card_name= "Lattice Strain"
    menu_icon=r":/images/src/images/scaling.svg"
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
        self.setTitle(self.tr("Make Cell Strain"))

        self.init_ui()

    def init_ui(self):
        """Build the form controls that expose the card configuration.
        """
        self.setObjectName("cell_strain_card_widget")


        self.engine_type_combo = SegmentedControl(parent=self.setting_widget)
        for value, label in (
            ("uniaxial", self.tr("Uniaxial")),
            ("biaxial", self.tr("Biaxial")),
            ("triaxial", self.tr("Triaxial")),
            ("isotropic", self.tr("Isotropic")),
            ("custom", self.tr("Custom")),
        ):
            self.engine_type_combo.addItem(label, userData=value)
        self.organic_checkbox=CheckBox(self.tr("Identify organic"), self.setting_widget)
        self.organic_checkbox.setChecked(False)
        self.organic_checkbox.setToolTip(self.tr("Treat organic molecules as rigid units"))
        self.organic_checkbox.installEventFilter(ToolTipFilter(self.organic_checkbox, 300, ToolTipPosition.TOP))

        axes_field = CompactField(self.tr("Axes"), self.engine_type_combo, self.setting_widget)
        axes_field.set_helper_text(self.tr("Choose a preset or enter axes such as X or XY."))
        self.custom_axes_edit = LineEdit(self.setting_widget)
        self.custom_axes_edit.setPlaceholderText(self.tr("For example: X or XY"))
        self.custom_axes_field = CompactField(
            self.tr("Custom axes"),
            self.custom_axes_edit,
            self.setting_widget,
        )
        self.custom_axes_field.hide()
        self.engine_type_combo.currentIndexChanged.connect(self._on_axes_mode_changed)

        self.strain_x_frame = RangeTripletInputFrame(self)
        self.strain_x_frame.setRange(-100,100)
        self.strain_x_frame.set_input_value([-5,5,1])

        self.strain_y_frame = RangeTripletInputFrame(self)
        self.strain_y_frame.setRange(-100,100)
        self.strain_y_frame.set_input_value([-5,5,1])

        self.strain_z_frame = RangeTripletInputFrame(self)
        self.strain_z_frame.setRange(-100,100)
        self.strain_z_frame.set_input_value([-5,5,1])

        setup_section = InspectorSection(self.tr("Setup"), self.setting_widget)
        setup_section.addWidget(axes_field)
        setup_section.addWidget(self.custom_axes_field)
        setup_section.addWidget(self.organic_checkbox)

        ranges_section = InspectorSection(
            self.tr("Strain ranges"),
            self.setting_widget,
            self.tr("Values are percentages; each axis uses minimum, maximum, and step."),
        )
        ranges_grid = ResponsiveFormGrid(ranges_section, two_column_threshold=520)
        self.strain_x_field = CompactField(self.tr("X axis"), self.strain_x_frame, ranges_section)
        self.strain_y_field = CompactField(self.tr("Y axis"), self.strain_y_frame, ranges_section)
        self.strain_z_field = CompactField(self.tr("Z axis"), self.strain_z_frame, ranges_section)
        ranges_grid.add_field(self.strain_x_field)
        ranges_grid.add_field(self.strain_y_field)
        ranges_grid.add_field(self.strain_z_field)
        ranges_section.addWidget(ranges_grid)

        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.addWidget(setup_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(ranges_section, 1, 0, 1, 3)

    def _on_axes_mode_changed(self, _index: int) -> None:
        self.custom_axes_field.setVisible(
            self.engine_type_combo.currentData() == "custom"
        )

    def create_operation(self):
        """Return the UI-independent strain operation."""
        return CellStrainOperation()

    def get_summary_text(self) -> str:
        params = self.get_params()
        active = {
            "x": params.x_range,
            "y": params.y_range,
            "z": params.z_range,
        }
        ranges = ", ".join(
            f"{axis.upper()} {values[0]:g}…{values[1]:g}%"
            for axis, values in active.items()
            if axis in params.axes.lower() or params.axes in ("triaxial", "isotropic")
        )
        return self.tr("{axes} · {ranges}").format(axes=params.axes, ranges=ranges)

    def get_guidance_text(self) -> str:
        return self.tr(
            "Strain values modify the cell before relaxation. Validate volume, symmetry, "
            "and shortest distances in the generated structures."
        )

    def get_params(self) -> CellStrainParams:
        """Read strain parameters from the UI controls."""
        return CellStrainParams(
            axes=(
                self.custom_axes_edit.text().strip()
                if self.engine_type_combo.currentData() == "custom"
                else str(self.engine_type_combo.currentData())
            ),
            x_range=tuple(float(value) for value in self.strain_x_frame.get_input_value()),
            y_range=tuple(float(value) for value in self.strain_y_frame.get_input_value()),
            z_range=tuple(float(value) for value in self.strain_z_frame.get_input_value()),
            identify_organic=self.organic_checkbox.isChecked(),
        )

    def set_params(self, params: CellStrainParams) -> None:
        """Apply strain parameters to the UI controls."""
        self.organic_checkbox.setChecked(bool(params.identify_organic))
        index = self.engine_type_combo.findData(params.axes)
        if index >= 0:
            self.engine_type_combo.setCurrentIndex(index)
        else:
            self.custom_axes_edit.setText(params.axes)
            self.engine_type_combo.setCurrentIndex(
                self.engine_type_combo.findData("custom")
            )
        self.strain_x_frame.set_input_value(list(params.x_range))
        self.strain_y_frame.set_input_value(list(params.y_range))
        self.strain_z_frame.set_input_value(list(params.z_range))

    def process_structure(self, structure):
        """Generate strained lattices from UI-independent parameters."""
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
            params = CellStrainParams(
                axes=raw_params.get("axes", "uniaxial"),
                x_range=tuple(raw_params.get("x_range", [-5.0, 5.0, 1.0])),
                y_range=tuple(raw_params.get("y_range", [-5.0, 5.0, 1.0])),
                z_range=tuple(raw_params.get("z_range", [-5.0, 5.0, 1.0])),
                identify_organic=raw_params.get("identify_organic", False),
            )
        else:
            params = CellStrainParams(
                axes=data_dict.get("engine_type", "uniaxial"),
                x_range=tuple(data_dict.get("x_range", [-5.0, 5.0, 1.0])),
                y_range=tuple(data_dict.get("y_range", [-5.0, 5.0, 1.0])),
                z_range=tuple(data_dict.get("z_range", [-5.0, 5.0, 1.0])),
                identify_organic=data_dict.get("organic", False),
            )
        self.set_params(params)
