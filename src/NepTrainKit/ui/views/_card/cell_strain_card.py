"""Card for scanning normal strains along lattice vectors."""

from qfluentwidgets import CheckBox, ComboBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import CellStrainOperation, CellStrainParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    RangeTripletInputFrame,
    ResponsiveFormGrid,
)


@CardManager.register_card
class CellStrainCard(MakeDataCard):
    """Generate de-duplicated normal-strain paths and grids."""

    group = "Lattice"
    card_name = "Lattice Strain"
    menu_icon = r":/images/src/images/scaling.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Lattice Strain"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("cell_strain_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.engine_type_combo = ComboBox(self.setting_widget)
        for value, label in (
            ("uniaxial", self.tr("Single-axis paths (a, b, c)")),
            ("biaxial", self.tr("Paired-axis grids (ab, ac, bc)")),
            ("triaxial", self.tr("Three-axis grid (abc)")),
            ("isotropic", self.tr("Isotropic path (a = b = c)")),
            ("custom", self.tr("Selected lattice axes")),
        ):
            self.engine_type_combo.addItem(label, userData=value)
        mode_field = CompactField(
            self.tr("Scan mode"), self.engine_type_combo, self.setting_widget
        )
        self.mode_field = mode_field

        self.custom_axes_edit = LineEdit(self.setting_widget)
        self.custom_axes_edit.setPlaceholderText(self.tr("For example: a or ab"))
        self.custom_axes_field = CompactField(
            self.tr("Selected axes"),
            self.custom_axes_edit,
            self.setting_widget,
            self.tr("Use each lattice axis at most once."),
        )
        self.custom_axes_field.hide()

        self.strain_x_frame = self._range_frame()
        self.strain_y_frame = self._range_frame()
        self.strain_z_frame = self._range_frame()
        self.strain_x_field = CompactField(
            self.tr("Lattice a"), self.strain_x_frame, self.setting_widget
        )
        self.strain_y_field = CompactField(
            self.tr("Lattice b"), self.strain_y_frame, self.setting_widget
        )
        self.strain_z_field = CompactField(
            self.tr("Lattice c"), self.strain_z_frame, self.setting_widget
        )

        setup_section = InspectorSection(self.tr("Path definition"), self.setting_widget)
        setup_section.addWidget(mode_field)
        setup_section.addWidget(self.custom_axes_field)

        ranges_section = InspectorSection(
            self.tr("Strain ranges"),
            self.setting_widget,
            self.tr("Engineering strain in percent: minimum – maximum – step."),
        )
        ranges_grid = ResponsiveFormGrid(ranges_section, two_column_threshold=520)
        ranges_grid.add_field(self.strain_x_field)
        ranges_grid.add_field(self.strain_y_field)
        ranges_grid.add_field(self.strain_z_field)
        ranges_section.addWidget(ranges_grid)

        self.organic_checkbox = CheckBox(
            self.tr("Keep detected molecules rigid"), self.setting_widget
        )
        self.organic_checkbox.setChecked(False)
        molecule_section = InspectorSection(
            self.tr("Molecular handling"),
            self.setting_widget,
            self.tr(
                "After affine cell strain, restore the internal geometry of detected molecular clusters."
            ),
        )
        molecule_section.addWidget(self.organic_checkbox)

        self.settingLayout.addWidget(setup_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(ranges_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(molecule_section, 2, 0, 1, 3)

        self.engine_type_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.custom_axes_edit.textChanged.connect(self._on_custom_axes_changed)
        for frame in (
            self.strain_x_frame,
            self.strain_y_frame,
            self.strain_z_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self.refresh_compact_presentation)
        self.organic_checkbox.toggled.connect(self.refresh_compact_presentation)
        self._on_mode_changed()

    def _range_frame(self) -> RangeTripletInputFrame:
        frame = RangeTripletInputFrame(self)
        frame.object_list[0].setRange(-99.0, 100.0)
        frame.object_list[1].setRange(-99.0, 100.0)
        frame.object_list[2].setRange(0.001, 199.0)
        frame.set_input_value([-5.0, 5.0, 1.0])
        return frame

    def _mode_explanation(self) -> str:
        mode = combo_value(self.engine_type_combo)
        explanations = {
            "uniaxial": self.tr(
                "Scans a, b, and c separately, then removes identical strain states."
            ),
            "biaxial": self.tr(
                "Combines ab, ac, and bc grids, then removes overlapping strain states."
            ),
            "triaxial": self.tr("Combines the a, b, and c ranges into one 3D grid."),
            "isotropic": self.tr(
                "Uses the lattice-a range as one shared strain for a, b, and c."
            ),
            "custom": self.tr("Combines only the selected lattice-axis ranges."),
        }
        return explanations.get(mode, "")

    def _on_mode_changed(self, *_args) -> None:
        mode = combo_value(self.engine_type_combo)
        self.custom_axes_field.setVisible(mode == "custom")
        self.mode_field.set_helper_text(self._mode_explanation())
        self._refresh_range_visibility()
        self.refresh_compact_presentation()

    def _on_custom_axes_changed(self, *_args) -> None:
        self._refresh_range_visibility()
        self.refresh_compact_presentation()

    def _refresh_range_visibility(self) -> None:
        mode = combo_value(self.engine_type_combo)
        if mode == "isotropic":
            visible = {"a"}
            self.strain_x_field.set_label(self.tr("Shared strain (a = b = c)"))
        elif mode == "custom":
            selected = self.custom_axes_edit.text().strip().lower()
            visible = set(selected) & {"a", "b", "c"}
            if not visible:
                visible = {"a", "b", "c"}
            self.strain_x_field.set_label(self.tr("Lattice a"))
        else:
            visible = {"a", "b", "c"}
            self.strain_x_field.set_label(self.tr("Lattice a"))
        self.strain_x_field.setVisible("a" in visible)
        self.strain_y_field.setVisible("b" in visible)
        self.strain_z_field.setVisible("c" in visible)

    @staticmethod
    def _custom_axes_to_core(text: str) -> str:
        return text.strip().upper().translate(str.maketrans("ABC", "XYZ"))

    @staticmethod
    def _custom_axes_from_core(text: str) -> str:
        return text.strip().upper().translate(str.maketrans("XYZ", "ABC")).lower()

    @staticmethod
    def _dataset_count(dataset) -> int:
        if dataset is None:
            return 0
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return 1
        try:
            return len(dataset)
        except TypeError:
            return 0

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def create_operation(self):
        return CellStrainOperation()

    def get_summary_text(self) -> str:
        try:
            summary = self.create_operation().sampling_summary(self.get_params())
        except ValueError:
            return self.tr("Complete the strain ranges and axis selection")
        mode_names = {
            "uniaxial": self.tr("single-axis paths"),
            "biaxial": self.tr("paired-axis grids"),
            "triaxial": self.tr("three-axis grid"),
            "isotropic": self.tr("isotropic path"),
        }
        mode = str(summary["mode"])
        name = mode_names.get(mode)
        if name is None:
            name = self.tr("axes {axes}").format(
                axes=self._custom_axes_from_core(mode)
            )
        return self.tr("{mode} · {count}/input").format(
            mode=name, count=summary["outputs_per_input"]
        )

    def get_guidance_text(self) -> str:
        try:
            summary = self.create_operation().sampling_summary(self.get_params())
        except ValueError as exc:
            return translate_runtime_message(exc)
        per_input = int(summary["outputs_per_input"])
        input_count = getattr(self, "_preview_input_count", None)
        if input_count is None:
            input_count = self._dataset_count(getattr(self, "dataset", None)) or None
        elif input_count == 0:
            input_count = None
        if input_count is None:
            count_text = self.tr("Each input produces {count} unique structures.").format(
                count=per_input
            )
        else:
            count_text = self.tr("{inputs} × {count} = {total} outputs.").format(
                inputs=input_count,
                count=per_input,
                total=input_count * per_input,
            )
        return self.tr("{count} {mode}").format(
            count=count_text, mode=self._mode_explanation()
        )

    def get_params(self) -> CellStrainParams:
        mode = combo_value(self.engine_type_combo)
        axes = (
            self._custom_axes_to_core(self.custom_axes_edit.text())
            if mode == "custom"
            else str(mode)
        )
        return CellStrainParams(
            axes=axes,
            x_range=tuple(float(value) for value in self.strain_x_frame.get_input_value()),
            y_range=tuple(float(value) for value in self.strain_y_frame.get_input_value()),
            z_range=tuple(float(value) for value in self.strain_z_frame.get_input_value()),
            identify_organic=self.organic_checkbox.isChecked(),
        )

    def set_params(self, params: CellStrainParams) -> None:
        self.organic_checkbox.setChecked(bool(params.identify_organic))
        if self.engine_type_combo.findData(params.axes) >= 0:
            set_combo_value(self.engine_type_combo, params.axes)
        else:
            set_combo_value(self.engine_type_combo, "custom")
            self.custom_axes_edit.setText(self._custom_axes_from_core(str(params.axes)))
        self.strain_x_frame.set_input_value(list(params.x_range))
        self.strain_y_frame.set_input_value(list(params.y_range))
        self.strain_z_frame.set_input_value(list(params.z_range))
        self._on_mode_changed()

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
