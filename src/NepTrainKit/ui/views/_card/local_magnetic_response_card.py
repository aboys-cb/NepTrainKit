"""Thin UI for controlled local magnetic-response groups."""

from qfluentwidgets import CaptionLabel, CheckBox, ComboBox, LineEdit

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.magnetic_response import LocalMagneticResponseParams, MagneticResponseScanOperation
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    DirectionInput,
    ElementLineEdit,
    ElementPairLineEdit,
    InspectorSection,
    MakeDataCard,
    NumericScanInput,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class LocalMagneticResponseCard(MakeDataCard):
    """Build complete local rotation or moment-scale response groups."""

    group = "Magnetism"
    card_name = "Local Magnetic Response"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Local Magnetic Response"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("local_magnetic_response_card_widget")
        self._active_kind = "Atom pair canting"
        self._rotation_scan_text = "-2,-1,0,1,2"
        self._magnitude_scan_text = "0.8,0.9,1.0,1.1,1.2"
        self.kind_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.kind_combo,
            ["Single-spin tilt", "Atom pair canting", "Group pair canting", "Moment magnitude"],
        )
        set_combo_value(self.kind_combo, "Atom pair canting")

        self.scan_input = NumericScanInput(
            self.setting_widget, minimum=-180.0, maximum=180.0, decimals=3
        )
        self.scan_input.set_range(-2.0, 2.0, 1.0)
        self.scan_field = CompactField(
            self.tr("Rotation-angle scan (degrees)"),
            self.scan_input,
            self.setting_widget,
            self.tr("Production default is a symmetric five-point path: -2°, -1°, 0°, +1°, +2°."),
        )

        response_section = InspectorSection(
            self.tr("Response path"),
            self.setting_widget,
            self.tr("Choose the physical probe first; only parameters used by that probe are shown."),
        )
        response_section.addWidget(CompactField(self.tr("Probe"), self.kind_combo, response_section))
        response_section.addWidget(self.scan_field)

        self.target_edit = LineEdit(self.setting_widget)
        self.target_edit.setPlaceholderText(self.tr("For example: 1 or 1,3-5"))
        self.target_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.target_mode_combo,
            ["First eligible atom", "All eligible atoms", "Explicit indices"],
        )
        set_combo_value(self.target_mode_combo, "First eligible atom")
        self.target_mode_field = CompactField(
            self.tr("Target selection"), self.target_mode_combo, self.setting_widget
        )
        self.target_field = CompactField(
            self.tr("Atom indices (1-based)"),
            self.target_edit,
            self.setting_widget,
            self.tr("Ranges are allowed, for example 1,3-5."),
        )

        self.apply_edit = ElementLineEdit(self.setting_widget, multiple=True)
        self.apply_edit.setPlaceholderText(self.tr("For example: Fe,Co; empty includes every element"))
        self.apply_field = CompactField(
            self.tr("Eligible elements"),
            self.apply_edit,
            self.setting_widget,
            self.tr("Only atoms with non-zero moments and one of these elements can be selected."),
        )

        self.pair_source_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.pair_source_combo, ["Manual indices", "Auto by neighbor shell"])
        set_combo_value(self.pair_source_combo, "Manual indices")
        self.pair_source_field = CompactField(self.tr("Pair selection"), self.pair_source_combo, self.setting_widget)
        self.left_edit = LineEdit(self.setting_widget)
        self.left_edit.setText("1")
        self.right_edit = LineEdit(self.setting_widget)
        self.right_edit.setText("2")
        self.left_field = CompactField(self.tr("Left atom(s), 1-based"), self.left_edit, self.setting_widget)
        self.right_field = CompactField(self.tr("Right atom(s), 1-based"), self.right_edit, self.setting_widget)
        self.shell_frame = SpinBoxUnitInputFrame(self)
        self.shell_frame.set_input("", 1, "int")
        self.shell_frame.setRange(1, 100)
        self.shell_frame.set_input_value([1])
        self.shell_field = CompactField(
            self.tr("Neighbor shell"),
            self.shell_frame,
            self.setting_widget,
            inline=True,
            input_max_width=132,
        )

        self.pair_filters_checkbox = CheckBox(
            self.tr("Filter automatic pairs"), self.setting_widget
        )
        self.pair_tol_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.pair_tol_frame.set_input(self.tr("Å"), 1, "float")
        self.pair_tol_frame.setRange(0.0001, 5.0)
        self.pair_tol_frame.object_list[0].setDecimals(4)
        self.pair_tol_frame.set_input_value([0.05])
        self.pair_tol_field = CompactField(
            self.tr("Shell tolerance"),
            self.pair_tol_frame,
            self.setting_widget,
            self.tr("Distances within this tolerance are treated as the same neighbor shell."),
            inline=True,
            input_max_width=150,
        )
        self.pair_element_edit = ElementPairLineEdit(self.setting_widget)
        self.pair_element_edit.setPlaceholderText(self.tr("For example: Fe-Co or Fe-Fe,Fe-Co"))
        self.pair_element_field = CompactField(
            self.tr("Element pairs"),
            self.pair_element_edit,
            self.setting_widget,
            self.tr("Leave empty to accept every element pair."),
        )
        self.pair_group_edit = LineEdit(self.setting_widget)
        self.pair_group_edit.setPlaceholderText(self.tr("For example: A-B"))
        self.pair_group_field = CompactField(
            self.tr("Label pairs"),
            self.pair_group_edit,
            self.setting_widget,
            self.tr("Uses atoms.arrays['group']; leave empty to ignore group labels."),
        )
        self.bond_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.bond_mode_combo,
            [
                ("Any", "Any direction"),
                ("Near axis", "Near an axis"),
                ("Near plane", "Near a plane"),
            ],
        )
        set_combo_value(self.bond_mode_combo, "Any")
        self.bond_mode_field = CompactField(
            self.tr("Bond direction"), self.bond_mode_combo, self.setting_widget
        )
        self.bond_axis_input = DirectionInput(
            self.setting_widget, default=(0.0, 0.0, 1.0)
        )
        self.bond_axis_field = CompactField(
            self.tr("Reference axis / plane normal (Cartesian)"),
            self.bond_axis_input,
            self.setting_widget,
        )
        self.bond_tol_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.bond_tol_frame.set_input(self.tr("°"), 1, "float")
        self.bond_tol_frame.setRange(0.1, 90.0)
        self.bond_tol_frame.object_list[0].setDecimals(1)
        self.bond_tol_frame.set_input_value([20.0])
        self.bond_tol_field = CompactField(
            self.tr("Angular tolerance"),
            self.bond_tol_frame,
            self.setting_widget,
            inline=True,
            input_max_width=150,
        )
        self.pair_filters_section = InspectorSection(
            self.tr("Automatic pair filters"),
            self.setting_widget,
            self.tr("Optional filters are applied after the neighbor shell is selected."),
        )
        pair_filters_grid = ResponsiveFormGrid(self.pair_filters_section)
        pair_filters_grid.add_field(self.pair_tol_field)
        pair_filters_grid.add_field(self.bond_mode_field)
        pair_filters_grid.add_field(self.pair_element_field)
        pair_filters_grid.add_field(self.pair_group_field)
        pair_filters_grid.add_field(self.bond_axis_field, span=2)
        pair_filters_grid.add_field(self.bond_tol_field)
        self.pair_filters_section.addWidget(pair_filters_grid)
        self.pair_filters_section.hide()
        self.group_a_edit = LineEdit(self.setting_widget)
        self.group_a_edit.setText("A")
        self.group_b_edit = LineEdit(self.setting_widget)
        self.group_b_edit.setText("B")
        self.group_a_field = CompactField(self.tr("Left group name"), self.group_a_edit, self.setting_widget)
        self.group_b_field = CompactField(self.tr("Right group name"), self.group_b_edit, self.setting_widget)

        selection_section = InspectorSection(self.tr("Targets"), self.setting_widget)
        selection_grid = ResponsiveFormGrid(selection_section)
        for field, span in (
            (self.target_mode_field, 1),
            (self.target_field, 2),
            (self.apply_field, 2),
            (self.pair_source_field, 2),
            (self.left_field, 1),
            (self.right_field, 1),
            (self.shell_field, 2),
            (self.group_a_field, 1),
            (self.group_b_field, 1),
        ):
            selection_grid.add_field(field, span=span)
        selection_section.addWidget(selection_grid)
        selection_section.addWidget(self.pair_filters_checkbox)
        selection_section.addWidget(self.pair_filters_section)

        self.advanced_checkbox = CheckBox(self.tr("Show rotation axis and output limit"), self.setting_widget)
        self.axis_input = DirectionInput(self.setting_widget, default=(0.0, 1.0, 0.0))
        self.axis_field = CompactField(
            self.tr("Rotation axis (Cartesian)"), self.axis_input, self.setting_widget
        )
        self.limit_frame = SpinBoxUnitInputFrame(self)
        self.limit_frame.set_input("", 1, "int")
        self.limit_frame.setRange(3, 999999)
        self.limit_frame.set_input_value([100])
        self.limit_field = CompactField(
            self.tr("Maximum structures"),
            self.limit_frame,
            self.setting_widget,
            self.tr("Complete groups are kept together when the limit is reached."),
            inline=True,
            input_max_width=150,
        )
        self.advanced_section = InspectorSection(self.tr("Axis and limit"), self.setting_widget)
        advanced_grid = ResponsiveFormGrid(self.advanced_section)
        advanced_grid.add_field(self.axis_field)
        advanced_grid.add_field(self.limit_field)
        self.advanced_section.addWidget(advanced_grid)
        self.advanced_section.hide()

        self.output_preview = CaptionLabel("", self.setting_widget)
        self.output_preview.setWordWrap(True)
        output_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        output_section.addWidget(self.output_preview)

        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.addWidget(response_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(selection_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_checkbox, 2, 0, 1, 3)
        self.settingLayout.addWidget(self.advanced_section, 3, 0, 1, 3)
        self.settingLayout.addWidget(output_section, 4, 0, 1, 3)

        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        self.pair_source_combo.currentIndexChanged.connect(self._update_widgets)
        self.target_mode_combo.currentIndexChanged.connect(self._update_widgets)
        self.bond_mode_combo.currentIndexChanged.connect(self._update_widgets)
        self.pair_filters_checkbox.toggled.connect(self._update_widgets)
        self.advanced_checkbox.toggled.connect(self.advanced_section.setVisible)
        for spin in self.scan_input.range_frame.object_list:
            spin.valueChanged.connect(self._update_output_preview)
        self.scan_input.custom_edit.textChanged.connect(self._update_output_preview)
        self._update_widgets()

    def _on_kind_changed(self, *_args):
        kind = combo_value(self.kind_combo)
        if self._active_kind == "Moment magnitude":
            self._magnitude_scan_text = self.scan_input.scan_text()
        else:
            self._rotation_scan_text = self.scan_input.scan_text()
        self.scan_input.set_scan_text(
            self._magnitude_scan_text if kind == "Moment magnitude" else self._rotation_scan_text
        )
        self._active_kind = kind
        self._update_widgets()

    def _update_widgets(self, *_args):
        kind = combo_value(self.kind_combo)
        is_magnitude = kind == "Moment magnitude"
        self.scan_field.set_label(
            self.tr("Moment scale scan") if is_magnitude else self.tr("Rotation-angle scan (degrees)")
        )
        self.scan_field.set_helper_text(
            self.tr("Scale 1.0 preserves the original moment magnitude; direction is fixed.")
            if is_magnitude
            else self.tr("Angles are displayed in degrees and stored as radians in the response metadata.")
        )
        is_pair = kind == "Atom pair canting"
        is_group = kind == "Group pair canting"
        auto = is_pair and combo_value(self.pair_source_combo) == "Auto by neighbor shell"
        is_atom_target = kind in {"Single-spin tilt", "Moment magnitude"}
        explicit = is_atom_target and combo_value(self.target_mode_combo) == "Explicit indices"
        self.target_mode_field.setVisible(is_atom_target)
        self.target_field.setVisible(explicit)
        self.apply_field.setVisible(is_atom_target or auto)
        self.pair_source_field.setVisible(is_pair)
        self.left_field.setVisible(is_pair and not auto)
        self.right_field.setVisible(is_pair and not auto)
        self.shell_field.setVisible(auto)
        self.pair_filters_checkbox.setVisible(auto)
        show_pair_filters = auto and self.pair_filters_checkbox.isChecked()
        self.pair_filters_section.setVisible(show_pair_filters)
        show_bond_details = show_pair_filters and combo_value(self.bond_mode_combo) != "Any"
        self.bond_axis_field.setVisible(show_bond_details)
        self.bond_tol_field.setVisible(show_bond_details)
        self.group_a_field.setVisible(is_group)
        self.group_b_field.setVisible(is_group)
        self.axis_field.setVisible(not is_magnitude)
        self._update_output_preview()

    def _update_output_preview(self, *_args):
        try:
            count = self.scan_input.count()
            groups = max(0, int(self.limit_frame.get_input_value()[0]) // count)
            self.output_preview.setText(
                self.tr(
                    "{count} structures per complete group, including one reference; "
                    "the current limit can keep at most {groups} groups."
                ).format(count=count, groups=groups)
            )
        except ValueError as exc:
            self.output_preview.setText(str(exc))

    def get_summary_text(self) -> str:
        return self.tr("{probe} · {count} per group").format(
            probe=self.kind_combo.currentText(), count=self.scan_input.count()
        )

    def get_guidance_text(self) -> str:
        kind = combo_value(self.kind_combo)
        if kind == "Moment magnitude":
            return self.tr(
                "Scale 1.0 is the reference. Check that the selected magnitudes vary while their directions stay fixed."
            )
        if kind == "Atom pair canting":
            return self.tr(
                "Use a symmetric scan: the left atom rotates by +θ/2 and the right atom by −θ/2."
            )
        if kind == "Group pair canting":
            return self.tr(
                "The two group labels must exist on the input; group A rotates by +θ/2 and group B by −θ/2."
            )
        return self.tr(
            "Use a symmetric scan and verify that only the selected moment rotates around the Cartesian axis."
        )

    def create_operation(self):
        return MagneticResponseScanOperation()

    def get_params(self) -> LocalMagneticResponseParams:
        kind = combo_value(self.kind_combo)
        scan_text = self.scan_input.scan_text()
        if kind == "Moment magnitude":
            self._magnitude_scan_text = scan_text
        else:
            self._rotation_scan_text = scan_text
        return LocalMagneticResponseParams(
            response_kind=kind,
            coordinate_scan_deg=self._rotation_scan_text,
            target_mode=combo_value(self.target_mode_combo),
            target_indices=self.target_edit.text(),
            pair_source=combo_value(self.pair_source_combo),
            pair_left_indices=self.left_edit.text(),
            pair_right_indices=self.right_edit.text(),
            pair_shell=int(self.shell_frame.get_input_value()[0]),
            pair_shell_tolerance=float(self.pair_tol_frame.get_input_value()[0]),
            pair_element_filter=self.pair_element_edit.text(),
            pair_group_filter=self.pair_group_edit.text(),
            bond_filter_mode=combo_value(self.bond_mode_combo),
            bond_filter_axis=self.bond_axis_input.vector(),
            bond_filter_tolerance=float(self.bond_tol_frame.get_input_value()[0]),
            group_a=self.group_a_edit.text(),
            group_b=self.group_b_edit.text(),
            rotation_axis=self.axis_input.vector(),
            apply_elements=self.apply_edit.text(),
            moment_scale_scan=self._magnitude_scan_text,
            max_outputs=int(self.limit_frame.get_input_value()[0]),
        )

    def set_params(self, params: LocalMagneticResponseParams):
        set_combo_value(self.kind_combo, params.response_kind)
        self._rotation_scan_text = params.coordinate_scan_deg
        self._magnitude_scan_text = params.moment_scale_scan
        self._active_kind = params.response_kind
        self.scan_input.set_scan_text(
            params.moment_scale_scan if params.response_kind == "Moment magnitude" else params.coordinate_scan_deg
        )
        set_combo_value(self.target_mode_combo, params.target_mode)
        self.target_edit.setText(params.target_indices)
        set_combo_value(self.pair_source_combo, params.pair_source)
        self.left_edit.setText(params.pair_left_indices)
        self.right_edit.setText(params.pair_right_indices)
        self.shell_frame.set_input_value([params.pair_shell])
        self.pair_tol_frame.set_input_value([params.pair_shell_tolerance])
        self.pair_element_edit.setText(params.pair_element_filter)
        self.pair_group_edit.setText(params.pair_group_filter)
        set_combo_value(self.bond_mode_combo, params.bond_filter_mode)
        self.bond_axis_input.set_vector(params.bond_filter_axis)
        self.bond_tol_frame.set_input_value([params.bond_filter_tolerance])
        self.pair_filters_checkbox.setChecked(
            params.pair_shell_tolerance != 0.05
            or bool(params.pair_element_filter.strip())
            or bool(params.pair_group_filter.strip())
            or params.bond_filter_mode != "Any"
            or tuple(params.bond_filter_axis) != (0.0, 0.0, 1.0)
            or params.bond_filter_tolerance != 20.0
        )
        self.group_a_edit.setText(params.group_a)
        self.group_b_edit.setText(params.group_b)
        self.axis_input.set_vector(params.rotation_axis)
        self.apply_edit.setText(params.apply_elements)
        self.limit_frame.set_input_value([params.max_outputs])
        self.advanced_checkbox.setChecked(
            tuple(params.rotation_axis) != (0.0, 1.0, 0.0) or params.max_outputs != 100
        )
        self._update_widgets()

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.set_params(LocalMagneticResponseParams(**data_dict.get("params", {})))
