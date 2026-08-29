"""Card for setting or normalising initial magnetic moments."""

from __future__ import annotations

from qfluentwidgets import CheckBox, ComboBox

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.cards.magnetism import SetMagneticMomentsOperation, SetMagneticMomentsParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.widgets import (
    CompactField,
    ElementLineEdit,
    InspectorSection,
    KeyValueTableInput,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)

from .i18n_utils import add_translated_items, combo_value, set_combo_value


@CardManager.register_card
class SetMagneticMomentsCard(MakeDataCard):
    """Set or convert magnetic moments into a consistent scalar/vector representation."""

    group = "Magnetism"
    card_name = "Set Magnetic Moments"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Set Moments"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("set_magnetic_moments_card_widget")

        self.source_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.source_combo,
            ["Existing initial magmoms", "Map/default magnitude", "Constant magnitude"],
        )
        set_combo_value(self.source_combo, "Map/default magnitude")
        self.source_field = CompactField(
            self.tr("Moment source"),
            self.source_combo,
            self.setting_widget,
            self.tr("Reuse input moments, assign by element, or use one magnitude."),
        )
        self.source_label = self.source_field.caption

        self.format_combo = ComboBox(self.setting_widget)
        add_translated_items(self, self.format_combo, ["Collinear (scalar)", "Non-collinear (vector)"])
        set_combo_value(self.format_combo, "Non-collinear (vector)")
        self.format_field = CompactField(
            self.tr("Output representation"),
            self.format_combo,
            self.setting_widget,
            self.tr("Write one scalar or a Cartesian three-component vector per atom."),
        )
        self.format_label = self.format_field.caption

        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "int")
        self.axis_frame.setRange(-99, 99)
        self.axis_frame.set_input_value([0, 0, 1])
        self.axis_field = CompactField(
            self.tr("Reference axis (x, y, z)"),
            self.axis_frame,
            self.setting_widget,
            self.tr("Enter an integer Cartesian direction; it is normalized before use."),
        )
        self.axis_label = self.axis_field.caption

        self.map_edit = KeyValueTableInput(
            self.tr("Element"),
            self.tr("Moment or vector"),
            self.setting_widget,
            element_picker=True,
            new_element_value="1.0",
        )
        self.map_field = CompactField(
            self.tr("Element moments (μB)"),
            self.map_edit,
            self.setting_widget,
            self.tr("For example Fe:2.2, Co:1.7, or Cr:[0,0,1]."),
        )
        self.map_label = self.map_field.caption

        self.use_element_dir_checkbox = CheckBox(self.tr("Use element vector directions"), self.setting_widget)
        self.use_element_dir_checkbox.setChecked(False)
        self.use_element_dir_checkbox.setToolTip(
            self.tr(
                "If the map provides vectors, preserve their directions in vector output"
            )
        )

        self.default_frame = SpinBoxUnitInputFrame(self)
        self.default_frame.set_input("", 1, "float")
        self.default_frame.setRange(0.0, 20.0)
        self.default_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.default_frame.set_input_value([0.0])
        self.default_field = CompactField(
            self.tr("Unlisted element |m| (μB)"),
            self.default_frame,
            self.setting_widget,
            inline=True,
            input_max_width=180,
        )
        self.default_label = self.default_field.caption

        self.constant_frame = SpinBoxUnitInputFrame(self)
        self.constant_frame.set_input("", 1, "float")
        self.constant_frame.setRange(0.0, 20.0)
        self.constant_frame.object_list[0].setDecimals(6)  # pyright: ignore[reportAttributeAccessIssue]
        self.constant_frame.set_input_value([2.0])
        self.constant_field = CompactField(
            self.tr("Constant |m| (μB)"),
            self.constant_frame,
            self.setting_widget,
            inline=True,
            input_max_width=180,
        )
        self.constant_label = self.constant_field.caption

        self.lift_scalar_checkbox = CheckBox(self.tr("Lift scalar magmoms to vectors"), self.setting_widget)
        self.lift_scalar_checkbox.setChecked(True)
        self.lift_scalar_checkbox.setToolTip(
            self.tr(
                "When Source = Existing initial magmoms, lift scalar input onto Axis for vector output"
            )
        )

        self.apply_edit = ElementLineEdit(self.setting_widget, multiple=True)
        self.apply_edit.setPlaceholderText(self.tr("Fe,Co,Ni"))
        self.apply_field = CompactField(
            self.tr("Apply only to elements"),
            self.apply_edit,
            self.setting_widget,
            self.tr("Leave empty to apply to every atom."),
        )
        self.apply_label = self.apply_field.caption

        self.representation_section = InspectorSection(
            self.tr("Source and representation"), self.setting_widget
        )
        representation_form = ResponsiveFormGrid(
            self.representation_section, two_column_threshold=430
        )
        representation_form.add_field(self.source_field, span=2)
        representation_form.add_field(self.format_field, span=2)
        representation_form.add_field(self.axis_field, span=2)
        self.representation_section.addWidget(representation_form)

        self.moments_section = InspectorSection(
            self.tr("Moment values"), self.setting_widget
        )
        moments_form = ResponsiveFormGrid(
            self.moments_section, two_column_threshold=430
        )
        moments_form.add_field(self.map_field, span=2)
        moments_form.add_field(self.use_element_dir_checkbox, span=2)
        moments_form.add_field(self.default_field, span=2)
        moments_form.add_field(self.constant_field, span=2)
        moments_form.add_field(self.lift_scalar_checkbox, span=2)
        self.moments_section.addWidget(moments_form)

        self.scope_section = InspectorSection(self.tr("Scope"), self.setting_widget)
        self.scope_section.addWidget(self.apply_field)
        self.settingLayout.addWidget(self.representation_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(self.moments_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(self.scope_section, 2, 0, 1, 3)

        self.source_combo.currentTextChanged.connect(self._update_source_widgets)
        self.format_combo.currentTextChanged.connect(self._update_source_widgets)
        self.map_edit.editingFinished.connect(self.refresh_compact_presentation)
        self.apply_edit.editingFinished.connect(self.refresh_compact_presentation)
        for checkbox in (
            self.use_element_dir_checkbox,
            self.lift_scalar_checkbox,
        ):
            checkbox.stateChanged.connect(self.refresh_compact_presentation)
        for frame in (self.axis_frame, self.default_frame, self.constant_frame):
            for control in frame.object_list:
                control.valueChanged.connect(self.refresh_compact_presentation)
        self._update_source_widgets()

    def _update_source_widgets(self):
        source = combo_value(self.source_combo)
        vector_output = combo_value(self.format_combo) == "Non-collinear (vector)"

        use_map = source == "Map/default magnitude"
        use_constant = source == "Constant magnitude"
        use_existing = source == "Existing initial magmoms"

        self.map_field.setEnabled(use_map)
        self.map_edit.setEnabled(use_map)
        self.map_field.setVisible(use_map)

        self.use_element_dir_checkbox.setEnabled(use_map and vector_output)
        self.use_element_dir_checkbox.setVisible(use_map)

        self.default_field.setEnabled(use_map)
        self.default_frame.setEnabled(use_map)
        self.default_field.setVisible(use_map)

        self.constant_field.setEnabled(use_constant)
        self.constant_frame.setEnabled(use_constant)
        self.constant_field.setVisible(use_constant)

        self.lift_scalar_checkbox.setEnabled(use_existing and vector_output)
        self.lift_scalar_checkbox.setVisible(use_existing)
        self.refresh_compact_presentation()

    def get_summary_text(self) -> str:
        source = self.source_combo.currentText()
        representation = self.format_combo.currentText()
        scope = self.apply_edit.text().strip() or self.tr("all elements")
        return f"{source} · {representation} · {scope}"

    def get_guidance_text(self) -> str:
        source = combo_value(self.source_combo)
        if source == "Existing initial magmoms":
            return self.tr(
                "The input must contain initial magnetic moments. Check the reference "
                "axis when scalar moments are lifted to vectors."
            )
        if source == "Map/default magnitude":
            return self.tr(
                "List the elements that need explicit moments; the default magnitude "
                "is used for every unlisted selected element."
            )
        return self.tr(
            "The constant magnitude is assigned to every selected element along the reference axis."
        )

    def create_operation(self):
        return SetMagneticMomentsOperation()

    def get_params(self) -> SetMagneticMomentsParams:
        return SetMagneticMomentsParams(
            source=combo_value(self.source_combo),
            format=combo_value(self.format_combo),
            axis=self.axis_frame.get_input_value(),
            magmom_map=self.map_edit.text(),
            use_element_dirs=self.use_element_dir_checkbox.isChecked(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            constant_moment=float(self.constant_frame.get_input_value()[0]),
            lift_scalar=self.lift_scalar_checkbox.isChecked(),
            apply_elements=self.apply_edit.text(),
        )

    def set_params(self, params: SetMagneticMomentsParams) -> None:
        set_combo_value(self.source_combo, params.source)
        set_combo_value(self.format_combo, params.format)
        self.axis_frame.set_input_value([int(round(float(v))) for v in params.axis])
        self.map_edit.setText(params.magmom_map)
        self.use_element_dir_checkbox.setChecked(bool(params.use_element_dirs))
        self.default_frame.set_input_value([float(params.default_moment)])
        self.constant_frame.set_input_value([float(params.constant_moment)])
        self.lift_scalar_checkbox.setChecked(bool(params.lift_scalar))
        self.apply_edit.setText(params.apply_elements)
        self._update_source_widgets()

    def process_structure(self, structure):
        result = self.create_operation().run_structure(structure, self.get_params())
        if (
            len(result) == 1
            and result[0] is not structure
            and "MagSet(" not in str(result[0].info.get("Config_type", ""))
        ):
            MessageManager.send_warning_message("SetMagneticMoments: no usable initial_magmoms found.")
        return result

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if raw_params:
            params = SetMagneticMomentsParams(**raw_params)
        else:
            params = SetMagneticMomentsParams(
                source=data_dict.get("source", "Map/default magnitude"),
                format=data_dict.get("format", "Non-collinear (vector)"),
                axis=data_dict.get("axis", [0.0, 0.0, 1.0]),
                magmom_map=data_dict.get("magmom_map", ""),
                use_element_dirs=data_dict.get("use_element_dirs", False),
                default_moment=data_dict.get("default_moment", [0.0])[0],
                constant_moment=data_dict.get("constant_moment", [2.0])[0],
                lift_scalar=data_dict.get("lift_scalar", True),
                apply_elements=data_dict.get("apply_elements", ""),
            )
        self.set_params(params)
