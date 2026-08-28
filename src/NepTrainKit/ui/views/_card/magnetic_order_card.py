"""Card for generating FM, AFM, and random PM endpoint spin states."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    LineEdit,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.magnetism import MagneticOrderOperation, MagneticOrderParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items, combo_value, set_combo_value
from NepTrainKit.ui.widgets import KeyValueTableInput, MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class MagneticOrderCard(MakeDataCard):
    """Generate common magnetic-order endpoints without changing atomic geometry."""

    group = "Magnetism"
    card_name = "Magnetic Order"
    menu_icon = r":/images/src/images/perturb.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._input_count = 0
        self.setTitle(self.tr("Magnetic Order"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("magnetic_order_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.setColumnStretch(1, 1)

        self.format_label = BodyLabel(self.tr("Spin model"), self.setting_widget)
        self.format_label.setToolTip(
            self.tr("Collinear uses only +/- along the reference axis; non-collinear allows 3D directions")
        )
        self._install_tooltip(self.format_label)
        self.format_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.format_combo,
            [
                ("collinear", "Collinear along reference axis"),
                ("noncollinear", "Non-collinear 3D vectors"),
            ],
        )
        self._size_control(self.format_combo, minimum=280)

        self.axis_label = BodyLabel(self.tr("Reference axis"), self.setting_widget)
        self.axis_label.setToolTip(
            self.tr("Direction used by collinear moments and as the reference for non-collinear PM distributions")
        )
        self._install_tooltip(self.axis_label)
        self.axis_frame = SpinBoxUnitInputFrame(self)
        self.axis_frame.set_input("", 3, "float")
        self.axis_frame.setDecimals(6)
        self.axis_frame.setRange(-1.0, 1.0)
        self.axis_frame.set_input_value([0.0, 0.0, 1.0])

        self.map_label = BodyLabel(self.tr("Element moments (μB)"), self.setting_widget)
        self.map_label.setToolTip(
            self.tr('Moment magnitudes such as "Fe:2.2,Co:1.7"; vector values such as Cr:[0,0,1] are also accepted')
        )
        self._install_tooltip(self.map_label)
        self.map_edit = KeyValueTableInput(
            self.tr("Element"), self.tr("Moment magnitude"), self.setting_widget
        )
        self.map_edit.setAccessibleName(self.tr("Element moments (μB)"))
        self._size_control(self.map_edit)

        self.use_element_dir_checkbox = CheckBox(
            self.tr("Use directions from vector-valued element entries"),
            self.setting_widget,
        )
        self.use_element_dir_checkbox.setChecked(False)
        self.use_element_dir_checkbox.setToolTip(
            self.tr("Non-collinear FM/AFM will use each vector entry as its element reference direction")
        )
        self._install_tooltip(self.use_element_dir_checkbox)

        self.default_label = BodyLabel(
            self.tr("Unlisted element |m| (μB)"),
            self.setting_widget,
        )
        self.default_label.setToolTip(
            self.tr("Moment magnitude for selected elements not present in the element map")
        )
        self._install_tooltip(self.default_label)
        self.default_frame = SpinBoxUnitInputFrame(self)
        self.default_frame.set_input("", 1, "float")
        self.default_frame.setDecimals(6)
        self.default_frame.setRange(0.0, 20.0)
        self.default_frame.set_input_value([0.0])

        self.apply_label = BodyLabel(self.tr("Apply only to elements"), self.setting_widget)
        self.apply_label.setToolTip(
            self.tr("Optional comma-separated element list; leave empty to consider all elements")
        )
        self._install_tooltip(self.apply_label)
        self.apply_edit = LineEdit(self.setting_widget)
        self.apply_edit.setPlaceholderText(self.tr("Fe, Co, Ni"))
        self.apply_edit.setAccessibleName(self.tr("Apply only to elements"))
        self._size_control(self.apply_edit)

        self.fm_checkbox = CheckBox(self.tr("Generate FM"), self.setting_widget)
        self.fm_checkbox.setChecked(True)
        self.afm_checkbox = CheckBox(self.tr("Generate AFM"), self.setting_widget)
        self.afm_checkbox.setChecked(False)
        self.pm_checkbox = CheckBox(self.tr("Generate random PM"), self.setting_widget)
        self.pm_checkbox.setChecked(False)

        self.afm_mode_label = BodyLabel(self.tr("AFM assignment"), self.setting_widget)
        self.afm_mode_label.setToolTip(
            self.tr("Assign opposite signs by fractional-coordinate layers or existing group labels")
        )
        self._install_tooltip(self.afm_mode_label)
        self.afm_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.afm_mode_combo,
            [
                ("k_vector", "Coordinate-layer k-vector"),
                ("group_ab", "Existing group labels"),
            ],
        )
        self._size_control(self.afm_mode_combo)

        self.kvec_label = BodyLabel(self.tr("AFM layer vector"), self.setting_widget)
        self.kvec_label.setToolTip(
            self.tr("Fractional-coordinate phase direction; inspect the preview to confirm both signs occur")
        )
        self._install_tooltip(self.kvec_label)
        self.kvec_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.kvec_combo,
            [
                ("100", "100 (along lattice a)"),
                ("010", "010 (along lattice b)"),
                ("001", "001 (along lattice c)"),
                ("110", "110 (along lattice a+b)"),
                ("111", "111 (along lattice a+b+c)"),
            ],
        )
        set_combo_value(self.kvec_combo, "111")
        self._size_control(self.kvec_combo)

        self.group_a_label = BodyLabel(self.tr("Positive group label"), self.setting_widget)
        self.group_a_edit = LineEdit(self.setting_widget)
        self.group_a_edit.setText("A")
        self.group_a_edit.setAccessibleName(self.tr("Positive group label"))
        self._size_control(self.group_a_edit)

        self.group_b_label = BodyLabel(self.tr("Negative group label"), self.setting_widget)
        self.group_b_edit = LineEdit(self.setting_widget)
        self.group_b_edit.setText("B")
        self.group_b_edit.setAccessibleName(self.tr("Negative group label"))
        self._size_control(self.group_b_edit)

        self.zero_unknown_groups_checkbox = CheckBox(
            self.tr("Set other groups to zero moment"),
            self.setting_widget,
        )
        self.zero_unknown_groups_checkbox.setChecked(True)

        self.pm_count_label = BodyLabel(self.tr("PM structures per input"), self.setting_widget)
        self.pm_count_frame = SpinBoxUnitInputFrame(self)
        self.pm_count_frame.set_input("", 1, "int")
        self.pm_count_frame.setRange(1, 999999)
        self.pm_count_frame.set_input_value([10])

        self.pm_direction_label = BodyLabel(self.tr("PM direction distribution"), self.setting_widget)
        self.pm_direction_label.setToolTip(
            self.tr("Used only by non-collinear PM; collinear PM always samples +/- along the reference axis")
        )
        self._install_tooltip(self.pm_direction_label)
        self.pm_direction_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.pm_direction_combo,
            [
                ("sphere", "Full sphere"),
                ("cone", "Cone around reference axis"),
                ("plane", "Plane perpendicular to reference axis"),
                ("axis", "Along +/- reference axis"),
            ],
        )
        self._size_control(self.pm_direction_combo)

        self.pm_cone_label = BodyLabel(self.tr("PM cone half-angle"), self.setting_widget)
        self.pm_cone_label.setToolTip(
            self.tr("When opposite pairing is on, cone directions are paired around both +/- reference axes")
        )
        self._install_tooltip(self.pm_cone_label)
        self.pm_cone_frame = SpinBoxUnitInputFrame(self)
        self.pm_cone_frame.set_input("deg", 1, "float")
        self.pm_cone_frame.setDecimals(3)
        self.pm_cone_frame.setRange(0.0, 180.0)
        self.pm_cone_frame.set_input_value([30.0])

        self.pm_balanced_checkbox = CheckBox(
            self.tr("Pair opposite PM directions by magnitude"),
            self.setting_widget,
        )
        self.pm_balanced_checkbox.setChecked(True)
        self.pm_balanced_checkbox.setToolTip(
            self.tr("Complete equal-magnitude pairs cancel exactly; odd groups may leave one residual moment")
        )
        self._install_tooltip(self.pm_balanced_checkbox)

        self.seed_checkbox = CheckBox(self.tr("Use random seed"), self.setting_widget)
        self.seed_checkbox.setChecked(False)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("", 1, "int")
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.set_input_value([0])

        self.max_output_label = BodyLabel(self.tr("Maximum outputs per input"), self.setting_widget)
        self.max_output_label.setToolTip(
            self.tr("Reject the run when FM + AFM + PM outputs exceed this budget")
        )
        self._install_tooltip(self.max_output_label)
        self.max_output_frame = SpinBoxUnitInputFrame(self)
        self.max_output_frame.set_input("", 1, "int")
        self.max_output_frame.setRange(1, 999999)
        self.max_output_frame.set_input_value([100])

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("magneticOrderPreview")

        self.settingLayout.addWidget(self.format_label, 0, 0, 1, 1)
        self.settingLayout.addWidget(self.format_combo, 0, 1, 1, 2)
        self.settingLayout.addWidget(self.axis_label, 1, 0, 1, 1)
        self.settingLayout.addWidget(self.axis_frame, 1, 1, 1, 2)
        self.settingLayout.addWidget(self.map_label, 2, 0, 1, 1)
        self.settingLayout.addWidget(self.map_edit, 2, 1, 1, 2)
        self.settingLayout.addWidget(self.use_element_dir_checkbox, 3, 0, 1, 3)
        self.settingLayout.addWidget(self.default_label, 4, 0, 1, 1)
        self.settingLayout.addWidget(self.default_frame, 4, 1, 1, 2)
        self.settingLayout.addWidget(self.apply_label, 5, 0, 1, 1)
        self.settingLayout.addWidget(self.apply_edit, 5, 1, 1, 2)
        self.settingLayout.addWidget(self.fm_checkbox, 6, 0, 1, 1)
        self.settingLayout.addWidget(self.afm_checkbox, 6, 1, 1, 1)
        self.settingLayout.addWidget(self.pm_checkbox, 6, 2, 1, 1)
        self.settingLayout.addWidget(self.afm_mode_label, 7, 0, 1, 1)
        self.settingLayout.addWidget(self.afm_mode_combo, 7, 1, 1, 2)
        self.settingLayout.addWidget(self.kvec_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.kvec_combo, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.group_a_label, 8, 0, 1, 1)
        self.settingLayout.addWidget(self.group_a_edit, 8, 1, 1, 2)
        self.settingLayout.addWidget(self.group_b_label, 9, 0, 1, 1)
        self.settingLayout.addWidget(self.group_b_edit, 9, 1, 1, 2)
        self.settingLayout.addWidget(self.zero_unknown_groups_checkbox, 10, 0, 1, 3)
        self.settingLayout.addWidget(self.pm_count_label, 11, 0, 1, 1)
        self.settingLayout.addWidget(self.pm_count_frame, 11, 1, 1, 2)
        self.settingLayout.addWidget(self.pm_direction_label, 12, 0, 1, 1)
        self.settingLayout.addWidget(self.pm_direction_combo, 12, 1, 1, 2)
        self.settingLayout.addWidget(self.pm_cone_label, 13, 0, 1, 1)
        self.settingLayout.addWidget(self.pm_cone_frame, 13, 1, 1, 2)
        self.settingLayout.addWidget(self.pm_balanced_checkbox, 14, 0, 1, 3)
        self.settingLayout.addWidget(self.seed_checkbox, 15, 0, 1, 1)
        self.settingLayout.addWidget(self.seed_frame, 15, 1, 1, 2)
        self.settingLayout.addWidget(self.max_output_label, 16, 0, 1, 1)
        self.settingLayout.addWidget(self.max_output_frame, 16, 1, 1, 2)
        self.settingLayout.addWidget(self.preview_label, 17, 0, 1, 3)

        self.format_combo.currentIndexChanged.connect(self._update_dynamic_widgets)
        self.afm_checkbox.stateChanged.connect(self._update_dynamic_widgets)
        self.pm_checkbox.stateChanged.connect(self._update_dynamic_widgets)
        self.afm_mode_combo.currentIndexChanged.connect(self._update_dynamic_widgets)
        self.pm_direction_combo.currentIndexChanged.connect(self._update_dynamic_widgets)
        self.seed_checkbox.stateChanged.connect(self._update_dynamic_widgets)
        self.fm_checkbox.stateChanged.connect(self._refresh_preview)
        self.kvec_combo.currentIndexChanged.connect(self._refresh_preview)
        self.use_element_dir_checkbox.stateChanged.connect(self._refresh_preview)
        self.zero_unknown_groups_checkbox.stateChanged.connect(self._refresh_preview)
        self.pm_balanced_checkbox.stateChanged.connect(self._refresh_preview)
        self.map_edit.editingFinished.connect(self._refresh_preview)
        self.apply_edit.editingFinished.connect(self._refresh_preview)
        self.group_a_edit.editingFinished.connect(self._refresh_preview)
        self.group_b_edit.editingFinished.connect(self._refresh_preview)
        for frame in (
            self.axis_frame,
            self.default_frame,
            self.pm_count_frame,
            self.pm_cone_frame,
            self.seed_frame,
            self.max_output_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._refresh_preview)

        self._update_dynamic_widgets()

    @staticmethod
    def _install_tooltip(widget) -> None:
        widget.installEventFilter(ToolTipFilter(widget, 300, ToolTipPosition.TOP))

    @staticmethod
    def _size_control(widget, *, minimum: int = 0) -> None:
        if minimum:
            widget.setMinimumWidth(minimum)
        widget.setMaximumWidth(380)
        widget.setFixedHeight(28)

    @staticmethod
    def _set_visible(visible: bool, *widgets) -> None:
        for widget in widgets:
            widget.setVisible(visible)
            widget.setEnabled(visible)

    def _update_dynamic_widgets(self, *_args) -> None:
        noncollinear = combo_value(self.format_combo) == "noncollinear"
        afm_enabled = self.afm_checkbox.isChecked()
        use_groups = combo_value(self.afm_mode_combo) == "group_ab"
        pm_enabled = self.pm_checkbox.isChecked()
        use_cone = combo_value(self.pm_direction_combo) == "cone"

        self._set_visible(noncollinear, self.use_element_dir_checkbox)
        self._set_visible(
            afm_enabled,
            self.afm_mode_label,
            self.afm_mode_combo,
        )
        self._set_visible(
            afm_enabled and not use_groups,
            self.kvec_label,
            self.kvec_combo,
        )
        self._set_visible(
            afm_enabled and use_groups,
            self.group_a_label,
            self.group_a_edit,
            self.group_b_label,
            self.group_b_edit,
            self.zero_unknown_groups_checkbox,
        )
        self._set_visible(
            pm_enabled,
            self.pm_count_label,
            self.pm_count_frame,
            self.pm_balanced_checkbox,
            self.seed_checkbox,
            self.seed_frame,
            self.max_output_label,
            self.max_output_frame,
        )
        self._set_visible(
            pm_enabled and noncollinear,
            self.pm_direction_label,
            self.pm_direction_combo,
        )
        self._set_visible(
            pm_enabled and noncollinear and use_cone,
            self.pm_cone_label,
            self.pm_cone_frame,
        )
        self.seed_frame.setEnabled(pm_enabled and self.seed_checkbox.isChecked())
        self._update_tab_order()
        self._refresh_preview()

    @staticmethod
    def _first_structure(dataset):
        if dataset is None:
            return None
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_scaled_positions"):
            return dataset
        try:
            return next(iter(dataset))
        except (StopIteration, TypeError):
            return None

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        try:
            self._input_count = len(dataset)
        except TypeError:
            self._input_count = int(self._input_structure is not None)
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr("Load an upstream structure to preview magnetic atoms and output count.")
            )
            self.refresh_compact_presentation()
            return
        try:
            summary = self.create_operation().preview(
                self._input_structure,
                self.get_params(),
            )
        except (TypeError, ValueError) as exc:
            self.preview_label.setText(
                "⚠ " + self.tr("Preview unavailable: {error}").format(error=str(exc))
            )
            self.refresh_compact_presentation()
            return

        total = summary.output_count * max(self._input_count, 1)
        message = self.tr(
            "First input preview: magnetic atoms {magnetic}/{atoms} · outputs/input {outputs} · total {total}"
        ).format(
            magnetic=summary.magnetic_atoms,
            atoms=len(self._input_structure),
            outputs=summary.output_count,
            total=total,
        )
        if self.afm_checkbox.isChecked():
            message += " · " + self.tr("AFM signs: +{positive} / -{negative} / 0={zero}").format(
                positive=summary.afm_positive,
                negative=summary.afm_negative,
                zero=summary.afm_zero,
            )
        self.preview_label.setText(message)
        self.refresh_compact_presentation()

    def _update_tab_order(self) -> None:
        widgets = [
            self.format_combo,
            *self.axis_frame.object_list,
            self.map_edit,
            self.use_element_dir_checkbox,
            *self.default_frame.object_list,
            self.apply_edit,
            self.fm_checkbox,
            self.afm_checkbox,
            self.pm_checkbox,
            self.afm_mode_combo,
            self.kvec_combo,
            self.group_a_edit,
            self.group_b_edit,
            self.zero_unknown_groups_checkbox,
            *self.pm_count_frame.object_list,
            self.pm_direction_combo,
            *self.pm_cone_frame.object_list,
            self.pm_balanced_checkbox,
            self.seed_checkbox,
            *self.seed_frame.object_list,
            *self.max_output_frame.object_list,
        ]
        self.tab_order_widgets = [
            widget
            for widget in widgets
            if not widget.isHidden() and widget.isEnabled()
        ]
        for previous, current in zip(self.tab_order_widgets, self.tab_order_widgets[1:]):
            QWidget.setTabOrder(previous, current)

    def create_operation(self):
        return MagneticOrderOperation()

    def get_summary_text(self) -> str:
        orders = []
        if self.fm_checkbox.isChecked():
            orders.append("FM")
        if self.afm_checkbox.isChecked():
            orders.append("AFM")
        if self.pm_checkbox.isChecked():
            orders.append("PM")
        order_text = " + ".join(orders) or self.tr("no outputs")
        output_count = int(self.fm_checkbox.isChecked()) + int(
            self.afm_checkbox.isChecked()
        )
        if self.pm_checkbox.isChecked():
            output_count += int(self.pm_count_frame.get_input_value()[0])
        return self.tr("{orders} · {model} · n={count}").format(
            orders=order_text,
            model=self.format_combo.currentText(),
            count=output_count,
        )

    def get_guidance_text(self) -> str:
        if not any(
            checkbox.isChecked()
            for checkbox in (self.fm_checkbox, self.afm_checkbox, self.pm_checkbox)
        ):
            return self.tr("Select at least one magnetic order to generate output.")
        if self.afm_checkbox.isChecked():
            if combo_value(self.afm_mode_combo) == "group_ab":
                return self.tr(
                    "AFM uses existing group labels; confirm that both configured labels occur in the input."
                )
            return self.tr(
                "Inspect the AFM sign preview to confirm that the selected lattice-layer vector produces both signs."
            )
        if self.pm_checkbox.isChecked():
            return self.tr(
                "Random PM creates the configured number of structures for each input. "
                "Use a seed when the same directions must be reproduced."
            )
        return self.tr("FM creates one aligned magnetic structure for each input.")

    def get_params(self) -> MagneticOrderParams:
        return MagneticOrderParams(
            format=combo_value(self.format_combo),
            axis=self.axis_frame.get_input_value(),
            magmom_map=self.map_edit.text(),
            use_element_dirs=self.use_element_dir_checkbox.isChecked(),
            default_moment=float(self.default_frame.get_input_value()[0]),
            apply_elements=self.apply_edit.text(),
            gen_fm=self.fm_checkbox.isChecked(),
            gen_afm=self.afm_checkbox.isChecked(),
            afm_mode=combo_value(self.afm_mode_combo),
            afm_kvec=combo_value(self.kvec_combo),
            afm_group_a=self.group_a_edit.text(),
            afm_group_b=self.group_b_edit.text(),
            afm_zero_unknown=self.zero_unknown_groups_checkbox.isChecked(),
            gen_pm=self.pm_checkbox.isChecked(),
            pm_count=int(self.pm_count_frame.get_input_value()[0]),
            pm_direction=combo_value(self.pm_direction_combo),
            pm_cone_angle=float(self.pm_cone_frame.get_input_value()[0]),
            pm_balanced=self.pm_balanced_checkbox.isChecked(),
            use_seed=self.seed_checkbox.isChecked(),
            seed=int(self.seed_frame.get_input_value()[0]),
            max_outputs=int(self.max_output_frame.get_input_value()[0]),
        )

    def set_params(self, params: MagneticOrderParams) -> None:
        set_combo_value(
            self.format_combo,
            MagneticOrderOperation.normalize_format(params.format),
        )
        self.axis_frame.set_input_value([float(value) for value in params.axis])
        self.map_edit.setText(params.magmom_map)
        self.use_element_dir_checkbox.setChecked(bool(params.use_element_dirs))
        self.default_frame.set_input_value([float(params.default_moment)])
        self.apply_edit.setText(params.apply_elements)
        self.fm_checkbox.setChecked(bool(params.gen_fm))
        self.afm_checkbox.setChecked(bool(params.gen_afm))
        set_combo_value(
            self.afm_mode_combo,
            MagneticOrderOperation.normalize_afm_mode(params.afm_mode),
        )
        set_combo_value(self.kvec_combo, params.afm_kvec)
        self.group_a_edit.setText(params.afm_group_a)
        self.group_b_edit.setText(params.afm_group_b)
        self.zero_unknown_groups_checkbox.setChecked(bool(params.afm_zero_unknown))
        self.pm_checkbox.setChecked(bool(params.gen_pm))
        self.pm_count_frame.set_input_value([int(params.pm_count)])
        set_combo_value(self.pm_direction_combo, str(params.pm_direction).lower())
        self.pm_cone_frame.set_input_value([float(params.pm_cone_angle)])
        self.pm_balanced_checkbox.setChecked(bool(params.pm_balanced))
        self.seed_checkbox.setChecked(bool(params.use_seed))
        self.seed_frame.set_input_value([int(params.seed)])
        self.max_output_frame.set_input_value([int(params.max_outputs)])
        self._update_dynamic_widgets()

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
            params = MagneticOrderParams(**raw_params)
        else:
            params = MagneticOrderParams(
                format=data_dict.get("format", "Collinear (scalar)"),
                axis=data_dict.get("axis", [0.0, 0.0, 1.0]),
                magmom_map=data_dict.get("magmom_map", ""),
                use_element_dirs=data_dict.get("use_element_dirs", False),
                default_moment=self._legacy_scalar(
                    data_dict.get("default_moment", 0.0),
                    0.0,
                ),
                apply_elements=data_dict.get("apply_elements", ""),
                gen_fm=data_dict.get("gen_fm", True),
                gen_afm=data_dict.get("gen_afm", True),
                afm_mode=data_dict.get("afm_mode", "k-vector"),
                afm_kvec=data_dict.get("afm_kvec", "111"),
                afm_group_a=data_dict.get("afm_group_a", "A"),
                afm_group_b=data_dict.get("afm_group_b", "B"),
                afm_zero_unknown=data_dict.get("afm_zero_unknown", True),
                gen_pm=data_dict.get("gen_pm", False),
                pm_count=int(self._legacy_scalar(data_dict.get("pm_count", 10), 10)),
                pm_direction=data_dict.get("pm_direction", "sphere"),
                pm_cone_angle=self._legacy_scalar(
                    data_dict.get("pm_cone_angle", 30.0),
                    30.0,
                ),
                pm_balanced=data_dict.get("pm_balanced", True),
                use_seed=data_dict.get("use_seed", False),
                seed=int(self._legacy_scalar(data_dict.get("seed", 0), 0)),
                max_outputs=int(
                    self._legacy_scalar(data_dict.get("max_outputs", 100), 100)
                ),
            )
        self.set_params(params)

    @staticmethod
    def _legacy_scalar(value, default):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value
