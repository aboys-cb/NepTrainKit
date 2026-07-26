"""Card for explicit, unrelaxed GSFE displacement paths."""

from __future__ import annotations

from PySide6.QtCore import Qt
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    ToolTipFilter,
    ToolTipPosition,
)

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import StrictGSFEPathOperation, StrictGSFEPathParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame
from .i18n_utils import add_translated_items, combo_value, set_combo_value


@CardManager.register_card
class StrictGSFEPathCard(MakeDataCard):
    """Generate an unrelaxed GSFE path from explicit plane and slip geometry."""

    group = "Defect"
    card_name = "Stacking Fault / GSFE Path"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self.setTitle(self.tr("Stacking Fault / GSFE Path"))
        self.init_ui()

    def init_ui(self):
        """Build slip geometry, displacement, cut, and preview controls."""
        self.setObjectName("strict_gsfe_path_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)
        self.settingLayout.setColumnStretch(1, 1)

        self.description_label = CaptionLabel(
            self.tr(
                "Shift all atoms above an interlayer cut along an explicit in-plane direction. The input cell must already be oriented with its third vector normal to the fault plane."
            ),
            self.setting_widget,
        )
        self.description_label.setWordWrap(True)

        self.hkl_label = BodyLabel(
            self.tr("Fault plane (h k l)"),
            self.setting_widget,
        )
        self.hkl_label.setToolTip(
            self.tr(
                "Plane indices in the current cell; the third cell vector must be normal to this plane"
            )
        )
        self._install_tooltip(self.hkl_label)
        self.hkl_frame = SpinBoxUnitInputFrame(self)
        self.hkl_frame.set_input("", 3, "int")
        self.hkl_frame.setRange(-10, 10)
        self.hkl_frame.set_input_value([0, 0, 1])
        self.hkl_frame.setAccessibleName(self.tr("Fault plane (h k l)"))

        self.uvw_label = BodyLabel(
            self.tr("Slip direction [u v w]"),
            self.setting_widget,
        )
        self.uvw_label.setToolTip(
            self.tr(
                "Direction in the current cell basis; it must lie in the fault plane"
            )
        )
        self._install_tooltip(self.uvw_label)
        self.uvw_frame = SpinBoxUnitInputFrame(self)
        self.uvw_frame.set_input("", 3, "int")
        self.uvw_frame.setRange(-10, 10)
        self.uvw_frame.set_input_value([1, 0, 0])
        self.uvw_frame.setAccessibleName(self.tr("Slip direction [u v w]"))

        self.disp_label = BodyLabel(
            self.tr("Displacement range"),
            self.setting_widget,
        )
        self.disp_label.setToolTip(
            self.tr("Start, end, and positive step; both endpoints are included")
        )
        self._install_tooltip(self.disp_label)
        self.disp_frame = SpinBoxUnitInputFrame(self)
        self.disp_frame.set_input(["–", self.tr("step"), ""], 3, "float")
        self.disp_frame.setDecimals(4)
        self.disp_frame.setRange(-100.0, 100.0)
        self.disp_frame.set_input_value([0.0, 1.0, 0.5])
        self.disp_frame.setAccessibleName(self.tr("Displacement range"))

        self.unit_label = BodyLabel(
            self.tr("Displacement unit"),
            self.setting_widget,
        )
        self.unit_label.setToolTip(
            self.tr(
                "Use a fraction of the slip vector for a periodic path, or an actual distance in angstrom"
            )
        )
        self._install_tooltip(self.unit_label)
        self.unit_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.unit_combo,
            [
                ("fraction_of_vector", "Fraction of slip vector"),
                ("angstrom", "Actual distance (Å)"),
            ],
        )
        self.unit_combo.setAccessibleName(self.tr("Displacement unit"))

        self.cut_label = BodyLabel(
            self.tr("Cut position"),
            self.setting_widget,
        )
        self.cut_label.setToolTip(
            self.tr("Atoms above this interlayer cut are displaced together")
        )
        self._install_tooltip(self.cut_label)
        self.cut_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.cut_combo,
            [
                ("middle", "Middle interlayer gap"),
                ("fractional", "Fraction of projected thickness"),
                ("layer_index", "After a layer index"),
            ],
        )
        self.cut_combo.setAccessibleName(self.tr("Cut position"))

        self.cut_fraction_label = BodyLabel(
            self.tr("Thickness fraction"),
            self.setting_widget,
        )
        self.cut_fraction_label.setToolTip(
            self.tr("0 is the lowest projected atom plane and 1 is the highest")
        )
        self._install_tooltip(self.cut_fraction_label)
        self.cut_fraction_frame = SpinBoxUnitInputFrame(self)
        self.cut_fraction_frame.set_input("", 1, "float")
        self.cut_fraction_frame.setDecimals(4)
        self.cut_fraction_frame.setRange(0.0, 0.9999)
        self.cut_fraction_frame.set_input_value([0.5])
        self.cut_fraction_frame.setAccessibleName(self.tr("Thickness fraction"))

        self.layer_label = BodyLabel(
            self.tr("Lower layer index"),
            self.setting_widget,
        )
        self.layer_label.setToolTip(
            self.tr(
                "Zero-based projected layer; the cut is placed between this layer and the next"
            )
        )
        self._install_tooltip(self.layer_label)
        self.layer_frame = SpinBoxUnitInputFrame(self)
        self.layer_frame.set_input("", 1, "int")
        self.layer_frame.setRange(0, 999999)
        self.layer_frame.set_input_value([0])
        self.layer_frame.setAccessibleName(self.tr("Lower layer index"))

        self.wrap_checkbox = CheckBox(
            self.tr("Wrap displaced atoms into the cell"),
            self.setting_widget,
        )
        self.wrap_checkbox.setChecked(True)
        self.wrap_checkbox.setToolTip(
            self.tr(
                "Disable only when you need to inspect the unwrapped Cartesian displacement"
            )
        )
        self._install_tooltip(self.wrap_checkbox)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("strictGsfePathPreview")

        self.settingLayout.addWidget(self.description_label, 0, 0, 1, 3)
        rows = (
            (self.hkl_label, self.hkl_frame),
            (self.uvw_label, self.uvw_frame),
            (self.disp_label, self.disp_frame),
            (self.unit_label, self.unit_combo),
            (self.cut_label, self.cut_combo),
            (self.cut_fraction_label, self.cut_fraction_frame),
            (self.layer_label, self.layer_frame),
        )
        for row, (label, control) in enumerate(rows, start=1):
            self.settingLayout.addWidget(label, row, 0, 1, 1)
            self.settingLayout.addWidget(control, row, 1, 1, 2)
        self.settingLayout.addWidget(self.wrap_checkbox, len(rows) + 1, 0, 1, 3)
        self.settingLayout.addWidget(
            self.preview_label,
            len(rows) + 2,
            0,
            1,
            3,
        )

        self.cut_combo.currentIndexChanged.connect(self._update_cut_visibility)
        self.unit_combo.currentIndexChanged.connect(self._refresh_preview)
        for control in (
            *self.hkl_frame.object_list,
            *self.uvw_frame.object_list,
            *self.disp_frame.object_list,
            *self.cut_fraction_frame.object_list,
            *self.layer_frame.object_list,
        ):
            control.valueChanged.connect(self._refresh_preview)

        self._update_cut_visibility()
        self._refresh_preview()

    @staticmethod
    def _install_tooltip(widget) -> None:
        widget.installEventFilter(
            ToolTipFilter(widget, 300, ToolTipPosition.TOP)
        )

    def _update_cut_visibility(self, *_args) -> None:
        mode = combo_value(self.cut_combo)
        is_fractional = mode == "fractional"
        is_layer_index = mode == "layer_index"
        self.cut_fraction_label.setVisible(is_fractional)
        self.cut_fraction_frame.setVisible(is_fractional)
        self.layer_label.setVisible(is_layer_index)
        self.layer_frame.setVisible(is_layer_index)
        self._update_tab_order()
        self._refresh_preview()

    @staticmethod
    def _first_structure(dataset):
        if dataset is None:
            return None
        if hasattr(dataset, "arrays") and hasattr(dataset, "get_chemical_symbols"):
            return dataset
        try:
            return next(iter(dataset))
        except (StopIteration, TypeError):
            return None

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._input_structure = self._first_structure(dataset)
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr(
                    "Load an upstream slab-oriented structure to preview the cut and output count."
                )
            )
            return
        try:
            summary = self.create_operation().geometry_summary(
                self._input_structure,
                self.get_params(),
            )
        except (TypeError, ValueError, IndexError) as exc:
            self.preview_label.setText(
                "⚠ "
                + self.tr("Preview unavailable: {error}").format(
                    error=translate_runtime_message(exc)
                )
            )
            return

        self.preview_label.setText(
            self.tr(
                "First input: {atoms} atoms / {layers} projected layers · move {moved}, keep {stationary} · slip vector length {length} Å · {outputs} outputs"
            ).format(
                atoms=summary["atom_count"],
                layers=summary["layer_count"],
                moved=summary["moved_count"],
                stationary=summary["stationary_count"],
                length=f"{summary['slip_length']:.4g}",
                outputs=summary["output_count"],
            )
        )

    def _update_tab_order(self) -> None:
        if not hasattr(self, "cut_combo"):
            return
        widgets = [
            *self.hkl_frame.object_list,
            *self.uvw_frame.object_list,
            *self.disp_frame.object_list,
            self.unit_combo,
            self.cut_combo,
        ]
        mode = combo_value(self.cut_combo)
        if mode == "fractional":
            widgets.extend(self.cut_fraction_frame.object_list)
        elif mode == "layer_index":
            widgets.extend(self.layer_frame.object_list)
        widgets.append(self.wrap_checkbox)
        self.tab_order_widgets = widgets

    def create_operation(self):
        return StrictGSFEPathOperation()

    def get_params(self) -> StrictGSFEPathParams:
        return StrictGSFEPathParams(
            plane_hkl=tuple(int(v) for v in self.hkl_frame.get_input_value()),
            slip_uvw=tuple(int(v) for v in self.uvw_frame.get_input_value()),
            displacement_range=tuple(float(v) for v in self.disp_frame.get_input_value()),
            displacement_unit=combo_value(self.unit_combo),
            cut_mode=combo_value(self.cut_combo),
            cut_fraction=float(self.cut_fraction_frame.get_input_value()[0]),
            layer_index=int(self.layer_frame.get_input_value()[0]),
            wrap=self.wrap_checkbox.isChecked(),
        )

    def set_params(self, params: StrictGSFEPathParams) -> None:
        self.hkl_frame.set_input_value([int(v) for v in params.plane_hkl])
        self.uvw_frame.set_input_value([int(v) for v in params.slip_uvw])
        self.disp_frame.set_input_value([float(v) for v in params.displacement_range])
        set_combo_value(self.unit_combo, params.displacement_unit)
        set_combo_value(self.cut_combo, params.cut_mode)
        self.cut_fraction_frame.set_input_value([float(params.cut_fraction)])
        self.layer_frame.set_input_value([int(params.layer_index)])
        self.wrap_checkbox.setChecked(bool(params.wrap))
        self._update_cut_visibility()
        self._refresh_preview()

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.set_params(StrictGSFEPathParams(**data_dict.get("params", {})))
