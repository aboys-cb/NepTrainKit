"""Card for explicit, unrelaxed GSFE displacement paths."""

from __future__ import annotations

from PySide6.QtCore import Qt
from qfluentwidgets import CaptionLabel, CheckBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.defect import StrictGSFEPathOperation, StrictGSFEPathParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SegmentedControl,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class StrictGSFEPathCard(MakeDataCard):
    """Generate an unrelaxed GSFE path in the current cell's ab plane."""

    group = "Defect"
    card_name = "Stacking Fault / GSFE Path"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._setting_params = False
        self._stored_plane_hkl = (0, 0, 1)
        self._stored_slip_w = 0
        self.setTitle(self.tr("GSFE Path"))
        self.init_ui()

    def init_ui(self):
        """Build geometry, path, cut, coordinate, and preview sections."""
        self.setObjectName("strict_gsfe_path_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.slip_uv_frame = SpinBoxUnitInputFrame(self)
        self.slip_uv_frame.set_input("", 2, "int")
        self.slip_uv_frame.setRange(-100, 100)
        self.slip_uv_frame.set_input_value([1, 0])
        self.slip_uv_frame.setAccessibleName(self.tr("In-plane direction [u v]"))
        self.slip_uv_field = CompactField(
            self.tr("In-plane direction [u v]"),
            self.slip_uv_frame,
            self.setting_widget,
            self.tr("The shift vector is u·a + v·b in the current oriented cell."),
        )

        self.legacy_geometry_label = CaptionLabel("", self.setting_widget)
        self.legacy_geometry_label.setWordWrap(True)
        self.legacy_geometry_label.setStyleSheet("color:#b06a00;")
        self.legacy_geometry_label.setVisible(False)

        geometry_section = InspectorSection(
            self.tr("Slip geometry"),
            self.setting_widget,
            self.tr(
                "The current cell's ab plane is the fault plane (stored as 001); "
                "the third cell vector must be normal to it."
            ),
        )
        geometry_grid = ResponsiveFormGrid(geometry_section)
        geometry_grid.add_field(self.slip_uv_field, span=2)
        geometry_section.addWidget(geometry_grid)
        geometry_section.addWidget(self.legacy_geometry_label)

        self.disp_frame = SpinBoxUnitInputFrame(self)
        self.disp_frame.set_input(["–", self.tr("step"), ""], 3, "float")
        self.disp_frame.setDecimals(4)
        self.disp_frame.setSingleStep(0.1)
        self.disp_frame.setRange(-100.0, 100.0)
        self.disp_frame.object_list[2].setMinimum(0.0001)
        self.disp_frame.set_input_value([0.0, 1.0, 0.5])
        self.disp_frame.setAccessibleName(self.tr("Displacement path"))
        displacement_field = CompactField(
            self.tr("Path (start, end, step)"),
            self.disp_frame,
            self.setting_widget,
            self.tr("Both endpoints are included; step must be positive."),
        )

        self.unit_control = SegmentedControl(parent=self.setting_widget)
        self.unit_control.addItem(self.tr("Vector fraction"), userData="fraction_of_vector")
        self.unit_control.addItem(self.tr("Å distance"), userData="angstrom")
        self.unit_control.setAccessibleName(self.tr("Displacement unit"))
        self.unit_combo = self.unit_control
        unit_field = CompactField(
            self.tr("Displacement unit"),
            self.unit_control,
            self.setting_widget,
            self.tr(
                "A vector fraction scales u·a + v·b; Å distance uses only its direction."
            ),
        )

        path_section = InspectorSection(self.tr("Displacement path"), self.setting_widget)
        path_grid = ResponsiveFormGrid(path_section)
        path_grid.add_field(displacement_field, span=2)
        path_grid.add_field(unit_field, span=2)
        path_section.addWidget(path_grid)

        self.cut_control = SegmentedControl(parent=self.setting_widget)
        self.cut_control.addItem(self.tr("Middle"), userData="middle")
        self.cut_control.addItem(self.tr("Thickness"), userData="fractional")
        self.cut_control.addItem(self.tr("Layer index"), userData="layer_index")
        self.cut_control.setAccessibleName(self.tr("Interlayer cut"))
        self.cut_combo = self.cut_control
        cut_mode_field = CompactField(
            self.tr("Cut position"),
            self.cut_control,
            self.setting_widget,
            self.tr("Atoms above the resolved interlayer cut move together."),
        )

        self.cut_fraction_frame = SpinBoxUnitInputFrame(self)
        self.cut_fraction_frame.set_input("", 1, "float")
        self.cut_fraction_frame.setDecimals(4)
        self.cut_fraction_frame.setRange(0.0, 0.9999)
        self.cut_fraction_frame.set_input_value([0.5])
        self.cut_fraction_frame.setFixedWidth(144)
        self.cut_fraction_frame.setAccessibleName(self.tr("Thickness fraction"))
        self.cut_fraction_field = CompactField(
            self.tr("Thickness fraction"),
            self.cut_fraction_frame,
            self.setting_widget,
            self.tr("0 is the lowest projected layer and 1 is the highest."),
            inline=True,
            input_max_width=144,
        )

        self.layer_frame = SpinBoxUnitInputFrame(self)
        self.layer_frame.set_input("", 1, "int")
        self.layer_frame.setRange(0, 999999)
        self.layer_frame.set_input_value([0])
        self.layer_frame.setFixedWidth(144)
        self.layer_frame.setAccessibleName(self.tr("Lower layer index"))
        self.layer_field = CompactField(
            self.tr("Lower layer index"),
            self.layer_frame,
            self.setting_widget,
            self.tr("Zero-based; the cut is placed after this projected layer."),
            inline=True,
            input_max_width=144,
        )

        cut_section = InspectorSection(self.tr("Interlayer cut"), self.setting_widget)
        cut_grid = ResponsiveFormGrid(cut_section)
        cut_grid.add_field(cut_mode_field, span=2)
        cut_grid.add_field(self.cut_fraction_field, span=2)
        cut_grid.add_field(self.layer_field, span=2)
        cut_section.addWidget(cut_grid)

        self.wrap_checkbox = CheckBox(
            self.tr("Wrap displaced atoms into the cell"), self.setting_widget
        )
        self.wrap_checkbox.setChecked(True)
        coordinate_section = InspectorSection(
            self.tr("Coordinates"),
            self.setting_widget,
            self.tr(
                "Wrapping keeps periodic coordinates inside the cell; disable it only to inspect the raw Cartesian shift."
            ),
        )
        coordinate_section.addWidget(self.wrap_checkbox)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("strictGsfePathPreview")
        preview_section = InspectorSection(self.tr("Output preview"), self.setting_widget)
        preview_section.addWidget(self.preview_label)

        self.settingLayout.addWidget(geometry_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(path_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(cut_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(coordinate_section, 3, 0, 1, 3)
        self.settingLayout.addWidget(preview_section, 4, 0, 1, 3)

        self.cut_control.currentIndexChanged.connect(self._update_cut_visibility)
        self.unit_control.currentIndexChanged.connect(self._refresh_preview)
        for control in self.slip_uv_frame.object_list:
            control.valueChanged.connect(self._on_slip_edited)
        for control in (
            *self.disp_frame.object_list,
            *self.cut_fraction_frame.object_list,
            *self.layer_frame.object_list,
        ):
            control.valueChanged.connect(self._refresh_preview)
        self.wrap_checkbox.toggled.connect(self._refresh_preview)

        self._update_cut_visibility()

    def _on_slip_edited(self, *_args) -> None:
        if self._setting_params:
            return
        self._stored_plane_hkl = (0, 0, 1)
        self._stored_slip_w = 0
        self.legacy_geometry_label.setVisible(False)
        self._refresh_preview()

    def _update_cut_visibility(self, *_args) -> None:
        mode = self.cut_control.currentData()
        self.cut_fraction_field.setVisible(mode == "fractional")
        self.layer_field.setVisible(mode == "layer_index")
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

    def _cut_preview_text(self, summary: dict) -> str:
        params = self.get_params()
        position = f"{summary['cut_position']:.4g}"
        if params.cut_mode == "fractional":
            return self.tr("thickness {fraction}, cut at {position} Å").format(
                fraction=f"{params.cut_fraction:.4g}", position=position
            )
        if params.cut_mode == "layer_index":
            return self.tr("after layer {layer}, cut at {position} Å").format(
                layer=params.layer_index, position=position
            )
        return self.tr("middle cut at {position} Å").format(position=position)

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr(
                    "Load an oriented structure to preview the cut, displacement, and output count."
                )
            )
            return
        try:
            summary = self.create_operation().geometry_summary(
                self._input_structure, self.get_params()
            )
        except (TypeError, ValueError, IndexError) as exc:
            self.preview_label.setText(
                "⚠ "
                + self.tr("Preview unavailable: {error}").format(
                    error=translate_runtime_message(exc)
                )
            )
            return

        values = summary["values"]
        start = float(values[0])
        end = float(values[-1])
        if self.unit_control.currentData() == "fraction_of_vector":
            path_text = self.tr("{start}→{end} × vector = {a0}→{a1} Å").format(
                start=f"{start:g}",
                end=f"{end:g}",
                a0=f"{start * summary['slip_length']:.4g}",
                a1=f"{end * summary['slip_length']:.4g}",
            )
        else:
            path_text = self.tr("{start}→{end} Å").format(
                start=f"{start:g}", end=f"{end:g}"
            )
        self.preview_label.setText(
            self.tr(
                "First input: {layers} layers · {cut} · move {moved}, keep {stationary} · {path} · {outputs} outputs"
            ).format(
                layers=summary["layer_count"],
                cut=self._cut_preview_text(summary),
                moved=summary["moved_count"],
                stationary=summary["stationary_count"],
                path=path_text,
                outputs=len(values),
            )
        )

    def _update_tab_order(self) -> None:
        if not hasattr(self, "cut_control"):
            return
        widgets = [
            *self.slip_uv_frame.object_list,
            *self.disp_frame.object_list,
            self.unit_control,
            self.cut_control,
        ]
        mode = self.cut_control.currentData()
        if mode == "fractional":
            widgets.extend(self.cut_fraction_frame.object_list)
        elif mode == "layer_index":
            widgets.extend(self.layer_frame.object_list)
        widgets.append(self.wrap_checkbox)
        self.tab_order_widgets = widgets

    def create_operation(self):
        return StrictGSFEPathOperation()

    def get_params(self) -> StrictGSFEPathParams:
        u, v = (int(value) for value in self.slip_uv_frame.get_input_value())
        return StrictGSFEPathParams(
            plane_hkl=self._stored_plane_hkl,
            slip_uvw=(u, v, self._stored_slip_w),
            displacement_range=tuple(float(v) for v in self.disp_frame.get_input_value()),
            displacement_unit=str(self.unit_control.currentData()),
            cut_mode=str(self.cut_control.currentData()),
            cut_fraction=float(self.cut_fraction_frame.get_input_value()[0]),
            layer_index=int(self.layer_frame.get_input_value()[0]),
            wrap=self.wrap_checkbox.isChecked(),
        )

    def set_params(self, params: StrictGSFEPathParams) -> None:
        self._setting_params = True
        try:
            plane = tuple(int(v) for v in params.plane_hkl)
            slip = tuple(int(v) for v in params.slip_uvw)
            self._stored_plane_hkl = plane
            self._stored_slip_w = slip[2]
            self.slip_uv_frame.set_input_value([slip[0], slip[1]])
            self.disp_frame.set_input_value(
                [float(v) for v in params.displacement_range]
            )
            self.unit_control.setCurrentIndex(
                self.unit_control.findData(str(params.displacement_unit))
            )
            self.cut_control.setCurrentIndex(
                self.cut_control.findData(str(params.cut_mode))
            )
            self.cut_fraction_frame.set_input_value([float(params.cut_fraction)])
            self.layer_frame.set_input_value([int(params.layer_index)])
            self.wrap_checkbox.setChecked(bool(params.wrap))
        finally:
            self._setting_params = False

        legacy = plane != (0, 0, 1) or slip[2] != 0
        self.legacy_geometry_label.setVisible(legacy)
        if legacy:
            self.legacy_geometry_label.setText(
                self.tr(
                    "Loaded legacy geometry: plane {plane}, direction {direction}. "
                    "It is preserved until the in-plane direction is edited."
                ).format(plane=plane, direction=slip)
            )
        self._update_cut_visibility()

    def get_summary_text(self) -> str:
        params = self.get_params()
        u, v, w = params.slip_uvw
        direction = f"[{u} {v} {w}]"
        return self.tr("direction {direction} · {unit} · {cut}").format(
            direction=direction,
            unit=(
                self.tr("vector fraction")
                if params.displacement_unit == "fraction_of_vector"
                else self.tr("Å distance")
            ),
            cut={
                "middle": self.tr("middle cut"),
                "fractional": self.tr("thickness cut"),
                "layer_index": self.tr("layer-index cut"),
            }.get(params.cut_mode, params.cut_mode),
        )

    def get_guidance_text(self) -> str:
        return self.tr(
            "Indices use the current oriented cell. The third cell vector must be normal to the ab fault plane."
        )

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        self.set_params(StrictGSFEPathParams(**data_dict.get("params", {})))
