"""Card for Bain/tetragonal distortion paths."""

from __future__ import annotations

from PySide6.QtCore import Qt
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.lattice import BainPathOperation, BainPathParams
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import combo_value, set_combo_value
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class BainPathCard(MakeDataCard):
    """Generate Bain/tetragonal paths with an explicit path coordinate."""

    group = "Lattice"
    card_name = "Bain Path"
    menu_icon = r":/images/src/images/scaling.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.tr("Bain Path"))
        self.init_ui()

    def init_ui(self):
        self.setObjectName("bain_path_card_widget")
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setVerticalSpacing(4)

        self.axis_combo = ComboBox(self.setting_widget)
        self.axis_combo.addItem(self.tr("Lattice a"), userData="x")
        self.axis_combo.addItem(self.tr("Lattice b"), userData="y")
        self.axis_combo.addItem(self.tr("Lattice c"), userData="z")
        set_combo_value(self.axis_combo, "z")
        self.axis_combo.setAccessibleName(self.tr("Tetragonal lattice axis"))
        self.axis_field = CompactField(
            self.tr("Tetragonal axis"),
            self.axis_combo,
            self.setting_widget,
        )
        self.axis_label = self.axis_field

        self.coordinate_combo = ComboBox(self.setting_widget)
        self.coordinate_combo.addItem(self.tr("Relative c/a"), userData="relative_ca")
        self.coordinate_combo.addItem(self.tr("Axial scale (legacy)"), userData="axis_scale")
        self.coordinate_combo.setAccessibleName(self.tr("Path coordinate"))
        self.coordinate_field = CompactField(
            self.tr("Path coordinate"),
            self.coordinate_combo,
            self.setting_widget,
        )

        self.mode_combo = ComboBox(self.setting_widget)
        self.mode_combo.addItem(self.tr("Constant volume"), userData="constant_volume")
        self.mode_combo.addItem(self.tr("Shape × volume grid"), userData="scale_volume")
        self.mode_combo.addItem(self.tr("Free tetragonal axis"), userData="free_c")
        self.mode_combo.setAccessibleName(self.tr("Path mode"))
        self.mode_field = CompactField(
            self.tr("Path mode"),
            self.mode_combo,
            self.setting_widget,
        )
        self.mode_label = self.mode_field

        self.ca_frame = SpinBoxUnitInputFrame(self)
        self.ca_frame.set_input(["–", self.tr("step"), ""], 3, "float")
        self.ca_frame.setDecimals(4)
        self.ca_frame.setRange(0.0001, 100.0)
        self.ca_frame.set_input_value([0.95, 1.05, 0.05])
        self.ca_frame.setAccessibleName(self.tr("Relative c/a scan"))
        self.ca_field = CompactField(
            self.tr("Relative c/a scan"),
            self.ca_frame,
            self.setting_widget,
            self.tr(
                "Start – stop with a positive step. 1.0 preserves the input axial-to-transverse ratio."
            ),
        )
        self.ca_label = self.ca_field

        self.volume_frame = SpinBoxUnitInputFrame(self)
        self.volume_frame.set_input(["–", self.tr("step"), ""], 3, "float")
        self.volume_frame.setDecimals(4)
        self.volume_frame.setRange(0.0001, 100.0)
        self.volume_frame.set_input_value([1.0, 1.0, 1.0])
        self.volume_frame.setAccessibleName(self.tr("Relative volume scan"))
        self.volume_field = CompactField(
            self.tr("Relative volume scan"),
            self.volume_frame,
            self.setting_widget,
            self.tr("Start – stop with a positive step; 1.0 preserves the input volume."),
        )
        self.volume_label = self.volume_field

        self.path_section = InspectorSection(
            self.tr("Path definition"),
            self.setting_widget,
            self.tr("Use a cell already oriented with the intended tetragonal direction along lattice a, b, or c."),
        )
        path_grid = ResponsiveFormGrid(self.path_section)
        path_grid.add_field(self.axis_field)
        path_grid.add_field(self.coordinate_field)
        path_grid.add_field(self.mode_field, span=2)
        path_grid.add_field(self.ca_field, span=2)
        path_grid.add_field(self.volume_field, span=2)
        self.path_section.addWidget(path_grid)

        self.scale_atoms_checkbox = CheckBox(
            self.tr("Move atoms with the cell"),
            self.setting_widget,
        )
        self.scale_atoms_checkbox.setChecked(True)
        coordinate_section = InspectorSection(
            self.tr("Atomic coordinates"),
            self.setting_widget,
            self.tr(
                "Enabled keeps fractional coordinates fixed. Disabled keeps Cartesian "
                "positions fixed and should be used deliberately."
            ),
        )
        coordinate_section.addWidget(self.scale_atoms_checkbox)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_label.setObjectName("bainPathPreview")
        preview_section = InspectorSection(
            self.tr("Output preview"),
            self.setting_widget,
        )
        preview_section.addWidget(self.preview_label)

        self.settingLayout.addWidget(self.path_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(coordinate_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(preview_section, 2, 0, 1, 3)

        self.axis_combo.currentIndexChanged.connect(self._refresh_preview)
        self.coordinate_combo.currentIndexChanged.connect(self._on_coordinate_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        for control in self.ca_frame.object_list + self.volume_frame.object_list:
            control.valueChanged.connect(self._refresh_preview)
        self.scale_atoms_checkbox.toggled.connect(self._refresh_preview)

        self._on_coordinate_changed()
        self._on_mode_changed()

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

    def set_dataset(self, dataset) -> None:
        super().set_dataset(dataset)
        self._refresh_preview()

    def _on_coordinate_changed(self, *_args) -> None:
        relative_ca = combo_value(self.coordinate_combo) == "relative_ca"
        self.ca_field.set_label(self.tr("Relative c/a scan") if relative_ca else self.tr("Axial scale scan"))
        self.ca_field.set_helper_text(
            self.tr(
                "Start – stop with a positive step. 1.0 preserves the input axial-to-transverse ratio."
            )
            if relative_ca
            else self.tr("Legacy coordinate: the selected lattice vector is multiplied directly by this value.")
        )
        self.mode_field.set_helper_text(self._mode_explanation())
        self._refresh_preview()
        self._update_tab_order()

    def _on_mode_changed(self, *_args) -> None:
        show_volume = combo_value(self.mode_combo) == "scale_volume"
        self.volume_field.setVisible(show_volume)
        self.mode_field.set_helper_text(self._mode_explanation())
        self._refresh_preview()
        self._update_tab_order()

    def _mode_explanation(self) -> str:
        mode = combo_value(self.mode_combo)
        relative_ca = combo_value(self.coordinate_combo) == "relative_ca"
        if mode == "free_c":
            return self.tr("Changes only the selected lattice vector; transverse vectors stay fixed.")
        if mode == "scale_volume":
            return self.tr("Combines every tetragonal shape point with every relative-volume point.")
        if relative_ca:
            return self.tr("Changes the axial-to-transverse ratio while preserving cell volume.")
        return self.tr("Scales the selected vector directly and compensates transverse vectors to preserve volume.")

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        try:
            summary = self.create_operation().sampling_summary(self.get_params())
        except ValueError as exc:
            self.preview_label.setText(
                "⚠ " + self.tr("Preview unavailable: {error}").format(error=translate_runtime_message(exc))
            )
            return
        inputs = self._dataset_count(self.dataset)
        per_input = int(summary["outputs_per_input"])
        total = inputs * per_input
        coordinate = self._coordinate_name()
        if int(summary["volume_points"]) > 1:
            grid = self.tr("{path} shape points × {volume} volume points = {per_input} outputs/input").format(
                path=summary["path_points"],
                volume=summary["volume_points"],
                per_input=per_input,
            )
        else:
            grid = self.tr("{path} path points = {per_input} outputs/input").format(
                path=summary["path_points"],
                per_input=per_input,
            )
        if inputs:
            grid += self.tr(" · input structures: {inputs} → {total} outputs").format(
                inputs=inputs,
                total=total,
            )
        coordinate_text = (
            self.tr("atoms keep fractional coordinates")
            if self.scale_atoms_checkbox.isChecked()
            else self.tr("atoms keep Cartesian positions")
        )
        self.preview_label.setText(f"{coordinate} · {grid} · {coordinate_text}")

    def _coordinate_name(self) -> str:
        return (
            self.tr("relative c/a")
            if combo_value(self.coordinate_combo) == "relative_ca"
            else self.tr("legacy axial scale")
        )

    def _coordinate_summary_name(self) -> str:
        return (
            self.tr("relative c/a")
            if combo_value(self.coordinate_combo) == "relative_ca"
            else self.tr("legacy scale")
        )

    def _update_tab_order(self) -> None:
        if not hasattr(self, "volume_field"):
            return
        widgets = [
            self.axis_combo,
            self.coordinate_combo,
            self.mode_combo,
            *self.ca_frame.object_list,
        ]
        if not self.volume_field.isHidden():
            widgets.extend(self.volume_frame.object_list)
        widgets.append(self.scale_atoms_checkbox)
        self.tab_order_widgets = widgets

    def create_operation(self):
        return BainPathOperation()

    def get_params(self) -> BainPathParams:
        return BainPathParams(
            axis=combo_value(self.axis_combo),
            ca_range=tuple(float(v) for v in self.ca_frame.get_input_value()),
            coordinate_mode=combo_value(self.coordinate_combo),
            mode=combo_value(self.mode_combo),
            volume_scale_range=tuple(float(v) for v in self.volume_frame.get_input_value()),
            scale_atoms=self.scale_atoms_checkbox.isChecked(),
        )

    def set_params(self, params: BainPathParams) -> None:
        set_combo_value(self.axis_combo, params.axis)
        self.ca_frame.set_input_value(list(params.ca_range))
        set_combo_value(self.coordinate_combo, params.coordinate_mode)
        set_combo_value(self.mode_combo, params.mode)
        self.volume_frame.set_input_value(list(params.volume_scale_range))
        self.scale_atoms_checkbox.setChecked(bool(params.scale_atoms))
        self._on_coordinate_changed()
        self._on_mode_changed()

    def get_summary_text(self) -> str:
        params = self.get_params()
        summary = self.create_operation().sampling_summary(params)
        mode_names = {
            "constant_volume": self.tr("constant V"),
            "scale_volume": self.tr("shape × V"),
            "free_c": self.tr("free axis"),
        }
        return self.tr("{coordinate} · {mode} · {axis} · {outputs}/input").format(
            coordinate=self._coordinate_summary_name(),
            mode=mode_names[params.mode],
            axis={"x": "a", "y": "b", "z": "c"}[params.axis],
            outputs=summary["outputs_per_input"],
        )

    def get_guidance_text(self) -> str:
        params = self.get_params()
        summary = self.create_operation().sampling_summary(params)
        inputs = self._dataset_count(self.dataset)
        if inputs:
            count_text = self.tr(
                "Input structures: {inputs} × {outputs} outputs/input = {total} outputs."
            ).format(
                inputs=inputs,
                outputs=summary["outputs_per_input"],
                total=inputs * int(summary["outputs_per_input"]),
            )
        else:
            count_text = self.tr("Each input produces {outputs} path structures.").format(
                outputs=summary["outputs_per_input"]
            )
        return self.tr("Coordinate: {coordinate}. {count} {mode}").format(
            coordinate=self._coordinate_name(),
            count=count_text,
            mode=self._mode_explanation(),
        )

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self):
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict):
        super().from_dict(data_dict)
        raw = data_dict.get("params")
        if not isinstance(raw, dict):
            self.set_params(BainPathParams())
            return
        values = dict(raw)
        # Saved cards from before the coordinate fix stored direct axial factors.
        values.setdefault("coordinate_mode", "axis_scale")
        self.set_params(BainPathParams(**values))
