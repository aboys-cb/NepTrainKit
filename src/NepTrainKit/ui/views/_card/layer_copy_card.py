"""Card for stacking complete slabs along global Cartesian z."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from qfluentwidgets import CaptionLabel, CheckBox, ComboBox, LineEdit, TextEdit

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import LayerCopyOperation, LayerCopyParams
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import (
    add_translated_items,
    combo_value,
    set_combo_value,
)
from NepTrainKit.ui.widgets import (
    CompactField,
    InspectorSection,
    MakeDataCard,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
)


@CardManager.register_card
class LayerCopyCard(MakeDataCard):
    """Warp one slab, then stack complete copies along Cartesian z."""

    group = "Structure"
    card_name = "Layer Stack"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [{"name": "NepTrainKit", "role": "author"}]

    _PRESETS: list[tuple[str, str, str]] = [
        ("Custom", "", ""),
        ("Flat stack", "0", ""),
        (
            "2D sine ripple",
            "A*(sin(2*pi*x/Lx) + sin(2*pi*y/Ly))",
            "A=0.2, Lx=10, Ly=10",
        ),
        (
            "Gaussian bump",
            "A*exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))",
            "A=1, x0=0, y0=0, sigma=5",
        ),
        ("Paraboloid", "A*(x**2 + y**2)", "A=0.001"),
        ("Stripe ripple", "A*sin(2*pi*x/Lx)", "A=0.2, Lx=10"),
        ("Step at x=0", "where(x > 0, A, 0)", "A=1"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self._preview_input_count = None
        self.setTitle(self.tr("Layer Stack"))
        self._build_ui()

    def _number_input(
        self,
        unit: str,
        value: float,
        minimum: float,
        maximum: float,
        input_type: str,
    ) -> SpinBoxUnitInputFrame:
        frame = SpinBoxUnitInputFrame(self.setting_widget)
        frame.set_input(unit, 1, input_type=input_type)
        frame.setRange(minimum, maximum)
        frame.set_input_value([value])
        frame.setMaximumWidth(170)
        return frame

    def _build_ui(self) -> None:
        self.settingLayout.setContentsMargins(3, 0, 3, 0)
        self.settingLayout.setHorizontalSpacing(6)
        self.settingLayout.setVerticalSpacing(4)

        self.layers_frame = self._number_input("", 2, 1, 999, "int")
        self.distance_frame = self._number_input("Å", 3.35, 0.0, 1.0e4, "float")
        self.max_atoms_frame = self._number_input("", 100_000, 1, 100_000_000, "int")

        self.spacing_mode_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.spacing_mode_combo,
            [
                ("surface_gap", "Surface gap"),
                ("translation", "Copy translation (legacy)"),
            ],
        )
        set_combo_value(self.spacing_mode_combo, "surface_gap")
        self.spacing_mode_combo.setMinimumWidth(0)

        stack_section = InspectorSection(
            self.tr("Stack geometry"),
            self.setting_widget,
            self.tr(
                "Complete copies are placed along global Cartesian z. Surface gap is measured after the optional warp."
            ),
        )
        geometry_grid = ResponsiveFormGrid(stack_section)
        geometry_grid.add_field(
            CompactField(
                self.tr("Total layers"),
                self.layers_frame,
                stack_section,
                self.tr("Includes the original slab."),
            )
        )
        geometry_grid.add_field(
            CompactField(
                self.tr("Spacing value"),
                self.distance_frame,
                stack_section,
                self.tr(
                    "Surface gap is the empty z separation; legacy translation is the origin-to-origin copy shift."
                ),
            )
        )
        stack_section.addWidget(geometry_grid)
        stack_section.addWidget(
            CompactField(
                self.tr("Spacing definition"),
                self.spacing_mode_combo,
                stack_section,
            )
        )
        stack_section.addWidget(
            CompactField(
                self.tr("Atom budget per output"),
                self.max_atoms_frame,
                stack_section,
                self.tr("The exact output size is input atoms × total layers."),
            )
        )

        self.extend_cell_checkbox = CheckBox(
            self.tr("Extend cell along Cartesian z"), self.setting_widget
        )
        self.extend_cell_checkbox.setChecked(True)
        self.vacuum_frame = self._number_input("Å", 0.0, 0.0, 1.0e6, "float")
        self.vacuum_field = CompactField(
            self.tr("Additional top vacuum"),
            self.vacuum_frame,
            self.setting_widget,
        )
        self.wrap_checkbox = CheckBox(
            self.tr("Wrap atoms into the final cell"), self.setting_widget
        )
        self.wrap_checkbox.setChecked(False)
        cell_section = InspectorSection(self.tr("Final cell"), self.setting_widget)
        cell_section.addWidget(self.extend_cell_checkbox)
        cell_section.addWidget(self.vacuum_field)
        cell_section.addWidget(self.wrap_checkbox)

        self.show_warp_checkbox = CheckBox(
            self.tr("Apply a Cartesian-z surface warp"), self.setting_widget
        )
        self.show_warp_checkbox.setChecked(False)
        self.preset_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.preset_combo,
            [(str(index), name) for index, (name, _, _) in enumerate(self._PRESETS)],
        )
        self.preset_combo.setCurrentIndex(1)
        self.preset_combo.setMinimumWidth(0)
        self.expr_edit = TextEdit(self.setting_widget)
        self.expr_edit.setPlaceholderText(self.tr("For example: A*sin(2*pi*x/Lx)"))
        self.expr_edit.setFixedHeight(52)
        self.params_edit = LineEdit(self.setting_widget)
        self.params_edit.setPlaceholderText(self.tr("For example: A=0.2, Lx=10"))
        self.apply_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.apply_combo,
            [
                ("0", "All atoms"),
                ("1", "Selected elements"),
                ("2", "Cartesian z range"),
            ],
        )
        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setPlaceholderText(self.tr("For example: C, Si, O"))
        self.zrange_frame = SpinBoxUnitInputFrame(self.setting_widget)
        self.zrange_frame.set_input(["Å", "Å"], 2, input_type="float")
        self.zrange_frame.setRange(-1.0e6, 1.0e6)
        self.zrange_frame.set_input_value([-1.0e6, 1.0e6])

        warp_section = InspectorSection(
            self.tr("Optional surface warp"),
            self.setting_widget,
            self.tr(
                "The expression changes selected input atoms once; every full-slab copy then has the same shape."
            ),
        )
        warp_section.addWidget(self.show_warp_checkbox)
        self.preset_field = CompactField(
            self.tr("Warp preset"), self.preset_combo, warp_section
        )
        self.expression_field = CompactField(
            self.tr("z displacement expression"),
            self.expr_edit,
            warp_section,
            self.tr("Uses Cartesian x, y, z and returns displacement in Å."),
        )
        self.params_field = CompactField(
            self.tr("Expression parameters"), self.params_edit, warp_section
        )
        self.apply_field = CompactField(
            self.tr("Warp selection"),
            self.apply_combo,
            warp_section,
            self.tr("This limits only the warp; every atom is still copied."),
        )
        self.elements_field = CompactField(
            self.tr("Elements"), self.elements_edit, warp_section
        )
        self.zrange_field = CompactField(
            self.tr("Cartesian z range"), self.zrange_frame, warp_section
        )
        for field in (
            self.preset_field,
            self.expression_field,
            self.params_field,
            self.apply_field,
            self.elements_field,
            self.zrange_field,
        ):
            warp_section.addWidget(field)

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("layerCopyPreview")
        self.legacy_notice = CaptionLabel("", self.setting_widget)
        self.legacy_notice.setWordWrap(True)
        self.legacy_notice.setStyleSheet("color:#c56a00; font-weight:600;")
        self.legacy_notice.hide()
        preview_section = InspectorSection(self.tr("Exact preview"), self.setting_widget)
        preview_section.addWidget(self.preview_label)
        preview_section.addWidget(self.legacy_notice)

        self.settingLayout.addWidget(stack_section, 0, 0, 1, 3)
        self.settingLayout.addWidget(cell_section, 1, 0, 1, 3)
        self.settingLayout.addWidget(warp_section, 2, 0, 1, 3)
        self.settingLayout.addWidget(preview_section, 3, 0, 1, 3)

        self.expr_edit.setPlainText("0")
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.apply_combo.currentIndexChanged.connect(self._update_warp_visibility)
        self.show_warp_checkbox.toggled.connect(self._update_warp_visibility)
        self.extend_cell_checkbox.toggled.connect(self._update_cell_visibility)
        for frame in (
            self.layers_frame,
            self.distance_frame,
            self.max_atoms_frame,
            self.vacuum_frame,
            self.zrange_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self._refresh_preview)
        self.spacing_mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self.wrap_checkbox.toggled.connect(self._refresh_preview)
        self.expr_edit.textChanged.connect(self._refresh_preview)
        self.params_edit.textChanged.connect(self._refresh_preview)
        self.elements_edit.textChanged.connect(self._refresh_preview)
        self._update_cell_visibility()
        self._update_warp_visibility()
        self._refresh_preview()

    def _on_preset_changed(self, index: int) -> None:
        if 0 < index < len(self._PRESETS):
            _name, expr, params = self._PRESETS[index]
            self.expr_edit.setPlainText(expr)
            self.params_edit.setText(params)
        self._refresh_preview()

    def _update_warp_visibility(self, *_args) -> None:
        visible = self.show_warp_checkbox.isChecked()
        for field in (
            self.preset_field,
            self.expression_field,
            self.params_field,
            self.apply_field,
        ):
            field.setVisible(visible)
        mode = self.apply_combo.currentIndex()
        self.elements_field.setVisible(visible and mode == 1)
        self.zrange_field.setVisible(visible and mode == 2)
        self._refresh_preview()

    def _update_cell_visibility(self, *_args) -> None:
        self.vacuum_field.setVisible(self.extend_cell_checkbox.isChecked())
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

    def set_preview_structure(self, structure) -> None:
        self._input_structure = structure
        self._refresh_preview()

    def set_preview_input_count(self, count: int | None) -> None:
        self._preview_input_count = None if count is None else max(0, int(count))
        self.refresh_compact_presentation()

    def _summary(self):
        if self._input_structure is None:
            return None
        return self.create_operation().geometry_summary(
            self._input_structure,
            self.get_params(),
        )

    def _refresh_preview(self, *_args) -> None:
        if not hasattr(self, "preview_label"):
            return
        self.refresh_compact_presentation()
        if self._input_structure is None:
            self.preview_label.setText(
                self.tr("Load an upstream slab to preview the exact stack geometry.")
            )
            return
        try:
            summary = self._summary()
        except (TypeError, ValueError, IndexError) as exc:
            self.preview_label.setText(
                "⚠ "
                + self.tr("Preview unavailable: {error}").format(
                    error=translate_runtime_message(exc)
                )
            )
            return
        warp = (
            self.tr("flat")
            if abs(summary["dz_min"]) <= 1.0e-12
            and abs(summary["dz_max"]) <= 1.0e-12
            else self.tr("warp dz {minimum} to {maximum} Å").format(
                minimum=f"{summary['dz_min']:.4g}",
                maximum=f"{summary['dz_max']:.4g}",
            )
        )
        self.preview_label.setText(
            self.tr(
                "First input: thickness {thickness} Å · gap {gap} Å · copy translation {translation} Å · "
                "{layers} layers · {input_atoms} input atoms → {atoms} atoms/output · {warp} · "
                "lattice c {before} → {after} Å"
            ).format(
                thickness=f"{summary['slab_thickness']:.4g}",
                gap=f"{summary['surface_gap']:.4g}",
                translation=f"{summary['translation']:.4g}",
                layers=summary["layers"],
                input_atoms=summary["input_atoms"],
                atoms=summary["output_atoms"],
                warp=warp,
                before=f"{summary['cell_c_before']:.4g}",
                after=f"{summary['cell_c_after']:.4g}",
            )
        )

    def create_operation(self):
        return LayerCopyOperation()

    def get_summary_text(self) -> str:
        try:
            summary = self._summary()
        except (TypeError, ValueError, IndexError):
            return self.tr("Check the stack geometry")
        if summary is not None:
            return self.tr(
                "{layers} layers · gap {gap} Å · {atoms} atoms/out · 1/in"
            ).format(
                layers=summary["layers"],
                gap=f"{summary['surface_gap']:.4g}",
                atoms=summary["output_atoms"],
            )
        params = self.get_params()
        return self.tr("{layers} layers · spacing {spacing} Å · 1 output/input").format(
            layers=params.layers,
            spacing=f"{params.distance:.4g}",
        )

    def get_guidance_text(self) -> str:
        count = self._preview_input_count
        output_text = (
            self.tr("Outputs/input: 1.")
            if count is None or count <= 0
            else self.tr("Inputs {inputs} × 1 output/input = outputs {total}.").format(
                inputs=count,
                total=count,
            )
        )
        if self._input_structure is None:
            return output_text + " " + self.tr(
                "Load an upstream slab to verify thickness, gap, and output atoms."
            )
        try:
            summary = self._summary()
        except (TypeError, ValueError, IndexError) as exc:
            return output_text + " " + translate_runtime_message(exc)
        return output_text + " " + self.tr(
            "Each output has {atoms} atoms ({input_atoms} × {layers}); verify gap "
            "{gap} Å and copy translation {translation} Å."
        ).format(
            atoms=summary["output_atoms"],
            input_atoms=summary["input_atoms"],
            layers=summary["layers"],
            gap=f"{summary['surface_gap']:.4g}",
            translation=f"{summary['translation']:.4g}",
        )

    def get_params(self) -> LayerCopyParams:
        return LayerCopyParams(
            preset_index=self.preset_combo.currentIndex(),
            dz_expr=self.expr_edit.toPlainText(),
            expression_params=self.params_edit.text(),
            apply_mode=self.apply_combo.currentIndex(),
            elements=self.elements_edit.text(),
            z_range=tuple(float(value) for value in self.zrange_frame.get_input_value()),
            wrap=self.wrap_checkbox.isChecked(),
            extend_cell_z=self.extend_cell_checkbox.isChecked(),
            extra_vacuum=float(self.vacuum_frame.get_input_value()[0]),
            layers=int(self.layers_frame.get_input_value()[0]),
            distance_mode=combo_value(self.spacing_mode_combo),
            distance=float(self.distance_frame.get_input_value()[0]),
            max_output_atoms=int(self.max_atoms_frame.get_input_value()[0]),
        )

    def set_params(self, params: LayerCopyParams) -> None:
        self.preset_combo.setCurrentIndex(int(params.preset_index))
        self.expr_edit.setPlainText(params.dz_expr)
        self.params_edit.setText(params.expression_params)
        self.apply_combo.setCurrentIndex(int(params.apply_mode))
        self.elements_edit.setText(params.elements)
        self.zrange_frame.set_input_value([float(value) for value in params.z_range])
        self.wrap_checkbox.setChecked(bool(params.wrap))
        self.extend_cell_checkbox.setChecked(bool(params.extend_cell_z))
        self.vacuum_frame.set_input_value([float(params.extra_vacuum)])
        self.layers_frame.set_input_value([int(params.layers)])
        set_combo_value(self.spacing_mode_combo, params.distance_mode)
        self.distance_frame.set_input_value([float(params.distance)])
        self.max_atoms_frame.set_input_value([int(params.max_output_atoms)])
        self.show_warp_checkbox.setChecked(
            str(params.dz_expr).strip() not in {"", "0", "0.0"}
        )
        self._update_cell_visibility()
        self._update_warp_visibility()

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    @staticmethod
    def _legacy_scalar(value, default):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value

    def from_dict(self, data_dict: dict[str, Any]) -> None:
        super().from_dict(data_dict)
        self.legacy_notice.clear()
        self.legacy_notice.hide()
        raw_params = data_dict.get("params")
        if not isinstance(raw_params, dict):
            raw_params = data_dict.get("operation_params")
        legacy = not isinstance(raw_params, dict) or "distance_mode" not in raw_params
        if isinstance(raw_params, dict):
            params = LayerCopyParams(
                preset_index=raw_params.get("preset_index", 1),
                dz_expr=raw_params.get("dz_expr", "0"),
                expression_params=raw_params.get("expression_params", ""),
                apply_mode=raw_params.get("apply_mode", 0),
                elements=raw_params.get("elements", ""),
                z_range=tuple(raw_params.get("z_range", [-1.0e6, 1.0e6])),
                wrap=raw_params.get("wrap", False),
                extend_cell_z=raw_params.get("extend_cell_z", True),
                extra_vacuum=raw_params.get("extra_vacuum", 0.0),
                layers=raw_params.get("layers", 3 if legacy else 2),
                distance_mode=raw_params.get("distance_mode", "translation"),
                distance=raw_params.get("distance", 3.0 if legacy else 3.35),
                max_output_atoms=raw_params.get("max_output_atoms", 100_000),
            )
        else:
            z_range = data_dict.get("z_range", [-1.0e6, 1.0e6])
            params = LayerCopyParams(
                preset_index=data_dict.get("preset_index", 1),
                dz_expr=data_dict.get("dz_expr", "0"),
                expression_params=data_dict.get("params", ""),
                apply_mode=data_dict.get("apply_mode", 0),
                elements=data_dict.get("elements", ""),
                z_range=tuple(z_range if isinstance(z_range, (list, tuple)) else [-1.0e6, 1.0e6]),
                wrap=data_dict.get("wrap", False),
                extend_cell_z=data_dict.get("extend_cell_z", True),
                extra_vacuum=self._legacy_scalar(data_dict.get("extra_vacuum", 0.0), 0.0),
                layers=self._legacy_scalar(data_dict.get("layers", 3), 3),
                distance_mode="translation",
                distance=self._legacy_scalar(data_dict.get("distance", 3.0), 3.0),
                max_output_atoms=data_dict.get("max_output_atoms", 100_000),
            )
        self.set_params(params)
        if legacy:
            message = self.tr(
                "Legacy Layer Stack loaded: distance keeps its old copy-translation meaning. "
                "A negative surface gap is now rejected; verify the exact preview before running."
            )
            self.legacy_notice.setText("⚠ " + message)
            self.legacy_notice.show()
            MessageManager.send_warning_message(message)
