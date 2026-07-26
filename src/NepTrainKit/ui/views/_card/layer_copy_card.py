"""Card for warping a structure by dz=f(x,y) and copying it into stacked layers."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    LineEdit,
    TextEdit,
    ToolTipFilter,
    ToolTipPosition,
    TransparentToolButton,
    FluentIcon,
)

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.core.cards.operation import params_to_dict
from NepTrainKit.core.cards.structure import (
    LayerCopyOperation,
    LayerCopyParams,
    evaluate_dz_expression,
    parse_dz_params,
)
from NepTrainKit.ui.messages import translate_runtime_message
from NepTrainKit.ui.views._card.i18n_utils import add_translated_items
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class LayerCopyCard(MakeDataCard):
    """Warp structure by dz=f(x,y) then copy-translate along z into a single stacked structure."""

    group = "Structure"
    card_name = "Layer Copy"
    menu_icon = r":/images/src/images/defect.svg"
    contributors = [
        {"name": "NepTrainKit", "role": "author"},
    ]

    _PRESETS: list[tuple[str, str, str]] = [
        ("Custom", "", ""),
        ("Flat stack (no warp)", "0", ""),
        ("2D sine ripple", "A*(sin(2*pi*x/Lx) + sin(2*pi*y/Ly))", "A=0.2, Lx=10, Ly=10"),
        ("Gaussian bump", "A*exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))", "A=1, x0=0, y0=0, sigma=5"),
        ("Paraboloid", "A*(x**2 + y**2)", "A=0.001"),
        ("Stripe ripple", "A*sin(2*pi*x/Lx)", "A=0.2, Lx=10"),
        ("Step (x>0)", "where(x > 0, A, 0)", "A=1"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_structure = None
        self.setTitle(self.tr("Layer Stack (optional z warp)"))
        self._build_ui()

    def _build_ui(self):
        layout: QGridLayout = self.settingLayout
        layout.setContentsMargins(3, 0, 3, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        layout.setColumnStretch(1, 1)
        row = 0

        self.layers_label = BodyLabel(self.tr("Total layers"), self.setting_widget)
        self.layers_label.setToolTip(
            self.tr("Includes the original layer; 2 produces one original plus one copy")
        )
        self.layers_label.installEventFilter(
            ToolTipFilter(self.layers_label, 300, ToolTipPosition.TOP)
        )
        self.layers_frame = SpinBoxUnitInputFrame(self)
        self.layers_frame.set_input("layers", 1, input_type="int")
        self.layers_frame.setRange(1, 999)
        self.layers_frame.set_input_value([2])
        layout.addWidget(self.layers_label, row, 0, 1, 1)
        layout.addWidget(self.layers_frame, row, 1, 1, 2)
        row += 1

        self.distance_label = BodyLabel(
            self.tr("Copy translation along z"),
            self.setting_widget,
        )
        self.distance_label.setToolTip(
            self.tr(
                "Origin-to-origin translation between copies, not the surface-to-surface gap"
            )
        )
        self.distance_label.installEventFilter(
            ToolTipFilter(self.distance_label, 300, ToolTipPosition.TOP)
        )
        self.distance_frame = SpinBoxUnitInputFrame(self)
        self.distance_frame.set_input("Å", 1, input_type="float")
        self.distance_frame.setRange(0.001, 1e4)
        self.distance_frame.set_input_value([3.35])
        layout.addWidget(self.distance_label, row, 0, 1, 1)
        layout.addWidget(self.distance_frame, row, 1, 1, 2)
        row += 1

        self.extend_cell_checkbox = CheckBox(
            self.tr("Extend the cell along Cartesian z"),
            self.setting_widget,
        )
        self.extend_cell_checkbox.setChecked(True)
        self.extend_cell_checkbox.setToolTip(
            self.tr("Adds the stack height and optional vacuum to the z component of lattice c")
        )
        layout.addWidget(self.extend_cell_checkbox, row, 0, 1, 3)
        row += 1

        self.vacuum_label = BodyLabel(
            self.tr("Additional top vacuum"),
            self.setting_widget,
        )
        self.vacuum_frame = SpinBoxUnitInputFrame(self)
        self.vacuum_frame.set_input("Å", 1, input_type="float")
        self.vacuum_frame.setRange(0.0, 1e6)
        self.vacuum_frame.set_input_value([0.0])
        layout.addWidget(self.vacuum_label, row, 0, 1, 1)
        layout.addWidget(self.vacuum_frame, row, 1, 1, 2)
        row += 1

        self.wrap_checkbox = CheckBox(
            self.tr("Wrap stacked atoms into the final periodic cell"),
            self.setting_widget,
        )
        self.wrap_checkbox.setChecked(False)
        layout.addWidget(self.wrap_checkbox, row, 0, 1, 3)
        row += 1

        self.show_warp_checkbox = CheckBox(
            self.tr("Show optional surface-warp settings"),
            self.setting_widget,
        )
        self.show_warp_checkbox.setChecked(False)
        layout.addWidget(self.show_warp_checkbox, row, 0, 1, 3)
        row += 1

        self.preset_label = BodyLabel(self.tr("Warp preset"), self.setting_widget)
        self.preset_combo = ComboBox(self.setting_widget)
        add_translated_items(
            self,
            self.preset_combo,
            [(str(index), name) for index, (name, _, _) in enumerate(self._PRESETS)],
        )
        self.preset_combo.setCurrentIndex(1)
        self.preset_label.setToolTip(
            self.tr("Choose a Cartesian-z displacement expression; Custom keeps your input")
        )
        self.preset_label.installEventFilter(ToolTipFilter(self.preset_label, 300, ToolTipPosition.TOP))

        self.test_button = TransparentToolButton(FluentIcon.PLAY, self.setting_widget)
        self.test_button.setToolTip(self.tr("Preview the displacement range on the first input"))
        self.test_button.installEventFilter(ToolTipFilter(self.test_button, 300, ToolTipPosition.TOP))

        self.expr_label = BodyLabel(self.tr("z displacement expression"), self.setting_widget)
        self.expr_label.setToolTip(
            self.tr("Evaluated in Å using Cartesian x, y, and z coordinates")
        )
        self.expr_edit = TextEdit(self.setting_widget)
        self.expr_edit.setPlaceholderText(self.tr("e.g. A*sin(2*pi*x/Lx)"))
        self.expr_edit.setFixedHeight(58)

        self.params_label = BodyLabel(self.tr("Expression parameters"), self.setting_widget)
        self.params_edit = LineEdit(self.setting_widget)
        self.params_edit.setPlaceholderText(self.tr("A=0.2, Lx=10  (optional)"))

        self.apply_label = BodyLabel(self.tr("Warp which atoms"), self.setting_widget)
        self.apply_label.setToolTip(
            self.tr("Only limits the optional warp; every atom is copied into every layer")
        )
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
        self.elements_edit.setPlaceholderText(self.tr("e.g. C, Si, O"))
        self.elements_edit.setVisible(False)

        self.zrange_frame = SpinBoxUnitInputFrame(self)
        self.zrange_frame.set_input(["Å", "Å"], 2, input_type="float")
        self.zrange_frame.setRange(-1e6, 1e6)
        self.zrange_frame.set_input_value([-1e6, 1e6])
        self.zrange_frame.setVisible(False)

        layout.addWidget(self.preset_label, row, 0, 1, 1)
        layout.addWidget(self.preset_combo, row, 1, 1, 1)
        layout.addWidget(self.test_button, row, 2, 1, 1)
        row += 1
        layout.addWidget(self.expr_label, row, 0, 1, 1)
        layout.addWidget(self.expr_edit, row, 1, 1, 2)
        row += 1
        layout.addWidget(self.params_label, row, 0, 1, 1)
        layout.addWidget(self.params_edit, row, 1, 1, 2)
        row += 1
        layout.addWidget(self.apply_label, row, 0, 1, 1)
        layout.addWidget(self.apply_combo, row, 1, 1, 2)
        row += 1
        layout.addWidget(self.elements_edit, row, 1, 1, 2)
        row += 1
        layout.addWidget(self.zrange_frame, row, 1, 1, 2)
        row += 1

        self.preview_label = CaptionLabel("", self.setting_widget)
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.preview_label.setObjectName("layerCopyPreview")
        layout.addWidget(self.preview_label, row, 0, 1, 3)

        self.warp_controls = (
            self.preset_label,
            self.preset_combo,
            self.test_button,
            self.expr_label,
            self.expr_edit,
            self.params_label,
            self.params_edit,
            self.apply_label,
            self.apply_combo,
            self.elements_edit,
            self.zrange_frame,
        )

        self.expr_edit.setPlainText("0")

        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.apply_combo.currentIndexChanged.connect(self._on_apply_changed)
        self.test_button.clicked.connect(self._test_expression)
        self.show_warp_checkbox.stateChanged.connect(self._update_warp_visibility)
        self.extend_cell_checkbox.stateChanged.connect(self._update_cell_visibility)
        for control in self.layers_frame.object_list + self.distance_frame.object_list + self.vacuum_frame.object_list:
            control.valueChanged.connect(self._refresh_preview)
        self.wrap_checkbox.stateChanged.connect(self._refresh_preview)
        self.expr_edit.textChanged.connect(self._refresh_preview)
        self.params_edit.textChanged.connect(self._refresh_preview)
        self.elements_edit.textChanged.connect(self._refresh_preview)
        for control in self.zrange_frame.object_list:
            control.valueChanged.connect(self._refresh_preview)
        self._update_cell_visibility()
        self._update_warp_visibility()
        self._refresh_preview()

    def _on_preset_changed(self, index: int) -> None:
        if index <= 0 or index >= len(self._PRESETS):
            self._refresh_preview()
            return
        _, expr, params = self._PRESETS[index]
        if expr:
            self.expr_edit.setPlainText(expr)
        self.params_edit.setText(params or "")
        self._refresh_preview()

    def _on_apply_changed(self, index: int) -> None:
        # 0: all, 1: elements, 2: z-range
        visible = self.show_warp_checkbox.isChecked()
        self.elements_edit.setVisible(visible and index == 1)
        self.zrange_frame.setVisible(visible and index == 2)
        self._refresh_preview()

    def _update_warp_visibility(self, *_args) -> None:
        visible = self.show_warp_checkbox.isChecked()
        for widget in self.warp_controls:
            widget.setVisible(visible)
        self._on_apply_changed(self.apply_combo.currentIndex())

    def _update_cell_visibility(self, *_args) -> None:
        visible = self.extend_cell_checkbox.isChecked()
        self.vacuum_label.setVisible(visible)
        self.vacuum_frame.setVisible(visible)
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
                    "Load an upstream slab to preview selected atoms, displacement range, and final stack size."
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
        if abs(summary["dz_min"]) <= 1e-12 and abs(summary["dz_max"]) <= 1e-12:
            warp_text = self.tr("no surface warp")
        else:
            warp_text = self.tr(
                "warp {selected} atoms / dz {minimum} to {maximum} Å"
            ).format(
                selected=summary["selected_atoms"],
                minimum=f"{summary['dz_min']:.4g}",
                maximum=f"{summary['dz_max']:.4g}",
            )
        if summary["extend_cell"]:
            cell_text = self.tr("lattice c {before} → {after} Å").format(
                before=f"{summary['cell_c_before']:.4g}",
                after=f"{summary['cell_c_after']:.4g}",
            )
        else:
            cell_text = self.tr("lattice c unchanged at {length} Å").format(
                length=f"{summary['cell_c_before']:.4g}",
            )
        self.preview_label.setText(
            self.tr(
                "First input: {atoms} atoms · {warp} · {layers} total layers at {translation} Å translation · output {output} atoms · {cell}"
            ).format(
                atoms=summary["input_atoms"],
                warp=warp_text,
                layers=summary["layers"],
                translation=f"{summary['translation']:.4g}",
                output=summary["output_atoms"],
                cell=cell_text,
            )
        )

    def create_operation(self):
        """Return the UI-independent layer-copy operation."""
        return LayerCopyOperation()

    def get_params(self) -> LayerCopyParams:
        """Read layer-copy parameters from UI controls."""
        return LayerCopyParams(
            preset_index=self.preset_combo.currentIndex(),
            dz_expr=self.expr_edit.toPlainText(),
            expression_params=self.params_edit.text(),
            apply_mode=self.apply_combo.currentIndex(),
            elements=self.elements_edit.text(),
            z_range=tuple(float(v) for v in self.zrange_frame.get_input_value()),
            wrap=self.wrap_checkbox.isChecked(),
            extend_cell_z=self.extend_cell_checkbox.isChecked(),
            extra_vacuum=float(self.vacuum_frame.get_input_value()[0]),
            layers=int(self.layers_frame.get_input_value()[0]),
            distance=float(self.distance_frame.get_input_value()[0]),
        )

    def set_params(self, params: LayerCopyParams) -> None:
        """Apply layer-copy parameters to UI controls."""
        self.preset_combo.setCurrentIndex(int(params.preset_index))
        self.expr_edit.setPlainText(params.dz_expr)
        self.params_edit.setText(params.expression_params)
        self.apply_combo.setCurrentIndex(int(params.apply_mode))
        self.elements_edit.setText(params.elements)
        self.zrange_frame.set_input_value([float(v) for v in params.z_range])
        self.wrap_checkbox.setChecked(bool(params.wrap))
        self.extend_cell_checkbox.setChecked(bool(params.extend_cell_z))
        self.vacuum_frame.set_input_value([float(params.extra_vacuum)])
        self.layers_frame.set_input_value([int(params.layers)])
        self.distance_frame.set_input_value([float(params.distance)])
        if str(params.dz_expr).strip() not in {"", "0", "0.0"}:
            self.show_warp_checkbox.setChecked(True)
        self._update_cell_visibility()
        self._update_warp_visibility()
        self._on_apply_changed(self.apply_combo.currentIndex())
        self._refresh_preview()

    def _test_expression(self) -> None:
        if self._input_structure is None:
            MessageManager.send_warning_message("No input structure available to test.")
            return
        structure = self._input_structure
        try:
            params = self.get_params()
            expr_params = parse_dz_params(params.expression_params)
            positions = np.asarray(structure.get_positions(), dtype=float)
            mask = LayerCopyOperation.apply_mask(structure, params)
            if not np.any(mask):
                MessageManager.send_warning_message("No atoms selected by 'apply to' settings.")
                return
            dz = evaluate_dz_expression(
                params.dz_expr.strip(),
                x=positions[mask, 0],
                y=positions[mask, 1],
                z=positions[mask, 2],
                params=expr_params,
            )
            MessageManager.send_info_message(
                f"dz test ok: n={int(mask.sum())}, min={float(np.min(dz)):.6g}, max={float(np.max(dz)):.6g}"
            )
        except Exception as e:  # noqa: BLE001
            MessageManager.send_error_message(f"dz test failed: {e}")

    def process_structure(self, structure):
        return self.create_operation().run_structure(structure, self.get_params())

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["params"] = params_to_dict(self.get_params())
        return data

    def from_dict(self, data_dict: dict[str, Any]) -> None:
        super().from_dict(data_dict)
        raw_params = data_dict.get("params")
        if not isinstance(raw_params, dict):
            raw_params = data_dict.get("operation_params")
        if isinstance(raw_params, dict):
            params = LayerCopyParams(
                preset_index=raw_params.get("preset_index", 1),
                dz_expr=raw_params.get("dz_expr", "0"),
                expression_params=raw_params.get("expression_params", ""),
                apply_mode=raw_params.get("apply_mode", 0),
                elements=raw_params.get("elements", ""),
                z_range=tuple(raw_params.get("z_range", [-1e6, 1e6])),
                wrap=raw_params.get("wrap", False),
                extend_cell_z=raw_params.get("extend_cell_z", True),
                extra_vacuum=raw_params.get("extra_vacuum", 0.0),
                layers=raw_params.get("layers", 3),
                distance=raw_params.get("distance", 3.0),
            )
        else:
            z_range = data_dict.get("z_range", [-1e6, 1e6])
            extra_vacuum = data_dict.get("extra_vacuum", [0.0])
            layers = data_dict.get("layers", [3])
            distance = data_dict.get("distance", [3.0])
            params = LayerCopyParams(
                preset_index=data_dict.get("preset_index", 1),
                dz_expr=data_dict.get("dz_expr", "0"),
                expression_params=data_dict.get("params", ""),
                apply_mode=data_dict.get("apply_mode", 0),
                elements=data_dict.get("elements", ""),
                z_range=tuple(z_range if isinstance(z_range, (list, tuple)) else [-1e6, 1e6]),
                wrap=data_dict.get("wrap", False),
                extend_cell_z=data_dict.get("extend_cell_z", True),
                extra_vacuum=extra_vacuum[0] if isinstance(extra_vacuum, (list, tuple)) else extra_vacuum,
                layers=layers[0] if isinstance(layers, (list, tuple)) else layers,
                distance=distance[0] if isinstance(distance, (list, tuple)) else distance,
            )
        self.set_params(params)
