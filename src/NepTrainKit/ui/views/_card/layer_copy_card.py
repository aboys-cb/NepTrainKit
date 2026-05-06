"""Card for warping a structure by dz=f(x,y) and copying it into stacked layers."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtWidgets import QGridLayout
from qfluentwidgets import (
    BodyLabel,
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
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


@CardManager.register_card
class LayerCopyCard(MakeDataCard):
    """Warp structure by dz=f(x,y) then copy-translate along z into a single stacked structure."""

    group = "Structure"
    card_name = "Layer Copy"
    menu_icon = r":/images/src/images/defect.svg"

    _PRESETS: list[tuple[str, str, str]] = [
        ("Custom", "", ""),
        ("Script: sin(x/pi)+sin(y/pi)", "sin(x/pi) + sin(y/pi)", ""),
        ("Sine (2D, params)", "A*(sin(x/Lx) + sin(y/Ly))", "A=1, Lx=3.141592653589793, Ly=3.141592653589793"),
        ("Gaussian bump", "A*exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))", "A=1, x0=0, y0=0, sigma=5"),
        ("Paraboloid", "A*(x**2 + y**2)", "A=0.001"),
        ("Ripple (stripe)", "A*sin(x/Lx)", "A=1, Lx=3.141592653589793"),
        ("Step (x>0)", "where(x > 0, A, 0)", "A=1"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Surface Warp (dz=f(x,y)) + Layer Copy")
        self._build_ui()

    def _build_ui(self):
        layout: QGridLayout = self.settingLayout

        self.preset_label = BodyLabel("dz preset", self.setting_widget)
        self.preset_combo = ComboBox(self.setting_widget)
        self.preset_combo.addItems([name for name, _, _ in self._PRESETS])
        self.preset_combo.setCurrentIndex(1)
        self.preset_label.setToolTip("Choose a preset dz(x,y) expression (Custom keeps your input).")
        self.preset_label.installEventFilter(ToolTipFilter(self.preset_label, 300, ToolTipPosition.TOP))

        self.test_button = TransparentToolButton(FluentIcon.PLAY, self.setting_widget)
        self.test_button.setToolTip("Test dz expression on current structure")
        self.test_button.installEventFilter(ToolTipFilter(self.test_button, 300, ToolTipPosition.TOP))

        self.expr_label = BodyLabel("dz expression (Å)", self.setting_widget)
        self.expr_edit = TextEdit(self.setting_widget)
        self.expr_edit.setPlaceholderText("e.g. sin(x/pi) + sin(y/pi)")
        self.expr_edit.setFixedHeight(70)

        self.params_label = BodyLabel("params", self.setting_widget)
        self.params_edit = LineEdit(self.setting_widget)
        self.params_edit.setPlaceholderText("A=1, Lx=3.14, Ly=3.14  (optional)")

        self.apply_label = BodyLabel("apply to", self.setting_widget)
        self.apply_combo = ComboBox(self.setting_widget)
        self.apply_combo.addItems(["All atoms", "Elements", "Z-range"])

        self.elements_edit = LineEdit(self.setting_widget)
        self.elements_edit.setPlaceholderText("C, Si, O")
        self.elements_edit.setVisible(False)

        self.zrange_frame = SpinBoxUnitInputFrame(self)
        self.zrange_frame.set_input(["Å", "Å"], 2, input_type="float")
        self.zrange_frame.setRange(-1e6, 1e6)
        self.zrange_frame.set_input_value([-1e6, 1e6])
        self.zrange_frame.setVisible(False)

        self.wrap_checkbox = CheckBox("Wrap after warp/copy", self.setting_widget)
        self.wrap_checkbox.setChecked(False)

        self.extend_cell_checkbox = CheckBox("Extend cell along z", self.setting_widget)
        self.extend_cell_checkbox.setChecked(True)

        self.vacuum_label = BodyLabel("extra vacuum (Å)", self.setting_widget)
        self.vacuum_frame = SpinBoxUnitInputFrame(self)
        self.vacuum_frame.set_input("Å", 1, input_type="float")
        self.vacuum_frame.setRange(0.0, 1e6)
        self.vacuum_frame.set_input_value([0.0])

        self.layers_label = BodyLabel("Number of layers", self.setting_widget)
        self.layers_frame = SpinBoxUnitInputFrame(self)
        self.layers_frame.set_input("layers", 1, input_type="int")
        self.layers_frame.setRange(1, 999)

        self.distance_label = BodyLabel("Layer spacing (Å)", self.setting_widget)
        self.distance_frame = SpinBoxUnitInputFrame(self)
        self.distance_frame.set_input("Å", 1, input_type="float")
        self.distance_frame.setRange(-1e4, 1e4)

        layout.addWidget(self.preset_label, 0, 0, 1, 1)
        layout.addWidget(self.preset_combo, 0, 1, 1, 1)
        layout.addWidget(self.test_button, 0, 2, 1, 1)

        layout.addWidget(self.expr_label, 1, 0, 1, 1)
        layout.addWidget(self.expr_edit, 1, 1, 1, 2)

        layout.addWidget(self.params_label, 2, 0, 1, 1)
        layout.addWidget(self.params_edit, 2, 1, 1, 2)

        layout.addWidget(self.apply_label, 3, 0, 1, 1)
        layout.addWidget(self.apply_combo, 3, 1, 1, 2)
        layout.addWidget(self.elements_edit, 4, 1, 1, 2)
        layout.addWidget(self.zrange_frame, 5, 1, 1, 2)

        layout.addWidget(self.extend_cell_checkbox, 6, 0, 1, 1)
        layout.addWidget(self.vacuum_label, 6, 1, 1, 1)
        layout.addWidget(self.vacuum_frame, 6, 2, 1, 1)
        layout.addWidget(self.wrap_checkbox, 7, 0, 1, 3)

        layout.addWidget(self.layers_label, 8, 0, 1, 1)
        layout.addWidget(self.layers_frame, 8, 1, 1, 2)
        layout.addWidget(self.distance_label, 9, 0, 1, 1)
        layout.addWidget(self.distance_frame, 9, 1, 1, 2)

        # Defaults mirroring the provided script
        self.layers_frame.set_input_value([3])
        self.distance_frame.set_input_value([3.0])
        self.expr_edit.setPlainText("sin(x/pi) + sin(y/pi)")

        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.apply_combo.currentIndexChanged.connect(self._on_apply_changed)
        self.test_button.clicked.connect(self._test_expression)

    def _on_preset_changed(self, index: int) -> None:
        if index <= 0 or index >= len(self._PRESETS):
            return
        _, expr, params = self._PRESETS[index]
        if expr:
            self.expr_edit.setPlainText(expr)
        self.params_edit.setText(params or "")

    def _on_apply_changed(self, index: int) -> None:
        # 0: all, 1: elements, 2: z-range
        self.elements_edit.setVisible(index == 1)
        self.zrange_frame.setVisible(index == 2)

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
        self._on_apply_changed(self.apply_combo.currentIndex())

    def _test_expression(self) -> None:
        if not hasattr(self, "dataset") or not self.dataset:
            MessageManager.send_warning_message("No input structure available to test.")
            return
        structure = self.dataset[0]
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
        params = self.get_params()
        operation_params = params_to_dict(params)
        operation_params["z_range"] = list(params.z_range)
        data.update(
            {
                "operation_params": operation_params,
                "preset_index": params.preset_index,
                "dz_expr": params.dz_expr,
                "params": params.expression_params,
                "apply_mode": params.apply_mode,
                "elements": params.elements,
                "z_range": list(params.z_range),
                "wrap": params.wrap,
                "extend_cell_z": params.extend_cell_z,
                "extra_vacuum": [params.extra_vacuum],
                "layers": [params.layers],
                "distance": [params.distance],
            }
        )
        return data

    def from_dict(self, data_dict: dict[str, Any]) -> None:
        super().from_dict(data_dict)
        raw_params = data_dict.get("operation_params")
        if raw_params:
            params = LayerCopyParams(
                preset_index=raw_params.get("preset_index", 1),
                dz_expr=raw_params.get("dz_expr", "sin(x/pi) + sin(y/pi)"),
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
                dz_expr=data_dict.get("dz_expr", "sin(x/pi) + sin(y/pi)"),
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
