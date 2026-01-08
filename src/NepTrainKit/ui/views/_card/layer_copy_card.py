"""Card for warping a structure by dz=f(x,y) and copying it into stacked layers."""

from __future__ import annotations

import ast
import hashlib
import math
import re
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

from NepTrainKit.core import CardManager
from NepTrainKit.core import MessageManager
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


_ALLOWED_FUNCS: dict[str, Any] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "where": np.where,
    "clip": np.clip,
    "min": np.minimum,
    "max": np.maximum,
}


def _validate_dz_expr(expr: str, allowed_names: set[str]) -> ast.Expression:
    expr = expr.strip()
    if not expr:
        raise ValueError("dz expression is empty")
    tree = ast.parse(expr, mode="eval")

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Compare,
        ast.BoolOp,
    )
    allowed_binops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
    )
    allowed_unaryops = (ast.UAdd, ast.USub)
    allowed_boolops = (ast.And, ast.Or)
    allowed_cmpops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    for node in ast.walk(tree):
        # Operator nodes (ast.Add, ast.And, etc.) are part of the AST walk.
        if isinstance(node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
            continue
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.BinOp) and not isinstance(node.op, allowed_binops):
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, allowed_unaryops):
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        if isinstance(node, ast.BoolOp) and not isinstance(node.op, allowed_boolops):
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
        if isinstance(node, ast.Compare) and not all(isinstance(op, allowed_cmpops) for op in node.ops):
            raise ValueError("Unsupported comparison operator")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed (e.g. sin(x))")
            func_name = node.func.id
            if func_name not in _ALLOWED_FUNCS:
                raise ValueError(f"Function '{func_name}' is not allowed")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unknown name '{node.id}'")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            raise ValueError("String constants are not allowed")
    return tree  # pyright: ignore[reportReturnType]


def _parse_params(text: str) -> dict[str, float]:
    params: dict[str, float] = {}
    chunks = [c.strip() for c in re.split(r"[,\n;]+", text or "") if c.strip()]
    for chunk in chunks:
        if "=" not in chunk:
            raise ValueError(f"Invalid param '{chunk}', expected name=value")
        name, value_expr = chunk.split("=", 1)
        name = name.strip()
        value_expr = value_expr.strip()
        if not _NAME_RE.match(name):
            raise ValueError(f"Invalid parameter name '{name}'")

        allowed_names = set(_ALLOWED_FUNCS) | {"pi", "e"} | set(params)
        tree = _validate_dz_expr(value_expr, allowed_names=allowed_names)
        code = compile(tree, "<param>", "eval")
        env: dict[str, Any] = dict(_ALLOWED_FUNCS)
        env.update(params)
        env["pi"] = math.pi
        env["e"] = math.e
        # Evaluate as scalar; numpy functions return numpy scalars/arrays.
        val = eval(code, {"__builtins__": {}}, env)  # noqa: S307
        val = float(np.asarray(val).reshape(-1)[0])
        if not np.isfinite(val):
            raise ValueError(f"Parameter '{name}' is not finite")
        params[name] = val
    return params


def evaluate_dz_expression(expr: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, params: dict[str, float]) -> np.ndarray:
    allowed_names = set(_ALLOWED_FUNCS) | {"x", "y", "z", "pi", "e"} | set(params)
    tree = _validate_dz_expr(expr, allowed_names=allowed_names)
    code = compile(tree, "<dz_expr>", "eval")
    env: dict[str, Any] = dict(_ALLOWED_FUNCS)
    env.update(params)
    env["x"] = x
    env["y"] = y
    env["z"] = z
    env["pi"] = math.pi
    env["e"] = math.e
    out = eval(code, {"__builtins__": {}}, env)  # noqa: S307
    out_arr = np.asarray(out, dtype=float)
    if out_arr.ndim == 0:
        out_arr = np.full_like(x, float(out_arr))
    if out_arr.shape != x.shape:
        raise ValueError(f"dz expression returned shape {out_arr.shape}, expected {x.shape}")
    if not np.all(np.isfinite(out_arr)):
        raise ValueError("dz expression produced NaN/Inf values")
    return out_arr


def build_layers(base_positions: np.ndarray, num_layers: int, layer_distance: float) -> list[np.ndarray]:
    """Stack copies of the positions along z."""
    num_layers = max(1, int(num_layers))
    offsets = np.arange(num_layers, dtype=float) * float(layer_distance)
    layers = []
    for offset in offsets:
        shifted = base_positions.copy()
        shifted[:, 2] = shifted[:, 2] + offset
        layers.append(shifted)
    return layers


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

    def _get_apply_mask(self, structure) -> np.ndarray:
        n = len(structure)
        mode = int(self.apply_combo.currentIndex())
        if mode == 0:
            return np.ones(n, dtype=bool)
        if mode == 1:
            text = self.elements_edit.text().strip()
            elems = [t.strip() for t in re.split(r"[,\s]+", text) if t.strip()]
            if not elems:
                return np.zeros(n, dtype=bool)
            symbols = np.asarray(structure.get_chemical_symbols(), dtype=object)
            return np.isin(symbols, np.asarray(elems, dtype=object))
        z_min, z_max = [float(v) for v in self.zrange_frame.get_input_value()]
        positions = structure.get_positions()
        z = positions[:, 2]
        if z_min > z_max:
            z_min, z_max = z_max, z_min
        return (z >= z_min) & (z <= z_max)

    def _test_expression(self) -> None:
        if not hasattr(self, "dataset") or not self.dataset:
            MessageManager.send_warning_message("No input structure available to test.")
            return
        structure = self.dataset[0]
        expr = self.expr_edit.toPlainText().strip()
        params_text = self.params_edit.text().strip()
        try:
            params = _parse_params(params_text)
            positions = np.asarray(structure.get_positions(), dtype=float)
            mask = self._get_apply_mask(structure)
            if not np.any(mask):
                MessageManager.send_warning_message("No atoms selected by 'apply to' settings.")
                return
            dz = evaluate_dz_expression(
                expr,
                x=positions[mask, 0],
                y=positions[mask, 1],
                z=positions[mask, 2],
                params=params,
            )
            MessageManager.send_info_message(
                f"dz test ok: n={int(mask.sum())}, min={float(np.min(dz)):.6g}, max={float(np.max(dz)):.6g}"
            )
        except Exception as e:  # noqa: BLE001
            MessageManager.send_error_message(f"dz test failed: {e}")

    def process_structure(self, structure):
        num_layers = int(self.layers_frame.get_input_value()[0])
        layer_distance = float(self.distance_frame.get_input_value()[0])
        expr = self.expr_edit.toPlainText().strip()
        params_text = self.params_edit.text().strip()
        wrap = bool(self.wrap_checkbox.isChecked())
        extend_cell = bool(self.extend_cell_checkbox.isChecked())
        extra_vac = float(self.vacuum_frame.get_input_value()[0])

        if not expr:
            MessageManager.send_warning_message("LayerCopy: dz expression is empty.")
            return [structure]

        base = structure.copy()
        positions = np.asarray(base.get_positions(), dtype=float)
        mask = self._get_apply_mask(base)
        if not np.any(mask):
            MessageManager.send_warning_message("LayerCopy: no atoms selected by 'apply to' settings.")
            return [structure]

        try:
            params = _parse_params(params_text)
            warped_positions = positions.copy()
            dz = evaluate_dz_expression(
                expr,
                x=positions[mask, 0],
                y=positions[mask, 1],
                z=positions[mask, 2],
                params=params,
            )
            warped_positions[mask, 2] = warped_positions[mask, 2] + dz
        except Exception as e:  # noqa: BLE001
            MessageManager.send_error_message(f"LayerCopy: dz evaluation failed: {e}")
            return [structure]

        layers = build_layers(warped_positions, num_layers=num_layers, layer_distance=layer_distance)
        if not layers:
            return []

        # Follow the original script semantics: always write a single structure
        # containing num_layers * num_atoms atoms.
        combined = base.copy()
        combined.set_positions(layers[0])
        for layer_pos in layers[1:]:
            layer_struct = base.copy()
            layer_struct.set_positions(layer_pos)
            combined += layer_struct

        if extend_cell and hasattr(combined, "cell"):
            try:
                base_cell = np.asarray(base.cell.array, dtype=float)
                if base_cell.shape == (3, 3) and num_layers > 1:
                    dz_total = abs(layer_distance) * (num_layers - 1) + max(0.0, extra_vac)
                    if dz_total > 0.0:
                        base_cell = base_cell.copy()
                        base_cell[2, 2] = base_cell[2, 2] + dz_total
                        combined.set_cell(base_cell, scale_atoms=False)
            except Exception:
                pass

        if wrap and hasattr(combined, "wrap"):
            try:
                combined.wrap()
            except Exception:
                pass

        expr_hash = hashlib.sha1((expr + "\n" + params_text).encode("utf-8")).hexdigest()[:10]
        combined.info["dz_expression"] = expr
        combined.info["dz_params"] = params_text
        combined.info["dz_hash"] = expr_hash

        tag = combined.info.get("Config_type", "")
        combined.info["Config_type"] = f"{tag} SurfaceWarpCopy(layers={num_layers},dz={layer_distance:g},expr={expr_hash})".strip()
        return [combined]

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "preset_index": self.preset_combo.currentIndex(),
                "dz_expr": self.expr_edit.toPlainText(),
                "params": self.params_edit.text(),
                "apply_mode": self.apply_combo.currentIndex(),
                "elements": self.elements_edit.text(),
                "z_range": self.zrange_frame.get_input_value(),
                "wrap": self.wrap_checkbox.isChecked(),
                "extend_cell_z": self.extend_cell_checkbox.isChecked(),
                "extra_vacuum": self.vacuum_frame.get_input_value(),
                "layers": self.layers_frame.get_input_value(),
                "distance": self.distance_frame.get_input_value(),
            }
        )
        return data

    def from_dict(self, data_dict: dict[str, Any]) -> None:
        super().from_dict(data_dict)
        try:
            self.preset_combo.setCurrentIndex(int(data_dict.get("preset_index", 1)))
        except Exception:
            self.preset_combo.setCurrentIndex(1)
        self.expr_edit.setPlainText(data_dict.get("dz_expr", "sin(x/pi) + sin(y/pi)"))
        self.params_edit.setText(data_dict.get("params", ""))
        try:
            self.apply_combo.setCurrentIndex(int(data_dict.get("apply_mode", 0)))
        except Exception:
            self.apply_combo.setCurrentIndex(0)
        self.elements_edit.setText(data_dict.get("elements", ""))
        z_range = data_dict.get("z_range", [-1e6, 1e6])
        self.zrange_frame.set_input_value(z_range if isinstance(z_range, (list, tuple)) else [-1e6, 1e6])
        self.wrap_checkbox.setChecked(bool(data_dict.get("wrap", False)))
        self.extend_cell_checkbox.setChecked(bool(data_dict.get("extend_cell_z", True)))
        vac = data_dict.get("extra_vacuum", [0.0])
        self.vacuum_frame.set_input_value(vac if isinstance(vac, (list, tuple)) else [vac])

        self.layers_frame.set_input_value(data_dict.get("layers", [3]))
        self.distance_frame.set_input_value(data_dict.get("distance", [3.0]))

        # Ensure conditional widgets match loaded state.
        self._on_apply_changed(self.apply_combo.currentIndex())
