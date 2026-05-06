"""UI-independent structure-generation Make Dataset operations."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from NepTrainKit.core.config_type import append_config_tag

from .operation import StructureOperation


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


def validate_dz_expr(expr: str, allowed_names: set[str]) -> ast.Expression:
    """Validate a dz expression and return its AST."""
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
    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor)
    allowed_unaryops = (ast.UAdd, ast.USub)
    allowed_boolops = (ast.And, ast.Or)
    allowed_cmpops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    for node in ast.walk(tree):
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


def parse_dz_params(text: str) -> dict[str, float]:
    """Parse scalar expression parameters from ``name=value`` chunks."""
    params: dict[str, float] = {}
    chunks = [chunk.strip() for chunk in re.split(r"[,\n;]+", text or "") if chunk.strip()]
    for chunk in chunks:
        if "=" not in chunk:
            raise ValueError(f"Invalid param '{chunk}', expected name=value")
        name, value_expr = chunk.split("=", 1)
        name = name.strip()
        value_expr = value_expr.strip()
        if not _NAME_RE.match(name):
            raise ValueError(f"Invalid parameter name '{name}'")
        allowed_names = set(_ALLOWED_FUNCS) | {"pi", "e"} | set(params)
        tree = validate_dz_expr(value_expr, allowed_names=allowed_names)
        code = compile(tree, "<param>", "eval")
        env: dict[str, Any] = dict(_ALLOWED_FUNCS)
        env.update(params)
        env["pi"] = math.pi
        env["e"] = math.e
        val = eval(code, {"__builtins__": {}}, env)  # noqa: S307
        val = float(np.asarray(val).reshape(-1)[0])
        if not np.isfinite(val):
            raise ValueError(f"Parameter '{name}' is not finite")
        params[name] = val
    return params


def evaluate_dz_expression(expr: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Evaluate a validated dz expression over selected coordinates."""
    allowed_names = set(_ALLOWED_FUNCS) | {"x", "y", "z", "pi", "e"} | set(params)
    tree = validate_dz_expr(expr, allowed_names=allowed_names)
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
    """Stack copies of positions along z."""
    num_layers = max(1, int(num_layers))
    offsets = np.arange(num_layers, dtype=float) * float(layer_distance)
    layers = []
    for offset in offsets:
        shifted = base_positions.copy()
        shifted[:, 2] = shifted[:, 2] + offset
        layers.append(shifted)
    return layers


@dataclass(frozen=True)
class LayerCopyParams:
    """Parameters for surface warp and layer-copy generation."""

    preset_index: int = 1
    dz_expr: str = "sin(x/pi) + sin(y/pi)"
    expression_params: str = ""
    apply_mode: int = 0
    elements: str = ""
    z_range: tuple[float, float] = (-1000000.0, 1000000.0)
    wrap: bool = False
    extend_cell_z: bool = True
    extra_vacuum: float = 0.0
    layers: int = 3
    distance: float = 3.0


class LayerCopyOperation(StructureOperation):
    """Warp selected atoms by dz=f(x,y,z) and copy into stacked layers."""

    def run_structure(self, structure, params: LayerCopyParams) -> list:
        if not params.dz_expr.strip():
            raise ValueError("LayerCopy: dz expression is empty.")

        base = structure.copy()
        positions = np.asarray(base.get_positions(), dtype=float)
        mask = self.apply_mask(base, params)
        if not np.any(mask):
            raise ValueError("LayerCopy: no atoms selected by apply settings.")

        expr_params = parse_dz_params(params.expression_params)
        warped_positions = positions.copy()
        dz = evaluate_dz_expression(
            params.dz_expr.strip(),
            x=positions[mask, 0],
            y=positions[mask, 1],
            z=positions[mask, 2],
            params=expr_params,
        )
        warped_positions[mask, 2] = warped_positions[mask, 2] + dz

        layers = build_layers(warped_positions, num_layers=int(params.layers), layer_distance=float(params.distance))
        combined = base.copy()
        combined.set_positions(layers[0])
        for layer_pos in layers[1:]:
            layer_struct = base.copy()
            layer_struct.set_positions(layer_pos)
            combined += layer_struct

        if params.extend_cell_z and hasattr(combined, "cell"):
            base_cell = np.asarray(base.cell.array, dtype=float)
            if base_cell.shape == (3, 3) and int(params.layers) > 1:
                dz_total = abs(float(params.distance)) * (int(params.layers) - 1) + max(0.0, float(params.extra_vacuum))
                if dz_total > 0.0:
                    base_cell = base_cell.copy()
                    base_cell[2, 2] = base_cell[2, 2] + dz_total
                    combined.set_cell(base_cell, scale_atoms=False)

        if params.wrap and hasattr(combined, "wrap"):
            combined.wrap()

        append_config_tag(combined, f"SWC(L={int(params.layers)},dz={float(params.distance):g})")
        return [combined]

    @staticmethod
    def apply_mask(structure, params: LayerCopyParams) -> np.ndarray:
        n_atoms = len(structure)
        mode = int(params.apply_mode)
        if mode == 0:
            return np.ones(n_atoms, dtype=bool)
        if mode == 1:
            elems = [token.strip() for token in re.split(r"[,\s]+", params.elements) if token.strip()]
            if not elems:
                return np.zeros(n_atoms, dtype=bool)
            symbols = np.asarray(structure.get_chemical_symbols(), dtype=object)
            return np.isin(symbols, np.asarray(elems, dtype=object))
        z_min, z_max = [float(value) for value in params.z_range]
        if z_min > z_max:
            z_min, z_max = z_max, z_min
        z = structure.get_positions()[:, 2]
        return (z >= z_min) & (z <= z_max)
