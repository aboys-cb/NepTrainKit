"""Card for replacing atoms based on spatial conditions in the current dataset."""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np
from PySide6.QtWidgets import QGridLayout
from qfluentwidgets import BodyLabel, LineEdit, ToolTipFilter, ToolTipPosition

from NepTrainKit.core import CardManager, MessageManager
from NepTrainKit.ui.widgets import MakeDataCard, SpinBoxUnitInputFrame


def _normalize_condition_expr(expr: str) -> str:
    """Convert custom condition syntax to a Python boolean expression."""
    expr = expr.strip()
    if not expr or expr.lower() == "all":
        return "True"
    # Normalise logical keywords
    expr = re.sub(r"\bAND\b", "and", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bOR\b", "or", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bNOT\b", "not", expr, flags=re.IGNORECASE)
    # Replace bare "=" with "==" while keeping ">=" and "<=" intact
    expr = re.sub(r"(?<![<>!])=(?!=)", "==", expr)
    return expr


def _is_allowed_node(node: ast.AST) -> bool:
    """Restrict AST to safe boolean/compare forms."""
    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.UnaryOp,
        ast.BinOp,
        ast.Not,
    )
    allowed_ops = (
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Not,
    )
    if isinstance(node, allowed_nodes):
        for child in ast.iter_child_nodes(node):
            if not _is_allowed_node(child):
                return False
        if isinstance(node, ast.BoolOp) and not isinstance(node.op, (ast.And, ast.Or)):
            return False
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, (ast.UAdd, ast.USub)):
            return False
        if isinstance(node, ast.BinOp) and not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            return False
        if isinstance(node, ast.Compare):
            if not all(isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
                return False
        return True
    return False


def _eval_node(node: ast.AST, env: dict[str, float], tol: float) -> float | bool:
    """Safely evaluate an AST using a limited set of ops with tolerance for equality."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env, tol)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id, 0.0)
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, env, tol)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.Not):
            return not bool(val)
        raise ValueError("Unsupported unary op")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env, tol)
        right = _eval_node(node.right, env, tol)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("Unsupported binary op")
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env, tol)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, env, tol)
            if isinstance(op, ast.Eq):
                ok = abs(left - right) <= tol
            elif isinstance(op, ast.NotEq):
                ok = abs(left - right) > tol
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right or abs(left - right) <= tol
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right or abs(left - right) <= tol
            else:
                ok = False
            result = result and ok
            left = right
            if not result:
                break
        return result
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, env, tol) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(bool(v) for v in vals)
        if isinstance(node.op, ast.Or):
            return any(bool(v) for v in vals)
    raise ValueError("Unsupported expression")


def evaluate_condition(expr: str, coords: np.ndarray) -> bool | np.ndarray:
    """Safely evaluate a boolean expression against coordinates with tolerance on equality.

    Supports both a single coordinate vector (shape (3,)) and an array of positions
    (shape (N, 3)), returning a boolean or a boolean array respectively.
    """
    expr_py = _normalize_condition_expr(expr)
    coords_arr = np.asarray(coords, dtype=float)

    def _eval_single(pos) -> bool:
        try:
            tree = ast.parse(expr_py, mode="eval")
            if not _is_allowed_node(tree):
                raise ValueError("disallowed AST")
            x, y, z = map(float, pos[:3])
            return bool(_eval_node(tree, {"x": x, "y": y, "z": z}, tol=1e-4))
        except Exception:
            try:
                x, y, z = map(float, pos[:3])
                return bool(eval(expr_py, {"__builtins__": {}}, {"x": x, "y": y, "z": z}))
            except Exception:
                return False

    if coords_arr.ndim == 1:
        return _eval_single(coords_arr)
    if coords_arr.ndim == 2:
        return np.array([_eval_single(p) for p in coords_arr], dtype=bool)
    # Unsupported shape
    return False


def replace_atoms_with_conditions(
    structure,
    atom_to_replace: str,
    new_atoms: list[str],
    probabilities: list[float],
    condition: str,
    seed: int | None = None,
):
    """Replace atoms in a structure using a probability distribution and coordinate condition."""
    rng = np.random.default_rng(seed)
    symbols = structure.get_chemical_symbols()
    positions = structure.get_positions()

    target_indices = []
    for idx, (sym, pos) in enumerate(zip(symbols, positions)):
        if sym != atom_to_replace:
            continue
        if evaluate_condition(condition, np.asarray(pos, dtype=float)):
            target_indices.append(idx)

    if len(target_indices) == 0:
        return structure, 0

    probs = np.asarray(probabilities, dtype=float)
    if probs.size != len(new_atoms) or probs.size == 0:
        return structure, 0
    if np.all(probs <= 0):
        probs = np.ones_like(probs, dtype=float)
    probs = probs / probs.sum()

    shuffled = rng.permutation(target_indices)
    sampled = rng.choice(new_atoms, size=len(shuffled), p=probs, replace=True)

    new_structure = structure.copy()
    for idx, elem in zip(shuffled, sampled):
        new_structure[idx].symbol = elem
    return new_structure, len(shuffled)


@CardManager.register_card
class ConditionalReplaceCard(MakeDataCard):
    """Replace atoms in the active structures using spatial conditions and ratios."""

    group = "Defect"
    card_name = "Conditional Replace"
    menu_icon = r":/images/src/images/defect.svg"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Conditional Atom Replacement")
        self._build_ui()

    def _build_ui(self):
        self.target_label = BodyLabel("Target element", self.setting_widget)
        self.target_edit = LineEdit(self.setting_widget)
        self.target_edit.setPlaceholderText("e.g., O")

        self.new_atoms_label = BodyLabel("New elements", self.setting_widget)
        self.new_atoms_edit = LineEdit(self.setting_widget)
        self.new_atoms_edit.setPlaceholderText("Comma-separated, e.g., F,K,C")

        self.ratio_label = BodyLabel("Ratios", self.setting_widget)
        self.ratio_edit = LineEdit(self.setting_widget)
        self.ratio_edit.setPlaceholderText("Comma-separated, same length as new elements, e.g., 0.1,0.1,0.1")

        self.condition_label = BodyLabel("Condition", self.setting_widget)
        self.condition_edit = LineEdit(self.setting_widget)
        self.condition_edit.setPlaceholderText('Use x, y, z; e.g., "z==2.658", "z>=1.0 and z<=3.0", or "all"')
        self.condition_label.setToolTip('Expression on coordinates; supports x, y, z with >, <, ==, >=, <=, and/or.')
        self.condition_label.installEventFilter(ToolTipFilter(self.condition_label, 300, ToolTipPosition.TOP))

        self.seed_label = BodyLabel("Seed", self.setting_widget)
        self.seed_frame = SpinBoxUnitInputFrame(self)
        self.seed_frame.set_input("seed", 1)
        self.seed_frame.setRange(0, 2**31 - 1)
        self.seed_frame.setToolTip("Random seed (0 leaves it random)")
        self.seed_label.installEventFilter(ToolTipFilter(self.seed_label, 300, ToolTipPosition.TOP))

        layout: QGridLayout = self.settingLayout
        layout.addWidget(self.target_label, 0, 0, 1, 1)
        layout.addWidget(self.target_edit, 0, 1, 1, 2)
        layout.addWidget(self.new_atoms_label, 1, 0, 1, 1)
        layout.addWidget(self.new_atoms_edit, 1, 1, 1, 2)
        layout.addWidget(self.ratio_label, 2, 0, 1, 1)
        layout.addWidget(self.ratio_edit, 2, 1, 1, 2)
        layout.addWidget(self.condition_label, 3, 0, 1, 1)
        layout.addWidget(self.condition_edit, 3, 1, 1, 2)
        layout.addWidget(self.seed_label, 4, 0, 1, 1)
        layout.addWidget(self.seed_frame, 4, 1, 1, 2)

    def _parse_list(self, text: str) -> list[str]:
        return [item.strip() for item in text.split(",") if item.strip()]

    def _parse_float_list(self, text: str) -> list[float]:
        vals: list[float] = []
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                vals.append(float(item))
            except ValueError:
                continue
        return vals

    def process_structure(self, structure):
        target = self.target_edit.text().strip()
        if not target:
            return [structure]
        new_atoms = self._parse_list(self.new_atoms_edit.text())
        ratios = self._parse_float_list(self.ratio_edit.text())
        condition = self.condition_edit.text().strip() or "all"
        seed_val = int(self.seed_frame.get_input_value()[0])
        seed = seed_val if seed_val != 0 else None

        if not new_atoms or len(ratios) != len(new_atoms):
            MessageManager.send_warning_message("Please provide new elements and matching ratios.")
            return [structure]

        symbols = structure.get_chemical_symbols()
        positions = structure.get_positions()
        # Pre-count matches for logging
        target_mask = [s == target for s in symbols]
        target_count = sum(target_mask)
        if target_count == 0:
            MessageManager.send_info_message(f"No atoms with symbol '{target}' found in structure.")
            return [structure]
        target_positions = positions[target_mask]

        x_min = float(np.min(target_positions[:, 0])) if target_positions.size else 0.0
        x_max = float(np.max(target_positions[:, 0])) if target_positions.size else 0.0
        matched = 0
        for sym, pos in zip(symbols, positions):
            if sym == target and evaluate_condition(condition, pos):
                matched += 1

        new_structure, replaced = replace_atoms_with_conditions(
            structure,
            atom_to_replace=target,
            new_atoms=new_atoms,
            probabilities=ratios,
            condition=condition,
            seed=seed,
        )
        if replaced == 0:
            MessageManager.send_info_message(
                f"No atoms matched target='{target}' with condition '{condition}'. "
                f"Target count: {target_count}, matched: {matched}, x-range=({x_min:.4f}, {x_max:.4f})."
            )
        if replaced:
            new_structure.info["Config_type"] = new_structure.info.get("Config_type", "") + f" Replace({target}->{','.join(new_atoms)})"
            MessageManager.send_info_message(
                f"ConditionalReplace: target='{target}', matched={matched}, replaced={replaced}, condition='{condition}', x-range=({x_min:.4f}, {x_max:.4f})."
            )
        return [new_structure]

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "target": self.target_edit.text(),
                "new_atoms": self.new_atoms_edit.text(),
                "ratios": self.ratio_edit.text(),
                "condition": self.condition_edit.text(),
                "seed": self.seed_frame.get_input_value(),
            }
        )
        return data

    def from_dict(self, data_dict: dict[str, Any]) -> None:
        super().from_dict(data_dict)
        self.target_edit.setText(data_dict.get("target", ""))
        self.new_atoms_edit.setText(data_dict.get("new_atoms", ""))
        self.ratio_edit.setText(data_dict.get("ratios", ""))
        self.condition_edit.setText(data_dict.get("condition", ""))
        seed_val = data_dict.get("seed", [0])
        self.seed_frame.set_input_value(seed_val if isinstance(seed_val, (list, tuple)) else [seed_val])
