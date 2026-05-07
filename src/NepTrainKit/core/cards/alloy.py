"""UI-independent alloy and composition Make Dataset operations."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from NepTrainKit.core.alloy import (
    assign_random_occupancy,
    parse_composition,
    parse_element_list,
    simplex_grid_points,
    simplex_sobol_points,
)
from NepTrainKit.core.config_type import append_config_tag, stable_config_id

from .operation import StructureOperation


def sample_dopants(
    dopant_list,
    ratios,
    n_items,
    exact: bool = False,
    rng: np.random.Generator | None = None,
    ratio_type: str = "atom",
) -> list:
    """Sample dopant elements from atom or mass ratios."""
    if rng is None:
        rng = np.random.default_rng()

    dopant_list = list(dopant_list)
    ratios = np.array(ratios, dtype=float)

    if ratio_type == "mass":
        masses = np.array([atomic_masses[atomic_numbers.get(elem, 1)] for elem in dopant_list])
        atom_ratios = ratios / masses
        atom_ratios = atom_ratios / atom_ratios.sum()
    else:
        atom_ratios = ratios / ratios.sum()

    if not exact:
        return list(rng.choice(dopant_list, size=n_items, p=atom_ratios, replace=True))

    counts = (atom_ratios * n_items).astype(int)
    diff = n_items - counts.sum()
    if diff != 0:
        max_idx = np.argmax(atom_ratios)
        counts[max_idx] += diff

    arr = np.repeat(dopant_list, counts)
    rng.shuffle(arr)
    return list(arr)


@dataclass(frozen=True)
class RandomDopingParams:
    """Parameters for random site doping."""

    rules: list[dict[str, Any]] = field(default_factory=list)
    doping_type: str = "Random"
    max_structures: int = 1
    use_seed: bool = False
    seed: int = 0


class RandomDopingOperation(StructureOperation):
    """Perform random atomic substitutions according to explicit rules."""

    def run_structure(self, structure, params: RandomDopingParams) -> list:
        if not isinstance(params.rules, list) or not params.rules:
            return [structure]

        structure_list = []
        exact = params.doping_type == "Exact"
        base_seed = int(params.seed) if params.use_seed else None
        rng = np.random.default_rng(base_seed)

        for _ in range(int(params.max_structures)):
            new_structure = structure.copy()
            symbols = np.asarray(new_structure.get_chemical_symbols(), dtype=object)
            total_doping = 0
            for rule in params.rules:
                target = rule.get("target")
                dopants = rule.get("dopants", {})
                if not target or not dopants:
                    continue

                groups = rule.get("group")
                if groups and "group" in new_structure.arrays:
                    group_values = np.asarray(new_structure.arrays["group"], dtype=object)
                    candidate_indices = np.nonzero((symbols == target) & np.isin(group_values, list(groups)))[0]
                else:
                    candidate_indices = np.nonzero(symbols == target)[0]

                if len(candidate_indices) == 0:
                    continue

                doping_num = self._doping_count(new_structure, candidate_indices, target, dopants, rule, rng)
                doping_num = min(doping_num, len(candidate_indices))
                idxs = rng.choice(candidate_indices, doping_num, replace=False)

                dopant_list = list(dopants.keys())
                ratios = np.array(list(dopants.values()), dtype=float)
                sample = sample_dopants(
                    dopant_list,
                    ratios,
                    doping_num,
                    exact,
                    rng=rng,
                    ratio_type=rule.get("ratio_type", "atom"),
                )

                symbols[np.asarray(idxs, dtype=int)] = np.asarray(sample, dtype=object)
                total_doping += doping_num

            if total_doping:
                new_structure.set_chemical_symbols(symbols.tolist())
                append_config_tag(new_structure, f"Dop(n={total_doping})")
            structure_list.append(new_structure)

        return structure_list

    def _doping_count(self, structure, candidate_indices, target, dopants, rule, rng) -> int:
        use_mode = rule.get("use", "atomic_percent")

        if use_mode == "atomic_percent":
            percent_min, percent_max = rule.get("percent", [0.0, 100.0])
            value = rng.uniform(float(percent_min), float(percent_max)) / 100.0
            return max(1, int(len(candidate_indices) * value))

        if use_mode == "mass_percent":
            percent_min, percent_max = rule.get("percent", [0.0, 100.0])
            target_mass_percent = rng.uniform(float(percent_min), float(percent_max)) / 100.0

            target_mass = atomic_masses[atomic_numbers.get(target, 1)]
            total_target_mass = len(candidate_indices) * target_mass
            dopant_elements = list(dopants.keys())
            if dopant_elements:
                avg_dopant_mass = np.mean(
                    [atomic_masses[atomic_numbers.get(elem, 1)] for elem in dopant_elements]
                )
            else:
                avg_dopant_mass = target_mass

            doped_mass = total_target_mass * target_mass_percent
            return max(1, int(doped_mass / avg_dopant_mass))

        if use_mode == "count":
            count_values = list(rule.get("count", [1, 1]))
            if not count_values:
                return 1
            count_min = int(count_values[0])
            count_max = int(count_values[-1])
            count_mode = str(rule.get("count_mode", "")).lower()
            if count_mode == "fixed" or (not count_mode and count_min == count_max):
                return count_min
            if count_max < count_min:
                count_min, count_max = count_max, count_min
            return int(rng.integers(count_min, count_max + 1))

        return len(candidate_indices)


@dataclass(frozen=True)
class CompositionSweepParams:
    """Parameters for composition-space sweeps."""

    elements: str = "Co,Cr,Ni"
    order: str = "2,3,4,5"
    method: str = "Grid"
    step: float = 0.1
    n_points: int = 50
    min_fraction: float = 0.0
    include_endpoints: bool = True
    use_seed: bool = False
    seed: int = 0
    max_outputs: int = 500
    budget_mode: str = "Equal+Reflow"


class CompositionSweepOperation(StructureOperation):
    """Create composition-tagged copies of each input structure."""

    def run_structure(self, structure, params: CompositionSweepParams) -> list:
        elements = parse_element_list(params.elements)
        if len(elements) < 2:
            return [structure]

        orders = [order for order in self._target_orders(params.order) if len(elements) >= order]
        max_outputs = int(params.max_outputs)
        if max_outputs <= 0 or not orders:
            return [structure]

        out = []
        seed = int(params.seed) if params.use_seed else None
        combo_rng = np.random.default_rng(seed) if seed is not None else None

        order_data = []
        capacities = {}
        for order in orders:
            points = self._simplex_points(order, params)
            if not points:
                continue
            combos = list(combinations(elements, order))
            if combo_rng is not None and combos:
                combo_rng.shuffle(combos)
            unique_total = len(combos) * len(points)
            if unique_total <= 0:
                continue
            capacities[order] = int(unique_total)
            order_data.append({"order": order, "points": points, "combos": combos, "capacity": int(unique_total)})

        if not order_data:
            return [structure]

        active_orders = [int(item["order"]) for item in order_data]
        mode = self._budget_mode(params.budget_mode)
        if mode == "weighted_reflow":
            budgets = self._allocate_weighted(active_orders, capacities, max_outputs)
        else:
            budgets = self._allocate_equal(active_orders, max_outputs)

        if budgets.get(active_orders[0], 0) == 0 and max_outputs > 0:
            budgets[active_orders[0]] = 1

        if mode == "equal_legacy":
            emit = {
                order: int(min(max(0, int(budgets.get(order, 0))), max(0, int(capacities.get(order, 0)))))
                for order in active_orders
            }
        else:
            emit = self._reflow_budget(active_orders, budgets, capacities, max_outputs)

        for item in order_data:
            order = int(item["order"])
            points = item["points"]
            combos = item["combos"]
            order_budget = int(emit.get(order, 0))
            if order_budget <= 0:
                continue
            unique_total = int(item["capacity"])
            n_emit = min(order_budget, unique_total)
            slot_seed = None if seed is None else int(seed + order * 104729)
            slots = self._spread_slots(unique_total, n_emit, seed=slot_seed)
            for slot in slots:
                combo_idx = int(slot % len(combos))
                point_idx = int(slot // len(combos))
                elems = combos[combo_idx]
                frac = points[point_idx]
                comp = {elem: float(value) for elem, value in zip(elems, frac)}
                new_structure = structure.copy()
                tag = ",".join(f"{elem}={comp[elem]:.4g}" for elem in elems)
                append_config_tag(new_structure, f"Comp({tag})")
                out.append(new_structure)
                if len(out) >= max_outputs:
                    return out
        return out or [structure]

    def _target_orders(self, text: str) -> list[int]:
        text = (text or "").strip()
        legacy_map = {
            "Binary": [2],
            "Ternary": [3],
            "Quaternary": [4],
            "Quinary": [5],
            "Quaternary+Quinary": [4, 5],
            "Binary+Ternary+Quaternary+Quinary": [2, 3, 4, 5],
        }
        if text in legacy_map:
            return legacy_map[text]

        orders = []
        for token in text.replace(" ", "").split(","):
            if not token:
                continue
            try:
                value = int(token)
            except ValueError:
                continue
            if value not in (2, 3, 4, 5):
                continue
            if value not in orders:
                orders.append(value)
        return orders or [2, 3]

    def _simplex_points(self, order: int, params: CompositionSweepParams) -> list[tuple[float, ...]]:
        seed = int(params.seed) if params.use_seed else None
        if params.method == "Sobol":
            return simplex_sobol_points(order, int(params.n_points), seed=seed, min_fraction=float(params.min_fraction))
        points = simplex_grid_points(
            order,
            float(params.step),
            include_endpoints=bool(params.include_endpoints),
            min_fraction=float(params.min_fraction),
        )
        if seed is not None and points:
            rng = np.random.default_rng(seed)
            rng.shuffle(points)
        return points

    def _budget_mode(self, text: str) -> str:
        text = (text or "").strip().lower()
        if "legacy" in text:
            return "equal_legacy"
        if "weight" in text:
            return "weighted_reflow"
        return "equal_reflow"

    @staticmethod
    def _allocate_equal(orders: list[int], max_outputs: int) -> dict[int, int]:
        budgets = {order: 0 for order in orders}
        if not orders or max_outputs <= 0:
            return budgets
        base = max_outputs // len(orders)
        remainder = max_outputs - base * len(orders)
        for i, order in enumerate(orders):
            budgets[order] = base + (1 if i < remainder else 0)
        return budgets

    @staticmethod
    def _allocate_weighted(orders: list[int], capacities: dict[int, int], max_outputs: int) -> dict[int, int]:
        budgets = {order: 0 for order in orders}
        if not orders or max_outputs <= 0:
            return budgets
        total_cap = int(sum(max(0, int(capacities.get(order, 0))) for order in orders))
        if total_cap <= 0:
            return budgets

        raw = [float(max_outputs) * float(max(0, int(capacities.get(order, 0)))) / float(total_cap) for order in orders]
        floors = [int(np.floor(value)) for value in raw]
        for order, value in zip(orders, floors):
            budgets[order] = int(value)
        remaining = int(max_outputs - sum(floors))
        if remaining > 0:
            frac_rank = sorted(range(len(orders)), key=lambda i: (raw[i] - floors[i], -i), reverse=True)
            for i in frac_rank[:remaining]:
                budgets[orders[i]] += 1
        return budgets

    @staticmethod
    def _reflow_budget(
        orders: list[int],
        budget: dict[int, int],
        capacities: dict[int, int],
        max_outputs: int,
    ) -> dict[int, int]:
        emit = {
            order: int(min(max(0, int(budget.get(order, 0))), max(0, int(capacities.get(order, 0)))))
            for order in orders
        }
        remaining = int(max_outputs) - int(sum(emit.values()))
        if remaining <= 0:
            return emit

        active = [order for order in orders if int(capacities.get(order, 0)) > emit[order]]
        while remaining > 0 and active:
            n_active = len(active)
            share = max(remaining // n_active, 1)
            next_active = []
            progressed = False
            for order in active:
                room = int(capacities.get(order, 0)) - emit[order]
                if room <= 0:
                    continue
                add = int(min(room, share))
                if add > 0:
                    emit[order] += add
                    remaining -= add
                    progressed = True
                if emit[order] < int(capacities.get(order, 0)):
                    next_active.append(order)
                if remaining <= 0:
                    break
            if not progressed:
                break
            active = next_active
        return emit

    @staticmethod
    def _coprime_stride(total: int, hint: int) -> int:
        total = int(total)
        if total <= 1:
            return 1
        stride = int(hint) % total
        if stride <= 0:
            stride = 1
        while math.gcd(stride, total) != 1:
            stride += 1
            if stride >= total:
                stride = 1
        return stride

    @classmethod
    def _spread_slots(cls, total: int, n_pick: int, *, seed: int | None = None) -> list[int]:
        total = int(total)
        n_pick = int(n_pick)
        if total <= 0 or n_pick <= 0:
            return []
        if n_pick >= total:
            return list(range(total))
        if seed is None:
            start = 0
            stride_hint = int(total * 0.6180339887498949)
        else:
            rng = np.random.default_rng(int(seed))
            start = int(rng.integers(0, total))
            stride_hint = int(rng.integers(1, total))
        stride = cls._coprime_stride(total, stride_hint)
        return [int((start + i * stride) % total) for i in range(n_pick)]


@dataclass(frozen=True)
class CompositionGradientParams:
    elements: str = "Ni,Co"
    start_composition: str = "Ni:1,Co:0"
    end_composition: str = "Ni:0,Co:1"
    axis: str = "x"
    bins: int = 8
    target_elements: str = ""
    samples: int = 1
    use_seed: bool = False
    seed: int = 0


class CompositionGradientOperation(StructureOperation):
    """Assign site species from a composition gradient along one Cartesian axis."""

    AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

    def run_structure(self, structure, params: CompositionGradientParams) -> list:
        elements = parse_element_list(params.elements)
        if len(elements) < 2:
            raise ValueError("Composition Gradient requires at least two elements.")
        start_comp = self._normalized_composition(params.start_composition, elements)
        end_comp = self._normalized_composition(params.end_composition, elements)
        if not start_comp or not end_comp:
            raise ValueError("Composition Gradient requires valid start and end compositions.")

        candidate_indices = self._candidate_indices(structure, params.target_elements)
        if candidate_indices.size == 0:
            raise ValueError("Composition Gradient found no atoms matching target_elements.")

        bins = max(1, int(params.bins))
        axis_idx = self.AXIS_INDEX.get(str(params.axis).strip().lower(), 0)
        coord = self._axis_coordinate(structure, axis_idx)
        order = candidate_indices[np.argsort(coord[candidate_indices], kind="mergesort")]
        groups = [group for group in np.array_split(order, bins) if len(group) > 0]
        if not groups:
            raise ValueError("Composition Gradient could not build nonempty coordinate bins.")

        base_seed = int(params.seed) if params.use_seed else None
        cfg_id = stable_config_id(structure)
        outputs = []
        for sample_idx in range(max(int(params.samples), 1)):
            if base_seed is None:
                rng = np.random.default_rng()
                seed_tag = ""
            else:
                derived_seed = int(base_seed + cfg_id * 1000003 + sample_idx)
                rng = np.random.default_rng(derived_seed)
                seed_tag = f",s={derived_seed}"
            atoms = structure.copy()
            symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
            for group_idx, indices in enumerate(groups):
                t = 0.0 if len(groups) == 1 else float(group_idx) / float(len(groups) - 1)
                comp = {
                    element: (1.0 - t) * float(start_comp[element]) + t * float(end_comp[element])
                    for element in elements
                }
                assigned = self._exact_layer_assignment(elements, comp, len(indices), rng)
                symbols[np.asarray(indices, dtype=int)] = assigned
            atoms.set_chemical_symbols(symbols.tolist())
            append_config_tag(atoms, f"CompGrad(ax={self._axis_name(axis_idx)},b={len(groups)}{seed_tag})")
            outputs.append(atoms)
        return outputs

    @staticmethod
    def _normalized_composition(text: str, elements: list[str]) -> dict[str, float]:
        parsed = parse_composition(text)
        values = np.asarray([float(parsed.get(element, 0.0)) for element in elements], dtype=float)
        if values.size != len(elements) or np.any(values < 0.0) or float(values.sum()) <= 0.0:
            return {}
        values = values / float(values.sum())
        return {element: float(value) for element, value in zip(elements, values)}

    @staticmethod
    def _candidate_indices(structure, target_elements: str) -> np.ndarray:
        targets = set(parse_element_list(target_elements))
        if not targets:
            return np.arange(len(structure), dtype=int)
        return np.asarray(
            [idx for idx, symbol in enumerate(structure.get_chemical_symbols()) if symbol in targets],
            dtype=int,
        )

    @staticmethod
    def _axis_coordinate(structure, axis_idx: int) -> np.ndarray:
        if bool(np.asarray(structure.pbc, dtype=bool)[axis_idx]) and float(structure.get_volume()) > 0.0:
            return np.asarray(structure.get_scaled_positions(wrap=True), dtype=float)[:, axis_idx]
        return np.asarray(structure.get_positions(), dtype=float)[:, axis_idx]

    @staticmethod
    def _axis_name(axis_idx: int) -> str:
        return ("x", "y", "z")[int(axis_idx)]

    @staticmethod
    def _exact_layer_assignment(elements: list[str], comp: dict[str, float], n_sites: int, rng: np.random.Generator) -> np.ndarray:
        fractions = np.asarray([float(comp[element]) for element in elements], dtype=float)
        fractions = fractions / float(fractions.sum())
        raw = fractions * int(n_sites)
        counts = np.floor(raw).astype(int)
        remainder = int(n_sites) - int(counts.sum())
        if remainder > 0:
            residual_order = np.argsort(-(raw - counts))
            for i in range(remainder):
                counts[int(residual_order[i % len(residual_order)])] += 1
        assigned: list[str] = []
        for element, count in zip(elements, counts):
            assigned.extend([element] * int(count))
        rng.shuffle(assigned)
        return np.asarray(assigned, dtype=object)


@dataclass(frozen=True)
class RandomOccupancyParams:
    """Parameters for random occupancy assignment."""

    source: str = "Auto (Comp tag)"
    manual: str = ""
    mode: str = "Exact"
    samples: int = 1
    group_filter: str = ""
    use_seed: bool = False
    seed: int = 0


class RandomOccupancyOperation(StructureOperation):
    """Assign elements to sites from a target composition."""

    def run_structure(self, structure, params: RandomOccupancyParams) -> list:
        comp = self._read_composition(structure, params)
        if not comp:
            return [structure]

        indices = self._eligible_indices(structure, params.group_filter)
        base_seed = int(params.seed) if params.use_seed else None
        cfg_id = stable_config_id(structure)

        out = []
        for sample_idx in range(max(int(params.samples), 1)):
            if base_seed is None:
                rng = np.random.default_rng()
                seed_note = ""
            else:
                derived_seed = int(base_seed + cfg_id * 1000003 + sample_idx)
                rng = np.random.default_rng(derived_seed)
                seed_note = f",s={derived_seed}"

            new_atoms = assign_random_occupancy(structure, comp, indices=indices, mode=params.mode, rng=rng)
            mode_tag = "E" if params.mode.lower().startswith("exact") else "R"
            append_config_tag(new_atoms, f"Occ({mode_tag}{seed_note})")
            out.append(new_atoms)
        return out

    def _read_composition(self, structure, params: RandomOccupancyParams) -> dict[str, float]:
        if params.source.lower().startswith("auto"):
            comp = self._read_comp_from_config_type(structure)
            if comp:
                return comp
        manual = params.manual.strip()
        if not manual:
            return {}
        return parse_composition(manual)

    @staticmethod
    def _read_comp_from_config_type(structure) -> dict[str, float]:
        cfg = str(structure.info.get("Config_type", "") or "")
        if not cfg:
            return {}
        for token in cfg.split("|"):
            token = token.strip()
            if token.startswith("Comp(") and token.endswith(")"):
                inner = token[5:-1].strip()
                if inner:
                    return parse_composition(inner)
        return {}

    @staticmethod
    def _eligible_indices(structure, groups_text: str) -> np.ndarray | None:
        groups_text = groups_text.strip()
        if not groups_text or "group" not in structure.arrays:
            return None
        allowed = {group.strip() for group in groups_text.split(",") if group.strip()}
        if not allowed:
            return None
        groups = structure.arrays["group"]
        return np.array([i for i, group in enumerate(groups) if str(group) in allowed], dtype=int)


def normalize_condition_expr(expr: str) -> str:
    """Convert card condition syntax to a Python boolean expression."""
    expr = expr.strip()
    if not expr or expr.lower() == "all":
        return "True"
    expr = re.sub(r"\bAND\b", "and", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bOR\b", "or", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bNOT\b", "not", expr, flags=re.IGNORECASE)
    return re.sub(r"(?<![<>!])=(?!=)", "==", expr)


def _is_allowed_condition_node(node: ast.AST) -> bool:
    if isinstance(node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
        return True
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
    if not isinstance(node, allowed_nodes):
        return False
    for child in ast.iter_child_nodes(node):
        if not _is_allowed_condition_node(child):
            return False
    if isinstance(node, ast.BoolOp) and not isinstance(node.op, (ast.And, ast.Or)):
        return False
    if isinstance(node, ast.UnaryOp) and not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        return False
    if isinstance(node, ast.BinOp) and not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        return False
    if isinstance(node, ast.Compare):
        return all(isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops)
    return True


def _eval_condition_node(node: ast.AST, env: dict[str, float], tol: float) -> float | bool:
    if isinstance(node, ast.Expression):
        return _eval_condition_node(node.body, env, tol)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"Unknown name '{node.id}'")
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        val = _eval_condition_node(node.operand, env, tol)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.Not):
            return not bool(val)
    if isinstance(node, ast.BinOp):
        left = _eval_condition_node(node.left, env, tol)
        right = _eval_condition_node(node.right, env, tol)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    if isinstance(node, ast.Compare):
        left = _eval_condition_node(node.left, env, tol)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_condition_node(comparator, env, tol)
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
        vals = [_eval_condition_node(value, env, tol) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(bool(value) for value in vals)
        if isinstance(node.op, ast.Or):
            return any(bool(value) for value in vals)
    raise ValueError("Unsupported expression")


def evaluate_condition(expr: str, coords: np.ndarray) -> bool | np.ndarray:
    """Safely evaluate a coordinate condition against one or more positions."""
    expr_py = normalize_condition_expr(expr)
    tree = ast.parse(expr_py, mode="eval")
    if not _is_allowed_condition_node(tree):
        raise ValueError("Condition expression contains unsupported syntax.")
    coords_arr = np.asarray(coords, dtype=float)

    def eval_single(pos) -> bool:
        x, y, z = map(float, pos[:3])
        return bool(_eval_condition_node(tree, {"x": x, "y": y, "z": z}, tol=1e-4))

    if coords_arr.ndim == 1:
        return eval_single(coords_arr)
    if coords_arr.ndim == 2:
        return np.array([eval_single(position) for position in coords_arr], dtype=bool)
    raise ValueError(f"Unsupported coordinate shape: {coords_arr.shape}")


def parse_replacements(text: str) -> tuple[list[str], list[float]]:
    """Parse replacement spec like ``Cs:0.6,Na:0.4`` or a JSON mapping."""
    names: list[str] = []
    ratios: list[float] = []
    text = (text or "").strip()
    if not text:
        return names, ratios

    if text.startswith("{") and text.endswith("}"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Replacement JSON must be an object.")
        for key, value in data.items():
            name = str(key).strip()
            ratio = float(value)
            if name and ratio >= 0:
                names.append(name)
                ratios.append(ratio)
        return names, ratios

    for token in (item for item in text.split(",") if item.strip()):
        if ":" in token:
            key, value = token.split(":", 1)
            name = key.strip()
            ratio = float(value)
        else:
            name = token.strip()
            ratio = 1.0
        if name and ratio >= 0:
            names.append(name)
            ratios.append(ratio)
    return names, ratios


@dataclass(frozen=True)
class ConditionalReplaceParams:
    """Parameters for coordinate-gated atomic replacement."""

    target: str = ""
    replacements: str = ""
    condition: str = ""
    seed: int = 0
    mode: int = 0


class ConditionalReplaceOperation(StructureOperation):
    """Replace atoms that match target species and coordinate condition."""

    def run_structure(self, structure, params: ConditionalReplaceParams) -> list:
        target = params.target.strip()
        if not target:
            return [structure]

        new_atoms, ratios = parse_replacements(params.replacements)
        if not new_atoms or len(ratios) != len(new_atoms):
            raise ValueError("Replacements must be provided as elem:ratio entries.")

        seed = int(params.seed) if int(params.seed) != 0 else None
        exact = int(params.mode) == 1
        new_structure, replaced = replace_atoms_with_conditions(
            structure,
            atom_to_replace=target,
            new_atoms=new_atoms,
            probabilities=ratios,
            condition=params.condition.strip() or "all",
            seed=seed,
            exact=exact,
        )
        if replaced:
            append_config_tag(new_structure, f"Repl({target}->{','.join(new_atoms)})")
        return [new_structure]


def replace_atoms_with_conditions(
    structure,
    atom_to_replace: str,
    new_atoms: list[str],
    probabilities: list[float],
    condition: str,
    seed: int | None = None,
    exact: bool = False,
):
    """Replace atoms in a structure using a probability distribution and coordinate condition."""
    symbols = structure.get_chemical_symbols()
    positions = structure.get_positions()
    target_mask = np.asarray(symbols, dtype=object) == atom_to_replace
    condition_result = evaluate_condition(condition, np.asarray(positions, dtype=float))
    if isinstance(condition_result, np.ndarray):
        condition_mask = np.asarray(condition_result, dtype=bool)
    else:
        condition_mask = np.full(len(symbols), bool(condition_result), dtype=bool)
    target_indices = np.nonzero(target_mask & condition_mask)[0]
    if len(target_indices) == 0:
        return structure, 0

    probs = np.asarray(probabilities, dtype=float)
    if probs.size != len(new_atoms) or probs.size == 0:
        raise ValueError("Replacement probabilities must match replacement atoms.")
    if np.all(probs <= 0):
        raise ValueError("At least one replacement probability must be positive.")
    probs = probs / probs.sum()

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(target_indices)
    if exact:
        sampled = _exact_replacement_sample(new_atoms, probs, len(shuffled), rng)
    else:
        sampled = rng.choice(new_atoms, size=len(shuffled), p=probs, replace=True)

    new_structure = structure.copy()
    new_symbols = list(symbols)
    for idx, elem in zip(shuffled.tolist(), sampled.tolist()):
        new_symbols[int(idx)] = str(elem)
    new_structure.set_chemical_symbols(new_symbols)
    return new_structure, len(shuffled)


def _exact_replacement_sample(new_atoms: list[str], probs: np.ndarray, total: int, rng: np.random.Generator) -> np.ndarray:
    raw_counts = probs * total
    counts = np.floor(raw_counts).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        residuals = raw_counts - counts
        order = np.argsort(-residuals)
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    sampled: list[str] = []
    for name, count in zip(new_atoms, counts):
        sampled.extend([name] * int(count))
    rng.shuffle(sampled)
    return np.array(sampled, dtype=object)
