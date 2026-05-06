"""UI-independent alloy and composition Make Dataset operations."""

from __future__ import annotations

import math
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
            total_doping = 0
            for rule in params.rules:
                target = rule.get("target")
                dopants = rule.get("dopants", {})
                if not target or not dopants:
                    continue

                groups = rule.get("group")
                if groups and "group" in new_structure.arrays:
                    candidate_indices = [
                        i
                        for i, elem, group in zip(
                            range(len(new_structure)),
                            new_structure,
                            new_structure.arrays["group"],
                        )
                        if elem.symbol == target and group in groups
                    ]
                else:
                    candidate_indices = [i for i, atom in enumerate(new_structure) if atom.symbol == target]

                if not candidate_indices:
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

                for idx, elem in zip(idxs, sample):
                    new_structure[idx].symbol = elem
                total_doping += doping_num

            if total_doping:
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
            count_min, count_max = rule.get("count", [1, 1])
            return int(rng.integers(int(count_min), int(count_max) + 1))

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
