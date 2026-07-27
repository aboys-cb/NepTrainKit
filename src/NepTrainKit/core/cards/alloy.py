"""UI-independent alloy and composition Make Dataset operations."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.data import atomic_masses, atomic_numbers

from NepTrainKit.core.alloy import (
    assign_random_occupancy,
    best_supercell_factors_max_atoms,
    fractions_to_counts_exact,
    parse_composition,
    parse_element_list,
    simplex_grid_points,
    simplex_sobol_points,
)
from NepTrainKit.core.config_type import append_config_tag, stable_config_id

from .geometry import scaled_positions
from .operation import GeneratorOperation, StructureOperation


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def _range_pair(value: Any, *, label: str) -> tuple[float, float]:
    values = _as_list(value)
    if not values:
        raise ValueError(f"{label} must not be empty.")
    if len(values) == 1:
        low = high = float(values[0])
    elif len(values) == 2:
        low, high = [float(item) for item in values]
    else:
        raise ValueError(f"{label} must contain one or two values.")
    if not np.all(np.isfinite([low, high])):
        raise ValueError(f"{label} values must be finite.")
    if low > high:
        raise ValueError(f"{label} minimum must be <= maximum.")
    return low, high


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
    n_items = int(n_items)
    if n_items < 0:
        raise ValueError("Dopant item count must be non-negative.")
    if not dopant_list:
        raise ValueError("At least one dopant is required.")
    if ratios.size != len(dopant_list) or ratios.size == 0:
        raise ValueError("Dopant ratios must match dopant elements.")
    if np.any(~np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError("Dopant ratios must be finite and non-negative.")
    invalid_elements = [str(elem) for elem in dopant_list if str(elem) not in atomic_numbers]
    if invalid_elements:
        raise ValueError(
            "Unknown dopant element symbol(s): " + ", ".join(invalid_elements) + "."
        )
    if ratio_type not in {"atom", "mass"}:
        raise ValueError("Dopant ratio_type must be 'atom' or 'mass'.")

    if ratio_type == "mass":
        masses = np.array([atomic_masses[atomic_numbers[elem]] for elem in dopant_list])
        atom_ratios = ratios / masses
        total = float(atom_ratios.sum())
    else:
        atom_ratios = ratios
        total = float(atom_ratios.sum())
    if total <= 0.0:
        raise ValueError("At least one dopant ratio must be positive.")
    atom_ratios = atom_ratios / total

    if not exact:
        return list(rng.choice(dopant_list, size=n_items, p=atom_ratios, replace=True))

    counts = fractions_to_counts_exact(atom_ratios, n_items)
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
            return [structure.copy()]

        structure_list = []
        doping_type = str(params.doping_type).strip()
        if doping_type not in {"Random", "Exact"}:
            raise ValueError("RandomDoping: doping_type must be Random or Exact.")
        exact = doping_type == "Exact"
        max_structures = int(params.max_structures)
        if max_structures <= 0:
            raise ValueError("RandomDoping: max_structures must be >= 1.")
        seed = int(params.seed)
        if params.use_seed and seed < 0:
            raise ValueError("RandomDoping: seed must be >= 0.")
        base_seed = seed if params.use_seed else None
        rng = np.random.default_rng(base_seed)

        for _ in range(max_structures):
            new_structure = structure.copy()
            symbols = np.asarray(new_structure.get_chemical_symbols(), dtype=object)
            total_doping = 0
            for rule_index, rule in enumerate(params.rules, start=1):
                label = f"RandomDoping rule {rule_index}"
                if not isinstance(rule, dict):
                    raise ValueError(f"{label} must be a mapping.")
                target = str(rule.get("target", "") or "").strip()
                dopants = rule.get("dopants", {})
                if not target:
                    raise ValueError(f"{label} requires a target element.")
                if not isinstance(dopants, dict):
                    raise ValueError(f"{label} dopants must be an element->ratio mapping.")
                if not dopants:
                    raise ValueError(f"{label} requires at least one dopant element.")
                if target not in atomic_numbers:
                    raise ValueError(f"{label} has an unknown target element '{target}'.")
                invalid_dopants = [
                    str(element)
                    for element in dopants
                    if str(element) not in atomic_numbers
                ]
                if invalid_dopants:
                    raise ValueError(
                        f"{label} has unknown dopant element(s): "
                        + ", ".join(invalid_dopants)
                        + "."
                    )
                if target in dopants:
                    raise ValueError(
                        f"{label} includes target element '{target}' as its own dopant."
                    )

                groups = rule.get("group")
                requested_groups = [
                    str(value).strip()
                    for value in _as_list(groups)
                    if str(value).strip()
                ]
                if requested_groups:
                    if "group" not in new_structure.arrays:
                        raise ValueError(
                            f"{label} requests group labels, but the input structure has no group array."
                        )
                    group_values = np.asarray(new_structure.arrays["group"], dtype=object)
                    candidate_indices = np.nonzero(
                        (symbols == target)
                        & np.isin(group_values, requested_groups)
                    )[0]
                else:
                    candidate_indices = np.nonzero(symbols == target)[0]

                if len(candidate_indices) == 0:
                    scope = (
                        f" in group {','.join(requested_groups)}"
                        if requested_groups
                        else ""
                    )
                    raise ValueError(
                        f"{label} matched no '{target}' atoms{scope}."
                    )

                doping_num = self._doping_count(new_structure, candidate_indices, target, dopants, rule, rng)
                if doping_num < 0:
                    raise ValueError(f"{label} replacement count must be >= 0.")
                if doping_num > len(candidate_indices):
                    raise ValueError(
                        f"{label} requests {doping_num} replacements, but only "
                        f"{len(candidate_indices)} eligible atoms are available."
                    )
                if doping_num == 0:
                    continue
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
            percent_min, percent_max = _range_pair(rule.get("percent", [0.0, 100.0]), label="percent")
            if percent_min < 0.0 or percent_max > 100.0:
                raise ValueError("percent must be within [0, 100].")
            value = rng.uniform(float(percent_min), float(percent_max)) / 100.0
            return int(len(candidate_indices) * value)

        if use_mode == "mass_percent":
            percent_min, percent_max = _range_pair(rule.get("percent", [0.0, 100.0]), label="percent")
            if percent_min < 0.0 or percent_max > 100.0:
                raise ValueError("percent must be within [0, 100].")
            target_mass_percent = rng.uniform(float(percent_min), float(percent_max)) / 100.0

            target_mass = atomic_masses[atomic_numbers[target]]
            total_target_mass = len(candidate_indices) * target_mass
            dopant_elements = list(dopants.keys())
            if dopant_elements:
                avg_dopant_mass = np.mean(
                    [atomic_masses[atomic_numbers[elem]] for elem in dopant_elements]
                )
            else:
                avg_dopant_mass = target_mass

            doped_mass = total_target_mass * target_mass_percent
            return int(doped_mass / avg_dopant_mass)

        if use_mode == "count":
            count_min_f, count_max_f = _range_pair(rule.get("count", [1, 1]), label="count")
            if not float(count_min_f).is_integer() or not float(count_max_f).is_integer():
                raise ValueError("count values must be integers.")
            count_min = int(count_min_f)
            count_max = int(count_max_f)
            if count_min < 0:
                raise ValueError("count values must be >= 0.")
            count_mode = str(rule.get("count_mode", "")).lower()
            if count_mode and count_mode not in {"fixed", "random"}:
                raise ValueError("count_mode must be fixed or random.")
            if count_mode == "fixed" or (not count_mode and count_min == count_max):
                if count_min != count_max:
                    raise ValueError("fixed count must use the same minimum and maximum.")
                return count_min
            return int(rng.integers(count_min, count_max + 1))

        raise ValueError(
            "RandomDoping rule use must be atomic_percent, mass_percent, or count."
        )


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
            return [structure.copy()]

        orders = [order for order in self._target_orders(params.order) if len(elements) >= order]
        max_outputs = int(params.max_outputs)
        if max_outputs <= 0 or not orders:
            return [structure.copy()]

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
            return [structure.copy()]

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
class OrderedAlloyPrototypeParams:
    """Parameters for ordered-alloy prototype generation."""

    prototype: str = "L12/A3B"
    a_range: tuple[float, float, float] = (3.6, 3.6, 0.1)
    covera: float = 1.0
    sublattice_elements: str = "A:X,B:X"
    auto_supercell: bool = True
    max_atoms: int = 128
    rep: tuple[int, int, int] = (2, 2, 2)
    max_outputs: int = 200


@dataclass(frozen=True)
class _PrototypeDefinition:
    key: str
    labels: tuple[str, ...]
    scaled_positions: tuple[tuple[float, float, float], ...]
    cell_kind: str


_ORDERED_PROTOTYPES = {
    "A1": _PrototypeDefinition(
        key="A1",
        labels=("A", "A", "A", "A"),
        scaled_positions=((0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)),
        cell_kind="cubic",
    ),
    "A2": _PrototypeDefinition(
        key="A2",
        labels=("A", "A"),
        scaled_positions=((0, 0, 0), (0.5, 0.5, 0.5)),
        cell_kind="cubic",
    ),
    "A3": _PrototypeDefinition(
        key="A3",
        labels=("A", "A"),
        scaled_positions=((0, 0, 0), (2 / 3, 1 / 3, 0.5)),
        cell_kind="hexagonal",
    ),
    "L12": _PrototypeDefinition(
        key="L12",
        labels=("B", "A", "A", "A"),
        scaled_positions=((0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)),
        cell_kind="cubic",
    ),
    "B2": _PrototypeDefinition(
        key="B2",
        labels=("A", "B"),
        scaled_positions=((0, 0, 0), (0.5, 0.5, 0.5)),
        cell_kind="cubic",
    ),
    "L10": _PrototypeDefinition(
        key="L10",
        labels=("A", "A", "B", "B"),
        scaled_positions=((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)),
        cell_kind="tetragonal",
    ),
}


def _canonical_prototype_name(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]", "", str(text or "")).upper()
    aliases = {
        "A1": "A1",
        "FCC": "A1",
        "A1FCC": "A1",
        "A2": "A2",
        "BCC": "A2",
        "A2BCC": "A2",
        "A3": "A3",
        "HCP": "A3",
        "A3HCP": "A3",
        "L12": "L12",
        "L12A3B": "L12",
        "A3B": "L12",
        "B2": "B2",
        "B2AB": "B2",
        "L10": "L10",
        "L10AB": "L10",
    }
    if normalized not in aliases:
        supported = ", ".join(_ORDERED_PROTOTYPES)
        raise ValueError(f"Ordered Alloy Prototype: unsupported prototype {text!r}; choose one of {supported}.")
    return aliases[normalized]


def _canonical_element(text: str) -> str:
    symbol = str(text or "").strip()
    if not symbol:
        symbol = "X"
    symbol = symbol[0].upper() + symbol[1:].lower()
    if symbol not in atomic_numbers:
        raise ValueError(
            f"Ordered Alloy Prototype: invalid element or placeholder {text!r}. "
            "Use an element symbol or X."
        )
    return symbol


def _parse_sublattice_elements(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        raw: dict[str, Any] = {}
    elif raw_text.startswith("{"):
        try:
            loaded = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ordered Alloy Prototype: invalid sublattice_elements JSON: {exc.msg}.") from exc
        if not isinstance(loaded, dict):
            raise ValueError("Ordered Alloy Prototype: sublattice_elements JSON must be an object.")
        raw = loaded
    else:
        raw = {}
        for token in raw_text.split(","):
            if not token.strip():
                continue
            if ":" not in token:
                raise ValueError("Ordered Alloy Prototype: use label:element entries such as A:Cu,B:Au.")
            label, value = token.split(":", 1)
            raw[label.strip()] = value.strip()

    required = tuple(dict.fromkeys(labels))
    known_labels = {label for definition in _ORDERED_PROTOTYPES.values() for label in definition.labels}
    extra = sorted(set(str(key) for key in raw) - known_labels)
    if extra:
        raise ValueError(f"Ordered Alloy Prototype: unknown sublattice labels: {', '.join(extra)}.")
    return {label: _canonical_element(raw.get(label, "X")) for label in required}


def _scan_lattice_values(values: tuple[float, float, float]) -> list[float]:
    if len(values) != 3:
        raise ValueError("Ordered Alloy Prototype: a_range must contain start, stop, and step.")
    start, stop, step = (float(value) for value in values)
    if not np.all(np.isfinite([start, stop, step])) or start <= 0.0 or stop <= 0.0 or step <= 0.0:
        raise ValueError("Ordered Alloy Prototype: a_range values must be finite and positive.")
    if stop < start:
        start, stop = stop, start
    return [float(value) for value in np.arange(start, stop + 0.5 * step, step, dtype=float)]


class OrderedAlloyPrototypeOperation(GeneratorOperation):
    """Generate periodic prototypes with an independent crystallographic sublattice array."""

    def generate(self, params: OrderedAlloyPrototypeParams) -> list:
        prototype = _canonical_prototype_name(params.prototype)
        definition = _ORDERED_PROTOTYPES[prototype]
        occupants = _parse_sublattice_elements(params.sublattice_elements, definition.labels)
        max_atoms = int(params.max_atoms)
        max_outputs = int(params.max_outputs)
        if max_atoms <= 0:
            raise ValueError("Ordered Alloy Prototype: max_atoms must be >= 1.")
        if max_outputs <= 0:
            raise ValueError("Ordered Alloy Prototype: max_outputs must be >= 1.")
        if len(definition.labels) > max_atoms:
            raise ValueError(
                f"Ordered Alloy Prototype: {prototype} primitive/conventional cell has "
                f"{len(definition.labels)} atoms, exceeding max_atoms={max_atoms}."
            )

        outputs = []
        for a in _scan_lattice_values(params.a_range):
            base = self._build_base(definition, occupants, a, float(params.covera))
            if params.auto_supercell:
                factors = best_supercell_factors_max_atoms(base, max_atoms)
                rep = (factors.na, factors.nb, factors.nc)
            else:
                rep = tuple(int(value) for value in params.rep)
                if len(rep) != 3 or any(value <= 0 for value in rep):
                    raise ValueError("Ordered Alloy Prototype: rep must contain three positive integers.")
            atom_count = len(base) * math.prod(rep)
            if atom_count > max_atoms:
                raise ValueError(
                    f"Ordered Alloy Prototype: rep={rep} produces {atom_count} atoms, "
                    f"exceeding max_atoms={max_atoms}."
                )

            atoms = make_supercell(base, np.diag(rep))
            atoms.wrap()
            metadata = {
                "prototype": prototype,
                "a": float(a),
                "covera": self._effective_covera(definition, float(params.covera)),
                "rep": list(rep),
                "sublattice_elements": occupants,
                "sublattice_counts": {
                    label: int(np.count_nonzero(np.asarray(atoms.arrays["sublattice"], dtype=str) == label))
                    for label in dict.fromkeys(definition.labels)
                },
            }
            atoms.info["ordered_alloy_prototype"] = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            append_config_tag(atoms, f"OrderedProto({prototype},a={a:.6g},rep={rep[0]}x{rep[1]}x{rep[2]})")
            outputs.append(atoms)
            if len(outputs) >= max_outputs:
                break
        return outputs

    @staticmethod
    def _effective_covera(definition: _PrototypeDefinition, covera: float) -> float:
        if definition.cell_kind == "cubic":
            return 1.0
        if not np.isfinite(covera) or covera <= 0.0:
            raise ValueError("Ordered Alloy Prototype: c/a must be finite and positive.")
        return float(covera)

    @classmethod
    def _build_base(
        cls,
        definition: _PrototypeDefinition,
        occupants: dict[str, str],
        a: float,
        covera: float,
    ) -> Atoms:
        effective_covera = cls._effective_covera(definition, covera)
        if definition.cell_kind == "hexagonal":
            cell = np.array(
                [
                    [a, 0.0, 0.0],
                    [-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
                    [0.0, 0.0, a * effective_covera],
                ],
                dtype=float,
            )
        else:
            cell = np.diag([a, a, a * effective_covera])
        symbols = [occupants[label] for label in definition.labels]
        atoms = Atoms(
            symbols=symbols,
            scaled_positions=np.asarray(definition.scaled_positions, dtype=float),
            cell=cell,
            pbc=True,
        )
        atoms.new_array("sublattice", np.asarray(definition.labels, dtype="U8"))
        return atoms


@dataclass(frozen=True)
class FiniteCellAlloyOccupancyParams:
    """Parameters for integer-authoritative finite-cell alloy occupancy."""

    site_rules: str = (
        '{"A":{"composition":{"X":1.0},"elements":["X"],"mode":"fixed_fraction"},'
        '"B":{"composition":{"X":1.0},"elements":["X"],"mode":"fixed_fraction"}}'
    )
    arrangements_per_composition: int = 1
    use_seed: bool = True
    seed: int = 0
    max_outputs: int = 200


@dataclass(frozen=True)
class FiniteCellAlloyEstimate:
    """Queryable pre-run upper-bound estimate for one input structure."""

    composition_count: int
    arrangements_per_composition: int
    estimated_total_outputs: int
    max_outputs: int
    site_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _CountSpace:
    label: str
    elements: tuple[str, ...]
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    n_sites: int
    realization: str
    requested: tuple[tuple[str, tuple[float, ...]], ...]

    def _ways(self, position: int, remaining: int) -> int:
        return _bounded_composition_ways(self.lower, self.upper, position, remaining)

    @property
    def capacity(self) -> int:
        return self._ways(0, self.n_sites)

    def unrank(self, rank: int) -> tuple[int, ...]:
        if rank < 0 or rank >= self.capacity:
            raise IndexError(f"Composition rank {rank} is outside site set {self.label!r}.")
        remaining = self.n_sites
        counts: list[int] = []
        for position in range(len(self.elements)):
            for count in range(self.lower[position], min(self.upper[position], remaining) + 1):
                ways = self._ways(position + 1, remaining - count)
                if rank < ways:
                    counts.append(count)
                    remaining -= count
                    break
                rank -= ways
            else:  # pragma: no cover - guarded by capacity/unrank invariants
                raise RuntimeError("Finite-cell composition unranking failed.")
        return tuple(counts)


@lru_cache(maxsize=65_536)
def _bounded_composition_ways(
    lower: tuple[int, ...],
    upper: tuple[int, ...],
    position: int,
    remaining: int,
) -> int:
    if position == len(lower):
        return int(remaining == 0)
    minimum = lower[position]
    maximum = min(upper[position], remaining)
    if maximum < minimum:
        return 0
    return sum(
        _bounded_composition_ways(lower, upper, position + 1, remaining - count)
        for count in range(minimum, maximum + 1)
    )


class FiniteCellAlloyOccupancyOperation(StructureOperation):
    """Assign unique alloy occupancies from feasible integer counts per site set."""

    _ENUMERATION_LIMIT = 20_000

    def __init__(self, *, require_normalized_fixed_fractions: bool = False) -> None:
        self.require_normalized_fixed_fractions = bool(
            require_normalized_fixed_fractions
        )

    def estimate(self, structure, params: FiniteCellAlloyOccupancyParams) -> FiniteCellAlloyEstimate:
        site_indices = self._site_indices(structure)
        spaces = self._build_spaces(
            site_indices,
            params.site_rules,
            require_normalized_fixed_fractions=self.require_normalized_fixed_fractions,
        )
        composition_count = math.prod(space.capacity for space in spaces)
        per_composition = int(params.arrangements_per_composition)
        max_outputs = int(params.max_outputs)
        if per_composition <= 0:
            raise ValueError("Finite-Cell Alloy Occupancy: arrangements_per_composition must be >= 1.")
        if max_outputs <= 0:
            raise ValueError("Finite-Cell Alloy Occupancy: max_outputs must be >= 1.")
        return FiniteCellAlloyEstimate(
            composition_count=int(composition_count),
            arrangements_per_composition=per_composition,
            estimated_total_outputs=int(min(composition_count * per_composition, max_outputs)),
            max_outputs=max_outputs,
            site_counts=tuple((label, len(indices)) for label, indices in site_indices.items()),
        )

    def run_structure(self, structure, params: FiniteCellAlloyOccupancyParams) -> list:
        site_indices = self._site_indices(structure)
        spaces = self._build_spaces(
            site_indices,
            params.site_rules,
            require_normalized_fixed_fractions=self.require_normalized_fixed_fractions,
        )
        estimate = self.estimate(structure, params)
        plan_count = estimate.composition_count
        base_seed = int(params.seed) if params.use_seed else None
        cfg_id = int(stable_config_id(structure))
        selection_seed = None if base_seed is None else int(base_seed + cfg_id * 1_000_003)
        plan_indices = self._plan_indices(
            plan_count,
            min(plan_count, estimate.max_outputs),
            seed=selection_seed,
        )

        plans = [self._decode_plan(spaces, plan_index) for plan_index in plan_indices]
        theoretical_arrangements = [
            self._theoretical_arrangements(plan, spaces)
            for plan in plans
        ]
        arrangement_targets = self._arrangement_targets(
            theoretical_arrangements,
            estimate.arrangements_per_composition,
            estimate.max_outputs,
        )

        prepared = []
        for plan_index, plan, theoretical, target in zip(
            plan_indices,
            plans,
            theoretical_arrangements,
            arrangement_targets,
        ):
            if target <= 0:
                continue
            rng, derived_seed = self._rng(base_seed, cfg_id, plan_index)
            arrangements = self._arrangements(plan, spaces, target, theoretical, rng)
            composition_id = self._composition_id(plan, spaces)
            counts_meta = {
                space.label: {
                    element: int(count)
                    for element, count in zip(space.elements, counts_for_space)
                }
                for space, counts_for_space in zip(spaces, plan)
            }
            fractions_meta = {
                label: {
                    element: count / len(site_indices[label])
                    for element, count in counts.items()
                }
                for label, counts in counts_meta.items()
            }
            prepared.append(
                (
                    arrangements,
                    composition_id,
                    derived_seed,
                    counts_meta,
                    fractions_meta,
                )
            )

        outputs = []
        max_rounds = max((len(item[0]) for item in prepared), default=0)
        for arrangement_index in range(max_rounds):
            for arrangements, composition_id, derived_seed, counts_meta, fractions_meta in prepared:
                if arrangement_index >= len(arrangements):
                    continue
                assignments = arrangements[arrangement_index]
                atoms = structure.copy()
                if "sublattice" not in atoms.arrays:
                    atoms.new_array("sublattice", np.full(len(atoms), "all", dtype="U8"))
                symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
                for space, assigned in zip(spaces, assignments):
                    symbols[site_indices[space.label]] = np.asarray(assigned, dtype=object)
                atoms.set_chemical_symbols(symbols.tolist())

                arrangement_id = self._arrangement_id(assignments, spaces)
                metadata = {
                    "composition_id": composition_id,
                    "arrangement_id": arrangement_id,
                    "arrangement_index": int(arrangement_index),
                    "counts": counts_meta,
                    "fractions": fractions_meta,
                    "realization": {space.label: space.realization for space in spaces},
                    "requested": {
                        space.label: {
                            element: values[0] if len(values) == 1 else list(values)
                            for element, values in space.requested
                        }
                        for space in spaces
                    },
                    "seed": derived_seed,
                }
                atoms.info["finite_cell_alloy"] = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                append_config_tag(atoms, f"FiniteAlloy(comp={composition_id},arr={arrangement_id})")
                outputs.append(atoms)
                if len(outputs) >= estimate.max_outputs:
                    return outputs
        return outputs

    @staticmethod
    def _arrangement_targets(
        theoretical_arrangements: list[int],
        requested_per_composition: int,
        max_outputs: int,
    ) -> list[int]:
        """Allocate the output budget by arrangement round across compositions."""
        targets = [0] * len(theoretical_arrangements)
        remaining = int(max_outputs)
        for _ in range(int(requested_per_composition)):
            progressed = False
            for index, theoretical in enumerate(theoretical_arrangements):
                if remaining <= 0:
                    return targets
                if targets[index] >= int(theoretical):
                    continue
                targets[index] += 1
                remaining -= 1
                progressed = True
            if not progressed:
                break
        return targets

    @staticmethod
    def _site_indices(structure) -> dict[str, np.ndarray]:
        if len(structure) <= 0:
            raise ValueError("Finite-Cell Alloy Occupancy: input structure has no sites.")
        if "sublattice" not in structure.arrays:
            return {"all": np.arange(len(structure), dtype=int)}
        raw = np.asarray(structure.arrays["sublattice"], dtype=str)
        if raw.shape != (len(structure),):
            raise ValueError("Finite-Cell Alloy Occupancy: atoms.arrays['sublattice'] must be one label per atom.")
        labels = list(dict.fromkeys(str(value).strip() for value in raw))
        if any(not label for label in labels):
            raise ValueError("Finite-Cell Alloy Occupancy: sublattice labels must be non-empty.")
        return {label: np.nonzero(raw == label)[0].astype(int) for label in labels}

    @staticmethod
    def _plan_indices(total: int, n_pick: int, *, seed: int | None) -> list[int]:
        total = int(total)
        n_pick = min(max(int(n_pick), 0), total)
        if n_pick <= 0:
            return []
        if n_pick == total:
            return list(range(total))
        rng = random.Random(seed)
        start = rng.randrange(total)
        stride_hint = rng.randrange(1, total)
        stride = CompositionSweepOperation._coprime_stride(total, stride_hint)
        return [int((start + index * stride) % total) for index in range(n_pick)]

    @classmethod
    def _build_spaces(
        cls,
        site_indices: dict[str, np.ndarray],
        rules_text: str,
        *,
        require_normalized_fixed_fractions: bool = False,
    ) -> tuple[_CountSpace, ...]:
        try:
            rules = json.loads(str(rules_text or ""))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Finite-Cell Alloy Occupancy: invalid site_rules JSON: {exc.msg}.") from exc
        if not isinstance(rules, dict) or not rules:
            raise ValueError("Finite-Cell Alloy Occupancy: site_rules must be a non-empty JSON object.")
        expected = set(site_indices)
        provided = {str(label) for label in rules}
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing rules for {', '.join(missing)}")
            if extra:
                details.append(f"unknown site sets {', '.join(extra)}")
            raise ValueError("Finite-Cell Alloy Occupancy: " + "; ".join(details) + ".")
        return tuple(
            cls._space_from_rule(
                label,
                len(site_indices[label]),
                rules[label],
                require_normalized_fixed_fractions=require_normalized_fixed_fractions,
            )
            for label in site_indices
        )

    @classmethod
    def _space_from_rule(
        cls,
        label: str,
        n_sites: int,
        raw_rule: Any,
        *,
        require_normalized_fixed_fractions: bool = False,
    ) -> _CountSpace:
        if not isinstance(raw_rule, dict):
            raise ValueError(f"Finite-Cell Alloy Occupancy: rule for {label!r} must be an object.")
        raw_elements = raw_rule.get("elements", [])
        if isinstance(raw_elements, str):
            elements = tuple(parse_element_list(raw_elements))
        elif isinstance(raw_elements, list):
            elements = tuple(parse_element_list(",".join(str(value) for value in raw_elements)))
        else:
            raise ValueError(f"Finite-Cell Alloy Occupancy: elements for {label!r} must be a list or string.")
        if not elements:
            raise ValueError(f"Finite-Cell Alloy Occupancy: site set {label!r} has no allowed elements.")
        if "X" in elements:
            raise ValueError(
                f"Finite-Cell Alloy Occupancy: site set {label!r} still contains placeholder element X; "
                "replace it with real element symbols."
            )

        mode = str(raw_rule.get("mode", "fixed_fraction")).strip().lower()
        if mode == "fixed_fraction":
            composition = cls._mapping_for_elements(raw_rule.get("composition"), elements, label, "composition")
            values = np.asarray([composition[element] for element in elements], dtype=float)
            if np.any(~np.isfinite(values)) or np.any(values < 0.0) or float(values.sum()) <= 0.0:
                raise ValueError(f"Finite-Cell Alloy Occupancy: fixed composition for {label!r} is invalid.")
            if require_normalized_fixed_fractions and not np.isclose(
                float(values.sum()),
                1.0,
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Finite-Cell Alloy Occupancy: fixed fractions for {label!r} must sum to 1."
                )
            fractions = values / float(values.sum())
            counts = fractions_to_counts_exact(fractions, n_sites)
            exact = bool(np.allclose(fractions * n_sites, counts, atol=1e-10, rtol=0.0))
            requested = tuple(
                (element, (float(value),))
                for element, value in zip(elements, fractions)
            )
            return _CountSpace(
                label=label,
                elements=elements,
                lower=tuple(int(value) for value in counts),
                upper=tuple(int(value) for value in counts),
                n_sites=n_sites,
                realization="exact" if exact else "nearest_integer",
                requested=requested,
            )

        if mode == "fraction_range":
            ranges = cls._range_mapping(raw_rule.get("fractions"), elements, label, fraction=True)
            lower = tuple(int(math.ceil(ranges[element][0] * n_sites - 1e-12)) for element in elements)
            upper = tuple(int(math.floor(ranges[element][1] * n_sites + 1e-12)) for element in elements)
            requested = tuple((element, tuple(float(value) for value in ranges[element])) for element in elements)
            realization = "fraction_range"
        elif mode == "count_range":
            ranges = cls._range_mapping(raw_rule.get("counts"), elements, label, fraction=False)
            lower = tuple(int(ranges[element][0]) for element in elements)
            upper = tuple(int(ranges[element][1]) for element in elements)
            requested = tuple((element, tuple(float(value) for value in ranges[element])) for element in elements)
            realization = "count_range"
        else:
            raise ValueError(
                f"Finite-Cell Alloy Occupancy: unsupported mode {mode!r} for {label!r}; "
                "use fixed_fraction, fraction_range, or count_range."
            )

        space = _CountSpace(
            label=label,
            elements=elements,
            lower=lower,
            upper=upper,
            n_sites=n_sites,
            realization=realization,
            requested=requested,
        )
        if space.capacity <= 0:
            raise ValueError(
                f"Finite-Cell Alloy Occupancy: constraints for site set {label!r} "
                f"have no integer count solution for {n_sites} sites."
            )
        return space

    @staticmethod
    def _mapping_for_elements(raw: Any, elements: tuple[str, ...], label: str, field_name: str) -> dict[str, float]:
        if not isinstance(raw, dict):
            raise ValueError(f"Finite-Cell Alloy Occupancy: {field_name} for {label!r} must be an object.")
        extra = sorted(set(str(key) for key in raw) - set(elements))
        missing = sorted(set(elements) - set(str(key) for key in raw))
        if extra or missing:
            raise ValueError(
                f"Finite-Cell Alloy Occupancy: {field_name} keys for {label!r} "
                "must exactly match its allowed elements."
            )
        return {element: float(raw[element]) for element in elements}

    @classmethod
    def _range_mapping(
        cls,
        raw: Any,
        elements: tuple[str, ...],
        label: str,
        *,
        fraction: bool,
    ) -> dict[str, tuple[float, float]]:
        field_name = "fractions" if fraction else "counts"
        if not isinstance(raw, dict):
            raise ValueError(f"Finite-Cell Alloy Occupancy: {field_name} for {label!r} must be an object.")
        extra = sorted(set(str(key) for key in raw) - set(elements))
        missing = sorted(set(elements) - set(str(key) for key in raw))
        if extra or missing:
            raise ValueError(
                f"Finite-Cell Alloy Occupancy: {field_name} keys for {label!r} "
                "must exactly match its allowed elements."
            )
        ranges: dict[str, tuple[float, float]] = {}
        for element in elements:
            value = raw[element]
            if isinstance(value, (list, tuple)):
                if len(value) != 2:
                    raise ValueError(f"Finite-Cell Alloy Occupancy: range for {label}.{element} needs two values.")
                low, high = float(value[0]), float(value[1])
            else:
                low = high = float(value)
            if not np.all(np.isfinite([low, high])) or low < 0.0 or high < low:
                raise ValueError(f"Finite-Cell Alloy Occupancy: invalid range for {label}.{element}.")
            if fraction and high > 1.0 + 1e-12:
                raise ValueError(f"Finite-Cell Alloy Occupancy: fraction for {label}.{element} exceeds 1.")
            if not fraction and (not float(low).is_integer() or not float(high).is_integer()):
                raise ValueError(f"Finite-Cell Alloy Occupancy: count bounds for {label}.{element} must be integers.")
            ranges[element] = (low, high)
        return ranges

    @staticmethod
    def _decode_plan(spaces: tuple[_CountSpace, ...], index: int) -> tuple[tuple[int, ...], ...]:
        ranks = []
        residual = int(index)
        for space in reversed(spaces):
            ranks.append(residual % space.capacity)
            residual //= space.capacity
        return tuple(space.unrank(rank) for space, rank in zip(spaces, reversed(ranks)))

    @staticmethod
    def _theoretical_arrangements(plan: tuple[tuple[int, ...], ...], spaces: tuple[_CountSpace, ...]) -> int:
        total = 1
        for counts, space in zip(plan, spaces):
            ways = math.factorial(space.n_sites)
            for count in counts:
                ways //= math.factorial(int(count))
            total *= ways
        return int(total)

    @staticmethod
    def _rng(base_seed: int | None, cfg_id: int, plan_index: int) -> tuple[np.random.Generator, int | None]:
        if base_seed is None:
            return np.random.default_rng(), None
        seed_sequence = np.random.SeedSequence([int(base_seed), int(cfg_id) & 0xFFFFFFFF, int(plan_index)])
        derived_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        return np.random.default_rng(derived_seed), derived_seed

    @classmethod
    def _arrangements(
        cls,
        plan: tuple[tuple[int, ...], ...],
        spaces: tuple[_CountSpace, ...],
        target: int,
        theoretical: int,
        rng: np.random.Generator,
    ) -> list[tuple[tuple[str, ...], ...]]:
        if target <= 0:
            return []
        if theoretical <= cls._ENUMERATION_LIMIT:
            all_arrangements = [()]
            for counts, space in zip(plan, spaces):
                site_assignments = list(cls._iter_multiset_assignments(space.elements, counts))
                all_arrangements = [
                    prefix + (assignment,)
                    for prefix in all_arrangements
                    for assignment in site_assignments
                ]
            selected = rng.choice(len(all_arrangements), size=target, replace=False)
            return [all_arrangements[int(index)] for index in np.atleast_1d(selected)]

        seen: set[tuple[tuple[str, ...], ...]] = set()
        outputs = []
        max_attempts = max(1000, target * 100)
        attempts = 0
        while len(outputs) < target and attempts < max_attempts:
            assignment_group = []
            for counts, space in zip(plan, spaces):
                pool = np.concatenate(
                    [np.repeat(element, int(count)) for element, count in zip(space.elements, counts) if count > 0]
                )
                rng.shuffle(pool)
                assignment_group.append(tuple(str(value) for value in pool.tolist()))
            signature = tuple(assignment_group)
            if signature not in seen:
                seen.add(signature)
                outputs.append(signature)
            attempts += 1
        if len(outputs) != target:
            raise RuntimeError(
                f"Finite-Cell Alloy Occupancy: generated {len(outputs)}/{target} unique arrangements "
                f"after {attempts} deterministic attempts."
            )
        return outputs

    @staticmethod
    def _iter_multiset_assignments(
        elements: tuple[str, ...],
        counts: tuple[int, ...],
    ):
        n_sites = int(sum(counts))
        assignment = [""] * n_sites

        def visit(element_index: int, available: tuple[int, ...]):
            if element_index == len(elements) - 1:
                for position in available:
                    assignment[position] = elements[element_index]
                yield tuple(assignment)
                return
            count = int(counts[element_index])
            for selected in combinations(available, count):
                selected_set = set(selected)
                for position in selected:
                    assignment[position] = elements[element_index]
                remaining = tuple(position for position in available if position not in selected_set)
                yield from visit(element_index + 1, remaining)

        yield from visit(0, tuple(range(n_sites)))

    @staticmethod
    def _composition_id(plan: tuple[tuple[int, ...], ...], spaces: tuple[_CountSpace, ...]) -> str:
        payload = "|".join(
            f"{space.label}:" + ",".join(f"{element}={count}" for element, count in zip(space.elements, counts))
            for space, counts in zip(spaces, plan)
        )
        return "c" + hashlib.blake2b(payload.encode("utf-8"), digest_size=5).hexdigest()

    @staticmethod
    def _arrangement_id(
        assignments: tuple[tuple[str, ...], ...],
        spaces: tuple[_CountSpace, ...],
    ) -> str:
        payload = "|".join(
            f"{space.label}:" + ",".join(assignment)
            for space, assignment in zip(spaces, assignments)
        )
        return "a" + hashlib.blake2b(payload.encode("utf-8"), digest_size=6).hexdigest()


@dataclass(frozen=True)
class CompositionGradientParams:
    elements: str = "Ni,Co"
    start_composition: str = "Ni:1,Co:0"
    end_composition: str = "Ni:0,Co:1"
    axis: str = "a"
    bins: int = 8
    target_elements: str = ""
    samples: int = 1
    use_seed: bool = False
    seed: int = 0


class CompositionGradientOperation(StructureOperation):
    """Assign site species from a composition gradient along one lattice coordinate."""

    AXIS_INDEX = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}

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
        axis_key = str(params.axis).strip().lower()
        if axis_key not in self.AXIS_INDEX:
            raise ValueError("Composition Gradient axis must be one of a, b, or c.")
        axis_idx = self.AXIS_INDEX[axis_key]
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
        if values.size != len(elements) or np.any(~np.isfinite(values)) or np.any(values < 0.0) or float(values.sum()) <= 0.0:
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
        if int(getattr(structure.cell, "rank", 0)) < 3:
            raise ValueError(
                "Composition Gradient requires a non-singular 3D cell "
                "to use lattice directions a, b, or c."
            )
        return scaled_positions(structure, wrap=True)[:, axis_idx]

    @staticmethod
    def _axis_name(axis_idx: int) -> str:
        return ("a", "b", "c")[int(axis_idx)]

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
            raise ValueError(
                "RandomOccupancy requires a Comp(...) tag or a non-empty manual composition."
            )
        invalid_elements = [element for element in comp if element not in atomic_numbers]
        if invalid_elements:
            raise ValueError(
                "RandomOccupancy has unknown element symbol(s): "
                + ", ".join(invalid_elements)
                + "."
            )

        indices = self._eligible_indices(structure, params.group_filter)
        mode = str(params.mode).strip()
        if mode not in {"Exact", "Random"}:
            raise ValueError("RandomOccupancy: mode must be Exact or Random.")
        samples = int(params.samples)
        if samples <= 0:
            raise ValueError("RandomOccupancy: samples must be >= 1.")
        seed = int(params.seed)
        if params.use_seed and seed < 0:
            raise ValueError("RandomOccupancy: seed must be >= 0.")
        base_seed = seed if params.use_seed else None
        cfg_id = stable_config_id(structure)

        out = []
        for sample_idx in range(samples):
            if base_seed is None:
                rng = np.random.default_rng()
                seed_note = ""
            else:
                derived_seed = int(base_seed + cfg_id * 1000003 + sample_idx)
                rng = np.random.default_rng(derived_seed)
                seed_note = f",s={derived_seed}"

            new_atoms = assign_random_occupancy(structure, comp, indices=indices, mode=mode, rng=rng)
            mode_tag = "E" if mode == "Exact" else "R"
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
        if not groups_text:
            return None
        allowed = {group.strip() for group in groups_text.split(",") if group.strip()}
        if not allowed:
            return None
        if "group" not in structure.arrays:
            raise ValueError(
                "RandomOccupancy group_filter requires atoms.arrays['group'] on the input structure."
            )
        groups = structure.arrays["group"]
        indices = np.array(
            [i for i, group in enumerate(groups) if str(group) in allowed],
            dtype=int,
        )
        if len(indices) == 0:
            raise ValueError(
                "RandomOccupancy group_filter matched no atoms: "
                + ",".join(sorted(allowed))
                + "."
            )
        return indices


def normalize_condition_expr(expr: str) -> str:
    """Convert card condition syntax to a Python boolean expression."""
    expr = expr.strip()
    if not expr or expr.lower() == "all":
        return "True"
    expr = re.sub(r"\bAND\b", "and", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bOR\b", "or", expr, flags=re.IGNORECASE)
    expr = re.sub(r"\bNOT\b", "not", expr, flags=re.IGNORECASE)
    return re.sub(r"(?<![<>=!])=(?!=)", "==", expr)


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
    try:
        tree = ast.parse(expr_py, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid condition expression {expr!r}: {exc.msg}.") from exc
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
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid replacement JSON: {exc.msg}") from exc
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
    condition: str = "all"
    seed: int = 0
    mode: int = 0


class ConditionalReplaceOperation(StructureOperation):
    """Replace atoms that match target species and coordinate condition."""

    def run_structure(self, structure, params: ConditionalReplaceParams) -> list:
        target = params.target.strip()
        if not target:
            return [structure.copy()]

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
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("Replacement probabilities must be finite and non-negative.")
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
