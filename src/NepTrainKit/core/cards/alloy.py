"""UI-independent alloy and composition Make Dataset operations."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from collections import Counter
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
    fractions_to_counts_exact,
    parse_composition,
    parse_element_list,
    simplex_grid_points,
    simplex_sobol_points,
)
from NepTrainKit.core.config_type import append_config_tag, stable_config_id

from .errors import CardOperationError
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


def _normalized_dopant_atom_ratios(
    dopant_list,
    ratios,
    ratio_type: str,
) -> tuple[list[str], np.ndarray]:
    """Validate dopant weights and return normalized atomic fractions."""
    dopant_list = [str(element) for element in dopant_list]
    ratios = np.array(list(ratios), dtype=float)
    if not dopant_list:
        raise ValueError("At least one dopant is required.")
    if ratios.size != len(dopant_list) or ratios.size == 0:
        raise ValueError("Dopant ratios must match dopant elements.")
    if np.any(~np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError("Dopant ratios must be finite and non-negative.")
    invalid_elements = [
        element for element in dopant_list if element not in atomic_numbers
    ]
    if invalid_elements:
        raise ValueError(
            "Unknown dopant element symbol(s): " + ", ".join(invalid_elements) + "."
        )
    if ratio_type not in {"atom", "mass"}:
        raise ValueError("Dopant ratio_type must be 'atom' or 'mass'.")

    if ratio_type == "mass":
        masses = np.array(
            [atomic_masses[atomic_numbers[element]] for element in dopant_list]
        )
        atom_ratios = ratios / masses
    else:
        atom_ratios = ratios
    total = float(atom_ratios.sum())
    if total <= 0.0:
        raise ValueError("At least one dopant ratio must be positive.")
    return dopant_list, atom_ratios / total


def sample_dopants(
    dopant_list,
    ratios,
    n_items,
    exact: bool = False,
    rng: np.random.Generator | None = None,
    ratio_type: str = "atom",
) -> list:
    """Sample dopant elements from atom or mass ratios."""
    n_items = int(n_items)
    if n_items < 0:
        raise ValueError("Dopant item count must be non-negative.")
    dopant_list, atom_ratios = _normalized_dopant_atom_ratios(
        dopant_list,
        ratios,
        ratio_type,
    )
    if rng is None:
        rng = np.random.default_rng()

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

    @staticmethod
    def _validated_rules(
        rules: list[dict[str, Any]],
    ) -> list[tuple[str, dict[str, Any], str, dict[str, float]]]:
        validated = []
        for rule_index, rule in enumerate(rules, start=1):
            label = f"RandomDoping rule {rule_index}"
            if not isinstance(rule, dict):
                raise ValueError(f"{label} must be a mapping.")
            target = str(rule.get("target", "") or "").strip()
            dopants = rule.get("dopants", {})
            if not target:
                raise ValueError(f"{label} requires a target element.")
            if not isinstance(dopants, dict):
                raise ValueError(
                    f"{label} dopants must be an element->ratio mapping."
                )
            if not dopants:
                raise ValueError(f"{label} requires at least one dopant element.")
            if target not in atomic_numbers:
                raise ValueError(
                    f"{label} has an unknown target element '{target}'."
                )
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
            ratio_type = str(rule.get("ratio_type", "atom"))
            try:
                _normalized_dopant_atom_ratios(
                    dopants.keys(),
                    dopants.values(),
                    ratio_type,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}: {exc}") from exc
            validated.append((label, rule, target, dopants))
        return validated

    @staticmethod
    def _maximum_prior_consumption(
        current_mask: np.ndarray,
        prior_rules: list[tuple[np.ndarray, int]],
    ) -> int:
        """Return the largest number of current sites earlier rules can consume."""
        active_rules = [
            (np.asarray(mask, dtype=bool), int(capacity))
            for mask, capacity in prior_rules
            if int(capacity) > 0 and np.any(current_mask & mask)
        ]
        if not active_rules:
            return 0

        signature_counts: dict[tuple[int, ...], int] = {}
        for atom_index in np.nonzero(current_mask)[0]:
            signature = tuple(
                rule_index
                for rule_index, (mask, _capacity) in enumerate(active_rules)
                if mask[atom_index]
            )
            if signature:
                signature_counts[signature] = signature_counts.get(signature, 0) + 1

        rule_count = len(active_rules)
        signatures = list(signature_counts)
        source = 0
        rule_offset = 1
        signature_offset = rule_offset + rule_count
        sink = signature_offset + len(signatures)
        graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

        def add_edge(left: int, right: int, capacity: int) -> None:
            graph[left].append([right, capacity, len(graph[right])])
            graph[right].append([left, 0, len(graph[left]) - 1])

        for rule_index, (_mask, capacity) in enumerate(active_rules):
            add_edge(source, rule_offset + rule_index, capacity)
        for signature_index, signature in enumerate(signatures):
            signature_node = signature_offset + signature_index
            site_count = signature_counts[signature]
            add_edge(signature_node, sink, site_count)
            for rule_index in signature:
                add_edge(rule_offset + rule_index, signature_node, site_count)

        total_flow = 0
        while True:
            levels = [-1] * len(graph)
            levels[source] = 0
            queue = [source]
            for node in queue:
                for target, capacity, _reverse in graph[node]:
                    if capacity > 0 and levels[target] < 0:
                        levels[target] = levels[node] + 1
                        queue.append(target)
            if levels[sink] < 0:
                return total_flow

            next_edges = [0] * len(graph)

            def send(node: int, flow: int) -> int:
                if node == sink:
                    return flow
                while next_edges[node] < len(graph[node]):
                    edge = graph[node][next_edges[node]]
                    target, capacity, reverse = edge
                    if capacity > 0 and levels[target] == levels[node] + 1:
                        pushed = send(target, min(flow, capacity))
                        if pushed:
                            edge[1] -= pushed
                            graph[target][reverse][1] += pushed
                            return pushed
                    next_edges[node] += 1
                return 0

            while True:
                pushed = send(source, len(current_mask))
                if not pushed:
                    break
                total_flow += pushed

    def _prepared_rules(
        self,
        structure,
        validated_rules,
    ) -> list[tuple[str, dict[str, Any], str, dict[str, float], list[str]]]:
        """Resolve rule scopes and reject seed-dependent capacity conflicts."""
        symbols = np.asarray(structure.get_chemical_symbols(), dtype=object)
        group_values = np.asarray(
            structure.arrays.get("group", np.empty(0, dtype=object)),
            dtype=object,
        )
        prepared = []
        prior_by_target: dict[str, list[tuple[np.ndarray, int]]] = {}
        for label, rule, target, dopants in validated_rules:
            groups = rule.get("group")
            requested_groups = [
                str(value).strip()
                for value in _as_list(groups)
                if str(value).strip()
            ]
            candidate_mask = symbols == target
            if requested_groups:
                if "group" not in structure.arrays:
                    raise ValueError(
                        f"{label} requests group labels, but the input structure "
                        "has no group array."
                    )
                candidate_mask &= np.isin(group_values, requested_groups)
            candidate_indices = np.nonzero(candidate_mask)[0]
            if len(candidate_indices) == 0:
                scope = (
                    f" in group {','.join(requested_groups)}"
                    if requested_groups
                    else ""
                )
                raise ValueError(f"{label} matched no '{target}' atoms{scope}.")

            initial_max = self._doping_count(
                structure,
                candidate_indices,
                target,
                dopants,
                rule,
                rng=None,
            )
            if initial_max > len(candidate_indices):
                raise ValueError(
                    f"{label} can request up to {initial_max} replacements, "
                    f"but only {len(candidate_indices)} eligible atoms are available."
                )

            prior_rules = prior_by_target.setdefault(target, [])
            prior_consumption = self._maximum_prior_consumption(
                candidate_mask,
                prior_rules,
            )
            guaranteed_count = len(candidate_indices) - prior_consumption
            guaranteed_indices = np.arange(guaranteed_count, dtype=int)
            guaranteed_max = self._doping_count(
                structure,
                guaranteed_indices,
                target,
                dopants,
                rule,
                rng=None,
            )
            if guaranteed_max > guaranteed_count:
                raise ValueError(
                    f"{label} can request up to {guaranteed_max} replacements, "
                    f"but earlier overlapping rules can leave only "
                    f"{guaranteed_count} eligible atoms."
                )

            prior_rules.append((candidate_mask, initial_max))
            prepared.append((label, rule, target, dopants, requested_groups))
        return prepared

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
        validated_rules = self._validated_rules(params.rules)
        prepared_rules = self._prepared_rules(structure, validated_rules)
        rng: np.random.Generator | None = None

        for _ in range(max_structures):
            new_structure = structure.copy()
            symbols = np.asarray(new_structure.get_chemical_symbols(), dtype=object)
            total_doping = 0
            for label, rule, target, dopants, requested_groups in prepared_rules:
                if requested_groups:
                    group_values = np.asarray(new_structure.arrays["group"], dtype=object)
                    candidate_indices = np.nonzero(
                        (symbols == target)
                        & np.isin(group_values, requested_groups)
                    )[0]
                else:
                    candidate_indices = np.nonzero(symbols == target)[0]

                max_doping_num = self._doping_count(
                    new_structure,
                    candidate_indices,
                    target,
                    dopants,
                    rule,
                    rng=None,
                )
                if max_doping_num > len(candidate_indices):
                    raise ValueError(
                        f"{label} can request up to {max_doping_num} replacements, "
                        f"but only {len(candidate_indices)} eligible atoms are available."
                    )
                if rng is None:
                    rng = np.random.default_rng(base_seed)
                doping_num = self._doping_count(
                    new_structure,
                    candidate_indices,
                    target,
                    dopants,
                    rule,
                    rng,
                )
                if doping_num < 0:
                    raise ValueError(f"{label} replacement count must be >= 0.")
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

    def _doping_count(
        self,
        structure,
        candidate_indices,
        target,
        dopants,
        rule,
        rng: np.random.Generator | None,
    ) -> int:
        use_mode = rule.get("use", "atomic_percent")

        if use_mode == "atomic_percent":
            percent_min, percent_max = _range_pair(
                rule.get("percent", [0.0, 100.0]),
                label="percent",
            )
            if percent_min < 0.0 or percent_max > 100.0:
                raise ValueError("percent must be within [0, 100].")
            value = (
                float(percent_max)
                if rng is None
                else rng.uniform(float(percent_min), float(percent_max))
            ) / 100.0
            return int(len(candidate_indices) * value)

        if use_mode == "mass_percent":
            percent_min, percent_max = _range_pair(
                rule.get("percent", [0.0, 100.0]),
                label="percent",
            )
            if percent_min < 0.0 or percent_max > 100.0:
                raise ValueError("percent must be within [0, 100].")
            target_mass_percent = (
                float(percent_max)
                if rng is None
                else rng.uniform(float(percent_min), float(percent_max))
            ) / 100.0

            target_mass = atomic_masses[atomic_numbers[target]]
            total_target_mass = len(candidate_indices) * target_mass
            dopant_elements, atom_ratios = _normalized_dopant_atom_ratios(
                dopants.keys(),
                dopants.values(),
                str(rule.get("ratio_type", "atom")),
            )
            dopant_masses = np.array(
                [
                    atomic_masses[atomic_numbers[element]]
                    for element in dopant_elements
                ],
                dtype=float,
            )
            avg_dopant_mass = float(np.dot(atom_ratios, dopant_masses))

            doped_mass = total_target_mass * target_mass_percent
            return int(doped_mass / avg_dopant_mass)

        if use_mode == "count":
            count_min_f, count_max_f = _range_pair(
                rule.get("count", [1, 1]),
                label="count",
            )
            if (
                not float(count_min_f).is_integer()
                or not float(count_max_f).is_integer()
            ):
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
            if rng is None:
                return count_max
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

    MAX_OUTPUTS_PER_INPUT = 10_000
    MAX_GRID_TEMPLATES = 100_000

    def sampling_summary(self, params: CompositionSweepParams) -> dict[str, object]:
        """Validate settings and build the exact unique target-composition plan."""
        elements = parse_element_list(params.elements)
        if len(elements) < 2:
            raise CardOperationError(
                "composition_sweep.too_few_elements",
                "Composition Space Sampling requires at least two valid elements.",
            )
        invalid_elements = [element for element in elements if element not in atomic_numbers]
        if invalid_elements:
            raise CardOperationError(
                "composition_sweep.invalid_elements",
                "Composition Space Sampling has unknown element symbol(s): {elements}.",
                elements=", ".join(invalid_elements),
            )

        requested_orders = self._target_orders(params.order)
        orders = [order for order in requested_orders if len(elements) >= order]
        skipped_orders = [order for order in requested_orders if len(elements) < order]
        if not orders:
            raise CardOperationError(
                "composition_sweep.no_feasible_orders",
                "None of the selected component counts is feasible for {count} elements.",
                count=len(elements),
            )
        method = str(params.method or "").strip()
        if method not in {"Grid", "Sobol"}:
            raise ValueError("Composition Space Sampling method must be Grid or Sobol.")
        max_outputs = int(params.max_outputs)
        if max_outputs < 1:
            raise CardOperationError(
                "composition_sweep.invalid_budget",
                "Maximum target compositions per input must be at least 1.",
            )
        if max_outputs > self.MAX_OUTPUTS_PER_INPUT:
            raise CardOperationError(
                "composition_sweep.budget_too_large",
                "Maximum target compositions per input cannot exceed {maximum}.",
                maximum=self.MAX_OUTPUTS_PER_INPUT,
            )
        min_fraction = float(params.min_fraction)
        if not np.isfinite(min_fraction) or min_fraction < 0.0 or min_fraction > 1.0:
            raise ValueError("Minimum element fraction must be between 0 and 1.")
        if params.use_seed and int(params.seed) < 0:
            raise ValueError("Composition Space Sampling seed must be non-negative.")

        seed = int(params.seed) if params.use_seed else None
        combo_rng = np.random.default_rng(seed) if seed is not None else None

        order_data: list[dict[str, object]] = []
        capacities: dict[int, int] = {}
        for order in orders:
            try:
                if method == "Grid":
                    self._validate_grid_size(order, params)
                points = self._simplex_points(
                    order,
                    params,
                    point_limit=max_outputs if method == "Sobol" else None,
                )
            except NotImplementedError as exc:
                raise CardOperationError(
                    "composition_sweep.grid_step",
                    "Grid sampling for four or five components requires a step of 1/n, such as 0.1 or 0.05.",
                ) from exc
            if not points:
                continue
            combos = list(combinations(elements, order))
            if combo_rng is not None and combos:
                combo_rng.shuffle(combos)
            available_total = len(combos) * len(points)
            if available_total <= 0:
                continue
            nominal_points = int(params.n_points) if method == "Sobol" else len(points)
            nominal_total = len(combos) * nominal_points
            capacities[order] = int(nominal_total)
            order_data.append(
                {
                    "order": order,
                    "points": points,
                    "combos": combos,
                    "capacity": int(nominal_total),
                    "available_capacity": int(available_total),
                }
            )

        if not order_data:
            raise CardOperationError(
                "composition_sweep.no_targets",
                "The current composition constraints produce no target compositions.",
            )

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

        iterators = {
            int(item["order"]): self._target_iterator(item, seed=seed)
            for item in order_data
        }
        exhausted = {order: False for order in active_orders}
        seen: set[tuple[tuple[str, float], ...]] = set()
        targets: list[tuple[int, tuple[tuple[str, float], ...]]] = []
        emitted_by_order = {order: 0 for order in active_orders}

        def take_unique(order: int, limit: int) -> None:
            iterator = iterators[order]
            while emitted_by_order[order] < limit and len(targets) < max_outputs:
                try:
                    key = next(iterator)
                except StopIteration:
                    exhausted[order] = True
                    return
                if key in seen:
                    continue
                seen.add(key)
                targets.append((order, key))
                emitted_by_order[order] += 1

        for order in active_orders:
            take_unique(order, int(emit.get(order, 0)))

        if mode != "equal_legacy":
            while len(targets) < max_outputs:
                progressed = False
                for order in active_orders:
                    if exhausted[order] or len(targets) >= max_outputs:
                        continue
                    before = len(targets)
                    take_unique(order, emitted_by_order[order] + 1)
                    progressed = progressed or len(targets) > before
                if not progressed:
                    break

        if not targets:
            raise CardOperationError(
                "composition_sweep.no_unique_targets",
                "The current settings produce no unique target compositions.",
            )
        return {
            "elements": tuple(elements),
            "method": method,
            "requested_orders": tuple(requested_orders),
            "active_orders": tuple(active_orders),
            "skipped_orders": tuple(skipped_orders),
            "nominal_capacities": dict(capacities),
            "emitted_by_order": dict(emitted_by_order),
            "targets": tuple(targets),
            "outputs_per_input": len(targets),
            "max_outputs": max_outputs,
            "budget_mode": mode,
        }

    def run_structure(self, structure, params: CompositionSweepParams) -> list:
        summary = self.sampling_summary(params)
        out = []
        for _order, composition in summary["targets"]:
            new_structure = structure.copy()
            tag = ",".join(
                f"{element}={fraction:.12g}" for element, fraction in composition
            )
            self._replace_composition_tag(new_structure, f"Comp({tag})")
            out.append(new_structure)
        return out

    @staticmethod
    def _canonical_target(
        elements: tuple[str, ...],
        fractions: tuple[float, ...],
    ) -> tuple[tuple[str, float], ...]:
        positive = [
            (element, float(fraction))
            for element, fraction in zip(elements, fractions)
            if float(fraction) > 1.0e-12
        ]
        total = float(sum(fraction for _element, fraction in positive))
        return tuple(
            sorted(
                (element, round(fraction / total, 12))
                for element, fraction in positive
            )
        )

    def _target_iterator(self, item: dict[str, object], *, seed: int | None):
        order = int(item["order"])
        points = item["points"]
        combos = item["combos"]
        total = int(item["available_capacity"])
        slot_seed = None if seed is None else int(seed + order * 104729)
        for slot in self._spread_slot_order(total, seed=slot_seed):
            combo_idx = int(slot % len(combos))
            point_idx = int(slot // len(combos))
            yield self._canonical_target(combos[combo_idx], points[point_idx])

    @classmethod
    def _spread_slot_order(cls, total: int, *, seed: int | None):
        total = int(total)
        if total <= 0:
            return
        if seed is None:
            start = 0
            stride_hint = int(total * 0.6180339887498949)
        else:
            rng = np.random.default_rng(int(seed))
            start = int(rng.integers(0, total))
            stride_hint = int(rng.integers(1, total)) if total > 1 else 1
        stride = cls._coprime_stride(total, stride_hint)
        for index in range(total):
            yield int((start + index * stride) % total)

    @staticmethod
    def _replace_composition_tag(structure, tag: str) -> None:
        current = str(structure.info.get("Config_type", "") or "")
        tokens = [token.strip() for token in re.split(r"[|\s]+", current) if token.strip()]
        structure.info["Config_type"] = "|".join(
            token
            for token in tokens
            if not (token.startswith("Comp(") and token.endswith(")"))
        )
        append_config_tag(structure, tag)

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
        if not orders:
            raise ValueError(
                "Composition Space Sampling component counts must select 2, 3, 4, or 5."
            )
        return orders

    def _simplex_points(
        self,
        order: int,
        params: CompositionSweepParams,
        *,
        point_limit: int | None = None,
    ) -> list[tuple[float, ...]]:
        seed = int(params.seed) if params.use_seed else None
        if params.method == "Sobol":
            n_points = int(params.n_points)
            if point_limit is not None:
                n_points = min(n_points, int(point_limit))
            return simplex_sobol_points(
                order,
                n_points,
                seed=seed,
                min_fraction=float(params.min_fraction),
            )
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

    def _validate_grid_size(
        self,
        order: int,
        params: CompositionSweepParams,
    ) -> None:
        """Reject grid templates that would block the synchronous exact preview."""
        step = float(params.step)
        if not np.isfinite(step) or step <= 0.0 or step > 1.0:
            return
        inverse = int(round(1.0 / step))
        near_rational = inverse > 0 and abs(step - 1.0 / inverse) <= 1.0e-9
        if near_rational:
            min_each = 0 if params.include_endpoints else 1
            if float(params.min_fraction) > 0.0:
                min_each = max(
                    min_each,
                    int(math.ceil(float(params.min_fraction) * inverse - 1.0e-12)),
                )
            remaining = inverse - min_each * order
            template_count = (
                0
                if remaining < 0
                else math.comb(remaining + order - 1, order - 1)
            )
        else:
            values = int(math.floor((1.0 + 1.0e-12) / step)) + 1
            template_count = values if order == 2 else values * values
        if template_count > self.MAX_GRID_TEMPLATES:
            raise CardOperationError(
                "composition_sweep.grid_too_dense",
                "The Grid settings require about {count} simplex points before budgeting. Increase the step or use Sobol; the safe limit is {maximum}.",
                count=template_count,
                maximum=self.MAX_GRID_TEMPLATES,
            )

    def _budget_mode(self, text: str) -> str:
        text = (text or "").strip().lower()
        if "legacy" in text:
            return "equal_legacy"
        if text in {"capacity-weighted", "favor larger composition spaces"}:
            return "weighted_reflow"
        if text in {"equal+reflow", "balance component counts"}:
            return "equal_reflow"
        raise ValueError(
            "Composition Space Sampling budget allocation must be Equal+Reflow, Capacity-weighted, or Equal (legacy)."
        )

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
    max_outputs: int = 200


@dataclass(frozen=True)
class OrderedAlloyPrototypePlan:
    """Exact base-cell preview for an ordered-alloy prototype request."""

    prototype: str
    a_values: tuple[float, ...]
    atoms_per_output: int
    cell_lengths: tuple[float, float, float]
    sublattice_counts: dict[str, int]
    sublattice_elements: dict[str, str]
    truncated: bool


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
        raise CardOperationError(
            "ordered-alloy-element-empty",
            "Enter one element symbol or the X placeholder for every visible sublattice.",
        )
    symbol = symbol[0].upper() + symbol[1:].lower()
    if symbol not in atomic_numbers:
        raise CardOperationError(
            "ordered-alloy-element-invalid",
            "Invalid element or placeholder {element}; use a chemical element symbol or X.",
            element=repr(text),
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

    required = tuple(sorted(dict.fromkeys(labels)))
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
        plan = self.plan(params)
        prototype = plan.prototype
        definition = _ORDERED_PROTOTYPES[prototype]
        occupants = plan.sublattice_elements
        max_outputs = int(params.max_outputs)

        outputs = []
        for a in plan.a_values[:max_outputs]:
            atoms = self._build_base(definition, occupants, a, float(params.covera))
            atoms.wrap()
            metadata = {
                "prototype": prototype,
                "a": float(a),
                "covera": self._effective_covera(definition, float(params.covera)),
                "sublattice_elements": occupants,
                "sublattice_counts": {
                    label: int(np.count_nonzero(np.asarray(atoms.arrays["sublattice"], dtype=str) == label))
                    for label in dict.fromkeys(definition.labels)
                },
            }
            atoms.info["ordered_alloy_prototype"] = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            append_config_tag(atoms, f"OrderedProto({prototype},a={a:.6g})")
            outputs.append(atoms)
            if len(outputs) >= max_outputs:
                break
        return outputs

    def plan(self, params: OrderedAlloyPrototypeParams) -> OrderedAlloyPrototypePlan:
        """Validate parameters and return the exact first base-cell preview."""
        prototype = _canonical_prototype_name(params.prototype)
        definition = _ORDERED_PROTOTYPES[prototype]
        occupants = _parse_sublattice_elements(params.sublattice_elements, definition.labels)
        max_outputs = int(params.max_outputs)
        if max_outputs <= 0:
            raise ValueError("Ordered Alloy Prototype: max_outputs must be >= 1.")
        a_values = tuple(_scan_lattice_values(params.a_range))
        base = self._build_base(definition, occupants, a_values[0], float(params.covera))
        labels = np.asarray(base.arrays["sublattice"], dtype=str)
        return OrderedAlloyPrototypePlan(
            prototype=prototype,
            a_values=a_values,
            atoms_per_output=len(base),
            cell_lengths=tuple(float(value) for value in base.cell.lengths()),
            sublattice_counts={
                label: int(np.count_nonzero(labels == label))
                for label in sorted(dict.fromkeys(definition.labels))
            },
            sublattice_elements=dict(occupants),
            truncated=len(a_values) > max_outputs,
        )

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
    target_mode: str = "all"
    target_elements: str = ""
    samples: int = 1
    use_seed: bool = False
    seed: int = 0


class CompositionGradientOperation(StructureOperation):
    """Assign site species from a composition gradient along one lattice coordinate."""

    AXIS_INDEX = {"a": 0, "b": 1, "c": 2, "x": 0, "y": 1, "z": 2}

    @classmethod
    def sampling_summary(
        cls,
        params: CompositionGradientParams,
        structure=None,
    ) -> dict[str, object]:
        elements = parse_element_list(params.elements)
        if len(elements) < 2:
            raise ValueError("Composition Gradient requires at least two elements.")
        start_comp = cls._normalized_composition(params.start_composition, elements)
        end_comp = cls._normalized_composition(params.end_composition, elements)
        if not start_comp or not end_comp:
            raise ValueError(
                "Composition Gradient requires valid start and end compositions."
            )

        axis_key = str(params.axis).strip().lower()
        if axis_key not in cls.AXIS_INDEX:
            raise ValueError("Composition Gradient axis must be one of a, b, or c.")
        bins = int(params.bins)
        if bins < 2:
            raise CardOperationError(
                "composition_gradient.too_few_groups",
                "Composition Gradient requires at least two equal-count groups.",
            )
        samples = int(params.samples)
        if samples < 1:
            raise CardOperationError(
                "composition_gradient.too_few_samples",
                "Composition Gradient requires at least one random sample.",
            )

        summary: dict[str, object] = {
            "elements": elements,
            "start_composition": start_comp,
            "end_composition": end_comp,
            "axis": cls._axis_name(cls.AXIS_INDEX[axis_key]),
            "requested_groups": bins,
            "samples": samples,
            "outputs_per_input": samples,
        }
        if structure is not None:
            candidate_indices = cls._candidate_indices(
                structure,
                params.target_elements,
                params.target_mode,
            )
            candidate_count = int(candidate_indices.size)
            if candidate_count < 2:
                raise CardOperationError(
                    "composition_gradient.too_few_candidates",
                    "Composition Gradient requires at least two eligible sites.",
                )
            effective_groups = min(bins, candidate_count)
            quotient, remainder = divmod(candidate_count, effective_groups)
            summary.update(
                {
                    "candidate_indices": candidate_indices,
                    "candidate_sites": candidate_count,
                    "effective_groups": effective_groups,
                    "min_group_size": quotient,
                    "max_group_size": quotient + (1 if remainder else 0),
                }
            )
        return summary

    def run_structure(self, structure, params: CompositionGradientParams) -> list:
        summary = self.sampling_summary(params, structure)
        elements = summary["elements"]
        start_comp = summary["start_composition"]
        end_comp = summary["end_composition"]
        candidate_indices = summary["candidate_indices"]
        bins = int(summary["effective_groups"])
        axis_idx = self.AXIS_INDEX[str(summary["axis"])]
        coord = self._axis_coordinate(structure, axis_idx)
        order = candidate_indices[np.argsort(coord[candidate_indices], kind="mergesort")]
        groups = [group for group in np.array_split(order, bins) if len(group) > 0]
        if not groups:
            raise ValueError("Composition Gradient could not build nonempty coordinate bins.")

        base_seed = int(params.seed) if params.use_seed else None
        cfg_id = stable_config_id(structure)
        outputs = []
        for sample_idx in range(int(summary["samples"])):
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
    def _candidate_indices(
        structure,
        target_elements: str,
        target_mode: str = "all",
    ) -> np.ndarray:
        mode = str(target_mode or "all").strip().lower()
        if mode not in {"all", "listed"}:
            raise ValueError("Composition Gradient target mode must be all or listed.")
        targets = set(parse_element_list(target_elements))
        # A nonempty legacy target list implied listed-site mode before the UI
        # gained an explicit scope selector.
        if mode == "all" and not targets:
            return np.arange(len(structure), dtype=int)
        if not targets:
            raise CardOperationError(
                "composition_gradient.missing_targets",
                "List one or more existing elements for the selected site scope.",
            )
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
        for token in reversed(cfg.split("|")):
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


def _eval_condition_array(
    node: ast.AST,
    env: dict[str, np.ndarray],
    tol: float,
):
    """Evaluate a validated condition tree against coordinate arrays."""
    if isinstance(node, ast.Expression):
        return _eval_condition_array(node.body, env, tol)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _eval_condition_array(node.operand, env, tol)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.Not):
            return np.logical_not(value)
    if isinstance(node, ast.BinOp):
        left = _eval_condition_array(node.left, env, tol)
        right = _eval_condition_array(node.right, env, tol)
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
        left = _eval_condition_array(node.left, env, tol)
        result = True
        for operator, comparator in zip(node.ops, node.comparators):
            right = _eval_condition_array(comparator, env, tol)
            if isinstance(operator, ast.Eq):
                current = np.isclose(left, right, rtol=0.0, atol=tol)
            elif isinstance(operator, ast.NotEq):
                current = np.logical_not(
                    np.isclose(left, right, rtol=0.0, atol=tol)
                )
            elif isinstance(operator, ast.Lt):
                current = left < right
            elif isinstance(operator, ast.LtE):
                current = np.logical_or(
                    left <= right,
                    np.isclose(left, right, rtol=0.0, atol=tol),
                )
            elif isinstance(operator, ast.Gt):
                current = left > right
            elif isinstance(operator, ast.GtE):
                current = np.logical_or(
                    left >= right,
                    np.isclose(left, right, rtol=0.0, atol=tol),
                )
            result = np.logical_and(result, current)
            left = right
        return result
    if isinstance(node, ast.BoolOp):
        values = [_eval_condition_array(value, env, tol) for value in node.values]
        result = values[0]
        for value in values[1:]:
            if isinstance(node.op, ast.And):
                result = np.logical_and(result, value)
            elif isinstance(node.op, ast.Or):
                result = np.logical_or(result, value)
        return result
    raise ValueError("Unsupported expression")


def _parse_condition_tree(expr: str) -> ast.Expression:
    """Parse and validate a Cartesian selection expression without evaluating it."""
    expr_py = normalize_condition_expr(expr)
    try:
        tree = ast.parse(expr_py, mode="eval")
    except SyntaxError as exc:
        raise CardOperationError(
            "conditional_replace.condition_syntax",
            "Invalid Cartesian position filter syntax: {reason}.",
            reason=exc.msg,
        ) from exc
    if not _is_allowed_condition_node(tree):
        raise CardOperationError(
            "conditional_replace.condition_unsupported",
            "Cartesian position filter contains unsupported syntax.",
        )
    unknown_names = sorted(
        {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        - {"x", "y", "z"}
    )
    if unknown_names:
        raise CardOperationError(
            "conditional_replace.condition_names",
            "Cartesian position filter may use only x, y, and z; unknown name(s): {names}.",
            names=", ".join(unknown_names),
        )
    invalid_constant = any(
        isinstance(node, ast.Constant)
        and not isinstance(node.value, bool)
        and (
            not isinstance(node.value, (int, float))
            or not np.isfinite(float(node.value))
        )
        for node in ast.walk(tree)
    )
    if invalid_constant:
        raise CardOperationError(
            "conditional_replace.condition_constants",
            "Cartesian position filter may use only finite numeric constants.",
        )
    if not isinstance(tree.body, (ast.BoolOp, ast.Compare)) and not (
        isinstance(tree.body, ast.UnaryOp) and isinstance(tree.body.op, ast.Not)
    ) and not (
        isinstance(tree.body, ast.Constant) and isinstance(tree.body.value, bool)
    ):
        raise CardOperationError(
            "conditional_replace.condition_boolean",
            "Cartesian position filter must be a comparison or a boolean expression.",
        )
    return tree


def evaluate_condition(expr: str, coords: np.ndarray) -> bool | np.ndarray:
    """Safely evaluate a coordinate condition against one or more positions."""
    tree = _parse_condition_tree(expr)
    coords_arr = np.asarray(coords, dtype=float)

    def eval_single(pos) -> bool:
        x, y, z = map(float, pos[:3])
        try:
            return bool(
                _eval_condition_node(tree, {"x": x, "y": y, "z": z}, tol=1e-4)
            )
        except ZeroDivisionError as exc:
            raise CardOperationError(
                "conditional_replace.condition_division",
                "Cartesian position filter divides by zero for at least one atom.",
            ) from exc

    if coords_arr.ndim == 1:
        return eval_single(coords_arr)
    if coords_arr.ndim == 2:
        if coords_arr.shape[1] < 3:
            raise ValueError(f"Unsupported coordinate shape: {coords_arr.shape}")
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                result = _eval_condition_array(
                    tree,
                    {
                        "x": coords_arr[:, 0],
                        "y": coords_arr[:, 1],
                        "z": coords_arr[:, 2],
                    },
                    tol=1.0e-4,
                )
        except FloatingPointError as exc:
            raise CardOperationError(
                "conditional_replace.condition_nonfinite",
                "Cartesian position filter produces non-finite arithmetic for at least one atom.",
            ) from exc
        result_array = np.asarray(result, dtype=bool)
        if result_array.ndim == 0:
            return np.full(len(coords_arr), bool(result_array), dtype=bool)
        if result_array.shape != (len(coords_arr),):
            raise ValueError("Cartesian position filter returned an invalid shape.")
        return result_array
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
            if not name:
                raise ValueError("Replacement element names must not be empty.")
            if not np.isfinite(ratio) or ratio < 0.0:
                raise ValueError(
                    "Replacement ratios must be finite and non-negative."
                )
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
        if not name:
            raise ValueError("Replacement element names must not be empty.")
        if not np.isfinite(ratio) or ratio < 0.0:
            raise ValueError("Replacement ratios must be finite and non-negative.")
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

    def selection_summary(
        self,
        params: ConditionalReplaceParams,
        structure=None,
    ) -> dict[str, object]:
        """Validate the request and optionally count matching Cartesian sites."""
        try:
            targets = parse_element_list(str(params.target or ""))
        except ValueError as exc:
            raise CardOperationError(
                "conditional_replace.invalid_target",
                "Enter one valid target element symbol, such as O, Si, or Fe.",
            ) from exc
        if len(targets) != 1:
            raise CardOperationError(
                "conditional_replace.invalid_target",
                "Enter one valid target element symbol, such as O, Si, or Fe.",
            )
        target = targets[0]
        if target not in atomic_numbers:
            raise CardOperationError(
                "conditional_replace.unknown_target",
                "Unknown target element symbol: {element}.",
                element=target,
            )

        try:
            raw_names, raw_ratios = parse_replacements(params.replacements)
        except (TypeError, ValueError) as exc:
            raise CardOperationError(
                "conditional_replace.invalid_ratios",
                "Replacement ratios must be finite and non-negative.",
            ) from exc
        if not raw_names:
            raise CardOperationError(
                "conditional_replace.no_replacements",
                "Add at least one replacement element with a positive relative ratio.",
            )

        names: list[str] = []
        ratios: list[float] = []
        invalid_names: list[str] = []
        for raw_name, raw_ratio in zip(raw_names, raw_ratios):
            try:
                parsed = parse_element_list(str(raw_name))
            except ValueError:
                parsed = []
            if len(parsed) != 1 or parsed[0] not in atomic_numbers:
                invalid_names.append(str(raw_name))
                continue
            name = parsed[0]
            if name in names:
                raise CardOperationError(
                    "conditional_replace.duplicate_replacement",
                    "Replacement element {element} appears more than once.",
                    element=name,
                )
            names.append(name)
            ratios.append(float(raw_ratio))
        if invalid_names:
            raise CardOperationError(
                "conditional_replace.unknown_replacements",
                "Unknown replacement element symbol(s): {elements}.",
                elements=", ".join(invalid_names),
            )
        if target in names:
            raise CardOperationError(
                "conditional_replace.self_replacement",
                "Replacement elements must not include the target element "
                "{element}; use Random Doping for partial replacement.",
                element=target,
            )

        positive = [
            (name, ratio) for name, ratio in zip(names, ratios) if ratio > 0.0
        ]
        if not positive:
            raise CardOperationError(
                "conditional_replace.zero_ratios",
                "Add at least one replacement element with a positive relative ratio.",
            )
        names = [name for name, _ratio in positive]
        ratios_arr = np.asarray([ratio for _name, ratio in positive], dtype=float)
        ratios_arr /= ratios_arr.sum()

        try:
            mode = int(params.mode)
        except (TypeError, ValueError) as exc:
            raise CardOperationError(
                "conditional_replace.invalid_mode",
                "Element allocation must be Independent random assignment or Match overall ratio.",
            ) from exc
        if mode not in (0, 1):
            raise CardOperationError(
                "conditional_replace.invalid_mode",
                "Element allocation must be Independent random assignment or Match overall ratio.",
            )
        try:
            seed = int(params.seed)
        except (TypeError, ValueError) as exc:
            raise CardOperationError(
                "conditional_replace.invalid_seed",
                "Conditional Replace seed must be a non-negative integer.",
            ) from exc
        if seed < 0:
            raise CardOperationError(
                "conditional_replace.invalid_seed",
                "Conditional Replace seed must be a non-negative integer.",
            )

        condition = str(params.condition or "").strip() or "all"
        _parse_condition_tree(condition)
        summary: dict[str, object] = {
            "target": target,
            "replacement_elements": tuple(names),
            "normalized_ratios": tuple(float(value) for value in ratios_arr),
            "condition": condition,
            "mode": mode,
            "seed": seed,
            "outputs_per_input": 1,
        }
        if structure is None:
            return summary

        symbols = np.asarray(structure.get_chemical_symbols(), dtype=object)
        positions = np.asarray(structure.get_positions(), dtype=float)
        if positions.shape != (len(symbols), 3) or np.any(~np.isfinite(positions)):
            raise CardOperationError(
                "conditional_replace.invalid_positions",
                "Conditional Replace requires finite Cartesian atom positions.",
            )
        target_mask = symbols == target
        target_count = int(np.count_nonzero(target_mask))
        if target_count == 0:
            raise CardOperationError(
                "conditional_replace.no_target_atoms",
                "The input structure contains no {element} atoms.",
                element=target,
            )
        condition_result = evaluate_condition(condition, positions)
        condition_mask = (
            np.asarray(condition_result, dtype=bool)
            if isinstance(condition_result, np.ndarray)
            else np.full(len(symbols), bool(condition_result), dtype=bool)
        )
        matched = int(np.count_nonzero(target_mask & condition_mask))
        if matched == 0:
            raise CardOperationError(
                "conditional_replace.no_matches",
                "The Cartesian position filter matches no {element} atoms.",
                element=target,
            )
        summary["target_sites"] = target_count
        summary["matched_sites"] = matched
        if mode == 1:
            counts = fractions_to_counts_exact(ratios_arr, matched)
            summary["replacement_counts"] = tuple(
                (name, int(count)) for name, count in zip(names, counts)
            )
        return summary

    def run_structure(self, structure, params: ConditionalReplaceParams) -> list:
        summary = self.selection_summary(params, structure)
        target = str(summary["target"])
        new_atoms = list(summary["replacement_elements"])
        ratios = list(summary["normalized_ratios"])
        seed_value = int(summary["seed"])
        seed = seed_value if seed_value != 0 else None
        exact = int(summary["mode"]) == 1
        new_structure, replaced = replace_atoms_with_conditions(
            structure,
            atom_to_replace=target,
            new_atoms=new_atoms,
            probabilities=ratios,
            condition=params.condition.strip() or "all",
            seed=seed,
            exact=exact,
        )
        if replaced != int(summary["matched_sites"]):
            raise RuntimeError("Conditional Replace matched-site count changed during execution.")
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
        return structure.copy(), 0

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
# ----------------------------------------------------------------------
# Interface thin-layer interdiffusion (界面随机互混)
#
# Detects a bilayer interface, picks near-interface atomic layers on both
# sides (L = below the interface, R = above it), and swaps atom species
# between the two selected regions at a fixed or gradient concentration.
# ----------------------------------------------------------------------

INTERFACE_AXIS_INDEX = {"a": 0, "b": 1, "c": 2}
MIN_CONTRAST = 1e-4
CONC_TOLERANCE = 1e-6


@dataclass(frozen=True)
class InterfaceLayerMixParams:
    axis: str = "auto"
    auto_position: bool = True
    interface_position: float = 0.5
    layer_tolerance: float = 0.25
    left_layers: int = 2
    right_layers: int = 2
    mode: str = "fixed"
    concentration: float = 0.5
    gradient_start: float = 0.0
    gradient_end: float = 1.0
    num_structures: int = 1
    use_seed: bool = False
    seed: int = 0


class InterfaceLayerMixOperation(StructureOperation):
    """Swap atom species between near-interface layers of a bilayer."""

    def run_structure(self, structure, params: InterfaceLayerMixParams) -> list:
        resolved = self._resolve(structure, params)
        c_schedule = self._concentration_schedule(params, resolved["c_max"])
        outputs = []
        base_seed = (
            self._validated_integer(params.seed, "Random seed")
            if bool(params.use_seed)
            else None
        )
        cfg_id = stable_config_id(structure)
        for sample_idx, c in enumerate(c_schedule):
            if base_seed is None:
                rng = np.random.default_rng()
                seed_tag = ""
            else:
                derived_seed = int(base_seed + cfg_id * 1000003 + sample_idx)
                rng = np.random.default_rng(derived_seed)
                seed_tag = f",s={derived_seed}"
            atoms = structure.copy()
            pair_count = self._pair_count(
                c, resolved["n_total"], resolved["pair_capacity"]
            )
            c_effective = 2.0 * pair_count / resolved["n_total"]
            changed = self._swap_atoms(atoms, resolved, pair_count, rng)
            if changed:
                self._invalidate_reference_labels(atoms)
            target_tag = (
                ""
                if math.isclose(c_effective, c, rel_tol=0.0, abs_tol=CONC_TOLERANCE)
                else f",target={c:.3g}"
            )
            append_config_tag(
                atoms,
                f"IfaceMix(L={int(params.left_layers)},R={int(params.right_layers)},"
                f"c={c_effective:.3g}{target_tag}{seed_tag})",
            )
            outputs.append(atoms)
        return outputs

    def interface_summary(self, structure, params: InterfaceLayerMixParams) -> dict:
        """Deterministic geometry summary for the preview panel (no RNG)."""
        resolved = self._resolve(structure, params)
        requested = self._concentration_schedule(params, resolved["c_max"])
        effective = [
            2.0
            * self._pair_count(c, resolved["n_total"], resolved["pair_capacity"])
            / resolved["n_total"]
            for c in requested
        ]
        return {
            "axis": resolved["axis"],
            "position": resolved["position"],
            "left_formula": resolved["left_formula"],
            "right_formula": resolved["right_formula"],
            "left_layers_available": resolved["left_layers_available"],
            "right_layers_available": resolved["right_layers_available"],
            "left_layers": int(params.left_layers),
            "right_layers": int(params.right_layers),
            "n_left": resolved["n_left"],
            "n_right": resolved["n_right"],
            "n_total": resolved["n_total"],
            "c_max": resolved["c_max"],
            "pair_capacity": resolved["pair_capacity"],
            "requested_concentrations": requested,
            "effective_concentrations": effective,
            "num_structures": int(params.num_structures),
            "mode": str(params.mode).strip(),
        }

    # ------------------------------------------------------------------
    # geometry / validation
    # ------------------------------------------------------------------

    def _resolve(self, structure, params: InterfaceLayerMixParams) -> dict:
        if int(getattr(structure.cell, "rank", 0)) < 3:
            raise CardOperationError(
                "interface.singular_cell",
                "Interface Layer Mixing requires a non-singular 3D cell.",
            )
        if len(structure) < 2:
            raise CardOperationError(
                "interface.too_few_atoms",
                "Interface Layer Mixing requires at least two atoms.",
            )
        cell = np.asarray(structure.cell.array, dtype=float)
        positions = np.asarray(structure.positions, dtype=float)
        if not np.all(np.isfinite(cell)) or not np.all(np.isfinite(positions)):
            raise CardOperationError(
                "interface.non_finite_geometry",
                "Interface Layer Mixing requires finite cell vectors and atom positions.",
            )

        symbols = np.asarray(structure.get_chemical_symbols(), dtype=object)
        species = sorted(set(symbols.tolist()), key=lambda s: atomic_numbers.get(s, 200))
        if len(species) < 2:
            raise CardOperationError(
                "interface.single_element",
                "Interface Layer Mixing found only one element ({element}); "
                "swapping would not change the structure.",
                element=species[0],
            )

        left_layers = self._validated_integer(
            params.left_layers, "L-side layer count"
        )
        right_layers = self._validated_integer(
            params.right_layers, "R-side layer count"
        )
        if left_layers < 1:
            raise CardOperationError(
                "interface.left_layers",
                "L-side layer count must be >= 1 (got {value}).",
                value=left_layers,
            )
        if right_layers < 1:
            raise CardOperationError(
                "interface.right_layers",
                "R-side layer count must be >= 1 (got {value}).",
                value=right_layers,
            )
        num_structures = self._validated_integer(
            params.num_structures, "Number of structures"
        )
        if num_structures < 1:
            raise CardOperationError(
                "interface.num_structures",
                "Number of structures must be >= 1 (got {value}).",
                value=num_structures,
            )
        if bool(params.use_seed):
            seed = self._validated_integer(params.seed, "Random seed")
            if seed < 0 or seed > 2**31 - 1:
                raise CardOperationError(
                    "interface.seed",
                    "Random seed must be between 0 and {maximum} (got {value}).",
                    maximum=2**31 - 1,
                    value=seed,
                )
        layer_tolerance = float(params.layer_tolerance)
        if not np.isfinite(layer_tolerance) or layer_tolerance <= 0.0:
            raise CardOperationError(
                "interface.layer_tolerance",
                "Layer tolerance must be a finite distance greater than 0 Å (got {value}).",
                value=params.layer_tolerance,
            )

        coord3 = scaled_positions(structure, wrap=True)
        axis_key = str(params.axis).strip().lower()
        axis_idx = INTERFACE_AXIS_INDEX.get(axis_key)
        if axis_key != "auto" and axis_idx is None:
            raise CardOperationError(
                "interface.invalid_axis",
                "Interface axis must be auto, a, b, or c (got {axis}).",
                axis=params.axis,
            )

        pos = None
        if axis_key == "auto":
            best = None
            for idx in range(3):
                contrast, candidate_pos = self._best_split(symbols, species, coord3[:, idx])
                if best is None or contrast > best[0]:
                    best = (contrast, idx, candidate_pos)
            contrast, axis_idx, pos = best
            axis_key = ("a", "b", "c")[axis_idx]
            if pos is None or contrast < MIN_CONTRAST:
                raise CardOperationError(
                    "interface.no_interface",
                    "Auto-detection found no interface with distinct compositions "
                    "(max contrast {contrast}). Check that the structure is a bilayer "
                    "or pick the interface normal axis manually.",
                    contrast=f"{contrast:.3g}",
                )

        if bool(params.auto_position):
            if pos is None:
                contrast, pos = self._best_split(symbols, species, coord3[:, axis_idx])
            if pos is None or contrast < MIN_CONTRAST:
                raise CardOperationError(
                    "interface.no_interface",
                    "Lattice axis {axis} shows no distinct-composition split "
                    "(contrast {contrast}). Try another axis, or disable auto-locate "
                    "and type the interface position.",
                    axis=axis_key,
                    contrast=f"{contrast:.3g}",
                )
        else:
            pos = float(params.interface_position)
            if not np.isfinite(pos) or not (0.0 < pos < 1.0):
                raise CardOperationError(
                    "interface.invalid_position",
                    "Interface fractional position must be strictly between 0 and 1 (got {pos}).",
                    pos=f"{pos:.4g}",
                )

        coord = coord3[:, axis_idx]
        l_mask = coord < pos
        r_mask = ~l_mask
        idx = np.arange(len(structure), dtype=int)

        normal_scale = 1.0 / float(np.linalg.norm(np.linalg.inv(cell)[:, axis_idx]))
        l_layer_id, n_left_avail = self._layer_ids(
            coord[l_mask], normal_scale, layer_tolerance
        )
        if n_left_avail < left_layers:
            raise CardOperationError(
                "interface.not_enough_layers",
                "Not enough atomic layers below the interface: need {need}, only "
                "{have} available. Reduce the L-side layer count.",
                need=left_layers,
                have=n_left_avail,
            )
        l_sel = np.zeros(len(structure), dtype=bool)
        l_sel[l_mask] = n_left_avail - 1 - l_layer_id < left_layers

        r_layer_id, n_right_avail = self._layer_ids(
            coord[r_mask], normal_scale, layer_tolerance
        )
        if n_right_avail < right_layers:
            raise CardOperationError(
                "interface.not_enough_layers",
                "Not enough atomic layers above the interface: need {need}, only "
                "{have} available. Reduce the R-side layer count.",
                need=right_layers,
                have=n_right_avail,
            )
        r_sel = np.zeros(len(structure), dtype=bool)
        r_sel[r_mask] = r_layer_id < right_layers

        l_idx = idx[l_sel]
        r_idx = idx[r_sel]
        n_left = int(l_idx.size)
        n_right = int(r_idx.size)

        l_elements = set(symbols[l_idx].tolist())
        r_elements = set(symbols[r_idx].tolist())
        pair_capacity = self._unlike_pair_capacity(symbols[l_idx], symbols[r_idx])
        if pair_capacity == 0:
            raise CardOperationError(
                "interface.same_elements",
                "Both selected regions are the same single element {element}; "
                "swapping would not change the structure.",
                element=next(iter(l_elements & r_elements), "the same element"),
            )

        n_total = n_left + n_right
        c_max = 2.0 * pair_capacity / n_total
        return {
            "axis": axis_key,
            "position": pos,
            "n_left": n_left,
            "n_right": n_right,
            "n_total": n_total,
            "left_formula": self._formula(symbols[l_idx]),
            "right_formula": self._formula(symbols[r_idx]),
            "left_layers_available": n_left_avail,
            "right_layers_available": n_right_avail,
            "left_index": l_idx,
            "right_index": r_idx,
            "c_max": float(c_max),
            "pair_capacity": int(pair_capacity),
        }

    def _concentration_schedule(self, params: InterfaceLayerMixParams, c_max: float) -> list[float]:
        num = int(params.num_structures)
        mode = str(params.mode).strip()
        if mode == "fixed":
            c = float(params.concentration)
            if not np.isfinite(c) or c < 0.0 or c > c_max + CONC_TOLERANCE:
                raise CardOperationError(
                    "interface.concentration_exceeds_max",
                    "Target concentration {c} exceeds this interface's swap capacity "
                    "{c_max}. Lower the concentration or add more layers.",
                    c=f"{c:.4g}",
                    c_max=f"{c_max:.4g}",
                )
            return [c] * num
        if mode == "gradient":
            start = float(params.gradient_start)
            end = float(params.gradient_end)
            top = max(start, end)
            if (
                not np.all(np.isfinite([start, end]))
                or start < 0.0
                or end < 0.0
                or top > c_max + CONC_TOLERANCE
            ):
                raise CardOperationError(
                    "interface.concentration_exceeds_max",
                    "Gradient concentration bound {top} exceeds this interface's swap "
                    "capacity {c_max}. Lower the concentration or add more layers.",
                    top=f"{top:.4g}",
                    c_max=f"{c_max:.4g}",
                )
            if num == 1:
                return [start]
            return [start + (end - start) * i / (num - 1) for i in range(num)]
        raise CardOperationError(
            "interface.invalid_mode",
            "Concentration mode must be fixed or gradient (got {mode}).",
            mode=mode,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _best_split(self, symbols: np.ndarray, species: list[str], coord: np.ndarray) -> tuple[float, float | None]:
        """Return (contrast, split position) for the best split on one axis."""
        n_species = len(species)
        species_index = {s: i for i, s in enumerate(species)}
        sym_idx = np.asarray([species_index[s] for s in symbols], dtype=int)
        uniq_vals, inv = np.unique(coord, return_inverse=True)
        n_uniq = uniq_vals.size
        if n_uniq < 2:
            return 0.0, None
        uniq_hist = np.zeros((n_uniq, n_species), dtype=float)
        for s in range(n_species):
            uniq_hist[:, s] = np.bincount(inv[sym_idx == s], minlength=n_uniq)
        cum_hist = np.cumsum(uniq_hist, axis=0)
        cum_count = np.cumsum(uniq_hist.sum(axis=1))
        total = int(cum_count[-1])
        best_contrast = 0.0
        best_pos = None
        for i in range(n_uniq - 1):
            n_left = int(cum_count[i])
            n_right = total - n_left
            if n_left == 0 or n_right == 0:
                continue
            hist_l = cum_hist[i] / n_left
            hist_r = (cum_hist[-1] - cum_hist[i]) / n_right
            contrast = 1.0 - self._cosine(hist_l, hist_r)
            pos = 0.5 * (float(uniq_vals[i]) + float(uniq_vals[i + 1]))
            if contrast > best_contrast:
                best_contrast = contrast
                best_pos = pos
        return float(best_contrast), best_pos

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b)) / denom

    @staticmethod
    def _layer_ids(
        coords: np.ndarray, normal_scale: float, tolerance: float
    ) -> tuple[np.ndarray, int]:
        """Cluster nearby fractional coordinates using a physical Å tolerance."""
        if coords.size == 0:
            return np.zeros(0, dtype=int), 0
        order = np.argsort(coords, kind="stable")
        layer_ids = np.zeros(coords.size, dtype=int)
        layer = 0
        previous = float(coords[order[0]])
        for atom_index in order[1:]:
            current = float(coords[atom_index])
            if (current - previous) * normal_scale > tolerance:
                layer += 1
            layer_ids[atom_index] = layer
            previous = current
        return layer_ids, layer + 1

    @staticmethod
    def _formula(symbols_subset: np.ndarray) -> str:
        counts = Counter(symbols_subset.tolist())
        order = sorted(counts, key=lambda s: atomic_numbers.get(s, 200))
        g = 0
        for v in counts.values():
            g = math.gcd(g, v)
        parts = []
        for s in order:
            n = counts[s] // g if g else counts[s]
            parts.append(s if n == 1 else f"{s}{n}")
        return "".join(parts)

    @staticmethod
    def _validated_integer(value, label: str) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise CardOperationError(
                "interface.integer_parameter",
                "{label} must be an integer (got {value}).",
                label=label,
                value=value,
            ) from None
        if not np.isfinite(number) or not number.is_integer():
            raise CardOperationError(
                "interface.integer_parameter",
                "{label} must be an integer (got {value}).",
                label=label,
                value=value,
            )
        return int(number)

    @staticmethod
    def _unlike_pair_capacity(
        left_symbols: np.ndarray, right_symbols: np.ndarray
    ) -> int:
        left = Counter(left_symbols.tolist())
        right = Counter(right_symbols.tolist())
        n_left = int(len(left_symbols))
        n_right = int(len(right_symbols))
        dominant = max(
            (left[element] + right[element] for element in set(left) | set(right)),
            default=0,
        )
        return max(0, min(n_left, n_right, n_left + n_right - dominant))

    @staticmethod
    def _pair_count(c: float, n_total: int, capacity: int) -> int:
        return min(int(np.round(float(c) * n_total / 2.0)), int(capacity))

    @classmethod
    def _swap_atoms(
        cls, atoms, resolved: dict, pair_count: int, rng: np.random.Generator
    ) -> bool:
        if pair_count <= 0:
            return False
        symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
        l_idx = resolved["left_index"]
        r_idx = resolved["right_index"]
        lp, rp = cls._sample_unlike_pairs(symbols, l_idx, r_idx, pair_count, rng)
        old_l = symbols[lp].copy()
        symbols[lp] = symbols[rp]
        symbols[rp] = old_l
        for name in ("spin", "initial_magmoms", "initial_charges"):
            values = atoms.arrays.get(name)
            if values is None:
                continue
            old_values = np.asarray(values[lp]).copy()
            values[lp] = values[rp]
            values[rp] = old_values
        atoms.set_chemical_symbols(symbols.tolist())
        return True

    @staticmethod
    def _sample_unlike_pairs(symbols, l_idx, r_idx, count, rng):
        """Return exactly ``count`` cross-interface pairs with unlike species."""
        left_groups = {
            element: list(rng.permutation(l_idx[symbols[l_idx] == element]))
            for element in sorted(set(symbols[l_idx].tolist()))
        }
        right_groups = {
            element: list(rng.permutation(r_idx[symbols[r_idx] == element]))
            for element in sorted(set(symbols[r_idx].tolist()))
        }
        left_species = list(rng.permutation(list(left_groups)))
        right_species = list(rng.permutation(list(right_groups)))

        source = 0
        left_offset = 1
        right_offset = left_offset + len(left_species)
        sink = right_offset + len(right_species)
        capacity = np.zeros((sink + 1, sink + 1), dtype=int)
        for i, element in enumerate(left_species):
            capacity[source, left_offset + i] = len(left_groups[element])
        for j, element in enumerate(right_species):
            capacity[right_offset + j, sink] = len(right_groups[element])
        for i, left_element in enumerate(left_species):
            for j, right_element in enumerate(right_species):
                if left_element != right_element:
                    capacity[left_offset + i, right_offset + j] = count

        flow = np.zeros_like(capacity)
        total_flow = 0
        while total_flow < count:
            parent = np.full(sink + 1, -1, dtype=int)
            parent[source] = source
            queue = [source]
            for node in queue:
                for nxt in range(sink + 1):
                    if parent[nxt] < 0 and capacity[node, nxt] - flow[node, nxt] > 0:
                        parent[nxt] = node
                        queue.append(nxt)
                        if nxt == sink:
                            break
                if parent[sink] >= 0:
                    break
            if parent[sink] < 0:
                raise RuntimeError("Internal unlike-species matching failed.")
            amount = count - total_flow
            node = sink
            while node != source:
                prev = int(parent[node])
                amount = min(amount, int(capacity[prev, node] - flow[prev, node]))
                node = prev
            node = sink
            while node != source:
                prev = int(parent[node])
                flow[prev, node] += amount
                flow[node, prev] -= amount
                node = prev
            total_flow += amount

        left_pairs: list[int] = []
        right_pairs: list[int] = []
        for i, left_element in enumerate(left_species):
            for j, right_element in enumerate(right_species):
                amount = int(flow[left_offset + i, right_offset + j])
                for _ in range(amount):
                    left_pairs.append(left_groups[left_element].pop())
                    right_pairs.append(right_groups[right_element].pop())
        order = rng.permutation(len(left_pairs))
        return (
            np.asarray(left_pairs, dtype=int)[order],
            np.asarray(right_pairs, dtype=int)[order],
        )

    @staticmethod
    def _invalidate_reference_labels(atoms) -> None:
        """Drop labels invalidated by changing chemical species at fixed sites."""
        atoms.calc = None
        for key in (
            "energy",
            "free_energy",
            "virial",
            "stress",
            "dipole",
            "magmom",
        ):
            atoms.info.pop(key, None)
        for key in (
            "forces",
            "force",
            "energies",
            "atomic_energy",
            "magmoms",
            "charges",
            "bec",
            "born_effective_charges",
        ):
            atoms.arrays.pop(key, None)
