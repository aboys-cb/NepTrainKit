"""UI-independent defect and surface Make Dataset operations."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import comb
from typing import Any, Sequence

import numpy as np
from ase import Atom
from ase.build import surface
from ase.data import atomic_numbers
from ase.geometry import geometry
from loguru import logger
from scipy.stats.qmc import Sobol

from NepTrainKit.core.config_type import append_config_tag

from .geometry import wrapped_positions
from .operation import StructureOperation
from .sampling import derived_structure_seed


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def _count_range(value: Any, *, label: str) -> tuple[int, int]:
    values = _as_list(value)
    if not values:
        raise ValueError(f"{label} must not be empty.")
    if len(values) == 1:
        low = high = int(values[0])
    elif len(values) == 2:
        low, high = [int(item) for item in values]
    else:
        raise ValueError(f"{label} must contain one or two values.")
    if low > high:
        raise ValueError(f"{label} minimum must be <= maximum.")
    return low, high


def _range_values(values: Sequence[float], *, include_step: bool = False) -> np.ndarray:
    if len(values) != 3:
        raise ValueError("Range must contain exactly three values: start, stop, step.")
    start, end, step = values
    start, end, step = float(start), float(end), float(step)
    if not np.all(np.isfinite([start, end, step])):
        raise ValueError("Range values must be finite.")
    if step <= 0.0:
        raise ValueError("Range step must be positive.")
    if end < start:
        start, end = end, start
    if include_step:
        return np.arange(start, end + step, step)
    return np.arange(start, end + step / 2, step)


def _parse_insert_species(tokens: str) -> tuple[list[str], list[float]]:
    """Parse and validate comma-separated ``Element[:weight]`` entries."""
    raw_tokens = [item.strip() for item in str(tokens or "").split(",")]
    entries = [item for item in raw_tokens if item]
    if not entries:
        raise ValueError(
            "InsertDefect: species must contain at least one element."
        )

    combined: dict[str, float] = {}
    for item in entries:
        if ":" in item:
            symbol, weight_text = item.split(":", 1)
            symbol = symbol.strip()
            try:
                weight = float(weight_text.strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"InsertDefect: invalid weight for element {symbol or item}."
                ) from exc
        else:
            symbol = item.strip()
            weight = 1.0

        if symbol not in atomic_numbers:
            raise ValueError(
                f"InsertDefect: unknown chemical element '{symbol}'."
            )
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"InsertDefect: weight for {symbol} must be finite and positive."
            )
        combined[symbol] = combined.get(symbol, 0.0) + weight

    species = list(combined)
    raw_weights = np.asarray([combined[symbol] for symbol in species], dtype=float)
    weights = (raw_weights / raw_weights.sum()).tolist()
    return species, weights


@dataclass(frozen=True)
class RandomVacancyParams:
    """Parameters for rule-based random vacancy generation."""

    rules: list[dict[str, Any]] = field(default_factory=list)
    max_structures: int = 1
    use_seed: bool = False
    seed: int = 0


class RandomVacancyOperation(StructureOperation):
    """Create unique vacancy structures by removing atoms matched by explicit rules."""

    _MAX_ATTEMPTS_PER_OUTPUT = 20

    @classmethod
    def _validated_rules(cls, structure, rules: Any) -> list[dict[str, Any]]:
        if not isinstance(rules, list) or not rules:
            raise ValueError("RandomVacancy requires at least one vacancy rule.")
        if len(structure) <= 1:
            raise ValueError("RandomVacancy requires at least two atoms.")

        symbols = np.asarray(structure.get_chemical_symbols(), dtype=object)
        normalized: list[dict[str, Any]] = []
        for rule_index, rule in enumerate(rules, start=1):
            label = f"RandomVacancy rule {rule_index}"
            if not isinstance(rule, dict):
                raise ValueError(f"{label} must be a mapping.")

            element = str(rule.get("element", "") or "").strip()
            if not element:
                raise ValueError(f"{label} requires an element.")

            count_min, count_max = _count_range(rule.get("count", []), label=f"{label} count")
            if count_min < 0:
                raise ValueError(f"{label} count must be >= 0.")

            count_mode = str(rule.get("count_mode", "") or "").strip().lower()
            if not count_mode:
                count_mode = "fixed" if count_min == count_max else "random"
            if count_mode not in {"fixed", "random"}:
                raise ValueError(f"{label} count_mode must be fixed or random.")
            if count_mode == "fixed":
                if count_min != count_max:
                    raise ValueError(f"{label} fixed count must use the same minimum and maximum.")
                if count_min == 0:
                    raise ValueError(f"{label} fixed count must be >= 1.")
            elif count_max == 0:
                raise ValueError(f"{label} random range must allow at least one vacancy.")

            raw_groups = rule.get("group")
            groups = [str(value).strip() for value in _as_list(raw_groups) if str(value).strip()]
            candidate_mask = symbols == element
            group_constraint_requested = raw_groups is not None and not (
                isinstance(raw_groups, str) and not raw_groups.strip()
            )
            if isinstance(raw_groups, (list, tuple, np.ndarray)) and len(raw_groups) == 0:
                group_constraint_requested = False
            if group_constraint_requested:
                if not groups:
                    raise ValueError(f"{label} group must contain at least one non-empty label.")
                if "group" not in structure.arrays:
                    raise ValueError(
                        f"{label} requests group labels, but the input structure has no group array."
                    )
                group_values = np.asarray(structure.arrays["group"], dtype=object)
                candidate_mask &= np.isin(group_values, groups)

            candidate_count = int(np.count_nonzero(candidate_mask))
            target = element if not groups else f"{element} in group {','.join(groups)}"
            if candidate_count == 0:
                raise ValueError(f"{label} matched no atoms ({target}).")
            if count_max > candidate_count:
                raise ValueError(
                    f"{label} requests up to {count_max} vacancies, but only "
                    f"{candidate_count} atoms match ({target})."
                )

            normalized.append(
                {
                    "element": element,
                    "groups": groups,
                    "count_mode": count_mode,
                    "count_min": count_min,
                    "count_max": count_max,
                    "candidate_mask": candidate_mask,
                    "candidate_count": candidate_count,
                }
            )
        return normalized

    @classmethod
    def rule_match_summary(cls, structure, rules: Any) -> list[dict[str, Any]]:
        """Validate rules and return candidate counts for UI previews."""
        return [
            {
                "element": rule["element"],
                "groups": list(rule["groups"]),
                "count_mode": rule["count_mode"],
                "count_min": rule["count_min"],
                "count_max": rule["count_max"],
                "candidate_count": rule["candidate_count"],
            }
            for rule in cls._validated_rules(structure, rules)
        ]

    @staticmethod
    def _output_upper_bound(rules: list[dict[str, Any]], requested: int) -> int:
        possible_outputs = 1
        for rule in rules:
            possible_for_rule = 0
            for count in range(rule["count_min"], rule["count_max"] + 1):
                possible_for_rule += comb(rule["candidate_count"], count)
                if possible_for_rule >= requested:
                    possible_for_rule = requested
                    break
            possible_outputs = min(requested, possible_outputs * possible_for_rule)
            if possible_outputs >= requested:
                return requested
        return possible_outputs

    @classmethod
    def maximum_unique_outputs(cls, structure, rules: Any, requested: int) -> int:
        """Return a safe upper bound for distinct deletion patterns."""
        requested = int(requested)
        if requested <= 0:
            raise ValueError("RandomVacancy: max_structures must be >= 1.")
        return cls._output_upper_bound(cls._validated_rules(structure, rules), requested)

    def run_structure(self, structure, params: RandomVacancyParams) -> list:
        max_structures = int(params.max_structures)
        if max_structures <= 0:
            raise ValueError("RandomVacancy: max_structures must be >= 1.")
        seed = int(params.seed)
        if params.use_seed and seed < 0:
            raise ValueError("RandomVacancy: seed must be >= 0.")

        rules = self._validated_rules(structure, params.rules)
        if params.use_seed:
            rng = np.random.default_rng(
                derived_structure_seed(seed, structure)
            )
        else:
            rng = np.random.default_rng()

        structure_list = []
        seen_deletions: set[tuple[int, ...]] = set()
        target_outputs = self._output_upper_bound(rules, max_structures)
        max_attempts = max(100, target_outputs * self._MAX_ATTEMPTS_PER_OUTPUT)
        last_invalid_reason: str | None = None
        for _ in range(max_attempts):
            new_structure = structure.copy()
            keep_mask = np.ones(len(new_structure), dtype=bool)
            total_remove = 0
            attempt_is_valid = True
            for rule_index, rule in enumerate(rules, start=1):
                if rule["count_mode"] == "fixed":
                    remove_num = rule["count_min"]
                else:
                    remove_num = int(rng.integers(rule["count_min"], rule["count_max"] + 1))
                if remove_num <= 0:
                    continue

                candidate_indices = np.nonzero(keep_mask & rule["candidate_mask"])[0]
                if remove_num > len(candidate_indices):
                    last_invalid_reason = (
                        f"RandomVacancy rule {rule_index} sampled {remove_num} vacancies, "
                        f"but only {len(candidate_indices)} eligible atoms remain after earlier rules."
                    )
                    attempt_is_valid = False
                    break

                idxs = rng.choice(candidate_indices, remove_num, replace=False)
                keep_mask[np.asarray(idxs, dtype=int)] = False
                total_remove += remove_num

            if not attempt_is_valid:
                continue
            if total_remove >= len(structure):
                last_invalid_reason = (
                    "RandomVacancy sampled a combination that would remove every atom from the structure."
                )
                continue

            deletion_key = tuple(int(index) for index in np.nonzero(~keep_mask)[0])
            if deletion_key in seen_deletions:
                continue
            seen_deletions.add(deletion_key)

            if total_remove:
                del new_structure[~keep_mask]
                append_config_tag(new_structure, f"Vac(n={total_remove})")
            structure_list.append(new_structure)
            if len(structure_list) >= target_outputs:
                break
        if not structure_list:
            detail = last_invalid_reason or "no distinct deletion pattern could be sampled."
            raise ValueError(
                "RandomVacancy could not generate a valid non-empty structure from the rules: "
                f"{detail}"
            )
        return structure_list


@dataclass(frozen=True)
class VacancyDefectParams:
    """Parameters for stochastic vacancy-defect sampling."""

    engine_type: int = 1
    num_condition: int = 1
    use_num: bool = True
    concentration_condition: float = 0.01
    count_mode: str = "fixed"
    max_structures: int = 1
    use_seed: bool = False
    seed: int = 0


class VacancyDefectOperation(StructureOperation):
    """Sample unique global vacancy patterns by count or concentration."""

    _MAX_SOBOL_ATOMS = Sobol.MAXDIM - 1
    _MAX_ATTEMPTS_PER_OUTPUT = 20
    _SOBOL_BATCH_BYTES = 8 * 1024 * 1024

    @classmethod
    def _validated_settings(cls, structure, params: VacancyDefectParams) -> dict[str, Any]:
        n_atoms = len(structure)
        if n_atoms <= 1:
            raise ValueError("VacancyDefect requires at least two atoms.")

        try:
            engine_type = int(params.engine_type)
        except (TypeError, ValueError) as exc:
            raise ValueError("VacancyDefect: engine_type must be 0 (Sobol) or 1 (Uniform).") from exc
        if engine_type not in {0, 1}:
            raise ValueError("VacancyDefect: engine_type must be 0 (Sobol) or 1 (Uniform).")

        count_mode = str(params.count_mode).strip().lower()
        if count_mode not in {"fixed", "random"}:
            raise ValueError("VacancyDefect: count_mode must be fixed or random.")

        try:
            max_structures_value = float(params.max_structures)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "VacancyDefect: max_structures must be an integer."
            ) from exc
        if not np.isfinite(max_structures_value) or not max_structures_value.is_integer():
            raise ValueError("VacancyDefect: max_structures must be an integer.")
        max_structures = int(max_structures_value)
        if max_structures <= 0:
            raise ValueError("VacancyDefect: max_structures must be >= 1.")

        if params.use_num:
            try:
                count_value = float(params.num_condition)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "VacancyDefect: vacancy count must be an integer."
                ) from exc
            if not np.isfinite(count_value) or not count_value.is_integer():
                raise ValueError("VacancyDefect: vacancy count must be an integer.")
            max_defects = int(count_value)
            if max_defects <= 0:
                raise ValueError("VacancyDefect: vacancy count must be >= 1.")
            if max_defects >= n_atoms:
                raise ValueError(
                    f"VacancyDefect: vacancy count must be <= {n_atoms - 1} "
                    "so at least one atom remains."
                )
        else:
            try:
                fraction = float(params.concentration_condition)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "VacancyDefect: vacancy fraction must be greater than 0 and less than 1."
                ) from exc
            if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
                raise ValueError("VacancyDefect: vacancy fraction must be greater than 0 and less than 1.")
            max_defects = int(fraction * n_atoms)
            if max_defects <= 0:
                minimum = 1.0 / n_atoms
                raise ValueError(
                    "VacancyDefect: vacancy fraction is too small for this structure; "
                    f"use at least {minimum:.6g} to remove one atom."
                )

        if engine_type == 0 and n_atoms > cls._MAX_SOBOL_ATOMS:
            raise ValueError(
                f"VacancyDefect: Sobol sampling supports at most {cls._MAX_SOBOL_ATOMS} atoms; "
                "use Uniform sampling for larger structures."
            )

        try:
            seed = int(params.seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("VacancyDefect: seed must be an integer.") from exc
        if params.use_seed and seed < 0:
            raise ValueError("VacancyDefect: seed must be >= 0.")
        derived_seed = None
        if params.use_seed:
            derived_seed = derived_structure_seed(seed, structure)

        min_defects = max_defects if count_mode == "fixed" else 1
        possible_outputs = 0
        for count in range(min_defects, max_defects + 1):
            possible_outputs += comb(n_atoms, count)
            if possible_outputs >= max_structures:
                possible_outputs = max_structures
                break

        return {
            "n_atoms": n_atoms,
            "engine_type": engine_type,
            "count_mode": count_mode,
            "min_defects": min_defects,
            "max_defects": max_defects,
            "max_structures": max_structures,
            "target_outputs": possible_outputs,
            "derived_seed": derived_seed,
        }

    @classmethod
    def sampling_summary(cls, structure, params: VacancyDefectParams) -> dict[str, Any]:
        """Validate settings and return resolved counts for UI previews."""
        settings = cls._validated_settings(structure, params)
        return {
            key: settings[key]
            for key in (
                "n_atoms",
                "engine_type",
                "count_mode",
                "min_defects",
                "max_defects",
                "target_outputs",
            )
        }

    @staticmethod
    def _append_unique(
        structure,
        defect_indices,
        seen_deletions: set[tuple[int, ...]],
        structure_list: list,
    ) -> bool:
        deletion_key = tuple(int(index) for index in np.sort(defect_indices))
        if deletion_key in seen_deletions:
            return False
        seen_deletions.add(deletion_key)

        new_structure = structure.copy()
        mask = np.zeros(len(structure), dtype=bool)
        mask[np.asarray(deletion_key, dtype=int)] = True
        del new_structure[mask]
        append_config_tag(new_structure, f"Vac(n={len(deletion_key)})")
        structure_list.append(new_structure)
        return True

    def run_structure(self, structure, params: VacancyDefectParams) -> list:
        settings = self._validated_settings(structure, params)
        n_atoms = settings["n_atoms"]
        target_outputs = settings["target_outputs"]
        max_defects = settings["max_defects"]
        fixed_count = settings["count_mode"] == "fixed"
        base_seed = settings["derived_seed"]

        structure_list = []
        seen_deletions: set[tuple[int, ...]] = set()
        if settings["engine_type"] == 0:
            sobol_engine = Sobol(d=n_atoms + 1, scramble=True, seed=base_seed)
            sobol_draws = 1 << (max(1, target_outputs * 2) - 1).bit_length()
            row_bytes = 8 * (n_atoms + 1)
            batch_size = max(
                1,
                min(256, self._SOBOL_BATCH_BYTES // max(1, row_bytes)),
            )
            generated = 0
            while generated < sobol_draws and len(structure_list) < target_outputs:
                current_batch = min(batch_size, sobol_draws - generated)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="The balance properties of Sobol",
                        category=UserWarning,
                    )
                    sobol_seq = sobol_engine.random(current_batch)
                generated += current_batch
                for point in sobol_seq:
                    if fixed_count:
                        target_defects = max_defects
                    else:
                        target_defects = 1 + int(point[0] * max_defects)
                        target_defects = min(target_defects, max_defects)
                    position_scores = point[1:]
                    defect_indices = np.argpartition(
                        position_scores,
                        target_defects - 1,
                    )[:target_defects]
                    self._append_unique(
                        structure,
                        defect_indices,
                        seen_deletions,
                        structure_list,
                    )
                    if len(structure_list) >= target_outputs:
                        break
        else:
            rng = np.random.default_rng(base_seed)
            max_attempts = max(
                100,
                target_outputs * self._MAX_ATTEMPTS_PER_OUTPUT,
            )
            for _ in range(max_attempts):
                if fixed_count:
                    target_defects = max_defects
                else:
                    target_defects = int(rng.integers(1, max_defects + 1))
                defect_indices = rng.choice(
                    n_atoms,
                    target_defects,
                    replace=False,
                )
                self._append_unique(
                    structure,
                    defect_indices,
                    seen_deletions,
                    structure_list,
                )
                if len(structure_list) >= target_outputs:
                    break
        return structure_list


@dataclass(frozen=True)
class StackingFaultParams:
    """Parameters for stacking-fault displacement generation."""

    hkl: Sequence[int] = (1, 1, 1)
    step: Sequence[float] = (0.0, 1.0, 0.5)
    layers: int = 1


class StackingFaultOperation(StructureOperation):
    """Generate displaced structures across a stacking-fault plane."""

    def run_structure(self, structure, params: StackingFaultParams) -> list:
        if len(params.hkl) != 3:
            raise ValueError("StackingFault hkl must contain exactly three integers.")
        if len(params.step) != 3:
            raise ValueError("StackingFault step must contain exactly three values: start, stop, step.")
        h, k, l = [int(value) for value in params.hkl]
        step_start, step_end, step_step = [float(value) for value in params.step]
        num_layers = int(params.layers)
        if num_layers <= 0:
            raise ValueError("StackingFault layers must be >= 1.")

        cell = structure.cell.array
        if len(structure) == 0:
            raise ValueError("StackingFault requires at least one atom.")
        recip = np.linalg.inv(cell).T
        normal = h * recip[0] + k * recip[1] + l * recip[2]
        if np.linalg.norm(normal) < 1e-8:
            return [structure.copy()]
        normal = normal / np.linalg.norm(normal)

        positions = structure.get_positions()
        basis = np.eye(3)
        non_parallel_vector = basis[int(np.argmin(np.abs(basis @ normal)))]
        slip_direction = np.cross(normal, non_parallel_vector)
        slip_direction = slip_direction / np.linalg.norm(slip_direction)

        coord = positions @ normal
        unique_coords = np.unique(np.round(coord, 8))
        unique_coords.sort()
        if num_layers >= len(unique_coords):
            plane_pos = unique_coords[len(unique_coords) // 2]
        else:
            plane_pos = unique_coords[num_layers - 1]
        mask = coord >= plane_pos

        structure_list = []
        for displacement in _range_values((step_start, step_end, step_step)):
            new_structure = structure.copy()
            pos = new_structure.positions.copy()
            pos[mask] += slip_direction * displacement
            new_structure.set_positions(wrapped_positions(new_structure, pos))
            append_config_tag(new_structure, f"SF(hkl={h}{k}{l},d={displacement:g})")
            structure_list.append(new_structure)
        return structure_list


@dataclass(frozen=True)
class StrictGSFEPathParams:
    """Parameters for unrelaxed GSFE path generation with explicit slip geometry."""

    plane_hkl: Sequence[int] = (0, 0, 1)
    slip_uvw: Sequence[int] = (1, 0, 0)
    displacement_range: Sequence[float] = (0.0, 1.0, 0.5)
    displacement_unit: str = "fraction_of_vector"
    cut_mode: str = "middle"
    cut_fraction: float = 0.5
    layer_index: int = 0
    wrap: bool = True


class StrictGSFEPathOperation(StructureOperation):
    """Generate unrelaxed generalized stacking-fault structures."""

    @classmethod
    def geometry_summary(
        cls,
        structure,
        params: StrictGSFEPathParams,
    ) -> dict[str, Any]:
        """Validate the geometry and return resolved values for UI previews."""
        settings = cls._validated_settings(structure, params)
        return {
            "atom_count": len(structure),
            "layer_count": settings["layer_count"],
            "stationary_count": int(np.count_nonzero(~settings["mask"])),
            "moved_count": int(np.count_nonzero(settings["mask"])),
            "slip_length": settings["slip_norm"],
            "output_count": len(settings["values"]),
            "cut_position": settings["cut_position"],
        }

    @classmethod
    def _validated_settings(
        cls,
        structure,
        params: StrictGSFEPathParams,
    ) -> dict[str, Any]:
        if len(structure) == 0:
            raise ValueError("StrictGSFEPath requires at least one atom.")
        cell = np.asarray(structure.cell.array, dtype=float)
        if (
            cell.shape != (3, 3)
            or not np.all(np.isfinite(cell))
            or abs(float(np.linalg.det(cell))) <= 1e-12
        ):
            raise ValueError("StrictGSFEPath requires a finite, nonsingular 3x3 cell.")

        hkl = cls._int_triplet(params.plane_hkl, "plane_hkl")
        uvw = cls._int_triplet(params.slip_uvw, "slip_uvw")
        if not np.any(hkl):
            raise ValueError("StrictGSFEPath plane_hkl must not be (0,0,0).")
        if not np.any(uvw):
            raise ValueError("StrictGSFEPath slip_uvw must not be (0,0,0).")

        normal = cls.plane_normal(cell, hkl)
        cls._validate_slab_oriented(cell, normal)
        slip = np.asarray(uvw, dtype=float) @ cell
        slip_norm = float(np.linalg.norm(slip))
        if slip_norm <= 1e-12:
            raise ValueError("StrictGSFEPath slip_uvw produced a zero vector.")
        normal_component = float(np.dot(slip, normal))
        if abs(normal_component) > 1e-8 * slip_norm:
            raise ValueError(
                "StrictGSFEPath slip_uvw must lie in the fault plane."
            )

        unit = str(params.displacement_unit)
        if unit == "fraction_of_vector":
            displacement_vector = slip
        elif unit == "angstrom":
            displacement_vector = slip / slip_norm
        else:
            raise ValueError("StrictGSFEPath displacement_unit must be fraction_of_vector or angstrom.")

        positions = np.asarray(structure.get_positions(), dtype=float)
        if positions.shape != (len(structure), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("StrictGSFEPath requires finite Cartesian atom positions.")
        coord = positions @ normal
        mask, cut_position, layer_count = cls._resolved_cut(coord, params)
        values = _range_values(params.displacement_range)
        return {
            "cell": cell,
            "hkl": hkl,
            "uvw": uvw,
            "normal": normal,
            "slip": slip,
            "slip_norm": slip_norm,
            "displacement_vector": displacement_vector,
            "positions": positions,
            "mask": mask,
            "values": values,
            "cut_position": cut_position,
            "layer_count": layer_count,
        }

    def run_structure(self, structure, params: StrictGSFEPathParams) -> list:
        settings = self._validated_settings(structure, params)
        hkl = settings["hkl"]
        uvw = settings["uvw"]
        positions = settings["positions"]
        mask = settings["mask"]
        displacement_vector = settings["displacement_vector"]

        out = []
        for value in settings["values"]:
            atoms = structure.copy()
            if abs(float(value)) > 1e-15:
                new_positions = positions.copy()
                new_positions[mask] += float(value) * displacement_vector
                if params.wrap:
                    new_positions = wrapped_positions(atoms, new_positions)
                atoms.set_positions(new_positions)
            tag = f"GSFE(hkl={self._tag_triplet(hkl)},uvw={self._tag_triplet(uvw)},d={float(value):g})"
            append_config_tag(atoms, tag)
            out.append(atoms)
        return out

    @staticmethod
    def _int_triplet(values: Sequence[int], label: str) -> tuple[int, int, int]:
        if len(values) != 3:
            raise ValueError(f"StrictGSFEPath {label} must contain exactly three integers.")
        resolved = []
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"StrictGSFEPath {label} must contain exactly three integers."
                ) from exc
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(
                    f"StrictGSFEPath {label} must contain exactly three integers."
                )
            resolved.append(int(numeric))
        return tuple(resolved)  # pyright: ignore[reportReturnType]

    @staticmethod
    def _tag_triplet(values: Sequence[int]) -> str:
        return "".join(str(int(value)) for value in values)

    @staticmethod
    def plane_normal(cell: np.ndarray, hkl: Sequence[int]) -> np.ndarray:
        recip = np.linalg.inv(np.asarray(cell, dtype=float)).T
        normal = np.asarray(hkl, dtype=float) @ recip
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError("StrictGSFEPath plane_hkl produced a zero normal.")
        return normal / norm

    @staticmethod
    def _validate_slab_oriented(cell: np.ndarray, normal: np.ndarray) -> None:
        c_axis = np.asarray(cell, dtype=float)[2]
        c_norm = float(np.linalg.norm(c_axis))
        if c_norm <= 1e-12:
            raise ValueError("StrictGSFEPath requires a nonzero third cell vector.")
        parallel_error = float(np.linalg.norm(np.cross(c_axis / c_norm, normal)))
        if parallel_error > 1e-6:
            raise ValueError(
                "StrictGSFEPath requires a slab-oriented cell: the third cell vector must be normal to plane_hkl."
            )

    @staticmethod
    def _resolved_cut(
        coord: np.ndarray,
        params: StrictGSFEPathParams,
    ) -> tuple[np.ndarray, float, int]:
        coord = np.asarray(coord, dtype=float)
        if coord.ndim != 1 or len(coord) == 0 or not np.all(np.isfinite(coord)):
            raise ValueError("StrictGSFEPath requires finite projected atom coordinates.")
        layers = np.unique(np.round(coord, 8))
        layers.sort()
        if len(layers) < 2:
            raise ValueError(
                "StrictGSFEPath requires atoms on at least two distinct planes."
            )

        mode = str(params.cut_mode).strip().lower()
        if mode == "middle":
            lower_index = (len(layers) - 1) // 2
            cut = float(0.5 * (layers[lower_index] + layers[lower_index + 1]))
        elif mode == "fractional":
            fraction = float(params.cut_fraction)
            if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
                raise ValueError("StrictGSFEPath cut_fraction must be between 0 and 1.")
            cut = float(coord.min() + fraction * (coord.max() - coord.min()))
        elif mode == "layer_index":
            index = StrictGSFEPathOperation._strict_integer(
                params.layer_index,
                "layer_index",
            )
            if index < 0 or index >= len(layers) - 1:
                raise ValueError("StrictGSFEPath layer_index must select a layer below the top layer.")
            cut = float(0.5 * (layers[index] + layers[index + 1]))
        else:
            raise ValueError("StrictGSFEPath cut_mode must be middle, fractional, or layer_index.")
        mask = coord > cut + 1e-10
        moved = int(np.count_nonzero(mask))
        if moved == 0 or moved == len(coord):
            raise ValueError(
                "StrictGSFEPath cut must leave atoms on both sides; adjust the cut position."
            )
        return mask, cut, len(layers)

    @staticmethod
    def _strict_integer(value: Any, label: str) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"StrictGSFEPath {label} must be an integer.") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"StrictGSFEPath {label} must be an integer.")
        return int(numeric)

    @classmethod
    def cut_mask(cls, coord: np.ndarray, params: StrictGSFEPathParams) -> np.ndarray:
        """Return the atoms above the resolved interlayer cut."""
        mask, _, _ = cls._resolved_cut(coord, params)
        return mask


@dataclass(frozen=True)
class RandomSlabParams:
    """Parameters for surface-slab enumeration."""

    h_range: Sequence[int] = (0, 1, 1)
    k_range: Sequence[int] = (0, 1, 1)
    l_range: Sequence[int] = (1, 3, 1)
    layer_range: Sequence[int] = (3, 6, 1)
    vacuum_range: Sequence[float] = (10.0, 10.0, 1.0)


class RandomSlabOperation(StructureOperation):
    """Construct slabs across Miller-index, layer, and vacuum ranges."""

    def run_structure(self, structure, params: RandomSlabParams) -> list:
        structure_list = []
        h_range = _range_values(params.h_range, include_step=True)
        k_range = _range_values(params.k_range, include_step=True)
        l_range = _range_values(params.l_range, include_step=True)
        layer_range = _range_values(params.layer_range, include_step=True)
        vac_range = _range_values(params.vacuum_range, include_step=True)

        for h in h_range:
            for k in k_range:
                for l in l_range:
                    if h == 0 and k == 0 and l == 0:
                        continue
                    for layers in layer_range:
                        for vac in vac_range:
                            try:
                                vacuum = None if vac == 0 else vac
                                slab = surface(
                                    structure,
                                    (int(h), int(k), int(l)),
                                    int(layers),
                                    vacuum=vacuum,
                                    periodic=True,
                                )
                                slab.set_positions(wrapped_positions(slab, slab.positions))
                                slab.info["Config_type"] = structure.info.get("Config_type", "")
                                append_config_tag(
                                    slab,
                                    f"Slab(hkl={int(h)}{int(k)}{int(l)},L={int(layers)},vac={vacuum})",
                                )
                                structure_list.append(slab)
                            except Exception as exc:
                                logger.error(f"Failed to build slab {(h, k, l)}: {exc}")
        return structure_list


@dataclass(frozen=True)
class InsertDefectParams:
    """Parameters for interstitial and adsorbate insertion."""

    mode: int = 0
    species: str = ""
    insert_count: int = 1
    structure_count: int = 10
    min_distance: float = 1.4
    max_attempts: int = 200
    use_seed: bool = False
    seed: int = 0
    axis: int = 2
    offset: float = 1.5


class InsertDefectOperation(StructureOperation):
    """Insert atoms as bulk interstitials or surface adsorbates."""

    @staticmethod
    def _integer(value: Any, *, label: str) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"InsertDefect: {label} must be an integer.") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"InsertDefect: {label} must be an integer.")
        return int(numeric)

    @classmethod
    def _positive_integer(cls, value: Any, *, label: str) -> int:
        integer = cls._integer(value, label=label)
        if integer <= 0:
            raise ValueError(f"InsertDefect: {label} must be >= 1.")
        return integer

    @classmethod
    def _validated_settings(
        cls,
        structure,
        params: InsertDefectParams,
    ) -> dict[str, Any]:
        if len(structure) == 0:
            raise ValueError("InsertDefect requires at least one host atom.")

        try:
            mode = cls._integer(params.mode, label="mode")
        except ValueError as exc:
            raise ValueError(
                "InsertDefect: mode must be 0 (Interstitial) or 1 (Adsorption)."
            ) from exc
        if mode not in {0, 1}:
            raise ValueError(
                "InsertDefect: mode must be 0 (Interstitial) or 1 (Adsorption)."
            )

        count = cls._positive_integer(
            params.insert_count,
            label="insert_count",
        )
        structure_count = cls._positive_integer(
            params.structure_count,
            label="structure_count",
        )
        max_attempts = cls._positive_integer(
            params.max_attempts,
            label="max_attempts",
        )

        try:
            min_distance = float(params.min_distance)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "InsertDefect: min_distance must be finite and positive."
            ) from exc
        if not np.isfinite(min_distance) or min_distance <= 0.0:
            raise ValueError(
                "InsertDefect: min_distance must be finite and positive."
            )

        cell = np.asarray(structure.cell.array, dtype=float)
        if (
            cell.shape != (3, 3)
            or not np.all(np.isfinite(cell))
            or abs(float(np.linalg.det(cell))) <= 1e-12
        ):
            raise ValueError(
                "InsertDefect requires a finite, non-singular 3x3 cell."
            )

        try:
            axis = cls._integer(params.axis, label="axis")
        except ValueError as exc:
            raise ValueError(
                "InsertDefect: axis must be 0, 1, or 2."
            ) from exc
        if mode == 1 and axis not in {0, 1, 2}:
            raise ValueError("InsertDefect: axis must be 0, 1, or 2.")

        try:
            offset = float(params.offset)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "InsertDefect: adsorption height must be finite and positive."
            ) from exc
        if mode == 1 and (not np.isfinite(offset) or offset <= 0.0):
            raise ValueError(
                "InsertDefect: adsorption height must be finite and positive."
            )

        species, weights = _parse_insert_species(params.species)

        seed = cls._integer(params.seed, label="seed")
        if params.use_seed and seed < 0:
            raise ValueError("InsertDefect: seed must be >= 0.")
        derived_seed = None
        if params.use_seed:
            derived_seed = derived_structure_seed(seed, structure)

        surface_top_fraction = None
        surface_normal = None
        if mode == 1:
            surface_top_fraction, surface_normal = cls._adsorption_geometry(
                structure.get_positions(),
                cell,
                axis,
            )

        return {
            "mode": mode,
            "count": count,
            "structure_count": structure_count,
            "min_distance": min_distance,
            "max_attempts": max_attempts,
            "species": species,
            "weights": weights,
            "axis": axis,
            "offset": offset,
            "derived_seed": derived_seed,
            "cell": cell,
            "surface_top_fraction": surface_top_fraction,
            "surface_normal": surface_normal,
        }

    @classmethod
    def sampling_summary(
        cls,
        structure,
        params: InsertDefectParams,
    ) -> dict[str, Any]:
        """Validate settings and return resolved values for UI previews."""
        settings = cls._validated_settings(structure, params)
        return {
            key: settings[key]
            for key in (
                "mode",
                "count",
                "structure_count",
                "min_distance",
                "species",
                "weights",
                "axis",
                "offset",
            )
        }

    def run_structure(self, structure, params: InsertDefectParams) -> list:
        settings = self._validated_settings(structure, params)
        mode = settings["mode"]
        count = settings["count"]
        max_structs = settings["structure_count"]
        min_distance = settings["min_distance"]
        max_attempts = settings["max_attempts"]
        species = settings["species"]
        weights = np.asarray(settings["weights"], dtype=float)
        axis = settings["axis"]
        offset = settings["offset"]

        rng = np.random.default_rng(settings["derived_seed"])
        base_positions = np.asarray(structure.get_positions(), dtype=float)
        cell = settings["cell"]
        pbc = np.asarray(structure.get_pbc(), dtype=bool)

        results = []
        for output_index in range(max_structs):
            new_structure = structure.copy()
            positions_reference = np.array(base_positions, dtype=float)

            for insert_index in range(count):
                success = False
                for _attempt in range(max_attempts):
                    if mode == 0:
                        candidate = self._sample_interstitial(cell, rng=rng)
                    else:
                        candidate = self._sample_adsorbate_from_surface(
                            cell,
                            axis,
                            settings["surface_top_fraction"],
                            settings["surface_normal"],
                            offset,
                            rng=rng,
                        )

                    nearest = self._nearest_distance(
                        candidate,
                        positions_reference,
                        cell=cell,
                        pbc=pbc,
                    )
                    if nearest < min_distance:
                        continue

                    element = str(rng.choice(species, p=weights))
                    new_structure.append(Atom(element, candidate))
                    positions_reference = np.vstack([positions_reference, candidate])
                    success = True
                    break

                if not success:
                    mode_name = "adsorption" if mode == 1 else "interstitial"
                    raise ValueError(
                        "InsertDefect: could not place "
                        f"atom {insert_index + 1} of {count} for output "
                        f"{output_index + 1} after {max_attempts} attempts "
                        f"({mode_name}); reduce the minimum distance or insertion count."
                    )

            mode_tag = "ad" if mode == 1 else "int"
            append_config_tag(new_structure, f"Ins({mode_tag},n={count})")
            results.append(new_structure)
        return results

    @staticmethod
    def _sample_interstitial(cell: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        frac = rng.random(3)
        return frac @ cell

    @staticmethod
    def _nearest_distance(candidate: np.ndarray, positions: np.ndarray, *, cell: np.ndarray, pbc: np.ndarray) -> float:
        pos = np.asarray(positions, dtype=float)
        if pos.size == 0:
            return float("inf")
        cell_arr = np.asarray(cell, dtype=float)
        pbc_arr = np.asarray(pbc, dtype=bool)
        offdiag = cell_arr.copy()
        np.fill_diagonal(offdiag, 0.0)
        if cell_arr.shape == (3, 3) and np.all(np.isfinite(cell_arr)) and np.allclose(offdiag, 0.0, atol=1e-12):
            lengths = np.diag(cell_arr)
            if np.all(np.abs(lengths[pbc_arr]) > 1e-12):
                delta = pos - np.asarray(candidate, dtype=float).reshape(1, 3)
                for axis in range(3):
                    if pbc_arr[axis]:
                        length = float(lengths[axis])
                        delta[:, axis] -= np.rint(delta[:, axis] / length) * length
                return float(np.sqrt(np.min(np.sum(delta * delta, axis=1))))

        _, dists = geometry.get_distances(candidate, pos, cell=cell_arr, pbc=pbc_arr)
        flat = np.asarray(dists, dtype=float).ravel()
        return float(np.min(flat)) if flat.size else float("inf")

    @staticmethod
    def _adsorption_geometry(
        positions: np.ndarray,
        cell: np.ndarray,
        axis: int,
    ) -> tuple[float, np.ndarray]:
        """Return the upper host plane and true normal for one lattice direction."""
        cell_array = np.asarray(cell, dtype=float)
        inverse_cell = np.linalg.inv(cell_array)
        scaled = np.asarray(positions, dtype=float) @ inverse_cell
        top_frac = scaled[:, axis].max()
        surface_normal = inverse_cell[:, axis]
        normal_length = np.linalg.norm(surface_normal)
        if normal_length <= 1e-12:
            raise ValueError(
                "InsertDefect could not determine the adsorption surface normal."
            )
        return float(top_frac), surface_normal / normal_length

    @staticmethod
    def _sample_adsorbate_from_surface(
        cell: np.ndarray,
        axis: int,
        top_fraction: float,
        surface_normal: np.ndarray,
        offset: float,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample a lateral point at a fixed height above the upper host plane."""
        frac = rng.random(3)
        frac[axis] = top_fraction
        in_plane = frac @ cell
        return in_plane + np.asarray(surface_normal, dtype=float) * offset

    @classmethod
    def _sample_adsorbate(
        cls,
        structure,
        positions: np.ndarray,
        cell: np.ndarray,
        axis: int,
        offset: float,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Compatibility wrapper for sampling above the supplied host positions."""
        del structure
        top_fraction, surface_normal = cls._adsorption_geometry(
            positions,
            cell,
            axis,
        )
        return cls._sample_adsorbate_from_surface(
            cell,
            axis,
            top_fraction,
            surface_normal,
            offset,
            rng=rng,
        )
