"""UI-independent defect and surface Make Dataset operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from ase import Atom
from ase.build import surface
from ase.geometry import geometry
from loguru import logger
from scipy.stats.qmc import Sobol

from NepTrainKit.core.config_type import append_config_tag

from .operation import StructureOperation


def _range_values(values: Sequence[float], *, include_step: bool = False) -> np.ndarray:
    start, end, step = values
    if include_step:
        return np.arange(start, end + step, step)
    return np.arange(start, end + step / 2, step)


def _parse_species(tokens: str, fallback: Sequence[str]) -> tuple[list[str], list[float]]:
    """Parse comma-separated ``Element[:weight]`` entries."""
    species: list[str] = []
    weights: list[float] = []
    for token in tokens.split(","):
        item = token.strip()
        if not item:
            continue
        if ":" in item:
            symbol, weight_str = item.split(":", 1)
            symbol = symbol.strip()
            try:
                weight = float(weight_str.strip())
            except ValueError:
                weight = 1.0
        else:
            symbol = item
            weight = 1.0
        if symbol:
            species.append(symbol)
            weights.append(weight)

    if not species:
        default = list(dict.fromkeys(fallback))
        species = default or ["H"]
        weights = [1.0] * len(species)

    total = sum(weights)
    if total <= 0:
        weights = [1.0 / len(species)] * len(species)
    else:
        weights = [weight / total for weight in weights]
    return species, weights


@dataclass(frozen=True)
class RandomVacancyParams:
    """Parameters for rule-based random vacancy generation."""

    rules: list[dict[str, Any]] = field(default_factory=list)
    max_structures: int = 1
    use_seed: bool = False
    seed: int = 0


class RandomVacancyOperation(StructureOperation):
    """Create vacancy structures by probabilistically removing matched atoms."""

    def run_structure(self, structure, params: RandomVacancyParams) -> list:
        if not isinstance(params.rules, list) or not params.rules:
            return [structure]

        base_seed = int(params.seed) if params.use_seed else None
        rng = np.random.default_rng(base_seed)
        structure_list = []
        for _ in range(int(params.max_structures)):
            new_structure = structure.copy()
            total_remove = 0
            for rule in params.rules:
                element = rule.get("element")
                count_values = list(rule.get("count", [0, 0]))
                if not count_values:
                    continue
                count_min = int(count_values[0])
                count_max = int(count_values[-1])
                if not element or int(count_max) <= 0:
                    continue

                groups = rule.get("group")
                if groups and "group" in new_structure.arrays:
                    candidate_indices = [
                        i
                        for i, atom, group in zip(
                            range(len(new_structure)),
                            new_structure,
                            new_structure.arrays["group"],
                        )
                        if atom.symbol == element and group in groups
                    ]
                else:
                    candidate_indices = [i for i, atom in enumerate(new_structure) if atom.symbol == element]

                if not candidate_indices:
                    continue

                count_mode = str(rule.get("count_mode", "")).lower()
                if count_mode == "fixed" or (not count_mode and count_min == count_max):
                    remove_num = count_min
                else:
                    if count_max < count_min:
                        count_min, count_max = count_max, count_min
                    remove_num = int(rng.integers(count_min, count_max + 1))
                remove_num = min(remove_num, len(candidate_indices))
                if remove_num <= 0:
                    continue

                idxs = rng.choice(candidate_indices, remove_num, replace=False)
                for idx in sorted(idxs, reverse=True):
                    del new_structure[idx]
                total_remove += remove_num

            if total_remove:
                append_config_tag(new_structure, f"Vac(n={total_remove})")
            structure_list.append(new_structure)
        return structure_list


@dataclass(frozen=True)
class VacancyDefectParams:
    """Parameters for stochastic vacancy-defect sampling."""

    engine_type: int = 1
    num_condition: int = 1
    use_num: bool = True
    concentration_condition: float = 0.0
    count_mode: str = "fixed"
    max_structures: int = 1
    use_seed: bool = False
    seed: int = 0


class VacancyDefectOperation(StructureOperation):
    """Sample random vacancy defects by count or concentration."""

    def run_structure(self, structure, params: VacancyDefectParams) -> list:
        structure_list = []
        base_seed = int(params.seed) if params.use_seed else None
        rng = np.random.default_rng(base_seed)
        n_atoms = len(structure)

        if params.use_num:
            max_defects = int(params.num_condition)
        else:
            max_defects = int(float(params.concentration_condition) * n_atoms)
        if max_defects == n_atoms:
            max_defects -= 1
        if max_defects <= 0:
            raise ValueError("Vacancy defect settings must allow at least one vacancy.")

        max_num = int(params.max_structures)
        fixed_count = str(params.count_mode).lower() == "fixed"
        if int(params.engine_type) == 0:
            sobol_engine = Sobol(d=n_atoms + 1, scramble=True, seed=base_seed)
            sobol_seq = sobol_engine.random(max_num)
        elif fixed_count:
            defect_counts = np.full(max_num, max_defects, dtype=int)
        else:
            defect_counts = rng.integers(1, max_defects + 1, size=max_num)

        for i in range(max_num):
            new_structure = structure.copy()
            if int(params.engine_type) == 0:
                if fixed_count:
                    target_defects = max_defects
                else:
                    target_defects = 1 + int(sobol_seq[i, 0] * max_defects)
                    target_defects = int(min(target_defects, max_defects))
                position_scores = sobol_seq[i, 1:]
                defect_indices = np.argsort(position_scores)[:target_defects]
            else:
                target_defects = int(defect_counts[i])
                defect_indices = rng.choice(n_atoms, target_defects, replace=False)

            if target_defects == 0:
                structure_list.append(new_structure)
                continue

            mask = np.zeros(n_atoms, dtype=bool)
            mask[defect_indices] = True
            n_vacancies = np.sum(mask)
            del new_structure[mask]
            append_config_tag(new_structure, f"Vac(n={int(n_vacancies)})")
            structure_list.append(new_structure)
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
        h, k, l = [int(value) for value in params.hkl]
        step_start, step_end, step_step = [float(value) for value in params.step]
        num_layers = int(params.layers)

        cell = structure.cell.array
        recip = np.linalg.inv(cell).T
        normal = h * recip[0] + k * recip[1] + l * recip[2]
        if np.linalg.norm(normal) < 1e-8:
            return [structure]
        normal = normal / np.linalg.norm(normal)

        positions = structure.get_positions()
        non_parallel_vector = np.array([1, 0, 0]) if normal[0] != 1 else np.array([0, 1, 0])
        perpendicular_vector = np.cross(normal, non_parallel_vector)
        perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)

        coord = positions @ perpendicular_vector
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
            pos[mask] += normal * displacement
            new_structure.set_positions(pos)
            new_structure.wrap()
            append_config_tag(new_structure, f"SF(hkl={h}{k}{l},d={displacement:g})")
            structure_list.append(new_structure)
        return structure_list


@dataclass(frozen=True)
class RandomSlabParams:
    """Parameters for surface-slab enumeration."""

    h_range: Sequence[int] = (0, 1, 1)
    k_range: Sequence[int] = (0, 1, 1)
    l_range: Sequence[int] = (1, 3, 1)
    layer_range: Sequence[int] = (3, 6, 1)
    vacuum_range: Sequence[int] = (10, 10, 1)


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
                                slab.wrap()
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

    def run_structure(self, structure, params: InsertDefectParams) -> list:
        count = int(params.insert_count)
        max_structs = int(params.structure_count)
        min_distance = float(params.min_distance)
        max_attempts = int(params.max_attempts)
        species, weights = _parse_species(params.species, structure.get_chemical_symbols())
        axis = int(params.axis)
        offset = float(params.offset)

        base_seed = int(params.seed) if params.use_seed else None
        rng = np.random.default_rng(base_seed)
        base_positions = structure.get_positions()
        cell = structure.cell.array
        pbc = structure.get_pbc()

        results = []
        for _ in range(max_structs):
            new_structure = structure.copy()
            positions_reference = np.array(base_positions, dtype=float)
            inserted = 0

            for _ in range(count):
                success = False
                for _attempt in range(max_attempts):
                    if int(params.mode) == 0:
                        candidate = self._sample_interstitial(cell, rng=rng)
                    else:
                        candidate = self._sample_adsorbate(structure, positions_reference, cell, axis, offset, rng=rng)

                    if candidate is None:
                        continue

                    _, dists = geometry.get_distances(candidate, positions_reference, cell=cell, pbc=pbc)
                    if len(dists.ravel()) and np.min(dists) < max(min_distance, 0.0):
                        continue

                    element = str(rng.choice(species, p=np.array(weights, dtype=float)))
                    new_structure.append(Atom(element, candidate))
                    positions_reference = np.vstack([positions_reference, candidate])
                    inserted += 1
                    success = True
                    break

                if not success:
                    logger.warning(
                        "InsertDefectOperation: failed to place atom after "
                        f'{max_attempts} attempts (mode={"adsorption" if int(params.mode) == 1 else "interstitial"})'
                    )
                    break

            if inserted:
                mode_tag = "ad" if int(params.mode) == 1 else "int"
                append_config_tag(new_structure, f"Ins({mode_tag},n={inserted})")
            new_structure.wrap()
            results.append(new_structure)
        return results

    @staticmethod
    def _sample_interstitial(cell: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        frac = rng.random(3)
        return frac @ cell

    @staticmethod
    def _sample_adsorbate(
        structure,
        positions: np.ndarray,
        cell: np.ndarray,
        axis: int,
        offset: float,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        if positions.size == 0:
            return None

        scaled = structure.get_scaled_positions(wrap=False)
        top_frac = scaled[:, axis].max()
        frac = rng.random(3)
        frac[axis] = top_frac
        in_plane = frac @ cell

        axis_vec = cell[axis]
        norm = np.linalg.norm(axis_vec)
        if norm < 1e-8:
            return None
        direction = axis_vec / norm
        return in_plane + direction * offset
