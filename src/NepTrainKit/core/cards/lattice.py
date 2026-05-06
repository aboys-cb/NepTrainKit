"""UI-independent lattice Make Dataset operations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from ase.build import make_supercell
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from scipy.stats.qmc import Sobol

from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.core.structure import get_clusters, process_organic_clusters

from .operation import StructureOperation


@dataclass(frozen=True)
class CellStrainParams:
    """Parameters for axial lattice strain generation."""

    axes: str = "uniaxial"
    x_range: tuple[float, float, float] = (-5.0, 5.0, 1.0)
    y_range: tuple[float, float, float] = (-5.0, 5.0, 1.0)
    z_range: tuple[float, float, float] = (-5.0, 5.0, 1.0)
    identify_organic: bool = False


class CellStrainOperation(StructureOperation):
    """Generate strained lattices from explicit parameters."""

    def run_structure(self, structure, params: CellStrainParams) -> list:
        structure_list = []
        axes = params.axes
        identify_organic = params.identify_organic

        if identify_organic:
            clusters, is_organic_list = get_clusters(structure)

        strain_range = [
            np.arange(start=params.x_range[0], stop=params.x_range[1] + 0.001, step=params.x_range[2]),
            np.arange(start=params.y_range[0], stop=params.y_range[1] + 0.001, step=params.y_range[2]),
            np.arange(start=params.z_range[0], stop=params.z_range[1] + 0.001, step=params.z_range[2]),
        ]
        cell = structure.get_cell()
        all_axes = [0, 1, 2]

        if axes == "isotropic":
            for strain in strain_range[0]:
                new_structure = structure.copy()
                new_cell = cell.copy() * (1 + strain / 100)
                new_structure.set_cell(new_cell, scale_atoms=True)
                if identify_organic:
                    process_organic_clusters(structure, new_structure, clusters, is_organic_list)

                strain_info = [f"all={strain:g}%"]
                append_config_tag(new_structure, f"Str({','.join(strain_info)})")
                structure_list.append(new_structure)
            return structure_list

        if axes == "uniaxial":
            axes_combinations = [[i] for i in all_axes]
        elif axes == "biaxial":
            axes_combinations = list(combinations(all_axes, 2))
        elif axes == "triaxial":
            axes_combinations = [all_axes]
        else:
            axes_combinations = [["XYZ".index(i.upper()) for i in axes if i.upper() in "XYZ"]]

        for ax_comb in axes_combinations:
            if len(ax_comb) == 0:
                continue
            strain_combinations = np.array(
                np.meshgrid(*[strain_range[index] for index in ax_comb])
            ).T.reshape(-1, len(ax_comb))
            for strain_vals in strain_combinations:
                new_structure = structure.copy()
                new_cell = cell.copy()
                for ax_idx, strain in zip(ax_comb, strain_vals):
                    new_cell[ax_idx] *= 1 + strain / 100
                new_structure.set_cell(new_cell, scale_atoms=True)
                if identify_organic:
                    process_organic_clusters(structure, new_structure, clusters, is_organic_list)

                strain_info = [f"{'XYZ'[ax]}={float(s):g}%" for ax, s in zip(ax_comb, strain_vals)]
                append_config_tag(new_structure, f"Str({','.join(strain_info)})")
                structure_list.append(new_structure)

        return structure_list


@dataclass(frozen=True)
class CellScalingParams:
    """Parameters for random lattice scaling perturbations."""

    engine_type: int = 1
    max_scaling: float = 0.04
    max_num: int = 50
    perturb_angle: bool = True
    identify_organic: bool = False
    use_seed: bool = False
    seed: int = 0


class CellScalingOperation(StructureOperation):
    """Generate stochastic lattice perturbations without Qt widget state."""

    def run_structure(self, structure, params: CellScalingParams) -> list:
        structure_list = []
        base_seed = int(params.seed) if params.use_seed else None
        rng = np.random.default_rng(base_seed)
        dim = 6 if params.perturb_angle else 3

        if params.engine_type == 0:
            sobol_engine = Sobol(d=dim, scramble=True, seed=base_seed)
            sobol_seq = sobol_engine.random(int(params.max_num))
            perturbation_factors = 1 + (sobol_seq - 0.5) * 2 * float(params.max_scaling)
        else:
            perturbation_factors = 1 + rng.uniform(
                -float(params.max_scaling),
                float(params.max_scaling),
                (int(params.max_num), dim),
            )

        orig_lattice = structure.cell.array
        orig_lengths = np.linalg.norm(orig_lattice, axis=1)
        unit_vectors = orig_lattice / orig_lengths[:, np.newaxis]

        if params.identify_organic:
            clusters, is_organic_list = get_clusters(structure)

        for i in range(int(params.max_num)):
            new_structure = structure.copy()
            length_factors = perturbation_factors[i, :3]
            new_lengths = orig_lengths * length_factors
            new_lattice = unit_vectors * new_lengths[:, np.newaxis]

            if params.perturb_angle:
                angle_factors = perturbation_factors[i, 3:]
                angles = np.arccos(
                    [
                        np.dot(orig_lattice[1], orig_lattice[2]) / (orig_lengths[1] * orig_lengths[2]),
                        np.dot(orig_lattice[0], orig_lattice[2]) / (orig_lengths[0] * orig_lengths[2]),
                        np.dot(orig_lattice[0], orig_lattice[1]) / (orig_lengths[0] * orig_lengths[1]),
                    ]
                )
                new_angles = angles * angle_factors
                new_lattice = np.zeros((3, 3), dtype=np.float32)
                new_lattice[0] = [new_lengths[0], 0, 0]
                new_lattice[1] = [
                    new_lengths[1] * np.cos(new_angles[2]),
                    new_lengths[1] * np.sin(new_angles[2]),
                    0,
                ]
                cx = new_lengths[2] * np.cos(new_angles[1])
                cy = new_lengths[2] * (
                    np.cos(new_angles[0]) - np.cos(new_angles[1]) * np.cos(new_angles[2])
                ) / np.sin(new_angles[2])
                cz = np.sqrt(max(new_lengths[2] ** 2 - cx ** 2 - cy ** 2, 0))
                new_lattice[2] = [cx, cy, cz]

            eng = "U" if params.engine_type == 1 else "S"
            append_config_tag(new_structure, f"LSc(max={params.max_scaling},{eng})")
            new_structure.set_cell(new_lattice, scale_atoms=True)
            if params.identify_organic:
                process_organic_clusters(structure, new_structure, clusters, is_organic_list)

            structure_list.append(new_structure)
        return structure_list


@dataclass(frozen=True)
class ShearMatrixParams:
    """Parameters for shear-matrix strain generation."""

    xy_range: tuple[float, float, float] = (-5.0, 5.0, 1.0)
    yz_range: tuple[float, float, float] = (-5.0, 5.0, 1.0)
    xz_range: tuple[float, float, float] = (-5.0, 5.0, 1.0)
    symmetric: bool = True
    identify_organic: bool = False


class ShearMatrixOperation(StructureOperation):
    """Apply shear matrices from explicit parameters."""

    def run_structure(self, structure, params: ShearMatrixParams) -> list:
        structure_list = []
        if params.identify_organic:
            clusters, is_organic_list = get_clusters(structure)

        xy_range = np.arange(params.xy_range[0], params.xy_range[1] + 0.001, params.xy_range[2])
        yz_range = np.arange(params.yz_range[0], params.yz_range[1] + 0.001, params.yz_range[2])
        xz_range = np.arange(params.xz_range[0], params.xz_range[1] + 0.001, params.xz_range[2])
        cell = structure.get_cell()

        for sxy in xy_range:
            for syz in yz_range:
                for sxz in xz_range:
                    new_structure = structure.copy()
                    shear_matrix = np.eye(3)
                    shear_matrix[0, 1] += sxy / 100
                    shear_matrix[1, 2] += syz / 100
                    shear_matrix[0, 2] += sxz / 100
                    if params.symmetric:
                        shear_matrix[1, 0] += sxy / 100
                        shear_matrix[2, 1] += syz / 100
                        shear_matrix[2, 0] += sxz / 100

                    new_structure.set_cell(np.matmul(cell, shear_matrix), scale_atoms=True)
                    if params.identify_organic:
                        process_organic_clusters(structure, new_structure, clusters, is_organic_list)

                    info_list = []
                    if abs(sxy) > 1e-8:
                        info_list.append(f"xy={sxy:g}%")
                    if abs(syz) > 1e-8:
                        info_list.append(f"yz={syz:g}%")
                    if abs(sxz) > 1e-8:
                        info_list.append(f"xz={sxz:g}%")
                    info_str = ",".join(info_list)
                    append_config_tag(new_structure, f"Shr({info_str},sym={int(bool(params.symmetric))})")
                    structure_list.append(new_structure)
        return structure_list


@dataclass(frozen=True)
class ShearAngleParams:
    """Parameters for lattice angle perturbations."""

    alpha_range: tuple[float, float, float] = (-2.0, 2.0, 1.0)
    beta_range: tuple[float, float, float] = (-2.0, 2.0, 1.0)
    gamma_range: tuple[float, float, float] = (-2.0, 2.0, 1.0)
    identify_organic: bool = False


class ShearAngleOperation(StructureOperation):
    """Perturb lattice angles while preserving cell lengths."""

    def run_structure(self, structure, params: ShearAngleParams) -> list:
        structure_list = []
        if params.identify_organic:
            clusters, is_organic_list = get_clusters(structure)

        alpha_range = np.arange(params.alpha_range[0], params.alpha_range[1] + 0.001, params.alpha_range[2])
        beta_range = np.arange(params.beta_range[0], params.beta_range[1] + 0.001, params.beta_range[2])
        gamma_range = np.arange(params.gamma_range[0], params.gamma_range[1] + 0.001, params.gamma_range[2])
        cellpar = cell_to_cellpar(structure.get_cell())
        lengths = cellpar[:3]
        angles0 = cellpar[3:]

        for da in alpha_range:
            for db in beta_range:
                for dg in gamma_range:
                    new_structure = structure.copy()
                    new_angles = angles0 + np.array([da, db, dg])
                    new_lattice = cellpar_to_cell([*lengths, *new_angles])
                    new_structure.set_cell(new_lattice, scale_atoms=True)
                    if params.identify_organic:
                        process_organic_clusters(structure, new_structure, clusters, is_organic_list)

                    info_list = []
                    if abs(da) > 1e-8:
                        info_list.append(f"a={da:g}")
                    if abs(db) > 1e-8:
                        info_list.append(f"b={db:g}")
                    if abs(dg) > 1e-8:
                        info_list.append(f"g={dg:g}")
                    info_str = ",".join(info_list)
                    append_config_tag(new_structure, f"Ang({info_str})")
                    structure_list.append(new_structure)
        return structure_list


@dataclass(frozen=True)
class PerturbParams:
    """Parameters for random atomic perturbations."""

    engine_type: int = 1
    max_distance: float = 0.3
    max_num: int = 50
    identify_organic: bool = False
    use_element_scaling: bool = False
    element_scalings: dict[str, float] | None = None
    use_seed: bool = False
    seed: int = 0


class PerturbOperation(StructureOperation):
    """Apply random atomic displacements from explicit parameters."""

    def run_structure(self, structure, params: PerturbParams) -> list:
        structure_list = []
        n_atoms = len(structure)
        dim = n_atoms * 3
        symbols = structure.get_chemical_symbols()
        element_scalings = params.element_scalings or {}
        per_atom_scaling = (
            np.array([element_scalings.get(sym, params.max_distance) for sym in symbols])
            if params.use_element_scaling
            else np.full(n_atoms, params.max_distance)
        )

        base_seed = int(params.seed) if params.use_seed else None
        rng = np.random.default_rng(base_seed)

        if params.engine_type == 0:
            sobol_engine = Sobol(d=dim, scramble=True, seed=base_seed)
            perturbation_factors = (sobol_engine.random(int(params.max_num)) - 0.5) * 2
        else:
            perturbation_factors = rng.uniform(-1, 1, (int(params.max_num), dim))

        if params.identify_organic:
            clusters, is_organic_list = get_clusters(structure)
            organic_clusters = [cluster for cluster, is_org in zip(clusters, is_organic_list) if is_org]
            inorganic_clusters = [cluster for cluster, is_org in zip(clusters, is_organic_list) if not is_org]

        orig_positions = structure.positions
        for i in range(int(params.max_num)):
            delta = perturbation_factors[i].reshape(n_atoms, 3) * per_atom_scaling[:, None]
            new_positions = orig_positions + delta

            if params.identify_organic:
                new_positions = orig_positions.copy()
                for cluster in organic_clusters:
                    cluster_delta = delta[cluster[0]]
                    new_positions[cluster] += cluster_delta
                for cluster in inorganic_clusters:
                    new_positions[cluster] += delta[cluster]

            new_structure = structure.copy()
            new_structure.set_positions(new_positions)
            new_structure.wrap()
            eng = "U" if params.engine_type == 1 else "S"
            append_config_tag(new_structure, f"Pert(d={params.max_distance},{eng})")
            structure_list.append(new_structure)

        return structure_list


SuperCellMode = Literal["scale", "cell", "max_atoms"]


@dataclass(frozen=True)
class SuperCellParams:
    """Parameters for supercell generation."""

    behavior_type: int = 0
    mode: SuperCellMode = "scale"
    super_scale: tuple[int, int, int] = (3, 3, 3)
    target_cell: tuple[float, float, float] = (20.0, 20.0, 20.0)
    max_atoms: int = 100
    fixed_axis_flags: tuple[bool, bool, bool] = (False, False, False)
    fixed_axis_scale: tuple[int, int, int] = (1, 1, 1)


class SuperCellOperation(StructureOperation):
    """Create supercells without depending on Qt widget state."""

    def run_structure(self, structure, params: SuperCellParams) -> list:
        if params.mode == "scale":
            expansion_factors = self._get_scale_factors(params)
        elif params.mode == "cell":
            expansion_factors = self._get_cell_factors(structure, params)
        elif params.mode == "max_atoms":
            expansion_factors = self._get_max_atoms_factors(structure, params)
        else:
            expansion_factors = [(1, 1, 1)]

        expansion_factors = self._dedupe_factors(expansion_factors, params)
        return self._generate_structures(structure, expansion_factors, params)

    def _apply_fixed_axes(
        self,
        scale_factors: tuple[int, int, int],
        params: SuperCellParams,
    ) -> tuple[int, int, int]:
        return tuple(
            int(params.fixed_axis_scale[i]) if params.fixed_axis_flags[i] else max(int(scale_factors[i]), 1)
            for i in range(3)
        )

    def _dedupe_factors(
        self,
        expansion_factors: list[tuple[int, int, int]],
        params: SuperCellParams,
    ) -> list[tuple[int, int, int]]:
        unique_factors = []
        seen = set()
        for scale_factors in expansion_factors:
            adjusted = self._apply_fixed_axes(scale_factors, params)
            if adjusted in seen:
                continue
            seen.add(adjusted)
            unique_factors.append(adjusted)
        return unique_factors

    def _get_iteration_axis_values(
        self,
        scale_factors: tuple[int, int, int],
        params: SuperCellParams,
    ) -> tuple[list[int], list[int], list[int]]:
        axis_values = []
        for axis, limit in enumerate(scale_factors):
            if params.fixed_axis_flags[axis]:
                axis_values.append([int(params.fixed_axis_scale[axis])])
            else:
                axis_values.append(list(range(1, max(int(limit), 1) + 1)))
        return axis_values[0], axis_values[1], axis_values[2]

    def _get_scale_factors(self, params: SuperCellParams) -> list[tuple[int, int, int]]:
        na, nb, nc = params.super_scale
        return [self._apply_fixed_axes((int(na), int(nb), int(nc)), params)]

    def _get_cell_factors(self, structure, params: SuperCellParams) -> list[tuple[int, int, int]]:
        target_a, target_b, target_c = params.target_cell
        lattice = structure.cell.array
        a_len = np.linalg.norm(lattice[0])
        b_len = np.linalg.norm(lattice[1])
        c_len = np.linalg.norm(lattice[2])

        if params.behavior_type == 2:
            na = self._fixed_or_minimum_factor(0, target_a, a_len, params)
            nb = self._fixed_or_minimum_factor(1, target_b, b_len, params)
            nc = self._fixed_or_minimum_factor(2, target_c, c_len, params)
        else:
            na = self._fixed_or_maximum_factor(0, target_a, a_len, params)
            nb = self._fixed_or_maximum_factor(1, target_b, b_len, params)
            nc = self._fixed_or_maximum_factor(2, target_c, c_len, params)

        return [(max(na, 1), max(nb, 1), max(nc, 1))]

    def _fixed_or_minimum_factor(
        self,
        axis: int,
        target: float,
        length: float,
        params: SuperCellParams,
    ) -> int:
        if params.fixed_axis_flags[axis]:
            return int(params.fixed_axis_scale[axis])
        return int(target / length) + 1 if length > 0 else 1

    def _fixed_or_maximum_factor(
        self,
        axis: int,
        target: float,
        length: float,
        params: SuperCellParams,
    ) -> int:
        if params.fixed_axis_flags[axis]:
            return int(params.fixed_axis_scale[axis])
        value = max(int(target / length) if length > 0 else 0, 1)
        if value * length > target and value > 1:
            value -= 1
        return value

    def _get_max_atoms_factors(self, structure, params: SuperCellParams) -> list[tuple[int, int, int]]:
        num_atoms_orig = len(structure)
        if num_atoms_orig <= 0:
            return []

        max_n = max(int(params.max_atoms / num_atoms_orig), 1)
        axis_ranges = []
        for axis, is_fixed in enumerate(params.fixed_axis_flags):
            if is_fixed:
                fixed = int(params.fixed_axis_scale[axis])
                axis_ranges.append(range(fixed, fixed + 1))
            else:
                axis_ranges.append(range(1, max_n + 1))

        expansion_factors = []
        for na in axis_ranges[0]:
            for nb in axis_ranges[1]:
                for nc in axis_ranges[2]:
                    total_atoms = num_atoms_orig * na * nb * nc
                    if total_atoms <= params.max_atoms:
                        expansion_factors.append((na, nb, nc))

        expansion_factors.sort(key=lambda value: num_atoms_orig * value[0] * value[1] * value[2])
        return expansion_factors

    def _generate_structures(
        self,
        structure,
        expansion_factors: list[tuple[int, int, int]],
        params: SuperCellParams,
    ) -> list:
        if not expansion_factors:
            return [structure.copy()]

        structure_list = []
        if params.behavior_type == 0:
            na, nb, nc = expansion_factors[-1]
            if na == 1 and nb == 1 and nc == 1:
                return [structure.copy()]
            structure_list.append(self._make_supercell(structure, na, nb, nc))

        elif params.behavior_type == 1:
            if params.mode == "max_atoms":
                for na, nb, nc in expansion_factors:
                    if na == 1 and nb == 1 and nc == 1:
                        supercell = structure.copy()
                    else:
                        supercell = self._make_supercell(structure, na, nb, nc)
                    structure_list.append(supercell)
            else:
                na, nb, nc = expansion_factors[0]
                a_values, b_values, c_values = self._get_iteration_axis_values((na, nb, nc), params)
                for i in a_values:
                    for j in b_values:
                        for k in c_values:
                            if i == 1 and j == 1 and k == 1:
                                supercell = structure.copy()
                            else:
                                supercell = self._make_supercell(structure, i, j, k)
                            structure_list.append(supercell)

        elif params.behavior_type == 2:
            na, nb, nc = expansion_factors[0]
            if na == 1 and nb == 1 and nc == 1:
                return [structure.copy()]
            structure_list.append(self._make_supercell(structure, na, nb, nc))

        return structure_list

    def _make_supercell(self, structure, na: int, nb: int, nc: int):
        supercell = make_supercell(structure, np.diag([na, nb, nc]), order="atom-major")
        supercell.info["Config_type"] = structure.info.get("Config_type", "")
        append_config_tag(supercell, f"SC({na}x{nb}x{nc})")
        return supercell
