"""UI-independent lattice Make Dataset operations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from ase.build import make_supercell
from ase.geometry import cell_to_cellpar
from scipy.stats.qmc import Sobol

from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.core.structure import get_clusters, process_organic_clusters

from .errors import CardOperationError
from .geometry import wrapped_positions as fast_wrapped_positions
from .operation import StructureOperation
from .sampling import derived_structure_seed


def _scan_values(values, *, label: str) -> np.ndarray:
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three values: start, stop, step.")
    start, stop, step = [float(value) for value in values]
    if not np.all(np.isfinite([start, stop, step])):
        raise ValueError(f"{label} values must be finite.")
    if step <= 0.0:
        raise ValueError(f"{label} step must be positive.")
    if stop < start:
        start, stop = stop, start
    return np.arange(start, stop + step * 0.5, step, dtype=float)


def _cell_from_lengths_angles(lengths, angles_deg) -> np.ndarray:
    """Build a cell from lengths and degree angles using ASE's default convention."""
    a, b, c = [float(value) for value in lengths]
    alpha, beta, gamma = np.radians(np.asarray(angles_deg, dtype=float))
    sin_gamma = np.sin(gamma)
    if abs(float(sin_gamma)) <= 1e-12:
        raise ValueError("ShearAngle produced a singular gamma angle.")

    cell = np.zeros((3, 3), dtype=float)
    cell[0] = [a, 0.0, 0.0]
    cell[1] = [b * np.cos(gamma), b * sin_gamma, 0.0]
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    cell[2] = [cx, cy, cz]
    return cell


@dataclass(frozen=True)
class CellStrainParams:
    """Parameters for axial lattice strain generation."""

    axes: str = "uniaxial"
    x_range: tuple[float, float, float] = (-5.0, 5.0, 1.0)
    y_range: tuple[float, float, float] = (-5.0, 5.0, 1.0)
    z_range: tuple[float, float, float] = (-5.0, 5.0, 1.0)
    identify_organic: bool = False


@dataclass(frozen=True)
class BainPathParams:
    """Parameters for Bain/tetragonal distortion path generation."""

    axis: Literal["x", "y", "z"] = "z"
    ca_range: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mode: Literal["constant_volume", "scale_volume", "free_c"] = "constant_volume"
    volume_scale_range: tuple[float, float, float] = (1.0, 1.0, 1.0)
    scale_atoms: bool = True


class BainPathOperation(StructureOperation):
    """Generate fixed-structure Bain/tetragonal distortion structures."""

    def run_structure(self, structure, params: BainPathParams) -> list:
        axis_map = {"x": 0, "y": 1, "z": 2}
        axis = str(params.axis).lower()
        if axis not in axis_map:
            raise ValueError("BainPath axis must be x, y, or z.")
        if params.mode not in {"constant_volume", "scale_volume", "free_c"}:
            raise ValueError("BainPath mode must be constant_volume, scale_volume, or free_c.")

        cell = np.asarray(structure.cell.array, dtype=float)
        if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) <= 1e-12:
            raise ValueError("BainPath requires a nonsingular 3x3 cell.")

        ca_values = _scan_values(params.ca_range, label="ca_range")
        if np.any(ca_values <= 0.0):
            raise ValueError("BainPath ca_range values must be positive.")
        volume_values = (
            _scan_values(params.volume_scale_range, label="volume_scale_range")
            if params.mode == "scale_volume"
            else np.array([1.0], dtype=float)
        )
        if np.any(volume_values <= 0.0):
            raise ValueError("BainPath volume_scale_range values must be positive.")

        c_axis = axis_map[axis]
        out = []
        base_volume = abs(float(np.linalg.det(cell)))
        for r in ca_values:
            factors = np.ones(3, dtype=float)
            factors[c_axis] = float(r)
            if params.mode != "free_c":
                for i in range(3):
                    if i != c_axis:
                        factors[i] = 1.0 / np.sqrt(float(r))
            for vscale in volume_values:
                new_cell = cell * factors[:, None]
                if params.mode == "scale_volume":
                    new_cell = new_cell * (float(vscale) ** (1.0 / 3.0))
                atoms = structure.copy()
                atoms.set_cell(new_cell, scale_atoms=bool(params.scale_atoms))
                v_over_v0 = abs(float(np.linalg.det(new_cell))) / base_volume
                append_config_tag(atoms, f"Bain(ax={axis},ca={float(r):g},V={v_over_v0:g},mode={params.mode})")
                out.append(atoms)
        return out


class CellStrainOperation(StructureOperation):
    """Generate strained lattices from explicit parameters."""

    def run_structure(self, structure, params: CellStrainParams) -> list:
        structure_list = []
        axes = str(params.axes).strip()
        named_modes = {"isotropic", "uniaxial", "biaxial", "triaxial"}
        if axes not in named_modes:
            custom_axes = axes.upper()
            if not custom_axes or any(axis not in "XYZ" for axis in custom_axes):
                raise ValueError(
                    "CellStrain axes must be isotropic, uniaxial, biaxial, "
                    "triaxial, or a nonempty combination of X/Y/Z."
                )
            axes = custom_axes
        identify_organic = params.identify_organic

        if identify_organic:
            clusters, is_organic_list = get_clusters(structure)

        strain_range = [
            _scan_values(params.x_range, label="x_range"),
            _scan_values(params.y_range, label="y_range"),
            _scan_values(params.z_range, label="z_range"),
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
        max_num = int(params.max_num)
        if max_num <= 0:
            raise ValueError("CellScaling: max_num must be >= 1.")
        engine_type = int(params.engine_type)
        if engine_type not in {0, 1}:
            raise ValueError("CellScaling: engine_type must be 0 (Sobol) or 1 (Uniform).")
        max_scaling = float(params.max_scaling)
        if not np.isfinite(max_scaling) or max_scaling < 0.0:
            raise ValueError("CellScaling: max_scaling must be finite and non-negative.")
        base_seed = (
            derived_structure_seed(int(params.seed), structure)
            if params.use_seed
            else None
        )
        rng = np.random.default_rng(base_seed)
        dim = 6 if params.perturb_angle else 3

        if engine_type == 0:
            sobol_engine = Sobol(d=dim, scramble=True, seed=base_seed)
            sobol_seq = sobol_engine.random(max_num)
            perturbation_factors = 1 + (sobol_seq - 0.5) * 2 * max_scaling
        else:
            perturbation_factors = 1 + rng.uniform(
                -max_scaling,
                max_scaling,
                (max_num, dim),
            )

        orig_lattice = structure.cell.array
        orig_lengths = np.linalg.norm(orig_lattice, axis=1)
        if orig_lattice.shape != (3, 3) or np.any(~np.isfinite(orig_lengths)) or np.any(orig_lengths <= 1e-12):
            raise ValueError("CellScaling requires three nonzero lattice vectors.")
        unit_vectors = orig_lattice / orig_lengths[:, np.newaxis]

        if params.identify_organic:
            clusters, is_organic_list = get_clusters(structure)

        for i in range(max_num):
            new_structure = structure.copy()
            length_factors = perturbation_factors[i, :3]
            new_lengths = orig_lengths * length_factors
            new_lattice = unit_vectors * new_lengths[:, np.newaxis]

            if params.perturb_angle:
                angle_factors = perturbation_factors[i, 3:]
                cosines = np.array(
                    [
                        np.dot(orig_lattice[1], orig_lattice[2]) / (orig_lengths[1] * orig_lengths[2]),
                        np.dot(orig_lattice[0], orig_lattice[2]) / (orig_lengths[0] * orig_lengths[2]),
                        np.dot(orig_lattice[0], orig_lattice[1]) / (orig_lengths[0] * orig_lengths[1]),
                    ],
                    dtype=float,
                )
                angles = np.arccos(np.clip(cosines, -1.0, 1.0))
                new_angles = angles * angle_factors
                if abs(float(np.sin(new_angles[2]))) <= 1e-12:
                    raise ValueError("CellScaling produced a singular gamma angle.")
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

            eng = "U" if engine_type == 1 else "S"
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

        xy_range = _scan_values(params.xy_range, label="xy_range")
        yz_range = _scan_values(params.yz_range, label="yz_range")
        xz_range = _scan_values(params.xz_range, label="xz_range")
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

        alpha_range = _scan_values(params.alpha_range, label="alpha_range")
        beta_range = _scan_values(params.beta_range, label="beta_range")
        gamma_range = _scan_values(params.gamma_range, label="gamma_range")
        cellpar = cell_to_cellpar(structure.get_cell())
        lengths = cellpar[:3]
        angles0 = cellpar[3:]

        for da in alpha_range:
            for db in beta_range:
                for dg in gamma_range:
                    new_structure = structure.copy()
                    new_angles = angles0 + np.array([da, db, dg])
                    new_lattice = _cell_from_lengths_angles(lengths, new_angles)
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

    _MAX_SOBOL_ATOMS = Sobol.MAXDIM // 3

    @staticmethod
    def unit_ball_displacements(samples: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Map [0, 1]^3 samples to displacement vectors inside per-atom balls."""
        raw = np.asarray(samples, dtype=float)
        if raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError("Perturb: samples must have shape (n_structures, n_atoms, 3).")
        limits = np.asarray(radii, dtype=float)
        if limits.ndim != 1 or limits.shape[0] != raw.shape[1]:
            raise ValueError("Perturb: radii must contain one value per atom.")
        if not np.all(np.isfinite(limits)) or np.any(limits < 0.0):
            raise ValueError("Perturb: max_distance values must be finite and non-negative.")

        cos_theta = 2.0 * raw[:, :, 0] - 1.0
        phi = 2.0 * np.pi * raw[:, :, 1]
        radius = np.cbrt(raw[:, :, 2]) * limits[None, :]
        sin_theta = np.sqrt(np.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
        return np.stack(
            (
                radius * sin_theta * np.cos(phi),
                radius * sin_theta * np.sin(phi),
                radius * cos_theta,
            ),
            axis=2,
        )

    @staticmethod
    def wrapped_positions(structure, positions: np.ndarray) -> np.ndarray:
        """Wrap Cartesian positions through fractional coordinates without ASE's per-call solve."""
        return fast_wrapped_positions(structure, positions)

    def run_structure(self, structure, params: PerturbParams) -> list:
        structure_list = []
        n_atoms = len(structure)
        max_num = int(params.max_num)
        if max_num <= 0:
            raise ValueError("Perturb: max_num must be >= 1.")
        engine_type = int(params.engine_type)
        if engine_type not in {0, 1}:
            raise ValueError("Perturb: engine_type must be 0 (Sobol) or 1 (Uniform).")
        if n_atoms == 0:
            return [structure.copy()]
        dim = n_atoms * 3
        if engine_type == 0 and n_atoms > self._MAX_SOBOL_ATOMS:
            raise CardOperationError(
                "perturb.sobol_dimension_limit",
                "Perturb: Sobol sampling supports at most {max_atoms} atoms; "
                "use Uniform sampling for larger structures.",
                max_atoms=self._MAX_SOBOL_ATOMS,
            )
        symbols = structure.get_chemical_symbols()
        element_scalings = params.element_scalings or {}
        per_atom_scaling = (
            np.array([element_scalings.get(sym, params.max_distance) for sym in symbols])
            if params.use_element_scaling
            else np.full(n_atoms, params.max_distance)
        )
        if not np.all(np.isfinite(per_atom_scaling)) or np.any(per_atom_scaling < 0.0):
            raise ValueError("Perturb: max_distance values must be finite and non-negative.")

        base_seed = (
            derived_structure_seed(int(params.seed), structure)
            if params.use_seed
            else None
        )
        rng = np.random.default_rng(base_seed)

        if engine_type == 0:
            sobol_engine = Sobol(d=dim, scramble=True, seed=base_seed)
            unit_samples = sobol_engine.random(max_num).reshape(max_num, n_atoms, 3)
        else:
            unit_samples = rng.random((max_num, n_atoms, 3))
        displacements = self.unit_ball_displacements(unit_samples, per_atom_scaling)

        if params.identify_organic:
            clusters, is_organic_list = get_clusters(structure)
            organic_clusters = [cluster for cluster, is_org in zip(clusters, is_organic_list) if is_org]
            inorganic_clusters = [cluster for cluster, is_org in zip(clusters, is_organic_list) if not is_org]

        orig_positions = structure.positions
        for i in range(max_num):
            delta = displacements[i]

            if params.identify_organic:
                new_positions = orig_positions.copy()
                for cluster in organic_clusters:
                    cluster_delta = delta[cluster[0]]
                    new_positions[cluster] += cluster_delta
                for cluster in inorganic_clusters:
                    new_positions[cluster] += delta[cluster]
            else:
                new_positions = orig_positions + delta

            new_structure = structure.copy()
            new_structure.set_positions(self.wrapped_positions(structure, new_positions))
            eng = "U" if engine_type == 1 else "S"
            append_config_tag(new_structure, f"Pert(d={params.max_distance},{eng})")
            structure_list.append(new_structure)

        return structure_list


SuperCellMode = Literal["scale", "cell", "max_atoms"]
SuperCellOutputMode = Literal["single", "enumerate"]
SuperCellTargetPolicy = Literal["at_least", "at_most"]


@dataclass(frozen=True)
class SuperCellParams:
    """Parameters for supercell generation."""

    mode: SuperCellMode = "scale"
    output_mode: SuperCellOutputMode = "single"
    target_policy: SuperCellTargetPolicy = "at_least"
    super_scale: tuple[int, int, int] = (3, 3, 3)
    target_cell: tuple[float, float, float] = (20.0, 20.0, 20.0)
    max_atoms: int = 100
    fixed_axis_flags: tuple[bool, bool, bool] = (False, False, False)
    fixed_axis_scale: tuple[int, int, int] = (1, 1, 1)


class SuperCellOperation(StructureOperation):
    """Create supercells without depending on Qt widget state."""

    MAX_ENUMERATED_OUTPUTS = 1000

    def run_structure(self, structure, params: SuperCellParams) -> list:
        expansion_factors = self.plan_factors(structure, params)
        return [self._make_supercell_or_copy(structure, factors) for factors in expansion_factors]

    def plan_factors(
        self,
        structure,
        params: SuperCellParams,
    ) -> list[tuple[int, int, int]]:
        """Return the exact integer repeat factors without building output structures."""
        self._validate_params(structure, params)
        if params.mode == "scale":
            expansion_factors = self._get_scale_factors(params)
        elif params.mode == "cell":
            expansion_factors = self._get_cell_factors(structure, params)
        elif params.mode == "max_atoms":
            expansion_factors = self._get_max_atoms_factors(structure, params)
        else:
            raise ValueError("SuperCell: mode must be scale, cell, or max_atoms.")

        expansion_factors = self._dedupe_factors(expansion_factors, params)
        if params.output_mode == "single":
            expansion_factors = [self._select_single_factor(structure, expansion_factors)]
        elif len(expansion_factors) > self.MAX_ENUMERATED_OUTPUTS:
            raise CardOperationError(
                "supercell_too_many_outputs",
                "Supercell enumeration would create {count} structures; the limit is {limit}. "
                "Use single-output mode or reduce the requested size.",
                count=len(expansion_factors),
                limit=self.MAX_ENUMERATED_OUTPUTS,
            )
        return expansion_factors

    def _validate_params(self, structure, params: SuperCellParams) -> None:
        if params.mode not in {"scale", "cell", "max_atoms"}:
            raise ValueError("SuperCell: mode must be scale, cell, or max_atoms.")
        if params.output_mode not in {"single", "enumerate"}:
            raise ValueError("SuperCell: output_mode must be single or enumerate.")
        if params.target_policy not in {"at_least", "at_most"}:
            raise ValueError("SuperCell: target_policy must be at_least or at_most.")
        if len(structure) <= 0:
            raise CardOperationError(
                "supercell_empty_input",
                "Supercell generation requires an input structure with at least one atom.",
            )
        active_triplets = [("fixed_axis_scale", params.fixed_axis_scale)]
        if params.mode == "scale":
            active_triplets.append(("super_scale", params.super_scale))
        elif params.mode == "cell":
            active_triplets.append(("target_cell", params.target_cell))
        for name, values in active_triplets:
            if len(values) != 3 or not np.all(np.isfinite(values)):
                raise ValueError(f"SuperCell: {name} must contain three finite values.")
        if params.mode == "scale" and any(int(value) < 1 for value in params.super_scale):
            raise ValueError("SuperCell: super_scale values must be positive integers.")
        if params.mode == "cell" and any(float(value) <= 0.0 for value in params.target_cell):
            raise ValueError("SuperCell: target_cell values must be positive.")
        if any(int(value) < 1 for value in params.fixed_axis_scale):
            raise ValueError("SuperCell: fixed_axis_scale values must be positive integers.")
        if len(params.fixed_axis_flags) != 3:
            raise ValueError("SuperCell: fixed_axis_flags must contain three values.")
        cell_lengths = np.asarray(structure.cell.lengths(), dtype=float)
        if not np.all(np.isfinite(cell_lengths)) or np.any(cell_lengths <= 0.0):
            raise CardOperationError(
                "supercell_invalid_cell",
                "Supercell generation requires three finite, non-zero lattice vectors.",
            )
        if params.mode == "max_atoms" and int(params.max_atoms) < len(structure):
            raise CardOperationError(
                "supercell_atom_budget_below_input",
                "The atom limit ({limit}) is smaller than the input structure ({input_atoms} atoms).",
                limit=int(params.max_atoms),
                input_atoms=len(structure),
            )

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
        scale_factors = self._apply_fixed_axes((int(na), int(nb), int(nc)), params)
        if params.output_mode == "single":
            return [scale_factors]
        axis_values = self._get_iteration_axis_values(scale_factors, params)
        return [
            (na, nb, nc)
            for na in axis_values[0]
            for nb in axis_values[1]
            for nc in axis_values[2]
        ]

    def _get_cell_factors(self, structure, params: SuperCellParams) -> list[tuple[int, int, int]]:
        target_a, target_b, target_c = params.target_cell
        lattice = structure.cell.array
        a_len = np.linalg.norm(lattice[0])
        b_len = np.linalg.norm(lattice[1])
        c_len = np.linalg.norm(lattice[2])

        if params.target_policy == "at_least":
            na = self._fixed_or_at_least_factor(0, target_a, a_len, params)
            nb = self._fixed_or_at_least_factor(1, target_b, b_len, params)
            nc = self._fixed_or_at_least_factor(2, target_c, c_len, params)
        else:
            na = self._fixed_or_at_most_factor(0, target_a, a_len, params)
            nb = self._fixed_or_at_most_factor(1, target_b, b_len, params)
            nc = self._fixed_or_at_most_factor(2, target_c, c_len, params)

        factors = (max(na, 1), max(nb, 1), max(nc, 1))
        if params.output_mode == "single":
            return [factors]
        axis_values = self._get_iteration_axis_values(factors, params)
        return [
            (na, nb, nc)
            for na in axis_values[0]
            for nb in axis_values[1]
            for nc in axis_values[2]
        ]

    def _fixed_or_at_least_factor(
        self,
        axis: int,
        target: float,
        length: float,
        params: SuperCellParams,
    ) -> int:
        if params.fixed_axis_flags[axis]:
            return int(params.fixed_axis_scale[axis])
        return max(int(np.ceil(float(target) / float(length) - 1e-12)), 1)

    def _fixed_or_at_most_factor(
        self,
        axis: int,
        target: float,
        length: float,
        params: SuperCellParams,
    ) -> int:
        if params.fixed_axis_flags[axis]:
            return int(params.fixed_axis_scale[axis])
        return max(int(np.floor(float(target) / float(length) + 1e-12)), 1)

    def _get_max_atoms_factors(self, structure, params: SuperCellParams) -> list[tuple[int, int, int]]:
        num_atoms_orig = len(structure)
        max_n = max(int(params.max_atoms // num_atoms_orig), 1)
        axis_ranges = [
            range(int(params.fixed_axis_scale[axis]), int(params.fixed_axis_scale[axis]) + 1)
            if is_fixed
            else range(1, max_n + 1)
            for axis, is_fixed in enumerate(params.fixed_axis_flags)
        ]

        if params.output_mode == "single":
            best_factor = None
            best_score = None
            base_lengths = tuple(float(value) for value in structure.cell.lengths())
            for na in axis_ranges[0]:
                for nb in axis_ranges[1]:
                    remaining = int(params.max_atoms) // (num_atoms_orig * na * nb)
                    if remaining < 1:
                        break
                    nc = (
                        int(params.fixed_axis_scale[2])
                        if params.fixed_axis_flags[2]
                        else min(max_n, remaining)
                    )
                    factor = (na, nb, nc)
                    total_atoms = num_atoms_orig * na * nb * nc
                    if total_atoms > params.max_atoms:
                        continue
                    output_lengths = (
                        base_lengths[0] * na,
                        base_lengths[1] * nb,
                        base_lengths[2] * nc,
                    )
                    aspect = max(output_lengths) / min(output_lengths)
                    score = (total_atoms, -aspect, tuple(-value for value in factor))
                    if best_score is None or score > best_score:
                        best_factor = factor
                        best_score = score
            if best_factor is None:
                raise CardOperationError(
                    "supercell_fixed_axes_over_budget",
                    "The fixed-axis multipliers require more than the {limit}-atom budget.",
                    limit=int(params.max_atoms),
                )
            return [best_factor]

        expansion_factors = []
        for na in axis_ranges[0]:
            for nb in axis_ranges[1]:
                remaining = int(params.max_atoms) // (num_atoms_orig * na * nb)
                if remaining < 1:
                    break
                nc_values = axis_ranges[2]
                if not params.fixed_axis_flags[2]:
                    nc_values = range(1, min(max_n, remaining) + 1)
                for nc in nc_values:
                    total_atoms = num_atoms_orig * na * nb * nc
                    if total_atoms <= params.max_atoms:
                        expansion_factors.append((na, nb, nc))
                        if len(expansion_factors) > self.MAX_ENUMERATED_OUTPUTS:
                            raise CardOperationError(
                                "supercell_too_many_outputs",
                                "Supercell enumeration would create more than {limit} structures. "
                                "Use single-output mode or reduce the atom limit.",
                                limit=self.MAX_ENUMERATED_OUTPUTS,
                            )

        if not expansion_factors:
            raise CardOperationError(
                "supercell_fixed_axes_over_budget",
                "The fixed-axis multipliers require more than the {limit}-atom budget.",
                limit=int(params.max_atoms),
            )
        return expansion_factors

    @staticmethod
    def _shape_ratio(structure, factors: tuple[int, int, int]) -> float:
        base_lengths = structure.cell.lengths()
        lengths = tuple(float(base_lengths[index]) * factors[index] for index in range(3))
        positive = tuple(length for length in lengths if length > 1e-12)
        if len(positive) < 2:
            return float("inf")
        return max(positive) / min(positive)

    def _select_single_factor(
        self,
        structure,
        expansion_factors: list[tuple[int, int, int]],
    ) -> tuple[int, int, int]:
        if not expansion_factors:
            return (1, 1, 1)
        return max(
            expansion_factors,
            key=lambda factors: (
                len(structure) * int(np.prod(factors)),
                -self._shape_ratio(structure, factors),
                tuple(-value for value in factors),
            ),
        )

    def _make_supercell_or_copy(self, structure, factors: tuple[int, int, int]):
        if factors == (1, 1, 1):
            return structure.copy()
        return self._make_supercell(structure, *factors)

    def _make_supercell(self, structure, na: int, nb: int, nc: int):
        supercell = make_supercell(structure, np.diag([na, nb, nc]), order="atom-major")
        supercell.info["Config_type"] = structure.info.get("Config_type", "")
        append_config_tag(supercell, f"SC({na}x{nb}x{nc})")
        return supercell
