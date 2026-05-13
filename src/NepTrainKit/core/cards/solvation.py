"""UI-independent solvation Make Dataset operations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from io import StringIO
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read as ase_read

from NepTrainKit.core.config_type import append_config_tag, stable_config_id

from .operation import StructureOperation
from .structure import OrganicMolConfigPBCOperation, OrganicMolConfigPBCParams


AVOGADRO = 6.02214076e23

DEFAULT_WATER_XYZ = """3
water
O 0.00000000 0.00000000 0.00000000
H 0.95720000 0.00000000 0.00000000
H -0.23998720 0.92662721 0.00000000
"""

ION_ELEMENTS = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Fe", "Cu", "Al"}

COLLISION_RADIUS = {
    "H": 0.75,
    "Li": 0.90,
    "Na": 1.15,
    "K": 1.50,
    "Rb": 1.65,
    "Cs": 1.80,
    "Mg": 1.05,
    "Ca": 1.35,
    "Sr": 1.50,
    "Ba": 1.65,
    "Al": 1.00,
    "C": 1.10,
    "N": 1.00,
    "O": 1.00,
    "F": 0.95,
    "P": 1.25,
    "S": 1.25,
    "Cl": 1.20,
    "Fe": 1.25,
    "Cu": 1.20,
    "Zn": 1.20,
    "Br": 1.30,
    "I": 1.45,
}

MODE_PROFILES = {
    "general": {
        "shell": (2.2, 4.5),
        "collision_scale": 0.70,
        "default_per_center": 2,
        "max_default": 80,
        "dipolar": False,
        "density_packing_fraction": 0.45,
    },
    "water": {
        "shell": (2.1, 4.2),
        "collision_scale": 0.70,
        "default_per_center": 3,
        "max_default": 120,
        "dipolar": True,
        "density_packing_fraction": 0.50,
    },
    "ion-water": {
        "shell": (2.5, 3.4),
        "collision_scale": 0.72,
        "default_per_center": 6,
        "max_default": 120,
        "dipolar": True,
        "density_packing_fraction": 0.65,
    },
    "loose": {
        "shell": (2.5, 6.0),
        "collision_scale": 0.62,
        "default_per_center": 2,
        "max_default": 100,
        "dipolar": False,
        "density_packing_fraction": 0.35,
    },
    "dense": {
        "shell": (1.9, 3.5),
        "collision_scale": 0.78,
        "default_per_center": 5,
        "max_default": 150,
        "dipolar": True,
        "density_packing_fraction": 0.60,
    },
}

ION_WATER_PROFILES = {
    "Li": {"coordination": 4, "ion_o": (1.85, 2.10), "first_shell_com": (2.1, 2.8)},
    "Na": {"coordination": 6, "ion_o": (2.25, 2.55), "first_shell_com": (2.5, 3.3)},
    "K": {"coordination": 8, "ion_o": (2.65, 3.05), "first_shell_com": (3.0, 4.0)},
    "Mg": {"coordination": 6, "ion_o": (1.95, 2.20), "first_shell_com": (2.3, 3.0)},
    "Ca": {"coordination": 6, "ion_o": (2.30, 2.60), "first_shell_com": (2.6, 3.4)},
    "Sr": {"coordination": 8, "ion_o": (2.50, 2.85), "first_shell_com": (2.8, 3.7)},
    "Ba": {"coordination": 8, "ion_o": (2.70, 3.10), "first_shell_com": (3.0, 4.0)},
    "Zn": {"coordination": 6, "ion_o": (1.95, 2.25), "first_shell_com": (2.3, 3.1)},
}

DEFAULT_ION_WATER_PROFILE = {"coordination": 6, "ion_o": (2.30, 2.60), "first_shell_com": (2.5, 3.4)}


@dataclass(frozen=True)
class LocalSolvationParams:
    """Parameters for local solvent-shell generation."""

    solvent_xyz: str = DEFAULT_WATER_XYZ
    structures: int = 1
    solvent_count: int = 30
    sampling_mode: str = "auto"
    center_mode: str = "all"
    center_elements: str = ""
    center_indices: str = ""
    z_range: tuple[float, float] = (0.0, 0.0)
    shell: tuple[float, float] = (2.2, 4.5)
    min_distance: float = 0.0
    collision_scale: float = 0.0
    max_attempts: int = 3000
    strict_count: bool = True
    auto_box: bool = False
    box_size: float = 100.0
    box_padding: float = 8.0
    min_box: float = 0.0
    flex_solvent: bool = False
    flex_pool: int = 32
    flex_torsion_range: tuple[float, float] = (-180.0, 180.0)
    flex_max_torsions: int = 5
    flex_gaussian_sigma: float = 0.03
    use_seed: bool = False
    seed: int = 0


@dataclass(frozen=True)
class SolventBoxFillParams:
    """Parameters for filling an existing periodic cell with solvent."""

    solvent_xyz: str = DEFAULT_WATER_XYZ
    structures: int = 1
    count_mode: str = "fixed"
    solvent_count: int = 100
    density: float = 1.0
    sampling_mode: str = "auto"
    fill_packing: float = 1.0
    min_distance: float = 0.0
    collision_scale: float = 0.0
    max_attempts_per_solvent: int = 500
    strict_count: bool = True
    flex_solvent: bool = False
    flex_pool: int = 32
    flex_torsion_range: tuple[float, float] = (-180.0, 180.0)
    flex_max_torsions: int = 5
    flex_gaussian_sigma: float = 0.03
    use_seed: bool = False
    seed: int = 0


@dataclass(frozen=True)
class SolventConformer:
    """Precomputed solvent conformer data used in repeated placement attempts."""

    symbols: list[str]
    positions: np.ndarray
    centered_positions: np.ndarray
    water_like: bool
    oxygen_index: int | None
    hydrogen_indices: tuple[int, ...]
    oxygen_shifted_positions: np.ndarray | None


class LocalSolvationOperation(StructureOperation):
    """Insert solvent molecules around selected atoms."""

    def run_structure(self, structure, params: LocalSolvationParams) -> list[Atoms]:
        self._validate_common(params.solvent_xyz, params.structures, params.solvent_count)
        if int(params.max_attempts) <= 0:
            raise ValueError("Local Solvation: max_attempts must be >= 1.")

        solvent = parse_solvent_xyz(params.solvent_xyz)
        centers = select_center_indices(structure, params.center_mode, params.center_elements, params.center_indices, params.z_range)
        if not centers:
            raise ValueError("Local Solvation: no center atoms selected.")

        selected_elements = {structure.get_chemical_symbols()[idx] for idx in centers}
        mode = resolve_mode(params.sampling_mode, solvent, selected_elements)
        profile = MODE_PROFILES[mode]
        shell = tuple(float(v) for v in params.shell)
        if mode == "auto":
            raise AssertionError("mode should have been resolved")
        if shell[1] <= shell[0]:
            raise ValueError("Local Solvation: shell outer radius must be larger than inner radius.")

        collision_scale = float(params.collision_scale) if float(params.collision_scale) > 0 else float(profile["collision_scale"])
        outputs = []
        for sample_idx in range(int(params.structures)):
            rng = make_rng(structure, bool(params.use_seed), int(params.seed), sample_idx)
            conformers = solvent_conformer_pool(solvent, params, rng_seed=None if not params.use_seed else int(params.seed) + sample_idx)
            target_count = int(params.solvent_count)
            atoms, placed = self._solvate_one(
                structure,
                centers=centers,
                solvent_pool=conformers,
                target_count=target_count,
                mode=mode,
                shell=shell,
                collision_scale=collision_scale,
                min_distance=float(params.min_distance),
                max_attempts=int(params.max_attempts),
                rng=rng,
            )
            if params.strict_count and placed != target_count:
                hint = ""
                if placed == 0 and has_valid_cell(structure) and np.any(np.asarray(structure.pbc, dtype=bool)):
                    hint = " Input appears to be a periodic dense structure with no solvent-accessible void; use a slab/box with free volume, lower solvent_count, or use Solvent Box Fill on a larger cell."
                raise ValueError(f"Local Solvation: placed {placed}/{target_count} solvent molecules.{hint}")
            if params.auto_box:
                apply_auto_box(atoms, float(params.box_padding), float(params.min_box))
            elif not has_valid_cell(structure):
                center_in_fixed_box(atoms, float(params.box_size))
            append_config_tag(atoms, f"SolvLocal(mode={mode},n={placed},sel={len(centers)})")
            outputs.append(atoms)
        return outputs

    @staticmethod
    def _validate_common(solvent_xyz: str, structures: int, solvent_count: int) -> None:
        if not solvent_xyz.strip():
            raise ValueError("Solvation: solvent_xyz is empty.")
        if int(structures) <= 0:
            raise ValueError("Solvation: structures must be >= 1.")
        if int(solvent_count) <= 0:
            raise ValueError("Solvation: solvent_count must be >= 1.")

    def _solvate_one(
        self,
        structure: Atoms,
        *,
        centers: list[int],
        solvent_pool: list[Atoms],
        target_count: int,
        mode: str,
        shell: tuple[float, float],
        collision_scale: float,
        min_distance: float,
        max_attempts: int,
        rng: np.random.Generator,
    ) -> tuple[Atoms, int]:
        base_symbols = structure.get_chemical_symbols()
        base_positions = np.asarray(structure.get_positions(), dtype=float)
        out_symbols = list(base_symbols)
        out_positions = [np.array(pos, dtype=float) for pos in base_positions]
        cell = np.asarray(structure.cell.array, dtype=float)
        pbc = np.asarray(structure.pbc, dtype=bool)
        has_periodic_cell = has_valid_cell(structure) and bool(np.any(pbc))
        spatial_hash = (
            PeriodicSpatialHash(
                out_symbols,
                np.asarray(out_positions, dtype=float),
                cell=cell,
                pbc=pbc,
                cutoff=max_collision_cutoff(solvent_pool, out_symbols, collision_scale, min_distance),
            )
            if has_periodic_cell
            else NonPeriodicSpatialHash(
                out_symbols,
                np.asarray(out_positions, dtype=float),
                cutoff=max_collision_cutoff(solvent_pool, out_symbols, collision_scale, min_distance),
            )
        )
        center_positions = base_positions[centers]
        center_symbols = [base_symbols[idx] for idx in centers]
        ion_counts = {idx: 0 for idx in centers}
        conformers = prepare_solvent_conformers(solvent_pool)

        placed = 0
        attempts = 0
        no_progress_limit = min(max_attempts, max(300, 20 * max(target_count, 1)))
        while placed < target_count and attempts < max_attempts:
            attempts += 1
            solvent = conformers[int(rng.integers(0, len(conformers)))]
            candidate = self._propose_local_candidate(
                solvent,
                center_indices=centers,
                center_positions=center_positions,
                center_symbols=center_symbols,
                ion_counts=ion_counts,
                mode=mode,
                shell=shell,
                rng=rng,
            )
            if candidate is None:
                continue
            cand_symbols, cand_positions, used_center = candidate
            collision = spatial_hash.has_collision(
                cand_symbols,
                cand_positions,
                collision_scale=collision_scale,
                min_distance=min_distance,
            )
            if collision:
                if placed == 0 and attempts >= no_progress_limit:
                    break
                continue
            out_symbols.extend(cand_symbols)
            out_positions.extend(np.asarray(cand_positions, dtype=float))
            spatial_hash.add_atoms(cand_symbols, cand_positions)
            if used_center is not None:
                ion_counts[used_center] = ion_counts.get(used_center, 0) + 1
            placed += 1

        atoms = Atoms(
            symbols=out_symbols,
            positions=np.asarray(out_positions, dtype=float),
            cell=structure.cell,
            pbc=structure.pbc,
            info=dict(structure.info),
        )
        return atoms, placed

    @staticmethod
    def _propose_local_candidate(
        solvent: SolventConformer,
        *,
        center_indices: list[int],
        center_positions: np.ndarray,
        center_symbols: list[str],
        ion_counts: dict[int, int],
        mode: str,
        shell: tuple[float, float],
        rng: np.random.Generator,
    ) -> tuple[list[str], np.ndarray, int | None] | None:
        local_center_idx = int(rng.integers(0, len(center_indices)))
        center_index = center_indices[local_center_idx]
        center = center_positions[local_center_idx]
        center_symbol = center_symbols[local_center_idx]

        if mode == "ion-water" and center_symbol in ION_ELEMENTS and solvent.water_like:
            ion_profile = ION_WATER_PROFILES.get(center_symbol, DEFAULT_ION_WATER_PROFILE)
            if ion_counts.get(center_index, 0) >= int(ion_profile["coordination"]):
                inner, outer = shell
            else:
                inner, outer = ion_profile["ion_o"]
                direction = random_unit_vector(rng)
                target_oxygen = center + direction * random_radius(inner, outer, rng)
                oriented = orient_water_oxygen_conformer(solvent, target_oxygen, center, rng)
                return solvent.symbols, oriented, center_index
        else:
            inner, outer = shell

        target = center + random_unit_vector(rng) * random_radius(inner, outer, rng)
        oriented = orient_centered_by_center(solvent.centered_positions, target, dipolar=MODE_PROFILES[mode]["dipolar"], center=center, rng=rng)
        return solvent.symbols, oriented, center_index


class SolventBoxFillOperation(StructureOperation):
    """Fill a periodic cell with solvent molecules."""

    def run_structure(self, structure, params: SolventBoxFillParams) -> list[Atoms]:
        LocalSolvationOperation._validate_common(params.solvent_xyz, params.structures, params.solvent_count)
        if params.count_mode not in {"fixed", "density"}:
            raise ValueError("Solvent Box Fill: count_mode must be 'fixed' or 'density'.")
        if float(params.fill_packing) <= 0:
            raise ValueError("Solvent Box Fill: fill_packing must be positive.")
        if int(params.max_attempts_per_solvent) <= 0:
            raise ValueError("Solvent Box Fill: max_attempts_per_solvent must be >= 1.")
        if params.sampling_mode not in {"auto", "general", "water", "loose", "dense"}:
            raise ValueError("Solvent Box Fill: sampling_mode must be one of auto, general, water, loose, dense.")

        cell = np.asarray(structure.cell.array, dtype=float)
        if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) <= 1e-12:
            raise ValueError("Solvent Box Fill requires a non-singular input cell.")
        if not np.any(np.asarray(structure.pbc, dtype=bool)):
            raise ValueError("Solvent Box Fill requires periodic boundary conditions.")

        solvent = parse_solvent_xyz(params.solvent_xyz)
        mode = resolve_mode(params.sampling_mode, solvent, set())
        profile = MODE_PROFILES[mode]
        collision_scale = float(params.collision_scale) if float(params.collision_scale) > 0 else float(profile["collision_scale"])
        target_count = int(params.solvent_count)
        if params.count_mode == "density":
            target_count = estimate_solvent_count_from_density(solvent, float(params.density), cell, float(params.fill_packing))

        outputs = []
        for sample_idx in range(int(params.structures)):
            rng = make_rng(structure, bool(params.use_seed), int(params.seed), sample_idx)
            conformers = solvent_conformer_pool(solvent, params, rng_seed=None if not params.use_seed else int(params.seed) + sample_idx)
            atoms, placed, attempts, consecutive_failures = self._fill_one(
                structure,
                solvent_pool=conformers,
                target_count=target_count,
                collision_scale=collision_scale,
                min_distance=float(params.min_distance),
                max_attempts_per_solvent=int(params.max_attempts_per_solvent),
                rng=rng,
            )
            if params.strict_count and placed != target_count:
                raise ValueError(
                    f"Solvent Box Fill: placed {placed}/{target_count} solvent molecules. "
                    f"Stopped after {attempts} attempts"
                    + (f" and {consecutive_failures} consecutive rejected placements." if consecutive_failures else ".")
                    + " The cell may be too small/dense for the requested solvent_count, min_distance, or density."
                )
            append_config_tag(atoms, f"SolvBox(mode={mode},n={placed})")
            outputs.append(atoms)
        return outputs

    @staticmethod
    def _fill_one(
        structure: Atoms,
        *,
        solvent_pool: list[Atoms],
        target_count: int,
        collision_scale: float,
        min_distance: float,
        max_attempts_per_solvent: int,
        rng: np.random.Generator,
    ) -> tuple[Atoms, int, int, int]:
        out_symbols = list(structure.get_chemical_symbols())
        out_positions = [np.array(pos, dtype=float) for pos in np.asarray(structure.get_positions(), dtype=float)]
        cell = np.asarray(structure.cell.array, dtype=float)
        pbc = np.asarray(structure.pbc, dtype=bool)
        spatial_hash = PeriodicSpatialHash(
            out_symbols,
            np.asarray(out_positions, dtype=float),
            cell=cell,
            pbc=pbc,
            cutoff=max_collision_cutoff(solvent_pool, out_symbols, collision_scale, min_distance),
        )
        wrap_lengths = spatial_hash.ortho_lengths
        conformers = prepare_solvent_conformers(solvent_pool)
        placed = 0
        attempts = 0
        max_attempts = max_attempts_per_solvent * target_count
        consecutive_failures = 0
        stall_limit = min(max_attempts, max(300, 20 * max(target_count, 1)))

        while placed < target_count and attempts < max_attempts:
            attempts += 1
            solvent = conformers[int(rng.integers(0, len(conformers)))]
            target = rng.random(3) @ cell
            cand_positions = orient_centered_by_center(solvent.centered_positions, target, dipolar=False, center=target, rng=rng)
            cand_positions = wrap_positions(cand_positions, cell, wrap_lengths)
            if spatial_hash.has_collision(
                solvent.symbols,
                cand_positions,
                collision_scale=collision_scale,
                min_distance=min_distance,
            ):
                consecutive_failures += 1
                if consecutive_failures >= stall_limit:
                    break
                continue
            out_symbols.extend(solvent.symbols)
            out_positions.extend(cand_positions)
            spatial_hash.add_atoms(solvent.symbols, cand_positions)
            placed += 1
            consecutive_failures = 0

        atoms = Atoms(
            symbols=out_symbols,
            positions=np.asarray(out_positions, dtype=float),
            cell=structure.cell,
            pbc=structure.pbc,
            info=dict(structure.info),
        )
        try:
            atoms.wrap()
        except Exception:
            pass
        return atoms, placed, attempts, consecutive_failures


def parse_solvent_xyz(text: str) -> Atoms:
    """Parse a single solvent molecule from XYZ/extxyz text."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Solvation: solvent_xyz is empty.")
    try:
        atoms = ase_read(StringIO(stripped), index=0, format="extxyz")
    except Exception as exc:
        raise ValueError(f"Solvation: cannot parse solvent_xyz: {exc}") from exc
    if len(atoms) == 0:
        raise ValueError("Solvation: solvent molecule contains no atoms.")
    return center_solvent(atoms)


def center_solvent(atoms: Atoms) -> Atoms:
    centered = atoms.copy()
    positions = np.asarray(centered.get_positions(), dtype=float)
    centered.set_positions(positions - positions.mean(axis=0))
    centered.set_cell(np.zeros((3, 3)))
    centered.set_pbc(False)
    return centered


def solvent_conformer_pool(solvent: Atoms, params: LocalSolvationParams | SolventBoxFillParams, rng_seed: int | None) -> list[Atoms]:
    if not bool(params.flex_solvent):
        return [solvent]
    pool_size = int(params.flex_pool)
    if pool_size <= 0:
        raise ValueError("Solvation: flex_pool must be >= 1.")
    op = OrganicMolConfigPBCOperation()
    flex_params = OrganicMolConfigPBCParams(
        perturb_per_frame=pool_size,
        torsion_range_deg=tuple(map(float, params.flex_torsion_range)),
        max_torsions_per_conf=int(params.flex_max_torsions),
        gaussian_sigma=float(params.flex_gaussian_sigma),
        pbc_mode="no",
        use_seed=rng_seed is not None,
        seed=0 if rng_seed is None else int(rng_seed),
    )
    return [center_solvent(atoms) for atoms in op.run_structure(solvent, flex_params)]


def prepare_solvent_conformers(solvent_pool: list[Atoms]) -> list[SolventConformer]:
    conformers: list[SolventConformer] = []
    for solvent in solvent_pool:
        symbols = solvent.get_chemical_symbols()
        positions = np.asarray(solvent.get_positions(), dtype=float)
        centered = positions - positions.mean(axis=0)
        oxygen_indices = tuple(idx for idx, symbol in enumerate(symbols) if symbol == "O")
        hydrogen_indices = tuple(idx for idx, symbol in enumerate(symbols) if symbol == "H")
        oxygen_index = oxygen_indices[0] if oxygen_indices else None
        oxygen_shifted = None if oxygen_index is None else positions - positions[oxygen_index]
        conformers.append(
            SolventConformer(
                symbols=symbols,
                positions=positions,
                centered_positions=centered,
                water_like=is_water_like(symbols),
                oxygen_index=oxygen_index,
                hydrogen_indices=hydrogen_indices,
                oxygen_shifted_positions=oxygen_shifted,
            )
        )
    return conformers


def resolve_mode(mode: str, solvent: Atoms, selected_elements: set[str]) -> str:
    if mode not in {"auto", "general", "water", "ion-water", "loose", "dense"}:
        raise ValueError(f"Solvation: unsupported sampling_mode '{mode}'.")
    if mode != "auto":
        return mode
    symbols = solvent.get_chemical_symbols()
    if is_water_like(symbols) and selected_elements & ION_ELEMENTS:
        return "ion-water"
    if is_water_like(symbols):
        return "water"
    return "general"


def select_center_indices(
    atoms: Atoms,
    mode: str,
    elements_text: str,
    indices_text: str,
    z_range: tuple[float, float],
) -> list[int]:
    mode = mode.strip().lower()
    if mode == "all":
        return list(range(len(atoms)))
    if mode == "elements":
        elements = {normalize_symbol(part) for part in re.split(r"[,;\s]+", elements_text) if part.strip()}
        if not elements:
            raise ValueError("Local Solvation: center_elements is empty.")
        return [idx for idx, symbol in enumerate(atoms.get_chemical_symbols()) if symbol in elements]
    if mode == "indices":
        return parse_index_ranges(indices_text, len(atoms))
    if mode == "z_range":
        z0, z1 = sorted(float(v) for v in z_range)
        z = np.asarray(atoms.get_positions(), dtype=float)[:, 2]
        return [int(idx) for idx in np.where((z >= z0) & (z <= z1))[0]]
    raise ValueError(f"Local Solvation: unsupported center_mode '{mode}'.")


def parse_index_ranges(text: str, natoms: int) -> list[int]:
    selected: set[int] = set()
    if not text.strip():
        raise ValueError("Local Solvation: center_indices is empty.")
    for chunk in re.split(r"[,;\s]+", text.strip()):
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                start, end = end, start
            for value in range(start, end + 1):
                add_1based_index(selected, value, natoms)
        else:
            add_1based_index(selected, int(chunk), natoms)
    return sorted(selected)


def add_1based_index(selected: set[int], value: int, natoms: int) -> None:
    if value < 1 or value > natoms:
        raise ValueError(f"Local Solvation: atom index {value} out of range 1..{natoms}.")
    selected.add(value - 1)


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if not symbol:
        return symbol
    return symbol[0].upper() + symbol[1:].lower()


def make_rng(structure: Atoms, use_seed: bool, seed: int, sample_idx: int) -> np.random.Generator:
    if not use_seed:
        return np.random.default_rng()
    return np.random.default_rng(int(seed) + stable_config_id(structure) * 1000003 + int(sample_idx))


def is_water_like(symbols: Iterable[str]) -> bool:
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts.get("O", 0) == 1 and counts.get("H", 0) == 2 and sum(counts.values()) == 3


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


def random_radius(inner: float, outer: float, rng: np.random.Generator) -> float:
    return float((inner**3 + rng.random() * (outer**3 - inner**3)) ** (1.0 / 3.0))


def rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def orient_by_center(solvent_positions: np.ndarray, target: np.ndarray, *, dipolar: bool, center: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = solvent_positions - solvent_positions.mean(axis=0)
    return orient_centered_by_center(positions, target, dipolar=dipolar, center=center, rng=rng)


def orient_centered_by_center(centered_positions: np.ndarray, target: np.ndarray, *, dipolar: bool, center: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = np.asarray(centered_positions, dtype=float)
    rotated = positions @ rotation_matrix(rng).T
    if dipolar:
        dipole = rotated.mean(axis=0)
        desired = target - center
        if np.linalg.norm(dipole) > 1e-12 and np.linalg.norm(desired) > 1e-12:
            rotated = (rotated @ rotation_between(dipole, desired).T)
    return rotated + target


def orient_water_oxygen(solvent_positions: np.ndarray, symbols: list[str], target_oxygen: np.ndarray, ion_center: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    oxygen_indices = [idx for idx, symbol in enumerate(symbols) if symbol == "O"]
    if not oxygen_indices:
        return orient_by_center(solvent_positions, target_oxygen, dipolar=True, center=ion_center, rng=rng)
    oxygen_idx = oxygen_indices[0]
    shifted = solvent_positions - solvent_positions[oxygen_idx]
    rotated = shifted @ rotation_matrix(rng).T
    hydrogens = [idx for idx, symbol in enumerate(symbols) if symbol == "H"]
    if hydrogens:
        oh_mean = rotated[hydrogens].mean(axis=0) - rotated[oxygen_idx]
        away_from_ion = target_oxygen - ion_center
        if np.linalg.norm(oh_mean) > 1e-12 and np.linalg.norm(away_from_ion) > 1e-12:
            rotated = rotated @ rotation_between(oh_mean, away_from_ion).T
    return rotated + target_oxygen


def orient_water_oxygen_conformer(solvent: SolventConformer, target_oxygen: np.ndarray, ion_center: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if solvent.oxygen_shifted_positions is None or solvent.oxygen_index is None:
        return orient_centered_by_center(solvent.centered_positions, target_oxygen, dipolar=True, center=ion_center, rng=rng)
    rotated = solvent.oxygen_shifted_positions @ rotation_matrix(rng).T
    if solvent.hydrogen_indices:
        oh_mean = rotated[list(solvent.hydrogen_indices)].mean(axis=0) - rotated[solvent.oxygen_index]
        away_from_ion = target_oxygen - ion_center
        if np.linalg.norm(oh_mean) > 1e-12 and np.linalg.norm(away_from_ion) > 1e-12:
            rotated = rotated @ rotation_between(oh_mean, away_from_ion).T
    return rotated + target_oxygen


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-12:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return axis_angle_rotation(axis, math.pi)
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=float,
    )


def has_collision(
    trial_symbols: list[str],
    trial_positions: np.ndarray,
    existing_symbols: list[str],
    existing_positions: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    collision_scale: float,
    min_distance: float,
) -> bool:
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    trial_positions = np.asarray(trial_positions, dtype=float)
    existing_positions = np.asarray(existing_positions, dtype=float)
    use_periodic = cell.shape == (3, 3) and abs(float(np.linalg.det(cell))) > 1e-12 and bool(np.any(pbc))
    if not use_periodic:
        for symbol, position in zip(trial_symbols, trial_positions):
            deltas = existing_positions - position
            distances2 = np.einsum("ij,ij->i", deltas, deltas)
            cutoffs = np.fromiter(
                (pair_cutoff(symbol, other_symbol, collision_scale, min_distance) for other_symbol in existing_symbols),
                dtype=float,
                count=len(existing_symbols),
            )
            if bool(np.any(distances2 < cutoffs * cutoffs)):
                return True
        return False

    inv_cell_t = np.linalg.inv(cell.T)
    ortho_lengths = orthorhombic_lengths(cell)
    for symbol, position in zip(trial_symbols, trial_positions):
        for other_symbol, other_position in zip(existing_symbols, existing_positions):
            delta = minimum_image_delta_fast(np.asarray(position) - np.asarray(other_position), cell, pbc, inv_cell_t, ortho_lengths)
            distance = float(np.linalg.norm(delta))
            cutoff = pair_cutoff(symbol, other_symbol, collision_scale, min_distance)
            if distance < cutoff:
                return True
    return False


class NonPeriodicSpatialHash:
    """Spatial hash for local structures without periodic minimum-image checks."""

    def __init__(self, symbols: list[str], positions: np.ndarray, *, cutoff: float):
        self.cutoff = max(float(cutoff), 1e-6)
        self.buckets: dict[tuple[int, int, int], list[tuple[str, np.ndarray]]] = {}
        self.neighbor_offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        self.add_atoms(symbols, positions)

    def add_atoms(self, symbols: list[str], positions: np.ndarray) -> None:
        for symbol, position in zip(symbols, np.asarray(positions, dtype=float)):
            key = self._bucket_key(position)
            self.buckets.setdefault(key, []).append((symbol, np.asarray(position, dtype=float)))

    def has_collision(
        self,
        trial_symbols: list[str],
        trial_positions: np.ndarray,
        *,
        collision_scale: float,
        min_distance: float,
    ) -> bool:
        for symbol, position in zip(trial_symbols, np.asarray(trial_positions, dtype=float)):
            base = self._bucket_key(position)
            for dx, dy, dz in self.neighbor_offsets:
                bucket = self.buckets.get((base[0] + dx, base[1] + dy, base[2] + dz))
                if not bucket:
                    continue
                for other_symbol, other_position in bucket:
                    cutoff = pair_cutoff(symbol, other_symbol, collision_scale, min_distance)
                    delta = position - other_position
                    if float(np.dot(delta, delta)) < cutoff * cutoff:
                        return True
        return False

    def _bucket_key(self, position: np.ndarray) -> tuple[int, int, int]:
        scaled = np.floor(np.asarray(position, dtype=float) / self.cutoff)
        return int(scaled[0]), int(scaled[1]), int(scaled[2])


class PeriodicSpatialHash:
    """Small PBC-aware spatial hash for repeated solvent insertion checks."""

    def __init__(self, symbols: list[str], positions: np.ndarray, *, cell: np.ndarray, pbc: np.ndarray, cutoff: float):
        self.cell = np.asarray(cell, dtype=float)
        self.pbc = np.asarray(pbc, dtype=bool)
        self.inv_cell_t = np.linalg.inv(self.cell.T)
        self.ortho_lengths = orthorhombic_lengths(self.cell)
        self.cutoff = max(float(cutoff), 1e-6)
        lengths = np.abs(self.ortho_lengths) if self.ortho_lengths is not None else np.linalg.norm(self.cell, axis=1)
        self.grid_shape = tuple(max(1, int(math.floor(float(length) / self.cutoff))) for length in lengths)
        min_bin = min(float(length) / n for length, n in zip(lengths, self.grid_shape) if length > 1e-12)
        self.neighbor_range = 1 if self.ortho_lengths is not None else max(1, int(math.ceil(self.cutoff / max(min_bin, 1e-6))) + 1)
        neighbor_axis = range(-self.neighbor_range, self.neighbor_range + 1)
        self.neighbor_offsets = [(dx, dy, dz) for dx in neighbor_axis for dy in neighbor_axis for dz in neighbor_axis]
        self.buckets: dict[tuple[int, int, int], list[tuple[str, np.ndarray]]] = {}
        self.add_atoms(symbols, positions)

    def add_atoms(self, symbols: list[str], positions: np.ndarray) -> None:
        for symbol, position in zip(symbols, np.asarray(positions, dtype=float)):
            key = self._bucket_key(position)
            self.buckets.setdefault(key, []).append((symbol, np.asarray(position, dtype=float)))

    def has_collision(
        self,
        trial_symbols: list[str],
        trial_positions: np.ndarray,
        *,
        collision_scale: float,
        min_distance: float,
    ) -> bool:
        for symbol, position in zip(trial_symbols, np.asarray(trial_positions, dtype=float)):
            base = self._bucket_key(position)
            for dx, dy, dz in self.neighbor_offsets:
                bucket = self.buckets.get(self._offset_key(base, dx, dy, dz))
                if not bucket:
                    continue
                for other_symbol, other_position in bucket:
                    delta = minimum_image_delta_fast(position - other_position, self.cell, self.pbc, self.inv_cell_t, self.ortho_lengths)
                    cutoff = pair_cutoff(symbol, other_symbol, collision_scale, min_distance)
                    if float(np.dot(delta, delta)) < cutoff * cutoff:
                        return True
        return False

    def _nearby_atoms(self, position: np.ndarray):
        base = self._bucket_key(position)
        rx = range(-self.neighbor_range, self.neighbor_range + 1)
        for dx in rx:
            for dy in rx:
                for dz in rx:
                    key = self._normalize_key((base[0] + dx, base[1] + dy, base[2] + dz))
                    yield from self.buckets.get(key, ())

    def _bucket_key(self, position: np.ndarray) -> tuple[int, int, int]:
        position = np.asarray(position, dtype=float)
        if self.ortho_lengths is not None:
            frac = position / self.ortho_lengths
        else:
            frac = self.inv_cell_t @ position
        x = int(math.floor((frac[0] % 1.0 if self.pbc[0] else frac[0]) * self.grid_shape[0]))
        y = int(math.floor((frac[1] % 1.0 if self.pbc[1] else frac[1]) * self.grid_shape[1]))
        z = int(math.floor((frac[2] % 1.0 if self.pbc[2] else frac[2]) * self.grid_shape[2]))
        return self._normalize_key((x, y, z))

    def _normalize_key(self, key: tuple[int, int, int]) -> tuple[int, int, int]:
        x, y, z = key
        if self.pbc[0]:
            x %= self.grid_shape[0]
        if self.pbc[1]:
            y %= self.grid_shape[1]
        if self.pbc[2]:
            z %= self.grid_shape[2]
        return x, y, z

    def _offset_key(self, base: tuple[int, int, int], dx: int, dy: int, dz: int) -> tuple[int, int, int]:
        x = base[0] + dx
        y = base[1] + dy
        z = base[2] + dz
        if self.pbc[0]:
            x %= self.grid_shape[0]
        if self.pbc[1]:
            y %= self.grid_shape[1]
        if self.pbc[2]:
            z %= self.grid_shape[2]
        return x, y, z


def max_collision_cutoff(solvent_pool: list[Atoms], existing_symbols: list[str], collision_scale: float, min_distance: float) -> float:
    if min_distance > 0:
        return float(min_distance)
    solvent_symbols = [symbol for atoms in solvent_pool for symbol in atoms.get_chemical_symbols()]
    max_solvent = max(collision_radius(symbol) for symbol in solvent_symbols)
    max_existing = max(collision_radius(symbol) for symbol in existing_symbols)
    return float(collision_scale) * (max_solvent + max_existing)


def orthorhombic_lengths(cell: np.ndarray) -> np.ndarray | None:
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        return None
    diagonal = np.diag(cell)
    off_diagonal = cell - np.diag(diagonal)
    if np.any(np.abs(diagonal) <= 1e-12) or not np.all(np.abs(off_diagonal) <= 1e-12):
        return None
    return diagonal


def minimum_image_delta(delta: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) <= 1e-12 or not np.any(pbc):
        return delta
    return minimum_image_delta_fast(delta, cell, pbc, np.linalg.inv(cell.T), orthorhombic_lengths(cell))


def minimum_image_delta_fast(
    delta: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    inv_cell_t: np.ndarray,
    ortho_lengths: np.ndarray | None,
) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    if ortho_lengths is not None:
        out = delta.copy()
        for axis in range(3):
            if pbc[axis]:
                out[axis] -= np.rint(out[axis] / ortho_lengths[axis]) * ortho_lengths[axis]
        return out
    frac = inv_cell_t @ delta
    for axis in range(3):
        if pbc[axis]:
            frac[axis] -= np.rint(frac[axis])
    return frac @ cell


def pair_cutoff(left: str, right: str, collision_scale: float, min_distance: float) -> float:
    if min_distance > 0.0:
        return float(min_distance)
    return float(collision_scale) * (collision_radius(left) + collision_radius(right))


def collision_radius(symbol: str) -> float:
    return COLLISION_RADIUS.get(symbol, 1.20)


def wrap_positions(positions: np.ndarray, cell: np.ndarray, ortho_lengths: np.ndarray | None = None) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    lengths = ortho_lengths if ortho_lengths is not None else orthorhombic_lengths(cell)
    if lengths is not None:
        frac = positions / lengths
        return (frac - np.floor(frac)) * lengths
    frac = np.linalg.solve(cell.T, positions.T).T
    frac = frac - np.floor(frac)
    return frac @ cell


def estimate_solvent_count_from_density(solvent: Atoms, density: float, cell: np.ndarray, packing: float) -> int:
    if density <= 0:
        raise ValueError("Solvent Box Fill: density must be positive.")
    volume_a3 = abs(float(np.linalg.det(cell)))
    mass_g_per_mol = molecule_mass(solvent.get_chemical_symbols())
    if mass_g_per_mol <= 0:
        raise ValueError("Solvent Box Fill: solvent mass must be positive.")
    volume_cm3 = volume_a3 * 1e-24
    molecule_mass_g = mass_g_per_mol / AVOGADRO
    packing = max(0.0, min(float(packing), 1.0))
    return max(1, int(round(float(density) * volume_cm3 * packing / molecule_mass_g)))


def molecule_mass(symbols: Iterable[str]) -> float:
    total = 0.0
    for symbol in symbols:
        number = atomic_numbers.get(symbol)
        if number is None:
            raise ValueError(f"Solvation: unknown element '{symbol}'.")
        total += float(atomic_masses[number])
    return total


def apply_auto_box(atoms: Atoms, padding: float, min_box: float) -> None:
    positions = np.asarray(atoms.get_positions(), dtype=float)
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    lengths = np.maximum(hi - lo + 2.0 * float(padding), float(min_box))
    atoms.set_positions(positions - lo + float(padding))
    atoms.set_cell(np.diag(lengths))
    atoms.set_pbc(False)


def center_in_fixed_box(atoms: Atoms, box_size: float) -> None:
    if box_size <= 0.0:
        raise ValueError("Local Solvation: box_size must be positive.")
    positions = np.asarray(atoms.get_positions(), dtype=float)
    center = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    atoms.set_positions(positions - center + 0.5 * float(box_size))
    atoms.set_cell(np.diag([float(box_size), float(box_size), float(box_size)]))
    atoms.set_pbc(False)


def has_valid_cell(atoms: Atoms) -> bool:
    cell = np.asarray(atoms.cell.array, dtype=float)
    return cell.shape == (3, 3) and abs(float(np.linalg.det(cell))) > 1e-12
