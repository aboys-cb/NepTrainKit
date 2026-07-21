"""UI-independent dataset filtering operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import neighbor_list

from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.io import (
    allocate_sqrt_quotas,
    centered_fps,
    farthest_point_sampling,
    structure_element_set_key,
)
from NepTrainKit.core.io.importers import import_structures
from NepTrainKit.core.types import NepBackend

from .operation import DatasetOperation


@dataclass(frozen=True)
class FPSFilterParams:
    """Parameters for descriptor-space farthest point sampling."""

    nep_path: str
    n_samples: int = 100
    min_distance: float = 0.01
    backend: str = "auto"
    chunk_max_atoms: int = 100000
    strategy: str = "global"
    existing_dataset_path: str = ""


@dataclass(frozen=True)
class FPSGroupReport:
    """Candidate, warm-start, and selected counts for one element set."""

    candidate_count: int
    existing_count: int
    selected_count: int


class FPSFilterOperation(DatasetOperation):
    """Select representative structures using NEP descriptors and FPS."""

    VALID_STRATEGIES = {"global", "element_set"}

    def __init__(self) -> None:
        self.last_group_report: dict[tuple[str, ...], FPSGroupReport] = {}

    def run_dataset(self, dataset, params: FPSFilterParams) -> list:
        self.last_group_report = {}
        if not dataset:
            return []
        strategy = str(params.strategy).strip().lower()
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unsupported FPS strategy '{params.strategy}'. "
                f"Expected one of {sorted(self.VALID_STRATEGIES)}."
            )
        nep_path = Path(params.nep_path)
        if not nep_path.exists():
            raise FileNotFoundError(f"NEP file does not exist: {nep_path}")

        nep_calc = NepCalculator(
            model_file=str(nep_path),
            backend=NepBackend(params.backend),
            chunk_max_atoms=int(params.chunk_max_atoms),
        )
        desc_array = nep_calc.descriptors(dataset)
        if strategy == "element_set":
            return self._run_element_set_fps(dataset, desc_array, nep_calc, params)

        remaining_indices = farthest_point_sampling(
            desc_array,
            n_samples=int(params.n_samples),
            min_dist=float(params.min_distance),
        )
        return [dataset[i] for i in remaining_indices]

    def _run_element_set_fps(self, dataset, descriptors, nep_calc, params: FPSFilterParams) -> list:
        groups = self.group_indices_by_element_set(dataset)
        quotas = self.allocate_sqrt_quotas(
            {key: len(indices) for key, indices in groups.items()},
            int(params.n_samples),
        )
        existing_groups: dict[tuple[str, ...], list] = {}
        existing_descriptors = np.empty((0, descriptors.shape[1]), dtype=float)
        existing_structures: list = []
        if params.existing_dataset_path.strip():
            existing_path = Path(params.existing_dataset_path).expanduser()
            if not existing_path.exists():
                raise FileNotFoundError(f"Existing training dataset does not exist: {existing_path}")
            existing_structures = list(import_structures(existing_path))
            if not existing_structures:
                raise ValueError(f"Existing training dataset contains no structures: {existing_path}")
            existing_descriptors = nep_calc.descriptors(existing_structures)
            existing_groups = self.group_indices_by_element_set(existing_structures)

        selected_global_indices: list[int] = []
        for key in sorted(groups):
            candidate_indices = groups[key]
            candidate_descriptors = np.asarray(descriptors[candidate_indices], dtype=float)
            warm_indices = existing_groups.get(key, [])
            warm_descriptors = (
                np.asarray(existing_descriptors[warm_indices], dtype=float)
                if warm_indices
                else None
            )
            local_indices = self.centered_fps(
                candidate_descriptors,
                n_samples=quotas[key],
                min_dist=float(params.min_distance),
                selected_data=warm_descriptors,
            )
            chosen = [candidate_indices[index] for index in local_indices]
            selected_global_indices.extend(chosen)
            self.last_group_report[key] = FPSGroupReport(
                candidate_count=len(candidate_indices),
                existing_count=len(warm_indices),
                selected_count=len(chosen),
            )

        selected_set = set(selected_global_indices)
        return [structure for index, structure in enumerate(dataset) if index in selected_set]

    @staticmethod
    def element_set_key(structure) -> tuple[str, ...]:
        """Return a stable element-set key for a structure."""
        return structure_element_set_key(structure)

    @classmethod
    def group_indices_by_element_set(cls, structures) -> dict[tuple[str, ...], list[int]]:
        """Group structure indices by their set of chemical elements."""
        groups: dict[tuple[str, ...], list[int]] = {}
        for index, structure in enumerate(structures):
            groups.setdefault(cls.element_set_key(structure), []).append(index)
        return groups

    @staticmethod
    def allocate_sqrt_quotas(
        group_sizes: dict,
        n_samples: int,
    ) -> dict:
        """Allocate one slot per group, then distribute the rest by sqrt(size)."""
        return allocate_sqrt_quotas(group_sizes, n_samples)

    @staticmethod
    def centered_fps(
        points,
        n_samples: int,
        min_dist: float,
        selected_data=None,
    ) -> list[int]:
        """Run FPS from the feature-space center, or from a warm-start set."""
        return centered_fps(points, n_samples, min_dist, selected_data=selected_data)


@dataclass(frozen=True)
class GeometryFilterParams:
    """Parameters for explicit geometry-quality filtering."""

    min_pair_distance: float = 1.0
    min_volume_per_atom: float = 0.0
    max_volume_per_atom: float = 0.0
    min_density: float = 0.0
    max_density: float = 0.0
    require_finite_cell: bool = False


class GeometryFilterOperation(DatasetOperation):
    """Reject structures that violate explicit distance, volume, or density bounds."""

    AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660

    def run_dataset(self, dataset, params: GeometryFilterParams) -> list:
        return [structure for structure in dataset if self.keep_structure(structure, params)]

    @classmethod
    def keep_structure(cls, structure, params: GeometryFilterParams) -> bool:
        natoms = len(structure)
        if natoms <= 0:
            return False

        volume = float(structure.get_volume())
        checks_need_cell = (
            bool(params.require_finite_cell)
            or float(params.min_volume_per_atom) > 0.0
            or float(params.max_volume_per_atom) > 0.0
            or float(params.min_density) > 0.0
            or float(params.max_density) > 0.0
        )
        if checks_need_cell and volume <= 0.0:
            return False

        if bool(params.require_finite_cell):
            cell = np.asarray(structure.cell.array, dtype=float)
            if cell.shape != (3, 3) or not np.all(np.isfinite(cell)) or abs(float(np.linalg.det(cell))) <= 1e-12:
                return False

        min_pair_distance = float(params.min_pair_distance)
        if min_pair_distance > 0.0 and natoms > 1 and cls.has_pair_closer_than(structure, min_pair_distance):
            return False

        if volume > 0.0:
            volume_per_atom = volume / float(natoms)
            if float(params.min_volume_per_atom) > 0.0 and volume_per_atom < float(params.min_volume_per_atom):
                return False
            if float(params.max_volume_per_atom) > 0.0 and volume_per_atom > float(params.max_volume_per_atom):
                return False

            density = cls.mass_density(structure, volume)
            if float(params.min_density) > 0.0 and density < float(params.min_density):
                return False
            if float(params.max_density) > 0.0 and density > float(params.max_density):
                return False

        return True

    @staticmethod
    def shortest_pair_distance(structure) -> float:
        distances = np.asarray(structure.get_all_distances(mic=True), dtype=float)
        if distances.shape[0] < 2:
            return float("inf")
        distances[distances <= 1e-12] = np.inf
        return float(np.min(distances))

    @staticmethod
    def has_pair_closer_than(structure, cutoff: float) -> bool:
        try:
            indices = neighbor_list("i", structure, float(cutoff), self_interaction=False)
        except Exception:
            return GeometryFilterOperation.shortest_pair_distance(structure) < float(cutoff)
        return bool(len(indices) > 0)

    @classmethod
    def mass_density(cls, structure, volume: float) -> float:
        if float(volume) <= 0.0:
            raise ValueError("GeometryFilter mass_density requires positive volume.")
        total_mass = 0.0
        for symbol in structure.get_chemical_symbols():
            if symbol not in atomic_numbers:
                raise ValueError(f"Unknown chemical symbol '{symbol}'.")
            total_mass += float(atomic_masses[atomic_numbers[symbol]])
        return total_mass / float(volume) * cls.AMU_PER_A3_TO_G_PER_CM3
