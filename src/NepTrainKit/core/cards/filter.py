"""UI-independent dataset filtering operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.io import farthest_point_sampling
from NepTrainKit.core.types import NepBackend

from .operation import DatasetOperation


@dataclass(frozen=True)
class FPSFilterParams:
    """Parameters for descriptor-space farthest point sampling."""

    nep_path: str
    n_samples: int = 100
    min_distance: float = 0.01
    backend: str = "auto"
    batch_size: int = 1000


class FPSFilterOperation(DatasetOperation):
    """Select representative structures using NEP descriptors and FPS."""

    def run_dataset(self, dataset, params: FPSFilterParams) -> list:
        nep_path = Path(params.nep_path)
        if not nep_path.exists():
            raise FileNotFoundError(f"NEP file does not exist: {nep_path}")

        nep_calc = NepCalculator(
            model_file=str(nep_path),
            backend=NepBackend(params.backend),
            batch_size=int(params.batch_size),
        )
        desc_array = nep_calc.get_structures_descriptor(dataset)
        remaining_indices = farthest_point_sampling(
            desc_array,
            n_samples=int(params.n_samples),
            min_dist=float(params.min_distance),
        )
        return [dataset[i] for i in remaining_indices]


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
        if min_pair_distance > 0.0 and natoms > 1 and cls.shortest_pair_distance(structure) < min_pair_distance:
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

    @classmethod
    def mass_density(cls, structure, volume: float) -> float:
        total_mass = 0.0
        for symbol in structure.get_chemical_symbols():
            total_mass += float(atomic_masses[atomic_numbers[symbol]])
        return total_mass / float(volume) * cls.AMU_PER_A3_TO_G_PER_CM3
