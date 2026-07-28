"""UI-independent dataset filtering operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import neighbor_list

from NepTrainKit.core.audit.neighbor_scan import find_short_distance_structure_rows
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
    min_distance: float = 0.0
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

    @staticmethod
    def _integer(value: object, name: str, *, minimum: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"FPS Filter: {name} must be an integer.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"FPS Filter: {name} must be an integer.") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"FPS Filter: {name} must be an integer.")
        result = int(numeric)
        if result < minimum:
            raise ValueError(f"FPS Filter: {name} must be >= {minimum}.")
        return result

    @staticmethod
    def _finite(value: object, name: str, *, minimum: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"FPS Filter: {name} must be a finite number.") from exc
        if not np.isfinite(result):
            raise ValueError(f"FPS Filter: {name} must be a finite number.")
        if result < minimum:
            raise ValueError(f"FPS Filter: {name} must be >= {minimum:g}.")
        return result

    @classmethod
    def _validated_settings(
        cls,
        dataset,
        params: FPSFilterParams,
        *,
        require_model: bool,
    ) -> dict:
        n_samples = cls._integer(params.n_samples, "n_samples", minimum=1)
        min_distance = cls._finite(
            params.min_distance,
            "min_distance",
            minimum=0.0,
        )
        chunk_max_atoms = cls._integer(
            params.chunk_max_atoms,
            "chunk_max_atoms",
            minimum=1,
        )
        strategy = str(params.strategy).strip().lower()
        if strategy not in cls.VALID_STRATEGIES:
            raise ValueError(
                f"Unsupported FPS strategy '{params.strategy}'. "
                f"Expected one of {sorted(cls.VALID_STRATEGIES)}."
            )
        try:
            backend = NepBackend(str(params.backend).strip().lower())
        except ValueError as exc:
            raise ValueError(
                "FPS Filter: backend must be auto, cpu, or cuda."
            ) from exc
        empty_indices = [index for index, structure in enumerate(dataset) if len(structure) < 1]
        if empty_indices:
            raise ValueError(
                f"FPS Filter: input structure {empty_indices[0] + 1} contains no atoms."
            )

        model_path = Path(str(params.nep_path)).expanduser()
        if require_model and not model_path.is_file():
            raise FileNotFoundError(f"NEP file does not exist: {model_path}")
        existing_text = str(params.existing_dataset_path).strip()
        existing_path = Path(existing_text).expanduser() if existing_text else None
        return {
            "n_samples": n_samples,
            "min_distance": min_distance,
            "chunk_max_atoms": chunk_max_atoms,
            "strategy": strategy,
            "backend": backend,
            "model_path": model_path,
            "existing_path": existing_path,
        }

    @staticmethod
    def _validate_descriptors(descriptors, expected_rows: int, label: str) -> np.ndarray:
        array = np.asarray(descriptors, dtype=float)
        if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] < 1:
            raise ValueError(
                f"FPS Filter: {label} descriptors must have shape "
                f"({expected_rows}, D) with D >= 1."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"FPS Filter: {label} descriptors contain NaN/Inf.")
        return array

    @classmethod
    def selection_summary(cls, dataset, params: FPSFilterParams) -> dict:
        """Return the count and group-quota plan without calculating descriptors."""
        structures = list(dataset) if dataset is not None else []
        settings = cls._validated_settings(
            structures,
            params,
            require_model=False,
        )
        groups = cls.group_indices_by_element_set(structures)
        quotas = (
            cls.allocate_sqrt_quotas(
                {key: len(indices) for key, indices in groups.items()},
                settings["n_samples"],
            )
            if settings["strategy"] == "element_set" and structures
            else {}
        )
        return {
            "input_count": len(structures),
            "max_output": min(settings["n_samples"], len(structures)),
            "strategy": settings["strategy"],
            "group_count": len(groups),
            "quotas": quotas,
            "model_exists": settings["model_path"].is_file(),
            "model_name": settings["model_path"].name or str(settings["model_path"]),
            "existing_path": settings["existing_path"],
            "min_distance": settings["min_distance"],
        }

    def run_dataset(self, dataset, params: FPSFilterParams) -> list:
        self.last_group_report = {}
        if not dataset:
            return []
        structures = list(dataset)
        settings = self._validated_settings(
            structures,
            params,
            require_model=True,
        )

        nep_calc = NepCalculator(
            model_file=str(settings["model_path"]),
            backend=settings["backend"],
            chunk_max_atoms=settings["chunk_max_atoms"],
        )
        desc_array = self._validate_descriptors(
            nep_calc.descriptors(structures),
            len(structures),
            "candidate",
        )
        existing_structures: list = []
        existing_descriptors = None
        existing_path = settings["existing_path"]
        if existing_path is not None:
            if not existing_path.exists():
                raise FileNotFoundError(
                    f"Existing training dataset does not exist: {existing_path}"
                )
            existing_structures = list(import_structures(existing_path))
            if not existing_structures:
                raise ValueError(
                    f"Existing training dataset contains no structures: {existing_path}"
                )
            if any(len(structure) < 1 for structure in existing_structures):
                raise ValueError(
                    "FPS Filter: existing training dataset contains an empty structure."
                )
            existing_descriptors = self._validate_descriptors(
                nep_calc.descriptors(existing_structures),
                len(existing_structures),
                "existing",
            )
            if existing_descriptors.shape[1] != desc_array.shape[1]:
                raise ValueError(
                    "FPS Filter: candidate and existing descriptor dimensions differ."
                )

        if settings["strategy"] == "element_set":
            return self._run_element_set_fps(
                structures,
                desc_array,
                settings["n_samples"],
                settings["min_distance"],
                existing_structures,
                existing_descriptors,
            )

        remaining_indices = farthest_point_sampling(
            desc_array,
            n_samples=settings["n_samples"],
            min_dist=settings["min_distance"],
            selected_data=existing_descriptors,
        )
        return [structures[i] for i in remaining_indices]

    def _run_element_set_fps(
        self,
        dataset,
        descriptors,
        n_samples: int,
        min_distance: float,
        existing_structures,
        existing_descriptors,
    ) -> list:
        groups = self.group_indices_by_element_set(dataset)
        quotas = self.allocate_sqrt_quotas(
            {key: len(indices) for key, indices in groups.items()},
            n_samples,
        )
        existing_groups: dict[tuple[str, ...], list] = {}
        if existing_structures:
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
                min_dist=min_distance,
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

    min_pair_distance: float = 0.0
    min_volume_per_atom: float = 0.0
    max_volume_per_atom: float = 0.0
    min_density: float = 0.0
    max_density: float = 0.0
    require_finite_cell: bool = False


class GeometryFilterOperation(DatasetOperation):
    """Reject structures that violate explicit distance, volume, or density bounds."""

    AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660

    @staticmethod
    def _finite_threshold(value: object, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Geometry Filter: {name} must be a finite non-negative number."
            ) from exc
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(
                f"Geometry Filter: {name} must be a finite non-negative number."
            )
        return result

    @classmethod
    def _validated_params(cls, params: GeometryFilterParams) -> GeometryFilterParams:
        normalized = GeometryFilterParams(
            min_pair_distance=cls._finite_threshold(
                params.min_pair_distance,
                "min_pair_distance",
            ),
            min_volume_per_atom=cls._finite_threshold(
                params.min_volume_per_atom,
                "min_volume_per_atom",
            ),
            max_volume_per_atom=cls._finite_threshold(
                params.max_volume_per_atom,
                "max_volume_per_atom",
            ),
            min_density=cls._finite_threshold(
                params.min_density,
                "min_density",
            ),
            max_density=cls._finite_threshold(
                params.max_density,
                "max_density",
            ),
            require_finite_cell=bool(params.require_finite_cell),
        )
        if (
            normalized.min_volume_per_atom > 0.0
            and normalized.max_volume_per_atom > 0.0
            and normalized.min_volume_per_atom > normalized.max_volume_per_atom
        ):
            raise ValueError(
                "Geometry Filter: minimum volume/atom must not exceed maximum volume/atom."
            )
        if (
            normalized.min_density > 0.0
            and normalized.max_density > 0.0
            and normalized.min_density > normalized.max_density
        ):
            raise ValueError(
                "Geometry Filter: minimum density must not exceed maximum density."
            )
        return normalized

    def run_dataset(self, dataset, params: GeometryFilterParams) -> list:
        normalized = self._validated_params(params)
        structures = list(dataset) if dataset is not None else []
        reasons = self._batch_rejection_reasons(structures, normalized)
        return [
            structure
            for structure, reason in zip(structures, reasons)
            if reason is None
        ]

    @classmethod
    def keep_structure(cls, structure, params: GeometryFilterParams) -> bool:
        normalized = cls._validated_params(params)
        return cls._rejection_reason(structure, normalized) is None

    @classmethod
    def _rejection_reason(
        cls,
        structure,
        params: GeometryFilterParams,
        *,
        pair_is_close: bool | None = None,
    ) -> str | None:
        natoms = len(structure)
        if natoms <= 0:
            return "empty"

        positions = np.asarray(structure.get_positions(), dtype=float)
        if positions.shape != (natoms, 3) or not np.all(np.isfinite(positions)):
            return "nonfinite_positions"

        cell = np.asarray(structure.cell.array, dtype=float)
        valid_cell = (
            cell.shape == (3, 3)
            and np.all(np.isfinite(cell))
            and abs(float(np.linalg.det(cell))) > 1e-12
        )
        volume = abs(float(np.linalg.det(cell))) if valid_cell else 0.0
        checks_need_cell = (
            bool(params.require_finite_cell)
            or params.min_volume_per_atom > 0.0
            or params.max_volume_per_atom > 0.0
            or params.min_density > 0.0
            or params.max_density > 0.0
        )
        if checks_need_cell and not valid_cell:
            return "invalid_cell"

        if (
            params.min_pair_distance > 0.0
            and natoms > 1
            and (
                pair_is_close
                if pair_is_close is not None
                else cls.has_pair_closer_than(structure, params.min_pair_distance)
            )
        ):
            return "pair_distance"

        if volume > 0.0:
            volume_per_atom = volume / float(natoms)
            if params.min_volume_per_atom > 0.0 and volume_per_atom < params.min_volume_per_atom:
                return "volume_per_atom"
            if params.max_volume_per_atom > 0.0 and volume_per_atom > params.max_volume_per_atom:
                return "volume_per_atom"

            density = cls.mass_density(structure, volume)
            if params.min_density > 0.0 and density < params.min_density:
                return "density"
            if params.max_density > 0.0 and density > params.max_density:
                return "density"

        return None

    @classmethod
    def _batch_rejection_reasons(
        cls,
        structures: list,
        params: GeometryFilterParams,
    ) -> list[str | None]:
        """Return first-failure reasons with one native pair scan per dataset."""
        close_pair_rows: set[int] = set()
        if params.min_pair_distance > 0.0:
            checks_need_cell = (
                bool(params.require_finite_cell)
                or params.min_volume_per_atom > 0.0
                or params.max_volume_per_atom > 0.0
                or params.min_density > 0.0
                or params.max_density > 0.0
            )
            source_rows: list[int] = []
            positions_by_structure: list[np.ndarray] = []
            cells: list[np.ndarray] = []
            pbc_flags: list[np.ndarray] = []
            for source_row, structure in enumerate(structures):
                natoms = len(structure)
                if natoms <= 1:
                    continue
                positions = np.asarray(structure.get_positions(), dtype=float)
                if (
                    positions.shape != (natoms, 3)
                    or not np.all(np.isfinite(positions))
                ):
                    continue
                cell = np.asarray(structure.cell.array, dtype=float)
                valid_cell = (
                    cell.shape == (3, 3)
                    and np.all(np.isfinite(cell))
                    and abs(float(np.linalg.det(cell))) > 1e-12
                )
                if checks_need_cell and not valid_cell:
                    continue
                source_rows.append(source_row)
                positions_by_structure.append(positions)
                cells.append(cell)
                pbc_flags.append(np.asarray(structure.pbc, dtype=bool))

            if source_rows:
                # The card rejects distances strictly below the threshold.
                # The shared native primitive uses an inclusive cutoff, so move
                # by one representable float to preserve the existing contract.
                strict_cutoff = float(
                    np.nextafter(params.min_pair_distance, -np.inf)
                )
                relative_rows = find_short_distance_structure_rows(
                    positions_by_structure,
                    cells,
                    pbc_flags,
                    strict_cutoff,
                )
                close_pair_rows = {
                    source_rows[relative_row]
                    for relative_row in relative_rows
                }

        return [
            cls._rejection_reason(
                structure,
                params,
                pair_is_close=(source_row in close_pair_rows),
            )
            for source_row, structure in enumerate(structures)
        ]

    @classmethod
    def filter_summary(cls, dataset, params: GeometryFilterParams) -> dict:
        """Return kept and first-failure counts without modifying structures."""
        normalized = cls._validated_params(params)
        reasons = {
            "empty": 0,
            "nonfinite_positions": 0,
            "invalid_cell": 0,
            "pair_distance": 0,
            "volume_per_atom": 0,
            "density": 0,
        }
        kept = 0
        structures = list(dataset) if dataset is not None else []
        for reason in cls._batch_rejection_reasons(structures, normalized):
            if reason is None:
                kept += 1
            else:
                reasons[reason] += 1
        return {
            "input_count": len(structures),
            "kept_count": kept,
            "rejected_count": len(structures) - kept,
            "reasons": reasons,
        }

    @staticmethod
    def shortest_pair_distance(structure) -> float:
        distances = np.asarray(structure.get_all_distances(mic=True), dtype=float)
        if distances.shape[0] < 2:
            return float("inf")
        upper = distances[np.triu_indices(distances.shape[0], k=1)]
        if not np.all(np.isfinite(upper)):
            return float("nan")
        return float(np.min(upper))

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
