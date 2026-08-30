"""NepTrainKit policy layer over the public :mod:`nep_adapters` interface."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress

from NepTrainKit.core.adapter_api import (
    AdapterCalculator,
    BackendStatus,
    BackendUnavailableError,
    ChargePrediction,
    ModelInfo,
    OutOfMemoryError,
    Prediction,
    SpinPrediction,
    UnsupportedModelError,
    backend_status,
)
from NepTrainKit.core.types import NepBackend, parse_nep_backend
from NepTrainKit.core.utils import aggregate_per_atom_to_structure
from NepTrainKit.paths import PathLike, as_path
from NepTrainKit.utils import timeit


ProgressCallback = Callable[[int, int], None]
DEFAULT_CHUNK_MAX_ATOMS = 100_000


@dataclass(frozen=True)
class BackendSelection:
    """One immutable backend decision made while loading a model."""

    requested: NepBackend
    resolved: NepBackend
    cuda_status: BackendStatus
    reason: str

    @property
    def summary(self) -> str:
        if self.requested is NepBackend.AUTO:
            return f"Auto → {self.resolved.value.upper()}"
        return self.resolved.value.upper()


def _structure_list(structures):
    if hasattr(structures, "get_chemical_symbols") or hasattr(structures, "positions"):
        return [structures]
    if isinstance(structures, list):
        return structures
    return list(structures)


def _merge_predictions(parts: Sequence[Prediction]) -> Prediction:
    if not parts:
        raise ValueError("prediction parts must not be empty")
    cls = type(parts[0])
    common = dict(
        energy=np.concatenate([part.energy for part in parts]),
        potential=np.concatenate([part.potential for part in parts]),
        forces=np.concatenate([part.forces for part in parts], axis=0),
        virials=np.concatenate([part.virials for part in parts], axis=0),
        structure_virials=np.concatenate(
            [part.structure_virials for part in parts], axis=0
        ),
        atom_counts=np.concatenate([part.atom_counts for part in parts]),
    )
    if cls is ChargePrediction:
        return ChargePrediction(
            **common,
            charges=np.concatenate([part.charges for part in parts]),
            becs=np.concatenate([part.becs for part in parts], axis=0),
        )
    if cls is SpinPrediction:
        return SpinPrediction(
            **common,
            mforces=np.concatenate([part.mforces for part in parts], axis=0),
        )
    return Prediction(**common)


class NepCalculator:
    """Select one backend, then expose typed NEP predictions to the application."""

    def __init__(
        self,
        model_file: PathLike = "nep.txt",
        backend: NepBackend | str | None = None,
        chunk_max_atoms: int | None = None,
    ) -> None:
        self.model_path = as_path(model_file)
        self.requested_backend = parse_nep_backend(backend)
        self.chunk_max_atoms = int(chunk_max_atoms or DEFAULT_CHUNK_MAX_ATOMS)
        if self.chunk_max_atoms <= 0:
            raise ValueError("chunk_max_atoms must be positive")

        self.cuda_status = backend_status("cuda")
        self._cpu_operation_calculator: AdapterCalculator | None = None
        self._calculator, self.selection = self._select_backend()
        self.backend = self.selection.resolved
        self.model_info: ModelInfo = self._calculator.model_info
        self.is_charge_model = self.model_info.model_type == "charge"
        self.is_spin_model = self.model_info.model_type == "spin"
        self.element_list = list(self.model_info.elements)
        self.type_dict = {
            element: index for index, element in enumerate(self.element_list)
        }
        self.initialized = True

    def _select_backend(self) -> tuple[AdapterCalculator, BackendSelection]:
        if self.requested_backend is NepBackend.CPU:
            return (
                AdapterCalculator(self.model_path, backend="cpu"),
                BackendSelection(
                    requested=self.requested_backend,
                    resolved=NepBackend.CPU,
                    cuda_status=self.cuda_status,
                    reason="cpu_requested",
                ),
            )

        if self.requested_backend is NepBackend.CUDA:
            if not self.cuda_status.available:
                raise BackendUnavailableError(
                    "CUDA was requested but is unavailable: "
                    f"{self.cuda_status.detail} Select CPU or install a Linux "
                    "CPU+CUDA nep-adapters wheel.",
                    backend="cuda",
                    operation="select_backend",
                )
            return (
                AdapterCalculator(self.model_path, backend="cuda"),
                BackendSelection(
                    requested=self.requested_backend,
                    resolved=NepBackend.CUDA,
                    cuda_status=self.cuda_status,
                    reason="cuda_requested",
                ),
            )

        if self.cuda_status.available:
            try:
                calculator = AdapterCalculator(self.model_path, backend="cuda")
            except (BackendUnavailableError, UnsupportedModelError) as error:
                reason = f"cuda_model_unavailable:{error.code}"
            else:
                return (
                    calculator,
                    BackendSelection(
                        requested=NepBackend.AUTO,
                        resolved=NepBackend.CUDA,
                        cuda_status=self.cuda_status,
                        reason="cuda_available",
                    ),
                )
        else:
            reason = f"cuda_unavailable:{self.cuda_status.reason}"

        return (
            AdapterCalculator(self.model_path, backend="cpu"),
            BackendSelection(
                requested=NepBackend.AUTO,
                resolved=NepBackend.CPU,
                cuda_status=self.cuda_status,
                reason=reason,
            ),
        )

    def close(self) -> None:
        self._calculator.close()
        if self._cpu_operation_calculator is not None:
            self._cpu_operation_calculator.close()
            self._cpu_operation_calculator = None

    def cancel(self) -> None:
        self._calculator.cancel()
        if self._cpu_operation_calculator is not None:
            self._cpu_operation_calculator.cancel()

    def reset_cancel(self) -> None:
        self._calculator.reset_cancel()
        if self._cpu_operation_calculator is not None:
            self._cpu_operation_calculator.reset_cancel()

    def _cpu_calculator_for_operation(self) -> AdapterCalculator:
        """Return the CPU engine required by CPU-only calculation families."""
        if self.backend is NepBackend.CPU:
            return self._calculator
        if self._cpu_operation_calculator is None:
            self._cpu_operation_calculator = AdapterCalculator(
                self.model_path, backend="cpu"
            )
        return self._cpu_operation_calculator

    def _effective_chunk_max_atoms(self, structures: Sequence[Atoms]) -> int:
        maximum = self.chunk_max_atoms
        if self.backend is not NepBackend.CUDA:
            return maximum
        recommended = self._calculator.recommend_max_atoms()
        if recommended is None:
            return maximum
        largest_structure = max((len(structure) for structure in structures), default=0)
        if largest_structure > recommended:
            raise OutOfMemoryError(
                "The largest structure has "
                f"{largest_structure} atoms, above the CUDA workspace estimate "
                f"of {recommended}. Use CPU for this dataset; a single structure "
                "cannot be split safely.",
                backend="cuda",
                operation="plan_chunks",
            )
        return min(maximum, recommended)

    @staticmethod
    def _chunks(
        structures: Sequence[Atoms],
        max_atoms: int,
    ) -> list[list[Atoms]]:
        chunks: list[list[Atoms]] = []
        current: list[Atoms] = []
        current_atoms = 0
        for structure in structures:
            natoms = len(structure)
            if current and current_atoms + natoms > max_atoms:
                chunks.append(current)
                current = []
                current_atoms = 0
            current.append(structure)
            current_atoms += natoms
        if current:
            chunks.append(current)
        return chunks

    def _predict_chunk(self, structures: Sequence[Atoms]) -> Prediction:
        if self.is_spin_model:
            return self._calculator.predict_spin_structures(structures)
        if self.is_charge_model:
            return self._calculator.predict_charge_structures(structures)
        return self._calculator.predict_structures(structures)

    def _predict_with_oom_split(
        self,
        structures: Sequence[Atoms],
        *,
        operation: Callable[[Sequence[Atoms]], Prediction],
        progress: ProgressCallback | None,
        completed: list[int],
        total: int,
    ) -> list[Prediction]:
        try:
            prediction = operation(structures)
        except OutOfMemoryError:
            if len(structures) <= 1:
                raise
            midpoint = len(structures) // 2
            return self._predict_with_oom_split(
                structures[:midpoint],
                operation=operation,
                progress=progress,
                completed=completed,
                total=total,
            ) + self._predict_with_oom_split(
                structures[midpoint:],
                operation=operation,
                progress=progress,
                completed=completed,
                total=total,
            )
        completed[0] += len(structures)
        if progress is not None:
            progress(completed[0], total)
        return [prediction]

    def _run_prediction(
        self,
        structures,
        operation: Callable[[Sequence[Atoms]], Prediction],
        *,
        progress: ProgressCallback | None = None,
        cpu_only: bool = False,
    ) -> Prediction:
        structure_list = _structure_list(structures)
        if not structure_list:
            return operation([])
        max_atoms = (
            self.chunk_max_atoms
            if cpu_only
            else self._effective_chunk_max_atoms(structure_list)
        )
        parts: list[Prediction] = []
        completed = [0]
        for chunk in self._chunks(structure_list, max_atoms):
            parts.extend(
                self._predict_with_oom_split(
                    chunk,
                    operation=operation,
                    progress=progress,
                    completed=completed,
                    total=len(structure_list),
                )
            )
        return _merge_predictions(parts)

    @timeit
    def predict(
        self,
        structures: Iterable[Atoms] | Atoms,
        *,
        progress: ProgressCallback | None = None,
    ) -> Prediction:
        return self._run_prediction(
            structures,
            self._predict_chunk,
            progress=progress,
        )

    def _array_with_oom_split(
        self,
        structures: Sequence[Atoms],
        operation: Callable[[Sequence[Atoms]], np.ndarray],
    ) -> list[np.ndarray]:
        try:
            return [np.asarray(operation(structures), dtype=np.float64)]
        except OutOfMemoryError:
            if len(structures) <= 1:
                raise
            midpoint = len(structures) // 2
            return self._array_with_oom_split(
                structures[:midpoint], operation
            ) + self._array_with_oom_split(structures[midpoint:], operation)

    def _prediction_and_descriptors_with_oom_split(
        self,
        structures: Sequence[Atoms],
    ) -> list[tuple[Prediction, np.ndarray]]:
        try:
            if (
                not self.is_spin_model
                and not self.is_charge_model
                and self.model_info.supports("evaluate_with_descriptors")
            ):
                prediction, descriptors = (
                    self._calculator.predict_with_descriptors_structures(
                        structures
                    )
                )
                # Preserve the existing descriptor contract: the adapter-level
                # values are narrowed to float32 before NepTrainKit aggregates
                # them per structure.
                descriptors = np.asarray(descriptors, dtype=np.float32)
            else:
                prediction = self._predict_chunk(structures)
                descriptor_operation = (
                    self._calculator.predict_spin_descriptors
                    if self.is_spin_model
                    else lambda items: self._calculator.get_structures_descriptor(
                        items, mean_descriptor=False
                    )
                )
                descriptors = descriptor_operation(structures)
            return [
                (
                    prediction,
                    np.asarray(descriptors, dtype=np.float64),
                )
            ]
        except OutOfMemoryError:
            if len(structures) <= 1:
                raise
            midpoint = len(structures) // 2
            return self._prediction_and_descriptors_with_oom_split(
                structures[:midpoint]
            ) + self._prediction_and_descriptors_with_oom_split(
                structures[midpoint:]
            )

    @timeit
    def predict_with_descriptors(
        self,
        structures: Iterable[Atoms] | Atoms,
        *,
        mean: bool = True,
        progress: ProgressCallback | None = None,
    ) -> tuple[Prediction, npt.NDArray[np.float64]]:
        """Return prediction and descriptors while sharing CPU descriptor work."""
        structure_list = _structure_list(structures)
        if not structure_list:
            prediction = self._predict_chunk([])
            descriptors = np.empty(
                (0, self.model_info.descriptor_dim), dtype=np.float64
            )
            return prediction, descriptors

        parts: list[Prediction] = []
        descriptor_parts: list[np.ndarray] = []
        completed = 0
        max_atoms = self._effective_chunk_max_atoms(structure_list)
        for chunk in self._chunks(structure_list, max_atoms):
            for prediction, descriptors in (
                self._prediction_and_descriptors_with_oom_split(chunk)
            ):
                parts.append(prediction)
                descriptor_parts.append(descriptors)
                completed += len(prediction.atom_counts)
                if progress is not None:
                    progress(completed, len(structure_list))

        prediction = _merge_predictions(parts)
        per_atom = np.concatenate(descriptor_parts, axis=0)
        if not mean:
            return prediction, per_atom
        return prediction, aggregate_per_atom_to_structure(
            per_atom,
            [len(structure) for structure in structure_list],
            map_func=np.mean,
            axis=0,
        )

    @timeit
    def descriptors(
        self,
        structures: Iterable[Atoms] | Atoms,
        *,
        mean: bool = True,
        progress: ProgressCallback | None = None,
    ) -> npt.NDArray[np.float64]:
        structure_list = _structure_list(structures)
        if not structure_list:
            return np.empty((0, self.model_info.descriptor_dim), dtype=np.float64)
        max_atoms = self._effective_chunk_max_atoms(structure_list)
        blocks: list[np.ndarray] = []
        completed = 0
        for chunk in self._chunks(structure_list, max_atoms):
            operation = (
                self._calculator.predict_spin_descriptors
                if self.is_spin_model
                else lambda items: self._calculator.get_structures_descriptor(
                    items, mean_descriptor=False
                )
            )
            blocks.extend(self._array_with_oom_split(chunk, operation))
            completed += len(chunk)
            if progress is not None:
                progress(completed, len(structure_list))
        per_atom = np.concatenate(blocks, axis=0)
        if not mean:
            return per_atom
        return aggregate_per_atom_to_structure(
            per_atom,
            [len(structure) for structure in structure_list],
            map_func=np.mean,
            axis=0,
        )

    def dipoles(self, structures: Iterable[Atoms] | Atoms) -> npt.NDArray[np.float64]:
        structure_list = _structure_list(structures)
        if not structure_list:
            return np.empty((0, 3), dtype=np.float64)
        blocks = []
        for chunk in self._chunks(
            structure_list, self._effective_chunk_max_atoms(structure_list)
        ):
            blocks.extend(
                self._array_with_oom_split(
                    chunk, self._calculator.get_structures_dipole
                )
            )
        return np.concatenate(blocks, axis=0)

    @timeit
    def polarizabilities(
        self, structures: Iterable[Atoms] | Atoms
    ) -> npt.NDArray[np.float64]:
        structure_list = _structure_list(structures)
        if not structure_list:
            return np.empty((0, 6), dtype=np.float64)
        blocks = []
        for chunk in self._chunks(
            structure_list, self._effective_chunk_max_atoms(structure_list)
        ):
            blocks.extend(
                self._array_with_oom_split(
                    chunk, self._calculator.get_structures_polarizability
                )
            )
        return np.concatenate(blocks, axis=0)

    @timeit
    def predict_dftd3(
        self,
        structures: Iterable[Atoms] | Atoms,
        *,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
    ) -> Prediction:
        calculator = self._cpu_calculator_for_operation()
        return self._run_prediction(
            structures,
            lambda items: calculator.predict_dftd3_structures(
                items, functional, cutoff, cutoff_cn
            ),
            cpu_only=True,
        )

    @timeit
    def predict_with_dftd3(
        self,
        structures: Iterable[Atoms] | Atoms,
        *,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
    ) -> Prediction:
        calculator = self._cpu_calculator_for_operation()
        return self._run_prediction(
            structures,
            lambda items: calculator.predict_with_dftd3_structures(
                items, functional, cutoff, cutoff_cn
            ),
            cpu_only=True,
        )

    def calculate_to_ase(
        self,
        atoms_list: Atoms | Iterable[Atoms],
        calc_descriptor: bool = False,
    ) -> None:
        structures = _structure_list(atoms_list)
        prediction = self.predict(structures)
        descriptor_blocks: list[np.ndarray] | None = None
        if calc_descriptor:
            descriptors = self.descriptors(structures, mean=False)
            descriptor_blocks = list(
                np.split(descriptors, np.cumsum(prediction.atom_counts)[:-1])
            )
        mforce_blocks = (
            prediction.mforce_blocks()
            if isinstance(prediction, SpinPrediction)
            else None
        )
        for index, atoms in enumerate(structures):
            virial = prediction.structure_virials[index]
            stress = virial.reshape(3, 3) * len(atoms) / atoms.get_volume()
            calculator = SinglePointCalculator(
                atoms,
                energy=float(prediction.energy[index]),
                forces=prediction.force_blocks()[index],
                stress=full_3x3_to_voigt_6_stress(stress),
            )
            if descriptor_blocks is not None:
                calculator.results["descriptor"] = descriptor_blocks[index]
            if mforce_blocks is not None:
                calculator.results["mforces"] = mforce_blocks[index]
            atoms.calc = calculator


Nep3Calculator = NepCalculator


class NepAseCalculator(Calculator):
    implemented_properties = [
        "energy",
        "energies",
        "forces",
        "stress",
        "descriptor",
        "mforces",
    ]

    def __init__(
        self,
        model_file: PathLike = "nep.txt",
        backend: NepBackend | str | None = None,
        chunk_max_atoms: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        self._calc = NepCalculator(model_file, backend, chunk_max_atoms)
        super().__init__(*args, **kwargs)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        prediction = self._calc.predict(atoms)
        self.results["energy"] = float(prediction.energy[0])
        self.results["energies"] = prediction.potential
        self.results["forces"] = prediction.forces
        virial = prediction.structure_virials[0]
        stress = virial.reshape(3, 3) * len(atoms) / atoms.get_volume()
        self.results["stress"] = full_3x3_to_voigt_6_stress(stress)
        if "descriptor" in properties:
            self.results["descriptor"] = self._calc.descriptors(atoms, mean=False)
        if isinstance(prediction, SpinPrediction):
            self.results["mforces"] = prediction.mforces
