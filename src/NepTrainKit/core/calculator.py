#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Runtime NEP calculator wrapper handling CPU/GPU backends."""
import contextlib
import io
import sys
import traceback
from collections.abc import Iterable
from pathlib import Path
import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from loguru import logger
from NepTrainKit.utils import timeit
from NepTrainKit.core import Structure, MessageManager
from NepTrainKit.paths import PathLike, as_path
from NepTrainKit.core.types import NepBackend

try:
    from NepTrainKit.nep_cpu import CpuNep
except ImportError:
    logger.debug("no found NepTrainKit.nep_cpu")

    try:
        from nep_cpu import CpuNep
    except ImportError:
        logger.debug("no found nep_cpu")


        CpuNep = None
try:
    from NepTrainKit.nep_gpu import GpuNep
except ImportError:
    logger.debug("no found NepTrainKit.nep_gpu")
    try:
        from nep_gpu import GpuNep
    except ImportError:
        logger.debug("no found nep_gpu")
        GpuNep = None

class NepCalculator:

    def __init__(
        self,
        model_file: PathLike = "nep.txt",
        backend: NepBackend | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialise the NEP calculator and load a CPU/GPU backend.

        Parameters
        ----------
        model_file : str or pathlib.Path, default="nep.txt"
            Path to the NEP model file.
        backend : NepBackend or None, optional
            Preferred backend; ``AUTO`` tries GPU then CPU.
        batch_size : int or None, optional
            NEP backend batch size. Defaults to 1000 when not specified.

        Notes
        -----
        If neither CPU nor GPU backends are importable, a message box will be
        shown via :class:`MessageManager` and the instance remains uninitialised.

        Examples
        --------
        >>> c = NepCalculator("nep.txt","gpu")  # doctest: +SKIP
        >>> structure_list=Structure.read_multiple("train.xyz") # doctest: +SKIP
        >>> energy,forces,virial = c.calculate(structure_list) # doctest: +SKIP
        >>> structures_desc = c.get_structures_descriptor(structure_list) # doctest: +SKIP
        """
        super().__init__()
        self.model_path = as_path(model_file)
        if isinstance(backend,str):
            backend = NepBackend(backend)
        self.backend = backend or NepBackend.AUTO
        self.batch_size = batch_size or 1000
        self.initialized = False
        self.nep3 = None
        self.element_list: list[str] = []
        self.type_dict: dict[str, int] = {}
        if CpuNep is None and GpuNep is None:
            MessageManager.send_message_box(
                "Failed to import NEP.\n To use the display functionality normally, please prepare the *.out and descriptor.out files.",
                "Error",
            )
            return
        if self.model_path.exists():
            self.load_nep()
            if getattr(self, "nep3", None) is not None:
                self.element_list = self.nep3.get_element_list()
                self.type_dict = {element: index for index, element in enumerate(self.element_list)}
                self.initialized = True
        else:
            logger.warning("NEP model file not found: %s", self.model_path)

    def cancel(self) -> None:
        """Forward a cancel request to the underlying NEP backend."""
        self.nep3.cancel()

    def load_nep(self) -> None:
        """Attempt to load the NEP backend using the configured preference."""
        if self.backend == NepBackend.AUTO:
            if not self._load_nep_backend(NepBackend.GPU):
                self._load_nep_backend(NepBackend.CPU)
        elif self.backend == NepBackend.GPU:
            if not self._load_nep_backend(NepBackend.GPU):
                MessageManager.send_warning_message("The NEP backend you selected is GPU, but it failed to load on your device; the program has switched to the CPU backend.")
                self._load_nep_backend(NepBackend.CPU)
        else:
            self._load_nep_backend(NepBackend.CPU)
    def _load_nep_backend(self, backend: NepBackend) -> bool:
        """Attempt to initialise ``backend`` and return ``True`` when successful."""
        try:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                if backend == NepBackend.GPU:
                    if GpuNep is None:
                        return False
                    try:
                        self.nep3 = GpuNep(str(self.model_path))
                        self.nep3.set_batch_size(self.batch_size)
                    except RuntimeError as exc:
                        logger.error(exc)
                        MessageManager.send_warning_message(str(exc))
                        return False
                else:
                    if CpuNep is None:
                        return False
                    self.nep3 = CpuNep(str(self.model_path))
                self.backend = backend
                return True
        except Exception:
            logger.debug(traceback.format_exc())
            return False

    @staticmethod
    def _ensure_structure_list(
        structures: Iterable[Structure] | Structure,
    ) -> list[Structure]:
        """Normalise ``structures`` to a list of ``Structure`` instances."""
        if isinstance(structures, (Structure,Atoms)):
            return [structures]
        if isinstance(structures, list):
            return structures
        return list(structures)
    def compose_structures(
        self,
        structures: Iterable[Structure] | Structure,
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]], list[int]]:
        """Convert ``structures`` into backend-ready arrays of types, boxes, and positions."""
        structure_list = self._ensure_structure_list(structures)
        group_sizes: list[int] = []
        atom_types: list[list[int]] = []
        boxes: list[list[float]] = []
        positions: list[list[float]] = []
        for structure in structure_list:
            symbols = structure.get_chemical_symbols()
            mapped_types = [self.type_dict[symbol] for symbol in symbols]
            box = structure.cell.transpose(1, 0).reshape(-1).tolist()
            coords = structure.positions.transpose(1, 0).reshape(-1).tolist()
            atom_types.append(mapped_types)
            boxes.append(box)
            positions.append(coords)
            group_sizes.append(len(mapped_types))
        return atom_types, boxes, positions, group_sizes
    @timeit
    def calculate(
        self,
        structures: Iterable[Structure] | Structure,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Compute energies, forces, and virials for one or more structures.

        Parameters
        ----------
        structures : Structure or Iterable[Structure]
            Single structure or an iterable of structures to evaluate.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(potentials, forces, virials)`` arrays with ``float32`` dtype.
            Potentials are per-structure, forces per-atom, and virials per-structure.

        Examples
        --------
        >>> # c = NepCalculator(...); e, f, v = c.calculate(structs)  # doctest: +SKIP
        """
        structure_list = self._ensure_structure_list(structures)
        if not self.initialized or not structure_list:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty
        atom_types, boxes, positions, group_sizes = self.compose_structures(structure_list)
        self.nep3.reset_cancel()
        potentials, forces, virials = self.nep3.calculate(atom_types, boxes, positions)
        split_indices = np.cumsum(group_sizes)[:-1]
        potentials = np.hstack(potentials)
        split_potential_arrays = np.split(potentials, split_indices) if split_indices.size else [potentials]
        potentials_array = np.array([np.sum(chunk) for chunk in split_potential_arrays], dtype=np.float32)
        reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]
        if reshaped_forces:
            forces_array = np.vstack(reshaped_forces).astype(np.float32, copy=False)
        else:
            forces_array = np.empty((0, 3), dtype=np.float32)
        reshaped_virials = [np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials]
        if reshaped_virials:
            virials_array = np.vstack(reshaped_virials).astype(np.float32, copy=False)
        else:
            virials_array = np.empty((0, 9), dtype=np.float32)
        return potentials_array, forces_array, virials_array

    @timeit
    def calculate_dftd3(
        self,
        structures: Iterable[Structure] | Structure,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Evaluate structures using the DFT-D3 variant of the NEP backend.

        Parameters
        ----------
        structures : Structure or Iterable[Structure]
            Structures to evaluate.
        functional : str
            Exchange-correlation functional identifier.
        cutoff : float
            Real-space cutoff for dispersion corrections.
        cutoff_cn : float
            Coordination number cutoff.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(potentials, forces, virials)`` arrays with ``float32`` dtype.
        """
        structure_list = self._ensure_structure_list(structures)
        if not self.initialized or not structure_list:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty
        atom_types, boxes, positions, group_sizes = self.compose_structures(structure_list)
        self.nep3.reset_cancel()
        potentials, forces, virials = self.nep3.calculate_dftd3(
            functional,
            cutoff,
            cutoff_cn,
            atom_types,
            boxes,
            positions,
        )
        split_indices = np.cumsum(group_sizes)[:-1]
        potentials = np.hstack(potentials)
        split_potential_arrays = np.split(potentials, split_indices) if split_indices.size else [potentials]
        potentials_array = np.array([np.sum(chunk) for chunk in split_potential_arrays], dtype=np.float32)
        reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]
        if reshaped_forces:
            forces_array = np.vstack(reshaped_forces).astype(np.float32, copy=False)
        else:
            forces_array = np.empty((0, 3), dtype=np.float32)
        reshaped_virials = [np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials]
        if reshaped_virials:
            virials_array = np.vstack(reshaped_virials).astype(np.float32, copy=False)
        else:
            virials_array = np.empty((0, 9), dtype=np.float32)
        return potentials_array, forces_array, virials_array
    @timeit
    def calculate_with_dftd3(
        self,
        structures: Iterable[Structure] | Structure,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Run coupled NEP + DFT-D3 calculation and return results.

        Parameters
        ----------
        structures : Structure or Iterable[Structure]
            Structures to evaluate.
        functional : str
            Exchange-correlation functional identifier.
        cutoff : float
            Real-space cutoff for dispersion corrections.
        cutoff_cn : float
            Coordination number cutoff.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(potentials, forces, virials)`` arrays with ``float32`` dtype.
        """
        structure_list = self._ensure_structure_list(structures)
        if not self.initialized or not structure_list:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty
        atom_types, boxes, positions, group_sizes = self.compose_structures(structure_list)
        self.nep3.reset_cancel()
        potentials, forces, virials = self.nep3.calculate_with_dftd3(
            functional,
            cutoff,
            cutoff_cn,
            atom_types,
            boxes,
            positions,
        )
        split_indices = np.cumsum(group_sizes)[:-1]
        potentials = np.hstack(potentials)
        split_potential_arrays = np.split(potentials, split_indices) if split_indices.size else [potentials]
        potentials_array = np.array([np.sum(chunk) for chunk in split_potential_arrays], dtype=np.float32)
        reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]
        if reshaped_forces:
            forces_array = np.vstack(reshaped_forces).astype(np.float32, copy=False)
        else:
            forces_array = np.empty((0, 3), dtype=np.float32)
        reshaped_virials = [np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials]
        if reshaped_virials:
            virials_array = np.vstack(reshaped_virials).astype(np.float32, copy=False)
        else:
            virials_array = np.empty((0, 9), dtype=np.float32)
        return potentials_array, forces_array, virials_array

    def get_descriptor(self, structure: Structure) -> npt.NDArray[np.float32]:
        """Return the per-atom descriptor matrix for a single ``structure``."""
        if not self.initialized:
            return np.array([])
        symbols = structure.get_chemical_symbols()
        mapped_types = [self.type_dict[symbol] for symbol in symbols]
        box = structure.cell.transpose(1, 0).reshape(-1).tolist()
        positions = structure.positions.transpose(1, 0).reshape(-1).tolist()
        self.nep3.reset_cancel()
        descriptor = self.nep3.get_descriptor(mapped_types, box, positions)
        descriptors_per_atom = np.array(descriptor, dtype=np.float32).reshape(-1, len(structure)).T
        return descriptors_per_atom
    @timeit
    def get_structures_descriptor(
        self,
        structures: list[Structure],
    ) -> npt.NDArray[np.float32]:
        """Return descriptors for multiple structures without additional averaging."""
        if not self.initialized:
            return np.array([])
        types, boxes, positions, _ = self.compose_structures(structures)
        self.nep3.reset_cancel()

        descriptor = self.nep3.get_structures_descriptor(types, boxes, positions)
        return np.array(descriptor, dtype=np.float32)

    @timeit
    def get_structures_polarizability(
        self,
        structures: list[Structure],
    ) -> npt.NDArray[np.float32]:
        """Compute polarizability tensors for each structure."""
        if not self.initialized:
            return np.array([])
        types, boxes, positions, _ = self.compose_structures(structures)
        self.nep3.reset_cancel()

        polarizability = self.nep3.get_structures_polarizability(types, boxes, positions)
        return np.array(polarizability, dtype=np.float32)

    def get_structures_dipole(
        self,
        structures: list[Structure],
    ) -> npt.NDArray[np.float32]:
        """Compute dipole vectors for each structure."""
        if not self.initialized:
            return np.array([])
        self.nep3.reset_cancel()

        types, boxes, positions, _ = self.compose_structures(structures)
        dipole = self.nep3.get_structures_dipole(types, boxes, positions)
        return np.array(dipole, dtype=np.float32)

Nep3Calculator = NepCalculator

from ase.calculators.calculator import Calculator, all_changes


class NepAseCalculator(Calculator):
    """
    Encapsulated ase calculator, the input parameters are the same as NepCalculator

    Examples
    --------
    >>>from ase.io import read
    >>>from NepTrainKit.core.calculator import NepAseCalculator
    >>>atoms=read('9.vasp')
    >>>calc = NepAseCalculator("./Config/nep89.txt","gpu")
    >>>atoms.calc=calc
    >>>print('Energy (eV):', atoms.get_potential_energy())
    >>>print('Forces (eV/Å):\n', atoms.get_forces())
    >>>print('Stress (eV/Å^3):\n', atoms.get_stress())
    """
    implemented_properties=[
        "energy",
        "energies",
        "forces",
        "stress",
        "descriptor",
    ]
    def __init__(self,
                 model_file: PathLike = "nep.txt",
                backend: NepBackend | None = None,
                batch_size: int | None = None,*args,**kwargs) -> None:

        self._calc=NepCalculator(model_file,backend,batch_size)
        Calculator.__init__(self,*args,**kwargs)

    def calculate(
        self, atoms=None, properties=['energy'], system_changes=all_changes
    ):

        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms,properties,system_changes)
        if "descriptor" in properties:
            descriptor = self._calc.get_descriptor(atoms)
            self.results["descriptor"]=descriptor
        energy,forces,virial = self._calc.calculate(atoms)

        self.results["energy"]=energy[0]
        self.results["forces"]=forces
        virial=virial[0].reshape(3,3)*len(atoms)
        stress = virial/atoms.get_volume()
        self.results["stress"]=full_3x3_to_voigt_6_stress(stress)


if __name__ == "__main__":
    structures = Structure.read_multiple(Path("D:/Desktop/nep/nep-data-main/2023_Zhao_PdCuNiP/train.xyz"))
    nep = NepCalculator(Path("D:/Desktop/nep/nep-data-main/2023_Zhao_PdCuNiP/nep.txt"))
    nep.calculate(structures)
