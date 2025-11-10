#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Runtime NEP calculator wrapper handling CPU/GPU backends."""
import contextlib
import io
import os
import sys
import traceback
from collections.abc import Iterable
from pathlib import Path
import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from loguru import logger
from NepTrainKit.utils import timeit
from NepTrainKit.core import   MessageManager
from NepTrainKit.core.structure import Structure
from NepTrainKit.paths import PathLike, as_path
from NepTrainKit.core.types import NepBackend
from NepTrainKit.core.utils import split_by_natoms,aggregate_per_atom_to_structure

try:
    from NepTrainKit.nep_cpu import CpuNep
except ImportError:
    logger.debug("no found NepTrainKit.nep_cpu")
    #logger.error(traceback.format_exc())
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
if GpuNep is not None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class NepCalculator:
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
    >>> from NepTrainKit.core.structure import Structure
    >>> c = NepCalculator("nep.txt","gpu")
    >>> structure_list=Structure.read_multiple("train.xyz")
    >>> energy,forces,virial = c.calculate(structure_list)
    >>> structures_desc = c.get_structures_descriptor(structure_list)
    """
    def __init__(
        self,
        model_file: PathLike = "nep.txt",
        backend: NepBackend | None = None,
        batch_size: int | None = None,
    ) -> None:

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
            logger.warning(f"NEP model file not found: { self.model_path}" )

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
    @timeit
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
    ) -> tuple[list[np.float32], list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]]]:
        """Compute energies, forces, and virials for one or more structures.

        Parameters
        ----------
        structures : Structure or Iterable[Structure]
            Single structure or an iterable of structures to evaluate.

        Returns
        -------
        tuple[list, list, list]
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
        potentials = np.hstack(potentials)
        potentials_array = aggregate_per_atom_to_structure(potentials,group_sizes,map_func=np.sum,axis=None)
        reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]
        reshaped_virials = [np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials]

        return potentials_array.tolist(), reshaped_forces, reshaped_virials

    @timeit
    def calculate_dftd3(
        self,
        structures: Iterable[Structure] | Structure,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
    ) -> tuple[list[np.float32], list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]]]:
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
        tuple[list, list, list]
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
        potentials = np.hstack(potentials)
        potentials_array = aggregate_per_atom_to_structure(potentials, group_sizes, map_func=np.sum, axis=None)
        reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]
        reshaped_virials = [np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials]

        return potentials_array.tolist(), reshaped_forces, reshaped_virials
    @timeit
    def calculate_with_dftd3(
        self,
        structures: Iterable[Structure] | Structure,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
    ) -> tuple[list[np.float32], list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]]]:
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
        tuple[list, list, list]
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
        potentials = np.hstack(potentials)
        potentials_array = aggregate_per_atom_to_structure(potentials, group_sizes, map_func=np.sum, axis=None)
        reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]
        reshaped_virials = [np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials]

        return potentials_array.tolist(), reshaped_forces, reshaped_virials

    def get_descriptor(self, structure: Structure) -> npt.NDArray[np.float32]:
        """Return the per-atom descriptor matrix for a single ``structure``."""
        if not self.initialized:
            return np.array([])
        return self.get_structures_descriptor(structure,mean_descriptor=False)
    @timeit
    def get_structures_descriptor(
        self,
        structures: list[Structure],
        mean_descriptor: bool=True
    ) -> npt.NDArray[np.float32]:
        """Return per-atom NEP descriptors stacked across ``structures``."""
        if not self.initialized:
            return np.array([])
        types, boxes, positions, group_sizes = self.compose_structures(structures)
        self.nep3.reset_cancel()
        descriptor = self.nep3.get_structures_descriptor(types, boxes, positions)
        # Ensure numpy array without unnecessary copy when already ndarray
        descriptor = np.asarray(descriptor, dtype=np.float32)
        if not mean_descriptor:
            return descriptor
        structure_descriptor = aggregate_per_atom_to_structure(descriptor, group_sizes, map_func=np.mean, axis=0)
        return structure_descriptor

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

    def calculate_to_ase(
            self,
            atoms_list: Atoms | Iterable[Atoms],
            calc_descriptor=False,

    ):
        """
        Perform single-point calculations for one or many ASE Atoms objects **in-place**
        and attach a ``SinglePointCalculator`` holding the results.

        Parameters
        ----------
        atoms_list : Atoms or iterable of Atoms
            Atomic structure(s) to be evaluated.  The **same** object(s) are
            modified in place; no copy is returned.
        calc_descriptor : bool, optional
            If True the descriptor vector is also computed and stored in
            ``atoms.calc.results['descriptor']``.

        Returns
        -------
        None
            Results are attached to the original ``atoms`` object(s) under
            ``atoms.calc.results``.

        Examples
        --------
        >>> from ase.io import read
        >>> from NepTrainKit.core.calculator import NepCalculator
        >>> frames = read('train.xyz', index=':')   # list[Atoms]
        >>> NepCalculator("nep.txt","gpu").calculate_to_ase(frames)
        >>> for atoms in frames:
        ...     print(atoms.get_potential_energy(), atoms.get_forces())
        """
        if isinstance(atoms_list, Atoms):
            atoms_list = [atoms_list]
        descriptor_blocks: list[np.ndarray] | None = None
        if calc_descriptor:
            per_atom_descriptor = self.get_structures_descriptor(atoms_list,mean_descriptor=False)
            atom_counts = [len(atoms) for atoms in atoms_list]
            descriptor_blocks = split_by_natoms(per_atom_descriptor, atom_counts)

        energy,forces,virial = self.calculate(atoms_list)

        for index,atoms in enumerate(atoms_list):
            _e= energy[index]
            _f= forces[index]
            _vi= virial[index]
            _s = _vi.reshape(3, 3) * len(atoms) / atoms.get_volume()
            spc = SinglePointCalculator(
                atoms,
                energy=_e,
                forces=_f,
                stress=full_3x3_to_voigt_6_stress(_s),

            )
            if calc_descriptor:
                spc.results["descriptor"]=descriptor_blocks[index]
            atoms.calc = spc


Nep3Calculator = NepCalculator



class NepAseCalculator(Calculator):
    """Encapsulated ASE calculator mirroring the :class:`NepCalculator` interface.

    :param model_file: Path to the NEP model file. Defaults to ``"nep.txt"``.
    :param backend: Preferred backend; ``AUTO`` tries GPU then CPU.
    :param batch_size: Optional NEP backend batch size. Defaults to ``1000``.

    Examples
    --------

    >>> from ase.io import read
    >>> from NepTrainKit.core.calculator import NepAseCalculator
    >>> atoms = read('9.vasp')
    >>> calc = NepAseCalculator('./Config/nep89.txt', 'gpu')
    >>> atoms.calc = calc
    >>> print('Energy (eV):', atoms.get_potential_energy())
    >>> print('Forces (eV/Angstrom):', atoms.get_forces())
    >>> print('Stress (eV/Angstrom^3):', atoms.get_stress())

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
        self.results["forces"]=forces[0]
        virial=virial[0].reshape(3,3)*len(atoms)
        stress = virial/atoms.get_volume()
        self.results["stress"]=full_3x3_to_voigt_6_stress(stress)


