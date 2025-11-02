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
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from loguru import logger
from NepTrainKit.utils import timeit
from NepTrainKit.core import   MessageManager
from NepTrainKit.core.structure import Structure
from NepTrainKit.paths import PathLike, as_path
from NepTrainKit.core.types import NepBackend
from NepTrainKit.core.utils import split_by_natoms,aggregate_per_atom_to_structure
from NepTrainKit.core.cstdio_redirect import redirect_c_stdout_stderr
from typing import Literal

try:
    from NepTrainKit.nep_cpu import CpuNep
except ImportError:
    logger.debug("no found NepTrainKit.nep_cpu")
    logger.error(traceback.format_exc())
    try:
        from nep_cpu import CpuNep
    except ImportError:
        logger.debug("no found nep_cpu")


        CpuNep = None
try:
    from NepTrainKit.nep_gpu import GpuNep
except ImportError:
    logger.debug("no found NepTrainKit.nep_gpu")
    logger.debug(traceback.format_exc())
    try:
        from nep_gpu import GpuNep
    except ImportError:
        logger.debug("no found nep_gpu")
        GpuNep = None

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
        native_stdio: str | Path | Literal["inherit", "silent"] | None = "silent",
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
        # Native stdio behavior for C/C++ (printf) in backends
        #   - "silent": suppress to devnull (default)
        #   - "inherit": leave as-is (print to terminal)
        #   - path-like: redirect native prints to this file
        self._native_stdio = native_stdio
        self._persistent_stdio_guard = None
        # Install process-wide C stdio redirection early to catch any async prints
        if self._native_stdio != "inherit":
            target = None if self._native_stdio in (None, "silent") else as_path(self._native_stdio)
            self._persistent_stdio_guard = redirect_c_stdout_stderr(target)
            # Activate and keep until object deletion
            self._persistent_stdio_guard.__enter__()

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

    def __del__(self):
        # Restore stdio on deletion if we installed a persistent guard
        try:
            if getattr(self, "_persistent_stdio_guard", None) is not None:
                self._persistent_stdio_guard.__exit__(None, None, None)
        except Exception:
            pass
    def _load_nep_backend(self, backend: NepBackend) -> bool:
        """Attempt to initialise ``backend`` and return ``True`` when successful."""
        try:
            sink = io.StringIO()
            with self._native_stdio_ctx(), contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
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

    def _native_stdio_ctx(self):
        if self._native_stdio == "inherit":
            return contextlib.nullcontext()
        if self._native_stdio in (None, "silent"):
            return redirect_c_stdout_stderr()
        return redirect_c_stdout_stderr(as_path(self._native_stdio))

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
        structures: Iterable[Structure] | Structure,has_spin=False
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]], list[int]] | tuple[list[list[int]], list[list[float]], list[list[float]], list[list[float]],list[int]]:
        """Convert ``structures`` into backend-ready arrays of types, boxes, and positions."""
        structure_list = self._ensure_structure_list(structures)
        group_sizes: list[int] = []
        atom_types: list[list[int]] = []
        boxes: list[list[float]] = []
        spins: list[list[float]] = []
        positions: list[list[float]] = []
        for structure in structure_list:
            symbols = structure.get_chemical_symbols()
            mapped_types = [self.type_dict[symbol] for symbol in symbols]
            box = structure.cell.transpose(1, 0).reshape(-1).tolist()
            coords = structure.positions.transpose(1, 0).reshape(-1).tolist()
            atom_types.append(mapped_types)
            boxes.append(box)
            positions.append(coords)
            if has_spin:
                spins.append(structure.atomic_properties["spin"].transpose(1, 0).reshape(-1).tolist())
            group_sizes.append(len(mapped_types))
        if has_spin:
            return atom_types, boxes, positions,spins, group_sizes
        else:
            return atom_types, boxes, positions, group_sizes

    @timeit
    def calculate_flat(
        self,
        structures: Iterable[Structure] | Structure,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Return atom-level arrays across all frames.

        Returns (potentials_atom, forces, virials) with shapes:
        - potentials_atom: (total_atoms,)
        - forces: (total_atoms, 3)
        - virials: (total_atoms, 9)
        """
        structure_list = self._ensure_structure_list(structures)
        if not self.initialized or not structure_list:
            empty = np.array([], dtype=np.float64)
            return empty, empty, empty
        atom_types, boxes, positions, _ = self.compose_structures(structure_list)
        self.nep3.reset_cancel()
        with self._native_stdio_ctx():
            potentials, forces, virials = self.nep3.calculate(atom_types, boxes, positions)
        # Ensure numpy arrays; keep dtype from backend (GPU: float32, CPU: float64)
        potentials = np.asarray(potentials)
        forces = np.asarray(forces)
        virials = np.asarray(virials)
        return potentials, forces, virials

    @timeit
    def calculate_spin_flat(
        self,
        structures: Iterable[Structure] | Structure,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """Return atom-level arrays across all frames for spin models.

        Returns (potentials_atom, forces, mforces, virials) with shapes:
        - potentials_atom: (total_atoms,)
        - forces: (total_atoms, 3)
        - mforces: (total_atoms, 3)
        - virials: (total_atoms, 9)
        """
        structure_list = self._ensure_structure_list(structures)
        if not self.initialized or not structure_list:
            empty = np.array([], dtype=np.float64)
            return empty, empty, empty, empty
        atom_types, boxes, positions, spins, _ = self.compose_structures(structure_list, True)
        self.nep3.reset_cancel()
        with self._native_stdio_ctx():
            potentials, forces, mforce, virials = self.nep3.calculate(atom_types, boxes, positions, spins)
        return np.asarray(potentials), np.asarray(forces), np.asarray(mforce), np.asarray(virials)

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
            return [], [], []
        atom_types, boxes, positions, group_sizes = self.compose_structures(structure_list)
        self.nep3.reset_cancel()
        with self._native_stdio_ctx():
            potentials, forces, virials = self.nep3.calculate(atom_types, boxes, positions)

        # Handle GPU (np.ndarray) vs CPU (list of arrays) outputs uniformly
        if isinstance(potentials, np.ndarray):
            pot_concat = potentials
        else:
            pot_concat = np.hstack(potentials).astype(np.float32, copy=False)
        potentials_array = aggregate_per_atom_to_structure(pot_concat, group_sizes, map_func=np.sum, axis=None)

        if isinstance(forces, np.ndarray):
            # GPU/CPU: forces is (total_atoms, 3); split per structure
            forces_blocks = split_by_natoms(forces, group_sizes)
        else:
            # CPU: per-structure flat 3N; reshape each to (N, 3)
            forces_blocks = [np.array(block, dtype=np.float32).reshape(3, -1).T for block in forces]

        # if isinstance(virials, np.ndarray):
        #     # Keep atom-level virials per structure: (Ni, 9)
        #     virials_blocks = split_by_natoms(virials, group_sizes)
        # else:
            # CPU legacy list: per-structure flat 9N; reshape to (Ni, 9)
        virials_blocks = [np.array(block, dtype=np.float32).reshape(9, -1).mean(axis=1).T for block in virials]

        return potentials_array.tolist(), forces_blocks, virials_blocks

    @timeit
    def calculate_spin(
        self,
        structures: Iterable[Structure] | Structure,
    ) -> tuple[list[np.float32], list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]]]:
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
            return empty, empty, empty, empty
        atom_types, boxes, positions,spins, group_sizes = self.compose_structures(structure_list,True)
        self.nep3.reset_cancel()
        # Order from GPU extension: potentials, forces, magnetic-forces, virials
        with self._native_stdio_ctx():
            potentials, forces, mforce, virials = self.nep3.calculate(atom_types, boxes, positions,spins)




        potentials_array = aggregate_per_atom_to_structure(potentials, group_sizes, map_func=np.sum, axis=None)


        forces_blocks = split_by_natoms(forces, group_sizes)



        mforce_blocks = split_by_natoms(mforce, group_sizes)

        # Keep atom-level virials per structure: (Ni, 9)
        virials_blocks = split_by_natoms(virials, group_sizes)

        return potentials_array.tolist(), forces_blocks, mforce_blocks, virials_blocks



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
        with self._native_stdio_ctx():
            potentials, forces, virials = self.nep3.calculate_dftd3(
            functional,
            cutoff,
            cutoff_cn,
            atom_types,
            boxes,
            positions,
        )
        # Unify ndarray/list returns
        if isinstance(potentials, np.ndarray):
            pot_concat = potentials
        else:
            pot_concat = np.hstack(potentials).astype(np.float32, copy=False)
        potentials_array = aggregate_per_atom_to_structure(pot_concat, group_sizes, map_func=np.sum, axis=None)

        if isinstance(forces, np.ndarray):
            forces_blocks = split_by_natoms(forces, group_sizes)
        else:
            forces_blocks = [np.array(force).reshape(3, -1).T for force in forces]

        if isinstance(virials, np.ndarray):
            vir_blocks = split_by_natoms(virials, group_sizes)
            virials_blocks = [vb.mean(axis=0) for vb in vir_blocks]
        else:
            virials_blocks = [np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials]

        return potentials_array.tolist(), forces_blocks, virials_blocks
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
        with self._native_stdio_ctx():
            potentials, forces, virials = self.nep3.calculate_with_dftd3(
            functional,
            cutoff,
            cutoff_cn,
            atom_types,
            boxes,
            positions,
        )
        if isinstance(potentials, np.ndarray):
            pot_concat = potentials
        else:
            pot_concat = np.hstack(potentials).astype(np.float32, copy=False)
        potentials_array = aggregate_per_atom_to_structure(pot_concat, group_sizes, map_func=np.sum, axis=None)

        if isinstance(forces, np.ndarray):
            forces_blocks = split_by_natoms(forces, group_sizes)
        else:
            forces_blocks = [np.array(force).reshape(3, -1).T for force in forces]

        if isinstance(virials, np.ndarray):
            vir_blocks = split_by_natoms(virials, group_sizes)
            virials_blocks = [vb.mean(axis=0) for vb in vir_blocks]
        else:
            virials_blocks = [np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials]

        return potentials_array.tolist(), forces_blocks, virials_blocks

    def get_descriptor(self, structure: Structure) -> npt.NDArray[np.float32]:
        """Return the per-atom descriptor matrix for a single ``structure``."""
        if not self.initialized:
            return np.array([])
        symbols = structure.get_chemical_symbols()
        mapped_types = [self.type_dict[symbol] for symbol in symbols]
        box = structure.cell.transpose(1, 0).reshape(-1).tolist()
        positions = structure.positions.transpose(1, 0).reshape(-1).tolist()
        self.nep3.reset_cancel()
        with self._native_stdio_ctx():
            descriptor = self.nep3.get_descriptor(mapped_types, box, positions)
        descriptors_per_atom = np.array(descriptor, dtype=np.float32).reshape(-1, len(structure)).T
        return descriptors_per_atom
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
        with self._native_stdio_ctx():
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
        with self._native_stdio_ctx():
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
        with self._native_stdio_ctx():
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
            per_atom_descriptor = self.get_structures_descriptor(atoms_list)
            atom_counts = [len(atoms) for atoms in atoms_list]
            descriptor_blocks = split_by_natoms(per_atom_descriptor, atom_counts)

        energy,forces,virial = self.calculate(atoms_list)

        for index,atoms in enumerate(atoms_list):
            _e= energy[index]
            _f= forces[index]
            _vi = np.asarray(virial[index])  # (Ni, 9)
            if _vi.ndim == 2 and _vi.shape[1] == 9:
                _vi_avg = _vi.mean(axis=0)
            else:
                _vi_avg = np.asarray(_vi).reshape(9)
            _s = _vi_avg.reshape(3, 3) * len(atoms) / atoms.get_volume()
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


