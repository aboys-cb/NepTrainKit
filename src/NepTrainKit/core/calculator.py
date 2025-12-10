#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Runtime NEP calculator wrapper handling CPU/GPU backends."""
import contextlib
import os
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from loguru import logger
from NepTrainKit.utils import timeit
from NepTrainKit.core import MessageManager
from NepTrainKit.core.structure import Structure
from NepTrainKit.paths import PathLike, as_path
from NepTrainKit.core.types import NepBackend
from NepTrainKit.core.utils import split_by_natoms, aggregate_per_atom_to_structure, is_charge_model
from NepTrainKit.core.cstdio_redirect import redirect_c_stdout_stderr

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
    from NepTrainKit.qnep_cpu import CpuQNep
except ImportError:
    logger.debug("no found NepTrainKit.qnep_cpu")
    try:
        from qnep_cpu import CpuQNep
    except ImportError:
        logger.debug("no found qnep_cpu")
        CpuQNep = None
try:
    from NepTrainKit.nep_gpu import GpuNep
except ImportError:
    logger.debug("no found NepTrainKit.nep_gpu")
    try:
        from nep_gpu import GpuNep
    except ImportError:
        logger.debug("no found nep_gpu")
        GpuNep = None
try:
    from NepTrainKit.nep_gpu import GpuQNep  # type: ignore[attr-defined]
except ImportError:
    GpuQNep = None
if GpuNep is not None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class NepCalculator:
    """Initialise the NEP calculator and load a CPU/GPU backend."""
    def __init__(
        self,
        model_file: PathLike = "nep.txt",
        backend: NepBackend | None = None,
        batch_size: int | None = None,
        native_stdio: str | Path | Literal["inherit", "silent"] | None = "inherit",
    ) -> None:

        super().__init__()
        self.model_path = as_path(model_file)
        if isinstance(backend, str):
            backend = NepBackend(backend)
        self.backend = backend or NepBackend.AUTO
        self.batch_size = batch_size or 1000
        self.initialized = False
        self.nep3 = None
        self.is_charge_model = is_charge_model(self.model_path)
        self.element_list: list[str] = []
        self.type_dict: dict[str, int] = {}
        # Native stdio behavior for C/C++ (printf) in backends
        self._native_stdio = native_stdio
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

    def __del__(self):
        return

    def _native_stdio_ctx(self):
        if self._native_stdio == "inherit":
            return contextlib.nullcontext()
        if self._native_stdio in (None, "silent"):
            return redirect_c_stdout_stderr()
        return redirect_c_stdout_stderr(as_path(self._native_stdio))

    def cancel(self) -> None:
        if getattr(self, "nep3", None) is not None:
            self.nep3.cancel()

    def load_nep(self) -> None:
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
        try:
            if backend == NepBackend.GPU:
                target_cls = GpuQNep if self.is_charge_model else GpuNep
                if target_cls is None:
                    return False
                try:
                    with self._native_stdio_ctx():
                        self.nep3 = target_cls(str(self.model_path))
                        if hasattr(self.nep3, "set_batch_size"):
                            self.nep3.set_batch_size(self.batch_size)
                except RuntimeError as exc:
                    logger.error(exc)
                    MessageManager.send_warning_message(str(exc))
                    return False
            else:
                target_cls = CpuQNep if self.is_charge_model else CpuNep
                if target_cls is None:
                    return False
                with self._native_stdio_ctx():
                    self.nep3 = target_cls(str(self.model_path))
            self.backend = backend
            return True
        except Exception:
            logger.debug(traceback.format_exc())
            return False

    @staticmethod
    def _ensure_structure_list(
        structures: Iterable[Structure] | Structure,
    ) -> list[Structure]:
        if isinstance(structures, (Structure, Atoms)):
            return [structures]
        if isinstance(structures, list):
            return structures
        return list(structures)

    @timeit
    def compose_structures(
        self,
        structures: Iterable[Structure] | Structure,
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]], list[int]]:
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
        return_charge: bool = False,
        mean_virial: bool = True,
    ):
        structure_list = self._ensure_structure_list(structures)
        if not self.initialized or not structure_list:
            empty = np.array([], dtype=np.float32)
            if self.is_charge_model and return_charge:
                return empty, [], [], [], []
            return empty, [], []
        atom_types, boxes, positions, group_sizes = self.compose_structures(structure_list)
        self.nep3.reset_cancel()
        try:
            with self._native_stdio_ctx():
                if self.is_charge_model and hasattr(self.nep3, "calculate_qnep"):
                    outputs = self.nep3.calculate_qnep(atom_types, boxes, positions)
                else:
                    outputs = self.nep3.calculate(atom_types, boxes, positions)
        except Exception as exc:
            logger.error(exc)
            MessageManager.send_warning_message(str(exc))
            empty = np.array([], dtype=np.float32)
            if self.is_charge_model and return_charge:
                return empty, [], [], [], []
            return empty, [], []

        if self.is_charge_model:
            potentials, forces, virials, charges, becs = outputs
            pot_arr = np.asarray(potentials)
            if pot_arr.dtype != object:
                pot_arr = pot_arr.astype(np.float32, copy=False)
                forces_arr = np.asarray(forces, dtype=np.float32)
                virials_arr = np.asarray(virials, dtype=np.float32)
                charges_arr = np.asarray(charges, dtype=np.float32)
                becs_arr = np.asarray(becs, dtype=np.float32)
                if forces_arr.ndim == 1:
                    forces_arr = forces_arr.reshape(-1, 3)
                if virials_arr.ndim == 1:
                    virials_arr = virials_arr.reshape(-1, 9)
                if charges_arr.ndim > 1:
                    charges_arr = charges_arr.reshape(-1)
                if becs_arr.ndim == 1:
                    becs_arr = becs_arr.reshape(-1, 9)
                potentials_array = aggregate_per_atom_to_structure(pot_arr, group_sizes, map_func=np.sum, axis=None).tolist()
                force_blocks = split_by_natoms(forces_arr, group_sizes)
                virial_blocks = aggregate_per_atom_to_structure(virials_arr, group_sizes, map_func=np.mean, axis=0).tolist()
                charge_blocks = [blk.reshape(-1) for blk in split_by_natoms(charges_arr, group_sizes)]
                bec_blocks = split_by_natoms(becs_arr.reshape(-1, 9), group_sizes)
                if not return_charge:
                    return potentials_array, force_blocks, virial_blocks
                return potentials_array, force_blocks, virial_blocks, charge_blocks, bec_blocks
            else:
                potentials = np.hstack(potentials) if len(potentials) else np.array([])
                potentials_array = aggregate_per_atom_to_structure(potentials, group_sizes, map_func=np.sum, axis=None)
                reshaped_forces = [np.array(force).reshape(3, -1).T for force in forces]
                reshaped_virials = [np.array(virial).reshape(9, -1).mean(axis=1) for virial in virials]
                if not return_charge:
                    return potentials_array.tolist(), reshaped_forces, reshaped_virials

                charges_list = [np.asarray(charge, dtype=np.float32).reshape(-1) for charge in (charges or [])]
                becs_list = [np.asarray(bec, dtype=np.float32).reshape(9, -1).T for bec in (becs or [])]
                return potentials_array.tolist(), reshaped_forces, reshaped_virials, charges_list, becs_list

        potentials, forces, virials = outputs
        potentials_arr = np.asarray(potentials, dtype=np.float32)
        forces_arr = np.asarray(forces, dtype=np.float32)
        virials_arr = np.asarray(virials, dtype=np.float32)
        if potentials_arr.size == 0:
            return [], [], []
        if forces_arr.ndim == 1:
            forces_arr = forces_arr.reshape(-1, 3)
        if virials_arr.ndim == 1:
            virials_arr = virials_arr.reshape(-1, 9)
        potentials_array = aggregate_per_atom_to_structure(potentials_arr, group_sizes, map_func=np.sum, axis=None).tolist()
        forces_blocks = split_by_natoms(forces_arr, group_sizes)
        if mean_virial:
            virials_blocks = aggregate_per_atom_to_structure(virials_arr, group_sizes, map_func=np.mean, axis=0).tolist()
        else:
            virials_blocks = split_by_natoms(virials_arr, group_sizes)

        return potentials_array, forces_blocks, virials_blocks

    @timeit
    def calculate_dftd3(
        self,
        structures: Iterable[Structure] | Structure,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
        mean_virial: bool = True,
    ):
        structure_list = self._ensure_structure_list(structures)
        if not self.initialized or not structure_list:
            return [], [], []
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
        potentials_arr = np.asarray(potentials, dtype=np.float32)
        forces_arr = np.asarray(forces, dtype=np.float32)
        virials_arr = np.asarray(virials, dtype=np.float32)
        if forces_arr.ndim == 1:
            forces_arr = forces_arr.reshape(-1, 3)
        if virials_arr.ndim == 1:
            virials_arr = virials_arr.reshape(-1, 9)
        potentials_array = aggregate_per_atom_to_structure(potentials_arr, group_sizes, map_func=np.sum, axis=None).tolist()
        forces_blocks = split_by_natoms(forces_arr, group_sizes)
        if mean_virial:
            virials_blocks = aggregate_per_atom_to_structure(virials_arr, group_sizes, map_func=np.mean, axis=0).tolist()
        else:
            virials_blocks = split_by_natoms(virials_arr, group_sizes)
        return potentials_array, forces_blocks, virials_blocks

    @timeit
    def calculate_with_dftd3(
        self,
        structures: Iterable[Structure] | Structure,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
        mean_virial: bool = True,
    ):
        structure_list = self._ensure_structure_list(structures)
        if not self.initialized or not structure_list:
            return [], [], []
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

        potentials_arr = np.asarray(potentials, dtype=np.float32)
        forces_arr = np.asarray(forces, dtype=np.float32)
        virials_arr = np.asarray(virials, dtype=np.float32)
        if forces_arr.ndim == 1:
            forces_arr = forces_arr.reshape(-1, 3)
        if virials_arr.ndim == 1:
            virials_arr = virials_arr.reshape(-1, 9)
        potentials_array = aggregate_per_atom_to_structure(potentials_arr, group_sizes, map_func=np.sum, axis=None).tolist()
        forces_blocks = split_by_natoms(forces_arr, group_sizes)
        if mean_virial:
            virials_blocks = aggregate_per_atom_to_structure(virials_arr, group_sizes, map_func=np.mean, axis=0).tolist()
        else:
            virials_blocks = split_by_natoms(virials_arr, group_sizes)
        return potentials_array, forces_blocks, virials_blocks

    def get_descriptor(self, structure: Structure) -> npt.NDArray[np.float32]:
        if not self.initialized:
            return np.array([])
        return self.get_structures_descriptor([structure], mean_descriptor=False)

    @timeit
    def get_structures_descriptor(
        self,
        structures: list[Structure],
        mean_descriptor: bool = True
    ) -> npt.NDArray[np.float32]:
        if not self.initialized:
            return np.array([])
        types, boxes, positions, group_sizes = self.compose_structures(structures)
        self.nep3.reset_cancel()
        with self._native_stdio_ctx():
            descriptor = self.nep3.get_structures_descriptor(types, boxes, positions)
        descriptor = np.asarray(descriptor, dtype=np.float32)
        if descriptor.size == 0:
            return descriptor
        if not mean_descriptor:
            return descriptor
        structure_descriptor = aggregate_per_atom_to_structure(descriptor, group_sizes, map_func=np.mean, axis=0)
        return structure_descriptor

    @timeit
    def get_structures_polarizability(
        self,
        structures: list[Structure],
    ) -> npt.NDArray[np.float32]:
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
        if isinstance(atoms_list, Atoms):
            atoms_list = [atoms_list]
        descriptor_blocks: list[np.ndarray] | None = None
        if calc_descriptor:
            per_atom_descriptor = self.get_structures_descriptor(atoms_list, mean_descriptor=False)
            atom_counts = [len(atoms) for atoms in atoms_list]
            descriptor_blocks = split_by_natoms(per_atom_descriptor, atom_counts)

        energy, forces, virial = self.calculate(atoms_list)

        for index, atoms in enumerate(atoms_list):
            _e = energy[index]
            _f = forces[index]
            _vi = np.asarray(virial[index])
            if _vi.ndim == 2 and _vi.shape[1] == 9:
                _vi_avg = _vi.mean(axis=0)
            else:
                _vi_avg = _vi.reshape(9)
            _s = _vi_avg.reshape(3, 3) * len(atoms) / atoms.get_volume()
            spc = SinglePointCalculator(
                atoms,
                energy=_e,
                forces=_f,
                stress=full_3x3_to_voigt_6_stress(_s),

            )
            if calc_descriptor:
                spc.results["descriptor"] = descriptor_blocks[index]
            atoms.calc = spc


Nep3Calculator = NepCalculator


class NepAseCalculator(Calculator):
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
        virial=np.asarray(virial[0])
        if virial.ndim == 2 and virial.shape[1] == 9:
            virial_val = virial.mean(axis=0)
        else:
            virial_val = virial.reshape(9)
        stress = virial_val.reshape(3,3)*len(atoms)/atoms.get_volume()
        self.results["stress"]=full_3x3_to_voigt_6_stress(stress)
