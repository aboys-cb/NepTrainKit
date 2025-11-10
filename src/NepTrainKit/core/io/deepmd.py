#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utilities for interacting with DeepMD-generated datasets."""
import traceback
from pathlib import Path
import numpy as np
import numpy.typing as npt
from loguru import logger
from .base import StructureData, ResultData, DPPlotData, StructureSyncRule, NepPlotData
from NepTrainKit.core.structure import Structure, load_npy_structure,save_npy_structure
from NepTrainKit.paths import PathLike, as_path
from NepTrainKit.core.utils import aggregate_per_atom_to_structure, read_nep_out_file, concat_nep_dft_array
from NepTrainKit.config import Config
from .. import   MessageManager
from ... import module_path
def is_deepmd_path(folder: PathLike) -> bool:
    """Return ``True`` when ``folder`` looks like a DeepMD dataset directory."""
    candidate = as_path(folder)
    if (candidate / "type.raw").exists():
        return True
    return any(candidate.rglob("type.raw"))
class DeepmdResultData(ResultData):
    """Result loader that adapts DeepMD outputs to the ResultData interface.

    The loader reads DeepMD numpy exports, normalises them, and exposes consistent
    NEP plot datasets for downstream visualisation.
    """
    _energy_dataset:DPPlotData
    _force_dataset:DPPlotData
    _spin_dataset:DPPlotData
    _virial_dataset:DPPlotData
    def __init__(self, nep_txt_path: Path|str,
                 data_xyz_path: Path|str,
                 energy_out_path: Path|str,
                 force_out_path: Path|str,
                 virial_out_path: Path|str,
                 descriptor_path: Path|str,
                 spin_out_path: Path|str|None=None
                 ):
        """Initialise DeepMD result paths and optional spin output.

        Parameters
        ----------
        nep_txt_path : Path or str
            Location of the NEP model text file.
        data_xyz_path : Path or str
            Directory containing DeepMD ``.npy`` structural data.
        energy_out_path : Path or str
            Target file for per-atom energy comparisons.
        force_out_path : Path or str
            Target file for per-atom force comparisons.
        virial_out_path : Path or str
            Target file for per-atom virial comparisons.
        descriptor_path : Path or str
            Path to the descriptor metadata produced by DeepMD.
        spin_out_path : Path or str, optional
            Optional file storing spin-related outputs.
        """
        super().__init__(nep_txt_path, data_xyz_path,descriptor_path)
        self.energy_out_path = Path(energy_out_path)
        self.force_out_path = Path(force_out_path)
        self.spin_out_path = Path(spin_out_path) if spin_out_path is not None else None
        self.virial_out_path = Path(virial_out_path)

    @staticmethod
    def _collect_energy_sync(result_data: 'DeepmdResultData', dataset: NepPlotData, structure_indices):
        """Collect reference energies for the provided structure indices.

        Parameters
        ----------
        result_data : DeepmdResultData
            Source container providing structures and metadata.
        dataset : NepPlotData
            Plot dataset whose columns are being synchronised.
        structure_indices : Sequence[int]
            Indices of structures selected in the UI.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Row indices and energy values aligned with the dataset layout.
        """
        total_cols = dataset.data.all_data.shape[1] if dataset.data.all_data.ndim > 1 else 0
        target_width = max(total_cols - dataset.cols, 0)
        if target_width == 0:
            return np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        indices = result_data._normalize_structure_indices(structure_indices)
        if indices.size == 0:
            return np.array([], dtype=np.int64), np.empty((0, target_width), dtype=np.float32)
        structures = [result_data.structure.all_data[i] for i in indices]
        values = np.array([s.per_atom_energy for s in structures], dtype=np.float32).reshape(-1, target_width)
        return indices, values

    @staticmethod
    def _collect_force_sync(result_data: 'DeepmdResultData', dataset: NepPlotData, structure_indices):
        """Collect force values aligned with the provided structures.

        Parameters
        ----------
        result_data : DeepmdResultData
            Source container providing structures and metadata.
        dataset : NepPlotData
            Plot dataset whose columns are being synchronised.
        structure_indices : Sequence[int]
            Indices of structures selected in the UI.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Dataset row indices and force components arranged per atom or per structure.
        """
        total_cols = dataset.data.all_data.shape[1] if dataset.data.all_data.ndim > 1 else 0
        target_width = max(total_cols - dataset.cols, 0)
        if target_width == 0:
            return np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        indices = result_data._normalize_structure_indices(structure_indices)
        if indices.size == 0:
            return np.array([], dtype=np.int64), np.empty((0, target_width), dtype=np.float32)
        group_vals = dataset.group_array.all_data
        per_atom = bool(group_vals.size and np.unique(group_vals).size != group_vals.size)
        structures = [result_data.structure.all_data[i] for i in indices]
        if per_atom:
            row_idx = dataset.convert_index(indices)
            values = np.vstack([s.forces for s in structures]).astype(np.float32, copy=False)
        else:
            row_idx = indices
            values = np.array([np.linalg.norm(s.forces, axis=0) for s in structures], dtype=np.float32)
        return row_idx, values

    @staticmethod
    def _collect_virial_sync(result_data: 'DeepmdResultData', dataset: NepPlotData, structure_indices):
        """Collect virial tensors for structures that provide virial information.

        Parameters
        ----------
        result_data : DeepmdResultData
            Source container providing structures and metadata.
        dataset : NepPlotData
            Plot dataset whose columns are being synchronised.
        structure_indices : Sequence[int]
            Indices of structures selected in the UI.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Structure indices with virials and the corresponding tensor components.
        """
        total_cols = dataset.data.all_data.shape[1] if dataset.data.all_data.ndim > 1 else 0
        target_width = max(total_cols - dataset.cols, 0)
        if target_width == 0:
            return np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        indices = result_data._normalize_structure_indices(structure_indices)
        if indices.size == 0:
            return np.array([], dtype=np.int64), np.empty((0, target_width), dtype=np.float32)
        structures = [result_data.structure.all_data[i] for i in indices]
        mask = np.array([s.has_virial for s in structures], dtype=bool)
        if not mask.any():
            return np.array([], dtype=np.int64), np.empty((0, target_width), dtype=np.float32)
        selected_indices = indices[mask]
        values = np.vstack([structures[i].nep_virial for i, flag in enumerate(mask) if flag]).astype(np.float32,
                                                                                                     copy=False)
        return selected_indices, values

    @staticmethod
    def _collect_stress_sync(result_data: 'DeepmdResultData', dataset: NepPlotData, structure_indices):
        """Collect stress tensors derived from virials for the selected structures.

        Parameters
        ----------
        result_data : DeepmdResultData
            Source container providing structures and metadata.
        dataset : NepPlotData
            Plot dataset whose columns are being synchronised.
        structure_indices : Sequence[int]
            Indices of structures selected in the UI.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Structure indices and stress values expressed in eV/Angstrom^3.
        """
        total_cols = dataset.data.all_data.shape[1] if dataset.data.all_data.ndim > 1 else 0
        target_width = max(total_cols - dataset.cols, 0)
        if target_width == 0:
            return np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        indices = result_data._normalize_structure_indices(structure_indices)
        if indices.size == 0:
            return np.array([], dtype=np.int64), np.empty((0, target_width), dtype=np.float32)
        structures = [result_data.structure.all_data[i] for i in indices]
        mask = np.array([s.has_virial for s in structures], dtype=bool)
        if not mask.any():
            return np.array([], dtype=np.int64), np.empty((0, target_width), dtype=np.float32)
        selected_indices = indices[mask]
        virial_values = np.vstack([structures[i].nep_virial for i, flag in enumerate(mask) if flag]).astype(np.float32,
                                                                                                            copy=False)
        atoms = result_data.atoms_num_list[selected_indices].astype(np.float32)
        volumes = np.array([structures[i].volume for i, flag in enumerate(mask) if flag], dtype=np.float32)
        coeff = np.divide(atoms, volumes, out=np.zeros_like(atoms, dtype=np.float32), where=volumes != 0)[:, np.newaxis]
        stress_values = virial_values * coeff * 160.21766208
        return selected_indices, stress_values.astype(np.float32, copy=False)

    STRUCTURE_SYNC_RULES = {
        'energy': StructureSyncRule('energy', 'x_cols', _collect_energy_sync),
        'force': StructureSyncRule('force', 'x_cols', _collect_force_sync),
        'virial': StructureSyncRule('virial', 'x_cols', _collect_virial_sync),
        'stress': StructureSyncRule('stress', 'x_cols', _collect_stress_sync),
    }

    @classmethod
    def from_path(cls, path, *, structures: list[Structure] | None = None):
        """Create an instance from a DeepMD dataset directory.

        Parameters
        ----------
        path : Path or str
            Directory that contains DeepMD ``set.*`` data and outputs.
        structures : list[Structure], optional
            Pre-loaded structures to attach instead of reading from disk.

        Returns
        -------
        DeepmdResultData
            Configured loader pointing at the resolved dataset.
        """
        dataset_path = as_path(path)
        # file_name=dataset_path.name
        nep_txt_path = dataset_path.with_name(f"nep.txt")
        if not nep_txt_path.exists():
            nep_txt_path = module_path/ "Config/nep89.txt"
        descriptor_path = dataset_path.with_name(f"descriptor.out")
        e_path = list(dataset_path.parent.glob("*.e_peratom.out") )
        if e_path:
            e_path = e_path[0]
            suffix = (e_path.name.replace(".e_peratom.out",""))
        else:
            suffix="detail"
        energy_out_path = dataset_path.with_name(f"{suffix}.e_peratom.out")
        force_out_path = dataset_path.with_name(f"{suffix}.fr.out")
        if  not force_out_path.exists():
            force_out_path = dataset_path.with_name(f"{suffix}.f.out")
        # stress_out_path = dataset_path.with_name(f"{suffix}.v.out")
        virial_out_path = dataset_path.with_name(f"{suffix}.v_peratom.out")
        spin_out_path=  dataset_path.with_name(f"{suffix}.fm.out")
        if not spin_out_path.exists():
            spin_out_path = None
        inst = cls(nep_txt_path,dataset_path,energy_out_path,force_out_path,virial_out_path,descriptor_path,spin_out_path=spin_out_path)
        # DeepMD loader ignores in-memory structures; it reads its own format.
        return inst
    def load_structures(self):
        """Load structures from DeepMD numpy sets into the local dataset.

        Notes
        -----
        - Recognises DeepMD ``set.*`` partitions and aggregates per-set arrays.
        - Respects the optional ``cancel_event`` for graceful cancellation.

        Examples
        --------
        >>> # Constructed via DeepmdResultData.from_path(...)  # doctest: +SKIP
        """
        structures = load_npy_structure(self.data_xyz_path,order_file=self.energy_out_path, cancel_event=self.cancel_event)
        self._atoms_dataset = StructureData(structures)
        self.atoms_num_list = np.array([len(s) for s in structures])
    @property
    def datasets(self):
        """Return the datasets exposed to the UI in canonical order."""
        if self.spin_out_path is None:
            return [self.energy, self.force,  self.virial, self.descriptor]
        else:
            return [self.energy, self.force,self.spin, self.virial, self.descriptor]
    @property
    def energy(self):
        """Return the per-atom energy dataset."""
        return self._energy_dataset
    @property
    def force(self):
        """Return the per-atom force dataset."""
        return self._force_dataset
    @property
    def spin(self):
        """Return the per-atom spin dataset."""
        return self._spin_dataset
    @property
    def virial(self):
        """Return the per-atom virial dataset."""
        return self._virial_dataset
    def _load_dataset(self) -> None:
        """Populate plot datasets from cached files or NEP recalculation."""
        if self._should_recalculate( ):
            energy_array, force_array, virial_array = self._recalculate_and_save( )
        else:
            energy_array=read_nep_out_file(self.energy_out_path,ndmin=2)
            force_array=read_nep_out_file(self.force_out_path,ndmin=2)
            virial_array=read_nep_out_file(self.virial_out_path,ndmin=2)
            if energy_array.shape[0]!=self.atoms_num_list.shape[0]:
                self.energy_out_path.unlink(True)
                self.force_out_path.unlink(True)
                if self.spin_out_path is not None:
                    self.spin_out_path.unlink(True)
                self.virial_out_path.unlink(True)
                return self._load_dataset()
        self._energy_dataset = DPPlotData(energy_array, title="energy")
        default_forces = Config.get("widget", "forces_data", "Row")
        if force_array.size != 0 and default_forces == "Norm":
            force_array = aggregate_per_atom_to_structure(force_array, self.atoms_num_list, map_func=np.linalg.norm, axis=0)
            self._force_dataset = DPPlotData(force_array, title="force")
        else:
            self._force_dataset = DPPlotData(force_array, group_list=self.atoms_num_list, title="force")
        if self.spin_out_path is not None:
            spin_array=read_nep_out_file(self.spin_out_path,ndmin=2)
            group_list=[s.spin_num for s in self.structure.now_data]
            if (np.sum(group_list))!=0:
                self._spin_dataset = DPPlotData(spin_array,  group_list=group_list, title="spin")
            else:
                self.spin_out_path = None
        self._virial_dataset = DPPlotData(virial_array, title="virial")
    def _should_recalculate(self  ) -> bool:
        """Return ``True`` when cached output files are missing.

        Returns
        -------
        bool
            ``True`` if any required DeepMD output file is absent.
        """
        output_files_exist = any([
            self.energy_out_path.exists(),
            self.force_out_path.exists(),
            self.virial_out_path.exists()
        ])
        return   not output_files_exist
    def _save_energy_data(self, potentials: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Persist reference and predicted per-atom energies side by side.

        Parameters
        ----------
        potentials : numpy.ndarray
            Predicted per-structure potential energies from the NEP calculator.

        Returns
        -------
        numpy.ndarray
            Two-column array containing reference and predicted per-atom energies.
        """

        ref_energies = np.array([s.energy if s.has_energy else np.nan for s in self.structure.now_data], dtype=np.float32)
        energy_array = concat_nep_dft_array(potentials,ref_energies,deepmd=True)


        energy_array=energy_array/ self.atoms_num_list.reshape(-1, 1)
        energy_array = energy_array.astype(np.float32)
        if energy_array.size != 0:
            np.savetxt(self.energy_out_path, energy_array, fmt='%10.8f')
        return energy_array
    def _save_force_data(self, forces: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Persist reference and predicted forces for every atom.

        Parameters
        ----------
        forces : numpy.ndarray
            Predicted forces arranged either per-atom or per-structure.

        Returns
        -------
        numpy.ndarray
            Two-column array storing reference and predicted forces.
        """
        ref_forces = np.vstack([s.forces if s.has_forces else np.full((len(s),3 ), np.nan) for s in self.structure.now_data], dtype=np.float32)

        forces_array = concat_nep_dft_array(forces,ref_forces,deepmd=True)
        if forces_array.size != 0:
            np.savetxt(self.force_out_path, forces_array, fmt='%10.8f')
        return forces_array
    def _save_virial_and_data(self, virials: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Persist reference and predicted virial or stress tensors.

        Parameters
        ----------
        virials : numpy.ndarray
            Predicted virial tensors returned by the NEP calculator.

        Returns
        -------
        numpy.ndarray
            Two-column array with reference and predicted virial components.
        """
        ref_virials = np.vstack([s.nep_virial if s.has_virial else [np.nan]*6 for s in self.structure.now_data ], dtype=np.float32)
        virials_array = concat_nep_dft_array(virials,ref_virials,deepmd=True)

        if virials_array.size != 0:
            np.savetxt(self.virial_out_path, virials_array, fmt='%10.8f')
        return virials_array
    def _recalculate_and_save(self ):
        """Recompute NEP predictions and persist freshly generated outputs.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Energy, force, and virial arrays that were written to disk.
        """
        try:
            nep_potentials_list, nep_forces_list, nep_virials_list = self.nep_calc.calculate(self.structure.now_data.tolist())
            nep_potentials_array=np.array(nep_potentials_list)
            nep_forces_array=np.vstack(nep_forces_list)
            nep_virials_array=np.vstack(nep_virials_list)
            if nep_potentials_array.size == 0:
                MessageManager.send_warning_message("The nep calculator fails to calculate the potentials, use the original potentials instead.")
            energy_array = self._save_energy_data(nep_potentials_array)
            force_array = self._save_force_data(nep_forces_array)
            virial_array = self._save_virial_and_data(nep_virials_array[:, [0, 4, 8, 1, 5, 6]])
            return energy_array,force_array,virial_array
        except Exception as e:
            # logger.debug(traceback.format_exc())
            MessageManager.send_error_message(f"An error occurred while running NEP3 calculator: {e}")
            return np.array([]), np.array([]), np.array([])
    def export_model_xyz(self, save_path: PathLike) -> None:
        """Export current and removed structures into dedicated directories.

        Parameters
        ----------
        save_path : Path or str
            Destination directory that will receive ``export_good_model`` and
            ``export_remove_model`` folders.
        """
        target = as_path(save_path)
        try:
            save_npy_structure(str(target / 'export_good_model'), self.structure.now_data)
            save_npy_structure(str(target / 'export_remove_model'), self.structure.remove_data)
            MessageManager.send_info_message(f'File exported to: {target}')
        except Exception:
            MessageManager.send_info_message(
                'An unknown error occurred while saving. The error message has been output to the log!'
            )
            logger.error(traceback.format_exc())
