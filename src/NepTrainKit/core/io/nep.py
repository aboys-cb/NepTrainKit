#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Core NEP result data loaders and helpers."""
import traceback
from loguru import logger
from pathlib import Path
import numpy.typing as npt
import numpy as np
from NepTrainKit import module_path
from NepTrainKit.core import MessageManager
from NepTrainKit.core.structure import Structure
from NepTrainKit.paths import as_path
from NepTrainKit.config import Config
from .base import NepPlotData, ResultData, StructureSyncRule
from NepTrainKit.core.utils import read_nep_out_file, check_fullbatch, read_nep_in, aggregate_per_atom_to_structure,concat_nep_dft_array
from NepTrainKit.core.types import ForcesMode


class NepTrainResultData(ResultData):
    """Result loader for NEP training outputs with energy, force, stress, and virial datasets.

    The loader normalises NEP predictions into plot-ready datasets and registers
    synchronisation rules used by the UI.

    Examples
    --------
    >>> from NepTrainKit.core.io import NepTrainResultData
    # Load the xyz file
    >>> result_dataset = NepTrainResultData.from_path(r"D:/Desktop/dataset3635-addD3/train.xyz")
    >>> result_dataset.load()
    >>> print(result_dataset)
    # Select structures at indices 0 and 10
    >>> result_dataset.select([0, 10])
    >>> print(result_dataset)
    # Delete the selected structures
    >>> result_dataset.delete_selected()
    >>> print(result_dataset)
    # Get the indices of the 10 points with the largest energy error
    >>> index = result_dataset.energy.get_max_error_index(10)
    # Select the 10 points with the largest energy error and delete them
    >>> result_dataset.select(index)
    >>> result_dataset.delete_selected()
    >>> print(result_dataset)
    # Revoke the last deletion
    >>> result_dataset.revoke()
    # Perform farthest point sampling (normal global sampling)
    >>> index, reverse = result_dataset.sparse_descriptor_selection(100, 0.001, False)
    # Perform sampling within a region (select the first 300 structures)
    >>> index = result_dataset.select_structures_by_index(":300")
    >>> result_dataset.select(index)
    >>> index, reverse = result_dataset.sparse_descriptor_selection(100, 0.001, True)
    # Uncheck or inverse select based on the reverse flag
    >>> if reverse:
    >>>     result_dataset.uncheck(index)
    >>> else:
    >>>     result_dataset.select(index)
    >>>     result_dataset.inverse_select()
    >>> print(result_dataset)

    """
    _energy_dataset: NepPlotData
    _force_dataset: NepPlotData
    _stress_dataset: NepPlotData
    _virial_dataset: NepPlotData
    @staticmethod
    def _collect_energy_sync(result_data: 'NepTrainResultData', dataset: NepPlotData, structure_indices):
        """Collect reference energies for the provided structure indices.
        
        Parameters
        ----------
        result_data : NepTrainResultData
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
    def _collect_force_sync(result_data: 'NepTrainResultData', dataset: NepPlotData, structure_indices):
        """Collect force values aligned with the provided structures.
        
        Parameters
        ----------
        result_data : NepTrainResultData
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
    def _collect_virial_sync(result_data: 'NepTrainResultData', dataset: NepPlotData, structure_indices):
        """Collect virial tensors for structures that provide virial information.
        
        Parameters
        ----------
        result_data : NepTrainResultData
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
        values = np.vstack([structures[i].nep_virial for i, flag in enumerate(mask) if flag]).astype(np.float32, copy=False)
        return selected_indices, values
    @staticmethod
    def _collect_stress_sync(result_data: 'NepTrainResultData', dataset: NepPlotData, structure_indices):
        """Collect stress tensors derived from virials for the selected structures.
        
        Parameters
        ----------
        result_data : NepTrainResultData
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
        virial_values = np.vstack([structures[i].nep_virial for i, flag in enumerate(mask) if flag]).astype(np.float32, copy=False)
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

    def __init__(self,
                 nep_txt_path: Path|str,
                 data_xyz_path: Path|str,
                 energy_out_path: Path|str,
                 force_out_path: Path|str,
                 stress_out_path: Path|str,
                 virial_out_path: Path|str,
                 descriptor_path: Path|str
                 ):
        """Initialise NEP training result paths and metadata.
        
        Parameters
        ----------
        nep_txt_path : Path or str
            Path to the NEP model file.
        data_xyz_path : Path or str
            Directory containing NEP dataset structures.
        energy_out_path : Path or str
            Output file capturing NEP versus reference energies.
        force_out_path : Path or str
            Output file capturing NEP versus reference forces.
        stress_out_path : Path or str
            Output file capturing NEP versus reference stresses.
        virial_out_path : Path or str
            Output file capturing NEP versus reference virials.
        descriptor_path : Path or str
            Descriptor file produced alongside the dataset.
        """
        super().__init__(nep_txt_path,data_xyz_path,descriptor_path)
        self.energy_out_path = Path(energy_out_path)
        self.force_out_path = Path(force_out_path)
        self.stress_out_path = Path(stress_out_path)
        self.virial_out_path = Path(virial_out_path)
        self.has_virial_structure_index_list = None
    @property
    def datasets(self):
        """Return datasets exposed to the UI in display order."""
        # return [self.energy, self.stress,self.virial, self.descriptor]
        return [self.energy,self.force,self.stress,self.virial, self.descriptor]
    @property
    def energy(self):
        """Return the per-structure energy dataset."""
        return self._energy_dataset
    @property
    def force(self):
        """Return the force dataset respecting per-atom settings."""
        return self._force_dataset
    @property
    def stress(self):
        """Return the stress dataset derived from predicted virials."""
        return self._stress_dataset
    @property
    def virial(self):
        """Return the per-structure virial dataset."""
        return self._virial_dataset
    @classmethod
    def from_path(cls, path ,model_type=0, *, structures: list[Structure] | None = None)->"NepTrainResultData":
        """Create an instance from a NEP result directory.
        
        Parameters
        ----------
        path : PathLike
            Directory containing NEP outputs and descriptors.
        model_type : int, optional
            NEP model type hint used to select descriptor fallbacks.
        structures : list[Structure], optional
            Pre-loaded structures to attach instead of reading from disk.
        
        Returns
        -------
        NepTrainResultData
            Configured loader bound to the resolved directory.
        """
        dataset_path = as_path(path)
        file_name=dataset_path.stem
        nep_txt_path = dataset_path.with_name(f"nep.txt")
        if not nep_txt_path.exists()  :
            nep_txt_path = module_path/ "Config/nep89.txt"
            MessageManager.send_warning_message(f"no find nep.txt; the program will use nep89 instead.")
        elif model_type>2:
            nep_txt_path = module_path/ "Config/nep89.txt"
            MessageManager.send_warning_message(f"NEPKit currently does not support model_type={model_type}; the program will use nep89 instead.")
        energy_out_path = dataset_path.with_name(f"energy_{file_name}.out")
        force_out_path = dataset_path.with_name(f"force_{file_name}.out")
        stress_out_path = dataset_path.with_name(f"stress_{file_name}.out")
        virial_out_path = dataset_path.with_name(f"virial_{file_name}.out")
        if file_name=="train":
            descriptor_path = dataset_path.with_name(f"descriptor.out")
        else:
            descriptor_path = dataset_path.with_name(f"descriptor_{file_name}.out")
        inst = cls(nep_txt_path,dataset_path,energy_out_path,force_out_path,stress_out_path,virial_out_path,descriptor_path)
        if structures is not None:
            try:
                inst.set_structures(structures)
            except Exception:
                pass
        return inst
    def _load_dataset(self) -> None:
        """Populate plot datasets from cached outputs or by recalculating with NEP."""
        nep_in = read_nep_in(self.data_xyz_path.with_name("nep.in"))
        if self._should_recalculate(nep_in):
            energy_array, force_array, virial_array, stress_array = self._recalculate_and_save( )
        else:
            energy_array = read_nep_out_file(self.energy_out_path, dtype=np.float32,ndmin=2)
            force_array = read_nep_out_file(self.force_out_path, dtype=np.float32,ndmin=2)
            virial_array = read_nep_out_file(self.virial_out_path, dtype=np.float32,ndmin=2)
            stress_array = read_nep_out_file(self.stress_out_path, dtype=np.float32,ndmin=2)
            if energy_array.shape[0]!=self.atoms_num_list.shape[0]:
                self.energy_out_path.unlink(True)
                self.force_out_path.unlink(True)
                self.virial_out_path.unlink(True)
                self.stress_out_path.unlink(True)
                return self._load_dataset()
        self._energy_dataset = NepPlotData(energy_array, title="energy")
        default_forces = Config.get("widget", "forces_data", ForcesMode.Raw)
        if force_array.size != 0 and default_forces == ForcesMode.Norm:
            force_array = aggregate_per_atom_to_structure(force_array, self.atoms_num_list, map_func=np.linalg.norm, axis=0)
            self._force_dataset = NepPlotData(force_array, title="force")
        else:
            self._force_dataset = NepPlotData(force_array, group_list=self.atoms_num_list, title="force")
        if float(nep_in.get("lambda_v", 1)) != 0:
            self._stress_dataset = NepPlotData(stress_array,  title="stress")
            self._virial_dataset = NepPlotData(virial_array, title="virial")
        else:
            self._stress_dataset = NepPlotData([], title="stress")
            self._virial_dataset = NepPlotData([], title="virial")
    def _should_recalculate(self, nep_in: dict) -> bool:
        """Return ``True`` when cached outputs are missing or inconsistent.
        
        Parameters
        ----------
        nep_in : dict
            Parsed contents of ``nep.in`` controlling batching behaviour.
        
        Returns
        -------
        bool
            ``True`` if NEP predictions need to be regenerated.
        """
        output_files_exist = all([
            self.energy_out_path.exists(),
            self.force_out_path.exists(),
            self.stress_out_path.exists(),
            self.virial_out_path.exists()
        ])
        return not check_fullbatch(nep_in, len(self.atoms_num_list)) or not output_files_exist
    def _save_energy_data(self, potentials:npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Persist per-structure energy comparisons to disk.
        
        Parameters
        ----------
        potentials : numpy.ndarray
            Potential energies predicted by the NEP calculator.
        
        Returns
        -------
        numpy.ndarray
            Two-column array with predicted and reference energies per structure.
        """

        ref_energies = np.array([s.energy if s.has_energy else np.nan for s in self.structure.now_data], dtype=np.float32)
        energy_array = concat_nep_dft_array(potentials,ref_energies)

        energy_array=energy_array/ self.atoms_num_list.reshape(-1, 1)
        energy_array = energy_array.astype(np.float32)
        if energy_array.size != 0:
            np.savetxt(self.energy_out_path, energy_array, fmt='%10.8f')
        return energy_array
    def _save_force_data(self, forces: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Persist force comparisons to disk with reference and predicted values.
        
        Parameters
        ----------
        forces : numpy.ndarray
            Forces predicted by the NEP calculator.
        
        Returns
        -------
        numpy.ndarray
            Two-column array containing reference and predicted forces.
        """

        ref_forces = np.vstack([s.forces if s.has_forces else np.full((len(s),3 ), np.nan) for s in self.structure.now_data], dtype=np.float32)
        forces_array = concat_nep_dft_array(forces,ref_forces)

        if forces_array.size != 0:
            np.savetxt(self.force_out_path, forces_array, fmt='%10.8f')
        return forces_array
    def _save_virial_and_stress_data(self, virials: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32],npt.NDArray[np.float32]]:
        """Persist virial tensors and derived stresses to disk.
        
        Parameters
        ----------
        virials : numpy.ndarray
            Predicted virial components arranged per structure.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Tuple of (virial_array, stress_array) stored for later plotting.
        """
        coefficient = (self.atoms_num_list / np.array([s.volume for s in self.structure.now_data ]))[:, np.newaxis]

        ref_virials = np.vstack([s.nep_virial if s.has_virial else [np.nan]*6 for s in self.structure.now_data ], dtype=np.float32)
        virials_array = concat_nep_dft_array(virials,ref_virials)

        stress_array = virials_array * coefficient * 160.21766208  # Unit conversion to MPa
        stress_array = stress_array.astype(np.float32)
        if virials_array.size != 0:
            np.savetxt(self.virial_out_path, virials_array, fmt='%10.8f')
        if stress_array.size != 0:
            np.savetxt(self.stress_out_path, stress_array, fmt='%10.8f')
        return virials_array, stress_array
    def _recalculate_and_save(self ):
        """Recompute NEP predictions and update on-disk comparison files.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Energy, force, virial, and stress arrays generated by the recalculation.
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
            virial_array, stress_array = self._save_virial_and_stress_data(nep_virials_array[:, [0, 4, 8, 1, 5, 6]])
            self.write_prediction()
            return energy_array,force_array,virial_array, stress_array
        except Exception as e:

            logger.debug(traceback.format_exc())
            MessageManager.send_error_message(f"An error occurred while running NEP3 calculator: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
class NepPolarizabilityResultData(ResultData):
    """Result loader for NEP polarizability evaluations."""
    _polarizability_diagonal_dataset: NepPlotData
    _polarizability_no_diagonal_dataset: NepPlotData
    def __init__(self,
                 nep_txt_path: Path|str,
                 data_xyz_path: Path|str,
                 polarizability_out_path: Path|str,
                 descriptor_path: Path|str
                 ):
        """Initialise NEP polarizability result paths.
        
        Parameters
        ----------
        nep_txt_path : Path or str
            Path to the NEP model file.
        data_xyz_path : Path or str
            Directory containing NEP dataset structures.
        polarizability_out_path : Path or str
            Output file storing polarizability comparisons.
        descriptor_path : Path or str
            Descriptor file produced alongside the dataset.
        """
        super().__init__(nep_txt_path,data_xyz_path,descriptor_path)
        self.polarizability_out_path = Path(polarizability_out_path)
        # self.nep_calc = NepCalculator(model_file=self.nep_txt_path.as_posix(),
        #                               backend=NepBackend.CPU,
        #                               batch_size=Config.getint("nep", "gpu_batch_size", 1000)
        #                               )
    @property
    def datasets(self):
        """Return the polarizability datasets in display order."""
        return [self.polarizability_diagonal,self.polarizability_no_diagonal, self.descriptor]
    @property
    def polarizability_diagonal(self):
        """Return the diagonal polarizability dataset."""
        return self._polarizability_diagonal_dataset
    @property
    def polarizability_no_diagonal(self):
        """Return the off-diagonal polarizability dataset."""
        return self._polarizability_no_diagonal_dataset
    @property
    def descriptor(self):
        """Return the descriptor dataset associated with the polarizability run."""
        return self._descriptor_dataset
    @classmethod
    def from_path(cls, path, *, structures: list[Structure] | None = None ):
        """Create a polarizability loader from a NEP dataset directory.
        
        Parameters
        ----------
        path : PathLike
            Directory containing NEP outputs.
        structures : list[Structure], optional
            Pre-loaded structures to attach instead of reading from disk.
        
        Returns
        -------
        NepPolarizabilityResultData
            Configured loader bound to the resolved directory.
        """
        dataset_path = as_path(path)
        file_name = dataset_path.stem
        nep_txt_path = dataset_path.with_name(f"nep.txt")
        polarizability_out_path = dataset_path.with_name(f"polarizability_{file_name}.out")
        if file_name == "train":
            descriptor_path = dataset_path.with_name(f"descriptor.out")
        else:
            descriptor_path = dataset_path.with_name(f"descriptor_{file_name}.out")
        inst = cls(nep_txt_path, dataset_path, polarizability_out_path, descriptor_path)
        if structures is not None:
            try:
                inst.set_structures(structures)
            except Exception:
                pass
        return inst
    def _should_recalculate(self, nep_in: dict) -> bool:
        """Return ``True`` when cached polarizability outputs are missing or stale.
        
        Parameters
        ----------
        nep_in : dict
            Parsed ``nep.in`` metadata controlling batching behaviour.
        
        Returns
        -------
        bool
            ``True`` if NEP polarizability predictions must be regenerated.
        """
        output_files_exist = all([
            self.polarizability_out_path.exists(),
        ])
        return not check_fullbatch(nep_in, len(self.atoms_num_list)) or not output_files_exist
    def _recalculate_and_save(self ):
        """Recompute polarizability predictions and persist them to disk.
        
        Returns
        -------
        numpy.ndarray
            Combined NEP and reference polarizability values.
        """
        try:
            # nep_polarizability_array = run_nep3_calculator_process(self.nep_txt_path.as_posix(),
            #                                                        self.structure.now_data, "polarizability")
            nep_polarizability_array = self.nep_calc.get_structures_polarizability(self.structure.now_data.tolist())
            # nep_polarizability_array=self.nep_calc_thread.func_result
            if nep_polarizability_array.size == 0:
                MessageManager.send_warning_message("The nep calculator fails to calculate the polarizability, use the original polarizability instead.")
            nep_polarizability_array = self._save_polarizability_data(  nep_polarizability_array)
            self.write_prediction()
        except Exception as e:
            # logger.debug(traceback.format_exc())
            MessageManager.send_error_message(f"An error occurred while running NEP3 calculator: {e}")
            nep_polarizability_array = np.array([])
        return nep_polarizability_array
    def _save_polarizability_data(self, polarizability: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Persist polarizability comparisons to disk.
        
        Parameters
        ----------
        polarizability : numpy.ndarray
            Polarizability values predicted by the NEP calculator.
        
        Returns
        -------
        numpy.ndarray
            Array containing predicted and reference polarizability components.
        """
        nep_polarizability_array = polarizability / (self.atoms_num_list[:, np.newaxis])
        try:
            ref_polarizability = np.vstack([s.nep_polarizability for s in self.structure.now_data], dtype=np.float32)
            if polarizability.size == 0:
                polarizability_array = np.column_stack([ref_polarizability, ref_polarizability])
            else:
                polarizability_array = np.column_stack([nep_polarizability_array,
                                                        ref_polarizability
                                                        ])
        except Exception:
            # logger.debug(traceback.format_exc())
            polarizability_array = np.column_stack([polarizability, polarizability])
        polarizability_array = polarizability_array.astype(np.float32)
        if polarizability_array.size != 0:
            np.savetxt(self.polarizability_out_path, polarizability_array, fmt='%10.8f')
        return polarizability_array
    def _load_dataset(self) -> None:
        """Populate polarizability datasets from cached outputs or by recalculating."""
        nep_in = read_nep_in(self.data_xyz_path.with_name("nep.in"))
        if self._should_recalculate(nep_in):
            polarizability_array = self._recalculate_and_save( )
        else:
            polarizability_array= read_nep_out_file(self.polarizability_out_path, dtype=np.float32,ndmin=2)
            if polarizability_array.shape[0]!=self.atoms_num_list.shape[0]:
                self.polarizability_out_path.unlink()
                return self._load_dataset()
        self._polarizability_diagonal_dataset = NepPlotData(polarizability_array[:, [0,1,2,6,7,8]], title="Polar Diag")
        self._polarizability_no_diagonal_dataset = NepPlotData(polarizability_array[:, [3,4,5,9,10,11]], title="Polar NoDiag")
class NepDipoleResultData(ResultData):
    """Result loader for NEP dipole predictions."""
    _dipole_dataset: NepPlotData
    def __init__(self,
                 nep_txt_path: Path|str,
                 data_xyz_path: Path|str,
                 dipole_out_path: Path|str,
                 descriptor_path: Path|str
                 ):
        """Initialise NEP dipole result paths.
        
        Parameters
        ----------
        nep_txt_path : Path or str
            Path to the NEP model file.
        data_xyz_path : Path or str
            Directory containing NEP dataset structures.
        dipole_out_path : Path or str
            Output file storing dipole comparisons.
        descriptor_path : Path or str
            Descriptor file produced alongside the dataset.
        """
        super().__init__(nep_txt_path, data_xyz_path, descriptor_path)
        self.dipole_out_path = Path(dipole_out_path)
        # self.nep_calc = NepCalculator(model_file=self.nep_txt_path.as_posix(),
        #                      backend=NepBackend.CPU,
        #                      batch_size=Config.getint("nep", "gpu_batch_size", 1000)
        #                      )
    @property
    def datasets(self):
        """Return the dipole datasets in display order."""
        return [self.dipole , self.descriptor]
    @property
    def dipole(self):
        """Return the dipole dataset."""
        return self._dipole_dataset
    @property
    def descriptor(self):
        """Return the descriptor dataset associated with the dipole run."""
        return self._descriptor_dataset
    @classmethod
    def from_path(cls, path, *, structures: list[Structure] | None = None ):
        """Create a dipole loader from a NEP dataset directory.
        
        Parameters
        ----------
        path : PathLike
            Directory containing NEP outputs.
        structures : list[Structure], optional
            Pre-loaded structures to attach instead of reading from disk.
        
        Returns
        -------
        NepDipoleResultData
            Configured loader bound to the resolved directory.
        """
        dataset_path = as_path(path)
        file_name = dataset_path.stem
        nep_txt_path = dataset_path.with_name(f"nep.txt")
        polarizability_out_path = dataset_path.with_name(f"dipole_{file_name}.out")
        if file_name == "train":
            descriptor_path = dataset_path.with_name(f"descriptor.out")
        else:
            descriptor_path = dataset_path.with_name(f"descriptor_{file_name}.out")
        inst = cls(nep_txt_path, dataset_path, polarizability_out_path, descriptor_path)
        if structures is not None:
            try:
                inst.set_structures(structures)
            except Exception:
                pass
        return inst
    def _should_recalculate(self, nep_in: dict) -> bool:
        """Return ``True`` when cached dipole outputs are missing or stale.
        
        Parameters
        ----------
        nep_in : dict
            Parsed ``nep.in`` metadata controlling batching behaviour.
        
        Returns
        -------
        bool
            ``True`` if NEP dipole predictions must be regenerated.
        """
        output_files_exist = all([
            self.dipole_out_path.exists(),
        ])
        return not check_fullbatch(nep_in, len(self.atoms_num_list)) or not output_files_exist
    def _recalculate_and_save(self ):
        """Recompute dipole predictions and persist them to disk.
        
        Returns
        -------
        numpy.ndarray
            Dipole array written to disk.
        """
        try:
            # nep_dipole_array = run_nep3_calculator_process(self.nep_txt_path.as_posix(),
            #                                                self.structure.now_data, "dipole")
            nep_dipole_array = self.nep_calc.get_structures_dipole(self.structure.now_data.tolist())
            # nep_dipole_array=self.nep_calc_thread.func_result
            if nep_dipole_array.size == 0:
                MessageManager.send_warning_message("The nep calculator fails to calculate the dipole, use the original dipole instead.")
            nep_dipole_array = self._save_dipole_data(  nep_dipole_array)
            self.write_prediction()
        except Exception as e:
            # logger.debug(traceback.format_exc())
            MessageManager.send_error_message(f"An error occurred while running NEP3 calculator: {e}")
            nep_dipole_array = np.array([])
        return nep_dipole_array
    def _save_dipole_data(self, dipole: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Persist dipole comparisons to disk.
        
        Parameters
        ----------
        dipole : numpy.ndarray
            Dipole values predicted by the NEP calculator.
        
        Returns
        -------
        numpy.ndarray
            Array containing predicted and reference dipole components.
        """
        nep_dipole_array = dipole / (self.atoms_num_list[:, np.newaxis])
        try:
            ref_dipole = np.vstack([s.nep_dipole for s in self.structure.now_data], dtype=np.float32)
            if dipole.size == 0:
                dipole_array = np.column_stack([ref_dipole, ref_dipole])
            else:
                dipole_array = np.column_stack([nep_dipole_array,
                                            ref_dipole
                                                    ])
        except Exception:
            # logger.debug(traceback.format_exc())
            dipole_array = np.column_stack([nep_dipole_array, nep_dipole_array])
        dipole_array = dipole_array.astype(np.float32)
        if dipole_array.size != 0:
            np.savetxt(self.dipole_out_path, dipole_array, fmt='%10.8f')
        return dipole_array
    def _load_dataset(self) -> None:
        """Populate dipole datasets from cached outputs or by recalculating."""
        nep_in = read_nep_in(self.data_xyz_path.with_name("nep.in"))
        if self._should_recalculate(nep_in):
            dipole_array = self._recalculate_and_save( )
        else:
            dipole_array= read_nep_out_file(self.dipole_out_path, dtype=np.float32,ndmin=2)
            if dipole_array.shape[0]!=self.atoms_num_list.shape[0]:
                self.dipole_out_path.unlink()
                return self._load_dataset()
        self._dipole_dataset = NepPlotData(dipole_array, title="dipole")
