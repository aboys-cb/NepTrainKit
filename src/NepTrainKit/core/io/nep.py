#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Core NEP result data loaders and helpers."""
import traceback
import json
import hashlib
from loguru import logger
from pathlib import Path
from typing import Any
import numpy.typing as npt
import numpy as np
try:
    from nep_adapters import (
        ChargePrediction,
        SpinPrediction,
        __version__ as nep_adapters_version,
        inspect_model,
    )
except ImportError:
    ChargePrediction = None  # type: ignore[assignment]
    SpinPrediction = None  # type: ignore[assignment]
    nep_adapters_version = "0.0.0"

    def inspect_model(model_path, **kwargs):
        raise ImportError("nep-adapters is not installed")
from NepTrainKit import module_path
from NepTrainKit.core import MessageManager
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.precision import get_storage_float_dtype
from NepTrainKit.paths import as_path
from NepTrainKit.paths import get_bundled_nep89_path
from NepTrainKit.config import Config
from .base import (
    NepPlotData,
    ResultData,
    StructureSyncRule,
    collect_energy_sync,
    collect_force_sync,
    collect_stress_sync,
    collect_virial_sync,
)
from NepTrainKit.core.utils import (
    aggregate_per_atom_to_structure,
    check_fullbatch,
    concat_nep_dft_array,
    read_nep_in,
    read_nep_out_file,
)
from NepTrainKit.core.types import ForcesMode, parse_forces_mode


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
    _spin_force_dataset: NepPlotData | None
    STRUCTURE_SYNC_RULES = {
        'energy': StructureSyncRule('energy', 'x_cols', collect_energy_sync),
        'force': StructureSyncRule('force', 'x_cols', collect_force_sync),
        'virial': StructureSyncRule('virial', 'x_cols', collect_virial_sync),
        'stress': StructureSyncRule('stress', 'x_cols', collect_stress_sync),
    }

    def __init__(self,
                 nep_txt_path: Path|str,
                 data_xyz_path: Path|str,
                 energy_out_path: Path|str,
                 force_out_path: Path|str,
                 stress_out_path: Path|str,
                 virial_out_path: Path|str,
                 descriptor_path: Path|str,
                 charge_out_path: Path|str|None = None,
                 bec_out_path: Path|str|None = None,
                 charge_model: bool | None = None,
                 spin_force_out_path: Path|str|None = None,
                 spin_model: bool | None = None,
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
        spin_force_out_path : Path or str, optional
            Optional file capturing magnetic forces (mforce) when available.
        """
        super().__init__(nep_txt_path,data_xyz_path,descriptor_path)
        self.energy_out_path = Path(energy_out_path)
        self.force_out_path = Path(force_out_path)
        self.stress_out_path = Path(stress_out_path)
        self.virial_out_path = Path(virial_out_path)
        self.charge_out_path = Path(charge_out_path) if charge_out_path else None
        self.bec_out_path = Path(bec_out_path) if bec_out_path else None
        if charge_model is None and spin_model is None:
            model_info = inspect_model(self.nep_txt_path)
            detected_charge = model_info.model_type == "charge"
            detected_spin = model_info.model_type == "spin"
        else:
            detected_charge = bool(charge_model)
            detected_spin = bool(spin_model)
        self.is_charge_model = detected_charge if charge_model is None else bool(charge_model)
        self.is_spin_model = detected_spin if spin_model is None else bool(spin_model)
        self.spin_force_out_path = Path(spin_force_out_path) if spin_force_out_path else None
        self.prediction_meta_path = self.energy_out_path.parent / "prediction.meta.json"
        self.has_virial_structure_index_list = None
        self._bec_dataset = None
        self._spin_force_dataset = None
        self._force_vector_dataset = None
        self._spin_force_vector_dataset = None
        self._pending_prediction: Prediction | None = None
    @property
    def datasets(self):
        """Return datasets exposed to the UI in display order."""
        items = [self.energy, self.force]
        if not getattr(self, "is_spin_model", False):
            items.append(self.stress)
        items.append(self.virial)
        if getattr(self, "_spin_force_dataset", None) is not None:
            items.append(self.spin_force)
        if getattr(self, "_bec_dataset", None) is not None:
            items.append(self.bec)
        items.append(self.descriptor)
        return items
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
    @property
    def bec(self):
        """Return the per-atom Born effective charge dataset when available."""
        return self._bec_dataset
    @property
    def spin_force(self):
        """Return the magnetic force dataset when available."""
        return self._spin_force_dataset
    @classmethod
    def from_path(cls, path ,model_type=0, *, structures: list[Structure] | None = None, nep_txt_path: Path | str | None = None)->"NepTrainResultData":
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

        # Normalise optional nep_txt_path
        explicit_nep = Path(nep_txt_path) if nep_txt_path is not None else None

        # Try to find nep.txt first when no explicit NEP file is provided
        if explicit_nep is not None:
            nep_txt_path = explicit_nep
        else:
            nep_txt_path = dataset_path.with_name("nep.txt")

            # If nep.txt not found, search for any txt file containing "nep" in filename
            if not nep_txt_path.exists():
                dir_path = dataset_path.parent
                nep_files: list[Path] = []
                for txt_file in dir_path.glob("*.txt"):
                    if "nep" in txt_file.stem.lower():
                        nep_files.append(txt_file)

                # Sort: prefer files starting with "nep" and shorter names
                if nep_files:
                    nep_files.sort(key=lambda p: (not p.stem.lower().startswith("nep"), len(p.stem), p.stem))
                    nep_txt_path = nep_files[0]
                    logger.info(f"Using detected NEP file: {Path(nep_txt_path).name}")
                else:
                    # No NEP file found, use fallback
                    nep_txt_path = get_bundled_nep89_path()
                    MessageManager.send_warning_message("No NEP model file found; the program will use nep89 instead.")

        # Coerce to Path for downstream logic
        nep_txt_path = Path(nep_txt_path)

        # Determine output directory based on NEP model filename
        nep_stem = nep_txt_path.stem
        if nep_stem == "nep":
            # Standard nep.txt, output to current directory
            output_dir = dataset_path.parent
            output_suffix = file_name
        else:
            # Other NEP files (nep1.txt, nep2.txt, etc.), create file_name_XXX directory
            output_dir = dataset_path.parent / f"{file_name}_{nep_stem}"
            output_dir.mkdir(exist_ok=True)
            output_suffix = file_name
            logger.info(f"Output files will be saved to: {output_dir.name}/")
        
        # Build output paths in the appropriate directory
        energy_out_path = output_dir / f"energy_{output_suffix}.out"
        force_out_path = output_dir / f"force_{output_suffix}.out"
        stress_out_path = output_dir / f"stress_{output_suffix}.out"
        virial_out_path = output_dir / f"virial_{output_suffix}.out"
        
        # Optional spin-force output (magnetic).  Its presence also identifies
        # cached spin results without requiring the model parser.
        spin_force_out_path_candidate = output_dir / f"mforce_{output_suffix}.out"
        
        if file_name=="train":
            descriptor_path = output_dir / f"descriptor.out"
        else:
            descriptor_path = output_dir / f"descriptor_{output_suffix}.out"
        
        charge_out_path = output_dir / f"charge_{output_suffix}.out"
        bec_out_path = output_dir / f"bec_{output_suffix}.out"

        # Official training outputs are self-describing enough to display.  Do
        # not make an older/unsupported model format a prerequisite for opening
        # results that GPUMD has already written.
        base_outputs = (energy_out_path, force_out_path, virial_out_path)
        has_base_outputs = all(output.exists() for output in base_outputs)
        has_spin_output = has_base_outputs and spin_force_out_path_candidate.exists()
        has_charge_outputs = (
            has_base_outputs and charge_out_path.exists() and bec_out_path.exists()
        )
        has_standard_outputs = has_base_outputs and stress_out_path.exists()
        if has_spin_output or has_charge_outputs or has_standard_outputs:
            has_spin = has_spin_output
            has_charge = has_charge_outputs and not has_spin
        else:
            model_info = inspect_model(nep_txt_path)
            has_spin = model_info.model_type == "spin"
            has_charge = model_info.model_type == "charge"

        spin_force_out_path = spin_force_out_path_candidate if has_spin else None
        
        inst = cls(
            nep_txt_path,
            dataset_path,
            energy_out_path,
            force_out_path,
            stress_out_path,
            virial_out_path,
            descriptor_path,
            charge_out_path,
            bec_out_path,
            has_charge,
            spin_force_out_path,
            has_spin,
        )
        if structures is not None:
            try:
                inst.set_structures(structures)
            except Exception:
                pass
        return inst

    def _can_load_without_calculator(self) -> bool:
        """Return whether complete official outputs can be displayed directly."""
        required_paths = [
            self.energy_out_path,
            self.force_out_path,
            self.virial_out_path,
            self.spin_force_out_path if self.is_spin_model else self.stress_out_path,
        ]
        if self.is_charge_model:
            required_paths.extend((self.charge_out_path, self.bec_out_path))
        return all(path is not None and path.exists() for path in required_paths)

    def _load_dataset(self) -> None:
        """Populate plot datasets from cached outputs or by recalculating with NEP."""
        nep_in = read_nep_in(self.data_xyz_path.with_name("nep.in"))
        bec_array = np.array([])
        charge_array = np.array([])
        spin_force_array = np.array([])
        if self._should_recalculate(nep_in):
            results = self._recalculate_and_save()
            if getattr(self, "is_charge_model", False):
                energy_array, force_array, virial_array, stress_array, charge_array, bec_array = results
            elif getattr(self, "is_spin_model", False):
                energy_array, force_array, virial_array, stress_array, spin_force_array = results
            else:
                energy_array, force_array, virial_array, stress_array = results
        else:
            storage_dtype = get_storage_float_dtype()
            energy_array = read_nep_out_file(self.energy_out_path, dtype=storage_dtype, ndmin=2)
            force_array = read_nep_out_file(self.force_out_path, dtype=storage_dtype, ndmin=2)
            virial_array = read_nep_out_file(self.virial_out_path, dtype=storage_dtype, ndmin=2)
            stress_array = read_nep_out_file(self.stress_out_path, dtype=storage_dtype, ndmin=2)
            if self.spin_force_out_path:
                spin_force_array = read_nep_out_file(self.spin_force_out_path, dtype=storage_dtype, ndmin=2)
            if getattr(self, "is_charge_model", False):
                if self.charge_out_path:
                    charge_array = read_nep_out_file(self.charge_out_path, dtype=storage_dtype, ndmin=2)
                if self.bec_out_path:
                    bec_array = read_nep_out_file(self.bec_out_path, dtype=storage_dtype, ndmin=2)
            if energy_array.shape[0] != self.atoms_num_list.shape[0]:
                raise ValueError(
                    f"{self.energy_out_path.name} contains {energy_array.shape[0]} structures, "
                    f"but {self.data_xyz_path.name} contains {self.atoms_num_list.shape[0]}. "
                    "Move the existing official .out files aside before generating new predictions."
                )
        self._energy_dataset = NepPlotData(energy_array, title="energy")
        default_forces = parse_forces_mode(Config.get("widget", "forces_data", ForcesMode.Raw))
        self._force_vector_dataset = (
            NepPlotData(force_array, group_list=self.atoms_num_list, title="force")
            if force_array.size != 0
            else None
        )
        if force_array.size != 0 and default_forces == ForcesMode.Norm:
            force_array = aggregate_per_atom_to_structure(force_array, self.atoms_num_list, map_func=np.linalg.norm, axis=0)
            self._force_dataset = NepPlotData(force_array, title="force")
        else:
            self._force_dataset = NepPlotData(force_array, group_list=self.atoms_num_list, title="force")
        # Spin force (magnetic force) dataset, display only
        if spin_force_array.size != 0:
            self._spin_force_vector_dataset = NepPlotData(spin_force_array, group_list=self.atoms_num_list, title="mforce")
            self._spin_force_dataset = NepPlotData(spin_force_array, group_list=self.atoms_num_list, title="mforce")
        else:
            self._spin_force_vector_dataset = None
            self._spin_force_dataset = None
        if float(nep_in.get("lambda_v", 1)) != 0 or (getattr(self, "is_spin_model", False) and virial_array.size != 0):
            self._stress_dataset = NepPlotData(stress_array, title="stress")
            self._virial_dataset = NepPlotData(virial_array, title="virial")
        else:
            self._stress_dataset = NepPlotData([], title="stress")
            self._virial_dataset = NepPlotData([], title="virial")
        if getattr(self, "is_charge_model", False):
            # build bec dataset if present
            if bec_array.size != 0:
                self._bec_dataset = NepPlotData(bec_array, group_list=self.atoms_num_list, title="bec")
            else:
                self._bec_dataset = None
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
        if not self.cache_outputs_enabled():
            return True
        required_paths = [
            self.energy_out_path,
            self.force_out_path,
            self.virial_out_path,
        ]
        if not getattr(self, "is_spin_model", False):
            required_paths.append(self.stress_out_path)
        else:
            required_paths.append(self.spin_force_out_path)
        if getattr(self, "is_charge_model", False):
            if self.charge_out_path:
                required_paths.append(self.charge_out_path)
            if self.bec_out_path:
                required_paths.append(self.bec_out_path)
        required_paths = [path for path in required_paths if path is not None]
        existing = [path.exists() for path in required_paths]
        metadata = None
        record = None
        if self.prediction_meta_path.exists():
            try:
                metadata = json.loads(
                    self.prediction_meta_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Cannot validate {self.prediction_meta_path.name}: {error}. "
                    "Move the manifest aside to treat the .out files as external results."
                ) from error
            if metadata.get("schema_version") != 1:
                raise ValueError(
                    f"Unsupported {self.prediction_meta_path.name} schema. "
                    "Move the manifest aside to treat the .out files as external results."
                )
            record = metadata.get("predictions", {}).get(self.data_xyz_path.name)
        if all(existing):
            if record is None or getattr(self, "nep_calc", None) is None:
                return False
            expected_model = self.nep_calc.model_info.sha256
            expected_dataset = self._sha256(self.data_xyz_path)
            return not (
                record.get("model", {}).get("sha256") == expected_model
                and record.get("dataset", {}).get("sha256") == expected_dataset
                and record.get("dataset", {}).get("structures")
                == int(len(self.atoms_num_list))
            )
        if record is not None:
            return True
        if any(existing):
            existing_names = ", ".join(
                path.name for path, exists in zip(required_paths, existing) if exists
            )
            raise FileExistsError(
                "Found partial official NEP outputs without prediction.meta.json: "
                f"{existing_names}. NepTrainKit will not overwrite them automatically; "
                "move the existing .out files aside and retry."
            )
        return True
    def _save_energy_data(self, potentials:npt.NDArray[np.floating]) -> npt.NDArray[Any]:
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

        ref_energies = np.array([s.energy if s.has_energy else np.nan for s in self.structure.now_data], dtype=np.float64)
        energy_array = concat_nep_dft_array(np.asarray(potentials, dtype=np.float64), ref_energies, quantity="energies")

        energy_array=energy_array/ self.atoms_num_list.reshape(-1, 1)
        energy_array = np.asarray(energy_array, dtype=get_storage_float_dtype())
        if energy_array.size != 0 and self.cache_outputs_enabled():
            np.savetxt(self.energy_out_path, energy_array, fmt='%.17g')
        return energy_array
    def _save_force_data(self, forces: npt.NDArray[np.floating]) -> npt.NDArray[Any]:
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

        ref_forces = np.vstack([
            s.forces if s.has_forces else np.full((len(s), 3), np.nan, dtype=np.float64)
            for s in self.structure.now_data
        ], dtype=np.float64)
        forces_array = concat_nep_dft_array(np.asarray(forces, dtype=np.float64), ref_forces, quantity="forces")
        forces_array = np.asarray(forces_array, dtype=get_storage_float_dtype())

        if forces_array.size != 0 and self.cache_outputs_enabled():
            np.savetxt(self.force_out_path, forces_array, fmt='%.17g')
        return forces_array
    def _save_virial_and_stress_data(self, virials: npt.NDArray[np.floating]) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
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

        ref_virials = np.vstack([
            s.nep_virial if s.has_virial else np.full(6, np.nan, dtype=np.float64)
            for s in self.structure.now_data
        ], dtype=np.float64)
        virials_array = concat_nep_dft_array(np.asarray(virials, dtype=np.float64), ref_virials, quantity="virials")
        virials_array = np.asarray(virials_array, dtype=get_storage_float_dtype())

        stress_array = virials_array * coefficient * 160.21766208  # Unit conversion to MPa
        stress_array = np.asarray(stress_array, dtype=get_storage_float_dtype())
        if virials_array.size != 0 and self.cache_outputs_enabled():
            np.savetxt(self.virial_out_path, virials_array, fmt='%.17g')
        if (
            stress_array.size != 0
            and self.cache_outputs_enabled()
            and not getattr(self, "is_spin_model", False)
        ):
            np.savetxt(self.stress_out_path, stress_array, fmt='%.17g')
        return virials_array, stress_array
    def _save_charge_data(self, charges: npt.NDArray[np.floating]) -> npt.NDArray[Any]:
        """Persist per-atom charges (NEP prediction only)."""
        if charges.size == 0:
            return np.array([])
        charge_arr = np.asarray(charges.reshape(-1, 1), dtype=get_storage_float_dtype())
        if getattr(self, "charge_out_path", None) and charge_arr.size != 0 and self.cache_outputs_enabled():
            np.savetxt(self.charge_out_path, charge_arr, fmt='%.17g')
        return charge_arr
    def _save_bec_data(self, becs: npt.NDArray[np.floating]) -> npt.NDArray[Any]:
        """Persist per-atom BEC with optional reference pairing."""
        if becs.size == 0:
            return np.array([])
        nep_bec = np.asarray(becs, dtype=np.float64)
        ref_bec = np.vstack([
            np.asarray(s.bec, dtype=np.float64).reshape(-1, 9) if getattr(s, "has_bec", False)
            else np.full((len(s), 9), np.nan, dtype=np.float64)
            for s in self.structure.now_data
        ])

        bec_array = concat_nep_dft_array(nep_bec, ref_bec, quantity="BEC values")
        bec_array = np.asarray(bec_array, dtype=get_storage_float_dtype())
        if getattr(self, "bec_out_path", None) and bec_array.size != 0 and self.cache_outputs_enabled():
            np.savetxt(self.bec_out_path, bec_array, fmt='%.17g')
        return bec_array

    def _save_spin_force_data(self, mforces: npt.NDArray[np.floating]) -> npt.NDArray[Any]:
        """Persist magnetic force comparisons in the official ``mforce_*.out`` format."""
        reference = np.vstack(
            [
                np.asarray(structure.atomic_properties["force_mag"], dtype=np.float64)
                if "force_mag" in structure.atomic_properties
                else np.full((len(structure), 3), np.nan, dtype=np.float64)
                for structure in self.structure.now_data
            ]
        )
        values = concat_nep_dft_array(
            np.asarray(mforces, dtype=np.float64),
            reference,
            quantity="magnetic forces",
        )
        values = np.asarray(values, dtype=get_storage_float_dtype())
        if values.size and self.spin_force_out_path and self.cache_outputs_enabled():
            np.savetxt(self.spin_force_out_path, values, fmt="%.17g")
        return values

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _write_prediction_meta(self) -> None:
        if not self.cache_outputs_enabled():
            return
        output_paths = [
            self.energy_out_path,
            self.force_out_path,
            self.virial_out_path,
        ]
        if not self.is_spin_model:
            output_paths.append(self.stress_out_path)
        if self.spin_force_out_path:
            output_paths.append(self.spin_force_out_path)
        if self.is_charge_model:
            output_paths.extend(
                path for path in (self.charge_out_path, self.bec_out_path) if path
            )
        record = {
            "nep_adapters_version": nep_adapters_version,
            "model": {
                "path": str(self.nep_txt_path.resolve()),
                "sha256": self.nep_calc.model_info.sha256,
                "type": self.nep_calc.model_info.model_type,
            },
            "dataset": {
                "path": str(self.data_xyz_path.resolve()),
                "sha256": self._sha256(self.data_xyz_path),
                "structures": int(len(self.atoms_num_list)),
                "atoms": int(np.sum(self.atoms_num_list)),
            },
            "backend": {
                "requested": self.nep_calc.selection.requested.value,
                "resolved": self.nep_calc.selection.resolved.value,
                "reason": self.nep_calc.selection.reason,
            },
            "chunk_max_atoms": self.nep_calc.chunk_max_atoms,
            "outputs": [
                {"name": path.name, "size": path.stat().st_size}
                for path in output_paths
                if path.exists()
            ],
        }
        payload = {
            "schema_version": 1,
            "producer": "NepTrainKit",
            "predictions": {},
        }
        if self.prediction_meta_path.exists():
            try:
                existing = json.loads(
                    self.prediction_meta_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                existing = None
            if isinstance(existing, dict) and existing.get("schema_version") == 1:
                payload["predictions"].update(existing.get("predictions", {}))
        payload["predictions"][self.data_xyz_path.name] = record
        temporary = self.prediction_meta_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.prediction_meta_path)

    def _recalculate_and_save(self ):
        """Recompute NEP predictions and update on-disk comparison files."""
        prediction = self._pending_prediction
        self._pending_prediction = None
        if prediction is None:
            prediction = self.nep_calc.predict(
                self.structure.now_data.tolist(),
                progress=lambda done, total: self.predictionStatusSignal.emit(
                    self.tr("Running NEP prediction: {done}/{total} structures").format(
                        done=done, total=total
                    )
                ),
            )
        energy_array = self._save_energy_data(prediction.energy)
        force_array = self._save_force_data(prediction.forces)
        virial_array, stress_array = self._save_virial_and_stress_data(
            prediction.structure_virials[:, [0, 4, 8, 1, 5, 6]]
        )

        if isinstance(prediction, ChargePrediction):
            charge_array = self._save_charge_data(prediction.charges)
            bec_array = self._save_bec_data(prediction.becs)
            result = (
                energy_array,
                force_array,
                virial_array,
                stress_array,
                charge_array,
                bec_array,
            )
        elif isinstance(prediction, SpinPrediction):
            spin_force_array = self._save_spin_force_data(prediction.mforces)
            result = (
                energy_array,
                force_array,
                virial_array,
                stress_array,
                spin_force_array,
            )
        else:
            result = energy_array, force_array, virial_array, stress_array

        self.write_prediction()
        self._write_prediction_meta()
        return result

    def _generate_missing_descriptors(self) -> npt.NDArray[np.float64]:
        nep_in = read_nep_in(self.data_xyz_path.with_name("nep.in"))
        if not self._should_recalculate(nep_in):
            return super()._generate_missing_descriptors()

        self.predictionStatusSignal.emit(
            self.tr(
                "Generating NEP descriptors and predictions together to avoid duplicate work."
            )
        )
        prediction, descriptors = self.nep_calc.predict_with_descriptors(
            self.structure.now_data.tolist(),
            progress=lambda done, total: self.predictionStatusSignal.emit(
                self.tr(
                    "Running combined NEP calculation: {done}/{total} structures"
                ).format(done=done, total=total)
            ),
        )
        self._pending_prediction = prediction
        return descriptors
class NepPolarizabilityResultData(ResultData):
    """Result loader for NEP polarizability evaluations."""
    FORCE_CPU_BACKEND = True
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
            nep_polarizability_array = self.nep_calc.polarizabilities(self.structure.now_data.tolist())
            # nep_polarizability_array=self.nep_calc_thread.func_result
            if nep_polarizability_array.size == 0:
                MessageManager.send_warning_message("The nep calculator fails to calculate the polarizability, use the original polarizability instead.")
            nep_polarizability_array = self._save_polarizability_data(  nep_polarizability_array)
            self.write_prediction()
        except Exception as e:
            # logger.debug(traceback.format_exc())
            MessageManager.send_error_message(f"An error occurred while running NEP calculator: {e}")
            nep_polarizability_array = np.array([])
        return nep_polarizability_array
    def _save_polarizability_data(self, polarizability: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
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
        nep_polarizability_array = np.asarray(polarizability, dtype=np.float64) / np.asarray(self.atoms_num_list[:, np.newaxis], dtype=np.float64)
        try:
            ref_polarizability = np.vstack([s.nep_polarizability for s in self.structure.now_data], dtype=np.float64)
            if polarizability.size == 0:
                polarizability_array = np.column_stack([ref_polarizability, ref_polarizability])
            else:
                polarizability_array = np.column_stack([nep_polarizability_array,
                                                        ref_polarizability
                                                        ])
        except Exception:
            # logger.debug(traceback.format_exc())
            polarizability_array = np.column_stack([polarizability, polarizability])
        polarizability_array = np.asarray(polarizability_array, dtype=get_storage_float_dtype())
        if polarizability_array.size != 0 and self.cache_outputs_enabled():
            np.savetxt(self.polarizability_out_path, polarizability_array, fmt='%.17g')
        return polarizability_array
    def _load_dataset(self) -> None:
        """Populate polarizability datasets from cached outputs or by recalculating."""
        nep_in = read_nep_in(self.data_xyz_path.with_name("nep.in"))
        if self._should_recalculate(nep_in):
            polarizability_array = self._recalculate_and_save( )
        else:
            polarizability_array = read_nep_out_file(self.polarizability_out_path, dtype=get_storage_float_dtype(), ndmin=2)
            if polarizability_array.shape[0]!=self.atoms_num_list.shape[0]:
                if self.cache_outputs_enabled():
                    self.polarizability_out_path.unlink()
                    return self._load_dataset()
                polarizability_array = self._recalculate_and_save()
        self._polarizability_diagonal_dataset = NepPlotData(polarizability_array[:, [0,1,2,6,7,8]], title="Polar Diag")
        self._polarizability_no_diagonal_dataset = NepPlotData(polarizability_array[:, [3,4,5,9,10,11]], title="Polar NoDiag")
class NepDipoleResultData(ResultData):
    """Result loader for NEP dipole predictions."""
    FORCE_CPU_BACKEND = True
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
            nep_dipole_array = self.nep_calc.dipoles(self.structure.now_data.tolist())
            # nep_dipole_array=self.nep_calc_thread.func_result
            if nep_dipole_array.size == 0:
                MessageManager.send_warning_message("The nep calculator fails to calculate the dipole, use the original dipole instead.")
            nep_dipole_array = self._save_dipole_data(  nep_dipole_array)
            self.write_prediction()
        except Exception as e:
            # logger.debug(traceback.format_exc())
            MessageManager.send_error_message(f"An error occurred while running NEP calculator: {e}")
            nep_dipole_array = np.array([])
        return nep_dipole_array
    def _save_dipole_data(self, dipole: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
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
        nep_dipole_array = np.asarray(dipole, dtype=np.float64) / np.asarray(self.atoms_num_list[:, np.newaxis], dtype=np.float64)
        try:
            ref_dipole = np.vstack([s.nep_dipole for s in self.structure.now_data], dtype=np.float64)
            if dipole.size == 0:
                dipole_array = np.column_stack([ref_dipole, ref_dipole])
            else:
                dipole_array = np.column_stack([nep_dipole_array,
                                            ref_dipole
                                                    ])
        except Exception:
            # logger.debug(traceback.format_exc())
            dipole_array = np.column_stack([nep_dipole_array, nep_dipole_array])
        dipole_array = np.asarray(dipole_array, dtype=get_storage_float_dtype())
        if dipole_array.size != 0 and self.cache_outputs_enabled():
            np.savetxt(self.dipole_out_path, dipole_array, fmt='%.17g')
        return dipole_array
    def _load_dataset(self) -> None:
        """Populate dipole datasets from cached outputs or by recalculating."""
        nep_in = read_nep_in(self.data_xyz_path.with_name("nep.in"))
        if self._should_recalculate(nep_in):
            dipole_array = self._recalculate_and_save( )
        else:
            dipole_array = read_nep_out_file(self.dipole_out_path, dtype=get_storage_float_dtype(), ndmin=2)
            if dipole_array.shape[0]!=self.atoms_num_list.shape[0]:
                if self.cache_outputs_enabled():
                    self.dipole_out_path.unlink()
                    return self._load_dataset()
                dipole_array = self._recalculate_and_save()
        self._dipole_dataset = NepPlotData(dipole_array, title="dipole")
