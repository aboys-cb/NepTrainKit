#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""Core NEP result data loaders and helpers."""
import traceback
import json
from loguru import logger
from pathlib import Path
from typing import Any
import numpy.typing as npt
import numpy as np
from PySide6.QtCore import QCoreApplication
from NepTrainKit import module_path
from NepTrainKit.core.adapter_api import (
    ChargePrediction,
    Prediction,
    SpinPrediction,
    inspect_model,
    nep_adapters_version,
)
from NepTrainKit.core import MessageManager
from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.prediction_cache import prediction_record_error, sha256_file
from NepTrainKit.core.precision import get_storage_float_dtype
from NepTrainKit.paths import as_path
from NepTrainKit.paths import get_bundled_nep89_path
from NepTrainKit.config import Config
from .atomic import atomic_text_output
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
    concat_nep_dft_array,
    read_nep_in,
    read_nep_out_file,
)
from NepTrainKit.core.types import ForcesMode, parse_forces_mode
from NepTrainKit.core.ward import VIRIAL_ORDER_KEY, WARD_UNIT, ward_labels


def associated_nep_model(path: Path) -> Path | None:
    """Find the model path for a dataset with its own model or official outputs."""
    standard = path.with_name("nep.txt")
    if standard.is_file():
        return standard

    def has_output(name):
        return path.with_name(f"{name}_{path.stem}.out").is_file()

    if all(has_output(name) for name in ("energy", "force", "virial")) and (
        has_output("stress") or has_output("mforce")
    ):
        return standard
    candidates = [file for file in path.parent.glob("*.txt") if "nep" in file.stem.lower() and file.is_file()]
    return min(
        candidates,
        key=lambda file: (not file.stem.lower().startswith("nep"), len(file.stem), file.name),
        default=None,
    )


def compatible_default_nep_model(path: Path, structures) -> Path:
    """Use preferred weights only when they cover the elements of every frame."""
    bundled = get_bundled_nep89_path()
    if path == bundled:
        return path
    elements = set(inspect_model(path).elements)
    return path if all(elements.issuperset(frame.elements) for frame in structures) else bundled


class WardReferenceResultData(ResultData):
    """Show raw XYZ labels in Dataset Display without opening a potential."""

    def __init__(self, path: Path, virial_order: str):
        if virial_order not in ("", "row_major", "column_major"):
            raise ValueError(f"Unsupported XYZ virial order: {virial_order!r}")
        super().__init__(None, path, path.with_name("descriptor.out"))
        self.virial_order = virial_order

    def load_structures(self):
        super().load_structures()
        for frame in self.structure.all_data:
            fields = frame.additional_fields
            virial = fields.get("virial")
            if self.virial_order and virial is not None and virial.size == 9 and VIRIAL_ORDER_KEY not in fields:
                if self.virial_order == "column_major":
                    fields["virial"] = virial.reshape(3, 3).T.reshape(9)
                fields[VIRIAL_ORDER_KEY] = "row_major"

    def _load_descriptors(self):
        self._descriptor_dataset = NepPlotData([], title="descriptor", parity_mode=False)

    def _load_dataset(self):
        indices = []
        points = []
        for index, frame in enumerate(self.structure.all_data):
            ward = ward_labels(frame)
            norm = ward.norm if ward is not None else None
            if norm is not None:
                indices.append(index)
                points.append((norm, index))
        self._ward_plot = NepPlotData(
            np.asarray(points, dtype=float).reshape(-1, 2), index_list=indices,
            title=self.tr("Ward residual"), parity_mode=False,
            x_label=self.tr("Original index"), y_label=f"|r| ({WARD_UNIT})",
        )

    @property
    def datasets(self):
        return [self._ward_plot]


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
        'force_vector': StructureSyncRule('_force_vector_dataset', 'x_cols', collect_force_sync),
        'virial': StructureSyncRule('virial', 'x_cols', collect_virial_sync),
        'stress': StructureSyncRule('stress', 'x_cols', collect_stress_sync),
    }

    def __init__(self,
                 nep_txt_path: Path | str | None,
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
        self.outputs_only = False
        self._check_default_elements = False
        self._pending_prediction: Prediction | None = None
        self._force_recalculate_outputs = False
        self._validated_energy_array: npt.NDArray[Any] | None = None
        self._validated_force_array: npt.NDArray[Any] | None = None
    @property
    def datasets(self):
        """Return datasets exposed to the UI in display order."""
        items = [self.energy, self.force]
        if self.outputs_only or not self.is_spin_model:
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
    def from_path(
        cls,
        path,
        model_type=0,
        *,
        structures: list[Structure] | None = None,
        nep_txt_path: Path | str | None = None,
        cache_outputs: bool | None = None,
        default_nep_path: Path | None = None,
        outputs_only: bool = False,
    ) -> "NepTrainResultData":
        """Create an instance from a NEP result directory.
        
        Parameters
        ----------
        path : PathLike
            Directory containing NEP outputs and descriptors.
        model_type : int, optional
            NEP model type hint used to select descriptor fallbacks.
        structures : list[Structure], optional
            Pre-loaded structures to attach instead of reading from disk.
        cache_outputs : bool, optional
            Per-result cache policy. ``False`` keeps predictions in memory.
        default_nep_path : Path, optional
            Prefer these weights when no model accompanies the dataset and all elements match.
        outputs_only : bool, optional
            Read existing outputs beside the dataset without loading a calculator.
        
        Returns
        -------
        NepTrainResultData
            Configured loader bound to the resolved directory.
        """
        dataset_path = as_path(path)
        standard = dataset_path.with_name("nep.txt")
        selected = None
        if not outputs_only:
            if nep_txt_path is not None:
                selected = Path(nep_txt_path)
            elif structures is None or dataset_path.is_file():
                selected = associated_nep_model(dataset_path)
        check_default = selected is None and not outputs_only and default_nep_path is not None
        if check_default:
            selected = Path(default_nep_path)
            check_default = selected != get_bundled_nep89_path()
        if selected is None and not outputs_only:
            selected = get_bundled_nep89_path()
            MessageManager.send_warning_message("No NEP model file found; the program will use nep89 instead.")

        paths, has_charge, has_spin = cls._output_paths(dataset_path, selected or standard, outputs_only)
        if not check_default and not outputs_only and cache_outputs is not False:
            paths["energy_out_path"].parent.mkdir(exist_ok=True)
        inst = cls(selected, dataset_path, **paths, charge_model=has_charge, spin_model=has_spin)
        inst.outputs_only = outputs_only
        inst._check_default_elements = check_default
        inst.set_cache_outputs_override(cache_outputs)
        if structures is not None:
            inst.set_structures(structures)
        return inst

    @staticmethod
    def _output_paths(dataset_path: Path, nep_file: Path, outputs_only: bool = False):
        stem = dataset_path.stem
        directory = dataset_path.parent
        if nep_file.stem != "nep":
            directory = directory / f"{stem}_{nep_file.stem}"
        paths = {
            f"{kind}_out_path": directory / f"{kind}_{stem}.out"
            for kind in ("energy", "force", "stress", "virial", "charge", "bec")
        }
        paths["spin_force_out_path"] = directory / f"mforce_{stem}.out"
        paths["descriptor_path"] = directory / ("descriptor.out" if stem == "train" else f"descriptor_{stem}.out")
        base_outputs = all(paths[f"{kind}_out_path"].is_file() for kind in ("energy", "force", "virial"))
        spin = paths["spin_force_out_path"].is_file()
        charge = paths["charge_out_path"].is_file() and paths["bec_out_path"].is_file()
        if outputs_only:
            charge = paths["bec_out_path"].is_file()
        elif not (base_outputs and (spin or charge or paths["stress_out_path"].is_file())):
            model_type = inspect_model(nep_file).model_type
            spin, charge = model_type == "spin", model_type == "charge"
        if not spin:
            paths["spin_force_out_path"] = None
        return paths, charge and (outputs_only or not spin), spin

    def _prepare_load(self) -> bool:
        if not self._check_default_elements:
            return False
        self.load_structures()
        if self.cancel_event.is_set():
            return True
        selected = compatible_default_nep_model(self.nep_txt_path, self.structure.all_data)
        if selected != self.nep_txt_path:
            paths, self.is_charge_model, self.is_spin_model = self._output_paths(self.data_xyz_path, selected)
            for name, path in paths.items():
                setattr(self, name, path)
            self.nep_txt_path = selected
            self.prediction_meta_path = self.energy_out_path.parent / "prediction.meta.json"
            self._loaded_model_hash = self._sha256(selected)
        if self.cache_outputs_enabled():
            self.energy_out_path.parent.mkdir(exist_ok=True)
        return True

    def cache_outputs_enabled(self) -> bool:
        return not self.outputs_only and super().cache_outputs_enabled()

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

    def _prediction_manifest_alignment_error(self) -> str | None:
        """Validate available NepTrainKit provenance without requiring a calculator."""
        if not self.prediction_meta_path.exists():
            return None
        try:
            metadata = json.loads(
                self.prediction_meta_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return None
        if metadata.get("schema_version") != 1:
            return None
        record = metadata.get("predictions", {}).get(self.data_xyz_path.name)
        if not isinstance(record, dict):
            return None

        output_error = prediction_record_error(
            record, model_hash=None, dataset_hash=self._sha256(self.data_xyz_path),
            structures=len(self.atoms_num_list), atoms=int(np.sum(self.atoms_num_list)),
            directory=self.prediction_meta_path.parent,
        )
        if output_error:
            return output_error

        dataset_record = record.get("dataset", {})
        recorded_dataset_hash = dataset_record.get("sha256")
        if recorded_dataset_hash and recorded_dataset_hash != self._sha256(
            self.data_xyz_path
        ):
            return "prediction.meta.json records a different structure file"
        recorded_structures = dataset_record.get("structures")
        if recorded_structures is not None and int(recorded_structures) != int(
            len(self.atoms_num_list)
        ):
            return "prediction.meta.json records a different structure count"
        recorded_atoms = dataset_record.get("atoms")
        if recorded_atoms is not None and int(recorded_atoms) != int(
            np.sum(self.atoms_num_list)
        ):
            return "prediction.meta.json records a different atom count"

        model_record = record.get("model", {})
        recorded_model_hash = model_record.get("sha256")
        if (
            recorded_model_hash
            and self.nep_txt_path.exists()
            and recorded_model_hash != self._sha256(self.nep_txt_path)
        ):
            return "prediction.meta.json records a different NEP model"
        return None

    def _prepare_cached_output_alignment(self) -> None:
        """Validate complete cached outputs before descriptors are attached."""
        if getattr(self, "nep_calc", None) is not None:
            return
        if self._validated_energy_array is not None:
            return
        storage_dtype = get_storage_float_dtype()
        try:
            energy_array = read_nep_out_file(
                self.energy_out_path, dtype=storage_dtype, ndmin=2
            )
            force_array = read_nep_out_file(
                self.force_out_path, dtype=storage_dtype, ndmin=2
            )
            alignment_error = self._prediction_manifest_alignment_error()
            if alignment_error is None:
                alignment_error = self._cached_output_alignment_error(
                    energy_array, force_array
                )
        except Exception as error:
            alignment_error = f"cached outputs could not be validated: {error}"
            energy_array = np.array([])
            force_array = np.array([])

        if alignment_error is None:
            self._validated_energy_array = energy_array
            self._validated_force_array = force_array
            return

        self._force_recalculate_outputs = True
        MessageManager.send_warning_message(
            self.tr(
                "Existing NEP outputs do not match the loaded structure order "
                "({reason}). NepTrainKit will recalculate them."
            ).format(reason=alignment_error)
        )
        self.nep_calc = NepCalculator(
            model_file=self.nep_txt_path.as_posix(),
            backend=self._calculation_backend(),
            chunk_max_atoms=Config.getint("nep", "chunk_max_atoms", 100000),
        )

    def _cached_descriptors_are_usable(self) -> bool:
        """Reject descriptors whenever the paired official outputs are stale."""
        return not self._force_recalculate_outputs

    def _load_descriptors(self) -> None:
        """Validate cached output alignment before loading paired descriptors."""
        if not self.outputs_only:
            self._prepare_cached_output_alignment()
        super()._load_descriptors()

    def _read_existing_output(self, path: Path | None, shape: tuple[int, int]) -> np.ndarray:
        if path is None or not path.is_file():
            return np.array([])
        try:
            values = read_nep_out_file(path, dtype=get_storage_float_dtype(), ndmin=2)
        except (OSError, ValueError):
            values = np.array([])
        if values.shape == shape:
            return values
        MessageManager.send_warning_message(
            self.tr("Skipped invalid NEP output {file}.").format(file=path.name)
        )
        return np.array([])

    def _load_dataset(self) -> None:
        """Populate plot datasets from cached outputs or by recalculating with NEP."""
        nep_in = {} if self.outputs_only else read_nep_in(self.data_xyz_path.with_name("nep.in"))
        bec_array = np.array([])
        charge_array = np.array([])
        spin_force_array = np.array([])
        if self.outputs_only:
            frames = len(self.atoms_num_list)
            atoms = int(np.sum(self.atoms_num_list))
            energy_array = self._read_existing_output(self.energy_out_path, (frames, 2))
            force_array = self._read_existing_output(self.force_out_path, (atoms, 6))
            virial_array = self._read_existing_output(self.virial_out_path, (frames, 12))
            stress_array = self._read_existing_output(self.stress_out_path, (frames, 12))
            spin_force_array = self._read_existing_output(self.spin_force_out_path, (atoms, 6))
            bec_array = self._read_existing_output(self.bec_out_path, (atoms, 18))
        elif self._should_recalculate(nep_in):
            results = self._recalculate_and_save()
            if getattr(self, "is_charge_model", False):
                energy_array, force_array, virial_array, stress_array, charge_array, bec_array = results
            elif getattr(self, "is_spin_model", False):
                energy_array, force_array, virial_array, stress_array, spin_force_array = results
            else:
                energy_array, force_array, virial_array, stress_array = results
        else:
            storage_dtype = get_storage_float_dtype()
            energy_array = self._validated_energy_array
            if energy_array is None:
                energy_array = read_nep_out_file(self.energy_out_path, dtype=storage_dtype, ndmin=2)
            force_array = self._validated_force_array
            if force_array is None:
                force_array = read_nep_out_file(self.force_out_path, dtype=storage_dtype, ndmin=2)
            self._validated_energy_array = None
            self._validated_force_array = None
            virial_array = read_nep_out_file(self.virial_out_path, dtype=storage_dtype, ndmin=2)
            stress_array = read_nep_out_file(self.stress_out_path, dtype=storage_dtype, ndmin=2)
            if self.spin_force_out_path:
                spin_force_array = read_nep_out_file(self.spin_force_out_path, dtype=storage_dtype, ndmin=2)
                spin_force_array = self._restore_cached_spin_force_reference(spin_force_array)
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
            groups = self.atoms_num_list if force_array.size else 1
            self._force_dataset = NepPlotData(force_array, group_list=groups, title="force")
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
        if self.outputs_only:
            for dataset in self.datasets:
                if dataset.num == 0:
                    dataset.parity_mode = False
                    dataset.show_rmse = False

    def _should_recalculate(self, nep_in: dict | None = None) -> bool:
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
        if self._force_recalculate_outputs or not self.cache_outputs_enabled():
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
            if record is not None and self._prediction_manifest_alignment_error() is not None:
                return True
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

    def _reference_spin_forces(self) -> npt.NDArray[np.float64]:
        """Read DFT magnetic forces from either supported EXTXYZ label."""
        return np.vstack(
            [
                np.asarray(
                    structure.atomic_properties[
                        "force_mag" if "force_mag" in structure.atomic_properties else "mforce"
                    ],
                    dtype=np.float64,
                )
                if "force_mag" in structure.atomic_properties or "mforce" in structure.atomic_properties
                else np.full((len(structure), 3), np.nan, dtype=np.float64)
                for structure in self.structure.now_data
            ]
        )

    def _restore_cached_spin_force_reference(self, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Use EXTXYZ labels when cached mforce DFT columns contain fallback predictions."""
        reference = self._reference_spin_forces()
        if values.shape != (reference.shape[0], 6):
            raise ValueError(
                f"Cached magnetic-force output {self.spin_force_out_path.name} has shape {values.shape}, "
                f"expected ({reference.shape[0]}, 6)."
            )
        labelled = np.isfinite(reference)
        if labelled.any():
            values[:, 3:][labelled] = reference[labelled]
        return values

    def _save_spin_force_data(self, mforces: npt.NDArray[np.floating]) -> npt.NDArray[Any]:
        """Persist magnetic force comparisons in the official ``mforce_*.out`` format."""
        reference = self._reference_spin_forces()
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
        return sha256_file(path)

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
                {"name": path.name, "size": path.stat().st_size, "sha256": self._sha256(path)}
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
        self._force_recalculate_outputs = False
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
class _NepResponseResultData(ResultData):
    """Load paired per-atom predictions and reference labels for one NEP response."""

    FORCE_CPU_BACKEND = True

    @property
    def _response_path(self) -> Path:
        return getattr(self, f"{self._quantity}_out_path")

    @classmethod
    def from_path(cls, path, *, structures: list[Structure] | None = None):
        """Open a response dataset with the model and outputs next to its XYZ."""
        source = as_path(path)
        descriptor = "descriptor.out" if source.stem == "train" else f"descriptor_{source.stem}.out"
        result = cls(
            source.with_name("nep.txt"),
            source,
            source.with_name(f"{cls._quantity}_{source.stem}.out"),
            source.with_name(descriptor),
        )
        if structures is not None:
            result.set_structures(structures)
        return result

    def _response_references(self) -> np.ndarray:
        references = np.full((len(self.atoms_num_list), self._width), np.nan)
        for row, structure in zip(references, self.structure.now_data):
            try:
                values = getattr(structure, f"nep_{self._quantity}")
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
            if values.shape == row.shape and np.isfinite(values).all():
                row[:] = values
        return references

    def _should_recalculate(self, nep_in: dict | None = None) -> bool:
        return self._force_recalculate_outputs or not self._response_path.is_file()

    def _read_cached_response(self) -> np.ndarray | None:
        if self._should_recalculate():
            return None
        path = self._response_path
        expected = self._response_references()
        if np.isnan(expected).any():
            # Unlabeled rows cannot verify the order of a cached prediction.
            self._force_recalculate_outputs = True
            return None
        try:
            array = read_nep_out_file(path, dtype=get_storage_float_dtype(), ndmin=2)
            error = self._cached_single_output_alignment_error(array, expected, path)
        except (OSError, ValueError) as exc:
            error = str(exc)
        if error is None:
            return array
        self._force_recalculate_outputs = True
        MessageManager.send_warning_message(
            QCoreApplication.translate(
                "NepTrainResultData",
                "Existing NEP outputs do not match the loaded structure order "
                "({reason}). NepTrainKit will recalculate them.",
            ).format(reason=error)
        )
        return None

    def _load_descriptors(self) -> None:
        self._read_cached_response()
        super()._load_descriptors()

    def _cached_descriptors_are_usable(self) -> bool:
        return not self._force_recalculate_outputs

    def _load_response(self) -> np.ndarray:
        cached = self._read_cached_response()
        if cached is not None:
            return cached

        references = self._response_references()
        calculate = getattr(self.nep_calc, self._prediction_method)
        predictions = np.asarray(calculate(self.structure.now_data.tolist()), dtype=np.float64)
        if predictions.size == 0:
            MessageManager.send_warning_message(self._empty_prediction_warning)
            missing = np.full_like(references, np.nan)
            return np.column_stack([missing, references]).astype(get_storage_float_dtype())

        if predictions.shape != references.shape:
            raise ValueError(QCoreApplication.translate(
                "RuntimeMessage", "{filename} has shape {actual}; expected {expected}."
            ).format(filename=self._response_path.name, actual=predictions.shape, expected=references.shape))
        predictions = predictions / self.atoms_num_list[:, None]
        response = np.column_stack([predictions, references]).astype(get_storage_float_dtype())
        if self.cache_outputs_enabled():
            with atomic_text_output(self._response_path) as file:
                np.savetxt(file, response, fmt="%.17g")
        self.write_prediction()
        self._force_recalculate_outputs = False
        return response


class NepPolarizabilityResultData(_NepResponseResultData):
    """Result loader for NEP polarizability evaluations."""

    _quantity = "polarizability"
    _prediction_method = "polarizabilities"
    _width = 6
    _empty_prediction_warning = (
        "The nep calculator fails to calculate the polarizability, use the original polarizability instead."
    )

    def __init__(
        self, nep_txt_path: Path | str, data_xyz_path: Path | str,
        polarizability_out_path: Path | str, descriptor_path: Path | str,
    ):
        super().__init__(nep_txt_path, data_xyz_path, descriptor_path)
        self.polarizability_out_path = Path(polarizability_out_path)

    @property
    def datasets(self):
        return [self.polarizability_diagonal, self.polarizability_no_diagonal, self.descriptor]

    @property
    def polarizability_diagonal(self):
        return self._polarizability_diagonal_dataset

    @property
    def polarizability_no_diagonal(self):
        return self._polarizability_no_diagonal_dataset

    def _load_dataset(self) -> None:
        values = self._load_response()
        self._polarizability_diagonal_dataset = NepPlotData(values[:, [0, 1, 2, 6, 7, 8]], title="Polar Diag")
        self._polarizability_no_diagonal_dataset = NepPlotData(values[:, [3, 4, 5, 9, 10, 11]], title="Polar NoDiag")


class NepDipoleResultData(_NepResponseResultData):
    """Result loader for NEP dipole predictions."""

    _quantity = "dipole"
    _prediction_method = "dipoles"
    _width = 3
    _empty_prediction_warning = "The nep calculator fails to calculate the dipole, use the original dipole instead."

    def __init__(
        self, nep_txt_path: Path | str, data_xyz_path: Path | str,
        dipole_out_path: Path | str, descriptor_path: Path | str,
    ):
        super().__init__(nep_txt_path, data_xyz_path, descriptor_path)
        self.dipole_out_path = Path(dipole_out_path)

    @property
    def datasets(self):
        return [self.dipole, self.descriptor]

    @property
    def dipole(self):
        return self._dipole_dataset

    def _load_dataset(self) -> None:
        self._dipole_dataset = NepPlotData(self._load_response(), title="dipole")
