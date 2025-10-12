#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base datasets, PCA helper, and the abstract result container.

This module defines the common data abstractions used by result loaders. It
includes a PCA helper, dataset containers with masking/undo support, and an
abstract :class:`ResultData` class centralising structure IO, selection, and
dataset synchronisation.


"""
import os
import json
import threading
import re
import traceback
from functools import cached_property
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PySide6.QtCore import QObject, Signal
from loguru import logger
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
import numpy.typing as npt
from NepTrainKit.utils import timeit, parse_index_string
from NepTrainKit.config import Config
from NepTrainKit.core import   MessageManager
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.utils import read_nep_out_file, aggregate_per_atom_to_structure, get_rmse, split_by_natoms

from .sampler import SparseSampler,farthest_point_sampling,pca
from NepTrainKit.core.types import Brushes, SearchType, NepBackend
from NepTrainKit.core.energy_shift import shift_dataset_energy
from NepTrainKit.core.calculator import   NepCalculator

class DataBase:
    """Container that tracks active rows and supports undo operations.


    Parameters
    ----------
    data_list : Sequence[Any] or numpy.ndarray
        Initial payload that is coerced to ``numpy.ndarray`` so masking remains ``O(1)``.
    """
    def __init__(self, data_list: Sequence[Any] | npt.NDArray[Any]):
        """Initialise the container state for masking and undo.


        Parameters
        ----------
        data_list : Sequence[Any] or numpy.ndarray
            Source values that are converted to ``numpy.ndarray`` for vectorised masking.

        Notes
        -----
        A boolean mask is initialised with all entries marked as active.
        Each call to :meth:`remove` pushes the affected indices onto an undo stack
        that can later be restored with :meth:`revoke`.
        """
        self._data = np.asarray(data_list)
        self._active_mask = np.ones(len(self._data), dtype=bool)
        self._history: list[npt.NDArray[np.int_]] = []
    @property
    def mask_array(self) -> npt.NDArray[np.bool_]:
        """Boolean mask highlighting the active rows."""
        return self._active_mask
    @property
    def num(self) -> int:
        """Number of rows currently marked as active."""
        return int(np.sum(self._active_mask))
    @property
    def all_data(self) -> npt.NDArray[Any]:
        """Return the unmanaged backing array."""
        return self._data
    @property
    def now_data(self) -> npt.NDArray[Any]:
        """Return a view that only exposes active rows."""
        return self._data[self._active_mask]
    @property
    def remove_data(self) -> npt.NDArray[Any]:
        """Return rows that were deactivated via :meth:`remove`."""
        return self._data[~self._active_mask]
    @property
    def now_indices(self) -> npt.NDArray[np.int32]:
        """Indices of the rows that remain active."""
        return np.where(self._active_mask)[0]
    @property
    def remove_indices(self) -> npt.NDArray[np.int32]:
        """Indices of rows that were marked inactive."""
        return np.where(~self._active_mask)[0]
    def remove(self, indices: Sequence[int] | int) -> None:
        """Deactivate items denoted by ``indices``.

        Parameters
        ----------
        indices : int or Sequence[int]
            Positions in :attr:`all_data` that should be marked inactive.
            Invalid indices are ignored silently.
        """
        if isinstance(indices, Sequence) and not isinstance(indices, (str, bytes)):
            idx = np.asarray(indices, dtype=int).ravel()
        else:
            idx = np.asarray([indices], dtype=int)
        idx = np.unique(idx)
        idx = idx[(idx >= 0) & (idx < len(self._data))]
        if idx.size == 0:
            return
        self._history.append(idx)
        self._active_mask[idx] = False
    def revoke(self) -> None:
        """Undo the most recent :meth:`remove` call, if any."""
        if self._history:
            last_indices = self._history.pop()
            self._active_mask[last_indices] = True
    def __getitem__(self, item: Any) -> Any:
        """Return a slice or element from the active view."""
        return self.now_data[item]
class NepData:
    """Base accessor that pairs a data matrix with structure group metadata.

    Parameters
    ----------
    data_list : Sequence[Any] or numpy.ndarray
        Array-like object that stores the target/property values. The input is
        converted to ``numpy.ndarray``.
    group_list : int or Sequence[int], default=1
        Describes how property rows map onto structures. A scalar means
        one-to-one, while a sequence contains repetition counts for each
        structure.
    index_list : Sequence[int] or numpy.ndarray, optional
        Custom index map used when ``group_list`` is already expanded.
    **kwargs
        Arbitrary attributes that should be attached to the instance.
    """
    title: str
    def __init__(
        self,
        data_list: Sequence[Any] | npt.NDArray[Any],
        group_list: int | Sequence[int] = 1,
        index_list: Sequence[int] | npt.NDArray[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise dataset values and grouping arrays.

        Parameters
        ----------
        data_list : Sequence[Any] or numpy.ndarray
            Property values to manage; they are converted to ``numpy.ndarray``.
        group_list : int or Sequence[int], optional
            Controls how rows map onto structures. A scalar means one-to-one, while
            a sequence supplies repetition counts per structure.
        index_list : Sequence[int] or numpy.ndarray, optional
            Custom index map when ``group_list`` is already expanded.
        **kwargs
            Additional attributes to attach to the instance.
        """
        data = np.asarray(data_list)
        self.data = DataBase(data)
        if index_list is None:
            if isinstance(group_list, int):
                group = np.arange(data.shape[0], dtype=np.uint32)
            else:
                counts = np.asarray(group_list, dtype=np.int64)
                if counts.ndim != 1:
                    raise ValueError("group_list must be one dimensional")
                group = np.arange(len(counts), dtype=np.uint32).repeat(counts)
        else:
            group = np.asarray(index_list, dtype=np.uint32)
            if not isinstance(group_list, int):
                group = group.repeat(group_list)
        self.group_array = DataBase(group)
        for key, value in kwargs.items():
            setattr(self, key, value)
    @property
    def num(self) -> int:
        """Return the number of active rows in :attr:`data`."""
        return self.data.num
    @cached_property
    def cols(self) -> int:
        """Half the number of columns, assuming NEP/DFT pairs."""
        if self.now_data.size == 0:
            return 0
        return self.now_data.shape[1] // 2
    @property
    def now_data(self) -> npt.NDArray[Any]:
        """Active slices of the underlying data matrix."""
        return self.data.now_data
    @property
    def now_indices(self) -> npt.NDArray[np.int32]:
        """Indices of active items relative to :attr:`all_data`."""
        return self.data.now_indices
    @property
    def all_data(self) -> npt.NDArray[Any]:
        """Return the full (unmasked) data matrix."""
        return self.data.all_data
    def is_visible(self, index: int) -> bool:
        """Return ``True`` if the row referenced by ``index`` is active."""
        if self.data.all_data.size == 0:
            return False
        return bool(self.data.mask_array[index].all())
    @property
    def remove_data(self) -> npt.NDArray[Any]:
        """Return rows that were removed from the active view."""
        return self.data.remove_data
    def convert_index(self, index_list: Sequence[int] | npt.NDArray[np.number] | int) -> npt.NDArray[np.int32]:
        """Translate original structure indices to positions in the dataset.

        Parameters
        ----------
        index_list : int or Sequence[int]
            Original structure indices.
        Returns
        -------
        numpy.ndarray
            Positions in :attr:`group_array` that match the supplied indices.
        """
        if isinstance(index_list, (int, np.number)):
            index_array = np.array([int(index_list)], dtype=np.int64)
        else:
            index_array = np.asarray(index_list, dtype=np.int64)
        mask = np.isin(self.group_array.all_data, index_array)
        return np.nonzero(mask)[0].astype(np.int32)
    def remove(self, remove_index: Sequence[int] | int) -> None:
        """Remove rows associated with the provided structure indices."""
        remove_indices = self.convert_index(remove_index)
        self.data.remove(remove_indices)
        self.group_array.remove(remove_indices)
    def revoke(self) -> None:
        """Restore the last removal across data and grouping arrays."""
        self.data.revoke()
        self.group_array.revoke()
    def get_rmse(self) -> float:
        """Return the RMSE between NEP and reference columns."""
        if not self.cols:
            return 0.0
        return float(get_rmse(self.now_data[:, : self.cols], self.now_data[:, self.cols :]))
    def get_formart_rmse(self) -> str:  # noqa: D401 - keep legacy name
        """Return the formatted RMSE string with units inferred from ``title``."""
        rmse = self.get_rmse()
        unit = ""
        scale = 1.0
        if self.title == "energy":
            unit, scale = "meV/atom", 1000
        elif self.title == "force":
            unit, scale = "meV/Å", 1000
        elif self.title == "virial":
            unit, scale = "meV/atom", 1000
        elif self.title == "stress":
            unit, scale = "MPa", 1000
        elif "Polar" in self.title:
            unit, scale = "(m.a.u./atom)", 1000
        elif self.title == "dipole":
            unit, scale = "(m.a.u./atom)", 1000
        elif self.title == "spin":
            unit, scale = "meV/μB", 1000
        else:
            return ""
        return f"{rmse * scale:.2f} {unit}"
    def get_max_error_index(self, nmax: int) -> list[int]:
        """Return the ``nmax`` structure indices with the largest absolute error."""
        if not self.cols:
            return []
        error = np.sum(np.abs(self.now_data[:, : self.cols] - self.now_data[:, self.cols :]), axis=1)
        sorted_idx = np.argsort(-error)
        structure_index = self.group_array.now_data[sorted_idx]
        _, unique_indices = np.unique(structure_index, return_index=True)
        return structure_index[np.sort(unique_indices)][:nmax].tolist()
class NepPlotData(NepData):
    """Two-column plot helper that separates NEP predictions from references."""
    def __init__(self, data_list: Sequence[Any] | npt.NDArray[Any], **kwargs: Any) -> None:
        """Initialise the plot dataset and cache column slices."""
        super().__init__(data_list, **kwargs)
        self.x_cols = slice(self.cols, None)
        self.y_cols = slice(None, self.cols)
    @property
    def x(self) -> npt.NDArray[Any]:
        """Flattened NEP predictions suitable for scatter plots."""
        if self.cols == 0:
            return self.now_data
        return self.now_data[:, self.x_cols].ravel()
    @property
    def y(self) -> npt.NDArray[Any]:
        """Flattened reference values."""
        if self.cols == 0:
            return self.now_data
        return self.now_data[:, self.y_cols].ravel()
    @property
    def structure_index(self) -> npt.NDArray[np.int32]:
        """Map each flattened point back to its parent structure index."""
        if self.cols == 0:
            return self.group_array.now_data.astype(np.int32)
        return self.group_array[:].repeat(self.cols).astype(np.int32)
class DPPlotData(NepData):
    """Plot helper for DP datasets where columns are ordered differently."""
    def __init__(self, data_list: Sequence[Any] | npt.NDArray[Any], **kwargs: Any) -> None:
        """Initialise slices for DP-format data (reference first)."""
        super().__init__(data_list, **kwargs)
        self.x_cols = slice(None, self.cols)
        self.y_cols = slice(self.cols, None)
    @property
    def x(self) -> npt.NDArray[Any]:
        """Flattened reference values."""
        if self.cols == 0:
            return self.now_data
        return self.now_data[:, self.x_cols].ravel()
    @property
    def y(self) -> npt.NDArray[Any]:
        """Flattened DP predictions."""
        if self.cols == 0:
            return self.now_data
        return self.now_data[:, self.y_cols].ravel()
    @property
    def structure_index(self) -> npt.NDArray[np.int32]:
        """Return structure indices replicated per column pair."""
        if self.cols == 0:
            return self.group_array.now_data.astype(np.int32)
        return self.group_array[:].repeat(self.cols).astype(np.int32)
class StructureData(NepData):
    """Utility mixin for structure-level queries."""
    @timeit
    def get_all_config(self, search_type: SearchType | None = None) -> list[str]:
        """Return structure metadata used for filtering.

        Parameters
        ----------
        search_type : SearchType, optional
            Metadata selector. Defaults to :data:`SearchType.TAG`.
        Returns
        -------
        list[str]
            Value per active structure matching ``search_type``.
        """
        if search_type is None:
            search_type = SearchType.TAG
        if search_type == SearchType.TAG:
            return [structure.tag for structure in self.now_data]
        if search_type == SearchType.FORMULA:
            return [structure.formula for structure in self.now_data]
        MessageManager.send_warning_message(f"Unsupported search type: {search_type}")
        return []
    def search_config(self, config: str, search_type: SearchType) -> list[int]:
        """Return structure indices whose metadata match ``config``.

        Parameters
        ----------
        config : str
            Regular expression used for matching.
        search_type : SearchType
            Attribute family to inspect.
        Returns
        -------
        list[int]
            Structure indices satisfying the pattern; empty on failure.
        """
        if search_type == SearchType.TAG:
            result_index = [i for i, structure in enumerate(self.now_data) if re.search(config, structure.tag)]
        elif search_type == SearchType.FORMULA:
            result_index = [i for i, structure in enumerate(self.now_data) if re.search(config, structure.formula)]
        else:
            MessageManager.send_warning_message(f"Unsupported search type: {search_type}")
            return []
        return self.group_array[result_index].tolist()
@dataclass(frozen=True)
class StructureSyncRule:
    """Declarative instruction that synchronises structure attributes into datasets."""
    dataset_attr: str
    target: str | slice | Callable[[Any], Any]
    collector: Callable[["ResultData", Any, Optional[np.ndarray]], tuple[np.ndarray, npt.NDArray[np.float32]]]
    precondition: Callable[["ResultData"], bool] = lambda _: True
    dtype: Any = np.float32
    def _resolve_target(self, dataset: Any) -> Any:
        """Return the concrete column selector for ``dataset``."""
        if callable(self.target):
            return self.target(dataset)
        if isinstance(self.target, str):
            return getattr(dataset, self.target)
        return self.target
    def apply(self, result_data: "ResultData", structure_indices: Optional[np.ndarray] = None) -> None:
        """Execute the rule on ``result_data`` if the precondition passes."""
        dataset = getattr(result_data, self.dataset_attr, None)
        if dataset is None or getattr(dataset, "num", 0) == 0:
            return
        if not self.precondition(result_data):
            return
        row_idx, values = self.collector(result_data, dataset, structure_indices)
        if row_idx is None or values is None:
            return
        row_idx = np.asarray(row_idx, dtype=np.int64)
        if row_idx.size == 0:
            return
        values = np.asarray(values, dtype=self.dtype)
        if values.size == 0:
            return
        target = self._resolve_target(dataset)
        dataset.all_data[row_idx, target] = values
class ResultData(QObject):
    """Manage structures, descriptors, and plots for NEP result files.
    Subclasses implement :meth:`_load_dataset` and expose their plot datasets
    through :py:attr:`datasets`. The class also centralises selection and
    synchronisation utilities shared by the GUI.
    """
    STRUCTURE_SYNC_RULES: dict[str, StructureSyncRule] = {}
    updateInfoSignal = Signal( )
    loadFinishedSignal = Signal()
    atoms_num_list: npt.NDArray
    _atoms_dataset: StructureData
    def __init__(self,
                 nep_txt_path: Path,
                 data_xyz_path: Path,
                 descriptor_path: Path,
                 calculator_factory: Optional[Callable[[str], Any]] = None,
                 import_options: Optional[dict] = None):
        """Initialise the result container with file locations and factories.

        Parameters
        ----------
        nep_txt_path : str or pathlib.Path
            Path to the NEP model file.
        data_xyz_path : str or pathlib.Path
            Path to the trajectory/structure file.
        descriptor_path : str or pathlib.Path
            Destination of cached descriptor values.
        calculator_factory : Callable[[str], Any], optional
            Factory returning a calculator compatible with the subclass.
            Defaults to :class:`NepCalculator`.
        import_options : dict, optional
            Extra keyword arguments forwarded to :func:`import_structures`.
        """
        super().__init__()
        self.load_flag=False
        # cooperative cancel for long-running loads
        self.cancel_event = threading.Event()
        self.descriptor_path=Path(descriptor_path)
        self.data_xyz_path=Path(data_xyz_path)
        self.nep_txt_path=Path(nep_txt_path)
        self.select_index=set()
        # Optional pre-fetched structures to skip IO in load_structures
        self._prefetched_structures: Optional[list[Structure]] = None
        # Optional importer options forwarded to importers.import_structures
        self._import_options: dict = dict(import_options or {})
        self.calculator_factory=calculator_factory
        self._structure_sync_rules = dict(getattr(self, "STRUCTURE_SYNC_RULES", {}))
        self._pending_non_physical_indices: list[int] = []
        self._sampler = SparseSampler(self)
    def request_cancel(self):
        """Request cooperative cancel during load. Also forward to calculator."""
        self.cancel_event.set()
        try:
            if hasattr(self, "nep_calc") and self.nep_calc is not None:
                self.nep_calc.cancel()
        except Exception:
            pass
    def reset_cancel(self):
        """Clear the cancellation flag so future operations proceed."""
        self.cancel_event.clear()
    @timeit
    def load_structures(self):
        """Populate :attr:`structure` from disk or a prefetched cache.
        The method honours :attr:`_prefetched_structures` first; otherwise it
        delegates to the importer registry and honours ``import_options``.
        """

        # If structures were provided upfront, use them; otherwise parse from file
        if self._prefetched_structures is not None:
            structures = self._prefetched_structures
        else:
            # Unified path: delegate to importers for all formats, including EXTXYZ.
            # ExtxyzImporter internally uses Structure.iter_read_multiple with cancel support.
            from NepTrainKit.core.io import importers as _imps
            opts = dict(self._import_options)
            opts.setdefault("cancel_event", self.cancel_event)
            structures = _imps.import_structures(self.data_xyz_path.as_posix(), **opts)
        self._atoms_dataset = StructureData(structures)
        self.atoms_num_list = np.array([len(struct) for struct in self.structure.now_data])
    def set_structures(self, structures: list[Structure]):
        """
        Provide pre-parsed structures so load_structures can skip file IO.
        """
        self._prefetched_structures = list(structures)
    def _normalize_structure_indices(self, structure_indices: Sequence[int] | npt.NDArray[Any] | None) -> npt.NDArray[np.int64]:
        """Return indices intersected with the currently active structure set.

        Parameters
        ----------
        structure_indices : Sequence[int] or numpy.ndarray, optional
            Candidate indices referring to :attr:`structure`. ``None`` means
            all active indices.
        Returns
        -------
        numpy.ndarray
            Sorted indices that are active within :attr:`structure`.
        """
        dataset = getattr(self, '_atoms_dataset', None)
        if dataset is None or dataset.num == 0:
            return np.array([], dtype=np.int64)
        active_indices = dataset.now_indices
        if structure_indices is None:
            return active_indices.copy()
        idx = np.asarray(structure_indices, dtype=np.int64).ravel()
        if idx.size == 0:
            return np.array([], dtype=np.int64)
        return np.intersect1d(active_indices, idx, assume_unique=False)
    def sync_structures(self, fields: Iterable[str] | None = None, structure_indices: Sequence[int] | None = None) -> None:
        """Apply registered :class:`StructureSyncRule` objects to datasets.

        Parameters
        ----------
        fields : Iterable[str] or str, optional
            Subset of rule names to apply. ``None`` means all registered rules.
        structure_indices : Sequence[int], optional
            Visible structure indices affected by the update. ``None`` uses all
            active structures.
        """

        if not getattr(self, '_structure_sync_rules', None):
            return
        dataset = getattr(self, '_atoms_dataset', None)
        if dataset is None or dataset.num == 0:
            return
        indices = self._normalize_structure_indices(structure_indices)
        if isinstance(fields, str):
            field_iter = [fields]
        elif fields is None:
            field_iter = list(self._structure_sync_rules.keys())
        else:
            field_iter = list(fields)
        for name in field_iter:
            rule = self._structure_sync_rules.get(name)
            if rule is None:
                continue
            rule.apply(self, indices)
    def write_prediction(self):
        """Create a ``nep.in`` stub when large datasets require prediction mode.
        The GUI expects a ``nep.in`` file to mark prediction runs for large
        (>1000) structure collections.
        """
        if self.atoms_num_list.shape[0] > 1000:
            #
            if not self.data_xyz_path.with_name("nep.in").exists():
                with open(self.data_xyz_path.with_name("nep.in"),
                          "w", encoding="utf8") as f:
                    f.write("prediction 1 ")
    def load(self):
        """Load structures, descriptors, and dataset arrays in sequence.
        The routine instantiates a calculator (optionally via ``calculator_factory``),
        parses structures, and then delegates to subclass hooks for descriptors and
        dataset-specific properties.
        """
        try:
            # Calculator injection (default to NEP). Subclasses can pass in a factory for other ML potentials.
            if self.calculator_factory is None:
                self.nep_calc = NepCalculator(
                    model_file=self.nep_txt_path.as_posix(),
                    backend=NepBackend(Config.get("nep", "backend", "auto")),
                    batch_size=Config.getint("nep", "gpu_batch_size", 1000)
                )
            else:
                # Factory is responsible for creating a calculator compatible with this ResultData subclass
                try:
                    self.nep_calc = self.calculator_factory(self.nep_txt_path.as_posix())
                except Exception:
                    logger.debug(traceback.format_exc())
                    MessageManager.send_warning_message("Failed to create custom calculator; falling back to NEP.")
                    self.nep_calc = NepCalculator(
                        model_file=self.nep_txt_path.as_posix(),
                        backend=NepBackend(Config.get("nep", "backend", "auto")),
                        batch_size=Config.getint("nep", "gpu_batch_size", 1000)
                    )
            # If subclass overrides load_structures, defer to it; otherwise do cancel-aware read
            self.load_structures()
            if self._atoms_dataset.num!=0:
                if not self.cancel_event.is_set():
                    self._load_descriptors()
                if not self.cancel_event.is_set():
                    self._load_dataset()
                if not self.cancel_event.is_set():
                    self.load_flag=True
            else:
                MessageManager.send_warning_message("No structures were loaded.")
        except:
            logger.error(traceback.format_exc())
            MessageManager.send_error_message("load dataset error!")
        self.loadFinishedSignal.emit()
    def _load_dataset(self):
        """Populate subclass-specific datasets (must be implemented by subclasses)."""
        raise NotImplementedError()
    @property
    def datasets(self) -> list["NepPlotData"]:
        """Return the plot datasets exposed by the subclass."""
        raise NotImplementedError()
    @property
    def descriptor(self):
        """Return the descriptor dataset prepared in :meth:`_load_descriptors`."""
        return self._descriptor_dataset
    @property
    def num(self):
        """Return the number of active structures in the dataset."""
        return self._atoms_dataset.num
    @property
    def structure(self):
        """Return the :class:`StructureData` wrapper for the active structures."""
        return self._atoms_dataset
    def is_select(self, i: int) -> bool:
        """Return ``True`` if the structure index is marked as selected."""
        return i in self.select_index
    def select(self, indices: Sequence[int] | int) -> None:
        """Mark structures denoted by ``indices`` as selected."""
        if isinstance(indices, (int, np.integer)):
            idx = np.array([int(indices)], dtype=int)
        else:
            idx = np.asarray(indices, dtype=int).ravel()
        idx = np.unique(idx)
        valid = (idx >= 0) & (idx < len(self.structure.all_data))
        valid &= self.structure.data.mask_array[idx]
        idx = idx[valid]
        self.select_index.update(idx.tolist())
        self.updateInfoSignal.emit()
    def uncheck(self, indices: Sequence[int] | int) -> None:
        """Remove structures denoted by ``indices`` from the selection set."""
        if isinstance(indices, (int, np.integer)):
            iter_indices = [int(indices)]
        else:
            iter_indices = (int(i) for i in np.asarray(indices).ravel())
        for idx in iter_indices:
            self.select_index.discard(idx)
        self.updateInfoSignal.emit()
    def inverse_select(self) -> None:
        """Invert the current selection over the active structure set."""
        active_indices = set(self.structure.data.now_indices.tolist())
        selected_indices = set(self.select_index)
        to_unselect = list(selected_indices)
        to_select = list(active_indices - selected_indices)
        if to_unselect:
            self.uncheck(to_unselect)
        if to_select:
            self.select(to_select)
    def select_structures_by_index(self, index_expression: str, use_origin: bool = True) -> list[int]:
        """Resolve an index expression into raw structure indices."""
        if not index_expression:
            return []
        text = index_expression.strip()
        if not text:
            return []
        structure = getattr(self, "structure", None)
        if structure is None:
            return []
        total = structure.all_data.shape[0] if use_origin else structure.now_data.shape[0]
        indices = parse_index_string(text, total)
        if not indices:
            return []
        idx_array = np.asarray(indices, dtype=np.int64)
        if use_origin:
            return idx_array.tolist()
        mapped = structure.group_array.now_data[idx_array]
        return np.asarray(mapped, dtype=np.int64).tolist()

    def select_structures_by_range(
        self,
        dataset: "NepPlotData",
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        use_and: bool = True,
    ) -> list[int]:
        """Return structure indices whose scatter positions fall in the given bounds."""
        if dataset is None or dataset.now_data.size == 0:
            return []
        x_low, x_high = sorted((float(x_min), float(x_max)))
        y_low, y_high = sorted((float(y_min), float(y_max)))
        mask_x = (dataset.x >= x_low) & (dataset.x <= x_high)
        mask_y = (dataset.y >= y_low) & (dataset.y <= y_high)
        mask = mask_x & mask_y if use_and else mask_x | mask_y
        if not np.any(mask):
            return []
        return np.unique(dataset.structure_index[mask]).astype(int).tolist()

    def get_selected_structures(self) -> list[Structure]:
        """Return the selected structures in the order of their raw index."""
        indices = list(self.select_index)
        mapped = self.structure.convert_index(indices)
        return self.structure.all_data[mapped].tolist()
    def export_selected_xyz(self, save_file_path: str | Path) -> None:
        """Write the currently selected structures to ``save_file_path``."""
        indices = list(self.select_index)
        try:
            with open(save_file_path, "w", encoding="utf8") as handle:
                mapped = self.structure.convert_index(indices)
                for structure in self.structure.all_data[mapped]:
                    structure.write(handle)
            MessageManager.send_info_message(f"File exported to: {save_file_path}")
        except Exception:
            MessageManager.send_info_message("An unknown error occurred while saving. The error message has been output to the log!")
            logger.error(traceback.format_exc())
    def export_model_xyz(self, save_path: str | Path) -> None:
        """Export active and removed structures into ``save_path`` folder."""
        try:
            good_path = Path(save_path).joinpath("export_good_model.xyz")
            with open(good_path, "w", encoding="utf8") as handle:
                for structure in self.structure.now_data:
                    structure.write(handle)
            removed_path = Path(save_path).joinpath("export_remove_model.xyz")
            with open(removed_path, "w", encoding="utf8") as handle:
                for structure in self.structure.remove_data:
                    structure.write(handle)
            MessageManager.send_info_message(f"File exported to: {save_path}")
        except Exception:
            MessageManager.send_info_message("An unknown error occurred while saving. The error message has been output to the log!")
            logger.error(traceback.format_exc())
    def get_atoms(self, index: int):
        """Return the ASE atoms object for the original ``index``."""
        mapped = self.structure.convert_index(index)
        return self.structure.all_data[mapped][0]
    def remove(self, i: int) -> None:
        """Remove the structure ``i`` across all datasets."""
        self.structure.remove(i)
        for dataset in self.datasets:
            dataset.remove(i)
        self.updateInfoSignal.emit()
    @property
    def is_revoke(self) -> bool:
        """Return ``True`` if any structures have been removed."""
        return self.structure.remove_data.size != 0
    def revoke(self) -> None:
        """Undo the most recent removal across structures and datasets."""
        self.structure.revoke()
        for dataset in self.datasets:
            dataset.revoke()
        self.updateInfoSignal.emit()
    @timeit
    def delete_selected(self):
        """Remove and clear all currently selected structures."""
        self.remove(list(self.select_index))
        self.select_index.clear()
        self.updateInfoSignal.emit()
    def iter_non_physical_structure_indices(self, radius_coefficient: float):
        """Yield progress increments while collecting non-physical structures."""
        structures = self.structure.now_data
        group_array = self.structure.group_array.now_data
        pending: list[int] = []
        for structure, index in zip(structures, group_array):
            if not structure.adjust_reasonable(radius_coefficient):
                pending.append(int(index))
            yield 1
        self._pending_non_physical_indices = pending

    def consume_non_physical_structure_indices(self) -> list[int]:
        """Return and clear indices collected by the non-physical scan."""
        indices = getattr(self, "_pending_non_physical_indices", [])
        self._pending_non_physical_indices = []
        return list(indices)

    def sparse_descriptor_selection(
        self,
        n_samples: int,
        distance: float,
        restrict_to_selection: bool=False,
    ) -> tuple[list[int], bool]:
        """Return FPS-selected structure indices and whether they should be deselected."""
        dataset = getattr(self, "descriptor", None)
        if dataset is None or dataset.now_data.size == 0:
            MessageManager.send_message_box("No descriptor data available", "Error")
            return [], False

        reverse = False
        points = dataset.now_data
        mask = np.ones(points.shape[0], dtype=bool)

        if restrict_to_selection:
            sel = np.asarray(list(self.select_index), dtype=np.int64)
            if sel.size == 0:
                MessageManager.send_info_message("No selection found; FPS will run on full data.")
            else:
                struct_ids = dataset.group_array.now_data
                mask = np.isin(struct_ids, sel)
                if not np.any(mask):
                    MessageManager.send_info_message(
                        "Current selection has no points on this plot; FPS will run on full data."
                    )
                    mask = np.ones(points.shape[0], dtype=bool)
                else:
                    reverse = True
                    MessageManager.send_info_message(
                        "When FPS sampling is performed in the designated area, the program will automatically deselect it, just click to delete!"
                    )

        if np.any(mask):
            subset = points[mask]
            idx_local = farthest_point_sampling(subset, n_samples=n_samples, min_dist=distance)
            if len(idx_local) == 0:
                global_rows = np.array([], dtype=np.int64)
            else:
                global_rows = np.where(mask)[0][np.asarray(idx_local, dtype=np.int64)]
        else:
            global_rows = np.array([], dtype=np.int64)

        structures = dataset.group_array[global_rows]
        return structures.tolist(), reverse




    def sparse_point_selection(
        self,
        n_samples: int,
        distance: float,
        descriptor_source: str = "reduced",
        restrict_to_selection: bool = False,
        training_path: str | None = None,
    ) -> tuple[list[int], bool]:
        """Delegate sparse sampling to the sampler helper."""
        return self._sampler.sparse_point_selection(
            n_samples=n_samples,
            distance=distance,
            descriptor_source=descriptor_source,
            restrict_to_selection=restrict_to_selection,
            training_path=training_path,
        )

    def export_descriptor_data(self, path: str | Path) -> None:
        """Write descriptor values for the current selection to ``path``."""
        if len(self.select_index) == 0:
            MessageManager.send_info_message("No data selected!")
            return

        descriptor = getattr(self, "descriptor", None)
        if descriptor is None:
            MessageManager.send_warning_message("Descriptor dataset is unavailable.")
            return

        select_index = descriptor.convert_index(list(self.select_index))
        descriptor_data = descriptor.all_data[select_index, :]

        if hasattr(self, "energy") and getattr(self.energy, "num", 0) != 0:
            energy_index = self.energy.convert_index(list(self.select_index))
            energy_data = self.energy.all_data[energy_index, 1]
            descriptor_data = np.column_stack((descriptor_data, energy_data))

        with open(path, "w", encoding="utf8") as handle:
            np.savetxt(handle, descriptor_data, fmt="%.6g", delimiter="\t")

    def get_editable_structure_tags(self) -> set[str]:
        """Return the editable tags for currently selected structures."""
        selected = self.get_selected_structures()
        tags = {item for structure in selected for item in structure.get_prop_key(True, True)}
        tags.discard("species")
        tags.discard("pos")
        return tags

    def update_structure_metadata(
        self,
        remove_tags: Iterable[str],
        new_tag_info: Mapping[str, str],
    ) -> None:
        """Apply metadata removals and additions to the selected structures."""
        selected_structures = self.get_selected_structures()
        if not selected_structures:
            MessageManager.send_info_message("No data selected!")
            return

        for structure in selected_structures:
            for remove_tag in remove_tags:
                if remove_tag in structure.additional_fields:
                    structure.additional_fields.pop(remove_tag)
                elif remove_tag in structure.atomic_properties:
                    structure.remove_atomic_properties(remove_tag)

            for new_tag, value_text in new_tag_info.items():
                if value_text is None:
                    continue
                value_text = value_text.strip()
                if value_text == "":
                    continue
                try:
                    value = json.loads(value_text)
                    if isinstance(value, list):
                        value = np.array(value)
                except Exception:
                    try:
                        value = float(value_text)
                    except Exception:
                        value = value_text
                structure.additional_fields[new_tag] = value

        MessageManager.send_info_message("Edit completed")
        self.updateInfoSignal.emit()

    def iter_shift_energy_baseline(
        self,
        group_patterns: Sequence[str],
        alignment_mode: str,
        max_generations: int,
        population_size: int,
        convergence_tol: float,
        reference_indices: Optional[Sequence[int]] = None,
    ):
        """Shift dataset energies and yield progress units for UI hooks."""
        if reference_indices is None:
            ref_index = list(self.select_index)
        else:
            ref_index = list(reference_indices)

        reference_structures = self.structure.all_data[ref_index] if ref_index else []
        nep_energy_array = None
        if hasattr(self, "energy"):
            nep_energy_array = getattr(self.energy, "y", None)

        for progress in shift_dataset_energy(
            structures=self.structure.now_data,
            reference_structures=reference_structures,
            max_generations=max_generations,
            population_size=population_size,
            convergence_tol=convergence_tol,
            group_patterns=list(group_patterns),
            alignment_mode=alignment_mode,
            nep_energy_array=nep_energy_array,
        ):
            yield progress

        self.sync_structures(["energy"])

    def apply_dft_d3_correction(
        self,
        mode: int,
        functional: str,
        cutoff: float,
        cutoff_cn: float,
    ) -> None:
        """Apply DFT-D3 corrections and synchronise dependent datasets."""
        nep_calc = NepCalculator(
            model_file=self.nep_txt_path.as_posix(),
            backend=NepBackend.CPU,
            batch_size=Config.getint("nep", "gpu_batch_size", 1000),
        )

        potentials, forces, virials = nep_calc.calculate_dftd3(
            self.structure.now_data.tolist(),
            functional=functional,
            cutoff=cutoff,
            cutoff_cn=cutoff_cn,
        )

        if self.structure.now_data.size == 0:
            return
        factor = 1 if mode == 0 else -1
        for idx, structure in enumerate(self.structure.now_data):
            try:
                structure.energy += potentials[idx] * factor
            except Exception:
                pass
            try:
                structure.forces += forces[idx] * factor
            except Exception:
                pass
            if getattr(structure, "has_virial", False):
                try:
                    structure.virial += virials[idx]*len(structure) * factor
                except Exception:
                    pass

        self.sync_structures(["energy", "force", "virial", "stress"])

    def _load_descriptors(self):
        """Load cached descriptors or generate them with the calculator."""
        if self.descriptor_path.exists():
            desc_array = read_nep_out_file(self.descriptor_path,dtype=np.float32,ndmin=2)
        else:
            desc_array = np.array([])
        if desc_array.size == 0:
            desc_array = self.nep_calc.get_structures_descriptor(self.structure.now_data.tolist())

            if desc_array.size != 0:
                np.savetxt(self.descriptor_path, desc_array, fmt='%.6g')
        else:
            if desc_array.shape[0] == np.sum(self.atoms_num_list):
                desc_array = aggregate_per_atom_to_structure(desc_array, self.atoms_num_list, map_func=np.mean, axis=0)
            elif desc_array.shape[0] == self.atoms_num_list.shape[0]:
                pass
            else:
                self.descriptor_path.unlink(True)
                return self._load_descriptors()
        # Cache raw (pre-PCA) per-structure descriptors to avoid reloading later
        # This enables advanced sampling to use original descriptor space when requested.
        if desc_array.size != 0:
            # Ensure float32 and store an immutable copy for later masking
            self._descriptor_raw_all = np.asarray(desc_array, dtype=np.float32)
        else:
            self._descriptor_raw_all = np.array([], dtype=np.float32)

        # Prepare reduced (PCA) descriptors for plotting
        reduced = self._descriptor_raw_all
        if reduced.size != 0 and reduced.shape[1] > 2:
            try:
                reduced = pca(reduced, 2)
            except Exception:
                MessageManager.send_error_message("PCA dimensionality reduction fails")
                reduced = np.array([], dtype=np.float32)
        self._descriptor_dataset = NepPlotData(reduced, title="descriptor")
    def __repr__(self):
        info = f"{self.__class__.__name__}(Orig: {self.atoms_num_list.shape[0]} Now: {self.structure.now_data.shape[0]} " \
               f"Rm: {self.structure.remove_data.shape[0]} Sel: {len(self.select_index)} Unsel: {self.structure.now_data.shape[0] - len(self.select_index)})"
        return info
