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
from NepTrainKit.core.structure import Structure, atomic_numbers, get_type_map, save_npy_structure
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
        elif self.title == "bec":
            unit, scale = "e", 1000

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

    @staticmethod
    def _normalise_search_type(search_type: SearchType | str | None) -> SearchType:
        if search_type is None:
            return SearchType.TAG
        if isinstance(search_type, SearchType):
            return search_type
        val = str(search_type).strip()
        if val.startswith(f"{SearchType.__name__}.") and "." in val:
            name = val.split(".")[-1]
            try:
                return SearchType[name]
            except Exception:
                pass
        try:
            return SearchType(val)
        except Exception:
            MessageManager.send_warning_message(f"Unsupported search type: {search_type}")
            return SearchType.TAG

    @staticmethod
    def _normalise_element_symbol(symbol: str) -> str:
        symbol = symbol.strip()
        if not symbol:
            return ""
        if len(symbol) == 1:
            return symbol.upper()
        return symbol[0].upper() + symbol[1:].lower()

    @classmethod
    def _parse_elements_query(cls, config: str) -> tuple[set[str], set[str], set[str]]:
        """Parse an element query into (allowed, required, excluded) sets.

        Query syntax
        ------------
        - ``Fe,O``: only elements from this set (subset constraint)
        - ``+Fe,+O``: must include these elements (no subset constraint)
        - ``-H`` / ``!H``: must not include this element

        Tokens can be separated by commas or whitespace, e.g. ``Fe O -H``.
        """
        allowed: set[str] = set()
        required: set[str] = set()
        excluded: set[str] = set()

        if not config:
            return allowed, required, excluded

        raw_tokens = re.split(r"[,\s]+", config.strip())
        for raw in raw_tokens:
            if not raw:
                continue
            op = ""
            token = raw.strip()
            if token[:1] in {"+", "-", "!"}:
                op = token[0]
                token = token[1:]
            token = cls._normalise_element_symbol(token)
            if not token:
                continue
            if token not in atomic_numbers:
                MessageManager.send_warning_message(f"Unknown element symbol: {token}")
                continue
            if op == "+":
                required.add(token)
            elif op in {"-", "!"}:
                excluded.add(token)
            else:
                allowed.add(token)
        return allowed, required, excluded

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
        search_type = self._normalise_search_type(search_type)
        if search_type == SearchType.TAG:
            return [structure.tag for structure in self.now_data]
        if search_type == SearchType.FORMULA:
            return [structure.formula for structure in self.now_data]
        if search_type == SearchType.ELEMENTS:
            words: list[str] = []
            for structure in self.now_data:
                try:
                    words.extend(sorted(set(map(str, structure.elements))))
                except Exception:
                    continue
            return words
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
        search_type = self._normalise_search_type(search_type)
        if search_type == SearchType.TAG:
            result_index = [i for i, structure in enumerate(self.now_data) if re.search(config, structure.tag)]
        elif search_type == SearchType.FORMULA:
            result_index = [i for i, structure in enumerate(self.now_data) if re.search(config, structure.formula)]
        elif search_type == SearchType.ELEMENTS:
            allowed, required, excluded = self._parse_elements_query(config)
            result_index = []
            for i, structure in enumerate(self.now_data):
                try:
                    elem_set = set(map(str, structure.elements))
                except Exception:
                    continue
                if excluded and elem_set.intersection(excluded):
                    continue
                if required and not required.issubset(elem_set):
                    continue
                if allowed and not elem_set.issubset(allowed):
                    continue
                result_index.append(i)
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
    _abcs: npt.NDArray[np.float32]
    _angles: npt.NDArray[np.float32]
    def __init__(self,
                 nep_txt_path: Path,
                 data_xyz_path: Path,
                 descriptor_path: Path,
                 calculator_factory: Optional[Callable[[str], Any]] = None,
                 import_options: Optional[dict[str, Any]] = None):
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
        # Mark structures as "bad/reject" without interfering with selection.
        # Uses original structure indices aligned with StructureData/group_array indices.
        self.reject_index: set[int] = set()
        # Optional pre-fetched structures to skip IO in load_structures
        self._prefetched_structures: Optional[list[Structure]] = None
        # Optional importer options forwarded to importers.import_structures
        self._import_options: dict[str, Any] = dict(import_options or {})
        self.calculator_factory=calculator_factory
        self._structure_sync_rules = dict(getattr(self, "STRUCTURE_SYNC_RULES", {}))
        self._pending_non_physical_indices: list[int] = []
        self._sampler = SparseSampler(self)
        self._abcs = np.empty((0, 3), dtype=np.float32)
        self._angles = np.empty((0, 3), dtype=np.float32)
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
        # Cache lattice parameters for all structures to avoid repeated calculations
        self._abcs = np.array([s.abc for s in structures], dtype=np.float32)
        self._angles = np.array([s.angles for s in structures], dtype=np.float32)
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
        if not self.cache_outputs_enabled():
            return
        if self.atoms_num_list.shape[0] > 1000:
            #
            if not self.data_xyz_path.with_name("nep.in").exists():
                with open(self.data_xyz_path.with_name("nep.in"),
                          "w", encoding="utf8") as f:
                    f.write("prediction 1 ")

    @staticmethod
    def cache_outputs_enabled() -> bool:
        """Return whether loader-generated cache files should be written."""
        return bool(Config.getboolean("io", "cache_outputs", True))
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

    @property
    def abcs(self) -> npt.NDArray[np.float32]:
        """Return the cached lattice vector lengths (a, b, c) for all structures."""
        return self._abcs

    @property
    def angles(self) -> npt.NDArray[np.float32]:
        """Return the cached lattice angles (alpha, beta, gamma) for all structures."""
        return self._angles

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

    def select_structures_by_lattice_range(
        self,
        a_range: tuple[float, float],
        b_range: tuple[float, float],
        c_range: tuple[float, float],
        alpha_range: tuple[float, float],
        beta_range: tuple[float, float],
        gamma_range: tuple[float, float],
    ) -> list[int]:
        """Return structure indices whose lattice parameters fall within the given ranges.
        
        Uses a fixed tolerance of 1e-4 to handle floating-point precision loss from
        float32 storage of lattice vectors, independent of range size.
        """
        # Use vectorized comparison on cached lattice parameters for performance
        now_indices = self.structure.now_indices
        if now_indices.size == 0:
            return []
            
        abcs = self._abcs[now_indices]
        angles = self._angles[now_indices]
        
        # Fixed tolerance for float32 precision loss
        tolerance = 1e-4

        mask = (
            (a_range[0] - tolerance <= abcs[:, 0]) & (abcs[:, 0] <= a_range[1] + tolerance) &
            (b_range[0] - tolerance <= abcs[:, 1]) & (abcs[:, 1] <= b_range[1] + tolerance) &
            (c_range[0] - tolerance <= abcs[:, 2]) & (abcs[:, 2] <= c_range[1] + tolerance) &
            (alpha_range[0] - tolerance <= angles[:, 0]) & (angles[:, 0] <= alpha_range[1] + tolerance) &
            (beta_range[0] - tolerance <= angles[:, 1]) & (angles[:, 1] <= beta_range[1] + tolerance) &
            (gamma_range[0] - tolerance <= angles[:, 2]) & (angles[:, 2] <= gamma_range[1] + tolerance)
        )
        
        indices = self.structure.group_array.now_data
        return indices[mask].astype(int).tolist()

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

    def export_selected_npy(self, save_path: str | Path) -> None:
        """Export selected structures as a DeepMD-style ``deepmd/npy`` dataset."""
        try:
            selected = self.get_selected_structures()
            if not selected:
                MessageManager.send_info_message("Please select some structures first!")
                return
            target = Path(save_path).joinpath("export_selected_model")
            all_structures = self.structure.all_data.tolist()
            type_map = get_type_map(all_structures) if all_structures else None
            save_npy_structure(str(target), selected, type_map=type_map)
            MessageManager.send_info_message(f"File exported to: {target}")
        except Exception:
            MessageManager.send_info_message(
                "An unknown error occurred while saving. The error message has been output to the log!"
            )
            logger.error(traceback.format_exc())

    def export_active_xyz(self, save_file_path: str | Path) -> None:
        """Write active (non-removed) structures to ``save_file_path``."""
        try:
            active = self.structure.now_data
            if getattr(active, "size", 0) == 0:
                MessageManager.send_info_message("No active structures to export.")
                return
            with open(save_file_path, "w", encoding="utf8") as handle:
                for structure in active:
                    structure.write(handle)
            MessageManager.send_info_message(f"File exported to: {save_file_path}")
        except Exception:
            MessageManager.send_info_message(
                "An unknown error occurred while saving. The error message has been output to the log!"
            )
            logger.error(traceback.format_exc())

    def export_active_npy(self, save_path: str | Path) -> None:
        """Export active (non-removed) structures as a DeepMD-style ``deepmd/npy`` dataset."""
        try:
            active = self.structure.now_data.tolist()
            if not active:
                MessageManager.send_info_message("No active structures to export.")
                return
            target = Path(save_path).joinpath("export_active_model")
            all_structures = self.structure.all_data.tolist()
            type_map = get_type_map(all_structures) if all_structures else None
            save_npy_structure(str(target), active, type_map=type_map)
            MessageManager.send_info_message(f"File exported to: {target}")
        except Exception:
            MessageManager.send_info_message(
                "An unknown error occurred while saving. The error message has been output to the log!"
            )
            logger.error(traceback.format_exc())

    def export_removed_xyz(self, save_file_path: str | Path) -> None:
        """Write removed structures (if any) to ``save_file_path``."""
        try:
            removed = self.structure.remove_data
            if getattr(removed, "size", 0) == 0:
                MessageManager.send_info_message("No removed structures to export.")
                return
            with open(save_file_path, "w", encoding="utf8") as handle:
                for structure in removed:
                    structure.write(handle)
            MessageManager.send_info_message(f"File exported to: {save_file_path}")
        except Exception:
            MessageManager.send_info_message(
                "An unknown error occurred while saving. The error message has been output to the log!"
            )
            logger.error(traceback.format_exc())

    def export_removed_npy(self, save_path: str | Path) -> None:
        """Export removed structures as a DeepMD-style ``deepmd/npy`` dataset."""
        try:
            removed = self.structure.remove_data.tolist()
            if not removed:
                MessageManager.send_info_message("No removed structures to export.")
                return
            target = Path(save_path).joinpath("export_remove_model")
            all_structures = self.structure.all_data.tolist()
            type_map = get_type_map(all_structures) if all_structures else None
            save_npy_structure(str(target), removed, type_map=type_map)
            MessageManager.send_info_message(f"File exported to: {target}")
        except Exception:
            MessageManager.send_info_message(
                "An unknown error occurred while saving. The error message has been output to the log!"
            )
            logger.error(traceback.format_exc())

    def export_current_npy(self, save_path: str | Path, index: int) -> None:
        """Export a single structure as DeepMD-style ``deepmd/npy`` dataset."""
        try:
            mapped = self.structure.convert_index(index)
            structure = self.structure.all_data[mapped][0]
            target = Path(save_path).joinpath(f"structure_{int(index)}")
            all_structures = self.structure.all_data.tolist()
            type_map = get_type_map(all_structures) if all_structures else None
            save_npy_structure(str(target), [structure], type_map=type_map)
            MessageManager.send_info_message(f"File exported to: {target}")
        except Exception:
            MessageManager.send_info_message(
                "An unknown error occurred while saving. The error message has been output to the log!"
            )
            logger.error(traceback.format_exc())

    def export_model_extxyz(self, save_path: str | Path) -> None:
        """Export active and removed structures into ``save_path`` folder as extxyz."""
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
            MessageManager.send_info_message(
                "An unknown error occurred while saving. The error message has been output to the log!"
            )
            logger.error(traceback.format_exc())

    def export_model_npy(self, save_path: str | Path) -> None:
        """Export active and removed structures into ``save_path`` folder as deepmd/npy."""
        try:
            target = Path(save_path)
            good_target = target / "export_good_model"
            removed_target = target / "export_remove_model"
            all_structures = self.structure.all_data.tolist()
            type_map = get_type_map(all_structures) if all_structures else None
            save_npy_structure(str(good_target), self.structure.now_data.tolist(), type_map=type_map)
            save_npy_structure(str(removed_target), self.structure.remove_data.tolist(), type_map=type_map)
            MessageManager.send_info_message(f"File exported to: {target}")
        except Exception:
            MessageManager.send_info_message(
                "An unknown error occurred while saving. The error message has been output to the log!"
            )
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

    def iter_unbalanced_force_indices(self, threshold: float):
        """Yield progress units while collecting structures with non-zero net force.

        Parameters
        ----------
        threshold : float
            Minimum allowed magnitude of the summed force vector |ΣF|. Structures
            whose net force exceeds this value are recorded for later selection.
        """
        structures = self.structure.now_data
        group_array = self.structure.group_array.now_data
        pending: list[int] = []
        if structures.size == 0:
            return
        thr = float(threshold)
        for structure, index in zip(structures, group_array):
            if getattr(structure, "has_forces", False):
                try:
                    forces = np.asarray(structure.forces, dtype=np.float64)
                    if forces.size != 0:
                        net = forces.sum(axis=0)
                        norm = float(np.linalg.norm(net))
                        if norm > thr:
                            pending.append(int(index))
                except Exception:
                    logger.debug(traceback.format_exc())
            yield 1
        self._pending_unbalanced_force_indices = pending

    def consume_unbalanced_force_indices(self) -> list[int]:
        """Return and clear indices collected by the net-force scan."""
        indices = getattr(self, "_pending_unbalanced_force_indices", [])
        self._pending_unbalanced_force_indices = []
        return list(indices)

    def iter_dataset_summary(self, group_by: SearchType = SearchType.TAG):
        """Aggregate dataset-wide statistics for use in summary dialogs.

        Notes
        -----
        This generator yields a progress unit after each structure so that
        callers can drive a progress dialog. Results are cached on the
        instance and later returned by :meth:`get_dataset_summary`.

        Parameters
        ----------
        group_by : SearchType, default=SearchType.TAG
            Attribute used for grouping the distribution table. ``TAG`` uses
            ``Structure.tag`` (Config_type), while ``FORMULA`` uses
            ``Structure.formula``.
        """
        summary: dict[str, Any] = {}
        structures = getattr(self, "structure", None)
        if structures is None or structures.now_data.size == 0:
            self._dataset_summary = {}
            return

        active_structures = int(structures.now_data.shape[0])
        orig_structures = int(self.atoms_num_list.shape[0]) if hasattr(self, "atoms_num_list") else active_structures
        removed_structures = int(structures.remove_data.shape[0])
        selected_structures = len(self.select_index)
        unselected_structures = max(0, active_structures - selected_structures)

        atoms_per_struct: list[int] = []
        total_atoms_active = 0
        element_atom_counts: dict[str, int] = {}
        element_structure_counts: dict[str, int] = {}
        group_counts: dict[str, int] = {}

        # Numeric distributions for HTML export (exact histograms, no sampling)
        # We keep exact min/max/total stats and build histograms online.
        dist_metrics: dict[str, dict[str, Any]] = {}
        hist_bins_initial = 160
        hist_bins_max = 2000

        class _StreamHist:
            def __init__(self, *, bins_initial: int, bins_max: int):
                self._bins_initial = int(bins_initial)
                self._bins_max = int(bins_max)
                self.left: float | None = None
                self.width: float | None = None
                self.counts: npt.NDArray[np.int64] = np.zeros(0, dtype=np.int64)
                self.total = 0
                self.sum = 0.0
                self.sum_sq = 0.0
                self.min = float("inf")
                self.max = float("-inf")

            def _ensure_range(self, mn: float, mx: float) -> None:
                if self.left is None or self.width is None:
                    return
                left = float(self.left)
                width = float(self.width)
                if not (np.isfinite(left) and np.isfinite(width) and width > 0.0):
                    return
                if self.counts.size == 0:
                    self.counts = np.zeros(self._bins_initial, dtype=np.int64)
                min_idx = int(np.floor((mn - left) / width))
                max_idx = int(np.floor((mx - left) / width))
                if min_idx < 0:
                    add = -min_idx
                    left -= float(add) * width
                    self.counts = np.concatenate([np.zeros(add, dtype=np.int64), self.counts])
                if max_idx >= self.counts.size:
                    add = max_idx - int(self.counts.size) + 1
                    self.counts = np.concatenate([self.counts, np.zeros(add, dtype=np.int64)])
                self.left = left

            def _shrink_if_needed(self) -> None:
                if self.width is None:
                    return
                while self.counts.size > self._bins_max:
                    if self.counts.size % 2 == 1:
                        self.counts = np.concatenate([self.counts, np.zeros(1, dtype=np.int64)])
                    self.counts = self.counts.reshape(-1, 2).sum(axis=1)
                    self.width = float(self.width) * 2.0

            def update(self, values: Any) -> None:
                try:
                    arr = np.asarray(values, dtype=np.float64).reshape(-1)
                except Exception:
                    return
                if arr.size == 0:
                    return
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return

                self.total += int(arr.size)
                self.sum += float(arr.sum())
                self.sum_sq += float(np.square(arr).sum())

                mn = float(arr.min())
                mx = float(arr.max())
                self.min = float(min(self.min, mn))
                self.max = float(max(self.max, mx))

                if self.left is None or self.width is None:
                    rng = mx - mn
                    width = float(rng / float(self._bins_initial)) if rng > 0 else 1.0
                    if not np.isfinite(width) or width <= 0:
                        width = 1.0
                    width = max(width, 1e-12)
                    self.width = width
                    self.left = float(np.floor(mn / width) * width)
                    self.counts = np.zeros(self._bins_initial, dtype=np.int64)

                self._ensure_range(mn, mx)
                if self.left is None or self.width is None or self.counts.size == 0:
                    return
                idx = np.floor((arr - float(self.left)) / float(self.width)).astype(np.int64)
                mask = (idx >= 0) & (idx < self.counts.size)
                if np.any(mask):
                    bc = np.bincount(idx[mask], minlength=self.counts.size).astype(np.int64, copy=False)
                    self.counts += bc
                self._shrink_if_needed()

            def export(self) -> dict[str, Any]:
                if self.left is None or self.width is None:
                    return {"hist": [], "hist_left": 0.0, "hist_right": 0.0, "bins": 0}
                bins = int(self.counts.size)
                left = float(self.left)
                right = left + float(self.width) * float(bins)
                return {
                    "hist": self.counts.astype(np.int64, copy=False).tolist(),
                    "hist_left": float(left),
                    "hist_right": float(right),
                    "bins": bins,
                }

        class _StreamHistMulti:
            def __init__(self, names: Sequence[str], *, bins_initial: int, bins_max: int):
                self.names = [str(n) for n in names]
                self._bins_initial = int(bins_initial)
                self._bins_max = int(bins_max)
                self.left: float | None = None
                self.width: float | None = None
                self.counts: npt.NDArray[np.int64] = np.zeros((len(self.names), 0), dtype=np.int64)
                self.total = 0
                self.min = float("inf")
                self.max = float("-inf")

            def _ensure_range(self, mn: float, mx: float) -> None:
                if self.left is None or self.width is None:
                    return
                left = float(self.left)
                width = float(self.width)
                if not (np.isfinite(left) and np.isfinite(width) and width > 0.0):
                    return
                if self.counts.shape[1] == 0:
                    self.counts = np.zeros((len(self.names), self._bins_initial), dtype=np.int64)
                min_idx = int(np.floor((mn - left) / width))
                max_idx = int(np.floor((mx - left) / width))
                if min_idx < 0:
                    add = -min_idx
                    left -= float(add) * width
                    pad = np.zeros((len(self.names), add), dtype=np.int64)
                    self.counts = np.concatenate([pad, self.counts], axis=1)
                if max_idx >= self.counts.shape[1]:
                    add = max_idx - int(self.counts.shape[1]) + 1
                    pad = np.zeros((len(self.names), add), dtype=np.int64)
                    self.counts = np.concatenate([self.counts, pad], axis=1)
                self.left = left

            def _shrink_if_needed(self) -> None:
                if self.width is None:
                    return
                while self.counts.shape[1] > self._bins_max:
                    if self.counts.shape[1] % 2 == 1:
                        pad = np.zeros((len(self.names), 1), dtype=np.int64)
                        self.counts = np.concatenate([self.counts, pad], axis=1)
                    # merge adjacent bins
                    self.counts = self.counts.reshape(len(self.names), -1, 2).sum(axis=2)
                    self.width = float(self.width) * 2.0

            def update(self, values: Any) -> None:
                try:
                    arr = np.asarray(values, dtype=np.float64).reshape(len(self.names))
                except Exception:
                    return
                if arr.size != len(self.names):
                    return
                if not np.all(np.isfinite(arr)):
                    return
                self.total += 1
                mn = float(arr.min())
                mx = float(arr.max())
                self.min = float(min(self.min, mn))
                self.max = float(max(self.max, mx))

                if self.left is None or self.width is None:
                    rng = mx - mn
                    width = float(rng / float(self._bins_initial)) if rng > 0 else 1.0
                    if not np.isfinite(width) or width <= 0:
                        width = 1.0
                    width = max(width, 1e-12)
                    self.width = width
                    self.left = float(np.floor(mn / width) * width)
                    self.counts = np.zeros((len(self.names), self._bins_initial), dtype=np.int64)

                self._ensure_range(mn, mx)
                if self.left is None or self.width is None or self.counts.shape[1] == 0:
                    return
                idx = np.floor((arr - float(self.left)) / float(self.width)).astype(np.int64)
                for i in range(len(self.names)):
                    j = int(idx[i])
                    if 0 <= j < self.counts.shape[1]:
                        self.counts[i, j] += 1
                self._shrink_if_needed()

            def update_many(self, values: Any) -> None:
                try:
                    arr2 = np.asarray(values, dtype=np.float64).reshape(-1, len(self.names))
                except Exception:
                    return
                if arr2.size == 0:
                    return
                mask = np.all(np.isfinite(arr2), axis=1)
                if not np.any(mask):
                    return
                arr2 = arr2[mask]
                self.total += int(arr2.shape[0])
                mn = float(arr2.min())
                mx = float(arr2.max())
                self.min = float(min(self.min, mn))
                self.max = float(max(self.max, mx))

                if self.left is None or self.width is None:
                    rng = mx - mn
                    width = float(rng / float(self._bins_initial)) if rng > 0 else 1.0
                    if not np.isfinite(width) or width <= 0:
                        width = 1.0
                    width = max(width, 1e-12)
                    self.width = width
                    self.left = float(np.floor(mn / width) * width)
                    self.counts = np.zeros((len(self.names), self._bins_initial), dtype=np.int64)

                self._ensure_range(mn, mx)
                if self.left is None or self.width is None or self.counts.shape[1] == 0:
                    return
                idx = np.floor((arr2 - float(self.left)) / float(self.width)).astype(np.int64)
                for i in range(len(self.names)):
                    col = idx[:, i]
                    m2 = (col >= 0) & (col < self.counts.shape[1])
                    if np.any(m2):
                        bc = np.bincount(col[m2], minlength=self.counts.shape[1]).astype(np.int64, copy=False)
                        self.counts[i] += bc
                self._shrink_if_needed()

            def export(self) -> dict[str, Any]:
                if self.left is None or self.width is None:
                    return {"bins": 0, "hist_left": 0.0, "hist_right": 0.0, "series": []}
                bins = int(self.counts.shape[1])
                left = float(self.left)
                right = left + float(self.width) * float(bins)
                series = []
                for i, name in enumerate(self.names):
                    series.append(
                        {"name": name, "hist": self.counts[i].astype(np.int64, copy=False).tolist()}
                    )
                return {"bins": bins, "hist_left": float(left), "hist_right": float(right), "series": series}

        def _update_metric(
            key: str,
            *,
            label: str,
            unit: str = "",
            values: npt.NDArray[np.floating[Any]]
            | npt.NDArray[np.integer[Any]]
            | Sequence[float]
            | float
            | int,
            sample_per_structure: int = 512,
        ) -> None:
            metric = dist_metrics.get(key)
            if metric is None:
                metric = {
                    "key": key,
                    "label": str(label),
                    "unit": str(unit or ""),
                    "hist": _StreamHist(bins_initial=hist_bins_initial, bins_max=hist_bins_max),
                }
                dist_metrics[key] = metric
            metric["label"] = str(label)
            metric["unit"] = str(unit or "")
            try:
                metric["hist"].update(values)
            except Exception:
                return

        # Force RMS summary (text only; no plot)
        force_sum_sq = 0.0
        force_count = 0
        force_rms_per_structure: list[float] = []
        force_structures = 0

        mag_key_re = re.compile(r"(mag|spin)", re.IGNORECASE)

        for s, idx in zip(structures.now_data, structures.group_array.now_data):
            try:
                n_atoms = int(len(s))
            except Exception:
                n_atoms = 0
            atoms_per_struct.append(n_atoms)
            total_atoms_active += n_atoms

            # Element statistics
            try:
                elems = [str(e) for e in s.elements]
            except Exception:
                elems = []
            if elems:
                # atom counts
                for e in elems:
                    element_atom_counts[e] = element_atom_counts.get(e, 0) + 1
                # per-structure presence counts
                for e in set(elems):
                    element_structure_counts[e] = element_structure_counts.get(e, 0) + 1

            # Group distribution (Config_type via tag, or formula)
            if group_by == SearchType.FORMULA:
                group_value = getattr(s, "formula", "") or ""
            else:
                group_value = getattr(s, "tag", "") or ""
            if isinstance(group_value, str) and group_value:
                group_counts[group_value] = group_counts.get(group_value, 0) + 1

            # Magnetism/spin numeric arrays (DFT/extxyz per-atom fields)
            try:
                atomic_props = getattr(s, "atomic_properties", {}) or {}
                for prop_key, prop_val in atomic_props.items():
                    if prop_key in {"pos", "species", "species_id"}:
                        continue
                    # Avoid duplicating forces handled above
                    if prop_key == getattr(s, "force_label", ""):
                        continue
                    if mag_key_re.search(str(prop_key)) is None:
                        continue
                    try:
                        arr = np.asarray(prop_val)
                    except Exception:
                        continue
                    if arr.size == 0:
                        continue
                    if not np.issubdtype(arr.dtype, np.number):
                        continue
                    arr = np.asarray(arr, dtype=np.float64)
                    if arr.ndim == 1:
                        _update_metric(
                            f"mag.{prop_key}",
                            label=f"{prop_key} (all atoms)",
                            unit="",
                            values=arr,
                            sample_per_structure=1024,
                        )
                    elif arr.ndim == 2 and arr.shape[1] == 1:
                        _update_metric(
                            f"mag.{prop_key}",
                            label=f"{prop_key} (all atoms)",
                            unit="",
                            values=arr.reshape(-1),
                            sample_per_structure=1024,
                        )
                    elif arr.ndim == 2 and arr.shape[1] == 3:
                        _update_metric(
                            f"mag.{prop_key}.component",
                            label=f"{prop_key} components (all atoms)",
                            unit="",
                            values=arr.reshape(-1),
                            sample_per_structure=2048,
                        )
            except Exception:
                logger.debug(traceback.format_exc())

            # Yield a progress unit for UI hooks
            yield 1

        # ------------------------------------------------------------------
        # Prefer plot datasets for numeric distributions.
        # In this codebase, NepPlotData/DPPlotData `.x` corresponds to the DFT/reference axis.
        # ------------------------------------------------------------------
        try:
            energy_ds = getattr(self, "energy", None)
            if energy_ds is not None and getattr(energy_ds, "now_data", None) is not None:
                e_vals = np.asarray(energy_ds.x, dtype=np.float64).reshape(-1)
                e_vals = e_vals[np.isfinite(e_vals)]
                if e_vals.size:
                    _update_metric("energy.per_atom", label="Energy per atom", unit="eV/atom", values=e_vals)
        except Exception:
            logger.debug(traceback.format_exc())

        try:
            force_ds = getattr(self, "force", None)
            if force_ds is not None and getattr(force_ds, "now_data", None) is not None:
                # Use plotted x-axis directly (DFT/reference values).
                f_vals = np.asarray(force_ds.x, dtype=np.float64).reshape(-1)
                f_vals = f_vals[np.isfinite(f_vals)]
                if f_vals.size:
                    _update_metric("force.component", label="Force components (DFT, x/y/z)", unit="eV/Å", values=f_vals)
                # RMS + atom count when x contains 3 components per atom.
                if f_vals.size and (f_vals.size % 3 == 0):
                    force_mat = f_vals.reshape(-1, 3)
                    mags = np.linalg.norm(force_mat, axis=1)
                    mags = mags[np.isfinite(mags)]
                    if mags.size:
                        force_count = int(mags.size)
                        force_sum_sq = float(np.sum(np.square(mags)))
                        try:
                            force_structures = int(np.unique(force_ds.group_array.now_data).size)
                        except Exception:
                            force_structures = 0
                        force_rms_per_structure = []  # not displayed; keep empty
        except Exception:
            logger.debug(traceback.format_exc())

        try:
            virial_ds = getattr(self, "virial", None)
            if virial_ds is not None and getattr(virial_ds, "now_data", None) is not None and virial_ds.now_data.size != 0:
                # Use plotted x-axis directly (DFT/reference values).
                v_flat = np.asarray(virial_ds.x, dtype=np.float64).reshape(-1)
                if v_flat.size and (v_flat.size % 6 == 0):
                    v6 = v_flat.reshape(-1, 6)
                    v6 = v6[np.all(np.isfinite(v6), axis=1)]
                    if v6.size:
                        v_labels = ["xx", "yy", "zz", "xy", "yz", "zx"]
                        virial_hist = _StreamHistMulti(v_labels, bins_initial=hist_bins_initial, bins_max=hist_bins_max)
                        virial_hist.update_many(v6)
                else:
                    virial_hist = None  # type: ignore[assignment]
            else:
                virial_hist = None  # type: ignore[assignment]
        except Exception:
            logger.debug(traceback.format_exc())
            virial_hist = None  # type: ignore[assignment]

        try:
            stress_ds = getattr(self, "stress", None)
            if stress_ds is not None and getattr(stress_ds, "now_data", None) is not None and stress_ds.now_data.size != 0:
                # Use plotted x-axis directly (DFT/reference values).
                s_flat = np.asarray(stress_ds.x, dtype=np.float64).reshape(-1)
                if s_flat.size and (s_flat.size % 6 == 0):
                    s6 = s_flat.reshape(-1, 6)
                    s6 = s6[np.all(np.isfinite(s6), axis=1)]
                    if s6.size:
                        s_labels = ["xx", "yy", "zz", "xy", "yz", "zx"]
                        stress_hist = _StreamHistMulti(s_labels, bins_initial=hist_bins_initial, bins_max=hist_bins_max)
                        stress_hist.update_many(s6)
                        stress_unit = "GPa"
                else:
                    stress_hist = None  # type: ignore[assignment]
                    stress_unit = "GPa"
            else:
                stress_hist = None  # type: ignore[assignment]
                stress_unit = "GPa"
        except Exception:
            logger.debug(traceback.format_exc())
            stress_hist = None  # type: ignore[assignment]
            stress_unit = "GPa"

        # Prepare counts section
        summary_counts = {
            "orig_structures": orig_structures,
            "active_structures": active_structures,
            "removed_structures": removed_structures,
            "selected_structures": selected_structures,
            "unselected_structures": unselected_structures,
        }

        # Atom statistics
        atoms_array = np.asarray(atoms_per_struct, dtype=float) if atoms_per_struct else np.array([], dtype=float)
        if atoms_array.size:
            atoms_stats = {
                "total_atoms_active": int(total_atoms_active),
                "min_atoms": int(atoms_array.min()),
                "max_atoms": int(atoms_array.max()),
                "mean_atoms": float(atoms_array.mean()),
                "median_atoms": float(np.median(atoms_array)),
            }
        else:
            atoms_stats = {
                "total_atoms_active": 0,
                "min_atoms": 0,
                "max_atoms": 0,
                "mean_atoms": 0.0,
                "median_atoms": 0.0,
            }

        # Element statistics table
        elements_table: list[dict[str, Any]] = []
        if element_atom_counts:
            total_atoms = float(sum(element_atom_counts.values())) or 1.0
            for symbol in sorted(element_atom_counts.keys()):
                atoms = int(element_atom_counts[symbol])
                structs = int(element_structure_counts.get(symbol, 0))
                frac = atoms / total_atoms
                elements_table.append(
                    {
                        "symbol": symbol,
                        "atoms": atoms,
                        "structures": structs,
                        "fraction": frac,
                    }
                )

        # Group distribution (sorted by count desc)
        group_table: list[dict[str, Any]] = []
        if group_counts:
            total_groups = float(sum(group_counts.values())) or 1.0
            for name, count in sorted(group_counts.items(), key=lambda kv: kv[1], reverse=True):
                frac = count / total_groups
                group_table.append(
                    {
                        "name": name,
                        "count": int(count),
                        "fraction": frac,
                    }
                )

        # Energy statistics (prefer plot dataset x-axis = DFT/reference)
        energy_stats: dict[str, Any] = {"count": 0}
        try:
            energy_ds = getattr(self, "energy", None)
            if energy_ds is not None and getattr(energy_ds, "now_data", None) is not None:
                e_arr = np.asarray(energy_ds.x, dtype=np.float64).reshape(-1)
                e_arr = e_arr[np.isfinite(e_arr)]
                if e_arr.size:
                    energy_stats = {
                        "count": int(e_arr.size),
                        "min": float(e_arr.min()),
                        "max": float(e_arr.max()),
                        "mean": float(e_arr.mean()),
                        "std": float(e_arr.std()),
                        "median": float(np.median(e_arr)),
                    }
                    # Map back to original structure indices when one-to-one.
                    if int(e_arr.size) == int(active_structures):
                        try:
                            origin_map = np.asarray(structures.group_array.now_data, dtype=np.int64).reshape(-1)
                            if origin_map.size == e_arr.size:
                                energy_stats["min_index"] = int(origin_map[int(np.argmin(e_arr))])
                                energy_stats["max_index"] = int(origin_map[int(np.argmax(e_arr))])
                        except Exception:
                            pass
        except Exception:
            logger.debug(traceback.format_exc())

        # Health diagnostics
        health_messages: list[str] = []
        # Missing energy structures
        missing_energy = active_structures - energy_stats.get("count", 0)
        if missing_energy > 0:
            health_messages.append(
                f"{missing_energy} active structures are missing energy; they are ignored in energy statistics."
            )
        # Element coverage
        low_elements = [e for e in elements_table if e["fraction"] < 0.05]
        if low_elements:
            names = ", ".join(f"{e['symbol']} ({e['fraction']*100:.1f}%)" for e in low_elements[:5])
            health_messages.append(f"Elements with low atomic fraction (<5%): {names}.")
        # Dominant group label (Config_type or formula)
        if group_table:
            top_cfg = group_table[0]
            if top_cfg["fraction"] > 0.7:
                group_label = "Formula" if group_by == SearchType.FORMULA else "Config_type"
                health_messages.append(
                    f"{group_label} '{top_cfg['name']}' dominates {top_cfg['fraction']*100:.1f}% of active structures."
                )
        # Atom-count diversity
        if atoms_stats["min_atoms"] == atoms_stats["max_atoms"] and atoms_stats["min_atoms"] > 0:
            health_messages.append(
                "All active structures have the same atom count; consider adding systems with different sizes."
            )
        # Small dataset warning
        if active_structures < 100:
            health_messages.append(
                f"Only {active_structures} active structures; model training may be prone to overfitting."
            )

        summary["counts"] = summary_counts
        summary["atoms"] = atoms_stats
        summary["elements"] = elements_table
        summary["config_types"] = group_table
        summary["energy"] = energy_stats
        summary["health"] = health_messages
        summary["data_file"] = str(self.data_xyz_path.name)
        summary["model_file"] = str(self.nep_txt_path.name)
        summary["group_by"] = group_by.value
        if force_count > 0:
            per_struct = np.asarray(force_rms_per_structure, dtype=np.float64) if force_rms_per_structure else np.array([], dtype=np.float64)
            summary["force_rms"] = {
                "structures_with_forces": int(force_structures),
                "atoms_with_forces": int(force_count),
                "rms_all_atoms": float(np.sqrt(force_sum_sq / float(force_count))),
                "min_rms_per_structure": float(per_struct.min()) if per_struct.size else 0.0,
                "max_rms_per_structure": float(per_struct.max()) if per_struct.size else 0.0,
                "mean_rms_per_structure": float(per_struct.mean()) if per_struct.size else 0.0,
                "median_rms_per_structure": float(np.median(per_struct)) if per_struct.size else 0.0,
            }
        else:
            summary["force_rms"] = {
                "structures_with_forces": 0,
                "atoms_with_forces": 0,
                "rms_all_atoms": 0.0,
            }

        virial_total = int(getattr(virial_hist, "total", 0) or 0) if "virial_hist" in locals() else 0
        stress_total = int(getattr(stress_hist, "total", 0) or 0) if "stress_hist" in locals() else 0
        if dist_metrics or virial_total > 0 or stress_total > 0:
            metrics_list: list[dict[str, Any]] = []
            for item in dist_metrics.values():
                hist: _StreamHist | None = item.get("hist")
                if hist is None:
                    continue
                total = int(hist.total)
                if total <= 0:
                    continue
                s1 = float(hist.sum)
                s2 = float(hist.sum_sq)
                mean = s1 / float(total) if total > 0 else 0.0
                var = (s2 / float(total)) - mean * mean if total > 0 else 0.0
                std = float(np.sqrt(max(0.0, var)))
                exported = hist.export()
                metrics_list.append(
                    {
                        "key": item.get("key", ""),
                        "label": item.get("label", item.get("key", "")),
                        "unit": item.get("unit", ""),
                        "min": float(hist.min) if np.isfinite(hist.min) else 0.0,
                        "max": float(hist.max) if np.isfinite(hist.max) else 0.0,
                        "total": total,
                        "mean": float(mean),
                        "std": float(std),
                        **exported,
                    }
                )
            if virial_hist is not None and virial_total > 0 and np.isfinite(virial_hist.min) and np.isfinite(virial_hist.max):
                exported = virial_hist.export()
                metrics_list.append(
                    {
                        "key": "virial.per_atom.v6",
                        "label": "Virial per atom (DFT, 6 components)",
                        "unit": "eV/atom",
                        "min": float(virial_hist.min),
                        "max": float(virial_hist.max),
                        "total": int(virial_total),
                        **exported,
                    }
                )
            if stress_hist is not None and stress_total > 0 and np.isfinite(stress_hist.min) and np.isfinite(stress_hist.max):
                exported = stress_hist.export()
                metrics_list.append(
                    {
                        "key": "stress.v6",
                        "label": "Stress (DFT, 6 components)",
                        "unit": str(stress_unit),
                        "min": float(stress_hist.min),
                        "max": float(stress_hist.max),
                        "total": int(stress_total),
                        **exported,
                    }
                )
            metrics_list.sort(key=lambda d: str(d.get("key", "")))
            summary["numeric_distributions"] = {
                "metrics": metrics_list,
            }

        self._dataset_summary = summary

    def get_dataset_summary(self) -> dict[str, Any]:
        """Return the most recently computed dataset summary."""
        return dict(getattr(self, "_dataset_summary", {}) or {})

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
        sampling_mode: str = "count",
        r2_threshold: float = 0.9,
    ) -> tuple[list[int], bool]:
        """Delegate sparse sampling to the sampler helper."""
        return self._sampler.sparse_point_selection(
            n_samples=n_samples,
            distance=distance,
            descriptor_source=descriptor_source,
            restrict_to_selection=restrict_to_selection,
            training_path=training_path,
            sampling_mode=sampling_mode,
            r2_threshold=r2_threshold,
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
        tags.discard("species_id")
        tags.discard("pos")
        return tags

    def update_structure_metadata(
        self,
        remove_tags: Iterable[str],
        new_tag_info: Mapping[str, str],
        rename_map: Mapping[str, str] | None = None,
    ) -> None:
        """Apply metadata removals, additions, and key renames to the selected structures."""
        selected_structures = self.get_selected_structures()
        if not selected_structures:
            MessageManager.send_info_message("No data selected!")
            return

        for structure in selected_structures:
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

            if rename_map:
                for old_key, new_key in rename_map.items():
                    if not new_key or old_key == new_key:
                        continue

                    if old_key in structure.additional_fields:
                        value = structure.additional_fields.pop(old_key)
                        structure.additional_fields[new_key] = value
                        continue

                    if old_key in structure.atomic_properties:
                        value = structure.atomic_properties.pop(old_key)
                        structure.atomic_properties[new_key] = value

                        old_descriptor = None
                        for prop in structure.properties:
                            if prop.get("name") == old_key:
                                old_descriptor = prop
                                break
                        if old_descriptor is not None:
                            if new_key != old_key:
                                structure.properties = [
                                    prop for prop in structure.properties if prop.get("name") != new_key
                                ]
                            old_descriptor["name"] = new_key

            for remove_tag in remove_tags:
                if remove_tag in structure.additional_fields:
                    structure.additional_fields.pop(remove_tag)
                elif remove_tag in structure.atomic_properties:
                    structure.remove_atomic_properties(remove_tag)

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
        precomputed_baseline=None,
        baseline_store: Optional[dict] = None,
        source_summary: Optional[dict] = None,
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
            precomputed_baseline=precomputed_baseline,
            baseline_store=baseline_store,
            source_summary=source_summary,
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
        desc_array = np.array([])
        if self.descriptor_path.exists():
            try:
                desc_array = read_nep_out_file(self.descriptor_path, dtype=np.float32, ndmin=2)
            except Exception:
                desc_array = np.array([])

        if desc_array.size != 0:
            if desc_array.shape[0] == np.sum(self.atoms_num_list):
                desc_array = aggregate_per_atom_to_structure(desc_array, self.atoms_num_list, map_func=np.mean, axis=0)
            elif desc_array.shape[0] == self.atoms_num_list.shape[0]:
                pass
            else:
                if self.cache_outputs_enabled():
                    self.descriptor_path.unlink(True)
                    return self._load_descriptors()
                desc_array = np.array([])

        if desc_array.size == 0:
            desc_array = self.nep_calc.get_structures_descriptor(self.structure.now_data.tolist())
            if desc_array.size != 0 and self.cache_outputs_enabled():
                np.savetxt(self.descriptor_path, desc_array, fmt='%.6g')
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
