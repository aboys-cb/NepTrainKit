#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base datasets, PCA helper, and the abstract result container.

This module defines the common data abstractions used by result loaders. It
includes a PCA helper, dataset containers with masking/undo support, and an
abstract :class:`ResultData` class centralising structure IO, selection, and
dataset synchronisation.


"""
import ast
import hashlib
import os
import json
import threading
import re
import traceback
from collections import Counter
from functools import cached_property
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from loguru import logger
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
import numpy.typing as npt
from NepTrainKit.core.adapter_api import NepAdaptersError
from NepTrainKit.utils import timeit, parse_index_string
from NepTrainKit.config import Config
from NepTrainKit.core import   MessageManager
from NepTrainKit.core.precision import get_export_significant_digits, get_storage_float_dtype
from NepTrainKit.core.structure import (
    Structure,
    atomic_numbers,
    get_type_map,
    save_npy_structure,
    write_structures_extxyz_atomic,
)
from NepTrainKit.core.geometry_cache import GeometrySnapshot, StructureGeometryCache
from NepTrainKit.core.utils import read_nep_out_file, aggregate_per_atom_to_structure, get_rmse, split_by_natoms

from .sampler import SparseSampler,farthest_point_sampling,pca
from .distribution import (
    DistributionAnalysisMixin,
    DistributionRequest,
    FieldSpec,
)
from NepTrainKit.core.types import (
    Brushes,
    SearchType,
    NepBackend,
    FieldValueShape,
    FieldDomain,
    DistributionGroupMode,
    DistributionValueView,
    DistributionScope,
    DistributionSelectMode,
    DistributionCurveStyle,
)
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
        self._version = 0

    @property
    def version(self) -> int:
        """Monotonically increasing mutation counter for mask changes."""
        return int(self._version)
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
        # Only record indices that actually change state.
        try:
            idx = idx[self._active_mask[idx]]
        except Exception:
            pass
        if idx.size == 0:
            return
        self._history.append(idx)
        self._active_mask[idx] = False
        self._version += 1
    def revoke(self) -> None:
        """Undo the most recent :meth:`remove` call, if any."""
        if self._history:
            last_indices = self._history.pop()
            self._active_mask[last_indices] = True
            self._version += 1
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
        nmax = int(nmax)
        if nmax <= 0:
            return []
        data_version = int(getattr(self.data, "version", 0) or 0)
        cache = getattr(self, "_max_error_cache", None)
        if cache is not None and cache[0] == data_version and cache[1] == self.cols:
            error = cache[2]
        else:
            error = np.sum(np.abs(self.now_data[:, : self.cols] - self.now_data[:, self.cols :]), axis=1)
            self._max_error_cache = (data_version, self.cols, error)
        if error.size == 0:
            return []
        total = int(error.shape[0])
        k = min(total, max(nmax * 4, nmax + 64))
        while True:
            if k >= total:
                sorted_idx = np.argsort(-error)
            else:
                candidate_idx = np.argpartition(-error, k - 1)[:k]
                sorted_idx = candidate_idx[np.argsort(-error[candidate_idx])]
            structure_index = self.group_array.now_data[sorted_idx]
            _, unique_indices = np.unique(structure_index, return_index=True)
            if unique_indices.size >= nmax or k >= total:
                return structure_index[np.sort(unique_indices)][:nmax].tolist()
            k = min(total, k * 2)
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

    _geometry_cache_init_lock = threading.Lock()

    def geometry_snapshot(
        self,
        source_indices: Sequence[int] | npt.NDArray[np.int64] | None = None,
    ) -> GeometrySnapshot:
        """Return cached contiguous geometry for all or selected source rows."""
        cache = getattr(self, "_geometry_cache", None)
        if cache is None:
            with self._geometry_cache_init_lock:
                cache = getattr(self, "_geometry_cache", None)
                if cache is None:
                    cache = StructureGeometryCache(self.all_data)
                    self._geometry_cache = cache
        return cache.snapshot(source_indices)

    def cached_geometry_analysis(self, namespace, key, build):
        """Cache a geometry-derived result for this immutable dataset."""
        self.geometry_snapshot()
        return self._geometry_cache.analysis_result(namespace, key, build)

    def _completer_cache_lock(self) -> threading.Lock:
        lock = getattr(self, "_completer_cache_lock_obj", None)
        if lock is None:
            lock = threading.Lock()
            self._completer_cache_lock_obj = lock
        return lock

    @staticmethod
    def _truncate_counter(data: dict[str, int], max_items: int) -> tuple[dict[str, int], bool]:
        """Return a stable top-N projection of ``data``.

        Parameters
        ----------
        data : dict[str, int]
            Candidate mapping (already aggregated).
        max_items : int
            Maximum number of entries to keep.

        Returns
        -------
        dict[str, int]
            Truncated mapping, sorted by (count desc, key asc).
        bool
            Whether truncation happened.
        """
        if max_items is None or max_items <= 0:
            return {}, bool(data)
        if len(data) <= max_items:
            return data, False
        items = sorted(data.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        return dict(items[:max_items]), True

    def has_completer_cache(self, search_type: SearchType | str | None = None, max_items: int = 50000) -> bool:
        """Return True if a completer cache exists for ``search_type`` and ``max_items``."""
        search_type = self._normalise_search_type(search_type)
        cache = getattr(self, "_completer_cache", None)
        cache_max_items = getattr(self, "_completer_cache_max_items", None)
        cache_version = getattr(self, "_completer_cache_data_version", None)
        current_version = getattr(getattr(self, "data", None), "version", None)
        if cache is None or cache_max_items != int(max_items):
            return False
        if cache_version is None or current_version is None:
            return False
        if int(cache_version) != int(current_version):
            return False
        return search_type in cache

    def ensure_completer_cache(self, max_items: int = 50000) -> None:
        """Build and cache completer mappings for tag/formula/elements.

        Notes
        -----
        - Designed to run in a background thread (e.g. dataset load thread).
        - Results are stored as dict[SearchType, dict[str,int]] and can be fed
          directly into ConfigTypeSearchLineEdit.setCompleterKeyWord(...).
        """
        max_items = int(max_items or 0)
        with self._completer_cache_lock():
            cache = getattr(self, "_completer_cache", None)
            cache_max_items = getattr(self, "_completer_cache_max_items", None)
            cache_version = getattr(self, "_completer_cache_data_version", None)
            element_cache_version = getattr(self, "_element_count_cache_data_version", None)
            current_version = getattr(getattr(self, "data", None), "version", None)
            if (
                cache is not None
                and cache_max_items == max_items
                and cache_version is not None
                and element_cache_version is not None
                and current_version is not None
                and int(cache_version) == int(current_version)
                and int(element_cache_version) == int(current_version)
            ):
                return

            try:
                start_version = int(self.data.version)
            except Exception:
                start_version = -1

            tag_counter: Counter[str] = Counter()
            formula_counter: Counter[str] = Counter()
            elem_counter: Counter[str] = Counter()
            element_counts: dict[str, npt.NDArray[np.int32]] = {}
            active_count = int(self.now_data.shape[0])

            for row, structure in enumerate(self.now_data):
                try:
                    tag = str(getattr(structure, "tag", "") or "").strip()
                    if tag:
                        tag_counter[tag] += 1
                except Exception:
                    pass
                try:
                    formula = str(getattr(structure, "formula", "") or "").strip()
                    if formula:
                        formula_counter[formula] += 1
                except Exception:
                    pass
                try:
                    counts = Counter(self._normalise_element_symbol(str(elem)) for elem in getattr(structure, "elements"))
                except Exception:
                    counts = Counter()
                for elem, count in counts.items():
                    elem = str(elem or "").strip()
                    if elem:
                        elem_counter[elem] += 1
                        values = element_counts.get(elem)
                        if values is None:
                            values = np.zeros(active_count, dtype=np.int32)
                            element_counts[elem] = values
                        values[row] = int(count)

            tag_map, trunc_tag = self._truncate_counter(dict(tag_counter), max_items)
            formula_map, trunc_formula = self._truncate_counter(dict(formula_counter), max_items)
            elem_map, trunc_elem = self._truncate_counter(dict(elem_counter), max_items)

            try:
                end_version = int(self.data.version)
            except Exception:
                end_version = start_version
            if start_version != -1 and end_version != start_version:
                # Underlying mask changed while building; skip caching stale results.
                return

            truncated = bool(trunc_tag or trunc_formula or trunc_elem)
            warned = bool(getattr(self, "_completer_cache_trunc_warned", False))
            if truncated and not warned:
                try:
                    MessageManager.send_info_message(
                        f"Search completer candidates exceed {max_items}; suggestions were truncated."
                    )
                except Exception:
                    pass
                self._completer_cache_trunc_warned = True

            self._completer_cache = {
                SearchType.TAG: tag_map,
                SearchType.FORMULA: formula_map,
                SearchType.ELEMENTS: elem_map,
            }
            self._completer_cache_max_items = max_items
            self._completer_cache_data_version = start_version if start_version != -1 else 0
            self._element_count_cache = element_counts
            self._element_count_cache_data_version = start_version if start_version != -1 else 0

    def get_element_count_cache(self, elements: set[str] | None = None) -> dict[str, npt.NDArray[np.int32]]:
        """Return element count arrays aligned with active structures."""
        current_version = getattr(getattr(self, "data", None), "version", None)
        cache_version = getattr(self, "_element_count_cache_data_version", None)
        cache = getattr(self, "_element_count_cache", None)
        if cache is None or cache_version is None or current_version is None or int(cache_version) != int(current_version):
            self.ensure_completer_cache(max_items=int(getattr(self, "_completer_cache_max_items", 50000) or 50000))
            cache = getattr(self, "_element_count_cache", None) or {}
        if elements is None:
            return {elem: values for elem, values in cache.items()}
        wanted = {self._normalise_element_symbol(elem) for elem in elements if str(elem or "").strip()}
        return {elem: values for elem, values in cache.items() if elem in wanted}

    def get_completer_cache(self, search_type: SearchType | str | None = None, max_items: int = 50000) -> dict[str, int]:
        """Return cached completer mapping for ``search_type``; builds it if needed."""
        search_type = self._normalise_search_type(search_type)
        try:
            self.ensure_completer_cache(max_items=max_items)
        except Exception:
            logger.debug(traceback.format_exc())
            return {}
        cache = getattr(self, "_completer_cache", None) or {}
        return dict(cache.get(search_type, {}))

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
            try:
                pattern = re.compile(config)
            except re.error:
                MessageManager.send_warning_message("Invalid regex pattern.")
                return []
            result_index = [i for i, structure in enumerate(self.now_data) if pattern.search(structure.tag)]
        elif search_type == SearchType.FORMULA:
            try:
                pattern = re.compile(config)
            except re.error:
                MessageManager.send_warning_message("Invalid regex pattern.")
                return []
            result_index = [i for i, structure in enumerate(self.now_data) if pattern.search(structure.formula)]
        elif search_type == SearchType.ELEMENTS:
            allowed, required, excluded = self._parse_elements_query(config)
            element_counts = self.get_element_count_cache()
            active_count = int(self.now_data.shape[0])
            mask = np.ones(active_count, dtype=bool)
            for elem in required:
                values = element_counts.get(elem)
                mask &= values > 0 if values is not None else False
            for elem in excluded:
                values = element_counts.get(elem)
                if values is not None:
                    mask &= values == 0
            if allowed:
                outside = np.zeros(active_count, dtype=bool)
                for elem, values in element_counts.items():
                    if elem not in allowed:
                        outside |= values > 0
                mask &= ~outside
            result_index = np.nonzero(mask)[0]
        return self.group_array[result_index].tolist()

    def search_config_tags(self, filter_spec: dict, search_type: SearchType) -> list[int]:
        """Return structure indices matching a tag/formula filter spec.

        Uses simple substring matching (not regex) with group-based logic:
        groups are AND'd, conditions within a group use AND/OR per group mode.

        Parameters
        ----------
        filter_spec : dict
            Dictionary with ``groups`` (list of group dicts, each having
            ``conditions`` and ``mode`` keys).
        search_type : SearchType
            One of :attr:`SearchType.TAG` or :attr:`SearchType.FORMULA`.

        Returns
        -------
        list[int]
            Matching structure indices.
        """
        from NepTrainKit.core.types import TagFilterSpec

        spec = TagFilterSpec.from_dict(filter_spec) if isinstance(filter_spec, dict) else filter_spec
        if spec.is_empty():
            return []

        if search_type == SearchType.TAG:
            values = [getattr(s, "tag", "") or "" for s in self.now_data]
        elif search_type == SearchType.FORMULA:
            values = [getattr(s, "formula", "") or "" for s in self.now_data]
        else:
            return []

        active_count = len(values)
        mask = np.ones(active_count, dtype=bool)

        for group in spec.groups:
            if group.is_empty():
                continue
            group_mask: np.ndarray | None = None
            for cond in group.conditions:
                text = str(cond.text)
                if not text:
                    continue
                row_match = np.array([text in v for v in values], dtype=bool)
                if cond.negate:
                    row_match = ~row_match
                if group.mode == "and":
                    group_mask = row_match.copy() if group_mask is None else group_mask & row_match
                else:
                    group_mask = row_match if group_mask is None else group_mask | row_match
            if group_mask is not None:
                mask &= group_mask

        result_index = np.nonzero(mask)[0]
        return self.group_array[result_index].tolist()


@dataclass(frozen=True)
class StructureSyncRule:
    """Declarative instruction that synchronises structure attributes into datasets."""
    dataset_attr: str
    target: str | slice | Callable[[Any], Any]
    collector: Callable[["ResultData", Any, Optional[np.ndarray]], tuple[np.ndarray, npt.NDArray[Any]]]
    precondition: Callable[["ResultData"], bool] = lambda _: True
    dtype: Any = None
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
        values = np.asarray(values, dtype=self.dtype or get_storage_float_dtype())
        if values.size == 0:
            return
        target = self._resolve_target(dataset)
        dataset.all_data[row_idx, target] = values
        dataset._plot_coord_version = int(getattr(dataset, "_plot_coord_version", 0) or 0) + 1


def _sync_target_width(dataset: Any) -> int:
    total_cols = dataset.data.all_data.shape[1] if dataset.data.all_data.ndim > 1 else 0
    return max(total_cols - dataset.cols, 0)


def _empty_sync_result(width: int) -> tuple[np.ndarray, npt.NDArray[Any]]:
    return np.array([], dtype=np.int64), np.empty((0, width), dtype=get_storage_float_dtype())


def collect_energy_sync(result_data: "ResultData", dataset: NepPlotData, structure_indices):
    """Collect reference energies for structure-synchronised result datasets."""
    target_width = _sync_target_width(dataset)
    if target_width == 0:
        return _empty_sync_result(0)
    indices = result_data._normalize_structure_indices(structure_indices)
    if indices.size == 0:
        return _empty_sync_result(target_width)
    storage_dtype = get_storage_float_dtype()
    structures = [result_data.structure.all_data[i] for i in indices]
    values = np.array([s.per_atom_energy for s in structures], dtype=storage_dtype).reshape(-1, target_width)
    return indices, values


def collect_force_sync(result_data: "ResultData", dataset: NepPlotData, structure_indices):
    """Collect force values aligned with structure or atom rows."""
    target_width = _sync_target_width(dataset)
    if target_width == 0:
        return _empty_sync_result(0)
    indices = result_data._normalize_structure_indices(structure_indices)
    if indices.size == 0:
        return _empty_sync_result(target_width)
    storage_dtype = get_storage_float_dtype()
    group_vals = dataset.group_array.all_data
    per_atom = bool(group_vals.size and np.unique(group_vals).size != group_vals.size)
    structures = [result_data.structure.all_data[i] for i in indices]
    if per_atom:
        row_idx = dataset.convert_index(indices)
        values = np.vstack([s.forces for s in structures]).astype(storage_dtype, copy=False)
    else:
        row_idx = indices
        values = np.array([np.linalg.norm(s.forces, axis=0) for s in structures], dtype=storage_dtype)
    return row_idx, values


def collect_virial_sync(result_data: "ResultData", dataset: NepPlotData, structure_indices):
    """Collect virial tensors for structures that provide virial information."""
    target_width = _sync_target_width(dataset)
    if target_width == 0:
        return _empty_sync_result(0)
    indices = result_data._normalize_structure_indices(structure_indices)
    if indices.size == 0:
        return _empty_sync_result(target_width)
    storage_dtype = get_storage_float_dtype()
    structures = [result_data.structure.all_data[i] for i in indices]
    mask = np.array([s.has_virial for s in structures], dtype=bool)
    if not mask.any():
        return _empty_sync_result(target_width)
    selected_indices = indices[mask]
    values = np.vstack([structures[i].nep_virial for i, flag in enumerate(mask) if flag]).astype(storage_dtype, copy=False)
    return selected_indices, values


def collect_stress_sync(result_data: "ResultData", dataset: NepPlotData, structure_indices):
    """Collect stress tensors derived from virials for selected structures."""
    target_width = _sync_target_width(dataset)
    if target_width == 0:
        return _empty_sync_result(0)
    indices = result_data._normalize_structure_indices(structure_indices)
    if indices.size == 0:
        return _empty_sync_result(target_width)
    storage_dtype = get_storage_float_dtype()
    structures = [result_data.structure.all_data[i] for i in indices]
    mask = np.array([s.has_virial for s in structures], dtype=bool)
    if not mask.any():
        return _empty_sync_result(target_width)
    selected_indices = indices[mask]
    virial_values = np.vstack([structures[i].nep_virial for i, flag in enumerate(mask) if flag]).astype(storage_dtype, copy=False)
    atoms = result_data.atoms_num_list[selected_indices].astype(storage_dtype)
    volumes = np.array([structures[i].volume for i, flag in enumerate(mask) if flag], dtype=storage_dtype)
    coeff = np.divide(atoms, volumes, out=np.zeros_like(atoms, dtype=storage_dtype), where=volumes != 0)[:, np.newaxis]
    stress_values = virial_values * coeff * 160.21766208
    return selected_indices, stress_values.astype(storage_dtype, copy=False)


class ResultData(DistributionAnalysisMixin, QObject):
    """Manage structures, descriptors, and plots for NEP result files.
    Subclasses implement :meth:`_load_dataset` and expose their plot datasets
    through :py:attr:`datasets`. The class also centralises selection and
    synchronisation utilities shared by the GUI.
    """
    STRUCTURE_SYNC_RULES: dict[str, StructureSyncRule] = {}
    FORCE_CPU_BACKEND = False
    updateInfoSignal = Signal( )
    loadFinishedSignal = Signal()
    predictionStatusSignal = Signal(str)
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
        self._cache_outputs_override: bool | None = None
        self.select_index=set()
        self._selection_history: list[set[int]] = []
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
        self._distribution_cache_key: tuple[Any, ...] | None = None
        self._distribution_analysis: dict[str, Any] = {}
        self._distribution_bin_lookup: dict[tuple[int, str, str, int], list[int]] = {}
        self._distribution_analysis_id: int = 0
        self._load_origin_thread: QThread | None = None

    def move_to_load_thread(self, thread: QThread) -> None:
        """Move this long-lived result object to a loader and remember its owner."""
        origin = self.thread()
        if origin is not QThread.currentThread():
            raise RuntimeError("ResultData must be moved from its current owner thread")
        self._load_origin_thread = origin
        self.moveToThread(thread)

    @Slot()
    def _restore_load_thread_affinity(self) -> None:
        """Return to the original thread before publishing loaded UI state."""
        origin = self._load_origin_thread
        self._load_origin_thread = None
        if origin is None or self.thread() is origin:
            return
        if self.thread() is not QThread.currentThread():
            raise RuntimeError("ResultData affinity can only be restored by its load thread")
        self.moveToThread(origin)

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
    def _result_completer_cache_lock(self) -> threading.Lock:
        lock = getattr(self, "_result_completer_cache_lock_obj", None)
        if lock is None:
            lock = threading.Lock()
            self._result_completer_cache_lock_obj = lock
        return lock
    def _current_structure_version(self) -> int | None:
        try:
            return int(self.structure.data.version)
        except Exception:
            return None
    def has_completer_cache(self, search_type: SearchType | str | None = None, max_items: int = 50000) -> bool:
        """Return True if a completer cache exists for ``search_type`` and ``max_items``."""
        search_type = StructureData._normalise_search_type(search_type)
        if search_type != SearchType.EXPRESSION:
            return self.structure.has_completer_cache(search_type, max_items=max_items)
        cache = getattr(self, "_expression_completer_cache", None)
        cache_max_items = getattr(self, "_expression_completer_cache_max_items", None)
        cache_version = getattr(self, "_expression_completer_cache_version", None)
        current_version = self._current_structure_version()
        return (
            cache is not None
            and cache_max_items == int(max_items)
            and cache_version is not None
            and current_version is not None
            and int(cache_version) == int(current_version)
        )
    def ensure_completer_cache(self, search_type: SearchType | str | None = None, max_items: int = 50000) -> None:
        """Build and cache completer mappings for the requested search type."""
        search_type = StructureData._normalise_search_type(search_type)
        if search_type != SearchType.EXPRESSION:
            self.structure.ensure_completer_cache(max_items=max_items)
            return
        max_items = int(max_items or 0)
        with self._result_completer_cache_lock():
            current_version = self._current_structure_version()
            if (
                getattr(self, "_expression_completer_cache", None) is not None
                and getattr(self, "_expression_completer_cache_max_items", None) == max_items
                and getattr(self, "_expression_completer_cache_version", None) is not None
                and current_version is not None
                and int(getattr(self, "_expression_completer_cache_version")) == int(current_version)
            ):
                return
            cache = self._build_expression_completer_cache(max_items=max_items)
            self._expression_completer_cache = cache
            self._expression_completer_cache_max_items = max_items
            self._expression_completer_cache_version = 0 if current_version is None else int(current_version)
    def get_completer_cache(self, search_type: SearchType | str | None = None, max_items: int = 50000) -> dict[str, int]:
        """Return cached completer mapping for ``search_type``; builds it if needed."""
        search_type = StructureData._normalise_search_type(search_type)
        if search_type != SearchType.EXPRESSION:
            return self.structure.get_completer_cache(search_type, max_items=max_items)
        try:
            self.ensure_completer_cache(search_type, max_items=max_items)
        except Exception:
            logger.debug(traceback.format_exc())
            return {}
        return dict(getattr(self, "_expression_completer_cache", None) or {})
    @staticmethod
    def _expression_alias(text: str) -> str:
        alias = re.sub(r"[^0-9A-Za-z_]+", "_", str(text or "").strip().lower()).strip("_")
        return alias
    @staticmethod
    def _expression_component_aliases(components: Sequence[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []
        for comp in components:
            comp_text = str(comp or "").strip()
            if not comp_text:
                continue
            key = comp_text.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(comp_text)
        return tuple(result)
    @staticmethod
    def _normalise_expression_text(expr: str) -> str:
        text = str(expr or "").strip()
        text = re.sub(r"\bAND\b", " and ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bOR\b", " or ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bNOT\b", " not ", text, flags=re.IGNORECASE)
        text = text.replace("&&", " and ")
        text = text.replace("||", " or ")
        text = re.sub(r"(?<![<>=!])!(?!=)", " not ", text)
        return text.strip()
    @staticmethod
    def _contains_numeric_component_reference(expr: str) -> bool:
        pattern = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\.\d+\b")
        return bool(pattern.search(str(expr or "")))
    @staticmethod
    def _expression_ast_chain(node: ast.AST) -> tuple[str, ...] | None:
        if isinstance(node, ast.Name):
            return (node.id,)
        if isinstance(node, ast.Attribute):
            base = ResultData._expression_ast_chain(node.value)
            if base is None:
                return None
            return (*base, node.attr)
        return None
    @staticmethod
    def _expression_reference_chains(node: ast.AST) -> set[tuple[str, ...]]:
        """Return expression field references without counting nested attributes twice."""
        chain = ResultData._expression_ast_chain(node)
        if chain is not None:
            return {chain}
        refs: set[tuple[str, ...]] = set()
        for child in ast.iter_child_nodes(node):
            refs.update(ResultData._expression_reference_chains(child))
        return refs
    @staticmethod
    def _is_allowed_expression_node(node: ast.AST) -> bool:
        chain = ResultData._expression_ast_chain(node)
        if chain is not None:
            return True
        allowed_nodes = (
            ast.Expression,
            ast.BoolOp,
            ast.Compare,
            ast.Load,
            ast.Constant,
            ast.UnaryOp,
            ast.BinOp,
            ast.And,
            ast.Or,
            ast.Not,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.UAdd,
            ast.USub,
        )
        if not isinstance(node, allowed_nodes):
            return False
        if isinstance(node, ast.BoolOp) and not isinstance(node.op, (ast.And, ast.Or)):
            return False
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            return False
        if isinstance(node, ast.BinOp) and not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return False
        if isinstance(node, ast.Compare):
            if not all(isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
                return False
        return all(ResultData._is_allowed_expression_node(child) for child in ast.iter_child_nodes(node))
    @staticmethod
    def _is_expression_predicate(node: ast.AST) -> bool:
        """Return whether an expression node has explicit boolean semantics."""
        chain = ResultData._expression_ast_chain(node)
        if chain is not None:
            root = chain[0].lower()
            return root in {"has_energy", "has_forces", "has_virial", "has_bec"} or (
                root == "has" and len(chain) == 2
            )
        if isinstance(node, ast.Compare):
            return True
        if isinstance(node, ast.BoolOp):
            return all(ResultData._is_expression_predicate(value) for value in node.values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ResultData._is_expression_predicate(node.operand)
        return False
    def _discover_expression_fields(
        self,
        structure_indices: npt.NDArray[np.int64],
    ) -> tuple[dict[str, tuple[FieldSpec, Any]], dict[str, tuple[FieldSpec, str]]]:
        dataset_specs, dataset_lookup = self._discover_dataset_field_specs()
        dataset_fields: dict[str, tuple[FieldSpec, Any]] = {}
        for spec in dataset_specs:
            alias = self._expression_alias(spec.label or spec.key.split(":", 1)[-1])
            dataset = dataset_lookup.get(spec.key)
            if alias and dataset is not None and alias not in dataset_fields:
                dataset_fields[alias] = (spec, dataset)
        atomic_fields: dict[str, tuple[FieldSpec, str]] = {}
        for spec in self._discover_atomic_property_specs(structure_indices):
            prop_name = spec.key.split(":", 1)[-1]
            alias = self._expression_alias(prop_name)
            if alias and alias not in atomic_fields:
                atomic_fields[alias] = (spec, prop_name)
        return dataset_fields, atomic_fields
    def _build_expression_completer_cache(self, max_items: int = 50000) -> dict[str, int]:
        active_indices = self._normalize_structure_indices(None)
        if active_indices.size == 0:
            return {}
        structures = [self.structure.all_data[int(i)] for i in active_indices.tolist()]
        dataset_fields, atomic_fields = self._discover_expression_fields(active_indices)
        cache: dict[str, int] = {}

        def add_candidate(token: str, weight: int) -> None:
            key = str(token or "").strip()
            if not key:
                return
            cache[key] = int(weight)

        active_count = int(active_indices.shape[0])

        builtin_tokens = (
            "natoms",
            "n_atoms",
            "volume",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
            "spin_natoms",
            "energy",
            "energy_per_atom",
            "has_energy",
            "has_forces",
            "has_virial",
            "has_bec",
        )
        always_available_tokens = {"natoms", "n_atoms", "volume", "a", "b", "c", "alpha", "beta", "gamma", "spin_natoms"}
        builtin_values, element_values = self._build_expression_builtin_values(active_indices)
        for token in builtin_tokens:
            values = np.asarray(builtin_values.get(token, np.array([], dtype=np.float64)))
            if values.dtype == np.bool_:
                count = int(np.count_nonzero(values))
            else:
                count = int(np.count_nonzero(np.isfinite(values)))
            if token in always_available_tokens:
                count = active_count
            add_candidate(token, count)

        for elem, values in element_values.get("count", {}).items():
            count = int(np.count_nonzero(np.asarray(values, dtype=np.float64) > 0))
            add_candidate(f"count.{elem}", count)
            add_candidate(f"frac.{elem}", count)
            add_candidate(f"has.{elem}", count)

        for alias, (spec, _dataset) in dataset_fields.items():
            value_count = 0
            try:
                row_sids = np.asarray(_dataset.group_array.now_data, dtype=np.int64).reshape(-1)
                if row_sids.size > 0:
                    value_count = int(np.unique(row_sids[np.isin(row_sids, active_indices)]).shape[0])
            except Exception:
                value_count = 0
            if value_count <= 0:
                value_count = active_count
            components = list(spec.components)
            if not components:
                components = list(self._component_names(1))
            if len(components) == 1:
                add_candidate(alias, value_count)
            views = ("ref", "pred", "err") if bool(spec.has_prediction_pair) else ("ref",)
            for view in views:
                if len(components) == 1:
                    add_candidate(f"{alias}.{view}", value_count)
                    continue
                for comp in self._expression_component_aliases(components):
                    add_candidate(f"{alias}.{view}.{comp}", value_count)
                add_candidate(f"{alias}.{view}.norm", value_count)
            if len(components) > 1:
                for comp in self._expression_component_aliases(components):
                    add_candidate(f"{alias}.{comp}", value_count)
                add_candidate(f"{alias}.norm", value_count)

        for alias, (spec, _prop_name) in atomic_fields.items():
            value_count = 0
            for structure in structures:
                props = getattr(structure, "atomic_properties", {}) or {}
                if _prop_name in props:
                    value_count += 1
            if value_count <= 0:
                value_count = active_count
            components = list(spec.components)
            if not components:
                components = list(self._component_names(1))
            add_candidate(f"atomic.{alias}", value_count)
            if len(components) == 1:
                continue
            for comp in self._expression_component_aliases(components):
                add_candidate(f"atomic.{alias}.{comp}", value_count)
            add_candidate(f"atomic.{alias}.norm", value_count)

        return dict(StructureData._truncate_counter(cache, max_items)[0])
    def _build_expression_builtin_values(
        self,
        active_indices: npt.NDArray[np.int64],
        references: set[tuple[str, ...]] | None = None,
    ) -> tuple[dict[str, npt.NDArray[Any]], dict[str, dict[str, npt.NDArray[Any]]]]:
        builtin_tokens = {
            "natoms",
            "n_atoms",
            "volume",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
            "spin_natoms",
            "energy",
            "energy_per_atom",
            "has_energy",
            "has_forces",
            "has_virial",
            "has_bec",
        }
        if references is None:
            requested_builtins = set(builtin_tokens)
            requested_elements: set[str] | None = None
        else:
            requested_builtins = {chain[0].lower() for chain in references if chain and chain[0].lower() in builtin_tokens}
            requested_elements = {
                StructureData._normalise_element_symbol(chain[1])
                for chain in references
                if len(chain) == 2 and chain[0].lower() in {"count", "frac", "has"}
            }
        needs_structures = references is None or bool(
            requested_builtins.intersection(
                {"volume", "spin_natoms", "energy", "energy_per_atom", "has_energy", "has_forces", "has_virial", "has_bec"}
            )
            or requested_elements
        )
        structures = [self.structure.all_data[int(i)] for i in active_indices.tolist()] if needs_structures else []
        if {"natoms", "n_atoms"}.intersection(requested_builtins) or requested_elements:
            try:
                natoms = np.asarray(self.atoms_num_list[active_indices], dtype=np.float64)
            except Exception:
                source = structures or [self.structure.all_data[int(i)] for i in active_indices.tolist()]
                natoms = np.array([int(len(s)) for s in source], dtype=np.float64)
        else:
            natoms = np.array([], dtype=np.float64)
        abcs = (
            np.asarray(self.abcs[active_indices], dtype=np.float64)
            if active_indices.size and {"a", "b", "c"}.intersection(requested_builtins)
            else np.empty((0, 3), dtype=np.float64)
        )
        angles = (
            np.asarray(self.angles[active_indices], dtype=np.float64)
            if active_indices.size and {"alpha", "beta", "gamma"}.intersection(requested_builtins)
            else np.empty((0, 3), dtype=np.float64)
        )

        def safe_scalar(getter: Callable[[Structure], float]) -> npt.NDArray[np.float64]:
            values: list[float] = []
            for structure in structures:
                try:
                    values.append(float(getter(structure)))
                except Exception:
                    values.append(float("nan"))
            return np.asarray(values, dtype=np.float64)

        builtin_values: dict[str, npt.NDArray[Any]] = {}
        if "natoms" in requested_builtins:
            builtin_values["natoms"] = natoms
        if "n_atoms" in requested_builtins:
            builtin_values["n_atoms"] = natoms
        if "volume" in requested_builtins:
            builtin_values["volume"] = np.array([float(getattr(s, "volume", np.nan)) for s in structures], dtype=np.float64)
        if "a" in requested_builtins:
            builtin_values["a"] = abcs[:, 0] if abcs.size else np.array([], dtype=np.float64)
        if "b" in requested_builtins:
            builtin_values["b"] = abcs[:, 1] if abcs.size else np.array([], dtype=np.float64)
        if "c" in requested_builtins:
            builtin_values["c"] = abcs[:, 2] if abcs.size else np.array([], dtype=np.float64)
        if "alpha" in requested_builtins:
            builtin_values["alpha"] = angles[:, 0] if angles.size else np.array([], dtype=np.float64)
        if "beta" in requested_builtins:
            builtin_values["beta"] = angles[:, 1] if angles.size else np.array([], dtype=np.float64)
        if "gamma" in requested_builtins:
            builtin_values["gamma"] = angles[:, 2] if angles.size else np.array([], dtype=np.float64)
        if "spin_natoms" in requested_builtins:
            builtin_values["spin_natoms"] = np.array([int(getattr(s, "spin_num", 0) or 0) for s in structures], dtype=np.float64)
        if "energy" in requested_builtins:
            builtin_values["energy"] = safe_scalar(lambda s: s.energy)
        if "energy_per_atom" in requested_builtins:
            builtin_values["energy_per_atom"] = safe_scalar(lambda s: s.per_atom_energy)
        if "has_energy" in requested_builtins:
            builtin_values["has_energy"] = np.array([bool(getattr(s, "has_energy", False)) for s in structures], dtype=bool)
        if "has_forces" in requested_builtins:
            builtin_values["has_forces"] = np.array([bool(getattr(s, "has_forces", False)) for s in structures], dtype=bool)
        if "has_virial" in requested_builtins:
            builtin_values["has_virial"] = np.array([bool(getattr(s, "has_virial", False)) for s in structures], dtype=bool)
        if "has_bec" in requested_builtins:
            builtin_values["has_bec"] = np.array([bool(getattr(s, "has_bec", False)) for s in structures], dtype=bool)
        count_values: dict[str, npt.NDArray[np.float64]] = {}
        frac_values: dict[str, npt.NDArray[np.float64]] = {}
        has_values: dict[str, npt.NDArray[np.bool_]] = {}
        element_set: set[str] = set()
        counters: list[Counter[str]] = []
        cached_element_counts: dict[str, npt.NDArray[np.int32]] | None = None
        if requested_elements is not None and requested_elements:
            try:
                active_now = np.asarray(self.structure.group_array.now_data, dtype=np.int64).reshape(-1)
                if active_now.shape == active_indices.shape and np.array_equal(active_now, active_indices):
                    cached_element_counts = self.structure.get_element_count_cache(requested_elements)
            except Exception:
                cached_element_counts = None
        if cached_element_counts is not None:
            element_set = {e for e in requested_elements if e}
        elif requested_elements is None or requested_elements:
            for structure in structures:
                try:
                    cnt = Counter(StructureData._normalise_element_symbol(str(elem)) for elem in structure.elements)
                except Exception:
                    cnt = Counter()
                counters.append(cnt)
                element_set.update(StructureData._normalise_element_symbol(e) for e in cnt.keys())
        if requested_elements is not None:
            element_set = {e for e in requested_elements if e}
        for elem in sorted(e for e in element_set if e):
            if cached_element_counts is not None:
                counts = np.asarray(cached_element_counts.get(elem, np.zeros(active_indices.shape[0], dtype=np.int32)), dtype=np.float64)
            else:
                counts = np.asarray([float(counter.get(elem, 0)) for counter in counters], dtype=np.float64)
            count_values[elem] = counts
            with np.errstate(divide="ignore", invalid="ignore"):
                frac_values[elem] = np.divide(
                    counts,
                    natoms,
                    out=np.zeros_like(counts, dtype=np.float64),
                    where=natoms > 0,
                )
            has_values[elem] = counts > 0
        return builtin_values, {"count": count_values, "frac": frac_values, "has": has_values}
    @staticmethod
    def _resolve_expression_component(
        token: str | None,
        components: Sequence[str],
        label: str,
    ) -> tuple[str, int | None]:
        if token is None:
            if len(components) == 1:
                return "value", 0
            raise ValueError(f"Field '{label}' requires an explicit component or '.norm'.")
        norm_token = token.lower()
        if norm_token == "norm":
            return "norm", None
        if norm_token.isdigit():
            raise ValueError(f"Numeric component suffixes are not supported for field '{label}'.")
        for idx, comp in enumerate(components):
            if norm_token == str(comp).lower():
                return "component", idx
        raise ValueError(f"Unknown component '{token}' for field '{label}'.")
    @staticmethod
    def _apply_expression_reduction(
        matrix: npt.NDArray[np.float64],
        selector: tuple[str, int | None],
        aggregate_atomwise: bool,
    ) -> float:
        if matrix.size == 0:
            return float("nan")
        mode, comp_index = selector
        if mode == "norm":
            values = np.linalg.norm(matrix, axis=1)
        else:
            assert comp_index is not None
            values = matrix[:, comp_index]
        if values.size == 0:
            return float("nan")
        if aggregate_atomwise:
            return float(np.max(np.abs(values)))
        return float(values[0])
    def _dataset_expression_values(
        self,
        dataset: Any,
        spec: FieldSpec,
        component_token: str | None,
        view: str,
        active_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        try:
            rows = np.asarray(dataset.now_data)
            row_sids = np.asarray(dataset.group_array.now_data, dtype=np.int64).reshape(-1)
        except Exception as exc:
            raise ValueError(f"Failed to read dataset '{spec.label or spec.key}'.") from exc
        if rows.size == 0 or row_sids.size == 0:
            return np.full(active_indices.shape[0], np.nan, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(-1, 1)
        limit = min(rows.shape[0], row_sids.shape[0])
        rows = rows[:limit]
        row_sids = row_sids[:limit]
        scope_mask = np.isin(row_sids, active_indices)
        if not np.any(scope_mask):
            return np.full(active_indices.shape[0], np.nan, dtype=np.float64)
        rows = rows[scope_mask]
        row_sids = row_sids[scope_mask]
        cols = int(getattr(dataset, "cols", 0) or 0)
        if cols > 0:
            ref = np.asarray(rows[:, dataset.x_cols], dtype=np.float64)
            pred = np.asarray(rows[:, dataset.y_cols], dtype=np.float64)
            if ref.ndim == 1:
                ref = ref.reshape(-1, 1)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            if view == "pred":
                values = pred
            elif view == "err":
                values = pred - ref
            else:
                values = ref
        else:
            if view != "ref":
                raise ValueError(f"Field '{spec.label or spec.key}' does not support '.{view}'.")
            values = np.asarray(rows, dtype=np.float64).reshape(rows.shape[0], -1)
        components = list(spec.components) or list(self._component_names(int(values.shape[1]) if values.ndim == 2 else 1))
        selector = self._resolve_expression_component(component_token, components, spec.label or spec.key)
        sid_to_pos = {int(sid): pos for pos, sid in enumerate(active_indices.tolist())}
        grouped: dict[int, list[int]] = {}
        for ridx, sid_raw in enumerate(row_sids.tolist()):
            sid = int(sid_raw)
            if sid in sid_to_pos:
                grouped.setdefault(sid, []).append(ridx)
        out = np.full(active_indices.shape[0], np.nan, dtype=np.float64)
        aggregate_atomwise = spec.domain == FieldDomain.ATOM
        for sid, indices in grouped.items():
            out[sid_to_pos[sid]] = self._apply_expression_reduction(values[indices], selector, aggregate_atomwise)
        return out
    def _atomic_expression_values(
        self,
        prop_name: str,
        spec: FieldSpec,
        component_token: str | None,
        active_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        components = list(spec.components)
        selector = self._resolve_expression_component(component_token, components, f"atomic.{prop_name}")
        out = np.full(active_indices.shape[0], np.nan, dtype=np.float64)
        for pos, sid in enumerate(active_indices.tolist()):
            structure = self.structure.all_data[int(sid)]
            props = getattr(structure, "atomic_properties", {}) or {}
            if prop_name not in props:
                continue
            try:
                arr = np.asarray(props[prop_name], dtype=np.float64)
            except Exception:
                continue
            if arr.size == 0:
                continue
            out[pos] = self._apply_expression_reduction(arr.reshape(arr.shape[0], -1), selector, True)
        return out
    def _resolve_expression_reference(
        self,
        chain: tuple[str, ...],
        builtin_values: Mapping[str, npt.NDArray[Any]],
        element_values: Mapping[str, Mapping[str, npt.NDArray[Any]]],
        dataset_fields: Mapping[str, tuple[FieldSpec, Any]],
        atomic_fields: Mapping[str, tuple[FieldSpec, str]],
        active_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[Any]:
        if not chain:
            raise ValueError("Empty expression reference.")
        parts = [str(raw) for raw in chain]
        base = parts[0].lower()
        if base in {"count", "frac", "has"}:
            if len(parts) != 2:
                raise ValueError(f"'{base}' expects an element symbol like '{base}.Fe'.")
            symbol = StructureData._normalise_element_symbol(parts[1])
            values = element_values.get(base, {}).get(symbol)
            if values is None:
                raise ValueError(f"Unknown element symbol in expression: {parts[1]}")
            return values
        if base == "atomic":
            if len(parts) < 2:
                raise ValueError("Atomic expression fields must use 'atomic.<name>'.")
            alias = self._expression_alias(parts[1])
            entry = atomic_fields.get(alias)
            if entry is None:
                raise ValueError(f"Unknown atomic field: {parts[1]}")
            spec, prop_name = entry
            tail = parts[2:]
            if tail and tail[0].lower() in {"ref", "pred", "err"}:
                raise ValueError(f"Atomic field 'atomic.{alias}' does not support value views.")
            component_token = tail[0] if tail else None
            if len(tail) > 1:
                raise ValueError(f"Unsupported suffix for atomic field 'atomic.{alias}'.")
            return self._atomic_expression_values(prop_name, spec, component_token, active_indices)
        if base in builtin_values:
            if len(parts) == 1:
                return builtin_values[base]
            if base in dataset_fields and parts[1].lower() in {"ref", "pred", "err"}:
                spec, dataset = dataset_fields[base]
                tail = parts[2:]
                component_token = tail[0] if tail else None
                if len(tail) > 1:
                    raise ValueError(f"Unsupported suffix for field '{base}'.")
                return self._dataset_expression_values(dataset, spec, component_token, parts[1].lower(), active_indices)
            raise ValueError(f"Builtin field '{base}' does not support attribute '{parts[1]}'.")
        entry = dataset_fields.get(base)
        if entry is not None:
            spec, dataset = entry
            tail = parts[1:]
            view = "ref"
            if tail and tail[0].lower() in {"ref", "pred", "err"}:
                if not bool(spec.has_prediction_pair):
                    raise ValueError(f"Field '{base}' does not support value views.")
                view = tail[0].lower()
                tail = tail[1:]
            component_token = tail[0] if tail else None
            if len(tail) > 1:
                raise ValueError(f"Unsupported suffix for field '{base}'.")
            return self._dataset_expression_values(dataset, spec, component_token, view, active_indices)
        raise ValueError(f"Unknown field in expression: {parts[0]}")
    def _eval_expression_ast(
        self,
        node: ast.AST,
        builtin_values: Mapping[str, npt.NDArray[Any]],
        element_values: Mapping[str, Mapping[str, npt.NDArray[Any]]],
        dataset_fields: Mapping[str, tuple[FieldSpec, Any]],
        atomic_fields: Mapping[str, tuple[FieldSpec, str]],
        active_indices: npt.NDArray[np.int64],
    ) -> Any:
        chain = self._expression_ast_chain(node)
        if chain is not None:
            return self._resolve_expression_reference(chain, builtin_values, element_values, dataset_fields, atomic_fields, active_indices)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BoolOp):
            values = [
                self._eval_expression_ast(v, builtin_values, element_values, dataset_fields, atomic_fields, active_indices)
                for v in node.values
            ]
            if isinstance(node.op, ast.And):
                result = np.asarray(values[0], dtype=bool)
                for value in values[1:]:
                    result = np.logical_and(result, np.asarray(value, dtype=bool))
                return result
            result = np.asarray(values[0], dtype=bool)
            for value in values[1:]:
                result = np.logical_or(result, np.asarray(value, dtype=bool))
            return result
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_expression_ast(node.operand, builtin_values, element_values, dataset_fields, atomic_fields, active_indices)
            if isinstance(node.op, ast.Not):
                return np.logical_not(np.asarray(operand, dtype=bool))
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
        if isinstance(node, ast.BinOp):
            left = self._eval_expression_ast(node.left, builtin_values, element_values, dataset_fields, atomic_fields, active_indices)
            right = self._eval_expression_ast(node.right, builtin_values, element_values, dataset_fields, atomic_fields, active_indices)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        if isinstance(node, ast.Compare):
            left = self._eval_expression_ast(node.left, builtin_values, element_values, dataset_fields, atomic_fields, active_indices)
            result = None
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_expression_ast(comparator, builtin_values, element_values, dataset_fields, atomic_fields, active_indices)
                if isinstance(op, ast.Eq):
                    current = left == right
                elif isinstance(op, ast.NotEq):
                    current = left != right
                elif isinstance(op, ast.Gt):
                    current = left > right
                elif isinstance(op, ast.GtE):
                    current = left >= right
                elif isinstance(op, ast.Lt):
                    current = left < right
                elif isinstance(op, ast.LtE):
                    current = left <= right
                else:
                    raise ValueError("Unsupported comparison operator.")
                result = current if result is None else np.logical_and(result, current)
                left = right
            return result
        raise ValueError("Unsupported expression syntax.")
    def _search_expression(self, expr: str) -> list[int]:
        text = self._normalise_expression_text(expr)
        if self._contains_numeric_component_reference(text):
            raise ValueError("Numeric component suffixes are not supported in expressions.")
        try:
            parsed = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Invalid expression syntax.") from exc
        if not self._is_allowed_expression_node(parsed):
            raise ValueError("Expression contains unsupported syntax.")
        references = self._expression_reference_chains(parsed.body)
        if not references:
            raise ValueError("Expression must reference at least one structure field.")
        if not self._is_expression_predicate(parsed.body):
            raise ValueError(
                "Expression must be a condition. Add a comparison, for example: natoms > 100."
            )
        active_indices = self._normalize_structure_indices(None)
        if active_indices.size == 0:
            return []
        builtin_tokens = {
            "natoms",
            "n_atoms",
            "volume",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
            "spin_natoms",
            "energy",
            "energy_per_atom",
            "has_energy",
            "has_forces",
            "has_virial",
            "has_bec",
        }
        dynamic_field_needed = any(
            chain
            and chain[0].lower() not in builtin_tokens
            and chain[0].lower() not in {"count", "frac", "has"}
            for chain in references
        )
        builtin_values, element_values = self._build_expression_builtin_values(active_indices, references)
        if dynamic_field_needed:
            dataset_fields, atomic_fields = self._discover_expression_fields(active_indices)
        else:
            dataset_fields, atomic_fields = {}, {}
        result = self._eval_expression_ast(parsed.body, builtin_values, element_values, dataset_fields, atomic_fields, active_indices)
        mask = np.asarray(result, dtype=bool)
        if mask.ndim == 0:
            mask = np.full(active_indices.shape[0], bool(mask.item()), dtype=bool)
        else:
            mask = mask.reshape(-1)
        if mask.shape[0] != active_indices.shape[0]:
            raise ValueError("Expression result shape does not match active structures.")
        return active_indices[mask].astype(int).tolist()
    def search_config(self, config: str, search_type: SearchType) -> list[int]:
        """Return structure indices matching the selected search mode."""
        search_type = StructureData._normalise_search_type(search_type)
        if search_type == SearchType.EXPRESSION:
            return self._search_expression(config)
        return self.structure.search_config(config, search_type)

    def search_config_tags(self, filter_spec: dict, search_type: SearchType) -> list[int]:
        """Return structure indices matching a tag/formula filter spec."""
        return self.structure.search_config_tags(filter_spec, search_type)

    def search_structures(self, filter_spec):
        """Evaluate a typed composite structure filter without changing selection."""
        from NepTrainKit.core.search import StructureFilterEngine

        return StructureFilterEngine.evaluate(self, filter_spec)
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

    def set_cache_outputs_override(self, enabled: bool | None) -> None:
        """Override cache persistence for this result instance only."""
        self._cache_outputs_override = (
            None if enabled is None else bool(enabled)
        )

    def cache_outputs_enabled(self) -> bool:
        """Return whether loader-generated cache files should be written."""
        if self._cache_outputs_override is not None:
            return self._cache_outputs_override
        return bool(Config.getboolean("io", "cache_outputs", True))

    def _can_load_without_calculator(self) -> bool:
        """Return whether this result can be loaded from existing outputs alone."""
        return False

    def _calculation_backend(self) -> NepBackend:
        """Resolve the backend used when this result requires calculations."""
        if self.FORCE_CPU_BACKEND:
            return NepBackend.CPU
        return NepBackend(Config.get("nep", "backend", "auto"))

    def load(self):
        """Load structures, descriptors, and dataset arrays in sequence.
        The routine instantiates a calculator (optionally via ``calculator_factory``),
        parses structures, and then delegates to subclass hooks for descriptors and
        dataset-specific properties.
        """
        try:
            load_from_outputs = self._can_load_without_calculator()
            if load_from_outputs:
                self.nep_calc = None
                if self.descriptor_path.exists():
                    status = (
                        "Loading existing official NEP .out files without opening "
                        "the model."
                    )
                    notify = MessageManager.send_info_message
                else:
                    status = (
                        "Loading existing official NEP .out files without opening "
                        "the model. descriptor.out is missing, so descriptor plots "
                        "and FPS are unavailable. Install a nep-adapters version that "
                        "supports this model to generate descriptors."
                    )
                    notify = MessageManager.send_warning_message
                self.predictionStatusSignal.emit(status)
                notify(status)
            else:
                calculation_backend = self._calculation_backend()
                # Calculator injection (default to NEP). Subclasses can pass in a factory for other ML potentials.
                if self.calculator_factory is None:
                    self.nep_calc = NepCalculator(
                        model_file=self.nep_txt_path.as_posix(),
                        backend=calculation_backend,
                        chunk_max_atoms=Config.getint("nep", "chunk_max_atoms", 100000),
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
                            backend=calculation_backend,
                            chunk_max_atoms=Config.getint("nep", "chunk_max_atoms", 100000),
                        )
                selection = getattr(self.nep_calc, "selection", None)
                if self.FORCE_CPU_BACKEND:
                    MessageManager.send_info_message(
                        "Dipole and polarizability models are CPU-only; "
                        "NepTrainKit will use CPU regardless of the selected NEP backend."
                    )
                elif selection is not None and selection.requested is NepBackend.AUTO:
                    if selection.resolved is NepBackend.CUDA:
                        MessageManager.send_info_message(
                            "NEP Auto selected CUDA acceleration for this model."
                        )
                    else:
                        detail = getattr(selection.cuda_status, "detail", selection.reason)
                        MessageManager.send_warning_message(
                            "NEP Auto selected CPU because CUDA is unavailable "
                            f"({detail}). The calculation will continue on CPU. "
                            "To enable CUDA, install a Linux CPU+CUDA nep-adapters wheel "
                            "with a compatible NVIDIA driver."
                        )
            # If subclass overrides load_structures, defer to it; otherwise do cancel-aware read
            self.load_structures()
            # Pre-build completer caches so UI mode switching remains smooth for large datasets.
            if not self.cancel_event.is_set():
                try:
                    max_items = Config.getint("widget", "completer_max_items", 50000)
                except Exception:
                    max_items = 50000
                try:
                    self.structure.ensure_completer_cache(max_items=max_items)
                except Exception:
                    logger.debug(traceback.format_exc())
            if self._atoms_dataset.num!=0:
                if not self.cancel_event.is_set():
                    self._load_descriptors()
                if not self.cancel_event.is_set():
                    self._load_dataset()
                if not self.cancel_event.is_set():
                    try:
                        self.get_completer_cache(SearchType.EXPRESSION, max_items=max_items)
                    except Exception:
                        logger.debug(traceback.format_exc())
                if not self.cancel_event.is_set():
                    self.load_flag=True
            else:
                MessageManager.send_warning_message("No structures were loaded.")
        except NepAdaptersError as error:
            logger.error(traceback.format_exc())
            message = (
                f"NEP calculation failed [{error.code}]: {error} "
                "Check the selected backend, model type, spin fields, and chunk size."
            )
            self.predictionStatusSignal.emit(message)
            MessageManager.send_error_message(message)
        except Exception as error:
            logger.error(traceback.format_exc())
            message = f"Failed to load dataset: {error}"
            self.predictionStatusSignal.emit(message)
            MessageManager.send_error_message(message)
        try:
            self._restore_load_thread_affinity()
        except RuntimeError:
            logger.error(traceback.format_exc())
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

    def get_reference_per_atom_energy_array(self, use_active: bool = False) -> npt.NDArray[np.float64]:
        """Return reference per-atom energies as a flat float64 array."""
        dataset = getattr(self, "energy", None)
        if dataset is None or getattr(dataset, "cols", 0) == 0:
            return np.array([], dtype=np.float64)
        data = dataset.now_data if use_active else dataset.all_data
        if data.size == 0:
            return np.array([], dtype=np.float64)
        return np.asarray(data[:, dataset.x_cols], dtype=np.float64).reshape(-1)

    def get_predicted_per_atom_energy_array(self, use_active: bool = False) -> npt.NDArray[np.float64]:
        """Return predicted per-atom energies as a flat float64 array."""
        dataset = getattr(self, "energy", None)
        if dataset is None or getattr(dataset, "cols", 0) == 0:
            return np.array([], dtype=np.float64)
        data = dataset.now_data if use_active else dataset.all_data
        if data.size == 0:
            return np.array([], dtype=np.float64)
        return np.asarray(data[:, dataset.y_cols], dtype=np.float64).reshape(-1)

    def is_select(self, i: int) -> bool:
        """Return ``True`` if the structure index is marked as selected."""
        return i in self.select_index
    def _active_selection(self, indices: Iterable[int]) -> set[int]:
        """Return valid active structure indices from ``indices``."""
        active_mask = self.structure.data.mask_array
        total = len(self.structure.all_data)
        selected: set[int] = set()
        for value in indices:
            idx = int(value)
            if 0 <= idx < total and active_mask[idx]:
                selected.add(idx)
        return selected
    def _set_selection(self, selected: set[int], *, record: bool = True) -> bool:
        """Replace the selection, optionally recording one undo step."""
        selected = self._active_selection(selected)
        if selected == self.select_index:
            return False
        if record:
            self._selection_history.append(set(self.select_index))
        self.select_index.clear()
        self.select_index.update(selected)
        self.updateInfoSignal.emit()
        return True
    @property
    def can_undo_selection(self) -> bool:
        """Return whether a previous selection state is available."""
        return bool(self._selection_history)
    def clear_selection_history(self) -> None:
        """Drop stored selection undo states."""
        self._selection_history.clear()
    def undo_selection(self) -> bool:
        """Restore the previous selection state."""
        while self._selection_history:
            selected = self._selection_history.pop()
            if self._set_selection(selected, record=False):
                return True
        return False
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
        selected = set(self.select_index)
        selected.update(idx.tolist())
        self._set_selection(selected)
    def uncheck(self, indices: Sequence[int] | int) -> None:
        """Remove structures denoted by ``indices`` from the selection set."""
        if isinstance(indices, (int, np.integer)):
            iter_indices = [int(indices)]
        else:
            iter_indices = (int(i) for i in np.asarray(indices).ravel())
        selected = set(self.select_index)
        for idx in iter_indices:
            selected.discard(idx)
        self._set_selection(selected)
    def inverse_select(self) -> None:
        """Invert the current selection over the active structure set."""
        active_indices = set(self.structure.data.now_indices.tolist())
        self._set_selection(active_indices - set(self.select_index))

    def apply_selection(self, indices: Iterable[int], mode: str) -> bool:
        """Apply one cached result to selection as a single undoable change."""
        matched = self._active_selection(indices)
        current = set(self.select_index)
        if mode == "replace":
            target = matched
        elif mode == "add":
            target = current | matched
        elif mode == "remove":
            target = current - matched
        elif mode == "clear":
            target = set()
        else:
            raise ValueError(f"Unsupported selection mode: {mode}")
        return self._set_selection(target)
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
            atomic_float_digits = get_export_significant_digits()
            mapped = self.structure.convert_index(indices)
            write_structures_extxyz_atomic(
                save_file_path,
                self.structure.all_data[mapped],
                atomic_float_digits=atomic_float_digits,
            )
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
            atomic_float_digits = get_export_significant_digits()
            active = self.structure.now_data
            if getattr(active, "size", 0) == 0:
                MessageManager.send_info_message("No active structures to export.")
                return
            write_structures_extxyz_atomic(
                save_file_path,
                active,
                atomic_float_digits=atomic_float_digits,
            )
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
            atomic_float_digits = get_export_significant_digits()
            removed = self.structure.remove_data
            if getattr(removed, "size", 0) == 0:
                MessageManager.send_info_message("No removed structures to export.")
                return
            write_structures_extxyz_atomic(
                save_file_path,
                removed,
                atomic_float_digits=atomic_float_digits,
            )
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
            atomic_float_digits = get_export_significant_digits()
            good_path = Path(save_path).joinpath("export_good_model.xyz")
            write_structures_extxyz_atomic(
                good_path,
                self.structure.now_data,
                atomic_float_digits=atomic_float_digits,
            )
            removed_path = Path(save_path).joinpath("export_remove_model.xyz")
            write_structures_extxyz_atomic(
                removed_path,
                self.structure.remove_data,
                atomic_float_digits=atomic_float_digits,
            )
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
            atomic_float_digits = get_export_significant_digits()
            good_path = Path(save_path).joinpath("export_good_model.xyz")
            write_structures_extxyz_atomic(
                good_path,
                self.structure.now_data,
                atomic_float_digits=atomic_float_digits,
            )
            removed_path = Path(save_path).joinpath("export_remove_model.xyz")
            write_structures_extxyz_atomic(
                removed_path,
                self.structure.remove_data,
                atomic_float_digits=atomic_float_digits,
            )
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
        self._selection_history.clear()
        self.updateInfoSignal.emit()
    def iter_non_physical_structure_indices(self, radius_coefficient: float):
        """Yield progress increments while collecting non-physical structures."""
        from NepTrainKit.core.audit.neighbor_scan import (
            find_scaled_radii_collision_structure_indices,
        )

        active_indices = self.structure.now_indices
        geometry = self.structure.geometry_snapshot(active_indices)
        pending = list(
            find_scaled_radii_collision_structure_indices(
                geometry,
                float(radius_coefficient),
            )
        )
        for _ in active_indices:
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
        selection_strategy: str = "global",
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
            selection_strategy=selection_strategy,
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
                if new_tag in {"energy", "energy_original"} and isinstance(value, (float, int, np.number)):
                    value = float(np.float64(value))
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
        nep_energy_array = self.get_predicted_per_atom_energy_array(use_active=True)

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
        MessageManager.send_info_message(
            "DFT-D3 calculations are CPU-only; NepTrainKit will use CPU "
            "regardless of the selected NEP backend."
        )
        nep_calc = NepCalculator(
            model_file=self.nep_txt_path.as_posix(),
            backend=NepBackend.CPU,
            chunk_max_atoms=Config.getint("nep", "chunk_max_atoms", 100000),
        )

        prediction = nep_calc.predict_dftd3(
            self.structure.now_data.tolist(),
            functional=functional,
            cutoff=cutoff,
            cutoff_cn=cutoff_cn,
        )
        potentials = prediction.energy
        forces = prediction.force_blocks()
        virials = prediction.structure_virials

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
            if getattr(self, "nep_calc", None) is None:
                self._descriptor_raw_all = np.array([], dtype=np.float32)
                self._descriptor_dataset = NepPlotData(
                    [],
                    title="descriptor",
                    parity_mode=False,
                    show_rmse=False,
                )
                return
            desc_array = self._generate_missing_descriptors()
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
            reduced = self._load_or_compute_descriptor_pca(reduced)
        self._descriptor_dataset = NepPlotData(
            reduced,
            title="descriptor",
            parity_mode=False,
            show_rmse=False,
        )

    def _generate_missing_descriptors(self) -> npt.NDArray[np.float64]:
        """Generate descriptors when no usable descriptor cache exists."""
        return self.nep_calc.descriptors(
            self.structure.now_data.tolist(),
            progress=lambda done, total: self.predictionStatusSignal.emit(
                self.tr(
                    "Generating NEP descriptors: {done}/{total} structures"
                ).format(done=done, total=total)
            ),
        )

    def _descriptor_pca_cache_paths(self) -> tuple[Path, Path]:
        cache_path = self.descriptor_path.with_suffix(".pca2.npy")
        meta_path = self.descriptor_path.with_suffix(".pca2.json")
        return cache_path, meta_path

    def _descriptor_pca_cache_metadata(self, desc_array: npt.NDArray[Any]) -> dict[str, Any] | None:
        try:
            stat = self.descriptor_path.stat()
        except OSError:
            return None
        atoms = np.asarray(getattr(self, "atoms_num_list", []), dtype=np.int64)
        return {
            "descriptor_path": self.descriptor_path.name,
            "descriptor_size": int(stat.st_size),
            "descriptor_mtime_ns": int(stat.st_mtime_ns),
            "descriptor_shape": [int(v) for v in np.asarray(desc_array).shape],
            "atoms_num_hash": hashlib.sha256(atoms.tobytes()).hexdigest(),
        }

    def _load_or_compute_descriptor_pca(self, desc_array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        metadata = self._descriptor_pca_cache_metadata(desc_array)
        cache_path, meta_path = self._descriptor_pca_cache_paths()
        if metadata is not None and cache_path.exists() and meta_path.exists():
            try:
                cached_meta = json.loads(meta_path.read_text(encoding="utf8"))
                if cached_meta == metadata:
                    cached = np.load(cache_path)
                    if cached.shape[0] == desc_array.shape[0] and cached.shape[1] == 2:
                        return np.asarray(cached, dtype=np.float32)
            except Exception:
                logger.debug(traceback.format_exc())

        try:
            reduced = pca(desc_array, 2)
        except Exception:
            MessageManager.send_error_message("PCA dimensionality reduction fails")
            return np.array([], dtype=np.float32)

        if metadata is not None and self.cache_outputs_enabled():
            try:
                np.save(cache_path, np.asarray(reduced, dtype=np.float32))
                meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")
            except Exception:
                logger.debug(traceback.format_exc())
        return reduced
    def __repr__(self):
        info = f"{self.__class__.__name__}(Orig: {self.atoms_num_list.shape[0]} Now: {self.structure.now_data.shape[0]} " \
               f"Rm: {self.structure.remove_data.shape[0]} Sel: {len(self.select_index)} Unsel: {self.structure.now_data.shape[0] - len(self.select_index)})"
        return info
