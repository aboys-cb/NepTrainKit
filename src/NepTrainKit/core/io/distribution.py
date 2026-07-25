#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Numeric-field discovery and distribution analysis for result datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from NepTrainKit.config import Config
from NepTrainKit.core.types import (
    DistributionCurveStyle,
    DistributionGroupMode,
    DistributionScope,
    DistributionSelectMode,
    DistributionValueView,
    FieldDomain,
    FieldValueShape,
)


def _coerce_enum(value: Any, enum_cls: Any, default: Any):
    """Best-effort conversion to an enum value with a fallback."""
    if isinstance(value, enum_cls):
        return value
    text = str(value or "").strip()
    if not text:
        return default
    if text.startswith(f"{enum_cls.__name__}.") and "." in text:
        text = text.split(".")[-1]
        try:
            return enum_cls[text]
        except Exception:
            pass
    try:
        return enum_cls(text)
    except Exception:
        pass
    lower = text.lower()
    for item in enum_cls:
        if lower in {str(item.value).lower(), str(item.name).lower()}:
            return item
    return default


@dataclass(frozen=True)
class FieldSpec:
    """Description of a discoverable numeric field for distribution analysis."""

    key: str
    source: str
    shape: FieldValueShape
    components: tuple[str, ...]
    has_prediction_pair: bool
    unit_guess: str = "unknown"
    domain: FieldDomain = FieldDomain.ATOM
    label: str = ""


@dataclass(frozen=True)
class DistributionRequest:
    """Serializable request object for distribution analysis jobs."""

    field_keys: tuple[str, ...] = ()
    include_norm: bool = True
    value_view: DistributionValueView = DistributionValueView.REFERENCE
    group_mode: DistributionGroupMode = DistributionGroupMode.FORMULA
    scope: DistributionScope = DistributionScope.ACTIVE
    bins: int = 120
    select_mode: DistributionSelectMode = DistributionSelectMode.REPLACE
    groups: tuple[str, ...] = ()
    curve_style: DistributionCurveStyle = DistributionCurveStyle.KDE
    curve_points: int = 240
    selected_value_views: tuple[str, ...] = ()
    custom_group_specs: tuple[dict[str, Any], ...] = ()

    @classmethod
    def from_any(cls, request: Any) -> "DistributionRequest":
        """Create a normalized request from a dataclass or mapping."""
        if isinstance(request, cls):
            bins = int(max(2, min(5000, int(request.bins or 120))))
            curve_points = int(max(64, min(1024, int(request.curve_points or 240))))
            return cls(
                field_keys=tuple(str(i) for i in request.field_keys),
                include_norm=bool(request.include_norm),
                value_view=_coerce_enum(request.value_view, DistributionValueView, DistributionValueView.REFERENCE),
                group_mode=_coerce_enum(request.group_mode, DistributionGroupMode, DistributionGroupMode.FORMULA),
                scope=_coerce_enum(request.scope, DistributionScope, DistributionScope.ACTIVE),
                bins=bins,
                select_mode=_coerce_enum(request.select_mode, DistributionSelectMode, DistributionSelectMode.REPLACE),
                groups=tuple(str(i) for i in request.groups if str(i)),
                curve_style=_coerce_enum(
                    request.curve_style, DistributionCurveStyle, DistributionCurveStyle.KDE
                ),
                curve_points=curve_points,
                selected_value_views=tuple(str(i) for i in getattr(request, "selected_value_views", ()) if str(i)),
                custom_group_specs=tuple(
                    dict(i) if isinstance(i, dict) else dict(i) if hasattr(i, "items") else {}
                    for i in getattr(request, "custom_group_specs", ())
                ),
            )
        if isinstance(request, Mapping):
            field_keys = request.get("field_keys", ())
            groups = request.get("groups", ())
            if isinstance(field_keys, str):
                field_keys = [field_keys]
            if isinstance(groups, str):
                groups = [groups]
            bins = int(max(2, min(5000, int(request.get("bins", 120) or 120))))
            curve_points = int(max(64, min(1024, int(request.get("curve_points", 240) or 240))))
            raw_views = request.get("selected_value_views", ())
            if isinstance(raw_views, str):
                raw_views = [raw_views]
            raw_specs = request.get("custom_group_specs", ())
            if isinstance(raw_specs, dict):
                raw_specs = [raw_specs]
            return cls(
                field_keys=tuple(str(i) for i in field_keys if str(i)),
                include_norm=bool(request.get("include_norm", True)),
                value_view=_coerce_enum(
                    request.get("value_view", DistributionValueView.REFERENCE),
                    DistributionValueView,
                    DistributionValueView.REFERENCE,
                ),
                group_mode=_coerce_enum(
                    request.get("group_mode", DistributionGroupMode.FORMULA),
                    DistributionGroupMode,
                    DistributionGroupMode.FORMULA,
                ),
                scope=_coerce_enum(
                    request.get("scope", DistributionScope.ACTIVE),
                    DistributionScope,
                    DistributionScope.ACTIVE,
                ),
                bins=bins,
                select_mode=_coerce_enum(
                    request.get("select_mode", DistributionSelectMode.REPLACE),
                    DistributionSelectMode,
                    DistributionSelectMode.REPLACE,
                ),
                groups=tuple(str(i) for i in groups if str(i)),
                curve_style=_coerce_enum(
                    request.get("curve_style", DistributionCurveStyle.KDE),
                    DistributionCurveStyle,
                    DistributionCurveStyle.KDE,
                ),
                curve_points=curve_points,
                selected_value_views=tuple(str(i) for i in raw_views if str(i)),
                custom_group_specs=tuple(dict(i) if isinstance(i, dict) else dict(i) if hasattr(i, "items") else {} for i in raw_specs),
            )
        return cls()


class DistributionAnalysisMixin:
    """Distribution-analysis responsibility shared by result loaders."""

    @staticmethod
    def _guess_field_unit(*, source: str, title: str = "", key: str = "") -> str:
        """Best-effort unit guess for UI display."""
        if source == "dataset":
            t = str(title or "").lower()
            if t == "energy":
                return "eV/atom"
            if "force" in t:
                return "eV/A"
            if t == "virial":
                return "eV/atom"
            if t == "stress":
                return "GPa"
            if t == "bec":
                return "e"
            if t == "dipole":
                return "e*A/atom"
        _ = key
        return "unknown"

    @staticmethod
    def _component_names(n_comp: int) -> tuple[str, ...]:
        """Return component names for flattened numeric vectors."""
        n_comp = int(max(1, n_comp))
        if n_comp == 1:
            return ("value",)
        if n_comp == 3:
            return ("x", "y", "z")
        if n_comp == 6:
            return ("xx", "yy", "zz", "xy", "yz", "zx")
        return tuple(f"c{i}" for i in range(n_comp))

    @staticmethod
    def _classify_field_shape(n_comp: int, rank: int) -> FieldValueShape:
        """Classify a flattened numeric field by shape and rank."""
        n_comp = int(max(1, n_comp))
        rank = int(max(1, rank))
        if n_comp == 1:
            return FieldValueShape.SCALAR
        if n_comp == 3:
            return FieldValueShape.VECTOR3
        if rank > 2:
            return FieldValueShape.TENSOR
        return FieldValueShape.VECTORN

    def _iter_scope_structure_indices(self, scope: DistributionScope) -> npt.NDArray[np.int64]:
        """Return active/selected structure indices according to ``scope``."""
        active = np.asarray(self.structure.now_indices, dtype=np.int64).reshape(-1)
        if scope != DistributionScope.SELECTED:
            return active
        selected = np.asarray(sorted(int(i) for i in self.select_index), dtype=np.int64).reshape(-1)
        if selected.size == 0:
            return np.array([], dtype=np.int64)
        return np.intersect1d(active, selected, assume_unique=False)

    def _discover_dataset_field_specs(self) -> tuple[list[FieldSpec], dict[str, Any]]:
        """Return field specs backed by result datasets and a key->dataset map."""
        specs: list[FieldSpec] = []
        lookup: dict[str, Any] = {}
        used_keys: set[str] = set()

        for dataset in getattr(self, "datasets", []):
            try:
                title = str(getattr(dataset, "title", "") or "").strip()
            except Exception:
                title = ""
            if not title or title == "descriptor":
                continue
            now_data = getattr(dataset, "now_data", None)
            if now_data is None or getattr(now_data, "size", 0) == 0:
                continue
            cols = int(getattr(dataset, "cols", 0) or 0)
            if cols <= 0:
                continue

            try:
                groups = np.asarray(dataset.group_array.now_data, dtype=np.int64).reshape(-1)
                per_atom = bool(groups.size and np.unique(groups).size != groups.size)
            except Exception:
                per_atom = False

            base_key = f"dataset:{title}"
            key = base_key
            suffix = 2
            while key in used_keys:
                key = f"{base_key}#{suffix}"
                suffix += 1
            used_keys.add(key)

            components = self._component_names(cols)
            shape = self._classify_field_shape(cols, rank=2)
            domain = FieldDomain.ATOM if per_atom else FieldDomain.STRUCTURE
            spec = FieldSpec(
                key=key,
                source="dataset",
                shape=shape,
                components=components,
                has_prediction_pair=True,
                unit_guess=self._guess_field_unit(source="dataset", title=title),
                domain=domain,
                label=title,
            )
            specs.append(spec)
            lookup[key] = dataset
        return specs, lookup

    def _discover_atomic_property_specs(self, structure_indices: npt.NDArray[np.int64]) -> list[FieldSpec]:
        """Return discoverable numeric atomic-property field specs."""
        if structure_indices.size == 0:
            return []

        blacklist = {"species", "species_id", "pos"}
        stats: dict[str, dict[str, Any]] = {}

        for idx in structure_indices.tolist():
            structure = self.structure.all_data[int(idx)]
            props = getattr(structure, "atomic_properties", {}) or {}
            n_atoms = int(len(structure))
            for prop_key, prop_val in props.items():
                name = str(prop_key)
                if name in blacklist:
                    continue
                try:
                    arr = np.asarray(prop_val)
                except Exception:
                    continue
                if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
                    continue
                if arr.ndim == 0:
                    continue
                if int(arr.shape[0]) != n_atoms:
                    continue

                n_comp = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
                entry = stats.setdefault(
                    name,
                    {
                        "n_comp": n_comp,
                        "max_rank": int(arr.ndim),
                        "compatible": True,
                    },
                )
                if int(entry["n_comp"]) != n_comp:
                    entry["compatible"] = False
                entry["max_rank"] = max(int(entry["max_rank"]), int(arr.ndim))

        specs: list[FieldSpec] = []
        for prop_key in sorted(stats.keys()):
            entry = stats[prop_key]
            if not bool(entry.get("compatible", False)):
                continue
            n_comp = int(entry.get("n_comp", 1) or 1)
            rank = int(entry.get("max_rank", 1) or 1)
            components = self._component_names(n_comp)
            shape = self._classify_field_shape(n_comp, rank)
            specs.append(
                FieldSpec(
                    key=f"atomic:{prop_key}",
                    source="atomic_property",
                    shape=shape,
                    components=components,
                    has_prediction_pair=False,
                    unit_guess=self._guess_field_unit(source="atomic_property", key=prop_key),
                    domain=FieldDomain.ATOM,
                    label=prop_key,
                )
            )
        return specs

    def discover_atomic_numeric_fields(self, scope: str | DistributionScope = DistributionScope.ACTIVE) -> list[FieldSpec]:
        """Discover available numeric fields for distribution analysis.

        Parameters
        ----------
        scope : str | DistributionScope, default="active"
            Structure subset to inspect. ``selected`` inspects only selected
            active structures.
        """
        scope_enum = _coerce_enum(scope, DistributionScope, DistributionScope.ACTIVE)
        structure_indices = self._iter_scope_structure_indices(scope_enum)
        dataset_specs, _ = self._discover_dataset_field_specs()
        atomic_specs = self._discover_atomic_property_specs(structure_indices)
        return [*dataset_specs, *atomic_specs]

    def _dataset_row_element_map(self, dataset: Any) -> npt.NDArray[np.object_] | None:
        """Map each dataset row to an element symbol when possible.

        Returns ``None`` when the dataset row cardinality does not match per-atom
        element counts (e.g. sparse spin-only outputs).
        """
        try:
            row_groups = np.asarray(dataset.group_array.now_data, dtype=np.int64).reshape(-1)
        except Exception:
            return None
        if row_groups.size == 0:
            return np.empty((0,), dtype=object)

        elems_by_sid: dict[int, npt.NDArray[np.object_]] = {}
        counts_by_sid: dict[int, int] = {}
        unique_sid, sid_counts = np.unique(row_groups, return_counts=True)
        for sid, sid_count in zip(unique_sid.tolist(), sid_counts.tolist()):
            try:
                elems = np.asarray(self.structure.all_data[int(sid)].elements, dtype=object).reshape(-1)
            except Exception:
                return None
            if elems.size != int(sid_count):
                return None
            elems_by_sid[int(sid)] = elems
            counts_by_sid[int(sid)] = 0

        out = np.empty((row_groups.size,), dtype=object)
        for i, sid_raw in enumerate(row_groups.tolist()):
            sid = int(sid_raw)
            elems = elems_by_sid.get(sid)
            if elems is None:
                return None
            pos = int(counts_by_sid.get(sid, 0))
            if pos < 0 or pos >= elems.size:
                return None
            out[i] = str(elems[pos])
            counts_by_sid[sid] = pos + 1
        return out

    @staticmethod
    def _normal_pdf(x: npt.NDArray[np.float64], mean: float, std: float) -> npt.NDArray[np.float64]:
        """Evaluate the normal PDF with safeguards for invalid ``std``."""
        sigma = float(std)
        if not np.isfinite(sigma) or sigma <= 0.0:
            return np.zeros_like(x, dtype=np.float64)
        z = (x - float(mean)) / sigma
        coeff = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        return coeff * np.exp(-0.5 * np.square(z))

    def _build_distribution_curve(
        self,
        vals: npt.NDArray[np.float64],
        lo: float,
        hi: float,
        bins: int,
        style: DistributionCurveStyle,
        points: int,
    ) -> tuple[str, list[float], list[float], str | None]:
        """Build an optional curve overlay scaled to histogram counts."""
        style_norm = _coerce_enum(style, DistributionCurveStyle, DistributionCurveStyle.KDE)
        if style_norm == DistributionCurveStyle.NONE:
            return style_norm.value, [], [], None
        if vals.size < 2:
            return "none", [], [], "Curve overlay requires at least 2 samples."
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return "none", [], [], "Invalid histogram range; curve overlay disabled."

        bins_norm = int(max(1, bins))
        points_norm = int(max(64, min(1024, points)))
        x = np.linspace(lo, hi, points_norm, dtype=np.float64)
        bin_width = float(hi - lo) / float(bins_norm)
        if not np.isfinite(bin_width) or bin_width <= 0.0:
            return "none", [], [], "Invalid histogram width; curve overlay disabled."

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if not np.isfinite(std) or std <= 1e-15:
            return "none", [], [], "Series variance is too small for curve fitting."

        if style_norm == DistributionCurveStyle.NORMAL:
            dens = self._normal_pdf(x, mean=mean, std=std)
            dens = np.where(np.isfinite(dens), dens, 0.0)
            y = np.maximum(0.0, dens) * float(vals.size) * bin_width
            return style_norm.value, x.tolist(), y.tolist(), None

        # KDE path (default)
        sample = vals
        if sample.size > 50000:
            rng = np.random.default_rng(0)
            sample = rng.choice(sample, size=50000, replace=False)
        try:
            from scipy.stats import gaussian_kde
        except Exception:
            return "none", [], [], "SciPy KDE is unavailable; curve overlay disabled."
        try:
            kde = gaussian_kde(sample)
            dens = np.asarray(kde.evaluate(x), dtype=np.float64)
        except Exception:
            return "none", [], [], "KDE fitting failed; curve overlay disabled."
        dens = np.where(np.isfinite(dens), dens, 0.0)
        y = np.maximum(0.0, dens) * float(vals.size) * bin_width
        return DistributionCurveStyle.KDE.value, x.tolist(), y.tolist(), None

    @staticmethod
    def _histogram_bin_indices(
        vals: npt.NDArray[np.float64],
        lo: float,
        hi: float,
        bins: int,
    ) -> npt.NDArray[np.int64]:
        """Map values to histogram bins using ``numpy.histogram`` edge semantics."""
        bins_norm = int(max(1, bins))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return np.zeros((vals.size,), dtype=np.int64)
        edges = np.linspace(lo, hi, bins_norm + 1, dtype=np.float64)
        idx = np.searchsorted(edges, vals, side="right") - 1
        return np.clip(idx.astype(np.int64, copy=False), 0, bins_norm - 1)

    def iter_distribution_analysis(self, request: DistributionRequest | Mapping[str, Any] | None = None):
        """Build distribution statistics for selected numeric fields.

        Notes
        -----
        Results are cached by ``(structure.version, request, force_mode)`` and
        stored for retrieval via :meth:`get_distribution_analysis`.
        """
        req = DistributionRequest.from_any(request or {})
        structure_version = int(getattr(self.structure.data, "version", 0) or 0)
        force_mode = str(Config.get("widget", "forces_data", "Raw"))
        scope_indices = self._iter_scope_structure_indices(req.scope)

        dataset_specs, dataset_lookup = self._discover_dataset_field_specs()
        atomic_specs = self._discover_atomic_property_specs(scope_indices)
        all_specs = [*dataset_specs, *atomic_specs]
        spec_lookup = {spec.key: spec for spec in all_specs}

        selected_field_keys = tuple(k for k in req.field_keys if k in spec_lookup)
        if not selected_field_keys and all_specs:
            selected_field_keys = (all_specs[0].key,)

        req_norm = DistributionRequest(
            field_keys=selected_field_keys,
            include_norm=bool(req.include_norm),
            value_view=req.value_view,
            group_mode=req.group_mode,
            scope=req.scope,
            bins=int(max(2, min(5000, req.bins))),
            select_mode=req.select_mode,
            groups=tuple(str(i) for i in req.groups if str(i)),
            curve_style=req.curve_style,
            curve_points=int(max(64, min(1024, req.curve_points))),
            selected_value_views=tuple(str(i) for i in req.selected_value_views if str(i)),
            custom_group_specs=tuple(dict(i) if isinstance(i, dict) else {} for i in req.custom_group_specs),
        )
        scope_key = tuple(int(i) for i in scope_indices.tolist()) if req_norm.scope == DistributionScope.SELECTED else ()
        cache_key = (structure_version, req_norm, force_mode, scope_key)
        if cache_key == self._distribution_cache_key:
            return

        self._distribution_analysis_id += 1
        analysis_id = int(self._distribution_analysis_id)
        groups_filter = set(req_norm.groups) if req_norm.groups else None
        bins = int(req_norm.bins)

        formula_map: dict[int, str] = {}
        elem_set_map: dict[int, set[str]] = {}
        elem_array_map: dict[int, npt.NDArray[np.object_]] = {}
        for sid in scope_indices.tolist():
            try:
                structure = self.structure.all_data[int(sid)]
            except Exception:
                continue
            formula_map[int(sid)] = str(getattr(structure, "formula", "") or "")
            try:
                elems = np.asarray(structure.elements, dtype=object).reshape(-1)
            except Exception:
                elems = np.empty((0,), dtype=object)
            elem_array_map[int(sid)] = elems
            elem_set_map[int(sid)] = set(str(e) for e in elems.tolist())

        # Custom group evaluation: map structure_id -> list of group labels
        custom_group_map: dict[int, list[str]] = {}
        custom_group_labels: list[str] = []
        if req_norm.group_mode == DistributionGroupMode.CUSTOM and req_norm.custom_group_specs:
            from NepTrainKit.core.search import StructureFilterEngine, StructureFilterValidationError
            from NepTrainKit.core.types import StructureFilterSpec

            for spec_data in req_norm.custom_group_specs:
                label = str(spec_data.get("label", "") or "").strip()
                if not label:
                    continue
                custom_group_labels.append(label)
                try:
                    spec = StructureFilterSpec.from_dict(spec_data.get("spec", {}))
                except Exception:
                    continue
                if spec.is_empty():
                    continue
                try:
                    result = StructureFilterEngine.evaluate(self, spec)
                    matched_sids = set(result.indices)
                except StructureFilterValidationError:
                    continue
                for sid in matched_sids:
                    sid_int = int(sid)
                    if sid_int in formula_map or sid_int in elem_set_map:
                        custom_group_map.setdefault(sid_int, []).append(label)

        # Value view group evaluation: which value views to include
        selected_views: list[DistributionValueView] = []
        if req_norm.group_mode == DistributionGroupMode.VALUE_VIEW:
            for v in req_norm.selected_value_views:
                try:
                    selected_views.append(DistributionValueView(v))
                except ValueError:
                    pass
            if not selected_views:
                selected_views = [DistributionValueView.REFERENCE]
        elif req_norm.group_mode == DistributionGroupMode.CUSTOM:
            for v in req_norm.selected_value_views:
                try:
                    selected_views.append(DistributionValueView(v))
                except ValueError:
                    pass
            # CUSTOM: if no views selected, leave empty -> group by custom label

        metric_values: dict[str, dict[str, list[float]]] = {}
        metric_structs: dict[str, dict[str, list[int]]] = {}
        metric_meta: dict[str, dict[str, Any]] = {}
        messages: list[str] = []
        warned_view_downgrade = False

        def _append_sample(metric_key: str, series_key: str, value: float, sid: int, meta: dict[str, Any]) -> None:
            if not np.isfinite(value):
                return
            s_key = str(series_key or "").strip()
            if not s_key:
                return
            if groups_filter is not None and s_key not in groups_filter:
                return
            metric_values.setdefault(metric_key, {}).setdefault(s_key, []).append(float(value))
            metric_structs.setdefault(metric_key, {}).setdefault(s_key, []).append(int(sid))
            if metric_key not in metric_meta:
                metric_meta[metric_key] = dict(meta)

        for field_key in req_norm.field_keys:
            spec = spec_lookup.get(field_key)
            if spec is None:
                continue

            if spec.source == "dataset":
                dataset = dataset_lookup.get(field_key)
                if dataset is None:
                    continue
                try:
                    data_all = np.asarray(dataset.now_data)
                    row_struct_all = np.asarray(dataset.group_array.now_data, dtype=np.int64).reshape(-1)
                except Exception:
                    continue
                if data_all.size == 0 or row_struct_all.size == 0:
                    continue
                if data_all.ndim == 1:
                    data_all = data_all.reshape(-1, 1)
                if data_all.shape[0] != row_struct_all.shape[0]:
                    lim = int(min(data_all.shape[0], row_struct_all.shape[0]))
                    data_all = data_all[:lim]
                    row_struct_all = row_struct_all[:lim]
                scope_mask = np.isin(row_struct_all, scope_indices)
                if not np.any(scope_mask):
                    continue

                rows = data_all[scope_mask]
                row_struct = row_struct_all[scope_mask]
                cols = int(getattr(dataset, "cols", 0) or 0)
                if cols <= 0:
                    continue

                ref = np.asarray(rows[:, dataset.x_cols], dtype=np.float64)
                pred = np.asarray(rows[:, dataset.y_cols], dtype=np.float64)
                if ref.ndim == 1:
                    ref = ref.reshape(-1, 1)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)

                view = req_norm.value_view
                if view == DistributionValueView.PREDICTION:
                    values = pred
                elif view == DistributionValueView.ERROR:
                    values = pred - ref
                else:
                    values = ref

                comp_names = list(spec.components)
                n_comp = int(values.shape[1])
                if len(comp_names) != n_comp:
                    comp_names = list(self._component_names(n_comp))

                row_elem = None
                if req_norm.group_mode == DistributionGroupMode.ELEMENT and spec.domain == FieldDomain.ATOM:
                    row_elem_all = self._dataset_row_element_map(dataset)
                    if row_elem_all is not None and row_elem_all.shape[0] == data_all.shape[0]:
                        row_elem = np.asarray(row_elem_all[scope_mask], dtype=object).reshape(-1)

                # Determine which value views to process
                # - VALUE_VIEW: process all selected views as separate series
                # - CUSTOM with single group + selected views: also process multiple views
                # - Otherwise: single view only
                use_multi_view = (
                    req_norm.group_mode == DistributionGroupMode.VALUE_VIEW
                    or (req_norm.group_mode == DistributionGroupMode.CUSTOM and selected_views)
                )
                if use_multi_view and spec.has_prediction_pair and selected_views:
                    views_to_process = [(vv, vv.value) for vv in selected_views]
                else:
                    views_to_process = [(view, view.value)]

                for proc_view, proc_view_label in views_to_process:
                    if proc_view == DistributionValueView.PREDICTION:
                        proc_values = pred
                    elif proc_view == DistributionValueView.ERROR:
                        proc_values = pred - ref
                    else:
                        proc_values = ref
                    if proc_values.ndim == 1:
                        proc_values = proc_values.reshape(-1, 1)

                    for ci, comp_name in enumerate(comp_names):
                        metric_key = f"{field_key}|{comp_name}"
                        meta = {
                            "field_key": field_key,
                            "field_label": spec.label or field_key,
                            "component": str(comp_name),
                            "unit": spec.unit_guess,
                            "value_view": proc_view_label,
                            "has_prediction_pair": bool(spec.has_prediction_pair),
                            "available_views": [
                                DistributionValueView.REFERENCE.value,
                                DistributionValueView.PREDICTION.value,
                                DistributionValueView.ERROR.value,
                            ],
                        }
                        col = proc_values[:, ci]
                        for ridx, sid_raw in enumerate(row_struct.tolist()):
                            sid = int(sid_raw)
                            v = float(col[ridx])
                            if req_norm.group_mode == DistributionGroupMode.FORMULA:
                                groups = [formula_map.get(sid, "")]
                            elif req_norm.group_mode == DistributionGroupMode.ELEMENT:
                                if row_elem is not None:
                                    groups = [str(row_elem[ridx])]
                                else:
                                    groups = sorted(elem_set_map.get(sid, set()))
                            elif req_norm.group_mode == DistributionGroupMode.VALUE_VIEW:
                                groups = [proc_view_label]
                            elif req_norm.group_mode == DistributionGroupMode.CUSTOM:
                                if selected_views:
                                    # CUSTOM + multi-value: group by value view, filter by custom groups
                                    if sid in custom_group_map:
                                        groups = [proc_view_label]
                                    else:
                                        groups = []
                                else:
                                    # CUSTOM + single value: group by custom group label
                                    groups = custom_group_map.get(sid, [])
                            else:
                                groups = [formula_map.get(sid, "")]
                            for grp in groups:
                                _append_sample(metric_key, grp, v, sid, meta)

                    if req_norm.include_norm and n_comp > 1:
                        metric_key = f"{field_key}|norm"
                        meta = {
                            "field_key": field_key,
                            "field_label": spec.label or field_key,
                            "component": "norm",
                            "unit": spec.unit_guess,
                            "value_view": proc_view_label,
                            "has_prediction_pair": bool(spec.has_prediction_pair),
                            "available_views": [
                                DistributionValueView.REFERENCE.value,
                                DistributionValueView.PREDICTION.value,
                                DistributionValueView.ERROR.value,
                            ],
                        }
                        norm_vals = np.linalg.norm(proc_values, axis=1)
                        for ridx, sid_raw in enumerate(row_struct.tolist()):
                            sid = int(sid_raw)
                            v = float(norm_vals[ridx])
                            if req_norm.group_mode == DistributionGroupMode.FORMULA:
                                groups = [formula_map.get(sid, "")]
                            elif req_norm.group_mode == DistributionGroupMode.ELEMENT:
                                if row_elem is not None:
                                    groups = [str(row_elem[ridx])]
                                else:
                                    groups = sorted(elem_set_map.get(sid, set()))
                            elif req_norm.group_mode == DistributionGroupMode.VALUE_VIEW:
                                groups = [proc_view_label]
                            elif req_norm.group_mode == DistributionGroupMode.CUSTOM:
                                if selected_views:
                                    if sid in custom_group_map:
                                        groups = [proc_view_label]
                                    else:
                                        groups = []
                                else:
                                    groups = custom_group_map.get(sid, [])
                            else:
                                groups = [formula_map.get(sid, "")]
                            for grp in groups:
                                _append_sample(metric_key, grp, v, sid, meta)

                yield 1
                continue

            if spec.source == "atomic_property":
                prop_name = field_key.split(":", 1)[-1]
                view = req_norm.value_view
                if view != DistributionValueView.REFERENCE and not warned_view_downgrade:
                    warned_view_downgrade = True
                    messages.append(
                        "Atomic-property fields do not provide prediction pairs; falling back to reference view."
                    )
                for sid in scope_indices.tolist():
                    sid_int = int(sid)
                    structure = self.structure.all_data[sid_int]
                    props = getattr(structure, "atomic_properties", {}) or {}
                    if prop_name not in props:
                        continue
                    try:
                        arr = np.asarray(props[prop_name])
                    except Exception:
                        continue
                    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
                        continue
                    n_atoms = int(len(structure))
                    if arr.ndim == 0 or int(arr.shape[0]) != n_atoms:
                        continue
                    mat = np.asarray(arr, dtype=np.float64).reshape(n_atoms, -1)
                    n_comp = int(mat.shape[1])
                    comp_names = list(spec.components)
                    if len(comp_names) != n_comp:
                        comp_names = list(self._component_names(n_comp))

                    elems = elem_array_map.get(sid_int, np.empty((0,), dtype=object))

                    for ci, comp_name in enumerate(comp_names):
                        metric_key = f"{field_key}|{comp_name}"
                        meta = {
                            "field_key": field_key,
                            "field_label": spec.label or field_key,
                            "component": str(comp_name),
                            "unit": spec.unit_guess,
                            "value_view": DistributionValueView.REFERENCE.value,
                            "has_prediction_pair": False,
                            "available_views": [DistributionValueView.REFERENCE.value],
                        }
                        col = mat[:, ci]
                        if req_norm.group_mode == DistributionGroupMode.FORMULA:
                            grp = formula_map.get(sid_int, "")
                            for v in col.tolist():
                                _append_sample(metric_key, grp, float(v), sid_int, meta)
                        elif req_norm.group_mode == DistributionGroupMode.VALUE_VIEW:
                            for v in col.tolist():
                                _append_sample(metric_key, DistributionValueView.REFERENCE.value, float(v), sid_int, meta)
                        elif req_norm.group_mode == DistributionGroupMode.CUSTOM:
                            if selected_views and sid_int in custom_group_map:
                                for v in col.tolist():
                                    _append_sample(metric_key, DistributionValueView.REFERENCE.value, float(v), sid_int, meta)
                            elif not selected_views:
                                for grp in custom_group_map.get(sid_int, []):
                                    for v in col.tolist():
                                        _append_sample(metric_key, grp, float(v), sid_int, meta)
                        else:
                            for aidx, v in enumerate(col.tolist()):
                                if aidx >= elems.size:
                                    continue
                                grp = str(elems[aidx])
                                _append_sample(metric_key, grp, float(v), sid_int, meta)

                    if req_norm.include_norm and n_comp > 1:
                        metric_key = f"{field_key}|norm"
                        meta = {
                            "field_key": field_key,
                            "field_label": spec.label or field_key,
                            "component": "norm",
                            "unit": spec.unit_guess,
                            "value_view": DistributionValueView.REFERENCE.value,
                            "has_prediction_pair": False,
                            "available_views": [DistributionValueView.REFERENCE.value],
                        }
                        norm_vals = np.linalg.norm(mat, axis=1)
                        if req_norm.group_mode == DistributionGroupMode.FORMULA:
                            grp = formula_map.get(sid_int, "")
                            for v in norm_vals.tolist():
                                _append_sample(metric_key, grp, float(v), sid_int, meta)
                        elif req_norm.group_mode == DistributionGroupMode.ELEMENT:
                            for aidx, v in enumerate(norm_vals.tolist()):
                                if aidx >= elems.size:
                                    continue
                                grp = str(elems[aidx])
                                _append_sample(metric_key, grp, float(v), sid_int, meta)
                        elif req_norm.group_mode == DistributionGroupMode.VALUE_VIEW:
                            for v in norm_vals.tolist():
                                _append_sample(
                                    metric_key,
                                    DistributionValueView.REFERENCE.value,
                                    float(v),
                                    sid_int,
                                    meta,
                                )
                        elif req_norm.group_mode == DistributionGroupMode.CUSTOM:
                            if selected_views and sid_int in custom_group_map:
                                for v in norm_vals.tolist():
                                    _append_sample(
                                        metric_key,
                                        DistributionValueView.REFERENCE.value,
                                        float(v),
                                        sid_int,
                                        meta,
                                    )
                            elif not selected_views:
                                for grp in custom_group_map.get(sid_int, []):
                                    for v in norm_vals.tolist():
                                        _append_sample(metric_key, grp, float(v), sid_int, meta)
                    yield 1

        lookup: dict[tuple[int, str, str, int], list[int]] = {}
        metrics: list[dict[str, Any]] = []
        for metric_key in sorted(metric_values.keys()):
            series_map = metric_values.get(metric_key, {})
            if not series_map:
                continue
            all_vals = np.concatenate([np.asarray(v, dtype=np.float64) for v in series_map.values() if len(v)])
            if all_vals.size == 0:
                continue
            all_vals = all_vals[np.isfinite(all_vals)]
            if all_vals.size == 0:
                continue
            lo = float(np.min(all_vals))
            hi = float(np.max(all_vals))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if hi <= lo:
                delta = max(1e-12, abs(lo) * 1e-9 + 1e-12)
                lo -= delta
                hi += delta

            meta = metric_meta.get(metric_key, {})
            series_list: list[dict[str, Any]] = []
            for series_key in sorted(series_map.keys()):
                vals = np.asarray(series_map[series_key], dtype=np.float64)
                if vals.size == 0:
                    continue
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                sid_arr = np.asarray(metric_structs.get(metric_key, {}).get(series_key, []), dtype=np.int64)
                if sid_arr.size != vals.size:
                    size = int(min(sid_arr.size, vals.size))
                    sid_arr = sid_arr[:size]
                    vals = vals[:size]
                hist, _ = np.histogram(vals, bins=bins, range=(lo, hi))
                bin_idx = self._histogram_bin_indices(vals, lo=lo, hi=hi, bins=bins)

                bucket: dict[int, set[int]] = {}
                for b, sid in zip(bin_idx.tolist(), sid_arr.tolist()):
                    bucket.setdefault(int(b), set()).add(int(sid))
                for b, sid_set in bucket.items():
                    lookup[(analysis_id, metric_key, str(series_key), int(b))] = sorted(sid_set)

                curve_type, curve_x, curve_y, curve_msg = self._build_distribution_curve(
                    vals=vals,
                    lo=lo,
                    hi=hi,
                    bins=bins,
                    style=req_norm.curve_style,
                    points=req_norm.curve_points,
                )
                if curve_msg:
                    messages.append(f"{metric_key} [{series_key}]: {curve_msg}")

                series_list.append(
                    {
                        "series_key": str(series_key),
                        "name": str(series_key),
                        "hist": hist.astype(np.int64, copy=False).tolist(),
                        "total": int(vals.size),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "curve_type": str(curve_type),
                        "curve_x": list(curve_x),
                        "curve_y": list(curve_y),
                        "curve_y_mode": "count",
                    }
                )
            if not series_list:
                continue

            metrics.append(
                {
                    "metric_key": metric_key,
                    "field_key": str(meta.get("field_key", "")),
                    "field_label": str(meta.get("field_label", meta.get("field_key", ""))),
                    "component": str(meta.get("component", "")),
                    "unit": str(meta.get("unit", "unknown")),
                    "value_view": str(meta.get("value_view", DistributionValueView.REFERENCE.value)),
                    "has_prediction_pair": bool(meta.get("has_prediction_pair", False)),
                    "available_views": list(meta.get("available_views", [DistributionValueView.REFERENCE.value])),
                    "bins": int(bins),
                    "hist_left": float(lo),
                    "hist_right": float(hi),
                    "series": series_list,
                }
            )

        result = {
            "analysis_id": analysis_id,
            "request": {
                "field_keys": list(req_norm.field_keys),
                "include_norm": bool(req_norm.include_norm),
                "value_view": req_norm.value_view.value,
                "group_mode": req_norm.group_mode.value,
                "scope": req_norm.scope.value,
                "bins": int(req_norm.bins),
                "select_mode": req_norm.select_mode.value,
                "groups": list(req_norm.groups),
                "curve_style": req_norm.curve_style.value,
                "curve_points": int(req_norm.curve_points),
                "selected_value_views": list(req_norm.selected_value_views),
                "custom_group_specs": list(req_norm.custom_group_specs),
            },
            "field_specs": [
                {
                    "key": s.key,
                    "source": s.source,
                    "shape": s.shape.value,
                    "components": list(s.components),
                    "has_prediction_pair": bool(s.has_prediction_pair),
                    "unit_guess": s.unit_guess,
                    "domain": s.domain.value,
                    "label": s.label,
                }
                for s in all_specs
            ],
            "messages": messages,
            "metrics": metrics,
        }

        self._distribution_cache_key = cache_key
        self._distribution_analysis = result
        self._distribution_bin_lookup = lookup

    def get_distribution_analysis(self) -> dict[str, Any]:
        """Return the last computed distribution-analysis payload."""
        return dict(getattr(self, "_distribution_analysis", {}) or {})

    def resolve_distribution_bin_indices(
        self,
        analysis_id: int,
        metric_key: str,
        series_key: str,
        bin_index: int,
    ) -> list[int]:
        """Resolve structure indices represented by a histogram bin."""
        key = (int(analysis_id), str(metric_key), str(series_key), int(bin_index))
        return list(getattr(self, "_distribution_bin_lookup", {}).get(key, []))
