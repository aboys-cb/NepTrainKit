#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import pytest

from NepTrainKit.core.io import NepPlotData, StructureData, ResultData
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.types import (
    SearchType,
    FieldValueShape,
    DistributionGroupMode,
    DistributionValueView,
    DistributionScope,
    DistributionSelectMode,
    DistributionCurveStyle,
)
from NepTrainKit.core.io.base import DistributionRequest

@pytest.fixture
def test_setup():
    test_data = np.random.rand(10, 6)
    test_indices = np.arange(10)
    test_dir = Path(__file__).parent
    return test_data, test_indices, test_dir
def test_single_remove_and_revoke(test_setup):
    """Removing one row keeps 2-D shape and revoke restores it"""
    test_data, _, _ = test_setup
    data = NepPlotData(test_data)
    data.remove(0)
    assert data.now_data.shape == (9, 6)
    assert data.remove_data.shape == (1, 6)
    data.revoke()
    assert data.now_data.shape == (10, 6)
    assert data.remove_data.shape == (0, 6)
def test_nep_plot_data(test_setup):
    """测试NepPlotData基本功能"""
    test_data, _, _ = test_setup
    data = NepPlotData(test_data)
    assert data.num == 10
    assert data.now_data.shape == (10, 6)
    data.remove([0, 1])
    assert data.now_data.shape == (8, 6)
    assert data.remove_data.shape == (2, 6)
    data.revoke()
    assert data.now_data.shape == (10, 6)

def test_structure_data(test_setup):
    """测试StructureData基本功能"""
    _, _, test_dir = test_setup
    structures = Structure.read_multiple(os.path.join(test_dir, "data/nep/train.xyz"))
    data = StructureData(structures)
    assert data.num == 25


def _make_structure(species: list[str], tag: str) -> Structure:
    lattice = np.eye(3, dtype=np.float32)
    pos = np.zeros((len(species), 3), dtype=np.float32)
    atomic_properties = {
        "species": np.asarray(species, dtype=object),
        "pos": pos,
    }
    properties = [
        {"name": "species", "type": "S", "count": 1},
        {"name": "pos", "type": "R", "count": 3},
    ]
    additional_fields = {"Config_type": tag, "energy": 0.0}
    return Structure(lattice, atomic_properties, properties, additional_fields)


def test_structure_data_completer_cache_counts():
    structures = [
        _make_structure(["H", "O"], "alpha"),
        _make_structure(["Fe", "O"], "beta"),
        _make_structure(["Fe", "Fe"], "alpha"),
    ]
    data = StructureData(structures)

    tag_cache = data.get_completer_cache(SearchType.TAG, max_items=50000)
    assert isinstance(tag_cache, dict)
    assert tag_cache["alpha"] == 2
    assert tag_cache["beta"] == 1

    formula_cache = data.get_completer_cache(SearchType.FORMULA, max_items=50000)
    assert isinstance(formula_cache, dict)
    assert sum(formula_cache.values()) == 3

    elem_cache = data.get_completer_cache(SearchType.ELEMENTS, max_items=50000)
    assert elem_cache["H"] == 1
    assert elem_cache["O"] == 2
    assert elem_cache["Fe"] == 2


def test_completer_cache_respects_max_items():
    structures = [_make_structure(["H"], f"tag_{i:04d}") for i in range(100)]
    data = StructureData(structures)
    cache = data.get_completer_cache(SearchType.TAG, max_items=10)
    assert isinstance(cache, dict)
    assert len(cache) == 10


def _make_structure_with_numeric_props(species: list[str], tag: str, base: float = 0.0) -> Structure:
    n = len(species)
    lattice = np.eye(3, dtype=np.float32)
    pos = np.arange(n * 3, dtype=np.float32).reshape(n, 3) * 0.01
    forces = np.full((n, 3), base + 0.2, dtype=np.float32)
    spin_scalar = np.linspace(base, base + 0.3, num=n, dtype=np.float32)
    spin_vec = np.stack(
        [
            np.linspace(base + 0.1, base + 0.3, num=n, dtype=np.float32),
            np.linspace(base + 0.2, base + 0.4, num=n, dtype=np.float32),
            np.linspace(base + 0.3, base + 0.5, num=n, dtype=np.float32),
        ],
        axis=1,
    )
    spin_tensor = np.zeros((n, 2, 2), dtype=np.float32)
    spin_tensor[:, 0, 0] = base + 0.1
    spin_tensor[:, 0, 1] = base + 0.2
    spin_tensor[:, 1, 0] = base + 0.3
    spin_tensor[:, 1, 1] = base + 0.4

    atomic_properties = {
        "species": np.asarray(species, dtype=object),
        "pos": pos,
        "forces": forces,
        "spin_scalar": spin_scalar,
        "spin_vec": spin_vec,
        "spin_tensor": spin_tensor,
    }
    properties = [
        {"name": "species", "type": "S", "count": 1},
        {"name": "pos", "type": "R", "count": 3},
        {"name": "forces", "type": "R", "count": 3},
        {"name": "spin_scalar", "type": "R", "count": 1},
        {"name": "spin_vec", "type": "R", "count": 3},
        {"name": "spin_tensor", "type": "R", "count": 4},
    ]
    additional_fields = {"Config_type": tag, "energy": float(base + n)}
    return Structure(lattice, atomic_properties, properties, additional_fields)


class _DummyResultData(ResultData):
    def __init__(self, structures: list[Structure]):
        super().__init__(Path("nep.txt"), Path("train.xyz"), Path("descriptor.out"), calculator_factory=lambda _m: None)
        self._atoms_dataset = StructureData(structures)
        self.atoms_num_list = np.array([len(s) for s in structures], dtype=np.int32)
        self._abcs = np.array([s.abc for s in structures], dtype=np.float32)
        self._angles = np.array([s.angles for s in structures], dtype=np.float32)

        ref_energy = np.array([float(s.per_atom_energy) for s in structures], dtype=np.float32).reshape(-1, 1)
        pred_energy = ref_energy + 0.05
        energy_data = np.hstack([pred_energy, ref_energy]).astype(np.float32, copy=False)
        self._energy_dataset = NepPlotData(energy_data, title="energy")

        ref_force = np.vstack([np.asarray(s.forces, dtype=np.float32) for s in structures], dtype=np.float32)
        pred_force = ref_force + 0.01
        force_data = np.hstack([pred_force, ref_force]).astype(np.float32, copy=False)
        self._force_dataset = NepPlotData(force_data, group_list=self.atoms_num_list, title="force")

    @property
    def datasets(self):
        return [self.energy, self.force]

    @property
    def energy(self):
        return self._energy_dataset

    @property
    def force(self):
        return self._force_dataset

    def _load_dataset(self) -> None:
        return


def _build_dummy_result() -> _DummyResultData:
    structures = [
        _make_structure_with_numeric_props(["H", "O"], "alpha", base=0.0),
        _make_structure_with_numeric_props(["Fe", "O", "Fe"], "beta", base=1.0),
    ]
    return _DummyResultData(structures)


def test_discover_atomic_numeric_fields_excludes_blacklist_and_classifies():
    data = _build_dummy_result()
    fields = data.discover_atomic_numeric_fields(scope="active")
    field_by_key = {f.key: f for f in fields}

    assert "dataset:energy" in field_by_key
    assert "dataset:force" in field_by_key
    assert "atomic:spin_scalar" in field_by_key
    assert "atomic:spin_vec" in field_by_key
    assert "atomic:spin_tensor" in field_by_key
    assert "atomic:species" not in field_by_key
    assert "atomic:pos" not in field_by_key

    assert field_by_key["atomic:spin_scalar"].shape == FieldValueShape.SCALAR
    assert field_by_key["atomic:spin_vec"].shape == FieldValueShape.VECTOR3
    assert field_by_key["atomic:spin_tensor"].shape == FieldValueShape.TENSOR


def test_distribution_formula_group_vector_has_norm_metric():
    data = _build_dummy_result()
    req = DistributionRequest(
        field_keys=("atomic:spin_vec",),
        include_norm=True,
        value_view=DistributionValueView.REFERENCE,
        group_mode=DistributionGroupMode.FORMULA,
        scope=DistributionScope.ACTIVE,
        bins=20,
        select_mode=DistributionSelectMode.REPLACE,
    )
    list(data.iter_distribution_analysis(req))
    analysis = data.get_distribution_analysis()
    metrics = analysis.get("metrics", [])
    metric_by_key = {m.get("metric_key"): m for m in metrics}

    assert "atomic:spin_vec|norm" in metric_by_key
    total = sum(int(s.get("total", 0)) for s in metric_by_key["atomic:spin_vec|norm"].get("series", []))
    assert total == 5  # two structures: 2 + 3 atoms


def test_distribution_element_group_counts_match_element_atoms():
    data = _build_dummy_result()
    req = DistributionRequest(
        field_keys=("atomic:spin_scalar",),
        include_norm=False,
        value_view=DistributionValueView.REFERENCE,
        group_mode=DistributionGroupMode.ELEMENT,
        scope=DistributionScope.ACTIVE,
        bins=16,
        select_mode=DistributionSelectMode.REPLACE,
    )
    list(data.iter_distribution_analysis(req))
    analysis = data.get_distribution_analysis()
    metrics = analysis.get("metrics", [])
    metric = next(m for m in metrics if m.get("metric_key") == "atomic:spin_scalar|value")
    totals = {s.get("series_key"): int(s.get("total", 0)) for s in metric.get("series", [])}

    assert totals.get("H", 0) == 1
    assert totals.get("O", 0) == 2
    assert totals.get("Fe", 0) == 2


def test_distribution_atomic_field_degrades_prediction_and_error_view():
    data = _build_dummy_result()
    req = DistributionRequest(
        field_keys=("atomic:spin_scalar",),
        include_norm=False,
        value_view=DistributionValueView.ERROR,
        group_mode=DistributionGroupMode.FORMULA,
        scope=DistributionScope.ACTIVE,
        bins=10,
        select_mode=DistributionSelectMode.REPLACE,
    )
    list(data.iter_distribution_analysis(req))
    analysis = data.get_distribution_analysis()
    assert analysis.get("messages")

    metric = next(m for m in analysis.get("metrics", []) if m.get("metric_key") == "atomic:spin_scalar|value")
    assert metric.get("value_view") == DistributionValueView.REFERENCE.value
    assert metric.get("available_views") == [DistributionValueView.REFERENCE.value]


def test_distribution_bin_reverse_lookup_returns_unique_sorted_indices():
    data = _build_dummy_result()
    req = DistributionRequest(
        field_keys=("atomic:spin_scalar",),
        include_norm=False,
        value_view=DistributionValueView.REFERENCE,
        group_mode=DistributionGroupMode.FORMULA,
        scope=DistributionScope.ACTIVE,
        bins=4,
        select_mode=DistributionSelectMode.REPLACE,
    )
    list(data.iter_distribution_analysis(req))
    analysis = data.get_distribution_analysis()

    analysis_id = int(analysis.get("analysis_id", 0))
    metric = next(m for m in analysis.get("metrics", []) if m.get("metric_key") == "atomic:spin_scalar|value")
    series = metric.get("series", [])[0]
    series_key = str(series.get("series_key"))
    hist = list(series.get("hist", []))
    bin_index = next(i for i, c in enumerate(hist) if int(c) > 0)

    indices = data.resolve_distribution_bin_indices(analysis_id, "atomic:spin_scalar|value", series_key, bin_index)
    assert indices == sorted(set(indices))
    assert len(indices) >= 1


def test_distribution_cache_invalidates_after_remove_and_revoke():
    data = _build_dummy_result()
    req = DistributionRequest(
        field_keys=("atomic:spin_scalar",),
        include_norm=False,
        value_view=DistributionValueView.REFERENCE,
        group_mode=DistributionGroupMode.FORMULA,
        scope=DistributionScope.ACTIVE,
        bins=8,
        select_mode=DistributionSelectMode.REPLACE,
    )

    list(data.iter_distribution_analysis(req))
    first = data.get_distribution_analysis()
    first_id = int(first.get("analysis_id", 0))

    data.remove(0)
    list(data.iter_distribution_analysis(req))
    second = data.get_distribution_analysis()
    second_id = int(second.get("analysis_id", 0))

    assert second_id > first_id
    totals_second = sum(
        int(s.get("total", 0))
        for m in second.get("metrics", [])
        if m.get("metric_key") == "atomic:spin_scalar|value"
        for s in m.get("series", [])
    )
    assert totals_second == 3

    data.revoke()
    list(data.iter_distribution_analysis(req))
    third = data.get_distribution_analysis()
    third_id = int(third.get("analysis_id", 0))
    assert third_id > second_id


def test_distribution_curve_kde_payload_generated():
    data = _build_dummy_result()
    req = DistributionRequest(
        field_keys=("atomic:spin_scalar",),
        include_norm=False,
        value_view=DistributionValueView.REFERENCE,
        group_mode=DistributionGroupMode.ELEMENT,
        scope=DistributionScope.ACTIVE,
        bins=24,
        select_mode=DistributionSelectMode.REPLACE,
        curve_style=DistributionCurveStyle.KDE,
        curve_points=160,
    )
    list(data.iter_distribution_analysis(req))
    analysis = data.get_distribution_analysis()
    metric = next(m for m in analysis.get("metrics", []) if m.get("metric_key") == "atomic:spin_scalar|value")
    series = next(s for s in metric.get("series", []) if s.get("series_key") == "Fe")
    assert series.get("curve_y_mode") == "count"
    assert int(len(series.get("curve_x", []))) in {0, 160}
    assert int(len(series.get("curve_y", []))) in {0, 160}
    if series.get("curve_type") == DistributionCurveStyle.NONE.value:
        assert any("KDE" in str(msg) or "SciPy" in str(msg) for msg in analysis.get("messages", []))
    else:
        assert series.get("curve_type") == DistributionCurveStyle.KDE.value


def test_distribution_curve_normal_payload_generated():
    data = _build_dummy_result()
    req = DistributionRequest(
        field_keys=("atomic:spin_scalar",),
        include_norm=False,
        value_view=DistributionValueView.REFERENCE,
        group_mode=DistributionGroupMode.FORMULA,
        scope=DistributionScope.ACTIVE,
        bins=20,
        select_mode=DistributionSelectMode.REPLACE,
        curve_style=DistributionCurveStyle.NORMAL,
        curve_points=128,
    )
    list(data.iter_distribution_analysis(req))
    analysis = data.get_distribution_analysis()
    metric = next(m for m in analysis.get("metrics", []) if m.get("metric_key") == "atomic:spin_scalar|value")
    series = metric.get("series", [])[0]
    assert series.get("curve_type") == DistributionCurveStyle.NORMAL.value
    assert len(series.get("curve_x", [])) == 128
    assert len(series.get("curve_y", [])) == 128


def test_distribution_curve_degrades_for_constant_series():
    data = _build_dummy_result()
    for structure in data.structure.all_data:
        n = len(structure)
        structure.atomic_properties["spin_scalar"] = np.ones((n,), dtype=np.float32)

    req = DistributionRequest(
        field_keys=("atomic:spin_scalar",),
        include_norm=False,
        value_view=DistributionValueView.REFERENCE,
        group_mode=DistributionGroupMode.ELEMENT,
        scope=DistributionScope.ACTIVE,
        bins=16,
        select_mode=DistributionSelectMode.REPLACE,
        curve_style=DistributionCurveStyle.KDE,
        curve_points=200,
    )
    list(data.iter_distribution_analysis(req))
    analysis = data.get_distribution_analysis()
    metric = next(m for m in analysis.get("metrics", []) if m.get("metric_key") == "atomic:spin_scalar|value")
    for series in metric.get("series", []):
        assert series.get("curve_type") == DistributionCurveStyle.NONE.value
        assert series.get("curve_x", []) == []
        assert series.get("curve_y", []) == []
    assert any("variance" in str(msg).lower() for msg in analysis.get("messages", []))


def test_distribution_cache_invalidates_when_curve_style_changes():
    data = _build_dummy_result()
    req_kde = DistributionRequest(
        field_keys=("atomic:spin_scalar",),
        include_norm=False,
        value_view=DistributionValueView.REFERENCE,
        group_mode=DistributionGroupMode.FORMULA,
        scope=DistributionScope.ACTIVE,
        bins=10,
        select_mode=DistributionSelectMode.REPLACE,
        curve_style=DistributionCurveStyle.KDE,
        curve_points=120,
    )
    list(data.iter_distribution_analysis(req_kde))
    first = data.get_distribution_analysis()
    first_id = int(first.get("analysis_id", 0))

    req_normal = DistributionRequest(
        field_keys=("atomic:spin_scalar",),
        include_norm=False,
        value_view=DistributionValueView.REFERENCE,
        group_mode=DistributionGroupMode.FORMULA,
        scope=DistributionScope.ACTIVE,
        bins=10,
        select_mode=DistributionSelectMode.REPLACE,
        curve_style=DistributionCurveStyle.NORMAL,
        curve_points=120,
    )
    list(data.iter_distribution_analysis(req_normal))
    second = data.get_distribution_analysis()
    second_id = int(second.get("analysis_id", 0))
    assert second_id > first_id


def test_distribution_nonzero_bins_always_resolve_structure_indices():
    data = _build_dummy_result()
    req = DistributionRequest(
        field_keys=("atomic:spin_scalar",),
        include_norm=False,
        value_view=DistributionValueView.REFERENCE,
        group_mode=DistributionGroupMode.ELEMENT,
        scope=DistributionScope.ACTIVE,
        bins=12,
        select_mode=DistributionSelectMode.REPLACE,
    )
    list(data.iter_distribution_analysis(req))
    analysis = data.get_distribution_analysis()
    analysis_id = int(analysis.get("analysis_id", 0))
    metric = next(m for m in analysis.get("metrics", []) if m.get("metric_key") == "atomic:spin_scalar|value")
    for series in metric.get("series", []):
        series_key = str(series.get("series_key", ""))
        hist = list(series.get("hist", []) or [])
        for bidx, count in enumerate(hist):
            if int(count) <= 0:
                continue
            indices = data.resolve_distribution_bin_indices(analysis_id, metric["metric_key"], series_key, bidx)
            assert len(indices) > 0

