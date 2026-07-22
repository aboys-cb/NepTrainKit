#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from NepTrainKit.core.io import NepPlotData, StructureData, ResultData
from NepTrainKit.config import Config
from NepTrainKit.core.energy_shift import DFT_TO_NEP_ALIGNMENT
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
from NepTrainKit.core.io.base import DistributionRequest, StructureSyncRule

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


def test_nep_plot_data_max_error_uses_unique_structure_order():
    data = np.array(
        [
            [10.0, 0.0],
            [9.0, 0.0],
            [8.0, 0.0],
            [7.0, 0.0],
            [6.0, 0.0],
            [5.0, 0.0],
            [4.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float64,
    )
    plot = NepPlotData(data, group_list=[4, 2, 2], title="force")

    assert plot.get_max_error_index(3) == [0, 1, 2]


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


def test_structure_geometry_snapshot_survives_mask_changes_and_reuses_storage():
    structures = [
        _make_structure(["H"], "single"),
        _make_structure(["Fe", "O"], "pair"),
        _make_structure(["Ni"], "single"),
    ]
    structures[1].positions[1] = np.asarray([0.25, 0.0, 0.0], dtype=np.float32)
    data = StructureData(structures)

    full = data.geometry_snapshot()
    assert data.geometry_snapshot() is full
    assert full.positions.dtype == np.float32
    assert full.atomic_numbers.tolist() == [1, 26, 8, 28]
    assert full.atom_offsets.tolist() == [0, 1, 3, 4]

    data.remove(1)
    active = data.geometry_snapshot(data.now_indices)
    assert active.source_indices.tolist() == [0, 2]
    assert active.atom_offsets.tolist() == [0, 1, 2]
    assert data.geometry_snapshot(data.now_indices) is active

    data.revoke()
    assert data.geometry_snapshot(data.now_indices) is full


def test_structure_geometry_cache_owns_versioned_derived_analysis():
    data = StructureData([_make_structure(["Ni"], "single")])
    calls = []

    first, first_hit = data.cached_geometry_analysis(
        "phase",
        ("method-v1", (0,)),
        lambda: calls.append("built") or {"phase": "fcc"},
    )
    second, second_hit = data.cached_geometry_analysis(
        "phase",
        ("method-v1", (0,)),
        lambda: calls.append("rebuilt") or {"phase": "bcc"},
    )

    assert first == {"phase": "fcc"}
    assert second is first
    assert first_hit is False
    assert second_hit is True
    assert calls == ["built"]


def test_non_physical_scan_uses_active_geometry_snapshot():
    safe = _make_structure(["H"], "safe")
    collision = _make_structure(["Fe", "O"], "collision")
    collision.positions[1] = np.asarray([0.1, 0.0, 0.0], dtype=np.float32)
    owner = type("Owner", (), {})()
    owner.structure = StructureData([safe, collision])
    owner._pending_non_physical_indices = []

    assert list(ResultData.iter_non_physical_structure_indices(owner, 0.7)) == [1, 1]
    assert ResultData.consume_non_physical_structure_indices(owner) == [1]

    owner.structure.remove(1)
    assert list(ResultData.iter_non_physical_structure_indices(owner, 0.7)) == [1]
    assert ResultData.consume_non_physical_structure_indices(owner) == []


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
    def _collect_energy_sync(result_data: "_DummyResultData", dataset: NepPlotData, structure_indices):
        total_cols = dataset.data.all_data.shape[1] if dataset.data.all_data.ndim > 1 else 0
        target_width = max(total_cols - dataset.cols, 0)
        if target_width == 0:
            return np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.float64)
        indices = result_data._normalize_structure_indices(structure_indices)
        if indices.size == 0:
            return np.array([], dtype=np.int64), np.empty((0, target_width), dtype=np.float64)
        structures = [result_data.structure.all_data[i] for i in indices]
        values = np.array([s.per_atom_energy for s in structures], dtype=np.float64).reshape(-1, target_width)
        return indices, values

    STRUCTURE_SYNC_RULES = {
        "energy": StructureSyncRule("energy", "x_cols", _collect_energy_sync, dtype=np.float64),
    }

    def __init__(self, structures: list[Structure]):
        super().__init__(Path("nep.txt"), Path("train.xyz"), Path("descriptor.out"), calculator_factory=lambda _m: None)
        self._atoms_dataset = StructureData(structures)
        self.atoms_num_list = np.array([len(s) for s in structures], dtype=np.int32)
        self._abcs = np.array([s.abc for s in structures], dtype=np.float32)
        self._angles = np.array([s.angles for s in structures], dtype=np.float32)

        ref_energy = np.array([float(s.per_atom_energy) for s in structures], dtype=np.float64).reshape(-1, 1)
        pred_energy = ref_energy + 0.05
        energy_data = np.hstack([pred_energy, ref_energy]).astype(np.float64, copy=False)
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


def test_descriptor_pca_cache_reuses_matching_file(tmp_path):
    previous_cache = Config.get("io", "cache_outputs", None)
    try:
        Config.set("io", "cache_outputs", True)
        data = _build_dummy_result()
        data.descriptor_path = tmp_path / "descriptor.out"
        data.descriptor_path.write_text("descriptor payload", encoding="utf8")

        descriptors = np.arange(30, dtype=np.float32).reshape(5, 6)
        reduced = np.column_stack(
            [
                np.linspace(0.0, 1.0, num=5, dtype=np.float32),
                np.linspace(1.0, 2.0, num=5, dtype=np.float32),
            ]
        )

        with patch("NepTrainKit.core.io.base.pca", return_value=reduced) as pca_mock:
            first = data._load_or_compute_descriptor_pca(descriptors)
            second = data._load_or_compute_descriptor_pca(descriptors)

        pca_mock.assert_called_once_with(descriptors, 2)
        np.testing.assert_allclose(first, reduced)
        np.testing.assert_allclose(second, reduced)
    finally:
        if previous_cache is None:
            Config.delete("io", "cache_outputs")
        else:
            Config.set("io", "cache_outputs", previous_cache)


def test_export_model_xyz_reads_export_digits_once(tmp_path):
    previous_digits = Config.get("io", "export_significant_digits", None)
    try:
        Config.delete("io", "export_significant_digits")
        data = _build_dummy_result()

        with patch.object(Config, "getint", wraps=Config.getint) as getint_mock:
            data.export_model_xyz(tmp_path)

        digit_calls = [
            call for call in getint_mock.call_args_list
            if call.args[:2] == ("io", "export_significant_digits")
        ]
        assert len(digit_calls) == 1

        lines = (tmp_path / "export_good_model.xyz").read_text(encoding="utf8").splitlines()
        assert lines[2].startswith("H 0 0.009999999776 0.01999999955")
    finally:
        if previous_digits is None:
            Config.delete("io", "export_significant_digits")
        else:
            Config.set("io", "export_significant_digits", previous_digits)


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


def test_iter_shift_energy_baseline_uses_float64_predicted_energy_array():
    data = _build_dummy_result()

    with patch("NepTrainKit.core.io.base.shift_dataset_energy", return_value=iter(())) as shift_mock:
        list(
            data.iter_shift_energy_baseline(
                group_patterns=[],
                alignment_mode=DFT_TO_NEP_ALIGNMENT,
                max_generations=10,
                population_size=8,
                convergence_tol=1e-8,
            )
        )

    nep_energy_array = shift_mock.call_args.kwargs["nep_energy_array"]
    assert nep_energy_array.dtype == np.float64
    np.testing.assert_allclose(
        nep_energy_array,
        data.get_predicted_per_atom_energy_array(use_active=True),
        atol=0.0,
    )


def test_apply_dft_d3_correction_keeps_energy_float64():
    data = _build_dummy_result()
    original = np.array([structure.energy for structure in data.structure.now_data], dtype=np.float64)
    potentials = [
        np.float64("0.12345678901234566"),
        np.float64("0.23456789012345677"),
    ]
    zero_forces = [np.zeros_like(structure.forces) for structure in data.structure.now_data]
    zero_virials = [np.zeros(9, dtype=np.float32) for _ in data.structure.now_data]

    with patch("NepTrainKit.core.io.base.NepCalculator") as calc_cls:
        prediction = type(
            "Prediction",
            (),
            {
                "energy": np.asarray(potentials),
                "force_blocks": lambda self: zero_forces,
                "structure_virials": np.asarray(zero_virials),
            },
        )()
        calc_cls.return_value.predict_dftd3.return_value = prediction
        data.apply_dft_d3_correction(mode=0, functional="pbe", cutoff=12.0, cutoff_cn=10.0)

    shifted = np.array([structure.energy for structure in data.structure.now_data], dtype=np.float64)
    np.testing.assert_allclose(shifted, original + np.asarray(potentials, dtype=np.float64), atol=1e-15)
    assert data.energy.all_data.dtype == np.float64
    np.testing.assert_allclose(
        data.energy.all_data[:, data.energy.x_cols].reshape(-1),
        np.array([structure.per_atom_energy for structure in data.structure.now_data], dtype=np.float64),
        atol=1e-15,
    )


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


def test_expression_completer_cache_contains_dynamic_fields_and_elements():
    data = _build_dummy_result()
    cache = data.get_completer_cache(SearchType.EXPRESSION, max_items=50000)

    assert "natoms" in cache
    assert cache["natoms"] == 2
    assert "count.Fe" in cache
    assert cache["count.Fe"] == 1
    assert "frac.O" in cache
    assert cache["frac.O"] == 2
    assert "has.H" in cache
    assert cache["has.H"] == 1
    assert "force.ref.x" in cache
    assert cache["force.ref.x"] == 2
    assert "force.x" in cache
    assert "force.norm" in cache
    assert "atomic.spin_vec.y" in cache
    assert cache["atomic.spin_vec.y"] == 2
    assert "mforce.ref.x" not in cache
    assert "force.ref.1" not in cache


def test_expression_search_supports_builtins_elements_and_dynamic_fields():
    data = _build_dummy_result()

    assert data.search_config("natoms > 2", SearchType.EXPRESSION) == [1]
    assert data.search_config("energy > 3", SearchType.EXPRESSION) == [1]
    assert data.search_config("energy_per_atom > 1.2", SearchType.EXPRESSION) == [1]
    assert data.search_config("count.Fe >= 2 && !has.H", SearchType.EXPRESSION) == [1]
    assert data.search_config("frac.H > 0.4", SearchType.EXPRESSION) == [0]
    assert data.search_config("force.x > 1.0", SearchType.EXPRESSION) == [1]
    assert data.search_config("force.err.x > 0.005", SearchType.EXPRESSION) == [0, 1]


def test_expression_search_accepts_explicit_boolean_predicates():
    data = _build_dummy_result()

    assert data.search_config("has_energy", SearchType.EXPRESSION) == [0, 1]
    assert data.search_config("has.H", SearchType.EXPRESSION) == [0]
    assert data.search_config("!has.H", SearchType.EXPRESSION) == [1]
    assert data.search_config("has_energy && natoms > 2", SearchType.EXPRESSION) == [1]


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("natoms", "must be a condition"),
        ("energy_per_atom", "must be a condition"),
        ("natoms / 2", "must be a condition"),
        ("natoms && energy", "must be a condition"),
        ("True", "at least one structure field"),
        ("1 < 2", "at least one structure field"),
    ],
)
def test_expression_search_rejects_values_that_would_be_cast_to_boolean(expression, message):
    data = _build_dummy_result()

    with pytest.raises(ValueError, match=message):
        data.search_config(expression, SearchType.EXPRESSION)


def test_expression_validation_does_not_disappear_on_an_empty_dataset():
    data = _build_dummy_result()
    data.remove([0, 1])

    with pytest.raises(ValueError, match="must be a condition"):
        data.search_config("natoms", SearchType.EXPRESSION)
    assert data.search_config("natoms > 0", SearchType.EXPRESSION) == []


def test_expression_search_skips_dynamic_discovery_for_simple_references():
    data = _build_dummy_result()

    with patch.object(data, "_discover_expression_fields", side_effect=AssertionError("unexpected dynamic scan")):
        assert data.search_config("natoms > 2", SearchType.EXPRESSION) == [1]
        assert data.search_config("count.Fe >= 2", SearchType.EXPRESSION) == [1]


def test_expression_search_handles_atomic_fields_and_errors():
    data = _build_dummy_result()

    assert data.search_config("atomic.spin_vec.norm > 1.5", SearchType.EXPRESSION) == [1]

    with pytest.raises(ValueError, match="Unknown field"):
        data.search_config("mforce.ref.x > 1", SearchType.EXPRESSION)

    with pytest.raises(ValueError, match="does not support value views"):
        data.search_config("atomic.spin_vec.pred.x > 1", SearchType.EXPRESSION)

    with pytest.raises(ValueError, match="Numeric component suffixes are not supported"):
        data.search_config("force.1 > 1", SearchType.EXPRESSION)

    with pytest.raises(ValueError, match="Invalid expression syntax"):
        data.search_config("force.x >", SearchType.EXPRESSION)


def test_expression_completer_cache_refreshes_after_structure_removal():
    data = _build_dummy_result()
    initial = data.get_completer_cache(SearchType.EXPRESSION, max_items=50000)
    assert "count.Fe" in initial

    data.remove(1)
    refreshed = data.get_completer_cache(SearchType.EXPRESSION, max_items=50000)
    assert "count.Fe" not in refreshed
    assert "has.H" in refreshed
