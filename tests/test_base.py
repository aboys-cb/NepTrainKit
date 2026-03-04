#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from pathlib import Path
import os
from NepTrainKit.core.io import NepPlotData, StructureData
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.types import SearchType

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

