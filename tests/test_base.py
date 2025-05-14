#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import numpy as np
from pathlib import Path
import os
from NepTrainKit.core.io.base import NepPlotData, StructureData
from NepTrainKit.core.structure import Structure

@pytest.fixture
def test_setup():
    test_data = np.random.rand(10, 6)
    test_indices = np.arange(10)
    test_dir = Path(__file__).parent
    return test_data, test_indices, test_dir

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