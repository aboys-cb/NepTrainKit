#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from pathlib import Path
import os
from NepTrainKit.core.io.base import NepPlotData, StructureData
from NepTrainKit.core.structure import Structure
class TestBaseClasses(unittest.TestCase):
    def setUp(self):
        self.test_data = np.random.rand(10, 6)
        self.test_indices = np.arange(10)
        self.test_dir = Path(__file__).parent

    def test_nep_plot_data(self):
        """测试NepPlotData基本功能"""
        data = NepPlotData(self.test_data )
        self.assertEqual(data.num, 10)
        self.assertEqual(data.now_data.shape, (10, 6))
        data.remove([0,1])
        self.assertEqual(data.now_data.shape, (8, 6))
        self.assertEqual(data.remove_data.shape, (2, 6))
        data.revoke()
        self.assertEqual(data.now_data.shape, (10, 6))

    def test_structure_data(self):
        """测试StructureData基本功能"""

        structures =Structure.read_multiple(os.path.join(self.test_dir,"data/nep/train.xyz"))
        data = StructureData(structures)
        self.assertEqual(data.num, 25)
        
if __name__ == "__main__":
    unittest.main()