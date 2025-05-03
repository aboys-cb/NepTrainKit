#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import numpy as np
from NepTrainKit.core.structure import Structure

class TestStructure(unittest.TestCase):
    def setUp(self):
        # 创建测试用的晶格和原子结构
        self.lattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.structure_info = {
            'species': ['H', 'O'],
            'pos': np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=np.float32),
            'forces': np.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]], dtype=np.float32)
        }
        self.properties = [
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
            {"name": "forces", "type":  "R", "count": 3}
        ]
        self.additional_fields = {"energy": 1.0, "virial": "0 0 0 0 0 0 0 0 0"}
        self.structure = Structure(self.lattice, self.structure_info, self.properties, self.additional_fields)

    def test_basic_properties(self):
        # 测试基本属性
        self.assertEqual(len(self.structure), 2)
        self.assertEqual(self.structure.num_atoms, 2)
        self.assertEqual(self.structure.formula, "H1O1")
        self.assertEqual(self.structure.html_formula, "H<sub>1</sub>O<sub>1</sub>")
        self.assertListEqual(self.structure.numbers, [1, 8])

    def test_energy_calculations(self):
        # 测试能量相关计算
        self.assertEqual(self.structure.per_atom_energy, 0.5)

    def test_lattice_operations(self):
        # 测试晶格操作
        new_lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float32)
        new_structure = self.structure.set_lattice(new_lattice)
        np.testing.assert_array_equal(new_structure.lattice, new_lattice)
        np.testing.assert_allclose(new_structure.positions, 
                                  np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32))

    def test_virial_calculation(self):
        # 测试virial计算
        expected_virial = np.zeros(6, dtype=np.float32)
        np.testing.assert_array_equal(self.structure.nep_virial, expected_virial)

    def test_xyz_io(self):
        # 测试xyz文件读写
        test_file = "test.xyz"
        # 写入测试文件
        with open(test_file, 'w') as f:
            self.structure.write(f)

        # 读取测试文件并验证内容
        read_structure = Structure.read_xyz(test_file)
        self.assertEqual(len(read_structure), 2)
        self.assertEqual(read_structure.num_atoms, 2)
        np.testing.assert_array_equal(read_structure.lattice, self.lattice)
        np.testing.assert_array_equal(read_structure.positions, self.structure_info['pos'])
        np.testing.assert_array_equal(read_structure.elements, self.structure_info['species'])

        # 清理测试文件
        import os
        os.remove(test_file)

if __name__ == '__main__':
    unittest.main()