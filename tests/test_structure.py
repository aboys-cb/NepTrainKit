#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest
import os
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ["LOCALAPPDATA"] = str(Path(__file__).resolve().parent / "_localappdata")

from NepTrainKit.config import Config
from NepTrainKit.core.precision import get_storage_precision
from NepTrainKit.core.structure import Structure, load_npy_structure, save_npy_structure
from NepTrainKit.core.types import DataPrecision


class TestStructure(unittest.TestCase):
    lattice: np.ndarray = np.array([])
    structure_info: dict = {}
    properties: list[dict] = []
    additional_fields: dict = {}
    structure: Structure

    def setUp(self):
        self._prev_precision = Config.get("nep", "data_precision")
        self._prev_export_digits = Config.get("io", "export_significant_digits")
        Config.delete("nep", "data_precision")
        Config.delete("io", "export_significant_digits")
        self.lattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.structure_info = {
            "species": ["H", "O"],
            "pos": np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=np.float32),
            "forces": np.array([[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]], dtype=np.float32),
        }
        self.properties = [
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
            {"name": "forces", "type": "R", "count": 3},
        ]
        self.additional_fields = {
            "energy": 1.0,
            "virial": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
        }
        self.structure = Structure(self.lattice, self.structure_info, self.properties, self.additional_fields)

    def tearDown(self):
        if self._prev_precision is None:
            Config.delete("nep", "data_precision")
        else:
            Config.set("nep", "data_precision", self._prev_precision)
        if self._prev_export_digits is None:
            Config.delete("io", "export_significant_digits")
        else:
            Config.set("io", "export_significant_digits", self._prev_export_digits)

    def test_default_storage_precision_is_float32(self):
        self.assertEqual(get_storage_precision(), DataPrecision.FLOAT32)

    def test_basic_properties(self):
        self.assertEqual(len(self.structure), 2)
        self.assertEqual(self.structure.num_atoms, 2)
        self.assertEqual(self.structure.formula, "HO")
        self.assertEqual(self.structure.html_formula, "HO")
        self.assertListEqual(self.structure.numbers, [1, 8])
        self.assertEqual(self.structure.lattice.dtype, np.float32)
        self.assertEqual(self.structure.positions.dtype, np.float32)
        self.assertEqual(self.structure.forces.dtype, np.float32)
        self.assertEqual(self.structure.virial.dtype, np.float32)
        self.assertEqual(self.structure.angles.dtype, np.float64)

    def test_energy_calculations(self):
        self.assertEqual(self.structure.per_atom_energy, 0.5)

    def test_bad_bond_scan_can_skip_large_structures(self):
        large_structure = Structure(
            np.eye(3),
            {
                "species": ["H"] * 501,
                "pos": np.zeros((501, 3), dtype=np.float32),
            },
            [
                {"name": "species", "type": "S", "count": 1},
                {"name": "pos", "type": "R", "count": 3},
            ],
            {},
        )

        with patch.object(large_structure, "get_all_distances") as get_all_distances:
            self.assertEqual(large_structure.get_bad_bond_pairs(max_atoms=500), [])

        get_all_distances.assert_not_called()

    def test_bad_bond_scan_keeps_complete_analysis_by_default(self):
        with patch.object(
            self.structure,
            "get_all_distances",
            return_value=np.array([[0.0, 0.1], [0.1, 0.0]]),
        ) as get_all_distances:
            self.assertEqual(self.structure.get_bad_bond_pairs(), [(0, 1)])

        get_all_distances.assert_called_once_with()

    def test_lattice_operations(self):
        new_lattice = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float32)
        new_structure = self.structure.set_lattice(new_lattice)
        self.assertEqual(new_structure.lattice.dtype, np.float32)
        self.assertEqual(new_structure.positions.dtype, np.float32)
        np.testing.assert_array_equal(new_structure.lattice, new_lattice)
        np.testing.assert_allclose(
            new_structure.positions,
            np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        )

    def test_virial_calculation(self):
        expected_virial = np.zeros(6, dtype=np.float32)
        np.testing.assert_array_equal(self.structure.nep_virial, expected_virial)

    def test_xyz_io(self):
        test_file = "test.xyz"
        with open(test_file, "w", encoding="utf8") as f:
            self.structure.write(f)

        read_structure = Structure.read_xyz(test_file)
        self.assertEqual(len(read_structure), 2)
        self.assertEqual(read_structure.num_atoms, 2)
        self.assertEqual(read_structure.positions.dtype, np.float32)
        self.assertEqual(read_structure.forces.dtype, np.float32)
        np.testing.assert_array_equal(read_structure.lattice, self.lattice)
        np.testing.assert_array_equal(read_structure.positions, self.structure_info["pos"])
        np.testing.assert_array_equal(read_structure.elements, self.structure_info["species"])

        import os

        os.remove(test_file)

    def test_xyz_energy_roundtrip_preserves_float64_precision(self):
        test_file = "test_precision.xyz"
        Config.set("nep", "data_precision", DataPrecision.FLOAT64)
        precise_energy = 1.1234567890123457
        precise_original = 9.876543210987654
        precise_lattice = np.array(
            [
                [1.1234567890123457, 0.0, 0.0],
                [0.0, 2.2345678901234567, 0.0],
                [0.0, 0.0, 3.345678901234567],
            ],
            dtype=np.float64,
        )
        precise_positions = np.array(
            [[0.12345678901234567, 0.0, 0.0], [0.5, 0.5000000000000001, 0.5]],
            dtype=np.float64,
        )
        precise_forces = np.array(
            [[0.1111111111111111, 0.2222222222222222, 0.3333333333333333],
             [-0.4444444444444444, -0.5555555555555556, -0.6666666666666666]],
            dtype=np.float64,
        )
        precise_virial = np.array(
            [1.1234567890123457, 0.0, 0.0, 0.0, 2.2345678901234567, 0.0, 0.0, 0.0, 3.345678901234567],
            dtype=np.float64,
        )
        structure = Structure(
            precise_lattice,
            {
                "species": ["H", "O"],
                "pos": precise_positions,
                "forces": precise_forces,
            },
            self.properties,
            {
                "energy": precise_energy,
                "energy_original": precise_original,
                "virial": precise_virial,
            },
        )

        with open(test_file, "w", encoding="utf8") as f:
            structure.write(f, atomic_float_digits=17)

        read_structure = Structure.read_xyz(test_file)
        self.assertEqual(read_structure.lattice.dtype, np.float64)
        self.assertEqual(read_structure.positions.dtype, np.float64)
        self.assertEqual(read_structure.forces.dtype, np.float64)
        self.assertEqual(read_structure.virial.dtype, np.float64)
        np.testing.assert_allclose(read_structure.lattice, precise_lattice, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(read_structure.positions, precise_positions, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(read_structure.forces, precise_forces, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(read_structure.virial, precise_virial, rtol=0.0, atol=1e-15)
        self.assertAlmostEqual(read_structure.energy, precise_energy, places=15)
        self.assertAlmostEqual(read_structure.additional_fields["energy_original"], precise_original, places=15)

        import os

        os.remove(test_file)

    def test_xyz_escaped_json_metadata_roundtrip_preserves_following_fields(self):
        xyz = r'''1
Lattice="3 0 0 0 3 0 0 0 3" Properties=species:S:1:pos:R:3 rss_composition="_JSON {\"B\": 0.6, \"N\": 0.4}" prerelax_energy=-4.5 stress="1 0 0 0 1 0 0 0 1" energy=-10.25 pbc="T T T"
B 0 0 0
'''

        structure = Structure.parse_xyz(xyz)

        self.assertEqual(structure.additional_fields["rss_composition"], '_JSON {"B": 0.6, "N": 0.4}')
        self.assertEqual(structure.additional_fields["prerelax_energy"], -4.5)
        self.assertEqual(structure.energy, -10.25)
        np.testing.assert_array_equal(structure.additional_fields["stress"], np.eye(3).reshape(-1))
        self.assertEqual(structure.additional_fields["pbc"], "T T T")

        handle = StringIO()
        structure.write(handle)
        written = handle.getvalue()
        self.assertIn(r'rss_composition="_JSON {\"B\": 0.6, \"N\": 0.4}"', written.splitlines()[1])

        reparsed = Structure.parse_xyz(written)
        self.assertEqual(reparsed.additional_fields["rss_composition"], structure.additional_fields["rss_composition"])
        self.assertEqual(reparsed.energy, structure.energy)
        np.testing.assert_array_equal(reparsed.additional_fields["stress"], structure.additional_fields["stress"])

    def test_xyz_export_digits_only_affect_atomic_float_fields(self):
        Config.set("nep", "data_precision", DataPrecision.FLOAT64)
        Config.set("io", "export_significant_digits", 8)
        precise_lattice = np.array(
            [
                [1.1234567890123457, 0.0, 0.0],
                [0.0, 2.2345678901234567, 0.0],
                [0.0, 0.0, 3.345678901234567],
            ],
            dtype=np.float64,
        )
        precise_positions = np.array(
            [[0.12345678901234567, 0.0, 0.0], [0.5, 0.5000000000000001, 0.5]],
            dtype=np.float64,
        )
        precise_forces = np.array(
            [[0.1111111111111111, 0.2222222222222222, 0.3333333333333333],
             [-0.4444444444444444, -0.5555555555555556, -0.6666666666666666]],
            dtype=np.float64,
        )
        precise_virial = np.array(
            [1.1234567890123457, 0.0, 0.0, 0.0, 2.2345678901234567, 0.0, 0.0, 0.0, 3.345678901234567],
            dtype=np.float64,
        )
        structure = Structure(
            precise_lattice,
            {
                "species": ["H", "O"],
                "pos": precise_positions,
                "forces": precise_forces,
            },
            self.properties,
            {
                "energy": 1.1234567890123457,
                "virial": precise_virial,
            },
        )

        handle = StringIO()
        structure.write(handle)
        lines = handle.getvalue().splitlines()

        self.assertIn('Lattice="1.1234567890123457', lines[1])
        self.assertIn('virial="1.1234567890123457', lines[1])
        self.assertIn("energy=1.1234567890123457", lines[1])
        self.assertEqual(lines[2], "H 0.12345679 0 0 0.11111111 0.22222222 0.33333333")
        self.assertEqual(lines[3], "O 0.5 0.5 0.5 -0.44444444 -0.55555556 -0.66666667")

    def test_xyz2npy(self):
        save_npy_structure("./npy", [self.structure])
        read_structure = load_npy_structure("./npy")[0]
        self.assertEqual(read_structure.lattice.dtype, np.float32)
        self.assertEqual(read_structure.positions.dtype, np.float32)
        self.assertEqual(read_structure.forces.dtype, np.float32)
        self.assertEqual(read_structure.virial.dtype, np.float32)
        np.testing.assert_array_equal(read_structure.lattice, self.lattice)
        np.testing.assert_array_equal(read_structure.positions, self.structure_info["pos"])
        np.testing.assert_array_equal(read_structure.elements, self.structure_info["species"])
        self.assertEqual(np.load("./npy/HO/set.000/box.npy").dtype, np.float32)
        self.assertEqual(np.load("./npy/HO/set.000/coord.npy").dtype, np.float32)
        self.assertEqual(np.load("./npy/HO/set.000/forces.npy").dtype, np.float32)
        self.assertEqual(np.load("./npy/HO/set.000/virial.npy").dtype, np.float32)
        self.assertEqual(np.load("./npy/HO/set.000/energy.npy").dtype, np.float32)

        import shutil

        shutil.rmtree("./npy")

    def test_xyz2npy_uses_float64_when_enabled(self):
        Config.set("nep", "data_precision", DataPrecision.FLOAT64)
        structure = Structure(self.lattice, self.structure_info, self.properties, self.additional_fields)
        save_npy_structure("./npy64", [structure])
        read_structure = load_npy_structure("./npy64")[0]
        self.assertEqual(read_structure.lattice.dtype, np.float64)
        self.assertEqual(read_structure.positions.dtype, np.float64)
        self.assertEqual(read_structure.forces.dtype, np.float64)
        self.assertEqual(read_structure.virial.dtype, np.float64)
        self.assertEqual(np.load("./npy64/HO/set.000/box.npy").dtype, np.float64)
        self.assertEqual(np.load("./npy64/HO/set.000/coord.npy").dtype, np.float64)
        self.assertEqual(np.load("./npy64/HO/set.000/forces.npy").dtype, np.float64)
        self.assertEqual(np.load("./npy64/HO/set.000/virial.npy").dtype, np.float64)
        self.assertEqual(np.load("./npy64/HO/set.000/energy.npy").dtype, np.float64)

        import shutil

        shutil.rmtree("./npy64")


if __name__ == "__main__":
    unittest.main()
