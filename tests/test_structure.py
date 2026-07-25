#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest
import os
import tempfile
import threading
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

os.environ["LOCALAPPDATA"] = str(Path(__file__).resolve().parent / "_localappdata")

from NepTrainKit.config import Config
from NepTrainKit.core.precision import get_storage_precision
import NepTrainKit.core.structure as structure_module
from NepTrainKit.core.structure import (
    FastStructure,
    Structure,
    load_npy_structure,
    save_npy_structure,
    write_structures_extxyz_atomic,
)
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
            "pbc": "T T T",
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

        with patch.object(structure_module, "_native_audit", None), patch.object(
            large_structure, "get_all_distances"
        ) as get_all_distances:
            self.assertEqual(large_structure.get_bad_bond_pairs(max_atoms=500), [])

        get_all_distances.assert_not_called()

    def test_bad_bond_scan_keeps_complete_analysis_by_default(self):
        with patch.object(structure_module, "_native_audit", None), patch.object(
            self.structure,
            "get_all_distances",
            return_value=np.array([[0.0, 0.1], [0.1, 0.0]]),
        ) as get_all_distances:
            self.assertEqual(self.structure.get_bad_bond_pairs(), [(0, 1)])

        get_all_distances.assert_called_once_with()

    def test_bad_bond_scan_uses_native_pairs_without_size_limit(self):
        native_scan = MagicMock()
        native_scan.scaled_radii_collision_pairs.return_value = np.array(
            [[0, 1]], dtype=np.int32
        )
        large_structure = Structure(
            np.eye(3) * 100,
            {
                "species": ["H"] * 501,
                "pos": np.arange(1503, dtype=np.float32).reshape(501, 3),
            },
            [
                {"name": "species", "type": "S", "count": 1},
                {"name": "pos", "type": "R", "count": 3},
            ],
            {},
        )

        with patch.object(structure_module, "_native_audit", native_scan), patch.object(
            large_structure, "get_all_distances"
        ) as get_all_distances:
            pairs = large_structure.get_bad_bond_pairs(max_atoms=500)

        self.assertEqual(pairs, [(0, 1)])
        native_scan.scaled_radii_collision_pairs.assert_called_once()
        get_all_distances.assert_not_called()

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

    def test_atomic_extxyz_write_preserves_existing_target_on_failure(self):
        class BrokenStructure:
            def write(self, handle, **_kwargs):
                handle.write("partial replacement")
                raise RuntimeError("injected write failure")

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "existing.xyz"
            target.write_text("trusted original", encoding="utf8")

            with self.assertRaisesRegex(RuntimeError, "injected write failure"):
                write_structures_extxyz_atomic(target, [BrokenStructure()])

            self.assertEqual(target.read_text(encoding="utf8"), "trusted original")
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

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

    def test_xyz_logical_fields_preserve_false_and_true_in_both_readers(self):
        text = """2
Lattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3:mask:L:1 pbc="T T T"
H 0 0 0 F
He 1 1 1 T
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "logical.extxyz"
            path.write_text(text, encoding="utf-8")

            with patch.dict(os.environ, {"NEPKIT_DISABLE_FASTXYZ": "1"}):
                python_result = Structure.read_multiple_fast(str(path))
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("NEPKIT_DISABLE_FASTXYZ", None)
                native_result = Structure.read_multiple_fast(str(path))

        np.testing.assert_array_equal(
            python_result[0].atomic_properties["mask"],
            np.array([False, True]),
        )
        np.testing.assert_array_equal(
            native_result[0].atomic_properties["mask"],
            np.array([False, True]),
        )

    def test_xyz_readers_reject_truncated_and_malformed_atom_rows(self):
        damaged_frames = (
            """2
Lattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3 pbc="T T T"
H 0 0 0
""",
            """1
Lattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3:force:R:3 pbc="T T T"
H 0 0 0 1 2
""",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            for index, text in enumerate(damaged_frames):
                path = Path(tmp_dir) / f"damaged-{index}.extxyz"
                path.write_text(text, encoding="utf-8")
                with patch.dict(os.environ, {"NEPKIT_DISABLE_FASTXYZ": "1"}):
                    with self.assertRaises(ValueError):
                        Structure.read_multiple_fast(str(path))
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("NEPKIT_DISABLE_FASTXYZ", None)
                    with self.assertRaises(ValueError):
                        Structure.read_multiple_fast(str(path))

    def test_xyz_reader_rejects_invalid_element_pbc_and_periodic_cell(self):
        invalid_frames = (
            """1
Lattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3 pbc="T T T"
Qq 0 0 0
""",
            """1
Lattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3 pbc="maybe"
H 0 0 0
""",
            """1
Lattice="0 0 0 0 0 0 0 0 0" Properties=species:S:1:pos:R:3 pbc="T T T"
H 0 0 0
""",
        )
        for text in invalid_frames:
            with self.assertRaises(ValueError):
                Structure.parse_xyz(text)

    def test_xyz_missing_pbc_defaults_to_nep_periodic_contract(self):
        structure = Structure.parse_xyz(
            """1
Lattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3
H 0 0 0
"""
        )

        self.assertEqual(structure.additional_fields["pbc"], "T T T")

    def test_fast_structure_iterator_and_pre_canceled_import(self):
        text = """1
Lattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3 pbc="T T T"
H 0 0 0
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "one.extxyz"
            path.write_text(text, encoding="utf-8")
            with patch.dict(os.environ, {"NEPKIT_DISABLE_FASTXYZ": "1"}):
                self.assertEqual(len(list(FastStructure.iter_read_multiple(str(path)))), 1)

            cancel_event = threading.Event()
            cancel_event.set()
            self.assertEqual(
                Structure.read_multiple_fast(str(path), cancel_event=cancel_event),
                [],
            )

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

    def test_deepmd_roundtrip_preserves_pbc_dtypes_metadata_and_alignment(self):
        properties = [
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
            {"name": "forces", "type": "R", "count": 3},
            {"name": "magmom", "type": "R", "count": 1},
            {"name": "group", "type": "I", "count": 1},
            {"name": "mask", "type": "L", "count": 1},
            {"name": "sublattice", "type": "S", "count": 1},
        ]
        structure = Structure(
            np.eye(3) * 4,
            {
                "species": ["O", "H"],
                "pos": np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                "forces": np.array([[20.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
                "magmom": np.array([2.0, 1.0]),
                "group": np.array([20, 10], dtype=np.int32),
                "mask": np.array([False, True]),
                "sublattice": np.array(["B", "A"]),
            },
            properties,
            {
                "Config_type": "phase/nonperiodic",
                "pbc": "F F F",
                "energy": -2.5,
                "virial": np.arange(9, dtype=float),
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "deepmd"
            save_npy_structure(root, [structure], type_map=["H", "O"])
            self.assertTrue((root / "phase" / "nonperiodic" / "nopbc").is_file())
            loaded = load_npy_structure(root)

        self.assertEqual(len(loaded), 1)
        restored = loaded[0]
        self.assertEqual(restored.tag, "phase/nonperiodic")
        self.assertEqual(restored.additional_fields["pbc"], "F F F")
        self.assertEqual(restored.elements.tolist(), ["H", "O"])
        np.testing.assert_allclose(restored.positions[:, 0], [1.0, 2.0])
        np.testing.assert_allclose(restored.forces[:, 0], [10.0, 20.0])
        np.testing.assert_allclose(restored.atomic_properties["magmom"], [1.0, 2.0])
        np.testing.assert_array_equal(restored.atomic_properties["group"], [10, 20])
        np.testing.assert_array_equal(restored.atomic_properties["mask"], [True, False])
        np.testing.assert_array_equal(restored.atomic_properties["sublattice"], ["A", "B"])
        self.assertEqual(restored.atomic_properties["group"].dtype, np.int32)
        self.assertEqual(restored.atomic_properties["mask"].dtype, np.bool_)
        self.assertEqual(restored.atomic_properties["sublattice"].dtype.kind, "U")
        property_types = {prop["name"]: prop["type"] for prop in restored.properties}
        self.assertEqual(property_types["group"], "I")
        self.assertEqual(property_types["mask"], "L")
        self.assertEqual(property_types["sublattice"], "S")
        self.assertAlmostEqual(restored.energy, -2.5)
        np.testing.assert_allclose(restored.virial, np.arange(9, dtype=float))

    def test_deepmd_single_atom_energy_remains_a_frame_label(self):
        structure = Structure(
            np.eye(3),
            {"species": ["H"], "pos": np.zeros((1, 3))},
            [
                {"name": "species", "type": "S", "count": 1},
                {"name": "pos", "type": "R", "count": 3},
            ],
            {"Config_type": "single", "pbc": "T T T", "energy": -1.25},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "deepmd"
            save_npy_structure(root, [structure])
            restored = load_npy_structure(root)[0]

        self.assertAlmostEqual(restored.energy, -1.25)
        self.assertNotIn("energy", restored.atomic_properties)

    def test_deepmd_import_fails_closed_for_missing_species_or_required_arrays(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "missing_species"
            set_dir = root / "set.000"
            set_dir.mkdir(parents=True)
            (root / "type.raw").write_text("0\n", encoding="utf8")
            np.save(set_dir / "box.npy", np.eye(3).reshape(1, 9))
            np.save(set_dir / "coord.npy", np.zeros((1, 3)))

            with self.assertRaisesRegex(ValueError, "missing type_map.raw"):
                load_npy_structure(root)

            (root / "type_map.raw").write_text("H\n", encoding="utf8")
            (set_dir / "box.npy").unlink()
            with self.assertRaisesRegex(ValueError, "missing required arrays: box.npy"):
                load_npy_structure(root)

    def test_deepmd_import_rejects_frame_and_atom_count_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "mismatched"
            set_dir = root / "set.000"
            set_dir.mkdir(parents=True)
            (root / "type.raw").write_text("0\n1\n", encoding="utf8")
            (root / "type_map.raw").write_text("H\nO\n", encoding="utf8")
            np.save(set_dir / "box.npy", np.tile(np.eye(3).reshape(1, 9), (2, 1)))
            np.save(set_dir / "coord.npy", np.zeros((1, 6)))

            with self.assertRaisesRegex(ValueError, "inconsistent frame counts"):
                load_npy_structure(root)

            np.save(set_dir / "box.npy", np.eye(3).reshape(1, 9))
            np.save(set_dir / "coord.npy", np.zeros((1, 3)))
            with self.assertRaisesRegex(ValueError, "type.raw declares 2 atoms"):
                load_npy_structure(root)

    def test_deepmd_export_fails_closed_for_unrepresentable_pbc_and_mixed_contracts(self):
        base = Structure(
            np.eye(3),
            {"species": ["H"], "pos": np.zeros((1, 3))},
            [
                {"name": "species", "type": "S", "count": 1},
                {"name": "pos", "type": "R", "count": 3},
            ],
            {"Config_type": "same", "pbc": "T T F"},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "partial"
            with self.assertRaisesRegex(ValueError, "partial PBC"):
                save_npy_structure(target, [base])
            self.assertFalse(target.exists())

            periodic = Structure(
                np.eye(3),
                {"species": ["H"], "pos": np.zeros((1, 3))},
                base.properties,
                {"Config_type": "same", "pbc": "T T T"},
            )
            nonperiodic = Structure(
                np.eye(3),
                {"species": ["H"], "pos": np.zeros((1, 3))},
                base.properties,
                {"Config_type": "same", "pbc": "F F F"},
            )
            target = Path(tmp_dir) / "mixed"
            with self.assertRaisesRegex(ValueError, "must share"):
                save_npy_structure(target, [periodic, nonperiodic])
            self.assertFalse(target.exists())

            different_species = Structure(
                np.eye(3),
                {"species": ["He"], "pos": np.zeros((1, 3))},
                base.properties,
                {"Config_type": "same", "pbc": "T T T"},
            )
            target = Path(tmp_dir) / "composition"
            with self.assertRaisesRegex(ValueError, "must share"):
                save_npy_structure(target, [periodic, different_species])
            self.assertFalse(target.exists())

    def test_deepmd_export_requires_explicit_pbc(self):
        structure = Structure(
            np.eye(3),
            {"species": ["H"], "pos": np.zeros((1, 3))},
            [
                {"name": "species", "type": "S", "count": 1},
                {"name": "pos", "type": "R", "count": 3},
            ],
            {"Config_type": "missing-pbc"},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "deepmd"
            with self.assertRaisesRegex(ValueError, "explicit PBC"):
                save_npy_structure(target, [structure])
            self.assertFalse(target.exists())

    def test_deepmd_export_io_failure_preserves_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            parent = Path(tmp_dir)
            target = parent / "deepmd"
            target.mkdir()
            sentinel = target / "existing.txt"
            sentinel.write_text("previous dataset", encoding="utf8")
            real_save = np.save
            call_count = 0

            def fail_on_third_array(path, array, *args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise OSError("simulated disk failure")
                return real_save(path, array, *args, **kwargs)

            with patch.object(structure_module.np, "save", side_effect=fail_on_third_array):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    save_npy_structure(target, [self.structure])

            self.assertEqual(sentinel.read_text(encoding="utf8"), "previous dataset")
            self.assertEqual([item.name for item in parent.iterdir()], ["deepmd"])

    def test_deepmd_export_success_replaces_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "deepmd"
            target.mkdir()
            (target / "stale.txt").write_text("old", encoding="utf8")

            save_npy_structure(target, [self.structure])

            self.assertFalse((target / "stale.txt").exists())
            restored = load_npy_structure(target)
            self.assertEqual(len(restored), 1)
            np.testing.assert_allclose(restored[0].positions, self.structure.positions)


if __name__ == "__main__":
    unittest.main()
