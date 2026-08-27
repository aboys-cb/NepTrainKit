import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
from ase import Atoms
from ase.build import fcc111

from NepTrainKit.core.cards.defect import StrictGSFEPathOperation, StrictGSFEPathParams
from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.core.cards.lattice import BainPathOperation, BainPathParams
from NepTrainKit.core.cards.structure import CrystalPrototypeBuilderOperation, CrystalPrototypeBuilderParams


class TestBenchmarkStage1Core(unittest.TestCase):
    def test_bain_constant_volume_and_count(self):
        atoms = Atoms(
            "H2",
            positions=[[0, 0, 0], [0.2, 0.3, 0.4]],
            cell=[[2, 0, 0], [0.5, 3, 0], [0.2, 0.4, 4]],
            pbc=True,
        )
        volume = atoms.get_volume()

        out = BainPathOperation().run_structure(
            atoms,
            BainPathParams(axis="z", ca_range=(0.8, 1.2, 0.2), mode="constant_volume"),
        )

        self.assertEqual(len(out), 3)
        for item in out:
            self.assertAlmostEqual(item.get_volume(), volume, places=12)
            self.assertIn("Bain(ax=z", item.info.get("Config_type", ""))

    def test_bain_free_c_only_changes_selected_cell_vector(self):
        atoms = Atoms("H", positions=[[0.1, 0.2, 0.3]], cell=[[2, 0, 0], [0.5, 3, 0], [0.2, 0.4, 4]], pbc=True)
        cell = atoms.cell.array.copy()

        out = BainPathOperation().run_structure(
            atoms,
            BainPathParams(axis="x", ca_range=(1.5, 1.5, 1.0), mode="free_c", scale_atoms=False),
        )[0]

        np.testing.assert_allclose(out.cell.array[0], cell[0] * 1.5)
        np.testing.assert_allclose(out.cell.array[1:], cell[1:])
        np.testing.assert_allclose(out.positions, atoms.positions)

    def test_bain_scale_volume_applies_volume_scan(self):
        atoms = Atoms("H", positions=[[0.1, 0.2, 0.3]], cell=[2, 3, 4], pbc=True)
        volume = atoms.get_volume()

        out = BainPathOperation().run_structure(
            atoms,
            BainPathParams(
                axis="z",
                ca_range=(1.2, 1.2, 1.0),
                mode="scale_volume",
                volume_scale_range=(0.5, 1.0, 0.5),
            ),
        )

        self.assertEqual(len(out), 2)
        np.testing.assert_allclose([item.get_volume() / volume for item in out], [0.5, 1.0], rtol=1e-12)
        self.assertTrue(all("mode=scale_volume" in item.info.get("Config_type", "") for item in out))

    def test_bain_rejects_invalid_axis_and_range(self):
        atoms = Atoms("H", positions=[[0, 0, 0]], cell=[1, 1, 1], pbc=True)
        with self.assertRaisesRegex(ValueError, "axis"):
            BainPathOperation().run_structure(atoms, BainPathParams(axis="q"))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "step"):
            BainPathOperation().run_structure(atoms, BainPathParams(ca_range=(1, 2, 0)))

    def test_strict_gsfe_zero_and_nonzero_displacements(self):
        atoms = Atoms(
            "H4",
            positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
            cell=[8, 8, 8],
            pbc=True,
        )
        params = StrictGSFEPathParams(
            plane_hkl=(0, 0, 1),
            slip_uvw=(1, 0, 0),
            displacement_range=(0.0, 0.5, 0.5),
            displacement_unit="angstrom",
            cut_mode="middle",
            wrap=False,
        )

        zero, shifted = StrictGSFEPathOperation().run_structure(atoms, params)

        np.testing.assert_allclose(zero.positions, atoms.positions)
        displacement = shifted.positions - atoms.positions
        np.testing.assert_allclose(displacement[:2], 0.0, atol=1e-12)
        np.testing.assert_allclose(displacement[2:], [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], atol=1e-12)
        self.assertIn("GSFE(hkl=001,uvw=100,d=0.5)", shifted.info.get("Config_type", ""))

    def test_strict_gsfe_rejects_a_slip_direction_with_normal_component(self):
        atoms = Atoms(
            "H4",
            positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
            cell=[8, 8, 8],
            pbc=True,
        )

        with self.assertRaisesRegex(ValueError, "must lie in the fault plane"):
            StrictGSFEPathOperation().run_structure(
                atoms,
                StrictGSFEPathParams(
                    plane_hkl=(0, 0, 1),
                    slip_uvw=(1, 0, 1),
                ),
            )

    def test_strict_gsfe_fraction_of_vector_uses_explicit_slip(self):
        atoms = Atoms("H4", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], cell=[8, 8, 8], pbc=True)

        shifted = StrictGSFEPathOperation().run_structure(
            atoms,
            StrictGSFEPathParams(
                plane_hkl=(0, 0, 1),
                slip_uvw=(1, 0, 0),
                displacement_range=(0.0, 0.25, 0.25),
                displacement_unit="fraction_of_vector",
                cut_mode="middle",
                wrap=False,
            ),
        )[1]

        displacement = shifted.positions - atoms.positions
        np.testing.assert_allclose(displacement[:2], 0.0, atol=1e-12)
        np.testing.assert_allclose(displacement[2:], [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], atol=1e-12)

    def test_strict_gsfe_cut_modes_and_wrap(self):
        atoms = Atoms("H4", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], cell=[4, 4, 4], pbc=True)
        op = StrictGSFEPathOperation()

        fractional = op.run_structure(
            atoms,
            StrictGSFEPathParams(
                plane_hkl=(0, 0, 1),
                slip_uvw=(1, 0, 0),
                displacement_range=(0.0, 1.0, 1.0),
                displacement_unit="angstrom",
                cut_mode="fractional",
                cut_fraction=0.25,
                wrap=False,
            ),
        )[1]
        moved = np.linalg.norm(fractional.positions - atoms.positions, axis=1) > 1e-12
        np.testing.assert_array_equal(moved, [False, True, True, True])

        layer = op.run_structure(
            atoms,
            StrictGSFEPathParams(
                plane_hkl=(0, 0, 1),
                slip_uvw=(1, 0, 0),
                displacement_range=(0.0, 1.0, 1.0),
                displacement_unit="angstrom",
                cut_mode="layer_index",
                layer_index=1,
                wrap=False,
            ),
        )[1]
        moved = np.linalg.norm(layer.positions - atoms.positions, axis=1) > 1e-12
        np.testing.assert_array_equal(moved, [False, False, True, True])

        wrapped = op.run_structure(
            atoms,
            StrictGSFEPathParams(
                plane_hkl=(0, 0, 1),
                slip_uvw=(1, 0, 0),
                displacement_range=(0.0, 5.0, 5.0),
                displacement_unit="angstrom",
                cut_mode="middle",
                wrap=True,
            ),
        )[1]
        self.assertTrue(np.all(wrapped.get_scaled_positions(wrap=False) >= -1e-12))
        self.assertTrue(np.all(wrapped.get_scaled_positions(wrap=False) < 1.0 + 1e-12))

    def test_strict_gsfe_middle_uses_an_interlayer_gap_and_previews_geometry(self):
        atoms = Atoms(
            "H6",
            positions=[
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 0, 2],
                [1, 0, 2],
            ],
            cell=[4, 4, 4],
            pbc=True,
        )
        params = StrictGSFEPathParams(
            displacement_range=(0.0, 0.5, 0.25),
            displacement_unit="angstrom",
            cut_mode="middle",
            wrap=False,
        )

        summary = StrictGSFEPathOperation.geometry_summary(atoms, params)

        self.assertEqual(summary["layer_count"], 3)
        self.assertEqual(summary["stationary_count"], 4)
        self.assertEqual(summary["moved_count"], 2)
        self.assertEqual(summary["output_count"], 3)
        self.assertEqual(summary["values"], (0.0, 0.25, 0.5))
        self.assertAlmostEqual(summary["cut_position"], 1.5)
        self.assertAlmostEqual(summary["slip_length"], 4.0)

    def test_strict_gsfe_rejects_invalid_geometry(self):
        atoms = Atoms("H3", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], cell=[5, 5, 5], pbc=True)
        op = StrictGSFEPathOperation()
        cases = [
            (StrictGSFEPathParams(plane_hkl=(0, 0, 0)), "gsfe_zero_plane"),
            (StrictGSFEPathParams(slip_uvw=(0, 0, 0)), "gsfe_zero_direction"),
            (
                StrictGSFEPathParams(plane_hkl=(0, 0, 1), slip_uvw=(0, 0, 1)),
                "gsfe_direction_out_of_plane",
            ),
            (
                StrictGSFEPathParams(cut_mode="layer_index", layer_index=2),
                "gsfe_invalid_layer_index",
            ),
            (
                StrictGSFEPathParams(plane_hkl=(0, 0, 1.5)),
                "gsfe_invalid_index_triplet",
            ),
            (
                StrictGSFEPathParams(
                    cut_mode="layer_index",
                    layer_index=0.5,  # type: ignore[arg-type]
                ),
                "gsfe_noninteger_value",
            ),
            (
                StrictGSFEPathParams(cut_mode="fractional", cut_fraction=1.0),
                "gsfe_empty_cut_side",
            ),
            (
                StrictGSFEPathParams(displacement_range=(0.0, 1.0, 0.0)),
                "gsfe_invalid_displacement_path",
            ),
        ]
        for params, code in cases:
            with self.subTest(params=params):
                with self.assertRaises(CardOperationError) as caught:
                    op.run_structure(atoms, params)
                self.assertEqual(caught.exception.code, code)
        single_layer = Atoms(
            "H2",
            positions=[[0, 0, 0], [1, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        )
        with self.assertRaises(CardOperationError) as caught:
            op.run_structure(single_layer, StrictGSFEPathParams())
        self.assertEqual(caught.exception.code, "gsfe_too_few_layers")

    def test_strict_gsfe_rejects_non_slab_oriented_plane(self):
        atoms = Atoms(
            "Ni4",
            scaled_positions=[[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
            cell=[3.6, 3.6, 3.6],
            pbc=True,
        )
        with self.assertRaises(CardOperationError) as caught:
            StrictGSFEPathOperation().run_structure(atoms, StrictGSFEPathParams(plane_hkl=(1, 1, 1)))
        self.assertEqual(caught.exception.code, "gsfe_cell_not_oriented")

    def test_strict_gsfe_fcc111_slab_geometry(self):
        atoms = fcc111("Ni", size=(2, 4, 6), a=3.6, vacuum=None, periodic=True, orthogonal=True)
        frames = StrictGSFEPathOperation().run_structure(
            atoms,
            StrictGSFEPathParams(
                plane_hkl=(0, 0, 1),
                slip_uvw=(1, 0, 0),
                displacement_range=(0.0, 1.0, 0.125),
            ),
        )

        self.assertEqual(len(frames), 9)
        for frame in frames:
            distances = frame.get_all_distances(mic=True)
            distances = distances[~np.eye(len(frame), dtype=bool)]
            self.assertGreater(float(distances.min()), 2.0)
            self.assertFalse(np.any(distances < 1e-6))
        np.testing.assert_allclose(frames[0].get_positions(), frames[-1].get_positions(), atol=1e-12)

    def test_strict_gsfe_reproduces_reference_fcc111_half_shift_path(self):
        atoms = fcc111(
            "Cu",
            size=(1, 2, 4),
            a=3.62,
            vacuum=None,
            periodic=True,
            orthogonal=True,
        )
        frames = StrictGSFEPathOperation().run_structure(
            atoms,
            StrictGSFEPathParams(
                plane_hkl=(0, 0, 1),
                slip_uvw=(0, 1, 0),
                displacement_range=(0.0, 1.47786, 0.073893),
                displacement_unit="angstrom",
                cut_mode="fractional",
                cut_fraction=0.5,
                wrap=False,
            ),
        )

        self.assertEqual(len(frames), 21)
        displacement = frames[-1].positions - atoms.positions
        np.testing.assert_allclose(displacement[:4], 0.0, atol=1e-12)
        np.testing.assert_allclose(
            displacement[4:],
            np.tile([0.0, 1.47786, 0.0], (4, 1)),
            atol=1e-12,
        )
        self.assertAlmostEqual(1.47786, 3.62 / np.sqrt(6.0), places=5)

    def test_crystal_prototype_builder_fcc111(self):
        frames = CrystalPrototypeBuilderOperation().generate(
            CrystalPrototypeBuilderParams(
                lattice="fcc111",
                element="Ni",
                a_range=(3.6, 3.6, 0.1),
                max_outputs=1,
            )
        )

        self.assertEqual(len(frames), 1)
        self.assertEqual(len(frames[0]), 6)
        self.assertAlmostEqual(frames[0].get_volume() / len(frames[0]), 11.664, places=12)
        self.assertAlmostEqual(frames[0].cell.angles()[0], 90.0, places=12)
        self.assertIn("Proto(fcc111", frames[0].info.get("Config_type", ""))


if __name__ == "__main__":
    unittest.main()
