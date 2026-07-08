import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
from ase import Atoms

from NepTrainKit.core.cards.defect import StrictGSFEPathOperation, StrictGSFEPathParams
from NepTrainKit.core.cards.lattice import BainPathOperation, BainPathParams


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

    def test_strict_gsfe_projected_slip_is_in_plane(self):
        cell = np.array([[2.0, 0.0, 0.0], [0.5, 2.5, 0.0], [0.1, 0.2, 3.0]])
        hkl = (1, 1, 1)
        uvw = (1, 1, -2)
        normal = StrictGSFEPathOperation.plane_normal(cell, hkl)
        slip = np.asarray(uvw, dtype=float) @ cell
        projected = slip - np.dot(slip, normal) * normal

        self.assertAlmostEqual(float(np.dot(projected, normal)), 0.0, places=12)

    def test_strict_gsfe_fraction_of_vector_uses_projected_slip(self):
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

    def test_strict_gsfe_rejects_invalid_geometry(self):
        atoms = Atoms("H3", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], cell=[5, 5, 5], pbc=True)
        op = StrictGSFEPathOperation()
        with self.assertRaisesRegex(ValueError, "plane_hkl"):
            op.run_structure(atoms, StrictGSFEPathParams(plane_hkl=(0, 0, 0)))
        with self.assertRaisesRegex(ValueError, "slip_uvw"):
            op.run_structure(atoms, StrictGSFEPathParams(slip_uvw=(0, 0, 0)))
        with self.assertRaisesRegex(ValueError, "parallel"):
            op.run_structure(atoms, StrictGSFEPathParams(plane_hkl=(0, 0, 1), slip_uvw=(0, 0, 1)))
        with self.assertRaisesRegex(ValueError, "layer_index"):
            op.run_structure(atoms, StrictGSFEPathParams(cut_mode="layer_index", layer_index=2))


if __name__ == "__main__":
    unittest.main()
