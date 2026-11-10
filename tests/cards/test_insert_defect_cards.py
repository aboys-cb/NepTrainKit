"""Insert/interstitial defect card."""

import warnings
from unittest.mock import patch

from ase.geometry import geometry
from PySide6.QtCore import Qt
from NepTrainKit.core.cards.errors import CardOperationError

from .card_test_base import *


class TestInsertDefectCards(BaseCardTest):
    def test_insert_defect_fast_nearest_distance_matches_ase(self):
        structure = self.structure.copy()
        candidate = np.asarray(structure.get_positions()[0], dtype=float) + np.array([0.2, 0.1, 0.0])

        nearest = InsertDefectOperation._nearest_distance(
            candidate,
            np.asarray(structure.get_positions(), dtype=float),
            cell=np.asarray(structure.cell.array, dtype=float),
            pbc=np.asarray(structure.pbc, dtype=bool),
        )
        _, dists = geometry.get_distances(
            candidate,
            np.asarray(structure.get_positions(), dtype=float),
            cell=np.asarray(structure.cell.array, dtype=float),
            pbc=np.asarray(structure.pbc, dtype=bool),
        )

        self.assertAlmostEqual(nearest, float(np.min(dists)), places=12)

    def test_insert_defect_interstitial_preserves_host_and_enforces_distance(self):
        structure = Atoms(
            "Si2",
            positions=[[10.2, 0.0, 0.0], [3.0, 3.0, 3.0]],
            cell=[10, 10, 10],
            pbc=True,
            info={"Config_type": "bulk"},
        )
        structure.arrays["site_id"] = np.arange(len(structure))
        original_positions = structure.positions.copy()

        result = InsertDefectOperation().run_structure(
            structure,
            InsertDefectParams(
                mode=0,
                species="H",
                insert_count=2,
                structure_count=1,
                min_distance=0.8,
                max_attempts=200,
                use_seed=True,
                seed=7,
            ),
        )[0]

        self.assertEqual(result.get_chemical_symbols()[-2:], ["H", "H"])
        np.testing.assert_allclose(result.positions[: len(structure)], original_positions)
        np.testing.assert_allclose(result.cell.array, structure.cell.array)
        np.testing.assert_array_equal(result.pbc, structure.pbc)
        np.testing.assert_array_equal(
            result.arrays["site_id"][: len(structure)],
            structure.arrays["site_id"],
        )
        for added_index in range(len(structure), len(result)):
            reference = np.delete(result.positions, added_index, axis=0)
            nearest = InsertDefectOperation._nearest_distance(
                result.positions[added_index],
                reference,
                cell=result.cell.array,
                pbc=result.pbc,
            )
            self.assertGreaterEqual(nearest, 0.8 - 1e-12)
        self.assertIn("bulk", result.info["Config_type"])
        self.assertIn("Ins(int,n=2)", result.info["Config_type"])

    def test_insert_defect_species_weights_are_strict_and_normalized(self):
        structure = Atoms(
            "Si2",
            positions=[[0, 0, 0], [2, 2, 2]],
            cell=[8, 8, 8],
            pbc=True,
        )
        operation = InsertDefectOperation()
        summary = operation.sampling_summary(
            structure,
            InsertDefectParams(species="Li:2, Na:3, Li:3"),
        )

        self.assertEqual(summary["species"], ["Li", "Na"])
        np.testing.assert_allclose(summary["weights"], [0.625, 0.375])

        cases = [
            ("", "at least one element"),
            ("Xx", "unknown chemical element"),
            ("Li:bad", "invalid weight"),
            ("Li:0", "finite and positive"),
            ("Li:nan", "finite and positive"),
        ]
        for species, message in cases:
            with self.subTest(species=species):
                with self.assertRaisesRegex(ValueError, message):
                    operation.run_structure(
                        structure,
                        InsertDefectParams(
                            species=species,
                            structure_count=1,
                        ),
                    )

    def test_insert_defect_rejects_invalid_settings(self):
        structure = Atoms(
            "Si2",
            positions=[[0, 0, 0], [2, 2, 2]],
            cell=[8, 8, 8],
            pbc=True,
        )
        cases = [
            (InsertDefectParams(mode=2, species="H"), "mode must be"),
            (
                InsertDefectParams(
                    mode=1.5,
                    species="H",
                ),
                "mode must be",
            ),
            (
                InsertDefectParams(
                    species="H",
                    insert_count=0,
                ),
                "insert_count must be >= 1",
            ),
            (
                InsertDefectParams(
                    species="H",
                    insert_count=1.5,
                ),
                "insert_count must be an integer",
            ),
            (
                InsertDefectParams(
                    species="H",
                    structure_count=0,
                ),
                "structure_count must be >= 1",
            ),
            (
                InsertDefectParams(
                    species="H",
                    min_distance=0.0,
                ),
                "min_distance must be finite and positive",
            ),
            (
                InsertDefectParams(
                    species="H",
                    max_attempts=0,
                ),
                "max_attempts must be >= 1",
            ),
            (
                InsertDefectParams(
                    mode=1,
                    species="H",
                    axis=3,
                ),
                "axis must be 0, 1, or 2",
            ),
            (
                InsertDefectParams(
                    mode=1,
                    species="H",
                    offset=0.0,
                ),
                "adsorption height must be finite and positive",
            ),
            (
                InsertDefectParams(
                    species="H",
                    use_seed=True,
                    seed=-1,
                ),
                "seed must be >= 0",
            ),
        ]
        operation = InsertDefectOperation()
        for params, message in cases:
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, message):
                    operation.run_structure(structure, params)

        with self.assertRaisesRegex(ValueError, "at least one host atom"):
            operation.run_structure(
                Atoms(cell=[8, 8, 8], pbc=True),
                InsertDefectParams(species="H"),
            )
        with self.assertRaisesRegex(ValueError, "non-singular"):
            operation.run_structure(
                Atoms("Si", positions=[[0, 0, 0]]),
                InsertDefectParams(species="H"),
            )

    def test_insert_defect_failed_placement_raises_instead_of_returning_partial(self):
        structure = Atoms(
            "Si",
            positions=[[0, 0, 0]],
            cell=[1, 1, 1],
            pbc=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "could not place atom 1 of 2 for output 1",
        ):
            InsertDefectOperation().run_structure(
                structure,
                InsertDefectParams(
                    species="H",
                    insert_count=2,
                    structure_count=3,
                    min_distance=10.0,
                    max_attempts=2,
                    use_seed=True,
                    seed=2,
                ),
            )

    def test_insert_defect_adsorbates_share_original_host_plane(self):
        structure = Atoms(
            "Si2",
            positions=[[1, 1, 1], [3, 3, 2]],
            cell=[5, 5, 12],
            pbc=[True, True, False],
        )

        result = InsertDefectOperation().run_structure(
            structure,
            InsertDefectParams(
                mode=1,
                species="H",
                insert_count=3,
                structure_count=1,
                min_distance=0.2,
                max_attempts=100,
                use_seed=True,
                seed=7,
                axis=2,
                offset=1.5,
            ),
        )[0]

        np.testing.assert_allclose(
            result.positions[len(structure) :, 2],
            [3.5, 3.5, 3.5],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.positions[: len(structure)],
            structure.positions,
        )
        self.assertIn("Ins(ad,n=3)", result.info["Config_type"])

    def test_insert_defect_adsorption_uses_true_normal_for_skew_cell(self):
        cell = np.asarray(
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 0.0, 6.0]]
        )
        structure = Atoms(
            "Si2",
            scaled_positions=[[0.2, 0.2, 0.2], [0.8, 0.8, 0.3]],
            cell=cell,
            pbc=[True, True, False],
        )
        offset = 1.5

        result = InsertDefectOperation().run_structure(
            structure,
            InsertDefectParams(
                mode=1,
                species="H",
                insert_count=2,
                structure_count=1,
                min_distance=0.2,
                max_attempts=100,
                use_seed=True,
                seed=4,
                axis=2,
                offset=offset,
            ),
        )[0]

        inverse_cell = np.linalg.inv(cell)
        normal = inverse_cell[:, 2].copy()
        normal /= np.linalg.norm(normal)
        top_fraction = (
            structure.positions @ inverse_cell
        )[:, 2].max()
        plane_point = np.asarray([0.0, 0.0, top_fraction]) @ cell
        heights = (
            result.positions[len(structure) :] - plane_point
        ) @ normal
        np.testing.assert_allclose(heights, offset, atol=1e-12)

    def test_insert_defect_seed_is_reproducible_and_structure_specific(self):
        structure_a = Atoms(
            "Si2",
            positions=[[0, 0, 0], [2, 2, 2]],
            cell=[10, 10, 10],
            pbc=True,
        )
        structure_b = structure_a.copy()
        structure_b.positions[0, 0] += 0.125
        params = InsertDefectParams(
            species="H",
            insert_count=2,
            structure_count=2,
            min_distance=0.01,
            use_seed=True,
            seed=42,
        )
        operation = InsertDefectOperation()

        first = operation.run_structure(structure_a, params)
        repeated = operation.run_structure(structure_a, params)
        second_structure = operation.run_structure(structure_b, params)
        first_added = [
            atoms.positions[len(structure_a) :].tolist()
            for atoms in first
        ]
        repeated_added = [
            atoms.positions[len(structure_a) :].tolist()
            for atoms in repeated
        ]
        second_added = [
            atoms.positions[len(structure_b) :].tolist()
            for atoms in second_structure
        ]

        self.assertEqual(first_added, repeated_added)
        self.assertNotEqual(first_added, second_added)

    def test_insert_defect_card_defaults_mode_visibility_and_preview(self):
        card = InsertDefectCard()
        self.assertEqual(
            card.getTitle(),
            "Insert / Adsorb",
        )
        self.assertEqual(card.get_params(), InsertDefectParams())
        self.assertTrue(card.axis_label.isHidden())
        self.assertTrue(card.max_attempts_field.isHidden())
        self.assertFalse(card.seed_frame.isVisible())
        self.assertIn(
            "Enter at least one inserted species",
            card.preview_label.text(),
        )

        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[12, 12, 12],
            pbc=True,
        )
        card.species_edit.setText("Li:7, Na:3")
        card.set_dataset([structure])
        self.assertIn("First input: 4 atoms", card.preview_label.text())
        self.assertIn("Li 70% / Na 30%", card.preview_label.text())
        self.assertIn("random positions inside the cell", card.preview_label.text())
        self.assertIn("1 inputs × 10 = 10 outputs", card.preview_label.text())
        self.assertIn("1 inputs × 10 outputs/input = 10 outputs", card.get_guidance_text())
        self.assertIn("Bulk interstitial", card.get_summary_text())

        card.mode_combo.setCurrentIndex(card.mode_combo.findData(1))
        self.assertFalse(card.axis_label.isHidden())
        self.assertIn("upper surface along lattice c", card.preview_label.text())
        self.assertIn("height 1.5 Å", card.preview_label.text())
        self.assertIn("Choose the slab vacuum direction", card.get_guidance_text())

        card.advanced_checkbox.setChecked(True)
        self.assertFalse(card.max_attempts_field.isHidden())

    def test_insert_defect_card_roundtrip(self):
        card = InsertDefectCard()
        card.mode_combo.setCurrentIndex(card.mode_combo.findData(1))
        card.species_edit.setText("H:2, O:1")
        card.insert_count_frame.set_input_value([2])
        card.structures_frame.set_input_value([3])
        card.min_distance_frame.set_input_value([0.5])
        card.max_attempts_frame.set_input_value([50])
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([13])
        card.axis_combo.setCurrentIndex(2)
        card.offset_frame.set_input_value([2.0])

        restored = InsertDefectCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

        legacy = InsertDefectCard()
        legacy.from_dict(
            {
                "check_state": True,
                "mode": [1],
                "species": "O",
                "insert_count": [2],
                "structure_count": [4],
                "min_distance": [1.2],
                "max_attempts": [80],
                "use_seed": True,
                "seed": [9],
                "axis": [1],
                "offset": [1.8],
            }
        )
        self.assertEqual(
            legacy.get_params(),
            InsertDefectParams(
                mode=1,
                species="O",
                insert_count=2,
                structure_count=4,
                min_distance=1.2,
                max_attempts=80,
                use_seed=True,
                seed=9,
                axis=1,
                offset=1.8,
            ),
        )

