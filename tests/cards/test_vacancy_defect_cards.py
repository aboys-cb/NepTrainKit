"""Vacancy defect card."""

import warnings
from unittest.mock import patch

from ase.geometry import geometry
from PySide6.QtCore import Qt
from NepTrainKit.core.cards.errors import CardOperationError

from .card_test_base import *


class TestVacancyDefectCards(BaseCardTest):
    def test_vacancy_defect_card_concentration(self):
        card = VacancyDefectCard()
        structure = self.structure.copy()
        card.engine_type_combo.setCurrentIndex(
            card.engine_type_combo.findData(1)
        )
        card.amount_mode_control.setCurrentIndex(
            card.amount_mode_control.findData("fraction")
        )
        card.concentration_condition_frame.set_input_value([0.6])
        card.num_condition_frame.set_input_value([1])
        card.count_mode_control.setCurrentIndex(
            card.count_mode_control.findData("random")
        )
        card.max_atoms_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(atoms) < len(structure) for atoms in results))

    def test_vacancy_defect_card_defaults_and_previews_first_input(self):
        card = VacancyDefectCard()

        self.assertEqual(card.getTitle(), "Global Vacancy")
        self.assertFalse(card.num_condition_field.isHidden())
        self.assertTrue(card.concentration_condition_field.isHidden())
        self.assertEqual(card.get_params(), VacancyDefectParams())
        self.assertIn("Load an upstream structure", card.preview_label.text())

        structure = Atoms(
            "SiO4",
            positions=np.arange(15, dtype=float).reshape(5, 3),
            cell=[12, 12, 12],
            pbc=True,
        )
        card.set_dataset([structure])

        self.assertIn("First input: 5 atoms", card.preview_label.text())
        self.assertIn("remove 1 atoms", card.preview_label.text())
        self.assertIn("all elements eligible", card.preview_label.text())

        card.amount_mode_control.setCurrentIndex(
            card.amount_mode_control.findData("fraction")
        )
        card.concentration_condition_frame.set_input_value([0.4])
        self.assertTrue(card.num_condition_field.isHidden())
        self.assertFalse(card.concentration_condition_field.isHidden())
        self.assertFalse(card.get_params().use_num)
        self.assertIn("remove 2 atoms", card.preview_label.text())

    def test_vacancy_defect_amount_and_generation_modes_are_unambiguous(self):
        card = VacancyDefectCard()
        card.show()
        QApplication.processEvents()

        card.amount_mode_control.setCurrentIndex(
            card.amount_mode_control.findData("fraction")
        )
        card.count_mode_control.setCurrentIndex(
            card.count_mode_control.findData("random")
        )
        QApplication.processEvents()

        self.assertFalse(card.get_params().use_num)
        self.assertTrue(card.num_condition_field.isHidden())
        self.assertTrue(card.concentration_condition_frame.isVisible())
        self.assertEqual(
            card.concentration_condition_field.caption.text(),
            "Maximum vacancy fraction",
        )
        self.assertEqual(card.get_params().count_mode, "random")

        card.count_mode_control.setCurrentIndex(
            card.count_mode_control.findData("fixed")
        )
        self.assertEqual(
            card.concentration_condition_field.caption.text(),
            "Vacancy fraction",
        )

    def test_vacancy_defect_seed_value_is_progressively_disclosed(self):
        card = VacancyDefectCard()
        card.show()
        QApplication.processEvents()
        self.assertFalse(card.seed_frame.isVisible())

        card.seed_checkbox.setChecked(True)
        QApplication.processEvents()
        self.assertTrue(card.seed_frame.isVisible())
        self.assertTrue(card.seed_frame.isEnabled())

    def test_vacancy_defect_card_roundtrip_and_legacy_restore(self):
        card = VacancyDefectCard()
        expected = VacancyDefectParams(
            engine_type=0,
            num_condition=3,
            use_num=False,
            concentration_condition=0.125,
            count_mode="random",
            max_structures=12,
            use_seed=True,
            seed=17,
        )
        card.set_params(expected)

        restored = VacancyDefectCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), expected)

        legacy = VacancyDefectCard()
        legacy.from_dict(
            {
                "check_state": True,
                "engine_type": [0],
                "num_condition": [2],
                "num_radio_button": False,
                "concentration_condition": [0.25],
                "count_mode": "random",
                "max_atoms_condition": [7],
                "use_seed": True,
                "seed": [9],
            }
        )
        self.assertEqual(
            legacy.get_params(),
            VacancyDefectParams(
                engine_type=0,
                num_condition=2,
                use_num=False,
                concentration_condition=0.25,
                count_mode="random",
                max_structures=7,
                use_seed=True,
                seed=9,
            ),
        )

        legacy_without_count_mode = VacancyDefectCard()
        legacy_without_count_mode.from_dict(
            {
                "check_state": True,
                "engine_type": [1],
                "num_condition": [2],
                "num_radio_button": True,
                "max_atoms_condition": [3],
            }
        )
        self.assertEqual(
            legacy_without_count_mode.get_params().count_mode,
            "random",
        )

    def test_vacancy_defect_fixed_count(self):
        structure = Atoms("Si5", positions=np.arange(15, dtype=float).reshape(5, 3), cell=[10, 10, 10], pbc=True)

        results = VacancyDefectOperation().run_structure(
            structure,
            VacancyDefectParams(
                engine_type=1,
                num_condition=3,
                use_num=True,
                count_mode="fixed",
                max_structures=3,
                use_seed=True,
                seed=2,
            ),
        )

        self.assertTrue(all(len(atoms) == 2 for atoms in results))

    def test_vacancy_defect_fraction_uses_floor_and_random_mode_uses_range(self):
        structure = Atoms(
            "Si10",
            positions=np.arange(30, dtype=float).reshape(10, 3),
            cell=[20, 20, 20],
            pbc=True,
        )
        operation = VacancyDefectOperation()

        fixed = operation.run_structure(
            structure,
            VacancyDefectParams(
                use_num=False,
                concentration_condition=0.29,
                count_mode="fixed",
                max_structures=4,
                use_seed=True,
                seed=3,
            ),
        )
        self.assertEqual(len(fixed), 4)
        self.assertTrue(all(len(atoms) == 8 for atoms in fixed))

        random_results = operation.run_structure(
            structure,
            VacancyDefectParams(
                num_condition=4,
                use_num=True,
                count_mode="random",
                max_structures=20,
                use_seed=True,
                seed=4,
            ),
        )
        removed_counts = {len(structure) - len(atoms) for atoms in random_results}
        self.assertTrue(removed_counts.issubset({1, 2, 3, 4}))
        self.assertGreater(len(removed_counts), 1)

    def test_vacancy_defect_rejects_invalid_settings_instead_of_clamping(self):
        structure = Atoms(
            "Si5",
            positions=np.arange(15, dtype=float).reshape(5, 3),
            cell=[10, 10, 10],
            pbc=True,
        )
        cases = [
            (
                VacancyDefectParams(engine_type=99),
                "global_vacancy_invalid_engine",
            ),
            (
                VacancyDefectParams(count_mode="typo"),
                "global_vacancy_invalid_count_mode",
            ),
            (
                VacancyDefectParams(max_structures=0),
                "global_vacancy_invalid_output_limit",
            ),
            (
                VacancyDefectParams(num_condition=5),
                "global_vacancy_count_exceeds_atoms",
            ),
            (
                VacancyDefectParams(
                    use_num=False,
                    concentration_condition=0.19,
                ),
                "global_vacancy_fraction_too_small",
            ),
            (
                VacancyDefectParams(
                    use_num=False,
                    concentration_condition=1.0,
                ),
                "global_vacancy_invalid_fraction",
            ),
            (
                VacancyDefectParams(use_seed=True, seed=-1),
                "global_vacancy_negative_seed",
            ),
        ]

        for params, code in cases:
            with self.subTest(params=params):
                with self.assertRaises(CardOperationError) as caught:
                    VacancyDefectOperation().run_structure(structure, params)
                self.assertEqual(caught.exception.code, code)

    def test_vacancy_defect_deduplicates_when_request_exceeds_combinations(self):
        structure = Atoms(
            "SiO",
            positions=[[0, 0, 0], [1, 1, 1]],
            cell=[5, 5, 5],
            pbc=True,
        )

        results = VacancyDefectOperation().run_structure(
            structure,
            VacancyDefectParams(
                num_condition=1,
                max_structures=20,
                use_seed=True,
                seed=8,
            ),
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(
            {atoms.get_chemical_formula() for atoms in results},
            {"O", "Si"},
        )

    def test_vacancy_defect_sobol_is_quiet_and_has_explicit_size_limit(self):
        structure = Atoms(
            "Si8",
            positions=np.arange(24, dtype=float).reshape(8, 3),
            cell=[20, 20, 20],
            pbc=True,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = VacancyDefectOperation().run_structure(
                structure,
                VacancyDefectParams(
                    engine_type=0,
                    num_condition=2,
                    count_mode="random",
                    max_structures=3,
                    use_seed=True,
                    seed=5,
                ),
            )
        self.assertEqual(len(results), 3)
        self.assertFalse(
            any("balance properties of Sobol" in str(item.message) for item in caught)
        )

        oversized = Atoms(
            numbers=np.ones(21201, dtype=int),
            positions=np.zeros((21201, 3)),
        )
        with self.assertRaisesRegex(ValueError, "at most 21200 atoms"):
            VacancyDefectOperation().run_structure(
                oversized,
                VacancyDefectParams(engine_type=0),
            )

    def test_vacancy_defect_seed_is_reproducible_and_structure_specific(self):
        structure_a = Atoms(
            "Si8",
            positions=np.arange(24, dtype=float).reshape(8, 3),
            cell=[30, 30, 30],
            pbc=True,
        )
        structure_b = structure_a.copy()
        structure_b.positions[0, 0] += 0.125
        for structure in (structure_a, structure_b):
            structure.arrays["site_id"] = np.arange(len(structure))
        params = VacancyDefectParams(
            num_condition=3,
            max_structures=3,
            use_seed=True,
            seed=42,
        )
        operation = VacancyDefectOperation()

        first = operation.run_structure(structure_a, params)
        repeated = operation.run_structure(structure_a, params)
        second_structure = operation.run_structure(structure_b, params)
        first_ids = [atoms.arrays["site_id"].tolist() for atoms in first]
        repeated_ids = [atoms.arrays["site_id"].tolist() for atoms in repeated]
        second_ids = [atoms.arrays["site_id"].tolist() for atoms in second_structure]

        self.assertEqual(first_ids, repeated_ids)
        self.assertNotEqual(first_ids, second_ids)

    def test_vacancy_defect_preserves_remaining_geometry_and_arrays(self):
        structure = Atoms(
            "SiO4",
            positions=np.arange(15, dtype=float).reshape(5, 3),
            cell=[12, 13, 14],
            pbc=[True, False, True],
            info={"Config_type": "bulk"},
        )
        structure.arrays["site_id"] = np.arange(len(structure))
        structure.arrays["group"] = np.asarray(
            ["a", "b", "b", "a", "b"],
            dtype=object,
        )

        result = VacancyDefectOperation().run_structure(
            structure,
            VacancyDefectParams(
                num_condition=2,
                max_structures=1,
                use_seed=True,
                seed=6,
            ),
        )[0]

        kept_ids = result.arrays["site_id"]
        np.testing.assert_allclose(result.positions, structure.positions[kept_ids])
        np.testing.assert_allclose(result.cell.array, structure.cell.array)
        np.testing.assert_array_equal(result.pbc, structure.pbc)
        np.testing.assert_array_equal(
            result.arrays["group"],
            structure.arrays["group"][kept_ids],
        )
        self.assertIn("bulk", result.info["Config_type"])
        self.assertIn("Vac(n=2)", result.info["Config_type"])

