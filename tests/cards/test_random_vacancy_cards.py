"""Random vacancy card."""

import warnings
from unittest.mock import patch

from ase.geometry import geometry
from PySide6.QtCore import Qt
from NepTrainKit.core.cards.errors import CardOperationError

from .card_test_base import *


class TestRandomVacancyCards(BaseCardTest):
    def test_random_vacancy_card(self):
        card = RandomVacancyCard()
        structure = self.structure.copy()
        card.rules_widget.from_rules([
            {"element": "Si", "count": [1, 1], "count_mode": "fixed"},
        ])
        card.max_atoms_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(atoms) == len(structure) - 1 for atoms in results))
        self.assertTrue(
            all("Vac(n=1)" in atoms.info.get("Config_type", "") for atoms in results)
        )

    def test_random_vacancy_card_defaults_to_one_editable_rule_and_previews_matches(self):
        card = RandomVacancyCard()
        self.assertEqual(len(card.rules_widget.rule_items()), 1)
        self.assertEqual(card.rules_widget.to_rules(), [])

        structure = Atoms(
            "SiO3",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[12, 12, 12],
            pbc=True,
        )
        structure.arrays["group"] = np.asarray(["bulk", "surface", "surface", "bulk"], dtype=object)
        card.rules_widget.from_rules(
            [
                {
                    "element": "O",
                    "group": ["surface"],
                    "count": [1, 1],
                    "count_mode": "fixed",
                }
            ]
        )
        card.max_atoms_condition_frame.set_input_value([20])
        card.set_dataset([structure])

        self.assertIn("2 matches", card.preview_label.text())
        self.assertIn("O/surface", card.preview_label.text())
        self.assertIn("up to 2 unique outputs", card.preview_label.text())

    def test_random_vacancy_card_roundtrip_accepts_legacy_scalar_and_group_string(self):
        card = RandomVacancyCard()
        card.from_dict(
            {
                "class": "RandomVacancyCard",
                "check_state": True,
                "rules": '[{"element":"O","group":"surface","count":[1,2],"count_mode":"random"}]',
                "max_atoms_condition": 7,
                "use_seed": True,
                "seed": 19,
            }
        )

        params = card.get_params()
        self.assertEqual(params.rules[0]["group"], ["surface"])
        self.assertEqual(params.max_structures, 7)
        self.assertTrue(params.use_seed)
        self.assertEqual(params.seed, 19)
        self.assertTrue(card.seed_frame.isEnabled())

        restored = RandomVacancyCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), params)

    def test_random_vacancy_operation_fixed_count(self):
        structure = Atoms("Si5", positions=np.arange(15, dtype=float).reshape(5, 3), cell=[10, 10, 10], pbc=True)

        results = RandomVacancyOperation().run_structure(
            structure,
            RandomVacancyParams(
                rules=[{"element": "Si", "count": [2, 2], "count_mode": "fixed"}],
                max_structures=4,
                use_seed=True,
                seed=1,
            ),
        )

        self.assertTrue(all(len(atoms) == 3 for atoms in results))
        self.assertEqual(len({tuple(atoms.positions.ravel()) for atoms in results}), 4)

    def test_random_vacancy_operation_requires_requested_group_array(self):
        structure = Atoms(
            "O4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[10, 10, 10],
            pbc=True,
        )

        with self.assertRaisesRegex(ValueError, "has no group array"):
            RandomVacancyOperation().run_structure(
                structure,
                RandomVacancyParams(
                    rules=[
                        {
                            "element": "O",
                            "group": ["surface"],
                            "count": [1, 1],
                            "count_mode": "fixed",
                        }
                    ],
                ),
            )

    def test_random_vacancy_operation_limits_deletion_to_existing_group(self):
        structure = Atoms(
            "O4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[10, 10, 10],
            pbc=True,
        )
        structure.arrays["group"] = np.asarray(
            ["surface", "surface", "bulk", "bulk"],
            dtype=object,
        )
        structure.arrays["site_id"] = np.arange(4)

        results = RandomVacancyOperation().run_structure(
            structure,
            RandomVacancyParams(
                rules=[
                    {
                        "element": "O",
                        "group": ["surface"],
                        "count": [1, 1],
                        "count_mode": "fixed",
                    }
                ],
                max_structures=2,
                use_seed=True,
                seed=3,
            ),
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(
            all({2, 3}.issubset(set(atoms.arrays["site_id"])) for atoms in results)
        )

    def test_random_vacancy_operation_rejects_invalid_or_destructive_counts(self):
        structure = Atoms(
            "O4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[10, 10, 10],
            pbc=True,
        )
        operation = RandomVacancyOperation()

        with self.assertRaises(CardOperationError) as missing_rules:
            operation.run_structure(structure, RandomVacancyParams(rules=[]))
        self.assertEqual(missing_rules.exception.code, "targeted_vacancy_missing_rules")
        with self.assertRaises(CardOperationError) as invalid_mode:
            operation.run_structure(
                structure,
                RandomVacancyParams(
                    rules=[{"element": "O", "count": [1, 2], "count_mode": "typo"}]
                ),
            )
        self.assertEqual(invalid_mode.exception.code, "targeted_vacancy_invalid_count_mode")
        with self.assertRaises(CardOperationError) as negative_count:
            operation.run_structure(
                structure,
                RandomVacancyParams(
                    rules=[{"element": "O", "count": [-1, 2], "count_mode": "random"}]
                ),
            )
        self.assertEqual(negative_count.exception.code, "targeted_vacancy_negative_count")
        with self.assertRaises(CardOperationError) as excessive_count:
            operation.run_structure(
                structure,
                RandomVacancyParams(
                    rules=[{"element": "O", "count": [5, 5], "count_mode": "fixed"}]
                ),
            )
        self.assertEqual(
            excessive_count.exception.code,
            "targeted_vacancy_count_exceeds_matches",
        )
        with self.assertRaises(CardOperationError) as destructive_count:
            operation.run_structure(
                structure,
                RandomVacancyParams(
                    rules=[{"element": "O", "count": [4, 4], "count_mode": "fixed"}]
                ),
            )
        self.assertEqual(
            destructive_count.exception.code,
            "targeted_vacancy_no_valid_output",
        )

    def test_random_vacancy_preview_marks_overlapping_rule_upper_bound(self):
        structure = Atoms(
            "O3",
            positions=np.arange(9, dtype=float).reshape(3, 3),
            cell=[10, 10, 10],
            pbc=True,
        )
        card = RandomVacancyCard()
        card.rules_widget.from_rules(
            [
                {"element": "O", "count": [1, 1], "count_mode": "fixed"},
                {"element": "O", "count": [1, 1], "count_mode": "fixed"},
            ]
        )
        card.max_atoms_condition_frame.set_input_value([20])
        card.set_dataset([structure])
        self.assertIn("up to 9 unique outputs", card.preview_label.text())
        self.assertIn("combinatorial upper bound", card.preview_label.text())

    def test_random_vacancy_operation_deduplicates_limited_site_combinations(self):
        structure = Atoms(
            "SiO",
            positions=[[0, 0, 0], [1, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        )
        params = RandomVacancyParams(
            rules=[{"element": "O", "count": [1, 1], "count_mode": "fixed"}],
            max_structures=20,
            use_seed=True,
            seed=9,
        )
        results = RandomVacancyOperation().run_structure(
            structure,
            params,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get_chemical_formula(), "Si")
        self.assertEqual(
            RandomVacancyOperation.maximum_unique_outputs(
                structure,
                params.rules,
                params.max_structures,
            ),
            1,
        )

    def test_random_vacancy_seed_is_reproducible_and_structure_specific(self):
        structure_a = Atoms(
            "O8",
            positions=np.arange(24, dtype=float).reshape(8, 3),
            cell=[30, 30, 30],
            pbc=True,
        )
        structure_b = structure_a.copy()
        structure_b.positions[0, 1] += 0.125
        for structure in (structure_a, structure_b):
            structure.arrays["site_id"] = np.arange(len(structure))
        params = RandomVacancyParams(
            rules=[{"element": "O", "count": [3, 3], "count_mode": "fixed"}],
            max_structures=3,
            use_seed=True,
            seed=42,
        )
        operation = RandomVacancyOperation()

        first = operation.run_structure(structure_a, params)
        repeated = operation.run_structure(structure_a, params)
        second_structure = operation.run_structure(structure_b, params)
        first_ids = [atoms.arrays["site_id"].tolist() for atoms in first]
        repeated_ids = [atoms.arrays["site_id"].tolist() for atoms in repeated]
        second_ids = [atoms.arrays["site_id"].tolist() for atoms in second_structure]

        self.assertEqual(first_ids, repeated_ids)
        self.assertNotEqual(first_ids, second_ids)

    def test_random_vacancy_overlapping_random_rules_skip_only_infeasible_draws(self):
        structure = Atoms(
            "O6",
            positions=np.arange(18, dtype=float).reshape(6, 3),
            cell=[30, 30, 30],
            pbc=True,
        )
        rules = [
            {"element": "O", "count": [1, 4], "count_mode": "random"},
            {"element": "O", "count": [1, 3], "count_mode": "random"},
        ]

        for seed in range(32):
            with self.subTest(seed=seed):
                results = RandomVacancyOperation().run_structure(
                    structure,
                    RandomVacancyParams(
                        rules=rules,
                        max_structures=4,
                        use_seed=True,
                        seed=seed,
                    ),
                )
                self.assertGreaterEqual(len(results), 1)
                for atoms in results:
                    removed = len(structure) - len(atoms)
                    self.assertGreaterEqual(removed, 2)
                    self.assertLessEqual(removed, 5)
                    self.assertGreater(len(atoms), 0)
                    self.assertIn(f"Vac(n={removed})", atoms.info.get("Config_type", ""))

    def test_random_vacancy_overlapping_fixed_rules_fail_if_no_nonempty_result_exists(self):
        structure = Atoms(
            "O3",
            positions=np.arange(9, dtype=float).reshape(3, 3),
            cell=[20, 20, 20],
            pbc=True,
        )

        with self.assertRaisesRegex(ValueError, "could not generate a valid non-empty structure"):
            RandomVacancyOperation().run_structure(
                structure,
                RandomVacancyParams(
                    rules=[
                        {"element": "O", "count": [2, 2], "count_mode": "fixed"},
                        {"element": "O", "count": [1, 1], "count_mode": "fixed"},
                    ],
                    use_seed=True,
                    seed=7,
                ),
            )

