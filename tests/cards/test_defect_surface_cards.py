import warnings

from ase.geometry import geometry
from NepTrainKit.core.cards.errors import CardOperationError

from .card_test_base import *


class TestDefectSurfaceCards(BaseCardTest):
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

    def test_random_slab_card(self):
        card = RandomSlabCard()
        structure = self.structure.copy()
        card.h_frame.set_input_value([1, 1, 1])
        card.k_frame.set_input_value([0, 0, 1])
        card.l_frame.set_input_value([0, 0, 1])
        card.layer_frame.set_input_value([1, 1, 1])
        card.vacuum_frame.set_input_value([0, 0, 1])

        results = card.process_structure(structure)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(len(atoms) >= len(structure) for atoms in results))

    def test_random_slab_operation_enumerates_layer_vacuum_product_and_roundtrips(self):
        structure = self.structure.copy()
        structure.info["Config_type"] = "bulk"
        params = RandomSlabParams(
            h_range=(1, 1, 1),
            k_range=(0, 0, 1),
            l_range=(0, 0, 1),
            layer_range=(1, 2, 1),
            vacuum_range=(0.0, 2.0, 2.0),
        )

        results = RandomSlabOperation().run_structure(structure, params)

        self.assertEqual(len(results), 4)
        tags = [atoms.info.get("Config_type", "") for atoms in results]
        self.assertTrue(all("Slab(hkl=100" in tag for tag in tags))
        self.assertEqual(sum("L=1" in tag for tag in tags), 2)
        self.assertEqual(sum("L=2" in tag for tag in tags), 2)
        self.assertEqual(sum("vac=None" in tag for tag in tags), 2)
        self.assertEqual(sum("vac=2.0" in tag for tag in tags), 2)
        self.assertTrue(all(np.asarray(atoms.pbc, dtype=bool).all() for atoms in results))

        card = RandomSlabCard()
        card.set_params(params)
        restored = RandomSlabCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), params)

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

    def test_vacancy_rule_count_mode_distinguishes_fixed_and_range(self):
        item = VacancyRuleItem()
        item.show()
        QApplication.processEvents()
        self.assertEqual(item.fixed_count_frame.object_list[0].minimum(), 1)
        self.assertEqual(item.count_range_frame.object_list[0].minimum(), 0)
        self.assertTrue(item.fixed_count_frame.isVisible())
        self.assertFalse(item.count_range_frame.isVisible())
        item.element_edit.setText("Si")
        item.fixed_count_frame.set_input_value([2])
        fixed = item.to_rule()
        self.assertEqual(fixed["count"], [2, 2])
        self.assertEqual(fixed["count_mode"], "fixed")

        item.count_mode_combo.setCurrentText("Random range")
        QApplication.processEvents()
        self.assertFalse(item.fixed_count_frame.isVisible())
        self.assertTrue(item.count_range_frame.isVisible())
        self.assertTrue(all(spin.isVisible() for spin in item.count_range_frame.object_list))
        item.count_range_frame.set_input_value([1, 3])
        ranged = item.to_rule()
        self.assertEqual(ranged["count"], [1, 3])
        self.assertEqual(ranged["count_mode"], "random")

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

    def test_stacking_fault_card(self):
        card = StackingFaultCard()
        structure = self.structure.copy()
        card.hkl_frame.set_input_value([1, 1, 1])
        card.layer_frame.set_input_value([1])
        card.step_frame.set_input_value([0.1, 0.1, 0.1])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 1)
        self.assertGreater(
            np.abs(results[0].get_positions() - structure.get_positions()).sum(),
            0.0,
        )

    def test_legacy_stacking_fault_json_remains_loadable(self):
        card_class = CardManager.card_info_dict["StackingFaultCard"]
        card = card_class()
        card.from_dict(
            {
                "class": "StackingFaultCard",
                "check_state": True,
                "params": {
                    "hkl": [0, 0, 1],
                    "step": [0.0, 0.4, 0.2],
                    "layers": 2,
                },
            }
        )

        self.assertEqual(
            card.get_params(),
            StackingFaultParams(
                hkl=(0, 0, 1),
                step=(0.0, 0.4, 0.2),
                layers=2,
            ),
        )

    def test_stacking_fault_displaces_selected_layers_in_plane(self):
        structure = Atoms(
            "Si3",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            cell=[5.0, 5.0, 5.0],
            pbc=False,
        )

        result = StackingFaultOperation().run_structure(
            structure,
            StackingFaultParams(hkl=(0, 0, 1), step=(0.5, 0.5, 0.1), layers=2),
        )[0]

        displacement = result.get_positions() - structure.get_positions()
        np.testing.assert_allclose(displacement[0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(displacement[1:, 2], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(displacement[1:], axis=1), [0.5, 0.5], atol=1e-12)

    def test_defect_surface_operations_are_ui_independent(self):
        structure = self.structure.copy()

        vac_results = RandomVacancyOperation().run_structure(
            structure,
            RandomVacancyParams(
                rules=[{"element": "Si", "count": [1, 1], "count_mode": "fixed"}],
                max_structures=2,
                use_seed=True,
                seed=3,
            ),
        )
        self.assertEqual(len(vac_results), 2)
        self.assertTrue(all(len(atoms) == len(structure) - 1 for atoms in vac_results))

        defect_results = VacancyDefectOperation().run_structure(
            structure,
            VacancyDefectParams(
                engine_type=1,
                use_num=False,
                concentration_condition=0.6,
                count_mode="random",
                max_structures=2,
                use_seed=True,
                seed=5,
            ),
        )
        self.assertEqual(len(defect_results), 2)
        self.assertTrue(all(len(atoms) < len(structure) for atoms in defect_results))

        fault_results = StackingFaultOperation().run_structure(
            structure,
            StackingFaultParams(hkl=(1, 1, 1), step=(0.1, 0.1, 0.1), layers=1),
        )
        self.assertEqual(len(fault_results), 1)
        self.assertGreater(
            np.abs(fault_results[0].get_positions() - structure.get_positions()).sum(),
            0.0,
        )
        self.assertIn("SF(", fault_results[0].info.get("Config_type", ""))

        slab_results = RandomSlabOperation().run_structure(
            structure,
            RandomSlabParams(
                h_range=(1, 1, 1),
                k_range=(0, 0, 1),
                l_range=(0, 0, 1),
                layer_range=(1, 1, 1),
                vacuum_range=(0, 0, 1),
            ),
        )
        self.assertEqual(len(slab_results), 1)
        self.assertGreaterEqual(len(slab_results[0]), len(structure))
        self.assertIn("Slab(", slab_results[0].info.get("Config_type", ""))

        insert_results = InsertDefectOperation().run_structure(
            structure,
            InsertDefectParams(
                mode=0,
                species="H",
                insert_count=1,
                structure_count=1,
                min_distance=0.1,
                max_attempts=20,
                use_seed=True,
                seed=7,
            ),
        )
        self.assertEqual(len(insert_results), 1)
        self.assertEqual(len(insert_results[0]), len(structure) + 1)
        self.assertIn("Ins(int", insert_results[0].info.get("Config_type", ""))

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
            "Interstitial and Surface Adsorption",
        )
        self.assertEqual(card.get_params(), InsertDefectParams())
        self.assertTrue(card.axis_label.isHidden())
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

        card.mode_combo.setCurrentIndex(card.mode_combo.findData(1))
        self.assertFalse(card.axis_label.isHidden())
        self.assertIn("upper surface along lattice c", card.preview_label.text())
        self.assertIn("height 1.5 Å", card.preview_label.text())

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

    def test_strict_gsfe_path_card_roundtrip(self):
        card = StrictGSFEPathCard()
        self.assertTrue(card.cut_fraction_field.isHidden())
        self.assertTrue(card.layer_field.isHidden())
        self.assertGreater(card.disp_frame.object_list[2].minimum(), 0.0)
        self.assertIn("Load an oriented structure", card.preview_label.text())

        card.set_dataset(
            Atoms(
                "H4",
                positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
                cell=[8, 8, 8],
                pbc=True,
            )
        )
        self.assertIn("4 layers", card.preview_label.text())
        self.assertIn("move 2, keep 2", card.preview_label.text())
        self.assertIn("0→1 × vector = 0→8 Å", card.preview_label.text())
        self.assertIn("3 outputs", card.preview_label.text())

        card.slip_uv_frame.set_input_value([1, -1])
        card.disp_frame.set_input_value([0.0, 0.5, 0.25])
        card.unit_control.setCurrentIndex(card.unit_control.findData("angstrom"))
        card.cut_control.setCurrentIndex(card.cut_control.findData("layer_index"))
        self.assertTrue(card.cut_fraction_field.isHidden())
        self.assertFalse(card.layer_field.isHidden())
        card.cut_fraction_frame.set_input_value([0.25])
        card.layer_frame.set_input_value([2])
        card.wrap_checkbox.setChecked(False)

        restored = StrictGSFEPathCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_strict_gsfe_cut_mode_fields_are_visible_and_operable(self):
        card = StrictGSFEPathCard()
        card.show()
        QApplication.processEvents()

        card.cut_control.setCurrentIndex(card.cut_control.findData("fractional"))
        QApplication.processEvents()
        self.assertTrue(card.cut_fraction_frame.isVisible())
        self.assertFalse(card.layer_field.isVisible())
        card.cut_fraction_frame.set_input_value([0.25])
        self.assertEqual(card.get_params().cut_fraction, 0.25)

        card.cut_control.setCurrentIndex(card.cut_control.findData("layer_index"))
        QApplication.processEvents()
        self.assertFalse(card.cut_fraction_field.isVisible())
        self.assertTrue(card.layer_frame.isVisible())
        card.layer_frame.set_input_value([2])
        self.assertEqual(card.get_params().layer_index, 2)

        card.cut_control.setCurrentIndex(card.cut_control.findData("middle"))
        QApplication.processEvents()
        self.assertFalse(card.cut_fraction_field.isVisible())
        self.assertFalse(card.layer_field.isVisible())

    def test_strict_gsfe_preserves_legacy_geometry_until_direction_edit(self):
        card = StrictGSFEPathCard()
        legacy = StrictGSFEPathParams(
            plane_hkl=(1, 1, 0),
            slip_uvw=(1, -1, 2),
        )
        card.set_params(legacy)

        self.assertEqual(card.get_params(), legacy)
        self.assertFalse(card.legacy_geometry_label.isHidden())

        card.slip_uv_frame.object_list[0].setValue(2)
        self.assertEqual(card.get_params().plane_hkl, (0, 0, 1))
        self.assertEqual(card.get_params().slip_uvw, (2, -1, 0))
        self.assertTrue(card.legacy_geometry_label.isHidden())

    def test_layer_copy_operation_is_ui_independent(self):
        structure = self.structure.copy()
        results = LayerCopyOperation().run_structure(
            structure,
            LayerCopyParams(
                dz_expr="1.0",
                layers=2,
                distance=3.0,
                extend_cell_z=True,
            ),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), len(structure) * 2)
        self.assertIn(
            "LayerStack(L=2,step=3)",
            results[0].info.get("Config_type", ""),
        )

    def test_layer_copy_card_roundtrip(self):
        card = LayerCopyCard()
        card.preset_combo.setCurrentIndex(0)
        card.expr_edit.setPlainText("A + z*0")
        card.params_edit.setText("A=1.5")
        card.apply_combo.setCurrentIndex(2)
        card.zrange_frame.set_input_value([0.0, 2.0])
        card.wrap_checkbox.setChecked(True)
        card.extend_cell_checkbox.setChecked(False)
        card.vacuum_frame.set_input_value([1.0])
        card.layers_frame.set_input_value([2])
        card.distance_frame.set_input_value([4.0])

        restored = LayerCopyCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_layer_copy_default_is_flat_and_preview_explains_full_copy(self):
        card = LayerCopyCard()
        slab = Atoms(
            "MoS2",
            positions=[[0, 0, 0], [0, 0, 1.5], [0, 0, -1.5]],
            cell=[8, 8, 15],
            pbc=True,
        )
        card.set_dataset([slab])

        self.assertEqual(card.get_params().dz_expr, "0")
        self.assertFalse(card.show_warp_checkbox.isChecked())
        self.assertTrue(card.preset_combo.isHidden())
        self.assertIn("3 atoms", card.preview_label.text())
        self.assertIn("2 total layers", card.preview_label.text())
        self.assertIn("output 6 atoms", card.preview_label.text())

        card.show_warp_checkbox.setChecked(True)
        card.apply_combo.setCurrentIndex(1)
        self.assertFalse(card.preset_combo.isHidden())
        self.assertFalse(card.elements_edit.isHidden())
        self.assertTrue(card.zrange_frame.isHidden())

    def test_layer_copy_selection_only_limits_warp_not_copy(self):
        structure = Atoms(
            "CSi",
            positions=[[0, 0, 0], [1, 0, 0]],
            cell=[5, 5, 10],
            pbc=True,
        )
        result = LayerCopyOperation().run_structure(
            structure,
            LayerCopyParams(
                dz_expr="1",
                apply_mode=1,
                elements="C",
                layers=2,
                distance=4,
            ),
        )[0]

        self.assertEqual(result.get_chemical_symbols(), ["C", "Si", "C", "Si"])
        np.testing.assert_allclose(
            result.positions,
            [[0, 0, 1], [1, 0, 0], [0, 0, 5], [1, 0, 4]],
        )

    def test_layer_copy_extends_cell_for_one_layer_vacuum(self):
        structure = Atoms(
            "C",
            positions=[[0, 0, 0]],
            cell=[[5, 0, 0], [0, 5, 0], [1, 0, 10]],
            pbc=True,
        )
        result = LayerCopyOperation().run_structure(
            structure,
            LayerCopyParams(
                dz_expr="0",
                layers=1,
                distance=3,
                extend_cell_z=True,
                extra_vacuum=4,
            ),
        )[0]

        np.testing.assert_allclose(result.cell.array[2], [1, 0, 14])

    def test_layer_copy_rejects_invalid_geometry_settings(self):
        structure = Atoms(
            "C",
            positions=[[0, 0, 0]],
            cell=[5, 5, 10],
            pbc=True,
        )
        operation = LayerCopyOperation()

        with self.assertRaisesRegex(ValueError, "layers must be an integer"):
            operation.run_structure(
                structure,
                LayerCopyParams(dz_expr="0", layers=1.5),
            )
        with self.assertRaisesRegex(ValueError, "translation must be positive"):
            operation.run_structure(
                structure,
                LayerCopyParams(dz_expr="0", layers=2, distance=0),
            )
        with self.assertRaisesRegex(ValueError, "finite number"):
            operation.run_structure(
                structure,
                LayerCopyParams(dz_expr="0", layers=1, extra_vacuum=float("nan")),
            )
        with self.assertRaisesRegex(ValueError, "no atoms selected"):
            operation.run_structure(
                structure,
                LayerCopyParams(
                    dz_expr="1",
                    apply_mode=1,
                    elements="Si",
                    layers=1,
                ),
            )
