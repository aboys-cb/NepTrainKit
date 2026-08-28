import json
import warnings
from unittest.mock import patch

from .card_test_base import *

from ase.calculators.singlepoint import SinglePointCalculator
from NepTrainKit.core.cards.alloy import sample_dopants
from NepTrainKit.core.cards.errors import CardOperationError


class TestAlloyCards(BaseCardTest):
    def test_random_doping_card(self):
        card = RandomDopingCard()
        structure = self.structure.copy()
        rules = [{
            "target": "Si",
            "dopants": {"Ge": 1.0},
            "use": "count",
            "count": [1, 1],
            "concentration": [0.0, 1.0],
        }]
        card.rules_widget.from_rules(rules)
        card.doping_type_combo.setCurrentText("Exact")
        card.max_atoms_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        for atoms in results:
            self.assertIn("Ge", atoms.get_chemical_symbols())

    def test_random_doping_and_occupancy_cards_roundtrip_nondefault_params(self):
        doping_params = RandomDopingParams(
            rules=[
                {
                    "target": "Si",
                    "dopants": {"Ge": 0.7, "C": 0.3},
                    "use": "count",
                    "count": [1, 2],
                    "count_mode": "random",
                    "group": ["surface"],
                }
            ],
            doping_type="Exact",
            max_structures=4,
            use_seed=True,
            seed=59,
        )
        doping = RandomDopingCard()
        doping.set_params(doping_params)
        normalized_doping_params = doping.get_params()
        restored_doping = RandomDopingCard()
        restored_doping.from_dict(doping.to_dict())
        self.assertEqual(
            restored_doping.get_params(),
            normalized_doping_params,
        )

        occupancy_params = RandomOccupancyParams(
            source="Manual",
            manual="Co:0.4,Ni:0.6",
            mode="Random",
            samples=5,
            group_filter="A,B",
            use_seed=True,
            seed=61,
        )
        occupancy = RandomOccupancyCard()
        occupancy.set_params(occupancy_params)
        normalized_occupancy_params = occupancy.get_params()
        restored_occupancy = RandomOccupancyCard()
        restored_occupancy.from_dict(occupancy.to_dict())
        self.assertEqual(
            restored_occupancy.get_params(),
            normalized_occupancy_params,
        )

    def test_random_doping_dopants_accept_bare_element(self):
        item = DopingRuleItem()
        item.target_edit.setText("Si")
        item.dopants_edit.setText("Ge")
        item.count_botton.setChecked(True)
        item._on_mode_changed()
        item.fixed_count_frame.set_input_value([1])

        rule = item.to_rule()

        self.assertEqual(rule["dopants"], {"Ge": 1.0})
        self.assertEqual(rule["count"], [1, 1])
        self.assertEqual(rule["count_mode"], "fixed")

    def test_random_doping_rule_restores_user_friendly_dopant_text(self):
        item = DopingRuleItem()

        item.from_rule({"target": "Si", "dopants": {"Ge": 1.0}})
        self.assertEqual(item.dopants_edit.text(), "Ge")

        item.from_rule({"target": "Si", "dopants": {"Ge": 0.7, "C": 0.3}})

        self.assertEqual(item.dopants_edit.text(), "Ge:0.7,C:0.3")
        self.assertEqual(item.to_rule()["dopants"], {"Ge": 0.7, "C": 0.3})

    def test_random_doping_ratio_button_label_matches_serialized_semantics(self):
        item = DopingRuleItem()

        self.assertTrue(item.ratio_type_button.isChecked())
        self.assertEqual(item.ratio_type_button.text(), "Atom ratio")
        self.assertEqual(item.to_rule()["ratio_type"], "atom")

        item.ratio_type_button.click()
        self.assertFalse(item.ratio_type_button.isChecked())
        self.assertEqual(item.ratio_type_button.text(), "Mass ratio")
        self.assertEqual(item.to_rule()["ratio_type"], "mass")

        item.from_rule({"target": "Si", "dopants": {"Ge": 1.0}, "ratio_type": "atom"})
        self.assertEqual(item.ratio_type_button.text(), "Atom ratio")
        self.assertEqual(item.to_rule()["ratio_type"], "atom")

        item.from_rule({"target": "Si", "dopants": {"Ge": 1.0}, "ratio_type": "mass"})
        self.assertEqual(item.ratio_type_button.text(), "Mass ratio")
        self.assertEqual(item.to_rule()["ratio_type"], "mass")

    def test_random_doping_count_mode_distinguishes_fixed_and_range(self):
        structure = Atoms("Si5", positions=np.arange(15, dtype=float).reshape(5, 3), cell=[10, 10, 10], pbc=True)

        fixed = RandomDopingOperation().run_structure(
            structure,
            RandomDopingParams(
                rules=[{"target": "Si", "dopants": {"Ge": 1.0}, "use": "count", "count": [3, 3], "count_mode": "fixed"}],
                max_structures=3,
                use_seed=True,
                seed=1,
            ),
        )
        self.assertTrue(all(atoms.get_chemical_symbols().count("Ge") == 3 for atoms in fixed))

        ranged = RandomDopingOperation().run_structure(
            structure,
            RandomDopingParams(
                rules=[{"target": "Si", "dopants": {"Ge": 1.0}, "use": "count", "count": [1, 3], "count_mode": "random"}],
                max_structures=10,
                use_seed=True,
                seed=1,
            ),
        )
        self.assertTrue(all(1 <= atoms.get_chemical_symbols().count("Ge") <= 3 for atoms in ranged))

    def test_random_doping_operation_is_ui_independent(self):
        params = RandomDopingParams(
            rules=[
                {
                    "target": "Si",
                    "dopants": {"Ge": 1.0},
                    "use": "count",
                    "count": [1, 1],
                }
            ],
            doping_type="Exact",
            max_structures=2,
            use_seed=True,
            seed=3,
        )
        results = RandomDopingOperation().run_structure(self.structure.copy(), params)

        self.assertEqual(len(results), 2)
        self.assertTrue(all("Ge" in atoms.get_chemical_symbols() for atoms in results))

    def test_random_doping_group_constraint_is_required_and_limits_changed_sites(self):
        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[20, 20, 20],
            pbc=True,
        )
        rule = {
            "target": "Si",
            "dopants": {"Ge": 1.0},
            "use": "count",
            "count": [1, 1],
            "count_mode": "fixed",
            "group": ["A"],
        }
        operation = RandomDopingOperation()

        with self.assertRaisesRegex(ValueError, "has no group array"):
            operation.run_structure(
                structure,
                RandomDopingParams(rules=[rule], use_seed=True, seed=1),
            )

        structure.new_array("group", np.array(["A", "A", "B", "B"], dtype=object))
        result = operation.run_structure(
            structure,
            RandomDopingParams(rules=[rule], use_seed=True, seed=1),
        )[0]
        symbols = result.get_chemical_symbols()
        self.assertEqual(symbols[:2].count("Ge"), 1)
        self.assertEqual(symbols[2:], ["Si", "Si"])

        no_match = structure.copy()
        no_match.arrays["group"][:] = "B"
        with self.assertRaisesRegex(ValueError, "matched no 'Si' atoms in group A"):
            operation.run_structure(
                no_match,
                RandomDopingParams(rules=[rule], use_seed=True, seed=1),
            )

    def test_random_doping_does_not_clamp_zero_percent_or_oversized_count(self):
        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[20, 20, 20],
            pbc=True,
        )
        operation = RandomDopingOperation()
        zero = operation.run_structure(
            structure,
            RandomDopingParams(
                rules=[
                    {
                        "target": "Si",
                        "dopants": {"Ge": 1.0},
                        "use": "atomic_percent",
                        "percent": [0.0, 0.0],
                    }
                ],
                use_seed=True,
                seed=1,
            ),
        )[0]
        self.assertEqual(zero.get_chemical_symbols(), structure.get_chemical_symbols())
        self.assertNotIn("Dop(", zero.info.get("Config_type", ""))

        with self.assertRaisesRegex(
            ValueError,
            "can request up to 10 replacements, but only 4",
        ):
            operation.run_structure(
                structure,
                RandomDopingParams(
                    rules=[
                        {
                            "target": "Si",
                            "dopants": {"Ge": 1.0},
                            "use": "count",
                            "count": [10, 10],
                            "count_mode": "fixed",
                        }
                    ],
                ),
            )

    def test_random_doping_random_count_capacity_is_seed_independent(self):
        structure = Atoms(
            "Si3",
            positions=np.arange(9, dtype=float).reshape(3, 3),
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        impossible = {
            "target": "Si",
            "dopants": {"Ge": 1.0},
            "use": "count",
            "count": [1, 4],
            "count_mode": "random",
        }
        feasible = {
            **impossible,
            "count": [0, 3],
        }
        operation = RandomDopingOperation()

        for seed in range(32):
            with self.subTest(case="impossible", seed=seed):
                with self.assertRaisesRegex(
                    ValueError,
                    "can request up to 4 replacements, but only 3",
                ):
                    operation.run_structure(
                        structure,
                        RandomDopingParams(
                            rules=[impossible],
                            use_seed=True,
                            seed=seed,
                        ),
                    )

            with self.subTest(case="feasible", seed=seed):
                result = operation.run_structure(
                    structure,
                    RandomDopingParams(
                        rules=[feasible],
                        use_seed=True,
                        seed=seed,
                    ),
                )[0]
                self.assertLessEqual(
                    result.get_chemical_symbols().count("Ge"),
                    3,
                )

    def test_random_doping_impossible_range_fails_before_rng(self):
        structure = Atoms(
            "Si3",
            positions=np.arange(9, dtype=float).reshape(3, 3),
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )

        with patch(
            "NepTrainKit.core.cards.alloy.np.random.default_rng"
        ) as rng_factory:
            with self.assertRaisesRegex(
                ValueError,
                "can request up to 4 replacements, but only 3",
            ):
                RandomDopingOperation().run_structure(
                    structure,
                    RandomDopingParams(
                        rules=[
                            {
                                "target": "Si",
                                "dopants": {"Ge": 1.0},
                                "use": "count",
                                "count": [1, 4],
                                "count_mode": "random",
                            }
                        ],
                        use_seed=True,
                        seed=7,
                    ),
                )
        rng_factory.assert_not_called()

    def test_random_doping_overlapping_rules_have_seed_independent_capacity(self):
        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[12.0, 12.0, 12.0],
            pbc=True,
        )
        impossible = [
            {
                "target": "Si",
                "dopants": {"Ge": 1.0},
                "use": "count",
                "count": [0, 2],
                "count_mode": "random",
            },
            {
                "target": "Si",
                "dopants": {"C": 1.0},
                "use": "count",
                "count": [3, 3],
                "count_mode": "fixed",
            },
        ]
        feasible = [
            {
                **impossible[0],
                "count": [0, 1],
            },
            impossible[1],
        ]
        operation = RandomDopingOperation()

        with patch(
            "NepTrainKit.core.cards.alloy.np.random.default_rng"
        ) as rng_factory:
            with self.assertRaisesRegex(
                ValueError,
                "earlier overlapping rules can leave only 2 eligible atoms",
            ):
                operation.run_structure(
                    structure,
                    RandomDopingParams(
                        rules=impossible,
                        use_seed=True,
                        seed=7,
                    ),
                )
        rng_factory.assert_not_called()

        for seed in range(32):
            with self.subTest(case="impossible", seed=seed):
                with self.assertRaisesRegex(
                    ValueError,
                    "earlier overlapping rules can leave only 2 eligible atoms",
                ):
                    operation.run_structure(
                        structure,
                        RandomDopingParams(
                            rules=impossible,
                            use_seed=True,
                            seed=seed,
                        ),
                    )

            with self.subTest(case="feasible", seed=seed):
                output = operation.run_structure(
                    structure,
                    RandomDopingParams(
                        rules=feasible,
                        use_seed=True,
                        seed=seed,
                    ),
                )[0]
                changed = sum(
                    symbol != "Si"
                    for symbol in output.get_chemical_symbols()
                )
                self.assertIn(changed, {3, 4})

    def test_random_doping_mass_percent_capacity_is_seed_independent(self):
        structure = Atoms(
            "Au4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[12.0, 12.0, 12.0],
            pbc=True,
        )
        impossible = {
            "target": "Au",
            "dopants": {"Li": 1.0},
            "use": "mass_percent",
            "percent": [5.0, 60.0],
        }
        feasible = {
            **impossible,
            "percent": [0.1, 3.0],
        }
        operation = RandomDopingOperation()

        for seed in range(24):
            with self.subTest(case="impossible", seed=seed):
                with self.assertRaisesRegex(
                    ValueError,
                    "can request up to .* replacements, but only 4",
                ):
                    operation.run_structure(
                        structure,
                        RandomDopingParams(
                            rules=[impossible],
                            use_seed=True,
                            seed=seed,
                        ),
                    )

            with self.subTest(case="feasible", seed=seed):
                result = operation.run_structure(
                    structure,
                    RandomDopingParams(
                        rules=[feasible],
                        use_seed=True,
                        seed=seed,
                    ),
                )[0]
                self.assertLessEqual(
                    result.get_chemical_symbols().count("Li"),
                    4,
                )

    def test_random_doping_mass_percent_uses_declared_dopant_ratio_type(self):
        structure = Atoms(
            "Au20",
            positions=np.arange(60, dtype=float).reshape(20, 3),
            cell=[80.0, 80.0, 80.0],
            pbc=True,
        )
        operation = RandomDopingOperation()
        candidates = np.arange(len(structure))
        common = {
            "target": "Au",
            "dopants": {"H": 9.0, "Pt": 1.0},
            "use": "mass_percent",
            "percent": [1.0, 1.0],
        }

        atom_ratio_count = operation._doping_count(
            structure,
            candidates,
            "Au",
            common["dopants"],
            {**common, "ratio_type": "atom"},
            rng=None,
        )
        mass_ratio_count = operation._doping_count(
            structure,
            candidates,
            "Au",
            common["dopants"],
            {**common, "ratio_type": "mass"},
            rng=None,
        )

        self.assertEqual(atom_ratio_count, 1)
        self.assertGreater(mass_ratio_count, atom_ratio_count)

    def test_random_doping_rejects_invalid_ratios_before_output_sampling(self):
        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[20.0, 20.0, 20.0],
            pbc=True,
        )
        operation = RandomDopingOperation()

        for dopants, ratio_type, message in (
            ({"Ge": -1.0}, "atom", "finite and non-negative"),
            ({"Ge": 0.0}, "atom", "[Aa]t least one dopant ratio must be positive"),
            ({"Ge": 1.0}, "typo", "ratio_type must be 'atom' or 'mass'"),
        ):
            with self.subTest(dopants=dopants, ratio_type=ratio_type):
                with self.assertRaisesRegex(ValueError, message):
                    operation.run_structure(
                        structure,
                        RandomDopingParams(
                            rules=[
                                {
                                    "target": "Si",
                                    "dopants": dopants,
                                    "ratio_type": ratio_type,
                                    "use": "count",
                                    "count": [1, 1],
                                }
                            ],
                            max_structures=3,
                            use_seed=True,
                            seed=7,
                        ),
                    )

    def test_random_doping_exact_ratios_use_largest_remainder_allocation(self):
        sampled = sample_dopants(
            ["Fe", "Co", "Ni"],
            [0.34, 0.33, 0.33],
            5,
            exact=True,
            rng=np.random.default_rng(1),
        )

        self.assertEqual(
            sorted(sampled.count(element) for element in ("Fe", "Co", "Ni")),
            [1, 2, 2],
        )

    def test_parse_composition_accepts_bare_elements(self):
        self.assertEqual(parse_composition("Ge"), {"Ge": 1.0})
        self.assertEqual(parse_composition("Ge,C"), {"Ge": 1.0, "C": 1.0})
        self.assertEqual(parse_composition("Ge:0.7,C"), {"Ge": 0.7, "C": 1.0})

    def test_conditional_replace_operation_is_ui_independent(self):
        base = Atoms(
            "Si6",
            positions=np.column_stack(
                [np.zeros(6), np.zeros(6), np.arange(6, dtype=float)]
            ),
            cell=[8.0, 8.0, 8.0],
            pbc=True,
        )
        result = ConditionalReplaceOperation().run_structure(
            base,
            ConditionalReplaceParams(
                target="Si",
                replacements="Ge:1",
                condition="z>=3",
                seed=1,
                mode=1,
            ),
        )[0]

        self.assertEqual(
            result.get_chemical_symbols(),
            ["Si", "Si", "Si", "Ge", "Ge", "Ge"],
        )
        np.testing.assert_allclose(result.positions, base.positions)
        np.testing.assert_allclose(result.cell.array, base.cell.array)
        np.testing.assert_array_equal(result.pbc, base.pbc)
        self.assertIn("Repl(Si->Ge)", result.info.get("Config_type", ""))

    def test_conditional_replace_allocation_modes_and_seed(self):
        base = Atoms(
            "Si20",
            positions=np.column_stack(
                [np.zeros(20), np.zeros(20), np.arange(20, dtype=float)]
            ),
            cell=[10.0, 10.0, 22.0],
            pbc=True,
        )
        operation = ConditionalReplaceOperation()
        random_params = ConditionalReplaceParams(
            target="Si",
            replacements="Ge:0.5,C:0.5",
            condition="all",
            seed=7,
            mode=0,
        )
        random_a = operation.run_structure(base, random_params)[0]
        random_b = operation.run_structure(base, random_params)[0]
        exact = operation.run_structure(
            base,
            ConditionalReplaceParams(
                target="Si",
                replacements="Ge:0.5,C:0.5",
                condition="all",
                seed=7,
                mode=1,
            ),
        )[0]

        self.assertEqual(
            random_a.get_chemical_symbols(),
            random_b.get_chemical_symbols(),
        )
        self.assertNotIn("Si", random_a.get_chemical_symbols())
        self.assertEqual(random_a.get_chemical_symbols().count("Ge"), 9)
        self.assertEqual(random_a.get_chemical_symbols().count("C"), 11)
        self.assertEqual(exact.get_chemical_symbols().count("Ge"), 10)
        self.assertEqual(exact.get_chemical_symbols().count("C"), 10)

    def test_conditional_replace_card_roundtrip(self):
        card = ConditionalReplaceCard()
        self.assertEqual(card.condition_edit.text(), "all")
        self.assertFalse(card.seed_checkbox.isChecked())
        self.assertEqual(card.get_params().seed, 0)
        card.target_edit.setText("Si")
        card.replacements_edit.setText("Ge:0.5,C:0.5")
        card.condition_edit.setText("z>=0")
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([9])
        card.mode_combo.setCurrentIndex(card.mode_combo.findData(1))

        restored = ConditionalReplaceCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())
        self.assertTrue(restored.seed_checkbox.isChecked())
        self.assertEqual(restored.mode_combo.currentData(), 1)

        legacy = ConditionalReplaceCard()
        legacy.from_dict(
            {
                "class": "ConditionalReplaceCard",
                "check_state": True,
                "target": "Si",
                "new_atoms": "Ge,C",
                "ratios": "0.5,0.5",
                "condition": "",
                "seed": [0],
                "mode": 1,
            }
        )
        self.assertEqual(legacy.get_params().condition, "all")
        self.assertEqual(legacy.get_params().replacements, "Ge:0.5,C:0.5")
        self.assertFalse(legacy.seed_checkbox.isChecked())
        self.assertEqual(legacy.mode_combo.currentData(), 1)

    def test_conditional_replace_exact_preview_and_magnetic_array_contract(self):
        base = Atoms(
            "O10",
            positions=np.column_stack(
                [np.linspace(0.0, 1.8, 10), np.zeros(10), np.arange(10.0)]
            ),
            cell=[[8.0, 0.0, 0.0], [1.2, 9.0, 0.0], [0.4, 0.6, 12.0]],
            pbc=[True, False, True],
        )
        spin = np.arange(30.0).reshape(10, 3)
        magmoms = np.flip(spin, axis=1).copy()
        base.new_array("spin", spin)
        base.set_initial_magnetic_moments(magmoms)
        base.info["Config_type"] = "surface"
        params = ConditionalReplaceParams(
            target="O",
            replacements="F:0.7,Cl:0.3",
            condition="z>=2 and z<=7",
            seed=13,
            mode=1,
        )

        operation = ConditionalReplaceOperation()
        summary = operation.selection_summary(params, base)
        self.assertEqual(summary["target_sites"], 10)
        self.assertEqual(summary["matched_sites"], 6)
        self.assertEqual(summary["replacement_counts"], (("F", 4), ("Cl", 2)))

        result = operation.run_structure(base, params)[0]
        self.assertEqual(result.get_chemical_symbols().count("O"), 4)
        self.assertEqual(result.get_chemical_symbols().count("F"), 4)
        self.assertEqual(result.get_chemical_symbols().count("Cl"), 2)
        np.testing.assert_array_equal(result.positions, base.positions)
        np.testing.assert_array_equal(result.cell.array, base.cell.array)
        np.testing.assert_array_equal(result.pbc, base.pbc)
        np.testing.assert_array_equal(result.arrays["spin"], spin)
        np.testing.assert_array_equal(result.arrays["initial_magmoms"], magmoms)
        self.assertEqual(result.info["Config_type"], "surface|Repl(O->F,Cl)")

        card = ConditionalReplaceCard()
        card.set_params(params)
        card.set_preview_structure(base)
        card.set_preview_input_count(2)
        self.assertIn("6 matched", card.get_summary_text())
        guidance = card.get_guidance_text()
        self.assertIn("outputs 2", guidance)
        self.assertIn("F:4", guidance)
        self.assertIn("spin", guidance)

    def test_conditional_replace_validation_and_truthful_tags(self):
        base = Atoms(
            "O4",
            positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        )
        operation = ConditionalReplaceOperation()
        invalid_cases = (
            ConditionalReplaceParams(target=""),
            ConditionalReplaceParams(target="Xx", replacements="F:1"),
            ConditionalReplaceParams(target="O", replacements=""),
            ConditionalReplaceParams(target="O", replacements="Xx:1"),
            ConditionalReplaceParams(target="O", replacements="F:-1,N:1"),
            ConditionalReplaceParams(target="O", replacements="F:0,N:0"),
            ConditionalReplaceParams(target="O", replacements="F:1,F:2"),
            ConditionalReplaceParams(target="O", replacements="O:1"),
            ConditionalReplaceParams(target="O", replacements="O:0,F:1"),
            ConditionalReplaceParams(target="O", replacements="F:1", mode=7),
            ConditionalReplaceParams(target="O", replacements="F:1", seed=-1),
            ConditionalReplaceParams(target="O", replacements="F:1", condition="q>2"),
            ConditionalReplaceParams(target="O", replacements="F:1", condition="x+1"),
            ConditionalReplaceParams(target="O", replacements="F:1", condition='x=="a"'),
            ConditionalReplaceParams(target="O", replacements="F:1", condition="x<1e999"),
            ConditionalReplaceParams(target="Si", replacements="Ge:1"),
            ConditionalReplaceParams(target="O", replacements="F:1", condition="z>99"),
        )
        for params in invalid_cases:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    operation.run_structure(base, params)

        invalid_positions = base.copy()
        invalid_positions.positions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite Cartesian"):
            operation.run_structure(
                invalid_positions,
                ConditionalReplaceParams(target="O", replacements="F:1"),
            )

        result = operation.run_structure(
            base,
            ConditionalReplaceParams(
                target="O",
                replacements="F:1,N:0",
                condition="z>=1 or x>99",
                seed=2,
                mode=1,
            ),
        )[0]
        self.assertEqual(result.get_chemical_symbols(), ["O", "F", "F", "F"])
        self.assertIn("Repl(O->F)", result.info["Config_type"])
        self.assertNotIn("N", result.info["Config_type"])

        for seed in range(20):
            random_result = operation.run_structure(
                base,
                ConditionalReplaceParams(
                    target="O",
                    replacements="F:1,Cl:1",
                    seed=seed,
                    mode=0,
                ),
            )[0]
            self.assertEqual(len(random_result), len(base))
            self.assertNotIn("O", random_result.get_chemical_symbols())

    def test_composition_sweep_and_random_occupancy_cards(self):
        base = self.structure.copy()
        base.info.setdefault("Config_type", "base")

        sweep = CompositionSweepCard()
        sweep.elements_edit.setText("Co,Ni")
        sweep.order_combo.setCurrentIndex(sweep.order_combo.findData("2"))
        sweep.method_combo.setCurrentText("Grid")
        sweep.step_frame.set_input_value([0.5])
        sweep.include_endpoints_checkbox.setChecked(True)
        sweep.minfrac_frame.set_input_value([0.0])
        sweep.max_output_frame.set_input_value([10])

        swept = sweep.process_structure(base)
        self.assertEqual(len(swept), 3)
        self.assertTrue(
            all("Comp(" in str(atoms.info.get("Config_type", "")) for atoms in swept)
        )

        occ = RandomOccupancyCard()
        occ.source_combo.setCurrentText("Auto (Comp tag)")
        occ.mode_combo.setCurrentText("Exact")
        occ.samples_frame.set_input_value([1])

        occupied = []
        for atoms in swept:
            occupied.extend(occ.process_structure(atoms))
        self.assertEqual(len(occupied), len(swept))
        for atoms in occupied:
            syms = set(atoms.get_chemical_symbols())
            self.assertTrue(syms.issubset({"Co", "Ni"}))

    def test_composition_and_occupancy_operations_are_ui_independent(self):
        base = self.structure.copy()
        base.info.setdefault("Config_type", "base")

        sweep_params = CompositionSweepParams(
            elements="Co,Ni",
            order="2",
            method="Grid",
            step=0.5,
            include_endpoints=True,
            max_outputs=3,
        )
        swept = CompositionSweepOperation().run_structure(base, sweep_params)

        self.assertEqual(len(swept), 3)
        self.assertTrue(all("Comp(" in atoms.info.get("Config_type", "") for atoms in swept))
        for atoms in swept:
            self.assertEqual(atoms.get_chemical_symbols(), base.get_chemical_symbols())
            np.testing.assert_allclose(atoms.positions, base.positions)
            np.testing.assert_allclose(atoms.cell.array, base.cell.array)
            np.testing.assert_array_equal(atoms.pbc, base.pbc)

        card = CompositionSweepCard()
        card.set_params(
            CompositionSweepParams(
                elements="Co,Cr,Ni,Al,Fe",
                order="5,4,3,2",
                method="Sobol",
                n_points=9,
                min_fraction=0.01,
                use_seed=True,
                seed=7,
                max_outputs=20,
                budget_mode="Capacity-weighted",
            )
        )
        restored = CompositionSweepCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())
        self.assertEqual(restored.order_combo.currentData(), "5,4,3,2")
        self.assertEqual(
            restored.budget_mode_combo.currentData(),
            "Capacity-weighted",
        )

        restored.set_params(
            CompositionSweepParams(
                elements="Co,Cr,Ni,Al",
                order="2,3,4",
            )
        )
        self.assertEqual(restored.order_combo.currentData(), "2,3,4")

        occ_params = RandomOccupancyParams(
            source="Auto (Comp tag)",
            mode="Exact",
            samples=1,
            use_seed=True,
            seed=5,
        )
        occupied = RandomOccupancyOperation().run_structure(swept[0], occ_params)

        self.assertEqual(len(occupied), 1)
        self.assertTrue(set(occupied[0].get_chemical_symbols()).issubset({"Co", "Ni"}))

    def test_composition_sweep_budget_modes_fill_cap_and_cover_requested_orders(self):
        base = self.structure.copy()
        operation = CompositionSweepOperation()

        for budget_mode in ("Equal+Reflow", "Capacity-weighted"):
            with self.subTest(budget_mode=budget_mode):
                params = CompositionSweepParams(
                    elements="Co,Cr,Ni,Al",
                    order="2,4",
                    method="Grid",
                    step=0.5,
                    include_endpoints=True,
                    use_seed=True,
                    seed=13,
                    max_outputs=8,
                    budget_mode=budget_mode,
                )
                first = operation.run_structure(base, params)
                repeated = operation.run_structure(base, params)
                tags = [atoms.info.get("Config_type", "") for atoms in first]
                repeated_tags = [
                    atoms.info.get("Config_type", "") for atoms in repeated
                ]

                self.assertEqual(len(first), 8)
                self.assertEqual(tags, repeated_tags)
                summary = operation.sampling_summary(params)
                self.assertGreater(summary["emitted_by_order"][2], 0)
                self.assertGreater(summary["emitted_by_order"][4], 0)

    def test_composition_sweep_summary_deduplicates_boundary_compositions(self):
        operation = CompositionSweepOperation()
        summary = operation.sampling_summary(CompositionSweepParams())

        self.assertEqual(summary["active_orders"], (2, 3))
        self.assertEqual(summary["skipped_orders"], (4, 5))
        self.assertEqual(summary["outputs_per_input"], 66)
        keys = [composition for _order, composition in summary["targets"]]
        self.assertEqual(len(keys), len(set(keys)))

        card = CompositionSweepCard()
        card.set_preview_input_count(2)
        self.assertIn("132", card.get_guidance_text())
        self.assertIn("Random Occupancy", card.get_guidance_text())

        card.method_combo.setCurrentIndex(card.method_combo.findData("Sobol"))
        self.assertFalse(card.n_points_field.isHidden())
        self.assertTrue(card.step_field.isHidden())
        card.method_combo.setCurrentIndex(card.method_combo.findData("Grid"))
        self.assertFalse(card.step_field.isHidden())
        self.assertTrue(card.n_points_field.isHidden())

    def test_composition_sweep_rejects_invalid_or_empty_plans(self):
        operation = CompositionSweepOperation()
        cases = (
            CompositionSweepParams(elements="Co"),
            CompositionSweepParams(elements="Co,Xx"),
            CompositionSweepParams(elements="Co,Ni", order="5"),
            CompositionSweepParams(elements="Co,Ni", min_fraction=0.6),
            CompositionSweepParams(max_outputs=0),
            CompositionSweepParams(max_outputs=10_001),
            CompositionSweepParams(order="nonsense"),
            CompositionSweepParams(
                elements="Co,Cr,Ni,Al",
                order="4",
                method="Grid",
                step=0.3,
            ),
        )
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    operation.sampling_summary(params)

    def test_composition_sweep_dense_sampling_stays_unique_and_budget_bounded(self):
        operation = CompositionSweepOperation()
        dense_sobol = CompositionSweepParams(
            elements="Co,Ni",
            order="2",
            method="Sobol",
            n_points=999_999,
            use_seed=True,
            seed=11,
            max_outputs=1_000,
        )

        summary = operation.sampling_summary(dense_sobol)
        self.assertEqual(summary["outputs_per_input"], 1_000)
        outputs = operation.run_structure(self.structure, dense_sobol)
        tags = [atoms.info["Config_type"].rsplit("|", 1)[-1] for atoms in outputs]
        self.assertEqual(len(tags), len(set(tags)))

        with self.assertRaisesRegex(ValueError, "safe limit"):
            operation.sampling_summary(
                CompositionSweepParams(step=1.0e-6, max_outputs=1)
            )

        constrained = operation.sampling_summary(
            CompositionSweepParams(
                elements="Co,Cr,Ni,Al,Fe",
                order="5",
                method="Sobol",
                n_points=999_999,
                min_fraction=0.19,
                max_outputs=500,
            )
        )
        self.assertEqual(constrained["outputs_per_input"], 500)
        self.assertTrue(
            all(
                min(fraction for _element, fraction in composition) >= 0.19 - 1e-12
                for _order, composition in constrained["targets"]
            )
        )

    def test_composition_sweep_replaces_old_target_and_preserves_structure_arrays(self):
        base = self.structure.copy()
        base.info["Config_type"] = "base|Comp(Fe=1)|old"
        spin = np.arange(len(base) * 3, dtype=float).reshape(-1, 3)
        initial_magmoms = np.flip(spin, axis=1).copy()
        base.new_array("spin", spin)
        base.set_initial_magnetic_moments(initial_magmoms)
        params = CompositionSweepParams(
            elements="Co,Ni",
            order="2",
            method="Grid",
            step=0.5,
            max_outputs=3,
        )

        outputs = CompositionSweepOperation().run_structure(base, params)
        self.assertEqual(len(outputs), 3)
        for atoms in outputs:
            tags = str(atoms.info["Config_type"]).split("|")
            self.assertEqual(sum(tag.startswith("Comp(") for tag in tags), 1)
            self.assertNotIn("Comp(Fe=1)", tags)
            np.testing.assert_array_equal(atoms.positions, base.positions)
            np.testing.assert_array_equal(atoms.cell.array, base.cell.array)
            np.testing.assert_array_equal(atoms.arrays["spin"], spin)
            np.testing.assert_array_equal(
                atoms.arrays["initial_magmoms"], initial_magmoms
            )

        occupancy = RandomOccupancyOperation().run_structure(
            outputs[1], RandomOccupancyParams(source="Auto (Comp tag)")
        )[0]
        self.assertNotIn("Fe", occupancy.get_chemical_symbols())

    def test_composition_sweep_loads_legacy_flat_order_name(self):
        card = CompositionSweepCard()
        card.from_dict(
            {
                "class": "CompositionSweepCard",
                "check_state": True,
                "elements": "Co,Ni",
                "order": "Binary",
                "method": "Grid",
                "step": [0.5],
                "n_points": [20],
                "min_fraction": [0.0],
                "max_outputs": [10],
                "seed": [0],
            }
        )
        self.assertEqual(card.get_params().order, "2")

    def test_random_occupancy_group_filter_is_required_and_limits_assignment(self):
        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[20, 20, 20],
            pbc=True,
        )
        params = RandomOccupancyParams(
            source="Manual",
            manual="Ge",
            group_filter="A",
            use_seed=True,
            seed=1,
        )
        operation = RandomOccupancyOperation()

        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(structure, params)
        self.assertEqual(raised.exception.code, "random_occupancy.missing_group_array")

        structure.new_array("group", np.array(["A", "A", "B", "B"], dtype=object))
        result = operation.run_structure(structure, params)[0]
        self.assertEqual(result.get_chemical_symbols(), ["Ge", "Ge", "Si", "Si"])
        self.assertIn("Occ(E", result.info.get("Config_type", ""))
        metadata = json.loads(result.info["random_occupancy"])
        self.assertEqual(metadata["eligible_sites"], 2)
        self.assertEqual(metadata["groups"], ["A"])
        self.assertEqual(metadata["actual_counts"], {"Ge": 2})

        no_match = structure.copy()
        no_match.arrays["group"][:] = "B"
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(no_match, params)
        self.assertEqual(raised.exception.code, "random_occupancy.no_matching_groups")

    def test_random_occupancy_missing_composition_and_invalid_counts_fail(self):
        operation = RandomOccupancyOperation()
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                self.structure,
                RandomOccupancyParams(source="Manual", manual=""),
            )
        self.assertEqual(raised.exception.code, "random_occupancy.empty_manual")
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                self.structure,
                RandomOccupancyParams(
                    source="Manual",
                    manual="Si",
                    samples=0,
                ),
            )
        self.assertEqual(raised.exception.code, "random_occupancy.invalid_samples")

    def test_random_occupancy_sources_are_strict_and_mutually_exclusive(self):
        structure = Atoms("Si4", positions=np.zeros((4, 3)))
        structure.info["Config_type"] = "base|Comp(Fe=1)|Comp(Co=1)"
        operation = RandomOccupancyOperation()

        auto = operation.run_structure(
            structure,
            RandomOccupancyParams(
                source="Auto (Comp tag)",
                manual="Ni:1",
                use_seed=True,
                seed=1,
            ),
        )[0]
        self.assertEqual(set(auto.get_chemical_symbols()), {"Co"})

        manual = operation.run_structure(
            structure,
            RandomOccupancyParams(
                source="Manual",
                manual="Ni:1",
                use_seed=True,
                seed=1,
            ),
        )[0]
        self.assertEqual(set(manual.get_chemical_symbols()), {"Ni"})

        without_tag = structure.copy()
        without_tag.info["Config_type"] = "base"
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                without_tag,
                RandomOccupancyParams(
                    source="Auto (Comp tag)",
                    manual="Ni:1",
                ),
            )
        self.assertEqual(raised.exception.code, "random_occupancy.missing_comp_tag")

    def test_random_occupancy_summary_and_exact_nondivisible_metadata(self):
        structure = Atoms(
            "Si3",
            scaled_positions=((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5)),
            cell=np.eye(3) * 3.6,
            pbc=(True, False, True),
        )
        structure.new_array("marker", np.arange(3, dtype=np.int64))
        structure.info["source_note"] = "keep-me"
        params = RandomOccupancyParams(
            source="Manual",
            manual="Fe:1,Co:1",
            mode="Exact",
            samples=2,
            use_seed=True,
            seed=7,
        )
        operation = RandomOccupancyOperation()
        summary = operation.sampling_summary(structure, params)

        self.assertEqual(summary["target"], {"Fe": 0.5, "Co": 0.5})
        self.assertEqual(summary["eligible_indices"], (0, 1, 2))
        self.assertEqual(summary["eligible_count"], 3)
        self.assertEqual(summary["fixed_counts"], {"Fe": 2, "Co": 1})
        self.assertEqual(
            summary["fixed_fractions"],
            {"Fe": 2 / 3, "Co": 1 / 3},
        )
        self.assertEqual(summary["outputs_per_input"], 2)

        input_symbols = structure.get_chemical_symbols()
        input_positions = structure.positions.copy()
        input_cell = structure.cell.array.copy()
        input_pbc = structure.pbc.copy()
        input_marker = structure.arrays["marker"].copy()
        outputs = operation.run_structure(structure, params)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(structure.get_chemical_symbols(), input_symbols)
        self.assertNotIn("random_occupancy", structure.info)

        for sample_index, output in enumerate(outputs):
            np.testing.assert_array_equal(output.positions, input_positions)
            np.testing.assert_array_equal(output.cell.array, input_cell)
            np.testing.assert_array_equal(output.pbc, input_pbc)
            np.testing.assert_array_equal(output.arrays["marker"], input_marker)
            self.assertEqual(output.info["source_note"], "keep-me")
            metadata = json.loads(output.info["random_occupancy"])
            self.assertEqual(metadata["target"], {"Co": 0.5, "Fe": 0.5})
            self.assertEqual(metadata["actual_counts"], {"Co": 1, "Fe": 2})
            self.assertEqual(
                metadata["actual_fractions"],
                {"Co": 1 / 3, "Fe": 2 / 3},
            )
            self.assertEqual(metadata["eligible_sites"], 3)
            self.assertEqual(metadata["groups"], [])
            self.assertEqual(metadata["mode"], "Exact")
            self.assertEqual(metadata["sample_index"], sample_index)
            self.assertIsInstance(metadata["seed"], int)

        unseeded = operation.run_structure(
            structure,
            RandomOccupancyParams(
                source="Manual",
                manual="Fe:1,Co:1",
                use_seed=False,
                seed=0,
            ),
        )[0]
        self.assertIsNone(
            json.loads(unseeded.info["random_occupancy"])["seed"]
        )

    def test_random_occupancy_rejects_invalid_inputs_without_mutation(self):
        operation = RandomOccupancyOperation()
        structure = Atoms("Si4", positions=np.zeros((4, 3)))
        structure.info["Config_type"] = "base"
        original_symbols = structure.get_chemical_symbols()
        original_info = dict(structure.info)
        cases = [
            ("random_occupancy.invalid_source", RandomOccupancyParams(source="Auto", manual="Fe:1")),
            ("random_occupancy.invalid_mode", RandomOccupancyParams(source="Manual", manual="Fe:1", mode="Other")),
            ("random_occupancy.invalid_seed", RandomOccupancyParams(source="Manual", manual="Fe:1", use_seed=True, seed=-1)),
            ("random_occupancy.invalid_seed", RandomOccupancyParams(source="Manual", manual="Fe:1", use_seed=False, seed=-1)),
            ("random_occupancy.zero_target", RandomOccupancyParams(source="Manual", manual="Fe:0,Co:0")),
            ("random_occupancy.empty_group_filter", RandomOccupancyParams(source="Manual", manual="Fe:1", group_filter=" , , ")),
            ("random_occupancy.unknown_elements", RandomOccupancyParams(source="Manual", manual="Qq:1")),
            ("random_occupancy.invalid_composition", RandomOccupancyParams(source="Manual", manual="Fe:not-a-number")),
        ]
        for code, params in cases:
            with self.subTest(code=code):
                with self.assertRaises(CardOperationError) as raised:
                    operation.run_structure(structure, params)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(structure.get_chemical_symbols(), original_symbols)
                self.assertEqual(structure.info, original_info)

        empty = Atoms()
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                empty,
                RandomOccupancyParams(source="Manual", manual="Fe:1"),
            )
        self.assertEqual(raised.exception.code, "random_occupancy.empty_structure")
        self.assertEqual(empty.info, {})

    def test_random_occupancy_clears_species_dependent_reference_data(self):
        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=np.diag([8.0, 7.0, 6.0]),
            pbc=(True, False, True),
        )
        structure.calc = SinglePointCalculator(
            structure,
            energy=-2.0,
            forces=np.ones((4, 3)),
            stress=np.arange(6, dtype=float),
        )
        structure.info.update(
            {
                "energy": -2.0,
                "free_energy": -2.1,
                "stress": [1.0] * 6,
                "virial": [2.0] * 9,
                "dipole": [0.0, 0.0, 1.0],
                "magmom": 3.0,
                "source_note": "keep-me",
            }
        )
        structure.new_array("group", np.array(["A", "A", "B", "B"], dtype="U1"))
        structure.new_array("sublattice", np.array(["A", "A", "B", "B"], dtype="U1"))
        structure.new_array("marker", np.arange(4, dtype=np.int64))
        for name, values in {
            "forces": np.ones((4, 3)),
            "force": np.ones((4, 3)) * 2,
            "energies": np.arange(4, dtype=float),
            "atomic_energy": np.arange(4, dtype=float) + 1,
            "magmoms": np.ones((4, 3)),
            "charges": np.arange(4, dtype=float),
            "bec": np.ones((4, 3, 3)),
            "born_effective_charges": np.ones((4, 3, 3)) * 2,
            "spin": np.ones((4, 3)) * 3,
            "initial_magmoms": np.ones((4, 3)) * 4,
            "initial_charges": np.ones(4) * 5,
        }.items():
            structure.new_array(name, values)

        output = RandomOccupancyOperation().run_structure(
            structure,
            RandomOccupancyParams(
                source="Manual",
                manual="Ge:1",
                mode="Exact",
                samples=1,
                use_seed=True,
                seed=3,
            ),
        )[0]

        self.assertIsNone(output.calc)
        for key in ("energy", "free_energy", "stress", "virial", "dipole", "magmom"):
            self.assertNotIn(key, output.info)
        for key in (
            "forces",
            "force",
            "energies",
            "atomic_energy",
            "magmoms",
            "charges",
            "bec",
            "born_effective_charges",
            "spin",
            "initial_magmoms",
            "initial_charges",
        ):
            self.assertNotIn(key, output.arrays)

        self.assertEqual(output.info["source_note"], "keep-me")
        for key in ("group", "sublattice", "marker"):
            np.testing.assert_array_equal(output.arrays[key], structure.arrays[key])
        np.testing.assert_array_equal(output.positions, structure.positions)
        np.testing.assert_array_equal(output.cell.array, structure.cell.array)
        np.testing.assert_array_equal(output.pbc, structure.pbc)

        self.assertIsNotNone(structure.calc)
        self.assertIn("energy", structure.info)
        self.assertIn("spin", structure.arrays)

    def test_random_occupancy_random_mode_is_seeded_and_respects_sample_contract(self):
        structure = Atoms(
            "Si12",
            positions=np.column_stack(
                [np.arange(12, dtype=float), np.zeros(12), np.zeros(12)]
            ),
            cell=[14.0, 4.0, 4.0],
            pbc=True,
        )
        operation = RandomOccupancyOperation()
        params = RandomOccupancyParams(
            source="Manual",
            manual="Co:0.25,Ni:0.75",
            mode="Random",
            samples=4,
            use_seed=True,
            seed=23,
        )

        first = operation.run_structure(structure, params)
        repeated = operation.run_structure(structure, params)

        self.assertEqual(len(first), 4)
        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first],
            [atoms.get_chemical_symbols() for atoms in repeated],
        )
        for atoms in first:
            self.assertEqual(len(atoms), len(structure))
            self.assertTrue(
                set(atoms.get_chemical_symbols()).issubset({"Co", "Ni"})
            )
            metadata = json.loads(atoms.info["random_occupancy"])
            self.assertEqual(sum(metadata["actual_counts"].values()), 12)
            self.assertEqual(metadata["mode"], "Random")

        observed_counts = set()
        for seed in range(20):
            output = operation.run_structure(
                structure,
                RandomOccupancyParams(
                    source="Manual",
                    manual="Co:0.25,Ni:0.75",
                    mode="Random",
                    samples=1,
                    use_seed=True,
                    seed=seed,
                ),
            )[0]
            observed_counts.add(
                json.loads(output.info["random_occupancy"])["actual_counts"]["Co"]
            )
        self.assertGreater(len(observed_counts), 1)

    def test_random_occupancy_samples_allow_duplicate_arrangements(self):
        structure = Atoms("Si2", positions=np.zeros((2, 3)))
        outputs = RandomOccupancyOperation().run_structure(
            structure,
            RandomOccupancyParams(
                source="Manual",
                manual="Fe:1,Co:1",
                mode="Exact",
                samples=10,
                use_seed=True,
                seed=3,
            ),
        )
        self.assertEqual(len(outputs), 10)
        self.assertLessEqual(
            len({tuple(output.get_chemical_symbols()) for output in outputs}),
            2,
        )
        self.assertEqual(
            [json.loads(output.info["random_occupancy"])["sample_index"] for output in outputs],
            list(range(10)),
        )

    def test_random_occupancy_legacy_json_roundtrip(self):
        legacy = {
            "class": "RandomOccupancyCard",
            "check_state": True,
            "source": "Manual",
            "manual": "Fe:0.4,Co:0.6",
            "mode": "Random",
            "samples": [5],
            "group_filter": "A,B",
            "use_seed": True,
            "seed": [61],
        }
        card = RandomOccupancyCard()
        card.from_dict(legacy)
        restored = RandomOccupancyCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_composition_gradient_operation_and_card_roundtrip(self):
        base = Atoms(
            symbols=["Ni"] * 8,
            positions=[[float(i), 0.0, 0.0] for i in range(8)],
            cell=np.diag([8.0, 2.0, 2.0]),
            pbc=[True, False, False],
        )
        params = CompositionGradientParams(
            elements="Ni,Co",
            start_composition="Ni:1,Co:0",
            end_composition="Ni:0,Co:1",
            axis="a",
            bins=4,
            samples=2,
            use_seed=True,
            seed=3,
        )
        results = CompositionGradientOperation().run_structure(base, params)
        self.assertEqual(len(results), 2)
        self.assertTrue(all("CompGrad(ax=a,b=4" in str(atoms.info.get("Config_type", "")) for atoms in results))
        self.assertEqual(results[0].get_chemical_symbols()[:2], ["Ni", "Ni"])
        self.assertEqual(results[0].get_chemical_symbols()[-2:], ["Co", "Co"])

        card = CompositionGradientCard()
        card.elements_edit.setText("Ni,Co")
        self.assertEqual(card.axis_combo.currentText(), "Lattice a")
        self.assertEqual(card.get_params().axis, "a")
        card.bins_frame.set_input_value([4])
        card.samples_frame.set_input_value([2])
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([3])
        self.assertFalse(card.seed_field.isHidden())
        restored = CompositionGradientCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_composition_gradient_uses_lattice_coordinates_and_loads_legacy_axis(self):
        cell = np.asarray(
            [
                [4.0, 0.0, 0.0],
                [-3.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
        scaled = np.asarray(
            [
                [0.1, 0.9, 0.5],
                [0.9, 0.1, 0.5],
                [0.2, 0.8, 0.5],
                [0.8, 0.2, 0.5],
            ]
        )
        base = Atoms("Ni4", cell=cell, pbc=True)
        base.set_scaled_positions(scaled)

        result = CompositionGradientOperation().run_structure(
            base,
            CompositionGradientParams(axis="a", bins=2, use_seed=True, seed=1),
        )[0]

        self.assertEqual(result.get_chemical_symbols(), ["Ni", "Co", "Ni", "Co"])

        legacy = CompositionGradientCard()
        legacy.from_dict(
            {
                "class": "CompositionGradientCard",
                "check_state": True,
                "params": {
                    **params_to_dict(CompositionGradientParams()),
                    "axis": "x",
                },
            }
        )
        self.assertEqual(legacy.get_params().axis, "a")
        self.assertEqual(legacy.axis_combo.currentText(), "Lattice a")

    def test_composition_gradient_target_elements_preserve_other_sublattice(self):
        cell = np.asarray(
            [
                [5.0, 0.0, 0.0],
                [1.5, 4.0, 0.0],
                [0.3, 0.5, 5.0],
            ]
        )
        structure = Atoms(
            "NiONiO",
            scaled_positions=[
                [0.1, 0.1, 0.2],
                [0.2, 0.2, 0.3],
                [0.8, 0.8, 0.7],
                [0.9, 0.9, 0.8],
            ],
            cell=cell,
            pbc=True,
        )
        result = CompositionGradientOperation().run_structure(
            structure,
            CompositionGradientParams(
                elements="Ni,Co",
                start_composition="Ni:1,Co:0",
                end_composition="Ni:0,Co:1",
                axis="c",
                bins=2,
                target_mode="listed",
                target_elements="Ni",
                samples=1,
                use_seed=True,
                seed=5,
            ),
        )[0]

        self.assertEqual(result.get_chemical_symbols(), ["Ni", "O", "Co", "O"])
        np.testing.assert_allclose(result.positions, structure.positions)
        np.testing.assert_allclose(result.cell.array, structure.cell.array)

    def test_composition_gradient_requires_two_groups_and_reports_effective_groups(self):
        structure = Atoms(
            "Ni8",
            scaled_positions=[[index / 8.0, 0.0, 0.0] for index in range(8)],
            cell=np.diag([8.0, 2.0, 2.0]),
            pbc=True,
        )
        operation = CompositionGradientOperation()

        with self.assertRaisesRegex(ValueError, "at least two equal-count groups"):
            operation.run_structure(
                structure,
                CompositionGradientParams(bins=1),
            )

        summary = operation.sampling_summary(
            CompositionGradientParams(bins=20, samples=3), structure
        )
        self.assertEqual(summary["candidate_sites"], 8)
        self.assertEqual(summary["requested_groups"], 20)
        self.assertEqual(summary["effective_groups"], 8)
        self.assertEqual(summary["min_group_size"], 1)
        self.assertEqual(summary["max_group_size"], 1)
        self.assertEqual(summary["outputs_per_input"], 3)

    def test_composition_gradient_explicit_scope_and_magnetic_array_contract(self):
        structure = Atoms(
            "NiONiO",
            scaled_positions=[
                [0.1, 0.1, 0.2],
                [0.2, 0.2, 0.3],
                [0.8, 0.8, 0.7],
                [0.9, 0.9, 0.8],
            ],
            cell=np.diag([4.0, 4.0, 4.0]),
            pbc=True,
        )
        spin = np.arange(12, dtype=float).reshape(4, 3)
        initial_magmoms = np.flip(spin, axis=1).copy()
        structure.new_array("spin", spin)
        structure.set_initial_magnetic_moments(initial_magmoms)

        result = CompositionGradientOperation().run_structure(
            structure,
            CompositionGradientParams(
                bins=2,
                target_mode="listed",
                target_elements="Ni",
                use_seed=True,
                seed=5,
            ),
        )[0]
        self.assertEqual(result.get_chemical_symbols(), ["Ni", "O", "Co", "O"])
        np.testing.assert_array_equal(result.arrays["spin"], spin)
        np.testing.assert_array_equal(
            result.arrays["initial_magmoms"], initial_magmoms
        )

        card = CompositionGradientCard()
        card.set_dataset([structure, structure.copy()])
        card.set_params(
            CompositionGradientParams(
                bins=8,
                target_mode="listed",
                target_elements="Ni",
                samples=3,
            )
        )
        self.assertEqual(card.get_params().target_mode, "listed")
        self.assertFalse(card.target_field.isHidden())
        self.assertIn("Inputs 2", card.get_guidance_text())
        self.assertIn("outputs 6", card.get_guidance_text())
        self.assertIn("Eligible sites 2", card.get_guidance_text())
        self.assertIn("second jump", card.get_guidance_text())
        self.assertIn("magnetic moments", card.get_guidance_text())
        self.assertIn("may repeat", card.get_guidance_text())

        restored = CompositionGradientCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_composition_sweep_quaternary_quinary(self):
        base = self.structure.copy()
        base.info.setdefault("Config_type", "base")

        sweep4 = CompositionSweepCard()
        sweep4.elements_edit.setText("Co,Cr,Ni,Al,Fe")
        sweep4.order_combo.setCurrentIndex(sweep4.order_combo.findData("4"))
        sweep4.method_combo.setCurrentText("Sobol")
        sweep4.n_points_frame.set_input_value([8])
        sweep4.max_output_frame.set_input_value([8])

        swept4 = sweep4.process_structure(base)
        self.assertEqual(len(swept4), 8)
        for atoms in swept4:
            cfg = str(atoms.info.get("Config_type", ""))
            comp_tokens = [t.strip() for t in cfg.split("|") if t.strip().startswith("Comp(") and t.strip().endswith(")")]
            self.assertTrue(comp_tokens)
            comp_items = [p for p in comp_tokens[-1][5:-1].split(",") if p.strip()]
            self.assertEqual(len(comp_items), 4)

        sweep5 = CompositionSweepCard()
        sweep5.elements_edit.setText("Co,Cr,Ni,Al,Fe")
        sweep5.order_combo.setCurrentIndex(sweep5.order_combo.findData("5"))
        sweep5.method_combo.setCurrentText("Sobol")
        sweep5.n_points_frame.set_input_value([6])
        sweep5.max_output_frame.set_input_value([6])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            swept5 = sweep5.process_structure(base)
        self.assertEqual(len(swept5), 6)
        self.assertFalse(
            any("balance properties of Sobol" in str(item.message) for item in caught)
        )
        for atoms in swept5:
            cfg = str(atoms.info.get("Config_type", ""))
            comp_tokens = [t.strip() for t in cfg.split("|") if t.strip().startswith("Comp(") and t.strip().endswith(")")]
            self.assertTrue(comp_tokens)
            comp_items = [p for p in comp_tokens[-1][5:-1].split(",") if p.strip()]
            self.assertEqual(len(comp_items), 5)
