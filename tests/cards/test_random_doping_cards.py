"""Random doping card."""

import json
import warnings
from unittest.mock import patch

from .card_test_base import *

from ase.calculators.singlepoint import SinglePointCalculator
from NepTrainKit.core.cards.alloy import sample_dopants
from NepTrainKit.core.cards.errors import CardOperationError


class TestRandomDopingCards(BaseCardTest):
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

    def test_random_doping_legacy_json_restores_existing_values(self):
        legacy_rule = {
            "target": "Si",
            "dopants": {"Ge": 0.6, "C": 0.4},
            "ratio_type": "mass",
            "use": "count",
            "count": [2, 4],
            "count_mode": "random",
            "group": ["surface"],
        }
        card = RandomDopingCard()
        card.from_dict(
            {
                "class": "RandomDopingCard",
                "check_state": True,
                "rules": json.dumps([legacy_rule]),
                "doping_type": "Exact",
                "max_atoms_condition": [3],
                "use_seed": True,
                "seed": [23],
            }
        )

        params = card.get_params()
        self.assertEqual(params.doping_type, "Exact")
        self.assertEqual(params.max_structures, 3)
        self.assertTrue(params.use_seed)
        self.assertEqual(params.seed, 23)
        self.assertEqual(len(params.rules), 1)
        restored_rule = params.rules[0]
        for key, value in legacy_rule.items():
            self.assertEqual(restored_rule[key], value)

    def test_random_doping_dopants_accept_bare_element(self):
        item = DopingRuleItem()
        item.target_edit.setText("Si")
        item.dopants_edit.setText("Ge")
        item.amount_mode_control.setCurrentIndex(
            item.amount_mode_control.findData("count")
        )
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

    def test_random_doping_ratio_control_matches_serialized_semantics(self):
        item = DopingRuleItem()
        item.target_edit.setText("Si")
        item.dopants_edit.setText("Ge")

        self.assertEqual(item.ratio_type_control.currentData(), "atom")
        self.assertEqual(item.to_rule()["ratio_type"], "atom")

        item.ratio_type_control.setCurrentIndex(
            item.ratio_type_control.findData("mass")
        )
        self.assertEqual(item.to_rule()["ratio_type"], "mass")

        item.from_rule({"target": "Si", "dopants": {"Ge": 1.0}, "ratio_type": "atom"})
        self.assertEqual(item.ratio_type_control.currentData(), "atom")
        self.assertEqual(item.to_rule()["ratio_type"], "atom")

        item.from_rule({"target": "Si", "dopants": {"Ge": 1.0}, "ratio_type": "mass"})
        self.assertEqual(item.ratio_type_control.currentData(), "mass")
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

        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                structure,
                RandomDopingParams(rules=[rule], use_seed=True, seed=1),
            )
        self.assertEqual(raised.exception.code, "random_doping.missing_group_array")

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
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                no_match,
                RandomDopingParams(rules=[rule], use_seed=True, seed=1),
            )
        self.assertEqual(raised.exception.code, "random_doping.no_candidates")

    def test_random_doping_rejects_deterministic_no_effect_and_oversized_count(self):
        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[20, 20, 20],
            pbc=True,
        )
        operation = RandomDopingOperation()
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
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
            )
        self.assertEqual(raised.exception.code, "random_doping.no_effect")

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

    def test_random_doping_summary_and_output_count_are_strict(self):
        structure = Atoms(
            "Si10",
            positions=np.arange(30, dtype=float).reshape(10, 3),
            cell=[40, 40, 40],
            pbc=[True, False, True],
        )
        operation = RandomDopingOperation()
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                structure,
                RandomDopingParams(rules=[], max_structures=4),
            )
        self.assertEqual(raised.exception.code, "random_doping.empty_rules")

        params = RandomDopingParams(
            rules=[
                {
                    "target": "Si",
                    "dopants": {"Ge": 1.0},
                    "use": "atomic_percent",
                    "percent": [20.0, 40.0],
                }
            ],
            max_structures=4,
            use_seed=True,
            seed=9,
        )
        summary = operation.sampling_summary(structure, params)
        outputs = operation.run_structure(structure, params)

        self.assertEqual(summary["outputs_per_input"], 4)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(summary["rules"][0]["eligible_sites"], 10)
        self.assertEqual(summary["rules"][0]["replacement_min"], 2)
        self.assertEqual(summary["rules"][0]["replacement_max"], 4)
        for output in outputs:
            np.testing.assert_allclose(output.cell.array, structure.cell.array)
            np.testing.assert_array_equal(output.pbc, structure.pbc)

    def test_random_doping_records_actual_output_and_clears_stale_labels(self):
        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[[8.0, 0.0, 0.0], [1.0, 9.0, 0.0], [0.5, 1.0, 10.0]],
            pbc=[True, False, True],
        )
        structure.info.update(
            {
                "Config_type": "base",
                "energy": -2.0,
                "free_energy": -2.1,
                "stress": np.ones(6),
                "marker": "keep",
            }
        )
        structure.new_array("forces", np.ones((4, 3)))
        structure.new_array("spin", np.ones((4, 3)))
        structure.new_array("group", np.array(["A", "A", "B", "B"], dtype=object))
        structure.calc = SinglePointCalculator(
            structure,
            energy=-2.0,
            forces=np.ones((4, 3)),
        )

        output = RandomDopingOperation().run_structure(
            structure,
            RandomDopingParams(
                rules=[
                    {
                        "target": "Si",
                        "dopants": {"Ge": 1.0},
                        "use": "count",
                        "count": [1, 1],
                        "count_mode": "fixed",
                        "group": ["A"],
                    }
                ],
                doping_type="Exact",
                max_structures=1,
                use_seed=True,
                seed=5,
            ),
        )[0]

        metadata = json.loads(output.info["random_doping"])
        self.assertEqual(metadata["total_replacements"], 1)
        self.assertEqual(metadata["rules"][0]["actual_dopant_counts"], {"Ge": 1})
        self.assertEqual(metadata["rules"][0]["eligible_sites"], 2)
        self.assertIn("Dop(n=1)", output.info["Config_type"])
        self.assertEqual(output.info["marker"], "keep")
        self.assertIsNone(output.calc)
        for key in ("energy", "free_energy", "stress"):
            self.assertNotIn(key, output.info)
        for key in ("forces", "spin"):
            self.assertNotIn(key, output.arrays)
        np.testing.assert_array_equal(output.arrays["group"], structure.arrays["group"])
        np.testing.assert_allclose(output.positions, structure.positions)
        np.testing.assert_allclose(output.cell.array, structure.cell.array)
        np.testing.assert_array_equal(output.pbc, structure.pbc)

    def test_random_doping_seed_is_derived_per_input_and_sample(self):
        base = Atoms(
            "Si6",
            positions=np.arange(18, dtype=float).reshape(6, 3),
            cell=[20, 20, 20],
            pbc=True,
        )
        first = base.copy()
        second = base.copy()
        first.info["Config_type"] = "input-A"
        second.info["Config_type"] = "input-B"
        params = RandomDopingParams(
            rules=[
                {
                    "target": "Si",
                    "dopants": {"Ge": 1.0},
                    "use": "count",
                    "count": [2, 2],
                    "count_mode": "fixed",
                }
            ],
            max_structures=2,
            use_seed=True,
            seed=17,
        )
        operation = RandomDopingOperation()
        first_run = operation.run_structure(first, params)
        repeated = operation.run_structure(first, params)
        second_run = operation.run_structure(second, params)

        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first_run],
            [atoms.get_chemical_symbols() for atoms in repeated],
        )
        first_seeds = [json.loads(atoms.info["random_doping"])["seed"] for atoms in first_run]
        second_seeds = [json.loads(atoms.info["random_doping"])["seed"] for atoms in second_run]
        self.assertEqual(first_seeds[1], first_seeds[0] + 1)
        self.assertNotEqual(first_seeds[0], second_seeds[0])

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

