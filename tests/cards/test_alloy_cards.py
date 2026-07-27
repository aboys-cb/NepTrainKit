from .card_test_base import *

from NepTrainKit.core.cards.alloy import sample_dopants


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

        with self.assertRaisesRegex(ValueError, "requests 10 replacements, but only 4"):
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

        with self.assertRaisesRegex(ValueError, "requires atoms.arrays\\['group'\\]"):
            operation.run_structure(structure, params)

        structure.new_array("group", np.array(["A", "A", "B", "B"], dtype=object))
        result = operation.run_structure(structure, params)[0]
        self.assertEqual(result.get_chemical_symbols(), ["Ge", "Ge", "Si", "Si"])
        self.assertIn("Occ(E", result.info.get("Config_type", ""))

        no_match = structure.copy()
        no_match.arrays["group"][:] = "B"
        with self.assertRaisesRegex(ValueError, "matched no atoms: A"):
            operation.run_structure(no_match, params)

    def test_random_occupancy_missing_composition_and_invalid_counts_fail(self):
        operation = RandomOccupancyOperation()
        with self.assertRaisesRegex(ValueError, "requires a Comp"):
            operation.run_structure(
                self.structure,
                RandomOccupancyParams(source="Manual", manual=""),
            )
        with self.assertRaisesRegex(ValueError, "samples must be >= 1"):
            operation.run_structure(
                self.structure,
                RandomOccupancyParams(
                    source="Manual",
                    manual="Si",
                    samples=0,
                ),
            )

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

        swept5 = sweep5.process_structure(base)
        self.assertEqual(len(swept5), 6)
        for atoms in swept5:
            cfg = str(atoms.info.get("Config_type", ""))
            comp_tokens = [t.strip() for t in cfg.split("|") if t.strip().startswith("Comp(") and t.strip().endswith(")")]
            self.assertTrue(comp_tokens)
            comp_items = [p for p in comp_tokens[-1][5:-1].split(",") if p.strip()]
            self.assertEqual(len(comp_items), 5)
