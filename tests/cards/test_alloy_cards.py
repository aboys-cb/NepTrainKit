from .card_test_base import *


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

    def test_parse_composition_accepts_bare_elements(self):
        self.assertEqual(parse_composition("Ge"), {"Ge": 1.0})
        self.assertEqual(parse_composition("Ge,C"), {"Ge": 1.0, "C": 1.0})
        self.assertEqual(parse_composition("Ge:0.7,C"), {"Ge": 0.7, "C": 1.0})

    def test_conditional_replace_operation_is_ui_independent(self):
        results = ConditionalReplaceOperation().run_structure(
            self.structure.copy(),
            ConditionalReplaceParams(
                target="Si",
                replacements="Ge:1",
                condition="all",
                seed=1,
                mode=1,
            ),
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(all(symbol == "Ge" for symbol in results[0].get_chemical_symbols()))
        self.assertIn("Repl(Si->Ge)", results[0].info.get("Config_type", ""))

    def test_conditional_replace_card_roundtrip(self):
        card = ConditionalReplaceCard()
        card.target_edit.setText("Si")
        card.replacements_edit.setText("Ge:0.5,C:0.5")
        card.condition_edit.setText("z>=0")
        card.seed_frame.set_input_value([9])
        card.mode_combo.setCurrentIndex(1)

        restored = ConditionalReplaceCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_composition_sweep_and_random_occupancy_cards(self):
        base = self.structure.copy()
        base.info.setdefault("Config_type", "base")

        sweep = CompositionSweepCard()
        sweep.elements_edit.setText("Co,Ni")
        sweep.order_combo.setCurrentText("2")
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
            axis="x",
            bins=4,
            samples=2,
            use_seed=True,
            seed=3,
        )
        results = CompositionGradientOperation().run_structure(base, params)
        self.assertEqual(len(results), 2)
        self.assertTrue(all("CompGrad(ax=x,b=4" in str(atoms.info.get("Config_type", "")) for atoms in results))
        self.assertEqual(results[0].get_chemical_symbols()[:2], ["Ni", "Ni"])
        self.assertEqual(results[0].get_chemical_symbols()[-2:], ["Co", "Co"])

        card = CompositionGradientCard()
        card.elements_edit.setText("Ni,Co")
        card.bins_frame.set_input_value([4])
        card.samples_frame.set_input_value([2])
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([3])
        restored = CompositionGradientCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_composition_sweep_quaternary_quinary(self):
        base = self.structure.copy()
        base.info.setdefault("Config_type", "base")

        sweep4 = CompositionSweepCard()
        sweep4.elements_edit.setText("Co,Cr,Ni,Al,Fe")
        sweep4.order_combo.setCurrentText("4")
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
        sweep5.order_combo.setCurrentText("5")
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
