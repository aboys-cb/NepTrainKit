from ase.geometry import geometry

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

    def test_vacancy_rule_count_mode_distinguishes_fixed_and_range(self):
        item = VacancyRuleItem()
        item.element_edit.setText("Si")
        item.fixed_count_frame.set_input_value([2])
        fixed = item.to_rule()
        self.assertEqual(fixed["count"], [2, 2])
        self.assertEqual(fixed["count_mode"], "fixed")

        item.count_mode_combo.setCurrentText("Random range")
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

    def test_vacancy_defect_card_concentration(self):
        card = VacancyDefectCard()
        structure = self.structure.copy()
        card.engine_type_combo.setCurrentIndex(1)
        card.concentration_radio_button.setChecked(True)
        card.num_radio_button.setChecked(False)
        card.concentration_condition_frame.set_input_value([0.6])
        card.num_condition_frame.set_input_value([1])
        card.count_mode_combo.setCurrentText("Random up to value")
        card.max_atoms_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(atoms) < len(structure) for atoms in results))

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

    def test_insert_defect_card_roundtrip(self):
        card = InsertDefectCard()
        card.mode_combo.setCurrentIndex(1)
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
        self.assertIn("SWC(L=2,dz=3)", results[0].info.get("Config_type", ""))

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
