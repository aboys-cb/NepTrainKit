from .card_test_base import *


class TestLatticeCards(BaseCardTest):
    def test_supercell_card_variants(self):
        card = SuperCellCard()
        structure = self.structure.copy()
        original_cell = np.array(structure.get_cell())

        card.super_scale_condition_frame.set_input_value([2, 1, 1])
        card.super_scale_radio_button.setChecked(True)
        card.super_cell_radio_button.setChecked(False)
        card.max_atoms_radio_button.setChecked(False)
        card.behavior_type_combo.setCurrentIndex(0)

        direct_results = card.process_structure(structure)
        self.assertEqual(len(direct_results), 1)
        self.assertEqual(len(direct_results[0]), len(structure) * 2)
        new_cell = np.array(direct_results[0].get_cell())
        self.assertGreater(np.linalg.norm(new_cell[0]), np.linalg.norm(original_cell[0]))

        card.super_cell_radio_button.setChecked(True)
        card.super_scale_radio_button.setChecked(False)
        card.max_atoms_radio_button.setChecked(False)
        lengths = structure.cell.lengths()
        card.super_cell_condition_frame.set_input_value([
            lengths[0] * 2.1,
            lengths[1] * 1.1,
            lengths[2] * 1.1,
        ])

        cell_results = card.process_structure(structure)
        self.assertEqual(len(cell_results), 1)
        self.assertGreater(len(cell_results[0]), len(structure))

        card.behavior_type_combo.setCurrentIndex(1)
        card.max_atoms_radio_button.setChecked(True)
        card.super_cell_radio_button.setChecked(False)
        card.super_scale_radio_button.setChecked(False)
        card.max_atoms_condition_frame.set_input_value([len(structure) * 2])

        atoms_results = card.process_structure(structure)
        self.assertGreaterEqual(len(atoms_results), 1)
        self.assertTrue(any(len(atoms) > len(structure) for atoms in atoms_results))

    def test_supercell_card_fixed_axis_lock(self):
        card = SuperCellCard()
        structure = self.structure.copy()
        original_cell = np.array(structure.get_cell())
        original_lengths = np.linalg.norm(original_cell, axis=1)

        card.behavior_type_combo.setCurrentIndex(1)
        card.super_cell_radio_button.setChecked(True)
        card.super_scale_radio_button.setChecked(False)
        card.max_atoms_radio_button.setChecked(False)
        card.fixed_axis_c_checkbox.setChecked(True)
        card.fixed_scale_condition_frame.set_input_value([1, 1, 1])
        card.super_cell_condition_frame.set_input_value([
            original_lengths[0] * 2.1,
            original_lengths[1] * 2.1,
            original_lengths[2] * 4.0,
        ])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 4)
        self.assertTrue(
            all(
                np.isclose(np.linalg.norm(np.array(atoms.get_cell())[2]), original_lengths[2], atol=1e-6)
                for atoms in results
            )
        )
        self.assertTrue(
            any(
                np.linalg.norm(np.array(atoms.get_cell())[0]) > original_lengths[0] + 1e-6
                or np.linalg.norm(np.array(atoms.get_cell())[1]) > original_lengths[1] + 1e-6
                for atoms in results
            )
        )

        data = card.to_dict()
        restored = SuperCellCard()
        restored.from_dict(data)
        self.assertTrue(restored.fixed_axis_c_checkbox.isChecked())
        self.assertEqual(restored.fixed_scale_condition_frame.get_input_value(), [1, 1, 1])
        self.assertEqual(restored.super_cell_condition_frame.get_input_value(), list(card.super_cell_condition_frame.get_input_value()))

    def test_supercell_operation_matches_card_params(self):
        card = SuperCellCard()
        structure = self.structure.copy()
        card.super_scale_radio_button.setChecked(True)
        card.super_cell_radio_button.setChecked(False)
        card.max_atoms_radio_button.setChecked(False)
        card.super_scale_condition_frame.set_input_value([2, 1, 1])

        params = card.get_params()
        self.assertIsInstance(params, SuperCellParams)
        card_result = card.process_structure(structure)
        op_result = SuperCellOperation().run_structure(structure, params)

        self.assertEqual(len(card_result), len(op_result))
        self.assertEqual(len(card_result[0]), len(op_result[0]))
        self.assertIn("params", card.to_dict())

    def test_perturb_card_with_organic(self):
        card = PerturbCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.engine_type_combo.setCurrentIndex(1)
        card.scaling_condition_frame.set_input_value([0.1])
        card.num_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        displacements = [
            np.linalg.norm(atoms.get_positions() - structure.get_positions(), axis=1).max()
            for atoms in results
        ]
        self.assertTrue(any(delta > 0 for delta in displacements))

    def test_perturb_operation_is_ui_independent(self):
        params = PerturbParams(
            engine_type=1,
            max_distance=0.1,
            max_num=2,
            use_seed=True,
            seed=11,
        )
        results = PerturbOperation().run_structure(self.structure.copy(), params)

        self.assertEqual(len(results), 2)
        self.assertTrue(all("Pert(d=0.1,U)" in atoms.info.get("Config_type", "") for atoms in results))
        for atoms in results:
            displacements = atoms.get_positions() - self.structure.get_positions()
            self.assertLessEqual(float(np.linalg.norm(displacements, axis=1).max()), 0.1 + 1e-12)

    def test_perturb_max_distance_is_displacement_norm_limit(self):
        structure = Atoms(
            "HHeLi",
            positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=False,
        )
        params = PerturbParams(
            engine_type=1,
            max_distance=0.2,
            max_num=8,
            use_element_scaling=True,
            element_scalings={"H": 0.05, "Li": 0.0},
            use_seed=True,
            seed=11,
        )

        results = PerturbOperation().run_structure(structure.copy(), params)

        limits = np.array([0.05, 0.2, 0.0])
        for atoms in results:
            displacements = atoms.get_positions() - structure.get_positions()
            norms = np.linalg.norm(displacements, axis=1)
            np.testing.assert_array_less(norms, limits + 1e-12)

    def test_perturb_rejects_invalid_distance_limits(self):
        with self.assertRaisesRegex(ValueError, "max_distance"):
            PerturbOperation().run_structure(
                self.structure.copy(),
                PerturbParams(max_distance=-0.1, max_num=1),
            )
        with self.assertRaisesRegex(ValueError, "max_distance"):
            PerturbOperation().run_structure(
                self.structure.copy(),
                PerturbParams(
                    max_distance=0.1,
                    max_num=1,
                    use_element_scaling=True,
                    element_scalings={"Si": float("nan")},
                ),
            )

    def test_cell_scaling_card_options(self):
        card = CellScalingCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.perturb_angle_checkbox.setChecked(False)
        card.engine_type_combo.setCurrentIndex(1)
        card.scaling_condition_frame.set_input_value([0.05])
        card.num_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        original_lengths = np.array(structure.cell.cellpar()[:3])
        self.assertTrue(
            any(
                not np.allclose(
                    np.array(atoms.cell.cellpar()[:3]),
                    original_lengths,
                    atol=1e-6,
                )
                for atoms in results
            )
        )

    def test_cell_scaling_operation_is_ui_independent(self):
        params = CellScalingParams(
            engine_type=1,
            max_scaling=0.05,
            max_num=2,
            perturb_angle=False,
            use_seed=True,
            seed=7,
        )
        results = CellScalingOperation().run_structure(self.structure.copy(), params)

        self.assertEqual(len(results), 2)
        self.assertTrue(all("LSc(max=0.05,U)" in atoms.info.get("Config_type", "") for atoms in results))

    def test_cell_strain_card_uniaxial(self):
        card = CellStrainCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.engine_type_combo.setText("uniaxial")
        card.strain_x_frame.set_input_value([1.0, 1.0, 1.0])
        card.strain_y_frame.set_input_value([0.0, 0.0, 1.0])
        card.strain_z_frame.set_input_value([0.0, 0.0, 1.0])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 3)
        original_lengths = np.array(structure.cell.cellpar()[:3])
        self.assertTrue(
            any(
                not np.allclose(
                    np.array(atoms.cell.cellpar()[:3]),
                    original_lengths,
                    atol=1e-6,
                )
                for atoms in results
            )
        )

    def test_cell_strain_operation_is_ui_independent(self):
        params = CellStrainParams(
            axes="X",
            x_range=(1.0, 1.0, 1.0),
            y_range=(0.0, 0.0, 1.0),
            z_range=(0.0, 0.0, 1.0),
            identify_organic=False,
        )
        results = CellStrainOperation().run_structure(self.structure.copy(), params)

        self.assertEqual(len(results), 1)
        self.assertIn("Str(X=1%)", results[0].info.get("Config_type", ""))

    def test_shear_matrix_card(self):
        card = ShearMatrixCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.symmetric_checkbox.setChecked(False)
        card.xy_frame.set_input_value([1.0, 1.0, 1.0])
        card.yz_frame.set_input_value([0.0, 0.0, 1.0])
        card.xz_frame.set_input_value([0.0, 0.0, 1.0])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 1)
        self.assertFalse(
            np.allclose(
                np.array(results[0].get_cell()),
                np.array(structure.get_cell()),
                atol=1e-6,
            )
        )

    def test_shear_matrix_operation_is_ui_independent(self):
        params = ShearMatrixParams(
            xy_range=(1.0, 1.0, 1.0),
            yz_range=(0.0, 0.0, 1.0),
            xz_range=(0.0, 0.0, 1.0),
            symmetric=False,
        )
        results = ShearMatrixOperation().run_structure(self.structure.copy(), params)

        self.assertEqual(len(results), 1)
        self.assertIn("Shr(xy=1%,sym=0)", results[0].info.get("Config_type", ""))

    def test_shear_angle_card(self):
        card = ShearAngleCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.alpha_frame.set_input_value([1.0, 1.0, 1.0])
        card.beta_frame.set_input_value([0.0, 0.0, 1.0])
        card.gamma_frame.set_input_value([0.0, 0.0, 1.0])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 1)
        original_angles = np.array(structure.cell.cellpar()[3:])
        new_angles = np.array(results[0].cell.cellpar()[3:])
        self.assertFalse(np.allclose(new_angles, original_angles, atol=1e-6))

    def test_shear_angle_operation_is_ui_independent(self):
        params = ShearAngleParams(
            alpha_range=(1.0, 1.0, 1.0),
            beta_range=(0.0, 0.0, 1.0),
            gamma_range=(0.0, 0.0, 1.0),
        )
        results = ShearAngleOperation().run_structure(self.structure.copy(), params)

        self.assertEqual(len(results), 1)
        self.assertIn("Ang(a=1)", results[0].info.get("Config_type", ""))

    def test_vibration_mode_perturb_card(self):
        card = VibrationModePerturbCard()
        structure = self.structure.copy()
        natoms = len(structure)
        n_modes = min(3 * natoms, 6)
        mode_vectors = np.zeros((n_modes, natoms, 3))
        for idx in range(n_modes):
            atom_index = idx % natoms
            component = idx % 3
            mode_vectors[idx, atom_index, component] = 1.0
        freq_values = np.linspace(50.0, 300.0, n_modes)
        for mode_idx in range(n_modes):
            structure.new_array(f"vibration_mode_{mode_idx}_x", mode_vectors[mode_idx, :, 0])
            structure.new_array(f"vibration_mode_{mode_idx}_y", mode_vectors[mode_idx, :, 1])
            structure.new_array(f"vibration_mode_{mode_idx}_z", mode_vectors[mode_idx, :, 2])
            structure.new_array(
                f"vibration_frequency_{mode_idx}",
                np.full(natoms, freq_values[mode_idx], dtype=float),
            )

        card.amplitude_frame.set_input_value([0.05])
        card.modes_frame.set_input_value([2])
        card.min_freq_frame.set_input_value([1.0])
        card.num_condition_frame.set_input_value([3])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 3)
        base_positions = structure.get_positions()
        displacements = [
            np.linalg.norm(atoms.get_positions() - base_positions, axis=1).max()
            for atoms in results
        ]
        self.assertTrue(all(delta > 0 for delta in displacements))
        self.assertTrue(all(len(atoms) == len(structure) for atoms in results))

        op_results = VibrationModePerturbOperation().run_structure(
            structure,
            VibrationModePerturbParams(
                amplitude=0.05,
                modes_per_sample=2,
                min_frequency=1.0,
                max_num=2,
                use_seed=True,
                seed=4,
            ),
        )
        self.assertEqual(len(op_results), 2)

        restored = VibrationModePerturbCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())
