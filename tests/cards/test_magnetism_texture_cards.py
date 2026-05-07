from .magnetism_test_base import *


class TestMagnetismTextureCards(MagnetismCardTest):
    def test_spin_spiral_card_periods_and_chirality(self):
        structure = self._spin_chain()

        card = SpinSpiralCard()
        card.source_combo.setCurrentText("Map/default magnitude")
        card.map_edit.setText("Fe:2.0")
        card.period_frame.set_input_value([2.0, 4.0, 2.0])
        card.phase_frame.set_input_value([0.0, 0.0, 15.0])
        card.mz_frame.set_input_value([0.0, 0.0, 0.1])
        card.chirality_combo.setCurrentText("Both")
        card.max_output_frame.set_input_value([10])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 4)

        for atoms in results:
            moments = np.array(atoms.get_initial_magnetic_moments(), dtype=float)
            self.assertEqual(moments.shape, (4, 3))
            self.assertTrue(np.allclose(np.linalg.norm(moments, axis=1), 2.0, atol=1e-6))
            self.assertTrue(np.allclose(moments[:, 2], 0.0, atol=1e-6))
            self.assertIn("Helix(", str(atoms.info.get("Config_type", "")))

        cw = next(
            atoms for atoms in results
            if "L=4" in str(atoms.info.get("Config_type", "")) and "chi=cw" in str(atoms.info.get("Config_type", ""))
        )
        ccw = next(
            atoms for atoms in results
            if "L=4" in str(atoms.info.get("Config_type", "")) and "chi=ccw" in str(atoms.info.get("Config_type", ""))
        )
        cw_m = np.array(cw.get_initial_magnetic_moments(), dtype=float)
        ccw_m = np.array(ccw.get_initial_magnetic_moments(), dtype=float)
        self.assertAlmostEqual(cw_m[1, 0], -ccw_m[1, 0], places=6)
        self.assertAlmostEqual(cw_m[1, 1], ccw_m[1, 1], places=6)

    def test_spin_spiral_card_mz_scan_generates_multiple_conical_states(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        card = SpinSpiralCard()
        card.period_frame.set_input_value([6.0, 6.0, 1.0])
        card.phase_frame.set_input_value([0.0, 0.0, 15.0])
        card.mz_frame.set_input_value([0.0, 0.8, 0.4])
        card.chirality_combo.setCurrentText("Clockwise")
        card.max_output_frame.set_input_value([10])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 3)

        expected_mz_values = [0.0, 0.4, 0.8]
        for atoms, expected_mz in zip(results, expected_mz_values):
            moments = np.array(atoms.get_initial_magnetic_moments(), dtype=float)
            norms = np.linalg.norm(moments, axis=1)
            self.assertTrue(np.allclose(norms, 2.0, atol=1e-6))
            self.assertTrue(np.allclose(moments[:, 2], norms * expected_mz, atol=1e-6))
            tag = str(atoms.info.get("Config_type", ""))
            self.assertIn(f"mz={expected_mz:.6g}", tag)
            if expected_mz == 0.0:
                self.assertIn("Helix(", tag)
            else:
                self.assertIn("Spiral(", tag)

    def test_spin_spiral_card_commensurate_period_filter(self):
        structure = self._spin_chain()

        card = SpinSpiralCard()
        card.source_combo.setCurrentText("Map/default magnitude")
        card.map_edit.setText("Fe:2.0")
        card.period_frame.set_input_value([4.0, 8.0, 2.0])
        card.phase_frame.set_input_value([0.0, 0.0, 15.0])
        card.mz_frame.set_input_value([0.0, 0.0, 0.1])
        card.chirality_combo.setCurrentText("Both")
        card.commensurate_checkbox.setChecked(True)
        card.max_output_frame.set_input_value([10])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 4)

        tags = [str(atoms.info.get("Config_type", "")) for atoms in results]
        self.assertTrue(all(("L=4" in tag or "L=8" in tag) for tag in tags))
        self.assertFalse(any("L=6" in tag for tag in tags))

        data = card.to_dict()
        restored = SpinSpiralCard()
        restored.from_dict(data)
        self.assertTrue(restored.commensurate_checkbox.isChecked())

    def test_spin_spiral_card_commensurate_period_suggests_supercell(self):
        structure = self._spin_chain()
        suggestion = SpinSpiralCard._suggest_supercell_multipliers(
            [6.0, 10.0],
            structure=structure,
            axis=np.array([0.0, 0.0, 1.0]),
        )
        self.assertIsNotNone(suggestion)
        suggested_period, multipliers = suggestion
        self.assertAlmostEqual(suggested_period, 6.0)
        self.assertEqual(multipliers, [1, 1, 3])

    def test_spin_spiral_card_commensurate_period_discovery_uses_continuous_range(self):
        structure = self._spin_chain()
        structure.set_cell(np.diag([6.0, 6.0, 31.0]))

        card = SpinSpiralCard()
        card.source_combo.setCurrentText("Map/default magnitude")
        card.map_edit.setText("Fe:2.0")
        card.period_frame.set_input_value([15.0, 16.0, 1.0])
        card.phase_frame.set_input_value([0.0, 0.0, 15.0])
        card.mz_frame.set_input_value([0.0, 0.0, 0.1])
        card.chirality_combo.setCurrentText("Both")
        card.commensurate_checkbox.setChecked(True)
        card.max_output_frame.set_input_value([10])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        tags = [str(atoms.info.get("Config_type", "")) for atoms in results]
        self.assertTrue(all("L=15.5" in tag for tag in tags))

    def test_spin_spiral_card_layer_locked_phase_keeps_each_layer_rigid(self):
        structure = self._spin_bilayer_chain()
        structure.set_initial_magnetic_moments([2.0] * len(structure))

        continuous = SpinSpiralCard()
        continuous.period_frame.set_input_value([4.0, 4.0, 1.0])
        continuous.phase_frame.set_input_value([0.0, 0.0, 15.0])
        continuous.mz_frame.set_input_value([0.0, 0.0, 0.1])
        continuous.chirality_combo.setCurrentText("Clockwise")

        layer_locked = SpinSpiralCard()
        layer_locked.period_frame.set_input_value([4.0, 4.0, 1.0])
        layer_locked.phase_frame.set_input_value([0.0, 0.0, 15.0])
        layer_locked.mz_frame.set_input_value([0.0, 0.0, 0.1])
        layer_locked.chirality_combo.setCurrentText("Clockwise")
        layer_locked.phase_mode_combo.setCurrentText("Layer-locked")
        layer_locked.layer_tol_frame.set_input_value([0.05])

        continuous_result = continuous.process_structure(structure)[0]
        layer_locked_result = layer_locked.process_structure(structure)[0]

        continuous_moments = np.array(continuous_result.get_initial_magnetic_moments(), dtype=float)
        layer_locked_moments = np.array(layer_locked_result.get_initial_magnetic_moments(), dtype=float)

        self.assertFalse(np.allclose(continuous_moments[0], continuous_moments[1], atol=1e-6))
        self.assertTrue(np.allclose(layer_locked_moments[0], layer_locked_moments[1], atol=1e-6))
        self.assertTrue(np.allclose(layer_locked_moments[2], layer_locked_moments[3], atol=1e-6))
        self.assertTrue(np.allclose(layer_locked_moments[4], layer_locked_moments[5], atol=1e-6))
        self.assertTrue(np.allclose(layer_locked_moments[6], layer_locked_moments[7], atol=1e-6))
        self.assertIn("pm=layer", str(layer_locked_result.info.get("Config_type", "")))

    def test_spin_spiral_card_existing_magnitudes_and_roundtrip(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([1.0, 2.0, 3.0, 4.0])

        card = SpinSpiralCard()
        card.axis_frame.set_input_value([0.0, 0.0, 1.0])
        card.period_frame.set_input_value([6.0, 6.0, 1.0])
        card.phase_frame.set_input_value([30.0, 30.0, 15.0])
        card.mz_frame.set_input_value([0.5, 0.5, 0.1])
        card.chirality_combo.setCurrentText("Clockwise")
        card.source_combo.setCurrentText("Existing initial magmoms")
        card.apply_edit.setText("Fe")
        card.max_output_frame.set_input_value([5])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 1)
        moments = np.array(results[0].get_initial_magnetic_moments(), dtype=float)
        norms = np.linalg.norm(moments, axis=1)
        self.assertTrue(np.allclose(norms, [1.0, 2.0, 3.0, 4.0], atol=1e-6))
        self.assertTrue(np.allclose(moments[:, 2], norms * 0.5, atol=1e-6))
        self.assertIn("Spiral(", str(results[0].info.get("Config_type", "")))
        self.assertIn("chi=cw", str(results[0].info.get("Config_type", "")))

        data = card.to_dict()
        restored = SpinSpiralCard()
        restored.from_dict(data)
        self.assertEqual(restored.parameter_mode_combo.currentText(), "Period (L_D)")
        self.assertEqual(restored.chirality_combo.currentText(), "Clockwise")
        self.assertEqual(restored.source_combo.currentText(), "Existing initial magmoms")
        self.assertEqual(restored.apply_edit.text(), "Fe")
        self.assertEqual(restored.period_frame.get_input_value(), [6.0, 6.0, 1.0])
        self.assertEqual(restored.mz_frame.get_input_value(), [0.5, 0.5, 0.1])
        self.assertEqual(restored.phase_mode_combo.currentText(), "Continuous by position")
        self.assertEqual(restored.layer_tol_frame.get_input_value(), [0.05])
        self.assertFalse(restored.map_label.isVisible())
        self.assertFalse(restored.default_label.isVisible())

        legacy = SpinSpiralCard()
        legacy.from_dict({"check_state": True, "mz": [0.25]})
        self.assertEqual(legacy.mz_frame.get_input_value(), [0.25, 0.25, 0.1])

    def test_spin_spiral_card_layer_locked_roundtrip(self):
        card = SpinSpiralCard()
        card.phase_mode_combo.setCurrentText("Layer-locked")
        card.layer_tol_frame.set_input_value([0.08])

        data = card.to_dict()
        restored = SpinSpiralCard()
        restored.from_dict(data)

        self.assertEqual(restored.phase_mode_combo.currentText(), "Layer-locked")
        self.assertEqual(restored.layer_tol_frame.get_input_value(), [0.08])
        self.assertFalse(restored.layer_tol_label.isHidden())
        self.assertTrue(restored.layer_tol_frame.isEnabled())

    def test_spin_spiral_card_angle_gradient_mode_matches_period_mode(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        by_period = SpinSpiralCard()
        by_period.period_frame.set_input_value([4.0, 4.0, 1.0])
        by_period.phase_frame.set_input_value([0.0, 0.0, 15.0])
        by_period.chirality_combo.setCurrentText("Counterclockwise")
        period_result = by_period.process_structure(structure)[0]

        by_gradient = SpinSpiralCard()
        by_gradient.parameter_mode_combo.setCurrentText("Angle gradient (deg/A)")
        by_gradient.angle_gradient_frame.set_input_value([90.0, 90.0, 1.0])
        by_gradient.phase_frame.set_input_value([0.0, 0.0, 15.0])
        by_gradient.chirality_combo.setCurrentText("Counterclockwise")
        gradient_result = by_gradient.process_structure(structure)[0]

        self.assertTrue(
            np.allclose(
                np.array(period_result.get_initial_magnetic_moments(), dtype=float),
                np.array(gradient_result.get_initial_magnetic_moments(), dtype=float),
                atol=1e-6,
            )
        )

        data = by_gradient.to_dict()
        restored = SpinSpiralCard()
        restored.from_dict(data)
        self.assertEqual(restored.parameter_mode_combo.currentText(), "Angle gradient (deg/A)")
        self.assertEqual(restored.angle_gradient_frame.get_input_value(), [90.0, 90.0, 1.0])

    def test_folded_helix_card_layer_pattern(self):
        structure = self._spin_bilayer_chain()

        card = FoldedHelixCard()
        card.source_combo.setCurrentText("Map/default magnitude")
        card.map_edit.setText("Fe:2.0")
        card.layer_axis_frame.set_input_value([0.0, 0.0, 1.0])
        card.plane_normal_frame.set_input_value([0.0, 0.0, 1.0])
        card.layer_tol_frame.set_input_value([0.05])
        card.half_period_mode_combo.setCurrentText("Manual")
        card.half_period_frame.set_input_value([2, 2, 1])
        card.angle_step_frame.set_input_value([30.0, 30.0, 15.0])
        card.phase_frame.set_input_value([0.0, 0.0, 15.0])
        card.sequence_combo.setCurrentText("Clockwise then counterclockwise")
        card.max_output_frame.set_input_value([4])

        result = card.process_structure(structure)[0]
        moments = np.array(result.get_initial_magnetic_moments(), dtype=float)
        self.assertEqual(moments.shape, (8, 3))
        self.assertTrue(np.allclose(np.linalg.norm(moments, axis=1), 2.0, atol=1e-6))
        self.assertTrue(np.allclose(moments[:, 2], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(moments[0], moments[1], atol=1e-6))
        self.assertTrue(np.allclose(moments[2], moments[3], atol=1e-6))
        self.assertTrue(np.allclose(moments[4], moments[5], atol=1e-6))
        self.assertTrue(np.allclose(moments[6], moments[7], atol=1e-6))

        e1, e2, _ = orthonormal_frame(np.array([0.0, 0.0, 1.0], dtype=float))
        coeff_1 = moments @ e1 / 2.0
        coeff_2 = moments @ e2 / 2.0
        phase_deg = np.rad2deg(np.arctan2(coeff_2, coeff_1))
        expected = np.array([0.0, 0.0, -30.0, -30.0, -60.0, -60.0, -30.0, -30.0], dtype=float)
        wrapped = ((phase_deg - expected + 180.0) % 360.0) - 180.0
        self.assertTrue(np.allclose(wrapped, 0.0, atol=1e-6))
        self.assertIn("FoldedHelix(", str(result.info.get("Config_type", "")))

    def test_folded_helix_card_auto_half_period_from_layer_count(self):
        structure = self._spin_bilayer_chain()
        structure.set_initial_magnetic_moments([2.0] * len(structure))

        card = FoldedHelixCard()
        card.layer_axis_frame.set_input_value([0.0, 0.0, 1.0])
        card.plane_normal_frame.set_input_value([0.0, 0.0, 1.0])
        card.layer_tol_frame.set_input_value([0.05])
        card.half_period_mode_combo.setCurrentText("Auto from layer count")
        card.angle_step_frame.set_input_value([30.0, 30.0, 15.0])
        card.phase_frame.set_input_value([0.0, 0.0, 15.0])
        card.sequence_combo.setCurrentText("Clockwise then counterclockwise")

        result = card.process_structure(structure)[0]
        moments = np.array(result.get_initial_magnetic_moments(), dtype=float)
        e1, e2, _ = orthonormal_frame(np.array([0.0, 0.0, 1.0], dtype=float))
        coeff_1 = moments @ e1 / 2.0
        coeff_2 = moments @ e2 / 2.0
        phase_deg = np.rad2deg(np.arctan2(coeff_2, coeff_1))
        expected = np.array([0.0, 0.0, -30.0, -30.0, -30.0, -30.0, 0.0, 0.0], dtype=float)
        wrapped = ((phase_deg - expected + 180.0) % 360.0) - 180.0
        self.assertTrue(np.allclose(wrapped, 0.0, atol=1e-6))
        self.assertTrue(np.allclose(moments[0], moments[-1], atol=1e-6))
        self.assertFalse(card.half_period_frame.isEnabled())

    def test_folded_helix_card_both_sequences_and_roundtrip(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([1.0, 2.0, 3.0, 4.0])

        card = FoldedHelixCard()
        card.layer_axis_frame.set_input_value([0.0, 0.0, 1.0])
        card.plane_normal_frame.set_input_value([0.0, 0.0, 1.0])
        card.half_period_mode_combo.setCurrentText("Manual")
        card.half_period_frame.set_input_value([2, 2, 1])
        card.angle_step_frame.set_input_value([45.0, 45.0, 15.0])
        card.phase_frame.set_input_value([15.0, 15.0, 15.0])
        card.sequence_combo.setCurrentText("Both")
        card.apply_edit.setText("Fe")
        card.max_output_frame.set_input_value([10])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        tags = [str(atoms.info.get("Config_type", "")) for atoms in results]
        self.assertTrue(any("seq=cw-ccw" in tag for tag in tags))
        self.assertTrue(any("seq=ccw-cw" in tag for tag in tags))
        for atoms in results:
            moments = np.array(atoms.get_initial_magnetic_moments(), dtype=float)
            self.assertEqual(moments.shape, (4, 3))
            self.assertTrue(np.allclose(np.linalg.norm(moments, axis=1), [1.0, 2.0, 3.0, 4.0], atol=1e-6))

        data = card.to_dict()
        restored = FoldedHelixCard()
        restored.from_dict(data)
        self.assertEqual(restored.half_period_mode_combo.currentText(), "Manual")
        self.assertEqual(restored.sequence_combo.currentText(), "Both")
        self.assertEqual(restored.half_period_frame.get_input_value(), [2, 2, 1])
        self.assertEqual(restored.angle_step_frame.get_input_value(), [45.0, 45.0, 15.0])
        self.assertEqual(restored.phase_frame.get_input_value(), [15.0, 15.0, 15.0])
        self.assertEqual(restored.apply_edit.text(), "Fe")
        self.assertFalse(restored.map_label.isVisible())
