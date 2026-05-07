from .magnetism_test_base import *


class TestMagnetismTiltDisorderCards(MagnetismCardTest):
    def test_small_angle_spin_tilt_card_reference_and_explicit_index(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        card = SmallAngleSpinTiltCard()
        card.target_mode_combo.setCurrentText("Explicit indices (1-based)")
        card.target_indices_edit.setText("2")
        card.angle_edit.setText("1,5")
        card.include_reference_checkbox.setChecked(True)

        results = card.process_structure(structure)
        self.assertEqual(len(results), 3)

        reference = np.array(results[0].get_initial_magnetic_moments(), dtype=float)
        tilted_1 = np.array(results[1].get_initial_magnetic_moments(), dtype=float)
        tilted_5 = np.array(results[2].get_initial_magnetic_moments(), dtype=float)

        self.assertTrue(np.allclose(reference, np.tile([0.0, 0.0, 2.0], (4, 1)), atol=1e-6))
        self.assertTrue(np.allclose(tilted_1[[0, 2, 3]], reference[[0, 2, 3]], atol=1e-6))
        self.assertTrue(np.allclose(tilted_5[[0, 2, 3]], reference[[0, 2, 3]], atol=1e-6))
        self.assertAlmostEqual(np.linalg.norm(tilted_1[1]), 2.0, places=6)
        self.assertAlmostEqual(np.linalg.norm(tilted_5[1]), 2.0, places=6)
        self.assertAlmostEqual(tilted_1[1, 0], 2.0 * np.sin(np.deg2rad(1.0)), places=6)
        self.assertAlmostEqual(tilted_1[1, 2], 2.0 * np.cos(np.deg2rad(1.0)), places=6)
        self.assertAlmostEqual(tilted_5[1, 0], 2.0 * np.sin(np.deg2rad(5.0)), places=6)
        self.assertAlmostEqual(tilted_5[1, 2], 2.0 * np.cos(np.deg2rad(5.0)), places=6)
        self.assertIn("SpinTiltRef", str(results[0].info.get("Config_type", "")))
        self.assertIn("SpinTilt(i=2,a=5", str(results[2].info.get("Config_type", "")))
        self.assertFalse(card.map_label.isVisible())
        self.assertFalse(card.default_label.isVisible())

    def test_small_angle_spin_tilt_global_tilt(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        results = SmallAngleSpinTiltOperation().run_structure(
            structure,
            SmallAngleSpinTiltParams(
                canting_mode="Global tilt",
                angle_list="5",
                include_reference=False,
            ),
        )
        self.assertEqual(len(results), 1)
        moments = np.array(results[0].get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(np.allclose(moments[:, 0], 2.0 * np.sin(np.deg2rad(5.0)), atol=1e-6))
        self.assertTrue(np.allclose(moments[:, 2], 2.0 * np.cos(np.deg2rad(5.0)), atol=1e-6))
        self.assertIn("SpinTiltG(a=5,sg=pos)", str(results[0].info.get("Config_type", "")))

    def test_small_angle_spin_tilt_card_map_source_roundtrip_and_limit(self):
        structure = self._spin_chain()

        card = SmallAngleSpinTiltCard()
        card.source_combo.setCurrentText("Map/default magnitude")
        card.map_edit.setText("Fe:2.5")
        card.target_mode_combo.setCurrentText("All eligible atoms")
        card.angle_edit.setText("1,2")
        card.sign_combo.setCurrentText("Both (+/- pair)")
        card.include_reference_checkbox.setChecked(False)
        card.max_output_frame.set_input_value([3])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 3)
        for atoms in results:
            moments = np.array(atoms.get_initial_magnetic_moments(), dtype=float)
            self.assertEqual(moments.shape, (4, 3))
            self.assertTrue(np.allclose(np.linalg.norm(moments, axis=1), 2.5, atol=1e-6))
            self.assertIn("SpinTilt(", str(atoms.info.get("Config_type", "")))
            self.assertRegex(str(atoms.info.get("Config_type", "")), r"sg=(pos|neg)")

        data = card.to_dict()
        restored = SmallAngleSpinTiltCard()
        restored.from_dict(data)
        self.assertEqual(restored.source_combo.currentText(), "Map/default magnitude")
        self.assertEqual(restored.target_mode_combo.currentText(), "All eligible atoms")
        self.assertEqual(restored.angle_edit.text(), "1,2")
        self.assertEqual(restored.sign_combo.currentText(), "Both (+/- pair)")
        self.assertFalse(restored.include_reference_checkbox.isChecked())
        self.assertEqual(restored.max_output_frame.get_input_value(), [3])

    def test_small_angle_spin_tilt_card_atom_pair_canting(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        card = SmallAngleSpinTiltCard()
        card.canting_mode_combo.setCurrentText("Atom pair canting")
        card.pair_left_edit.setText("1")
        card.pair_right_edit.setText("2")
        card.angle_edit.setText("10")
        card.sign_combo.setCurrentText("Both (+/- pair)")
        card.include_reference_checkbox.setChecked(False)

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)

        pos = next(a for a in results if "sg=pos" in str(a.info.get("Config_type", "")))
        neg = next(a for a in results if "sg=neg" in str(a.info.get("Config_type", "")))
        pos_m = np.array(pos.get_initial_magnetic_moments(), dtype=float)
        neg_m = np.array(neg.get_initial_magnetic_moments(), dtype=float)

        self.assertAlmostEqual(pos_m[0, 0], 2.0 * np.sin(np.deg2rad(5.0)), places=6)
        self.assertAlmostEqual(pos_m[1, 0], -2.0 * np.sin(np.deg2rad(5.0)), places=6)
        self.assertAlmostEqual(neg_m[0, 0], -2.0 * np.sin(np.deg2rad(5.0)), places=6)
        self.assertAlmostEqual(neg_m[1, 0], 2.0 * np.sin(np.deg2rad(5.0)), places=6)
        self.assertIn("SpinPair(i=1,j=2,a=10,sg=pos)", str(pos.info.get("Config_type", "")))

    def test_small_angle_spin_tilt_card_group_pair_canting(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])
        structure.new_array("group", np.array(["A", "A", "B", "B"], dtype="<U1"))

        card = SmallAngleSpinTiltCard()
        card.canting_mode_combo.setCurrentText("Group pair canting")
        card.group_a_edit.setText("A")
        card.group_b_edit.setText("B")
        card.angle_edit.setText("6")
        card.sign_combo.setCurrentText("Positive only")
        card.include_reference_checkbox.setChecked(False)

        result = card.process_structure(structure)[0]
        moments = np.array(result.get_initial_magnetic_moments(), dtype=float)
        expected = 2.0 * np.sin(np.deg2rad(3.0))
        self.assertTrue(np.allclose(moments[[0, 1], 0], expected, atol=1e-6))
        self.assertTrue(np.allclose(moments[[2, 3], 0], -expected, atol=1e-6))
        self.assertIn("SpinPairG(A=A,B=B,a=6,sg=pos)", str(result.info.get("Config_type", "")))

    def test_small_angle_spin_tilt_card_auto_neighbor_shell_pair(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        card = SmallAngleSpinTiltCard()
        card.canting_mode_combo.setCurrentText("Atom pair canting")
        card.pair_source_combo.setCurrentText("Auto by neighbor shell")
        card.pair_shell_frame.set_input_value([1])
        card.pair_tol_frame.set_input_value([0.02])
        card.angle_edit.setText("4")
        card.sign_combo.setCurrentText("Positive only")
        card.include_reference_checkbox.setChecked(False)

        results = card.process_structure(structure)
        self.assertEqual(len(results), 3)
        tags = [str(a.info.get("Config_type", "")) for a in results]
        self.assertTrue(any("SpinPair(i=1,j=2,a=4,sg=pos)" in tag for tag in tags))
        self.assertTrue(any("SpinPair(i=2,j=3,a=4,sg=pos)" in tag for tag in tags))
        self.assertTrue(any("SpinPair(i=3,j=4,a=4,sg=pos)" in tag for tag in tags))

        first = next(a for a in results if "SpinPair(i=1,j=2,a=4,sg=pos)" in str(a.info.get("Config_type", "")))
        moments = np.array(first.get_initial_magnetic_moments(), dtype=float)
        expected = 2.0 * np.sin(np.deg2rad(2.0))
        self.assertAlmostEqual(moments[0, 0], expected, places=6)
        self.assertAlmostEqual(moments[1, 0], -expected, places=6)

    def test_small_angle_spin_tilt_card_auto_neighbor_shell_filters_and_roundtrip(self):
        structure = Atoms(
            symbols=["Fe", "Co", "Fe", "Co"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ],
            cell=np.diag([6.0, 6.0, 8.0]),
            pbc=[False, False, True],
        )
        structure.info["Config_type"] = "FeCo_chain"
        structure.new_array("group", np.array(["A", "A", "B", "B"], dtype="<U1"))
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        card = SmallAngleSpinTiltCard()
        card.canting_mode_combo.setCurrentText("Atom pair canting")
        card.pair_source_combo.setCurrentText("Auto by neighbor shell")
        card.pair_shell_frame.set_input_value([1])
        card.pair_tol_frame.set_input_value([0.02])
        card.pair_element_edit.setText("Fe-Co")
        card.pair_group_edit.setText("A-B")
        card.bond_mode_combo.setCurrentText("Near axis")
        card.bond_axis_frame.set_input_value([0.0, 0.0, 1.0])
        card.bond_tol_frame.set_input_value([5.0])
        card.angle_edit.setText("6")
        card.sign_combo.setCurrentText("Both (+/- pair)")
        card.include_reference_checkbox.setChecked(False)

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        tags = [str(a.info.get("Config_type", "")) for a in results]
        self.assertTrue(all("SpinPair(i=2,j=3,a=6" in tag for tag in tags))

        data = card.to_dict()
        restored = SmallAngleSpinTiltCard()
        restored.from_dict(data)
        self.assertEqual(restored.pair_source_combo.currentText(), "Auto by neighbor shell")
        self.assertEqual(restored.pair_element_edit.text(), "Fe-Co")
        self.assertEqual(restored.pair_group_edit.text(), "A-B")
        self.assertEqual(restored.bond_mode_combo.currentText(), "Near axis")
        self.assertEqual(restored.bond_axis_frame.get_input_value(), [0.0, 0.0, 1.0])
        self.assertEqual(restored.bond_tol_frame.get_input_value(), [5.0])
        self.assertFalse(restored.bond_axis_label.isHidden())
        self.assertFalse(restored.bond_tol_label.isHidden())

    def test_spin_disorder_operation_and_card_roundtrip(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        results = SpinDisorderOperation().run_structure(
            structure,
            SpinDisorderParams(
                mode="Flip fraction",
                fractions="0.5",
                samples_per_fraction=1,
                use_seed=True,
                seed=11,
            ),
        )
        self.assertEqual(len(results), 1)
        moments = np.array(results[0].get_initial_magnetic_moments(), dtype=float)
        self.assertEqual(int(np.sum(moments[:, 2] < 0.0)), 2)
        self.assertIn("SpinDis(f=0.5,n=2,mode=flip", str(results[0].info.get("Config_type", "")))

        cone = SpinDisorderOperation().run_structure(
            structure,
            SpinDisorderParams(
                mode="Cone disorder",
                fractions="0.5",
                cone_angle=10.0,
                use_seed=True,
                seed=11,
            ),
        )[0]
        cone_moments = np.array(cone.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(np.allclose(np.linalg.norm(cone_moments, axis=1), 2.0, atol=1e-6))

        card = SpinDisorderCard()
        card.mode_combo.setCurrentText("Cone disorder")
        card.fractions_edit.setText("0.25,0.5")
        card.samples_frame.set_input_value([2])
        card.cone_frame.set_input_value([12.0])
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([5])
        restored = SpinDisorderCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())
