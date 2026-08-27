from unittest.mock import patch

from ase.geometry import get_distances

from .magnetism_test_base import *


class TestMagnetismTiltDisorderCards(MagnetismCardTest):
    def test_retired_canting_card_remains_loadable_with_migration_guidance(self):
        self.assertFalse(SmallAngleSpinTiltCard.discoverable)

        card = SmallAngleSpinTiltCard()
        card.from_dict(
            {
                "check_state": True,
                "canting_mode": "Atom pair canting",
                "pair_left_indices": "1",
                "pair_right_indices": "2",
                "angle_list": "5",
            }
        )

        self.assertEqual(card.get_params().canting_mode, "Atom pair canting")
        self.assertEqual(card.get_params().pair_left_indices, "1")
        self.assertIn("Legacy", card.getTitle())
        self.assertIn("Local Magnetic Response", card.get_guidance_text())

    def test_small_angle_card_fails_closed_for_invalid_moment_map(self):
        card = SmallAngleSpinTiltCard()
        card.source_combo.setCurrentText("Map/default magnitude")
        card.map_edit.setText("not-an-element:2")
        with self.assertRaises(ValueError):
            card.process_structure(self._spin_chain())

    def test_small_angle_spin_tilt_fast_pair_distance_matches_ase(self):
        structure = self._spin_chain()
        positions = np.asarray(structure.get_positions(), dtype=float)

        vec_fast, dist_fast = SmallAngleSpinTiltOperation.pair_distance_matrix(
            positions,
            cell=np.asarray(structure.cell.array, dtype=float),
            pbc=np.asarray(structure.pbc, dtype=bool),
        )
        vec_ase, dist_ase = get_distances(
            positions,
            positions,
            cell=np.asarray(structure.cell.array, dtype=float),
            pbc=np.asarray(structure.pbc, dtype=bool),
        )

        self.assertTrue(np.allclose(dist_fast, dist_ase, atol=1e-12))
        self.assertTrue(np.allclose(np.linalg.norm(vec_fast, axis=2), np.linalg.norm(vec_ase, axis=2), atol=1e-12))

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

    def test_small_angle_spin_tilt_missing_prerequisites_fail_instead_of_returning_input(self):
        operation = SmallAngleSpinTiltOperation()
        no_moments = self._spin_chain()
        with self.assertRaisesRegex(ValueError, "requires usable initial magnetic moments"):
            operation.run_structure(
                no_moments,
                SmallAngleSpinTiltParams(include_reference=False),
            )

        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])
        with self.assertRaisesRegex(ValueError, "group-pair mode requires atoms.arrays"):
            operation.run_structure(
                structure,
                SmallAngleSpinTiltParams(
                    canting_mode="Group pair canting",
                    include_reference=False,
                ),
            )
        with self.assertRaisesRegex(ValueError, "atom-pair mode matched no valid pairs"):
            operation.run_structure(
                structure,
                SmallAngleSpinTiltParams(
                    canting_mode="Atom pair canting",
                    pair_source="Manual indices",
                    pair_left_indices="",
                    pair_right_indices="",
                    include_reference=True,
                ),
            )

    def test_small_angle_spin_tilt_reference_must_leave_output_budget_for_canting(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        with self.assertRaisesRegex(ValueError, "max_outputs must be >= 2"):
            SmallAngleSpinTiltOperation().run_structure(
                structure,
                SmallAngleSpinTiltParams(
                    include_reference=True,
                    max_outputs=1,
                ),
            )

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

    def test_spin_disorder_dynamic_controls_and_preview_are_reachable(self):
        card = SpinDisorderCard()

        self.assertEqual(card.getTitle(), "Moment Disorder")
        self.assertEqual(card.mode_field.caption.text(), "How selected moments change")
        self.assertEqual(card.fractions_field.caption.text(), "Fraction of moments changed")
        self.assertEqual(card.fractions_edit.custom_checkbox.text(), "Specify fractions to generate")
        self.assertTrue(card.cone_field.isHidden())
        cone_index = card.mode_combo.findData("Cone disorder")
        card.mode_combo.setCurrentIndex(cone_index)
        self.assertFalse(card.cone_field.isHidden())

        card.advanced_checkbox.setChecked(True)
        self.assertFalse(card.source_section.isHidden())
        map_index = card.source_combo.findData("Map/default magnitude")
        card.source_combo.setCurrentIndex(map_index)
        self.assertFalse(card.map_field.isHidden())
        self.assertFalse(card.default_field.isHidden())
        self.assertTrue(card.lift_scalar_checkbox.isHidden())

        card.samples_frame.set_input_value([2])
        self.assertIn("10% × 2 · 30% × 2 · 50% × 2 · 70% × 2", card.output_preview.text())
        self.assertIn("= 8 structures", card.output_preview.text())
        self.assertLessEqual(card.samples_frame.width(), 132)
        self.assertLessEqual(card.max_output_frame.width(), 132)

        card.max_output_frame.set_input_value([5])
        self.assertIn("10% × 2 · 30% × 2 · 50% × 1 · 70% × 0", card.output_preview.text())
        self.assertIn("= 5 structures (8 requested; output limit reached)", card.output_preview.text())

    def test_spin_disorder_fraction_editing_is_fail_soft(self):
        card = SpinDisorderCard()
        card.fractions_edit.custom_checkbox.setChecked(True)

        for text in ("", "abc", "0"):
            with self.subTest(text=text):
                card.fractions_edit.custom_edit.setText(text)
                self.assertTrue(card.get_summary_text())
                self.assertTrue(card.get_guidance_text())
                self.assertTrue(card.output_preview.text())

    def test_spin_disorder_randomize_ui_value_runs_core_operation(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])
        card = SpinDisorderCard()
        mode_index = card.mode_combo.findData("Randomize fraction")
        self.assertGreaterEqual(mode_index, 0)
        card.mode_combo.setCurrentIndex(mode_index)
        card.fractions_edit.setText("0.5")
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([13])

        params = card.get_params()
        self.assertEqual(params.mode, "Randomize fraction")
        result = card.create_operation().run_structure(structure, params)[0]
        moments = np.asarray(result.get_initial_magnetic_moments(), dtype=float)

        self.assertEqual(int(np.count_nonzero(np.linalg.norm(moments, axis=1))), 4)
        self.assertEqual(
            int(np.count_nonzero(~np.isclose(moments[:, 0], 0.0))),
            2,
        )
        self.assertIn("mode=rand", result.info.get("Config_type", ""))

    def test_spin_disorder_rejects_invalid_fraction_tokens_and_ranges(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])
        operation = SpinDisorderOperation()

        for fractions in ("abc", "0", "-0.1", "1.1", "nan", "inf"):
            with self.subTest(fractions=fractions):
                with self.assertRaisesRegex(ValueError, "Spin Disorder fraction"):
                    operation.run_structure(
                        structure,
                        SpinDisorderParams(fractions=fractions),
                    )

    def test_disorder_cards_reject_invalid_disabled_or_serialized_values(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        for params, message in (
            (
                SpinDisorderParams(
                    magnitude_source="typo",
                ),
                "magnitude_source",
            ),
            (
                SpinDisorderParams(
                    use_seed=True,
                    seed=-1,
                ),
                "seed must be >= 0",
            ),
            (
                SpinDisorderParams(
                    mode="Cone disorder",
                    cone_angle=181.0,
                ),
                "cone_angle",
            ),
        ):
            with self.subTest(card="spin", params=params):
                with self.assertRaisesRegex(ValueError, message):
                    SpinDisorderOperation().run_structure(structure, params)

        for params, message in (
            (
                CorrelatedRandomSpinParams(
                    magnitude_source="typo",
                ),
                "magnitude_source",
            ),
            (
                CorrelatedRandomSpinParams(
                    use_seed=True,
                    seed=-1,
                ),
                "seed must be >= 0",
            ),
            (
                CorrelatedRandomSpinParams(
                    mode="Cone around reference",
                    cone_angle=181.0,
                ),
                "cone_angle",
            ),
        ):
            with self.subTest(card="correlated", params=params):
                with self.assertRaisesRegex(ValueError, message):
                    CorrelatedRandomSpinOperation().run_structure(structure, params)

    def test_correlated_random_spin_rejects_invalid_mode_before_rng(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        with patch(
            "NepTrainKit.core.cards.magnetism.np.random.default_rng"
        ) as rng_factory:
            with self.assertRaisesRegex(ValueError, "unsupported mode"):
                CorrelatedRandomSpinOperation().run_structure(
                    structure,
                    CorrelatedRandomSpinParams(mode="typo"),
                )
        rng_factory.assert_not_called()

    def test_disorder_cards_map_source_and_apply_elements_limit_changed_spins(self):
        structure = Atoms(
            ["Fe", "Ni", "Fe", "Ni"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
            ],
            cell=[6.0, 6.0, 8.0],
            pbc=[False, False, True],
        )
        disordered = SpinDisorderOperation().run_structure(
            structure,
            SpinDisorderParams(
                mode="Randomize fraction",
                fractions="1.0",
                samples_per_fraction=1,
                magnitude_source="Map/default magnitude",
                magmom_map="Fe:2.0",
                default_moment=9.0,
                apply_elements="Fe",
                use_seed=True,
                seed=31,
            ),
        )[0]
        np.testing.assert_allclose(
            np.linalg.norm(disordered.arrays["spin"], axis=1),
            [2.0, 0.0, 2.0, 0.0],
            atol=1e-12,
        )

        correlated = CorrelatedRandomSpinOperation().run_structure(
            structure,
            CorrelatedRandomSpinParams(
                mode="Cone around reference",
                correlation_kernel="exponential",
                correlation_length=2.0,
                samples=1,
                cone_angle=10.0,
                magnitude_source="Map/default magnitude",
                magmom_map="Fe:2.0",
                default_moment=9.0,
                apply_elements="Fe",
                max_atoms_for_full=2,
                use_seed=True,
                seed=31,
            ),
        )[0]
        np.testing.assert_allclose(
            np.linalg.norm(correlated.arrays["spin"], axis=1),
            [2.0, 0.0, 2.0, 0.0],
            atol=1e-12,
        )

    def test_correlated_random_spin_cone_preserves_magnitudes_and_seed(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])
        params = CorrelatedRandomSpinParams(
            mode="Cone around reference",
            correlation_kernel="exponential",
            correlation_length=2.0,
            samples=2,
            cone_angle=10.0,
            use_seed=True,
            seed=17,
            max_atoms_for_full=10,
        )

        op = CorrelatedRandomSpinOperation()
        results = op.run_structure(structure, params)
        repeated = op.run_structure(structure, params)

        self.assertEqual(len(results), 2)
        for left, right in zip(results, repeated):
            moments = np.asarray(left.get_initial_magnetic_moments(), dtype=float)
            repeated_moments = np.asarray(right.get_initial_magnetic_moments(), dtype=float)
            self.assertEqual(moments.shape, (4, 3))
            self.assertTrue(np.allclose(moments, repeated_moments, atol=1e-12))
            self.assertTrue(np.allclose(np.linalg.norm(moments, axis=1), 2.0, atol=1e-8))
            cos_angles = np.clip(moments[:, 2] / 2.0, -1.0, 1.0)
            angles = np.degrees(np.arccos(cos_angles))
            self.assertTrue(np.all(angles <= 10.0 + 1e-8))
            self.assertIn("CorrSpin(xi=2,ker=exponential,mode=cone", str(left.info.get("Config_type", "")))

    def test_correlated_random_spin_full_mode_kernel_and_limit(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])

        exp_result = CorrelatedRandomSpinOperation().run_structure(
            structure,
            CorrelatedRandomSpinParams(
                mode="Full random directions",
                correlation_kernel="exponential",
                correlation_length=2.0,
                samples=1,
                use_seed=True,
                seed=3,
                max_atoms_for_full=10,
            ),
        )[0]
        sq_result = CorrelatedRandomSpinOperation().run_structure(
            structure,
            CorrelatedRandomSpinParams(
                mode="Full random directions",
                correlation_kernel="squared_exponential",
                correlation_length=2.0,
                samples=1,
                use_seed=True,
                seed=3,
                max_atoms_for_full=10,
            ),
        )[0]

        exp_moments = np.asarray(exp_result.get_initial_magnetic_moments(), dtype=float)
        sq_moments = np.asarray(sq_result.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(np.allclose(np.linalg.norm(exp_moments, axis=1), 2.0, atol=1e-8))
        self.assertFalse(np.allclose(exp_moments, sq_moments, atol=1e-8))

        with self.assertRaisesRegex(ValueError, "exact full covariance"):
            CorrelatedRandomSpinOperation().run_structure(
                structure,
                CorrelatedRandomSpinParams(max_atoms_for_full=2),
            )

    def test_correlated_random_spin_card_roundtrip(self):
        card = CorrelatedRandomSpinCard()
        card.mode_combo.setCurrentText("Full random directions")
        card.kernel_combo.setCurrentText("squared_exponential")
        card.xi_frame.set_input_value([4.5])
        card.samples_frame.set_input_value([3])
        card.cone_frame.set_input_value([15.0])
        card.source_combo.setCurrentText("Map/default magnitude")
        card.map_edit.setText("Fe:2.2")
        card.default_frame.set_input_value([0.1])
        card.apply_edit.setText("Fe")
        card.max_atoms_frame.set_input_value([50])
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([13])

        restored = CorrelatedRandomSpinCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())
