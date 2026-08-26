import tempfile

from NepTrainKit.core.io.importers import import_structures
from NepTrainKit.core.magnetism import parse_magmom_map_any
from NepTrainKit.ui.views._card.i18n_utils import set_combo_value

from .magnetism_test_base import *


class TestMagnetismOrderCards(MagnetismCardTest):
    def test_set_magnetic_moments_card_fails_closed_for_invalid_map(self):
        card = SetMagneticMomentsCard()
        card.map_edit.setText("Fe:not-a-number")
        with self.assertRaises(ValueError):
            card.process_structure(self._spin_chain())

    def test_magnetic_order_card_fm_afm(self):
        proto = CrystalPrototypeBuilderCard()
        set_combo_value(proto.structure_combo, "bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.max_output_frame.set_input_value([1])
        base = proto.create_operation().generate(proto.get_params())[0].repeat((2, 2, 2))

        card = MagneticOrderCard()
        card.map_edit.setText("Fe:2.2")
        card.fm_checkbox.setChecked(True)
        card.afm_checkbox.setChecked(True)
        card.kvec_combo.setCurrentIndex(0)
        card.pm_checkbox.setChecked(False)

        results = card.process_structure(base)
        self.assertEqual(len(results), 2)
        fm = [a for a in results if "MagFM" in str(a.info.get("Config_type", ""))][0]
        afm = [a for a in results if "MagAFM100" in str(a.info.get("Config_type", ""))][0]

        fm_m = np.array(fm.get_initial_magnetic_moments(), dtype=float)
        afm_m = np.array(afm.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(np.all(fm_m >= 0))
        self.assertTrue(np.any(afm_m > 0) and np.any(afm_m < 0))
        self.assertTrue(
            np.allclose(fm.arrays["spin"][:, 2], fm_m, atol=1e-12)
        )
        self.assertTrue(
            np.allclose(afm.arrays["spin"][:, 2], afm_m, atol=1e-12)
        )

    def test_magnetic_order_rejects_false_success_and_invalid_modes(self):
        structure = self._spin_chain()
        operation = MagneticOrderOperation()

        defaults = MagneticOrderParams()
        self.assertTrue(defaults.gen_fm)
        self.assertFalse(defaults.gen_afm)
        with self.assertRaisesRegex(ValueError, "no nonzero magnetic moments"):
            operation.run_structure(structure, defaults)
        with self.assertRaisesRegex(ValueError, "select at least one"):
            operation.run_structure(
                structure,
                MagneticOrderParams(
                    magmom_map="Fe:2.0",
                    gen_fm=False,
                    gen_afm=False,
                    gen_pm=False,
                ),
            )
        with self.assertRaisesRegex(ValueError, "unsupported spin model"):
            operation.run_structure(
                structure,
                MagneticOrderParams(format="typo", magmom_map="Fe:2.0"),
            )
        with self.assertRaisesRegex(ValueError, "finite nonzero"):
            operation.run_structure(
                structure,
                MagneticOrderParams(
                    axis=(0.0, 0.0, 0.0),
                    magmom_map="Fe:2.0",
                ),
            )
        with self.assertRaisesRegex(ValueError, "above the limit"):
            operation.run_structure(
                structure,
                MagneticOrderParams(
                    magmom_map="Fe:2.0",
                    gen_pm=True,
                    pm_count=10,
                    max_outputs=5,
                ),
            )

    def test_magnetic_order_group_mode_fails_closed(self):
        structure = self._spin_chain()
        operation = MagneticOrderOperation()
        params = MagneticOrderParams(
            magmom_map="Fe:2.0",
            gen_fm=False,
            gen_afm=True,
            afm_mode="group_ab",
        )
        with self.assertRaisesRegex(ValueError, "requires atoms.arrays"):
            operation.run_structure(structure, params)

        structure.new_array(
            "group",
            np.asarray(["A", "A", "other", "other"], dtype=object),
        )
        with self.assertRaisesRegex(ValueError, "both the positive and negative"):
            operation.run_structure(structure, params)
        with self.assertRaisesRegex(ValueError, "must differ"):
            operation.run_structure(
                structure,
                MagneticOrderParams(
                    magmom_map="Fe:2.0",
                    gen_fm=False,
                    gen_afm=True,
                    afm_mode="group_ab",
                    afm_group_a="A",
                    afm_group_b="A",
                ),
            )

    def test_magnetic_order_group_unknown_policy_is_explicit(self):
        structure = self._spin_chain()
        structure.arrays["group"] = np.asarray(
            ["A", "B", "other", "other"],
            dtype=object,
        )
        base = dict(
            format="collinear",
            magmom_map="Fe:2.0",
            gen_fm=False,
            gen_afm=True,
            afm_mode="group_ab",
            afm_group_a="A",
            afm_group_b="B",
        )

        zero_unknown = MagneticOrderOperation().run_structure(
            structure,
            MagneticOrderParams(**base, afm_zero_unknown=True),
        )[0]
        positive_unknown = MagneticOrderOperation().run_structure(
            structure,
            MagneticOrderParams(**base, afm_zero_unknown=False),
        )[0]

        np.testing.assert_allclose(
            zero_unknown.get_initial_magnetic_moments(),
            [2.0, -2.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            positive_unknown.get_initial_magnetic_moments(),
            [2.0, -2.0, 2.0, 2.0],
        )

    def test_magnetic_order_k_vector_preview_is_stable_on_repeated_cell(self):
        structure = CrystalPrototypeBuilderOperation().generate(
            CrystalPrototypeBuilderParams(
                lattice="bcc",
                element="Fe",
                a_range=(2.9, 2.9, 0.1),
                max_outputs=1,
            )
        )[0].repeat((4, 4, 4))
        params = MagneticOrderParams(
            magmom_map="Fe:2.2",
            gen_fm=False,
            gen_afm=True,
            afm_mode="k_vector",
            afm_kvec="111",
        )
        operation = MagneticOrderOperation()
        preview = operation.preview(structure, params)
        self.assertEqual(preview.magnetic_atoms, 128)
        self.assertEqual(preview.output_count, 1)
        self.assertEqual(
            (preview.afm_positive, preview.afm_negative, preview.afm_zero),
            (64, 64, 0),
        )
        result = operation.run_structure(structure, params)[0]
        self.assertIn("MagAFM111", result.info.get("Config_type", ""))

        with self.assertRaisesRegex(ValueError, "Unsupported k-vector"):
            operation.run_structure(
                structure,
                MagneticOrderParams(
                    magmom_map="Fe:2.2",
                    gen_fm=False,
                    gen_afm=True,
                    afm_kvec="typo",
                ),
            )

    def test_balanced_noncollinear_pm_preserves_cone_and_pairs(self):
        structure = CrystalPrototypeBuilderOperation().generate(
            CrystalPrototypeBuilderParams(
                lattice="bcc",
                element="Fe",
                a_range=(2.9, 2.9, 0.1),
                max_outputs=1,
            )
        )[0].repeat((2, 2, 2))
        result = MagneticOrderOperation().run_structure(
            structure,
            MagneticOrderParams(
                format="noncollinear",
                magmom_map="Fe:2.2",
                gen_fm=False,
                gen_afm=False,
                gen_pm=True,
                pm_count=1,
                pm_direction="cone",
                pm_cone_angle=30.0,
                pm_balanced=True,
                use_seed=True,
                seed=17,
            ),
        )[0]
        spin = np.asarray(result.arrays["spin"], dtype=float)
        norms = np.linalg.norm(spin, axis=1)
        nearest_axis_angle = np.degrees(
            np.arccos(np.clip(np.abs(spin[:, 2]) / norms, -1.0, 1.0))
        )
        self.assertTrue(np.allclose(norms, 2.2, atol=1e-12))
        self.assertLessEqual(float(np.max(nearest_axis_angle)), 30.0 + 1e-10)
        self.assertTrue(np.allclose(spin.sum(axis=0), 0.0, atol=1e-12))

    def test_magmom_map_accepts_vector_token_syntax(self):
        parsed = parse_magmom_map_any("Fe:2.2,Cr:[0,0,1.5]")
        self.assertEqual(parsed["Fe"], 2.2)
        np.testing.assert_allclose(parsed["Cr"], [0.0, 0.0, 1.5])

    def test_magnetic_order_card_progressive_ui_preview_and_roundtrip(self):
        structure = self._spin_chain()
        card = MagneticOrderCard()
        self.assertFalse(card.afm_checkbox.isChecked())
        self.assertTrue(card.afm_mode_combo.isHidden())
        self.assertTrue(card.pm_count_frame.isHidden())

        card.map_edit.setText("Fe:2.0")
        card.set_dataset([structure])
        self.assertIn("magnetic atoms 4/4", card.preview_label.text())
        self.assertIn("outputs/input 1", card.preview_label.text())

        card.format_combo.setCurrentIndex(1)
        card.afm_checkbox.setChecked(True)
        card.kvec_combo.setCurrentIndex(2)
        card.pm_checkbox.setChecked(True)
        card.pm_count_frame.set_input_value([3])
        card.pm_direction_combo.setCurrentIndex(2)
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([19])
        self.assertFalse(card.pm_direction_combo.isHidden())
        self.assertTrue(card.pm_cone_frame.isHidden())

        restored = MagneticOrderCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

        legacy = MagneticOrderCard()
        legacy.from_dict(
            {
                "class": "MagneticOrderCard",
                "check_state": True,
                "format": "Non-collinear (vector)",
                "magmom_map": "Fe:2.0",
                "gen_fm": True,
                "gen_afm": True,
                "afm_mode": "group A/B",
            }
        )
        self.assertEqual(legacy.get_params().format, "noncollinear")
        self.assertEqual(legacy.get_params().afm_mode, "group_ab")
        self.assertEqual(legacy.get_params().max_outputs, 100)

    def test_canonical_spin_takes_precedence_over_conflicting_ase_magmoms(self):
        structure = self._spin_chain()
        canonical = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, 2.0, 0.0),
                (0.0, 0.0, -3.0),
                (-4.0, 0.0, 0.0),
            ),
            dtype=float,
        )
        structure.set_initial_magnetic_moments([9.0, 9.0, 9.0, 9.0])
        structure.set_array("spin", canonical)

        result = SetMagneticMomentsOperation().run_structure(
            structure,
            SetMagneticMomentsParams(
                source="Existing initial magmoms",
                format="Non-collinear (vector)",
                axis=(0.0, 0.0, 1.0),
            ),
        )[0]

        self.assertTrue(
            np.allclose(result.get_initial_magnetic_moments(), canonical)
        )
        self.assertTrue(np.allclose(result.arrays["spin"], canonical))

    def test_card_export_uses_show_nep_spin_contract_for_collinear_output(self):
        structure = self._spin_chain()
        card = MagneticOrderCard()
        card.format_combo.setCurrentIndex(0)
        card.axis_frame.set_input_value([1.0, 0.0, 0.0])
        card.map_edit.setText("Fe:2.2")
        card.fm_checkbox.setChecked(True)
        card.afm_checkbox.setChecked(False)
        card.pm_checkbox.setChecked(False)
        result = card.process_structure(structure)[0]

        scalar = np.asarray(result.get_initial_magnetic_moments(), dtype=float)
        expected_spin = np.column_stack(
            (scalar, np.zeros(len(result)), np.zeros(len(result)))
        )
        self.assertTrue(np.allclose(result.arrays["spin"], expected_spin))

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "magnetic output.xyz"
            card.result_dataset = [result]
            card.write_result_dataset(path)
            text = path.read_text(encoding="utf-8")
            loaded = import_structures(path)[0]

        self.assertIn("spin:R:3", text)
        self.assertNotIn("initial_magmoms", text)
        self.assertTrue(
            np.allclose(loaded.atomic_properties["spin"], expected_spin)
        )

    def test_magnetic_operations_are_ui_independent(self):
        structure = self._spin_chain()

        set_result = SetMagneticMomentsOperation().run_structure(
            structure,
            SetMagneticMomentsParams(
                source="Map/default magnitude",
                format="Non-collinear (vector)",
                magmom_map="Fe:2.0",
                axis=[0.0, 0.0, 1.0],
            ),
        )[0]
        self.assertIn("MagSet(map,vec)", str(set_result.info.get("Config_type", "")))

        order_result = MagneticOrderOperation().run_structure(
            structure,
            MagneticOrderParams(magmom_map="Fe:2.0", gen_fm=True, gen_afm=False),
        )[0]
        self.assertIn("MagFM", str(order_result.info.get("Config_type", "")))

        spiral_result = SpinSpiralOperation().run_structure(
            set_result,
            SpinSpiralParams(
                period_range=[4.0, 4.0, 1.0],
                phase_range=[0.0, 0.0, 15.0],
                chirality="Clockwise",
                max_outputs=1,
            ),
        )[0]
        self.assertIn("Helix(", str(spiral_result.info.get("Config_type", "")))

        folded = FoldedHelixOperation().run_structure(
            set_result,
            FoldedHelixParams(
                half_period_mode="Manual",
                half_period_layers=[2, 2, 1],
                angle_step_range=[30.0, 30.0, 15.0],
                phase_range=[0.0, 0.0, 15.0],
                max_outputs=1,
            ),
        )[0]
        self.assertIn("FoldedHelix(", str(folded.info.get("Config_type", "")))

        tilt_result = SmallAngleSpinTiltOperation().run_structure(
            set_result,
            SmallAngleSpinTiltParams(
                target_mode="Explicit indices (1-based)",
                target_indices="2",
                angle_list="5",
                include_reference=False,
            ),
        )[0]
        self.assertIn("SpinTilt(i=2,a=5", str(tilt_result.info.get("Config_type", "")))

        rotated = MagneticMomentRotationOperation().run_structure(
            set_result,
            MagneticMomentRotationParams(
                elements="Fe",
                max_angle=10.0,
                num_structures=1,
                use_seed=True,
                seed=7,
            ),
        )[0]
        moments = np.array(rotated.get_initial_magnetic_moments(), dtype=float)
        self.assertEqual(moments.shape, (4, 3))
        for output in (
            set_result,
            order_result,
            spiral_result,
            folded,
            tilt_result,
            rotated,
        ):
            self.assertIn("spin", output.arrays)
            self.assertEqual(np.asarray(output.arrays["spin"]).shape, (4, 3))
            initial = np.asarray(output.get_initial_magnetic_moments())
            if initial.ndim == 2:
                self.assertTrue(
                    np.allclose(output.arrays["spin"], initial, atol=1e-12)
                )

    def test_rotation_reads_spin_without_legacy_initial_magmoms(self):
        structure = self._spin_chain()
        canonical = np.tile([0.0, 0.0, 2.0], (len(structure), 1))
        structure.set_array("spin", canonical)

        result = MagneticMomentRotationOperation().run_structure(
            structure,
            MagneticMomentRotationParams(
                max_angle=10.0,
                num_structures=1,
                lift_scalar=False,
                use_seed=True,
                seed=71,
            ),
        )[0]

        self.assertEqual(result.get_initial_magnetic_moments().shape, (4, 3))
        self.assertFalse(np.allclose(result.arrays["spin"], canonical))
        self.assertTrue(
            np.allclose(
                result.arrays["spin"],
                result.get_initial_magnetic_moments(),
                atol=1e-12,
            )
        )

    def test_malformed_canonical_spin_fails_closed(self):
        structure = self._spin_chain()
        structure.set_array("spin", np.ones((len(structure), 1)))
        structure.set_initial_magnetic_moments([2.0] * len(structure))

        with self.assertRaisesRegex(
            ValueError,
            "spin must be a finite numeric N x 3 array",
        ):
            SetMagneticMomentsOperation().run_structure(
                structure,
                SetMagneticMomentsParams(
                    source="Existing initial magmoms",
                    format="Non-collinear (vector)",
                ),
            )

    def test_scalar_output_requires_an_explicit_nonzero_spin_axis(self):
        structure = self._spin_chain()

        with self.assertRaisesRegex(
            ValueError,
            "scalar magmoms require a finite nonzero axis",
        ):
            SetMagneticMomentsOperation().run_structure(
                structure,
                SetMagneticMomentsParams(
                    source="Constant magnitude",
                    format="Collinear (scalar)",
                    constant_moment=2.0,
                    axis=(0.0, 0.0, 0.0),
                ),
            )

    def test_magnetic_order_card_noncollinear_pm(self):
        proto = CrystalPrototypeBuilderCard()
        set_combo_value(proto.structure_combo, "fcc")
        proto.element_edit.setText("Ni")
        proto.a_frame.set_input_value([3.5, 3.5, 0.1])
        proto.max_output_frame.set_input_value([1])
        base = proto.create_operation().generate(proto.get_params())[0].repeat((2, 2, 2))

        card = MagneticOrderCard()
        card.format_combo.setCurrentIndex(1)
        card.axis_frame.set_input_value([0.0, 0.0, 1.0])
        card.map_edit.setText("Ni:0.6")
        card.fm_checkbox.setChecked(False)
        card.afm_checkbox.setChecked(False)
        card.pm_checkbox.setChecked(True)
        card.pm_count_frame.set_input_value([2])
        card.pm_direction_combo.setCurrentIndex(0)
        card.pm_balanced_checkbox.setChecked(True)
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([123])

        results = card.process_structure(base)
        self.assertEqual(len(results), 2)
        for atoms in results:
            m = np.array(atoms.get_initial_magnetic_moments(), dtype=float)
            self.assertEqual(m.ndim, 2)
            self.assertEqual(m.shape[1], 3)
            self.assertTrue(np.any(np.linalg.norm(m, axis=1) > 0))
            self.assertIn("MagPMnc", str(atoms.info.get("Config_type", "")))

    def test_set_magnetic_moments_card_map_vector_roundtrip(self):
        structure = self._spin_chain()

        card = SetMagneticMomentsCard()
        card.source_combo.setCurrentText("Map/default magnitude")
        card.format_combo.setCurrentText("Non-collinear (vector)")
        card.map_edit.setText("Fe:2.5")
        card.axis_frame.set_input_value([0.0, 0.0, 1.0])

        result = card.process_structure(structure)[0]
        moments = np.array(result.get_initial_magnetic_moments(), dtype=float)
        self.assertEqual(moments.shape, (4, 3))
        self.assertTrue(np.allclose(moments, np.tile([0.0, 0.0, 2.5], (4, 1)), atol=1e-6))
        self.assertIn("MagSet(map,vec)", str(result.info.get("Config_type", "")))

        data = card.to_dict()
        restored = SetMagneticMomentsCard()
        restored.from_dict(data)
        self.assertEqual(restored.source_combo.currentText(), "Map/default magnitude")
        self.assertEqual(restored.format_combo.currentText(), "Non-collinear (vector)")
        self.assertEqual(restored.map_edit.text(), "Fe:2.5")

    def test_set_and_order_magnetic_maps_honor_defaults_directions_and_element_scope(self):
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
        mapped = SetMagneticMomentsOperation().run_structure(
            structure,
            SetMagneticMomentsParams(
                source="Map/default magnitude",
                format="Non-collinear (vector)",
                axis=(0.0, 0.0, 1.0),
                magmom_map="Fe:[2,0,0]",
                use_element_dirs=True,
                default_moment=1.5,
                apply_elements="Fe,Ni",
            ),
        )[0]
        np.testing.assert_allclose(
            mapped.arrays["spin"],
            [
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.5],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 1.5],
            ],
            atol=1e-12,
        )

        scoped = SetMagneticMomentsOperation().run_structure(
            structure,
            SetMagneticMomentsParams(
                source="Map/default magnitude",
                format="Non-collinear (vector)",
                magmom_map="Fe:2",
                default_moment=9.0,
                apply_elements="Fe",
            ),
        )[0]
        np.testing.assert_allclose(
            np.linalg.norm(scoped.arrays["spin"], axis=1),
            [2.0, 0.0, 2.0, 0.0],
            atol=1e-12,
        )

        ordered = MagneticOrderOperation().run_structure(
            structure,
            MagneticOrderParams(
                format="noncollinear",
                magmom_map="Fe:[0,2,0]",
                use_element_dirs=True,
                default_moment=1.0,
                apply_elements="Fe",
                gen_fm=True,
                gen_afm=False,
            ),
        )[0]
        np.testing.assert_allclose(
            ordered.arrays["spin"],
            [
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            atol=1e-12,
        )

    def test_set_magnetic_moments_card_existing_scalar_to_vector(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([1.0, 2.0, 3.0, 4.0])

        card = SetMagneticMomentsCard()
        card.source_combo.setCurrentText("Existing initial magmoms")
        card.format_combo.setCurrentText("Non-collinear (vector)")
        card.axis_frame.set_input_value([1.0, 0.0, 0.0])

        result = card.process_structure(structure)[0]
        moments = np.array(result.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(
            np.allclose(
                moments,
                np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
                atol=1e-6,
            )
        )
        self.assertIn("MagSet(existing,vec)", str(result.info.get("Config_type", "")))

    def test_magmom_rotation_lifts_scalar_to_vector(self):
        proto = CrystalPrototypeBuilderCard()
        set_combo_value(proto.structure_combo, "bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.max_output_frame.set_input_value([1])
        base = proto.create_operation().generate(proto.get_params())[0].repeat((2, 2, 2))

        order = MagneticOrderCard()
        order.format_combo.setCurrentIndex(0)
        order.map_edit.setText("Fe:2.2")
        order.fm_checkbox.setChecked(True)
        order.afm_checkbox.setChecked(False)
        order.pm_checkbox.setChecked(False)
        fm = order.process_structure(base)[0]

        rot = MagneticMomentRotationCard()
        rot.elements_input.setText("Fe")
        rot.angle_frame.set_input_value([45.0])
        rot.lift_scalar_checkbox.setChecked(True)
        rot.axis_frame.set_input_value([0.0, 0.0, 1.0])
        rot.count_frame.set_input_value([1])

        rotated = rot.process_structure(fm)[0]
        m = np.array(rotated.get_initial_magnetic_moments(), dtype=float)
        self.assertEqual(m.ndim, 2)
        self.assertEqual(m.shape[1], 3)

    def test_magmom_rotation_seed_reproducibility_and_magnitude_bounds(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([2.0, 2.0, 2.0, 2.0])
        params = MagneticMomentRotationParams(
            max_angle=10.0,
            num_structures=3,
            disturb_magnitude=True,
            magnitude_factor=(0.9, 1.1),
            use_seed=True,
            seed=21,
        )

        first = MagneticMomentRotationOperation().run_structure(structure, params)
        second = MagneticMomentRotationOperation().run_structure(structure, params)

        for a, b in zip(first, second):
            ma = np.array(a.get_initial_magnetic_moments(), dtype=float)
            mb = np.array(b.get_initial_magnetic_moments(), dtype=float)
            self.assertTrue(np.allclose(ma, mb, atol=1e-12))
            self.assertEqual(ma.shape, (4, 3))
            norms = np.linalg.norm(ma, axis=1)
            self.assertTrue(np.all(norms >= 1.8 - 1e-12))
            self.assertTrue(np.all(norms <= 2.2 + 1e-12))
            self.assertIn("MMR(", a.info.get("Config_type", ""))

    def test_magmom_rotation_card_roundtrips_selection_axis_and_magnitude_controls(self):
        params = MagneticMomentRotationParams(
            elements="Fe,Ni",
            max_angle=17.5,
            num_structures=6,
            lift_scalar=False,
            axis=(1.0, 1.0, 0.0),
            disturb_magnitude=False,
            magnitude_factor=(0.8, 1.2),
            use_seed=True,
            seed=67,
        )
        card = MagneticMomentRotationCard()
        card.set_params(params)
        normalized_params = card.get_params()
        restored = MagneticMomentRotationCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), normalized_params)
