from .magnetism_test_base import *


class TestMagnetismOrderCards(MagnetismCardTest):
    def test_magnetic_order_card_fm_afm(self):
        proto = CrystalPrototypeBuilderCard()
        proto.structure_combo.setCurrentText("bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.manual_supercell_button.setChecked(True)
        proto.auto_supercell_button.setChecked(False)
        proto.rep_frame.set_input_value([2, 2, 2])
        proto.max_output_frame.set_input_value([1])
        base = proto.create_operation().generate(proto.get_params())[0]

        card = MagneticOrderCard()
        card.map_edit.setText("Fe:2.2")
        card.fm_checkbox.setChecked(True)
        card.afm_checkbox.setChecked(True)
        card.kvec_combo.setCurrentText("100")
        card.pm_checkbox.setChecked(False)

        results = card.process_structure(base)
        self.assertEqual(len(results), 2)
        fm = [a for a in results if "MagFM" in str(a.info.get("Config_type", ""))][0]
        afm = [a for a in results if "MagAFM100" in str(a.info.get("Config_type", ""))][0]

        fm_m = np.array(fm.get_initial_magnetic_moments(), dtype=float)
        afm_m = np.array(afm.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(np.all(fm_m >= 0))
        self.assertTrue(np.any(afm_m > 0) and np.any(afm_m < 0))

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

    def test_magnetic_order_card_noncollinear_pm(self):
        proto = CrystalPrototypeBuilderCard()
        proto.structure_combo.setCurrentText("fcc")
        proto.element_edit.setText("Ni")
        proto.a_frame.set_input_value([3.5, 3.5, 0.1])
        proto.manual_supercell_button.setChecked(True)
        proto.auto_supercell_button.setChecked(False)
        proto.rep_frame.set_input_value([2, 2, 2])
        proto.max_output_frame.set_input_value([1])
        base = proto.create_operation().generate(proto.get_params())[0]

        card = MagneticOrderCard()
        card.format_combo.setCurrentText("Non-collinear (vector)")
        card.axis_frame.set_input_value([0.0, 0.0, 1.0])
        card.map_edit.setText("Ni:0.6")
        card.fm_checkbox.setChecked(False)
        card.afm_checkbox.setChecked(False)
        card.pm_checkbox.setChecked(True)
        card.pm_count_frame.set_input_value([2])
        card.pm_direction_combo.setCurrentText("sphere")
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
        proto.structure_combo.setCurrentText("bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.manual_supercell_button.setChecked(True)
        proto.auto_supercell_button.setChecked(False)
        proto.rep_frame.set_input_value([2, 2, 2])
        proto.max_output_frame.set_input_value([1])
        base = proto.create_operation().generate(proto.get_params())[0]

        order = MagneticOrderCard()
        order.format_combo.setCurrentText("Collinear (scalar)")
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
