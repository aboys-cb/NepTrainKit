from .card_test_base import *


class TestStructureGeneratorCards(BaseCardTest):
    def test_crystal_prototype_builder_card(self):
        card = CrystalPrototypeBuilderCard()
        card.structure_combo.setCurrentText("fcc")
        card.element_edit.setText("Cu")
        card.a_frame.set_input_value([3.6, 3.6, 0.1])
        card.auto_supercell_button.setChecked(True)
        card.manual_supercell_button.setChecked(False)
        card.max_atoms_frame.set_input_value([64])
        card.max_output_frame.set_input_value([10])

        results = card.create_operation().generate(card.get_params())
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(atoms.pbc.all() for atoms in results))
        self.assertTrue(all(len(atoms) <= 64 for atoms in results))

    def test_crystal_prototype_builder_operation_is_ui_independent(self):
        params = CrystalPrototypeBuilderParams(
            lattice="bcc",
            element="Fe",
            a_range=(2.9, 2.9, 0.1),
            auto_supercell=False,
            rep=(1, 1, 1),
            max_outputs=1,
        )
        results = CrystalPrototypeBuilderOperation().generate(params)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].pbc.all())
        self.assertIn("Proto(bcc", results[0].info.get("Config_type", ""))

    def test_crystal_prototype_builder_card_roundtrip(self):
        card = CrystalPrototypeBuilderCard()
        card.structure_combo.setCurrentText("hcp")
        card.element_edit.setText("Mg")
        card.a_frame.set_input_value([3.1, 3.3, 0.1])
        card.covera_frame.set_input_value([1.62])
        card.auto_supercell_button.setChecked(False)
        card.manual_supercell_button.setChecked(True)
        card.max_atoms_frame.set_input_value([128])
        card.rep_frame.set_input_value([2, 3, 4])
        card.max_output_frame.set_input_value([5])

        restored = CrystalPrototypeBuilderCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_group_label_card_and_group_afm(self):
        proto = CrystalPrototypeBuilderCard()
        proto.structure_combo.setCurrentText("bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.manual_supercell_button.setChecked(True)
        proto.auto_supercell_button.setChecked(False)
        proto.rep_frame.set_input_value([2, 2, 2])
        proto.max_output_frame.set_input_value([1])
        base = proto.create_operation().generate(proto.get_params())[0]

        gl = GroupLabelCard()
        gl.mode_combo.setCurrentText("k-vector layers (recommended)")
        gl.kvec_combo.setCurrentText("111")
        gl.group_a_edit.setText("A")
        gl.group_b_edit.setText("B")
        labeled = gl.process_structure(base)[0]
        self.assertIn("group", labeled.arrays)
        groups = set(str(g) for g in labeled.arrays["group"])
        self.assertTrue({"A", "B"}.issubset(groups))

        mag = MagneticOrderCard()
        mag.format_combo.setCurrentText("Collinear (scalar)")
        mag.map_edit.setText("Fe:2.2")
        mag.fm_checkbox.setChecked(False)
        mag.afm_checkbox.setChecked(True)
        mag.afm_mode_combo.setCurrentText("group A/B")
        mag.group_a_edit.setText("A")
        mag.group_b_edit.setText("B")
        mag.pm_checkbox.setChecked(False)
        res = mag.process_structure(labeled)
        self.assertEqual(len(res), 1)
        afm = res[0]
        m = np.array(afm.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(np.any(m > 0) and np.any(m < 0))

    def test_group_label_operation_is_ui_independent(self):
        base = CrystalPrototypeBuilderOperation().generate(
            CrystalPrototypeBuilderParams(
                lattice="bcc",
                element="Fe",
                a_range=(2.9, 2.9, 0.1),
                auto_supercell=False,
                rep=(2, 2, 2),
                max_outputs=1,
            )
        )[0]

        labeled = GroupLabelOperation().run_structure(
            base,
            GroupLabelParams(
                mode="k-vector layers (recommended)",
                kvec="111",
                group_a="up",
                group_b="down",
            ),
        )[0]

        self.assertIn("group", labeled.arrays)
        self.assertEqual(set(str(value) for value in labeled.arrays["group"]), {"up", "down"})
        self.assertIn("Grp(k111,up/down)", labeled.info.get("Config_type", ""))

    def test_group_label_card_roundtrip(self):
        card = GroupLabelCard()
        card.mode_combo.setCurrentText("fractional parity")
        card.kvec_combo.setCurrentText("110")
        card.group_a_edit.setText("alpha")
        card.group_b_edit.setText("beta")
        card.overwrite_checkbox.setChecked(False)

        restored = GroupLabelCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_organic_configuration_card(self):
        card = OrganicMolConfigPBCCard()
        structure = self.structure.copy()
        card.perturb_frame.set_input_value([2])
        card.torsion_frame.set_input_value([-30.0, 30.0])
        card.max_torsions_frame.set_input_value([1])
        card.sigma_frame.set_input_value([0.01])
        card.pbc_combo.setCurrentIndex(1)

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        for atoms in results:
            self.assertEqual(len(atoms), len(structure))
            self.assertFalse(np.allclose(atoms.get_positions(), structure.get_positions()))

    def test_organic_configuration_card_roundtrip(self):
        card = OrganicMolConfigPBCCard()
        card.perturb_frame.set_input_value([3])
        card.torsion_frame.set_input_value([-45.0, 60.0])
        card.max_torsions_frame.set_input_value([2])
        card.sigma_frame.set_input_value([0.02])
        card.pbc_combo.setCurrentIndex(2)
        card.local_cut_frame.set_input_value([80])
        card.local_sub_frame.set_input_value([30])
        card.bond_max_enable.setChecked(True)
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([11])

        restored = OrganicMolConfigPBCCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())
