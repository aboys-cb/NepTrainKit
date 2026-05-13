from .card_test_base import *


class TestStructureGeneratorCards(BaseCardTest):
    def test_local_solvation_ion_water_is_reproducible_and_tags_output(self):
        structure = Atoms(
            symbols=["Ca"],
            positions=[[0.0, 0.0, 0.0]],
            pbc=False,
        )
        structure.info["Config_type"] = "Ca_seed"
        params = LocalSolvationParams(
            structures=1,
            solvent_count=2,
            sampling_mode="auto",
            center_mode="elements",
            center_elements="Ca",
            shell=(2.4, 3.2),
            min_distance=0.8,
            max_attempts=1000,
            use_seed=True,
            seed=17,
        )

        first = LocalSolvationOperation().run_structure(structure, params)[0]
        second = LocalSolvationOperation().run_structure(structure, params)[0]

        self.assertEqual(len(first), 7)
        self.assertEqual(first.get_chemical_symbols().count("O"), 2)
        self.assertEqual(first.get_chemical_symbols().count("H"), 4)
        self.assertTrue(np.allclose(first.cell.array, np.diag([100.0, 100.0, 100.0])))
        self.assertTrue(np.allclose(first.get_positions(), second.get_positions()))
        self.assertIn("SolvLocal(mode=ion-water,n=2,sel=1)", first.info.get("Config_type", ""))

    def test_local_solvation_rejects_empty_selection(self):
        structure = Atoms(
            symbols=["Si"],
            positions=[[0.0, 0.0, 0.0]],
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=False,
        )
        with self.assertRaisesRegex(ValueError, "no center atoms selected"):
            LocalSolvationOperation().run_structure(
                structure,
                LocalSolvationParams(
                    solvent_count=1,
                    center_mode="elements",
                    center_elements="Ca",
                    use_seed=True,
                    seed=1,
                ),
            )

    def test_local_solvation_dense_periodic_solid_fails_with_actionable_message(self):
        structure = Atoms(
            symbols=["Cu", "Cu", "Cu", "Cu"],
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            cell=np.diag([2.5, 2.5, 2.5]),
            pbc=True,
        )

        with self.assertRaisesRegex(ValueError, "periodic dense structure"):
            LocalSolvationOperation().run_structure(
                structure,
                LocalSolvationParams(
                    solvent_count=2,
                    sampling_mode="water",
                    center_mode="all",
                    shell=(1.8, 2.5),
                    min_distance=1.0,
                    max_attempts=5000,
                    use_seed=True,
                    seed=3,
                ),
            )

    def test_local_solvation_card_roundtrip(self):
        card = LocalSolvationCard()
        card.structures_frame.set_input_value([2])
        card.count_frame.set_input_value([3])
        card.mode_combo.setCurrentText("water")
        card.center_mode_combo.setCurrentText("indices")
        card.indices_edit.setText("1")
        card.shell_frame.set_input_value([2.0, 3.5])
        card.min_distance_frame.set_input_value([0.75])
        card.strict_checkbox.setChecked(True)
        card.auto_box_checkbox.setChecked(True)
        card.box_size_frame.set_input_value([90.0])
        card.box_frame.set_input_value([6.0, 20.0])
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([23])

        restored = LocalSolvationCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_solvent_box_fill_fixed_count_preserves_cell_and_is_reproducible(self):
        structure = Atoms(
            symbols=["Si"],
            positions=[[1.0, 1.0, 1.0]],
            cell=np.diag([16.0, 16.0, 16.0]),
            pbc=True,
        )
        structure.info["Config_type"] = "box_seed"
        params = SolventBoxFillParams(
            structures=1,
            count_mode="fixed",
            solvent_count=2,
            min_distance=0.8,
            max_attempts_per_solvent=1000,
            use_seed=True,
            seed=5,
        )

        first = SolventBoxFillOperation().run_structure(structure, params)[0]
        second = SolventBoxFillOperation().run_structure(structure, params)[0]

        self.assertEqual(len(first), 7)
        self.assertTrue(np.allclose(first.cell.array, structure.cell.array))
        self.assertTrue(first.pbc.all())
        self.assertTrue(np.allclose(first.get_positions(), second.get_positions()))
        self.assertIn("SolvBox(mode=water,n=2)", first.info.get("Config_type", ""))

    def test_solvent_box_fill_density_mode_and_card_roundtrip(self):
        structure = Atoms(
            symbols=["Ar"],
            positions=[[2.0, 2.0, 2.0]],
            cell=np.diag([10.0, 10.0, 10.0]),
            pbc=True,
        )
        params = SolventBoxFillParams(
            count_mode="density",
            density=1.0,
            fill_packing=1.0,
            min_distance=0.5,
            max_attempts_per_solvent=1000,
            use_seed=True,
            seed=9,
        )
        result = SolventBoxFillOperation().run_structure(structure, params)[0]
        self.assertGreater(len(result), len(structure))

        card = SolventBoxFillCard()
        card.count_mode_combo.setCurrentText("density")
        card.count_frame.set_input_value([9])
        card.density_frame.set_input_value([0.7])
        card.mode_combo.setCurrentText("loose")
        card.fill_packing_frame.set_input_value([0.8])
        card.min_distance_frame.set_input_value([0.6])
        card.strict_checkbox.setChecked(True)
        card.flex_checkbox.setChecked(False)
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([19])

        restored = SolventBoxFillCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_solvent_box_fill_tiny_dense_cell_fails_quickly(self):
        structure = Atoms(
            symbols=["Cu", "Cu", "Cu", "Cu"],
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            cell=np.diag([2.5, 2.5, 2.5]),
            pbc=True,
        )

        with self.assertRaisesRegex(ValueError, "too small/dense"):
            SolventBoxFillOperation().run_structure(
                structure,
                SolventBoxFillParams(
                    solvent_count=10,
                    min_distance=1.0,
                    max_attempts_per_solvent=1000,
                    use_seed=True,
                    seed=3,
                ),
            )

    def test_solvent_box_fill_rejects_local_ion_water_mode(self):
        structure = Atoms(
            symbols=["Na"],
            positions=[[4.0, 4.0, 4.0]],
            cell=np.diag([12.0, 12.0, 12.0]),
            pbc=True,
        )

        with self.assertRaisesRegex(ValueError, "sampling_mode"):
            SolventBoxFillOperation().run_structure(
                structure,
                SolventBoxFillParams(
                    sampling_mode="ion-water",
                    solvent_count=1,
                    use_seed=True,
                    seed=2,
                ),
            )

    def test_local_solvation_respects_global_min_distance(self):
        structure = Atoms(
            symbols=["Ca"],
            positions=[[0.0, 0.0, 0.0]],
            pbc=False,
        )
        min_distance = 0.85
        result = LocalSolvationOperation().run_structure(
            structure,
            LocalSolvationParams(
                solvent_count=4,
                sampling_mode="ion-water",
                center_mode="elements",
                center_elements="Ca",
                shell=(2.4, 3.6),
                min_distance=min_distance,
                max_attempts=3000,
                use_seed=True,
                seed=31,
            ),
        )[0]

        symbols = result.get_chemical_symbols()
        positions = result.get_positions()
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if symbols[i] == "O" and symbols[j] == "H":
                    continue
                if symbols[i] == "H" and symbols[j] == "O":
                    continue
                if symbols[i] == symbols[j] == "H":
                    continue
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                self.assertGreaterEqual(distance + 1e-12, min_distance)

    def test_local_solvation_z_range_uses_selected_center_region(self):
        structure = Atoms(
            symbols=["Ca", "Ca"],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]],
            pbc=False,
        )
        result = LocalSolvationOperation().run_structure(
            structure,
            LocalSolvationParams(
                solvent_count=2,
                sampling_mode="ion-water",
                center_mode="z_range",
                z_range=(19.0, 21.0),
                shell=(2.4, 3.4),
                min_distance=0.8,
                max_attempts=1000,
                use_seed=True,
                seed=8,
            ),
        )[0]

        oxygen_positions = np.array([pos for sym, pos in zip(result.get_chemical_symbols(), result.get_positions()) if sym == "O"])
        self.assertEqual(len(oxygen_positions), 2)
        near_top = np.linalg.norm(oxygen_positions - np.array([0.0, 0.0, 20.0]), axis=1)
        near_bottom = np.linalg.norm(oxygen_positions - np.array([0.0, 0.0, 0.0]), axis=1)
        self.assertTrue(np.all(near_top < near_bottom))

    def test_solvent_box_fill_density_count_matches_formula(self):
        structure = Atoms(
            symbols=["Si"],
            positions=[[1.0, 1.0, 1.0]],
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=True,
        )
        solvent = parse_solvent_xyz(LocalSolvationParams().solvent_xyz)
        expected = estimate_solvent_count_from_density(solvent, 1.0, structure.cell.array, 1.0)
        result = SolventBoxFillOperation().run_structure(
            structure,
            SolventBoxFillParams(
                count_mode="density",
                density=1.0,
                fill_packing=1.0,
                min_distance=0.8,
                max_attempts_per_solvent=2000,
                use_seed=True,
                seed=7,
            ),
        )[0]

        self.assertEqual(result.get_chemical_symbols().count("O"), expected)
        self.assertEqual(result.get_chemical_symbols().count("H"), 2 * expected)

    def test_solvent_box_fill_nonorthogonal_output_has_no_pbc_collisions(self):
        structure = Atoms(
            symbols=["Si"],
            positions=[[1.0, 1.0, 1.0]],
            cell=np.array([[14.0, 0.0, 0.0], [3.0, 13.0, 0.0], [1.0, 2.0, 15.0]]),
            pbc=True,
        )
        min_distance = 0.8
        result = SolventBoxFillOperation().run_structure(
            structure,
            SolventBoxFillParams(
                solvent_count=8,
                min_distance=min_distance,
                max_attempts_per_solvent=1000,
                use_seed=True,
                seed=12,
            ),
        )[0]

        symbols = result.get_chemical_symbols()
        positions = result.get_positions()
        for i in range(len(result)):
            self.assertFalse(
                has_collision(
                    [symbols[i]],
                    positions[i : i + 1],
                    symbols[:i] + symbols[i + 1 :],
                    np.vstack([positions[:i], positions[i + 1 :]]),
                    cell=result.cell.array,
                    pbc=np.asarray(result.pbc, dtype=bool),
                    collision_scale=0.70,
                    min_distance=min_distance,
                )
            )

    def test_solvent_box_fill_partial_output_when_not_strict(self):
        structure = Atoms(
            symbols=["Cu", "Cu", "Cu", "Cu"],
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            cell=np.diag([2.5, 2.5, 2.5]),
            pbc=True,
        )
        result = SolventBoxFillOperation().run_structure(
            structure,
            SolventBoxFillParams(
                solvent_count=10,
                min_distance=1.0,
                max_attempts_per_solvent=1000,
                strict_count=False,
                use_seed=True,
                seed=3,
            ),
        )[0]

        self.assertEqual(len(result), len(structure))
        self.assertIn("SolvBox(mode=water,n=0)", result.info.get("Config_type", ""))

    def test_solvent_box_fill_flexible_solvent_branch_runs(self):
        butane = """14
butane
C -2.3100 0.0000 0.0000
C -0.7700 0.0000 0.0000
C 0.7700 0.0000 0.0000
C 2.3100 0.0000 0.0000
H -2.6700 1.0000 0.0000
H -2.6700 -0.5000 0.8660
H -2.6700 -0.5000 -0.8660
H -0.4100 0.5000 0.8660
H -0.4100 0.5000 -0.8660
H 0.4100 -0.5000 0.8660
H 0.4100 -0.5000 -0.8660
H 2.6700 -1.0000 0.0000
H 2.6700 0.5000 0.8660
H 2.6700 0.5000 -0.8660
"""
        structure = Atoms(
            symbols=["Ar"],
            positions=[[8.0, 8.0, 8.0]],
            cell=np.diag([30.0, 30.0, 30.0]),
            pbc=True,
        )
        result = SolventBoxFillOperation().run_structure(
            structure,
            SolventBoxFillParams(
                solvent_xyz=butane,
                solvent_count=1,
                sampling_mode="general",
                min_distance=0.8,
                flex_solvent=True,
                flex_pool=3,
                flex_max_torsions=1,
                flex_gaussian_sigma=0.01,
                use_seed=True,
                seed=4,
            ),
        )[0]

        self.assertEqual(len(result), len(structure) + 14)
        self.assertEqual(result.get_chemical_symbols().count("C"), 4)
        self.assertIn("SolvBox(mode=general,n=1)", result.info.get("Config_type", ""))

    def test_random_packing_preserves_cell_composition_and_distance_constraints(self):
        structure = Atoms(
            symbols=["Fe", "Fe", "O", "O"],
            positions=np.zeros((4, 3)),
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=True,
        )
        structure.info["Config_type"] = "FeO_seed"

        results = RandomPackingOperation().run_structure(
            structure,
            RandomPackingParams(
                structures=2,
                min_distance=1.0,
                pair_min_distances="Fe-O:2.0,O-O:1.5",
                use_seed=True,
                seed=7,
            ),
        )

        self.assertEqual(len(results), 2)
        for atoms in results:
            self.assertTrue(np.allclose(atoms.cell.array, structure.cell.array))
            self.assertTrue(np.array_equal(np.asarray(atoms.pbc, dtype=bool), np.asarray(structure.pbc, dtype=bool)))
            self.assertEqual(atoms.get_chemical_symbols().count("Fe"), 2)
            self.assertEqual(atoms.get_chemical_symbols().count("O"), 2)
            symbols = atoms.get_chemical_symbols()
            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):
                    dist = RandomPackingOperation.candidate_distances(
                        atoms.positions[i],
                        atoms.positions[j : j + 1],
                        cell=np.asarray(atoms.cell.array, dtype=float),
                        pbc=np.asarray(atoms.pbc, dtype=bool),
                    )[0]
                    expected = RandomPackingOperation.min_distance_for_pair(
                        symbols[i],
                        symbols[j],
                        1.0,
                        RandomPackingOperation.parse_pair_min_distances("Fe-O:2.0,O-O:1.5"),
                    )
                    self.assertGreaterEqual(dist + 1e-12, expected)
            self.assertIn("RandPack(n=4,d=1", atoms.info.get("Config_type", ""))

    def test_random_packing_manual_exact_composition_and_roundtrip(self):
        structure = Atoms(
            symbols=["Si"],
            positions=[[0.0, 0.0, 0.0]],
            cell=np.diag([7.0, 7.0, 7.0]),
            pbc=True,
        )
        structure.info["Config_type"] = "manual_pack"

        card = RandomPackingCard()
        card.structures_frame.set_input_value([1])
        card.composition_edit.setText("Fe:2,O:1")
        card.min_distance_frame.set_input_value([1.0])
        card.pair_distance_edit.setText("Fe-O:1.2")
        card.attempts_frame.set_input_value([200])
        card.strict_checkbox.setChecked(True)
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([9])

        result = card.process_structure(structure)[0]
        self.assertEqual(result.get_chemical_symbols().count("Fe"), 2)
        self.assertEqual(result.get_chemical_symbols().count("O"), 1)
        self.assertEqual(len(result), 3)

        restored = RandomPackingCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_random_packing_invalid_or_impossible_constraints_fail_explicitly(self):
        structure = Atoms(
            symbols=["He", "He"],
            positions=[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            cell=np.diag([1.0, 1.0, 1.0]),
            pbc=True,
        )
        with self.assertRaisesRegex(ValueError, "composition count"):
            RandomPackingOperation.symbols_from_params(structure, "Fe:0.5,O:0.5")

        with self.assertRaisesRegex(ValueError, "could not place"):
            RandomPackingOperation().run_structure(
                structure,
                RandomPackingParams(
                    structures=1,
                    min_distance=0.9,
                    max_attempts_per_atom=5,
                    strict_mode=True,
                    use_seed=True,
                    seed=1,
                ),
            )

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

    def test_crystal_prototype_builder_reuses_auto_supercell_factors(self):
        params = CrystalPrototypeBuilderParams(
            lattice="fcc",
            element="Cu",
            a_range=(3.5, 3.7, 0.1),
            auto_supercell=True,
            max_atoms=128,
            max_outputs=3,
        )
        operation = CrystalPrototypeBuilderOperation()
        calls = []
        original = operation.__class__.generate.__globals__["best_supercell_factors_max_atoms"]

        def counted(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        operation.__class__.generate.__globals__["best_supercell_factors_max_atoms"] = counted
        try:
            results = operation.generate(params)
        finally:
            operation.__class__.generate.__globals__["best_supercell_factors_max_atoms"] = original

        self.assertEqual(len(results), 3)
        self.assertEqual(len(calls), 1)
        self.assertTrue(all("Proto(fcc" in atoms.info.get("Config_type", "") for atoms in results))

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
