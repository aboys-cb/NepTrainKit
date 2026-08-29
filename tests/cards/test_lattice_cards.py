from .card_test_base import *
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.io import read as ase_read, write as ase_write
from unittest.mock import patch
import io
import json
import re
import warnings
from NepTrainKit.core.cards.sampling import derived_structure_seed


class TestLatticeCards(BaseCardTest):
    def test_supercell_card_variants(self):
        card = SuperCellCard()
        structure = self.structure.copy()
        original_cell = np.array(structure.get_cell())

        card.set_params(SuperCellParams(mode="scale", super_scale=(2, 1, 1)))

        direct_results = card.process_structure(structure)
        self.assertEqual(len(direct_results), 1)
        self.assertEqual(len(direct_results[0]), len(structure) * 2)
        new_cell = np.array(direct_results[0].get_cell())
        self.assertGreater(np.linalg.norm(new_cell[0]), np.linalg.norm(original_cell[0]))

        lengths = structure.cell.lengths()
        card.set_params(
            SuperCellParams(
                mode="cell",
                target_policy="at_least",
                target_cell=(lengths[0] * 2.1, lengths[1] * 1.1, lengths[2] * 1.1),
            )
        )

        cell_results = card.process_structure(structure)
        self.assertEqual(len(cell_results), 1)
        self.assertGreater(len(cell_results[0]), len(structure))

        card.set_params(
            SuperCellParams(
                mode="max_atoms",
                output_mode="enumerate",
                max_atoms=len(structure) * 2,
            )
        )

        atoms_results = card.process_structure(structure)
        self.assertGreaterEqual(len(atoms_results), 1)
        self.assertTrue(any(len(atoms) > len(structure) for atoms in atoms_results))

    def test_supercell_card_fixed_axis_lock(self):
        card = SuperCellCard()
        structure = self.structure.copy()
        original_cell = np.array(structure.get_cell())
        original_lengths = np.linalg.norm(original_cell, axis=1)

        card.set_params(
            SuperCellParams(
                mode="cell",
                output_mode="enumerate",
                target_policy="at_least",
                target_cell=(
                    original_lengths[0] * 2.1,
                    original_lengths[1] * 2.1,
                    original_lengths[2] * 4.0,
                ),
                fixed_axis_flags=(False, False, True),
                fixed_axis_scale=(1, 1, 1),
            )
        )

        results = card.process_structure(structure)
        self.assertEqual(len(results), 9)
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
        self.assertEqual(
            restored.super_cell_condition_frame.get_input_value(),
            list(card.super_cell_condition_frame.get_input_value()),
        )

    def test_supercell_operation_matches_card_params(self):
        card = SuperCellCard()
        structure = self.structure.copy()
        card.set_params(SuperCellParams(mode="scale", super_scale=(2, 1, 1)))

        params = card.get_params()
        self.assertIsInstance(params, SuperCellParams)
        card_result = card.process_structure(structure)
        op_result = SuperCellOperation().run_structure(structure, params)

        self.assertEqual(len(card_result), len(op_result))
        self.assertEqual(len(card_result[0]), len(op_result[0]))
        self.assertIn("params", card.to_dict())

    def test_supercell_target_policy_exact_multiple_and_nonorthogonal_cell(self):
        structure = Atoms(
            "Fe2",
            positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            cell=[[5.0, 0.0, 0.0], [1.0, 5.916079783, 0.0], [0.5, 0.5, 6.964194139]],
            pbc=[True, True, False],
        )
        operation = SuperCellOperation()
        lengths = tuple(float(value) for value in structure.cell.lengths())

        exact = operation.run_structure(
            structure,
            SuperCellParams(
                mode="cell",
                target_policy="at_least",
                target_cell=(lengths[0] * 4, lengths[1] * 3, lengths[2] * 2),
            ),
        )[0]
        np.testing.assert_allclose(exact.cell.array, np.diag([4, 3, 2]) @ structure.cell.array)
        np.testing.assert_array_equal(exact.pbc, structure.pbc)
        self.assertEqual(len(exact), len(structure) * 24)

        at_most = operation.plan_factors(
            structure,
            SuperCellParams(
                mode="cell",
                target_policy="at_most",
                target_cell=(lengths[0] * 4.9, lengths[1] * 3.9, lengths[2] * 2.9),
            ),
        )
        self.assertEqual(at_most, [(4, 3, 2)])

    def test_supercell_atom_budget_prefers_balanced_full_cell(self):
        structure = Atoms(
            "Fe2",
            positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            cell=[5.0, 6.0, 7.0],
            pbc=True,
        )
        operation = SuperCellOperation()
        factors = operation.plan_factors(
            structure,
            SuperCellParams(mode="max_atoms", max_atoms=100),
        )

        self.assertEqual(factors, [(5, 5, 2)])
        result = operation.run_structure(
            structure,
            SuperCellParams(mode="max_atoms", max_atoms=100),
        )[0]
        self.assertEqual(len(result), 100)
        self.assertLess(max(result.cell.lengths()) / min(result.cell.lengths()), 2.2)

    def test_supercell_enumeration_limit_and_impossible_budget_are_visible(self):
        structure = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=[2.0, 2.0, 2.0], pbc=True)
        operation = SuperCellOperation()
        with self.assertRaisesRegex(ValueError, "more than 1000"):
            operation.plan_factors(
                structure,
                SuperCellParams(mode="max_atoms", output_mode="enumerate", max_atoms=10000),
            )
        with self.assertRaisesRegex(ValueError, "smaller than the input"):
            operation.run_structure(
                Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[2, 2, 2], pbc=True),
                SuperCellParams(mode="max_atoms", max_atoms=1),
            )
        with self.assertRaisesRegex(ValueError, "fixed-axis multipliers"):
            operation.run_structure(
                structure,
                SuperCellParams(
                    mode="max_atoms",
                    max_atoms=4,
                    fixed_axis_flags=(True, True, True),
                    fixed_axis_scale=(2, 2, 2),
                ),
            )

    def test_supercell_ui_disclosure_preview_and_legacy_json_migration(self):
        card = SuperCellCard()
        self.assertFalse(card.super_scale_field.isHidden())
        self.assertTrue(card.target_cell_field.isHidden())
        self.assertTrue(card.max_atoms_field.isHidden())
        self.assertTrue(card.fixed_scale_field.isHidden())
        self.assertIn("27", card.output_preview.text())

        card.from_dict(
            {
                "class": "SuperCellCard",
                "check_state": True,
                "params": {
                    "behavior_type": 1,
                    "mode": "cell",
                    "target_cell": [20.0, 18.0, 14.0],
                    "fixed_axis_flags": [False, False, True],
                    "fixed_axis_scale": [1, 1, 1],
                },
            }
        )
        params = card.get_params()
        self.assertEqual(params.mode, "cell")
        self.assertEqual(params.output_mode, "enumerate")
        self.assertEqual(params.target_policy, "at_most")
        self.assertTrue(card.super_scale_field.isHidden())
        self.assertFalse(card.target_cell_field.isHidden())
        self.assertFalse(card.fixed_scale_field.isHidden())
        self.assertFalse(card.fixed_scale_condition_frame.object_list[0].isEnabled())
        self.assertFalse(card.fixed_scale_condition_frame.object_list[1].isEnabled())
        self.assertTrue(card.fixed_scale_condition_frame.object_list[2].isEnabled())

        restored = SuperCellCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), params)

    def test_supercell_documented_presets_load_the_declared_modes(self):
        doc_path = (
            PROJECT_ROOT
            / "docs/source/module/make-dataset-cards/cards/super-cell-card.md"
        )
        blocks = re.findall(
            r"```json\s*(.*?)```",
            doc_path.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
        self.assertEqual(len(blocks), 3)

        modes = []
        for block in blocks:
            payload = json.loads(block)
            card = SuperCellCard()
            card.from_dict(payload)
            modes.append(card.get_params().mode)

        self.assertEqual(modes, ["scale", "cell", "max_atoms"])

    def test_bain_path_card_roundtrip(self):
        card = BainPathCard()
        card.set_params(
            BainPathParams(
                axis="x",
                ca_range=(0.8, 1.2, 0.2),
                coordinate_mode="relative_ca",
                mode="scale_volume",
                volume_scale_range=(0.9, 1.1, 0.1),
                scale_atoms=False,
            )
        )

        restored = BainPathCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_bain_path_default_is_three_points_and_mode_controls_volume_scan(self):
        card = BainPathCard()
        self.assertEqual(card.getTitle(), "Bain Path")
        self.assertEqual(card.get_params(), BainPathParams())
        self.assertTrue(card.volume_field.isHidden())
        self.assertIn("3 path points = 3 outputs/input", card.preview_label.text())

        card.set_dataset([self.structure.copy(), self.structure.copy()])
        self.assertIn("input structures: 2 → 6 outputs", card.preview_label.text())
        self.assertIn(
            "Input structures: 2 × 3 outputs/input = 6 outputs",
            card.get_guidance_text(),
        )
        self.assertIn("relative c/a", card.get_summary_text())

        card.set_params(
            BainPathParams(
                mode="scale_volume",
                volume_scale_range=(0.95, 1.05, 0.05),
            )
        )
        self.assertFalse(card.volume_field.isHidden())
        self.assertIn("3 shape points × 3 volume points = 9 outputs/input", card.preview_label.text())

    def test_bain_path_restores_old_json_as_legacy_axis_scale(self):
        card = BainPathCard()
        card.from_dict(
            {
                "check_state": True,
                "params": {
                    "axis": "z",
                    "ca_range": [1.0, 1.2, 0.2],
                    "mode": "constant_volume",
                    "volume_scale_range": [1.0, 1.0, 1.0],
                    "scale_atoms": True,
                }
            }
        )

        self.assertEqual(card.get_params().coordinate_mode, "axis_scale")
        self.assertIn("Legacy coordinate", card.ca_field.helper_label.text())
        self.assertIn("legacy scale", card.get_summary_text())

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

    def test_perturb_element_scaling_labels_and_disclosure(self):
        card = PerturbCard()

        self.assertEqual(card.get_params().max_distance, 0.25)
        self.assertEqual(card.element_scaling_checkbox.text(), "Use per-element limits")
        self.assertIn("maximum displacement", card.element_scaling_label.text())
        self.assertTrue(card.element_scaling_label.isHidden())
        self.assertTrue(card.element_rows_frame.isHidden())
        self.assertFalse(card.add_element_button.isEnabled())

        card.element_scaling_checkbox.setChecked(True)

        self.assertFalse(card.element_scaling_label.isHidden())
        self.assertFalse(card.element_rows_frame.isHidden())
        self.assertTrue(card.add_element_button.isEnabled())

    def test_perturb_element_limits_normalize_and_fail_closed(self):
        card = PerturbCard()
        card.element_scaling_checkbox.setChecked(True)
        card._add_element_row("h", 0.05)
        self.assertEqual(card.get_params().element_scalings, {"H": 0.05})

        invalid = PerturbCard()
        invalid.element_scaling_checkbox.setChecked(True)
        invalid._add_element_row("Hh", 0.05)
        self.assertFalse(invalid.element_validation_label.isHidden())
        with self.assertRaisesRegex(ValueError, "not a valid element symbol"):
            invalid.process_structure(Atoms("H", positions=[[0, 0, 0]]))

        duplicate = PerturbCard()
        duplicate.element_scaling_checkbox.setChecked(True)
        duplicate._add_element_row("H", 0.05)
        duplicate._add_element_row("h", 0.1)
        self.assertFalse(duplicate.element_validation_label.isHidden())
        with self.assertRaisesRegex(ValueError, "more than one displacement limit"):
            duplicate.process_structure(Atoms("H", positions=[[0, 0, 0]]))

    def test_perturb_summary_reports_outputs_per_input(self):
        card = PerturbCard()
        card.set_dataset([self.structure.copy(), self.structure.copy()])
        card.num_condition_frame.set_input_value([3])
        self.assertIn("3 per input", card.get_summary_text())
        self.assertIn("2 × 3 = 6 outputs", card.get_guidance_text())
        self.assertIn("No collision check", card.get_guidance_text())

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
            metadata = json.loads(atoms.info["atomic_perturb"])
            self.assertEqual(metadata["engine"], "uniform")
            self.assertEqual(metadata["structures_per_input"], 2)
            self.assertEqual(metadata["seed"], 11)

    def test_perturb_sobol_is_seeded_bounded_and_quiet(self):
        structure = Atoms(
            "Si8",
            positions=np.arange(24, dtype=float).reshape(8, 3),
            cell=[30, 30, 30],
            pbc=False,
        )
        params = PerturbParams(
            engine_type=0,
            max_distance=0.2,
            max_num=4,
            use_seed=True,
            seed=17,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = PerturbOperation().run_structure(
                structure,
                PerturbParams(**{**params.__dict__, "max_num": 5}),
            )
        repeated = PerturbOperation().run_structure(structure, params)

        self.assertEqual(len(first), 5)
        self.assertFalse(any("balance properties of Sobol" in str(item.message) for item in caught))
        for output, repeated_output in zip(first[:4], repeated):
            self.assertIn("Pert(d=0.2,S)", output.info.get("Config_type", ""))
            np.testing.assert_allclose(output.positions, repeated_output.positions)
            displacement = output.positions - structure.positions
            self.assertLessEqual(
                float(np.linalg.norm(displacement, axis=1).max()),
                params.max_distance + 1e-12,
            )

    def test_perturb_rigid_organic_cluster_uses_strictest_element_limit(self):
        structure = Atoms(
            "CH4",
            positions=[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
            cell=[20, 20, 20],
            pbc=False,
        )
        params = PerturbParams(
            max_distance=0.2,
            max_num=4,
            identify_organic=True,
            use_element_scaling=True,
            element_scalings={"C": 0.15, "H": 0.05},
            use_seed=True,
            seed=7,
        )
        with patch(
            "NepTrainKit.core.cards.lattice.get_clusters",
            return_value=([np.arange(5)], [True]),
        ):
            outputs = PerturbOperation().run_structure(structure, params)

        for output in outputs:
            displacement = output.positions - structure.positions
            np.testing.assert_allclose(displacement, np.repeat(displacement[:1], 5, axis=0))
            self.assertLessEqual(float(np.linalg.norm(displacement[0])), 0.05 + 1e-12)

    def test_perturb_sobol_has_explicit_dimension_limit(self):
        oversized_count = PerturbOperation._MAX_SOBOL_ATOMS + 1
        oversized = Atoms(
            numbers=np.ones(oversized_count, dtype=int),
            positions=np.zeros((oversized_count, 3)),
        )

        with self.assertRaisesRegex(
            ValueError,
            rf"at most {PerturbOperation._MAX_SOBOL_ATOMS} atoms",
        ):
            PerturbOperation().run_structure(
                oversized,
                PerturbParams(engine_type=0, max_num=1),
            )

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
        with self.assertRaisesRegex(ValueError, "displacement limit for Si"):
            PerturbOperation().run_structure(
                self.structure.copy(),
                PerturbParams(
                    max_distance=0.1,
                    max_num=1,
                    use_element_scaling=True,
                    element_scalings={"Si": float("nan")},
                ),
            )

    def test_perturb_wrap_matches_ase_after_lattice_change(self):
        structure = CellStrainOperation().run_structure(
            self.structure.copy(),
            CellStrainParams(
                axes="triaxial",
                x_range=(1.0, 1.0, 1.0),
                y_range=(-1.0, -1.0, 1.0),
                z_range=(0.5, 0.5, 1.0),
            ),
        )[0]
        params = PerturbParams(max_distance=0.2, max_num=4, use_seed=True, seed=23)

        results = PerturbOperation().run_structure(structure.copy(), params)

        rng = np.random.default_rng(derived_structure_seed(params.seed, structure))
        unit_samples = rng.random((params.max_num, len(structure), 3))
        displacements = PerturbOperation.unit_ball_displacements(
            unit_samples,
            np.full(len(structure), params.max_distance),
        )
        for result, displacement in zip(results, displacements):
            expected = structure.copy()
            expected.set_positions(structure.positions + displacement)
            expected.wrap()
            np.testing.assert_allclose(result.positions, expected.positions, atol=1e-12)

    def test_seeded_random_engines_are_structure_specific_and_reorder_stable(self):
        structure_a = Atoms(
            "Si8",
            positions=np.arange(24, dtype=float).reshape(8, 3),
            cell=[30, 30, 30],
            pbc=False,
        )
        structure_b = structure_a.copy()
        structure_b.positions[0, 0] += 0.125

        for engine_type in (0, 1):
            with self.subTest(operation="Perturb", engine_type=engine_type):
                params = PerturbParams(
                    engine_type=engine_type,
                    max_distance=0.2,
                    max_num=2,
                    use_seed=True,
                    seed=42,
                )
                operation = PerturbOperation()
                a_first = operation.run_structure(structure_a, params)
                b_second = operation.run_structure(structure_b, params)
                b_first = operation.run_structure(structure_b, params)
                a_second = operation.run_structure(structure_a, params)
                duplicate = operation.run_structure(structure_a.copy(), params)

                np.testing.assert_allclose(a_first[0].positions, a_second[0].positions)
                np.testing.assert_allclose(b_second[0].positions, b_first[0].positions)
                np.testing.assert_allclose(a_first[0].positions, duplicate[0].positions)
                self.assertFalse(
                    np.allclose(
                        a_first[0].positions - structure_a.positions,
                        b_second[0].positions - structure_b.positions,
                    )
                )

            with self.subTest(operation="CellScaling", engine_type=engine_type):
                params = CellScalingParams(
                    engine_type=engine_type,
                    max_scaling=0.04,
                    max_num=2,
                    perturb_angle=False,
                    use_seed=True,
                    seed=42,
                )
                operation = CellScalingOperation()
                a_first = operation.run_structure(structure_a, params)
                b_second = operation.run_structure(structure_b, params)
                b_first = operation.run_structure(structure_b, params)
                a_second = operation.run_structure(structure_a, params)
                duplicate = operation.run_structure(structure_a.copy(), params)

                np.testing.assert_allclose(a_first[0].cell.array, a_second[0].cell.array)
                np.testing.assert_allclose(b_second[0].cell.array, b_first[0].cell.array)
                np.testing.assert_allclose(a_first[0].cell.array, duplicate[0].cell.array)
                self.assertFalse(
                    np.allclose(a_first[0].cell.array, b_second[0].cell.array)
                )

    def test_cell_scaling_card_options(self):
        card = CellScalingCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.perturb_angle_checkbox.setChecked(False)
        card.engine_type_combo.setCurrentIndex(1)
        card.scaling_condition_frame.set_input_value([5.0])
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

    def test_cell_scaling_sobol_angle_branch_is_seeded_and_bounded(self):
        structure = self.structure.copy()
        structure.set_cell(
            np.array(
                [
                    [5.4, 0.2, 0.1],
                    [0.4, 5.6, 0.3],
                    [0.2, 0.5, 5.7],
                ]
            ),
            scale_atoms=True,
        )
        params = CellScalingParams(
            engine_type=0,
            max_scaling=0.08,
            max_num=4,
            perturb_angle=True,
            use_seed=True,
            seed=29,
        )

        first = CellScalingOperation().run_structure(structure, params)
        repeated = CellScalingOperation().run_structure(structure, params)

        self.assertEqual(len(first), 4)
        original_cellpar = cell_to_cellpar(structure.cell.array)
        changed_angle = False
        for left, right in zip(first, repeated):
            np.testing.assert_allclose(left.cell.array, right.cell.array, atol=1e-12)
            cellpar = cell_to_cellpar(left.cell.array)
            ratios = cellpar[:3] / original_cellpar[:3]
            self.assertTrue(np.all(ratios >= 0.92 - 1e-7))
            self.assertTrue(np.all(ratios <= 1.08 + 1e-7))
            changed_angle = changed_angle or not np.allclose(
                cellpar[3:],
                original_cellpar[3:],
                atol=1e-7,
            )
            self.assertIn("LSc(max=0.08,S)", left.info.get("Config_type", ""))
        self.assertTrue(changed_angle)

    def test_cell_scaling_zero_angle_perturbation_preserves_global_frame(self):
        structure = self.structure.copy()
        base_cell = np.array(
            [
                [2.4591, 0.7518, -1.5451],
                [0.3314, 2.8161, 0.5386],
                [2.5667, -0.4913, 3.0693],
            ]
        )
        structure.set_cell(base_cell, scale_atoms=True)
        structure.arrays["spin"] = np.tile([0.0, 0.0, 1.0], (len(structure), 1))
        original_positions = structure.positions.copy()
        original_spin = structure.arrays["spin"].copy()

        result = CellScalingOperation().run_structure(
            structure,
            CellScalingParams(max_scaling=0.0, max_num=1, perturb_angle=True),
        )[0]

        np.testing.assert_array_equal(result.cell.array, base_cell)
        np.testing.assert_array_equal(result.positions, original_positions)
        np.testing.assert_array_equal(result.arrays["spin"], original_spin)

    def test_cell_scaling_card_uses_percent_display_and_exact_preview(self):
        card = CellScalingCard()
        card.scaling_condition_frame.set_input_value([4.0])
        card.num_condition_frame.set_input_value([50])
        card.set_preview_input_count(2)

        self.assertAlmostEqual(card.get_params().max_scaling, 0.04)
        self.assertIn("±4%", card.get_summary_text())
        self.assertIn("2 × 50 = 100", card.get_guidance_text())

    def test_cell_scaling_rejects_singular_angle_samples(self):
        structure = self.structure.copy()
        structure.set_cell([[3.0, 0.0, 0.0], [0.7, 2.8, 0.0], [0.4, 0.3, 4.0]])
        with self.assertRaisesRegex(ValueError, "invalid or singular cell"):
            CellScalingOperation().run_structure(
                structure,
                CellScalingParams(
                    max_scaling=1.0,
                    max_num=500,
                    perturb_angle=True,
                    use_seed=True,
                    seed=0,
                ),
            )

    def test_cell_strain_card_uniaxial(self):
        card = CellStrainCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.engine_type_combo.setCurrentIndex(
            card.engine_type_combo.findData("uniaxial")
        )
        card.strain_x_frame.set_input_value([1.0, 1.0, 1.0])
        card.strain_y_frame.set_input_value([0.0, 0.0, 1.0])
        card.strain_z_frame.set_input_value([0.0, 0.0, 1.0])

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
        self.assertIn("Str(a=1%,b=0%,c=0%)", results[0].info.get("Config_type", ""))

    def test_cell_strain_all_public_modes_scale_lattice_vectors(self):
        structure = self.structure.copy()
        structure.set_cell(
            np.array(
                [
                    [5.0, 0.2, 0.1],
                    [0.5, 5.5, 0.3],
                    [0.2, 0.4, 6.0],
                ]
            ),
            scale_atoms=True,
        )
        original = structure.cell.array.copy()
        ranges = {
            "x_range": (1.0, 1.0, 1.0),
            "y_range": (2.0, 2.0, 1.0),
            "z_range": (3.0, 3.0, 1.0),
        }
        operation = CellStrainOperation()

        isotropic = operation.run_structure(
            structure,
            CellStrainParams(axes="isotropic", **ranges),
        )
        self.assertEqual(len(isotropic), 1)
        np.testing.assert_allclose(isotropic[0].cell.array, original * 1.01)

        uniaxial = operation.run_structure(
            structure,
            CellStrainParams(axes="uniaxial", **ranges),
        )
        self.assertEqual(len(uniaxial), 3)
        for axis, result in enumerate(uniaxial):
            expected = original.copy()
            expected[axis] *= 1.0 + (axis + 1) / 100.0
            np.testing.assert_allclose(result.cell.array, expected)

        biaxial = operation.run_structure(
            structure,
            CellStrainParams(axes="biaxial", **ranges),
        )
        self.assertEqual(len(biaxial), 3)
        for axes, result in zip(((0, 1), (0, 2), (1, 2)), biaxial):
            expected = original.copy()
            for axis in axes:
                expected[axis] *= 1.0 + (axis + 1) / 100.0
            np.testing.assert_allclose(result.cell.array, expected)

        triaxial = operation.run_structure(
            structure,
            CellStrainParams(axes="triaxial", **ranges),
        )
        self.assertEqual(len(triaxial), 1)
        np.testing.assert_allclose(
            triaxial[0].cell.array,
            original * np.array([1.01, 1.02, 1.03])[:, None],
        )

        custom = operation.run_structure(
            structure,
            CellStrainParams(axes="XZ", **ranges),
        )
        expected = original.copy()
        expected[0] *= 1.01
        expected[2] *= 1.03
        self.assertEqual(len(custom), 1)
        np.testing.assert_allclose(custom[0].cell.array, expected)

    def test_cell_strain_default_counts_are_exact_and_deduplicated(self):
        operation = CellStrainOperation()
        expected = {
            "uniaxial": 31,
            "biaxial": 331,
            "triaxial": 1331,
            "isotropic": 11,
        }
        for mode, count in expected.items():
            with self.subTest(mode=mode):
                summary = operation.sampling_summary(CellStrainParams(axes=mode))
                self.assertEqual(summary["outputs_per_input"], count)
                results = operation.run_structure(
                    self.structure.copy(), CellStrainParams(axes=mode)
                )
                tags = [atoms.info.get("Config_type", "") for atoms in results]
                self.assertEqual(len(results), count)
                self.assertEqual(len(set(tags)), count)
                self.assertEqual(
                    sum("Str(a=0%,b=0%,c=0%)" in tag for tag in tags), 1
                )

    def test_cell_strain_card_mode_visibility_and_preview(self):
        card = CellStrainCard()
        card.set_preview_input_count(2)
        self.assertIn("31/input", card.get_summary_text())
        self.assertIn("2 × 31 = 62", card.get_guidance_text())
        card.set_preview_input_count(0)
        self.assertIn("Each input produces 31 unique structures", card.get_guidance_text())

        card.engine_type_combo.setCurrentIndex(
            card.engine_type_combo.findData("isotropic")
        )
        self.assertFalse(card.strain_x_field.isHidden())
        self.assertTrue(card.strain_y_field.isHidden())
        self.assertTrue(card.strain_z_field.isHidden())
        self.assertIn("11/input", card.get_summary_text())

        card.set_params(
            CellStrainParams(
                axes="XZ",
                x_range=(1.0, 1.0, 1.0),
                y_range=(2.0, 2.0, 1.0),
                z_range=(3.0, 3.0, 1.0),
            )
        )
        self.assertEqual(card.custom_axes_edit.text(), "ac")
        self.assertFalse(card.strain_x_field.isHidden())
        self.assertTrue(card.strain_y_field.isHidden())
        self.assertFalse(card.strain_z_field.isHidden())
        self.assertEqual(card.get_params().axes, "XZ")
        self.assertIn("axes ac", card.get_summary_text())

    def test_cell_strain_rejects_duplicate_axes_and_cell_collapse(self):
        operation = CellStrainOperation()
        with self.assertRaisesRegex(ValueError, "unique lattice axes"):
            operation.sampling_summary(CellStrainParams(axes="XX"))
        with self.assertRaisesRegex(ValueError, "greater than -100%"):
            operation.sampling_summary(
                CellStrainParams(x_range=(-100.0, -100.0, 1.0))
            )
        summary = operation.sampling_summary(
            CellStrainParams(
                axes="isotropic",
                x_range=(1.0, 1.0, 1.0),
                y_range=(0.0, 1.0, 0.0),
                z_range=(0.0, 1.0, 0.0),
            )
        )
        self.assertEqual(summary["outputs_per_input"], 1)

    def test_shear_matrix_card(self):
        card = ShearMatrixCard()
        structure = self.structure.copy()
        card.set_params(
            ShearMatrixParams(
                xy_range=(1.0, 1.0, 1.0),
                yz_range=(0.0, 0.0, 1.0),
                xz_range=(0.0, 0.0, 1.0),
                symmetric=False,
                identify_organic=True,
            )
        )

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
        self.assertIn(
            "Shr(xy=1%,yz=0%,xz=0%,mode=simple)",
            results[0].info.get("Config_type", ""),
        )

    def test_shear_matrix_summary_reports_cartesian_product(self):
        summary = ShearMatrixOperation.sampling_summary(ShearMatrixParams())

        self.assertEqual(summary["xy_points"], 3)
        self.assertEqual(summary["yz_points"], 3)
        self.assertEqual(summary["xz_points"], 3)
        self.assertEqual(summary["outputs_per_input"], 27)

        card = ShearMatrixCard()
        card.set_preview_input_count(2)
        self.assertIn("54", card.get_guidance_text())

    def test_shear_matrix_symmetric_branch_sets_both_tensor_halves(self):
        structure = self.structure.copy()
        structure.set_cell(
            np.array(
                [
                    [5.0, 0.2, 0.1],
                    [0.4, 5.5, 0.3],
                    [0.2, 0.5, 6.0],
                ]
            ),
            scale_atoms=True,
        )
        params = ShearMatrixParams(
            xy_range=(10.0, 10.0, 1.0),
            yz_range=(-5.0, -5.0, 1.0),
            xz_range=(2.0, 2.0, 1.0),
            symmetric=True,
        )
        result = ShearMatrixOperation().run_structure(structure, params)[0]
        shear = np.array(
            [
                [1.0, 0.10, 0.02],
                [0.10, 1.0, -0.05],
                [0.02, -0.05, 1.0],
            ]
        )

        np.testing.assert_allclose(
            result.cell.array,
            structure.cell.array @ shear,
            atol=1e-12,
        )
        self.assertIn("mode=symmetric", result.info.get("Config_type", ""))

    def test_shear_matrix_zero_is_exact_noop_and_preserves_magnetic_arrays(self):
        structure = self.structure.copy()
        structure.set_cell(
            np.array(
                [
                    [4.8, 1.1, 0.4],
                    [-0.7, 5.2, 0.8],
                    [0.3, -0.6, 5.9],
                ]
            ),
            scale_atoms=True,
        )
        spin = np.arange(len(structure) * 3, dtype=float).reshape(-1, 3) + 0.25
        initial_magmoms = np.flip(spin, axis=1).copy()
        structure.new_array("spin", spin)
        structure.set_initial_magnetic_moments(initial_magmoms)
        original_cell = structure.cell.array.copy()
        original_positions = structure.positions.copy()

        no_op = ShearMatrixOperation().run_structure(
            structure,
            ShearMatrixParams(
                xy_range=(0.0, 0.0, 1.0),
                yz_range=(0.0, 0.0, 1.0),
                xz_range=(0.0, 0.0, 1.0),
            ),
        )[0]

        np.testing.assert_array_equal(no_op.cell.array, original_cell)
        np.testing.assert_array_equal(no_op.positions, original_positions)
        np.testing.assert_array_equal(no_op.arrays["spin"], spin)
        np.testing.assert_array_equal(no_op.arrays["initial_magmoms"], initial_magmoms)
        self.assertIn(
            "Shr(xy=0%,yz=0%,xz=0%,mode=symmetric)",
            no_op.info.get("Config_type", ""),
        )

        strained = ShearMatrixOperation().run_structure(
            structure,
            ShearMatrixParams(
                xy_range=(2.0, 2.0, 1.0),
                yz_range=(-1.0, -1.0, 1.0),
                xz_range=(0.5, 0.5, 1.0),
                symmetric=False,
            ),
        )[0]
        np.testing.assert_allclose(
            strained.get_scaled_positions(),
            structure.get_scaled_positions(),
            atol=1e-12,
        )
        np.testing.assert_array_equal(strained.arrays["spin"], spin)
        np.testing.assert_array_equal(
            strained.arrays["initial_magmoms"], initial_magmoms
        )

    def test_shear_matrix_rejects_singular_and_left_handed_outputs(self):
        cases = (
            (100.0, 0.0, 0.0),
            (100.0, 100.0, -100.0),
        )
        for xy, yz, xz in cases:
            with self.subTest(xy=xy, yz=yz, xz=xz):
                with self.assertRaisesRegex(ValueError, "invalid or singular cell"):
                    ShearMatrixOperation().run_structure(
                        self.structure,
                        ShearMatrixParams(
                            xy_range=(xy, xy, 1.0),
                            yz_range=(yz, yz, 1.0),
                            xz_range=(xz, xz, 1.0),
                            symmetric=True,
                        ),
                    )

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
        self.assertIn(
            "Ang(alpha=1,beta=0,gamma=0)",
            results[0].info.get("Config_type", ""),
        )

    def test_shear_angle_summary_reports_cartesian_product(self):
        summary = ShearAngleOperation.sampling_summary(ShearAngleParams())

        self.assertEqual(summary["alpha_points"], 3)
        self.assertEqual(summary["beta_points"], 3)
        self.assertEqual(summary["gamma_points"], 3)
        self.assertEqual(summary["outputs_per_input"], 27)

        card = ShearAngleCard()
        card.set_preview_input_count(2)
        self.assertIn("54", card.get_guidance_text())

    def test_scaling_and_shear_cards_roundtrip_nondefault_params(self):
        cases = [
            (
                CellScalingCard,
                CellScalingParams(
                    engine_type=0,
                    max_scaling=0.075,
                    max_num=7,
                    perturb_angle=False,
                    identify_organic=True,
                    use_seed=True,
                    seed=53,
                ),
            ),
            (
                ShearMatrixCard,
                ShearMatrixParams(
                    xy_range=(-3.0, 4.0, 0.5),
                    yz_range=(-2.0, 2.0, 0.25),
                    xz_range=(1.0, 3.0, 0.5),
                    symmetric=False,
                    identify_organic=True,
                ),
            ),
            (
                ShearAngleCard,
                ShearAngleParams(
                    alpha_range=(-1.5, 2.5, 0.5),
                    beta_range=(-2.0, 3.0, 1.0),
                    gamma_range=(0.5, 2.0, 0.25),
                    identify_organic=True,
                ),
            ),
        ]
        for card_cls, params in cases:
            with self.subTest(card=card_cls.__name__):
                card = card_cls()
                card.set_params(params)
                restored = card_cls()
                restored.from_dict(card.to_dict())
                self.assertEqual(restored.get_params(), params)

    def test_shear_angle_operation_matches_ase_cellpar_for_skewed_cell(self):
        structure = self.structure.copy()
        structure.set_cell(
            np.array(
                [
                    [5.4, 0.2, 0.1],
                    [0.4, 5.6, 0.3],
                    [0.2, 0.5, 5.7],
                ]
            ),
            scale_atoms=True,
        )
        params = ShearAngleParams(
            alpha_range=(1.0, 1.0, 1.0),
            beta_range=(-1.0, -1.0, 1.0),
            gamma_range=(0.5, 0.5, 1.0),
        )

        result = ShearAngleOperation().run_structure(structure.copy(), params)[0]
        cellpar = cell_to_cellpar(structure.get_cell())
        reference_cell = np.asarray(structure.get_cell(), dtype=float)
        expected_cell = cellpar_to_cell(
            [*cellpar[:3], *(cellpar[3:] + np.array([1.0, -1.0, 0.5]))],
            ab_normal=np.cross(reference_cell[0], reference_cell[1]),
            a_direction=reference_cell[0],
        )
        expected = structure.copy()
        expected.set_cell(expected_cell, scale_atoms=True)

        np.testing.assert_allclose(result.cell.array, expected.cell.array, atol=1e-12)
        np.testing.assert_allclose(result.positions, expected.positions, atol=1e-12)

    def test_shear_angle_preserves_spin_arrays_in_input_global_frame(self):
        structure = self.structure.copy()
        structure.set_cell(
            np.array(
                [
                    [4.8, 1.1, 0.4],
                    [-0.7, 5.2, 0.8],
                    [0.3, -0.6, 5.9],
                ]
            ),
            scale_atoms=True,
        )
        spin = np.arange(len(structure) * 3, dtype=float).reshape(-1, 3) + 0.25
        initial_magmoms = np.flip(spin, axis=1).copy()
        structure.new_array("spin", spin)
        structure.set_initial_magnetic_moments(initial_magmoms)
        original_cell = structure.cell.array.copy()
        original_positions = structure.positions.copy()

        no_op = ShearAngleOperation().run_structure(
            structure,
            ShearAngleParams(
                alpha_range=(0.0, 0.0, 1.0),
                beta_range=(0.0, 0.0, 1.0),
                gamma_range=(0.0, 0.0, 1.0),
            ),
        )[0]
        np.testing.assert_array_equal(no_op.cell.array, original_cell)
        np.testing.assert_array_equal(no_op.positions, original_positions)
        np.testing.assert_array_equal(no_op.arrays["spin"], spin)
        np.testing.assert_array_equal(no_op.arrays["initial_magmoms"], initial_magmoms)

        strained = ShearAngleOperation().run_structure(
            structure,
            ShearAngleParams(
                alpha_range=(1.0, 1.0, 1.0),
                beta_range=(-0.5, -0.5, 1.0),
                gamma_range=(0.5, 0.5, 1.0),
            ),
        )[0]
        np.testing.assert_allclose(
            strained.get_scaled_positions(),
            structure.get_scaled_positions(),
            atol=1e-12,
        )
        np.testing.assert_array_equal(strained.arrays["spin"], spin)
        np.testing.assert_array_equal(
            strained.arrays["initial_magmoms"], initial_magmoms
        )

    def test_shear_angle_rejects_invalid_final_angles(self):
        structure = self.structure.copy()
        structure.set_cell(cellpar_to_cell([5.0, 5.0, 5.0, 90.0, 90.0, 10.0]))

        with self.assertRaisesRegex(ValueError, "invalid or singular cell"):
            ShearAngleOperation().run_structure(
                structure,
                ShearAngleParams(
                    alpha_range=(0.0, 0.0, 1.0),
                    beta_range=(0.0, 0.0, 1.0),
                    gamma_range=(-20.0, -20.0, 1.0),
                ),
            )

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

        skewed = structure.copy()
        skewed.set_cell(
            np.array(
                [
                    [5.4, 0.2, 0.1],
                    [0.3, 5.5, 0.2],
                    [0.1, 0.4, 5.6],
                ]
            ),
            scale_atoms=True,
        )
        shifted_positions = skewed.positions + np.array([6.0, -5.5, 5.8])
        expected = skewed.copy()
        expected.set_positions(shifted_positions)
        expected.wrap()
        wrapped = VibrationModePerturbOperation.wrapped_positions(skewed, shifted_positions)
        np.testing.assert_allclose(wrapped, expected.positions, atol=1e-12)

        restored = VibrationModePerturbCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_vibration_mode_vector_columns_survive_extxyz_and_run(self):
        structure = Atoms(
            "Si2",
            positions=[[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        )
        structure.new_array(
            "vibration_mode_0",
            np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
        )
        structure.new_array("vibration_frequency_0", np.array([100.0, 100.0]))

        buffer = io.StringIO()
        ase_write(buffer, structure, format="extxyz")
        extxyz = buffer.getvalue()
        self.assertIn("vibration_mode_0:R:3", extxyz)
        restored = ase_read(io.StringIO(extxyz), format="extxyz")

        results = VibrationModePerturbOperation().run_structure(
            restored,
            VibrationModePerturbParams(
                amplitude=0.05,
                modes_per_sample=1,
                min_frequency=1.0,
                max_num=2,
                use_seed=True,
                seed=5,
            ),
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(
            all(not np.allclose(atoms.positions, restored.positions) for atoms in results)
        )

    def test_vibration_mode_distribution_and_frequency_scaling_contract(self):
        structure = Atoms(
            "H",
            positions=[[2.0, 2.0, 2.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=False,
        )
        structure.new_array("vibration_mode_0_x", np.array([1.0]))
        structure.new_array("vibration_mode_0_y", np.array([0.0]))
        structure.new_array("vibration_mode_0_z", np.array([0.0]))
        structure.new_array("vibration_frequency_0", np.array([100.0]))
        operation = VibrationModePerturbOperation()

        for distribution in (0, 1):
            with self.subTest(distribution=distribution):
                params = VibrationModePerturbParams(
                    distribution=distribution,
                    amplitude=0.2,
                    modes_per_sample=1,
                    min_frequency=0.0,
                    max_num=3,
                    scale_by_frequency=False,
                    exclude_near_zero=False,
                    use_seed=True,
                    seed=17,
                )
                first = operation.run_structure(structure, params)
                repeated = operation.run_structure(structure, params)
                self.assertEqual(len(first), 3)
                for left, right in zip(first, repeated):
                    np.testing.assert_allclose(left.positions, right.positions)

        unscaled = operation.run_structure(
            structure,
            VibrationModePerturbParams(
                distribution=1,
                amplitude=0.2,
                modes_per_sample=1,
                min_frequency=0.0,
                max_num=1,
                scale_by_frequency=False,
                exclude_near_zero=False,
                use_seed=True,
                seed=7,
            ),
        )[0]
        scaled = operation.run_structure(
            structure,
            VibrationModePerturbParams(
                distribution=1,
                amplitude=0.2,
                modes_per_sample=1,
                min_frequency=0.0,
                max_num=1,
                scale_by_frequency=True,
                exclude_near_zero=False,
                use_seed=True,
                seed=7,
            ),
        )[0]
        unscaled_delta = unscaled.positions - structure.positions
        scaled_delta = scaled.positions - structure.positions
        np.testing.assert_allclose(scaled_delta, unscaled_delta / 10.0, atol=1e-12)

    def test_vibration_mode_zero_filter_controls_extraction_threshold(self):
        structure = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        frequencies = np.array([100.0])
        modes = np.array([[[1.0, 0.0, 0.0]]])

        with patch(
            "NepTrainKit.core.cards.structure.get_vibration_modes",
            return_value=(frequencies, modes),
        ) as get_modes:
            VibrationModePerturbOperation().run_structure(
                structure,
                VibrationModePerturbParams(
                    modes_per_sample=1,
                    min_frequency=12.5,
                    exclude_near_zero=True,
                    max_num=1,
                    use_seed=True,
                    seed=2,
                ),
            )
            self.assertEqual(get_modes.call_args.kwargs["min_frequency"], 12.5)

            VibrationModePerturbOperation().run_structure(
                structure,
                VibrationModePerturbParams(
                    modes_per_sample=1,
                    min_frequency=12.5,
                    exclude_near_zero=False,
                    max_num=1,
                    use_seed=True,
                    seed=2,
                ),
            )
            self.assertEqual(get_modes.call_args.kwargs["min_frequency"], 0.0)

    def test_vibration_mode_perturb_fails_closed_for_unusable_input(self):
        operation = VibrationModePerturbOperation()
        plain = Atoms("H", positions=[[0.0, 0.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "at least one usable mode"):
            operation.run_structure(plain, VibrationModePerturbParams(max_num=1))

        mode_only = plain.copy()
        mode_only.new_array("vibration_mode_0_x", np.array([1.0]))
        mode_only.new_array("vibration_mode_0_y", np.array([0.0]))
        mode_only.new_array("vibration_mode_0_z", np.array([0.0]))
        with self.assertRaisesRegex(ValueError, "Finite frequencies"):
            operation.run_structure(
                mode_only,
                VibrationModePerturbParams(modes_per_sample=1, max_num=1),
            )

        with_frequency = mode_only.copy()
        with_frequency.new_array("vibration_frequency_0", np.array([100.0]))
        with self.assertRaisesRegex(ValueError, "only 1 usable modes"):
            operation.run_structure(
                with_frequency,
                VibrationModePerturbParams(modes_per_sample=2, max_num=1),
            )

        zero_frequency = mode_only.copy()
        zero_frequency.new_array("vibration_frequency_0", np.array([0.0]))
        with self.assertRaisesRegex(ValueError, "non-zero frequencies"):
            operation.run_structure(
                zero_frequency,
                VibrationModePerturbParams(
                    modes_per_sample=1,
                    max_num=1,
                    scale_by_frequency=True,
                    exclude_near_zero=False,
                ),
            )

    def test_vibration_mode_perturb_records_provenance_and_uses_structure_seed(self):
        operation = VibrationModePerturbOperation()

        def make_structure(x: float) -> Atoms:
            atoms = Atoms("H", positions=[[x, 0.0, 0.0]], pbc=False)
            for mode_index, vector in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
                for axis, value in zip("xyz", vector):
                    atoms.new_array(
                        f"vibration_mode_{mode_index}_{axis}", np.array([value])
                    )
                atoms.new_array(
                    f"vibration_frequency_{mode_index}", np.array([100.0 + mode_index])
                )
            return atoms

        params = VibrationModePerturbParams(
            distribution=1,
            amplitude=0.1,
            modes_per_sample=1,
            max_num=2,
            scale_by_frequency=False,
            exclude_near_zero=False,
            use_seed=True,
            seed=42,
        )
        left = make_structure(0.0)
        right = make_structure(0.2)
        left_results = operation.run_structure(left, params)
        repeated = operation.run_structure(left, params)
        right_results = operation.run_structure(right, params)

        for first, second in zip(left_results, repeated):
            np.testing.assert_allclose(first.positions, second.positions)
        left_displacements = np.array([atoms.positions - left.positions for atoms in left_results])
        right_displacements = np.array([atoms.positions - right.positions for atoms in right_results])
        self.assertFalse(np.allclose(left_displacements, right_displacements))

        metadata = json.loads(left_results[0].info["vibration_mode_perturb"])
        self.assertEqual(metadata["candidate_mode_count"], 2)
        self.assertEqual(len(metadata["selected_mode_indices"]), 1)
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["sample_index"], 0)
