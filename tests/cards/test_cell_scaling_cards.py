"""Cell scaling card."""

from .card_test_base import *
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.io import read as ase_read, write as ase_write
from unittest.mock import patch
import io
import json
import re
import warnings
from NepTrainKit.core.cards.sampling import derived_structure_seed


class TestCellScalingCards(BaseCardTest):
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
                dispose_card(card, restored)

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

