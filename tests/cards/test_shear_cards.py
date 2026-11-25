"""Shear matrix and shear angle cards."""

from .card_test_base import *
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.io import read as ase_read, write as ase_write
from unittest.mock import patch
import io
import json
import re
import warnings
from NepTrainKit.core.cards.sampling import derived_structure_seed


class TestShearCards(BaseCardTest):
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

