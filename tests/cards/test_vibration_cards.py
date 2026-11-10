"""Vibration mode card."""

from .card_test_base import *
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.io import read as ase_read, write as ase_write
from unittest.mock import patch
import io
import json
import re
import warnings
from NepTrainKit.core.cards.sampling import derived_structure_seed


class TestVibrationCards(BaseCardTest):
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
