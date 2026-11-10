"""Remaining lattice coverage."""

from .card_test_base import *
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.io import read as ase_read, write as ase_write
from unittest.mock import patch
import io
import json
import re
import warnings
from NepTrainKit.core.cards.sampling import derived_structure_seed


class TestLatticeMisc(BaseCardTest):
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

