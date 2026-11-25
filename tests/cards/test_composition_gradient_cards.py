"""Composition gradient card."""

import json
import warnings
from unittest.mock import patch

from .card_test_base import *

from ase.calculators.singlepoint import SinglePointCalculator
from NepTrainKit.core.cards.alloy import sample_dopants
from NepTrainKit.core.cards.errors import CardOperationError


class TestCompositionGradientCards(BaseCardTest):
    def test_composition_gradient_operation_and_card_roundtrip(self):
        base = Atoms(
            symbols=["Ni"] * 8,
            positions=[[float(i), 0.0, 0.0] for i in range(8)],
            cell=np.diag([8.0, 2.0, 2.0]),
            pbc=[True, False, False],
        )
        params = CompositionGradientParams(
            elements="Ni,Co",
            start_composition="Ni:1,Co:0",
            end_composition="Ni:0,Co:1",
            axis="a",
            bins=4,
            samples=2,
            use_seed=True,
            seed=3,
        )
        results = CompositionGradientOperation().run_structure(base, params)
        self.assertEqual(len(results), 2)
        self.assertTrue(all("CompGrad(ax=a,b=4" in str(atoms.info.get("Config_type", "")) for atoms in results))
        self.assertEqual(results[0].get_chemical_symbols()[:2], ["Ni", "Ni"])
        self.assertEqual(results[0].get_chemical_symbols()[-2:], ["Co", "Co"])

        card = CompositionGradientCard()
        card.elements_edit.setText("Ni,Co")
        self.assertEqual(card.axis_combo.currentText(), "Lattice a")
        self.assertEqual(card.get_params().axis, "a")
        card.bins_frame.set_input_value([4])
        card.samples_frame.set_input_value([2])
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([3])
        self.assertFalse(card.seed_field.isHidden())
        restored = CompositionGradientCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_composition_gradient_uses_lattice_coordinates_and_loads_legacy_axis(self):
        cell = np.asarray(
            [
                [4.0, 0.0, 0.0],
                [-3.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        )
        scaled = np.asarray(
            [
                [0.1, 0.9, 0.5],
                [0.9, 0.1, 0.5],
                [0.2, 0.8, 0.5],
                [0.8, 0.2, 0.5],
            ]
        )
        base = Atoms("Ni4", cell=cell, pbc=True)
        base.set_scaled_positions(scaled)

        result = CompositionGradientOperation().run_structure(
            base,
            CompositionGradientParams(axis="a", bins=2, use_seed=True, seed=1),
        )[0]

        self.assertEqual(result.get_chemical_symbols(), ["Ni", "Co", "Ni", "Co"])

        legacy = CompositionGradientCard()
        legacy.from_dict(
            {
                "class": "CompositionGradientCard",
                "check_state": True,
                "params": {
                    **params_to_dict(CompositionGradientParams()),
                    "axis": "x",
                },
            }
        )
        self.assertEqual(legacy.get_params().axis, "a")
        self.assertEqual(legacy.axis_combo.currentText(), "Lattice a")

    def test_composition_gradient_target_elements_preserve_other_sublattice(self):
        cell = np.asarray(
            [
                [5.0, 0.0, 0.0],
                [1.5, 4.0, 0.0],
                [0.3, 0.5, 5.0],
            ]
        )
        structure = Atoms(
            "NiONiO",
            scaled_positions=[
                [0.1, 0.1, 0.2],
                [0.2, 0.2, 0.3],
                [0.8, 0.8, 0.7],
                [0.9, 0.9, 0.8],
            ],
            cell=cell,
            pbc=True,
        )
        result = CompositionGradientOperation().run_structure(
            structure,
            CompositionGradientParams(
                elements="Ni,Co",
                start_composition="Ni:1,Co:0",
                end_composition="Ni:0,Co:1",
                axis="c",
                bins=2,
                target_mode="listed",
                target_elements="Ni",
                samples=1,
                use_seed=True,
                seed=5,
            ),
        )[0]

        self.assertEqual(result.get_chemical_symbols(), ["Ni", "O", "Co", "O"])
        np.testing.assert_allclose(result.positions, structure.positions)
        np.testing.assert_allclose(result.cell.array, structure.cell.array)

    def test_composition_gradient_requires_two_groups_and_reports_effective_groups(self):
        structure = Atoms(
            "Ni8",
            scaled_positions=[[index / 8.0, 0.0, 0.0] for index in range(8)],
            cell=np.diag([8.0, 2.0, 2.0]),
            pbc=True,
        )
        operation = CompositionGradientOperation()

        with self.assertRaisesRegex(ValueError, "at least two equal-count groups"):
            operation.run_structure(
                structure,
                CompositionGradientParams(bins=1),
            )

        summary = operation.sampling_summary(
            CompositionGradientParams(bins=20, samples=3), structure
        )
        self.assertEqual(summary["candidate_sites"], 8)
        self.assertEqual(summary["requested_groups"], 20)
        self.assertEqual(summary["effective_groups"], 8)
        self.assertEqual(summary["min_group_size"], 1)
        self.assertEqual(summary["max_group_size"], 1)
        self.assertEqual(summary["outputs_per_input"], 3)

    def test_composition_gradient_explicit_scope_and_magnetic_array_contract(self):
        structure = Atoms(
            "NiONiO",
            scaled_positions=[
                [0.1, 0.1, 0.2],
                [0.2, 0.2, 0.3],
                [0.8, 0.8, 0.7],
                [0.9, 0.9, 0.8],
            ],
            cell=np.diag([4.0, 4.0, 4.0]),
            pbc=True,
        )
        spin = np.arange(12, dtype=float).reshape(4, 3)
        initial_magmoms = np.flip(spin, axis=1).copy()
        structure.new_array("spin", spin)
        structure.set_initial_magnetic_moments(initial_magmoms)

        result = CompositionGradientOperation().run_structure(
            structure,
            CompositionGradientParams(
                bins=2,
                target_mode="listed",
                target_elements="Ni",
                use_seed=True,
                seed=5,
            ),
        )[0]
        self.assertEqual(result.get_chemical_symbols(), ["Ni", "O", "Co", "O"])
        np.testing.assert_array_equal(result.arrays["spin"], spin)
        np.testing.assert_array_equal(
            result.arrays["initial_magmoms"], initial_magmoms
        )

        card = CompositionGradientCard()
        card.set_dataset([structure, structure.copy()])
        card.set_params(
            CompositionGradientParams(
                bins=8,
                target_mode="listed",
                target_elements="Ni",
                samples=3,
            )
        )
        self.assertEqual(card.get_params().target_mode, "listed")
        self.assertFalse(card.target_field.isHidden())
        self.assertIn("Inputs 2", card.get_guidance_text())
        self.assertIn("outputs 6", card.get_guidance_text())
        self.assertIn("Eligible sites 2", card.get_guidance_text())
        self.assertIn("second jump", card.get_guidance_text())
        self.assertIn("magnetic moments", card.get_guidance_text())
        self.assertIn("may repeat", card.get_guidance_text())

        restored = CompositionGradientCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

