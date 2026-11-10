"""Conditional replace card."""

import json
import warnings
from unittest.mock import patch

from .card_test_base import *

from ase.calculators.singlepoint import SinglePointCalculator
from NepTrainKit.core.cards.alloy import sample_dopants
from NepTrainKit.core.cards.errors import CardOperationError


class TestConditionalReplaceCards(BaseCardTest):
    def test_conditional_replace_operation_is_ui_independent(self):
        base = Atoms(
            "Si6",
            positions=np.column_stack(
                [np.zeros(6), np.zeros(6), np.arange(6, dtype=float)]
            ),
            cell=[8.0, 8.0, 8.0],
            pbc=True,
        )
        result = ConditionalReplaceOperation().run_structure(
            base,
            ConditionalReplaceParams(
                target="Si",
                replacements="Ge:1",
                condition="z>=3",
                seed=1,
                mode=1,
            ),
        )[0]

        self.assertEqual(
            result.get_chemical_symbols(),
            ["Si", "Si", "Si", "Ge", "Ge", "Ge"],
        )
        np.testing.assert_allclose(result.positions, base.positions)
        np.testing.assert_allclose(result.cell.array, base.cell.array)
        np.testing.assert_array_equal(result.pbc, base.pbc)
        self.assertIn("Repl(Si->Ge)", result.info.get("Config_type", ""))

    def test_conditional_replace_allocation_modes_and_seed(self):
        base = Atoms(
            "Si20",
            positions=np.column_stack(
                [np.zeros(20), np.zeros(20), np.arange(20, dtype=float)]
            ),
            cell=[10.0, 10.0, 22.0],
            pbc=True,
        )
        operation = ConditionalReplaceOperation()
        random_params = ConditionalReplaceParams(
            target="Si",
            replacements="Ge:0.5,C:0.5",
            condition="all",
            seed=7,
            mode=0,
        )
        random_a = operation.run_structure(base, random_params)[0]
        random_b = operation.run_structure(base, random_params)[0]
        exact = operation.run_structure(
            base,
            ConditionalReplaceParams(
                target="Si",
                replacements="Ge:0.5,C:0.5",
                condition="all",
                seed=7,
                mode=1,
            ),
        )[0]

        self.assertEqual(
            random_a.get_chemical_symbols(),
            random_b.get_chemical_symbols(),
        )
        self.assertNotIn("Si", random_a.get_chemical_symbols())
        self.assertEqual(random_a.get_chemical_symbols().count("Ge"), 9)
        self.assertEqual(random_a.get_chemical_symbols().count("C"), 11)
        self.assertEqual(exact.get_chemical_symbols().count("Ge"), 10)
        self.assertEqual(exact.get_chemical_symbols().count("C"), 10)

    def test_conditional_replace_card_roundtrip(self):
        card = ConditionalReplaceCard()
        self.assertEqual(card.condition_edit.text(), "all")
        self.assertEqual(card.target_edit.text(), "")
        self.assertEqual(card.replacements_edit.text(), "")
        self.assertFalse(card.seed_checkbox.isChecked())
        self.assertEqual(card.get_params().seed, 0)
        card.target_edit.setText("Si")
        card.replacements_edit.setText("Ge:0.5,C:0.5")
        card.condition_edit.setText("z>=0")
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([9])
        card.mode_combo.setCurrentIndex(card.mode_combo.findData(1))

        restored = ConditionalReplaceCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())
        self.assertTrue(restored.seed_checkbox.isChecked())
        self.assertEqual(restored.mode_combo.currentData(), 1)

        legacy = ConditionalReplaceCard()
        legacy.from_dict(
            {
                "class": "ConditionalReplaceCard",
                "check_state": True,
                "target": "Si",
                "new_atoms": "Ge,C",
                "ratios": "0.5,0.5",
                "condition": "",
                "seed": [0],
                "mode": 1,
            }
        )
        self.assertEqual(legacy.get_params().condition, "all")
        self.assertEqual(legacy.get_params().replacements, "Ge:0.5,C:0.5")
        self.assertFalse(legacy.seed_checkbox.isChecked())
        self.assertEqual(legacy.mode_combo.currentData(), 1)

    def test_conditional_replace_exact_preview_and_magnetic_array_contract(self):
        base = Atoms(
            "O10",
            positions=np.column_stack(
                [np.linspace(0.0, 1.8, 10), np.zeros(10), np.arange(10.0)]
            ),
            cell=[[8.0, 0.0, 0.0], [1.2, 9.0, 0.0], [0.4, 0.6, 12.0]],
            pbc=[True, False, True],
        )
        spin = np.arange(30.0).reshape(10, 3)
        magmoms = np.flip(spin, axis=1).copy()
        base.new_array("spin", spin)
        base.set_initial_magnetic_moments(magmoms)
        base.info["Config_type"] = "surface"
        params = ConditionalReplaceParams(
            target="O",
            replacements="F:0.7,Cl:0.3",
            condition="z>=2 and z<=7",
            seed=13,
            mode=1,
        )

        operation = ConditionalReplaceOperation()
        summary = operation.selection_summary(params, base)
        self.assertEqual(summary["target_sites"], 10)
        self.assertEqual(summary["matched_sites"], 6)
        self.assertEqual(summary["replacement_counts"], (("F", 4), ("Cl", 2)))

        result = operation.run_structure(base, params)[0]
        self.assertEqual(result.get_chemical_symbols().count("O"), 4)
        self.assertEqual(result.get_chemical_symbols().count("F"), 4)
        self.assertEqual(result.get_chemical_symbols().count("Cl"), 2)
        np.testing.assert_array_equal(result.positions, base.positions)
        np.testing.assert_array_equal(result.cell.array, base.cell.array)
        np.testing.assert_array_equal(result.pbc, base.pbc)
        np.testing.assert_array_equal(result.arrays["spin"], spin)
        np.testing.assert_array_equal(result.arrays["initial_magmoms"], magmoms)
        self.assertEqual(result.info["Config_type"], "surface|Repl(O->F,Cl)")

        card = ConditionalReplaceCard()
        card.set_params(params)
        card.set_preview_structure(base)
        card.set_preview_input_count(2)
        self.assertIn("6 matched", card.get_summary_text())
        guidance = card.get_guidance_text()
        self.assertIn("outputs 2", guidance)
        self.assertIn("F:4", guidance)
        self.assertIn("spin", guidance)

    def test_conditional_replace_validation_and_truthful_tags(self):
        base = Atoms(
            "O4",
            positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        )
        operation = ConditionalReplaceOperation()
        invalid_cases = (
            ConditionalReplaceParams(target=""),
            ConditionalReplaceParams(target="Xx", replacements="F:1"),
            ConditionalReplaceParams(target="O", replacements=""),
            ConditionalReplaceParams(target="O", replacements="Xx:1"),
            ConditionalReplaceParams(target="O", replacements="F:-1,N:1"),
            ConditionalReplaceParams(target="O", replacements="F:0,N:0"),
            ConditionalReplaceParams(target="O", replacements="F:1,F:2"),
            ConditionalReplaceParams(target="O", replacements="O:1"),
            ConditionalReplaceParams(target="O", replacements="O:0,F:1"),
            ConditionalReplaceParams(target="O", replacements="F:1", mode=7),
            ConditionalReplaceParams(target="O", replacements="F:1", seed=-1),
            ConditionalReplaceParams(target="O", replacements="F:1", condition="q>2"),
            ConditionalReplaceParams(target="O", replacements="F:1", condition="x+1"),
            ConditionalReplaceParams(target="O", replacements="F:1", condition='x=="a"'),
            ConditionalReplaceParams(target="O", replacements="F:1", condition="x<1e999"),
            ConditionalReplaceParams(target="Si", replacements="Ge:1"),
            ConditionalReplaceParams(target="O", replacements="F:1", condition="z>99"),
        )
        for params in invalid_cases:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    operation.run_structure(base, params)

        invalid_positions = base.copy()
        invalid_positions.positions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite Cartesian"):
            operation.run_structure(
                invalid_positions,
                ConditionalReplaceParams(target="O", replacements="F:1"),
            )

        result = operation.run_structure(
            base,
            ConditionalReplaceParams(
                target="O",
                replacements="F:1,N:0",
                condition="z>=1 or x>99",
                seed=2,
                mode=1,
            ),
        )[0]
        self.assertEqual(result.get_chemical_symbols(), ["O", "F", "F", "F"])
        self.assertIn("Repl(O->F)", result.info["Config_type"])
        self.assertNotIn("N", result.info["Config_type"])

        for seed in range(20):
            random_result = operation.run_structure(
                base,
                ConditionalReplaceParams(
                    target="O",
                    replacements="F:1,Cl:1",
                    seed=seed,
                    mode=0,
                ),
            )[0]
            self.assertEqual(len(random_result), len(base))
            self.assertNotIn("O", random_result.get_chemical_symbols())

