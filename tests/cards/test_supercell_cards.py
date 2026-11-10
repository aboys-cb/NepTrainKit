"""Supercell card."""

from .card_test_base import *
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.io import read as ase_read, write as ase_write
from unittest.mock import patch
import io
import json
import re
import warnings
from NepTrainKit.core.cards.sampling import derived_structure_seed


class TestSupercellCards(BaseCardTest):
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
            dispose_card(card)

        self.assertEqual(modes, ["scale", "cell", "max_atoms"])

