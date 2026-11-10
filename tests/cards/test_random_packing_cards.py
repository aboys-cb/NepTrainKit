"""Random packing card."""

from .card_test_base import *
import threading
import time
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

from ase.geometry import find_mic

from NepTrainKit.core.alloy import best_supercell_factors_max_atoms
from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.ui.views._card.i18n_utils import combo_value, set_combo_value


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QApplication.processEvents()
    return bool(predicate())


class TestRandomPackingCards(BaseCardTest):
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
        with self.assertRaisesRegex(ValueError, "Atom count for Fe"):
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

    def test_random_packing_plan_rejects_budget_empty_and_nonfinite_inputs_before_rng(self):
        operation = RandomPackingOperation()
        cell_template = Atoms(cell=np.diag([8.0, 8.0, 8.0]), pbc=True)

        with self.assertRaisesRegex(ValueError, "at least one atom"):
            operation.plan(cell_template, RandomPackingParams())
        manual = operation.plan(
            cell_template,
            RandomPackingParams(composition="Fe:2,O:1", structures=3),
        )
        self.assertEqual(manual.atoms_per_output, 3)
        self.assertEqual(manual.requested_generated_atoms, 9)

        with patch.object(operation, "pack_one") as pack_one:
            with self.assertRaisesRegex(ValueError, "exceeding the budget"):
                operation.run_structure(
                    Atoms("H10", cell=np.diag([10.0] * 3), pbc=True),
                    RandomPackingParams(structures=2, max_generated_atoms=19),
                )
            pack_one.assert_not_called()

        for invalid in (float("nan"), float("inf"), -1.0):
            with self.assertRaisesRegex(ValueError, "positive finite"):
                operation.plan(
                    Atoms("H", cell=np.diag([5.0] * 3), pbc=True),
                    RandomPackingParams(min_distance=invalid),
                )
        with self.assertRaisesRegex(ValueError, "Unknown chemical element Xx"):
            operation.plan(
                cell_template,
                RandomPackingParams(composition="Xx:1"),
            )

    def test_random_packing_mixed_pbc_nonorthogonal_seed_scan_and_array_contract(self):
        structure = Atoms(
            symbols=["Fe", "Fe", "O"],
            scaled_positions=[[0.0, 0.0, 0.0]] * 3,
            cell=[[7.0, 0.0, 0.0], [1.2, 6.5, 0.0], [0.4, 0.7, 8.0]],
            pbc=[True, False, True],
        )
        structure.info["source"] = "mixed"
        structure.arrays["spin"] = np.ones((3, 3))
        structure.arrays["group"] = np.asarray(["A", "B", "A"])

        operation = RandomPackingOperation()
        for seed in range(20):
            params = RandomPackingParams(
                structures=2,
                min_distance=0.8,
                pair_min_distances="Fe-O:1.0,O-O:0.9",
                use_seed=True,
                seed=seed,
            )
            first = operation.run_structure(structure, params)
            second = operation.run_structure(structure, params)
            self.assertEqual(len(first), 2)
            self.assertTrue(np.allclose(first[0].positions, second[0].positions))
            for atoms in first:
                self.assertTrue(np.allclose(atoms.cell.array, structure.cell.array))
                self.assertTrue(np.array_equal(atoms.pbc, structure.pbc))
                self.assertEqual(atoms.info["source"], "mixed")
                self.assertNotIn("spin", atoms.arrays)
                self.assertNotIn("group", atoms.arrays)

    def test_random_packing_nonstrict_mode_returns_partial_output(self):
        structure = Atoms("He2", cell=np.diag([5.0] * 3), pbc=True)
        successful = Atoms("He2", positions=[[0, 0, 0], [2, 0, 0]], cell=structure.cell, pbc=True)
        operation = RandomPackingOperation()
        with patch.object(
            operation,
            "pack_one",
            side_effect=[ValueError("failed sample"), successful],
        ):
            outputs = operation.run_structure(
                structure,
                RandomPackingParams(structures=2, strict_mode=False, use_seed=True),
            )
        self.assertEqual(len(outputs), 1)
        self.assertIn("RandPack", outputs[0].info.get("Config_type", ""))

    def test_random_packing_ui_disclosure_preview_and_legacy_json(self):
        structure = Atoms("Fe2O", cell=np.diag([7.0] * 3), pbc=True)
        card = RandomPackingCard()
        card.set_preview_structure(structure)
        card.set_preview_input_count(4)
        self.assertTrue(card.advanced_section.isHidden())
        card.advanced_checkbox.setChecked(True)
        self.assertFalse(card.advanced_section.isHidden())
        self.assertFalse(card.pair_distance_edit.isHidden())
        self.assertFalse(card.attempts_frame.isHidden())
        self.assertIn("3 atoms/output", card.get_summary_text())
        self.assertIn("outputs 4", card.get_guidance_text())
        self.assertIn("3 generated atoms", card.preview_label.text())
        set_combo_value(card.composition_mode_combo, "manual")
        self.assertIn("at least one atom", card.preview_label.text())
        manual_empty_data = card.to_dict()
        self.assertEqual(manual_empty_data["params"]["composition_mode"], "manual")
        manual_empty = RandomPackingCard()
        manual_empty.from_dict(manual_empty_data)
        self.assertEqual(combo_value(manual_empty.composition_mode_combo), "manual")
        self.assertEqual(manual_empty.get_params().composition, "")
        set_combo_value(card.composition_mode_combo, "input")

        legacy = RandomPackingCard()
        legacy_data = card.to_dict()
        legacy_data["params"] = {
            "structures": 2,
            "composition": "Fe:2,O:1",
            "min_distance": 1.0,
            "pair_min_distances": "",
            "max_attempts_per_atom": 500,
            "strict_mode": True,
            "use_seed": True,
            "seed": 9,
        }
        legacy.from_dict(legacy_data)
        self.assertEqual(legacy.get_params().max_generated_atoms, 10_000)
        self.assertEqual(combo_value(legacy.composition_mode_combo), "manual")

