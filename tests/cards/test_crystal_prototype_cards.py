"""Crystal prototype builder card."""

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


class TestCrystalPrototypeCards(BaseCardTest):
    def test_crystal_prototype_builder_card(self):
        card = CrystalPrototypeBuilderCard()
        set_combo_value(card.structure_combo, "fcc")
        card.element_edit.setText("Cu")
        card.a_frame.set_input_value([3.6, 3.6, 0.1])
        card.max_output_frame.set_input_value([10])

        results = card.create_operation().generate(card.get_params())
        self.assertEqual(len(results), 1)
        self.assertTrue(all(atoms.pbc.all() for atoms in results))
        self.assertEqual(len(results[0]), 4)

    def test_crystal_prototype_builder_operation_is_ui_independent(self):
        params = CrystalPrototypeBuilderParams(
            lattice="bcc",
            element="Fe",
            a_range=(2.9, 2.9, 0.1),
            max_outputs=1,
        )
        results = CrystalPrototypeBuilderOperation().generate(params)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 2)
        self.assertTrue(results[0].pbc.all())
        self.assertIn("Proto(bcc", results[0].info.get("Config_type", ""))
        self.assertNotIn("rep=", results[0].info.get("Config_type", ""))

    def test_crystal_prototype_builder_outputs_only_base_cells(self):
        operation = CrystalPrototypeBuilderOperation()
        for lattice, atoms_per_cell in (("fcc", 4), ("bcc", 2), ("hcp", 2), ("fcc111", 6)):
            with self.subTest(lattice=lattice):
                results = operation.generate(
                    CrystalPrototypeBuilderParams(lattice=lattice, max_outputs=1)
                )
                self.assertEqual(len(results), 1)
                self.assertEqual(len(results[0]), atoms_per_cell)
                self.assertTrue(results[0].pbc.all())

    def test_crystal_prototype_builder_chains_with_super_cell_for_expansion(self):
        base = CrystalPrototypeBuilderOperation().generate(
            CrystalPrototypeBuilderParams(lattice="fcc", max_outputs=1)
        )[0]
        expanded = SuperCellOperation().run_structure(
            base,
            SuperCellParams(mode="scale", super_scale=(2, 2, 2)),
        )[0]

        self.assertEqual(len(base), 4)
        self.assertEqual(len(expanded), 32)
        np.testing.assert_allclose(expanded.cell.lengths(), base.cell.lengths() * 2.0)

    def test_crystal_prototype_builder_card_roundtrip(self):
        card = CrystalPrototypeBuilderCard()
        set_combo_value(card.structure_combo, "hcp")
        card.element_edit.setText("Mg")
        card.a_frame.set_input_value([3.1, 3.3, 0.1])
        card.covera_frame.set_input_value([1.62])
        card.max_output_frame.set_input_value([5])

        restored = CrystalPrototypeBuilderCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())
        self.assertEqual(combo_value(restored.structure_combo), "hcp")

    def test_crystal_prototype_builder_validates_element(self):
        operation = CrystalPrototypeBuilderOperation()
        for element in ("", "FeNi", "X"):
            with self.subTest(element=element), self.assertRaisesRegex(
                ValueError, "one valid chemical element"
            ):
                operation.generate(CrystalPrototypeBuilderParams(element=element))

    def test_crystal_prototype_builder_plan_matches_truncated_output(self):
        params = CrystalPrototypeBuilderParams(
            lattice="fcc",
            element="Cu",
            a_range=(3.5, 3.9, 0.1),
            max_outputs=2,
        )
        operation = CrystalPrototypeBuilderOperation()
        plan = operation.plan(params)
        results = operation.generate(params)

        self.assertEqual(len(plan.a_values), 5)
        self.assertEqual(plan.atoms_per_output, 4)
        self.assertTrue(plan.truncated)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(atoms) == plan.atoms_per_output for atoms in results))

    def test_crystal_prototype_auto_factor_optimization_matches_exhaustive_oracle(self):
        atoms = Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=np.diag([2.0, 3.0, 5.0]),
            pbc=True,
        )
        for max_atoms in range(3, 31):
            max_factor = max_atoms // len(atoms)
            scores = []
            for na, nb, nc in product(range(1, max_factor + 1), repeat=3):
                total = len(atoms) * na * nb * nc
                if total > max_atoms:
                    continue
                lengths = atoms.cell.lengths() * np.array([na, nb, nc])
                scores.append((total, -float(lengths.max() / lengths.min()), (na, nb, nc)))
            expected = max(scores)[2]
            actual = best_supercell_factors_max_atoms(atoms, max_atoms)
            self.assertEqual((actual.na, actual.nb, actual.nc), expected)

    def test_crystal_prototype_builder_ui_shows_base_cell_scope(self):
        card = CrystalPrototypeBuilderCard()
        card.show()
        QApplication.processEvents()
        self.assertFalse(card.covera_field.isVisible())
        self.assertFalse(hasattr(card, "max_atoms_field"))
        self.assertFalse(hasattr(card, "rep_field"))
        self.assertIn("Super Cell", card.expansion_tip.text())

        set_combo_value(card.structure_combo, "hcp")
        card.a_frame.set_input_value([3.1, 3.5, 0.1])
        card.max_output_frame.set_input_value([2])
        QApplication.processEvents()

        self.assertTrue(card.covera_field.isVisible())
        self.assertIn("2 base-cell output", card.output_preview.text())
        self.assertIn("5 points", card.output_preview.text())

        card.window_state = "collapse"
        card.show_setting()
        card.element_edit.clear()
        QApplication.processEvents()
        self.assertIn("parameters need attention", card.summary_label.text())

    def test_crystal_prototype_builder_warns_and_ignores_removed_expansion_settings(self):
        card = CrystalPrototypeBuilderCard()
        legacy = {
            "class": "CrystalPrototypeBuilderCard",
            "check_state": True,
            "params": {
                "lattice": "bcc",
                "element": "Fe",
                "a_range": [2.9, 2.9, 0.1],
                "covera": 1.633,
                "auto_supercell": False,
                "max_atoms": 128,
                "rep": [3, 3, 3],
                "max_outputs": 1,
            },
        }
        with patch(
            "NepTrainKit.ui.views._card.crystal_prototype_builder_card.MessageManager.send_warning_message"
        ) as warning:
            card.from_dict(legacy)

        warning.assert_called_once()
        self.assertIn("Super Cell", warning.call_args.args[0])
        self.assertFalse(card.legacy_expansion_notice.isHidden())
        self.assertIn("Super Cell", card.legacy_expansion_notice.text())
        self.assertEqual(len(card.create_operation().generate(card.get_params())[0]), 2)
        self.assertNotIn("auto_supercell", card.to_dict()["params"])
        self.assertNotIn("max_atoms", card.to_dict()["params"])
        self.assertNotIn("rep", card.to_dict()["params"])

