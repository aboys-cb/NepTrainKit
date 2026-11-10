"""Ordered prototype rule templates and auto-matching against input partitions."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from ase.io import read, write

from NepTrainKit.core.alloy import simplex_sobol_points
from NepTrainKit.i18n import install_translator
from NepTrainKit.ui.views._card.i18n_utils import set_combo_value

from .card_test_base import *


class TestOrderedPrototypeRules(BaseCardTest):
    @staticmethod
    def _prototype(prototype: str, rep=(1, 1, 1)):
        base = OrderedAlloyPrototypeOperation().generate(
            OrderedAlloyPrototypeParams(
                prototype=prototype,
                a_range=(3.6, 3.6, 0.1),
                covera=1.2 if prototype == "L10/AB" else 1.633,
                sublattice_elements="A:Cu,B:Au",
                max_outputs=1,
            )
        )[0]
        if tuple(rep) == (1, 1, 1):
            return base
        return SuperCellOperation().run_structure(
            base,
            SuperCellParams(mode="scale", super_scale=tuple(rep)),
        )[0]

    @staticmethod
    def _metadata(atoms):
        return json.loads(atoms.info["finite_cell_alloy"])

    def test_default_templates_match_their_input_partition(self):
        card = FiniteCellAlloyOccupancyCard()
        plain = self._prototype("A1/fcc", rep=(2, 2, 2))
        del plain.arrays["sublattice"]
        card.set_dataset([plain])
        self.assertFalse(card.rules_editor.validation_errors(card._input_counts))
        self.assertIn("First input sites: all=32", card.estimate_label.text())

        ordered = self._prototype("B2/AB", rep=(2, 2, 2))
        card.set_dataset([ordered])
        self.assertFalse(card.rules_editor.validation_errors(card._input_counts))
        self.assertIn("A=8, B=8", card.estimate_label.text())

    def test_auto_rules_use_current_elements_and_fractions(self):
        atoms = self._prototype("A1/fcc")
        del atoms.arrays["sublattice"]
        atoms.set_chemical_symbols(["Fe", "Fe", "Fe", "Co"])
        card = FiniteCellAlloyOccupancyCard()
        card.set_dataset([atoms])
        rule = json.loads(card.get_params().site_rules)["all"]
        self.assertEqual(rule["elements"], ["Fe", "Co"])
        self.assertEqual(rule["composition"], {"Fe": 0.75, "Co": 0.25})
        self.assertNotIn("X", card.get_params().site_rules)

    def test_rule_templates_require_confirmation_after_manual_edits(self):
        card = FiniteCellAlloyOccupancyCard()
        editor = card.rules_editor
        editor.site_editors[0].element_rows[0].element_edit.setText("Fe")
        before = editor.to_rules()

        editor.set_replacement_confirmation(lambda: False)
        editor.single_template_button.click()
        self.assertEqual(editor.to_rules(), before)
        all_index = editor.partition_mode_combo.findData("all")
        editor.partition_mode_combo.setCurrentIndex(all_index)
        self.assertEqual(editor.to_rules(), before)
        self.assertEqual(editor.partition_mode(), "sublattices")

        editor.set_replacement_confirmation(lambda: True)
        editor.single_template_button.click()
        self.assertEqual(set(editor.to_rules()), {"all"})

    def test_untouched_rules_auto_match_all_a_and_ab_inputs(self):
        card = FiniteCellAlloyOccupancyCard()
        ordered = self._prototype("L12/A3B", rep=(2, 2, 2))
        card.set_dataset([ordered])
        ordered_rules = json.loads(card.get_params().site_rules)
        self.assertEqual(set(ordered_rules), {"A", "B"})
        self.assertEqual(ordered_rules["A"]["elements"], ["Cu"])
        self.assertEqual(ordered_rules["B"]["elements"], ["Au"])
        self.assertIn("A, B", card.auto_match_label.text())
        card.rules_editor.ab_template_button.click()
        self.assertNotIn("X", card.get_params().site_rules)

        single_a = self._prototype("A1/fcc", rep=(2, 2, 2))
        card.set_dataset([single_a])
        single_rules = json.loads(card.get_params().site_rules)
        self.assertEqual(set(single_rules), {"A"})
        self.assertEqual(single_rules["A"]["elements"], ["Cu"])
        self.assertIn("input: A", card.auto_match_label.text())
        self.assertIn("First input sites: A=32", card.estimate_label.text())

        plain = single_a.copy()
        del plain.arrays["sublattice"]
        card.set_dataset([plain])
        plain_rules = json.loads(card.get_params().site_rules)
        self.assertEqual(set(plain_rules), {"all"})
        self.assertEqual(plain_rules["all"]["elements"], ["Cu"])
        self.assertIn("input: all", card.auto_match_label.text())
        self.assertIn("First input sites: all=32", card.estimate_label.text())

    def test_user_owned_rules_are_never_auto_overwritten(self):
        single_a = self._prototype("A1/fcc", rep=(2, 2, 2))
        rules = {
            "A": {
                "elements": ["Fe"],
                "mode": "fixed_fraction",
                "composition": {"Fe": 1.0},
            },
            "B": {
                "elements": ["Al"],
                "mode": "fixed_fraction",
                "composition": {"Al": 1.0},
            },
        }
        card = FiniteCellAlloyOccupancyCard()
        self.assertTrue(card.apply_rule_json(json.dumps(rules)))
        card.set_dataset([single_a])
        self.assertEqual(json.loads(card.get_params().site_rules), rules)
        self.assertTrue(card.auto_match_label.isHidden())
        self.assertIn(
            "Rules reference site sets absent from the input: B",
            card.rules_editor.status_label.text(),
        )

        restored = FiniteCellAlloyOccupancyCard()
        restored.set_params(FiniteCellAlloyOccupancyParams())
        restored.set_dataset([single_a])
        self.assertEqual(
            set(json.loads(restored.get_params().site_rules)),
            {"A", "B"},
        )
        self.assertTrue(restored.auto_match_label.isHidden())

        template_selected = FiniteCellAlloyOccupancyCard()
        template_selected.rules_editor.single_template_button.click()
        template_selected.set_dataset([single_a])
        self.assertEqual(
            set(json.loads(template_selected.get_params().site_rules)),
            {"all"},
        )
        self.assertTrue(template_selected.auto_match_label.isHidden())

        visually_edited = FiniteCellAlloyOccupancyCard()
        visually_edited.rules_editor.site_editors[0].element_rows[0].element_edit.setText("Fe")
        visually_edited.set_dataset([single_a])
        edited_rules = json.loads(visually_edited.get_params().site_rules)
        self.assertEqual(set(edited_rules), {"A", "B"})
        self.assertEqual(edited_rules["A"]["elements"], ["Fe"])

    def test_collapsing_site_sets_does_not_disable_auto_matching(self):
        card = FiniteCellAlloyOccupancyCard()
        card.rules_editor.site_editors[1].toggle_expanded()
        self.assertTrue(card._rules_are_auto_managed)
        single_a = self._prototype("A1/fcc")
        card.set_dataset([single_a])
        self.assertEqual(set(json.loads(card.get_params().site_rules)), {"A"})

