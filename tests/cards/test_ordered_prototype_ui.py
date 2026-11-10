"""Ordered prototype card UI: preview, JSON roundtrip, sublattice rule editor."""

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


class TestOrderedPrototypeUI(BaseCardTest):
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

    def test_cards_roundtrip_all_fields(self):
        prototype = OrderedAlloyPrototypeCard()
        self.assertEqual(prototype.get_params(), OrderedAlloyPrototypeParams())
        set_combo_value(prototype.prototype_combo, "L10/AB")
        prototype.a_frame.set_input_value([3.5, 3.7, 0.1])
        prototype.covera_frame.set_input_value([1.18])
        prototype.element_a_edit.setText("Fe")
        prototype.element_b_edit.setText("Pt")
        prototype.max_outputs_frame.set_input_value([3])
        prototype_restored = OrderedAlloyPrototypeCard()
        prototype_restored.from_dict(prototype.to_dict())
        self.assertEqual(prototype_restored.get_params(), prototype.get_params())

        occupancy = FiniteCellAlloyOccupancyCard()
        self.assertEqual(occupancy.get_params(), FiniteCellAlloyOccupancyParams())
        self.assertTrue(
            occupancy.apply_rule_json(
                '{"all":{"elements":["Fe"],"mode":"count_range","counts":{"Fe":[2,2]}}}'
            )
        )
        occupancy.arrangements_frame.set_input_value([7])
        occupancy.seed_checkbox.setChecked(True)
        occupancy.seed_frame.set_input_value([19])
        occupancy.max_outputs_frame.set_input_value([23])
        occupancy_restored = FiniteCellAlloyOccupancyCard()
        occupancy_restored.from_dict(occupancy.to_dict())
        self.assertEqual(occupancy_restored.get_params(), occupancy.get_params())
        self.assertIn(
            "Load an upstream structure",
            occupancy_restored.estimate_label.text(),
        )

    def test_ordered_prototype_ui_previews_scope_and_x_next_step(self):
        card = OrderedAlloyPrototypeCard()
        card.show()
        self._app.processEvents()

        self.assertIn("4 sites", card.output_preview.text())
        self.assertIn("A=3 (Cu)", card.output_preview.text())
        self.assertIn("B=1 (Au)", card.output_preview.text())
        self.assertIn("fixed-stoichiometry", card.next_step_tip.text())
        self.assertFalse(hasattr(card, "max_atoms_frame"))
        self.assertFalse(hasattr(card, "rep_frame"))

        card.element_a_edit.setText("X")
        card.element_b_edit.setText("X")
        self._app.processEvents()
        self.assertIn("not ready for training", card.next_step_tip.text())
        self.assertIn("X/X", card.get_summary_text())

        card.element_a_edit.clear()
        self._app.processEvents()
        self.assertIn("Enter one element symbol", card.output_preview.text())
        self.assertIn("parameters need attention", card.get_summary_text())

    def test_ordered_prototype_warns_and_ignores_removed_expansion_settings(self):
        card = OrderedAlloyPrototypeCard()
        legacy = {
            "class": "OrderedAlloyPrototypeCard",
            "check_state": True,
            "params": {
                "prototype": "B2/AB",
                "a_range": [3.0, 3.0, 0.1],
                "covera": 1.0,
                "sublattice_elements": "A:Fe,B:Al",
                "auto_supercell": False,
                "max_atoms": 128,
                "rep": [3, 3, 3],
                "max_outputs": 1,
            },
        }
        with patch(
            "NepTrainKit.ui.views._card.ordered_alloy_prototype_card.MessageManager.send_warning_message"
        ) as warning:
            card.from_dict(legacy)

        warning.assert_called_once()
        self.assertFalse(card.legacy_expansion_notice.isHidden())
        self.assertIn("Super Cell", card.legacy_expansion_notice.text())
        self.assertEqual(len(card.create_operation().generate(card.get_params())[0]), 2)
        self.assertEqual(
            set(card.to_dict()["params"]),
            {"prototype", "a_range", "covera", "sublattice_elements", "max_outputs"},
        )

    def test_visual_rule_editor_generates_each_mode_and_roundtrips(self):
        cases = [
            {
                "all": {
                    "elements": ["Fe", "Co"],
                    "mode": "fixed_fraction",
                    "composition": {"Fe": 0.25, "Co": 0.75},
                }
            },
            {
                "all": {
                    "elements": ["Fe", "Co"],
                    "mode": "fraction_range",
                    "fractions": {"Fe": [0.25, 0.75], "Co": [0.25, 0.75]},
                }
            },
            {
                "all": {
                    "elements": ["Fe", "Co"],
                    "mode": "count_range",
                    "counts": {"Fe": [4, 12], "Co": [4, 12]},
                }
            },
        ]
        for rules in cases:
            with self.subTest(mode=rules["all"]["mode"]):
                card = FiniteCellAlloyOccupancyCard()
                text = json.dumps(rules)
                self.assertTrue(card.apply_rule_json(text))
                self.assertEqual(json.loads(card.get_params().site_rules), rules)
                restored = FiniteCellAlloyOccupancyCard()
                restored.set_params(card.get_params())
                self.assertEqual(json.loads(restored.get_params().site_rules), rules)
                row = restored.rules_editor.site_editors[0].element_rows[0]
                mode = rules["all"]["mode"]
                self.assertEqual(not row.fixed_fraction_spin.isHidden(), mode == "fixed_fraction")
                self.assertEqual(not row.fraction_min_spin.isHidden(), mode == "fraction_range")
                self.assertEqual(not row.count_min_spin.isHidden(), mode == "count_range")
                dispose_card(card, restored)

    def test_site_rule_modes_expose_real_userdata_and_matching_column_headers(self):
        card = FiniteCellAlloyOccupancyCard()
        editor = card.rules_editor.site_editors[0]
        expected = {
            "fixed_fraction": ("Target fraction", "", False),
            "fraction_range": ("Min fraction", "Max fraction", True),
            "count_range": ("Min count", "Max count", True),
        }

        self.assertEqual(
            {editor.mode_combo.itemData(index) for index in range(editor.mode_combo.count())},
            set(expected),
        )
        for mode, (first_header, second_header, second_visible) in expected.items():
            with self.subTest(mode=mode):
                editor.mode_combo.setCurrentIndex(editor.mode_combo.findData(mode))
                self._app.processEvents()
                self.assertEqual(editor.mode(), mode)
                self.assertEqual(editor.value_1_header.text(), first_header)
                self.assertEqual(editor.value_2_header.text(), second_header)
                self.assertEqual(not editor.value_2_header.isHidden(), second_visible)

    def test_switching_fraction_modes_to_count_uses_feasible_fixed_counts(self):
        cases = [
            {
                "all": {
                    "elements": ["Fe", "Co"],
                    "mode": "fixed_fraction",
                    "composition": {"Fe": 0.5, "Co": 0.5},
                }
            },
            {
                "all": {
                    "elements": ["Fe", "Co"],
                    "mode": "fraction_range",
                    "fractions": {"Fe": [0.2, 0.6], "Co": [0.4, 0.8]},
                }
            },
        ]
        for rules in cases:
            with self.subTest(previous_mode=rules["all"]["mode"]):
                card = FiniteCellAlloyOccupancyCard()
                self.assertTrue(card.apply_rule_json(json.dumps(rules)))
                card.rules_editor.set_input_counts({"all": 3})
                editor = card.rules_editor.site_editors[0]
                editor.mode_combo.setCurrentIndex(editor.mode_combo.findData("count_range"))
                self._app.processEvents()

                minima = [row.count_min_spin.value() for row in editor.element_rows]
                maxima = [row.count_max_spin.value() for row in editor.element_rows]
                self.assertEqual(minima, maxima)
                self.assertEqual(sum(minima), 3)
                dispose_card(card)

    def test_hidden_template_controls_do_not_replace_partition_switching(self):
        card = FiniteCellAlloyOccupancyCard()
        editor = card.rules_editor
        editor.set_replacement_confirmation(lambda: True)
        self.assertTrue(editor.template_label.isHidden())
        self.assertTrue(editor.single_template_button.isHidden())
        self.assertTrue(editor.ab_template_button.isHidden())

        editor.partition_mode_combo.setCurrentIndex(
            editor.partition_mode_combo.findData("all")
        )
        self.assertEqual(editor.partition_mode(), "all")
        self.assertEqual(set(editor.to_rules()), {"all"})

        editor.partition_mode_combo.setCurrentIndex(
            editor.partition_mode_combo.findData("sublattices")
        )
        self.assertEqual(editor.partition_mode(), "sublattices")
        self.assertEqual(set(editor.to_rules()), {"A", "B"})

    def test_preview_summary_and_guidance_report_nearest_integer_realization(self):
        atoms = Atoms(
            "Cu3",
            scaled_positions=((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5)),
            cell=np.eye(3) * 3.6,
            pbc=True,
        )
        rules = {
            "all": {
                "elements": ["Fe", "Co"],
                "mode": "fixed_fraction",
                "composition": {"Fe": 0.5, "Co": 0.5},
            }
        }
        card = FiniteCellAlloyOccupancyCard()
        self.assertTrue(card.apply_rule_json(json.dumps(rules)))
        card.set_preview_structure(atoms)

        summary = card.get_summary_text()
        guidance = card.get_guidance_text()
        realization = card.estimate_label.text()
        self.assertIn("feasible compositions 1", summary)
        self.assertIn("up to 1/input", summary)
        self.assertIn("first input structure", guidance)
        self.assertIn("same site partition", guidance)
        self.assertIn("Fixed realization", realization)
        self.assertIn("Fe 2/3", realization)
        self.assertIn("Co 1/3", realization)
        self.assertIn("nearest integer", realization)

    def test_visual_rule_editor_adds_and_removes_site_sets_and_elements(self):
        card = FiniteCellAlloyOccupancyCard()
        editor = card.rules_editor
        editor.load_template("ab")
        site_c = editor.add_site_set("C")
        self.assertEqual([item.label_edit.text() for item in editor.site_editors], ["A", "B", "C"])
        site_c.add_element("Fe", 0.5)
        self.assertEqual([row.element() for row in site_c.element_rows], ["X", "Fe"])
        site_c.remove_element(site_c.element_rows[0])
        self.assertEqual([row.element() for row in site_c.element_rows], ["Fe"])
        editor.remove_site_set(site_c)
        self.assertEqual([item.label_edit.text() for item in editor.site_editors], ["A", "B"])

    def test_advanced_json_roundtrip_and_invalid_json_is_transactional(self):
        rules = {
            "A": {
                "elements": ["Fe", "Co"],
                "mode": "count_range",
                "counts": {"Fe": [8, 16], "Co": [8, 16]},
            },
            "B": {
                "elements": ["Al"],
                "mode": "fixed_fraction",
                "composition": {"Al": 1.0},
            },
        }
        card = FiniteCellAlloyOccupancyCard()
        self.assertTrue(card.apply_rule_json(json.dumps(rules)))
        valid_before = json.loads(card.get_params().site_rules)
        self.assertEqual(valid_before, rules)
        card.advanced_json_edit.setPlainText("{not valid JSON")
        self.assertFalse(card.apply_advanced_json())
        self.assertEqual(json.loads(card.get_params().site_rules), valid_before)
        self.assertEqual(card.advanced_json_edit.toPlainText(), "{not valid JSON")
        self.assertIn("JSON was not applied", card.json_error_label.text())

    def test_dataset_counts_estimate_and_missing_sublattice_rules_are_visible(self):
        atoms = self._prototype("L12/A3B", rep=(2, 2, 2))
        rules = {
            "A": {
                "elements": ["Fe", "Co"],
                "mode": "count_range",
                "counts": {"Fe": [0, 24], "Co": [0, 24]},
            },
            "B": {
                "elements": ["Al"],
                "mode": "fixed_fraction",
                "composition": {"Al": 1.0},
            },
        }
        card = FiniteCellAlloyOccupancyCard()
        self.assertTrue(card.apply_rule_json(json.dumps(rules)))
        card.arrangements_frame.set_input_value([4])
        card.max_outputs_frame.set_input_value([12])
        card.set_dataset([atoms])
        counts = {
            editor.label_edit.text(): editor.site_count_label.text()
            for editor in card.rules_editor.site_editors
        }
        self.assertEqual(counts, {"A": "24 sites", "B": "8 sites"})
        self.assertIn("First input sites: A=24, B=8", card.estimate_label.text())
        self.assertIn("25 feasible integer compositions", card.estimate_label.text())
        self.assertIn("Output upper-bound estimate: 100", card.estimate_label.text())
        self.assertIn("Max outputs per input: 12", card.estimate_label.text())
        self.assertIn("different compositions are covered", card.estimate_label.text())

        self.assertTrue(card.apply_rule_json(json.dumps({"A": rules["A"]})))
        self.assertIn("Missing rules for input site sets: B", card.rules_editor.status_label.text())

        impossible = dict(rules)
        impossible["A"] = {
            "elements": ["Fe", "Co"],
            "mode": "count_range",
            "counts": {"Fe": [0, 1], "Co": [0, 1]},
        }
        self.assertTrue(card.apply_rule_json(json.dumps(impossible)))
        self.assertIn(
            "no integer count solution for 24 sites",
            card.rules_editor.site_editors[0].error_label.text(),
        )

    def test_visual_fixed_fractions_require_sum_one_but_core_keeps_legacy_weights(self):
        atoms = self._prototype("A1/fcc")
        del atoms.arrays["sublattice"]
        rules = {
            "all": {
                "elements": ["Fe", "Co"],
                "mode": "fixed_fraction",
                "composition": {"Fe": 0.8, "Co": 0.8},
            }
        }
        card = FiniteCellAlloyOccupancyCard()
        self.assertTrue(card.apply_rule_json(json.dumps(rules)))
        card.set_dataset([atoms])
        errors = card.rules_editor.validation_errors(card._input_counts)
        self.assertTrue(any("must sum to 1" in error for error in errors))
        with self.assertRaisesRegex(ValueError, "must sum to 1"):
            card.create_operation().run_structure(atoms, card.get_params())

        output = FiniteCellAlloyOccupancyOperation().run_structure(
            atoms,
            FiniteCellAlloyOccupancyParams(
                site_rules=json.dumps(rules),
                max_outputs=1,
            ),
        )[0]
        self.assertEqual(self._metadata(output)["requested"]["all"], {"Co": 0.5, "Fe": 0.5})

        restored = FiniteCellAlloyOccupancyCard()
        restored.set_params(
            FiniteCellAlloyOccupancyParams(
                site_rules=json.dumps(rules),
                max_outputs=1,
            )
        )
        restored.set_dataset([atoms])
        self.assertEqual(
            len(restored.create_operation().run_structure(atoms, restored.get_params())),
            1,
        )

    def test_prototype_switch_controls_covera_and_visible_sublattices(self):
        card = OrderedAlloyPrototypeCard()
        card.show()
        self._app.processEvents()
        for prototype, uses_covera, expected_elements in [
            ("A1/fcc", False, "A:Cu"),
            ("A2/bcc", False, "A:Cu"),
            ("A3/hcp", True, "A:Cu"),
            ("L12/A3B", False, "A:Cu,B:Au"),
            ("B2/AB", False, "A:Cu,B:Au"),
            ("L10/AB", True, "A:Cu,B:Au"),
        ]:
            with self.subTest(prototype=prototype):
                card.prototype_combo.setCurrentIndex(card.prototype_combo.findData(prototype))
                self._app.processEvents()
                self.assertEqual(card.covera_field.isVisible(), uses_covera)
                self.assertEqual(card.get_params().sublattice_elements, expected_elements)
                has_b = ",B:" in expected_elements
                self.assertEqual(card.element_b_field.isVisible(), has_b)
                self.assertIn("Base-cell sites:", card.sublattice_hint_label.text())
                self.assertEqual(card.single_sublattice_tip.isVisible(), not has_b)

    def test_rule_editor_text_layout_and_tab_order_are_explicit(self):
        try:
            for language in ("en_US", "zh_CN"):
                with self.subTest(language=language):
                    install_translator(self._app, language)
                    card = FiniteCellAlloyOccupancyCard()
                    card.resize(1180, 760)
                    card.show()
                    self._app.processEvents()
                    self.assertTrue(card.estimate_label.wordWrap())
                    self.assertTrue(card.rules_editor.status_label.wordWrap())
                    self.assertTrue(all(editor.error_label.wordWrap() for editor in card.rules_editor.site_editors))
                    self.assertEqual(
                        card.rules_editor.site_editors[0].mode_combo.minimumWidth(),
                        0,
                    )
                    self.assertLessEqual(
                        card.rules_editor.site_editors[0].element_rows[0].element_edit.height(),
                        28,
                    )
                    self.assertEqual(card.arrangements_frame.minimumWidth(), 0)
                    text_widgets = [
                        card.rules_editor.single_template_button,
                        card.rules_editor.ab_template_button,
                        card.rules_editor.add_site_button,
                        card.rules_editor.site_editors[0].mode_combo,
                        card.advanced_button,
                    ]
                    for widget in text_widgets:
                        text = widget.currentText() if hasattr(widget, "currentText") else widget.text()
                        self.assertLessEqual(
                            widget.fontMetrics().horizontalAdvance(text) + 24,
                            widget.width(),
                            f"{language} text is clipped: {text}",
                        )
                    self.assertGreater(len(card.tab_order_widgets), 8)
                    self.assertEqual(len(card.tab_order_widgets), len(set(card.tab_order_widgets)))
                    card.close()
        finally:
            install_translator(self._app, "en_US")

    def test_rich_ab_editor_stays_compact_and_collapses_secondary_site_set(self):
        card = FiniteCellAlloyOccupancyCard()
        rules = {
            "A": {
                "elements": ["Fe", "Co", "Ni"],
                "mode": "fraction_range",
                "fractions": {
                    "Fe": [0.25, 0.5],
                    "Co": [0.25, 0.5],
                    "Ni": [0.0, 0.5],
                },
            },
            "B": {
                "elements": ["Al", "Ta"],
                "mode": "count_range",
                "counts": {"Al": [4, 8], "Ta": [0, 4]},
            },
        }
        self.assertTrue(card.apply_rule_json(json.dumps(rules)))
        from NepTrainKit.ui.widgets import MakeWorkflowArea

        area = MakeWorkflowArea()
        area.add_card(card)
        area.resize(1280, 760)
        area.show()
        self._app.processEvents()
        viewport = area.guidance_panel.parameter_scroll.viewport()
        self.assertLessEqual(card.setting_widget.width(), viewport.width())
        self.assertEqual(
            area.guidance_panel.parameter_scroll.horizontalScrollBar().maximum(),
            0,
        )
        self.assertEqual(
            [editor._expanded for editor in card.rules_editor.site_editors],
            [True, False],
        )
        card.rules_editor.site_editors[1].toggle_expanded()
        self.assertTrue(card.rules_editor.site_editors[1]._expanded)
        area.close()

