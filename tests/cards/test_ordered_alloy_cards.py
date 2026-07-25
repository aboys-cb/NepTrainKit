import json
import tempfile
import time
from pathlib import Path

from ase.io import read, write

from NepTrainKit.core.alloy import simplex_sobol_points
from NepTrainKit.i18n import install_translator

from .card_test_base import *


class TestOrderedAlloyCards(BaseCardTest):
    @staticmethod
    def _prototype(prototype: str, rep=(1, 1, 1), max_atoms=128):
        return OrderedAlloyPrototypeOperation().generate(
            OrderedAlloyPrototypeParams(
                prototype=prototype,
                a_range=(3.6, 3.6, 0.1),
                covera=1.2 if prototype == "L10/AB" else 1.633,
                sublattice_elements="A:Cu,B:Au",
                auto_supercell=False,
                rep=rep,
                max_atoms=max_atoms,
                max_outputs=1,
            )
        )[0]

    @staticmethod
    def _metadata(atoms):
        return json.loads(atoms.info["finite_cell_alloy"])

    def test_ordered_prototype_sublattice_stoichiometries(self):
        expected = {
            "A1/fcc": {"A": 4},
            "A2/bcc": {"A": 2},
            "A3/hcp": {"A": 2},
            "L12/A3B": {"A": 3, "B": 1},
            "B2/AB": {"A": 1, "B": 1},
            "L10/AB": {"A": 2, "B": 2},
        }
        for prototype, counts in expected.items():
            with self.subTest(prototype=prototype):
                atoms = self._prototype(prototype)
                actual = {
                    label: int(np.count_nonzero(atoms.arrays["sublattice"] == label))
                    for label in np.unique(atoms.arrays["sublattice"])
                }
                self.assertEqual(actual, counts)
                self.assertTrue(atoms.pbc.all())
                self.assertIn("OrderedProto(", atoms.info["Config_type"])

    def test_l12_32_atoms_preserves_24_to_8_sublattices(self):
        atoms = self._prototype("L12/A3B", rep=(2, 2, 2), max_atoms=32)
        self.assertEqual(len(atoms), 32)
        self.assertEqual(int(np.count_nonzero(atoms.arrays["sublattice"] == "A")), 24)
        self.assertEqual(int(np.count_nonzero(atoms.arrays["sublattice"] == "B")), 8)

    def test_ordered_prototype_enforces_atom_and_output_limits(self):
        with self.assertRaisesRegex(ValueError, "exceeding max_atoms=31"):
            self._prototype("L12/A3B", rep=(2, 2, 2), max_atoms=31)

        outputs = OrderedAlloyPrototypeOperation().generate(
            OrderedAlloyPrototypeParams(
                prototype="B2/AB",
                a_range=(2.8, 3.2, 0.1),
                sublattice_elements="A:Fe,B:Al",
                auto_supercell=True,
                max_atoms=64,
                max_outputs=2,
            )
        )
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(len(atoms) <= 64 for atoms in outputs))

    def test_sublattice_survives_supercell_occupancy_and_extxyz(self):
        primitive = self._prototype("B2/AB", max_atoms=2)
        supercell = SuperCellOperation().run_structure(
            primitive,
            SuperCellParams(mode="scale", super_scale=(2, 2, 2)),
        )[0]
        rules = json.dumps(
            {
                "A": {
                    "elements": ["Fe", "Co"],
                    "mode": "fixed_fraction",
                    "composition": {"Fe": 0.5, "Co": 0.5},
                },
                "B": {
                    "elements": ["Al"],
                    "mode": "count_range",
                    "counts": {"Al": 8},
                },
            }
        )
        occupied = FiniteCellAlloyOccupancyOperation().run_structure(
            supercell,
            FiniteCellAlloyOccupancyParams(site_rules=rules, use_seed=True, seed=5, max_outputs=1),
        )[0]
        np.testing.assert_array_equal(occupied.arrays["sublattice"], supercell.arrays["sublattice"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "occupied.extxyz"
            write(path, occupied, format="extxyz")
            restored = read(path, format="extxyz")
        np.testing.assert_array_equal(restored.arrays["sublattice"], occupied.arrays["sublattice"])
        self.assertEqual(json.loads(restored.info["finite_cell_alloy"]), json.loads(occupied.info["finite_cell_alloy"]))

    def test_single_site_integer_compositions_are_unique_and_sum_to_32(self):
        atoms = self._prototype("A1/fcc", rep=(2, 2, 2), max_atoms=32)
        del atoms.arrays["sublattice"]
        rules = json.dumps(
            {
                "all": {
                    "elements": ["Fe", "Co"],
                    "mode": "count_range",
                    "counts": {"Fe": [0, 32], "Co": [0, 32]},
                }
            }
        )
        params = FiniteCellAlloyOccupancyParams(
            site_rules=rules,
            arrangements_per_composition=1,
            use_seed=True,
            seed=11,
            max_outputs=20,
        )
        outputs = FiniteCellAlloyOccupancyOperation().run_structure(atoms, params)
        composition_ids = []
        count_plans = []
        for output in outputs:
            metadata = self._metadata(output)
            counts = metadata["counts"]["all"]
            self.assertEqual(sum(counts.values()), 32)
            self.assertTrue(np.all(output.arrays["sublattice"] == "all"))
            composition_ids.append(metadata["composition_id"])
            count_plans.append(tuple(sorted(counts.items())))
        self.assertEqual(len(composition_ids), len(set(composition_ids)))
        self.assertEqual(len(count_plans), len(set(count_plans)))

    def test_multisublattice_counts_are_independently_satisfied(self):
        atoms = self._prototype("L12/A3B", rep=(2, 2, 2), max_atoms=32)
        atoms.new_array("group", np.asarray(["even" if index % 2 == 0 else "odd" for index in range(32)], dtype="U8"))
        rules = json.dumps(
            {
                "A": {
                    "elements": ["Fe", "Co"],
                    "mode": "count_range",
                    "counts": {"Fe": [8, 16], "Co": [8, 16]},
                },
                "B": {
                    "elements": ["Al", "Ni"],
                    "mode": "fraction_range",
                    "fractions": {"Al": [0.25, 0.75], "Ni": [0.25, 0.75]},
                },
            }
        )
        outputs = FiniteCellAlloyOccupancyOperation().run_structure(
            atoms,
            FiniteCellAlloyOccupancyParams(site_rules=rules, use_seed=True, seed=2, max_outputs=12),
        )
        for output in outputs:
            metadata = self._metadata(output)
            self.assertEqual(sum(metadata["counts"]["A"].values()), 24)
            self.assertEqual(sum(metadata["counts"]["B"].values()), 8)
            np.testing.assert_array_equal(output.arrays["group"], atoms.arrays["group"])

    def test_fixed_fraction_reports_nearest_integer_instead_of_exact(self):
        atoms = self._prototype("B2/AB", rep=(2, 2, 2), max_atoms=16)
        rules = json.dumps(
            {
                "A": {
                    "elements": ["Fe", "Co", "Ni"],
                    "mode": "fixed_fraction",
                    "composition": {"Fe": 1, "Co": 1, "Ni": 1},
                },
                "B": {
                    "elements": ["Al"],
                    "mode": "fixed_fraction",
                    "composition": {"Al": 1},
                },
            }
        )
        output = FiniteCellAlloyOccupancyOperation().run_structure(
            atoms,
            FiniteCellAlloyOccupancyParams(site_rules=rules, use_seed=True, seed=1, max_outputs=1),
        )[0]
        metadata = self._metadata(output)
        self.assertEqual(metadata["realization"]["A"], "nearest_integer")
        self.assertEqual(sum(metadata["counts"]["A"].values()), 8)
        self.assertAlmostEqual(sum(metadata["fractions"]["A"].values()), 1.0)

    def test_seed_reproducibility_and_arrangement_tracking(self):
        atoms = self._prototype("A1/fcc", rep=(2, 2, 2), max_atoms=32)
        rules = json.dumps(
            {
                "A": {
                    "elements": ["Fe", "Co"],
                    "mode": "fixed_fraction",
                    "composition": {"Fe": 0.5, "Co": 0.5},
                }
            }
        )
        operation = FiniteCellAlloyOccupancyOperation()

        def run(seed):
            return operation.run_structure(
                atoms,
                FiniteCellAlloyOccupancyParams(
                    site_rules=rules,
                    arrangements_per_composition=3,
                    use_seed=True,
                    seed=seed,
                    max_outputs=3,
                ),
            )

        first = run(7)
        repeated = run(7)
        different = run(8)
        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first],
            [atoms.get_chemical_symbols() for atoms in repeated],
        )
        self.assertEqual(
            [self._metadata(atoms)["arrangement_id"] for atoms in first],
            [self._metadata(atoms)["arrangement_id"] for atoms in repeated],
        )
        self.assertNotEqual(first[0].get_chemical_symbols(), different[0].get_chemical_symbols())
        self.assertEqual(self._metadata(first[0])["counts"], self._metadata(different[0])["counts"])

    def test_theoretical_arrangement_limit_and_max_outputs_are_strict(self):
        atoms = self._prototype("A2/bcc", max_atoms=2)
        rules = json.dumps(
            {
                "A": {
                    "elements": ["Fe", "Co"],
                    "mode": "count_range",
                    "counts": {"Fe": 1, "Co": 1},
                }
            }
        )
        operation = FiniteCellAlloyOccupancyOperation()
        all_arrangements = operation.run_structure(
            atoms,
            FiniteCellAlloyOccupancyParams(
                site_rules=rules,
                arrangements_per_composition=10,
                use_seed=True,
                seed=3,
                max_outputs=10,
            ),
        )
        self.assertEqual(len(all_arrangements), 2)
        self.assertEqual(len({tuple(item.get_chemical_symbols()) for item in all_arrangements}), 2)

        limited = operation.run_structure(
            atoms,
            FiniteCellAlloyOccupancyParams(
                site_rules=rules,
                arrangements_per_composition=10,
                use_seed=True,
                seed=3,
                max_outputs=1,
            ),
        )
        self.assertEqual(len(limited), 1)

    def test_estimate_and_invalid_site_rules(self):
        atoms = self._prototype("B2/AB", rep=(2, 2, 2), max_atoms=16)
        rules = json.dumps(
            {
                "A": {
                    "elements": ["Fe", "Co"],
                    "mode": "count_range",
                    "counts": {"Fe": [0, 8], "Co": [0, 8]},
                },
                "B": {
                    "elements": ["Al"],
                    "mode": "count_range",
                    "counts": {"Al": 8},
                },
            }
        )
        operation = FiniteCellAlloyOccupancyOperation()
        estimate = operation.estimate(
            atoms,
            FiniteCellAlloyOccupancyParams(
                site_rules=rules,
                arrangements_per_composition=4,
                max_outputs=12,
            ),
        )
        self.assertEqual(estimate.composition_count, 9)
        self.assertEqual(estimate.arrangements_per_composition, 4)
        self.assertEqual(estimate.estimated_total_outputs, 12)

        with self.assertRaisesRegex(ValueError, "missing rules for B"):
            operation.run_structure(
                atoms,
                FiniteCellAlloyOccupancyParams(
                    site_rules=json.dumps({"A": json.loads(rules)["A"]}),
                    max_outputs=1,
                ),
            )

    def test_cards_roundtrip_all_fields(self):
        prototype = OrderedAlloyPrototypeCard()
        self.assertEqual(prototype.get_params(), OrderedAlloyPrototypeParams())
        prototype.prototype_combo.setCurrentText("L10/AB")
        prototype.a_frame.set_input_value([3.5, 3.7, 0.1])
        prototype.covera_frame.set_input_value([1.18])
        prototype.elements_edit.setText("A:Fe,B:Pt")
        prototype.manual_supercell_button.setChecked(True)
        prototype.max_atoms_frame.set_input_value([64])
        prototype.rep_frame.set_input_value([2, 2, 1])
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
            "Run or load an upstream structure",
            occupancy_restored.estimate_label.text(),
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
        atoms = self._prototype("L12/A3B", rep=(2, 2, 2), max_atoms=32)
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
        self.assertIn("Detected sites: A=24, B=8", card.estimate_label.text())
        self.assertIn("Feasible integer compositions: 25", card.estimate_label.text())
        self.assertIn("Theoretical outputs before limit: 100", card.estimate_label.text())
        self.assertIn("Expected outputs: 12", card.estimate_label.text())
        self.assertIn("Truncated by max_outputs: Yes", card.estimate_label.text())

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

    def test_default_templates_match_their_input_partition(self):
        card = FiniteCellAlloyOccupancyCard()
        plain = self._prototype("A1/fcc", rep=(2, 2, 2), max_atoms=32)
        del plain.arrays["sublattice"]
        card.rules_editor.load_template("all")
        card.set_dataset([plain])
        self.assertFalse(card.rules_editor.validation_errors(card._input_counts))
        self.assertIn("Detected sites: all=32", card.estimate_label.text())

        ordered = self._prototype("B2/AB", rep=(2, 2, 2), max_atoms=16)
        card.rules_editor.load_template("ab")
        card.set_dataset([ordered])
        self.assertFalse(card.rules_editor.validation_errors(card._input_counts))
        self.assertIn("A=8, B=8", card.estimate_label.text())

    def test_untouched_rules_auto_match_all_a_and_ab_inputs(self):
        card = FiniteCellAlloyOccupancyCard()
        ordered = self._prototype("L12/A3B", rep=(2, 2, 2), max_atoms=32)
        card.set_dataset([ordered])
        self.assertEqual(set(json.loads(card.get_params().site_rules)), {"A", "B"})
        self.assertIn("A, B", card.auto_match_label.text())

        single_a = self._prototype("A1/fcc", rep=(2, 2, 2), max_atoms=32)
        card.set_dataset([single_a])
        self.assertEqual(set(json.loads(card.get_params().site_rules)), {"A"})
        self.assertIn("site sets: A", card.auto_match_label.text())
        self.assertIn("Detected sites: A=32", card.estimate_label.text())

        plain = single_a.copy()
        del plain.arrays["sublattice"]
        card.set_dataset([plain])
        self.assertEqual(set(json.loads(card.get_params().site_rules)), {"all"})
        self.assertIn("site sets: all", card.auto_match_label.text())
        self.assertIn("Detected sites: all=32", card.estimate_label.text())

    def test_user_owned_rules_are_never_auto_overwritten(self):
        single_a = self._prototype("A1/fcc", rep=(2, 2, 2), max_atoms=32)
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
        single_a = self._prototype("A1/fcc", max_atoms=4)
        card.set_dataset([single_a])
        self.assertEqual(set(json.loads(card.get_params().site_rules)), {"A"})

    def test_prototype_switch_controls_covera_and_visible_sublattices(self):
        card = OrderedAlloyPrototypeCard()
        for prototype, enabled, expected_elements in [
            ("A1/fcc", False, "A:X"),
            ("A2/bcc", False, "A:X"),
            ("A3/hcp", True, "A:X"),
            ("L12/A3B", False, "A:X,B:X"),
            ("B2/AB", False, "A:X,B:X"),
            ("L10/AB", True, "A:X,B:X"),
        ]:
            with self.subTest(prototype=prototype):
                card.prototype_combo.setCurrentIndex(card.prototype_combo.findData(prototype))
                self.assertEqual(card.covera_frame.isEnabled(), enabled)
                self.assertEqual(card.elements_edit.text(), expected_elements)
                if enabled:
                    self.assertEqual(card.covera_label.text(), "c/a")
                else:
                    self.assertEqual(card.covera_label.text(), "c/a (fixed at 1)")
                    self.assertEqual(card.covera_frame.get_input_value(), [1.0])
                required = "A, B" if ",B:" in expected_elements else "A"
                self.assertIn(f"Required sublattices: {required}", card.sublattice_hint_label.text())

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
                    self.assertGreaterEqual(card.rules_editor.site_editors[0].mode_combo.minimumWidth(), 136)
                    self.assertLessEqual(
                        card.rules_editor.site_editors[0].element_rows[0].element_edit.height(),
                        28,
                    )
                    self.assertLessEqual(card.arrangements_frame.maximumWidth(), 220)
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
        card.show()
        self._app.processEvents()
        self.assertLessEqual(card.sizeHint().height(), 520)
        self.assertEqual(
            [editor._expanded for editor in card.rules_editor.site_editors],
            [True, False],
        )
        card.rules_editor.site_editors[1].toggle_expanded()
        self.assertTrue(card.rules_editor.site_editors[1]._expanded)

    def test_operation_performance_smoke_32_64_128_atoms(self):
        operation = FiniteCellAlloyOccupancyOperation()
        for rep, atom_count in [((2, 2, 2), 32), ((2, 2, 4), 64), ((2, 4, 4), 128)]:
            with self.subTest(atom_count=atom_count):
                atoms = self._prototype("A1/fcc", rep=rep, max_atoms=atom_count)
                half = atom_count // 2
                rules = json.dumps(
                    {
                        "A": {
                            "elements": ["Fe", "Co"],
                            "mode": "count_range",
                            "counts": {"Fe": half, "Co": half},
                        }
                    }
                )
                started = time.perf_counter()
                outputs = operation.run_structure(
                    atoms,
                    FiniteCellAlloyOccupancyParams(
                        site_rules=rules,
                        arrangements_per_composition=8,
                        use_seed=True,
                        seed=4,
                        max_outputs=8,
                    ),
                )
                elapsed = time.perf_counter() - started
                self.assertEqual(len(outputs), 8)
                self.assertLess(elapsed, 2.0)

    def test_conditional_replace_accepts_all_comparison_spellings(self):
        point = np.array([0.0, 1.0, 2.0])
        for expression in ["x=0", "x==0", "y>=1", "y<=1", "z!=0"]:
            with self.subTest(expression=expression):
                self.assertTrue(evaluate_condition(expression, point))
        self.assertFalse(evaluate_condition("x!=0", point))
        with self.assertRaisesRegex(ValueError, "Invalid condition expression"):
            evaluate_condition("x===0", point)

    def test_sobol_min_fraction_returns_target_valid_count_or_clear_failure(self):
        points = simplex_sobol_points(4, 25, seed=3, min_fraction=0.1)
        self.assertEqual(len(points), 25)
        self.assertTrue(all(min(point) >= 0.1 - 1e-12 for point in points))
        self.assertTrue(all(abs(sum(point) - 1.0) <= 1e-12 for point in points))

        swept = CompositionSweepOperation().run_structure(
            self.structure,
            CompositionSweepParams(
                elements="Fe,Co,Ni,Cr",
                order="4",
                method="Sobol",
                n_points=25,
                min_fraction=0.1,
                use_seed=True,
                seed=3,
                max_outputs=25,
            ),
        )
        self.assertEqual(len(swept), 25)

        with self.assertRaisesRegex(ValueError, "permits only the equimolar point"):
            simplex_sobol_points(4, 2, seed=3, min_fraction=0.25)
