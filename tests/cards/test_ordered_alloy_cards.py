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


class TestOrderedAlloyCards(BaseCardTest):
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
        atoms = self._prototype("L12/A3B", rep=(2, 2, 2))
        self.assertEqual(len(atoms), 32)
        self.assertEqual(int(np.count_nonzero(atoms.arrays["sublattice"] == "A")), 24)
        self.assertEqual(int(np.count_nonzero(atoms.arrays["sublattice"] == "B")), 8)
        metadata = json.loads(atoms.info["ordered_alloy_prototype"])
        self.assertEqual(metadata["prototype"], "L12")
        self.assertEqual(metadata["sublattice_elements"], {"A": "Cu", "B": "Au"})
        self.assertEqual(metadata["sublattice_counts"], {"A": 24, "B": 8})

    def test_ordered_prototype_limits_only_lattice_scan_outputs(self):
        outputs = OrderedAlloyPrototypeOperation().generate(
            OrderedAlloyPrototypeParams(
                prototype="B2/AB",
                a_range=(2.8, 3.2, 0.1),
                sublattice_elements="A:Fe,B:Al",
                max_outputs=2,
            )
        )
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(len(atoms) == 2 for atoms in outputs))
        self.assertNotIn("rep=", outputs[0].info["Config_type"])

    def test_ordered_prototype_plan_matches_base_cell_and_truncation(self):
        params = OrderedAlloyPrototypeParams(
            prototype="L12/A3B",
            a_range=(3.5, 3.9, 0.1),
            sublattice_elements="A:Cu,B:Au",
            max_outputs=2,
        )
        operation = OrderedAlloyPrototypeOperation()
        plan = operation.plan(params)
        outputs = operation.generate(params)

        self.assertEqual(len(plan.a_values), 5)
        self.assertEqual(plan.atoms_per_output, 4)
        self.assertEqual(plan.sublattice_counts, {"B": 1, "A": 3})
        self.assertEqual(plan.sublattice_elements, {"B": "Au", "A": "Cu"})
        self.assertTrue(plan.truncated)
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(len(atoms) == 4 for atoms in outputs))

    def test_sublattice_survives_supercell_occupancy_and_extxyz(self):
        primitive = self._prototype("B2/AB")
        supercell = SuperCellOperation().run_structure(
            primitive,
            SuperCellParams(mode="scale", super_scale=(2, 2, 2)),
        )[0]
        prototype_metadata = json.loads(supercell.info["ordered_alloy_prototype"])
        self.assertEqual(prototype_metadata["sublattice_counts"], {"A": 8, "B": 8})
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
        self.assertEqual(
            json.loads(occupied.info["ordered_alloy_prototype"]),
            prototype_metadata,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "occupied.extxyz"
            write(path, occupied, format="extxyz")
            restored = read(path, format="extxyz")
        np.testing.assert_array_equal(restored.arrays["sublattice"], occupied.arrays["sublattice"])
        self.assertEqual(json.loads(restored.info["finite_cell_alloy"]), json.loads(occupied.info["finite_cell_alloy"]))
        self.assertEqual(json.loads(restored.info["ordered_alloy_prototype"]), prototype_metadata)

    def test_single_site_integer_compositions_are_unique_and_sum_to_32(self):
        atoms = self._prototype("A1/fcc", rep=(2, 2, 2))
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
        atoms = self._prototype("L12/A3B", rep=(2, 2, 2))
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
        atoms = self._prototype("B2/AB", rep=(2, 2, 2))
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
        atoms = self._prototype("A1/fcc", rep=(2, 2, 2))
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

    def test_seed_validation_and_unseeded_metadata(self):
        atoms = self._prototype("A1/fcc")
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

        with self.assertRaisesRegex(
            ValueError,
            r"Finite-Cell Alloy Occupancy: seed must be >= 0",
        ):
            operation.run_structure(
                atoms,
                FiniteCellAlloyOccupancyParams(
                    site_rules=rules,
                    use_seed=True,
                    seed=-1,
                    max_outputs=1,
                ),
            )

        output = operation.run_structure(
            atoms,
            FiniteCellAlloyOccupancyParams(
                site_rules=rules,
                use_seed=False,
                seed=-1,
                max_outputs=1,
            ),
        )[0]
        self.assertIsNone(self._metadata(output)["seed"])

    def test_truncated_composition_selection_is_seeded_without_locking_sequence(self):
        atoms = self._prototype("A1/fcc", rep=(2, 2, 2))
        rules = json.dumps(
            {
                "A": {
                    "elements": ["Fe", "Co"],
                    "mode": "count_range",
                    "counts": {"Fe": [0, 32], "Co": [0, 32]},
                }
            }
        )
        operation = FiniteCellAlloyOccupancyOperation()

        def selected_counts(seed):
            outputs = operation.run_structure(
                atoms,
                FiniteCellAlloyOccupancyParams(
                    site_rules=rules,
                    arrangements_per_composition=1,
                    use_seed=True,
                    seed=seed,
                    max_outputs=4,
                ),
            )
            return frozenset(
                tuple(sorted(self._metadata(output)["counts"]["A"].items()))
                for output in outputs
            )

        baseline = selected_counts(0)
        self.assertEqual(selected_counts(0), baseline)
        self.assertTrue(
            any(selected_counts(seed) != baseline for seed in range(1, 16)),
            "At least one alternate seed should select a different truncated composition subset.",
        )

    def test_occupancy_preserves_input_and_non_target_structure_data(self):
        atoms = self._prototype("B2/AB", rep=(2, 1, 1))
        atoms.pbc = (True, False, True)
        atoms.new_array("marker", np.arange(len(atoms), dtype=np.int64))
        atoms.info["source_note"] = "keep-me"
        input_symbols = atoms.get_chemical_symbols()
        input_positions = atoms.positions.copy()
        input_cell = atoms.cell.array.copy()
        input_pbc = atoms.pbc.copy()
        input_sublattice = atoms.arrays["sublattice"].copy()
        input_marker = atoms.arrays["marker"].copy()
        rules = json.dumps(
            {
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
        )

        output = FiniteCellAlloyOccupancyOperation().run_structure(
            atoms,
            FiniteCellAlloyOccupancyParams(
                site_rules=rules,
                use_seed=True,
                seed=4,
                max_outputs=1,
            ),
        )[0]

        self.assertEqual(atoms.get_chemical_symbols(), input_symbols)
        self.assertNotIn("finite_cell_alloy", atoms.info)
        np.testing.assert_array_equal(atoms.positions, input_positions)
        np.testing.assert_array_equal(atoms.cell.array, input_cell)
        np.testing.assert_array_equal(atoms.pbc, input_pbc)
        np.testing.assert_array_equal(atoms.arrays["sublattice"], input_sublattice)
        np.testing.assert_array_equal(atoms.arrays["marker"], input_marker)

        np.testing.assert_array_equal(output.positions, input_positions)
        np.testing.assert_array_equal(output.cell.array, input_cell)
        np.testing.assert_array_equal(output.pbc, input_pbc)
        np.testing.assert_array_equal(output.arrays["sublattice"], input_sublattice)
        np.testing.assert_array_equal(output.arrays["marker"], input_marker)
        self.assertEqual(output.info["source_note"], "keep-me")

    def test_theoretical_arrangement_limit_and_max_outputs_are_strict(self):
        atoms = self._prototype("A2/bcc")
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

    def test_placeholder_element_x_is_never_emitted(self):
        atoms = self._prototype("A2/bcc")
        rules = json.dumps(
            {
                "A": {
                    "elements": ["X"],
                    "mode": "fixed_fraction",
                    "composition": {"X": 1.0},
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "placeholder element X"):
            FiniteCellAlloyOccupancyOperation().run_structure(
                atoms,
                FiniteCellAlloyOccupancyParams(site_rules=rules, max_outputs=1),
            )

    def test_output_budget_covers_compositions_before_extra_arrangements(self):
        atoms = self._prototype("A1/fcc")
        rules = json.dumps(
            {
                "A": {
                    "elements": ["Fe", "Co"],
                    "mode": "count_range",
                    "counts": {"Fe": [0, 4], "Co": [0, 4]},
                }
            }
        )
        outputs = FiniteCellAlloyOccupancyOperation().run_structure(
            atoms,
            FiniteCellAlloyOccupancyParams(
                site_rules=rules,
                arrangements_per_composition=3,
                use_seed=True,
                seed=9,
                max_outputs=3,
            ),
        )
        self.assertEqual(len(outputs), 3)
        self.assertEqual(
            len({self._metadata(output)["composition_id"] for output in outputs}),
            3,
        )

    def test_estimate_and_invalid_site_rules(self):
        atoms = self._prototype("B2/AB", rep=(2, 2, 2))
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
        self.assertIn("A=3 (X)", card.output_preview.text())
        self.assertIn("B=1 (X)", card.output_preview.text())
        self.assertIn("not ready for training", card.next_step_tip.text())
        self.assertFalse(hasattr(card, "max_atoms_frame"))
        self.assertFalse(hasattr(card, "rep_frame"))

        card.element_a_edit.setText("Cu")
        card.element_b_edit.setText("Au")
        self._app.processEvents()
        self.assertIn("fixed-stoichiometry", card.next_step_tip.text())
        self.assertIn("Cu/Au", card.get_summary_text())

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

    def test_prototype_switch_controls_covera_and_visible_sublattices(self):
        card = OrderedAlloyPrototypeCard()
        card.show()
        self._app.processEvents()
        for prototype, uses_covera, expected_elements in [
            ("A1/fcc", False, "A:X"),
            ("A2/bcc", False, "A:X"),
            ("A3/hcp", True, "A:X"),
            ("L12/A3B", False, "A:X,B:X"),
            ("B2/AB", False, "A:X,B:X"),
            ("L10/AB", True, "A:X,B:X"),
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

    def test_operation_performance_smoke_32_64_128_atoms(self):
        operation = FiniteCellAlloyOccupancyOperation()
        for rep, atom_count in [((2, 2, 2), 32), ((2, 2, 4), 64), ((2, 4, 4), 128)]:
            with self.subTest(atom_count=atom_count):
                atoms = self._prototype("A1/fcc", rep=rep)
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
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        np.testing.assert_array_equal(
            evaluate_condition("z>=2 and z<=4", points), [True, False]
        )
        np.testing.assert_array_equal(
            evaluate_condition("x>2 or y<2", points), [True, True]
        )
        with self.assertRaisesRegex(ValueError, "position filter syntax"):
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
