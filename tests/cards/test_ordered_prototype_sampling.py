"""Ordered prototype sampling: seeds, truncation, arrangement limits, output budget."""

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


class TestOrderedPrototypeSampling(BaseCardTest):
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

