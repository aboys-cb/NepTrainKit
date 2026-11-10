"""Remaining ordered prototype coverage."""

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


class TestOrderedPrototypeMisc(BaseCardTest):
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
