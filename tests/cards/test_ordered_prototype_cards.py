"""Ordered prototype construction: L1u2082/Cu3Au geometry and sublattice stoichiometry."""

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


class TestOrderedPrototypeCards(BaseCardTest):
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

    def test_ordered_prototype_default_is_real_cu3au_l12(self):
        params = OrderedAlloyPrototypeParams()
        output = OrderedAlloyPrototypeOperation().generate(params)[0]

        self.assertEqual(params.a_range, (3.75, 3.75, 0.1))
        self.assertEqual(params.sublattice_elements, "A:Cu,B:Au")
        self.assertEqual(output.get_chemical_symbols().count("Cu"), 3)
        self.assertEqual(output.get_chemical_symbols().count("Au"), 1)
        self.assertNotIn("X", output.get_chemical_symbols())

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

