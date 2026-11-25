"""Random occupancy card."""

import json
import warnings
from unittest.mock import patch

from .card_test_base import *

from ase.calculators.singlepoint import SinglePointCalculator
from NepTrainKit.core.cards.alloy import sample_dopants
from NepTrainKit.core.cards.errors import CardOperationError


class TestRandomOccupancyCards(BaseCardTest):
    def test_composition_sweep_and_random_occupancy_cards(self):
        base = self.structure.copy()
        base.info.setdefault("Config_type", "base")

        sweep = CompositionSweepCard()
        sweep.elements_edit.setText("Co,Ni")
        sweep.order_combo.setCurrentIndex(sweep.order_combo.findData("2"))
        sweep.method_combo.setCurrentText("Grid")
        sweep.step_frame.set_input_value([0.5])
        sweep.include_endpoints_checkbox.setChecked(True)
        sweep.minfrac_frame.set_input_value([0.0])
        sweep.max_output_frame.set_input_value([10])

        swept = sweep.process_structure(base)
        self.assertEqual(len(swept), 3)
        self.assertTrue(
            all("Comp(" in str(atoms.info.get("Config_type", "")) for atoms in swept)
        )

        occ = RandomOccupancyCard()
        occ.source_combo.setCurrentText("Auto (Comp tag)")
        occ.mode_combo.setCurrentText("Exact")
        occ.samples_frame.set_input_value([1])

        occupied = []
        for atoms in swept:
            occupied.extend(occ.process_structure(atoms))
        self.assertEqual(len(occupied), len(swept))
        for atoms in occupied:
            syms = set(atoms.get_chemical_symbols())
            self.assertTrue(syms.issubset({"Co", "Ni"}))

    def test_composition_and_occupancy_operations_are_ui_independent(self):
        base = self.structure.copy()
        base.info.setdefault("Config_type", "base")

        sweep_params = CompositionSweepParams(
            elements="Co,Ni",
            order="2",
            method="Grid",
            step=0.5,
            include_endpoints=True,
            max_outputs=3,
        )
        swept = CompositionSweepOperation().run_structure(base, sweep_params)

        self.assertEqual(len(swept), 3)
        self.assertTrue(all("Comp(" in atoms.info.get("Config_type", "") for atoms in swept))
        for atoms in swept:
            self.assertEqual(atoms.get_chemical_symbols(), base.get_chemical_symbols())
            np.testing.assert_allclose(atoms.positions, base.positions)
            np.testing.assert_allclose(atoms.cell.array, base.cell.array)
            np.testing.assert_array_equal(atoms.pbc, base.pbc)

        card = CompositionSweepCard()
        card.set_params(
            CompositionSweepParams(
                elements="Co,Cr,Ni,Al,Fe",
                order="5,4,3,2",
                method="Sobol",
                n_points=9,
                min_fraction=0.01,
                use_seed=True,
                seed=7,
                max_outputs=20,
                budget_mode="Capacity-weighted",
            )
        )
        restored = CompositionSweepCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())
        self.assertEqual(restored.order_combo.currentData(), "5,4,3,2")
        self.assertEqual(
            restored.budget_mode_combo.currentData(),
            "Capacity-weighted",
        )

        restored.set_params(
            CompositionSweepParams(
                elements="Co,Cr,Ni,Al",
                order="2,3,4",
            )
        )
        self.assertEqual(restored.order_combo.currentData(), "2,3,4")

        occ_params = RandomOccupancyParams(
            source="Auto (Comp tag)",
            mode="Exact",
            samples=1,
            use_seed=True,
            seed=5,
        )
        occupied = RandomOccupancyOperation().run_structure(swept[0], occ_params)

        self.assertEqual(len(occupied), 1)
        self.assertTrue(set(occupied[0].get_chemical_symbols()).issubset({"Co", "Ni"}))

    def test_random_occupancy_group_filter_is_required_and_limits_assignment(self):
        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=[20, 20, 20],
            pbc=True,
        )
        params = RandomOccupancyParams(
            source="Manual",
            manual="Ge",
            group_filter="A",
            use_seed=True,
            seed=1,
        )
        operation = RandomOccupancyOperation()

        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(structure, params)
        self.assertEqual(raised.exception.code, "random_occupancy.missing_group_array")

        structure.new_array("group", np.array(["A", "A", "B", "B"], dtype=object))
        result = operation.run_structure(structure, params)[0]
        self.assertEqual(result.get_chemical_symbols(), ["Ge", "Ge", "Si", "Si"])
        self.assertIn("Occ(E", result.info.get("Config_type", ""))
        metadata = json.loads(result.info["random_occupancy"])
        self.assertEqual(metadata["eligible_sites"], 2)
        self.assertEqual(metadata["groups"], ["A"])
        self.assertEqual(metadata["actual_counts"], {"Ge": 2})

        no_match = structure.copy()
        no_match.arrays["group"][:] = "B"
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(no_match, params)
        self.assertEqual(raised.exception.code, "random_occupancy.no_matching_groups")

    def test_random_occupancy_missing_composition_and_invalid_counts_fail(self):
        operation = RandomOccupancyOperation()
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                self.structure,
                RandomOccupancyParams(source="Manual", manual=""),
            )
        self.assertEqual(raised.exception.code, "random_occupancy.empty_manual")
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                self.structure,
                RandomOccupancyParams(
                    source="Manual",
                    manual="Si",
                    samples=0,
                ),
            )
        self.assertEqual(raised.exception.code, "random_occupancy.invalid_samples")

    def test_random_occupancy_sources_are_strict_and_mutually_exclusive(self):
        structure = Atoms("Si4", positions=np.zeros((4, 3)))
        structure.info["Config_type"] = "base|Comp(Fe=1)|Comp(Co=1)"
        operation = RandomOccupancyOperation()

        auto = operation.run_structure(
            structure,
            RandomOccupancyParams(
                source="Auto (Comp tag)",
                manual="Ni:1",
                use_seed=True,
                seed=1,
            ),
        )[0]
        self.assertEqual(set(auto.get_chemical_symbols()), {"Co"})

        manual = operation.run_structure(
            structure,
            RandomOccupancyParams(
                source="Manual",
                manual="Ni:1",
                use_seed=True,
                seed=1,
            ),
        )[0]
        self.assertEqual(set(manual.get_chemical_symbols()), {"Ni"})

        without_tag = structure.copy()
        without_tag.info["Config_type"] = "base"
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                without_tag,
                RandomOccupancyParams(
                    source="Auto (Comp tag)",
                    manual="Ni:1",
                ),
            )
        self.assertEqual(raised.exception.code, "random_occupancy.missing_comp_tag")

    def test_random_occupancy_summary_and_exact_nondivisible_metadata(self):
        structure = Atoms(
            "Si3",
            scaled_positions=((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5)),
            cell=np.eye(3) * 3.6,
            pbc=(True, False, True),
        )
        structure.new_array("marker", np.arange(3, dtype=np.int64))
        structure.info["source_note"] = "keep-me"
        params = RandomOccupancyParams(
            source="Manual",
            manual="Fe:1,Co:1",
            mode="Exact",
            samples=2,
            use_seed=True,
            seed=7,
        )
        operation = RandomOccupancyOperation()
        summary = operation.sampling_summary(structure, params)

        self.assertEqual(summary["target"], {"Fe": 0.5, "Co": 0.5})
        self.assertEqual(summary["eligible_indices"], (0, 1, 2))
        self.assertEqual(summary["eligible_count"], 3)
        self.assertEqual(summary["fixed_counts"], {"Fe": 2, "Co": 1})
        self.assertEqual(
            summary["fixed_fractions"],
            {"Fe": 2 / 3, "Co": 1 / 3},
        )
        self.assertEqual(summary["outputs_per_input"], 2)

        input_symbols = structure.get_chemical_symbols()
        input_positions = structure.positions.copy()
        input_cell = structure.cell.array.copy()
        input_pbc = structure.pbc.copy()
        input_marker = structure.arrays["marker"].copy()
        outputs = operation.run_structure(structure, params)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(structure.get_chemical_symbols(), input_symbols)
        self.assertNotIn("random_occupancy", structure.info)

        for sample_index, output in enumerate(outputs):
            np.testing.assert_array_equal(output.positions, input_positions)
            np.testing.assert_array_equal(output.cell.array, input_cell)
            np.testing.assert_array_equal(output.pbc, input_pbc)
            np.testing.assert_array_equal(output.arrays["marker"], input_marker)
            self.assertEqual(output.info["source_note"], "keep-me")
            metadata = json.loads(output.info["random_occupancy"])
            self.assertEqual(metadata["target"], {"Co": 0.5, "Fe": 0.5})
            self.assertEqual(metadata["actual_counts"], {"Co": 1, "Fe": 2})
            self.assertEqual(
                metadata["actual_fractions"],
                {"Co": 1 / 3, "Fe": 2 / 3},
            )
            self.assertEqual(metadata["eligible_sites"], 3)
            self.assertEqual(metadata["groups"], [])
            self.assertEqual(metadata["mode"], "Exact")
            self.assertEqual(metadata["sample_index"], sample_index)
            self.assertIsInstance(metadata["seed"], int)

        unseeded = operation.run_structure(
            structure,
            RandomOccupancyParams(
                source="Manual",
                manual="Fe:1,Co:1",
                use_seed=False,
                seed=0,
            ),
        )[0]
        self.assertIsNone(
            json.loads(unseeded.info["random_occupancy"])["seed"]
        )

    def test_random_occupancy_rejects_invalid_inputs_without_mutation(self):
        operation = RandomOccupancyOperation()
        structure = Atoms("Si4", positions=np.zeros((4, 3)))
        structure.info["Config_type"] = "base"
        original_symbols = structure.get_chemical_symbols()
        original_info = dict(structure.info)
        cases = [
            ("random_occupancy.invalid_source", RandomOccupancyParams(source="Auto", manual="Fe:1")),
            ("random_occupancy.invalid_mode", RandomOccupancyParams(source="Manual", manual="Fe:1", mode="Other")),
            ("random_occupancy.invalid_seed", RandomOccupancyParams(source="Manual", manual="Fe:1", use_seed=True, seed=-1)),
            ("random_occupancy.invalid_seed", RandomOccupancyParams(source="Manual", manual="Fe:1", use_seed=False, seed=-1)),
            ("random_occupancy.zero_target", RandomOccupancyParams(source="Manual", manual="Fe:0,Co:0")),
            ("random_occupancy.empty_group_filter", RandomOccupancyParams(source="Manual", manual="Fe:1", group_filter=" , , ")),
            ("random_occupancy.unknown_elements", RandomOccupancyParams(source="Manual", manual="Qq:1")),
            ("random_occupancy.invalid_composition", RandomOccupancyParams(source="Manual", manual="Fe:not-a-number")),
        ]
        for code, params in cases:
            with self.subTest(code=code):
                with self.assertRaises(CardOperationError) as raised:
                    operation.run_structure(structure, params)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(structure.get_chemical_symbols(), original_symbols)
                self.assertEqual(structure.info, original_info)

        empty = Atoms()
        with self.assertRaises(CardOperationError) as raised:
            operation.run_structure(
                empty,
                RandomOccupancyParams(source="Manual", manual="Fe:1"),
            )
        self.assertEqual(raised.exception.code, "random_occupancy.empty_structure")
        self.assertEqual(empty.info, {})

    def test_random_occupancy_clears_species_dependent_reference_data(self):
        structure = Atoms(
            "Si4",
            positions=np.arange(12, dtype=float).reshape(4, 3),
            cell=np.diag([8.0, 7.0, 6.0]),
            pbc=(True, False, True),
        )
        structure.calc = SinglePointCalculator(
            structure,
            energy=-2.0,
            forces=np.ones((4, 3)),
            stress=np.arange(6, dtype=float),
        )
        structure.info.update(
            {
                "energy": -2.0,
                "free_energy": -2.1,
                "stress": [1.0] * 6,
                "virial": [2.0] * 9,
                "dipole": [0.0, 0.0, 1.0],
                "magmom": 3.0,
                "source_note": "keep-me",
            }
        )
        structure.new_array("group", np.array(["A", "A", "B", "B"], dtype="U1"))
        structure.new_array("sublattice", np.array(["A", "A", "B", "B"], dtype="U1"))
        structure.new_array("marker", np.arange(4, dtype=np.int64))
        for name, values in {
            "forces": np.ones((4, 3)),
            "force": np.ones((4, 3)) * 2,
            "energies": np.arange(4, dtype=float),
            "atomic_energy": np.arange(4, dtype=float) + 1,
            "magmoms": np.ones((4, 3)),
            "charges": np.arange(4, dtype=float),
            "bec": np.ones((4, 3, 3)),
            "born_effective_charges": np.ones((4, 3, 3)) * 2,
            "spin": np.ones((4, 3)) * 3,
            "initial_magmoms": np.ones((4, 3)) * 4,
            "initial_charges": np.ones(4) * 5,
        }.items():
            structure.new_array(name, values)

        output = RandomOccupancyOperation().run_structure(
            structure,
            RandomOccupancyParams(
                source="Manual",
                manual="Ge:1",
                mode="Exact",
                samples=1,
                use_seed=True,
                seed=3,
            ),
        )[0]

        self.assertIsNone(output.calc)
        for key in ("energy", "free_energy", "stress", "virial", "dipole", "magmom"):
            self.assertNotIn(key, output.info)
        for key in (
            "forces",
            "force",
            "energies",
            "atomic_energy",
            "magmoms",
            "charges",
            "bec",
            "born_effective_charges",
            "spin",
            "initial_magmoms",
            "initial_charges",
        ):
            self.assertNotIn(key, output.arrays)

        self.assertEqual(output.info["source_note"], "keep-me")
        for key in ("group", "sublattice", "marker"):
            np.testing.assert_array_equal(output.arrays[key], structure.arrays[key])
        np.testing.assert_array_equal(output.positions, structure.positions)
        np.testing.assert_array_equal(output.cell.array, structure.cell.array)
        np.testing.assert_array_equal(output.pbc, structure.pbc)

        self.assertIsNotNone(structure.calc)
        self.assertIn("energy", structure.info)
        self.assertIn("spin", structure.arrays)

    def test_random_occupancy_random_mode_is_seeded_and_respects_sample_contract(self):
        structure = Atoms(
            "Si12",
            positions=np.column_stack(
                [np.arange(12, dtype=float), np.zeros(12), np.zeros(12)]
            ),
            cell=[14.0, 4.0, 4.0],
            pbc=True,
        )
        operation = RandomOccupancyOperation()
        params = RandomOccupancyParams(
            source="Manual",
            manual="Co:0.25,Ni:0.75",
            mode="Random",
            samples=4,
            use_seed=True,
            seed=23,
        )

        first = operation.run_structure(structure, params)
        repeated = operation.run_structure(structure, params)

        self.assertEqual(len(first), 4)
        self.assertEqual(
            [atoms.get_chemical_symbols() for atoms in first],
            [atoms.get_chemical_symbols() for atoms in repeated],
        )
        for atoms in first:
            self.assertEqual(len(atoms), len(structure))
            self.assertTrue(
                set(atoms.get_chemical_symbols()).issubset({"Co", "Ni"})
            )
            metadata = json.loads(atoms.info["random_occupancy"])
            self.assertEqual(sum(metadata["actual_counts"].values()), 12)
            self.assertEqual(metadata["mode"], "Random")

        observed_counts = set()
        for seed in range(20):
            output = operation.run_structure(
                structure,
                RandomOccupancyParams(
                    source="Manual",
                    manual="Co:0.25,Ni:0.75",
                    mode="Random",
                    samples=1,
                    use_seed=True,
                    seed=seed,
                ),
            )[0]
            observed_counts.add(
                json.loads(output.info["random_occupancy"])["actual_counts"]["Co"]
            )
        self.assertGreater(len(observed_counts), 1)

    def test_random_occupancy_samples_allow_duplicate_arrangements(self):
        structure = Atoms("Si2", positions=np.zeros((2, 3)))
        outputs = RandomOccupancyOperation().run_structure(
            structure,
            RandomOccupancyParams(
                source="Manual",
                manual="Fe:1,Co:1",
                mode="Exact",
                samples=10,
                use_seed=True,
                seed=3,
            ),
        )
        self.assertEqual(len(outputs), 10)
        self.assertLessEqual(
            len({tuple(output.get_chemical_symbols()) for output in outputs}),
            2,
        )
        self.assertEqual(
            [json.loads(output.info["random_occupancy"])["sample_index"] for output in outputs],
            list(range(10)),
        )

    def test_random_occupancy_legacy_json_roundtrip(self):
        legacy = {
            "class": "RandomOccupancyCard",
            "check_state": True,
            "source": "Manual",
            "manual": "Fe:0.4,Co:0.6",
            "mode": "Random",
            "samples": [5],
            "group_filter": "A,B",
            "use_seed": True,
            "seed": [61],
        }
        card = RandomOccupancyCard()
        card.from_dict(legacy)
        restored = RandomOccupancyCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

