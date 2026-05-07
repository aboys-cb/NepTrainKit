from .card_test_base import *

from NepTrainKit.core.cards.alloy import sample_dopants
from NepTrainKit.core.cards.defect import _range_values as defect_range_values
from NepTrainKit.core.cards.magnetism import int_range_values, parse_pair_filter, range_values
from NepTrainKit.core.io import farthest_point_sampling


class TestOperationRegressionEdges(BaseCardTest):
    def test_noop_operations_return_copies(self):
        structure = self.structure.copy()
        structure.new_array("group", np.array(["A"] * len(structure), dtype=object))

        cases = [
            MagneticOrderOperation().run_structure(
                structure,
                MagneticOrderParams(gen_fm=False, gen_afm=False, gen_pm=False),
            )[0],
            RandomDopingOperation().run_structure(structure, RandomDopingParams(rules=[]))[0],
            RandomVacancyOperation().run_structure(structure, RandomVacancyParams(rules=[]))[0],
            RandomOccupancyOperation().run_structure(structure, RandomOccupancyParams(source="Manual", manual=""))[0],
            ConditionalReplaceOperation().run_structure(structure, ConditionalReplaceParams(target=""))[0],
            GroupLabelOperation().run_structure(structure, GroupLabelParams(overwrite=False))[0],
        ]

        for result in cases:
            self.assertIsNot(result, structure)

    def test_magnetism_invalid_scan_pair_and_bond_inputs_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "three values"):
            range_values([0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "step must be positive"):
            int_range_values([1, 3, 0])
        with self.assertRaisesRegex(ValueError, "Invalid pair filter"):
            parse_pair_filter("Fe--Co", normalize_case=True)

        structure = Atoms(
            symbols=["Fe", "Fe"],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            cell=np.diag([5.0, 5.0, 5.0]),
            pbc=True,
        )
        structure.set_initial_magnetic_moments([2.0, 2.0])
        base_moments = np.tile([0.0, 0.0, 2.0], (2, 1))
        with self.assertRaisesRegex(ValueError, "same number"):
            SmallAngleSpinTiltOperation().pair_targets(
                structure,
                base_moments,
                SmallAngleSpinTiltParams(
                    canting_mode="Atom pair canting",
                    pair_source="Manual indices",
                    pair_left_indices="1,2",
                    pair_right_indices="1",
                ),
            )
        with self.assertRaisesRegex(ValueError, "Unsupported bond_filter_mode"):
            SmallAngleSpinTiltOperation.passes_pair_filters(
                structure,
                0,
                1,
                np.array([0.0, 0.0, 1.0]),
                SmallAngleSpinTiltParams(bond_filter_mode="typo"),
            )

    def test_lattice_invalid_ranges_and_empty_perturb_are_handled(self):
        with self.assertRaisesRegex(ValueError, "x_range step"):
            CellStrainOperation().run_structure(
                self.structure,
                CellStrainParams(x_range=(0.0, 1.0, 0.0)),
            )

        bad_cell = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=np.diag([0.0, 1.0, 1.0]), pbc=True)
        with self.assertRaisesRegex(ValueError, "nonzero lattice"):
            CellScalingOperation().run_structure(bad_cell, CellScalingParams(max_num=1))

        empty = Atoms(cell=np.diag([1.0, 1.0, 1.0]), pbc=True)
        result = PerturbOperation().run_structure(empty, PerturbParams(max_num=3, engine_type=0))[0]
        self.assertIsNot(result, empty)
        self.assertEqual(len(result), 0)

    def test_alloy_invalid_rules_and_ratios_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            sample_dopants(["Fe", "Co"], [0.0, 0.0], 2)

        with self.assertRaisesRegex(ValueError, "mapping"):
            RandomDopingOperation().run_structure(
                self.structure,
                RandomDopingParams(
                    rules=[
                        {
                            "target": "Si",
                            "dopants": ["Ge"],
                            "use": "atomic_percent",
                            "percent": [10.0, 20.0],
                        }
                    ]
                ),
            )

        with self.assertRaisesRegex(ValueError, "minimum"):
            RandomDopingOperation().run_structure(
                self.structure,
                RandomDopingParams(
                    rules=[
                        {
                            "target": "Si",
                            "dopants": {"Ge": 1.0},
                            "use": "atomic_percent",
                            "percent": [20.0, 10.0],
                        }
                    ]
                ),
            )

        self.assertEqual(CompositionGradientOperation._normalized_composition("Fe:nan,Co:1", ["Fe", "Co"]), {})

    def test_defect_invalid_edges_fail_and_adsorbate_uses_current_positions(self):
        with self.assertRaisesRegex(ValueError, "step must be positive"):
            defect_range_values((0.0, 1.0, 0.0))

        empty = Atoms(cell=np.diag([1.0, 1.0, 1.0]), pbc=True)
        with self.assertRaisesRegex(ValueError, "at least two atoms"):
            VacancyDefectOperation().run_structure(empty, VacancyDefectParams())
        with self.assertRaisesRegex(ValueError, "at least one atom"):
            StackingFaultOperation().run_structure(empty, StackingFaultParams())
        with self.assertRaisesRegex(ValueError, "min_distance"):
            InsertDefectOperation().run_structure(
                self.structure,
                InsertDefectParams(min_distance=-1.0),
            )

        structure = Atoms("H", positions=[[0.0, 0.0, 1.0]], cell=np.diag([10.0, 10.0, 10.0]), pbc=True)
        candidate = InsertDefectOperation._sample_adsorbate(
            structure,
            np.array([[0.0, 0.0, 7.0]], dtype=float),
            np.asarray(structure.cell.array, dtype=float),
            2,
            1.0,
            rng=np.random.default_rng(1),
        )
        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(float(candidate[2]), 8.0, places=8)

    def test_filter_empty_fps_input_returns_empty_selection(self):
        indices = farthest_point_sampling(np.empty((0, 3), dtype=float), n_samples=10)
        self.assertEqual(indices, [])
