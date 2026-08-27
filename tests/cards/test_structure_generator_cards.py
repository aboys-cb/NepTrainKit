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


class TestStructureGeneratorCards(BaseCardTest):
    def test_local_solvation_ion_water_is_reproducible_and_tags_output(self):
        structure = Atoms(
            symbols=["Ca"],
            positions=[[0.0, 0.0, 0.0]],
            pbc=False,
        )
        structure.info["Config_type"] = "Ca_seed"
        params = LocalSolvationParams(
            structures=1,
            solvent_count=2,
            sampling_mode="auto",
            center_mode="elements",
            center_elements="Ca",
            shell=(2.4, 3.2),
            min_distance=0.8,
            max_attempts=1000,
            use_seed=True,
            seed=17,
        )

        first = LocalSolvationOperation().run_structure(structure, params)[0]
        second = LocalSolvationOperation().run_structure(structure, params)[0]

        self.assertEqual(len(first), 7)
        self.assertEqual(first.get_chemical_symbols().count("O"), 2)
        self.assertEqual(first.get_chemical_symbols().count("H"), 4)
        self.assertTrue(np.allclose(first.cell.array, np.diag([100.0, 100.0, 100.0])))
        self.assertTrue(np.allclose(first.get_positions(), second.get_positions()))
        self.assertIn("SolvLocal(mode=ion-water,n=2,sel=1)", first.info.get("Config_type", ""))

    def test_local_solvation_rejects_empty_selection(self):
        structure = Atoms(
            symbols=["Si"],
            positions=[[0.0, 0.0, 0.0]],
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=False,
        )
        with self.assertRaisesRegex(ValueError, "no center atoms selected"):
            LocalSolvationOperation().run_structure(
                structure,
                LocalSolvationParams(
                    solvent_count=1,
                    center_mode="elements",
                    center_elements="Ca",
                    use_seed=True,
                    seed=1,
                ),
            )

    def test_local_solvation_dense_periodic_solid_fails_with_actionable_message(self):
        structure = Atoms(
            symbols=["Cu", "Cu", "Cu", "Cu"],
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            cell=np.diag([2.5, 2.5, 2.5]),
            pbc=True,
        )

        with self.assertRaisesRegex(ValueError, "periodic dense structure"):
            LocalSolvationOperation().run_structure(
                structure,
                LocalSolvationParams(
                    solvent_count=2,
                    sampling_mode="water",
                    center_mode="all",
                    shell=(1.8, 2.5),
                    min_distance=1.0,
                    max_attempts=5000,
                    use_seed=True,
                    seed=3,
                ),
            )
        with self.assertRaisesRegex(ValueError, "no solvent molecule"):
            LocalSolvationOperation().run_structure(
                structure,
                LocalSolvationParams(
                    solvent_count=2,
                    sampling_mode="water",
                    center_mode="all",
                    shell=(1.8, 2.5),
                    min_distance=1.0,
                    max_attempts=500,
                    strict_count=False,
                    use_seed=True,
                    seed=3,
                ),
            )

    def test_local_solvation_card_roundtrip(self):
        card = LocalSolvationCard()
        self.assertTrue(card.solvent_edit.isHidden())
        self.assertTrue(card.min_distance_frame.isHidden())
        self.assertTrue(card.elements_edit.isHidden())
        self.assertIn("Load an upstream structure", card.preview_label.text())

        card.center_mode_combo.setCurrentIndex(
            card.center_mode_combo.findData("elements")
        )
        self.assertFalse(card.elements_edit.isHidden())
        card.elements_edit.setText("Ca")
        card.set_dataset([Atoms("Ca", positions=[[0, 0, 0]], pbc=False)])
        self.assertIn("centers 1 (Ca)", card.preview_label.text())
        self.assertIn("resolved profile Ion-water first shell", card.preview_label.text())
        self.assertIn("Ca 2.3–2.6 Å", card.preview_label.text())

        card.advanced_checkbox.setChecked(True)
        self.assertFalse(card.min_distance_frame.isHidden())
        self.assertFalse(card.auto_box_checkbox.isHidden())
        self.assertFalse(card.box_size_frame.isHidden())
        self.assertTrue(card.box_frame.isHidden())
        card.auto_box_checkbox.setChecked(True)
        self.assertTrue(card.box_size_frame.isHidden())
        self.assertFalse(card.box_frame.isHidden())

        card.structures_frame.set_input_value([2])
        card.count_frame.set_input_value([3])
        card.mode_combo.setCurrentIndex(card.mode_combo.findData("water"))
        card.center_mode_combo.setCurrentIndex(
            card.center_mode_combo.findData("indices")
        )
        card.indices_edit.setText("1")
        card.shell_frame.set_input_value([2.0, 3.5])
        card.min_distance_frame.set_input_value([0.75])
        card.strict_checkbox.setChecked(True)
        card.auto_box_checkbox.setChecked(True)
        card.box_size_frame.set_input_value([90.0])
        card.box_frame.set_input_value([6.0, 20.0])
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([23])

        restored = LocalSolvationCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_solvation_cards_roundtrip_all_flexible_solvent_controls(self):
        local_params = LocalSolvationParams(
            structures=2,
            solvent_count=3,
            sampling_mode="loose",
            center_mode="indices",
            center_indices="1,3",
            shell=(2.0, 3.5),
            min_distance=0.7,
            collision_scale=0.8,
            max_attempts=1234,
            strict_count=False,
            auto_box=True,
            box_size=80.0,
            box_padding=7.0,
            min_box=25.0,
            flex_solvent=True,
            flex_pool=7,
            flex_torsion_range=(-75.0, 95.0),
            flex_max_torsions=3,
            flex_gaussian_sigma=0.02,
            use_seed=True,
            seed=41,
        )
        local = LocalSolvationCard()
        local.set_params(local_params)
        restored_local = LocalSolvationCard()
        restored_local.from_dict(local.to_dict())
        self.assertEqual(restored_local.get_params(), local_params)

        box_params = SolventBoxFillParams(
            structures=2,
            count_mode="density",
            solvent_count=8,
            density=0.75,
            sampling_mode="loose",
            fill_packing=0.85,
            min_distance=0.65,
            collision_scale=0.78,
            max_attempts_per_solvent=321,
            strict_count=False,
            flex_solvent=True,
            flex_pool=9,
            flex_torsion_range=(-65.0, 85.0),
            flex_max_torsions=4,
            flex_gaussian_sigma=0.025,
            use_seed=True,
            seed=43,
        )
        box = SolventBoxFillCard()
        box.set_params(box_params)
        restored_box = SolventBoxFillCard()
        restored_box.from_dict(box.to_dict())
        self.assertEqual(restored_box.get_params(), box_params)

    def test_local_solvation_validates_counts_preserves_periodic_box_and_orients_water(self):
        operation = LocalSolvationOperation()
        periodic = Atoms(
            "Ca",
            positions=[[5.0, 5.0, 5.0]],
            cell=[20, 20, 20],
            pbc=True,
        )
        output = operation.run_structure(
            periodic,
            LocalSolvationParams(
                solvent_count=1,
                sampling_mode="water",
                center_mode="all",
                shell=(3.0, 3.1),
                min_distance=0.5,
                auto_box=True,
                use_seed=True,
                seed=9,
            ),
        )[0]

        np.testing.assert_allclose(output.cell.array, periodic.cell.array)
        np.testing.assert_array_equal(output.pbc, periodic.pbc)
        positions = output.positions
        water_com = positions[1:4].mean(axis=0)
        oh_bisector = positions[[2, 3]].mean(axis=0) - positions[1]
        self.assertGreater(
            float(np.dot(oh_bisector, water_com - positions[0])),
            0.0,
        )

        with self.assertRaisesRegex(ValueError, "structures must be an integer"):
            operation.run_structure(
                periodic,
                LocalSolvationParams(structures=1.5),  # type: ignore[arg-type]
            )
        invalid_periodic = Atoms(
            "Ca",
            positions=[[0, 0, 0]],
            pbc=True,
        )
        with self.assertRaisesRegex(ValueError, "periodic input"):
            operation.run_structure(
                invalid_periodic,
                LocalSolvationParams(solvent_count=1),
            )

    def test_solvent_box_fill_fixed_count_preserves_cell_and_is_reproducible(self):
        structure = Atoms(
            symbols=["Si"],
            positions=[[1.0, 1.0, 1.0]],
            cell=np.diag([16.0, 16.0, 16.0]),
            pbc=True,
        )
        structure.info["Config_type"] = "box_seed"
        params = SolventBoxFillParams(
            structures=1,
            count_mode="fixed",
            solvent_count=2,
            min_distance=0.8,
            max_attempts_per_solvent=1000,
            use_seed=True,
            seed=5,
        )

        first = SolventBoxFillOperation().run_structure(structure, params)[0]
        second = SolventBoxFillOperation().run_structure(structure, params)[0]

        self.assertEqual(len(first), 7)
        self.assertTrue(np.allclose(first.cell.array, structure.cell.array))
        self.assertTrue(first.pbc.all())
        self.assertTrue(np.allclose(first.get_positions(), second.get_positions()))
        self.assertIn("SolvBox(mode=water,req=2,ok=2)", first.info.get("Config_type", ""))

    def test_solvent_box_fill_density_mode_and_card_roundtrip(self):
        structure = Atoms(
            symbols=["Ar"],
            positions=[[2.0, 2.0, 2.0]],
            cell=np.diag([10.0, 10.0, 10.0]),
            pbc=True,
        )
        params = SolventBoxFillParams(
            count_mode="density",
            density=1.0,
            fill_packing=1.0,
            min_distance=0.5,
            max_attempts_per_solvent=1000,
            use_seed=True,
            seed=9,
        )
        result = SolventBoxFillOperation().run_structure(structure, params)[0]
        self.assertGreater(len(result), len(structure))

        card = SolventBoxFillCard()
        self.assertTrue(card.solvent_edit.isHidden())
        self.assertTrue(card.density_frame.isHidden())
        self.assertTrue(card.min_distance_frame.isHidden())
        self.assertIn("Load an upstream periodic cell", card.preview_label.text())

        card.set_dataset([structure])
        self.assertIn("cell 1000 Å³", card.preview_label.text())
        self.assertIn("target 100 molecules", card.preview_label.text())

        card.count_mode_combo.setCurrentIndex(
            card.count_mode_combo.findData("density")
        )
        self.assertTrue(card.count_frame.isHidden())
        self.assertFalse(card.density_frame.isHidden())
        card.count_frame.set_input_value([9])
        card.density_frame.set_input_value([0.7])
        card.mode_combo.setCurrentIndex(card.mode_combo.findData("loose"))
        card.fill_packing_frame.set_input_value([0.8])
        self.assertIn("nominal density count", card.preview_label.text())

        card.advanced_checkbox.setChecked(True)
        self.assertFalse(card.min_distance_frame.isHidden())
        self.assertTrue(card.flex_pool_frame.isHidden())
        card.flex_checkbox.setChecked(True)
        self.assertFalse(card.flex_pool_frame.isHidden())
        card.min_distance_frame.set_input_value([0.6])
        card.strict_checkbox.setChecked(True)
        card.flex_checkbox.setChecked(False)
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([19])

        restored = SolventBoxFillCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_solvent_box_fill_supports_empty_periodic_cell_and_validates_density_multiplier(self):
        empty = Atoms(cell=[10, 10, 10], pbc=True)
        operation = SolventBoxFillOperation()

        output = operation.run_structure(
            empty,
            SolventBoxFillParams(
                solvent_count=1,
                min_distance=0.8,
                use_seed=True,
                seed=4,
            ),
        )[0]

        self.assertEqual(len(output), 3)
        self.assertIn(
            "SolvBox(mode=water,req=1,ok=1)",
            output.info.get("Config_type", ""),
        )
        with self.assertRaisesRegex(ValueError, "at most 1"):
            operation.run_structure(
                empty,
                SolventBoxFillParams(
                    count_mode="density",
                    fill_packing=1.1,
                ),
            )

        summary_empty = operation.capacity_summary(
            empty,
            SolventBoxFillParams(count_mode="density"),
        )
        occupied = Atoms(
            "Ar20",
            positions=np.zeros((20, 3)),
            cell=[10, 10, 10],
            pbc=True,
        )
        summary_occupied = operation.capacity_summary(
            occupied,
            SolventBoxFillParams(count_mode="density"),
        )
        self.assertEqual(
            summary_empty["target_count"],
            summary_occupied["target_count"],
        )

    def test_solvent_box_fill_tiny_dense_cell_fails_quickly(self):
        structure = Atoms(
            symbols=["Cu", "Cu", "Cu", "Cu"],
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            cell=np.diag([2.5, 2.5, 2.5]),
            pbc=True,
        )

        with self.assertRaisesRegex(ValueError, "too small/dense"):
            SolventBoxFillOperation().run_structure(
                structure,
                SolventBoxFillParams(
                    solvent_count=10,
                    min_distance=1.0,
                    max_attempts_per_solvent=1000,
                    use_seed=True,
                    seed=3,
                ),
            )

    def test_solvent_box_fill_rejects_local_ion_water_mode(self):
        structure = Atoms(
            symbols=["Na"],
            positions=[[4.0, 4.0, 4.0]],
            cell=np.diag([12.0, 12.0, 12.0]),
            pbc=True,
        )

        with self.assertRaisesRegex(ValueError, "sampling_mode"):
            SolventBoxFillOperation().run_structure(
                structure,
                SolventBoxFillParams(
                    sampling_mode="ion-water",
                    solvent_count=1,
                    use_seed=True,
                    seed=2,
                ),
            )

    def test_local_solvation_respects_global_min_distance(self):
        structure = Atoms(
            symbols=["Ca"],
            positions=[[0.0, 0.0, 0.0]],
            pbc=False,
        )
        min_distance = 0.85
        result = LocalSolvationOperation().run_structure(
            structure,
            LocalSolvationParams(
                solvent_count=4,
                sampling_mode="ion-water",
                center_mode="elements",
                center_elements="Ca",
                shell=(2.4, 3.6),
                min_distance=min_distance,
                max_attempts=3000,
                use_seed=True,
                seed=31,
            ),
        )[0]

        symbols = result.get_chemical_symbols()
        positions = result.get_positions()
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if symbols[i] == "O" and symbols[j] == "H":
                    continue
                if symbols[i] == "H" and symbols[j] == "O":
                    continue
                if symbols[i] == symbols[j] == "H":
                    continue
                distance = float(np.linalg.norm(positions[i] - positions[j]))
                self.assertGreaterEqual(distance + 1e-12, min_distance)

    def test_local_solvation_ion_water_uses_shell_after_first_coordination(self):
        structure = Atoms(
            symbols=["Ca"],
            positions=[[0.0, 0.0, 0.0]],
            pbc=False,
        )
        result = LocalSolvationOperation().run_structure(
            structure,
            LocalSolvationParams(
                solvent_count=50,
                sampling_mode="auto",
                center_mode="all",
                shell=(0.0, 5.0),
                collision_scale=0.72,
                max_attempts=10000,
                use_seed=True,
                seed=1,
            ),
        )[0]

        self.assertEqual(result.get_chemical_symbols().count("O"), 50)
        self.assertEqual(result.get_chemical_symbols().count("H"), 100)
        self.assertIn("SolvLocal(mode=ion-water,n=50,sel=1)", result.info.get("Config_type", ""))

    def test_local_solvation_z_range_uses_selected_center_region(self):
        structure = Atoms(
            symbols=["Ca", "Ca"],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 20.0]],
            pbc=False,
        )
        result = LocalSolvationOperation().run_structure(
            structure,
            LocalSolvationParams(
                solvent_count=2,
                sampling_mode="ion-water",
                center_mode="z_range",
                z_range=(19.0, 21.0),
                shell=(2.4, 3.4),
                min_distance=0.8,
                max_attempts=1000,
                use_seed=True,
                seed=8,
            ),
        )[0]

        oxygen_positions = np.array([pos for sym, pos in zip(result.get_chemical_symbols(), result.get_positions()) if sym == "O"])
        self.assertEqual(len(oxygen_positions), 2)
        near_top = np.linalg.norm(oxygen_positions - np.array([0.0, 0.0, 20.0]), axis=1)
        near_bottom = np.linalg.norm(oxygen_positions - np.array([0.0, 0.0, 0.0]), axis=1)
        self.assertTrue(np.all(near_top < near_bottom))

    def test_local_solvation_indices_and_box_controls_have_observable_effect(self):
        structure = Atoms(
            symbols=["Ca", "Ca"],
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 12.0]],
            pbc=False,
        )
        operation = LocalSolvationOperation()
        auto = operation.run_structure(
            structure,
            LocalSolvationParams(
                solvent_count=1,
                sampling_mode="ion-water",
                center_mode="indices",
                center_indices="2",
                shell=(2.4, 2.8),
                min_distance=0.8,
                auto_box=True,
                box_padding=3.0,
                min_box=20.0,
                use_seed=True,
                seed=19,
            ),
        )[0]
        symbols = auto.get_chemical_symbols()
        ca_positions = np.asarray(
            [pos for sym, pos in zip(symbols, auto.positions) if sym == "Ca"]
        )
        oxygen = np.asarray(
            [pos for sym, pos in zip(symbols, auto.positions) if sym == "O"]
        )[0]

        self.assertTrue(np.all(auto.cell.lengths() >= 20.0 - 1e-12))
        self.assertLess(
            np.linalg.norm(oxygen - ca_positions[1]),
            np.linalg.norm(oxygen - ca_positions[0]),
        )

        fixed = operation.run_structure(
            structure,
            LocalSolvationParams(
                solvent_count=1,
                sampling_mode="water",
                center_mode="indices",
                center_indices="1",
                shell=(2.4, 2.8),
                min_distance=0.8,
                auto_box=False,
                box_size=30.0,
                use_seed=True,
                seed=20,
            ),
        )[0]
        np.testing.assert_allclose(fixed.cell.array, np.diag([30.0, 30.0, 30.0]))
        self.assertFalse(np.asarray(fixed.pbc, dtype=bool).any())

    def test_solvent_box_fill_density_count_matches_formula(self):
        structure = Atoms(
            symbols=["Si"],
            positions=[[1.0, 1.0, 1.0]],
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=True,
        )
        solvent = parse_solvent_xyz(LocalSolvationParams().solvent_xyz)
        expected = estimate_solvent_count_from_density(solvent, 1.0, structure.cell.array, 1.0)
        result = SolventBoxFillOperation().run_structure(
            structure,
            SolventBoxFillParams(
                count_mode="density",
                density=1.0,
                fill_packing=1.0,
                min_distance=0.8,
                max_attempts_per_solvent=2000,
                use_seed=True,
                seed=7,
            ),
        )[0]

        self.assertEqual(result.get_chemical_symbols().count("O"), expected)
        self.assertEqual(result.get_chemical_symbols().count("H"), 2 * expected)

    def test_solvent_box_fill_nonorthogonal_output_has_no_pbc_collisions(self):
        structure = Atoms(
            symbols=["Si"],
            positions=[[1.0, 1.0, 1.0]],
            cell=np.array([[14.0, 0.0, 0.0], [3.0, 13.0, 0.0], [1.0, 2.0, 15.0]]),
            pbc=True,
        )
        min_distance = 0.8
        result = SolventBoxFillOperation().run_structure(
            structure,
            SolventBoxFillParams(
                solvent_count=8,
                min_distance=min_distance,
                max_attempts_per_solvent=1000,
                use_seed=True,
                seed=12,
            ),
        )[0]

        symbols = result.get_chemical_symbols()
        positions = result.get_positions()
        for i in range(len(result)):
            self.assertFalse(
                has_collision(
                    [symbols[i]],
                    positions[i : i + 1],
                    symbols[:i] + symbols[i + 1 :],
                    np.vstack([positions[:i], positions[i + 1 :]]),
                    cell=result.cell.array,
                    pbc=np.asarray(result.pbc, dtype=bool),
                    collision_scale=0.70,
                    min_distance=min_distance,
                )
            )

    def test_solvent_box_fill_partial_output_when_not_strict(self):
        structure = Atoms(
            symbols=["Cu", "Cu", "Cu", "Cu"],
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            cell=np.diag([2.5, 2.5, 2.5]),
            pbc=True,
        )
        with self.assertRaisesRegex(ValueError, "no solvent molecule"):
            SolventBoxFillOperation().run_structure(
                structure,
                SolventBoxFillParams(
                    solvent_count=10,
                    min_distance=1.0,
                    max_attempts_per_solvent=1000,
                    strict_count=False,
                    use_seed=True,
                    seed=3,
                ),
            )

    def test_solvent_box_fill_flexible_solvent_branch_runs(self):
        butane = """14
butane
C -2.3100 0.0000 0.0000
C -0.7700 0.0000 0.0000
C 0.7700 0.0000 0.0000
C 2.3100 0.0000 0.0000
H -2.6700 1.0000 0.0000
H -2.6700 -0.5000 0.8660
H -2.6700 -0.5000 -0.8660
H -0.4100 0.5000 0.8660
H -0.4100 0.5000 -0.8660
H 0.4100 -0.5000 0.8660
H 0.4100 -0.5000 -0.8660
H 2.6700 -1.0000 0.0000
H 2.6700 0.5000 0.8660
H 2.6700 0.5000 -0.8660
"""
        structure = Atoms(
            symbols=["Ar"],
            positions=[[8.0, 8.0, 8.0]],
            cell=np.diag([30.0, 30.0, 30.0]),
            pbc=True,
        )
        result = SolventBoxFillOperation().run_structure(
            structure,
            SolventBoxFillParams(
                solvent_xyz=butane,
                solvent_count=1,
                sampling_mode="general",
                min_distance=0.8,
                flex_solvent=True,
                flex_pool=3,
                flex_max_torsions=1,
                flex_gaussian_sigma=0.01,
                use_seed=True,
                seed=4,
            ),
        )[0]

        self.assertEqual(len(result), len(structure) + 14)
        self.assertEqual(result.get_chemical_symbols().count("C"), 4)
        self.assertIn(
            "SolvBox(mode=general,req=1,ok=1)",
            result.info.get("Config_type", ""),
        )

    def test_random_packing_preserves_cell_composition_and_distance_constraints(self):
        structure = Atoms(
            symbols=["Fe", "Fe", "O", "O"],
            positions=np.zeros((4, 3)),
            cell=np.diag([8.0, 8.0, 8.0]),
            pbc=True,
        )
        structure.info["Config_type"] = "FeO_seed"

        results = RandomPackingOperation().run_structure(
            structure,
            RandomPackingParams(
                structures=2,
                min_distance=1.0,
                pair_min_distances="Fe-O:2.0,O-O:1.5",
                use_seed=True,
                seed=7,
            ),
        )

        self.assertEqual(len(results), 2)
        for atoms in results:
            self.assertTrue(np.allclose(atoms.cell.array, structure.cell.array))
            self.assertTrue(np.array_equal(np.asarray(atoms.pbc, dtype=bool), np.asarray(structure.pbc, dtype=bool)))
            self.assertEqual(atoms.get_chemical_symbols().count("Fe"), 2)
            self.assertEqual(atoms.get_chemical_symbols().count("O"), 2)
            symbols = atoms.get_chemical_symbols()
            for i in range(len(atoms)):
                for j in range(i + 1, len(atoms)):
                    dist = RandomPackingOperation.candidate_distances(
                        atoms.positions[i],
                        atoms.positions[j : j + 1],
                        cell=np.asarray(atoms.cell.array, dtype=float),
                        pbc=np.asarray(atoms.pbc, dtype=bool),
                    )[0]
                    expected = RandomPackingOperation.min_distance_for_pair(
                        symbols[i],
                        symbols[j],
                        1.0,
                        RandomPackingOperation.parse_pair_min_distances("Fe-O:2.0,O-O:1.5"),
                    )
                    self.assertGreaterEqual(dist + 1e-12, expected)
            self.assertIn("RandPack(n=4,d=1", atoms.info.get("Config_type", ""))

    def test_random_packing_manual_exact_composition_and_roundtrip(self):
        structure = Atoms(
            symbols=["Si"],
            positions=[[0.0, 0.0, 0.0]],
            cell=np.diag([7.0, 7.0, 7.0]),
            pbc=True,
        )
        structure.info["Config_type"] = "manual_pack"

        card = RandomPackingCard()
        card.structures_frame.set_input_value([1])
        card.composition_edit.setText("Fe:2,O:1")
        card.min_distance_frame.set_input_value([1.0])
        card.pair_distance_edit.setText("Fe-O:1.2")
        card.attempts_frame.set_input_value([200])
        card.strict_checkbox.setChecked(True)
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([9])

        result = card.process_structure(structure)[0]
        self.assertEqual(result.get_chemical_symbols().count("Fe"), 2)
        self.assertEqual(result.get_chemical_symbols().count("O"), 1)
        self.assertEqual(len(result), 3)

        restored = RandomPackingCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), card.get_params())

    def test_random_packing_invalid_or_impossible_constraints_fail_explicitly(self):
        structure = Atoms(
            symbols=["He", "He"],
            positions=[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            cell=np.diag([1.0, 1.0, 1.0]),
            pbc=True,
        )
        with self.assertRaisesRegex(ValueError, "composition count"):
            RandomPackingOperation.symbols_from_params(structure, "Fe:0.5,O:0.5")

        with self.assertRaisesRegex(ValueError, "could not place"):
            RandomPackingOperation().run_structure(
                structure,
                RandomPackingParams(
                    structures=1,
                    min_distance=0.9,
                    max_attempts_per_atom=5,
                    strict_mode=True,
                    use_seed=True,
                    seed=1,
                ),
            )

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

    def test_group_label_card_and_group_afm(self):
        proto = CrystalPrototypeBuilderCard()
        set_combo_value(proto.structure_combo, "bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.max_output_frame.set_input_value([1])
        base = proto.create_operation().generate(proto.get_params())[0].repeat((2, 2, 2))

        gl = GroupLabelCard()
        set_combo_value(gl.plane_combo, "111")
        gl.group_a_edit.setText("A")
        gl.group_b_edit.setText("B")
        labeled = gl.process_structure(base)[0]
        self.assertIn("group", labeled.arrays)
        groups = set(str(g) for g in labeled.arrays["group"])
        self.assertTrue({"A", "B"}.issubset(groups))

        mag = MagneticOrderCard()
        mag.format_combo.setCurrentIndex(0)
        mag.map_edit.setText("Fe:2.2")
        mag.fm_checkbox.setChecked(False)
        mag.afm_checkbox.setChecked(True)
        mag.afm_mode_combo.setCurrentIndex(1)
        mag.group_a_edit.setText("A")
        mag.group_b_edit.setText("B")
        mag.pm_checkbox.setChecked(False)
        res = mag.process_structure(labeled)
        self.assertEqual(len(res), 1)
        afm = res[0]
        m = np.array(afm.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(np.any(m > 0) and np.any(m < 0))

    def test_group_label_operation_is_ui_independent(self):
        base = CrystalPrototypeBuilderOperation().generate(
            CrystalPrototypeBuilderParams(
                lattice="bcc",
                element="Fe",
                a_range=(2.9, 2.9, 0.1),
                max_outputs=1,
            )
        )[0].repeat((2, 2, 2))

        labeled = GroupLabelOperation().run_structure(
            base,
            GroupLabelParams(
                miller_index="111",
                group_a="up",
                group_b="down",
            ),
        )[0]

        self.assertIn("group", labeled.arrays)
        self.assertEqual(set(str(value) for value in labeled.arrays["group"]), {"up", "down"})
        self.assertIn("Grp(hkl111,tol=0.05,up/down)", labeled.info.get("Config_type", ""))

    def test_group_label_alternates_detected_layers_after_expansion(self):
        base = Atoms(
            symbols=["Fe"] * 4,
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            cell=np.diag([4.0, 2.0, 2.0]),
            pbc=True,
        )
        params = GroupLabelParams(miller_index="100")
        labeled = GroupLabelOperation().run_structure(base, params)[0]
        self.assertEqual(list(labeled.arrays["group"]), ["A", "B", "A", "B"])

        expanded = base.repeat((2, 1, 1))
        repeated = GroupLabelOperation().run_structure(expanded, params)[0]
        self.assertEqual(
            list(repeated.arrays["group"]),
            ["A", "B", "A", "B", "A", "B", "A", "B"],
        )

    def test_group_label_uses_reciprocal_normal_in_nonorthogonal_cell(self):
        cell = np.array([[4.0, 0.0, 0.0], [1.5, 3.0, 0.0], [0.4, 0.7, 5.0]])
        base = Atoms(
            symbols=["Fe"] * 4,
            scaled_positions=[[0.0, 0.2, 0.3], [0.25, 0.2, 0.3], [0.5, 0.2, 0.3], [0.75, 0.2, 0.3]],
            cell=cell,
            pbc=[True, True, False],
        )
        base.new_array("foo", np.arange(4))
        labeled = GroupLabelOperation().run_structure(
            base,
            GroupLabelParams(miller_index="100", layer_tolerance=0.05),
        )[0]
        self.assertEqual(list(labeled.arrays["group"]), ["A", "B", "A", "B"])
        np.testing.assert_allclose(labeled.cell.array, base.cell.array)
        np.testing.assert_allclose(labeled.positions, base.positions)
        np.testing.assert_array_equal(labeled.arrays["foo"], base.arrays["foo"])

    def test_group_label_merges_a_thermally_split_periodic_boundary_layer(self):
        base = Atoms(
            symbols=["Fe"] * 5,
            scaled_positions=[
                [0.005, 0.0, 0.0],
                [0.995, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.50, 0.0, 0.0],
                [0.75, 0.0, 0.0],
            ],
            cell=np.diag([4.0, 2.0, 2.0]),
            pbc=True,
        )
        operation = GroupLabelOperation()
        layer_ids = operation.layer_ids(base, "100", 0.05)
        self.assertEqual(int(layer_ids.max()) + 1, 4)
        self.assertEqual(layer_ids[0], layer_ids[1])
        labeled = operation.run_structure(
            base,
            GroupLabelParams(miller_index="100", layer_tolerance=0.05),
        )[0]
        self.assertEqual(list(labeled.arrays["group"]), ["A", "A", "B", "A", "B"])

    def test_group_label_counts_periodic_110_layers_without_parallelepiped_duplicates(self):
        base = Atoms(
            symbols=["Fe"] * 4,
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.5, 0.5, 0.0],
            ],
            cell=np.diag([4.0, 4.0, 2.0]),
            pbc=True,
        )
        layer_ids = GroupLabelOperation().layer_ids(base, "110", 0.05)
        self.assertEqual(int(layer_ids.max()) + 1, 2)
        self.assertEqual(list(layer_ids), [0, 1, 1, 0])

    def test_group_label_rejects_invalid_settings_and_single_layer(self):
        base = self.structure.copy()
        operation = GroupLabelOperation()
        with self.assertRaisesRegex(ValueError, "Plane index"):
            operation.run_structure(base, GroupLabelParams(miller_index="112"))
        with self.assertRaisesRegex(ValueError, "positive finite"):
            operation.run_structure(base, GroupLabelParams(layer_tolerance=0.0))
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            operation.run_structure(base, GroupLabelParams(group_a=""))
        with self.assertRaisesRegex(ValueError, "must be different"):
            operation.run_structure(base, GroupLabelParams(group_a="same", group_b="same"))

        single_layer = Atoms("Fe", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3), pbc=True)
        with self.assertRaisesRegex(ValueError, "at least two detected atomic layers"):
            operation.run_structure(single_layer, GroupLabelParams())

    def test_group_label_default_preserves_existing_groups(self):
        base = self.structure.copy()
        base.new_array(
            "group",
            np.asarray(["surface"] * len(base), dtype=object),
        )
        base.info["Config_type"] = "seed"
        params = GroupLabelParams()
        self.assertFalse(params.overwrite)
        output = GroupLabelOperation().run_structure(base, params)[0]
        self.assertIsNot(output, base)
        np.testing.assert_array_equal(output.arrays["group"], base.arrays["group"])
        self.assertEqual(output.info["Config_type"], "seed")

    def test_group_label_card_roundtrip(self):
        card = GroupLabelCard()
        set_combo_value(card.plane_combo, "110")
        card.tolerance_frame.set_input_value([0.08])
        card.group_a_edit.setText("alpha")
        card.group_b_edit.setText("beta")
        card.overwrite_checkbox.setChecked(False)

        restored = GroupLabelCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())
        self.assertEqual(restored.get_params().miller_index, "110")
        self.assertAlmostEqual(restored.get_params().layer_tolerance, 0.08)

        legacy = GroupLabelCard()
        with patch(
            "NepTrainKit.ui.views._card.group_label_card.MessageManager.send_warning_message"
        ) as warning:
            legacy.from_dict(
                {
                    "class": "GroupLabelCard",
                    "check_state": True,
                    "mode": "k-vector layers (recommended)",
                    "kvec": "100",
                    "group_a": "old_a",
                    "group_b": "old_b",
                }
            )
        warning.assert_called_once()
        self.assertFalse(legacy.legacy_notice.isHidden())
        self.assertIn("real atomic layers", legacy.legacy_notice.text())
        self.assertEqual(legacy.get_params().miller_index, "100")
        self.assertTrue(legacy.get_params().overwrite)

    def test_group_label_preview_shows_layer_sequence_and_existing_labels(self):
        fcc = Atoms(
            symbols=["Ni"] * 4,
            scaled_positions=[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            cell=np.eye(3),
            pbc=True,
        )
        card = GroupLabelCard()
        card.set_dataset([fcc])
        self.assertIn("1 layers", card.preview_label.text())
        self.assertIn("A(4)", card.preview_label.text())
        self.assertIn("A=4", card.preview_label.text())
        self.assertIn("B=0", card.preview_label.text())
        self.assertIn("At least two layers", card.preview_label.text())

        existing = fcc.copy()
        existing.new_array(
            "group",
            np.asarray(["surface", "surface", "bulk", "bulk"], dtype=object),
        )
        card.set_dataset([existing])
        self.assertIn("output will be unchanged", card.preview_label.text())
        self.assertIn("bulk=2", card.preview_label.text())
        self.assertIn("surface=2", card.preview_label.text())

        card.overwrite_checkbox.setChecked(True)
        self.assertIn("will be overwritten", card.preview_label.text())

    def test_organic_configuration_card(self):
        card = OrganicMolConfigPBCCard()
        structure = self.structure.copy()
        card.perturb_frame.set_input_value([2])
        card.torsion_frame.set_input_value([-30.0, 30.0])
        card.max_torsions_frame.set_input_value([1])
        card.sigma_frame.set_input_value([0.01])
        card.pbc_combo.setCurrentIndex(1)

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        for atoms in results:
            self.assertEqual(len(atoms), len(structure))
            self.assertFalse(np.allclose(atoms.get_positions(), structure.get_positions()))

    def test_organic_configuration_topology_preview_and_progressive_controls(self):
        molecule = Atoms(
            "C4",
            positions=[
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [2.8, 0.2, 0.0],
                [4.1, 0.2, 1.0],
            ],
            pbc=False,
        )
        card = OrganicMolConfigPBCCard()
        try:
            self.assertTrue(card.topology_section.isHidden())
            self.assertTrue(card.seed_field.isHidden())
            self.assertFalse(card.box_field.isHidden())
            self.assertGreater(card.bond_detect_frame.object_list[0].minimum(), 0.0)
            self.assertIn("Load an upstream molecule", card.preview_label.text())
            card.set_dataset([molecule])

            self.assertIn("background", card.preview_label.text())
            self.assertTrue(_wait_until(lambda: "4 atoms" in card.preview_label.text()))
            self.assertIn("3 detected bonds / 1 rotatable", card.preview_label.text())
            self.assertIn("1 component", card.preview_label.text())
            self.assertIn("nonperiodic", card.preview_label.text())
            self.assertIn("up to 100 outputs", card.preview_label.text())
            self.assertIn("dataset maximum 100", card.preview_label.text())

            card.pbc_combo.setCurrentIndex(card.pbc_combo.findData("yes"))
            self.assertTrue(card.box_field.isHidden())
            card.pbc_combo.setCurrentIndex(card.pbc_combo.findData("no"))
            self.assertFalse(card.box_field.isHidden())

            card.seed_checkbox.setChecked(True)
            self.assertFalse(card.seed_field.isHidden())

            card.advanced_checkbox.setChecked(True)
            self.assertFalse(card.topology_section.isHidden())
            self.assertFalse(card.bond_max_frame.isEnabled())
            card.bond_max_enable.setChecked(True)
            self.assertTrue(card.bond_max_frame.isEnabled())
            card.bond_min_frame.set_input_value([1.2])
            self.assertGreaterEqual(
                card.bond_max_frame.object_list[0].minimum(), 1.2
            )
        finally:
            _wait_until(
                lambda: card._preview_task is None
                or not card._preview_task.isRunning()
            )
            card.close()
            QApplication.processEvents()

    def test_organic_configuration_preview_runs_in_background_and_uses_latest_request(self):
        molecule = Atoms(
            "C4",
            positions=[
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [2.8, 0.2, 0.0],
                [4.1, 0.2, 1.0],
            ],
            pbc=False,
        )
        started = threading.Event()
        release = threading.Event()
        calls = []
        main_thread_id = threading.get_ident()

        def fake_summary(_structure, params):
            calls.append((threading.get_ident(), params.bond_detect_factor))
            if len(calls) == 1:
                started.set()
                release.wait(timeout=3.0)
            bond_count = 4 if params.bond_detect_factor > 1.2 else 3
            return SimpleNamespace(
                atom_count=4,
                bond_count=bond_count,
                component_count=1,
                torsion_count=1,
                torsion_active=True,
                pbc_active=False,
                local_mode=False,
                requested_outputs=params.perturb_per_frame,
                gaussian_sigma=params.gaussian_sigma,
            )

        card = OrganicMolConfigPBCCard()
        try:
            with patch.object(
                OrganicMolConfigPBCOperation,
                "topology_summary",
                side_effect=fake_summary,
            ):
                card.set_dataset([molecule])
                self.assertTrue(_wait_until(started.is_set))
                self.assertNotEqual(calls[0][0], main_thread_id)

                card.bond_detect_frame.set_input_value([1.3])
                self.assertIn("background", card.preview_label.text())
                release.set()

                self.assertTrue(
                    _wait_until(
                        lambda: len(calls) >= 2
                        and "4 detected bonds" in card.preview_label.text()
                    )
                )
                self.assertEqual(calls[1][1], 1.3)
                self.assertNotEqual(calls[1][0], main_thread_id)
        finally:
            release.set()
            _wait_until(
                lambda: card._preview_task is None
                or not card._preview_task.isRunning()
            )
            card.close()
            QApplication.processEvents()

    def test_organic_configuration_operation_honors_boundary_state_and_guards(self):
        molecule = Atoms(
            "C4",
            positions=[
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [2.8, 0.2, 0.0],
                [4.1, 0.2, 1.0],
            ],
            pbc=False,
        )
        operation = OrganicMolConfigPBCOperation()
        params = OrganicMolConfigPBCParams(
            perturb_per_frame=2,
            torsion_range_deg=(60.0, 60.0),
            max_torsions_per_conf=1,
            gaussian_sigma=0.0,
            pbc_mode="auto",
            use_seed=True,
            seed=4,
        )

        outputs = operation.run_structure(molecule, params)

        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(not np.any(atoms.pbc) for atoms in outputs))
        self.assertTrue(all("TG(req=2,ok=2" in atoms.info["Config_type"] for atoms in outputs))

        angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
        ring = Atoms(
            "C6",
            positions=np.column_stack(
                [1.4 * np.cos(angles), 1.4 * np.sin(angles), np.zeros(6)]
            ),
            pbc=False,
        )
        ring_summary = operation.topology_summary(
            ring,
            OrganicMolConfigPBCParams(gaussian_sigma=0.01),
        )
        self.assertEqual(ring_summary.bond_count, 6)
        self.assertEqual(ring_summary.torsion_count, 0)

        no_bonds = operation.topology_summary(
            molecule,
            OrganicMolConfigPBCParams(
                gaussian_sigma=0.01,
                bond_detect_factor=0.5,
            ),
        )
        self.assertEqual(no_bonds.bond_count, 0)

        from NepTrainKit.core.torsion_guard_pbc import (
            bonds_within_range_nonpbc,
        )

        stretched = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.asarray([0.5, 0.5])
        self.assertTrue(
            bonds_within_range_nonpbc(
                stretched,
                [(0, 1)],
                radii,
                min_factor=0.0,
                max_factor=None,
                detect_factor=1.15,
            )
        )
        self.assertFalse(
            bonds_within_range_nonpbc(
                stretched,
                [(0, 1)],
                radii,
                min_factor=0.0,
                max_factor=1.5,
                detect_factor=1.15,
            )
        )

        mixed = molecule.copy()
        mixed.set_cell([8, 8, 8])
        mixed.set_pbc([True, True, False])
        with self.assertRaises(CardOperationError) as mixed_error:
            operation.run_structure(mixed, params)
        self.assertEqual(mixed_error.exception.code, "organic-mixed-pbc")

        impossible = Atoms(
            "H2",
            positions=[[0, 0, 0], [0.7, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        )
        with self.assertRaises(CardOperationError) as guard_error:
            operation.run_structure(
                impossible,
                OrganicMolConfigPBCParams(
                    perturb_per_frame=2,
                    gaussian_sigma=0.01,
                    bond_keep_min_factor=2.0,
                    max_retries=0,
                    use_seed=True,
                    seed=2,
                ),
            )
        self.assertEqual(
            guard_error.exception.code, "organic-all-guards-failed"
        )

    def test_organic_configuration_advanced_params_reach_topology_and_roundtrip(self):
        molecule = Atoms(
            "C4",
            positions=[
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [2.8, 0.2, 0.0],
                [4.1, 0.2, 1.0],
            ],
            pbc=False,
        )
        params = OrganicMolConfigPBCParams(
            perturb_per_frame=3,
            torsion_range_deg=(-80.0, 100.0),
            max_torsions_per_conf=2,
            gaussian_sigma=0.01,
            pbc_mode="no",
            local_cutoff=3,
            local_subtree=2,
            bond_detect_factor=1.2,
            bond_keep_min_factor=0.55,
            bond_keep_max_factor=1.35,
            bond_keep_max_enable=True,
            nonbond_min_factor=0.75,
            max_retries=5,
            mult_bond_factor=0.82,
            nonpbc_box_size=75.0,
            bo_c_const=0.35,
            bo_threshold=0.1,
            use_seed=True,
            seed=47,
        )
        settings = OrganicMolConfigPBCOperation._validated_settings(
            molecule,
            params,
        )
        summary = OrganicMolConfigPBCOperation.topology_summary(
            molecule,
            params,
        )

        self.assertEqual(settings["local_cutoff"], 3)
        self.assertEqual(settings["local_subtree"], 2)
        self.assertEqual(settings["bond_max"], 1.35)
        self.assertEqual(settings["nonbond_min"], 0.75)
        self.assertEqual(settings["mult_bond"], 0.82)
        self.assertEqual(settings["box_size"], 75.0)
        self.assertEqual(settings["bo_c"], 0.35)
        self.assertEqual(settings["bo_threshold"], 0.1)
        self.assertTrue(summary.local_mode)
        self.assertEqual(summary.requested_outputs, 3)

        card = OrganicMolConfigPBCCard()
        card.set_params(params)
        restored = OrganicMolConfigPBCCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), params)

    def test_organic_configuration_unwraps_pbc_spanning_molecule_before_rotation(self):
        molecule = Atoms(
            "C4",
            positions=[
                [8.3, 5.0, 5.0],
                [9.7, 5.0, 5.0],
                [1.1, 5.2, 5.0],
                [2.4, 5.2, 6.0],
            ],
            cell=[10, 10, 10],
            pbc=True,
        )
        output = OrganicMolConfigPBCOperation().run_structure(
            molecule,
            OrganicMolConfigPBCParams(
                perturb_per_frame=1,
                torsion_range_deg=(75.0, 75.0),
                max_torsions_per_conf=1,
                gaussian_sigma=0.0,
                pbc_mode="auto",
                use_seed=True,
                seed=7,
            ),
        )[0]

        distances = output.get_all_distances(mic=True)
        self.assertLess(distances[0, 1], 1.7)
        self.assertLess(distances[1, 2], 1.7)
        self.assertLess(distances[2, 3], 1.7)
        self.assertFalse(np.allclose(output.positions, molecule.positions))

    def test_organic_configuration_preserves_triclinic_pbc_molecule(self):
        cell = np.asarray(
            [
                [10.0, 0.0, 0.0],
                [2.0, 9.0, 0.0],
                [1.0, 1.0, 8.0],
            ]
        )
        molecule = Atoms(
            "C4",
            positions=[
                [8.3, 4.0, 4.0],
                [9.7, 4.0, 4.0],
                [11.1, 4.2, 4.0],
                [12.4, 4.2, 5.0],
            ],
            cell=cell,
            pbc=True,
        )
        molecule.wrap()
        molecule.info["Config_type"] = "triclinic_input"
        molecule.info["source"] = "triclinic_oracle"
        molecule.new_array("group", np.asarray([1, 1, 2, 2]))
        params = OrganicMolConfigPBCParams(
            perturb_per_frame=1,
            torsion_range_deg=(55.0, 55.0),
            max_torsions_per_conf=1,
            gaussian_sigma=0.0,
            pbc_mode="auto",
            use_seed=True,
            seed=3,
        )
        operation = OrganicMolConfigPBCOperation()

        summary = operation.topology_summary(molecule, params)
        output = operation.run_structure(molecule, params)[0]

        self.assertEqual(summary.bond_count, 3)
        self.assertEqual(summary.component_count, 1)
        self.assertEqual(summary.torsion_count, 1)
        input_bond_lengths = np.asarray(
            [molecule.get_distance(i, i + 1, mic=True) for i in range(3)]
        )
        output_bond_lengths = np.asarray(
            [output.get_distance(i, i + 1, mic=True) for i in range(3)]
        )
        self.assertTrue(np.allclose(output_bond_lengths, input_bond_lengths))
        self.assertTrue(np.all(output_bond_lengths < 1.7))
        self.assertFalse(np.allclose(output.positions, molecule.positions))
        self.assertTrue(np.allclose(output.cell.array, molecule.cell.array))
        self.assertTrue(np.array_equal(output.pbc, molecule.pbc))
        self.assertTrue(
            np.array_equal(output.arrays["group"], molecule.arrays["group"])
        )
        self.assertEqual(output.info["source"], "triclinic_oracle")
        self.assertIn(
            "triclinic_input|TG(req=1,ok=1,sig=0,pbc=auto)",
            output.info["Config_type"],
        )

    def test_organic_configuration_uses_general_triclinic_minimum_image(self):
        cell = np.asarray(
            [
                [8.0, 0.0, 0.0],
                [7.0, 2.0, 0.0],
                [0.0, 0.0, 8.0],
            ]
        )
        base = np.asarray([4.0, 1.0, 4.0])
        boundary_jump = np.asarray([-3.8345117, -6.86969904, -7.80144916])
        molecule = Atoms(
            "C4",
            positions=[
                base,
                base + boundary_jump,
                base + boundary_jump + [1.4, 0.2, 0.0],
                base + boundary_jump + [2.7, 0.2, 1.0],
            ],
            cell=cell,
            pbc=True,
        )
        molecule.wrap()
        delta = molecule.positions[1] - molecule.positions[0]
        fractional = delta @ np.linalg.inv(cell)
        component_wrapped = (fractional - np.round(fractional)) @ cell
        ase_mic, ase_length = find_mic(delta, cell, pbc=True)
        from NepTrainKit.core.torsion_guard_pbc import mic_delta

        resolved_batch = mic_delta(
            np.asarray([delta, -delta]),
            cell,
            np.linalg.inv(cell),
        )
        ase_batch, _ase_lengths = find_mic(
            np.asarray([delta, -delta]),
            cell,
            pbc=True,
        )
        self.assertGreater(np.linalg.norm(component_wrapped), 6.0)
        self.assertAlmostEqual(float(ase_length), 1.1594779315, places=8)
        self.assertTrue(np.allclose(resolved_batch[0], ase_mic))
        self.assertTrue(np.allclose(resolved_batch, ase_batch))

        params = OrganicMolConfigPBCParams(
            perturb_per_frame=1,
            torsion_range_deg=(55.0, 55.0),
            max_torsions_per_conf=1,
            gaussian_sigma=0.0,
            pbc_mode="auto",
            bond_keep_max_enable=True,
            bond_keep_max_factor=1.15,
            use_seed=True,
            seed=3,
        )
        operation = OrganicMolConfigPBCOperation()

        summary = operation.topology_summary(molecule, params)
        output = operation.run_structure(molecule, params)[0]

        self.assertEqual(summary.bond_count, 3)
        self.assertEqual(summary.component_count, 1)
        self.assertEqual(summary.torsion_count, 1)
        input_bond_lengths = np.asarray(
            [molecule.get_distance(i, i + 1, mic=True) for i in range(3)]
        )
        output_bond_lengths = np.asarray(
            [output.get_distance(i, i + 1, mic=True) for i in range(3)]
        )
        self.assertTrue(np.allclose(output_bond_lengths, input_bond_lengths))
        self.assertTrue(np.all(output_bond_lengths < 1.7))
        resolved_delta = output.positions[1] - output.positions[0]
        resolved_mic, _resolved_length = find_mic(
            resolved_delta,
            output.cell.array,
            pbc=True,
        )
        self.assertTrue(
            np.isclose(np.linalg.norm(resolved_mic), output_bond_lengths[0])
        )

    def test_organic_configuration_card_roundtrip(self):
        card = OrganicMolConfigPBCCard()
        card.perturb_frame.set_input_value([3])
        card.torsion_frame.set_input_value([-45.0, 60.0])
        card.max_torsions_frame.set_input_value([2])
        card.sigma_frame.set_input_value([0.02])
        card.pbc_combo.setCurrentIndex(2)
        card.local_cut_frame.set_input_value([80])
        card.local_sub_frame.set_input_value([30])
        card.bond_max_enable.setChecked(True)
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([11])

        restored = OrganicMolConfigPBCCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())
