"""Solvent box card."""

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


class TestSolventBoxCards(BaseCardTest):
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
        self.assertIn("SolvBox(mode=general,req=2,ok=2)", first.info.get("Config_type", ""))

    def test_solvent_box_fill_preserves_molecules_along_nonperiodic_axes(self):
        structure = Atoms(cell=[10.0, 4.0, 10.0], pbc=[True, False, True])
        for seed in range(20):
            result = SolventBoxFillOperation().run_structure(
                structure,
                SolventBoxFillParams(
                    count_mode="fixed",
                    solvent_count=1,
                    min_distance=0.1,
                    use_seed=True,
                    seed=seed,
                ),
            )[0]

            self.assertTrue(np.array_equal(result.pbc, structure.pbc))
            self.assertLess(abs(result.positions[1, 1] - result.positions[0, 1]), 1.1)
            self.assertLess(abs(result.positions[2, 1] - result.positions[0, 1]), 1.1)
            self.assertAlmostEqual(result.get_distance(0, 1, mic=True), 0.9572, places=6)
            self.assertAlmostEqual(result.get_distance(0, 2, mic=True), 0.9572, places=6)

    def test_solvent_box_fill_ui_exposes_only_distinct_clearance_presets(self):
        card = SolventBoxFillCard()
        self.assertEqual(
            [card.mode_combo.itemData(index) for index in range(card.mode_combo.count())],
            ["loose", "general", "dense"],
        )
        structure = Atoms(cell=[12.0, 12.0, 12.0], pbc=True)
        for mode, expected_scale in (
            ("loose", 0.62),
            ("general", 0.70),
            ("dense", 0.78),
        ):
            card.mode_combo.setCurrentIndex(card.mode_combo.findData(mode))
            summary = SolventBoxFillOperation().capacity_summary(
                structure,
                card.get_params(),
            )
            self.assertEqual(summary["mode"], mode)
            self.assertAlmostEqual(summary["collision_scale"], expected_scale)
        card.from_dict(
            {
                "check_state": True,
                "params": params_to_dict(
                    SolventBoxFillParams(sampling_mode="water")
                )
            }
        )
        self.assertEqual(card.mode_combo.currentData(), "general")

        card.advanced_checkbox.setChecked(True)
        card.collision_frame.set_input_value([0.8])
        self.assertFalse(card.mode_field.isEnabled())
        self.assertTrue(card.collision_field.isEnabled())
        card.min_distance_frame.set_input_value([0.9])
        self.assertFalse(card.mode_field.isEnabled())
        self.assertFalse(card.collision_field.isEnabled())

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
        self.assertTrue(card.count_frame.isHidden())
        self.assertFalse(card.density_frame.isHidden())
        self.assertEqual(card.get_params(), SolventBoxFillParams())
        self.assertTrue(card.min_distance_frame.isHidden())
        self.assertIn("No input loaded", card.preview_label.text())

        card.set_dataset([structure])
        self.assertIn("cell 1000 Å³", card.preview_label.text())
        self.assertIn("Target 33", card.preview_label.text())

        card.count_mode_combo.setCurrentIndex(
            card.count_mode_combo.findData("density")
        )
        self.assertTrue(card.count_frame.isHidden())
        self.assertFalse(card.density_frame.isHidden())
        card.count_frame.set_input_value([9])
        card.density_frame.set_input_value([0.7])
        card.mode_combo.setCurrentIndex(card.mode_combo.findData("loose"))
        card.fill_packing_frame.set_input_value([0.8])
        self.assertIn("Full-cell density estimate", card.preview_label.text())

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
                count_mode="fixed",
                solvent_count=1,
                min_distance=0.8,
                use_seed=True,
                seed=4,
            ),
        )[0]

        self.assertEqual(len(output), 3)
        self.assertIn(
            "SolvBox(mode=general,req=1,ok=1)",
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
                    count_mode="fixed",
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
                    count_mode="fixed",
                    solvent_count=1,
                    use_seed=True,
                    seed=2,
                ),
            )

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
                count_mode="fixed",
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
                    count_mode="fixed",
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
                count_mode="fixed",
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

