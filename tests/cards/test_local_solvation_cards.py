"""Local solvation card."""

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


class TestLocalSolvationCards(BaseCardTest):
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
        with self.assertRaisesRegex(ValueError, "does not match any host atoms"):
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

        with self.assertRaisesRegex(ValueError, "no solvent-accessible void"):
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
        with self.assertRaisesRegex(ValueError, "No solvent molecule"):
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
        self.assertTrue(card.solvent_section.isHidden())
        self.assertTrue(card.collision_section.isHidden())
        self.assertTrue(card.elements_field.isHidden())
        self.assertIn("No input loaded", card.preview_label.text())

        card.center_mode_combo.setCurrentIndex(
            card.center_mode_combo.findData("elements")
        )
        self.assertFalse(card.elements_field.isHidden())
        card.elements_edit.setText("Ca")
        card.set_dataset([Atoms("Ca", positions=[[0, 0, 0]], pbc=False)])
        self.assertIn("1 center(s) (Ca)", card.preview_label.text())
        self.assertIn("Supported ion hydration", card.preview_label.text())
        self.assertIn("Ca 2.3–2.6 Å", card.placement_note.text())
        self.assertTrue(card.shell_field.isHidden())

        card.advanced_checkbox.setChecked(True)
        self.assertFalse(card.collision_section.isHidden())
        self.assertFalse(card.box_section.isHidden())
        self.assertFalse(card.box_size_field.isHidden())
        self.assertTrue(card.box_field.isHidden())
        card.auto_box_checkbox.setChecked(True)
        self.assertTrue(card.box_size_field.isHidden())
        self.assertFalse(card.box_field.isHidden())

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

        with self.assertRaisesRegex(ValueError, "Independent outputs per input must be an integer"):
            operation.run_structure(
                periodic,
                LocalSolvationParams(structures=1.5),  # type: ignore[arg-type]
            )
        invalid_periodic = Atoms(
            "Ca",
            positions=[[0, 0, 0]],
            pbc=True,
        )
        with self.assertRaisesRegex(ValueError, "Periodic input"):
            operation.run_structure(
                invalid_periodic,
                LocalSolvationParams(solvent_count=1),
            )

    def test_local_solvation_uses_exact_triclinic_minimum_images(self):
        cell = np.array(
            [[8.0, 0.0, 0.0], [7.0, 2.0, 0.0], [0.0, 0.0, 8.0]],
            dtype=float,
        )
        pbc = np.array([True, True, True])
        delta = np.array([2.191387, -3.68341258, -7.34442362])
        reference = find_mic(delta, cell, pbc=pbc)[0]
        self.assertLess(float(np.linalg.norm(reference)), 1.2)

        self.assertTrue(
            has_collision(
                ["H"],
                np.zeros((1, 3)),
                ["H"],
                np.asarray([-delta]),
                cell=cell,
                pbc=pbc,
                collision_scale=0.0,
                min_distance=1.2,
            )
        )
        spatial_hash = PeriodicSpatialHash(
            ["H"], np.asarray([-delta]), cell=cell, pbc=pbc, cutoff=1.2
        )
        self.assertTrue(
            spatial_hash.has_collision(
                ["H"],
                np.zeros((1, 3)),
                collision_scale=0.0,
                min_distance=1.2,
            )
        )

        mixed_pbc = np.array([True, False, True])
        mixed_delta = np.array([-3.8345117, -6.86969904, -7.80144916])
        mixed_reference = min(
            (
                mixed_delta - i * cell[0] - k * cell[2]
                for i in range(-5, 6)
                for k in range(-5, 6)
            ),
            key=np.linalg.norm,
        )
        mixed_hash = PeriodicSpatialHash(
            ["H"],
            np.asarray([-mixed_delta]),
            cell=cell,
            pbc=mixed_pbc,
            cutoff=8.0,
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(mixed_reference)), 7.869921684088001, places=9
        )
        self.assertTrue(
            mixed_hash.has_collision(
                ["H"],
                np.zeros((1, 3)),
                collision_scale=0.0,
                min_distance=8.0,
            )
        )

    def test_local_solvation_ion_hydration_is_fail_closed(self):
        operation = LocalSolvationOperation()
        methane = """5
methane
C 0 0 0
H 0.6 0.6 0.6
H -0.6 -0.6 0.6
H -0.6 0.6 -0.6
H 0.6 -0.6 -0.6
"""
        ca = Atoms("Ca", positions=[[0.0, 0.0, 0.0]], pbc=False)
        with self.assertRaises(CardOperationError) as nonwater:
            operation.placement_summary(
                ca,
                LocalSolvationParams(
                    solvent_xyz=methane,
                    sampling_mode="ion-water",
                    solvent_count=1,
                ),
            )
        self.assertEqual(
            nonwater.exception.code,
            "local-solvation-ion-water-requires-water",
        )

        fe = Atoms("Fe", positions=[[0.0, 0.0, 0.0]], pbc=False)
        auto_summary = operation.placement_summary(
            fe, LocalSolvationParams(sampling_mode="auto", solvent_count=1)
        )
        self.assertEqual(auto_summary["mode"], "water")
        self.assertEqual(auto_summary["ion_oxygen_ranges"], {})
        with self.assertRaises(CardOperationError) as unsupported:
            operation.placement_summary(
                fe,
                LocalSolvationParams(
                    sampling_mode="ion-water", solvent_count=1
                ),
            )
        self.assertEqual(
            unsupported.exception.code,
            "local-solvation-ion-water-requires-profile",
        )

        ca_summary = operation.placement_summary(
            ca, LocalSolvationParams(sampling_mode="auto", solvent_count=6)
        )
        self.assertEqual(ca_summary["mode"], "ion-water")
        self.assertEqual(ca_summary["first_shell_capacity"], 6)
        self.assertFalse(ca_summary["fallback_needed"])

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

