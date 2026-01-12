#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
from ase.io import read
from PySide6.QtWidgets import QApplication
import json

from NepTrainKit.ui.views._card import (
    SuperCellCard,
    CrystalPrototypeBuilderCard,
    PerturbCard,
    CellScalingCard,
    CellStrainCard,
    ShearMatrixCard,
    ShearAngleCard,
    RandomSlabCard,
    RandomDopingCard,
    CompositionSweepCard,
    RandomOccupancyCard,
    MagneticOrderCard,
    MagneticMomentRotationCard,
    GroupLabelCard,
    RandomVacancyCard,
    VacancyDefectCard,
    StackingFaultCard,
    OrganicMolConfigPBCCard,
    # VibrationModePerturbCard,
)


class TestCard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])
        cls.test_dir = Path(__file__).parent
        cls.base_structure = read(cls.test_dir / "data" / "Si2.vasp")
        cls.base_structure.info.setdefault("Config_type", "Si2")

    @classmethod
    def tearDownClass(cls):
        if cls._app is not None:
            cls._app.quit()
            cls._app = None

    def setUp(self):
        np.random.seed(0)
        random.seed(0)
        self.structure = self.base_structure.copy()

    def test_supercell_card_variants(self):
        card = SuperCellCard()
        structure = self.structure.copy()
        original_cell = np.array(structure.get_cell())

        card.super_scale_condition_frame.set_input_value([2, 1, 1])
        card.super_scale_radio_button.setChecked(True)
        card.super_cell_radio_button.setChecked(False)
        card.max_atoms_radio_button.setChecked(False)
        card.behavior_type_combo.setCurrentIndex(0)

        direct_results = card.process_structure(structure)
        self.assertEqual(len(direct_results), 1)
        self.assertEqual(len(direct_results[0]), len(structure) * 2)
        new_cell = np.array(direct_results[0].get_cell())
        self.assertGreater(np.linalg.norm(new_cell[0]), np.linalg.norm(original_cell[0]))

        card.super_cell_radio_button.setChecked(True)
        card.super_scale_radio_button.setChecked(False)
        card.max_atoms_radio_button.setChecked(False)
        lengths = structure.cell.lengths()
        card.super_cell_condition_frame.set_input_value([
            lengths[0] * 2.1,
            lengths[1] * 1.1,
            lengths[2] * 1.1,
        ])

        cell_results = card.process_structure(structure)
        self.assertEqual(len(cell_results), 1)
        self.assertGreater(len(cell_results[0]), len(structure))

        card.behavior_type_combo.setCurrentIndex(1)
        card.max_atoms_radio_button.setChecked(True)
        card.super_cell_radio_button.setChecked(False)
        card.super_scale_radio_button.setChecked(False)
        card.max_atoms_condition_frame.set_input_value([len(structure) * 2])

        atoms_results = card.process_structure(structure)
        self.assertGreaterEqual(len(atoms_results), 1)
        self.assertTrue(any(len(atoms) > len(structure) for atoms in atoms_results))

    def test_perturb_card_with_organic(self):
        card = PerturbCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.engine_type_combo.setCurrentIndex(1)
        card.scaling_condition_frame.set_input_value([0.1])
        card.num_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        displacements = [
            np.linalg.norm(atoms.get_positions() - structure.get_positions(), axis=1).max()
            for atoms in results
        ]
        self.assertTrue(any(delta > 0 for delta in displacements))

    def test_cell_scaling_card_options(self):
        card = CellScalingCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.perturb_angle_checkbox.setChecked(False)
        card.engine_type_combo.setCurrentIndex(1)
        card.scaling_condition_frame.set_input_value([0.05])
        card.num_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        original_lengths = np.array(structure.cell.cellpar()[:3])
        self.assertTrue(
            any(
                not np.allclose(
                    np.array(atoms.cell.cellpar()[:3]),
                    original_lengths,
                    atol=1e-6,
                )
                for atoms in results
            )
        )

    def test_cell_strain_card_uniaxial(self):
        card = CellStrainCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.engine_type_combo.setText("uniaxial")
        card.strain_x_frame.set_input_value([1.0, 1.0, 1.0])
        card.strain_y_frame.set_input_value([0.0, 0.0, 1.0])
        card.strain_z_frame.set_input_value([0.0, 0.0, 1.0])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 3)
        original_lengths = np.array(structure.cell.cellpar()[:3])
        self.assertTrue(
            any(
                not np.allclose(
                    np.array(atoms.cell.cellpar()[:3]),
                    original_lengths,
                    atol=1e-6,
                )
                for atoms in results
            )
        )

    def test_shear_matrix_card(self):
        card = ShearMatrixCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.symmetric_checkbox.setChecked(False)
        card.xy_frame.set_input_value([1.0, 1.0, 1.0])
        card.yz_frame.set_input_value([0.0, 0.0, 1.0])
        card.xz_frame.set_input_value([0.0, 0.0, 1.0])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 1)
        self.assertFalse(
            np.allclose(
                np.array(results[0].get_cell()),
                np.array(structure.get_cell()),
                atol=1e-6,
            )
        )

    def test_shear_angle_card(self):
        card = ShearAngleCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.alpha_frame.set_input_value([1.0, 1.0, 1.0])
        card.beta_frame.set_input_value([0.0, 0.0, 1.0])
        card.gamma_frame.set_input_value([0.0, 0.0, 1.0])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 1)
        original_angles = np.array(structure.cell.cellpar()[3:])
        new_angles = np.array(results[0].cell.cellpar()[3:])
        self.assertFalse(np.allclose(new_angles, original_angles, atol=1e-6))

    def test_random_slab_card(self):
        card = RandomSlabCard()
        structure = self.structure.copy()
        card.h_frame.set_input_value([1, 1, 1])
        card.k_frame.set_input_value([0, 0, 1])
        card.l_frame.set_input_value([0, 0, 1])
        card.layer_frame.set_input_value([1, 1, 1])
        card.vacuum_frame.set_input_value([0, 0, 1])

        results = card.process_structure(structure)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(len(atoms) >= len(structure) for atoms in results))

    def test_random_doping_card(self):
        card = RandomDopingCard()
        structure = self.structure.copy()
        rules = [{
            "target": "Si",
            "dopants": {"Ge": 1.0},
            "use": "count",
            "count": [1, 1],
            "concentration": [0.0, 1.0],
        }]
        card.rules_widget.from_rules(rules)
        card.doping_type_combo.setCurrentText("Exact")
        card.max_atoms_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        for atoms in results:
            self.assertIn("Ge", atoms.get_chemical_symbols())

    def test_crystal_prototype_builder_card(self):
        card = CrystalPrototypeBuilderCard()
        card.structure_combo.setCurrentText("fcc")
        card.element_edit.setText("Cu")
        card.a_frame.set_input_value([3.6, 3.6, 0.1])
        card.auto_supercell_button.setChecked(True)
        card.manual_supercell_button.setChecked(False)
        card.max_atoms_frame.set_input_value([64])
        card.max_output_frame.set_input_value([10])

        results = card.process_structure(self.structure.copy())
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(atoms.pbc.all() for atoms in results))
        self.assertTrue(all(len(atoms) <= 64 for atoms in results))

    def test_composition_sweep_and_random_occupancy_cards(self):
        base = self.structure.copy()
        base.info.setdefault("Config_type", "base")

        sweep = CompositionSweepCard()
        sweep.elements_edit.setText("Co,Ni")
        sweep.order_combo.setCurrentText("2")
        sweep.method_combo.setCurrentText("Grid")
        sweep.step_frame.set_input_value([0.5])
        sweep.include_endpoints_checkbox.setChecked(True)
        sweep.minfrac_frame.set_input_value([0.0])
        sweep.max_output_frame.set_input_value([10])

        swept = sweep.process_structure(base)
        self.assertEqual(len(swept), 3)
        self.assertTrue(all("alloy_composition" in atoms.info for atoms in swept))

        occ = RandomOccupancyCard()
        occ.source_combo.setCurrentText("From info (alloy_composition)")
        occ.mode_combo.setCurrentText("Exact")
        occ.samples_frame.set_input_value([1])

        occupied = []
        for atoms in swept:
            occupied.extend(occ.process_structure(atoms))
        self.assertEqual(len(occupied), len(swept))
        for atoms in occupied:
            syms = set(atoms.get_chemical_symbols())
            self.assertTrue(syms.issubset({"Co", "Ni"}))

    def test_composition_sweep_quaternary_quinary(self):
        base = self.structure.copy()
        base.info.setdefault("Config_type", "base")

        sweep4 = CompositionSweepCard()
        sweep4.elements_edit.setText("Co,Cr,Ni,Al,Fe")
        sweep4.order_combo.setCurrentText("4")
        sweep4.method_combo.setCurrentText("Sobol")
        sweep4.n_points_frame.set_input_value([8])
        sweep4.max_output_frame.set_input_value([8])

        swept4 = sweep4.process_structure(base)
        self.assertEqual(len(swept4), 8)
        for atoms in swept4:
            comp = json.loads(atoms.info["alloy_composition"])
            self.assertEqual(len(comp), 4)

        sweep5 = CompositionSweepCard()
        sweep5.elements_edit.setText("Co,Cr,Ni,Al,Fe")
        sweep5.order_combo.setCurrentText("5")
        sweep5.method_combo.setCurrentText("Sobol")
        sweep5.n_points_frame.set_input_value([6])
        sweep5.max_output_frame.set_input_value([6])

        swept5 = sweep5.process_structure(base)
        self.assertEqual(len(swept5), 6)
        for atoms in swept5:
            comp = json.loads(atoms.info["alloy_composition"])
            self.assertEqual(len(comp), 5)

    def test_magnetic_order_card_fm_afm(self):
        proto = CrystalPrototypeBuilderCard()
        proto.structure_combo.setCurrentText("bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.manual_supercell_button.setChecked(True)
        proto.auto_supercell_button.setChecked(False)
        proto.rep_frame.set_input_value([2, 2, 2])
        proto.max_output_frame.set_input_value([1])
        base = proto.process_structure(self.structure.copy())[0]

        card = MagneticOrderCard()
        card.map_edit.setText("Fe:2.2")
        card.fm_checkbox.setChecked(True)
        card.afm_checkbox.setChecked(True)
        card.kvec_combo.setCurrentText("100")
        card.pm_checkbox.setChecked(False)

        results = card.process_structure(base)
        self.assertEqual(len(results), 2)
        fm = [a for a in results if a.info.get("mag_order") == "FM"][0]
        afm = [a for a in results if str(a.info.get("mag_order", "")).startswith("AFM")][0]

        fm_m = np.array(fm.get_initial_magnetic_moments(), dtype=float)
        afm_m = np.array(afm.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(np.all(fm_m >= 0))
        self.assertTrue(np.any(afm_m > 0) and np.any(afm_m < 0))

    def test_group_label_card_and_group_afm(self):
        proto = CrystalPrototypeBuilderCard()
        proto.structure_combo.setCurrentText("bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.manual_supercell_button.setChecked(True)
        proto.auto_supercell_button.setChecked(False)
        proto.rep_frame.set_input_value([2, 2, 2])
        proto.max_output_frame.set_input_value([1])
        base = proto.process_structure(self.structure.copy())[0]

        gl = GroupLabelCard()
        gl.mode_combo.setCurrentText("k-vector layers (recommended)")
        gl.kvec_combo.setCurrentText("111")
        gl.group_a_edit.setText("A")
        gl.group_b_edit.setText("B")
        labeled = gl.process_structure(base)[0]
        self.assertIn("group", labeled.arrays)
        groups = set(str(g) for g in labeled.arrays["group"])
        self.assertTrue({"A", "B"}.issubset(groups))

        mag = MagneticOrderCard()
        mag.format_combo.setCurrentText("Collinear (scalar)")
        mag.map_edit.setText("Fe:2.2")
        mag.fm_checkbox.setChecked(False)
        mag.afm_checkbox.setChecked(True)
        mag.afm_mode_combo.setCurrentText("group A/B")
        mag.group_a_edit.setText("A")
        mag.group_b_edit.setText("B")
        mag.pm_checkbox.setChecked(False)
        res = mag.process_structure(labeled)
        self.assertEqual(len(res), 1)
        afm = res[0]
        m = np.array(afm.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(np.any(m > 0) and np.any(m < 0))

    def test_magnetic_order_card_noncollinear_pm(self):
        proto = CrystalPrototypeBuilderCard()
        proto.structure_combo.setCurrentText("fcc")
        proto.element_edit.setText("Ni")
        proto.a_frame.set_input_value([3.5, 3.5, 0.1])
        proto.manual_supercell_button.setChecked(True)
        proto.auto_supercell_button.setChecked(False)
        proto.rep_frame.set_input_value([2, 2, 2])
        proto.max_output_frame.set_input_value([1])
        base = proto.process_structure(self.structure.copy())[0]

        card = MagneticOrderCard()
        card.format_combo.setCurrentText("Non-collinear (vector)")
        card.axis_frame.set_input_value([0.0, 0.0, 1.0])
        card.map_edit.setText("Ni:0.6")
        card.fm_checkbox.setChecked(False)
        card.afm_checkbox.setChecked(False)
        card.pm_checkbox.setChecked(True)
        card.pm_count_frame.set_input_value([2])
        card.pm_direction_combo.setCurrentText("sphere")
        card.pm_balanced_checkbox.setChecked(True)
        card.seed_checkbox.setChecked(True)
        card.seed_frame.set_input_value([123])

        results = card.process_structure(base)
        self.assertEqual(len(results), 2)
        for atoms in results:
            m = np.array(atoms.get_initial_magnetic_moments(), dtype=float)
            self.assertEqual(m.ndim, 2)
            self.assertEqual(m.shape[1], 3)
            self.assertTrue(np.any(np.linalg.norm(m, axis=1) > 0))
            self.assertEqual(atoms.info.get("magmom_format"), "vector")

    def test_magmom_rotation_lifts_scalar_to_vector(self):
        proto = CrystalPrototypeBuilderCard()
        proto.structure_combo.setCurrentText("bcc")
        proto.element_edit.setText("Fe")
        proto.a_frame.set_input_value([2.9, 2.9, 0.1])
        proto.manual_supercell_button.setChecked(True)
        proto.auto_supercell_button.setChecked(False)
        proto.rep_frame.set_input_value([2, 2, 2])
        proto.max_output_frame.set_input_value([1])
        base = proto.process_structure(self.structure.copy())[0]

        order = MagneticOrderCard()
        order.format_combo.setCurrentText("Collinear (scalar)")
        order.map_edit.setText("Fe:2.2")
        order.fm_checkbox.setChecked(True)
        order.afm_checkbox.setChecked(False)
        order.pm_checkbox.setChecked(False)
        fm = order.process_structure(base)[0]

        rot = MagneticMomentRotationCard()
        rot.elements_input.setText("Fe")
        rot.angle_frame.set_input_value([45.0])
        rot.lift_scalar_checkbox.setChecked(True)
        rot.axis_frame.set_input_value([0.0, 0.0, 1.0])
        rot.count_frame.set_input_value([1])

        rotated = rot.process_structure(fm)[0]
        m = np.array(rotated.get_initial_magnetic_moments(), dtype=float)
        self.assertEqual(m.ndim, 2)
        self.assertEqual(m.shape[1], 3)

    def test_random_vacancy_card(self):
        card = RandomVacancyCard()
        structure = self.structure.copy()
        card.rules_widget.from_rules([
            {"element": "Si", "count": [1, 1]},
        ])
        card.max_atoms_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(atoms) == len(structure) - 1 for atoms in results))

    def test_vacancy_defect_card_concentration(self):
        card = VacancyDefectCard()
        structure = self.structure.copy()
        card.engine_type_combo.setCurrentIndex(1)
        card.concentration_radio_button.setChecked(True)
        card.num_radio_button.setChecked(False)
        card.concentration_condition_frame.set_input_value([0.6])
        card.num_condition_frame.set_input_value([1])
        card.max_atoms_condition_frame.set_input_value([2])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(len(atoms) < len(structure) for atoms in results))

    def test_stacking_fault_card(self):
        card = StackingFaultCard()
        structure = self.structure.copy()
        card.hkl_frame.set_input_value([1, 1, 1])
        card.layer_frame.set_input_value([1])
        card.step_frame.set_input_value([0.1, 0.1, 0.1])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 1)
        self.assertGreater(
            np.abs(results[0].get_positions() - structure.get_positions()).sum(),
            0.0,
        )

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

    def test_vibration_mode_perturb_card(self):
        return 
        card = VibrationModePerturbCard()
        structure = self.structure.copy()
        natoms = len(structure)
        n_modes = min(3 * natoms, 6)
        mode_vectors = np.zeros((n_modes, natoms, 3))
        for idx in range(n_modes):
            atom_index = idx % natoms
            component = idx % 3
            mode_vectors[idx, atom_index, component] = 1.0
        freq_values = np.linspace(50.0, 300.0, n_modes)
        for mode_idx in range(n_modes):
            structure.new_array(f"vibration_mode_{mode_idx}_x", mode_vectors[mode_idx, :, 0])
            structure.new_array(f"vibration_mode_{mode_idx}_y", mode_vectors[mode_idx, :, 1])
            structure.new_array(f"vibration_mode_{mode_idx}_z", mode_vectors[mode_idx, :, 2])
            structure.new_array(
                f"vibration_frequency_{mode_idx}",
                np.full(natoms, freq_values[mode_idx], dtype=float),
            )

        card.amplitude_frame.set_input_value([0.05])
        card.modes_frame.set_input_value([2])
        card.min_freq_frame.set_input_value([1.0])
        card.num_condition_frame.set_input_value([3])

        results = card.process_structure(structure)
        self.assertEqual(len(results), 3)
        base_positions = structure.get_positions()
        displacements = [
            np.linalg.norm(atoms.get_positions() - base_positions, axis=1).max()
            for atoms in results
        ]
        self.assertTrue(all(delta > 0 for delta in displacements))
        self.assertTrue(all(len(atoms) == len(structure) for atoms in results))


if __name__ == "__main__":
    unittest.main()
