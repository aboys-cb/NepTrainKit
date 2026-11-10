"""Organic configuration card."""

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


class TestOrganicConfigurationCards(BaseCardTest):
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
            self.assertEqual(card.get_params(), OrganicMolConfigPBCParams())
            self.assertIn("up to 20 outputs", card.preview_label.text())
            self.assertIn("dataset maximum 20", card.preview_label.text())

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
