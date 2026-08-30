"""Tests for the Interface Layer Mixing card (界面层互混)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from unittest import mock

import numpy as np
from ase import Atoms

from .card_test_base import *
from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.core.cards.alloy import (
    InterfaceLayerMixOperation,
    InterfaceLayerMixParams,
)
from NepTrainKit.ui.views._card.i18n_utils import set_combo_value
from NepTrainKit.ui.views._card.interface_layer_mix_card import InterfaceLayerMixCard


def _bilayer(n_xy: int = 2, cell=None, symbols=("Al", "Ni"), pbc=True):
    """Synthetic bilayer: 3 Al layers below, 3 Ni layers above, normal along c."""
    a_sym, b_sym = symbols
    cell = np.eye(3) * 8.0 if cell is None else np.asarray(cell, dtype=float)
    grid_x = np.linspace(0.0, 1.0, n_xy, endpoint=False)
    grid_y = np.linspace(0.0, 1.0, n_xy, endpoint=False)
    positions: list[list[float]] = []
    symbol_list: list[str] = []
    layer_spec = [(z, a_sym) for z in (0.1, 0.2, 0.3)] + [
        (z, b_sym) for z in (0.7, 0.8, 0.9)
    ]
    for z, sym in layer_spec:
        for x in grid_x:
            for y in grid_y:
                positions.append([x, y, z])
                symbol_list.append(sym)
    return Atoms(symbol_list, scaled_positions=positions, cell=cell, pbc=pbc)


def _mixed_bilayer():
    symbols = ["Al", "Al", "Al", "Ni", "Al", "Ni", "Ni", "Ni"]
    scaled = [
        *[[x, 0.0, 0.25] for x in (0.0, 0.25, 0.5, 0.75)],
        *[[x, 0.0, 0.75] for x in (0.0, 0.25, 0.5, 0.75)],
    ]
    return Atoms(symbols, scaled_positions=scaled, cell=np.eye(3) * 8.0, pbc=True)


class TestInterfaceLayerMixOperation(BaseCardTest):
    def test_auto_detects_interface_along_c(self):
        op = InterfaceLayerMixOperation()
        summary = op.interface_summary(
            _bilayer(),
            InterfaceLayerMixParams(axis="auto"),
        )

        self.assertEqual(summary["axis"], "c")
        self.assertAlmostEqual(summary["position"], 0.5)
        self.assertEqual(summary["left_layers_available"], 3)
        self.assertEqual(summary["right_layers_available"], 3)
        self.assertEqual(summary["left_formula"], "Al")
        self.assertEqual(summary["right_formula"], "Ni")
        # 2 near-interface layers per side, 4 atoms per layer
        self.assertEqual(summary["n_left"], 8)
        self.assertEqual(summary["n_right"], 8)
        self.assertEqual(summary["n_total"], 16)
        self.assertAlmostEqual(summary["c_max"], 1.0)

    def test_manual_axis_and_position_override_detection(self):
        op = InterfaceLayerMixOperation()
        summary = op.interface_summary(
            _bilayer(),
            InterfaceLayerMixParams(axis="c", auto_position=False, interface_position=0.5),
        )

        self.assertEqual(summary["position"], 0.5)
        self.assertAlmostEqual(summary["c_max"], 1.0)

        summary_a = op.interface_summary(
            _bilayer(),
            InterfaceLayerMixParams(
                axis="a",
                auto_position=False,
                interface_position=0.5,
                left_layers=1,
                right_layers=1,
            ),
        )
        self.assertEqual(summary_a["axis"], "a")

    def test_fixed_mode_keeps_strict_quantity_contract_and_reproduces(self):
        op = InterfaceLayerMixOperation()
        atoms = _bilayer()
        params = InterfaceLayerMixParams(
            left_layers=2,
            right_layers=2,
            mode="fixed",
            concentration=0.5,
            num_structures=3,
            use_seed=True,
            seed=11,
        )

        outputs = op.run_structure(atoms, params)

        self.assertEqual(len(outputs), 3)
        baseline = np.asarray(atoms.get_chemical_symbols())
        for out in outputs:
            new = np.asarray(out.get_chemical_symbols())
            self.assertEqual(int(np.sum(new != baseline)), 8)
            self.assertEqual(len(out), len(atoms))

        rerun = op.run_structure(atoms, params)
        for first, second in zip(outputs, rerun):
            self.assertEqual(first.get_chemical_symbols(), second.get_chemical_symbols())
        self.assertEqual(
            outputs[0].info["Config_type"],
            "IfaceMix(L=2,R=2,c=0.5,s=11)",
        )

    def test_concentration_zero_is_a_valid_empty_output(self):
        atoms = _bilayer()
        outputs = InterfaceLayerMixOperation().run_structure(
            atoms,
            InterfaceLayerMixParams(mode="fixed", concentration=0.0, use_seed=True, seed=1),
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].get_chemical_symbols(), atoms.get_chemical_symbols())

    def test_gradient_mode_interpolates_linearly(self):
        op = InterfaceLayerMixOperation()
        atoms = _bilayer()
        params = InterfaceLayerMixParams(
            mode="gradient",
            gradient_start=0.0,
            gradient_end=1.0,
            num_structures=5,
            use_seed=True,
            seed=3,
        )

        outputs = op.run_structure(atoms, params)

        self.assertEqual(len(outputs), 5)
        baseline = np.asarray(atoms.get_chemical_symbols())
        # expected changed pairs: round(c * n_total / 2), c in [0, .25, .5, .75, 1]
        expected_changed = [0, 4, 8, 12, 16]
        for out, n_changed in zip(outputs, expected_changed):
            new = np.asarray(out.get_chemical_symbols())
            self.assertEqual(int(np.sum(new != baseline)), n_changed)
        self.assertEqual(outputs[-1].info["Config_type"], "IfaceMix(L=2,R=2,c=1,s=7)")

    def test_c_max_validation_happens_before_any_rng(self):
        op = InterfaceLayerMixOperation()
        atoms = _bilayer()

        def _boom(*args, **kwargs):
            raise AssertionError("RNG reached before concentration validation")

        with mock.patch("NepTrainKit.core.cards.alloy.np.random.default_rng", _boom):
            # over-limit fixed concentration -> structured error, never RNG
            with self.assertRaisesRegex(CardOperationError, "swap capacity"):
                op.run_structure(
                    atoms,
                    InterfaceLayerMixParams(mode="fixed", concentration=1.1, use_seed=True, seed=0),
                )
            # over-limit gradient bound -> structured error, never RNG
            with self.assertRaisesRegex(CardOperationError, "swap capacity"):
                op.run_structure(
                    atoms,
                    InterfaceLayerMixParams(
                        mode="gradient",
                        gradient_start=1.1,
                        gradient_end=1.1,
                        use_seed=True,
                        seed=0,
                    ),
                )

    def test_seed_scan_at_intermediate_concentration_satisfies_contract(self):
        op = InterfaceLayerMixOperation()
        atoms = _bilayer()
        baseline = np.asarray(atoms.get_chemical_symbols())

        for seed in range(20):
            outputs = op.run_structure(
                atoms,
                InterfaceLayerMixParams(
                    mode="fixed",
                    concentration=0.5,
                    use_seed=True,
                    seed=seed,
                ),
            )
            self.assertEqual(len(outputs), 1)
            self.assertEqual(
                int(np.sum(np.asarray(outputs[0].get_chemical_symbols()) != baseline)),
                8,
            )

    def test_mixed_interface_swaps_only_unlike_pairs_for_20_seeds(self):
        op = InterfaceLayerMixOperation()
        atoms = _mixed_bilayer()
        baseline = np.asarray(atoms.get_chemical_symbols())
        params = InterfaceLayerMixParams(
            axis="c",
            auto_position=False,
            interface_position=0.5,
            left_layers=1,
            right_layers=1,
            concentration=0.5,
            use_seed=True,
        )

        for seed in range(20):
            out = op.run_structure(atoms, replace(params, seed=seed))[0]
            changed = np.flatnonzero(
                np.asarray(out.get_chemical_symbols()) != baseline
            )
            self.assertEqual(len(changed), 4)
            self.assertEqual(int(np.sum(changed < 4)), 2)
            self.assertEqual(int(np.sum(changed >= 4)), 2)
            self.assertEqual(
                Counter(out.get_chemical_symbols()),
                Counter(atoms.get_chemical_symbols()),
            )

    def test_capacity_counts_only_unlike_pairs_and_validates_before_rng(self):
        atoms = Atoms(
            ["Al", "Al", "Al", "Ni"] * 2,
            scaled_positions=[
                *[[x, 0.0, 0.25] for x in (0.0, 0.25, 0.5, 0.75)],
                *[[x, 0.0, 0.75] for x in (0.0, 0.25, 0.5, 0.75)],
            ],
            cell=np.eye(3) * 8.0,
            pbc=True,
        )
        params = InterfaceLayerMixParams(
            axis="c",
            auto_position=False,
            interface_position=0.5,
            left_layers=1,
            right_layers=1,
            concentration=0.75,
        )
        op = InterfaceLayerMixOperation()
        summary = op.interface_summary(atoms, replace(params, concentration=0.5))
        self.assertEqual(summary["pair_capacity"], 2)
        self.assertAlmostEqual(summary["c_max"], 0.5)

        with mock.patch(
            "NepTrainKit.core.cards.alloy.np.random.default_rng",
            side_effect=AssertionError("RNG reached"),
        ):
            with self.assertRaises(CardOperationError) as caught:
                op.run_structure(atoms, params)
        self.assertEqual(caught.exception.code, "interface.concentration_exceeds_max")

    def test_effective_concentration_is_previewed_and_tagged(self):
        atoms = _bilayer()
        params = InterfaceLayerMixParams(
            concentration=1.0 / 6.0,
            use_seed=True,
            seed=3,
        )
        op = InterfaceLayerMixOperation()
        summary = op.interface_summary(atoms, params)
        out = op.run_structure(atoms, params)[0]
        baseline = np.asarray(atoms.get_chemical_symbols())
        changed = int(
            np.sum(np.asarray(out.get_chemical_symbols()) != baseline)
        )

        self.assertEqual(changed, 2)
        self.assertAlmostEqual(summary["requested_concentrations"][0], 1.0 / 6.0)
        self.assertAlmostEqual(summary["effective_concentrations"][0], 0.125)
        self.assertIn("c=0.125", out.info["Config_type"])
        self.assertIn("target=0.167", out.info["Config_type"])

    def test_changed_output_drops_stale_labels_and_moves_species_arrays(self):
        atoms = _bilayer(n_xy=1)
        symbols = np.asarray(atoms.get_chemical_symbols())
        spin = np.where(symbols == "Al", 1.0, 2.0)
        initial = np.where(symbols == "Al", 10.0, 20.0)
        groups = np.arange(len(atoms), dtype=int)
        atoms.new_array("spin", spin.copy())
        atoms.set_initial_magnetic_moments(initial.copy())
        atoms.new_array("group", groups.copy())
        atoms.new_array("forces", np.ones((len(atoms), 3)))
        atoms.new_array("energies", np.arange(len(atoms), dtype=float))
        atoms.new_array("magmoms", initial.copy())
        atoms.info.update(
            energy=-5.0,
            free_energy=-5.1,
            stress=np.arange(6.0),
            virial=np.arange(9.0),
        )

        out = InterfaceLayerMixOperation().run_structure(
            atoms,
            InterfaceLayerMixParams(
                concentration=0.5, use_seed=True, seed=2
            ),
        )[0]

        self.assertIsNone(out.calc)
        for key in ("energy", "free_energy", "stress", "virial"):
            self.assertNotIn(key, out.info)
        for key in ("forces", "energies", "magmoms"):
            self.assertNotIn(key, out.arrays)
        out_symbols = np.asarray(out.get_chemical_symbols())
        np.testing.assert_array_equal(
            out.arrays["spin"], np.where(out_symbols == "Al", 1.0, 2.0)
        )
        np.testing.assert_array_equal(
            out.arrays["initial_magmoms"],
            np.where(out_symbols == "Al", 10.0, 20.0),
        )
        np.testing.assert_array_equal(out.arrays["group"], groups)
        self.assertIn("energy", atoms.info)
        self.assertIn("forces", atoms.arrays)

    def test_non_orthogonal_cell_and_mixed_pbc(self):
        slab_cell = np.array(
            [[3.0, 0.5, 0.0], [0.0, 3.2, 0.0], [0.0, 0.0, 4.0]],
            dtype=float,
        )
        op = InterfaceLayerMixOperation()
        summary = op.interface_summary(
            _bilayer(cell=slab_cell),
            InterfaceLayerMixParams(axis="auto"),
        )
        self.assertEqual(summary["axis"], "c")
        self.assertAlmostEqual(summary["position"], 0.5)
        self.assertEqual(summary["n_left"], 8)

        outputs = op.run_structure(
            _bilayer(cell=slab_cell, pbc=[True, True, False]),
            InterfaceLayerMixParams(mode="fixed", concentration=0.5, use_seed=True, seed=5),
        )
        self.assertEqual(len(outputs), 1)

    def test_layer_tolerance_uses_angstroms_in_different_cell_lengths(self):
        op = InterfaceLayerMixOperation()
        for height, rumple_fraction in ((10.0, 0.005), (100.0, 0.0005)):
            atoms = Atoms(
                ["Al", "Al", "Al", "Ni", "Ni", "Ni"],
                scaled_positions=[
                    [0.0, 0.0, 0.1],
                    [0.5, 0.0, 0.1 + rumple_fraction],
                    [0.0, 0.0, 0.3],
                    [0.0, 0.0, 0.7],
                    [0.5, 0.0, 0.7 + rumple_fraction],
                    [0.0, 0.0, 0.9],
                ],
                cell=np.diag([5.0, 5.0, height]),
                pbc=True,
            )
            params = InterfaceLayerMixParams(
                axis="c",
                auto_position=False,
                interface_position=0.5,
                left_layers=1,
                right_layers=1,
                layer_tolerance=0.1,
            )
            summary = op.interface_summary(atoms, params)
            self.assertEqual(summary["left_layers_available"], 2)
            self.assertEqual(summary["right_layers_available"], 2)

            split = op.interface_summary(
                atoms, replace(params, layer_tolerance=0.01)
            )
            self.assertEqual(split["left_layers_available"], 3)
            self.assertEqual(split["right_layers_available"], 3)

    def test_degenerate_inputs_fail_cleanly(self):
        op = InterfaceLayerMixOperation()

        empty = Atoms(cell=np.diag([5.0, 5.0, 5.0]), pbc=True)
        with self.assertRaisesRegex(CardOperationError, "at least two atoms"):
            op.run_structure(empty, InterfaceLayerMixParams())

        single = Atoms(
            "Al",
            scaled_positions=[[0.1, 0.1, 0.1]],
            cell=np.diag([5.0, 5.0, 5.0]),
            pbc=True,
        )
        with self.assertRaisesRegex(CardOperationError, "at least two atoms"):
            op.run_structure(single, InterfaceLayerMixParams())

        one_element = Atoms(
            "Al4",
            scaled_positions=[[0.0, 0.0, 0.1], [0.5, 0.5, 0.1], [0.0, 0.0, 0.7], [0.5, 0.5, 0.7]],
            cell=np.diag([5.0, 5.0, 5.0]),
            pbc=True,
        )
        with self.assertRaisesRegex(CardOperationError, "only one element"):
            op.run_structure(one_element, InterfaceLayerMixParams())

        # both selected near-interface regions are the same single element
        with self.assertRaisesRegex(CardOperationError, "same single element"):
            op.run_structure(
                Atoms(
                    "Al12Ni12",
                    scaled_positions=[
                        *[[x, y, 0.1] for x in (0.0, 0.5) for y in (0.0, 0.5)],
                        *[[x, y, 0.2] for x in (0.0, 0.5) for y in (0.0, 0.5)],
                        *[[x, y, 0.3] for x in (0.0, 0.5) for y in (0.0, 0.5)],
                        *[[x, y, 0.4] for x in (0.0, 0.5) for y in (0.0, 0.5)],
                        *[[x, y, 0.5] for x in (0.0, 0.5) for y in (0.0, 0.5)],
                        *[[x, y, 0.6] for x in (0.0, 0.5) for y in (0.0, 0.5)],
                    ],
                    cell=np.diag([5.0, 5.0, 5.0]),
                    pbc=True,
                ),
                InterfaceLayerMixParams(
                    axis="c",
                    auto_position=False,
                    interface_position=0.25,
                    left_layers=1,
                    right_layers=1,
                ),
            )

        with self.assertRaisesRegex(CardOperationError, "Not enough atomic layers"):
            op.run_structure(
                _bilayer(),
                InterfaceLayerMixParams(left_layers=5, right_layers=2),
            )
        with self.assertRaisesRegex(CardOperationError, "Not enough atomic layers"):
            op.run_structure(
                _bilayer(),
                InterfaceLayerMixParams(left_layers=2, right_layers=5),
            )

        with self.assertRaisesRegex(CardOperationError, "Interface axis"):
            op.run_structure(_bilayer(), InterfaceLayerMixParams(axis="q"))
        with self.assertRaisesRegex(CardOperationError, "strictly between 0 and 1"):
            op.run_structure(
                _bilayer(),
                InterfaceLayerMixParams(
                    axis="c",
                    auto_position=False,
                    interface_position=1.5,
                ),
            )
        with self.assertRaisesRegex(CardOperationError, "Concentration mode"):
            op.run_structure(_bilayer(), InterfaceLayerMixParams(mode="typo"))
        with self.assertRaisesRegex(CardOperationError, "Number of structures"):
            op.run_structure(_bilayer(), InterfaceLayerMixParams(num_structures=0))
        for tolerance in (0.0, -0.1, np.nan, np.inf):
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(CardOperationError) as caught:
                    op.run_structure(
                        _bilayer(),
                        InterfaceLayerMixParams(layer_tolerance=tolerance),
                    )
                self.assertEqual(caught.exception.code, "interface.layer_tolerance")
        with self.assertRaises(CardOperationError) as caught:
            op.run_structure(
                _bilayer(),
                InterfaceLayerMixParams(use_seed=True, seed=-1),
            )
        self.assertEqual(caught.exception.code, "interface.seed")

    def test_atom_positions_and_lattice_are_unchanged_by_swap(self):
        atoms = _bilayer()
        positions_before = np.array(atoms.positions)
        cell_before = np.array(atoms.cell.array)

        out = InterfaceLayerMixOperation().run_structure(
            atoms,
            InterfaceLayerMixParams(mode="fixed", concentration=1.0, use_seed=True, seed=2),
        )[0]

        np.testing.assert_array_equal(out.positions, positions_before)
        np.testing.assert_array_equal(np.asarray(out.cell.array), cell_before)


class TestInterfaceLayerMixCard(BaseCardTest):
    def test_card_registers_with_group_and_metadata(self):
        self.assertIn("InterfaceLayerMixCard", CardManager.card_info_dict)
        metadata = CardManager.get_card_metadata("InterfaceLayerMixCard")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.card_name, "Interface Layer Mixing")
        self.assertEqual(metadata.group, "Alloy")
        self.assertTrue(metadata.contributors)

    def test_online_doc_url_matches_doc_page(self):
        card = InterfaceLayerMixCard()
        self.assertEqual(
            card.get_online_doc_url(),
            f"{DOCS_BASE_URL}module/make-dataset-cards/cards/interface-layer-mix-card.html",
        )
        self.assertFalse(card.doc_button.isHidden())

    def test_params_roundtrip_through_ui_and_serialization(self):
        card = InterfaceLayerMixCard()
        self.assertEqual(card.get_params(), InterfaceLayerMixParams())

        params = InterfaceLayerMixParams(
            axis="a",
            auto_position=False,
            interface_position=0.25,
            layer_tolerance=0.15,
            left_layers=3,
            right_layers=1,
            mode="gradient",
            concentration=0.4,
            gradient_start=0.1,
            gradient_end=0.9,
            num_structures=7,
            use_seed=True,
            seed=42,
        )
        card.set_params(params)
        self.assertEqual(card.get_params(), params)

        restored = InterfaceLayerMixCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), params)
        self.assertEqual(set(card.to_dict()["params"]), set(card.get_params().__dict__))

    def test_ui_mode_combo_drives_the_matching_operation_branch(self):
        card = InterfaceLayerMixCard()
        params = InterfaceLayerMixParams(
            mode="gradient",
            gradient_start=0.0,
            gradient_end=1.0,
            num_structures=4,
            use_seed=True,
            seed=1,
        )
        card.set_params(params)
        outputs = card.create_operation().run_structure(_bilayer(), card.get_params())

        self.assertEqual(len(outputs), 4)
        baseline = np.asarray(_bilayer().get_chemical_symbols())
        changed = [int(np.sum(np.asarray(o.get_chemical_symbols()) != baseline)) for o in outputs]
        # gradient 0 -> 1 over 4 steps: c = 0, 1/3, 2/3, 1
        self.assertEqual(changed, [0, 6, 10, 16])

    def test_ui_live_summary_shows_hint_then_exact_readout(self):
        card = InterfaceLayerMixCard()
        self.assertIn("output(s)/input", card.get_summary_text())

        card.set_preview_structure(_bilayer())
        text = card.get_summary_text()
        self.assertIn("fractional c @ 0.500", text)
        self.assertIn("L 8/8 R sites", text)
        self.assertIn("realized 50%", text)

    def test_ui_live_summary_tracks_param_changes_and_mode_combo(self):
        card = InterfaceLayerMixCard()
        card.set_preview_structure(_bilayer())
        card.set_params(
            InterfaceLayerMixParams(
                mode="gradient",
                gradient_start=0.0,
                gradient_end=1.0,
                num_structures=4,
                use_seed=True,
                seed=1,
            )
        )
        card.set_preview_input_count(2)
        self.assertIn("2 × 4/input = 8 outputs", card.get_guidance_text())
        self.assertTrue(card.concentration_field.isHidden())
        self.assertFalse(card.gradient_start_field.isHidden())
        self.assertFalse(card.gradient_end_field.isHidden())

        # switch straight through the combo, read it back, and run the branch
        set_combo_value(card.mode_combo, "fixed")
        self.assertEqual(card.get_params().mode, "fixed")
        outputs = card.create_operation().run_structure(_bilayer(), card.get_params())
        baseline = np.asarray(_bilayer().get_chemical_symbols())
        self.assertEqual(
            int(np.sum(np.asarray(outputs[0].get_chemical_symbols()) != baseline)),
            8,
        )

    def test_ui_live_summary_corresponds_to_operation_run(self):
        card = InterfaceLayerMixCard()
        card.set_preview_structure(_bilayer())
        params = InterfaceLayerMixParams(
            left_layers=2,
            right_layers=2,
            mode="fixed",
            concentration=0.5,
            num_structures=3,
            use_seed=True,
            seed=11,
        )
        card.set_params(params)
        # the preview is exact: same capacity and output budget the run consumes
        self.assertIn("realized 50%", card.get_summary_text())
        self.assertIn("Outputs/input: 3", card.get_guidance_text())

        outputs = card.create_operation().run_structure(_bilayer(), card.get_params())
        self.assertEqual(len(outputs), 3)
        baseline = np.asarray(_bilayer().get_chemical_symbols())
        self.assertEqual(
            int(np.sum(np.asarray(outputs[0].get_chemical_symbols()) != baseline)),
            8,
        )

    def test_ui_live_summary_reports_resolution_error(self):
        card = InterfaceLayerMixCard()
        card.set_preview_structure(Atoms(
            "Al4",
            scaled_positions=[[0.0, 0.0, 0.1], [0.5, 0.5, 0.1], [0.0, 0.0, 0.7], [0.5, 0.5, 0.7]],
            cell=np.diag([5.0, 5.0, 5.0]),
            pbc=True,
        ))
        self.assertTrue(
            card.get_summary_text().startswith("Preview unavailable:"),
            card.get_summary_text(),
        )

    def test_manual_position_and_conditional_fields_restore_from_old_json(self):
        card = InterfaceLayerMixCard()
        card.from_dict({
            "check_state": True,
            "params": {
                "axis": "c",
                "auto_position": False,
                "interface_position": 0.25,
                "left_layers": 1,
                "right_layers": 3,
                "mode": "fixed",
                "concentration": 0.4,
                "gradient_start": 0.0,
                "gradient_end": 1.0,
                "num_structures": 2,
                "use_seed": True,
                "seed": 7,
            }
        })
        params = card.get_params()
        self.assertFalse(params.auto_position)
        self.assertAlmostEqual(params.interface_position, 0.25)
        self.assertAlmostEqual(params.layer_tolerance, 0.25)
        self.assertFalse(card.interface_position_field.isHidden())
        self.assertFalse(card.seed_field.isHidden())

        restored = InterfaceLayerMixCard()
        restored.from_dict(card.to_dict())
        self.assertEqual(restored.get_params(), params)

    def test_interface_layer_mix_translations_have_no_unfinished_entries(self):
        import xml.etree.ElementTree as ET

        ts_path = Path(__file__).resolve().parents[2] / "src" / "NepTrainKit" / "translations" / "neptrainkit_zh_CN.ts"
        root = ET.parse(ts_path).getroot()
        found = False
        for context in root.findall("context"):
            if context.findtext("name") != "InterfaceLayerMixCard":
                continue
            found = True
            for message in context.findall("message"):
                translation = message.find("translation")
                self.assertNotEqual(
                    translation.get("type"),
                    "unfinished",
                    f"unfinished translation for: {message.findtext('source')}",
                )
        self.assertTrue(found, "InterfaceLayerMixCard translation context missing")
