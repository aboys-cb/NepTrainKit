"""Tests for the Interface Layer Mixing card (界面随机互混)."""

from __future__ import annotations

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
        with self.assertRaisesRegex(CardOperationError, "must be in \\[0, 1\\]"):
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
        self.assertEqual(
            card.summary_label.text(),
            "Preview appears after attaching an input dataset.",
        )

        card.set_dataset([_bilayer()])
        text = card.summary_label.text()
        self.assertIn("c-axis interface @ 0.500", text)
        self.assertIn("L=2 (Al)", text)
        self.assertIn("R=2 (Ni)", text)
        self.assertIn("c_max=1", text)
        self.assertIn("outputs: 1", text)

    def test_ui_live_summary_tracks_param_changes_and_mode_combo(self):
        card = InterfaceLayerMixCard()
        card.set_dataset([_bilayer()])
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
        self.assertIn("outputs: 4", card.summary_label.text())
        self.assertTrue(card.concentration_frame.isHidden())
        self.assertFalse(card.gradient_container.isHidden())

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
        card.set_dataset([_bilayer()])
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
        self.assertIn("c_max=1", card.summary_label.text())
        self.assertIn("outputs: 3", card.summary_label.text())

        outputs = card.create_operation().run_structure(_bilayer(), card.get_params())
        self.assertEqual(len(outputs), 3)
        baseline = np.asarray(_bilayer().get_chemical_symbols())
        self.assertEqual(
            int(np.sum(np.asarray(outputs[0].get_chemical_symbols()) != baseline)),
            8,
        )

    def test_ui_live_summary_reports_resolution_error(self):
        card = InterfaceLayerMixCard()
        card.set_dataset(
            [
                Atoms(
                    "Al4",
                    scaled_positions=[[0.0, 0.0, 0.1], [0.5, 0.5, 0.1], [0.0, 0.0, 0.7], [0.5, 0.5, 0.7]],
                    cell=np.diag([5.0, 5.0, 5.0]),
                    pbc=True,
                )
            ]
        )
        self.assertTrue(
            card.summary_label.text().startswith("Preview unavailable:"),
            card.summary_label.text(),
        )

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
