"""Composition sweep card."""

import json
import warnings
from unittest.mock import patch

from .card_test_base import *

from ase.calculators.singlepoint import SinglePointCalculator
from NepTrainKit.core.cards.alloy import sample_dopants
from NepTrainKit.core.cards.errors import CardOperationError


class TestCompositionSweepCards(BaseCardTest):
    def test_composition_sweep_budget_modes_fill_cap_and_cover_requested_orders(self):
        base = self.structure.copy()
        operation = CompositionSweepOperation()

        for budget_mode in ("Equal+Reflow", "Capacity-weighted"):
            with self.subTest(budget_mode=budget_mode):
                params = CompositionSweepParams(
                    elements="Co,Cr,Ni,Al",
                    order="2,4",
                    method="Grid",
                    step=0.5,
                    include_endpoints=True,
                    use_seed=True,
                    seed=13,
                    max_outputs=8,
                    budget_mode=budget_mode,
                )
                first = operation.run_structure(base, params)
                repeated = operation.run_structure(base, params)
                tags = [atoms.info.get("Config_type", "") for atoms in first]
                repeated_tags = [
                    atoms.info.get("Config_type", "") for atoms in repeated
                ]

                self.assertEqual(len(first), 8)
                self.assertEqual(tags, repeated_tags)
                summary = operation.sampling_summary(params)
                self.assertGreater(summary["emitted_by_order"][2], 0)
                self.assertGreater(summary["emitted_by_order"][4], 0)

    def test_composition_sweep_summary_deduplicates_boundary_compositions(self):
        operation = CompositionSweepOperation()
        summary = operation.sampling_summary(CompositionSweepParams())

        self.assertEqual(summary["active_orders"], (2, 3))
        self.assertEqual(summary["skipped_orders"], ())
        self.assertEqual(summary["outputs_per_input"], 66)
        keys = [composition for _order, composition in summary["targets"]]
        self.assertEqual(len(keys), len(set(keys)))

        card = CompositionSweepCard()
        card.set_preview_input_count(2)
        self.assertIn("132", card.get_guidance_text())
        self.assertIn("Random Occupancy", card.get_guidance_text())

        card.method_combo.setCurrentIndex(card.method_combo.findData("Sobol"))
        self.assertFalse(card.n_points_field.isHidden())
        self.assertTrue(card.step_field.isHidden())
        card.method_combo.setCurrentIndex(card.method_combo.findData("Grid"))
        self.assertFalse(card.step_field.isHidden())
        self.assertTrue(card.n_points_field.isHidden())

    def test_composition_sweep_rejects_invalid_or_empty_plans(self):
        operation = CompositionSweepOperation()
        cases = (
            CompositionSweepParams(elements="Co"),
            CompositionSweepParams(elements="Co,Xx"),
            CompositionSweepParams(elements="Co,Ni", order="5"),
            CompositionSweepParams(elements="Co,Ni", min_fraction=0.6),
            CompositionSweepParams(max_outputs=0),
            CompositionSweepParams(max_outputs=10_001),
            CompositionSweepParams(order="nonsense"),
            CompositionSweepParams(
                elements="Co,Cr,Ni,Al",
                order="4",
                method="Grid",
                step=0.3,
            ),
        )
        for params in cases:
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    operation.sampling_summary(params)

    def test_composition_sweep_dense_sampling_stays_unique_and_budget_bounded(self):
        operation = CompositionSweepOperation()
        dense_sobol = CompositionSweepParams(
            elements="Co,Ni",
            order="2",
            method="Sobol",
            n_points=999_999,
            use_seed=True,
            seed=11,
            max_outputs=1_000,
        )

        summary = operation.sampling_summary(dense_sobol)
        self.assertEqual(summary["outputs_per_input"], 1_000)
        outputs = operation.run_structure(self.structure, dense_sobol)
        tags = [atoms.info["Config_type"].rsplit("|", 1)[-1] for atoms in outputs]
        self.assertEqual(len(tags), len(set(tags)))

        with self.assertRaisesRegex(ValueError, "safe limit"):
            operation.sampling_summary(
                CompositionSweepParams(step=1.0e-6, max_outputs=1)
            )

        constrained = operation.sampling_summary(
            CompositionSweepParams(
                elements="Co,Cr,Ni,Al,Fe",
                order="5",
                method="Sobol",
                n_points=999_999,
                min_fraction=0.19,
                max_outputs=500,
            )
        )
        self.assertEqual(constrained["outputs_per_input"], 500)
        self.assertTrue(
            all(
                min(fraction for _element, fraction in composition) >= 0.19 - 1e-12
                for _order, composition in constrained["targets"]
            )
        )

    def test_composition_sweep_replaces_old_target_and_preserves_structure_arrays(self):
        base = self.structure.copy()
        base.info["Config_type"] = "base|Comp(Fe=1)|old"
        spin = np.arange(len(base) * 3, dtype=float).reshape(-1, 3)
        initial_magmoms = np.flip(spin, axis=1).copy()
        base.new_array("spin", spin)
        base.set_initial_magnetic_moments(initial_magmoms)
        params = CompositionSweepParams(
            elements="Co,Ni",
            order="2",
            method="Grid",
            step=0.5,
            max_outputs=3,
        )

        outputs = CompositionSweepOperation().run_structure(base, params)
        self.assertEqual(len(outputs), 3)
        for atoms in outputs:
            tags = str(atoms.info["Config_type"]).split("|")
            self.assertEqual(sum(tag.startswith("Comp(") for tag in tags), 1)
            self.assertNotIn("Comp(Fe=1)", tags)
            np.testing.assert_array_equal(atoms.positions, base.positions)
            np.testing.assert_array_equal(atoms.cell.array, base.cell.array)
            np.testing.assert_array_equal(atoms.arrays["spin"], spin)
            np.testing.assert_array_equal(
                atoms.arrays["initial_magmoms"], initial_magmoms
            )

        occupancy = RandomOccupancyOperation().run_structure(
            outputs[1], RandomOccupancyParams(source="Auto (Comp tag)")
        )[0]
        self.assertNotIn("Fe", occupancy.get_chemical_symbols())

    def test_composition_sweep_loads_legacy_flat_order_name(self):
        card = CompositionSweepCard()
        card.from_dict(
            {
                "class": "CompositionSweepCard",
                "check_state": True,
                "elements": "Co,Ni",
                "order": "Binary",
                "method": "Grid",
                "step": [0.5],
                "n_points": [20],
                "min_fraction": [0.0],
                "max_outputs": [10],
                "seed": [0],
            }
        )
        self.assertEqual(card.get_params().order, "2")

    def test_composition_sweep_quaternary_quinary(self):
        base = self.structure.copy()
        base.info.setdefault("Config_type", "base")

        sweep4 = CompositionSweepCard()
        sweep4.elements_edit.setText("Co,Cr,Ni,Al,Fe")
        sweep4.order_combo.setCurrentIndex(sweep4.order_combo.findData("4"))
        sweep4.method_combo.setCurrentText("Sobol")
        sweep4.n_points_frame.set_input_value([8])
        sweep4.max_output_frame.set_input_value([8])

        swept4 = sweep4.process_structure(base)
        self.assertEqual(len(swept4), 8)
        for atoms in swept4:
            cfg = str(atoms.info.get("Config_type", ""))
            comp_tokens = [t.strip() for t in cfg.split("|") if t.strip().startswith("Comp(") and t.strip().endswith(")")]
            self.assertTrue(comp_tokens)
            comp_items = [p for p in comp_tokens[-1][5:-1].split(",") if p.strip()]
            self.assertEqual(len(comp_items), 4)

        sweep5 = CompositionSweepCard()
        sweep5.elements_edit.setText("Co,Cr,Ni,Al,Fe")
        sweep5.order_combo.setCurrentIndex(sweep5.order_combo.findData("5"))
        sweep5.method_combo.setCurrentText("Sobol")
        sweep5.n_points_frame.set_input_value([6])
        sweep5.max_output_frame.set_input_value([6])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            swept5 = sweep5.process_structure(base)
        self.assertEqual(len(swept5), 6)
        self.assertFalse(
            any("balance properties of Sobol" in str(item.message) for item in caught)
        )
        for atoms in swept5:
            cfg = str(atoms.info.get("Config_type", ""))
            comp_tokens = [t.strip() for t in cfg.split("|") if t.strip().startswith("Comp(") and t.strip().endswith(")")]
            self.assertTrue(comp_tokens)
            comp_items = [p for p in comp_tokens[-1][5:-1].split(",") if p.strip()]
            self.assertEqual(len(comp_items), 5)

    def test_parse_composition_accepts_bare_elements(self):
        self.assertEqual(parse_composition("Ge"), {"Ge": 1.0})
        self.assertEqual(parse_composition("Ge,C"), {"Ge": 1.0, "C": 1.0})
        self.assertEqual(parse_composition("Ge:0.7,C"), {"Ge": 0.7, "C": 1.0})
