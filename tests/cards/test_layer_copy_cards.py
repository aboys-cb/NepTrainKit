"""Layer copy card."""

import warnings
from unittest.mock import patch

from ase.geometry import geometry
from PySide6.QtCore import Qt
from NepTrainKit.core.cards.errors import CardOperationError

from .card_test_base import *


class TestLayerCopyCards(BaseCardTest):
    def test_layer_copy_operation_is_ui_independent(self):
        structure = self.structure.copy()
        results = LayerCopyOperation().run_structure(
            structure,
            LayerCopyParams(
                dz_expr="1.0",
                layers=2,
                distance=3.0,
                extend_cell_z=True,
            ),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), len(structure) * 2)
        self.assertIn(
            "LayerStack(L=2,gap=3,step=",
            results[0].info.get("Config_type", ""),
        )

    def test_layer_copy_card_roundtrip(self):
        card = LayerCopyCard()
        card.preset_combo.setCurrentIndex(0)
        card.expr_edit.setPlainText("A + z*0")
        card.params_edit.setText("A=1.5")
        card.apply_combo.setCurrentIndex(2)
        card.zrange_frame.set_input_value([0.0, 2.0])
        card.wrap_checkbox.setChecked(True)
        card.extend_cell_checkbox.setChecked(False)
        card.vacuum_frame.set_input_value([1.0])
        card.layers_frame.set_input_value([2])
        card.distance_frame.set_input_value([4.0])

        restored = LayerCopyCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_layer_copy_default_is_flat_and_preview_explains_full_copy(self):
        card = LayerCopyCard()
        slab = Atoms(
            "MoS2",
            positions=[[0, 0, 0], [0, 0, 1.5], [0, 0, -1.5]],
            cell=[8, 8, 15],
            pbc=True,
        )
        card.set_preview_structure(slab)
        card.set_preview_input_count(2)

        self.assertEqual(card.get_params().dz_expr, "0")
        self.assertFalse(card.show_warp_checkbox.isChecked())
        self.assertTrue(card.preset_field.isHidden())
        self.assertIn("3 input atoms", card.preview_label.text())
        self.assertIn("thickness 3 Å", card.preview_label.text())
        self.assertIn("gap 3.35 Å", card.preview_label.text())
        self.assertIn("copy translation 6.35 Å", card.preview_label.text())
        self.assertIn("6 atoms/output", card.preview_label.text())
        self.assertIn("6 atoms/out", card.get_summary_text())
        self.assertIn("outputs 2", card.get_guidance_text())
        self.assertIn("3 × 2", card.get_guidance_text())

        card.show_warp_checkbox.setChecked(True)
        card.apply_combo.setCurrentIndex(1)
        self.assertFalse(card.preset_field.isHidden())
        self.assertFalse(card.elements_field.isHidden())
        self.assertTrue(card.zrange_field.isHidden())

    def test_layer_copy_selection_only_limits_warp_not_copy(self):
        structure = Atoms(
            "CSi",
            positions=[[0, 0, 0], [1, 0, 0]],
            cell=[5, 5, 10],
            pbc=True,
        )
        result = LayerCopyOperation().run_structure(
            structure,
            LayerCopyParams(
                dz_expr="1",
                apply_mode=1,
                elements="C",
                layers=2,
                distance=4,
            ),
        )[0]

        self.assertEqual(result.get_chemical_symbols(), ["C", "Si", "C", "Si"])
        np.testing.assert_allclose(
            result.positions,
            [[0, 0, 1], [1, 0, 0], [0, 0, 6], [1, 0, 5]],
        )

    def test_layer_copy_extends_cell_for_one_layer_vacuum(self):
        structure = Atoms(
            "C",
            positions=[[0, 0, 0]],
            cell=[[5, 0, 0], [0, 5, 0], [1, 0, 10]],
            pbc=True,
        )
        result = LayerCopyOperation().run_structure(
            structure,
            LayerCopyParams(
                dz_expr="0",
                layers=1,
                distance=3,
                extend_cell_z=True,
                extra_vacuum=4,
            ),
        )[0]

        np.testing.assert_allclose(result.cell.array[2], [1, 0, 14])

    def test_layer_copy_rejects_invalid_geometry_settings(self):
        structure = Atoms(
            "C",
            positions=[[0, 0, 0]],
            cell=[5, 5, 10],
            pbc=True,
        )
        operation = LayerCopyOperation()

        with self.assertRaisesRegex(ValueError, "Total layers must be an integer"):
            operation.run_structure(
                structure,
                LayerCopyParams(dz_expr="0", layers=1.5),
            )
        with self.assertRaisesRegex(ValueError, "Copy translation must be positive"):
            operation.run_structure(
                structure,
                LayerCopyParams(
                    dz_expr="0",
                    layers=2,
                    distance_mode="translation",
                    distance=0,
                ),
            )
        with self.assertRaisesRegex(ValueError, "vacuum must be a finite"):
            operation.run_structure(
                structure,
                LayerCopyParams(dz_expr="0", layers=1, extra_vacuum=float("nan")),
            )
        with self.assertRaisesRegex(ValueError, "no atoms selected"):
            operation.run_structure(
                structure,
                LayerCopyParams(
                    dz_expr="1",
                    apply_mode=1,
                    elements="Si",
                    layers=1,
                ),
            )

    def test_layer_copy_surface_gap_legacy_translation_and_safety_limits(self):
        slab = Atoms(
            "MoS2",
            positions=[[0, 0, 0], [0, 0, 1.5], [0, 0, -1.5]],
            cell=[8, 8, 15],
            pbc=[True, True, False],
        )
        spin = np.arange(9.0).reshape(3, 3)
        magmoms = np.array([1.0, -1.0, 0.5])
        slab.new_array("spin", spin)
        slab.set_initial_magnetic_moments(magmoms)
        slab.new_array("group", np.asarray(["A", "B", "A"], dtype=object))
        slab.info["Config_type"] = "seed"
        operation = LayerCopyOperation()

        summary = operation.geometry_summary(slab, LayerCopyParams())
        self.assertEqual(summary["slab_thickness"], 3.0)
        self.assertEqual(summary["surface_gap"], 3.35)
        self.assertEqual(summary["translation"], 6.35)
        self.assertEqual(summary["output_atoms"], 6)

        output = operation.run_structure(slab, LayerCopyParams())[0]
        np.testing.assert_array_equal(output.arrays["spin"], np.tile(spin, (2, 1)))
        np.testing.assert_array_equal(
            output.arrays["initial_magmoms"], np.tile(magmoms, 2)
        )
        np.testing.assert_array_equal(
            output.arrays["group"], np.tile(slab.arrays["group"], 2)
        )
        np.testing.assert_array_equal(output.pbc, slab.pbc)
        self.assertIn("seed|LayerStack(L=2,gap=3.35,step=6.35)", output.info["Config_type"])

        with self.assertRaisesRegex(ValueError, "negative surface gap"):
            operation.run_structure(
                slab,
                LayerCopyParams(
                    distance_mode="translation",
                    distance=2.5,
                ),
            )
        with self.assertRaisesRegex(ValueError, "atom budget"):
            operation.run_structure(
                slab,
                LayerCopyParams(layers=4, max_output_atoms=11),
            )

        negative_c = slab.copy()
        negative_c.set_cell([[8, 0, 0], [0, -8, 0], [0, 0, -15]])
        with self.assertRaisesRegex(ValueError, "positive Cartesian z component"):
            operation.run_structure(negative_c, LayerCopyParams())

        legacy = LayerCopyCard()
        with patch(
            "NepTrainKit.ui.views._card.layer_copy_card.MessageManager.send_warning_message"
        ) as warning:
            legacy.from_dict(
                {
                    "class": "LayerCopyCard",
                    "check_state": True,
                    "layers": [2],
                    "distance": [4.0],
                }
            )
        warning.assert_called_once()
        self.assertEqual(legacy.get_params().distance_mode, "translation")
        self.assertEqual(legacy.get_params().distance, 4.0)
        self.assertFalse(legacy.legacy_notice.isHidden())
