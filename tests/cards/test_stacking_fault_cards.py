"""Stacking fault and strict GSFE path cards."""

import warnings
from unittest.mock import patch

from ase.geometry import geometry
from PySide6.QtCore import Qt
from NepTrainKit.core.cards.errors import CardOperationError

from .card_test_base import *


class TestStackingFaultCards(BaseCardTest):
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

    def test_legacy_stacking_fault_json_remains_loadable(self):
        card_class = CardManager.card_info_dict["StackingFaultCard"]
        card = card_class()
        card.from_dict(
            {
                "class": "StackingFaultCard",
                "check_state": True,
                "params": {
                    "hkl": [0, 0, 1],
                    "step": [0.0, 0.4, 0.2],
                    "layers": 2,
                },
            }
        )

        self.assertEqual(
            card.get_params(),
            StackingFaultParams(
                hkl=(0, 0, 1),
                step=(0.0, 0.4, 0.2),
                layers=2,
            ),
        )

    def test_stacking_fault_displaces_selected_layers_in_plane(self):
        structure = Atoms(
            "Si3",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            cell=[5.0, 5.0, 5.0],
            pbc=False,
        )

        result = StackingFaultOperation().run_structure(
            structure,
            StackingFaultParams(hkl=(0, 0, 1), step=(0.5, 0.5, 0.1), layers=2),
        )[0]

        displacement = result.get_positions() - structure.get_positions()
        np.testing.assert_allclose(displacement[0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(displacement[1:, 2], [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(displacement[1:], axis=1), [0.5, 0.5], atol=1e-12)

    def test_strict_gsfe_path_card_roundtrip(self):
        card = StrictGSFEPathCard()
        self.assertEqual(card.get_params(), StrictGSFEPathParams())
        self.assertTrue(card.cut_fraction_field.isHidden())
        self.assertTrue(card.layer_field.isHidden())
        self.assertGreater(card.disp_frame.object_list[2].minimum(), 0.0)
        self.assertIn("Load an oriented structure", card.preview_label.text())

        card.set_dataset(
            Atoms(
                "H4",
                positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
                cell=[8, 8, 8],
                pbc=True,
            )
        )
        self.assertIn("4 layers", card.preview_label.text())
        self.assertIn("move 2, keep 2", card.preview_label.text())
        self.assertIn("0→1 × vector = 0→8 Å", card.preview_label.text())
        self.assertIn("11 outputs", card.preview_label.text())

        card.slip_uv_frame.set_input_value([1, -1])
        card.disp_frame.set_input_value([0.0, 0.5, 0.25])
        card.unit_control.setCurrentIndex(card.unit_control.findData("angstrom"))
        card.cut_control.setCurrentIndex(card.cut_control.findData("layer_index"))
        self.assertTrue(card.cut_fraction_field.isHidden())
        self.assertFalse(card.layer_field.isHidden())
        card.cut_fraction_frame.set_input_value([0.25])
        card.layer_frame.set_input_value([2])
        card.wrap_checkbox.setChecked(False)

        restored = StrictGSFEPathCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_strict_gsfe_cut_mode_fields_are_visible_and_operable(self):
        card = StrictGSFEPathCard()
        card.show()
        QApplication.processEvents()

        card.cut_control.setCurrentIndex(card.cut_control.findData("fractional"))
        QApplication.processEvents()
        self.assertTrue(card.cut_fraction_frame.isVisible())
        self.assertFalse(card.layer_field.isVisible())
        card.cut_fraction_frame.set_input_value([0.25])
        self.assertEqual(card.get_params().cut_fraction, 0.25)

        card.cut_control.setCurrentIndex(card.cut_control.findData("layer_index"))
        QApplication.processEvents()
        self.assertFalse(card.cut_fraction_field.isVisible())
        self.assertTrue(card.layer_frame.isVisible())
        card.layer_frame.set_input_value([2])
        self.assertEqual(card.get_params().layer_index, 2)

        card.cut_control.setCurrentIndex(card.cut_control.findData("middle"))
        QApplication.processEvents()
        self.assertFalse(card.cut_fraction_field.isVisible())
        self.assertFalse(card.layer_field.isVisible())

    def test_strict_gsfe_preserves_legacy_geometry_until_direction_edit(self):
        card = StrictGSFEPathCard()
        legacy = StrictGSFEPathParams(
            plane_hkl=(1, 1, 0),
            slip_uvw=(1, -1, 2),
        )
        card.set_params(legacy)

        self.assertEqual(card.get_params(), legacy)
        self.assertFalse(card.legacy_geometry_label.isHidden())

        card.slip_uv_frame.object_list[0].setValue(2)
        self.assertEqual(card.get_params().plane_hkl, (0, 0, 1))
        self.assertEqual(card.get_params().slip_uvw, (2, -1, 0))
        self.assertTrue(card.legacy_geometry_label.isHidden())

