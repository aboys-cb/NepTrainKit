"""Set magnetic moments card: map syntax, element picker, scalar-to-vector."""

import json
import tempfile

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSpinBox

from NepTrainKit.core.io.importers import import_structures
from NepTrainKit.core.magnetism import parse_magmom_map_any
from NepTrainKit.ui.views._card.i18n_utils import set_combo_value

from .magnetism_test_base import *


class TestSetMagneticMomentsCards(MagnetismCardTest):
    def test_set_magnetic_moments_card_fails_closed_for_invalid_map(self):
        card = SetMagneticMomentsCard()
        card.map_edit.setText("Fe:not-a-number")
        with self.assertRaises(ValueError):
            card.process_structure(self._spin_chain())

    def test_magmom_map_accepts_vector_token_syntax(self):
        parsed = parse_magmom_map_any("Fe:2.2,Cr:[0,0,1.5]")
        self.assertEqual(parsed["Fe"], 2.2)
        np.testing.assert_allclose(parsed["Cr"], [0.0, 0.0, 1.5])

    def test_set_magnetic_moments_card_map_vector_roundtrip(self):
        structure = self._spin_chain()

        card = SetMagneticMomentsCard()
        card.source_combo.setCurrentText("Map/default magnitude")
        card.format_combo.setCurrentText("Non-collinear (vector)")
        card.map_edit.setText("Fe:2.5")
        card.axis_frame.set_input_value([0.0, 0.0, 1.0])

        result = card.process_structure(structure)[0]
        moments = np.array(result.get_initial_magnetic_moments(), dtype=float)
        self.assertEqual(moments.shape, (4, 3))
        self.assertTrue(np.allclose(moments, np.tile([0.0, 0.0, 2.5], (4, 1)), atol=1e-6))
        self.assertIn("MagSet(map,vec)", str(result.info.get("Config_type", "")))

        data = card.to_dict()
        restored = SetMagneticMomentsCard()
        restored.from_dict(data)
        self.assertEqual(restored.source_combo.currentText(), "Map/default magnitude")
        self.assertEqual(restored.format_combo.currentText(), "Non-collinear (vector)")
        self.assertEqual(restored.map_edit.text(), "Fe:2.5")

    def test_set_magnetic_moments_reference_axis_uses_compact_integer_inputs(self):
        card = SetMagneticMomentsCard()

        self.assertEqual(card.axis_frame.get_input_value(), [0, 0, 1])
        self.assertEqual(card.axis_frame.maximumWidth(), 16777215)
        self.assertTrue(all(isinstance(control, QSpinBox) for control in card.axis_frame.object_list))

    def test_set_magnetic_moments_element_picker_adds_validated_row(self):
        card = SetMagneticMomentsCard()

        card.map_edit._apply_element_selection("Fe")
        self.assertEqual(card.map_edit.table.item(0, 0).text(), "Fe")
        self.assertEqual(card.map_edit.table.item(0, 1).text(), "1.0")
        self.assertEqual(card.map_edit.text(), "Fe:1.0")
        self.assertFalse(
            bool(card.map_edit.table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable)
        )

        card.map_edit._apply_element_selection("Fe")
        self.assertEqual(card.map_edit.table.rowCount(), 1)

    def test_set_magnetic_moments_card_existing_scalar_to_vector(self):
        structure = self._spin_chain()
        structure.set_initial_magnetic_moments([1.0, 2.0, 3.0, 4.0])

        card = SetMagneticMomentsCard()
        card.source_combo.setCurrentText("Existing initial magmoms")
        card.format_combo.setCurrentText("Non-collinear (vector)")
        card.axis_frame.set_input_value([1.0, 0.0, 0.0])

        result = card.process_structure(structure)[0]
        moments = np.array(result.get_initial_magnetic_moments(), dtype=float)
        self.assertTrue(
            np.allclose(
                moments,
                np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
                atol=1e-6,
            )
        )
        self.assertIn("MagSet(existing,vec)", str(result.info.get("Config_type", "")))

