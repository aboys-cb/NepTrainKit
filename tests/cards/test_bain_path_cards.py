"""Bain path card."""

from .card_test_base import *
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.io import read as ase_read, write as ase_write
from unittest.mock import patch
import io
import json
import re
import warnings
from NepTrainKit.core.cards.sampling import derived_structure_seed


class TestBainPathCards(BaseCardTest):
    def test_bain_path_card_roundtrip(self):
        card = BainPathCard()
        card.set_params(
            BainPathParams(
                axis="x",
                ca_range=(0.8, 1.2, 0.2),
                coordinate_mode="relative_ca",
                mode="scale_volume",
                volume_scale_range=(0.9, 1.1, 0.1),
                scale_atoms=False,
            )
        )

        restored = BainPathCard()
        restored.from_dict(card.to_dict())

        self.assertEqual(restored.get_params(), card.get_params())

    def test_bain_path_default_is_three_points_and_mode_controls_volume_scan(self):
        card = BainPathCard()
        self.assertEqual(card.getTitle(), "Bain Path")
        self.assertEqual(card.get_params(), BainPathParams())
        self.assertTrue(card.volume_field.isHidden())
        self.assertIn("3 path points = 3 outputs/input", card.preview_label.text())

        card.set_dataset([self.structure.copy(), self.structure.copy()])
        self.assertIn("input structures: 2 → 6 outputs", card.preview_label.text())
        self.assertIn(
            "Input structures: 2 × 3 outputs/input = 6 outputs",
            card.get_guidance_text(),
        )
        self.assertIn("relative c/a", card.get_summary_text())

        card.set_params(
            BainPathParams(
                mode="scale_volume",
                volume_scale_range=(0.95, 1.05, 0.05),
            )
        )
        self.assertFalse(card.volume_field.isHidden())
        self.assertIn("3 shape points × 3 volume points = 9 outputs/input", card.preview_label.text())

    def test_bain_path_restores_old_json_as_legacy_axis_scale(self):
        card = BainPathCard()
        card.from_dict(
            {
                "check_state": True,
                "params": {
                    "axis": "z",
                    "ca_range": [1.0, 1.2, 0.2],
                    "mode": "constant_volume",
                    "volume_scale_range": [1.0, 1.0, 1.0],
                    "scale_atoms": True,
                }
            }
        )

        self.assertEqual(card.get_params().coordinate_mode, "axis_scale")
        self.assertIn("Legacy coordinate", card.ca_field.helper_label.text())
        self.assertIn("legacy scale", card.get_summary_text())

