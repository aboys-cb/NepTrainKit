"""Cell strain card."""

from .card_test_base import *
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from ase.io import read as ase_read, write as ase_write
from unittest.mock import patch
import io
import json
import re
import warnings
from NepTrainKit.core.cards.sampling import derived_structure_seed


class TestCellStrainCards(BaseCardTest):
    def test_cell_strain_card_uniaxial(self):
        card = CellStrainCard()
        structure = self.structure.copy()
        card.organic_checkbox.setChecked(True)
        card.engine_type_combo.setCurrentIndex(
            card.engine_type_combo.findData("uniaxial")
        )
        card.strain_x_frame.set_input_value([1.0, 1.0, 1.0])
        card.strain_y_frame.set_input_value([0.0, 0.0, 1.0])
        card.strain_z_frame.set_input_value([0.0, 0.0, 1.0])

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

    def test_cell_strain_operation_is_ui_independent(self):
        params = CellStrainParams(
            axes="X",
            x_range=(1.0, 1.0, 1.0),
            y_range=(0.0, 0.0, 1.0),
            z_range=(0.0, 0.0, 1.0),
            identify_organic=False,
        )
        results = CellStrainOperation().run_structure(self.structure.copy(), params)

        self.assertEqual(len(results), 1)
        self.assertIn("Str(a=1%,b=0%,c=0%)", results[0].info.get("Config_type", ""))

    def test_cell_strain_all_public_modes_scale_lattice_vectors(self):
        structure = self.structure.copy()
        structure.set_cell(
            np.array(
                [
                    [5.0, 0.2, 0.1],
                    [0.5, 5.5, 0.3],
                    [0.2, 0.4, 6.0],
                ]
            ),
            scale_atoms=True,
        )
        original = structure.cell.array.copy()
        ranges = {
            "x_range": (1.0, 1.0, 1.0),
            "y_range": (2.0, 2.0, 1.0),
            "z_range": (3.0, 3.0, 1.0),
        }
        operation = CellStrainOperation()

        isotropic = operation.run_structure(
            structure,
            CellStrainParams(axes="isotropic", **ranges),
        )
        self.assertEqual(len(isotropic), 1)
        np.testing.assert_allclose(isotropic[0].cell.array, original * 1.01)

        uniaxial = operation.run_structure(
            structure,
            CellStrainParams(axes="uniaxial", **ranges),
        )
        self.assertEqual(len(uniaxial), 3)
        for axis, result in enumerate(uniaxial):
            expected = original.copy()
            expected[axis] *= 1.0 + (axis + 1) / 100.0
            np.testing.assert_allclose(result.cell.array, expected)

        biaxial = operation.run_structure(
            structure,
            CellStrainParams(axes="biaxial", **ranges),
        )
        self.assertEqual(len(biaxial), 3)
        for axes, result in zip(((0, 1), (0, 2), (1, 2)), biaxial):
            expected = original.copy()
            for axis in axes:
                expected[axis] *= 1.0 + (axis + 1) / 100.0
            np.testing.assert_allclose(result.cell.array, expected)

        triaxial = operation.run_structure(
            structure,
            CellStrainParams(axes="triaxial", **ranges),
        )
        self.assertEqual(len(triaxial), 1)
        np.testing.assert_allclose(
            triaxial[0].cell.array,
            original * np.array([1.01, 1.02, 1.03])[:, None],
        )

        custom = operation.run_structure(
            structure,
            CellStrainParams(axes="XZ", **ranges),
        )
        expected = original.copy()
        expected[0] *= 1.01
        expected[2] *= 1.03
        self.assertEqual(len(custom), 1)
        np.testing.assert_allclose(custom[0].cell.array, expected)

    def test_cell_strain_default_counts_are_exact_and_deduplicated(self):
        operation = CellStrainOperation()
        expected = {
            "uniaxial": 31,
            "biaxial": 331,
            "triaxial": 1331,
            "isotropic": 11,
        }
        for mode, count in expected.items():
            with self.subTest(mode=mode):
                summary = operation.sampling_summary(CellStrainParams(axes=mode))
                self.assertEqual(summary["outputs_per_input"], count)
                results = operation.run_structure(
                    self.structure.copy(), CellStrainParams(axes=mode)
                )
                tags = [atoms.info.get("Config_type", "") for atoms in results]
                self.assertEqual(len(results), count)
                self.assertEqual(len(set(tags)), count)
                self.assertEqual(
                    sum("Str(a=0%,b=0%,c=0%)" in tag for tag in tags), 1
                )

    def test_cell_strain_card_mode_visibility_and_preview(self):
        card = CellStrainCard()
        card.set_preview_input_count(2)
        self.assertIn("31/input", card.get_summary_text())
        self.assertIn("2 × 31 = 62", card.get_guidance_text())
        card.set_preview_input_count(0)
        self.assertIn("Each input produces 31 unique structures", card.get_guidance_text())

        card.engine_type_combo.setCurrentIndex(
            card.engine_type_combo.findData("isotropic")
        )
        self.assertFalse(card.strain_x_field.isHidden())
        self.assertTrue(card.strain_y_field.isHidden())
        self.assertTrue(card.strain_z_field.isHidden())
        self.assertIn("11/input", card.get_summary_text())

        card.set_params(
            CellStrainParams(
                axes="XZ",
                x_range=(1.0, 1.0, 1.0),
                y_range=(2.0, 2.0, 1.0),
                z_range=(3.0, 3.0, 1.0),
            )
        )
        self.assertEqual(card.custom_axes_edit.text(), "ac")
        self.assertFalse(card.strain_x_field.isHidden())
        self.assertTrue(card.strain_y_field.isHidden())
        self.assertFalse(card.strain_z_field.isHidden())
        self.assertEqual(card.get_params().axes, "XZ")
        self.assertIn("axes ac", card.get_summary_text())

    def test_cell_strain_rejects_duplicate_axes_and_cell_collapse(self):
        operation = CellStrainOperation()
        with self.assertRaisesRegex(ValueError, "unique lattice axes"):
            operation.sampling_summary(CellStrainParams(axes="XX"))
        with self.assertRaisesRegex(ValueError, "greater than -100%"):
            operation.sampling_summary(
                CellStrainParams(x_range=(-100.0, -100.0, 1.0))
            )
        summary = operation.sampling_summary(
            CellStrainParams(
                axes="isotropic",
                x_range=(1.0, 1.0, 1.0),
                y_range=(0.0, 1.0, 0.0),
                z_range=(0.0, 1.0, 0.0),
            )
        )
        self.assertEqual(summary["outputs_per_input"], 1)

