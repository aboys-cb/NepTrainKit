#!/usr/bin/env python
import unittest

from PySide6.QtWidgets import QApplication, QWidget

from NepTrainKit.ui.widgets.dialog import ExportFormatMessageBox


class TestExportFormatMessageBox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()

    def test_offers_standard_and_mixed_deepmd_formats(self):
        box = ExportFormatMessageBox(
            self.parent,
            default_format="deepmd/npy/mixed",
            mixed_atom_numb_pad=16,
        )

        self.assertEqual(box.selected_format(), "deepmd/npy/mixed")
        self.assertEqual(box.formatCombo.findData("deepmd/npy"), 1)
        self.assertEqual(box.formatCombo.findData("deepmd/npy/mixed"), 2)
        self.assertTrue(box.standardGroupingWidget.isHidden())
        self.assertFalse(box.mixedPaddingWidget.isHidden())
        self.assertEqual(box.mixed_atom_numb_pad(), 16)

    def test_grouping_is_visible_only_for_standard_npy(self):
        box = ExportFormatMessageBox(
            self.parent,
            default_format="deepmd/npy",
            group_by_config_type=False,
        )

        self.assertFalse(box.standardGroupingWidget.isHidden())
        self.assertFalse(box.group_by_config_type())

        box.standardGroupingCombo.setCurrentIndex(
            box.standardGroupingCombo.findData("config_type")
        )
        self.assertTrue(box.group_by_config_type())

        box.formatCombo.setCurrentIndex(box.formatCombo.findData("xyz"))
        self.assertTrue(box.standardGroupingWidget.isHidden())
        self.assertTrue(box.mixedPaddingWidget.isHidden())

        box.formatCombo.setCurrentIndex(
            box.formatCombo.findData("deepmd/npy/mixed")
        )
        self.assertTrue(box.standardGroupingWidget.isHidden())
        self.assertFalse(box.mixedPaddingWidget.isHidden())

        box.mixedPaddingSpinBox.setValue(0)
        self.assertIsNone(box.mixed_atom_numb_pad())


if __name__ == "__main__":
    unittest.main()
