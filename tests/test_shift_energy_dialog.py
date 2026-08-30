#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from PySide6.QtWidgets import QApplication, QWidget

from NepTrainKit.ui.widgets.dialog import ShiftEnergyMessageBox


class TestShiftEnergyMessageBox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()
        self.parent.resize(640, 480)

    def test_set_preset_names_keeps_placeholder(self):
        box = ShiftEnergyMessageBox(self.parent)
        box.set_preset_names(["preset_a", "preset_b"], placeholder="None")

        self.assertEqual(box.presetCombo.itemText(0), "None")
        self.assertEqual(box.presetCombo.count(), 3)
        self.assertEqual(box.presetCombo.itemText(1), "preset_a")
        self.assertEqual(box.presetCombo.itemText(2), "preset_b")

    def test_collect_values_handles_empty_pattern_and_placeholder(self):
        box = ShiftEnergyMessageBox(self.parent)
        box.set_preset_names([], placeholder="None")
        box.groupEdit.setText(" ; ; ")
        box.presetCombo.setCurrentText("None")
        box.savePresetCheck.setChecked(True)
        box.presetNameEdit.setText("   ")

        values = box.collect_values()

        self.assertEqual(values.group_patterns, [])
        self.assertEqual(values.selected_preset_name, "")
        self.assertTrue(values.save_preset)
        self.assertEqual(values.preset_name, "")

    def test_collect_values_splits_group_patterns_on_semicolons(self):
        box = ShiftEnergyMessageBox(self.parent)
        box.set_preset_names([], placeholder="None")
        box.groupEdit.setText("bulk.*; surface.*")

        values = box.collect_values()

        self.assertEqual(values.group_patterns, ["bulk.*", "surface.*"])

if __name__ == '__main__':
    unittest.main()
