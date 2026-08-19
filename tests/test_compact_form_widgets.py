#!/usr/bin/env python
"""Smoke tests for the shared card header/body building blocks."""
from __future__ import annotations

import unittest

from PySide6.QtWidgets import QApplication

from NepTrainKit.ui.widgets.compact_form import (
    CategoryTag,
    CompactField,
    SegmentedControl,
    StatusDot,
)


class CompactFormWidgetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_status_dot_defaults_to_idle_and_updates(self):
        dot = StatusDot()
        self.assertEqual(dot.state(), "idle")

        dot.set_state("succeeded")
        self.assertEqual(dot.state(), "succeeded")

        # Unknown states fall back to idle rather than raising.
        dot.set_state("not-a-real-state")
        self.assertEqual(dot.state(), "not-a-real-state")

    def test_category_tag_hides_when_empty_and_shows_when_set(self):
        tag = CategoryTag("")
        self.assertTrue(tag.isHidden())

        tag.setText("Doping")
        self.assertEqual(tag.text(), "Doping")
        self.assertFalse(tag.isHidden())

        tag.setText("")
        self.assertEqual(tag.text(), "")
        self.assertTrue(tag.isHidden())

    def test_compact_field_wraps_label_and_input(self):
        from PySide6.QtWidgets import QLineEdit

        line_edit = QLineEdit()
        field = CompactField("Target", line_edit)
        self.assertEqual(field.caption.text(), "Target")
        self.assertIs(field.input_widget, line_edit)

        field.set_label("Dopants")
        self.assertEqual(field.caption.text(), "Dopants")

    def test_segmented_control_selection(self):
        control = SegmentedControl(["Atomic %", "Mass %", "Count"])
        self.assertEqual(control.currentIndex(), 0)
        self.assertEqual(control.currentText(), "Atomic %")

        received = []
        control.currentIndexChanged.connect(received.append)

        control._buttons[2].click()
        self.assertEqual(control.currentIndex(), 2)
        self.assertEqual(control.currentText(), "Count")
        self.assertEqual(received, [2])

        control.setCurrentIndex(1)
        self.assertEqual(control.currentIndex(), 1)
        # setCurrentIndex is a programmatic set and should not re-emit.
        self.assertEqual(received, [2])

    def test_segmented_control_replacing_options_resets_selection(self):
        control = SegmentedControl(["A", "B"])
        control.setCurrentIndex(1)

        control.set_options(["X", "Y", "Z"])
        self.assertEqual(control.currentIndex(), 0)
        self.assertEqual(control.currentText(), "X")
        self.assertEqual(len(control._buttons), 3)


if __name__ == "__main__":
    unittest.main()
