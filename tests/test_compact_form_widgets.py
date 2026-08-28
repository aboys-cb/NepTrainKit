#!/usr/bin/env python
"""Smoke tests for the shared card header/body building blocks."""
from __future__ import annotations

import unittest

from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget

from NepTrainKit.ui.widgets.compact_form import (
    CategoryTag,
    CompactField,
    ResponsiveFormGrid,
    SegmentedControl,
    StatusBadge,
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

    def test_status_badge_pairs_state_colour_with_readable_text(self):
        badge = StatusBadge()
        self.assertEqual(badge.state(), "idle")
        self.assertEqual(badge.label.text(), "Ready")

        badge.set_state("succeeded")
        self.assertEqual(badge.state(), "succeeded")
        self.assertEqual(badge.label.text(), "Done")
        self.assertIn("Done", badge.accessibleName())

        badge.set_state("succeeded", "12→48")
        self.assertEqual(badge.label.text(), "Done · 12→48")

        badge.set_state("partial")
        self.assertEqual(badge.state(), "partial")
        self.assertEqual(badge.label.text(), "Partial")

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

    def test_compact_field_can_opt_into_a_bounded_inline_control(self):
        line_edit = QLineEdit()
        field = CompactField(
            "Short scalar",
            line_edit,
            inline=True,
            input_max_width=132,
        )

        self.assertFalse(field.caption.wordWrap())
        self.assertEqual(line_edit.maximumWidth(), 132)
        self.assertIs(field.input_widget, line_edit)
        inline_layout = field.layout().itemAt(0).widget().layout()
        self.assertIs(inline_layout.itemAt(0).widget(), field.caption)
        self.assertIs(inline_layout.itemAt(1).widget(), line_edit)
        self.assertIsNotNone(inline_layout.itemAt(2).spacerItem())

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
        # Match QComboBox: a programmatic value change emits so dependent
        # visibility and preview logic stays synchronized during JSON restore.
        self.assertEqual(received, [2, 1])

    def test_segmented_control_replacing_options_resets_selection(self):
        control = SegmentedControl(["A", "B"])
        control.setCurrentIndex(1)

        control.set_options(["X", "Y", "Z"])
        self.assertEqual(control.currentIndex(), 0)
        self.assertEqual(control.currentText(), "X")
        self.assertEqual(len(control._buttons), 3)

    def test_responsive_grid_clears_stale_second_column_after_narrowing(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        grid = ResponsiveFormGrid(host, two_column_threshold=320)
        first = CompactField("First", QLineEdit(), grid)
        second = CompactField("Second", QLineEdit(), grid)
        grid.add_field(first)
        grid.add_field(second)
        layout.addWidget(grid)

        host.resize(420, 180)
        host.show()
        self._app.processEvents()
        self.assertEqual(grid.column_count(), 2)

        host.resize(280, 180)
        self._app.processEvents()
        self.assertEqual(grid.column_count(), 1)
        self.assertEqual(grid._layout.columnStretch(1), 0)
        self.assertGreater(first.width(), grid.width() * 0.8)
        host.close()


if __name__ == "__main__":
    unittest.main()
