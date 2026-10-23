#!/usr/bin/env python
"""Dialog-level coverage for the LAMMPS dump import wizard."""
import unittest
from pathlib import Path

from PySide6.QtWidgets import QApplication, QWidget

from NepTrainKit.core.io.lammps_dump import ColumnOverrides, DumpProbe, ElementSuggestion
from NepTrainKit.ui.widgets.dump_import_dialog import (
    DumpFieldMappingDialog,
    DumpImportDialog,
    _cell_text,
    _describe_plan,
    _find_data,
)

_SPIN_COLUMNS = (
    "id", "type", "x", "y", "z",
    "c_spin[1]", "c_spin[2]", "c_spin[3]", "c_spin[4]",
    "c_spin[5]", "c_spin[6]", "c_spin[7]",
)


def _probe(**overrides):
    values = dict(
        path=Path("/tmp/traj.dump"),
        frame_count=120,
        declared_frame_count=120,
        atom_counts=(16,),
        timesteps=(0,),
        signatures=(_SPIN_COLUMNS,),
        columns=_SPIN_COLUMNS,
        types=(1, 2),
        species_source="type",
        cell_styles=("ortho",),
        truncated=False,
        compressed=None,
        lattice=(6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 6.0),
        pbc="T T T",
    )
    values.update(overrides)
    return DumpProbe(**values)


class TestDumpImportDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()

    def test_prefills_known_types_and_gates_the_import_button(self):
        dialog = DumpImportDialog(ElementSuggestion(_probe(), {1: "Fe"}, (2,)), self.parent)

        self.assertEqual(len(dialog._edits), 2)
        self.assertEqual(dialog._edits[0].text(), "Fe")
        self.assertEqual(dialog._edits[1].text(), "")
        self.assertFalse(dialog.yesButton.isEnabled())

        dialog._edits[1].setText("Co")

        self.assertTrue(dialog.yesButton.isEnabled())
        self.assertTrue(dialog.validate())
        options = dialog.options()
        self.assertEqual(options.element_map, {1: "Fe", 2: "Co"})
        self.assertIsNone(options.overrides)
        self.assertTrue(options.keep_extra)
        self.assertTrue(options.remember)
        self.assertFalse(options.unwrap)

    def test_typed_symbols_and_atomic_numbers_are_accepted(self):
        dialog = DumpImportDialog(ElementSuggestion(_probe(), {}, (1, 2)), self.parent)

        dialog._edits[0].setText("fe")
        dialog._edits[1].setText("27")

        self.assertEqual(dialog._missing_types(), [])
        self.assertEqual(dialog.options().element_map, {1: "Fe", 2: "Co"})

    def test_invalid_typed_text_keeps_the_import_disabled(self):
        dialog = DumpImportDialog(ElementSuggestion(_probe(), {}, (1, 2)), self.parent)

        dialog._edits[0].setText("Fe")
        dialog._edits[1].setText("not-an-element")

        self.assertEqual(dialog._missing_types(), [2])
        self.assertFalse(dialog.yesButton.isEnabled())

    def test_duplicate_elements_are_allowed(self):
        dialog = DumpImportDialog(ElementSuggestion(_probe(), {1: "Fe"}, (2,)), self.parent)
        dialog._edits[1].setText("Fe")

        self.assertTrue(dialog.yesButton.isEnabled())
        self.assertEqual(dialog.options().element_map, {1: "Fe", 2: "Fe"})

    def test_unwrap_is_only_enabled_with_image_columns(self):
        without = DumpImportDialog(ElementSuggestion(_probe(), {}, (1, 2)), self.parent)
        self.assertFalse(without.unwrapCheck.isEnabled())

        columns = _SPIN_COLUMNS + ("ix", "iy", "iz")
        with_images = DumpImportDialog(
            ElementSuggestion(_probe(columns=columns, signatures=(columns,)), {}, (1, 2)),
            self.parent,
        )
        self.assertTrue(with_images.unwrapCheck.isEnabled())

    def test_options_round_trip(self):
        columns = _SPIN_COLUMNS + ("ix", "iy", "iz")
        dialog = DumpImportDialog(
            ElementSuggestion(_probe(columns=columns, signatures=(columns,)), {1: "Fe", 2: "Co"}, ()),
            self.parent,
        )

        dialog.unwrapCheck.setChecked(True)
        dialog.keepExtraCheck.setChecked(False)
        dialog.strictCheck.setChecked(True)
        dialog.rememberCheck.setChecked(False)
        options = dialog.options()

        self.assertEqual(options.element_map, {1: "Fe", 2: "Co"})
        self.assertTrue(options.unwrap)
        self.assertFalse(options.keep_extra)
        self.assertTrue(options.strict)
        self.assertFalse(options.remember)

    def test_field_mapping_button_applies_overrides(self):
        from unittest.mock import patch

        overrides = ColumnOverrides(spin_mode="off")
        dialog = DumpImportDialog(ElementSuggestion(_probe(), {1: "Fe", 2: "Co"}, ()), self.parent)

        with patch.object(DumpFieldMappingDialog, "ask", return_value=(True, overrides)) as ask:
            dialog._edit_fields()

        ask.assert_called_once()
        self.assertEqual(dialog.options().overrides, overrides)
        self.assertTrue(dialog.detectedLabel.text())

    def test_cancelling_the_field_mapping_keeps_auto(self):
        from unittest.mock import patch

        dialog = DumpImportDialog(ElementSuggestion(_probe(), {1: "Fe", 2: "Co"}, ()), self.parent)

        with patch.object(DumpFieldMappingDialog, "ask", return_value=(False, None)):
            dialog._edit_fields()

        self.assertIsNone(dialog.options().overrides)

    def test_detected_columns_describe_spin_definition(self):
        text = _describe_plan(_probe(), lambda source: source)

        self.assertIn("c_spin[1] × (c_spin[2], c_spin[3], c_spin[4])", text)

    def test_ask_on_gui_thread_marshals_worker_calls(self):
        import time
        from unittest.mock import patch

        from PySide6.QtCore import QThread

        app = QApplication.instance()
        seen: dict = {}
        holder: dict = {}

        def fake_ask(cls, suggestion, parent=None, overrides=None):
            seen["thread"] = QThread.currentThread()
            return "ok"

        worker = QThread()

        def run():
            holder["value"] = DumpImportDialog.ask_on_gui_thread(object())

        worker.started.connect(run)
        with patch.object(DumpImportDialog, "ask", classmethod(fake_ask)):
            worker.start()
            deadline = time.monotonic() + 5
            while "value" not in holder and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            worker.quit()
            worker.wait(2000)

        self.assertEqual(holder.get("value"), "ok")
        self.assertIs(seen.get("thread"), app.thread())

    def test_summary_reports_cell_and_frames(self):
        dialog = DumpImportDialog(ElementSuggestion(_probe(), {}, (1, 2)), self.parent)

        self.assertIn("120", dialog.subtitleLabel.text())
        self.assertEqual(_cell_text(_probe()), "6.000 × 6.000 × 6.000 Å")
        self.assertEqual(_cell_text(_probe(lattice=())), "-")


class TestDumpFieldMappingDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()

    def _dialog(self, columns=_SPIN_COLUMNS, overrides=None):
        return DumpFieldMappingDialog(columns, overrides, _probe(columns=columns, signatures=(columns,)), self.parent)

    def test_defaults_follow_automatic_detection(self):
        dialog = self._dialog()

        self.assertEqual(dialog.spinModeCombo.currentData(), None)
        self.assertIsNone(dialog.result_overrides())

    def test_spin_columns_can_be_changed(self):
        dialog = self._dialog()
        dialog.spinModeCombo.setCurrentIndex(dialog.spinModeCombo.findData("magnitude_direction"))
        direction = dialog._spin_combos[1]
        direction.setCurrentIndex(_find_data(direction, "c_spin[5]"))

        overrides = dialog.result_overrides()

        self.assertIsNotNone(overrides)
        self.assertEqual(overrides.spin_mode, "magnitude_direction")
        self.assertEqual(
            overrides.spin_names, ("c_spin[1]", "c_spin[5]", "c_spin[3]", "c_spin[4]")
        )

    def test_extra_columns_can_be_ignored(self):
        dialog = self._dialog()
        self.assertIn("dump.c_spin[5]", dialog._extra_checks)

        dialog._extra_checks["dump.c_spin[5]"].setChecked(False)

        overrides = dialog.result_overrides()
        self.assertIsNotNone(overrides)
        self.assertEqual(overrides.ignored_extras, frozenset({"dump.c_spin[5]"}))

    def test_disabling_forces_is_an_override(self):
        columns = ("id", "type", "x", "y", "z", "fx", "fy", "fz")
        dialog = self._dialog(columns)
        dialog.forceCheck.setChecked(False)

        overrides = dialog.result_overrides()
        self.assertIsNotNone(overrides)
        self.assertEqual(overrides.force_names, ())

    def test_coordinate_triplet_can_be_switched(self):
        columns = ("id", "type", "x", "y", "z", "xs", "ys", "zs")
        dialog = self._dialog(columns)
        index = _find_data(dialog.coordCombo, ("x", "y", "z"))
        dialog.coordCombo.setCurrentIndex(index)

        overrides = dialog.result_overrides()
        self.assertIsNotNone(overrides)
        self.assertEqual(overrides.coordinate_names, ("x", "y", "z"))

    def test_spin_labels_explain_magnitude_and_direction(self):
        dialog = self._dialog()
        dialog.spinModeCombo.setCurrentIndex(dialog.spinModeCombo.findData("magnitude_direction"))
        self.assertEqual(
            [label.text() for label in dialog.spinLabels],
            ["Magnitude", "Direction X", "Direction Y", "Direction Z"],
        )

        dialog.spinModeCombo.setCurrentIndex(dialog.spinModeCombo.findData("vector"))
        self.assertEqual(
            [label.text() for label in dialog.spinLabels],
            ["Direction X", "Direction Y", "Direction Z", ""],
        )
        self.assertTrue(dialog._spin_combos[3].isHidden())

    def test_saved_renames_and_ignores_are_restored(self):
        overrides = ColumnOverrides(
            ignored_extras=frozenset({"dump.c_spin[5]"}),
            extra_names=(("dump.c_spin[6]", "mag_y"),),
        )
        dialog = self._dialog(overrides=overrides)

        self.assertFalse(dialog._extra_checks["dump.c_spin[5]"].isChecked())
        self.assertEqual(dialog._extra_edits["dump.c_spin[6]"].text(), "mag_y")
        result = dialog.result_overrides()
        self.assertIsNotNone(result)
        self.assertEqual(result.ignored_extras, frozenset({"dump.c_spin[5]"}))
        self.assertIn(("dump.c_spin[6]", "mag_y"), result.extra_names)

    def test_extra_fields_can_be_renamed(self):
        dialog = self._dialog()
        dialog._extra_edits["dump.c_spin[5]"].setText("mag_force_x")

        overrides = dialog.result_overrides()
        self.assertIsNotNone(overrides)
        self.assertIn(("dump.c_spin[5]", "mag_force_x"), overrides.extra_names)

    def test_duplicate_extra_names_disable_confirm(self):
        dialog = self._dialog()
        first = dialog._extra_edits["dump.c_spin[5]"]
        second = dialog._extra_edits["dump.c_spin[6]"]
        second.setText(first.text())

        self.assertFalse(dialog.yesButton.isEnabled())
        second.setText("mag_force_y")
        self.assertTrue(dialog.yesButton.isEnabled())

    def test_custom_field_merges_columns(self):
        dialog = self._dialog()
        dialog._on_add_field()
        row = dialog._custom_rows[-1]
        row.name.setText("foo")
        row.columns.setText("c_spin[5], c_spin[6]")
        dialog._refresh_extras()

        overrides = dialog.result_overrides()
        self.assertIsNotNone(overrides)
        self.assertIn(("foo", ("c_spin[5]", "c_spin[6]")), overrides.custom_fields)
        self.assertNotIn("dump.c_spin[5]", dialog._extra_checks)
        self.assertEqual(row.preview.text(), "→ foo:R:2")
        self.assertNotIn("dump.c_spin[6]", dialog._extra_checks)

    def test_unknown_custom_column_disables_confirm(self):
        dialog = self._dialog()
        dialog._on_add_field()
        row = dialog._custom_rows[-1]
        row.name.setText("foo")
        row.columns.setText("not_a_column")

        self.assertFalse(dialog.yesButton.isEnabled())
        self.assertTrue(dialog.errorLabel.text())

    def test_saved_custom_fields_are_restored(self):
        overrides = ColumnOverrides(custom_fields=(("foo", ("c_spin[5]", "c_spin[6]")),))
        dialog = self._dialog(overrides=overrides)

        self.assertEqual(len(dialog._custom_rows), 1)
        row = dialog._custom_rows[0]
        self.assertEqual(row.name.text(), "foo")
        self.assertEqual(row.columns.text(), "c_spin[5], c_spin[6]")
        result = dialog.result_overrides()
        self.assertIn(("foo", ("c_spin[5]", "c_spin[6]")), result.custom_fields)

    def test_reset_restores_automatic(self):
        dialog = self._dialog()
        direction = dialog._spin_combos[1]
        direction.setCurrentIndex(_find_data(direction, "c_spin[5]"))
        dialog._extra_checks["dump.c_spin[5]"].setChecked(False)
        dialog._extra_edits["dump.c_spin[6]"].setText("renamed")
        self.assertIsNotNone(dialog.result_overrides())

        dialog._reset()

        self.assertIsNone(dialog.result_overrides())
        self.assertEqual(dialog._extra_edits["dump.c_spin[6]"].text(), "dump.c_spin[6]")
