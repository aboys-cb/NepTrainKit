from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QDialog

from NepTrainKit.ui.widgets import PeriodicTableDialog


def _close(dialog: PeriodicTableDialog, app: QApplication) -> None:
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_periodic_table_exposes_all_elements_in_standard_positions():
    app = QApplication.instance() or QApplication([])
    dialog = PeriodicTableDialog()

    assert len(dialog.element_buttons) == 118
    assert dialog._grid_position(1, 1, 1) == (0, 0)
    assert dialog._grid_position(2, 18, 1) == (0, 17)
    assert dialog._grid_position(57, 0, 6) == (7, 2)
    assert dialog._grid_position(89, 0, 7) == (8, 2)
    assert dialog.element_buttons["Fe"].toolTip() == "26 · Fe · Iron"
    assert dialog.element_buttons["Fe"].accessibleName()
    assert dialog.choose_button.isEnabled() is False

    _close(dialog, app)


def test_periodic_table_search_selects_and_commits_one_element():
    app = QApplication.instance() or QApplication([])
    dialog = PeriodicTableDialog()
    selected = QSignalSpy(dialog.elementSelected)

    dialog.search_edit.setText("iron")
    matches = [button.symbol for button in dialog.element_buttons.values() if button.search_match]
    assert matches == ["Fe"]

    dialog.search_edit.returnPressed.emit()
    assert dialog.selected_symbol == "Fe"
    assert dialog.element_buttons["Fe"].isChecked()
    assert dialog.selection_label.text() == "Fe · Iron · atomic number 26"
    assert dialog.choose_button.text() == "Choose Fe"
    assert dialog.choose_button.isEnabled()

    dialog.choose_button.click()
    assert selected.count() == 1
    assert selected.at(0)[0] == "Fe"
    assert dialog.result() == QDialog.DialogCode.Accepted

    _close(dialog, app)


def test_periodic_table_double_click_commits_and_closes():
    app = QApplication.instance() or QApplication([])
    dialog = PeriodicTableDialog()
    selected = QSignalSpy(dialog.elementSelected)
    iron = dialog.element_buttons["Fe"]

    QTest.mouseDClick(iron, Qt.MouseButton.LeftButton)

    assert dialog.selected_symbol == "Fe"
    assert iron.isChecked()
    assert selected.count() == 1
    assert selected.at(0)[0] == "Fe"
    assert dialog.result() == QDialog.DialogCode.Accepted

    _close(dialog, app)
