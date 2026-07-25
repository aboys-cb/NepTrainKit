import sys

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from NepTrainKit.core.search import StructureFilterValidationError
from NepTrainKit.core.types import (
    FilterField,
    SearchType,
    StructureFilterCondition,
    StructureFilterSpec,
    TextMatchMode,
)
from NepTrainKit.ui.widgets import structure_filter_bar as filter_bar_module
from NepTrainKit.ui.widgets.structure_filter_bar import StructureFilterBar


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def bar(qapp):
    widget = StructureFilterBar()
    widget.resize(700, 38)
    widget.show()
    widget.raise_()
    widget.activateWindow()
    qapp.processEvents()
    yield widget
    widget._popup._debounce.stop()
    widget._popup.close()
    widget.close()
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _spec(*conditions):
    return StructureFilterSpec(conditions=tuple(conditions))


def _condition(condition_id, field, *values, mode=None, enabled=True):
    return StructureFilterCondition(
        condition_id=condition_id,
        field=field,
        enabled=enabled,
        text_values=tuple(values),
        match_mode=mode,
        case_sensitive=field == FilterField.FORMULA,
    )


def test_expression_unit_uses_magnetic_force_unit():
    assert filter_bar_module._detect_expression_unit("atomic.force_mag.norm > 0") == "μB/Å"
    assert filter_bar_module._detect_expression_unit("atomic.mforce.x < 1") == "μB/Å"


def test_empty_state_disables_preview_and_apply_without_redundant_clear_action(bar):
    assert not bar.preview_button.isEnabled()
    assert not bar.apply_button.isEnabled()
    assert not hasattr(bar, "clear_button")
    assert not hasattr(bar, "add_button")
    entry = bar.chip_layout.itemAt(0).widget()
    assert isinstance(entry, QPushButton)
    assert entry.text() == bar.tr("Filter conditions")


def test_spec_builds_chips_and_result_enables_cached_actions(bar):
    empty_entry = bar.chip_layout.itemAt(0).widget()
    spec = _spec(
        _condition("tag", FilterField.CONFIG_TYPE, "surface", mode=TextMatchMode.CONTAINS),
        _condition("elements", FilterField.ELEMENT_REQUIRED, "Fe", "O"),
    )
    emitted = []
    bar.specChanged.connect(emitted.append)
    bar.set_spec(spec)
    assert emitted == [spec]
    assert empty_entry.isHidden()
    assert len(bar._chips) == 2
    assert bar.preview_button.isEnabled()
    assert not bar.apply_button.isEnabled()

    bar.set_result(12, 100, 4.5)
    assert bar.apply_button.isEnabled()
    assert "12" in bar.match_button.text()


def test_changing_or_removing_condition_marks_result_stale(bar):
    spec = _spec(_condition("tag", FilterField.CONFIG_TYPE, "surface"))
    bar.set_spec(spec)
    bar.set_result(3, 10, 1.0)
    bar._remove_condition("tag")
    assert bar.spec.is_empty()
    assert not bar._result_current


def test_element_editor_preserves_all_three_set_relations(bar):
    spec = _spec(
        _condition("required", FilterField.ELEMENT_REQUIRED, "Fe", "O"),
        _condition("excluded", FilterField.ELEMENT_EXCLUDED, "H"),
        _condition("allowed", FilterField.ELEMENT_ALLOWED, "Fe", "O", "C"),
    )
    bar.set_spec(spec)
    edited = bar._popup.spec()
    assert [condition.field for condition in edited.conditions] == [
        FilterField.ELEMENT_REQUIRED,
        FilterField.ELEMENT_EXCLUDED,
        FilterField.ELEMENT_ALLOWED,
    ]
    assert edited.conditions[0].text_values == ("Fe", "O")


def test_error_is_attached_to_matching_condition_row(bar):
    bar.set_spec(_spec(_condition("bad", FilterField.CONFIG_TYPE, "[", mode=TextMatchMode.REGEX)))
    error = StructureFilterValidationError("invalid_regex", "Invalid regex", "bad")
    bar.set_error(error)
    assert not bar._popup.error_label.isHidden()
    assert bar._popup._rows[0]._error


def test_editor_input_and_controls_emit_without_signal_arity_errors(bar, qapp):
    bar.set_spec(_spec(_condition("tag", FilterField.CONFIG_TYPE, "surface")))
    emitted = []
    previews = []
    bar.specChanged.connect(emitted.append)
    bar.previewRequested.connect(lambda: previews.append(True))

    row = bar._popup._rows[0]
    row.value_edit.setText("bulk")
    row.mode_combo.setCurrentIndex(1)
    row.case_button.setChecked(True)
    row.enabled_switch.setChecked(False)
    row.enabled_switch.setChecked(True)
    bar._popup.logic_combo.setCurrentIndex(1)
    bar.preview_button.click()
    qapp.processEvents()

    assert emitted
    assert previews == [True]


def test_match_case_button_is_explicit_and_preserved_in_spec(bar, qapp):
    bar.set_spec(_spec(_condition("tag", FilterField.CONFIG_TYPE, "Surface")))
    row = bar._popup._rows[0]

    assert not row.case_button.isHidden()
    assert not row.case_button.isChecked()
    row.case_button.click()
    qapp.processEvents()
    assert row.to_condition().case_sensitive

    row.field_combo.setCurrentIndex(1)
    qapp.processEvents()
    assert not row.case_button.isHidden()
    assert row.case_button.isChecked()

    row.field_combo.setCurrentIndex(2)
    qapp.processEvents()
    assert row.case_button.isHidden()
    assert not row.to_condition().case_sensitive


def test_narrow_width_collapses_extra_chips(bar, qapp):
    bar.set_spec(
        _spec(
            *(
                _condition(str(index), FilterField.CONFIG_TYPE, f"condition-{index}")
                for index in range(5)
            )
        )
    )
    bar.resize(430, 38)
    qapp.processEvents()
    bar._update_chip_overflow()
    assert bar._overflow_button is not None


def test_continuous_add_remove_and_resize_keeps_overflow_widgets_alive(bar, qapp):
    bar.resize(430, 38)
    bar.set_spec(
        _spec(
            *(
                _condition(str(index), FilterField.CONFIG_TYPE, f"condition-{index}")
                for index in range(5)
            )
        )
    )
    qapp.processEvents()
    old_overflow = bar._overflow_button
    assert old_overflow is not None
    assert shiboken6.isValid(old_overflow)

    for _ in range(8):
        bar._popup.add_condition()
        bar.resize(431, 38)
        bar.resize(430, 38)
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        if bar._overflow_button is not None:
            assert shiboken6.isValid(bar._overflow_button)

    assert len(bar.spec.conditions) == 13
    assert bar._overflow_button is not old_overflow

    for row in list(bar._popup._rows[-6:]):
        bar._popup._remove_row(row)
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        if bar._overflow_button is not None:
            assert shiboken6.isValid(bar._overflow_button)
    bar._popup._debounce.stop()


def test_real_click_typing_add_remove_clear_flow_has_no_qt_slot_errors(bar, qapp):
    slot_errors = []
    previous_hook = sys.excepthook
    sys.excepthook = lambda exc_type, value, traceback: slot_errors.append(value)
    try:
        entry = bar.chip_layout.itemAt(0).widget()
        entry.click()
        qapp.processEvents()
        assert bar.editor_is_open
        assert len(bar._popup._rows) == 1

        first = bar._popup._rows[0]
        first.value_edit.setText("surface")
        first.mode_combo.setCurrentIndex(1)
        first.enabled_switch.setChecked(False)
        first.enabled_switch.setChecked(True)

        for index in range(10):
            bar._popup.add_button.click()
            row = bar._popup._rows[-1]
            row.field_combo.setCurrentIndex(2 if index % 2 else 1)
            row.mode_combo.setCurrentIndex(index % max(1, row.mode_combo.count()))
            row.value_edit.setText("Fe, O" if index % 2 else f"Fe{index + 1}O")
            bar.resize(430 + index % 3, 38)
            qapp.processEvents()

        for row in list(bar._popup._rows[-4:]):
            row.remove_button.click()
            qapp.processEvents()

        QTest.qWait(320)
        bar._popup.clear_button.click()
        qapp.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert bar.spec.conditions == ()
        assert slot_errors == []
    finally:
        sys.excepthook = previous_hook
        bar._popup._debounce.stop()
        bar._popup.close()


def test_popup_grows_to_five_rows_then_scrolls_and_inputs_have_field_hints(bar, qapp):
    entry = bar.chip_layout.itemAt(0).widget()
    entry.click()
    qapp.processEvents()
    popup = bar._popup
    one_row_height = popup.height()
    row = popup._rows[0]
    assert "surface; bulk" in row.value_edit.placeholderText()
    assert "separate multiple values" in row.value_edit.toolTip().lower()

    row.field_combo.setCurrentIndex(1)
    assert "Fe2O3" in row.value_edit.placeholderText()
    row.field_combo.setCurrentIndex(2)
    assert "Fe, O" in row.value_edit.placeholderText()
    row.field_combo.setCurrentIndex(3)
    assert "natoms > 100" in row.value_edit.placeholderText()
    assert "must be conditions" in row.value_edit.toolTip().lower()

    for _ in range(4):
        popup.add_button.click()
        qapp.processEvents()
    five_row_height = popup.height()
    assert five_row_height > one_row_height
    assert all(current.height() == popup._ROW_HEIGHT for current in popup._rows)

    popup.add_button.click()
    qapp.processEvents()
    assert popup.height() == five_row_height
    assert popup.scroll.verticalScrollBar().maximum() > 0
    assert popup.scroll.verticalScrollBar().value() > 0

    for current in list(popup._rows[-2:]):
        current.remove_button.click()
        qapp.processEvents()
    assert popup.height() < five_row_height
    popup._debounce.stop()
    popup.close()


def test_dataset_suggestions_follow_field_and_replace_only_the_active_token(bar, qapp):
    bar.set_suggestions(
        {
            SearchType.TAG: {"surface_relax": 12, "bulk": 7},
            SearchType.FORMULA: {"Fe2O3": 5},
            SearchType.ELEMENTS: {"Fe": 20, "O": 18},
            SearchType.EXPRESSION: {"energy_per_atom": 10, "natoms": 20},
        }
    )
    entry = bar.chip_layout.itemAt(0).widget()
    entry.click()
    qapp.processEvents()
    row = bar._popup._rows[0]
    edit = row.value_edit

    edit.setText("bulk; sur")
    edit.setCursorPosition(len(edit.text()))
    edit.completer().setCompletionPrefix("SUR")
    assert edit.completer().completionModel().rowCount() == 1
    edit._accept_completion("surface_relax")
    assert edit.text() == "bulk; surface_relax"

    row.field_combo.setCurrentIndex(2)
    edit.setText("Fe, o")
    edit.setCursorPosition(len(edit.text()))
    edit._accept_completion("O")
    assert edit.text() == "Fe, O"

    row.field_combo.setCurrentIndex(3)
    edit.setText("natoms > ene")
    edit.setCursorPosition(len(edit.text()))
    edit._accept_completion("energy_per_atom")
    assert edit.text() == "natoms > energy_per_atom"
    bar._popup._debounce.stop()
    bar._popup.close()


def test_saved_filters_menu_is_in_the_editor_header_and_tracks_current_spec(bar, monkeypatch):
    monkeypatch.setattr(
        filter_bar_module,
        "list_structure_filter_preset_names",
        lambda: ["Fe-O cleanup", "Surface structures"],
    )
    popup = bar._popup
    popup.refresh_presets()

    assert popup.preset_button.text() == popup.tr("Saved filters")
    assert popup.preset_button.height() == popup.logic_combo.height()
    assert [action.text() for action in popup.preset_menu.actions()[:2]] == [
        "Fe-O cleanup",
        "Surface structures",
    ]
    assert not popup._save_preset_action.isEnabled()

    bar.set_spec(_spec(_condition("tag", FilterField.CONFIG_TYPE, "surface")))
    assert popup._save_preset_action.isEnabled()


def test_loading_saved_filter_replaces_conditions_and_never_applies_selection(bar, monkeypatch, qapp):
    saved = _spec(
        _condition("saved", FilterField.CONFIG_TYPE, "bulk", mode=TextMatchMode.EXACT),
        _condition("oxygen", FilterField.ELEMENT_REQUIRED, "O"),
    )
    monkeypatch.setattr(filter_bar_module, "load_structure_filter_preset", lambda name: saved)
    applied = []
    bar.applyRequested.connect(applied.append)

    bar._popup._load_preset("Fe-O cleanup")
    qapp.processEvents()

    assert bar.spec == saved
    assert len(bar._popup._rows) == 2
    assert applied == []
    bar._popup._debounce.stop()


def test_save_menu_action_consumes_qaction_signal_and_persists_only_the_spec(
    bar,
    monkeypatch,
    qapp,
):
    bar.set_spec(
        _spec(
            _condition(
                "tag",
                FilterField.CONFIG_TYPE,
                "surface",
                mode=TextMatchMode.CONTAINS,
            )
        )
    )
    saved = []
    monkeypatch.setattr(bar._popup, "_prompt_preset_name", lambda *args: "My filter")
    monkeypatch.setattr(bar._popup, "_restore_after_dialog", lambda: None)
    monkeypatch.setattr(filter_bar_module, "structure_filter_preset_exists", lambda name: False)
    monkeypatch.setattr(
        filter_bar_module,
        "save_structure_filter_preset",
        lambda name, spec: saved.append((name, spec)),
    )
    monkeypatch.setattr(filter_bar_module, "list_structure_filter_preset_names", lambda: [])
    bar._popup.refresh_presets()

    bar._popup._save_preset_action.trigger()
    qapp.processEvents()

    assert saved == [("My filter", bar.spec)]


def test_refusing_overwrite_keeps_the_existing_saved_filter(bar, monkeypatch):
    bar.set_spec(
        _spec(_condition("tag", FilterField.CONFIG_TYPE, "surface", mode=TextMatchMode.CONTAINS))
    )
    saved = []
    monkeypatch.setattr(bar._popup, "_prompt_preset_name", lambda *args: "Existing")
    monkeypatch.setattr(bar._popup, "_confirm", lambda *args: False)
    monkeypatch.setattr(bar._popup, "_restore_after_dialog", lambda: None)
    monkeypatch.setattr(filter_bar_module, "structure_filter_preset_exists", lambda name: True)
    monkeypatch.setattr(
        filter_bar_module,
        "save_structure_filter_preset",
        lambda name, spec: saved.append((name, spec)),
    )

    bar._popup._save_current_preset()

    assert saved == []


def test_saved_filter_can_be_renamed_and_deleted_from_the_menu(bar, monkeypatch):
    names = ["Old name"]
    monkeypatch.setattr(filter_bar_module, "list_structure_filter_preset_names", lambda: list(names))
    monkeypatch.setattr(bar._popup, "_restore_after_dialog", lambda: None)
    monkeypatch.setattr(bar._popup, "_prompt_preset_name", lambda *args: "New name")
    monkeypatch.setattr(bar._popup, "_confirm", lambda *args: True)
    monkeypatch.setattr(filter_bar_module, "structure_filter_preset_exists", lambda name: False)

    renamed = []

    def rename(old_name, new_name):
        renamed.append((old_name, new_name))
        names[:] = [new_name]
        return True

    deleted = []

    def delete(name):
        deleted.append(name)
        names.clear()
        return True

    monkeypatch.setattr(filter_bar_module, "rename_structure_filter_preset", rename)
    monkeypatch.setattr(filter_bar_module, "delete_structure_filter_preset", delete)
    bar._popup.refresh_presets()

    bar._popup._rename_preset_menu.actions()[0].trigger()
    assert renamed == [("Old name", "New name")]
    assert bar._popup.preset_menu.actions()[0].text() == "New name"

    bar._popup._delete_preset_menu.actions()[0].trigger()
    assert deleted == ["New name"]
    assert bar._popup.preset_menu.actions()[0].text() == bar._popup.tr("No saved filters")


def test_damaged_saved_filter_does_not_change_current_conditions(bar, monkeypatch):
    current = _spec(
        _condition("tag", FilterField.CONFIG_TYPE, "surface", mode=TextMatchMode.CONTAINS)
    )
    bar.set_spec(current)
    warnings = []
    applied = []
    bar.applyRequested.connect(applied.append)
    monkeypatch.setattr(filter_bar_module, "load_structure_filter_preset", lambda name: None)
    monkeypatch.setattr(filter_bar_module, "list_structure_filter_preset_names", lambda: [])
    monkeypatch.setattr(
        filter_bar_module.MessageManager,
        "send_warning_message",
        lambda message: warnings.append(message),
    )

    bar._popup._load_preset("Broken")

    assert bar.spec == current
    assert applied == []
    assert warnings


def test_rebuilding_saved_filter_menu_does_not_reuse_deleted_qt_actions(bar, monkeypatch, qapp):
    monkeypatch.setattr(filter_bar_module, "list_structure_filter_preset_names", lambda: ["Common"])
    loaded = []
    saved = _spec(
        _condition("saved", FilterField.CONFIG_TYPE, "bulk", mode=TextMatchMode.CONTAINS)
    )
    monkeypatch.setattr(filter_bar_module, "load_structure_filter_preset", lambda name: saved)
    monkeypatch.setattr(filter_bar_module.MessageManager, "send_success_message", lambda message: None)
    bar.specChanged.connect(lambda spec: loaded.append(spec))

    for _ in range(20):
        bar._popup.refresh_presets()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
        assert bar._popup.preset_menu.view.count() == 4

    action = bar._popup.preset_menu.actions()[0]
    assert shiboken6.isValid(action)
    action.trigger()
    qapp.processEvents()
    assert loaded


def test_narrow_english_layout_does_not_clip_header_rows_footer_or_bar_actions(bar, qapp):
    bar.resize(560, 38)
    entry = bar.chip_layout.itemAt(0).widget()
    entry.click()
    popup = bar._popup
    for _ in range(4):
        popup.add_button.click()
    popup.set_estimate(128, 2460)
    qapp.processEvents()

    assert popup.width() == popup.minimumWidth() == 620
    assert popup.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert popup.graphicsEffect() is None
    assert popup.card.graphicsEffect() is not None
    assert popup.preset_button.geometry().right() < popup.logic_combo.geometry().left()
    assert popup.preset_button.width() >= popup.preset_button.sizeHint().width()
    assert popup.logic_combo.width() >= popup.logic_combo.sizeHint().width()
    assert popup.done_button.width() >= popup.done_button.sizeHint().width()
    for widget in (popup.add_button, popup.estimate_label, popup.clear_button, popup.done_button):
        assert widget.geometry().right() <= popup.width() - 10

    row = popup._rows[0]
    assert row.enabled_switch.width() >= row.enabled_switch.minimumSizeHint().width()
    assert row.case_button.width() >= row.case_button.minimumSizeHint().width()
    field_text_width = max(
        row.field_combo.fontMetrics().horizontalAdvance(row.field_combo.itemText(index))
        for index in range(row.field_combo.count())
    )
    mode_text_width = max(
        row.mode_combo.fontMetrics().horizontalAdvance(row.mode_combo.itemText(index))
        for index in range(row.mode_combo.count())
    )
    assert row.field_combo.width() >= field_text_width + 30
    assert row.mode_combo.width() >= mode_text_width + 30
    assert bar.preview_button.width() >= bar.preview_button.sizeHint().width()
    assert bar.apply_button.width() >= bar.apply_button.sizeHint().width()
    popup._debounce.stop()
    popup.close()
