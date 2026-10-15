import sys

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from NepTrainKit.core.search import StructureFilterValidationError
from NepTrainKit.core.types import (
    FilterField,
    SearchType,
    StructureFilterCondition,
    StructureFilterSpec,
    TextMatchMode,
)
from NepTrainKit.ui.widgets import fluent_overlays
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
    assert filter_bar_module._detect_expression_unit("atomic.force_mag.norm > 0") == "eV/μB"
    assert filter_bar_module._detect_expression_unit("atomic.mforce.x < 1") == "eV/μB"
    assert filter_bar_module._detect_expression_unit("any(atomic.spin.norm < 1)") == "μB"
    assert filter_bar_module._detect_expression_unit("all(force.err.norm < 0.2)") == "eV/Å"
    assert filter_bar_module._detect_expression_unit("not any(atomic.charge < 0)") == "e"


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
    # A stale or missing preview can still be applied: it is recomputed first.
    assert bar.apply_button.isEnabled()

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


def test_real_click_typing_add_remove_clear_flow_has_no_qt_slot_errors(bar, qapp, monkeypatch):
    monkeypatch.setattr(bar._popup, "_confirm", lambda *args: True)
    monkeypatch.setattr(bar._popup, "_restore_after_dialog", lambda: None)
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
    assert all(current.height() == popup._rows[0].height() for current in popup._rows)

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
    if sys.platform == "win32":
        # The Windows completion popup must stay out of the layered-window path.
        assert edit._completerMenu.view.graphicsEffect() is None

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
        # Preset, separator, save, add-to-current, manage.
        assert bar._popup.preset_menu.view.count() == 5

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
    assert popup.graphicsEffect() is None
    assert popup.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    if sys.platform == "win32":
        # Windows keeps the rounded surface but drops the overflowing shadow.
        assert popup.card.graphicsEffect() is None
    else:
        assert popup.card.graphicsEffect() is not None
    assert not popup.preset_button.geometry().intersects(popup.logic_combo.geometry())
    assert popup.preset_button.width() >= (
        popup.preset_button.fontMetrics().horizontalAdvance(
            popup.preset_button.text()
        )
        + 64
    )
    logic_text_width = max(
        popup.logic_combo.fontMetrics().horizontalAdvance(
            popup.logic_combo.itemText(index)
        )
        for index in range(popup.logic_combo.count())
    )
    assert popup.logic_combo.width() >= logic_text_width + 48
    assert popup.done_button.width() >= popup.done_button.sizeHint().width()
    for widget in (popup.add_button, popup.estimate_label, popup.clear_button, popup.done_button):
        assert widget.geometry().right() <= popup.width() - 10

    row = popup._rows[0]
    assert row.enabled_switch.width() >= row.enabled_switch.minimumSizeHint().width()
    assert row.case_button.width() >= row.case_button.minimumSizeHint().width()
    for index in range(row.field_combo.count()):
        row.field_combo.setCurrentIndex(index)
        qapp.processEvents()
        field_text_width = row.field_combo.fontMetrics().horizontalAdvance(
            row.field_combo.currentText()
        )
        mode_text_width = max(
            row.mode_combo.fontMetrics().horizontalAdvance(row.mode_combo.itemText(mode_index))
            for mode_index in range(row.mode_combo.count())
        )
        assert row.field_combo.width() >= field_text_width + 30
        assert row.mode_combo.width() >= mode_text_width + 30
        assert row.remove_button.geometry().right() <= row.width()
    assert bar.preview_button.width() >= bar.preview_button.sizeHint().width()
    assert bar.apply_button.width() >= bar.apply_button.sizeHint().width()
    popup._debounce.stop()
    popup.close()


def test_windows_popup_keeps_every_effect_inside_its_window(qapp, monkeypatch):
    """Windows drops layered updates whose dirty region leaves the popup window."""
    monkeypatch.setattr(fluent_overlays, "_IS_WINDOWS", True)
    popup = filter_bar_module.StructureFilterEditorPopup()
    try:
        popup.set_spec(
            _spec(
                _condition(
                    "tag",
                    FilterField.CONFIG_TYPE,
                    "surface",
                    "bulk",
                    mode=TextMatchMode.CONTAINS,
                )
            )
        )
        popup.resize(687, popup.sizeHint().height())
        popup.show()
        qapp.processEvents()
        assert popup.mask().boundingRect() == popup.rect()
        window = popup.rect()
        for widget in [popup, *popup.findChildren(QWidget)]:
            effect = widget.graphicsEffect()
            # Menus and tooltips are top-level windows of their own; qfluentwidgets
            # clips their shadows with a mask, so only the popup window matters here.
            if effect is None or widget.window() is not popup:
                continue
            offset = widget.mapTo(popup, QPoint(0, 0))
            bounds = effect.boundingRectFor(QRectF(widget.rect())).translated(
                float(offset.x()), float(offset.y())
            )
            assert window.contains(bounds.toAlignedRect()), widget.objectName()
    finally:
        popup._debounce.stop()
        popup.close()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_match_result_details_use_the_fluent_tooltip(bar, qapp):
    bar.set_result(12, 100, 4.5)
    bar.match_button.click()
    qapp.processEvents()
    tooltip = bar._result_tip
    assert isinstance(tooltip, filter_bar_module.ToolTip)
    assert "Matched structures: 12" in tooltip.text()


def _atomic_suggestions():
    return {
        SearchType.ELEMENTS: {'Fe': 5, 'O': 5},
        SearchType.EXPRESSION: {
            'atomic.element': 6, 'atomic.spin': 6, 'atomic.spin.norm': 6, 'atomic.spin.z': 6,
            'atomic.charge': 6, 'force.norm': 6, 'force.ref.norm': 6, 'force.err.norm': 6,
            'natoms': 6, 'energy': 6, 'stress.norm': 6,
        },
    }


def _choose(combo, data):
    for index in range(combo.count()):
        if combo.itemData(index) == data:
            combo.setCurrentIndex(index)
            return
    raise AssertionError(f'Missing choice {data}')


def test_atomic_controls_generate_element_scoped_any_all_and_round_trip(bar, qapp):
    bar.set_suggestions(_atomic_suggestions())
    bar._popup.add_condition()
    row = bar._popup._rows[0]
    _choose(row.field_combo, 'atomic')
    editor = row.atomic_editor
    _choose(editor.element_combo, 'Fe')
    _choose(editor.field_combo, 'atomic.spin.norm')
    editor.value_edit.setText('1')
    assert row.to_condition().text_values == ('any(atomic.element == "Fe" && atomic.spin.norm < 1)',)
    assert row.value_edit.isHidden()
    assert row.mode_combo.isHidden()
    assert not editor.isHidden()
    assert editor.unit_label.text() == 'μB'
    assert 'Only Fe' in editor.hint_label.text()
    _choose(editor.quantifier_combo, 'all')
    expected = 'has.Fe && all(atomic.element != "Fe" || atomic.spin.norm < 1)'
    assert row.to_condition().text_values == (expected,)
    saved = StructureFilterSpec.from_dict(bar._popup.spec().to_dict())
    bar.set_spec(saved)
    restored = bar._popup._rows[0]
    assert restored.field_combo.currentData() == 'atomic'
    assert restored.atomic_editor.quantifier_combo.currentData() == 'all'
    assert restored.to_condition() == saved.conditions[0]
    assert 'All Fe' in bar._condition_summary(saved.conditions[0])[1]
    # Switching to the advanced entry preserves the exact generated condition.
    _choose(restored.field_combo, FilterField.EXPRESSION)
    assert restored.value_edit.text() == expected
    assert restored.atomic_editor.isHidden()
    _choose(restored.field_combo, 'atomic')
    assert restored.to_condition().text_values == (expected,)


def test_atomic_controls_limit_available_fields_and_keep_custom_expressions(bar):
    bar.set_suggestions(_atomic_suggestions())
    expression = 'any(atomic.spin.norm < 1 && force.norm > 10)'
    bar.set_spec(_spec(_condition('complex', FilterField.EXPRESSION, expression)))
    row = bar._popup._rows[0]
    assert row.field_combo.currentData() == FilterField.EXPRESSION
    assert row.to_condition().text_values == (expression,)
    fields = [row.atomic_editor.field_combo.itemData(i) for i in range(row.atomic_editor.field_combo.count())]
    assert 'atomic.spin.norm' in fields and 'force.err.norm' in fields
    assert 'atomic.spin' not in fields and 'atomic.element' not in fields
    assert 'natoms' not in fields and 'stress.norm' not in fields
    assert 'force.ref.norm' not in fields  # No duplicate of the default reference view.


def test_atomic_controls_keep_negative_scientific_thresholds_and_reject_incomplete_input(bar):
    bar.set_suggestions(_atomic_suggestions())
    expression = 'any(atomic.element == "Fe" && atomic.spin.z < -1e-3)'
    bar.set_spec(_spec(_condition('signed', FilterField.EXPRESSION, expression)))
    row = bar._popup._rows[0]
    assert row.to_condition().text_values == (expression,)
    for text in ('', '-', '1e', 'nan', 'inf', '1e999'):
        row.atomic_editor.value_edit.setText(text)
        assert row.to_condition().text_values == ()
    row.atomic_editor.value_edit.setText('0.25')
    _choose(row.atomic_editor.element_combo, '')
    assert row.to_condition().text_values == ('any(atomic.spin.z < 0.25)',)
    row.enabled_switch.setChecked(False)
    assert not row.to_condition().enabled


@pytest.mark.parametrize('long_field', [False, True])
def test_atomic_popup_grows_on_mode_switch_and_remains_within_width(bar, qapp, long_field):
    bar.set_suggestions(_atomic_suggestions())
    bar.open_editor(add_if_empty=True)
    row = bar._popup._rows[0]
    original_height = row.height()
    _choose(row.field_combo, 'atomic')
    if long_field:
        # Custom property names and wider platform fonts must not force a
        # hidden horizontal scrollbar or clip the threshold input.
        field = 'atomic.custom_long_property_name_for_local_environment.norm'
        row.atomic_editor.field_combo.addItem(field, userData=field)
        _choose(row.atomic_editor.field_combo, field)
    qapp.processEvents()
    assert row.height() > original_height
    assert bar._popup.scroll.height() >= row.height()
    editor = row.atomic_editor
    for control in (editor.element_combo, editor.quantifier_combo, editor.field_combo,
                    editor.operator_combo, editor.value_edit, editor.unit_label):
        assert control.geometry().right() < editor.width()
        assert control.geometry().bottom() < editor.height()
    assert editor.value_edit.width() >= 70
    assert bar._popup.scroll.horizontalScrollBar().maximum() == 0
    _choose(row.field_combo, FilterField.EXPRESSION)
    qapp.processEvents()
    assert row.height() == original_height


def test_atomic_visual_condition_uses_existing_backend_for_mixed_elements(bar):
    from test_base import _build_element_quantifier_result

    data = _build_element_quantifier_result()
    bar.set_suggestions({kind: data.get_completer_cache(kind) for kind in (
        SearchType.ELEMENTS, SearchType.EXPRESSION,
    )})
    bar._popup.add_condition()
    row = bar._popup._rows[0]
    _choose(row.field_combo, 'atomic')
    _choose(row.atomic_editor.element_combo, 'Fe')
    _choose(row.atomic_editor.field_combo, 'atomic.spin.norm')
    assert data.search_structures(bar._popup.spec()).indices == (1, 2, 3, 5)
    _choose(row.atomic_editor.quantifier_combo, 'all')
    assert data.search_structures(bar._popup.spec()).indices == (1, 3, 5)
    assert not data.select_index


def test_atomic_editor_handles_no_data_and_structure_norm_force_suggestions(bar):
    bar._popup.add_condition()
    row = bar._popup._rows[0]
    _choose(row.field_combo, 'atomic')
    assert row.to_condition().text_values == ()
    assert 'Load a dataset' in row.atomic_editor.hint_label.text()
    bar.set_suggestions({SearchType.EXPRESSION: {'force': 3, 'force.ref': 3, 'force.err': 3}})
    fields = [row.atomic_editor.field_combo.itemData(i) for i in range(row.atomic_editor.field_combo.count())]
    assert fields == ['force.norm', 'force.err.norm']
    assert row.to_condition().text_values == ('any(force.norm < 1)',)
    assert bar.spec.conditions[0].text_values == ('any(force.norm < 1)',)


def test_atomic_editor_distinguishes_properties_with_the_same_friendly_label(bar):
    bar.set_suggestions({SearchType.EXPRESSION: {'atomic.spin.norm': 2, 'atomic.spin_vec.norm': 2}})
    bar._popup.add_condition()
    combo = bar._popup._rows[0].atomic_editor.field_combo
    assert combo.itemText(0) != combo.itemText(1)
    assert '(spin)' in combo.itemText(0)
    assert '(spin_vec)' in combo.itemText(1)


def test_switching_fields_keeps_each_draft_without_leaking_values(bar):
    bar.set_suggestions(_atomic_suggestions())
    custom = 'natoms > 100 && energy < 0'
    bar.set_spec(_spec(_condition('expr', FilterField.EXPRESSION, custom)))
    row = bar._popup._rows[0]

    # An untouched detour through the visual editor restores the custom expression.
    _choose(row.field_combo, 'atomic')
    assert row.to_condition().text_values == ('any(atomic.spin.norm < 1)',)
    _choose(row.field_combo, FilterField.EXPRESSION)
    assert row.value_edit.text() == custom
    assert row.to_condition().text_values == (custom,)

    # Generated atomic expressions never become text values of other fields.
    _choose(row.field_combo, 'atomic')
    row.atomic_editor.value_edit.setText('3')
    _choose(row.field_combo, FilterField.CONFIG_TYPE)
    assert row.value_edit.text() == ''
    row.value_edit.setText('surface')
    _choose(row.field_combo, FilterField.FORMULA)
    assert row.value_edit.text() == ''
    _choose(row.field_combo, FilterField.CONFIG_TYPE)
    assert row.to_condition().text_values == ('surface',)

    # Edited visual conditions continue as raw expressions.
    _choose(row.field_combo, 'atomic')
    row.atomic_editor.value_edit.setText('2')
    _choose(row.field_combo, FilterField.EXPRESSION)
    assert row.to_condition().text_values == ('any(atomic.spin.norm < 2)',)


def test_incomplete_atomic_threshold_is_flagged_instead_of_silently_ignored(bar):
    bar.set_suggestions(_atomic_suggestions())
    bar._popup.add_condition()
    row = bar._popup._rows[0]
    _choose(row.field_combo, 'atomic')
    assert not row._local_error

    row.atomic_editor.value_edit.setText('')
    assert row.to_condition().text_values == ()
    message = row.atomic_editor.tr('Enter a finite numeric threshold.')
    assert row._local_error == message
    assert row.atomic_editor.hint_label.text() == message
    assert row.atomic_editor.toolTip() == message
    assert 'border: 1px solid transparent' not in row.styleSheet()
    # Re-emitting the spec clears engine errors but keeps the local hint.
    bar._popup._emit_spec()
    assert row._local_error == message
    assert bar._condition_summary(bar.spec.conditions[0])[1] == bar.tr('Incomplete')
    assert not bar.preview_button.isEnabled()
    assert bar.preview_button.toolTip() == bar.tr('Fill in the filter conditions to preview matches')

    row.atomic_editor.value_edit.setText('0.5')
    assert not row._local_error
    assert 'border: 1px solid transparent' in row.styleSheet()


def test_popup_validation_errors_are_translated(bar, qapp):
    from NepTrainKit import i18n

    bar.set_spec(_spec(_condition('tag', FilterField.CONFIG_TYPE, '(')))
    row = bar._popup._rows[0]
    i18n.install_translator(qapp, 'zh_CN')
    try:
        bar.set_error(
            StructureFilterValidationError(
                'invalid_regex',
                'Invalid regular expression: missing ), unterminated subpattern at position 0',
                'tag',
            )
        )
        expected = '正则表达式无效：missing ), unterminated subpattern at position 0'
        assert bar._popup.error_label.text() == expected
        assert row.value_edit.toolTip() == expected
        assert bar.match_button.toolTip() == expected

        bar.set_error(StructureFilterValidationError('empty_condition', 'The filter condition has no value.', 'tag'))
        assert bar._popup.error_label.text() == '该筛选条件尚未填写值。'
    finally:
        i18n.install_translator(qapp, 'en_US')


def test_not_toggle_round_trips_and_chips_use_editor_wording(bar, qapp):
    bar.set_spec(
        _spec(
            _condition("tag", FilterField.CONFIG_TYPE, "surface", "bulk", mode=TextMatchMode.PREFIX),
            _condition("h", FilterField.ELEMENT_EXCLUDED, "H"),
            _condition("formula", FilterField.FORMULA, "Fe2O3", enabled=False),
        )
    )
    row = bar._popup._rows[0]
    assert not row.to_condition().negated
    row.not_button.click()
    qapp.processEvents()
    assert row.to_condition().negated
    assert bar.spec.conditions[0].negated

    tag, excluded, formula = bar.spec.conditions
    assert bar._condition_summary(tag) == (
        bar.tr("NOT {label}").format(label="Config type starts with"),
        "surface; bulk",
    )
    assert bar._condition_summary(excluded) == ("Must not contain", "H")
    assert bar._condition_summary(formula) == ("Formula equals (Aa)", "Fe2O3")
    assert bar.logic_label is not None and bar.logic_label.text() == "All of:"

    bar._popup.logic_combo.setCurrentIndex(1)
    assert bar.logic_label.text() == "Any of:"
    disabled_chip = next(chip for chip in bar._chips if chip.condition_id == "formula")
    assert disabled_chip.value_label.font().strikeOut()


def test_long_chip_values_are_elided_with_full_text_in_tooltip(bar):
    value = "surface_" + "x" * 80
    bar.set_spec(_spec(_condition("tag", FilterField.CONFIG_TYPE, value)))
    chip = bar._chips[0]
    assert chip.value_label.text().endswith("…")
    assert chip.value_label.fontMetrics().horizontalAdvance(chip.value_label.text()) <= chip._MAX_VALUE_WIDTH
    assert value in chip.toolTip()
    assert bar.logic_label is None


def test_enter_submits_and_explicit_preview_flags_errors_but_auto_preview_stays_quiet(bar, qapp):
    bar.set_spec(_spec(_condition("bad", FilterField.CONFIG_TYPE, "surface")))
    bar.open_editor()
    popup = bar._popup
    row = popup._rows[0]
    previews = []
    bar.previewRequested.connect(lambda: previews.append(True))

    row.value_edit.setText("[")
    assert popup.estimate_label.text() == popup.tr("Updating estimate…")
    popup._debounce.stop()
    popup._auto_preview_requested()
    error = StructureFilterValidationError("invalid_regex", "Invalid regular expression: bad", "bad")
    bar.set_error(error)
    assert not row._error
    assert popup.error_label.isHidden()
    assert "Invalid regular expression" in popup.estimate_label.text()

    QTest.keyClick(row.value_edit, Qt.Key.Key_Return)
    qapp.processEvents()
    assert previews == [True, True]
    assert not popup.isVisible()
    bar.open_editor()
    bar.set_error(error)
    assert popup._rows[0]._error
    assert not popup.error_label.isHidden()
    popup._debounce.stop()
    popup.close()


def test_clear_button_confirms_before_discarding_filled_conditions(bar, monkeypatch):
    bar.set_spec(_spec(_condition("tag", FilterField.CONFIG_TYPE, "surface")))
    popup = bar._popup
    answers = []
    monkeypatch.setattr(popup, "_confirm", lambda *args: answers.pop(0))
    monkeypatch.setattr(popup, "_restore_after_dialog", lambda: None)

    answers.append(False)
    popup.clear_button.click()
    assert bar.spec.conditions and not answers

    answers.append(True)
    popup.clear_button.click()
    assert bar.spec.conditions == ()

    popup.add_condition()
    popup.clear_button.click()  # A blank draft needs no confirmation.
    assert bar.spec.conditions == ()
    popup._debounce.stop()


def test_saved_filter_can_be_appended_to_current_conditions(bar, monkeypatch, qapp):
    saved = StructureFilterSpec(
        conditions=(_condition("saved", FilterField.ELEMENT_REQUIRED, "O"),),
        logic=filter_bar_module.FilterLogic.ANY,
    )
    monkeypatch.setattr(filter_bar_module, "list_structure_filter_preset_names", lambda: ["Oxygen"])
    monkeypatch.setattr(filter_bar_module, "load_structure_filter_preset", lambda name: saved)
    monkeypatch.setattr(filter_bar_module.MessageManager, "send_success_message", lambda message: None)
    bar.set_spec(_spec(_condition("tag", FilterField.CONFIG_TYPE, "surface")))
    bar._popup.add_condition()  # Blank drafts are dropped when appending.
    bar._popup.refresh_presets()

    bar._popup._append_preset_menu.actions()[0].trigger()
    qapp.processEvents()

    assert [condition.condition_id for condition in bar.spec.conditions] == ["tag", "saved"]
    assert bar.spec.logic == filter_bar_module.FilterLogic.ALL
    bar._popup._debounce.stop()


def test_expression_field_picker_inserts_dataset_fields_at_the_cursor(bar, monkeypatch):
    bar.set_suggestions({SearchType.EXPRESSION: {"natoms": 5, "force.norm": 9}})
    bar.set_spec(_spec(_condition("expr", FilterField.EXPRESSION, " > 3")))
    row = bar._popup._rows[0]
    assert not row.field_picker.isHidden()
    assert row.mode_combo.isHidden()
    assert row._expression_fields() == ["force.norm", "natoms"]

    menus = []
    monkeypatch.setattr(filter_bar_module.RoundMenu, "exec", lambda self, *args, **kwargs: menus.append(self))
    row.field_picker.click()
    texts = [action.text() for action in menus[0].actions()]
    assert texts == ["force.norm  (eV/Å)", "natoms  (count)"]

    row.value_edit.setCursorPosition(0)
    menus[0].actions()[1].trigger()
    assert row.to_condition().text_values == ("natoms > 3",)

    _choose(row.field_combo, FilterField.CONFIG_TYPE)
    assert row.field_picker.isHidden()
    assert not row.mode_combo.isHidden()


def test_stale_or_missing_preview_offers_preview_then_apply(bar, monkeypatch):
    menus = []
    monkeypatch.setattr(filter_bar_module.RoundMenu, "exec", lambda self, *args, **kwargs: menus.append(self))
    applied = []
    bar.applyRequested.connect(applied.append)
    bar.set_spec(_spec(_condition("tag", FilterField.CONFIG_TYPE, "surface")))

    bar.apply_button.click()
    actions = menus[-1].actions()
    assert actions[0].text() == bar.tr("Preview, then replace current selection")
    assert actions[0].isEnabled()
    actions[1].trigger()
    assert applied == ["add"]

    bar.set_result(4, 10, 1.0)
    bar.apply_button.click()
    assert "(4)" in menus[-1].actions()[0].text()
