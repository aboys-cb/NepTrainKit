from __future__ import annotations

from PySide6.QtWidgets import QApplication

from NepTrainKit.ui.views._card.spin_spiral_card import SpinSpiralCard
from NepTrainKit.ui.widgets import (
    AdaptiveCompactDoubleSpinBox,
    AdaptiveCompactSpinBox,
    ResponsiveFormGrid,
    SpinBoxUnitInputFrame,
    adapt_legacy_inspector_form,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_adaptive_float_input_hint_keeps_step_control_visible():
    app = _app()
    spin = AdaptiveCompactDoubleSpinBox()
    spin.setDecimals(6)
    spin.setValue(45.0)
    spin.resize(spin.readable_width_hint(), 30)
    spin.show()
    app.processEvents()

    assert not spin.compactSpinButton.isHidden()
    spin.close()


def test_adaptive_integer_input_hint_keeps_step_control_visible():
    app = _app()
    spin = AdaptiveCompactSpinBox()
    spin.setRange(1, 500000)
    spin.setValue(512)
    spin.resize(spin.readable_width_hint(), 30)
    spin.show()
    app.processEvents()

    assert not spin.compactSpinButton.isHidden()
    spin.close()


def test_multi_integer_frame_keeps_one_equal_width_row_when_narrow():
    app = _app()
    frame = SpinBoxUnitInputFrame()
    frame.set_input("", 3, "int")
    frame.setRange(1, 999)
    frame.set_input_value([4, 4, 4])
    frame.resize(180, 30)
    frame.show()
    for _ in range(3):
        app.processEvents()

    assert frame._column_count == 3
    widths = [control.width() for control in frame.object_list]
    assert max(widths) - min(widths) <= 1
    frame.close()


def test_multi_float_frame_tracks_parent_width_without_wrapping():
    app = _app()
    frame = SpinBoxUnitInputFrame()
    frame.set_input("", 3, "float")
    frame.setDecimals(6)
    frame.set_input_value([15.0, 45.0, 15.0])
    frame.resize(240, 30)
    frame.show()
    app.processEvents()
    narrow_widths = [control.width() for control in frame.object_list]

    frame.resize(420, 30)
    app.processEvents()
    wide_widths = [control.width() for control in frame.object_list]

    assert frame._column_count == 3
    assert max(narrow_widths) - min(narrow_widths) <= 1
    assert max(wide_widths) - min(wide_widths) <= 1
    assert all(wide > narrow for wide, narrow in zip(wide_widths, narrow_widths))
    frame.close()


def _assert_compact_grid_has_no_holes(form: ResponsiveFormGrid) -> None:
    rows: dict[int, list[tuple[int, int]]] = {}
    for field, _requested_span in form._fields:
        span = int(field.property("responsiveGridSpan"))
        if field.isHidden():
            assert span == 0
            continue
        row = int(field.property("responsiveGridRow"))
        column = int(field.property("responsiveGridColumn"))
        rows.setdefault(row, []).append((column, span))
    assert rows
    assert all(
        sorted(cells) in ([(0, 2)], [(0, 1), (1, 1)])
        for cells in rows.values()
    )


def test_legacy_inspector_reflows_when_conditional_fields_change():
    app = _app()
    card = SpinSpiralCard()
    assert adapt_legacy_inspector_form(card.setting_widget, card.settingLayout)
    card.setting_widget.resize(340, 900)
    card.setting_widget.show()
    for _ in range(4):
        app.processEvents()

    form = card.setting_widget.findChild(ResponsiveFormGrid)
    assert form is not None
    _assert_compact_grid_has_no_holes(form)

    mode_index = card.parameter_mode_combo.findData("Angle gradient (deg/A)")
    phase_index = card.phase_mode_combo.findData("Layer-locked")
    source_index = card.source_combo.findData("Map/default magnitude")
    assert min(mode_index, phase_index, source_index) >= 0
    card.advanced_checkbox.setChecked(True)
    card.parameter_mode_combo.setCurrentIndex(mode_index)
    card.phase_mode_combo.setCurrentIndex(phase_index)
    card.source_combo.setCurrentIndex(source_index)
    for _ in range(4):
        app.processEvents()

    _assert_compact_grid_has_no_holes(form)
    assert not card.angle_gradient_frame.isHidden()
    assert not card.layer_tol_frame.isHidden()
    assert not card.map_edit.isHidden()
    card.close()
