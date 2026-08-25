from __future__ import annotations

from PySide6.QtWidgets import QApplication

from NepTrainKit.ui.widgets import (
    AdaptiveCompactDoubleSpinBox,
    AdaptiveCompactSpinBox,
    SpinBoxUnitInputFrame,
    adapt_legacy_inspector_form,
)
from NepTrainKit.ui.views._card.spin_spiral_card import SpinSpiralCard
from NepTrainKit.ui.widgets.compact_form import ResponsiveFormGrid


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_adaptive_float_input_prioritizes_text_when_narrow():
    app = _app()
    spin = AdaptiveCompactDoubleSpinBox()
    spin.setDecimals(6)
    spin.setValue(45.0)
    spin.resize(spin.readable_width_hint(), 30)
    spin.show()
    app.processEvents()

    text_width = spin.fontMetrics().horizontalAdvance(spin.text())
    assert spin.compactSpinButton.isHidden()
    assert spin.lineEdit().width() >= text_width + 4

    spin.resize(180, 30)
    app.processEvents()
    assert not spin.compactSpinButton.isHidden()
    spin.close()


def test_adaptive_integer_input_prioritizes_text_when_narrow():
    app = _app()
    spin = AdaptiveCompactSpinBox()
    spin.setRange(1, 500000)
    spin.setValue(512)
    spin.resize(spin.readable_width_hint(), 30)
    spin.show()
    app.processEvents()

    text_width = spin.fontMetrics().horizontalAdvance(spin.text())
    assert spin.compactSpinButton.isHidden()
    assert spin.lineEdit().width() >= text_width + 4

    spin.resize(120, 30)
    app.processEvents()
    assert not spin.compactSpinButton.isHidden()
    spin.close()


def test_multi_integer_frame_keeps_each_value_readable_in_narrow_row():
    app = _app()
    frame = SpinBoxUnitInputFrame()
    frame.set_input("", 3, "int")
    frame.setRange(1, 999)
    frame.set_input_value([4, 4, 4])
    frame.resize(110, 160)
    frame.show()
    for _ in range(3):
        app.processEvents()

    assert 1 <= frame._column_count <= 3
    for spin in frame.object_list:
        text_width = spin.fontMetrics().horizontalAdvance(spin.text())
        assert spin.compactSpinButton.isHidden()
        assert spin.lineEdit().width() >= text_width + 4
    frame.close()


def test_multi_float_frame_reflows_before_values_become_unreadable():
    _app()
    frame = SpinBoxUnitInputFrame()
    frame.set_input(["-", "step", "deg"], 3, "float")
    frame.setDecimals(6)
    frame.set_input_value([15.0, 45.0, 15.0])

    readable_widths = [
        spin.readable_width_hint() + label.sizeHint().width() + 4
        for spin, label in zip(frame.object_list, frame._unit_labels)
    ]
    spacing = frame._layout.horizontalSpacing()
    three_column_width = sum(readable_widths) + spacing * 2

    frame._reflow_inputs(three_column_width - 1)
    assert frame._column_count < 3

    frame._reflow_inputs(three_column_width)
    assert frame._column_count == 3
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
