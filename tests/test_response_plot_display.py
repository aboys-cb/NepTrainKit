"""Visible data determine plot bounds and categorical labels stay readable."""

import pytest

from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QApplication

from NepTrainKit.ui.canvas.curve_factory import category_ticks
from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace
from test_reference_response import four_spec, four_states, mae_frames
from test_validation_framework import FakePredictor, _import, _label
from test_validation_ui import scored as scored, make_service as make_service


def test_category_ticks_reduce_density_and_elide_long_labels():
    font = QFont()
    font.setPixelSize(15)
    metrics = QFontMetrics(font)
    points = [(i, str(i)) for i in range(1, 29)]
    ticks = category_ticks(points, .5, 28.5, 320, metrics)
    assert 1 <= len(ticks) <= 5
    assert all(text == str(index) for index, text in ticks)
    long = [(i, "geometry=very_long_structure_name_" + str(i)) for i in range(1, 9)]
    ticks = category_ticks(long, 2.5, 7.5, 240, metrics)
    assert all(2.5 <= x <= 7.5 and metrics.horizontalAdvance(text) <= 228 / len(ticks)
               for x, text in ticks)


def test_hiding_outlying_model_refits_mae_without_prediction(scored, make_service, tmp_path):
    make_service(**{scored.old["content_hash"]: (100., 0.), scored.new["content_hash"]: (1., 0.)})
    frames = mae_frames()
    for a in frames:
        a.set_chemical_symbols(["Pb", "Te"])
        _label(a, a.info.pop("energy"))
    from NepTrainKit.core.validation.protocols.response import suggest_response_settings

    item = _import(scored.service, tmp_path, frames, "angles", protocol="response",
                   analysis=suggest_response_settings(frames))
    suite = scored.service.create_suite("bounds", [item["id"]])
    for model in (scored.old, scored.new):
        scored.service.run_suite(model["id"], suite["id"])
    scored.service.set_baseline(suite["id"], scored.old["id"])
    w = ValidationWorkspace(service=scored.service)
    w.resize(1100, 760)
    w.open_scorecard(suite["id"], scored.new["id"])
    w.show()
    w.open_item(item["id"])
    QApplication.processEvents()
    plot, = w.detail.eos_plots
    if hasattr(plot, "view"):
        old_span = plot.view.camera.rect.height
    else:
        low, high = plot.getViewBox().viewRange()[1]
        old_span = high - low
    calls = FakePredictor.calls
    w.detail.set_series_visible(scored.old["id"], False)
    QApplication.processEvents()
    if hasattr(plot, "view"):
        low, high = plot.view.camera.rect.bottom, plot.view.camera.rect.top
    else:
        low, high = plot.getViewBox().viewRange()[1]
    assert high - low < old_span / 10
    assert all(low <= y <= high for c in plot.listDataItems() if c.isVisible() for y in c.yData)
    assert FakePredictor.calls == calls
    w.close()
    w.deleteLater()


def test_four_state_plot_uses_indices_and_keeps_full_hover_labels(scored, tmp_path, monkeypatch):
    frames = []
    for i in range(28):
        for a in four_states("long_structure_name_" + str(i)):
            a.set_chemical_symbols(["Pb", "Te"])
            _label(a, a.info.pop("energy"))
            frames.append(a)
    item = _import(scored.service, tmp_path, frames, "four-state", protocol="response",
                   analysis=four_spec().to_dict())
    suite = scored.service.create_suite("categories", [item["id"]])
    w = ValidationWorkspace(service=scored.service)
    w.resize(1100, 760)
    w.open_scorecard(suite["id"], scored.new["id"])
    w.show()
    w.open_item(item["id"])
    QApplication.processEvents()
    plot, = w.detail.eos_plots
    assert plot.axis_titles[0] == "Combination index"
    assert len(plot.category_labels) == 28
    curve, = plot.listDataItems()
    assert list(curve.xData) == list(range(1, 29))
    plot.cursor_coordinates = lambda pos: (7., curve.yData[6], .01, .01)
    plot.point_hover._follow(QPoint(100, 100))
    assert "long_structure_name_6" in plot.point_hover.text()
    assert plot.point_hover.width() <= 420
    from PySide6.QtWidgets import QDialog
    from qfluentwidgets import TableWidget
    from NepTrainKit.ui.widgets.validation_values import show_curve_values

    def check(dialog):
        table = dialog.findChild(TableWidget)
        assert table.horizontalHeaderItem(0).text() == "Combination index"
        assert table.horizontalHeaderItem(1).text() == "State / combination"
        assert "long_structure_name_6" in table.item(6, 1).text()
        return 0

    monkeypatch.setattr(QDialog, "exec", check)
    show_curve_values(w.detail, plot, {item["id"]: item["name"]})
    w.close()
    w.deleteLater()


def test_four_state_facets_use_one_structure_order_and_keep_conditions_separate():
    import random
    from NepTrainKit.core.validation.protocols.response import ReferenceResponseProtocol, ResponseSpec

    frames = []
    for geometry in ("variant10", "variant2"):
        for theta in (5., 2.5):
            for left, right in ((1, 1), (1, 2), (2, 1), (2, 2)):
                for frame in four_states(geometry):
                    frame.info.update(axis_i=left, axis_j=right, site_i=0, site_j=1, theta_deg=theta)
                    frames.append(frame)
    random.Random(7).shuffle(frames)
    spec = ResponseSpec(mode="four_state", group_fields=("geometry", "axis_i", "axis_j", "theta_deg"),
                        state_fields=("sign_i", "sign_j"), normalization="total")
    summary = ReferenceResponseProtocol(spec.to_dict()).summarize_reference(frames)
    branches = summary["plot_series"]
    assert len(branches) == 8
    for branch in branches:
        assert branch["labels"] == ["variant2", "variant10"]
        assert branch["x"] == [1, 2] and branch["connected"] is False
        assert [summary["reference"][i] for i in branch["indices"]] == pytest.approx([4., 4.])
        assert branch["condition"] in (["0", "1", "5.0"], ["0", "1", "2.5"])
    assert len({tuple(b["condition"]) for b in branches}) == 2
