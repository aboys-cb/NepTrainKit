"""Regression comparisons preserve frame identity, masks and baseline ordering."""
import pytest
from ase.io import write
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtGui import QColor
from qfluentwidgets import PushButton, TableWidget

from NepTrainKit.core.validation.accuracy import (accuracy_labels, compare_frame_errors,
                                                 frame_error_statistics, frame_group_labels, prediction_values)
from NepTrainKit.core.validation.protocols.response import ResponseSpec
from NepTrainKit.config import Config
from NepTrainKit.ui.views.validation_workspace import ValidationWorkspace
from test_validation_accuracy import accuracy_case as accuracy_case
from test_validation_framework import FakePredictor, _import, efv_frames
from test_validation_ui import scored as scored, make_service as make_service


@pytest.fixture(params=("pyqtgraph", "vispy"))
def curve_mode(request):
    previous = Config.get("widget", "canvas_type", "auto")
    Config.set("widget", "canvas_type", request.param)
    yield request.param
    Config.set("widget", "canvas_type", previous)


def test_frame_errors_respect_missing_labels_and_original_indices(accuracy_case):
    case = accuracy_case
    labels = accuracy_labels(case.frames)
    errors = frame_error_statistics(labels, prediction_values(case.prediction, labels))
    for quantity, expected, mask in (("energy", 1, [0, 2]), ("force", 2, [0]),
                                     ("virial", 3, [0]), ("mforce", 4, [0, 1])):
        assert [i for i, v in enumerate(errors[quantity]) if v is not None] == mask
        assert [errors[quantity][i] for i in mask] == pytest.approx([expected] * len(mask))
    case.frames[0].info["geometry"] = "  polar  "
    case.frames[2].info["geometry"] = "polar"
    assert frame_group_labels(case.frames) == {"geometry": ["polar", None, "polar"]}
    assert frame_error_statistics(labels, {"energy": labels.values["energy"]})["energy"] == [0, None, 0]


def test_comparisons_share_baseline_order_and_keep_missing_frame_gaps():
    baseline = [2., 10., None, 4., 10.]
    rows = compare_frame_errors([1., None, 8., 7., 9.], baseline)
    assert [(r["rank"], r["frame"]) for r in rows] == [(2, 4), (3, 3), (4, 0)]
    assert [r["delta"] for r in rows] == [-1., 3., -1.]
    assert [r["frame"] for r in compare_frame_errors([1., 5., None, 3., 7.], baseline)] == [1, 4, 3, 0]
    assert [r["frame"] for r in compare_frame_errors([1., 5., None], selected=[2, 0])] == [0]


def test_legacy_results_restore_verified_cache_without_prediction(accuracy_case, tmp_path):
    case = accuracy_case
    for a, group in zip(case.frames, ("C", "A", "B")):
        a.info["geometry"] = group
    source = tmp_path / "regression.xyz"
    write(source, case.frames, format="extxyz")
    item = case.service.import_set(source, name="regression", protocol="response", provenance={},
                                   analysis=ResponseSpec(mode="slices", group_fields=("geometry",)).to_dict())
    run = case.service.evaluate(case.model["id"], item["id"])
    assert run["status"] == "succeeded", run
    details = case.service.run_details(run)["details"]
    legacy = {k: v for k, v in details.items() if k not in ("frame_errors", "frame_groups")}
    restored = case.service.accuracy_view_data(item, run, legacy)
    assert restored["frame_errors"]["mforce"][:2] == pytest.approx([4, 4])
    assert restored["frame_groups"]["geometry"] == ["C", "A", "B"]
    assert case.service.accuracy_view_data(item, dict(run, environment={}), legacy) == legacy
    assert len(case.calls) == 1
    cache = case.service._prediction_cache(run["environment"])
    cache.path.unlink()
    case.service._accuracy_view_cache.clear()
    fallback = case.service.accuracy_view_data(item, run, legacy)
    assert fallback["frame_errors"]["energy"][1] is None
    assert [fallback["frame_errors"]["energy"][i] for i in (0, 2)] == pytest.approx([1, 1])
    assert "mforce" not in fallback["frame_errors"] and len(case.calls) == 1


def test_regression_cards_use_frame_error_changes_and_filter_without_prediction(scored, tmp_path, monkeypatch, curve_mode):
    frames = efv_frames()
    for i, frame in enumerate(frames):
        if i != 1:
            frame.info["geometry"] = "polar" if i < 2 else "reference"
    item = _import(scored.service, tmp_path, frames, "regression", protocol="response",
                   analysis=ResponseSpec(mode="slices").to_dict())
    suite = scored.service.create_suite("regression", [item["id"]])
    for model in (scored.old, scored.new, scored.fresh):
        scored.service.run_suite(model["id"], suite["id"])
    scored.service.set_baseline(suite["id"], scored.old["id"])
    w = ValidationWorkspace(service=scored.service)
    w.resize(1100, 760)
    w.open_scorecard(suite["id"], scored.new["id"])
    w.show(); w.open_item(item["id"]); QApplication.processEvents()
    panel = w.detail
    plot, = panel.eos_plots
    assert panel.accuracy_quantity == "force" and len(plot.comparison_points) == 4
    assert all(p["delta"] < 0 for p in plot.comparison_points)
    curve, = plot.listDataItems()
    assert curve.opts["pen"] is None and curve.property("modelId") == scored.new["id"]
    assert "baseline error descending" in plot.axis_titles[0]
    assert panel.legend_panel.isHidden() and not panel.accuracy_tools.isHidden()
    calls = FakePredictor.calls
    combo = panel.accuracy_filter
    combo.setCurrentIndex(next(i for i in range(combo.count()) if combo.itemData(i) == ("geometry", None)))
    plot, = panel.eos_plots
    assert [p["frame"] for p in plot.comparison_points] == [1]
    from NepTrainKit.ui.widgets.accuracy_comparison_plot import _show_frames
    statistics_opened = []

    def check(dialog):
        table = dialog.findChild(TableWidget)
        assert table.rowCount() == 1 and table.item(0, 0).data(0) == 1
        assert table.item(0, 5).text() == "—"
        dialog.findChild(PushButton).click()
        assert statistics_opened == [True]
        return 0

    monkeypatch.setattr(QDialog, "exec", check)
    _show_frames(panel, "test", plot.comparison_points, {"geometry": ["polar", None, "reference", "reference"]}, "meV/Å",
                 lambda: statistics_opened.append(True))
    panel.accuracy_picker.items["energy"].click()
    assert panel.accuracy_quantity == "energy"
    w._choose_model_role(scored.fresh["id"], "comparison")
    QApplication.processEvents()
    assert len(panel.eos_plots) == 2
    assert panel.eos_plots[0].comparison_y_range == panel.eos_plots[1].comparison_y_range
    assert all(len(p.comparison_points) == 1 for p in panel.eos_plots)
    panel.set_series_color(scored.new["id"], QColor("#123456"))
    assert panel._pg.mkPen(panel._curves[scored.new["id"]][0].opts["symbolPen"]).color().name() == "#123456"
    panel.set_series_visible(scored.fresh["id"], False)
    assert len(panel.eos_plots) == 1
    assert FakePredictor.calls == calls
    w.close(); w.deleteLater()
