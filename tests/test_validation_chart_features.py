"""Plot annotations and alternative views preserve saved values and model state."""
from copy import deepcopy

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QDialog
from qfluentwidgets import CaptionLabel, TableWidget

from NepTrainKit.core.validation.protocols.response import ReferenceResponseProtocol, ResponseSpec
from NepTrainKit.ui.views.model_library import RunDetailPanel
from test_accuracy_comparison_plot import curve_mode as curve_mode
from test_reference_response import exchange_frames
from test_validation_framework import FakePredictor
from test_validation_ui import scored as scored, make_service as make_service


def test_eos_fit_annotations_follow_models_and_stay_out_of_values(scored, curve_mode):
    benchmark = scored.eos
    entries = [(m['id'], m['name'], run, scored.service.run_details(run)['details'])
               for m in (scored.old, scored.new)
               for run in [scored.service.latest_run(m['id'], benchmark['id'])]]
    panel = RunDetailPanel()
    panel.set_adaptive_layout(True)
    panel.resize(900, 650)
    panel.show_runs(benchmark, entries)
    panel.show(); QApplication.processEvents()
    plot, = panel.eos_plots
    assert len(plot.listDataItems()) == 3
    annotations = [c for c in panel._curves[scored.new['id']] if c.property('auxiliary')]
    assert len(annotations) == 3
    assert {c.property('annotation') for c in annotations} == {'eos_fit', 'equilibrium', 'equilibrium_volume'}
    fit = entries[1][3]['model_fit']
    marker = next(c for c in annotations if c.property('annotation') == 'equilibrium')
    assert marker.xData == pytest.approx([fit['v0']])
    pen = annotations[0].opts['pen']
    assert pen.style() == Qt.PenStyle.DotLine
    calls = FakePredictor.calls
    panel.set_series_color(scored.new['id'], QColor('#d73377'))
    assert annotations[0].opts['pen'].color().name() == '#d73377'
    assert annotations[0].opts['pen'].style() == Qt.PenStyle.DotLine
    panel.set_series_visible(scored.new['id'], False)
    assert all(not c.isVisible() for c in annotations)
    panel.set_series_visible(scored.new['id'], True)
    assert all(c.isVisible() for c in annotations)
    panel._change_eos_view(1)
    assert not any(c.property('auxiliary') for curves in panel._curves.values() for c in curves)
    assert FakePredictor.calls == calls
    panel.close(); panel.deleteLater()


def test_discrete_offsets_and_exchange_views_keep_category_identity(curve_mode, monkeypatch):
    spec = ResponseSpec(mode='four_state', group_fields=('geometry', 'axis_i', 'axis_j', 'site_i', 'site_j'),
                        state_fields=('sign_i', 'sign_j'), normalization='total', extract_exchange=True)
    summary = ReferenceResponseProtocol(spec.to_dict()).summarize_reference(exchange_frames())
    model = deepcopy(summary)
    model.update(predicted=[v*1.1 for v in summary['reference']], errors=[v*.1 for v in summary['reference']])
    for item in model['exchange']['projections']:
        item['predicted'] = item['reference'] * 1.1
    for item in model['exchange']['symmetric_chiral']:
        item['predicted_symmetric'] = item['reference_symmetric'] * 1.1
        item['predicted_d_projection'] = item['reference_d_projection'] * 1.1
    old = {k: v for k, v in model.items() if k != 'exchange'}
    benchmark = dict(id='pair', name='pair', protocol='response', provenance={}, summary=summary,
                     source_ref={'analysis': spec.to_dict()})
    panel = RunDetailPanel()
    panel.set_adaptive_layout(True)
    panel.model_colors.update({1: '#2266aa', 2: '#ff6600'})
    panel.resize(1000, 700)
    panel.show_runs(benchmark, [(1, 'new', {'status': 'succeeded'}, model), (2, 'old', {'status': 'succeeded'}, old)])
    panel.show(); QApplication.processEvents()
    plot = panel.eos_plots[0]
    curves = plot.listDataItems()
    assert len(curves) == 3
    assert curves[0].xData[0] < curves[1].xData[0] < curves[2].xData[0]
    assert all(list(c.sample_x) == [1] for c in curves)
    plot.cursor_coordinates = lambda _: (curves[1].xData[0], curves[1].yData[0], .001, .001)
    plot.point_hover._follow(QPoint(90, 90))
    assert plot.point_hover.text().count('meV') == 3
    assert '#2266aa' in plot.point_hover.text() and '#ff6600' in plot.point_hover.text()
    panel.exchange_picker.items['d_projection'].click()
    plot, = panel.eos_plots
    assert 'Chiral D projection' in plot.axis_titles[1]
    ref, predicted = plot.listDataItems()
    np.testing.assert_allclose(ref.yData, [.3])
    np.testing.assert_allclose(predicted.yData, [.33])
    low, high = ((plot.view.camera.rect.bottom, plot.view.camera.rect.top) if hasattr(plot, 'view') else
                 plot.getViewBox().viewRange()[1])
    assert low <= 0 <= high
    assert any('Projection unavailable: old' == label.text() for label in panel.eos_cards[0].findChildren(CaptionLabel))
    assert 'exchange' not in old and model['predicted'] != [.33]
    from NepTrainKit.ui.widgets.reference_response_plot import _show_values

    def check(dialog):
        tables = dialog.findChildren(TableWidget)
        frames = next(t for t in tables if t.horizontalHeaderItem(0).text() == 'Group / state')
        components = next(t for t in tables if t.horizontalHeaderItem(0).text() == 'Ordered pair / group')
        assert frames.rowCount() == 4
        assert components.rowCount() == 1
        assert all(t.item(0, t.columnCount()-1).text() == '—' for t in (frames, components))
        return 0

    monkeypatch.setattr(QDialog, 'exec', check)
    _show_values(panel, summary, [0, 1, 2, 3], panel._args[1])
    panel.close(); panel.deleteLater()


def test_native_canvas_selection_during_build_finishes_with_latest_group(scored, monkeypatch):
    from NepTrainKit.ui.widgets import eos_plot

    panel = RunDetailPanel()
    panel.set_adaptive_layout(True)
    create = eos_plot.create_curve_plot
    calls = []

    def selecting_canvas(parent):
        calls.append(parent)
        if len(calls) == 1:
            panel.show_runs(scored.eos, [])
        return create(parent)

    monkeypatch.setattr(eos_plot, 'create_curve_plot', selecting_canvas)
    panel.show()
    panel.show_groups([('G1', scored.eos, []), ('G2', dict(scored.eos, id='other'), [])])
    QApplication.processEvents()
    QApplication.processEvents()
    assert len(panel.eos_cards) == 1 and panel.eos_cards[0].isVisible()
    assert len(panel.eos_plots[0].listDataItems()) == 1
    assert panel.eos_plots[0].listDataItems()[0].property('groupId') == scored.eos['id']
    assert not panel._redrawing and not panel._redraw_pending
    panel.close(); panel.deleteLater()
