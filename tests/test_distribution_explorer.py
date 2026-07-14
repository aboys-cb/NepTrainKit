from PySide6.QtWidgets import QApplication

from NepTrainKit.core.io.base import DistributionRequest, FieldSpec
from NepTrainKit.core.types import FieldDomain, FieldValueShape
from NepTrainKit.ui.pages.training_set_audit import TrainingSetAuditWidget
from NepTrainKit.ui.widgets.dialog import DistributionExplorerWidget


class _DistributionData:
    def __init__(self):
        self.resolve_calls = []

    def discover_atomic_numeric_fields(self, scope="active"):
        return [
            FieldSpec(
                key="atomic:spin_vec",
                source="atomic",
                shape=FieldValueShape.VECTOR3,
                components=("x", "y", "z"),
                has_prediction_pair=True,
                unit_guess="mu_B",
                domain=FieldDomain.ATOM,
                label="spin_vec",
            )
        ]

    def resolve_distribution_bin_indices(self, analysis_id, metric_key, series_key, bin_index):
        self.resolve_calls.append((analysis_id, metric_key, series_key, bin_index))
        return [3, 7]


def _analysis_payload():
    return {
        "analysis_id": 12,
        "metrics": [
            {
                "metric_key": "atomic:spin_vec|norm",
                "field_key": "atomic:spin_vec",
                "field_label": "spin_vec",
                "component": "norm",
                "value_view": "reference",
                "series": [
                    {
                        "series_key": "Fe",
                        "name": "Fe",
                        "hist": [2, 1],
                        "bin_edges": [0.0, 1.0, 2.0],
                        "total": 3,
                    }
                ],
            }
        ],
        "messages": [],
    }


def test_distribution_explorer_keeps_request_and_bin_selection_contract():
    app = QApplication.instance() or QApplication([])
    data = _DistributionData()
    requests = []
    selections = []
    explorer = DistributionExplorerWidget(
        data=data,
        run_analysis_callback=lambda request: requests.append(request) or _analysis_payload(),
        apply_selection_callback=lambda indices, mode: selections.append((indices, mode)),
        canvas_type="pyqtgraph",
    )

    explorer.analyzeButton.click()

    assert len(requests) == 1
    assert isinstance(requests[0], DistributionRequest)
    assert requests[0].field_keys == ("atomic:spin_vec",)
    assert explorer.selectModeCombo.isHidden()
    assert explorer.binsSpin.isHidden()
    assert explorer.metricCombo.count() == 1
    assert explorer.seriesCombo.currentData() == "Fe"

    explorer._select_bin(0)

    assert data.resolve_calls == [(12, "atomic:spin_vec|norm", "Fe", 0)]
    assert selections == [([3, 7], "replace")]

    explorer.selectModeCombo.setCurrentIndex(1)
    explorer._select_bin(1)
    explorer.selectModeCombo.setCurrentIndex(2)
    explorer._select_bin(1)

    assert selections[-2:] == [([3, 7], "add"), ([3, 7], "intersect")]
    explorer.deleteLater()
    app.processEvents()


def test_audit_page_embeds_the_same_distribution_explorer():
    app = QApplication.instance() or QApplication([])
    data = _DistributionData()
    widget = TrainingSetAuditWidget()

    widget.set_distribution_context(
        data=data,
        run_analysis_callback=lambda request: _analysis_payload(),
        apply_selection_callback=lambda indices, mode: None,
    )
    widget.show_distribution_explorer()

    assert widget.page_tabs.currentIndex() == 1
    assert widget.data_map_tabs.currentWidget() is widget.distribution_tab
    assert widget.distribution_explorer.fieldCombo.count() == 1
    assert widget.distribution_explorer._data is data
    widget.deleteLater()
    app.processEvents()
