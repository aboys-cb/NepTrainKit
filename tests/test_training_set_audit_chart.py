from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from NepTrainKit.ui.widgets.audit_chart import AuditChartWidget


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def histogram_payload():
    return {
        "kind": "histogram",
        "id": "composition:Fe",
        "title": "Fe concentration distribution",
        "x_label": "Atomic fraction",
        "y_label": "Structures",
        "series": (
            {
                "id": "Fe",
                "label": "Fe",
                "bin_edges": (0.0, 0.5, 1.0),
                "counts": (3, 1),
                "highlighted_bins": (1,),
                "structure_indices": ((10, 20, 30), (40,)),
            },
        ),
    }


@pytest.fixture
def categorical_payload():
    return {
        "kind": "categorical_bars",
        "id": "pair_contacts:radial",
        "title": "Radial pair contact coverage",
        "x_label": "Structures",
        "y_label": "Element pair",
        "series": (
            {
                "id": "radial_pair_contact",
                "labels": ("Fe-Ni", "Fe-Ta"),
                "counts": (4, 2),
                "structure_indices": ((1, 4, 9, 16), (25, 36)),
            },
        ),
    }


@pytest.fixture
def composition_payload():
    return {
        "kind": "composition_stems",
        "id": "inventory:composition:Ni",
        "title": "Exact Ni composition support",
        "x_label": "Ni atomic fraction",
        "y_label": "Structures",
        "x_min": -0.01,
        "x_max": 1.0,
        "log_scale": True,
        "target_points": (0.0, 0.25, 0.5),
        "series": (
            {
                "x_values": (0.0, 0.5, 1.0),
                "labels": ("Fe 100%", "Fe 50% · Ni 50%", "Ni 100%"),
                "counts": (596, 28746, 943),
                "structure_indices": ((1,), (2, 3), (4,)),
            },
        ),
    }


def test_histogram_state_can_be_set_and_cleared(app, histogram_payload):
    widget = AuditChartWidget()

    widget.set_plot(histogram_payload)

    assert widget.plot_id == "composition:Fe"
    assert widget.has_data is True
    assert widget.minimumHeight() == 220
    assert widget.sizeHint().width() == 640
    assert widget.sizeHint().height() == 260

    widget.clear()

    assert widget.plot_id == ""
    assert widget.has_data is False


def test_categorical_payload_is_accepted(app, categorical_payload):
    widget = AuditChartWidget()

    widget.set_plot(categorical_payload)

    assert widget.plot_id == "pair_contacts:radial"
    assert widget.has_data is True


def test_composition_stems_accept_exact_points_and_emit_indices(app, composition_payload):
    widget = AuditChartWidget()
    received = []
    widget.selectedGroupSignal.connect(received.append)
    widget.set_plot(composition_payload)
    widget.resize(640, 260)
    widget.show()
    app.processEvents()

    QTest.mouseClick(
        widget,
        Qt.MouseButton.LeftButton,
        pos=widget._bar_rects[1][0].center().toPoint(),
    )

    assert widget.plot_id == "inventory:composition:Ni"
    assert widget._bar_rects[0][0].center().x() > 58
    assert received == [[2, 3]]


def test_histogram_accepts_negative_finite_strictly_increasing_bin_edges(app, histogram_payload):
    histogram_payload["series"][0]["bin_edges"] = (-3.0, -0.5, 1.0)
    widget = AuditChartWidget()

    widget.set_plot(histogram_payload)

    assert widget.plot_id == "composition:Fe"
    assert widget.has_data is True


@pytest.mark.parametrize(
    "structure_indices",
    (
        42,
        ((10,),),
        ((10.5,), (40,)),
    ),
)
def test_malformed_supplied_structure_indices_use_empty_fallback(app, histogram_payload, structure_indices):
    histogram_payload["series"][0]["structure_indices"] = structure_indices
    widget = AuditChartWidget()

    widget.set_plot(histogram_payload)

    assert widget.plot_id == ""
    assert widget.has_data is False


def test_absent_structure_indices_leave_histogram_bars_unmapped(app, histogram_payload):
    histogram_payload["series"][0]["structure_indices"] = None
    widget = AuditChartWidget()
    received = []
    widget.selectedGroupSignal.connect(received.append)
    widget.set_plot(histogram_payload)
    widget.resize(640, 260)
    widget.show()
    app.processEvents()

    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=widget._bar_rects[0][0].center().toPoint())

    assert widget.has_data is True
    assert received == []


@pytest.mark.parametrize(
    "payload",
    (
        None,
        {},
        {"kind": "histogram", "id": "broken", "series": ()},
        {
            "kind": "categorical_bars",
            "id": "broken",
            "series": ({"labels": ("bulk",), "counts": ()},),
        },
    ),
)
def test_malformed_payload_uses_empty_fallback(app, payload):
    widget = AuditChartWidget()

    widget.set_plot(payload)

    assert widget.plot_id == ""
    assert widget.has_data is False


@pytest.mark.parametrize(
    "payload", ("histogram_payload", "categorical_payload", "composition_payload")
)
def test_plot_renders_multiple_colors(app, request, payload):
    widget = AuditChartWidget()
    widget.set_plot(request.getfixturevalue(payload))
    widget.resize(640, 260)
    widget.show()
    app.processEvents()

    image = widget.grab().toImage()
    sampled_colors = {
        image.pixelColor(x, y).rgba()
        for x in range(0, image.width(), 16)
        for y in range(0, image.height(), 16)
    }

    assert not image.isNull()
    assert len(sampled_colors) > 1


def test_clicking_histogram_bar_emits_original_structure_indices(app, histogram_payload):
    widget = AuditChartWidget()
    received = []
    widget.selectedGroupSignal.connect(received.append)
    widget.set_plot(histogram_payload)
    widget.resize(640, 260)
    widget.show()
    app.processEvents()

    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=widget._bar_rects[1][0].center().toPoint())

    assert received == [[40]]
