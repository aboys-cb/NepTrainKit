from __future__ import annotations

import csv

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
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


@pytest.fixture
def composition_phase_payload():
    return {
        "kind": "composition_phase_stacks",
        "id": "inventory:composition-phase:Ni",
        "title": "Phase distribution by Ni concentration",
        "x_label": "Ni atomic fraction",
        "y_label": "Structures",
        "x_min": -0.01,
        "x_max": 1.0,
        "x_values": (0.0, 0.5, 1.0),
        "labels": ("Ni 0%", "Ni 50%", "Ni 100%"),
        "series": (
            {
                "id": "fcc",
                "label": "FCC",
                "counts": (1, 2, 1),
                "structure_indices": ((0,), (1, 2), (4,)),
            },
            {
                "id": "bcc",
                "label": "BCC",
                "counts": (0, 1, 0),
                "structure_indices": ((), (3,), ()),
            },
        ),
    }


@pytest.fixture
def category_share_payload():
    return {
        "kind": "category_share_stacks",
        "id": "magnetic_evidence:phase_to_order",
        "title": "Magnetic types inside each structural phase",
        "x_label": "Share of structure frames",
        "y_label": "Structural phase",
        "row_ids": ("fcc", "bcc"),
        "row_labels": ("FCC", "BCC"),
        "series": (
            {
                "id": "fm",
                "label": "FM",
                "counts": (3, 1),
                "structure_indices": ((1, 2, 3), (6,)),
            },
            {
                "id": "afm_double_layered",
                "label": "Double-layer AFM",
                "counts": (2, 0),
                "structure_indices": ((4, 5), ()),
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


def test_plot_change_signal_tracks_export_availability(app, histogram_payload):
    widget = AuditChartWidget()
    states = []
    widget.plotChangedSignal.connect(states.append)

    widget.set_plot(histogram_payload)
    widget.clear()

    assert states == [True, False]


def test_chart_exports_two_x_png_without_focus_marker(app, histogram_payload, tmp_path):
    widget = AuditChartWidget()
    widget.set_plot(histogram_payload)
    widget.resize(640, 260)
    widget.show()
    widget.setFocus()
    app.processEvents()
    target = tmp_path / "histogram.png"

    widget.save_png(target, scale=2.0)

    image = QImage(str(target))
    assert not image.isNull()
    assert image.width() == 1280
    assert image.height() == 520


@pytest.mark.parametrize(
    ("payload_fixture", "expected_field", "expected_rows"),
    (
        ("histogram_payload", "bin_left", 2),
        ("categorical_payload", "category", 2),
        ("composition_payload", "x_value", 3),
        ("composition_phase_payload", "series_id", 6),
        ("category_share_payload", "row_id", 4),
    ),
)
def test_chart_exports_tidy_csv_for_every_plot_kind(
    app,
    request,
    tmp_path,
    payload_fixture,
    expected_field,
    expected_rows,
):
    widget = AuditChartWidget()
    widget.set_plot(request.getfixturevalue(payload_fixture))
    target = tmp_path / f"{payload_fixture}.csv"

    widget.write_csv(target)

    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == expected_rows
    assert expected_field in rows[0]
    assert rows[0]["plot_id"] == widget.plot_id
    assert "structure_indices" not in rows[0]


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


def test_composition_phase_stacks_emit_the_clicked_phase_group(
    app, composition_phase_payload
):
    widget = AuditChartWidget()
    received = []
    widget.selectedGroupSignal.connect(received.append)
    widget.set_plot(composition_phase_payload)
    widget.resize(760, 300)
    widget.show()
    app.processEvents()

    bcc_segment = next(
        rect
        for rect, indices in widget._bar_rects
        if indices == [3]
    )
    QTest.mouseClick(
        widget,
        Qt.MouseButton.LeftButton,
        pos=bcc_segment.center().toPoint(),
    )

    assert widget._plot["counts"] == (1.0, 3.0, 1.0)
    assert received == [[3]]


def test_category_share_stacks_show_frame_shares_and_support_keyboard_selection(
    app, category_share_payload
):
    widget = AuditChartWidget()
    received = []
    widget.selectedGroupSignal.connect(received.append)
    widget.set_plot(category_share_payload)
    widget.resize(760, 300)
    widget.show()
    widget.setFocus()
    app.processEvents()

    assert widget._plot["counts"] == (5.0, 1.0)
    target = next(rect for rect, indices in widget._bar_rects if indices == [4, 5])
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=target.center().toPoint())
    assert received[-1] == [4, 5]

    QTest.keyClick(widget, Qt.Key.Key_Return)
    assert received[-1] == [4, 5]


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
    "payload",
    (
        "histogram_payload",
        "categorical_payload",
        "composition_payload",
        "composition_phase_payload",
    ),
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


def test_tiny_categorical_bar_keeps_clickable_hit_area(app, categorical_payload):
    categorical_payload["series"][0]["counts"] = (10_000, 1)
    categorical_payload["series"][0]["structure_indices"] = ((10,), (99,))
    widget = AuditChartWidget()
    received = []
    widget.selectedGroupSignal.connect(received.append)
    widget.set_plot(categorical_payload)
    widget.resize(640, 260)
    widget.show()
    app.processEvents()

    rare_hit_rect = widget._bar_rects[1][0]
    QTest.mouseClick(
        widget,
        Qt.MouseButton.LeftButton,
        pos=rare_hit_rect.center().toPoint(),
    )

    assert rare_hit_rect.width() >= 20
    assert received == [[99]]
