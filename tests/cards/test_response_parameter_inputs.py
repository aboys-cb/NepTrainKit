from __future__ import annotations

import math

import pytest
from PySide6.QtWidgets import QApplication

from NepTrainKit.core.magnetic_response import (
    LocalMagneticResponseParams,
    MagnetoelasticResponseParams,
    TextureMagneticResponseParams,
)
from NepTrainKit.ui.views._card.local_magnetic_response_card import LocalMagneticResponseCard
from NepTrainKit.ui.views._card.magnetoelastic_response_card import MagnetoelasticResponseCard
from NepTrainKit.ui.views._card.soc_texture_response_card import SOCTextureResponseCard
from NepTrainKit.ui.widgets import DirectionInput, KeyValueTableInput, NumericScanInput


@pytest.fixture(scope="module", autouse=True)
def application():
    app = QApplication.instance() or QApplication([])
    yield app


def test_numeric_scan_uses_range_by_default_and_preserves_irregular_legacy_lists():
    editor = NumericScanInput(minimum=-10.0, maximum=10.0, decimals=3)
    editor.set_range(-2.0, 2.0, 1.0)
    assert editor.scan_text() == "-2,-1,0,1,2"
    assert not editor.custom_checkbox.isChecked()

    editor.set_scan_text("-5,-2,0,3,5")
    assert editor.custom_checkbox.isChecked()
    assert editor.scan_text() == "-5,-2,0,3,5"


def test_direction_presets_normalize_custom_vectors_and_mapping_table_keeps_contract():
    direction = DirectionInput(default=(1.0, 1.0, 0.0))
    assert direction.vector() == pytest.approx((1 / math.sqrt(2), 1 / math.sqrt(2), 0.0))
    direction.set_input_value((2.0, 1.0, 0.0))
    assert math.sqrt(sum(value * value for value in direction.get_input_value())) == pytest.approx(1.0)

    table = KeyValueTableInput("Element", "Ratio")
    table.setText("Fe:2,O:1")
    assert table.text() == "Fe:2,O:1"


def test_magnetoelastic_ui_exposes_percent_range_but_keeps_fraction_params():
    card = MagnetoelasticResponseCard()
    card.struct_scan.set_range(-2.0, 2.0, 1.0)
    params = card.get_params()
    assert params.structural_scan == "-0.02,-0.01,0,0.01,0.02"
    assert "5 lattice points" in card.get_summary_text()

    legacy = MagnetoelasticResponseParams(structural_scan="-0.02,-0.005,0,0.02")
    card.set_params(legacy)
    assert card.struct_scan.custom_checkbox.isChecked()
    assert card.get_params() == legacy


def test_local_response_switches_scan_meaning_without_losing_serialized_values():
    card = LocalMagneticResponseCard()
    params = LocalMagneticResponseParams(
        response_kind="Moment magnitude",
        moment_scale_scan="0.7,0.9,1.0,1.15",
        target_mode="Explicit indices",
        target_indices="2,4",
    )
    card.set_params(params)
    assert card.scan_field.caption.text() == card.tr("Moment scale scan")
    assert not card.target_field.isHidden()
    assert card.scan_input.custom_checkbox.isChecked()
    assert card.get_params() == params


def test_soc_response_roundtrip_splits_q_direction_and_magnitude():
    card = SOCTextureResponseCard()
    params = TextureMagneticResponseParams(
        response_kind="General spiral",
        coordinate_scan="-2,-1,0,1,2",
        q_vector_cart=(0.1, 0.2, 0.0),
        plane_normal=(0.0, 0.0, 1.0),
        phase_deg=12.5,
    )
    card.set_params(params)
    restored = card.get_params()
    assert restored.q_vector_cart == pytest.approx(params.q_vector_cart)
    assert restored.phase_deg == pytest.approx(12.5)
    assert card.scan_field.caption.text() == card.tr("Signed q scan (multiples of base q)")
    assert not card.plane_field.isHidden()
