from __future__ import annotations

import math

import pytest
from PySide6.QtWidgets import QApplication

from NepTrainKit.core.magnetic_response import (
    LocalMagneticResponseParams,
    MagnetoelasticResponseParams,
    TextureMagneticResponseParams,
)
from NepTrainKit.ui.views._card.i18n_utils import combo_value, set_combo_value
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


def test_local_response_round_trip_preserves_every_automatic_pair_filter():
    card = LocalMagneticResponseCard()
    params = LocalMagneticResponseParams(
        response_kind="Atom pair canting",
        coordinate_scan_deg="-3,-1,0,1,3",
        target_mode="All eligible atoms",
        target_indices="2,4",
        pair_source="Auto by neighbor shell",
        pair_left_indices="2,4",
        pair_right_indices="3,5",
        pair_shell=2,
        pair_shell_tolerance=0.125,
        pair_element_filter="Fe-Co",
        pair_group_filter="A-B",
        bond_filter_mode="Near plane",
        bond_filter_axis=(1.0, 0.0, 0.0),
        bond_filter_tolerance=7.5,
        group_a="up",
        group_b="down",
        rotation_axis=(0.0, 0.0, 1.0),
        apply_elements="Fe,Co",
        moment_scale_scan="0.7,1.0,1.3",
        max_outputs=45,
    )

    card.set_params(params)

    assert card.get_params() == params
    assert card.pair_filters_checkbox.isChecked()
    assert not card.pair_filters_section.isHidden()
    assert not card.bond_axis_field.isHidden()
    assert card.advanced_checkbox.isChecked()


def test_local_response_only_shows_controls_used_by_the_selected_mode():
    card = LocalMagneticResponseCard()

    set_combo_value(card.kind_combo, "Single-spin tilt")
    set_combo_value(card.target_mode_combo, "First eligible atom")
    assert card.target_field.isHidden()
    assert not card.apply_field.isHidden()

    set_combo_value(card.target_mode_combo, "Explicit indices")
    assert not card.target_field.isHidden()

    set_combo_value(card.kind_combo, "Atom pair canting")
    set_combo_value(card.pair_source_combo, "Auto by neighbor shell")
    assert not card.pair_filters_checkbox.isHidden()
    assert card.pair_filters_section.isHidden()

    card.pair_filters_checkbox.setChecked(True)
    assert not card.pair_filters_section.isHidden()
    assert card.bond_axis_field.isHidden()

    set_combo_value(card.bond_mode_combo, "Near axis")
    assert not card.bond_axis_field.isHidden()


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


def test_soc_response_defaults_to_cell_reciprocal_q_and_reveals_only_active_inputs():
    card = SOCTextureResponseCard()
    set_combo_value(card.kind_combo, "Bulk / Bloch")

    assert combo_value(card.q_definition_combo) == "Cell reciprocal vector"
    assert not card.q_reciprocal_field.isHidden()
    assert card.q_direction_field.isHidden()
    assert card.get_params().q_reciprocal_index == (1, 0, 0)

    set_combo_value(card.q_definition_combo, "Cartesian vector")
    assert card.q_reciprocal_field.isHidden()
    assert not card.q_direction_field.isHidden()
    assert not card.q_magnitude_field.isHidden()


def test_soc_response_restores_legacy_cartesian_q_and_opens_nondefault_advanced_values():
    card = SOCTextureResponseCard()
    payload = card.to_dict()
    payload["params"].update(
        response_kind="General spiral",
        q_vector_cart=[0.1, 0.2, 0.0],
        cone_component=0.25,
        phase_deg=30.0,
        require_commensurate=False,
        max_outputs=12,
    )
    payload["params"].pop("q_definition", None)
    payload["params"].pop("q_reciprocal_index", None)

    card.from_dict(payload)
    restored = card.get_params()

    assert combo_value(card.q_definition_combo) == "Cartesian vector"
    assert restored.q_definition == "Cartesian vector"
    assert restored.q_vector_cart == pytest.approx((0.1, 0.2, 0.0))
    assert card.advanced_checkbox.isChecked()
    assert not card.advanced_section.isHidden()
