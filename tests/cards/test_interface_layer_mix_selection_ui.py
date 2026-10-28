"""Exercise selective exchange through real card controls and saved workflows."""

import json
from collections import Counter

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from NepTrainKit.core.cards.alloy import InterfaceLayerMixParams
from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.core.io.importers import ase_atoms_to_structure
from NepTrainKit.ui.pages.makedata import MakeDataWidget
from NepTrainKit.ui.views._card.i18n_utils import set_combo_value
from NepTrainKit.ui.views._card.interface_layer_mix_card import InterfaceLayerMixCard


def _silicon_gap():
    symbols, positions = [], []
    for z in (0.2, 0.3, 0.7, 0.8):
        for i in range(20):
            symbols.append("Si" if z < 0.5 else ("Ga" if i % 2 else "P"))
            positions.append([i / 20, 0.25, z])
    return Atoms(symbols, scaled_positions=positions, cell=np.diag([20, 4, 20]), pbc=[True, True, False])


def _dispose(widget):
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("pair_text,expected", [
    ("Si-P", Counter({("Si", "P"): 8})),
    ("Si-Ga", Counter({("Si", "Ga"): 8})),
    ("Si-P,Si-Ga", Counter({("Si", "P"): 4, ("Si", "Ga"): 4})),
])
def test_selected_pairs_from_controls_run_the_actual_branch(pair_text, expected):
    atoms = _silicon_gap()
    card = InterfaceLayerMixCard()
    card.set_params(InterfaceLayerMixParams(axis="c", concentration=0.2, use_seed=True, seed=11))
    set_combo_value(card.pair_selection_combo, "selected")
    card.exchange_pairs_edit.setText(pair_text)
    set_combo_value(card.pair_distribution_combo, "equal")
    params = card.get_params()
    out = card.create_operation().run_structure(atoms, params)[0]
    before, after = atoms.get_chemical_symbols(), out.get_chemical_symbols()
    counts = Counter((old, new) for old, new in zip(before[:40], after[:40]) if old != new)
    assert counts == expected
    assert Counter(after) == Counter(before)
    np.testing.assert_array_equal(out.positions, atoms.positions)
    assert not card.exchange_pairs_field.isHidden()
    _dispose(card)


@pytest.mark.parametrize("selection", ["all", "selected"])
@pytest.mark.parametrize("allocation", ["random", "equal"])
def test_new_json_preserves_tuple_contract_and_every_enum(selection, allocation):
    card = InterfaceLayerMixCard()
    expected = InterfaceLayerMixParams(
        axis="c", concentration=0.2, pair_selection=selection,
        exchange_pairs=(("Si", "Ga"), ("Si", "P")),
        pair_distribution=allocation, layer_pair_counts=(4, 4), use_seed=True, seed=4,
    )
    card.set_params(expected)
    payload = json.loads(json.dumps(card.to_dict()))
    assert payload["params"]["exchange_pairs"] == [["Si", "Ga"], ["Si", "P"]]
    assert payload["params"]["layer_pair_counts"] == [4, 4]
    restored = InterfaceLayerMixCard()
    restored.from_dict(payload)
    assert restored.get_params() == expected
    assert len(restored.create_operation().run_structure(_silicon_gap(), restored.get_params())) == 1
    _dispose(restored)
    _dispose(card)


def test_quotas_disable_concentration_and_show_derived_amount():
    card = InterfaceLayerMixCard()
    card.set_preview_structure(_silicon_gap())
    set_combo_value(card.mode_combo, "gradient")
    card.layer_pair_counts_edit.setText("4,4")
    assert card.get_params().mode == "fixed"
    assert not card.mode_field.isEnabled()
    assert not card.concentration_field.isEnabled()
    assert "realized 20%" in card.get_summary_text()
    assert "nearest first" in card.get_guidance_text()
    card.layer_pair_counts_edit.clear()
    assert card.mode_field.isEnabled()
    assert card.concentration_field.isEnabled()
    _dispose(card)


def test_random_window_is_default_and_layer_quotas_are_opt_in():
    card = InterfaceLayerMixCard()
    assert card.get_params().layer_pair_counts == ()
    assert card.get_params().pair_distribution == "random"
    assert not card.layer_counts_checkbox.isChecked()
    assert card.layer_pair_counts_field.isHidden()
    card.layer_counts_checkbox.setChecked(True)
    card.layer_pair_counts_edit.setText("4,4")
    assert card.get_params().layer_pair_counts == (4, 4)
    card.layer_counts_checkbox.setChecked(False)
    assert card.layer_pair_counts_field.isHidden()
    assert card.get_params().layer_pair_counts == ()
    assert card.concentration_field.isEnabled()
    _dispose(card)


def test_invalid_gradient_quota_json_is_not_silently_changed():
    card = InterfaceLayerMixCard()
    with pytest.raises(CardOperationError, match="fixed mixing schedule"):
        card.from_dict({"check_state": True, "params": {"mode": "gradient", "layer_pair_counts": [4, 4]}})
    _dispose(card)


def test_fixed_four_output_summary_does_not_look_like_a_gradient():
    card = InterfaceLayerMixCard()
    card.set_preview_structure(_silicon_gap())
    card.set_params(InterfaceLayerMixParams(axis="c", concentration=0.2, num_structures=4))
    assert "realized 20%" in card.get_summary_text()
    assert "20%–20%" not in card.get_summary_text()
    _dispose(card)


def test_malformed_quota_is_visible_in_preview_instead_of_crashing():
    card = InterfaceLayerMixCard()
    card.set_preview_structure(_silicon_gap())
    card.layer_pair_counts_edit.setText("4,oops")
    assert "comma-separated" in card.get_summary_text()
    assert "comma-separated" in card.get_guidance_text()
    _dispose(card)


def test_old_json_runs_the_original_all_pairs_mode():
    card = InterfaceLayerMixCard()
    card.from_dict({"check_state": True, "params": {"axis": "c", "concentration": 0.2, "use_seed": True, "seed": 12}})
    assert card.get_params().pair_selection == "all"
    assert card.get_params().pair_distribution == "random"
    assert card.get_params().exchange_pairs == ()
    assert card.get_params().layer_pair_counts == ()
    assert card.exchange_pairs_field.isHidden()
    assert len(card.create_operation().run_structure(_silicon_gap(), card.get_params())) == 1
    _dispose(card)


def test_selection_fits_real_inspector_after_wide_to_default_resize():
    app = QApplication.instance()
    widget = MakeDataWidget()
    widget.dataset = [_silicon_gap()]
    card = widget.add_card("InterfaceLayerMixCard")
    card.set_params(InterfaceLayerMixParams(
        axis="c", concentration=0.2, pair_selection="selected",
        exchange_pairs=(("Si", "P"), ("Si", "Ga")),
        pair_distribution="equal", layer_pair_counts=(4, 4),
    ))
    area = widget.workspace_card_widget
    widget.show()
    area.select_card(card, expand=True)
    for width, height in ((1600, 900), (1151, 700), (1600, 900), (1151, 700)):
        widget.resize(width, height)
        for _ in range(3):
            app.processEvents()
        scroll = area.guidance_panel.parameter_scroll
        assert area.guidance_panel.isVisible()
        assert area.library_panel.isVisible()
        assert scroll.horizontalScrollBar().maximum() == 0
        assert card.exchange_pairs_edit.width() > 100
        assert card.left_layers_frame.object_list[0].width() > 30
    _dispose(widget)


@pytest.mark.parametrize("pairs,expected", [
    ((("Si", "Ga"),), Counter({("Si", "Ga"): 8})),
    ((("Si", "P"),), Counter({("Si", "P"): 8})),
    ((("Si", "Ga"), ("Si", "P")), Counter({("Si", "Ga"): 4, ("Si", "P"): 4})),
])
def test_compact_interface_xyz_keeps_labels_and_arrays_across_real_export_paths(tmp_path, pairs, expected):
    source = _silicon_gap()
    source.set_masses()
    source.new_array("group", np.column_stack((np.repeat(np.arange(4), 20), np.zeros(80, dtype=int))))
    stale_record = '{"pairs":[' + '"old-record",' * 2000 + ']}'
    source.info["interface_mix_audit"] = stale_record
    widget = MakeDataWidget()
    widget.dataset = [source]
    card = widget.add_card("InterfaceLayerMixCard")
    card.set_params(InterfaceLayerMixParams(
        axis="c", concentration=0.2, pair_selection="selected", exchange_pairs=pairs,
        pair_distribution="equal", num_structures=2, use_seed=True, seed=11,
    ))
    outputs = card.create_operation().run_structure(source, card.get_params())
    card.result_dataset = outputs
    direct = tmp_path / "direct.xyz"
    workflow = tmp_path / "workflow.xyz"
    handoff = tmp_path / "handoff.xyz"
    card.write_result_dataset(direct)
    widget._export_file(str(workflow), [card])
    with handoff.open("w", encoding="utf-8") as handle:
        for output in outputs:
            ase_atoms_to_structure(output).write(handle)

    assert source.info["interface_mix_audit"] == stale_record
    for path in (direct, workflow, handoff):
        text = path.read_text(encoding="utf-8")
        assert "interface_mix_audit" not in text
        assert '\\"' not in text
        headers = text.splitlines()[1::len(source) + 2]
        assert len(headers) == 2
        assert all(len(header) < 500 for header in headers)
        assert all("L=2,R=2,c=0.2" in header and ",s=" in header for header in headers)
        loaded = read(path, index=":", format="extxyz")
        assert len(loaded) == len(outputs)
        for expected_output, restored in zip(outputs, loaded):
            assert "interface_mix_audit" not in expected_output.info
            assert restored.info["Config_type"] == expected_output.info["Config_type"]
            np.testing.assert_array_equal(restored.numbers, expected_output.numbers)
            np.testing.assert_allclose(restored.positions, source.positions, atol=1e-7, rtol=0)
            np.testing.assert_allclose(restored.cell.array, source.cell.array, atol=1e-10, rtol=0)
            np.testing.assert_array_equal(restored.pbc, source.pbc)
            np.testing.assert_array_equal(restored.arrays["group"], source.arrays["group"])
            # The existing Dataset Display handoff stores per-atom reals as
            # float32; direct card/workflow exports retain ASE precision.
            mass_tolerance = 2e-6 if path == handoff else 1e-7
            np.testing.assert_allclose(restored.get_masses(), expected_output.get_masses(), atol=mass_tolerance, rtol=0)
            transitions = Counter(
                (old, new) for old, new in zip(source.get_chemical_symbols()[:40], restored.get_chemical_symbols()[:40])
                if old != new
            )
            assert transitions == expected
    assert {path.name for path in tmp_path.iterdir()} == {"direct.xyz", "workflow.xyz", "handoff.xyz"}
    _dispose(widget)
