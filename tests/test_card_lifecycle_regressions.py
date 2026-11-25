"""Regression cases for the card-module audit: inputs, edits, exports and forks."""

import threading
from unittest.mock import Mock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from test_makedata_failure_semantics import _wait_until
from NepTrainKit.core.cards.lattice import SuperCellOperation, SuperCellParams
from NepTrainKit.core.workflow import WorkflowError, load_workflow
from NepTrainKit.ui.pages import makedata


@pytest.fixture
def page():
    viewer = makedata.MakeDataWidget()
    yield viewer
    viewer.stop_run_card()
    assert _wait_until(lambda: not viewer._has_running_card())
    viewer.close()


def add_scale(page, factor=2):
    card = page.add_card("SuperCellCard", at_workflow_root=True)
    card.set_params(SuperCellParams(super_scale=(factor, 1, 1)))
    return card


def run_page(page, last):
    page.run_card()
    assert _wait_until(lambda: last.run_outcome == "succeeded" and not page._has_running_card())


def setup_container(page, name):
    root = page.add_card(name, at_workflow_root=True)
    child = add_scale(page)
    page.workspace_card_widget.remove_card(child, close=False)
    if name == "WorkflowFork":
        root.add_card(child, root.branches[0])
        root.branches[1].enabled_checkbox.setChecked(False)
    else:
        root.add_card(child)
    page.dataset = [Atoms("H", cell=[3, 3, 3], pbc=True)]
    return root, child


@pytest.mark.parametrize("container", ["CardGroup", "WorkflowFork"])
def test_completed_container_edit_clears_every_nested_output_and_can_rerun(page, container):
    root, child = setup_container(page, container)
    run_page(page, root)
    assert len(child.result_dataset[0]) == 2
    child.super_scale_condition_frame.set_input_value([3, 1, 1])
    assert root.result_dataset == child.result_dataset == []
    assert root.run_outcome == child.run_outcome == "idle"
    assert not page._cards_for_export(True)
    run_page(page, root)
    assert len(child.result_dataset[0]) == 3
    assert page._cards_for_export(True)


@pytest.mark.parametrize("tag", ["stress_sample", "bcc|stress", "stress sample"])
def test_new_input_invalidates_outputs_and_preserves_config_tokens(page, tmp_path, tag):
    card = add_scale(page)
    page.dataset = [Atoms("H", cell=[3, 3, 3], pbc=True)]
    run_page(page, card)
    replacement = Atoms("He", cell=[3, 3, 3], pbc=True)
    replacement.info["Config_type"] = tag
    path = tmp_path / "replacement.xyz"
    write(path, replacement)
    page.load_base_structure([str(path)])
    assert card.result_dataset == []
    assert not page._cards_for_export(False)
    assert page.dataset[0].info["Config_type"] == tag.replace(" ", "|")
    run_page(page, card)
    assert card.result_dataset[0].get_chemical_formula() == "He2"


def test_disabling_stage_invalidates_downstream_and_changes_next_run(page):
    first, last = add_scale(page, 2), add_scale(page, 3)
    page.dataset = [Atoms("H", cell=[3, 3, 3], pbc=True)]
    run_page(page, last)
    assert len(last.result_dataset[0]) == 6
    first.state_checkbox.setChecked(False)
    assert first.result_dataset == last.result_dataset == []
    assert not page._cards_for_export(False)
    run_page(page, last)
    assert len(last.result_dataset[0]) == 3


def test_reordering_while_running_cancels_old_schedule_before_callback(page, monkeypatch):
    first, last = add_scale(page, 2), add_scale(page, 3)
    started, release = threading.Event(), threading.Event()

    class GatedOperation(SuperCellOperation):
        def run_structure(self, atoms, params):
            started.set()
            assert release.wait(5)
            return super().run_structure(atoms, params)

    monkeypatch.setattr(first, "create_operation", lambda: GatedOperation())
    page.dataset = [Atoms("H", cell=[3, 3, 3], pbc=True)]
    page.run_card()
    try:
        assert started.wait(2)
        page.workspace_card_widget.move_card(first, 1)
        assert first.run_outcome == "canceling"
    finally:
        release.set()
    assert _wait_until(lambda: not hasattr(first, "worker_thread"))
    assert first.run_outcome == "canceled"
    assert last.run_outcome == "idle"
    assert not page._cards_for_export(True)
    assert page._last_completed_card_index is None
    monkeypatch.setattr(first, "create_operation", lambda: SuperCellOperation())
    run_page(page, first)
    assert len(first.result_dataset[0]) == 6


def test_new_dynamic_rule_marks_saved_workflow_dirty_and_invalidates_results(page):
    card = page.add_card("RandomDopingCard", at_workflow_root=True)
    card.rules_widget.from_rules([{"target": "Fe", "dopants": {"Co": 1.}, "use": "count", "count": [1, 1]}])
    card.max_atoms_condition_frame.set_input_value([1])
    page.dataset = [Atoms("Fe2", positions=[[0, 0, 0], [1, 1, 1]], cell=[4, 4, 4], pbc=True)]
    run_page(page, card)
    page._workflow_dirty = False
    row = card.rules_widget.add_rule()
    row.from_rule({"target": "Fe", "dopants": {"Ni": 1.}, "use": "count", "count": [1, 1]})
    assert page._workflow_dirty
    assert card.result_dataset == []
    assert not page._cards_for_export(True)


def test_new_element_row_controls_are_tracked_after_add_button(page):
    from NepTrainKit.core.cards.lattice import PerturbParams
    card = page.add_card("PerturbCard", at_workflow_root=True)
    card.set_params(PerturbParams(engine_type=1, max_num=1, use_element_scaling=True))
    page.dataset = [Atoms("H", cell=[3, 3, 3], pbc=True)]
    run_page(page, card)
    page._workflow_dirty = False
    card.add_element_button.click()
    card.element_rows[-1].element_input.setText("H")
    assert card.result_dataset == []
    assert page._workflow_dirty


def test_table_cell_edit_invalidates_results_and_marks_workflow_dirty(page):
    from NepTrainKit.core.cards.alloy import ConditionalReplaceParams
    card = page.add_card("ConditionalReplaceCard", at_workflow_root=True)
    card.set_params(ConditionalReplaceParams(target="Fe", replacements="Co:1"))
    page.dataset = [Atoms("Fe", cell=[3, 3, 3], pbc=True)]
    run_page(page, card)
    page._workflow_dirty = False
    card.replacements_edit.table.item(0, 1).setText("2")
    assert card.result_dataset == []
    assert page._workflow_dirty


@pytest.mark.parametrize("container", [None, "CardGroup", "WorkflowFork"])
def test_direct_export_uses_deep_snapshot_after_card_is_reset(page, tmp_path, container):
    if container:
        root, child = setup_container(page, container)
        if container == "WorkflowFork":
            root.merge_checkbox.setChecked(True)
    else:
        root = child = add_scale(page)
        page.dataset = [Atoms("H", cell=[3, 3, 3], pbc=True)]
    run_page(page, root)
    original = root.result_dataset[0]
    original.info["custom"] = np.array([3., 4.])
    path = tmp_path / "snapshot.xyz"
    worker = Mock()
    module = __import__(type(root).__module__, fromlist=["BackgroundTask"])
    if container is None:
        from NepTrainKit.ui.widgets import card_widget as module
    with patch.object(module, "BackgroundTask", return_value=worker), \
         patch.object(module, "call_path_dialog", return_value=str(path)):
        root.export_data()
    original.info["custom"][:] = 100.
    original.positions[:] = 2.
    root.set_dataset([])
    func, *args = worker.start_work.call_args.args
    func(*args)
    exported = read(path)
    assert exported.get_chemical_formula() == "H2"
    np.testing.assert_equal(exported.info["custom"], [3., 4.])
    assert not np.all(exported.positions == 2.)


@pytest.mark.parametrize("include_all", [False, True])
def test_workflow_export_snapshot_survives_input_replacement(page, tmp_path, include_all):
    card = add_scale(page)
    page.dataset = [Atoms("H", cell=[3, 3, 3], pbc=True)]
    run_page(page, card)
    path = tmp_path / "export.xyz"
    worker = Mock()
    with patch.object(makedata, "BackgroundTask", return_value=worker), \
         patch.object(makedata, "call_path_dialog", return_value=str(path)):
        page._start_export(include_all)
    page._invalidate_workflow_results()
    func, *args = worker.start_work.call_args.args
    func(*args)
    assert read(path).get_chemical_formula() == "H2"


def test_cancelled_partial_output_is_unavailable_through_every_entry(page, monkeypatch):
    card = add_scale(page)
    started, release = threading.Event(), threading.Event()

    class PartialOperation(SuperCellOperation):
        count = 0

        def run_structure(self, atoms, params):
            self.count += 1
            if self.count == 2:
                started.set()
                assert release.wait(5)
            return [atoms.copy()]

    monkeypatch.setattr(card, "create_operation", lambda: PartialOperation())
    card.set_dataset([Atoms("H"), Atoms("He")])
    card.run()
    try:
        assert started.wait(2)
        card.stop()
    finally:
        release.set()
    assert _wait_until(lambda: not hasattr(card, "worker_thread"))
    assert card.run_outcome == "canceled" and card.result_dataset
    assert not page._cards_for_export(True)
    handoff = Mock()
    page.finalOutputRequestedSignal.connect(handoff)
    with patch.object(makedata.MessageManager, "send_info_message"):
        page.request_card_output(card)
        page.request_selected_outputs()
    handoff.assert_not_called()
    with patch("NepTrainKit.ui.widgets.card_widget.call_path_dialog") as dialog:
        card.export_data()
    dialog.assert_not_called()


def test_generator_first_branch_runs_without_external_input(page):
    fork = page.add_card("WorkflowFork", at_workflow_root=True)
    generator = page.add_card("CrystalPrototypeBuilderCard", at_workflow_root=True)
    stage = add_scale(page)
    for card in (generator, stage):
        page.workspace_card_widget.remove_card(card, close=False)
        fork.add_card(card, fork.branches[0])
    fork.branches[1].enabled_checkbox.setChecked(False)
    assert not fork.requires_input_dataset
    run_page(page, fork)
    assert len(stage.result_dataset[0]) == len(generator.result_dataset[0]) * 2
    assert page._cards_for_export(True) == [stage]


def test_headless_fork_requires_explicit_merge_and_preserves_terminal_stages():
    def scale(n):
        return {"class": "SuperCellCard", "params": {"super_scale": [n, 1, 1]}}

    fork = {"class": "WorkflowFork", "merge": False, "branches": [
        {"cards": [scale(2), scale(3)]}, {"cards": [scale(4)]}]}
    with pytest.raises(WorkflowError, match="explicit Merge"):
        load_workflow({"cards": [fork, scale(5)]})
    seed = Atoms("H", cell=[3, 3, 3], pbc=True)
    result = load_workflow({"cards": [fork]}).run([seed])
    assert result.final() == []
    assert [len(output.structures[0]) for output in result.card_outputs[:-1]] == [2, 6, 4]
    fork["merge"] = True
    merged = load_workflow({"cards": [fork, scale(5)]}).run([seed])
    assert [len(atoms) for atoms in merged.final()] == [30, 20]
    generated = load_workflow({"cards": [{"class": "CrystalPrototypeBuilderCard", "params": {}}, scale(2)]})
    assert not generated.requires_input()
    assert generated.run().final()
