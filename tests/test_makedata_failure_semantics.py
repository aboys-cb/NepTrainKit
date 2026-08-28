from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from ase import Atoms
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication

from NepTrainKit.core.cards.operation import DatasetOperation, StructureOperation
from NepTrainKit.ui.pages.makedata import MakeDataWidget
from NepTrainKit.ui.views._card.card_group import CardGroup
from NepTrainKit.ui.widgets.card_widget import FilterDataCard, MakeDataCard


@pytest.fixture(scope="module", autouse=True)
def _application():
    """Keep Qt creation in test execution instead of collection."""
    app = QApplication.instance() or QApplication([])
    yield app


class _CopyOperation(StructureOperation):
    def __init__(self):
        self.calls = 0

    def run_structure(self, structure, params):
        self.calls += 1
        return [structure.copy()]


class _FailAfterFirstOperation(StructureOperation):
    def __init__(self):
        self.calls = 0

    def run_structure(self, structure, params):
        self.calls += 1
        if self.calls > 1:
            raise ValueError("intentional failure")
        return [structure.copy()]


class _SlowOperation(StructureOperation):
    def run_structure(self, structure, params):
        time.sleep(0.08)
        return [structure.copy()]


class _EmptyOperation(StructureOperation):
    def run_structure(self, structure, params):
        return []


class _KeepFirstOperation(DatasetOperation):
    def run_dataset(self, dataset, params):
        return list(dataset[:1])


class _OperationCard(MakeDataCard):
    def __init__(self, operation, parent=None):
        self._operation = operation
        super().__init__(parent)

    def create_operation(self):
        return self._operation


class _FilterCard(FilterDataCard):
    def create_operation(self):
        return _KeepFirstOperation()


def _wait_until(predicate, timeout_ms=3000):
    app = QApplication.instance() or QApplication([])
    loop = QEventLoop()
    timer = QTimer()

    def poll():
        if predicate():
            loop.quit()

    timer.timeout.connect(poll)
    timer.start(10)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    return predicate()


def test_card_header_icon_actions_have_accessible_names():
    card = _OperationCard(_CopyOperation())

    assert card.doc_button.accessibleName() == "Open online documentation"
    assert card.info_button.accessibleName() == "Show card information and contributors"
    assert card.copy_json_button.accessibleName() == "Copy card JSON"
    assert card.view_output_button.accessibleName() == "View this card output"
    assert card.export_button.accessibleName() == "Export data"
    assert card.close_button.accessibleName() == "Close card"
    assert card.collapse_button.accessibleName() == "Collapse or expand card"


def test_failed_card_discards_partial_output_and_stops_page_chain():
    page = MakeDataWidget()
    failing = _OperationCard(_FailAfterFirstOperation(), page)
    next_operation = _CopyOperation()
    following = _OperationCard(next_operation, page)
    page.workspace_card_widget.add_card(failing)
    page.workspace_card_widget.add_card(following)
    page.dataset = [Atoms("H"), Atoms("He")]
    following.result_dataset = [Atoms("Li")]
    following.set_output_available(True)
    completion = QSignalSpy(failing.runFinishedSignal)

    with patch(
        "NepTrainKit.ui.widgets.card_widget.MessageManager.send_error_message"
    ) as error_message, patch(
        "NepTrainKit.ui.pages.makedata.MessageManager.send_success_message"
    ) as success_message:
        page.run_card()
        assert completion.count() or completion.wait(10000)
        assert failing.run_outcome == "failed"

    assert failing.result_dataset == []
    assert not failing.view_output_button.isEnabled()
    assert following.run_outcome == "idle"
    assert following.result_dataset == []
    assert not following.view_output_button.isEnabled()
    assert next_operation.calls == 0
    assert page._last_completed_card_index is None
    error_message.assert_called_once()
    success_message.assert_not_called()
    page.close()
    page.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def test_card_group_propagates_child_failure_without_running_later_children():
    group = CardGroup()
    failing = _OperationCard(_FailAfterFirstOperation(), group)
    next_operation = _CopyOperation()
    following = _OperationCard(next_operation, group)
    group.add_card(failing)
    group.add_card(following)
    group.set_dataset([Atoms("H"), Atoms("He")])
    following.result_dataset = [Atoms("Li")]
    following.set_output_available(True)

    with patch(
        "NepTrainKit.ui.widgets.card_widget.MessageManager.send_error_message"
    ):
        group.run()
        assert _wait_until(lambda: group.run_outcome == "failed")

    assert group.result_dataset == []
    assert following.result_dataset == []
    assert not following.view_output_button.isEnabled()
    assert next_operation.calls == 0
    assert not group.view_output_button.isEnabled()


def test_card_group_finishes_after_enabled_post_filter():
    group = CardGroup()
    child = _OperationCard(_CopyOperation(), group)
    filter_card = _FilterCard(group)
    group.add_card(child)
    group.set_filter_card(filter_card)
    group.set_dataset([Atoms("H"), Atoms("He")])
    completions = []
    group.runFinishedSignal.connect(completions.append)

    group.run()
    assert _wait_until(lambda: group.run_outcome == "succeeded")

    assert completions == [group.index]
    assert len(group.result_dataset) == 1
    assert group.result_dataset == filter_card.result_dataset
    assert group._merged_count == 2
    assert "2 merged → 1 kept" in group.merge_count_label.text()
    assert "2 input → 2 merged → 1 kept" in group.summary_label.text()
    assert group.status_badge.state() == "succeeded"
    assert group.status_badge.label.text() == "Done · 2→1"


def test_enabled_card_group_requires_at_least_one_enabled_branch():
    group = CardGroup()
    child = _OperationCard(_CopyOperation(), group)
    filter_card = _FilterCard(group)
    group.add_card(child)
    group.set_filter_card(filter_card)
    child.state_checkbox.setChecked(False)
    group.set_dataset([Atoms("H")])
    completions = []
    group.runFinishedSignal.connect(completions.append)

    with patch.object(filter_card, "run") as filter_run, patch(
        "NepTrainKit.ui.views._card.card_group.MessageManager.send_error_message"
    ) as error_message:
        group.run()

    assert group.run_outcome == "failed"
    assert group.result_dataset == []
    assert not group.view_output_button.isEnabled()
    assert completions == [group.index]
    assert "Run failed" in group.merge_count_label.text()
    assert group.status_badge.state() == "failed"
    filter_run.assert_not_called()
    error_message.assert_called_once_with(
        "Branch Merge needs at least one enabled branch."
    )


def test_disabled_card_group_still_bypasses_input():
    group = CardGroup()
    filter_card = _FilterCard(group)
    group.set_filter_card(filter_card)
    source = [Atoms("H")]
    group.state_checkbox.setChecked(False)
    group.set_dataset(source)
    completions = []
    group.runFinishedSignal.connect(completions.append)

    with patch.object(filter_card, "run") as filter_run:
        group.run()

    assert group.run_outcome == "succeeded"
    assert group.result_dataset is source
    assert group.view_output_button.isEnabled()
    assert completions == [group.index]
    assert "1 merged structures" in group.merge_count_label.text()
    assert "kept" not in group.merge_count_label.text()
    assert group.status_badge.state() == "disabled"
    filter_run.assert_not_called()


def test_workflow_reports_legal_empty_output_without_generated_success_message():
    page = MakeDataWidget()
    group = CardGroup(page)
    group.add_card(_OperationCard(_EmptyOperation(), group))
    page.workspace_card_widget.add_card(group)
    page.dataset = [Atoms("H")]

    with patch(
        "NepTrainKit.ui.pages.makedata.MessageManager.send_success_message"
    ) as success_message, patch(
        "NepTrainKit.ui.pages.makedata.MessageManager.send_info_message"
    ) as info_message:
        page.run_card()
        assert _wait_until(lambda: group.run_outcome == "succeeded")

    assert group.result_dataset == []
    assert group.status_badge.state() == "succeeded"
    assert group.status_badge.label.text() == "Done · 1→0"
    success_message.assert_not_called()
    info_message.assert_called_once_with(
        "Workflow completed with 0 output structures."
    )


def test_card_group_toggle_preserves_individual_branch_choices():
    group = CardGroup()
    first = _OperationCard(_CopyOperation(), group)
    second = _OperationCard(_CopyOperation(), group)
    group.add_card(first)
    group.add_card(second)
    second.state_checkbox.setChecked(False)

    group.state_checkbox.setChecked(False)
    group.state_checkbox.setChecked(True)

    assert first.check_state is True
    assert second.check_state is False
    assert "1/2 branch cards enabled" in group.summary_label.text()


def test_card_group_allows_only_one_post_filter_without_losing_it():
    group = CardGroup()
    first_filter = _FilterCard(group)
    second_filter = _FilterCard(group)

    assert group.set_filter_card(first_filter) is True
    with patch(
        "NepTrainKit.ui.views._card.card_group.MessageManager.send_warning_message"
    ) as warning:
        assert group.set_filter_card(second_filter) is False

    assert group.filter_card is first_filter
    assert group.filter_layout.indexOf(first_filter) >= 0
    assert group.filter_layout.indexOf(second_filter) == -1
    warning.assert_called_once()


def test_stopped_card_reports_cancellation_and_disables_partial_output():
    card = _OperationCard(_SlowOperation())
    card.set_dataset([Atoms("H") for _ in range(8)])
    card.run()
    assert _wait_until(
        lambda: hasattr(card, "worker_thread") and card.worker_thread.isRunning()
    )

    card.stop()
    assert _wait_until(lambda: card.run_outcome == "canceled")

    assert "Stopped" in card.status_label.text()
    assert not card.view_output_button.isEnabled()
