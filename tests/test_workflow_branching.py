from __future__ import annotations

from unittest.mock import patch

from ase import Atoms
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QPoint, QTimer
from PySide6.QtWidgets import QApplication

from NepTrainKit.core.cards.operation import StructureOperation
from NepTrainKit.ui.pages.makedata import MakeDataWidget
from NepTrainKit.ui.views._card.card_group import CardGroup
from NepTrainKit.ui.views._card.cell_strain_card import CellStrainCard
from NepTrainKit.ui.views._card.perturb_card import PerturbCard
from NepTrainKit.ui.views._card.workflow_fork import WorkflowFork
from NepTrainKit.ui.widgets import MakeDataCard, MakeWorkflowArea


def _app():
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout_ms=3000):
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


def _dispose(widget) -> None:
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class _HistoryOperation(StructureOperation):
    def __init__(self, name: str):
        self.name = name

    def run_structure(self, structure, params):
        result = structure.copy()
        history = list(result.info.get("history", []))
        history.append(self.name)
        result.info["history"] = history
        return [result]


class _FailOperation(StructureOperation):
    def run_structure(self, structure, params):
        raise ValueError("intentional branch failure")


class _OperationCard(MakeDataCard):
    def __init__(self, operation, parent=None):
        self._operation = operation
        super().__init__(parent)
        self.setTitle(operation.__class__.__name__)

    def create_operation(self):
        return self._operation


def test_compact_workflow_nodes_edit_the_selected_card_in_the_inspector():
    app = _app()
    area = MakeWorkflowArea()
    first = PerturbCard()
    second = CellStrainCard()

    area.add_card(first)
    area.add_card(second)
    area.resize(1280, 760)
    area.show()
    app.processEvents()

    assert area.cards == [first, second]
    assert first.isVisible()
    assert second.isVisible()
    assert first.window_state == "collapse"
    assert second.window_state == "collapse"
    assert area.guidance_panel._editor_widget is second.setting_widget
    assert area.guidance_panel.title_label.text() == second.getTitle()
    assert "Type: Lattice" in area.guidance_panel.about_label.text()
    assert "Contributors: NepTrainKit" in area.guidance_panel.about_label.text()
    assert area.guidance_panel.docs_button.isEnabled()
    assert area.guidance_panel.copy_card_button.isEnabled()
    assert area.guidance_panel.tabs.count() == 2
    assert area.guidance_panel.tabs.tabText(0) == "Parameters"
    assert area.guidance_panel.tabs.tabText(1) == "Guidance"

    area.select_card(first, expand=True)
    app.processEvents()
    assert first.window_state == "collapse"
    assert second.window_state == "collapse"
    assert area.guidance_panel._editor_widget is first.setting_widget
    assert area.guidance_panel.title_label.text() == first.getTitle()
    area.guidance_panel.copy_card_button.click()
    assert QApplication.clipboard().text() == first.to_json_text()
    _dispose(area)


def test_workflow_cards_use_a_centered_readable_width_on_wide_windows():
    app = _app()
    area = MakeWorkflowArea()
    card = PerturbCard()
    area.add_card(card)
    area.resize(1600, 760)
    area.show()
    app.processEvents()

    assert card.width() == area._CARD_MAX_WIDTH
    assert card.geometry().left() > 80
    _dispose(area)


def test_fork_keeps_extra_width_needed_for_parallel_lanes():
    app = _app()
    area = MakeWorkflowArea()
    card = PerturbCard()
    fork = WorkflowFork()
    area.add_card(card)
    area.add_card(fork)
    area.resize(1600, 760)
    area.show()
    app.processEvents()

    assert card.width() == area._CARD_MAX_WIDTH
    assert fork.width() == min(
        area._FORK_MAX_WIDTH,
        area.scroll_area.viewport().width() - 72,
    )
    assert fork.width() > card.width()
    assert fork.connector.isVisible()
    _dispose(area)


def test_close_removes_root_card_state_and_new_cards_still_work():
    app = _app()
    area = MakeWorkflowArea()
    first = PerturbCard()
    closed = CellStrainCard()
    area.add_card(first)
    area.add_card(closed)
    area.show()
    app.processEvents()

    closed.close_button.click()
    app.processEvents()
    assert area.cards == [first]
    assert area.canvas_layout.count() == 3

    replacement = CellStrainCard()
    area.add_card(replacement)
    app.processEvents()
    assert area.cards == [first, replacement]
    assert replacement.isVisible()
    _dispose(area)


def test_close_removes_nested_branch_card_immediately():
    app = _app()
    fork = WorkflowFork()
    child = PerturbCard()
    fork.add_card(child, fork.branches[0])
    fork.show()
    app.processEvents()

    child.set_workflow_selected(True)
    child.close_button.click()
    app.processEvents()
    assert fork.branches[0].cards == []
    assert fork.branches[0].empty_label.isVisible()
    _dispose(fork)


def test_cards_expose_drag_handle_and_canvas_tracks_insertion_slot():
    app = _app()
    area = MakeWorkflowArea()
    first = PerturbCard()
    second = CellStrainCard()
    area.add_card(first)
    area.add_card(second)
    area.show()
    app.processEvents()

    assert first.drag_handle.isVisible()
    assert first.drag_handle.toolTip() == "Drag to reorder card"
    assert first.category_tag.isVisible()
    assert first.copy_json_button.isHidden()
    assert first.status_badge.isVisible()
    assert first.status_dot.isHidden()
    assert first.export_button.isHidden()
    assert first.result_action_group.isHidden()
    assert first.headerView.height() == 74
    assert first.headerTopView.height() == 40
    assert first.headerInfoView.isVisible()
    assert first.headerLabel.geometry().bottom() <= first.headerTopView.height()
    assert first.summary_label.parentWidget() is first.headerInfoView
    assert first.status_badge.parentWidget() is first.headerInfoView
    assert first.result_action_group.parentWidget() is first.headerTopView
    assert first.copy_json_button.size().width() == 28
    assert first.copy_json_button.iconSize().width() == 14
    assert first.close_button.size().width() == 28
    assert first.close_button.iconSize().width() == 11
    checkbox_to_type_gap = (
        first.category_tag.geometry().left()
        - first.state_checkbox.geometry().right()
        - 1
    )
    assert first.state_checkbox.width() <= 25
    assert checkbox_to_type_gap <= 2
    first.set_output_available(True)
    assert first.view_output_button.isVisible()
    assert first.export_button.isVisible()
    assert first.result_action_group.isVisible()
    area.canvas.set_drop_index(1)
    assert area.canvas._drop_index == 1
    area.canvas.set_drop_index(None)
    assert area.canvas._drop_index is None
    _dispose(area)


def test_parameter_summary_stays_stable_after_runtime_result_moves_to_badge():
    app = _app()
    area = MakeWorkflowArea()
    card = PerturbCard()
    area.add_card(card)
    area.resize(1200, 720)
    area.show()
    app.processEvents()

    parameter_summary = card.get_summary_text()
    card.set_dataset([Atoms("H"), Atoms("He")])
    card.result_dataset = [Atoms("H"), Atoms("He"), Atoms("Li")]
    card.run_outcome = "succeeded"
    card.update_dataset_info()
    card._set_card_status("succeeded", card._status_count_detail())
    card.refresh_compact_presentation()

    assert card.summary_label.text() == parameter_summary
    assert card.status_badge.label.text() == "Done · 2→3"
    assert card.view_output_button.isVisible()
    assert card.export_button.isVisible()
    _dispose(area)


def test_top_level_drop_reorders_against_visible_insertion_position():
    app = _app()
    area = MakeWorkflowArea()
    first = PerturbCard()
    second = CellStrainCard()
    third = PerturbCard()
    for card in (first, second, third):
        area.add_card(card)
    area.resize(1200, 800)
    area.show()
    app.processEvents()

    class DropEvent:
        accepted = False

        def source(self):
            return first

        def acceptProposedAction(self):
            self.accepted = True

        def ignore(self):
            self.accepted = False

    canvas_point = QPoint(third.geometry().center().x(), third.geometry().bottom() + 8)
    viewport_point = area.scroll_area.viewport().mapFrom(area.canvas, canvas_point)
    event = DropEvent()
    area.canvas.set_drop_index(3)
    area._handle_top_level_drop(event, viewport_point)

    assert event.accepted
    assert area.cards == [second, third, first]
    assert area.canvas._drop_index is None
    _dispose(area)


def test_new_card_is_inserted_after_current_workflow_context():
    app = _app()
    area = MakeWorkflowArea()
    first = PerturbCard()
    second = CellStrainCard()
    third = PerturbCard()
    for card in (first, second, third):
        area.add_card(card)
    area.select_card(first, expand=False)

    inserted = CellStrainCard()
    area.add_card(inserted)
    app.processEvents()

    assert area.cards == [first, inserted, second, third]
    assert inserted.window_state == "collapse"
    assert first.window_state == "collapse"
    assert area.guidance_panel._editor_widget is inserted.setting_widget
    _dispose(area)


def test_new_card_stays_in_selected_branch_context():
    app = _app()
    area = MakeWorkflowArea()
    fork = WorkflowFork()
    area.add_card(fork)
    existing = PerturbCard()
    fork.add_card(existing, fork.branches[0])
    area.select_card(existing, expand=True)

    inserted = CellStrainCard()
    area.add_card(inserted)
    app.processEvents()

    assert area.cards == [fork]
    assert fork.branches[0].cards == [existing, inserted]
    assert inserted.window_state == "collapse"
    assert existing.window_state == "collapse"
    assert area.guidance_panel._editor_widget is inserted.setting_widget
    _dispose(area)


def test_top_level_drag_placeholder_reorders_and_cancel_restores_source():
    app = _app()
    area = MakeWorkflowArea()
    first = PerturbCard()
    second = CellStrainCard()
    third = PerturbCard()
    for card in (first, second, third):
        area.add_card(card)
    area.resize(1200, 800)
    area.show()
    app.processEvents()

    area._on_drag_started(first)
    app.processEvents()
    assert area._drag_placeholder is not None
    assert not first.isVisible()

    class DropEvent:
        accepted = False

        def source(self):
            return first

        def acceptProposedAction(self):
            self.accepted = True

        def ignore(self):
            self.accepted = False

    canvas_point = QPoint(third.geometry().center().x(), third.geometry().bottom() + 12)
    viewport_point = area.scroll_area.viewport().mapFrom(area.canvas, canvas_point)
    event = DropEvent()
    area._handle_top_level_drop(event, viewport_point)
    app.processEvents()

    assert event.accepted
    assert area.cards == [second, third, first]
    assert first.isVisible()
    assert area._drag_placeholder is None

    area._on_drag_started(second)
    app.processEvents()
    assert not second.isVisible()
    area._on_drag_finished(second, False)
    app.processEvents()
    assert area.cards == [second, third, first]
    assert second.isVisible()
    assert area._drag_placeholder is None
    _dispose(area)


def test_branch_drag_placeholder_reorders_without_expanding_other_cards():
    app = _app()
    area = MakeWorkflowArea()
    fork = WorkflowFork()
    area.add_card(fork)
    first = PerturbCard()
    second = CellStrainCard()
    branch = fork.branches[0]
    fork.add_card(first, branch)
    fork.add_card(second, branch)
    first.window_state = "collapse"
    second.window_state = "expand"
    area.show()
    app.processEvents()

    first.dragStartedSignal.emit(first)
    app.processEvents()
    assert branch._drag_placeholder is not None
    assert not first.isVisible()

    area._move_card_to_branch(first, branch, 2)
    first.dragFinishedSignal.emit(first, True)
    app.processEvents()

    assert branch.cards == [second, first]
    assert first.isVisible()
    assert branch._drag_placeholder is None
    assert first.window_state == "collapse"
    assert second.window_state == "expand"
    _dispose(area)


def test_drag_edge_auto_scroll_moves_long_workflow_viewport():
    app = _app()
    area = MakeWorkflowArea()
    for _ in range(10):
        area.add_card(PerturbCard())
    area.resize(900, 360)
    area.show()
    app.processEvents()

    bar = area.scroll_area.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(bar.maximum() // 2)
    before = bar.value()
    area._update_drag_auto_scroll(area.scroll_area.viewport().height() - 1)
    area._auto_scroll_drag()
    assert bar.value() > before

    before = bar.value()
    area._update_drag_auto_scroll(1)
    area._auto_scroll_drag()
    assert bar.value() < before
    area._stop_drag_auto_scroll()
    assert not area._drag_scroll_timer.isActive()
    _dispose(area)


def test_guidance_panel_hides_before_main_canvas_becomes_too_narrow():
    app = _app()
    area = MakeWorkflowArea()
    card = PerturbCard()
    area.add_card(card)
    area.resize(1000, 700)
    area.show()
    app.processEvents()
    assert area.guidance_panel.isVisible()
    assert area.guidance_panel._editor_widget is card.setting_widget

    area.resize(700, 700)
    app.processEvents()
    assert not area.guidance_panel.isVisible()
    assert area.guidance_panel._editor_widget is None
    assert card.window_state == "expand"
    assert card.setting_widget.parent() is card.view
    _dispose(area)


def test_console_stays_toolbar_height_while_canvas_takes_remaining_space():
    app = _app()
    page = MakeDataWidget()
    page.resize(1280, 800)
    page.show()
    app.processEvents()

    assert page.setting_group.height() == 54
    assert page.workspace_card_widget.height() > 650
    _dispose(page)


def test_fork_stacks_branches_on_narrow_canvas_and_limits_branch_count():
    app = _app()
    fork = WorkflowFork()
    fork.resize(1100, 700)
    fork.show()
    app.processEvents()
    assert fork.connector.isVisible()

    fork.add_branch()
    assert len(fork.branches) == 3
    assert not fork.add_branch_button.isEnabled()

    fork.resize(700, 900)
    app.processEvents()
    assert not fork.connector.isVisible()
    assert fork.branches[1].geometry().top() >= fork.branches[0].geometry().bottom()
    _dispose(fork)


def test_workflow_can_move_existing_card_into_group_and_fork_branch():
    area = MakeWorkflowArea()
    card = PerturbCard()
    group = CardGroup()
    fork = WorkflowFork()
    area.add_card(card)
    area.add_card(group)
    area.add_card(fork)

    assert area.move_card_to_group(card, group)
    assert card in group.card_list
    assert card not in area.cards

    area._move_card_to_branch(card, fork.branches[0], 0)
    assert card not in group.card_list
    assert fork.branches[0].cards == [card]
    assert card.category_tag.isHidden()
    assert card.copy_json_button.isHidden()
    assert card.status_badge.isHidden()
    assert not card.status_dot.isHidden()
    assert card.headerInfoView.isHidden()
    assert card.headerView.height() == 48
    _dispose(area)


def test_fork_keeps_branch_pipelines_independent_without_merge():
    fork = WorkflowFork()
    fork.add_card(_OperationCard(_HistoryOperation("A1")), fork.branches[0])
    fork.add_card(_OperationCard(_HistoryOperation("A2")), fork.branches[0])
    fork.add_card(_OperationCard(_HistoryOperation("B1")), fork.branches[1])
    fork.set_dataset([Atoms("H")])

    fork.run()
    assert _wait_until(lambda: fork.run_outcome == "succeeded")

    assert fork.result_dataset == []
    assert fork.branch_results["A"][0].info["history"] == ["A1", "A2"]
    assert fork.branch_results["B"][0].info["history"] == ["B1"]
    assert len(fork.available_output_cards()) == 2
    _dispose(fork)


def test_explicit_merge_concatenates_in_stable_branch_order():
    fork = WorkflowFork()
    fork.add_card(_OperationCard(_HistoryOperation("A")), fork.branches[0])
    fork.add_card(_OperationCard(_HistoryOperation("B")), fork.branches[1])
    fork.merge_checkbox.setChecked(True)
    fork.set_dataset([Atoms("H")])

    fork.run()
    assert _wait_until(lambda: fork.run_outcome == "succeeded")

    assert [item.info["history"] for item in fork.result_dataset] == [["A"], ["B"]]
    assert fork.available_output_cards() == [fork]
    _dispose(fork)


def test_unmerged_fork_preserves_successful_branch_when_another_fails():
    fork = WorkflowFork()
    fork.add_card(_OperationCard(_FailOperation()), fork.branches[0])
    fork.add_card(_OperationCard(_HistoryOperation("B")), fork.branches[1])
    fork.set_dataset([Atoms("H")])

    with patch(
        "NepTrainKit.ui.widgets.card_widget.MessageManager.send_error_message"
    ):
        fork.run()
        assert _wait_until(lambda: fork.run_outcome == "partial_failed")

    assert fork.branch_results["A"] == []
    assert fork.branch_results["B"][0].info["history"] == ["B"]
    assert fork.available_output_cards() == [fork.branches[1].cards[-1]]
    _dispose(fork)


def test_fork_json_round_trip_preserves_branches_cards_and_merge():
    fork = WorkflowFork()
    fork.branches[0].name_label.setText("Surface")
    fork.branches[0].spec.name = "Surface"
    fork.add_card(PerturbCard(), fork.branches[0])
    fork.add_card(CellStrainCard(), fork.branches[1])
    fork.merge_checkbox.setChecked(True)

    payload = fork.to_dict()
    restored = WorkflowFork()
    restored.from_dict(payload)

    assert payload["class"] == "WorkflowFork"
    assert restored.merge_enabled is True
    assert [branch.spec.name for branch in restored.branches] == ["Surface", "Branch B"]
    assert [card.__class__.__name__ for card in restored.branches[0].cards] == ["PerturbCard"]
    assert [card.__class__.__name__ for card in restored.branches[1].cards] == ["CellStrainCard"]
    _dispose(fork)
    _dispose(restored)


def test_page_rejects_shared_downstream_after_unmerged_fork():
    page = MakeDataWidget()
    fork = page.add_card("WorkflowFork")
    downstream = page.add_card("PerturbCard")
    assert fork is not None and downstream is not None
    page.dataset = [Atoms("H")]

    with patch(
        "NepTrainKit.ui.pages.makedata.MessageManager.send_info_message"
    ) as message:
        page.run_card()

    assert fork.run_outcome == "idle"
    assert downstream.run_outcome == "idle"
    assert "without Merge" in message.call_args.args[0]
    _dispose(page)
