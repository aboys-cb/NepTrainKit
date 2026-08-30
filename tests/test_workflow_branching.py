from __future__ import annotations

from unittest.mock import patch

from ase import Atoms
from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QEventLoop,
    QPoint,
    QPointF,
    Qt,
    QTimer,
)
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication, QWidget
from qfluentwidgets import ScrollBarHandleDisplayMode

from NepTrainKit.core import CardManager
from NepTrainKit.core.cards.operation import StructureOperation
from NepTrainKit.ui.pages.makedata import MakeDataWidget
from NepTrainKit.ui.views._card.bain_path_card import BainPathCard
from NepTrainKit.ui.views._card.card_group import CardGroup
from NepTrainKit.ui.views._card.cell_strain_card import CellStrainCard
from NepTrainKit.ui.views._card.interstitial_adsorbate_card import InsertDefectCard
from NepTrainKit.ui.views._card.interface_layer_mix_card import InterfaceLayerMixCard
from NepTrainKit.ui.views._card.perturb_card import PerturbCard
from NepTrainKit.ui.views._card.random_doping_card import RandomDopingCard
from NepTrainKit.ui.views._card.random_slab_card import RandomSlabCard
from NepTrainKit.ui.views._card.set_magnetic_moments_card import SetMagneticMomentsCard
from NepTrainKit.ui.views._card.soc_texture_response_card import SOCTextureResponseCard
from NepTrainKit.ui.views._card.workflow_fork import WorkflowFork
from NepTrainKit.ui.widgets import (
    AdaptiveCompactDoubleSpinBox,
    AdaptiveCompactSpinBox,
    KeyValueTableInput,
    MakeDataCard,
    MakeWorkflowArea,
    SpinBoxUnitInputFrame,
)


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
    second_editor_maximum = second.setting_widget.maximumWidth()

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
    assert "Type: Lattice" in area.guidance_panel._about_text
    assert "Contributors: NepTrainKit" in area.guidance_panel._about_text
    assert area.guidance_panel.docs_button.isEnabled()
    assert area.guidance_panel.info_button.isEnabled()
    assert area.guidance_panel.copy_card_button.isEnabled()
    assert area.guidance_panel.docs_button.size().width() == 28
    assert area.guidance_panel.info_button.size().width() == 28
    with patch("NepTrainKit.ui.widgets.docker.Flyout.create") as create_flyout:
        area.guidance_panel.info_button.click()
        assert "Contributors: NepTrainKit" in create_flyout.call_args.kwargs["content"]
    assert area.guidance_panel.tabs.count() == 1
    assert area.guidance_panel.tabs.tabText(0) == "Parameters"
    assert area.guidance_panel.tabs.tabBar().isHidden()
    assert area.guidance_panel.context_widget.isVisible()
    assert area.guidance_panel.current_context_label.text() == second.get_summary_text()

    area.select_card(first, expand=True)
    app.processEvents()
    assert first.window_state == "collapse"
    assert second.window_state == "collapse"
    assert area.guidance_panel._editor_widget is first.setting_widget
    assert area.guidance_panel.title_label.text() == first.getTitle()
    assert second.setting_widget.maximumWidth() == second_editor_maximum
    first.num_condition_frame.set_input_value([3])
    app.processEvents()
    assert area.guidance_panel.current_context_label.text() == first.get_summary_text()
    area.guidance_panel.copy_card_button.click()
    assert QApplication.clipboard().text() == first.to_json_text()
    _dispose(area)


def test_inspector_context_refreshes_when_preview_dataset_changes():
    app = _app()
    area = MakeWorkflowArea()
    card = RandomSlabCard()
    area.add_card(card)
    area.resize(1024, 640)
    area.show()
    area.select_card(card)
    app.processEvents()
    assert "exact scan" in area.guidance_panel.current_context_label.text()

    structure = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True)
    card.set_preview_input_count(2)
    card.set_dataset([structure])
    app.processEvents()

    assert area.guidance_panel.current_context_label.text() == card.get_summary_text()
    assert "12/input" in area.guidance_panel.current_context_label.text()
    assert area.guidance_panel.recommend_label.text() == card.get_guidance_text()
    assert "Inputs 2 × 12" in area.guidance_panel.recommend_label.text()
    _dispose(area)


def test_inspector_shows_transient_parameter_errors_instead_of_logging_tracebacks():
    app = _app()
    area = MakeWorkflowArea()
    card = SOCTextureResponseCard()
    area.add_card(card)
    area.resize(1024, 640)
    area.show()
    area.select_card(card)
    app.processEvents()

    card.scan_input.range_frame.set_input_value([2.0, -2.0, 1.0])
    for _ in range(3):
        app.processEvents()

    assert area.guidance_panel.current_context_label.isHidden()
    assert area.guidance_panel.recommend_caption.text() == "Parameter issue"
    assert (
        area.guidance_panel.recommend_label.text()
        == "Scan maximum must be greater than or equal to the minimum."
    )

    card.scan_input.range_frame.set_input_value([-2.0, 2.0, 1.0])
    for _ in range(3):
        app.processEvents()

    assert area.guidance_panel.current_context_label.isVisibleTo(area)
    assert area.guidance_panel.recommend_caption.text() == "Recommended checks"
    _dispose(area)


def test_insert_defect_adsorption_fields_remain_visible_in_real_inspector():
    app = _app()
    area = MakeWorkflowArea()
    card = InsertDefectCard()
    area.add_card(card)
    area.resize(1024, 640)
    area.show()
    area.select_card(card)
    for _ in range(3):
        app.processEvents()

    assert area.guidance_panel._editor_widget is card.setting_widget
    assert card.axis_field.isHidden()
    assert card.offset_field.isHidden()

    card.mode_combo.setCurrentIndex(card.mode_combo.findData(1))
    for _ in range(3):
        app.processEvents()

    assert card.axis_field.isVisibleTo(area)
    assert card.offset_field.isVisibleTo(area)
    assert card.axis_combo.isVisibleTo(area)
    assert card.offset_frame.isVisibleTo(area)
    assert area.guidance_panel.parameter_scroll.horizontalScrollBar().maximum() == 0
    _dispose(area)


def test_bain_volume_scan_only_appears_for_the_shape_volume_grid():
    app = _app()
    area = MakeWorkflowArea()
    card = BainPathCard()
    area.add_card(card)
    area.resize(1024, 640)
    area.show()
    area.select_card(card)
    for _ in range(3):
        app.processEvents()

    assert area.guidance_panel._editor_widget is card.setting_widget
    assert card.volume_field.isHidden()
    assert "3 path points = 3 outputs/input" in card.preview_label.text()

    card.mode_combo.setCurrentIndex(card.mode_combo.findData("scale_volume"))
    for _ in range(3):
        app.processEvents()

    assert card.volume_field.isVisibleTo(area)
    assert card.volume_frame.isVisibleTo(area)
    assert area.guidance_panel.parameter_scroll.horizontalScrollBar().maximum() == 0
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


def test_all_builtin_parameter_cards_fit_the_narrow_inspector_without_horizontal_scroll():
    app = _app()
    area = MakeWorkflowArea()
    area.resize(1024, 900)
    area.show()
    app.processEvents()

    checked = []
    for class_name, card_cls in sorted(CardManager.card_info_dict.items()):
        metadata = CardManager.card_metadata_dict[class_name]
        if "_card" not in metadata.source_path:
            continue
        card = card_cls()
        if not hasattr(card, "setting_widget"):
            continue
        area.add_card(card)
        area.select_card(card)
        for _ in range(3):
            app.processEvents()
        assert area.guidance_panel.parameter_scroll.horizontalScrollBar().maximum() == 0, class_name
        title_width = card.headerLabel.fontMetrics().horizontalAdvance(
            card.headerLabel.text()
        )
        assert card.headerLabel.width() >= title_width, class_name
        for frame in card.setting_widget.findChildren(SpinBoxUnitInputFrame):
            if not frame.isVisibleTo(area) or len(frame.object_list) < 2:
                continue
            assert frame._column_count == len(frame.object_list), class_name
            widths = [control.width() for control in frame.object_list]
            assert max(widths) - min(widths) <= 1, class_name
        for table_editor in card.setting_widget.findChildren(KeyValueTableInput):
            if not table_editor.isVisibleTo(area):
                continue
            assert table_editor.height() + 1 >= table_editor.sizeHint().height(), class_name
            assert (
                table_editor.table.geometry().bottom()
                < table_editor.add_button.geometry().top()
            ), class_name
        for widget in card.setting_widget.findChildren(QWidget):
            if not isinstance(
                widget,
                (AdaptiveCompactSpinBox, AdaptiveCompactDoubleSpinBox),
            ):
                continue
            frame = widget.parentWidget()
            if widget.isVisibleTo(area) and not (
                isinstance(frame, SpinBoxUnitInputFrame)
                and len(frame.object_list) > 1
            ):
                assert not widget.compactSpinButton.isHidden(), class_name
        checked.append(class_name)
        area.remove_card(card)
        app.processEvents()

    assert len(checked) >= 35
    _dispose(area)


def test_parameter_inspector_scroll_handle_is_discoverable_without_hover():
    app = _app()
    area = MakeWorkflowArea()
    area.resize(1024, 640)
    area.show()
    card = PerturbCard()
    area.add_card(card)
    area.select_card(card)
    card.element_scaling_checkbox.setChecked(True)
    card._add_element_row("H", 0.1)
    app.processEvents()

    scroll = area.guidance_panel.parameter_scroll
    assert scroll.verticalScrollBar().maximum() > 0
    assert (
        scroll.scrollDelagate.vScrollBar.handleDisplayMode
        == ScrollBarHandleDisplayMode.ALWAYS
    )
    assert scroll.horizontalScrollBar().maximum() == 0
    _dispose(area)


def test_parameter_inspector_reflows_after_element_table_rows_are_added():
    app = _app()
    area = MakeWorkflowArea()
    area.resize(1024, 720)
    area.show()
    card = SetMagneticMomentsCard()
    area.add_card(card)
    area.select_card(card)
    app.processEvents()

    row_count_changes = QSignalSpy(card.map_edit.rowCountChanged)
    initial_editor_height = card.setting_widget.height()
    card.map_edit._apply_element_selection("Lu")
    assert _wait_until(
        lambda: area.guidance_panel.parameter_scroll.viewport().updatesEnabled()
    )
    card.map_edit._apply_element_selection("H")
    viewport = area.guidance_panel.parameter_scroll.viewport()
    assert not viewport.updatesEnabled()
    assert _wait_until(
        lambda: area.guidance_panel._editor_height_animation is None
        and card.setting_widget.height() > initial_editor_height
    )

    assert row_count_changes.count() == 2
    assert row_count_changes.at(0)[0] == 1
    assert row_count_changes.at(1)[0] == 2
    assert viewport.updatesEnabled()
    assert card.map_edit.height() >= card.map_edit.sizeHint().height()
    assert card.map_field.height() >= card.map_field.sizeHint().height()
    # Qt can round the hosted widget's final animated height one pixel below
    # its freshly recomputed size hint on the offscreen platform.
    assert card.setting_widget.height() + 1 >= card.setting_widget.sizeHint().height()
    assert card.map_edit.table.geometry().bottom() < card.map_edit.add_button.geometry().top()
    assert area.guidance_panel.parameter_scroll.verticalScrollBar().maximum() > 0

    expanded_height = card.setting_widget.height()
    card.map_edit.table.selectRow(1)
    QTest.mouseClick(card.map_edit.remove_button, Qt.MouseButton.LeftButton)
    assert not viewport.updatesEnabled()
    assert _wait_until(lambda: area.guidance_panel._editor_height_animation is not None)
    animation = area.guidance_panel._editor_height_animation
    animation_values = QSignalSpy(animation.valueChanged)
    assert _wait_until(lambda: area.guidance_panel._editor_height_animation is None)

    assert row_count_changes.count() == 3
    assert row_count_changes.at(2)[0] == 1
    assert viewport.updatesEnabled()
    assert animation_values.count() >= 2
    assert card.setting_widget.height() < expanded_height
    assert card.map_edit.table.geometry().bottom() < card.map_edit.add_button.geometry().top()
    _dispose(area)


def test_random_doping_rule_editor_reflows_without_clipping_controls():
    app = _app()
    area = MakeWorkflowArea()
    card = RandomDopingCard()
    card.rules_widget.from_rules(
        [
            {
                "target": "Si",
                "dopants": {"Ge": 0.7, "C": 0.3},
                "use": "atomic_percent",
                "percent": [3.0, 8.0],
                "ratio_type": "atom",
                "group": ["surface"],
            }
        ]
    )
    area.add_card(card)
    area.resize(1600, 760)
    area.show()
    area.select_card(card)
    app.processEvents()
    area.resize(1024, 640)
    app.processEvents()

    scroll = area.guidance_panel.parameter_scroll
    rule = card.rules_widget.rule_layout.itemAt(0).widget()
    assert scroll.horizontalScrollBar().maximum() == 0
    assert rule.width() <= scroll.viewport().width()
    assert rule.group_edit.isVisibleTo(area)
    assert rule.delete_button.isVisibleTo(area)
    assert rule.amount_mode_control.isVisibleTo(area)
    _dispose(area)


def test_first_perturb_card_previews_exact_imported_output_count():
    app = _app()
    widget = MakeDataWidget()
    widget.dataset = [Atoms("H"), Atoms("He")]
    card = widget.add_card("PerturbCard")
    card.num_condition_frame.set_input_value([3])
    widget._refresh_input_count_previews()
    app.processEvents()

    assert "2 × 3 = 6 outputs" in card.get_guidance_text()
    assert "2 × 3 = 6 outputs" in widget.workspace_card_widget.guidance_panel.recommend_label.text()
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_first_composition_gradient_previews_imported_structure_details():
    app = _app()
    widget = MakeDataWidget()
    structure = Atoms(
        "Ni8",
        scaled_positions=[[index / 8.0, 0.0, 0.0] for index in range(8)],
        cell=[8.0, 2.0, 2.0],
        pbc=True,
    )
    widget.dataset = [structure, structure.copy()]
    card = widget.add_card("CompositionGradientCard")
    card.bins_frame.set_input_value([20])
    card.samples_frame.set_input_value([3])
    widget._refresh_input_count_previews()
    app.processEvents()

    assert card.dataset is None
    assert "20 requested → 8 effective" in card.get_summary_text()
    assert "Inputs 2 × samples/input 3 = outputs 6" in card.get_guidance_text()
    assert "Eligible sites 8 → effective groups 8 → sites/group 1" in card.get_guidance_text()
    assert "second jump" in card.get_guidance_text()
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_first_interface_layer_mix_previews_imported_structure_and_output_count():
    app = _app()
    widget = MakeDataWidget()
    positions = []
    symbols = []
    for z, element in [
        (0.1, "Al"),
        (0.2, "Al"),
        (0.3, "Al"),
        (0.7, "Ni"),
        (0.8, "Ni"),
        (0.9, "Ni"),
    ]:
        for x in (0.0, 0.5):
            for y in (0.0, 0.5):
                positions.append([x, y, z])
                symbols.append(element)
    structure = Atoms(
        symbols,
        scaled_positions=positions,
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    widget.dataset = [structure, structure.copy()]
    card = widget.add_card("InterfaceLayerMixCard")
    card.num_structures_frame.set_input_value([3])
    widget._refresh_input_count_previews()
    app.processEvents()

    assert card.dataset is None
    assert "fractional c @ 0.500" in card.get_summary_text()
    assert "Inputs 2 × 3/input = 6 outputs" in card.get_guidance_text()
    assert "second interface" in card.get_guidance_text()
    _dispose(widget)


def test_interface_layer_mix_fields_fit_real_inspector_after_resize_roundtrip():
    app = _app()
    area = MakeWorkflowArea()
    card = InterfaceLayerMixCard()
    area.add_card(card)
    area.resize(1600, 760)
    area.show()
    area.select_card(card)
    for _ in range(3):
        app.processEvents()

    area.resize(1024, 640)
    for _ in range(3):
        app.processEvents()

    assert area.guidance_panel._editor_widget is card.setting_widget
    assert area.guidance_panel.parameter_scroll.horizontalScrollBar().maximum() == 0
    assert card.concentration_field.isVisibleTo(area)
    card.mode_combo.setCurrentIndex(card.mode_combo.findData("gradient"))
    card.position_mode_combo.setCurrentIndex(
        card.position_mode_combo.findData("manual")
    )
    card.seed_checkbox.setChecked(True)
    for _ in range(3):
        app.processEvents()
    assert card.gradient_start_field.isVisibleTo(area)
    assert card.gradient_end_field.isVisibleTo(area)
    assert card.interface_position_field.isVisibleTo(area)
    assert card.seed_field.isVisibleTo(area)
    assert area.guidance_panel.parameter_scroll.horizontalScrollBar().maximum() == 0
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


def test_fanout_merge_uses_parallel_paths_and_keeps_new_cards_in_group():
    app = _app()
    area = MakeWorkflowArea()
    group = CardGroup()
    first = PerturbCard()
    second = CellStrainCard()
    group.add_card(first)
    group.add_card(second)
    area.add_card(group)
    area.resize(1600, 820)
    area.show()
    area.select_card(group, expand=True)
    app.processEvents()
    app.processEvents()

    assert group.width() == min(
        area._FORK_MAX_WIDTH,
        area.scroll_area.viewport().width() - 72,
    )
    assert first.geometry().top() == second.geometry().top()
    assert first.geometry().right() < second.geometry().left()
    assert group.merge_frame.isVisible()
    assert "automatic merge" in area.guidance_panel.parameter_placeholder.text().lower()

    inserted = PerturbCard()
    area.add_card(inserted)
    app.processEvents()
    assert area.cards == [group]
    assert group.card_list == [first, second, inserted]
    assert area.guidance_panel._editor_widget is inserted.setting_widget
    _dispose(area)


def test_real_window_width_and_wide_restore_keep_structural_cards_stable():
    app = _app()
    area = MakeWorkflowArea()
    group = CardGroup()
    first = PerturbCard()
    second = CellStrainCard()
    group.add_card(first)
    group.add_card(second)
    area.add_card(group)

    area.resize(1550, 900)
    area.show()
    area.select_card(group, expand=True)
    for _ in range(3):
        app.processEvents()
    assert area.library_panel.isVisible()
    assert area.guidance_panel.isVisible()
    assert first.geometry().top() == second.geometry().top()

    # This is the Make Dataset page width inside the default 1200 px main window.
    area.resize(1151, 651)
    for _ in range(3):
        app.processEvents()

    assert area.library_panel.isVisible()
    assert area.guidance_panel.isVisible()
    assert area.canvas_column.width() >= 540
    assert group._grid_columns == 2
    assert first.geometry().top() == second.geometry().top()
    assert first.width() > 210
    assert second.width() > 210
    _dispose(area)


def test_three_path_group_keeps_portrait_tiles_in_one_parallel_row():
    app = _app()
    area = MakeWorkflowArea()
    group = CardGroup()
    cards = [PerturbCard(), CellStrainCard(), PerturbCard()]
    for card in cards:
        group.add_card(card)
    area.add_card(group)
    area.resize(1151, 700)
    area.show()
    area.select_card(group, expand=True)
    for _ in range(3):
        app.processEvents()

    assert area.library_panel.isVisible()
    assert group._grid_columns == 3
    assert len({card.geometry().top() for card in cards}) == 1
    assert all(card._group_tile_enabled for card in cards)
    assert all(140 <= card.width() <= 150 for card in cards)

    area.resize(1550, 900)
    for _ in range(3):
        app.processEvents()
    assert group._grid_columns == 3
    assert len({card.geometry().top() for card in cards}) == 1
    assert cards[0].geometry().right() < cards[1].geometry().left()
    assert cards[1].geometry().right() < cards[2].geometry().left()
    assert all(card.width() == 220 for card in cards)

    area.resize(1151, 700)
    for _ in range(3):
        app.processEvents()
    assert area.library_panel.isVisible()
    assert group._grid_columns == 3
    assert len({card.geometry().top() for card in cards}) == 1
    assert all(140 <= card.width() <= 150 for card in cards)
    _dispose(area)


def test_empty_fork_lane_selection_controls_where_the_next_card_is_added():
    app = _app()
    area = MakeWorkflowArea()
    fork = WorkflowFork()
    area.add_card(fork)
    area.resize(1500, 820)
    area.show()
    app.processEvents()

    target = fork.branches[1]
    area._activate_branch(fork, target)
    inserted = PerturbCard()
    area.add_card(inserted)
    app.processEvents()

    assert fork.branches[0].cards == []
    assert target.cards == [inserted]
    assert target.property("workflowBranchSelected") is False
    assert area.guidance_panel._editor_widget is inserted.setting_widget
    _dispose(area)


def test_clicking_empty_fork_lane_selects_it_for_the_next_card():
    app = _app()
    area = MakeWorkflowArea()
    fork = WorkflowFork()
    area.add_card(fork)
    area.resize(1200, 700)
    area.show()
    app.processEvents()

    target = fork.branches[1]
    QTest.mouseClick(target.empty_label, Qt.MouseButton.LeftButton)
    app.processEvents()
    inserted = PerturbCard()
    area.add_card(inserted)
    app.processEvents()

    assert fork.branches[0].cards == []
    assert target.cards == [inserted]
    _dispose(area)


def test_collapsed_fork_reopens_from_header_and_drag_hover_accepts_a_card():
    app = _app()
    area = MakeWorkflowArea()
    source = PerturbCard()
    fork = WorkflowFork()
    area.add_card(source)
    area.add_card(fork)
    area.resize(1500, 820)
    area.show()
    area.select_card(source, expand=True)
    app.processEvents()
    assert fork.window_state == "collapse"

    QTest.mouseClick(fork.headerLabel, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert fork.window_state == "expand"

    area.select_card(source, expand=True)
    assert fork.window_state == "collapse"

    class DragEvent:
        accepted = False

        def __init__(self, position):
            self._position = QPointF(position)

        def source(self):
            return source

        def position(self):
            return self._position

        def acceptProposedAction(self):
            self.accepted = True

        def ignore(self):
            self.accepted = False

    target = fork.branches[1]
    target_point = target.mapTo(fork, target.rect().center())
    drag = DragEvent(target_point)
    fork.dragEnterEvent(drag)
    app.processEvents()
    assert drag.accepted
    assert fork.window_state == "expand"

    drag._position = QPointF(target.mapTo(fork, target.rect().center()))
    fork.dropEvent(drag)
    app.processEvents()
    assert drag.accepted
    assert source not in area.cards
    assert target.cards == [source]
    _dispose(area)


def test_structural_headers_and_arrows_toggle_both_directions_manually():
    app = _app()
    area = MakeWorkflowArea()
    group = CardGroup()
    fork = WorkflowFork()
    area.add_card(group)
    area.add_card(fork)
    area.resize(1500, 900)
    area.show()
    app.processEvents()

    assert group.window_state == "collapse"
    assert fork.window_state == "expand"

    QTest.mouseClick(group.headerLabel, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert group.window_state == "expand"
    assert fork.window_state == "collapse"

    QTest.mouseClick(group.headerLabel, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert group.window_state == "collapse"

    QTest.mouseClick(fork.headerLabel, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert fork.window_state == "expand"
    assert group.window_state == "collapse"

    QTest.mouseClick(fork.headerLabel, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert fork.window_state == "collapse"

    QTest.mouseClick(group.collapse_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert group.window_state == "expand"
    QTest.mouseClick(group.collapse_button, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert group.window_state == "collapse"
    _dispose(area)


def test_group_nested_in_fork_keeps_group_context_and_manual_toggle():
    app = _app()
    area = MakeWorkflowArea()
    fork = WorkflowFork()
    area.add_card(fork)
    target = fork.branches[0]
    area._activate_branch(fork, target)

    group = CardGroup()
    area.add_card(group)
    assert target.cards == [group]

    first = PerturbCard()
    area.add_card(first)
    assert group.card_list == [first]
    assert target.cards == [group]

    second = CellStrainCard()
    area.add_card(second)
    app.processEvents()
    assert group.card_list == [first, second]
    assert target.cards == [group]
    assert fork.window_state == "expand"
    assert group.window_state == "expand"

    QTest.mouseClick(group.headerLabel, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert group.window_state == "collapse"
    assert fork.window_state == "expand"
    QTest.mouseClick(group.headerLabel, Qt.MouseButton.LeftButton)
    app.processEvents()
    assert group.window_state == "expand"

    other_branch = fork.branches[1]
    area._move_card_to_branch(group, other_branch, 0)
    app.processEvents()
    assert target.cards == []
    assert other_branch.cards == [group]
    assert group.card_list == [first, second]
    assert all(card._group_tile_enabled for card in group.card_list)

    area.select_card(first, expand=True)
    app.processEvents()
    assert fork.window_state == "expand"
    assert group.window_state == "expand"

    payload = fork.to_dict()
    restored = WorkflowFork()
    restored.from_dict(payload)
    restored_group = restored.branches[1].cards[0]
    assert isinstance(restored_group, CardGroup)
    assert [type(card) for card in restored_group.card_list] == [
        PerturbCard,
        CellStrainCard,
    ]
    assert all(card._group_tile_enabled for card in restored_group.card_list)
    _dispose(restored)
    _dispose(area)


def test_group_inside_fork_runs_as_one_branch_pipeline_stage():
    fork = WorkflowFork()
    group = CardGroup()
    group.add_card(_OperationCard(_HistoryOperation("A1")))
    group.add_card(_OperationCard(_HistoryOperation("A2")))
    fork.add_card(group, fork.branches[0])
    fork.add_card(_OperationCard(_HistoryOperation("after")), fork.branches[0])
    fork.add_card(_OperationCard(_HistoryOperation("B")), fork.branches[1])
    fork.set_dataset([Atoms("H")])

    fork.run()
    assert _wait_until(lambda: fork.run_outcome == "succeeded")

    assert [item.info["history"] for item in fork.branch_results["A"]] == [
        ["A1", "after"],
        ["A2", "after"],
    ]
    assert [item.info["history"] for item in fork.branch_results["B"]] == [["B"]]
    _dispose(fork)


def test_fork_output_mode_control_preserves_json_merge_contract():
    app = _app()
    fork = WorkflowFork()
    fork.add_card(PerturbCard(), fork.branches[0])
    fork.add_card(CellStrainCard(), fork.branches[1])

    assert fork.merge_checkbox.isHidden()
    assert fork.keep_separate_button.isChecked()
    assert fork.merge_enabled is False

    fork.merge_output_button.click()
    app.processEvents()
    payload = fork.to_dict()
    assert fork.merge_checkbox.isChecked()
    assert fork.merge_enabled is True
    assert fork.output_terminal_title.text() == "Explicit merge"
    assert payload["merge"] is True

    payload["metadata"]["card_name"] = "Permanent Fork"
    restored = WorkflowFork()
    restored.from_dict(payload)
    app.processEvents()
    assert restored.merge_enabled is True
    assert restored.merge_output_button.isChecked()
    assert [len(branch.cards) for branch in restored.branches] == [1, 1]
    _dispose(fork)
    _dispose(restored)


def test_legacy_card_group_json_loads_into_fanout_merge_without_schema_change():
    app = _app()
    source = CardGroup()
    source.add_card(PerturbCard())
    payload = source.to_dict()
    payload["metadata"]["card_name"] = "Card Group"

    restored = CardGroup()
    restored.from_dict(payload)
    app.processEvents()

    assert restored.to_dict()["class"] == "CardGroup"
    assert [card.__class__.__name__ for card in restored.card_list] == ["PerturbCard"]
    assert restored.filter_card is None
    _dispose(source)
    _dispose(restored)


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
    area.resize(1600, 700)
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
    viewport_point = area.scroll_area.viewport().mapFromGlobal(
        area.canvas.mapToGlobal(canvas_point)
    )
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
    viewport_point = area.scroll_area.viewport().mapFromGlobal(
        area.canvas.mapToGlobal(canvas_point)
    )
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


def test_guidance_panel_stays_visible_while_narrow_window_compresses_canvas():
    app = _app()
    area = MakeWorkflowArea()
    card = PerturbCard()
    area.add_card(card)
    area.resize(1000, 700)
    area.show()
    app.processEvents()
    assert area.guidance_panel.isVisible()
    assert area.guidance_panel._editor_widget is card.setting_widget

    # The page is about 728 px wide at the main window's minimum width.
    area.resize(728, 700)
    for _ in range(3):
        app.processEvents()
    assert area.library_panel.isVisible()
    assert area.guidance_panel.isVisible()
    assert area.guidance_panel._editor_widget is card.setting_widget
    assert card.setting_widget.parent() is area.guidance_panel.parameter_host
    assert area.library_panel.width() == 220
    assert area.guidance_panel.width() == 380
    assert area.canvas_column.width() < 140
    assert card.width() <= area.scroll_area.viewport().width()

    area.resize(1151, 700)
    for _ in range(3):
        app.processEvents()
    assert area.guidance_panel.isVisible()
    assert area.guidance_panel._editor_widget is card.setting_widget
    assert card.setting_widget.parent() is area.guidance_panel.parameter_host
    assert area.guidance_panel.parameter_scroll.horizontalScrollBar().maximum() == 0
    _dispose(area)


def test_guidance_panel_width_can_be_adjusted_with_right_splitter_handle():
    app = _app()
    area = MakeWorkflowArea()
    card = SetMagneticMomentsCard()
    area.add_card(card)
    area.select_card(card)
    area.resize(1200, 700)
    area.show()
    app.processEvents()

    handle = area.splitter.handle(2)
    initial_width = area.guidance_panel.width()
    initial_axis_width = card.axis_frame.width()
    assert handle.isEnabled()
    area.splitter.moveSplitter(handle.x() - 100, 2)
    for _ in range(3):
        app.processEvents()

    assert area.guidance_panel.width() > initial_width
    assert card.axis_frame.width() > initial_axis_width
    axis_widths = [control.width() for control in card.axis_frame.object_list]
    assert max(axis_widths) - min(axis_widths) <= 1
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
    area = MakeWorkflowArea()
    fork = WorkflowFork()
    area.add_card(fork)
    area.resize(1600, 900)
    area.show()
    app.processEvents()
    assert fork.connector.isVisible()

    fork.add_branch_button.click()
    assert len(fork.branches) == 3
    assert fork.branches[2].spec.branch_id == "C"
    assert fork.branches[2].spec.name == "Branch C"
    assert not fork.add_branch_button.isEnabled()

    area.resize(1280, 900)
    for _ in range(3):
        app.processEvents()
    assert fork.width() < fork._STACK_BRANCHES_BREAKPOINT
    assert not fork.connector.isVisible()
    assert fork.branches[1].geometry().top() >= fork.branches[0].geometry().bottom()

    area.resize(1024, 640)
    for _ in range(3):
        app.processEvents()
    assert fork._two_row_header
    assert fork.headerLabel.width() >= fork.headerLabel.sizeHint().width()
    _dispose(area)


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
    assert card._group_tile_enabled
    assert card.headerView.height() == 140

    area._move_card_to_branch(card, fork.branches[0], 0)
    assert card not in group.card_list
    assert fork.branches[0].cards == [card]
    assert not card._group_tile_enabled
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
    assert fork.status_badge.state() == "succeeded"
    assert fork.output_terminal_detail.text() == "A 1 + B 1 → 2 merged"
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
    assert fork.status_badge.state() == "partial"
    assert fork.status_badge.label.text() == "Partial"
    assert "1/2 branches completed" in fork.output_terminal_detail.text()
    _dispose(fork)


def test_fork_rejects_enabled_empty_branches_before_running_other_branches():
    fork = WorkflowFork()
    card = _OperationCard(_HistoryOperation("A"))
    fork.add_card(card, fork.branches[0])
    fork.set_dataset([Atoms("H")])
    completions = []
    fork.runFinishedSignal.connect(completions.append)

    with patch.object(card, "run") as card_run, patch(
        "NepTrainKit.ui.views._card.workflow_fork.MessageManager.send_error_message"
    ) as error_message:
        fork.run()

    assert fork.run_outcome == "failed"
    assert fork.status_badge.state() == "failed"
    assert fork.branches[1].output_label.text() == "Failed"
    assert completions == [fork.index]
    card_run.assert_not_called()
    error_message.assert_called_once_with(
        "Add at least one enabled card to: Branch B."
    )


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


def test_fork_branch_name_editor_updates_saved_display_name():
    fork = WorkflowFork()
    branch = fork.branches[0]

    branch.name_edit.setText("Surface path")
    branch.name_edit.editingFinished.emit()

    assert branch.spec.name == "Surface path"
    assert fork.to_dict()["branches"][0]["name"] == "Surface path"
    _dispose(fork)


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
