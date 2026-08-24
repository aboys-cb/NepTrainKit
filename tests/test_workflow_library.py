from __future__ import annotations

import json

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QStyleOptionViewItem, QWidget

from NepTrainKit.core.workflow_library import WorkflowLibrary
from NepTrainKit.ui.pages.makedata import MakeDataWidget
from NepTrainKit.ui.widgets.workflow_library import WorkflowLibraryPanel


def _payload():
    return {
        "software_version": "test",
        "workflow_schema": 2,
        "cards": [
            {
                "class": "PerturbCard",
                "check_state": True,
                "params": {"max_distance": 0.3},
                "dataset": ["must not persist"],
                "result_dataset": ["must not persist"],
                "run_outcome": "success",
            }
        ],
    }


def test_library_crud_and_runtime_data_exclusion(tmp_path):
    library = WorkflowLibrary(tmp_path)
    saved = library.save("Perturbation", _payload())

    assert saved.card_count == 1
    card = saved.workflow["cards"][0]
    assert card["params"] == {"max_distance": 0.3}
    assert "dataset" not in card
    assert "result_dataset" not in card
    assert "run_outcome" not in card

    renamed = library.rename(saved.workflow_id, "workflow", "Perturbation v2")
    assert renamed.name == "Perturbation v2"
    duplicate = library.duplicate(
        saved.workflow_id,
        "workflow",
        name="Reusable perturbation",
        target_kind="template",
    )
    assert duplicate.kind == "template"
    assert [entry.name for entry in library.list("template")] == [
        "Reusable perturbation"
    ]

    library.delete(saved.workflow_id, "workflow")
    assert library.list("workflow") == []


def test_library_import_and_export_keep_portable_workflow_record(tmp_path):
    library = WorkflowLibrary(tmp_path / "library")
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    imported = library.import_file(source, name="Imported")
    destination = tmp_path / "exported.json"
    library.export_file(imported.workflow_id, "workflow", destination)

    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record["name"] == "Imported"
    assert record["workflow"]["cards"][0]["class"] == "PerturbCard"
    assert "dataset" not in record["workflow"]["cards"][0]


def test_make_data_workbench_opens_saved_workflow_and_template_as_copy(tmp_path):
    app = QApplication.instance() or QApplication([])
    library = WorkflowLibrary(tmp_path)
    widget = MakeDataWidget(workflow_library=library)
    widget.add_card("PerturbCard")
    card = widget.workspace_card_widget.cards[0]
    card.dataset = ["runtime input"]
    card.result_dataset = ["runtime output"]

    saved = library.save("Saved perturbation", widget._current_card_config_payload())
    template = library.save(
        "Perturbation template",
        widget._current_card_config_payload(),
        kind="template",
    )
    widget._refresh_workflow_library()

    assert widget.workspace_card_widget.library_panel.workflow_list.count() == 1
    assert widget.workspace_card_widget.library_panel.template_list.count() == 1

    widget._load_library_entry(saved)
    restored = widget.workspace_card_widget.cards[0]
    assert widget._active_workflow_id == saved.workflow_id
    assert widget._workflow_dirty is False
    assert restored.dataset is None
    assert restored.result_dataset == []
    restored.scaling_condition_frame.set_input_value([0.4])
    assert widget._workflow_dirty is True
    assert "0.4" in restored.summary_label.text()

    widget._load_library_entry(template)
    assert widget._active_workflow_id is None
    assert widget._workflow_dirty is True
    assert widget._active_workflow_name == "New from Perturbation template"

    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_workflow_list_delegate_supports_fluent_hover_lifecycle(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = MakeDataWidget(workflow_library=WorkflowLibrary(tmp_path))
    workflow_list = widget.workspace_card_widget.library_panel.workflow_list

    workflow_list._setHoverRow(0)
    QApplication.sendEvent(workflow_list, QEvent(QEvent.Type.Leave))

    assert workflow_list.delegate.hoverRow == -1
    panel = widget.workspace_card_widget.library_panel
    assert panel.current_label.font().pixelSize() == 13
    assert all(item.font().pixelSize() == 14 for item in panel.pivot.items.values())
    assert all(item.height() == 36 for item in panel.pivot.items.values())
    assert (
        workflow_list.itemDelegate().sizeHint(QStyleOptionViewItem(), None).height()
        == 56
    )
    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_workflow_actions_live_in_current_workflow_panel():
    app = QApplication.instance() or QApplication([])
    panel = WorkflowLibraryPanel()
    copy_spy = QSignalSpy(panel.copyRequested)
    paste_spy = QSignalSpy(panel.pasteRequested)
    new_spy = QSignalSpy(panel.newRequested)

    assert not hasattr(panel, "new_button")
    assert not panel.copy_button.isEnabled()
    assert not panel.new_shortcut_action.isEnabled()
    panel.paste_button.click()
    assert paste_spy.count() == 1

    panel.set_current("Draft", dirty=True, has_cards=True)
    panel.copy_button.click()
    panel.new_shortcut_action.trigger()

    assert copy_spy.count() == 1
    assert new_spy.count() == 1
    assert panel.copy_button.toolTip() == "Copy workflow JSON"
    assert panel.paste_button.toolTip() == "Add cards from clipboard"
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_make_data_status_bar_spans_the_full_workbench(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = MakeDataWidget(workflow_library=WorkflowLibrary(tmp_path))
    widget.resize(1200, 720)
    widget.show()
    app.processEvents()

    workspace = widget.workspace_card_widget
    status_bar = workspace.findChild(QWidget, "makeDataStatusBar")
    assert status_bar is not None
    assert status_bar.parent() is workspace
    assert workspace.main_layout.indexOf(status_bar) > workspace.main_layout.indexOf(
        workspace.splitter
    )
    assert status_bar.geometry().left() == 0
    assert status_bar.width() == workspace.width()

    widget.close()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
