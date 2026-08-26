#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/20 17:18
# @email    : 1747193328@qq.com
import json
import os.path
import re
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QSizePolicy,
    QTextEdit,
    QWidget,
)
from ase import Atoms, Atom
from qfluentwidgets import FluentIcon, HyperlinkLabel, BodyLabel, SubtitleLabel

from NepTrainKit.core import MessageManager, CardManager
from NepTrainKit.core.workflow_library import WorkflowEntry, WorkflowLibrary
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.config import Config
from NepTrainKit.ui.widgets import MakeWorkflowArea

from NepTrainKit.ui.views import ConsoleWidget


from NepTrainKit.version import __version__
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.threads import BackgroundTask
from NepTrainKit.paths import get_user_config_path
from ase.io import read as ase_read


MAKE_DATA_STRUCTURE_FILE_FILTER = (
    "Structure files (*.xyz *.extxyz *.vasp *.cif POSCAR CONTCAR);;All files (*)"
)
_MAKE_DATA_STRUCTURE_SUFFIXES = {".xyz", ".extxyz", ".vasp", ".cif"}
_MAKE_DATA_STRUCTURE_NAMES = {"POSCAR", "CONTCAR"}


def is_make_data_structure_path(path: str | os.PathLike[str]) -> bool:
    """Return whether ``path`` is a structure file accepted by Make Dataset."""
    candidate = Path(path)
    return (
        candidate.suffix.lower() in _MAKE_DATA_STRUCTURE_SUFFIXES
        or candidate.name.upper() in _MAKE_DATA_STRUCTURE_NAMES
    )



class MakeDataWidget(QWidget):
    """Provide the workflow editor for assembling NEP training datasets.

    Parameters
    ----------
    parent : QWidget | None
        Optional owner widget that embeds this page.
    """

    finalOutputRequestedSignal = Signal(list)

    def __init__(self, parent=None, workflow_library: WorkflowLibrary | None = None):
        """Initialise the workflow editor and runtime state.

        Parameters
        ----------
        parent : QWidget | None
            Optional owner widget that embeds this page.
        """
        super().__init__(parent)
        self._parent = parent
        self.setObjectName("MakeDataWidget")
        self.setAcceptDrops(True)
        self.nep_result_data=None
        self._last_completed_card_index = None
        self._clipboard_shortcut_filter_installed = False
        self.workflow_library = workflow_library or WorkflowLibrary()
        self._active_workflow_id: str | None = None
        self._active_workflow_name: str | None = None
        self._workflow_dirty = False
        self.init_action()
        self.init_ui()
        self.dataset=None

    def eventFilter(self, watched, event):
        """Route Ctrl+V on the Make Dataset workspace to card JSON paste."""
        if (
            event.type() == QEvent.Type.KeyPress
            and self.isVisible()
            and QApplication.activeWindow() is self.window()
            and event.matches(QKeySequence.StandardKey.Paste)
            and self._focus_allows_card_json_paste()
        ):
            self.paste_card_config_from_clipboard()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _focus_allows_card_json_paste(self) -> bool:
        """Return True when Ctrl+V should create cards instead of editing text."""
        focus_widget = QApplication.focusWidget()
        editable_widgets = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)
        return not isinstance(focus_widget, editable_widgets)

    def _install_clipboard_shortcut_filter(self):
        """Install the application-level filter once while this page is visible."""
        if self._clipboard_shortcut_filter_installed:
            return
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._clipboard_shortcut_filter_installed = True

    def _remove_clipboard_shortcut_filter(self):
        """Remove the application-level Ctrl+V filter when leaving this page."""
        if not self._clipboard_shortcut_filter_installed:
            return
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._clipboard_shortcut_filter_installed = False


    def dragEnterEvent(self, event):
        """Accept drag events that contain supported file URLs.

        Parameters
        ----------
        event : QDragEnterEvent
            Drag event forwarded by Qt.

        Returns
        -------
        None
            The handler updates the event acceptance state.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Import dropped structure or card configuration files.

        Parameters
        ----------
        event : QDropEvent
            Drop event containing file URLs.

        Returns
        -------
        None
            Imported files are dispatched to the workflow widgets.
        """
        urls = event.mimeData().urls()

        if urls:
            structures_path = []
            for url in urls:
                file_path = url.toLocalFile()
                if is_make_data_structure_path(file_path):
                    structures_path.append(file_path)

                elif file_path.endswith(".json"):
                    self.parse_card_config(file_path)
                else:
                    MessageManager.send_info_message(
                        self.tr(
                            "Only .xyz, .extxyz, .vasp, .cif, POSCAR, CONTCAR, or JSON files are supported for import."
                        )
                    )
            if structures_path:
                self.load_base_structure(structures_path)

        # event.accept()

    def showEvent(self, event):
        """Attach menu actions when the widget becomes visible.

        Parameters
        ----------
        event : QShowEvent
            Show event forwarded by Qt.

        Returns
        -------
        None
            Menu actions are registered on the parent window.
        """
        if hasattr(self._parent,"load_menu"):
            self._parent.load_menu.addAction(self.load_card_config_action)  # pyright:ignore
            self._parent.load_menu.addAction(self.paste_card_config_action)  # pyright:ignore
        if hasattr(self._parent,"save_menu"):
            self._parent.save_menu.addAction(self.export_final_output_action)  # pyright:ignore
            self._parent.save_menu.addAction(self.export_all_outputs_action)  # pyright:ignore
            self._parent.save_menu.addAction(self.export_card_config_action)  # pyright:ignore
        self._install_clipboard_shortcut_filter()

    def hideEvent(self, event):
        """Detach menu actions when the widget is hidden.

        Parameters
        ----------
        event : QHideEvent
            Hide event forwarded by Qt.

        Returns
        -------
        None
            Menu actions are removed from the parent window.
        """
        if hasattr(self._parent,"load_menu"):
            self._parent.load_menu.removeAction(self.load_card_config_action)  # pyright:ignore
            self._parent.load_menu.removeAction(self.paste_card_config_action)  # pyright:ignore
        if hasattr(self._parent,"save_menu"):
            self._parent.save_menu.removeAction(self.export_final_output_action)  # pyright:ignore
            self._parent.save_menu.removeAction(self.export_all_outputs_action)  # pyright:ignore
            self._parent.save_menu.removeAction(self.export_card_config_action)   # pyright:ignore
        self._remove_clipboard_shortcut_filter()

    def init_action(self):
        """Create persistent actions shared with the main window.

        Returns
        -------
        None
            QAction instances are stored on the widget.
        """
        self.export_final_output_action = QAction(
            QIcon(r":/images/src/images/save.svg"),
            self.tr("Export final workflow output"),
        )
        self.export_final_output_action.triggered.connect(self.export_file)
        self.export_all_outputs_action = QAction(
            QIcon(r":/images/src/images/save.svg"),
            self.tr("Export all available card outputs"),
        )
        self.export_all_outputs_action.triggered.connect(self.export_all_outputs)
        self.export_card_config_action = QAction(
            QIcon(r":/images/src/images/save.svg"),
            self.tr("Export Card Config"),
        )
        self.export_card_config_action.triggered.connect(self.export_card_config)
        self.load_card_config_action = QAction(
            QIcon(r":/images/src/images/open.svg"),
            self.tr("Import Card Config"),
        )
        self.load_card_config_action.triggered.connect(self.load_card_config)
        self.paste_card_config_action = QAction(
            FluentIcon.PASTE.icon(),
            self.tr("Paste Card JSON"),
        )
        self.paste_card_config_action.triggered.connect(self.paste_card_config_from_clipboard)

    def init_ui(self):
        """Build the workflow canvas, console, and status widgets.

        Returns
        -------
        None
            All child widgets are created and added to the layout.
        """

        self.gridLayout = QGridLayout(self)
        self.gridLayout.setObjectName("make_data_gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.workspace_card_widget = MakeWorkflowArea(self)
        self._connect_workflow_library()
        self.workspace_card_widget.workflowChanged.connect(self._mark_workflow_dirty)
        self.workspace_card_widget.workflowChanged.connect(self._refresh_input_count_previews)
        self.setting_group=ConsoleWidget(self)
        self.setting_group.runSignal.connect(self.run_card)
        self.setting_group.stopSignal.connect(self.stop_run_card)
        self.setting_group.newCardSignal.connect(self.add_card)
        self.setting_group.viewOutputSignal.connect(self.request_selected_outputs)
        self.workspace_card_widget.set_command_bar(self.setting_group)

        self.path_label = HyperlinkLabel(self)
        self.path_label.setFixedHeight(30)
        user_config_path = get_user_config_path()
        self.path_label.setText(self.tr("Folder for Custom Cards"))

        self.path_label.setUrl(f"file:///{user_config_path}/cards")

        self.dataset_info_label = BodyLabel(self)
        self.dataset_info_label.setFixedHeight(30)

        status_bar = QWidget(self)
        status_bar.setObjectName("makeDataStatusBar")
        status_bar.setFixedHeight(30)
        status_bar.setStyleSheet(
            "QWidget#makeDataStatusBar {"
            "border-top: 1px solid rgba(100,120,128,38);"
            "background: rgba(255,255,255,232); }"
        )
        status_bar_layout = QHBoxLayout(status_bar)
        status_bar_layout.setContentsMargins(10, 0, 10, 0)
        status_bar_layout.setSpacing(8)
        status_bar_layout.addWidget(self.dataset_info_label, 1)
        status_bar_layout.addWidget(
            self.path_label, 0, Qt.AlignmentFlag.AlignRight
        )
        self.workspace_card_widget.set_status_bar(status_bar)

        self.gridLayout.addWidget(self.workspace_card_widget, 0, 0)
        self.gridLayout.setRowStretch(0, 1)
        self.workspace_card_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setLayout(self.gridLayout)

    def _connect_workflow_library(self) -> None:
        panel = self.workspace_card_widget.library_panel
        panel.newRequested.connect(self.new_workflow)
        panel.copyRequested.connect(self.copy_card_config_to_clipboard)
        panel.pasteRequested.connect(self.paste_card_config_from_clipboard)
        panel.saveRequested.connect(self.save_workflow)
        panel.saveAsRequested.connect(self.save_workflow_as)
        panel.openRequested.connect(self.open_library_workflow)
        panel.renameRequested.connect(self.rename_library_workflow)
        panel.duplicateRequested.connect(self.duplicate_library_workflow)
        panel.deleteRequested.connect(self.delete_library_workflow)
        panel.importRequested.connect(self.import_library_workflow)
        panel.exportRequested.connect(self.export_library_workflow)
        self._refresh_workflow_library()

    def _refresh_workflow_library(self) -> None:
        self.workspace_card_widget.library_panel.set_entries(
            self.workflow_library.list("workflow"),
            self.workflow_library.list("template"),
        )
        self.workspace_card_widget.library_panel.set_current(
            self._active_workflow_name,
            dirty=self._workflow_dirty,
            workflow_id=self._active_workflow_id,
            has_cards=bool(self.workspace_card_widget.cards),
        )

    def _set_active_workflow(
        self,
        entry: WorkflowEntry | None,
        *,
        dirty: bool,
        display_name: str | None = None,
    ) -> None:
        self._active_workflow_id = (
            entry.workflow_id if entry is not None and entry.kind == "workflow" else None
        )
        self._active_workflow_name = display_name or (
            entry.name if entry is not None and entry.kind == "workflow" else None
        )
        self._workflow_dirty = dirty
        self._refresh_workflow_library()

    def _mark_workflow_dirty(self) -> None:
        if self._workflow_dirty:
            self.workspace_card_widget.library_panel.set_current(
                self._active_workflow_name,
                dirty=True,
                workflow_id=self._active_workflow_id,
                has_cards=bool(self.workspace_card_widget.cards),
            )
            return
        self._workflow_dirty = True
        self.workspace_card_widget.library_panel.set_current(
            self._active_workflow_name,
            dirty=True,
            workflow_id=self._active_workflow_id,
            has_cards=bool(self.workspace_card_widget.cards),
        )

    def _track_card_parameter_changes(self, card) -> None:
        """Mark named workflows dirty when an editable card control changes."""
        widgets = [card, *card.findChildren(QWidget)]
        editor = getattr(card, "setting_widget", None)
        if editor is not None and editor not in widgets:
            widgets.extend([editor, *editor.findChildren(QWidget)])
        for widget in widgets:
            if bool(widget.property("workflowDirtyTracked")):
                continue
            connected = False
            for signal_name in (
                "toggled",
                "textChanged",
                "valueChanged",
                "currentIndexChanged",
            ):
                signal = getattr(widget, signal_name, None)
                if signal is None or not hasattr(signal, "connect"):
                    continue
                try:
                    signal.connect(
                        lambda *_args, selected=card: self._on_card_parameter_changed(
                            selected
                        )
                    )
                    connected = True
                except (RuntimeError, TypeError):
                    continue
            if connected:
                widget.setProperty("workflowDirtyTracked", True)

    def _on_card_parameter_changed(self, card) -> None:
        """Refresh the compact canvas readout and mark its workflow dirty."""
        refresh = getattr(card, "refresh_compact_presentation", None)
        if callable(refresh):
            refresh()
        self._mark_workflow_dirty()

    def _confirm_discard_workflow_changes(self) -> bool:
        if not self._workflow_dirty or not self.workspace_card_widget.cards:
            return True
        answer = QMessageBox.question(
            self,
            self.tr("Unsaved workflow"),
            self.tr("Discard the unsaved workflow changes?"),
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def new_workflow(self) -> None:
        if not self._confirm_discard_workflow_changes():
            return
        self.workspace_card_widget.clear_cards()
        self._set_active_workflow(None, dirty=False)

    def save_workflow(self) -> None:
        if self._active_workflow_id is None:
            self.save_workflow_as("workflow")
            return
        try:
            entry = self.workflow_library.save(
                self._active_workflow_name or self.tr("Untitled workflow"),
                self._current_card_config_payload(),
                workflow_id=self._active_workflow_id,
            )
        except (OSError, ValueError) as exc:
            MessageManager.send_error_message(str(exc))
            return
        self._set_active_workflow(entry, dirty=False)
        MessageManager.send_success_message(self.tr("Workflow saved."))

    def save_workflow_as(self, kind: str) -> None:
        default_name = self._active_workflow_name or self.tr("Untitled workflow")
        name, accepted = QInputDialog.getText(
            self,
            self.tr("Save workflow"),
            self.tr("Workflow name"),
            text=default_name,
        )
        if not accepted or not name.strip():
            return
        try:
            entry = self.workflow_library.save(
                name,
                self._current_card_config_payload(),
                kind=kind,
            )
        except (OSError, ValueError) as exc:
            MessageManager.send_error_message(str(exc))
            return
        if kind == "workflow":
            self._set_active_workflow(entry, dirty=False)
        else:
            self._refresh_workflow_library()
        MessageManager.send_success_message(
            self.tr("Workflow template saved.")
            if kind == "template"
            else self.tr("Workflow saved.")
        )

    def _load_library_entry(self, entry: WorkflowEntry) -> None:
        cards = self._normalise_card_config_payload(entry.workflow)
        self.workspace_card_widget.clear_cards()
        self._add_card_configs(cards, notify=False)
        if entry.kind == "template":
            self._set_active_workflow(
                None,
                dirty=True,
                display_name=self.tr("New from {name}").format(name=entry.name),
            )
        else:
            self._set_active_workflow(entry, dirty=False)

    def open_library_workflow(self, workflow_id: str, kind: str) -> None:
        if not self._confirm_discard_workflow_changes():
            return
        try:
            self._load_library_entry(self.workflow_library.get(workflow_id, kind))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            MessageManager.send_error_message(str(exc))

    def rename_library_workflow(self, workflow_id: str, kind: str) -> None:
        try:
            entry = self.workflow_library.get(workflow_id, kind)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            MessageManager.send_error_message(str(exc))
            return
        name, accepted = QInputDialog.getText(
            self,
            self.tr("Rename workflow"),
            self.tr("Workflow name"),
            text=entry.name,
        )
        if not accepted or not name.strip():
            return
        try:
            renamed = self.workflow_library.rename(workflow_id, kind, name)
        except (OSError, ValueError) as exc:
            MessageManager.send_error_message(str(exc))
            return
        if workflow_id == self._active_workflow_id and kind == "workflow":
            self._active_workflow_name = renamed.name
        self._refresh_workflow_library()

    def duplicate_library_workflow(self, workflow_id: str, kind: str) -> None:
        try:
            entry = self.workflow_library.get(workflow_id, kind)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            MessageManager.send_error_message(str(exc))
            return
        name, accepted = QInputDialog.getText(
            self,
            self.tr("Duplicate workflow"),
            self.tr("Workflow name"),
            text=self.tr("{name} copy").format(name=entry.name),
        )
        if not accepted or not name.strip():
            return
        try:
            self.workflow_library.duplicate(workflow_id, kind, name=name)
        except (OSError, ValueError) as exc:
            MessageManager.send_error_message(str(exc))
            return
        self._refresh_workflow_library()

    def delete_library_workflow(self, workflow_id: str, kind: str) -> None:
        answer = QMessageBox.question(
            self,
            self.tr("Delete workflow"),
            self.tr("Delete this saved workflow? This cannot be undone."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.workflow_library.delete(workflow_id, kind)
        except (OSError, ValueError) as exc:
            MessageManager.send_error_message(str(exc))
            return
        if workflow_id == self._active_workflow_id and kind == "workflow":
            self._active_workflow_id = None
            self._workflow_dirty = True
        self._refresh_workflow_library()

    def import_library_workflow(self, kind: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Import workflow"),
            str(Config.get_path()),
            self.tr("Workflow JSON (*.json)"),
        )
        if not path:
            return
        try:
            self.workflow_library.import_file(Path(path), kind=kind)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            MessageManager.send_error_message(str(exc))
            return
        self._refresh_workflow_library()

    def export_library_workflow(self, workflow_id: str, kind: str) -> None:
        try:
            entry = self.workflow_library.get(workflow_id, kind)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            MessageManager.send_error_message(str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Export workflow"),
            f"{entry.name}.json",
            self.tr("Workflow JSON (*.json)"),
        )
        if not path:
            return
        try:
            self.workflow_library.export_file(workflow_id, kind, Path(path))
        except (OSError, ValueError) as exc:
            MessageManager.send_error_message(str(exc))

    def load_base_structure(self,paths):
        """Load structures from disk and normalise metadata fields.

        Parameters
        ----------
        paths : Sequence[str]
            File paths for structure batches readable by ASE.

        Returns
        -------
        None
            Populates ``self.dataset`` and updates dataset statistics.
        """

        structures_list = []
        for path  in paths:
            try:
                atoms  = ase_read(path,":")
            except:
                MessageManager.send_warning_message(
                    self.tr("Load structure failed: {path}").format(path=path)
                )
                continue
            for atom in atoms:
                if isinstance(atom, Atom):
                    continue

                if 'config_type' in atom.info:
                    atom.info["Config_type"]=atom.info["config_type"]
                    del atom.info["config_type"]

                default_cfg = Config.get("widget", "default_config_type", "neptrainkit")
                raw_cfg = atom.info.get("Config_type", default_cfg)
                if isinstance(raw_cfg, np.ndarray):
                    tokens = [str(x) for x in raw_cfg.tolist() if str(x).strip()]
                elif isinstance(raw_cfg, (list, tuple, set)):
                    tokens = [str(x) for x in raw_cfg if str(x).strip()]
                else:
                    tokens = [t for t in re.split(r"[|\\s]+", str(raw_cfg).strip()) if t]

                atom.info["Config_type"] = ""
                for t in tokens:
                    append_config_tag(atom, t)
                if not atom.info.get("Config_type"):
                    atom.info["Config_type"] = default_cfg

                structures_list.append(atom)
        if len(structures_list)==0:
            return
        self.dataset=structures_list
        self._refresh_input_count_previews()
        MessageManager.send_success_message(
            self.tr("success load {count} structures.").format(count=len(structures_list))
        )
        self.dataset_info_label.setText(
            self.tr("Success load {count} structures.").format(count=len(structures_list))
        )

    def _refresh_input_count_previews(self) -> None:
        """Expose the imported count only where the first enabled card receives it exactly."""
        exact_target_found = False
        selected = getattr(self.workspace_card_widget.guidance_panel, "_card", None)
        for card in self.workspace_card_widget.cards:
            setter = getattr(card, "set_preview_input_count", None)
            if not callable(setter):
                if card.check_state:
                    exact_target_found = True
                continue
            exact_count = None
            if not exact_target_found and card.check_state:
                if bool(getattr(card, "requires_input_dataset", True)):
                    exact_count = len(self.dataset or [])
                exact_target_found = True
            setter(exact_count)
        if selected is not None:
            self.workspace_card_widget.guidance_panel._refresh_context()

    def open_file(self):
        """Open a file dialog and load selected structures.

        Returns
        -------
        None
            Selected files are passed to ``load_base_structure``.
        """
        path = call_path_dialog(
            self,
            self.tr("Please choose the structure files"),
            "selects",
            file_filter=self.tr(
                "Structure files (*.xyz *.extxyz *.vasp *.cif POSCAR CONTCAR);;All files (*)"
            ),
        )

        if path:
            self.load_base_structure(path)

    def _cards_for_export(self, include_all: bool):
        """Return available outputs for the requested workflow export scope."""
        enabled_cards = [
            card
            for card in self.workspace_card_widget.cards
            if card.check_state
        ]
        if not enabled_cards:
            return []
        if not include_all:
            if hasattr(self, "_last_completed_card_index"):
                completed_index = self._last_completed_card_index
                if completed_index is None:
                    return []
                cards = self.workspace_card_widget.cards
                if not 0 <= completed_index < len(cards):
                    return []
                final_card = cards[completed_index]
                if not final_card.check_state:
                    return []
            else:
                final_card = enabled_cards[-1]
            return MakeDataWidget._available_output_cards(final_card)
        outputs = []
        for card in enabled_cards:
            outputs.extend(MakeDataWidget._available_output_cards(card))
        return outputs

    @staticmethod
    def _available_output_cards(card):
        """Return concrete output-owning cards for a card or structural node."""
        getter = getattr(card, "available_output_cards", None)
        if callable(getter):
            return list(getter())
        output = getattr(card, "result_dataset", None) or []
        return [card] if output else []

    def _export_file(self, path, cards):
        """Write the explicitly selected card outputs to the given path.

        Parameters
        ----------
        path : str
            Destination file path.

        Returns
        -------
        None
            Serialises the provided card outputs into a single dataset file.
        """
        destination = Path(path)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                for index, card in enumerate(cards):
                    card.write_result_dataset(file, append=index > 0)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


    def export_file(self):
        """Export only the final available workflow output asynchronously.

        Returns
        -------
        None
            Starts a background job to write the dataset.
        """

        self._start_export(include_all=False)

    def export_all_outputs(self):
        """Export every available enabled card output for diagnostic use."""
        self._start_export(include_all=True)

    def _start_export(self, include_all: bool) -> None:
        cards = self._cards_for_export(include_all)
        if not cards:
            if not any(
                card.check_state for card in self.workspace_card_widget.cards
            ):
                message = self.tr("No enabled cards to export.")
            elif include_all:
                message = self.tr(
                    "No enabled card has output. Run the workflow first."
                )
            else:
                message = self.tr(
                    "The final enabled card has no output. Run the workflow first."
                )
            MessageManager.send_info_message(message)
            return
        if not include_all and len(cards) > 1:
            MessageManager.send_info_message(
                self.tr(
                    "The workflow has multiple independent branch outputs. Export each branch separately or insert an explicit Merge."
                )
            )
            return
        path = call_path_dialog(
            self,
            self.tr("Choose a file save location"),
            "file",
            default_filename="make_dataset.xyz",
        )
        if path:
            thread = BackgroundTask(
                self,
                show_tip=True,
                title=self.tr("Exporting data"),
            )
            thread.start_work(self._export_file, path, cards)

    def run_card(self):
        """Run the next enabled card using the currently loaded dataset.

        Returns
        -------
        None
            Starts the card execution chain or reports missing data.
        """
        if self._has_running_card():
            MessageManager.send_info_message(
                self.tr("Cards are still running. Please wait for the current run to finish.")
            )
            return
        self.stop_run_card()
        self._last_completed_card_index = None
        self.setting_group.set_output_available(False)
        for card in self.workspace_card_widget.cards:
            card.set_dataset([])
        for index, card in enumerate(self.workspace_card_widget.cards):
            if (
                card.__class__.__name__ == "WorkflowFork"
                and not bool(getattr(card, "merge_enabled", False))
                and any(
                    later.check_state
                    for later in self.workspace_card_widget.cards[index + 1 :]
                )
            ):
                MessageManager.send_info_message(
                    self.tr(
                        "A permanent fork without Merge must be the final workflow node. Insert an explicit Merge before adding a shared downstream card."
                    )
                )
                return
        first_card=self._next_card(-1)
        if first_card:
            needs_input = bool(getattr(first_card, "requires_input_dataset", True))
            if not self.dataset and needs_input:
                MessageManager.send_info_message(
                    self.tr("Please import the structure file first. You can drag it in directly or import it from the upper left corner!")
                )
                return
            first_card.set_dataset(self.dataset or [])

            first_card.runFinishedSignal.connect(self._run_next_card)
            first_card.run()
        else:
            MessageManager.send_info_message(
                self.tr("No card selected. Please select a card in the workspace.")
            )

    def _has_running_card(self):
        """Return True when any workspace card still owns a running worker."""
        return any(self._card_is_running(card) for card in self.workspace_card_widget.cards)

    def _card_is_running(self, card):
        worker = getattr(card, "worker_thread", None)
        if worker is not None and worker.isRunning():
            return True
        for child in getattr(card, "card_list", []):
            if self._card_is_running(child):
                return True
        for branch in getattr(card, "branches", []):
            for child in getattr(branch, "cards", []):
                if self._card_is_running(child):
                    return True
        filter_card = getattr(card, "filter_card", None)
        return bool(filter_card is not None and self._card_is_running(filter_card))

    def _next_card(self,current_card_index=-1):
        """Return the next enabled card after the given index.

        Parameters
        ----------
        current_card_index : int, default=-1
            Index of the previously executed card.

        Returns
        -------
        card_widget : MakeDataCard | None
            Next enabled card, or ``None`` when all cards are exhausted.
        """

        cards=self.workspace_card_widget.cards
        if current_card_index+1 >=len(cards):
            return None
        current_card_index+=1
        for i,card in enumerate(cards[current_card_index:]):

            if card.check_state:
                card.index=i+current_card_index
                return card
            else:
                continue
        return None

    def _run_next_card(self,current_card_index):
        """Run the next scheduled card once the current card finishes.

        Parameters
        ----------
        current_card_index : int
            Index of the card that just completed.

        Returns
        -------
        None
            Continues the execution chain until all cards finish.
        """

        cards=self.workspace_card_widget.cards
        current_card=cards[current_card_index]
        current_card.runFinishedSignal.disconnect(self._run_next_card)
        if getattr(current_card, "run_outcome", "succeeded") != "succeeded":
            self._last_completed_card_index = None
            self.setting_group.set_output_available(
                bool(self._cards_for_export(include_all=True))
            )
            return

        self._last_completed_card_index = current_card_index

        next_card=self._next_card(current_card_index )
        if current_card.result_dataset and next_card:
            next_card.set_dataset(current_card.result_dataset)

            next_card.runFinishedSignal.connect(self._run_next_card)
            next_card.run()
        else:
            self.setting_group.set_output_available(
                bool(self._cards_for_export(include_all=True))
            )
            MessageManager.send_success_message(
                self.tr("Training structures generated.")
            )

    def request_selected_outputs(self) -> None:
        """Send all checked card outputs to the main-window handoff."""
        cards = self._cards_for_export(include_all=True)
        if not cards:
            if any(card.check_state for card in self.workspace_card_widget.cards):
                message = self.tr(
                    "No checked card has output. Run the workflow first."
                )
            else:
                message = self.tr("No checked cards to view.")
            MessageManager.send_info_message(message)
            self.setting_group.set_output_available(False)
            return
        outputs = [
            structure
            for card in cards
            for structure in card.result_dataset
        ]
        self.finalOutputRequestedSignal.emit(outputs)

    def _refresh_selected_output_available(self, *_args) -> None:
        """Refresh the workflow-level view action after selection changes."""
        self.setting_group.set_output_available(
            bool(self._cards_for_export(include_all=True))
        )

    def _connect_card_output_actions(self, card) -> None:
        """Connect output actions for a card and any nested group cards."""
        if getattr(card, "_output_actions_owner", None) is not self:
            card.viewOutputSignal.connect(self.request_card_output)
            card.state_checkbox.stateChanged.connect(
                self._refresh_selected_output_available
            )
            card._output_actions_owner = self
        for child in getattr(card, "card_list", []):
            self._connect_card_output_actions(child)
        for branch in getattr(card, "branches", []):
            for child in getattr(branch, "cards", []):
                self._connect_card_output_actions(child)
        filter_card = getattr(card, "filter_card", None)
        if filter_card is not None:
            self._connect_card_output_actions(filter_card)

    def request_card_output(self, card) -> None:
        """Send one card's current output to the main-window handoff."""
        output_cards = self._available_output_cards(card)
        outputs = [
            structure
            for output_card in output_cards
            for structure in (getattr(output_card, "result_dataset", None) or [])
        ]
        if not outputs:
            card.set_output_available(False)
            MessageManager.send_info_message(
                self.tr("Run this card to create an output first.")
            )
            return
        self.finalOutputRequestedSignal.emit(outputs)

    def stop_run_card(self):
        """Stop all running cards and disconnect scheduling hooks.

        Returns
        -------
        None
            Ensures no card continues executing in the background.
        """
        import warnings

        for card in self.workspace_card_widget.cards:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    card.runFinishedSignal.disconnect(self._run_next_card)
            except Exception:
                pass
            card.stop()

    def add_card(self,card_name):
        """Instantiate and add a card widget by name.

        Parameters
        ----------
        card_name : str
            Class identifier registered in ``CardManager``.

        Returns
        -------
        card : QWidget | None
            The created card instance, or ``None`` when the name is unknown.
        """

        if card_name not in CardManager.card_info_dict:
            MessageManager.send_warning_message(self.tr("no card"))
            return None
        card=CardManager.card_info_dict[card_name](self)
        self._connect_card_output_actions(card)
        self.workspace_card_widget.add_card(card)
        self._track_card_parameter_changes(card)
        self._mark_workflow_dirty()
        return card

    def export_card_config(self):
        """Serialise the current card layout and settings to disk.

        Returns
        -------
        None
            Writes a JSON configuration file when cards exist.
        """
        cards=self.workspace_card_widget.cards
        if not cards:
            MessageManager.send_warning_message(self.tr("No cards in workspace."))

            return

        path = call_path_dialog(self, "Choose a file save location", "file", default_filename="card_config.json")
        if path:
            with open(path, "w",encoding="utf-8") as file:
                json.dump(self._current_card_config_payload(), file, indent=4, ensure_ascii=False)
            MessageManager.send_success_message(
                self.tr("Card configuration exported successfully.")
            )

    def _current_card_config_payload(self):
        """Return the current workflow card configuration payload."""
        return {
            "software_version": __version__,
            "workflow_schema": 2,
            "cards": [card.to_dict() for card in self.workspace_card_widget.cards],
        }

    def copy_card_config_to_clipboard(self):
        """Copy the current workflow card configuration JSON to the clipboard."""
        cards = self.workspace_card_widget.cards
        if not cards:
            MessageManager.send_warning_message(self.tr("No cards in workspace."))
            return
        QApplication.clipboard().setText(self.current_card_config_json())
        MessageManager.send_success_message(
            self.tr("Card configuration JSON copied to clipboard.")
        )

    def current_card_config_json(self):
        """Return the current workflow card configuration as pretty JSON text."""
        return json.dumps(self._current_card_config_payload(), indent=4, ensure_ascii=False)

    def load_card_config(self):
        """Load card configuration from a JSON file chosen by the user.

        Returns
        -------
        None
            Delegates parsing to ``parse_card_config`` when a file is picked.
        """
        path = call_path_dialog(self, "Choose a card configuration file", "select" )
        if path:

            self.parse_card_config(path)

    def paste_card_config_from_clipboard(self):
        """Append card configuration JSON from the system clipboard.

        Returns
        -------
        None
            Creates cards when the clipboard contains one card, a card list, or
            an exported workflow object with a ``cards`` field.
        """
        text = QApplication.clipboard().text().strip()
        if not text:
            MessageManager.send_warning_message(self.tr("Clipboard does not contain card JSON."))
            return
        try:
            payload = json.loads(text)
            cards = self._normalise_card_config_payload(payload)
        except ValueError as exc:
            MessageManager.send_warning_message(str(exc))
            return

        self._add_card_configs(cards)

    def parse_card_config(self,path):
        """Populate the workspace from a saved card configuration.

        Parameters
        ----------
        path : str
            Path to the JSON configuration file.

        Returns
        -------
        None
            Rebuilds the workspace cards when parsing succeeds.
        """
        try:
            with open(path, "r",encoding="utf-8") as file:
                config = json.load(file)
            cards = self._normalise_card_config_payload(config)
        except Exception as exc:
            MessageManager.send_warning_message(
                self.tr("Invalid card configuration file: {error}").format(error=exc)
            )
            return
        self.workspace_card_widget.clear_cards()
        self._add_card_configs(cards)
        self._set_active_workflow(None, dirty=True)

    def _normalise_card_config_payload(self, payload):
        """Return a validated list of card dictionaries from supported JSON shapes."""
        if isinstance(payload, dict) and "cards" in payload:
            cards = payload.get("cards")
        elif isinstance(payload, list):
            cards = payload
        elif isinstance(payload, dict):
            cards = [payload]
        else:
            raise ValueError("Card JSON must be an object, a list, or an exported workflow.")

        if not isinstance(cards, list) or not cards:
            raise ValueError("Card JSON does not contain any cards.")

        normalised_cards = []
        for card in cards:
            if not isinstance(card, dict):
                raise ValueError("Each card JSON entry must be an object.")
            name = card.get("class")
            if not isinstance(name, str) or not name:
                raise ValueError("Each card JSON entry must contain a class name.")
            if name not in CardManager.card_info_dict:
                raise ValueError(f"Unknown card class: {name}")
            card_data = dict(card)
            card_data.setdefault("check_state", True)
            normalised_cards.append(card_data)
        return normalised_cards

    def _add_card_configs(self, cards, *, notify: bool = True):
        """Create cards from validated card configuration dictionaries."""
        added_count = 0
        for card in cards:
            name=card.get("class")
            card_widget=self.add_card(name)
            if card_widget is not None:
                try:
                    card_widget.from_dict(card)
                except Exception as exc:
                    card_widget.close()
                    MessageManager.send_error_message(
                        self.tr("Failed to load {name}: {error}").format(name=name, error=exc)
                    )
                    continue
                self._track_card_parameter_changes(card_widget)
                self._connect_card_output_actions(card_widget)
                added_count += 1
        if added_count and notify:
            MessageManager.send_success_message(
                self.tr("Added {count} card configuration(s).").format(count=added_count)
            )


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    from NepTrainKit.core import Config
    Config()

    window = MakeDataWidget()
    window.resize( 800,600)
    window.show()
    sys.exit(app.exec())
