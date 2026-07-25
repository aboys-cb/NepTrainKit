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
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)
from ase import Atoms, Atom
from qfluentwidgets import FluentIcon, HyperlinkLabel, BodyLabel, SubtitleLabel

from NepTrainKit.core import MessageManager, CardManager
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

    def __init__(self,parent=None):
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
        self.setting_group=ConsoleWidget(self)
        self.setting_group.runSignal.connect(self.run_card)
        self.setting_group.stopSignal.connect(self.stop_run_card)
        self.setting_group.newCardSignal.connect(self.add_card)
        self.setting_group.pasteSignal.connect(self.paste_card_config_from_clipboard)
        self.setting_group.copySignal.connect(self.copy_card_config_to_clipboard)
        self.setting_group.viewOutputSignal.connect(self.request_selected_outputs)

        self.path_label = HyperlinkLabel(self)
        self.path_label.setFixedHeight(30)
        user_config_path = get_user_config_path()
        self.path_label.setText(self.tr("Folder for Custom Cards"))

        self.path_label.setUrl(f"file:///{user_config_path}/cards")

        self.dataset_info_label = BodyLabel(self)
        self.dataset_info_label.setFixedHeight(30)

        self.gridLayout.addWidget(self.setting_group, 0, 0, 1, 2)
        self.gridLayout.addWidget(self.workspace_card_widget, 1, 0, 1, 2)
        self.gridLayout.addWidget(self.dataset_info_label, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.path_label, 2, 1, 1, 1,alignment=Qt.AlignmentFlag.AlignRight)
        self.setLayout(self.gridLayout)

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
        MessageManager.send_success_message(
            self.tr("success load {count} structures.").format(count=len(structures_list))
        )
        self.dataset_info_label.setText(
            self.tr("Success load {count} structures.").format(count=len(structures_list))
        )

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
            final_output = getattr(final_card, "result_dataset", None) or []
            return [final_card] if len(final_output) > 0 else []
        return [
            card
            for card in enabled_cards
            if len(getattr(card, "result_dataset", None) or []) > 0
        ]

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
        filter_card = getattr(card, "filter_card", None)
        if filter_card is not None:
            self._connect_card_output_actions(filter_card)

    def request_card_output(self, card) -> None:
        """Send one card's current output to the main-window handoff."""
        outputs = list(getattr(card, "result_dataset", None) or [])
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

    def _add_card_configs(self, cards):
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
                self._connect_card_output_actions(card_widget)
                added_count += 1
        if added_count:
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
