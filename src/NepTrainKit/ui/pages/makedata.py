#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/20 17:18
# @email    : 1747193328@qq.com
import json
import os.path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QWidget, QGridLayout, QApplication
from ase import Atoms, Atom
from qfluentwidgets import HyperlinkLabel, BodyLabel, SubtitleLabel

from NepTrainKit.core import MessageManager, CardManager
from NepTrainKit.config import Config
from NepTrainKit.ui.widgets import MakeWorkflowArea

from NepTrainKit.ui.views import ConsoleWidget


from NepTrainKit.version import __version__
from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.threads import LoadingThread
from NepTrainKit.paths import get_user_config_path
from ase.io import read as ase_read



class MakeDataWidget(QWidget):
    """Provide the workflow editor for assembling NEP training datasets.

    Parameters
    ----------
    parent : QWidget | None
        Optional owner widget that embeds this page.
    """

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
        self.init_action()
        self.init_ui()
        self.dataset=None


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
                if (file_path.endswith(".xyz") or
                    file_path.endswith(".vasp") or
                    file_path.endswith(".cif")):
                    structures_path.append(file_path)

                elif file_path.endswith(".json"):
                    self.parse_card_config(file_path)
                else:
                    MessageManager.send_info_message("Only .xyz .vasp .cif or json files are supported for import.")
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
        if hasattr(self._parent,"save_menu"):
            self._parent.save_menu.addAction(self.export_card_config_action)  # pyright:ignore

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
        if hasattr(self._parent,"save_menu"):
            self._parent.save_menu.removeAction(self.export_card_config_action)   # pyright:ignore

    def init_action(self):
        """Create persistent actions shared with the main window.

        Returns
        -------
        None
            QAction instances are stored on the widget.
        """
        self.export_card_config_action = QAction(QIcon(r":/images/src/images/save.svg"), "Export Card Config")
        self.export_card_config_action.triggered.connect(self.export_card_config)
        self.load_card_config_action = QAction(QIcon(r":/images/src/images/open.svg"), "Import Card Config")
        self.load_card_config_action.triggered.connect(self.load_card_config)

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

        self.path_label = HyperlinkLabel(self)
        self.path_label.setFixedHeight(30)
        user_config_path = get_user_config_path()
        self.path_label.setText("Folder for Custom Cards  ")

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
                MessageManager.send_warning_message(f"load structure failed:{path}")
                continue
            for atom in atoms:
                if isinstance(atom, Atom):
                    continue

                if 'config_type' in atom.info:
                    atom.info["Config_type"]=atom.info["config_type"]
                    del atom.info["config_type"]



                if isinstance(atom.info.get("Config_type"),np.ndarray):
                    if atom.info["Config_type"].size==0:

                        atom.info["Config_type"] = Config.get("widget", "default_config_type", "neptrainkit")
                    else:
                        atom.info["Config_type"]=" ".join(atom.info["Config_type"])

                else:
                    atom.info["Config_type"]=str(atom.info.get("Config_type", Config.get("widget", "default_config_type", "neptrainkit")))

                structures_list.append(atom)
        if len(structures_list)==0:
            return
        self.dataset=structures_list
        MessageManager.send_success_message(f"success load {len(structures_list)} structures.")
        self.dataset_info_label.setText(f" Success load {len(structures_list)} structures.")

    def open_file(self):
        """Open a file dialog and load selected structures.

        Returns
        -------
        None
            Selected files are passed to ``load_base_structure``.
        """
        path = call_path_dialog(self,"Please choose the structure files",
                                      "selects",file_filter="Structure Files (*.xyz *.vasp *.cif)")

        if path:
            self.load_base_structure(path)

    def _export_file(self,path):
        """Write selected card datasets to the given path.

        Parameters
        ----------
        path : str
            Destination file path.

        Returns
        -------
        None
            Serialises the active cards into a single dataset file.
        """
        if os.path.exists(path):
            os.remove(path)
        with open(path, "w",encoding="utf8") as file:
            for card in self.workspace_card_widget.cards:
                if card.check_state:

                    card.write_result_dataset(file,append=True)


    def export_file(self):
        """Export selected card outputs to a dataset file asynchronously.

        Returns
        -------
        None
            Starts a background job to write the dataset.
        """

        path = call_path_dialog(self, "Choose a file save location", "file",default_filename="make_dataset.xyz")
        if path:
            thread = LoadingThread(self, show_tip=True, title="Exporting data")
            thread.start_work(self._export_file, path)

    def run_card(self):
        """Run the next enabled card using the currently loaded dataset.

        Returns
        -------
        None
            Starts the card execution chain or reports missing data.
        """
        if not  self.dataset  :
            MessageManager.send_info_message("Please import the structure file first. You can drag it in directly or import it from the upper left corner!")
            return
        self.stop_run_card()
        first_card=self._next_card(-1)
        if first_card:
            first_card.dataset = self.dataset

            first_card.runFinishedSignal.connect(self._run_next_card)
            first_card.run()
        else:
            MessageManager.send_info_message("No card selected. Please select a card in the workspace.")

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

        next_card=self._next_card(current_card_index )
        if current_card.result_dataset and next_card:
            next_card.set_dataset(current_card.result_dataset)

            next_card.runFinishedSignal.connect(self._run_next_card)
            next_card.run()
        else:
            MessageManager.send_success_message("Perturbation training set created successfully.")

    def stop_run_card(self):
        """Stop all running cards and disconnect scheduling hooks.

        Returns
        -------
        None
            Ensures no card continues executing in the background.
        """
        for card in self.workspace_card_widget.cards:
            try:
                card.runFinishedSignal.disconnect(self._run_next_card)
            except:
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
            MessageManager.send_warning_message("no card")
            return None
        card=CardManager.card_info_dict[card_name](self)
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
            MessageManager.send_warning_message("No cards in workspace.")

            return

        path = call_path_dialog(self, "Choose a file save location", "file", default_filename="card_config.json")
        if path:
            config={}
            config["software_version"]=__version__
            config["cards"]=[]
            for card in cards:
                config["cards"].append(card.to_dict())


            with open(path, "w",encoding="utf-8") as file:
                json.dump(config, file, indent=4,ensure_ascii=False)
            MessageManager.send_success_message("Card configuration exported successfully.")

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
        except Exception:
            MessageManager.send_warning_message("Invalid card configuration file.")
            return
        self.workspace_card_widget.clear_cards()
        cards=config.get("cards")
        for card in cards:
            name=card.get("class")
            card_widget=self.add_card(name)
            if card_widget is not None:
                card_widget.from_dict(card)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    from NepTrainKit.core import Config
    Config()

    window = MakeDataWidget()
    window.resize( 800,600)
    window.show()
    sys.exit(app.exec())
