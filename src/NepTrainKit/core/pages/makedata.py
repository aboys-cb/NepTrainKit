#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2024/12/20 17:18
# @Author  : 兵
# @email    : 1747193328@qq.com
import json
import os.path

import numpy as np
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QWidget, QGridLayout, QApplication

from NepTrainKit.core import MessageManager
from NepTrainKit.core.custom_widget import MakeWorkflowArea, CardGroup

from NepTrainKit.core.views.cards import   ConsoleWidget,card_info_dict


from NepTrainKit.version import __version__
from NepTrainKit import utils, module_path
from ase.io import read as ase_read



class MakeDataWidget(QWidget):
    """
微扰训练集制作
    """

    def __init__(self,parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setObjectName("MakeDataWidget")
        self.setAcceptDrops(True)
        self.nep_result_data=None
        self.init_action()
        self.init_ui()
        self.dataset=None


    def dragEnterEvent(self, event):

        # 检查拖拽的内容是否包含文件
        if event.mimeData().hasUrls():
            event.acceptProposedAction()  # 接受拖拽事件

        else:
            event.ignore()  # 忽略其他类型的拖拽

    def dropEvent(self, event):
        # 获取拖拽的文件路径
        # print("dropEvent",event)
        urls = event.mimeData().urls()

        if urls:
            # 获取第一个文件路径
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
        if hasattr(self._parent,"load_menu"):
            self._parent.load_menu.addAction(self.load_card_config_action)
        if hasattr(self._parent,"save_menu"):
            self._parent.save_menu.addAction(self.export_card_config_action)

    def hideEvent(self, event):
        if hasattr(self._parent,"load_menu"):
            self._parent.load_menu.removeAction(self.load_card_config_action)
        if hasattr(self._parent,"save_menu"):
            self._parent.save_menu.removeAction(self.export_card_config_action)

    def init_action(self):
        self.export_card_config_action = QAction(QIcon(r":/images/src/images/save.svg"), "Export Card Config")
        self.export_card_config_action.triggered.connect(self.export_card_config)
        self.load_card_config_action = QAction(QIcon(r":/images/src/images/open.svg"), "Import Card Config")
        self.load_card_config_action.triggered.connect(self.load_card_config)
    def init_ui(self):

        self.gridLayout = QGridLayout(self)
        self.gridLayout.setObjectName("make_data_gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.workspace_card_widget = MakeWorkflowArea(self)
        self.setting_group=ConsoleWidget(self)
        self.setting_group.runSignal.connect(self.run_card)
        self.setting_group.stopSignal.connect(self.stop_run_card)
        self.setting_group.newCardSignal.connect(self.add_card)
        self.gridLayout.addWidget(self.setting_group, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.workspace_card_widget, 1, 0, 1, 1)
        self.setLayout(self.gridLayout)

    def load_base_structure(self,paths):

        structures_list = []
        for path  in paths:
            atoms = ase_read(path,":")
            #ase有时候会将字符串解析成数组或者int  这里转换成str
            for atom in atoms:
                if isinstance(atom.info.get("Config_type"),np.ndarray):
                    if atom.info["Config_type"].size==0:

                        atom.info["Config_type"] = ""
                    else:
                        atom.info["Config_type"]=" ".join(atom.info["Config_type"])

                else:
                    atom.info["Config_type"]=str(atom.info.get("Config_type",""))
                structures_list.append(atom)

        self.dataset=structures_list
        MessageManager.send_success_message(f"success load base structure ({len(structures_list)} atoms).")

    def open_file(self):
        path = utils.call_path_dialog(self,"Please choose the structure files",
                                      "selects",file_filter="Structure Files (*.xyz;*.vasp;*.cif)")

        if path:
            self.load_base_structure(path)
    def _export_file(self,path):
        if os.path.exists(path):
            os.remove(path)
        with open(path, "w") as file:
            for card in self.workspace_card_widget.cards:
                if card.check_state:

                    card.write_result_dataset(file,append=True)


    def export_file(self):


        path = utils.call_path_dialog(self, "Choose a file save location", "file",default_filename="make_dataset.xyz")
        if path:
            thread = utils.LoadingThread(self, show_tip=True, title="Exporting data")
            thread.start_work(self._export_file, path)
    def run_card(self):
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
        for card in self.workspace_card_widget.cards:
            card.runFinishedSignal.disconnect(self._run_next_card)
            card.stop()

    def add_card(self,card_name):


        if card_name not in card_info_dict:
            MessageManager.send_warning_message("no card")
            return
        card=card_info_dict[card_name](self)
        self.workspace_card_widget.add_card(card)
        return card

    def export_card_config(self):
        cards=self.workspace_card_widget.cards
        if not cards:
            MessageManager.send_warning_message("No cards in workspace.")

            return

        path = utils.call_path_dialog(self, "Choose a file save location", "file", default_filename="card_config.json")
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
        path = utils.call_path_dialog(self, "Choose a card configuration file", "select" )
        if path:

            self.parse_card_config(path)
    def parse_card_config(self,path):
            try:
                with open(path, "r",encoding="utf-8") as file:
                    config = json.load(file)
            except:
                MessageManager.send_warning_message("Invalid card configuration file.")
                return
            self.workspace_card_widget.clear_cards()
            cards=config.get("cards")
            for card in cards:
                name=card.get("class")
                card_widget=self.add_card(name)
                card_widget.from_dict(card)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)

    window = MakeDataWidget()
    window.resize( 800,600)
    window.show()
    sys.exit(app.exec_())