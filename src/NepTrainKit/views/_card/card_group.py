#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/18 13:21
# @Author  : 兵
# @email    : 1747193328@qq.com
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout
from shiboken6 import isValid

from NepTrainKit import utils
from NepTrainKit.core import CardManager
from NepTrainKit.custom_widget import MakeDataCardWidget
from NepTrainKit.custom_widget.card_widget import MakeDataCard, FilterDataCard


@CardManager.register_card
class CardGroup(MakeDataCardWidget):
    separator=True
    card_name= "Card Group"
    menu_icon=r":/images/src/images/group.svg"
    #通知下一个card执行
    runFinishedSignal=Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Card Group")
        self.setAcceptDrops(True)
        self.index=0
        self.group_widget = QWidget(self)
        # self.setStyleSheet("CardGroup{boder: 2px solid #C0C0C0;}")
        self.viewLayout.addWidget(self.group_widget)
        self.group_layout = QVBoxLayout(self.group_widget)
        self.exportSignal.connect(self.export_data)
        self.windowStateChangedSignal.connect(self.show_card_setting)
        self.filter_widget = QWidget(self)
        self.filter_layout = QVBoxLayout(self.filter_widget)
        self.vBoxLayout.addWidget(self.filter_widget)

        self.filter_card=None
        self.dataset:list=None
        self.result_dataset=[]
        self.resize(400, 200)

    def set_filter_card(self,card):

        self.filter_card=card
        self.filter_layout.addWidget(card)

    def state_changed(self, state):
        super().state_changed(state)
        for card in self.card_list:
            card.state_checkbox.setChecked(state)

    @property
    def card_list(self)->["MakeDataCard"]:

        return [self.group_layout.itemAt(i).widget() for i in range(self.group_layout.count()) ]
    def show_card_setting(self):

        for card in self.card_list:
            card.window_state = self.window_state
            card.windowStateChangedSignal.emit()
    def set_dataset(self,dataset):
        self.dataset =dataset
        self.result_dataset=[]

    def add_card(self, card):
        self.group_layout.addWidget(card)

    def remove_card(self, card):
        self.group_layout.removeWidget(card)

    def clear_cards(self):
        for card in self.card_list:
            self.group_layout.removeWidget(card)

    def closeEvent(self, event):
        for card in self.card_list:
            card.close()
        self.deleteLater()
        super().closeEvent(event)

    def dragEnterEvent(self, event):

        widget = event.source()

        if widget == self:
            return
        if isinstance(widget, (MakeDataCard,CardGroup)):
            event.acceptProposedAction()
        else:
            event.ignore()  # 忽略其他类型的拖拽

    def dropEvent(self, event):

        widget = event.source()
        if widget == self:
            return
        if isinstance(widget, FilterDataCard):
            self.set_filter_card(widget)

        elif isinstance(widget, (MakeDataCard,CardGroup)):
            self.add_card(widget)
        event.acceptProposedAction()

    def on_card_finished(self, index):
        self.run_card_num -= 1
        self.card_list[index].runFinishedSignal.disconnect(self.on_card_finished)
        self.result_dataset.extend(self.card_list[index].result_dataset)

        if self.run_card_num == 0:
            self.runFinishedSignal.emit(self.index)
            if self.filter_card and isValid(self.filter_card) and self.filter_card.check_state:
                self.filter_card.set_dataset(self.result_dataset)
                self.filter_card.run()

    def stop(self):
        for card in self.card_list:
            try:
                card.runFinishedSignal.disconnect(self._run_next_card)
            except Exception:
                pass
            card.stop()
        if self.filter_card:
            self.filter_card.stop()

    def run(self):
        """Run child cards sequentially instead of in parallel."""
        self.run_card_num = len(self.card_list)

        if not (self.check_state and self.run_card_num > 0):
            self.result_dataset = self.dataset
            self.runFinishedSignal.emit(self.index)
            return

        self.result_dataset = []

        first_card = self._next_card(-1)
        if first_card:
            first_card.set_dataset(self.dataset)
            first_card.runFinishedSignal.connect(self._run_next_card)
            first_card.run()
        else:
            self.result_dataset = self.dataset
            self.runFinishedSignal.emit(self.index)

    def _next_card(self, current_card_index=-1):
        cards = self.card_list
        if current_card_index + 1 >= len(cards):
            return None
        current_card_index += 1
        for i, card in enumerate(cards[current_card_index:]):
            if card.check_state:
                card.index = i + current_card_index
                return card
        return None

    def _run_next_card(self, current_card_index):
        cards = self.card_list
        current_card = cards[current_card_index]
        current_card.runFinishedSignal.disconnect(self._run_next_card)
        self.result_dataset.extend(current_card.result_dataset)

        next_card = self._next_card(current_card_index)
        if next_card:
            next_card.set_dataset(current_card.result_dataset)
            next_card.runFinishedSignal.connect(self._run_next_card)
            next_card.run()
        else:
            self.runFinishedSignal.emit(self.index)
            if self.filter_card and isValid(self.filter_card) and self.filter_card.check_state:
                self.filter_card.set_dataset(self.result_dataset)
                self.filter_card.run()

    def write_result_dataset(self, file,**kwargs):
        if self.filter_card and self.filter_card.check_state:
            self.filter_card.write_result_dataset(file,**kwargs)
            return

        for index,card in enumerate(self.card_list):
            if index==0:
                if "append" not in kwargs:
                    kwargs["append"] = False
            else:
                kwargs["append"] = True
            if card.check_state:
                card.write_result_dataset(file,**kwargs)

    def export_data(self):
        if self.dataset is not None:
            path = utils.call_path_dialog(self, "Choose a file save location", "file",f"export_{self.getTitle()}_structure.xyz")
            if not path:
                return
            thread=utils.LoadingThread(self,show_tip=True,title="Exporting data")
            thread.start_work(self.write_result_dataset, path)
    def to_dict(self):
        data_dict = super().to_dict()

        data_dict["card_list"]=[]

        for card in self.card_list:
            data_dict["card_list"].append(card.to_dict())
        if self.filter_card:
            data_dict["filter_card"]=self.filter_card.to_dict()
        else:
            data_dict["filter_card"]=None

        return data_dict
    def from_dict(self,data_dict):
        self.state_checkbox.setChecked(data_dict['check_state'])
        for sub_card in data_dict.get("card_list",[]):
            card_name=sub_card["class"]
            card  = CardManager.card_info_dict[card_name](self)
            self.add_card(card)
            card.from_dict(sub_card)

        if data_dict.get("filter_card"):
            card_name=data_dict["filter_card"]["class"]
            filter_card  = CardManager.card_info_dict[card_name](self)
            filter_card.from_dict(data_dict["filter_card"])
            self.set_filter_card(filter_card)
