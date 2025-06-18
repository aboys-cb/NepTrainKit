#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/4/6 13:27
# @Author  : 兵
# @email    : 1747193328@qq.com
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/12/20 17:18
# @Author  : 兵
# @email    : 1747193328@qq.com

from PySide6.QtWidgets import QWidget, QApplication, QScrollArea

from NepTrainKit.custom_widget import FlowLayout, MakeDataCardWidget



class MakeWorkflowArea(QScrollArea):
    """
微扰训练集制作
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setObjectName("MakeWorkflowArea")
        self.setWidgetResizable(True)

        self.setAcceptDrops(True)

        self.init_ui()
    @property
    def cards(self):
        return [item.widget() for item in self.flow_layout.itemList]
    def dragEnterEvent(self, event):


        if isinstance(event.source(), (MakeDataCardWidget)):
            event.acceptProposedAction()
        else:
            event.ignore()  # 忽略其他类型的拖拽

    def dropEvent(self, event):

        if isinstance(event.source(), (MakeDataCardWidget)):


            dragged_widget = event.source()
            drag_start_index = self.flow_layout.findWidgetAt(dragged_widget)[0]
            drop_pos = event.position().toPoint()
            drop_index, _ = self.flow_layout.findItemAt(drop_pos)

            if drop_index == -1:
                drop_index = self.flow_layout.count() - 1

            drop_index = min(max(0, drop_index), self.flow_layout.count())

            if drag_start_index==-1:
                self.flow_layout.insertWidget(drop_index,dragged_widget)
            else:
                if drag_start_index != drop_index:
                    self.flow_layout.moveItem(drag_start_index, drop_index)

            self.flow_layout.update()


            event.acceptProposedAction()




    def init_ui(self):


        self.container = QWidget(self)
        self.flow_layout = FlowLayout(self.container)
        self.container.setLayout(self.flow_layout)
        self.setWidget(self.container)



    def add_card(self, card):
        self.flow_layout.addWidget(card)


    def clear_cards(self):
        for item in self.cards:
            item.close()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    window = MakeWorkflowArea()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())