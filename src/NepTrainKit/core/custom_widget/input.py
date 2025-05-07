#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2025/4/5 20:11
# @Author  : 兵
# @email    : 1747193328@qq.com
from PySide6.QtWidgets import QFrame, QHBoxLayout, QSpinBox, QDoubleSpinBox,QLineEdit
from qfluentwidgets import BodyLabel


class SpinBoxUnitInputFrame(QFrame):
    def __init__(self, parent=None):
        super(SpinBoxUnitInputFrame, self).__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.object_dict = []
    def set_input(self, unit_str,object_num ,input_type="int"):
        if  isinstance(unit_str,str):
            unit_str = [unit_str]*object_num
        elif isinstance(unit_str,list):
            unit_str=unit_str
        else:
            raise TypeError('unit_str must be str or list')

        if  isinstance(input_type,str):
            input_type = [input_type]*object_num
        elif isinstance(input_type,list):
            input_type=input_type
        else:
            raise TypeError('input_type must be str or list')

        for i in range(object_num):
            if input_type[i%len(unit_str)]=="int":
                input_object = QSpinBox(self)
                input_object.setButtonSymbols(QSpinBox.NoButtons)
            elif input_type[i%len(unit_str)]=="float":
                input_object = QDoubleSpinBox(self)
                input_object.setDecimals(3)
                input_object.setButtonSymbols(QSpinBox.NoButtons)
            else:
                raise TypeError('input_type must be int or float')

            input_object.setFixedHeight(25)
            self._layout.addWidget(input_object)
            self._layout.addWidget(BodyLabel(unit_str[i%len(unit_str)],self))
            self.object_dict.append(input_object)

    def setRange(self, min_value, max_value):
        for input_object in self.object_dict:
            input_object.setRange(min_value, max_value)

    def get_input_value(self):

        return [input_object.value() for input_object in self.object_dict]

    def set_input_value(self, value_list):
        for i, input_object in enumerate(self.object_dict):
            input_object.setValue(value_list[i])