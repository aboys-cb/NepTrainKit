#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Widget to edit doping rules in a user friendly way."""

from __future__ import annotations

import json
import traceback

from loguru import logger
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)
from qfluentwidgets import (
    BodyLabel,
    TransparentToolButton,
    FluentIcon,
    LineEdit,
    RadioButton,
    ToolTipFilter,
    ToolTipPosition,
)
from .input import SpinBoxUnitInputFrame


class DopingRuleItem(QFrame):
    """Single doping rule widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.setStyleSheet("background-color: rgb(239, 249, 254);")
        self.target_edit = QLineEdit(self)
        self.target_edit.setPlaceholderText("Cs")

        self.setFixedSize(300, 130)
        self.dopants_edit = QLineEdit(self)
         
        self.concentration_frame = SpinBoxUnitInputFrame(self)
        self.concentration_frame.set_input(["-", ""], 2, "float")
        self.concentration_frame.setRange(0, 1)
        self.concentration_frame.set_input_value([1.0, 1.0])

        self.concentration_botton = RadioButton("Conc", self)
        self.concentration_botton.setChecked(True)
        self.count_frame = SpinBoxUnitInputFrame(self)
        self.count_frame.set_input(["-", ""], 2, "int")
        self.count_frame.setRange(0, 10000)
        self.count_frame.set_input_value([10, 10])
        self.count_botton = RadioButton("Count", self)

        self.indices_edit = QLineEdit(self)
        self.delete_button = TransparentToolButton(QIcon(":/images/src/images/delete.svg"), self)
        self.delete_button.clicked.connect(self._delete_self)

        self.target_label = BodyLabel("Target", self)
        self.target_label.setToolTip("Element to replace,eg:Cs")
        self.target_label.installEventFilter(ToolTipFilter(self.target_label, 300, ToolTipPosition.TOP))
        self.layout.addWidget(self.target_label, 0, 0)
        self.layout.addWidget(self.target_edit, 0, 1)
        self.group_label = BodyLabel("Group", self)
        self.group_label.setToolTip("Optional group name")
        self.group_label.installEventFilter(ToolTipFilter(self.group_label, 300, ToolTipPosition.TOP))
        self.layout.addWidget(self.group_label, 0, 2)
        self.layout.addWidget(self.indices_edit, 0, 3, 1,1)

        self.dopants_label = BodyLabel("Dopants", self)
        self.dopants_label.setToolTip("Dopant elements and ratio,eg:Cs:0.6,Na:0.4")
        self.dopants_label.installEventFilter(ToolTipFilter(self.dopants_label, 300, ToolTipPosition.TOP))
        self.layout.addWidget(self.dopants_label, 1, 0)
        self.layout.addWidget(self.dopants_edit, 1, 1, 1, 3)
        self.concentration_botton.setToolTip("Use concentration")
        self.concentration_botton.installEventFilter(ToolTipFilter(self.concentration_botton, 300, ToolTipPosition.TOP))
        self.layout.addWidget(self.concentration_botton, 2, 0)
        self.layout.addWidget(self.concentration_frame, 2, 1,1,4)
        self.count_botton.setToolTip("Use count")
        self.count_botton.installEventFilter(ToolTipFilter(self.count_botton, 300, ToolTipPosition.TOP))
        self.layout.addWidget(self.count_botton, 3, 0)
        self.layout.addWidget(self.count_frame, 3, 1,1,4)

        self.delete_button.setToolTip("Delete rule")
        self.delete_button.installEventFilter(ToolTipFilter(self.delete_button, 300, ToolTipPosition.TOP))
        self.layout.addWidget(self.delete_button, 0, 4, 3, 1)

    def _delete_self(self) -> None:
        self.setParent(None)
        self.deleteLater()

    def to_rule(self) -> dict:
        rule: dict[str, object] = {}
        target = self.target_edit.text().strip()
        if target:
            rule["target"] = target
        try:
            dopant_text = self.dopants_edit.text().strip()
            if dopant_text.startswith("{") and dopant_text.endswith("}"):


                dopants = json.loads(self.dopants_edit.text()) if self.dopants_edit.text() else {}
                if isinstance(dopants, dict) and dopants:
                    rule["dopants"] = dopants
            else:
                #按照txt解析
                #要求格式 Cs:0.6,Na:0.4
                #或者 Cs:6,Na:4
                dopant_list = dopant_text.split(",")

                rule["dopants"] = {dopant.split(":")[0]:float(dopant.split(":")[1]) for dopant in dopant_list }
        except Exception:
            logger.error(traceback.format_exc())

        rule["concentration"] = [float(v) for v in self.concentration_frame.get_input_value()]

        rule["count"] = [int(v) for v in self.count_frame.get_input_value()]
        rule["use"] = "concentration" if self.concentration_botton.isChecked() else "count"
        indices_text = self.indices_edit.text().strip()
        if indices_text:
            try:
                idx = [i.strip() for i in indices_text.split(",") if i.strip()]
                rule["group"] = idx
            except Exception:
                pass
        return rule

    def from_rule(self, rule: dict) -> None:
        if not rule:
            return 
        self.target_edit.setText(str(rule.get("target", "")))
        dopants = rule.get("dopants")
        if dopants is not None:
            self.dopants_edit.setText(json.dumps(dopants))
        if "concentration" in rule:
            self.concentration_frame.set_input_value(rule["concentration"])
        if "count" in rule:
            self.count_frame.set_input_value(rule["count"])
        if "group" in rule:
            self.indices_edit.setText(",".join(str(i) for i in rule["group"]))
        if "use" in  rule:
            self.concentration_botton.setChecked(rule["use"]=="concentration")
            self.count_botton.setChecked(rule["use"]=="count")

class DopingRulesWidget(QWidget):
    """Container widget for multiple doping rules."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self.add_button = TransparentToolButton(FluentIcon.ADD, self)
        self.add_button.clicked.connect(self.add_rule)
        self.add_button.setToolTip("Add rule")
        self.add_button.installEventFilter(ToolTipFilter(self.add_button, 300, ToolTipPosition.TOP))
        btn_layout.addWidget(self.add_button, 0, Qt.AlignLeft)
        btn_layout.addStretch(1)
        self.layout.addLayout(btn_layout)

        self.rule_container = QWidget(self)
        self.rule_layout = QVBoxLayout(self.rule_container)
        self.rule_layout.setContentsMargins(0, 0, 0, 0)
        self.rule_layout.setSpacing(4)
        self.layout.addWidget(self.rule_container)

    def add_rule(self, rule: dict | None = None) -> DopingRuleItem:
        item = DopingRuleItem(self.rule_container)
        self.rule_layout.addWidget(item)
        if rule:
            item.from_rule(rule)
        return item

    def to_rules(self) -> list[dict]:
        rules: list[dict] = []
        for i in range(self.rule_layout.count()):
            widget = self.rule_layout.itemAt(i).widget()
            if isinstance(widget, DopingRuleItem):
                rule = widget.to_rule()
                if rule:
                    rules.append(rule)
        return rules

    def from_rules(self, rules: list[dict]) -> None:
        while self.rule_layout.count():
            item = self.rule_layout.takeAt(0).widget()
            if item is not None:
                item.deleteLater()
        for rule in rules or []:
            self.add_rule(rule)
