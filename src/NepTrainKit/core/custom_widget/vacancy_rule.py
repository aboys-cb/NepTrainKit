#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Widget to edit vacancy rules."""

from __future__ import annotations

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
    ToolTipFilter,
    ToolTipPosition,
)
from .input import SpinBoxUnitInputFrame


class VacancyRuleItem(QFrame):
    """Single vacancy rule widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.setStyleSheet("background-color: rgb(239, 249, 254);")

        self.element_edit = QLineEdit(self)
        self.element_edit.setPlaceholderText("Cs")
        self.group_edit = QLineEdit(self)
        self.count_frame = SpinBoxUnitInputFrame(self)
        self.count_frame.set_input(["-", ""], 2, "int")
        self.count_frame.setRange(0, 10000)
        self.count_frame.set_input_value([1, 1])

        self.delete_button = TransparentToolButton(QIcon(":/images/src/images/delete.svg"), self)
        self.delete_button.clicked.connect(self._delete_self)

        self.element_label = BodyLabel("Element", self)
        self.element_label.setToolTip("Element to remove")
        self.element_label.installEventFilter(ToolTipFilter(self.element_label, 300, ToolTipPosition.TOP))
        self.group_label = BodyLabel("Group", self)
        self.group_label.setToolTip("Optional group name")
        self.group_label.installEventFilter(ToolTipFilter(self.group_label, 300, ToolTipPosition.TOP))
        self.count_label = BodyLabel("Count", self)
        self.count_label.setToolTip("Number of atoms to remove")
        self.count_label.installEventFilter(ToolTipFilter(self.count_label, 300, ToolTipPosition.TOP))

        self.layout.addWidget(self.element_label, 0, 0)
        self.layout.addWidget(self.element_edit, 0, 1)
        self.layout.addWidget(self.group_label, 0, 2)
        self.layout.addWidget(self.group_edit, 0, 3)
        self.layout.addWidget(self.count_label, 1, 0)
        self.layout.addWidget(self.count_frame, 1, 1)
        self.layout.addWidget(self.delete_button, 0, 4, 2, 1)

    def _delete_self(self) -> None:
        self.setParent(None)
        self.deleteLater()

    def to_rule(self) -> dict:
        rule: dict[str, object] = {}
        element = self.element_edit.text().strip()
        if element:
            rule["element"] = element
        rule["count"] = [int(v) for v in self.count_frame.get_input_value()]
        groups = self.group_edit.text().strip()
        if groups:
            rule["group"] = [g.strip() for g in groups.split(",") if g.strip()]
        return rule

    def from_rule(self, rule: dict) -> None:
        if not rule:
            return
        self.element_edit.setText(str(rule.get("element", "")))
        if "count" in rule:
            self.count_frame.set_input_value(rule["count"])
        if "group" in rule:
            self.group_edit.setText(",".join(str(i) for i in rule["group"]))


class VacancyRulesWidget(QWidget):
    """Container widget for multiple vacancy rules."""

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

    def add_rule(self, rule: dict | None = None) -> VacancyRuleItem:
        item = VacancyRuleItem(self.rule_container)
        self.rule_layout.addWidget(item)
        if rule:
            item.from_rule(rule)
        return item

    def to_rules(self) -> list[dict]:
        rules: list[dict] = []
        for i in range(self.rule_layout.count()):
            widget = self.rule_layout.itemAt(i).widget()
            if isinstance(widget, VacancyRuleItem):
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
