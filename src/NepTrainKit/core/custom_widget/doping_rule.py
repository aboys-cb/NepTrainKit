#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Widget to edit doping rules in a user friendly way."""

from __future__ import annotations

import json

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
    SpinBox,
    DoubleSpinBox,
    FluentIcon,
)


class DopingRuleItem(QFrame):
    """Single doping rule widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)

        self.target_edit = QLineEdit(self)
        self.dopants_edit = QLineEdit(self)
        self.concentration_spin = DoubleSpinBox(self)
        self.concentration_spin.setDecimals(3)
        self.concentration_spin.setRange(0.0, 1.0)
        self.concentration_spin.setSingleStep(0.1)

        self.count_spin = SpinBox(self)
        self.count_spin.setRange(0, 9999)
        self.indices_edit = QLineEdit(self)
        self.delete_button = TransparentToolButton(QIcon(":/images/src/images/delete.svg"), self)
        self.delete_button.clicked.connect(self._delete_self)

        self.layout.addWidget(BodyLabel("Target", self), 0, 0)
        self.layout.addWidget(self.target_edit, 0, 1)
        self.layout.addWidget(BodyLabel("Dopants", self), 0, 2)
        self.layout.addWidget(self.dopants_edit, 0, 3)
        self.layout.addWidget(BodyLabel("Concentration", self), 1, 0)
        self.layout.addWidget(self.concentration_spin, 1, 1)
        self.layout.addWidget(BodyLabel("Count", self), 1, 2)
        self.layout.addWidget(self.count_spin, 1, 3)
        self.layout.addWidget(BodyLabel("Indices", self), 2, 0)
        self.layout.addWidget(self.indices_edit, 2, 1, 1, 2)
        self.layout.addWidget(self.delete_button, 2, 3)

    def _delete_self(self) -> None:
        self.setParent(None)
        self.deleteLater()

    def to_rule(self) -> dict:
        rule: dict[str, object] = {}
        target = self.target_edit.text().strip()
        if target:
            rule["target"] = target
        try:
            dopants = json.loads(self.dopants_edit.text()) if self.dopants_edit.text() else {}
            if isinstance(dopants, dict) and dopants:
                rule["dopants"] = dopants
        except Exception:
            pass
        if self.concentration_spin.value() > 0:
            rule["concentration"] = self.concentration_spin.value()
        if self.count_spin.value() > 0:
            rule["count"] = self.count_spin.value()
        indices_text = self.indices_edit.text().strip()
        if indices_text:
            try:
                idx = [int(i.strip()) for i in indices_text.split(",") if i.strip()]
                rule["indices"] = idx
            except Exception:
                pass
        return rule

    def from_rule(self, rule: dict) -> None:
        self.target_edit.setText(str(rule.get("target", "")))
        dopants = rule.get("dopants")
        if dopants is not None:
            self.dopants_edit.setText(json.dumps(dopants))
        if "concentration" in rule:
            self.concentration_spin.setValue(float(rule["concentration"]))
        if "count" in rule:
            self.count_spin.setValue(int(rule["count"]))
        if "indices" in rule:
            self.indices_edit.setText(",".join(str(i) for i in rule["indices"]))


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
