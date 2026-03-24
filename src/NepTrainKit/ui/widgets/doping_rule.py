"""Widgets for configuring random doping rules."""

import json
import traceback
from typing import Any

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
    QButtonGroup,
)
from qfluentwidgets import (
    BodyLabel,
    TransparentToolButton,
    FluentIcon,
    LineEdit,
    RadioButton,
    ToolTipFilter,
    ToolTipPosition,
    PushButton,
)
from .input import SpinBoxUnitInputFrame


class DopingRuleItem(QFrame):
    """Single doping rule widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.__layout = QGridLayout(self)
        self.__layout.setContentsMargins(6, 6, 6, 6)
        self.__layout.setSpacing(6)
        self.__layout.setVerticalSpacing(6)
        self.setStyleSheet("background-color: rgb(239, 249, 254);")

        self.setFixedSize(460, 140)

        self.target_edit = QLineEdit(self)
        self.target_edit.setPlaceholderText("Cs")
        self.target_edit.setFixedWidth(90)
        self.dopants_edit = QLineEdit(self)
        self.dopants_edit.setFixedWidth(160)

        self.percent_frame = SpinBoxUnitInputFrame(self)
        self.percent_frame.set_input(["~", ""], 2, "float")
        self.percent_frame.setDecimals(6)
        self.percent_frame.setRange(0, 100)
        self.percent_frame.set_input_value([0.0, 100.0])
        self.percent_frame.setFixedWidth(180)

        self.atomic_percent_radio = RadioButton("Atomic %", self)
        self.atomic_percent_radio.setChecked(True)
        self.mass_percent_radio = RadioButton("Mass %", self)
        self.count_botton = RadioButton("Count", self)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.atomic_percent_radio)
        self.mode_group.addButton(self.mass_percent_radio)
        self.mode_group.addButton(self.count_botton)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)

        self.count_frame = SpinBoxUnitInputFrame(self)
        self.count_frame.set_input(["-", ""], 2, "int")
        self.count_frame.setRange(0, 999999)
        self.count_frame.set_input_value([10, 10])
        self.count_frame.setFixedWidth(180)

        self.ratio_type_button = PushButton("Atom Ratio", self)
        self.ratio_type_button.setCheckable(True)
        self.ratio_type_button.setChecked(True)
        self.ratio_type_button.setFixedWidth(100)
        self.ratio_type_button.setToolTip("Toggle between atom ratio and mass ratio for dopants")
        self.ratio_type_button.installEventFilter(ToolTipFilter(self.ratio_type_button, 300, ToolTipPosition.TOP))
        self.ratio_type_button.clicked.connect(self._toggle_ratio_type)

        self.indices_edit = QLineEdit(self)
        self.indices_edit.setFixedWidth(60)
        self.delete_button = TransparentToolButton(QIcon(":/images/src/images/delete.svg"), self)
        self.delete_button.setFixedSize(32, 32)
        self.delete_button.clicked.connect(self._delete_self)

        self.target_label = BodyLabel("Target", self)
        self.target_label.setToolTip("Element to replace, e.g. Cs")
        self.target_label.installEventFilter(ToolTipFilter(self.target_label, 300, ToolTipPosition.TOP))
        self.__layout.addWidget(self.target_label, 0, 0)
        self.__layout.addWidget(self.target_edit, 0, 1)

        self.group_label = BodyLabel("Group", self)
        self.group_label.setToolTip("Optional group name")
        self.group_label.installEventFilter(ToolTipFilter(self.group_label, 300, ToolTipPosition.TOP))
        self.__layout.addWidget(self.group_label, 0, 2)
        self.__layout.addWidget(self.indices_edit, 0, 3)

        self.dopants_label = BodyLabel("Dopants", self)
        self.dopants_label.setToolTip("Dopant elements and ratio, e.g. Cs:0.6,Na:0.4")
        self.dopants_label.installEventFilter(ToolTipFilter(self.dopants_label, 300, ToolTipPosition.TOP))
        self.__layout.addWidget(self.dopants_label, 1, 0)
        self.__layout.addWidget(self.dopants_edit, 1, 1, 1, 2)
        self.__layout.addWidget(self.ratio_type_button, 1, 3)

        self.mode_label = BodyLabel("Mode", self)
        self.mode_label.setToolTip("Select replacement mode: Atomic %, Mass %, or Count")
        self.mode_label.installEventFilter(ToolTipFilter(self.mode_label, 300, ToolTipPosition.TOP))
        self.__layout.addWidget(self.mode_label, 2, 0)

        self.__layout.addWidget(self.atomic_percent_radio, 2, 1)
        self.__layout.addWidget(self.mass_percent_radio, 2, 2)
        self.__layout.addWidget(self.count_botton, 2, 3)

        self.value_label = BodyLabel("Value", self)
        self.value_label.setToolTip("Set value range for replacement")
        self.value_label.installEventFilter(ToolTipFilter(self.value_label, 300, ToolTipPosition.TOP))
        self.__layout.addWidget(self.value_label, 3, 0)
        self.__layout.addWidget(self.percent_frame, 3, 1, 1, 2)
        self.count_frame.setVisible(False)
        self.__layout.addWidget(self.count_frame, 3, 1, 1, 2)

        self.delete_button.setToolTip("Delete rule")
        self.delete_button.installEventFilter(ToolTipFilter(self.delete_button, 300, ToolTipPosition.TOP))
        self.__layout.addWidget(self.delete_button, 0, 4, 4, 1)

    def _delete_self(self) -> None:
        """Detach the widget from the layout and schedule deletion."""
        self.setParent(None)
        self.deleteLater()

    def _toggle_ratio_type(self) -> None:
        if self.ratio_type_button.isChecked():
            self.ratio_type_button.setText("Mass Ratio")
        else:
            self.ratio_type_button.setText("Atom Ratio")

    def _on_mode_changed(self) -> None:
        is_count = self.count_botton.isChecked()
        self.percent_frame.setVisible(not is_count)
        self.count_frame.setVisible(is_count)

    def to_rule(self) -> dict[str, Any]:
        """Serialize the current editor state into a rule mapping.

        Returns
        -------
        dict[str, Any]
            Mapping describing the configured doping rule.
        """
        rule: dict[str, Any] = {}
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
                dopant_list = dopant_text.split(",")
                rule["dopants"] = {dopant.split(":")[0]: float(dopant.split(":")[1]) for dopant in dopant_list}
        except Exception:
            logger.error(traceback.format_exc())

        rule["percent"] = [float(v) for v in self.percent_frame.get_input_value()]
        rule["count"] = [int(v) for v in self.count_frame.get_input_value()]

        if self.count_botton.isChecked():
            rule["use"] = "count"
        elif self.mass_percent_radio.isChecked():
            rule["use"] = "mass_percent"
        else:
            rule["use"] = "atomic_percent"

        rule["ratio_type"] = "atom" if self.ratio_type_button.isChecked() else "mass"

        indices_text = self.indices_edit.text().strip()
        if indices_text:
            try:
                idx = [i.strip() for i in indices_text.split(",") if i.strip()]
                rule["group"] = idx
            except Exception:
                pass
        return rule

    def from_rule(self, rule: dict[str, Any]) -> None:
        """Populate the inputs from a doping rule mapping.

        Parameters
        ----------
        rule : dict[str, Any]
            Mapping returned by `to_rule`.
        """
        if not rule:
            return
        self.target_edit.setText(str(rule.get("target", "")))
        dopants = rule.get("dopants")
        if dopants is not None:
            self.dopants_edit.setText(json.dumps(dopants))
        if "percent" in rule:
            self.percent_frame.set_input_value(rule["percent"])
        if "count" in rule:
            self.count_frame.set_input_value(rule["count"])
        if "group" in rule:
            self.indices_edit.setText(",".join(str(i) for i in rule["group"]))
        if "use" in rule:
            use_mode = rule["use"]
            if use_mode == "count":
                self.count_botton.setChecked(True)
            elif use_mode == "mass_percent":
                self.mass_percent_radio.setChecked(True)
            else:
                self.atomic_percent_radio.setChecked(True)
            self._on_mode_changed()
        if "ratio_type" in rule:
            self.ratio_type_button.setChecked(rule["ratio_type"] == "atom")
            self._toggle_ratio_type()


class DopingRulesWidget(QWidget):
    """Container widget for multiple doping rules."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the layout that hosts rule items and the add button."""
        super().__init__(parent)
        self.__layout = QVBoxLayout(self)
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(4)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self.add_button = TransparentToolButton(FluentIcon.ADD, self)
        self.add_button.clicked.connect(self.add_rule)
        self.add_button.setToolTip("Add rule")
        self.add_button.installEventFilter(ToolTipFilter(self.add_button, 300, ToolTipPosition.TOP))
        btn_layout.addWidget(self.add_button, 0, Qt.AlignmentFlag.AlignLeft)
        btn_layout.addStretch(1)
        self.__layout.addLayout(btn_layout)

        self.rule_container = QWidget(self)
        self.rule_layout = QVBoxLayout(self.rule_container)
        self.rule_layout.setContentsMargins(0, 0, 0, 0)
        self.rule_layout.setSpacing(4)
        self.__layout.addWidget(self.rule_container)

    def add_rule(self, rule: dict[str, Any] | None = None) -> DopingRuleItem:
        """Append a rule widget to the list.

        Parameters
        ----------
        rule : dict[str, Any], optional
            Optional rule used to initialize the new widget.

        Returns
        -------
        DopingRuleItem
            Newly created rule widget.
        """
        item = DopingRuleItem(self.rule_container)
        self.rule_layout.addWidget(item)
        if rule:
            item.from_rule(rule)
        return item

    def to_rules(self) -> list[dict[str, Any]]:
        """Serialize all rule widgets to a list of dictionaries."""
        rules: list[dict[str, Any]] = []
        for i in range(self.rule_layout.count()):
            widget = self.rule_layout.itemAt(i).widget()
            if isinstance(widget, DopingRuleItem):
                rule = widget.to_rule()
                if rule:
                    rules.append(rule)
        return rules

    def from_rules(self, rules: list[dict[str, Any]]) -> None:
        """Populate the rule list from serialized mappings.

        Parameters
        ----------
        rules : list[dict[str, Any]]
            Rules returned by `to_rules`.
        """
        while self.rule_layout.count():
            item = self.rule_layout.takeAt(0).widget()
            if item is not None:
                item.deleteLater()
        for rule in rules or []:
            self.add_rule(rule)
