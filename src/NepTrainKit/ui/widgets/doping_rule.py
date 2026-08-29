"""Widgets for configuring random doping rules."""

import traceback
from typing import Any

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from qfluentwidgets import (
    TransparentToolButton,
    FluentIcon,
    LineEdit,
    PushButton,
    StrongBodyLabel,
)
from NepTrainKit.core.alloy import format_composition, parse_composition

from .input import SpinBoxUnitInputFrame
from .compact_form import CompactField, ResponsiveFormGrid, SegmentedControl
from .parameter_inputs import ElementLineEdit


class DopingRuleItem(QFrame):
    """Single doping rule widget."""

    ruleChanged = Signal()
    deleteRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dopingRuleItem")
        self.setStyleSheet(
            "QFrame#dopingRuleItem {"
            "background: rgba(100, 120, 128, 10);"
            "border: 1px solid rgba(100, 120, 128, 32);"
            "border-radius: 7px; }"
        )
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        root = QVBoxLayout(self)
        root.setContentsMargins(7, 5, 7, 6)
        root.setSpacing(5)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.title_label = StrongBodyLabel(self.tr("Replacement rule"), self)
        header.addWidget(self.title_label, 1)
        self.delete_button = TransparentToolButton(
            QIcon(":/images/src/images/delete.svg"), self
        )
        self.delete_button.setFixedSize(28, 28)
        self.delete_button.setToolTip(self.tr("Delete rule"))
        self.delete_button.clicked.connect(self._delete_self)
        header.addWidget(self.delete_button)
        root.addLayout(header)

        self.target_edit = ElementLineEdit(self)
        self.target_edit.setPlaceholderText(self.tr("Cs"))
        self.target_edit.setMinimumWidth(0)
        self.group_edit = LineEdit(self)
        self.group_edit.setPlaceholderText(self.tr("Optional: A,B"))
        self.group_edit.setMinimumWidth(0)
        self.indices_edit = self.group_edit
        identity_grid = ResponsiveFormGrid(self, two_column_threshold=260)
        self.target_field = CompactField(
            self.tr("Replace element"), self.target_edit, self
        )
        self.group_field = CompactField(
            self.tr("Only in groups"), self.group_edit, self
        )
        identity_grid.add_field(self.target_field)
        identity_grid.add_field(self.group_field)
        root.addWidget(identity_grid)

        self.dopants_edit = ElementLineEdit(self, multiple=True)
        self.dopants_edit.setPlaceholderText(self.tr("Ge or Ge:0.7,C:0.3"))
        self.dopants_edit.setMinimumWidth(0)
        self.dopants_field = CompactField(
            self.tr("Replace with"),
            self.dopants_edit,
            self,
            self.tr("Enter one element or comma-separated relative weights."),
        )
        root.addWidget(self.dopants_field)

        self.ratio_type_control = SegmentedControl(parent=self)
        self.ratio_type_control.addItem(self.tr("Atom ratio"), userData="atom")
        self.ratio_type_control.addItem(self.tr("Mass ratio"), userData="mass")
        self.ratio_type_button = self.ratio_type_control
        self.ratio_type_field = CompactField(
            self.tr("Dopant weight basis"), self.ratio_type_control, self
        )

        self.amount_mode_control = SegmentedControl(parent=self)
        self.amount_mode_control.addItem(self.tr("Atomic %"), userData="atomic_percent")
        self.amount_mode_control.addItem(
            self.tr("Mass budget %"), userData="mass_percent"
        )
        self.amount_mode_control.addItem(self.tr("Count"), userData="count")
        self.atomic_percent_radio = self.amount_mode_control
        self.mass_percent_radio = self.amount_mode_control
        self.count_botton = self.amount_mode_control
        self.amount_mode_field = CompactField(
            self.tr("Replacement amount"), self.amount_mode_control, self
        )
        mode_grid = ResponsiveFormGrid(self, two_column_threshold=360)
        mode_grid.add_field(self.ratio_type_field, span=2)
        mode_grid.add_field(self.amount_mode_field, span=2)
        root.addWidget(mode_grid)

        self.percent_frame = SpinBoxUnitInputFrame(self)
        self.percent_frame.set_input(["~", ""], 2, "float")
        self.percent_frame.setDecimals(3)
        self.percent_frame.setRange(0, 100)
        self.percent_frame.set_input_value([0.0, 100.0])
        self.percent_field = CompactField(
            self.tr("Percentage range"),
            self.percent_frame,
            self,
            self.tr("A fixed percentage uses the same minimum and maximum."),
        )

        self.count_mode_combo = SegmentedControl(parent=self)
        self.count_mode_combo.addItem(self.tr("Fixed"), userData="fixed")
        self.count_mode_combo.addItem(self.tr("Random range"), userData="random")
        self.count_mode_field = CompactField(
            self.tr("Count behavior"), self.count_mode_combo, self
        )

        self.fixed_count_frame = SpinBoxUnitInputFrame(self)
        self.fixed_count_frame.set_input("", 1, "int")
        self.fixed_count_frame.setRange(0, 999999)
        self.fixed_count_frame.set_input_value([10])
        self.fixed_count_field = CompactField(
            self.tr("Atoms replaced"),
            self.fixed_count_frame,
            self,
            inline=True,
            input_max_width=150,
        )

        self.count_range_frame = SpinBoxUnitInputFrame(self)
        self.count_range_frame.set_input(["-", ""], 2, "int")
        self.count_range_frame.setRange(0, 999999)
        self.count_range_frame.set_input_value([1, 10])
        self.count_frame = self.count_range_frame
        self.count_range_field = CompactField(
            self.tr("Atom-count range"), self.count_range_frame, self
        )
        root.addWidget(self.percent_field)
        root.addWidget(self.count_mode_field)
        root.addWidget(self.fixed_count_field)
        root.addWidget(self.count_range_field)

        self.amount_mode_control.currentIndexChanged.connect(self._on_mode_changed)
        self.count_mode_combo.currentIndexChanged.connect(self._on_count_mode_changed)
        self.target_edit.textChanged.connect(self.ruleChanged)
        self.group_edit.textChanged.connect(self.ruleChanged)
        self.dopants_edit.textChanged.connect(self.ruleChanged)
        self.ratio_type_control.currentIndexChanged.connect(self.ruleChanged)
        self.amount_mode_control.currentIndexChanged.connect(self.ruleChanged)
        self.count_mode_combo.currentIndexChanged.connect(self.ruleChanged)
        for frame in (
            self.percent_frame,
            self.fixed_count_frame,
            self.count_range_frame,
        ):
            for control in frame.object_list:
                control.valueChanged.connect(self.ruleChanged)
        self._on_mode_changed()

    def _delete_self(self) -> None:
        """Ask the owning rule list to remove this rule."""
        self.deleteRequested.emit(self)

    def _toggle_ratio_type(self) -> None:
        """Compatibility hook retained for older callers."""
        self.ruleChanged.emit()

    def _on_mode_changed(self) -> None:
        is_count = self.amount_mode_control.currentData() == "count"
        self.percent_field.setVisible(not is_count)
        self.count_mode_field.setVisible(is_count)
        self._on_count_mode_changed()

    def _on_count_mode_changed(self) -> None:
        is_count = self.amount_mode_control.currentData() == "count"
        fixed = self.count_mode_combo.currentData() == "fixed"
        self.fixed_count_field.setVisible(is_count and fixed)
        self.count_range_field.setVisible(is_count and not fixed)

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
            dopants = parse_composition(dopant_text)
            if dopants:
                rule["dopants"] = dopants
        except Exception:
            logger.error(traceback.format_exc())
        if not target and not dopant_text:
            return {}

        rule["percent"] = [float(v) for v in self.percent_frame.get_input_value()]
        if self.count_mode_combo.currentData() == "fixed":
            count = int(self.fixed_count_frame.get_input_value()[0])
            count_values = [count, count]
            rule["count_mode"] = "fixed"
        else:
            count_values = [int(v) for v in self.count_range_frame.get_input_value()]
            rule["count_mode"] = "random"
        rule["count"] = count_values

        rule["use"] = str(self.amount_mode_control.currentData())

        rule["ratio_type"] = str(self.ratio_type_control.currentData())

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
            dopant_items = dict(dopants)
            if len(dopant_items) == 1 and float(next(iter(dopant_items.values()))) == 1.0:
                self.dopants_edit.setText(str(next(iter(dopant_items))))
            else:
                self.dopants_edit.setText(format_composition(dopant_items))
        if "percent" in rule:
            self.percent_frame.set_input_value(rule["percent"])
        if "count" in rule:
            count_values = list(rule["count"])
            if count_values:
                self.fixed_count_frame.set_input_value([int(count_values[0])])
                if len(count_values) == 1:
                    self.count_range_frame.set_input_value([int(count_values[0]), int(count_values[0])])
                else:
                    self.count_range_frame.set_input_value([int(count_values[0]), int(count_values[-1])])
        if "group" in rule:
            groups = rule["group"]
            self.indices_edit.setText(
                str(groups)
                if isinstance(groups, str)
                else ",".join(str(i) for i in groups)
            )
        if "use" in rule:
            use_mode = str(rule["use"])
            index = self.amount_mode_control.findData(use_mode)
            self.amount_mode_control.setCurrentIndex(
                index if index >= 0 else self.amount_mode_control.findData("atomic_percent")
            )
            self._on_mode_changed()
        count_values = list(rule.get("count", [1, 1]))
        count_mode = str(rule.get("count_mode", "")).lower()
        if count_mode == "fixed" or (not count_mode and count_values and count_values[0] == count_values[-1]):
            self.count_mode_combo.setCurrentIndex(0)
        else:
            self.count_mode_combo.setCurrentIndex(1)
        self._on_count_mode_changed()
        if "ratio_type" in rule:
            ratio_index = self.ratio_type_control.findData(str(rule["ratio_type"]))
            self.ratio_type_control.setCurrentIndex(
                ratio_index if ratio_index >= 0 else self.ratio_type_control.findData("atom")
            )


class DopingRulesWidget(QWidget):
    """Container widget for multiple doping rules."""

    rulesChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the layout that hosts rule items and the add button."""
        super().__init__(parent)
        self.__layout = QVBoxLayout(self)
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(4)

        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self.add_button = PushButton(FluentIcon.ADD, self.tr("Add replacement rule"), self)
        self.add_button.clicked.connect(self.add_rule)
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
        item.ruleChanged.connect(self.rulesChanged)
        item.deleteRequested.connect(self._remove_rule)
        if rule:
            item.from_rule(rule)
        self._refresh_rule_titles()
        self.rulesChanged.emit()
        return item

    def _remove_rule(self, item: DopingRuleItem) -> None:
        self.rule_layout.removeWidget(item)
        item.setParent(None)
        item.deleteLater()
        self._refresh_rule_titles()
        self.rulesChanged.emit()

    def _refresh_rule_titles(self) -> None:
        for index in range(self.rule_layout.count()):
            item = self.rule_layout.itemAt(index).widget()
            if isinstance(item, DopingRuleItem):
                item.title_label.setText(self.tr("Rule {index}").format(index=index + 1))

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
                item.hide()
                item.setParent(None)
                item.deleteLater()
        for rule in rules or [None]:
            self.add_rule(rule)
        self._refresh_rule_titles()
        self.rulesChanged.emit()
