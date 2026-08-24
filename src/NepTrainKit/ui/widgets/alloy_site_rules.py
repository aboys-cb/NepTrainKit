"""Visual editors for finite-cell alloy site-set rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from ase.data import atomic_numbers
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    FluentIcon,
    LineEdit,
    PushButton,
    StrongBodyLabel,
    TransparentToolButton,
)

from .compact_form import SegmentedControl
from .input import AdaptiveCompactSpinBox, AdaptiveInlineDoubleSpinBox


RULE_MODES = ("fixed_fraction", "fraction_range", "count_range")
COMPACT_CONTROL_HEIGHT = 28


def _element_symbol(text: object) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    return value[0].upper() + value[1:].lower()


class AlloyElementRuleRow(QWidget):
    """One element and its mode-dependent fraction or count fields."""

    changed = Signal()
    removeRequested = Signal(object)

    def __init__(self, element: str = "X", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode = "fixed_fraction"
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)

        self.element_edit = LineEdit(self)
        self.element_edit.setText(element)
        self.element_edit.setPlaceholderText(self.tr("Element"))
        self.element_edit.setMinimumWidth(0)
        self.element_edit.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.element_edit.setFixedHeight(COMPACT_CONTROL_HEIGHT)

        self.fixed_fraction_spin = AdaptiveInlineDoubleSpinBox(self)
        self.fixed_fraction_spin.setRange(0.0, 1.0)
        self.fixed_fraction_spin.setDecimals(6)
        self.fixed_fraction_spin.setSingleStep(0.05)
        self.fixed_fraction_spin.setValue(1.0)
        self.fixed_fraction_spin.setMinimumWidth(0)
        self.fixed_fraction_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.fixed_fraction_spin.setFixedHeight(COMPACT_CONTROL_HEIGHT)

        self.fraction_min_spin = AdaptiveInlineDoubleSpinBox(self)
        self.fraction_max_spin = AdaptiveInlineDoubleSpinBox(self)
        for spin in (self.fraction_min_spin, self.fraction_max_spin):
            spin.setRange(0.0, 1.0)
            spin.setDecimals(6)
            spin.setSingleStep(0.05)
            spin.setMinimumWidth(0)
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            spin.setFixedHeight(COMPACT_CONTROL_HEIGHT)
        self.fraction_min_spin.setValue(0.0)
        self.fraction_max_spin.setValue(1.0)

        self.count_min_spin = AdaptiveCompactSpinBox(self)
        self.count_max_spin = AdaptiveCompactSpinBox(self)
        for spin in (self.count_min_spin, self.count_max_spin):
            spin.setRange(0, 1_000_000)
            spin.setMinimumWidth(0)
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            spin.setFixedHeight(COMPACT_CONTROL_HEIGHT)
        self.count_min_spin.setValue(0)
        self.count_max_spin.setValue(1)

        self.delete_button = TransparentToolButton(FluentIcon.DELETE, self)
        self.delete_button.setToolTip(self.tr("Remove element"))
        self.delete_button.setAccessibleName(self.tr("Remove element"))
        self.delete_button.setFixedSize(COMPACT_CONTROL_HEIGHT, COMPACT_CONTROL_HEIGHT)
        self.delete_button.clicked.connect(lambda: self.removeRequested.emit(self))

        layout.addWidget(self.element_edit, 0, 0)
        layout.addWidget(self.fixed_fraction_spin, 0, 1, 1, 2)
        layout.addWidget(self.fraction_min_spin, 0, 1)
        layout.addWidget(self.fraction_max_spin, 0, 2)
        layout.addWidget(self.count_min_spin, 0, 1)
        layout.addWidget(self.count_max_spin, 0, 2)
        layout.addWidget(self.delete_button, 0, 3)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        self.element_edit.textChanged.connect(self.changed)
        for spin in (
            self.fixed_fraction_spin,
            self.fraction_min_spin,
            self.fraction_max_spin,
            self.count_min_spin,
            self.count_max_spin,
        ):
            spin.valueChanged.connect(self.changed)
        self.set_mode(self._mode)

    def set_mode(self, mode: str) -> None:
        self._mode = mode if mode in RULE_MODES else "fixed_fraction"
        fixed = self._mode == "fixed_fraction"
        fraction_range = self._mode == "fraction_range"
        count_range = self._mode == "count_range"
        self.fixed_fraction_spin.setVisible(fixed)
        self.fraction_min_spin.setVisible(fraction_range)
        self.fraction_max_spin.setVisible(fraction_range)
        self.count_min_spin.setVisible(count_range)
        self.count_max_spin.setVisible(count_range)

    def element(self) -> str:
        return _element_symbol(self.element_edit.text())

    def value(self, mode: str) -> float | int | list[float] | list[int]:
        if mode == "fixed_fraction":
            return float(self.fixed_fraction_spin.value())
        if mode == "fraction_range":
            return [float(self.fraction_min_spin.value()), float(self.fraction_max_spin.value())]
        return [int(self.count_min_spin.value()), int(self.count_max_spin.value())]

    def set_value(self, mode: str, value: Any) -> None:
        if mode == "fixed_fraction":
            self.fixed_fraction_spin.setValue(float(value))
            return
        values = list(value) if isinstance(value, (list, tuple)) else [value, value]
        if len(values) != 2:
            raise ValueError(self.tr("Range values must contain a minimum and maximum."))
        if mode == "fraction_range":
            self.fraction_min_spin.setValue(float(values[0]))
            self.fraction_max_spin.setValue(float(values[1]))
        else:
            self.count_min_spin.setValue(int(values[0]))
            self.count_max_spin.setValue(int(values[1]))

    def tab_widgets(self) -> list[QWidget]:
        widgets: list[QWidget] = [self.element_edit]
        if self._mode == "fixed_fraction":
            widgets.append(self.fixed_fraction_spin)
        elif self._mode == "fraction_range":
            widgets.extend([self.fraction_min_spin, self.fraction_max_spin])
        else:
            widgets.extend([self.count_min_spin, self.count_max_spin])
        widgets.append(self.delete_button)
        return widgets


class AlloySiteSetRuleEditor(CardWidget):
    """Collapsible visual editor for one site set."""

    changed = Signal()
    layoutChanged = Signal()
    removeRequested = Signal(object)

    def __init__(
        self,
        label: str = "A",
        rule: Mapping[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._expanded = True
        self._site_count: int | None = None
        self.element_rows: list[AlloyElementRuleRow] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 5, 8, 5)
        root.setSpacing(3)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        self.collapse_button = TransparentToolButton(FluentIcon.CARE_DOWN_SOLID, self)
        self.collapse_button.setToolTip(self.tr("Collapse or expand this site set"))
        self.collapse_button.setAccessibleName(self.tr("Collapse or expand this site set"))
        self.collapse_button.setFixedSize(26, 26)
        self.collapse_button.clicked.connect(self.toggle_expanded)

        self.title_label = StrongBodyLabel(self.tr("Site set"), self)
        self.label_edit = LineEdit(self)
        self.label_edit.setText(label)
        self.label_edit.setPlaceholderText(self.tr("Label"))
        self.label_edit.setMinimumWidth(0)
        self.label_edit.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.label_edit.setFixedHeight(COMPACT_CONTROL_HEIGHT)
        self.site_count_label = CaptionLabel(self.tr("Site count unknown"), self)

        self.mode_combo = SegmentedControl(parent=self)
        self.mode_combo.addItem(self.tr("Fixed"), userData="fixed_fraction")
        self.mode_combo.addItem(self.tr("Fraction"), userData="fraction_range")
        self.mode_combo.addItem(self.tr("Count"), userData="count_range")
        self.mode_combo.setMinimumWidth(136)
        self.mode_combo.setFixedHeight(COMPACT_CONTROL_HEIGHT)

        self.delete_button = TransparentToolButton(FluentIcon.DELETE, self)
        self.delete_button.setToolTip(self.tr("Remove site set"))
        self.delete_button.setAccessibleName(self.tr("Remove site set"))
        self.delete_button.setFixedSize(26, 26)
        self.delete_button.clicked.connect(lambda: self.removeRequested.emit(self))

        header.addWidget(self.collapse_button)
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.delete_button)
        root.addLayout(header)

        metadata = QGridLayout()
        metadata.setContentsMargins(30, 0, 0, 2)
        metadata.setHorizontalSpacing(6)
        metadata.setVerticalSpacing(3)
        label_caption = CaptionLabel(self.tr("Label"), self)
        mode_caption = CaptionLabel(self.tr("Composition mode"), self)
        label_caption.setStyleSheet("color:#8a95a0;")
        mode_caption.setStyleSheet("color:#8a95a0;")
        metadata.addWidget(label_caption, 0, 0)
        metadata.addWidget(mode_caption, 0, 1)
        metadata.addWidget(self.label_edit, 1, 0)
        metadata.addWidget(self.mode_combo, 1, 1)
        metadata.addWidget(self.site_count_label, 2, 0, 1, 2)
        metadata.setColumnStretch(0, 1)
        metadata.setColumnStretch(1, 2)
        root.addLayout(metadata)

        self.body_widget = QWidget(self)
        body = QVBoxLayout(self.body_widget)
        body.setContentsMargins(30, 2, 0, 0)
        body.setSpacing(6)

        self.column_header = QWidget(self.body_widget)
        column_layout = QGridLayout(self.column_header)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setHorizontalSpacing(4)
        self.element_header = CaptionLabel(self.tr("Element"), self.column_header)
        self.value_1_header = CaptionLabel("", self.column_header)
        self.value_2_header = CaptionLabel("", self.column_header)
        self.action_header = CaptionLabel("", self.column_header)
        for label_widget in (self.element_header, self.value_1_header, self.value_2_header):
            label_widget.setWordWrap(True)
        column_layout.addWidget(self.element_header, 0, 0)
        column_layout.addWidget(self.value_1_header, 0, 1)
        column_layout.addWidget(self.value_2_header, 0, 2)
        column_layout.addWidget(self.action_header, 0, 3)
        column_layout.setColumnStretch(0, 1)
        self.column_header.hide()

        self.rows_widget = QWidget(self.body_widget)
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)
        body.addWidget(self.rows_widget)

        self.add_element_button = PushButton(FluentIcon.ADD, self.tr("Add element"), self.body_widget)
        self.add_element_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.add_element_button.setFixedHeight(COMPACT_CONTROL_HEIGHT)
        self.add_element_button.clicked.connect(lambda: self.add_element())
        body.addWidget(self.add_element_button, 0, Qt.AlignmentFlag.AlignLeft)

        self.error_label = CaptionLabel("", self.body_widget)
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        body.addWidget(self.error_label)
        root.addWidget(self.body_widget)

        self.label_edit.textChanged.connect(self.changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        if rule is None:
            rule = {
                "elements": ["X"],
                "mode": "fixed_fraction",
                "composition": {"X": 1.0},
            }
        self.from_rule(rule)

    def toggle_expanded(self) -> None:
        self.set_expanded(not self._expanded)
        self.layoutChanged.emit()

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = bool(expanded)
        self.body_widget.setVisible(self._expanded)
        icon = FluentIcon.CARE_DOWN_SOLID if self._expanded else FluentIcon.CHEVRON_RIGHT
        self.collapse_button.setIcon(icon)

    def mode(self) -> str:
        return str(self.mode_combo.currentData() or "fixed_fraction")

    def set_label_editable(self, editable: bool) -> None:
        self.label_edit.setReadOnly(not editable)
        self.delete_button.setEnabled(editable)

    def set_site_count(self, count: int | None) -> None:
        self._site_count = None if count is None else int(count)
        if self._site_count is None:
            self.site_count_label.setText(self.tr("Site count unknown"))
        else:
            self.site_count_label.setText(
                self.tr("{count} sites").format(count=self._site_count)
            )

    def clear_elements(self) -> None:
        for row in self.element_rows:
            self.rows_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self.element_rows.clear()

    def add_element(
        self,
        element: str = "X",
        value: Any | None = None,
        *,
        emit_change: bool = True,
    ) -> AlloyElementRuleRow:
        row = AlloyElementRuleRow(element, self.rows_widget)
        row.set_mode(self.mode())
        if value is not None:
            row.set_value(self.mode(), value)
        row.changed.connect(self.changed)
        row.removeRequested.connect(self.remove_element)
        self.element_rows.append(row)
        self.rows_layout.addWidget(row)
        if emit_change:
            self.changed.emit()
        return row

    def remove_element(self, row: AlloyElementRuleRow) -> None:
        if row not in self.element_rows:
            return
        self.element_rows.remove(row)
        self.rows_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        self.changed.emit()

    def _on_mode_changed(self) -> None:
        mode = self.mode()
        if mode == "fixed_fraction":
            headers = (self.tr("Target fraction"), "")
        elif mode == "fraction_range":
            headers = (self.tr("Minimum fraction"), self.tr("Maximum fraction"))
        else:
            headers = (self.tr("Minimum count"), self.tr("Maximum count"))
        self.value_1_header.setText(headers[0])
        self.value_2_header.setText(headers[1])
        self.value_2_header.setVisible(bool(headers[1]))
        for row in self.element_rows:
            row.set_mode(mode)
        self.changed.emit()

    def to_rule(self) -> dict[str, Any]:
        mode = self.mode()
        elements = [row.element() for row in self.element_rows]
        if mode == "fixed_fraction":
            field_name = "composition"
        elif mode == "fraction_range":
            field_name = "fractions"
        else:
            field_name = "counts"
        return {
            "elements": elements,
            "mode": mode,
            field_name: {
                row.element(): row.value(mode)
                for row in self.element_rows
            },
        }

    def from_rule(self, rule: Mapping[str, Any]) -> None:
        mode = str(rule.get("mode", "fixed_fraction"))
        if mode not in RULE_MODES:
            raise ValueError(
                self.tr("Unsupported composition mode: {mode}").format(mode=mode)
            )
        field_name = {
            "fixed_fraction": "composition",
            "fraction_range": "fractions",
            "count_range": "counts",
        }[mode]
        values = rule.get(field_name)
        elements = rule.get("elements")
        if not isinstance(elements, list) or not elements:
            raise ValueError(self.tr("Each site set needs at least one element."))
        if not isinstance(values, Mapping):
            raise ValueError(
                self.tr("Mode {mode} requires a {field} mapping.").format(
                    mode=mode,
                    field=field_name,
                )
            )
        if set(str(key) for key in values) != set(str(element) for element in elements):
            raise ValueError(
                self.tr("Element names must match the keys in the active composition fields.")
            )

        index = self.mode_combo.findData(mode)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(max(index, 0))
        self.mode_combo.blockSignals(False)
        self.clear_elements()
        for element in elements:
            self.add_element(str(element), values[str(element)], emit_change=False)
        self._on_mode_changed()

    def validation_errors(self, *, allow_placeholder: bool = False) -> list[str]:
        errors: list[str] = []
        elements = [row.element() for row in self.element_rows]
        if not elements:
            errors.append(self.tr("Add at least one element."))
        if not allow_placeholder and "X" in elements:
            errors.append(self.tr("Replace placeholder X with real element symbols."))
        invalid = sorted(
            {
                element
                for element in elements
                if element not in atomic_numbers or (element == "X" and not allow_placeholder)
            }
            - {"X"}
        )
        if invalid:
            errors.append(
                self.tr("Invalid element symbols: {elements}.").format(
                    elements=", ".join(invalid)
                )
            )
        duplicates = sorted({element for element in elements if elements.count(element) > 1})
        if duplicates:
            errors.append(
                self.tr("Duplicate elements: {elements}.").format(
                    elements=", ".join(duplicates)
                )
            )

        mode = self.mode()
        if mode == "fixed_fraction":
            total = sum(float(row.fixed_fraction_spin.value()) for row in self.element_rows)
            if abs(total - 1.0) > 1e-6:
                errors.append(
                    self.tr("Fixed fractions must sum to 1 (current sum: {total:.6g}).").format(
                        total=total
                    )
                )
        elif mode == "fraction_range":
            for row in self.element_rows:
                low = float(row.fraction_min_spin.value())
                high = float(row.fraction_max_spin.value())
                if not 0.0 <= low <= 1.0 or not 0.0 <= high <= 1.0:
                    errors.append(
                        self.tr("Fractions for {element} must stay between 0 and 1.").format(
                            element=row.element() or self.tr("empty element")
                        )
                    )
                if low > high:
                    errors.append(
                        self.tr("Minimum fraction exceeds maximum for {element}.").format(
                            element=row.element() or self.tr("empty element")
                        )
                    )
        else:
            for row in self.element_rows:
                low = int(row.count_min_spin.value())
                high = int(row.count_max_spin.value())
                if low < 0 or high < 0:
                    errors.append(
                        self.tr("Counts for {element} must be non-negative integers.").format(
                            element=row.element() or self.tr("empty element")
                        )
                    )
                if low > high:
                    errors.append(
                        self.tr("Minimum count exceeds maximum for {element}.").format(
                            element=row.element() or self.tr("empty element")
                        )
                    )
        self.set_errors(errors)
        return errors

    def set_errors(self, errors: list[str]) -> None:
        if errors:
            self.set_expanded(True)
            self.error_label.setText("⚠ " + "\n".join(errors))
            self.error_label.show()
        else:
            self.error_label.clear()
            self.error_label.hide()

    def tab_widgets(self) -> list[QWidget]:
        widgets: list[QWidget] = [
            self.collapse_button,
            self.label_edit,
            self.mode_combo,
            self.delete_button,
        ]
        if self._expanded:
            for row in self.element_rows:
                widgets.extend(row.tab_widgets())
            widgets.append(self.add_element_button)
        return widgets


class AlloySiteRulesEditor(QWidget):
    """Manage site partition mode, templates, and site-set rule editors."""

    changed = Signal()
    layoutChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._loading = False
        self._confirm_replacement: Callable[[], bool] | None = None
        self.site_editors: list[AlloySiteSetRuleEditor] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        partition_row = QVBoxLayout()
        partition_row.setContentsMargins(0, 0, 0, 0)
        partition_row.setSpacing(4)
        self.partition_label = BodyLabel(self.tr("Site partition"), self)
        self.partition_mode_combo = SegmentedControl(parent=self)
        self.partition_mode_combo.addItem(self.tr("Entire structure"), userData="all")
        self.partition_mode_combo.addItem(self.tr("Sublattices"), userData="sublattices")
        self.partition_mode_combo.setMinimumWidth(0)
        self.partition_mode_combo.setFixedHeight(COMPACT_CONTROL_HEIGHT)

        self.single_template_button = PushButton(
            self.tr("No sublattice labels (all)"),
            self,
        )
        self.ab_template_button = PushButton(
            self.tr("A/B sublattices"),
            self,
        )
        self.add_site_button = PushButton(FluentIcon.ADD, self.tr("Add site set"), self)
        for button in (
            self.single_template_button,
            self.ab_template_button,
            self.add_site_button,
        ):
            button.setFixedHeight(COMPACT_CONTROL_HEIGHT)

        partition_row.addWidget(self.partition_label)
        partition_row.addWidget(self.partition_mode_combo)
        root.addLayout(partition_row)

        template_row = QHBoxLayout()
        template_row.setContentsMargins(0, 0, 0, 0)
        template_row.setSpacing(4)
        self.template_label = CaptionLabel(self.tr("Rule templates"), self)
        template_row.addWidget(self.single_template_button, 1)
        template_row.addWidget(self.ab_template_button, 1)
        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.addWidget(self.template_label)
        action_row.addStretch(1)
        action_row.addWidget(self.add_site_button)
        root.addLayout(action_row)
        root.addLayout(template_row)

        self.status_label = CaptionLabel("", self)
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.status_label)

        self.rules_widget = QWidget(self)
        self.rules_layout = QVBoxLayout(self.rules_widget)
        self.rules_layout.setContentsMargins(0, 0, 0, 0)
        self.rules_layout.setSpacing(4)
        root.addWidget(self.rules_widget)

        self.partition_mode_combo.currentIndexChanged.connect(self._on_partition_changed)
        self.single_template_button.clicked.connect(lambda: self.request_template("all"))
        self.ab_template_button.clicked.connect(lambda: self.request_template("ab"))
        self.add_site_button.clicked.connect(lambda: self.add_site_set())
        self.load_template("ab")

    def partition_mode(self) -> str:
        return str(self.partition_mode_combo.currentData() or "all")

    def _set_partition_mode(self, mode: str) -> None:
        index = self.partition_mode_combo.findData(mode)
        self.partition_mode_combo.blockSignals(True)
        self.partition_mode_combo.setCurrentIndex(max(index, 0))
        self.partition_mode_combo.blockSignals(False)
        self.add_site_button.setEnabled(mode == "sublattices")

    def _on_partition_changed(self) -> None:
        if self._loading:
            return
        template = "all" if self.partition_mode() == "all" else "ab"
        if not self.request_template(template):
            current_mode = "all" if set(self.to_rules()) == {"all"} else "sublattices"
            self._set_partition_mode(current_mode)

    def set_replacement_confirmation(self, callback: Callable[[], bool] | None) -> None:
        self._confirm_replacement = callback

    def request_template(self, template: str) -> bool:
        if self._confirm_replacement is not None and not self._confirm_replacement():
            return False
        self.load_template(template)
        return True

    def load_template(self, template: str) -> None:
        if template == "all":
            rules = {
                "all": {
                    "elements": ["X"],
                    "mode": "fixed_fraction",
                    "composition": {"X": 1.0},
                }
            }
        else:
            rules = {
                "A": {
                    "elements": ["X"],
                    "mode": "fixed_fraction",
                    "composition": {"X": 1.0},
                },
                "B": {
                    "elements": ["X"],
                    "mode": "fixed_fraction",
                    "composition": {"X": 1.0},
                },
            }
        self.from_rules(rules)

    def clear_site_sets(self) -> None:
        for editor in self.site_editors:
            self.rules_layout.removeWidget(editor)
            editor.setParent(None)
            editor.deleteLater()
        self.site_editors.clear()

    def add_site_set(
        self,
        label: str | None = None,
        rule: Mapping[str, Any] | None = None,
        *,
        emit_change: bool = True,
    ) -> AlloySiteSetRuleEditor:
        if label is None:
            used = {editor.label_edit.text().strip() for editor in self.site_editors}
            index = 1
            while f"S{index}" in used:
                index += 1
            label = f"S{index}"
        editor = AlloySiteSetRuleEditor(label, rule, self.rules_widget)
        editor.changed.connect(self.changed)
        editor.layoutChanged.connect(self.layoutChanged)
        editor.removeRequested.connect(self.remove_site_set)
        self.site_editors.append(editor)
        self.rules_layout.addWidget(editor)
        editable = self.partition_mode() == "sublattices"
        editor.set_label_editable(editable)
        if emit_change:
            self.changed.emit()
        return editor

    def remove_site_set(self, editor: AlloySiteSetRuleEditor) -> None:
        if editor not in self.site_editors:
            return
        self.site_editors.remove(editor)
        self.rules_layout.removeWidget(editor)
        editor.setParent(None)
        editor.deleteLater()
        self.changed.emit()

    @staticmethod
    def validate_rule_mapping(rules: Any) -> dict[str, Mapping[str, Any]]:
        if not isinstance(rules, Mapping) or not rules:
            raise ValueError("site_rules must be a non-empty JSON object.")
        normalized: dict[str, Mapping[str, Any]] = {}
        for raw_label, raw_rule in rules.items():
            label = str(raw_label).strip()
            if not label:
                raise ValueError("Site-set labels must be non-empty.")
            if not isinstance(raw_rule, Mapping):
                raise ValueError(f"Rule for {label!r} must be an object.")
            allowed = {"elements", "mode", "composition", "fractions", "counts"}
            extra = sorted(set(str(key) for key in raw_rule) - allowed)
            if extra:
                raise ValueError(
                    f"Rule for {label!r} contains unsupported fields: {', '.join(extra)}."
                )
            normalized[label] = raw_rule
        if "all" in normalized and len(normalized) > 1:
            raise ValueError("'all' cannot be combined with explicit sublattice labels.")
        return normalized

    def from_rules(self, rules: Mapping[str, Any]) -> None:
        normalized = self.validate_rule_mapping(rules)
        self._loading = True
        try:
            mode = "all" if set(normalized) == {"all"} else "sublattices"
            self._set_partition_mode(mode)
            self.clear_site_sets()
            for label, rule in normalized.items():
                self.add_site_set(label, rule, emit_change=False)
            for index, editor in enumerate(self.site_editors):
                editor.set_expanded(index == 0)
        finally:
            self._loading = False
        self.changed.emit()

    def to_rules(self) -> dict[str, dict[str, Any]]:
        return {
            editor.label_edit.text().strip(): editor.to_rule()
            for editor in self.site_editors
        }

    def set_input_counts(self, counts: Mapping[str, int] | None) -> None:
        count_map = {str(label): int(count) for label, count in dict(counts or {}).items()}
        for editor in self.site_editors:
            editor.set_site_count(count_map.get(editor.label_edit.text().strip()))

    def validation_errors(self, input_counts: Mapping[str, int] | None = None) -> list[str]:
        errors: list[str] = []
        labels = [editor.label_edit.text().strip() for editor in self.site_editors]
        if not labels:
            errors.append(self.tr("Add at least one site set."))
        if any(not label for label in labels):
            errors.append(self.tr("Site-set labels must be non-empty."))
        duplicates = sorted({label for label in labels if label and labels.count(label) > 1})
        if duplicates:
            errors.append(
                self.tr("Duplicate site-set labels: {labels}.").format(
                    labels=", ".join(duplicates)
                )
            )

        input_map = {str(label): int(count) for label, count in dict(input_counts or {}).items()}
        if input_counts is not None:
            missing = sorted(set(input_map) - set(labels))
            extra = sorted(set(labels) - set(input_map))
            if missing:
                errors.append(
                    self.tr("Missing rules for input site sets: {labels}.").format(
                        labels=", ".join(missing)
                    )
                )
            if extra:
                errors.append(
                    self.tr("Rules reference site sets absent from the input: {labels}.").format(
                        labels=", ".join(extra)
                    )
                )

        for editor in self.site_editors:
            editor_errors = editor.validation_errors(
                allow_placeholder=input_counts is None
            )
            label = editor.label_edit.text().strip()
            if input_counts is not None and label and label not in input_map:
                editor_errors.append(
                    self.tr("This label does not exist in the input structure.")
                )
                editor.set_errors(editor_errors)
            errors.extend(
                self.tr("{label}: {error}").format(label=label or "?", error=error)
                for error in editor_errors
            )
        self.set_status_errors(errors)
        return errors

    def set_status_errors(self, errors: list[str]) -> None:
        global_errors = [
            error
            for error in errors
            if not any(error.startswith(f"{editor.label_edit.text().strip()}:") for editor in self.site_editors)
        ]
        if global_errors:
            self.status_label.setText("⚠ " + "\n".join(global_errors))
            self.status_label.show()
        else:
            self.status_label.clear()
            self.status_label.hide()

    def tab_widgets(self) -> list[QWidget]:
        widgets: list[QWidget] = [
            self.partition_mode_combo,
            self.add_site_button,
            self.single_template_button,
            self.ab_template_button,
        ]
        for editor in self.site_editors:
            widgets.extend(editor.tab_widgets())
        return widgets
