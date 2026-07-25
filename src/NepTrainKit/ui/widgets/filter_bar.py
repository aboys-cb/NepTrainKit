"""Filter dialog widgets for tag/formula search."""

from __future__ import annotations

import json
import re
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    FluentIcon,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    ToolTipPosition,
    TransparentToolButton,
)

from NepTrainKit.core.types import SearchType, TagFilterSpec, FilterGroup, FilterCondition
from NepTrainKit.ui.widgets.button import TagPushButton
from NepTrainKit.ui.widgets.search_widget import ConfigTypeSearchLineEdit


class _GroupPanel(QFrame):
    """Filter group: tags OR'd within group, groups AND'd.
    Include/Exclude dropdown controls whether the group matches or excludes."""

    removed = Signal(object)
    changed = Signal()

    def __init__(self, group: FilterGroup, completer_words: dict[str, int], parent=None):
        super().__init__(parent)
        self._group = group
        self._completer_words = completer_words
        self._active_input: ConfigTypeSearchLineEdit | None = None
        self._pills: list[TagPushButton] = []
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(8)

        self._mode_combo = ComboBox(self)
        self._mode_combo.addItem(self.tr("Include"), userData=False)
        self._mode_combo.addItem(self.tr("Exclude"), userData=True)
        is_exclude = any(c.negate for c in self._group.conditions if c.text)
        self._mode_combo.setCurrentIndex(1 if is_exclude else 0)
        self._mode_combo.currentIndexChanged.connect(self._on_changed)

        header.addWidget(self._mode_combo)
        header.addStretch()

        remove_btn = TransparentToolButton(FluentIcon.CLOSE, self)
        remove_btn.setToolTip(self.tr("Remove group"))
        remove_btn.clicked.connect(lambda: self.removed.emit(self))
        header.addWidget(remove_btn)
        layout.addLayout(header)

        self._pills_container = QWidget(self)
        self._pills_layout = QVBoxLayout(self._pills_container)
        self._pills_layout.setContentsMargins(0, 0, 0, 0)
        self._pills_layout.setSpacing(3)

        add_btn = TransparentToolButton(FluentIcon.ADD, self)
        add_btn.setToolTip(self.tr("Add condition"))
        add_btn.clicked.connect(self._on_add_clicked)
        self._pills_layout.addWidget(add_btn)
        layout.addWidget(self._pills_container)

        for cond in self._group.conditions:
            if cond.text:
                self._add_pill(cond.text)

    def _on_changed(self):
        self._sync_to_group()
        self.changed.emit()

    def _sync_to_group(self):
        data = self._mode_combo.itemData(self._mode_combo.currentIndex())
        negate = bool(data) if data is not None else False
        texts = [p.text() for p in self._pills]
        self._group.conditions = [FilterCondition(text=t, negate=negate) for t in texts]
        self._group.mode = "or"

    def _add_pill(self, text: str):
        text = text.strip()
        if not text:
            return
        for p in self._pills:
            if p.text() == text:
                return
        pill = TagPushButton(text, self, FluentIcon.TAG)
        pill.closeClicked.connect(lambda p=pill: self._remove_pill(p))
        self._pills_layout.insertWidget(len(self._pills), pill)
        self._pills.append(pill)
        self._sync_to_group()
        self.changed.emit()

    def _remove_pill(self, pill: TagPushButton):
        self._pills_layout.removeWidget(pill)
        pill.deleteLater()
        self._pills.remove(pill)
        self._sync_to_group()
        self.changed.emit()

    def _on_add_clicked(self):
        if self._active_input is not None:
            self._active_input.deleteLater()
            self._active_input = None

        line = ConfigTypeSearchLineEdit(self)
        line.setFixedWidth(240)
        line.setCompleterKeyWord(self._completer_words)

        line.searchSignal.connect(lambda _t, _s: self._commit_input(line))
        line.returnPressed.connect(lambda: self._commit_input(line))
        line.searchButton.setVisible(False)
        line.checkButton.setVisible(False)
        line.uncheckButton.setVisible(False)

        completer = line.completer()
        if completer is not None:
            completer.activated.connect(lambda _text: self._commit_input(line))

        self._pills_layout.insertWidget(len(self._pills), line)
        line.setFocus()
        self._active_input = line

    def _commit_input(self, line: ConfigTypeSearchLineEdit):
        text = line.text().strip()
        if text:
            self._add_pill(text)
        line.deleteLater()
        if self._active_input is line:
            self._active_input = None

    def get_group(self) -> FilterGroup:
        self._sync_to_group()
        return self._group


class TagFilterDialog(QDialog):
    """Dialog for building tag/formula filter conditions with group logic."""

    filterChanged = Signal(object)

    def __init__(self, spec: TagFilterSpec, completer_words: dict[str, int],
                 search_type: SearchType, parent=None):
        super().__init__(parent)
        self._spec = spec
        self._completer_words = completer_words
        self._search_type = search_type
        self._group_panels: list[_GroupPanel] = []

        self._build_ui()
        self._rebuild_groups()

    def _build_ui(self):
        title = "Config_type" if self._search_type == SearchType.TAG else "Formula"
        self.setWindowTitle(self.tr("{title} Filter").format(title=title))
        self.resize(540, 440)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        hint = BodyLabel(self.tr("Groups = AND  |  Tags in group = OR"), self)
        hint.setStyleSheet("color: #888; font-size: 12px; padding: 2px 0;")
        layout.addWidget(hint)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._groups_widget = QWidget()
        self._groups_layout = QVBoxLayout(self._groups_widget)
        self._groups_layout.setSpacing(8)
        self._groups_layout.addStretch()
        self._scroll.setWidget(self._groups_widget)
        layout.addWidget(self._scroll, 1)

        add_group_btn = PushButton(self.tr("+ Add Group"), self)
        add_group_btn.clicked.connect(self._add_group)
        layout.addWidget(add_group_btn)

        self._preview_label = BodyLabel(self)
        self._preview_label.setStyleSheet("color: #666; padding: 4px;")
        self._preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._preview_label)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._save_btn = PushButton(self.tr("Save"), self)
        self._save_btn.setToolTip(self.tr("Save filter to file"))
        self._save_btn.clicked.connect(self._save_filter)

        self._load_btn = PushButton(self.tr("Load"), self)
        self._load_btn.setToolTip(self.tr("Load filter from file"))
        self._load_btn.clicked.connect(self._load_filter)

        toolbar.addWidget(self._save_btn)
        toolbar.addWidget(self._load_btn)
        toolbar.addStretch()

        self._copy_btn = PushButton(self.tr("Copy"), self)
        self._copy_btn.setFixedHeight(26)
        self._copy_btn.clicked.connect(self._copy_expression)
        toolbar.addWidget(self._copy_btn)
        layout.addLayout(toolbar)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = PushButton(self.tr("Cancel"), self)
        cancel_btn.clicked.connect(self.reject)
        clear_btn = PushButton(self.tr("Clear All"), self)
        clear_btn.clicked.connect(self._clear_all)
        apply_btn = PrimaryPushButton(self.tr("Apply"), self)
        apply_btn.clicked.connect(self._apply)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(apply_btn)
        layout.addLayout(btn_layout)

    def _rebuild_groups(self):
        for panel in self._group_panels:
            self._groups_layout.removeWidget(panel)
            panel.deleteLater()
        self._group_panels.clear()

        for group in self._spec.groups:
            self._add_group_panel(group)

        if not self._spec.groups:
            self._add_group_panel(FilterGroup())

        self._update_preview()

    def _add_group_panel(self, group: FilterGroup):
        panel = _GroupPanel(group, self._completer_words, self)
        panel.removed.connect(self._on_group_removed)
        panel.changed.connect(self._update_preview)
        insert_index = len(self._group_panels)
        self._groups_layout.insertWidget(insert_index, panel)
        self._group_panels.append(panel)

    def _add_group(self):
        self._add_group_panel(FilterGroup())
        self._update_preview()

    def _on_group_removed(self, panel: _GroupPanel):
        self._groups_layout.removeWidget(panel)
        panel.deleteLater()
        self._group_panels.remove(panel)
        if not self._group_panels:
            self._add_group_panel(FilterGroup())
        self._update_preview()

    def _update_preview(self):
        spec = self._collect_spec()
        expr = spec.to_expression()
        self._preview_label.setText(self.tr("Preview: {expr}").format(expr=expr) if expr else self.tr("Preview: (empty)"))

    def _collect_spec(self) -> TagFilterSpec:
        groups = [p.get_group() for p in self._group_panels]
        return TagFilterSpec(groups=groups)

    def _clear_all(self):
        self._spec = TagFilterSpec()
        self._rebuild_groups()

    def _copy_expression(self):
        spec = self._collect_spec()
        expr = spec.to_expression()
        if expr:
            QApplication.clipboard().setText(expr)

    def _save_filter(self):
        spec = self._collect_spec()
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Save filter"), "filter.json",
            self.tr("JSON (*.json)"),
        )
        if path:
            try:
                data = spec.to_dict()
                data["search_type"] = self._search_type.value
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except OSError:
                pass

    def _load_filter(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Load filter"), "",
            self.tr("JSON (*.json)"),
        )
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._spec = TagFilterSpec.from_dict(data)
                self._rebuild_groups()
            except (OSError, json.JSONDecodeError, KeyError):
                pass

    def _apply(self):
        self._spec = self._collect_spec()
        self.filterChanged.emit(self._spec.to_dict())
        self.accept()


_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


class ElementsFilterDialog(QDialog):
    """Dialog for building element-based structure filter with group logic."""

    filterChanged = Signal(str)

    def __init__(self, expression: str, completer_words: dict[str, int] | None = None, parent=None):
        super().__init__(parent)
        groups = self._parse_expression(expression)
        self._spec = TagFilterSpec(groups=groups) if groups else TagFilterSpec()
        self._completer_words = completer_words or {el: 1 for el in _ELEMENTS}
        self._group_panels: list[_GroupPanel] = []

        self._build_ui()
        self._rebuild_groups()

    @staticmethod
    def _parse_expression(expr: str) -> list[FilterGroup]:
        groups: list[FilterGroup] = []
        if not expr:
            return groups
        allowed: list[str] = []
        required: list[str] = []
        excluded: list[str] = []
        for token in expr.replace(",", " ").split():
            token = token.strip()
            if not token:
                continue
            if token.startswith("+") and len(token) > 1:
                required.append(token[1:])
            elif token.startswith("-") and len(token) > 1:
                excluded.append(token[1:])
            elif token.startswith("!") and len(token) > 1:
                excluded.append(token[1:])
            else:
                allowed.append(token)
        if allowed:
            groups.append(FilterGroup(conditions=[FilterCondition(text=t) for t in allowed], mode="or"))
        if required:
            groups.append(FilterGroup(conditions=[FilterCondition(text=t) for t in required], mode="and"))
        if excluded:
            groups.append(FilterGroup(conditions=[FilterCondition(text=t, negate=True) for t in excluded], mode="or"))
        return groups

    def _build_expression(self) -> str:
        spec = self._collect_spec()
        parts: list[str] = []
        for group in spec.groups:
            if group.is_empty():
                continue
            negate = any(c.negate for c in group.conditions)
            prefix = "-" if negate else ""
            for cond in group.conditions:
                parts.append(f"{prefix}{cond.text}")
        return ", ".join(parts) if parts else ""

    def _build_ui(self):
        self.setWindowTitle(self.tr("Elements Filter"))
        self.resize(540, 400)
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        hint = BodyLabel(self.tr("Groups = AND  |  Elements in group = OR"), self)
        hint.setStyleSheet("color: #888; font-size: 12px; padding: 2px 0;")
        layout.addWidget(hint)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._groups_widget = QWidget()
        self._groups_layout = QVBoxLayout(self._groups_widget)
        self._groups_layout.setSpacing(8)
        self._groups_layout.addStretch()
        self._scroll.setWidget(self._groups_widget)
        layout.addWidget(self._scroll, 1)

        add_group_btn = PushButton(self.tr("+ Add Group"), self)
        add_group_btn.clicked.connect(self._add_group)
        layout.addWidget(add_group_btn)

        self._preview_label = BodyLabel(self)
        self._preview_label.setStyleSheet("color: #666; padding: 4px;")
        self._preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._preview_label)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = PushButton(self.tr("Cancel"), self)
        cancel_btn.clicked.connect(self.reject)
        clear_btn = PushButton(self.tr("Clear All"), self)
        clear_btn.clicked.connect(self._clear_all)
        apply_btn = PrimaryPushButton(self.tr("Apply"), self)
        apply_btn.clicked.connect(self._apply)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(apply_btn)
        layout.addLayout(btn_layout)

    def _rebuild_groups(self):
        for panel in self._group_panels:
            self._groups_layout.removeWidget(panel)
            panel.deleteLater()
        self._group_panels.clear()
        for group in self._spec.groups:
            self._add_group_panel(group)
        if not self._spec.groups:
            self._add_group_panel(FilterGroup())
        self._update_preview()

    def _add_group_panel(self, group: FilterGroup):
        panel = _GroupPanel(group, self._completer_words, self)
        panel.removed.connect(self._on_group_removed)
        panel.changed.connect(self._update_preview)
        insert_index = len(self._group_panels)
        self._groups_layout.insertWidget(insert_index, panel)
        self._group_panels.append(panel)

    def _add_group(self):
        self._add_group_panel(FilterGroup())
        self._update_preview()

    def _on_group_removed(self, panel: _GroupPanel):
        self._groups_layout.removeWidget(panel)
        panel.deleteLater()
        self._group_panels.remove(panel)
        if not self._group_panels:
            self._add_group_panel(FilterGroup())
        self._update_preview()

    def _update_preview(self):
        expr = self._build_expression()
        self._preview_label.setText(expr if expr else self.tr("(empty)"))

    def _collect_spec(self) -> TagFilterSpec:
        groups = [p.get_group() for p in self._group_panels]
        return TagFilterSpec(groups=groups)

    def _clear_all(self):
        self._spec = TagFilterSpec()
        self._rebuild_groups()

    def _apply(self):
        self._result_expression = self._build_expression()
        self.filterChanged.emit(self._result_expression)
        self.accept()


_EXPRESSION_FIELDS = [
    "natoms", "volume", "a", "b", "c",
    "energy", "energy_per_atom",
    "force.ref.max", "force.ref.avg", "force.ref.norm",
    "force.pred.max", "force.pred.avg", "force.pred.norm",
    "force.error.max", "force.error.avg", "force.error.norm",
    "virial.ref.xx", "virial.ref.yy", "virial.ref.zz",
    "virial.ref.xy", "virial.ref.xz", "virial.ref.yz",
    "virial.pred.xx", "virial.pred.yy", "virial.pred.zz",
    "virial.pred.xy", "virial.pred.xz", "virial.pred.yz",
    "stress.ref.xx", "stress.ref.yy", "stress.ref.zz",
    "stress.ref.xy", "stress.ref.xz", "stress.ref.yz",
    "stress.pred.xx", "stress.pred.yy", "stress.pred.zz",
    "stress.pred.xy", "stress.pred.xz", "stress.pred.yz",
    "has_energy", "has_forces", "has_virial",
]

_OPERATORS = [">", "<", ">=", "<=", "=="]


_FIELD_UNITS = {
    "natoms": "count",
    "volume": "Å³",
    "a": "Å", "b": "Å", "c": "Å",
    "energy": "eV", "energy_per_atom": "eV/atom",
    "force.ref.max": "eV/Å", "force.ref.avg": "eV/Å", "force.ref.norm": "eV/Å",
    "force.pred.max": "eV/Å", "force.pred.avg": "eV/Å", "force.pred.norm": "eV/Å",
    "force.error.max": "eV/Å", "force.error.avg": "eV/Å", "force.error.norm": "eV/Å",
    "virial.ref.xx": "eV", "virial.ref.yy": "eV", "virial.ref.zz": "eV",
    "virial.ref.xy": "eV", "virial.ref.xz": "eV", "virial.ref.yz": "eV",
    "virial.pred.xx": "eV", "virial.pred.yy": "eV", "virial.pred.zz": "eV",
    "virial.pred.xy": "eV", "virial.pred.xz": "eV", "virial.pred.yz": "eV",
    "stress.ref.xx": "GPa", "stress.ref.yy": "GPa", "stress.ref.zz": "GPa",
    "stress.ref.xy": "GPa", "stress.ref.xz": "GPa", "stress.ref.yz": "GPa",
    "stress.pred.xx": "GPa", "stress.pred.yy": "GPa", "stress.pred.zz": "GPa",
    "stress.pred.xy": "GPa", "stress.pred.xz": "GPa", "stress.pred.yz": "GPa",
    "has_energy": "bool", "has_forces": "bool", "has_virial": "bool",
}


class _ConditionRow:
    """A single condition row (field + op + value + unit) within a group."""

    def __init__(self, panel: QFrame, field: str = "", op: str = ">", value: str = ""):
        self._field = field
        self._op = op
        self._value = value
        self._panel = panel

    def build_widgets(self, layout: QHBoxLayout):
        self._field_combo = ComboBox(self._panel)
        self._field_combo.addItems(_EXPRESSION_FIELDS)
        if self._field in _EXPRESSION_FIELDS:
            self._field_combo.setCurrentText(self._field)
        self._field_combo.setFixedWidth(140)

        self._unit_label = BodyLabel("", self._panel)
        self._unit_label.setFixedWidth(55)
        self._unit_label.setStyleSheet("color: #888; font-size: 11px;")

        self._op_combo = ComboBox(self._panel)
        self._op_combo.addItems(_OPERATORS)
        if self._op in _OPERATORS:
            self._op_combo.setCurrentText(self._op)
        self._op_combo.setFixedWidth(60)

        self._value_edit = LineEdit(self._panel)
        self._value_edit.setPlaceholderText(self._panel.tr("value"))
        self._value_edit.setFixedWidth(80)
        self._value_edit.setText(self._value)

        remove_btn = TransparentToolButton(FluentIcon.CLOSE, self._panel)
        remove_btn.setToolTip(self._panel.tr("Remove condition"))

        layout.addWidget(self._field_combo)
        layout.addWidget(self._op_combo)
        layout.addWidget(self._value_edit)
        layout.addWidget(self._unit_label)
        layout.addWidget(remove_btn)
        layout.addStretch()

        self._field_combo.currentIndexChanged.connect(lambda: self._update_unit())
        self._update_unit()

        return remove_btn

    def _update_unit(self):
        f = self._field_combo.currentText().strip()
        unit = _FIELD_UNITS.get(f, "")
        self._unit_label.setText(unit)

    def get_field(self) -> str:
        return self._field_combo.currentText().strip()

    def get_op(self) -> str:
        return self._op_combo.currentText().strip()

    def get_value(self) -> str:
        return self._value_edit.text().strip()

    def is_complete(self) -> bool:
        return bool(self.get_field() and self.get_op() and self.get_value())

    def connect_signals(self, slot):
        self._field_combo.currentIndexChanged.connect(slot)
        self._op_combo.currentIndexChanged.connect(slot)
        self._value_edit.textChanged.connect(slot)


class _ExpressionGroupPanel(QFrame):
    """Expression group: multiple conditions OR'd. Groups are AND'd."""

    removed = Signal(object)
    changed = Signal()

    def __init__(self, conditions: list[tuple[str, str, str]] | None = None,
                 negate: bool = False, parent=None):
        super().__init__(parent)
        self._negate = negate
        self._rows: list[_ConditionRow] = []
        self._row_layouts: list[QHBoxLayout] = []
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._build_ui()
        if conditions:
            for field, op, value in conditions:
                self._add_row(field, op, value)
        if not self._rows:
            self._add_row()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)

        header = QHBoxLayout()
        header.setSpacing(8)
        self._mode_combo = ComboBox(self)
        self._mode_combo.addItem(self.tr("Include"), userData=False)
        self._mode_combo.addItem(self.tr("Exclude"), userData=True)
        self._mode_combo.setCurrentIndex(1 if self._negate else 0)
        self._mode_combo.currentIndexChanged.connect(self.changed.emit)
        header.addWidget(self._mode_combo)
        header.addStretch()
        remove_btn = TransparentToolButton(FluentIcon.CLOSE, self)
        remove_btn.setToolTip(self.tr("Remove group"))
        remove_btn.clicked.connect(lambda: self.removed.emit(self))
        header.addWidget(remove_btn)
        layout.addLayout(header)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(3)
        layout.addLayout(self._rows_layout)

        add_btn = TransparentToolButton(FluentIcon.ADD, self)
        add_btn.setToolTip(self.tr("Add condition"))
        add_btn.clicked.connect(lambda: self._add_row())
        layout.addWidget(add_btn)

    def _add_row(self, field: str = "", op: str = ">", value: str = ""):
        row = _ConditionRow(self, field, op, value)
        row_layout = QHBoxLayout()
        row_layout.setSpacing(4)
        remove_btn = row.build_widgets(row_layout)
        row.connect_signals(self.changed.emit)
        insert = len(self._rows)
        self._rows_layout.insertLayout(insert, row_layout)
        self._rows.append(row)
        self._row_layouts.append(row_layout)

        def _on_remove():
            idx = self._rows.index(row) if row in self._rows else -1
            if idx >= 0:
                self._remove_row(idx)

        remove_btn.clicked.connect(_on_remove)
        self.changed.emit()

    def _remove_row(self, idx: int):
        rl = self._row_layouts[idx]
        while rl.count():
            w = rl.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._rows_layout.removeItem(rl)
        self._rows.pop(idx)
        self._row_layouts.pop(idx)
        if not self._rows:
            self._add_row()
        self.changed.emit()

    def get_conditions(self) -> list[tuple[str, str, str]]:
        return [(r.get_field(), r.get_op(), r.get_value()) for r in self._rows if r.is_complete()]

    def get_negate(self) -> bool:
        data = self._mode_combo.itemData(self._mode_combo.currentIndex())
        return bool(data) if data is not None else False

    def has_any(self) -> bool:
        return any(r.is_complete() for r in self._rows)


class ExpressionFilterDialog(QDialog):
    """Dialog for building expression-based structure filter."""

    filterChanged = Signal(str)

    def __init__(self, expression: str, parent=None):
        super().__init__(parent)
        self._result_expression: str = expression
        self._panels: list[_ExpressionGroupPanel] = []
        self._build_ui()
        self._parse_expression(expression)

    @staticmethod
    def _tokenize(expr: str) -> list[tuple[bool, list[tuple[str, str, str]]]]:
        """Parse expression into groups: [(negate, [(field, op, value)])]."""

        def _split_top_level(s: str, sep: str) -> list[str]:
            parts: list[str] = []
            depth = 0
            start = 0
            for i, c in enumerate(s):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif depth == 0 and s.startswith(sep, i):
                    parts.append(s[start:i])
                    start = i + len(sep)
            parts.append(s[start:])
            return [p.strip() for p in parts if p.strip()]

        def _strip_outer_parens(s: str) -> str:
            s = s.strip()
            while s.startswith("(") and s.endswith(")"):
                inner = s[1:-1].strip()
                if inner.count("(") == inner.count(")"):
                    s = inner
                else:
                    break
            return s

        results: list[tuple[bool, list[tuple[str, str, str]]]] = []
        if not expr.strip():
            return results

        expr = expr.replace("&&", " and ").replace("||", " or ")
        groups = _split_top_level(expr, " and ")

        for group in groups:
            negate = False
            group = group.strip()
            if group.lower().startswith("not "):
                negate = True
                group = group[3:].strip()
            elif group.lower().startswith("not("):
                negate = True
                group = group[3:].strip()
            group = _strip_outer_parens(group)

            conditions: list[tuple[str, str, str]] = []
            for cond in _split_top_level(group, " or "):
                cond = _strip_outer_parens(cond)
                match = re.search(r"(>=|<=|==|>|<)", cond)
                if not match:
                    continue
                op = match.group(1)
                field = cond[:match.start()].strip()
                value = cond[match.end():].strip()
                if field and op and value:
                    conditions.append((field, op, value))

            if conditions:
                results.append((negate, conditions))

        return results

    def _build_expression(self) -> str:
        groups = []
        for panel in self._panels:
            conds = panel.get_conditions()
            if not conds:
                continue
            parts = [f"{f} {op} {v}" for f, op, v in conds]
            if len(parts) == 1:
                inner = parts[0]
            else:
                inner = "(" + " or ".join(parts) + ")"
            if panel.get_negate():
                inner = f"not ({inner})"
            groups.append(inner)
        return " and ".join(groups) if groups else ""

    def _parse_expression(self, expr: str):
        groups = self._tokenize(expr)
        if groups:
            for negate, conditions in groups:
                self._add_panel(conditions, negate)
        else:
            self._add_panel()

    def _build_ui(self):
        self.setWindowTitle(self.tr("Expression Filter"))
        self.resize(500, 380)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        hint = BodyLabel(self.tr("Groups = AND  |  Rows in group = OR"), self)
        hint.setStyleSheet("color: #888; font-size: 12px; padding: 2px 0;")
        layout.addWidget(hint)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._groups_widget = QWidget()
        self._groups_layout = QVBoxLayout(self._groups_widget)
        self._groups_layout.setSpacing(8)
        self._groups_layout.addStretch()
        self._scroll.setWidget(self._groups_widget)
        layout.addWidget(self._scroll, 1)

        add_btn = PushButton(self.tr("+ Add Condition"), self)
        add_btn.clicked.connect(self._add_panel)
        layout.addWidget(add_btn)

        self._preview_label = BodyLabel(self)
        self._preview_label.setStyleSheet("color: #666; padding: 4px;")
        self._preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._preview_label)

        self._update_preview()

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = PushButton(self.tr("Cancel"), self)
        cancel_btn.clicked.connect(self.reject)
        clear_btn = PushButton(self.tr("Clear All"), self)
        clear_btn.clicked.connect(self._clear_all)
        apply_btn = PrimaryPushButton(self.tr("Apply"), self)
        apply_btn.clicked.connect(self._apply)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addWidget(apply_btn)
        layout.addLayout(btn_layout)

    def _add_panel(self, conditions: list[tuple[str, str, str]] | None = None, negate: bool = False):
        panel = _ExpressionGroupPanel(conditions, negate, self)
        panel.removed.connect(lambda p=panel: self._on_removed(p))
        panel.changed.connect(self._update_preview)
        insert = len(self._panels)
        self._groups_layout.insertWidget(insert, panel)
        self._panels.append(panel)
        self._update_preview()

    def _on_removed(self, panel: _ExpressionGroupPanel):
        self._groups_layout.removeWidget(panel)
        panel.deleteLater()
        self._panels.remove(panel)
        if not self._panels:
            self._add_panel()
        self._update_preview()

    def _update_preview(self):
        expr = self._build_expression()
        self._preview_label.setText(expr if expr else self.tr("(empty)"))

    def _clear_all(self):
        for p in list(self._panels):
            self._groups_layout.removeWidget(p)
            p.deleteLater()
        self._panels.clear()
        self._add_panel()
        self._update_preview()

    def _apply(self):
        self._result_expression = self._build_expression()
        self.filterChanged.emit(self._result_expression)
        self.accept()

    def result(self) -> str:
        return getattr(self, "_result_expression", "")

