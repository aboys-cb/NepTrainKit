"""Visual editor for simple atom predicates, stored as existing expressions."""

from __future__ import annotations

import math
import re
from collections import Counter

from PySide6.QtCore import QCoreApplication, QLocale, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QGridLayout, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, ComboBox, LineEdit

from NepTrainKit.core.structure import atomic_numbers
from NepTrainKit.core.types import SearchType


def _tr(text: str) -> str:
    return QCoreApplication.translate("AtomicFilterEditor", text)


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_FIELD = r"(?:atomic\.[a-zA-Z_]\w*(?:\.\w+)?|(?:force|mforce|bec)(?:\.(?:ref|pred|err))?(?:\.\w+)?)"
_COMPARISON = rf"(?P<field>{_FIELD})\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<value>{_NUMBER})"


def parse_atomic_condition(expression: str) -> dict[str, str] | None:
    """Recognize only the simple forms this editor can reproduce losslessly."""
    patterns = (
        rf'(?P<quantifier>any|all)\(\s*{_COMPARISON}\s*\)',
        rf'(?P<quantifier>any)\(\s*atomic\.element\s*==\s*"(?P<element>[A-Z][a-z]?)"\s*&&\s*{_COMPARISON}\s*\)',
        rf'has\.(?P<element>[A-Z][a-z]?)\s*&&\s*(?P<quantifier>all)\(\s*atomic\.element\s*!=\s*"(?P=element)"\s*\|\|\s*{_COMPARISON}\s*\)',
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, expression.strip())
        if match is not None:
            values = match.groupdict()
            values.setdefault("element", "")
            if values['element'] and values['element'] not in atomic_numbers:
                return None
            if values['field'].split('.')[:2] == ['atomic', 'element'] or not math.isfinite(float(values['value'])):
                return None
            return values
    return None


def atomic_condition_expression(values: dict[str, str]) -> str:
    field, threshold = values['field'], values['value'].strip()
    if not field or not re.fullmatch(_NUMBER, threshold) or not math.isfinite(float(threshold)):
        return ""
    predicate = f"{field} {values['op']} {threshold}"
    element, quantifier = values['element'], values['quantifier']
    if not element:
        return f"{quantifier}({predicate})"
    if quantifier == 'any':
        return f'any(atomic.element == "{element}" && {predicate})'
    return f'has.{element} && all(atomic.element != "{element}" || {predicate})'


def atomic_field_label(field: str) -> str:
    """Readable labels for familiar fields; preserve custom property names."""
    names = {
        'spin': _tr('Magnetic moment'), 'spin_vec': _tr('Magnetic moment'),
        'spin_scalar': _tr('Magnetic moment'), 'magmom': _tr('Magnetic moment'),
        'force': _tr('Force'), 'forces': _tr('Force'),
        'mforce': _tr('Magnetic force'), 'force_mag': _tr('Magnetic force'),
        'charge': _tr('Charge'), 'bec': _tr('Born effective charge'),
    }
    parts = field.split('.')
    atomic = parts[0] == 'atomic'
    if atomic:
        parts = parts[1:]
    name = names.get(parts[0], parts[0])
    tail = parts[1:]
    view_names = {'ref': _tr('Reference'), 'pred': _tr('Prediction'), 'err': _tr('Error')}
    if not atomic:
        view = tail.pop(0) if tail and tail[0] in view_names else 'ref'
        name = f"{name} · {view_names[view]}"
    if tail:
        component = _tr('Magnitude') if tail[0] == 'norm' else tail[0]
        name = f"{name} · {component}"
    return name


def atomic_condition_summary(values: dict[str, str]) -> str:
    quantifier = _tr('Any') if values['quantifier'] == 'any' else _tr('All')
    element = values['element'] or _tr('atoms')
    return f"{quantifier} {element} · {atomic_field_label(values['field'])} {values['op']} {values['value']}"


class AtomicFilterEditor(QWidget):
    """Two rows of controls plus a plain-language scope hint."""

    changed = Signal()

    def __init__(self, suggestions=None, values=None, parent=None):
        super().__init__(parent)
        self._syncing = True
        self.element_combo = ComboBox(self)
        self.quantifier_combo = ComboBox(self)
        self.quantifier_combo.addItem(self.tr('At least one atom'), userData='any')
        self.quantifier_combo.addItem(self.tr('All target atoms'), userData='all')
        self.field_combo = ComboBox(self)
        self.operator_combo = ComboBox(self)
        for label, op in ((self.tr('Less than'), '<'), (self.tr('At most'), '<='),
                          (self.tr('Greater than'), '>'), (self.tr('At least'), '>='),
                          (self.tr('Equal to'), '=='), (self.tr('Not equal to'), '!=')):
            self.operator_combo.addItem(f'{label} {op}', userData=op)
        self.value_edit = LineEdit(self)
        self.value_edit.setText('1')
        self.value_edit.setPlaceholderText(self.tr('Threshold'))
        validator = QDoubleValidator(self.value_edit)
        validator.setLocale(QLocale.c())
        self.value_edit.setValidator(validator)
        self.value_edit.setMinimumWidth(70)
        self.unit_label = CaptionLabel('', self)
        self.hint_label = CaptionLabel('', self)
        self.hint_label.setWordWrap(True)
        for widget in (self.element_combo, self.quantifier_combo, self.field_combo, self.operator_combo, self.value_edit):
            widget.setMinimumHeight(32)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # QPushButton's text-based minimum size (inherited by Fluent ComboBox)
        # can force the entire popup beyond its viewport on Windows or with
        # long custom property names. Let the grid allocate the available width.
        for combo in (self.element_combo, self.quantifier_combo, self.field_combo, self.operator_combo):
            combo.setMinimumWidth(80)
            combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(8)
        scope = QGridLayout()
        scope.setHorizontalSpacing(12)
        scope.setVerticalSpacing(4)
        controls = ((self.tr('Target element'), self.element_combo), (self.tr('Match'), self.quantifier_combo))
        for column, (label, widget) in enumerate(controls):
            caption = CaptionLabel(label, self)
            caption.setBuddy(widget)
            widget.setAccessibleName(label)
            scope.addWidget(caption, 0, column)
            scope.addWidget(widget, 1, column)
            scope.setColumnStretch(column, 1)
        layout.addLayout(scope)
        numeric = QGridLayout()
        numeric.setHorizontalSpacing(8)
        numeric.setVerticalSpacing(4)
        for column, (label, widget) in enumerate(((self.tr('Property'), self.field_combo),
                                                 (self.tr('Comparison'), self.operator_combo),
                                                 (self.tr('Threshold'), self.value_edit))):
            caption = CaptionLabel(label, self)
            caption.setBuddy(widget)
            widget.setAccessibleName(label)
            numeric.addWidget(caption, 0, column)
            numeric.addWidget(widget, 1, column)
        numeric.addWidget(self.unit_label, 1, 3)
        numeric.setColumnStretch(0, 3)
        numeric.setColumnStretch(1, 2)
        numeric.setColumnStretch(2, 1)
        layout.addLayout(numeric)
        layout.addWidget(self.hint_label)
        self.set_suggestions(suggestions or {})
        if values:
            self.set_values(values)
        for combo in (self.element_combo, self.quantifier_combo, self.field_combo, self.operator_combo):
            combo.currentIndexChanged.connect(self._changed)
        self.value_edit.textChanged.connect(self._changed)
        self._syncing = False
        self._update_hint()

    @staticmethod
    def _select(combo, data, label=None):
        for index in range(combo.count()):
            if combo.itemData(index) == data:
                combo.setCurrentIndex(index)
                return
        if data:
            combo.addItem(label or data, userData=data)
            combo.setCurrentIndex(combo.count() - 1)

    def set_suggestions(self, suggestions):
        was_syncing = self._syncing
        self._syncing = True
        old_element, old_field = self.element_combo.currentData(), self.field_combo.currentData()
        self.element_combo.clear()
        self.element_combo.addItem(self.tr('All elements'), userData='')
        for symbol in sorted(suggestions.get(SearchType.ELEMENTS, {})):
            if symbol in atomic_numbers:
                self.element_combo.addItem(symbol, userData=symbol)
        self._select(self.element_combo, old_element)
        tokens = set(suggestions.get(SearchType.EXPRESSION, {}))
        fields = []
        for token in tokens:
            if not re.fullmatch(_FIELD, token) or token.split('.')[:2] == ['atomic', 'element']:
                continue
            if token + '.norm' in tokens:
                continue  # A vector needs an explicit component.
            if not token.startswith('atomic.'):
                parts = token.split('.')
                if len(parts) > 1 and parts[1] == 'ref':
                    short = '.'.join([parts[0], *parts[2:]])
                    if short in tokens:
                        continue
                if len(parts) == 1 or (len(parts) == 2 and parts[1] in {'ref', 'pred', 'err'}):
                    token += '.norm'
            fields.append(token)
        preferred = ('atomic.spin.norm', 'atomic.spin_vec.norm', 'atomic.magmom.norm', 'force.norm')
        fields = sorted(set(fields), key=lambda field: (preferred.index(field) if field in preferred else len(preferred), field))
        self.field_combo.clear()
        labels = Counter(atomic_field_label(field) for field in fields)
        for field in fields:
            label = atomic_field_label(field)
            if labels[label] > 1:
                property_name = field.split('.')[1] if field.startswith('atomic.') else field.split('.')[0]
                label = f'{label} ({property_name})'
            self.field_combo.addItem(label, userData=field)
        self._select(self.field_combo, old_field, atomic_field_label(old_field) if old_field else None)
        self._syncing = was_syncing
        self._update_hint()

    def values(self):
        return dict(element=self.element_combo.currentData() or '', quantifier=self.quantifier_combo.currentData(),
                    field=self.field_combo.currentData() or '', op=self.operator_combo.currentData(),
                    value=self.value_edit.text())

    def set_values(self, values):
        self._syncing = True
        self._select(self.element_combo, values['element'])
        self._select(self.quantifier_combo, values['quantifier'])
        self._select(self.field_combo, values['field'], atomic_field_label(values['field']))
        self._select(self.operator_combo, values['op'])
        self.value_edit.setText(values['value'])
        self._syncing = False
        self._update_hint()

    def expression(self):
        return atomic_condition_expression(self.values())

    def _changed(self, *_args):
        if not self._syncing:
            self._update_hint()
            self.changed.emit()

    def _update_hint(self):
        for combo in (self.element_combo, self.quantifier_combo, self.field_combo, self.operator_combo):
            combo.setToolTip(combo.currentText())
        if not self.field_combo.currentData():
            hint = self.tr('Load a dataset with atomic properties to choose a field.')
        elif self.element_combo.currentData():
            hint = self.tr('Only {element} atoms are checked. Structures without {element} do not match.').format(element=self.element_combo.currentData())
        else:
            hint = self.tr('The condition is checked for each atom in the structure.')
        self.hint_label.setText(hint)
