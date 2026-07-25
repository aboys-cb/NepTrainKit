"""Compact composite structure filter used by the Show NEP page."""

from __future__ import annotations

import re
import uuid

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCompleter,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CaptionLabel,
    ComboBox,
    DropDownPushButton,
    FluentIcon,
    LineEdit,
    MessageBox,
    MessageBoxBase,
    PrimaryPushButton,
    PushButton,
    RoundMenu,
    StrongBodyLabel,
    SwitchButton,
    ToolButton,
    TogglePushButton,
    TransparentToolButton,
    isDarkTheme,
    qconfig,
    themeColor,
)
from qfluentwidgets.components.widgets.line_edit import CompleterMenu

from NepTrainKit.core.filter_presets import (
    delete_structure_filter_preset,
    list_structure_filter_preset_names,
    load_structure_filter_preset,
    rename_structure_filter_preset,
    save_structure_filter_preset,
    structure_filter_preset_exists,
)
from NepTrainKit.core.message import MessageManager
from NepTrainKit.core.search import StructureFilterValidationError
from NepTrainKit.core.types import (
    FilterField,
    FilterLogic,
    SearchType,
    StructureFilterCondition,
    StructureFilterSpec,
    TextMatchMode,
)
from NepTrainKit.ui.widgets.completer import CompleterModel, JoinDelegate


_TEXT_FIELDS = {FilterField.CONFIG_TYPE, FilterField.FORMULA}
_ELEMENT_FIELDS = {
    FilterField.ELEMENT_REQUIRED,
    FilterField.ELEMENT_EXCLUDED,
    FilterField.ELEMENT_ALLOWED,
}

_EXPRESSION_FIELD_UNITS: dict[str, str] = {
    "natoms": "count",
    "n_atoms": "count",
    "spin_natoms": "count",
    "volume": "Å³",
    "a": "Å",
    "b": "Å",
    "c": "Å",
    "alpha": "°",
    "beta": "°",
    "gamma": "°",
    "energy": "eV",
    "energy_per_atom": "eV/atom",
    "has_energy": "bool",
    "has_forces": "bool",
    "has_virial": "bool",
    "has_bec": "bool",
}

_EXPRESSION_UNIT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("force", "eV/Å"),
    ("stress", "GPa"),
    ("virial", "eV"),
    ("count.", "count"),
    ("frac.", "fraction"),
    ("has.", "bool"),
)


_ATOMIC_PROPERTY_UNITS: dict[str, str] = {
    "force": "eV/Å",
    "forces": "eV/Å",
    "pos": "Å",
    "spin_vec": "μB",
    "spin_scalar": "μB",
    "force_mag": "μB/Å",
    "mforce": "μB/Å",
    "magmom": "μB",
    "charge": "e",
    "bec": "e",
    "dipole": "e·Å",
    "velocity": "Å/fs",
}


def _detect_expression_unit(text: str) -> str:
    """Return the unit for the first field found in an expression string."""
    match = re.search(r"([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)", text)
    if not match:
        return ""
    field = match.group(1)
    unit = _EXPRESSION_FIELD_UNITS.get(field)
    if unit is not None:
        return unit
    for prefix, prefix_unit in _EXPRESSION_UNIT_PREFIXES:
        if field.startswith(prefix):
            return prefix_unit
    if field.startswith("atomic."):
        parts = field.split(".", 2)
        if len(parts) >= 2:
            prop = parts[1]
            return _ATOMIC_PROPERTY_UNITS.get(prop, "")
    return ""


def _surface_colors() -> tuple[str, str, str, str]:
    """Return theme-aware surface, border, text, and muted-text colors."""
    if isDarkTheme():
        return "#2b2b2b", "#555555", "#f2f2f2", "#a6a6a6"
    return "#ffffff", "#d0d0d0", "#202020", "#707070"


class FilterChip(QFrame):
    """One compact condition summary with independent edit and remove actions."""

    editRequested = Signal(str)
    removeRequested = Signal(str)

    def __init__(self, condition_id: str, kind: str, value: str, enabled: bool, parent=None):
        super().__init__(parent)
        self.condition_id = condition_id
        self.full_text = self.tr("{kind}: {value}").format(kind=kind, value=value)
        self.setObjectName("structureFilterChip")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(9, 0, 3, 0)
        layout.setSpacing(5)
        self.kind_label = CaptionLabel(self.tr("{kind}:").format(kind=kind), self)
        self.value_label = BodyLabel(value, self)
        self.value_label.setMaximumWidth(92)
        tooltip = self.full_text if enabled else self.tr("Disabled: {text}").format(text=self.full_text)
        self.setToolTip(tooltip)
        layout.addWidget(self.kind_label)
        layout.addWidget(self.value_label)
        close = TransparentToolButton(FluentIcon.CLOSE, self)
        close.setIconSize(QSize(10, 10))
        close.setFixedSize(18, 24)
        close.setToolTip(self.tr("Remove condition"))
        close.setAccessibleName(self.tr("Remove condition"))
        close.clicked.connect(lambda: self.removeRequested.emit(self.condition_id))
        layout.addWidget(close)
        self.setProperty("disabledCondition", not enabled)
        self._refresh_style()

    def _refresh_style(self) -> None:
        _, _, text, muted = _surface_colors()
        accent = themeColor().name()
        label_color = muted if self.property("disabledCondition") else text
        chip_surface = "#292929" if isDarkTheme() else "#fcfdff"
        chip_border = "#4b4b4b" if isDarkTheme() else "#d8dee8"
        self.setStyleSheet(
            "QFrame#structureFilterChip {"
            f" background: {chip_surface}; border: 1px solid {chip_border}; border-radius: 7px; color: {text}; }}"
            f"QFrame#structureFilterChip:hover {{ border-color: {accent}; }}"
            f"QFrame#structureFilterChip[disabledCondition='true'] {{ color: {muted}; }}"
            f"QFrame#structureFilterChip QLabel {{ border: none; background: transparent; color: {label_color}; }}"
        )
        self.kind_label.setStyleSheet(f"color: {muted}; border: none; background: transparent;")
        self.value_label.setStyleSheet(f"color: {label_color}; border: none; background: transparent; font-weight: 500;")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.editRequested.emit(self.condition_id)
        super().mouseReleaseEvent(event)


class _SuggestionLineEdit(LineEdit):
    """Fluent line edit with frequency-aware, token-aware suggestions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suggestion_data: dict[str, int] = {}
        self._token_separators = ""
        self._suggestion_model = CompleterModel({}, self)
        completer = QCompleter(self._suggestion_model, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setMaxVisibleItems(8)
        self.setCompleter(completer)
        menu = CompleterMenu(self)
        self.setCompleterMenu(menu)
        self._suggestion_delegate = JoinDelegate(self, {})
        menu.view.setItemDelegate(self._suggestion_delegate)
        menu.view.setMaxVisibleItems(8)
        completer.activated[str].connect(self._accept_completion)

    def set_suggestions(self, data: dict[str, int] | None, *, token_separators: str = "") -> None:
        self._suggestion_data = data or {}
        self._token_separators = token_separators
        self._suggestion_model.set_data(self._suggestion_data)
        self._suggestion_delegate.data = self._suggestion_data
        if not self._suggestion_data and self._completerMenu is not None:
            self._completerMenu.close()

    def _token_start(self) -> int:
        cursor = self.cursorPosition()
        text = self.text()
        start = cursor
        while start > 0 and text[start - 1] not in self._token_separators:
            start -= 1
        return start

    def _completion_prefix(self) -> str:
        return self.text()[self._token_start() : self.cursorPosition()].strip()

    def _showCompleterMenu(self) -> None:
        if not self._suggestion_data or self.completer() is None:
            return
        self.completer().setCompletionPrefix(self._completion_prefix())
        changed = self._completerMenu.setCompletion(
            self.completer().completionModel(),
            self.completer().completionColumn(),
        )
        self._completerMenu.setMaxVisibleItems(self.completer().maxVisibleItems())
        if changed:
            self._completerMenu.popup()

    def _accept_completion(self, value: str) -> None:
        value = str(value or "")
        if not value:
            return
        cursor = self.cursorPosition()
        start = self._token_start()
        text = self.text()
        segment = text[start:cursor]
        leading_space = segment[: len(segment) - len(segment.lstrip())]
        updated = f"{text[:start]}{leading_space}{value}{text[cursor:]}"
        self.setText(updated)
        self.setCursorPosition(start + len(leading_space) + len(value))

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self._suggestion_data:
            QTimer.singleShot(0, self._showCompleterMenu)


class _ConditionRow(QFrame):
    """Editable row that maps directly to one typed condition."""

    changed = Signal()
    removeRequested = Signal(object)

    def __init__(
        self,
        condition: StructureFilterCondition,
        suggestions: dict[SearchType, dict[str, int]] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.condition_id = condition.condition_id
        self._suggestions = suggestions or {}
        self._error = False
        self._input_hint = ""
        self.setObjectName("structureFilterConditionRow")

        self.enabled_switch = SwitchButton(self)
        self.enabled_switch.setOnText("")
        self.enabled_switch.setOffText("")
        self.enabled_switch.setFixedWidth(self.enabled_switch.sizeHint().width())
        self.enabled_switch.setChecked(condition.enabled)
        self.enabled_switch.setAccessibleName(self.tr("Enable condition"))

        self.field_combo = ComboBox(self)
        self.field_combo.setFixedWidth(112)
        self.field_combo.setFixedHeight(30)
        self.field_combo.addItem(self.tr("Config type"), userData=FilterField.CONFIG_TYPE)
        self.field_combo.addItem(self.tr("Formula"), userData=FilterField.FORMULA)
        self.field_combo.addItem(self.tr("Elements"), userData=FilterField.ELEMENT_REQUIRED)
        self.field_combo.addItem(self.tr("Custom expression"), userData=FilterField.EXPRESSION)

        self.mode_combo = ComboBox(self)
        self.mode_combo.setFixedWidth(124)
        self.mode_combo.setFixedHeight(30)

        self.case_button = TogglePushButton("Aa", self)
        self.case_button.setCheckable(True)
        self.case_button.setFixedSize(max(44, self.case_button.sizeHint().width()), 28)
        self.case_button.setAccessibleName(self.tr("Match case"))

        self.value_edit = _SuggestionLineEdit(self)
        self.value_edit.setClearButtonEnabled(True)
        self.value_edit.setFixedHeight(30)
        self.value_edit.setMinimumWidth(120)
        self.value_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.unit_label = CaptionLabel("", self)
        self.unit_label.setFixedHeight(30)
        self.unit_label.setMinimumWidth(0)
        self.unit_label.setMaximumWidth(64)
        self.unit_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.unit_label.setStyleSheet("color: #888; font-size: 11px;")
        self.unit_label.setVisible(False)

        self.remove_button = TransparentToolButton(FluentIcon.CLOSE, self)
        self.remove_button.setIconSize(QSize(10, 10))
        self.remove_button.setFixedSize(24, 28)
        self.remove_button.setToolTip(self.tr("Remove condition"))
        self.remove_button.setAccessibleName(self.tr("Remove condition"))
        self.remove_button.clicked.connect(lambda: self.removeRequested.emit(self))

        widest_field = max(
            self.field_combo.fontMetrics().horizontalAdvance(self.field_combo.itemText(index))
            for index in range(self.field_combo.count())
        )
        self._wrapped_layout = widest_field + 46 > 176
        self.setFixedHeight(68 if self._wrapped_layout else 34)
        if self._wrapped_layout:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(2, 1, 2, 1)
            layout.setSpacing(2)
            selector_row = QHBoxLayout()
            selector_row.setSpacing(4)
            selector_row.addWidget(self.enabled_switch)
            selector_row.addWidget(self.field_combo)
            selector_row.addWidget(self.mode_combo)
            selector_row.addWidget(self.case_button)
            selector_row.addStretch()
            selector_row.addWidget(self.remove_button)
            layout.addLayout(selector_row)
            value_row = QHBoxLayout()
            value_row.setSpacing(4)
            value_row.addWidget(self.value_edit, 1)
            value_row.addWidget(self.unit_label)
            layout.addLayout(value_row)
        else:
            layout = QHBoxLayout(self)
            layout.setContentsMargins(2, 1, 2, 1)
            layout.setSpacing(4)
            layout.addWidget(self.enabled_switch)
            layout.addWidget(self.field_combo)
            layout.addWidget(self.mode_combo)
            layout.addWidget(self.case_button)
            layout.addWidget(self.value_edit, 1)
            layout.addWidget(self.unit_label)
            layout.addWidget(self.remove_button)

        initial_field = condition.field
        if initial_field in _ELEMENT_FIELDS:
            initial_field = FilterField.ELEMENT_REQUIRED
        for index in range(self.field_combo.count()):
            if self.field_combo.itemData(index) == initial_field:
                self.field_combo.setCurrentIndex(index)
                break
        self._fit_current_combo_width(self.field_combo, 112, padding=46)
        self._rebuild_modes(condition.field, condition.match_mode)
        self._update_case_button(condition.case_sensitive)
        self.value_edit.setText(self._display_values(condition))
        self._update_placeholder()
        self._update_suggestions()

        self.enabled_switch.checkedChanged.connect(lambda _checked: self.changed.emit())
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        self.mode_combo.currentIndexChanged.connect(lambda _index: self.changed.emit())
        self.case_button.toggled.connect(self._on_case_toggled)
        self.value_edit.textChanged.connect(self._on_value_changed)
        self._refresh_style()
        self._update_unit_label()

    @staticmethod
    def _display_values(condition: StructureFilterCondition) -> str:
        if condition.field in _ELEMENT_FIELDS:
            return ", ".join(condition.text_values)
        if condition.field == FilterField.EXPRESSION:
            return condition.text_values[0] if condition.text_values else ""
        return "; ".join(condition.text_values)

    def _on_field_changed(self, _index=None):
        field = self.field_combo.currentData()
        self._fit_current_combo_width(self.field_combo, 112, padding=46)
        self._rebuild_modes(field, None)
        self._update_case_button(field == FilterField.FORMULA)
        self._update_placeholder()
        self._update_suggestions()
        self._update_unit_label()
        self.clear_error()
        self.changed.emit()

    def _update_unit_label(self) -> None:
        field = self.field_combo.currentData()
        if field != FilterField.EXPRESSION:
            self.unit_label.setVisible(False)
            self.unit_label.setText("")
            return
        unit = _detect_expression_unit(self.value_edit.text())
        if unit:
            self.unit_label.setText(unit)
            self.unit_label.setVisible(True)
        else:
            self.unit_label.setText("")
            self.unit_label.setVisible(False)

    def _on_case_toggled(self, _checked: bool) -> None:
        self._update_case_tooltip()
        self.changed.emit()

    def _on_value_changed(self, _text: str) -> None:
        self._update_unit_label()
        self.changed.emit()

    def _update_case_button(self, checked: bool) -> None:
        is_text = self.field_combo.currentData() in _TEXT_FIELDS
        self.case_button.setVisible(is_text)
        self.case_button.blockSignals(True)
        self.case_button.setChecked(bool(checked) if is_text else False)
        self.case_button.blockSignals(False)
        self._update_case_tooltip()

    def _update_case_tooltip(self) -> None:
        if self.case_button.isChecked():
            tooltip = self.tr("Match case: on")
        else:
            tooltip = self.tr("Match case: off")
        self.case_button.setToolTip(tooltip)

    def _rebuild_modes(self, field: FilterField, selected) -> None:
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        if field == FilterField.CONFIG_TYPE:
            options = (
                (self.tr("Contains"), TextMatchMode.CONTAINS),
                (self.tr("Equals"), TextMatchMode.EXACT),
                (self.tr("Starts with"), TextMatchMode.PREFIX),
                (self.tr("Ends with"), TextMatchMode.SUFFIX),
                (self.tr("Regex"), TextMatchMode.REGEX),
            )
            selected = selected or TextMatchMode.CONTAINS
        elif field == FilterField.FORMULA:
            options = (
                (self.tr("Equals"), TextMatchMode.EXACT),
                (self.tr("Contains"), TextMatchMode.CONTAINS),
                (self.tr("Starts with"), TextMatchMode.PREFIX),
                (self.tr("Ends with"), TextMatchMode.SUFFIX),
                (self.tr("Regex"), TextMatchMode.REGEX),
            )
            selected = selected or TextMatchMode.EXACT
        elif field in _ELEMENT_FIELDS:
            options = (
                (self.tr("Must contain"), FilterField.ELEMENT_REQUIRED),
                (self.tr("Must not contain"), FilterField.ELEMENT_EXCLUDED),
                (self.tr("Allow only"), FilterField.ELEMENT_ALLOWED),
            )
            selected = selected if selected in _ELEMENT_FIELDS else field
        else:
            options = ((self.tr("Expression"), FilterField.EXPRESSION),)
            selected = FilterField.EXPRESSION
        for text, value in options:
            self.mode_combo.addItem(text, userData=value)
        for index in range(self.mode_combo.count()):
            if self.mode_combo.itemData(index) == selected:
                self.mode_combo.setCurrentIndex(index)
                break
        self._fit_combo_width(self.mode_combo, 124, 280)
        self.mode_combo.blockSignals(False)

    @staticmethod
    def _fit_current_combo_width(combo: ComboBox, minimum: int, *, padding: int = 42) -> None:
        text_width = combo.fontMetrics().horizontalAdvance(combo.currentText())
        combo.setFixedWidth(max(minimum, text_width + padding))

    @staticmethod
    def _fit_combo_width(combo: ComboBox, minimum: int, maximum: int, *, padding: int = 42) -> None:
        text_width = max(
            (combo.fontMetrics().horizontalAdvance(combo.itemText(index)) for index in range(combo.count())),
            default=0,
        )
        combo.setFixedWidth(max(minimum, min(maximum, text_width + padding)))

    def _update_placeholder(self) -> None:
        field = self.field_combo.currentData()
        if field == FilterField.CONFIG_TYPE:
            placeholder = self.tr("e.g. surface; bulk")
            hint = self.tr("Separate multiple values with ;. Use Aa to control letter case.")
        elif field == FilterField.FORMULA:
            placeholder = self.tr("e.g. Fe2O3; FeO")
            hint = self.tr("Separate multiple values with ;. Use Aa to control letter case.")
        elif field in _ELEMENT_FIELDS:
            placeholder = self.tr("e.g. Fe, O")
            hint = self.tr("Element symbols are normalized (fe → Fe). Use commas or spaces. Example: Fe, O")
        else:
            placeholder = self.tr("e.g. natoms > 100")
            hint = self.tr("Expressions must be conditions. Add a comparison, for example: natoms > 100")
        self._input_hint = hint
        self.value_edit.setPlaceholderText(placeholder)
        self.value_edit.setToolTip(hint)

    def _update_suggestions(self) -> None:
        field = self.field_combo.currentData()
        if field == FilterField.CONFIG_TYPE:
            search_type = SearchType.TAG
            separators = ";"
        elif field == FilterField.FORMULA:
            search_type = SearchType.FORMULA
            separators = ";"
        elif field in _ELEMENT_FIELDS:
            search_type = SearchType.ELEMENTS
            separators = ", \t"
        else:
            search_type = SearchType.EXPRESSION
            separators = " \t()><=!&|+-*/,"
        self.value_edit.set_suggestions(self._suggestions.get(search_type), token_separators=separators)

    def set_suggestions(self, suggestions: dict[SearchType, dict[str, int]]) -> None:
        self._suggestions = suggestions
        self._update_suggestions()

    def to_condition(self) -> StructureFilterCondition:
        selected_field = self.field_combo.currentData()
        mode_data = self.mode_combo.currentData()
        raw = self.value_edit.text().strip()
        if selected_field in _ELEMENT_FIELDS:
            field = mode_data
            values = tuple(value for value in re.split(r"[,\s]+", raw) if value)
            match_mode = None
        elif selected_field in _TEXT_FIELDS:
            field = selected_field
            values = tuple(value.strip() for value in raw.split(";") if value.strip())
            match_mode = mode_data
        else:
            field = FilterField.EXPRESSION
            values = (raw,) if raw else ()
            match_mode = None
        return StructureFilterCondition(
            condition_id=self.condition_id,
            field=field,
            enabled=self.enabled_switch.isChecked(),
            text_values=values,
            match_mode=match_mode,
            case_sensitive=self.case_button.isChecked() if field in _TEXT_FIELDS else False,
        )

    def set_error(self, message: str) -> None:
        self._error = True
        self.value_edit.setToolTip(message)
        self._refresh_style()

    def clear_error(self) -> None:
        self._error = False
        self.value_edit.setToolTip(self._input_hint)
        self._refresh_style()

    def _refresh_style(self) -> None:
        dark = isDarkTheme()
        if self._error:
            border = "#a7545b" if dark else "#d95763"
            background = "#40292c" if dark else "#fff6f6"
        else:
            border = "transparent"
            background = "transparent"
        self.setStyleSheet(
            "QFrame#structureFilterConditionRow {"
            f" background: {background}; border: 1px solid {border}; border-radius: 6px; }}"
        )


class _PresetNameDialog(MessageBoxBase):
    """Small Fluent dialog used only to name or rename a saved filter."""

    def __init__(self, title: str, prompt: str, accept_text: str, initial: str = "", parent=None):
        super().__init__(parent)
        self.title_label = StrongBodyLabel(title, self)
        self.prompt_label = CaptionLabel(prompt, self)
        self.name_edit = LineEdit(self)
        self.name_edit.setMaxLength(80)
        self.name_edit.setText(initial)
        self.name_edit.selectAll()
        self.viewLayout.addWidget(self.title_label)
        self.viewLayout.addWidget(self.prompt_label)
        self.viewLayout.addWidget(self.name_edit)
        self.yesButton.setText(accept_text)
        self.cancelButton.setText(self.tr("Cancel"))
        self.yesButton.setEnabled(bool(initial.strip()))
        self.name_edit.textChanged.connect(lambda text: self.yesButton.setEnabled(bool(text.strip())))
        self.widget.setMinimumWidth(360)

    def value(self) -> str:
        return self.name_edit.text().strip()


class StructureFilterEditorPopup(QFrame):
    """Anchored, non-modal editor for a flat AND/OR condition list."""

    specChanged = Signal(object)
    previewRequested = Signal()
    _MAX_VISIBLE_ROWS = 5
    _ROW_SPACING = 4

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("structureFilterEditorPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlag(Qt.WindowType.NoDropShadowWindowHint, True)
        self.setMinimumWidth(620)
        self.setMaximumWidth(704)
        self._rows: list[_ConditionRow] = []
        self._suggestions: dict[SearchType, dict[str, int]] = {}
        self._syncing = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(280)
        self._debounce.timeout.connect(self.previewRequested.emit)
        self._build_ui()
        qconfig.themeChangedFinished.connect(self._refresh_style)

    def _build_ui(self) -> None:
        shell = QVBoxLayout(self)
        shell.setContentsMargins(10, 8, 10, 14)
        shell.setSpacing(0)
        self.card = QFrame(self)
        self.card.setObjectName("structureFilterEditorCard")
        shell.addWidget(self.card)
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(32, 45, 65, 64))
        self.card.setGraphicsEffect(shadow)

        outer = QVBoxLayout(self.card)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(6)

        self.title_label = StrongBodyLabel(self.tr("Edit structure filter"), self)
        self.preset_button = DropDownPushButton(FluentIcon.FILTER, self.tr("Saved filters"), self)
        self.preset_button.setFixedHeight(30)
        preset_text_width = self.preset_button.fontMetrics().horizontalAdvance(
            self.preset_button.text()
        )
        self.preset_button.setFixedWidth(preset_text_width + 64)
        self.preset_button.setToolTip(self.tr("Load or save frequently used filter conditions"))
        self.preset_button.setAccessibleName(self.tr("Saved filters"))
        self.preset_menu = None
        self.logic_combo = ComboBox(self)
        self.logic_combo.setFixedHeight(30)
        self.logic_combo.addItem(self.tr("Match all conditions (AND)"), userData=FilterLogic.ALL)
        self.logic_combo.addItem(self.tr("Match any condition (OR)"), userData=FilterLogic.ANY)
        logic_text_width = max(
            self.logic_combo.fontMetrics().horizontalAdvance(
                self.logic_combo.itemText(index)
            )
            for index in range(self.logic_combo.count())
        )
        self.logic_combo.setFixedWidth(logic_text_width + 48)
        self.logic_combo.currentIndexChanged.connect(lambda _index: self._emit_spec())

        header_width = self.minimumWidth() - 40
        header_spacing = 6
        single_row_width = (
            self.title_label.sizeHint().width()
            + self.preset_button.width()
            + self.logic_combo.width()
            + 2 * header_spacing
        )
        if single_row_width <= header_width:
            header = QHBoxLayout()
            header.setSpacing(header_spacing)
            header.addWidget(self.title_label)
            header.addStretch()
            header.addWidget(self.preset_button)
            header.addWidget(self.logic_combo)
            outer.addLayout(header)
        else:
            title_row = QHBoxLayout()
            title_row.setSpacing(header_spacing)
            title_row.addWidget(self.title_label)
            title_row.addStretch()
            title_row.addWidget(self.preset_button)
            outer.addLayout(title_row)

            logic_row = QHBoxLayout()
            logic_row.addStretch()
            logic_row.addWidget(self.logic_combo)
            outer.addLayout(logic_row)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.rows_widget = QWidget(self.scroll)
        self.rows_widget.setObjectName("structureFilterRows")
        self.rows_layout = QVBoxLayout(self.rows_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(self._ROW_SPACING)
        self.rows_layout.addStretch(1)
        self.scroll.setWidget(self.rows_widget)
        outer.addWidget(self.scroll)

        self.error_label = CaptionLabel("", self)
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        outer.addWidget(self.error_label)

        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFixedHeight(1)
        separator.setStyleSheet("background: rgba(128, 128, 128, 0.18); border: none;")
        outer.addWidget(separator)

        footer = QHBoxLayout()
        self.add_button = ToolButton(FluentIcon.ADD, self)
        self.add_button.setFixedSize(30, 30)
        self.add_button.setToolTip(self.tr("Add condition"))
        self.add_button.setAccessibleName(self.tr("Add condition"))
        self.add_button.clicked.connect(self.add_condition)
        footer.addWidget(self.add_button)
        self.estimate_label = CaptionLabel(self.tr("Set conditions to preview matches"), self)
        footer.addWidget(self.estimate_label)
        footer.addStretch()
        self.clear_button = TransparentToolButton(FluentIcon.DELETE, self)
        self.clear_button.setFixedSize(30, 30)
        self.clear_button.setToolTip(self.tr("Clear"))
        self.clear_button.setAccessibleName(self.tr("Clear"))
        self.clear_button.clicked.connect(self.clear_rows)
        footer.addWidget(self.clear_button)
        self.done_button = PrimaryPushButton(self.tr("Done and preview"), self)
        self.done_button.setFixedHeight(30)
        self.done_button.clicked.connect(self._done)
        footer.addWidget(self.done_button)
        outer.addLayout(footer)
        self._refresh_preset_menu()
        self._refresh_style()

    def _refresh_style(self) -> None:
        surface, border, text, muted = _surface_colors()
        self.setStyleSheet(
            "QFrame#structureFilterEditorPopup { background: transparent; border: none; }"
            "QFrame#structureFilterEditorCard {"
            f" background: {surface}; border: 1px solid {border}; border-radius: 10px; color: {text}; }}"
        )
        self.scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.scroll.viewport().setStyleSheet("background: transparent;")
        self.rows_widget.setStyleSheet("QWidget#structureFilterRows { background: transparent; }")
        self.title_label.setStyleSheet(f"color: {text}; border: none; background: transparent;")
        self.estimate_label.setStyleSheet(f"color: {muted}; border: none; background: transparent;")
        error_color = "#ff8a8a" if isDarkTheme() else "#b10e1e"
        self.error_label.setStyleSheet(f"color: {error_color}; border: none; background: transparent;")
        for row in self._rows:
            row._refresh_style()

    def _refresh_preset_menu(self) -> None:
        names = list_structure_filter_preset_names()
        old_menu = self.preset_menu
        self.preset_menu = RoundMenu(parent=self)
        self.preset_menu.view.setMaxVisibleItems(10)
        self.preset_button.setMenu(self.preset_menu)
        self._manage_preset_menu = None
        self._rename_preset_menu = None
        self._delete_preset_menu = None
        if names:
            for name in names:
                action = Action(
                    FluentIcon.FILTER,
                    name,
                    triggered=lambda _checked=False, preset_name=name: self._load_preset(preset_name),
                )
                self.preset_menu.addAction(action)
        else:
            empty = Action(FluentIcon.FILTER, self.tr("No saved filters"))
            empty.setEnabled(False)
            self.preset_menu.addAction(empty)

        self.preset_menu.addSeparator()
        self._save_preset_action = Action(
            FluentIcon.SAVE,
            self.tr("Save current conditions…"),
            triggered=lambda _checked=False: self._save_current_preset(),
        )
        self._save_preset_action.setEnabled(not self.spec().is_empty())
        self.preset_menu.addAction(self._save_preset_action)

        if names:
            manage_menu = RoundMenu(self.tr("Manage saved filters"), self.preset_menu)
            rename_menu = RoundMenu(self.tr("Rename"), manage_menu)
            delete_menu = RoundMenu(self.tr("Delete"), manage_menu)
            self._manage_preset_menu = manage_menu
            self._rename_preset_menu = rename_menu
            self._delete_preset_menu = delete_menu
            for name in names:
                rename_menu.addAction(
                    Action(
                        FluentIcon.EDIT,
                        name,
                        triggered=lambda _checked=False, preset_name=name: self._rename_preset(preset_name),
                    )
                )
                delete_menu.addAction(
                    Action(
                        FluentIcon.DELETE,
                        name,
                        triggered=lambda _checked=False, preset_name=name: self._delete_preset(preset_name),
                    )
                )
            manage_menu.addMenu(rename_menu)
            manage_menu.addMenu(delete_menu)
            self.preset_menu.addMenu(manage_menu)

        if old_menu is not None:
            old_menu.close()
            old_menu.deleteLater()

    def refresh_presets(self) -> None:
        """Refresh the menu from the user configuration database."""
        self._refresh_preset_menu()

    def _dialog_parent(self):
        owner = self.parentWidget()
        return owner.window() if owner is not None else self

    def _prompt_preset_name(self, title: str, accept_text: str, initial: str = "") -> str | None:
        dialog = _PresetNameDialog(
            title,
            self.tr("Preset name"),
            accept_text,
            initial,
            self._dialog_parent(),
        )
        if not dialog.exec():
            return None
        return dialog.value()

    def _confirm(self, title: str, message: str, accept_text: str) -> bool:
        box = MessageBox(title, message, self._dialog_parent())
        box.yesButton.setText(accept_text)
        box.cancelButton.setText(self.tr("Cancel"))
        return bool(box.exec())

    def _restore_after_dialog(self) -> None:
        owner = self.parentWidget()
        if owner is not None and hasattr(owner, "open_editor"):
            QTimer.singleShot(0, owner.open_editor)

    def _load_preset(self, name: str) -> None:
        spec = load_structure_filter_preset(name)
        if spec is None:
            MessageManager.send_warning_message(
                self.tr("Saved filter '{name}' is unavailable or damaged.").format(name=name)
            )
            self._refresh_preset_menu()
            return
        self.set_spec(spec)
        self._emit_spec()
        MessageManager.send_success_message(
            self.tr("Loaded saved filter: {name}").format(name=name)
        )

    def _save_current_preset(self) -> None:
        spec = self.spec()
        try:
            name = self._prompt_preset_name(
                self.tr("Save current conditions"),
                self.tr("Save"),
            )
            if not name:
                return
            if structure_filter_preset_exists(name) and not self._confirm(
                self.tr("Overwrite saved filter?"),
                self.tr("A saved filter named '{name}' already exists.").format(name=name),
                self.tr("Overwrite"),
            ):
                return
            save_structure_filter_preset(name, spec)
            self._refresh_preset_menu()
            MessageManager.send_success_message(
                self.tr("Saved filter: {name}").format(name=name)
            )
        except ValueError:
            MessageManager.send_warning_message(
                self.tr("Complete or remove empty conditions before saving.")
            )
        finally:
            self._restore_after_dialog()

    def _rename_preset(self, old_name: str) -> None:
        try:
            new_name = self._prompt_preset_name(
                self.tr("Rename saved filter"),
                self.tr("Rename"),
                old_name,
            )
            if not new_name or new_name == old_name:
                return
            if structure_filter_preset_exists(new_name) and not self._confirm(
                self.tr("Overwrite saved filter?"),
                self.tr("A saved filter named '{name}' already exists.").format(name=new_name),
                self.tr("Overwrite"),
            ):
                return
            if not rename_structure_filter_preset(old_name, new_name):
                MessageManager.send_warning_message(self.tr("Saved filter could not be renamed."))
                return
            self._refresh_preset_menu()
            MessageManager.send_success_message(
                self.tr("Renamed saved filter to: {name}").format(name=new_name)
            )
        except ValueError:
            MessageManager.send_warning_message(self.tr("Saved filter could not be renamed."))
        finally:
            self._restore_after_dialog()

    def _delete_preset(self, name: str) -> None:
        try:
            if not self._confirm(
                self.tr("Delete saved filter?"),
                self.tr("Delete saved filter '{name}'?").format(name=name),
                self.tr("Delete"),
            ):
                return
            if not delete_structure_filter_preset(name):
                MessageManager.send_warning_message(self.tr("Saved filter could not be deleted."))
                return
            self._refresh_preset_menu()
            MessageManager.send_success_message(
                self.tr("Deleted saved filter: {name}").format(name=name)
            )
        finally:
            self._restore_after_dialog()

    def set_spec(self, spec: StructureFilterSpec) -> None:
        self._syncing = True
        try:
            for row in self._rows:
                self.rows_layout.removeWidget(row)
                row.hide()
                row.deleteLater()
            self._rows.clear()
            self.logic_combo.setCurrentIndex(0 if spec.logic == FilterLogic.ALL else 1)
            for condition in spec.conditions:
                self._append_row(condition)
        finally:
            self._syncing = False
        self.clear_error()
        self._update_content_height()
        if hasattr(self, "_save_preset_action"):
            self._save_preset_action.setEnabled(not spec.is_empty())

    def _append_row(self, condition: StructureFilterCondition) -> _ConditionRow:
        row = _ConditionRow(condition, self._suggestions, self.rows_widget)
        row.changed.connect(self._emit_spec)
        row.removeRequested.connect(self._remove_row)
        self.rows_layout.insertWidget(len(self._rows), row)
        self._rows.append(row)
        if not self._syncing:
            self._update_content_height()
        return row

    def set_suggestions(self, suggestions: dict[SearchType, dict[str, int]]) -> None:
        self._suggestions = dict(suggestions)
        for row in self._rows:
            row.set_suggestions(self._suggestions)

    def add_condition(self) -> None:
        condition = StructureFilterCondition(
            condition_id=str(uuid.uuid4()),
            field=FilterField.CONFIG_TYPE,
            text_values=(),
            match_mode=TextMatchMode.CONTAINS,
        )
        row = self._append_row(condition)
        row.value_edit.setFocus(Qt.FocusReason.TabFocusReason)
        QTimer.singleShot(0, lambda: self._reveal_new_row(row))
        if not self._syncing:
            self._emit_spec()

    def _reveal_new_row(self, row: _ConditionRow) -> None:
        if row not in self._rows:
            return
        self.scroll.ensureWidgetVisible(row, 0, 4)
        row.value_edit.setFocus(Qt.FocusReason.TabFocusReason)

    def _remove_row(self, row: _ConditionRow) -> None:
        if row not in self._rows:
            return
        self.rows_layout.removeWidget(row)
        self._rows.remove(row)
        row.hide()
        row.deleteLater()
        self._update_content_height()
        self._emit_spec()

    def clear_rows(self) -> None:
        for row in list(self._rows):
            self.rows_layout.removeWidget(row)
            row.hide()
            row.deleteLater()
        self._rows.clear()
        self.clear_error()
        self._update_content_height()
        self._emit_spec()

    def _update_content_height(self) -> None:
        """Show up to five full rows, then keep a stable scroll viewport."""
        row_heights = [row.height() for row in self._rows]
        content_height = sum(row_heights) + max(0, len(self._rows) - 1) * self._ROW_SPACING
        self.rows_widget.setMinimumHeight(content_height)
        visible_rows = min(len(self._rows), self._MAX_VISIBLE_ROWS)
        self.scroll.setVisible(visible_rows > 0)
        if visible_rows:
            height = sum(row_heights[:visible_rows]) + (visible_rows - 1) * self._ROW_SPACING + 2
            self.scroll.setFixedHeight(height)
        self.card.layout().invalidate()
        self.card.adjustSize()
        self.layout().invalidate()
        self.layout().activate()
        if self.isVisible():
            target_height = self.sizeHint().height()
            screen = self.screen().availableGeometry() if self.screen() is not None else None
            if screen is not None:
                target_height = min(target_height, screen.height())
            self.resize(self.width(), target_height)

    def spec(self) -> StructureFilterSpec:
        logic = self.logic_combo.currentData() or FilterLogic.ALL
        return StructureFilterSpec(
            conditions=tuple(row.to_condition() for row in self._rows),
            logic=logic,
        )

    def _emit_spec(self) -> None:
        if self._syncing:
            return
        self.clear_error()
        spec = self.spec()
        if hasattr(self, "_save_preset_action"):
            self._save_preset_action.setEnabled(not spec.is_empty())
        self.specChanged.emit(spec)
        if not spec.is_empty():
            self._debounce.start()

    def _done(self) -> None:
        self._debounce.stop()
        self.specChanged.emit(self.spec())
        self.previewRequested.emit()
        self.close()

    def focus_condition(self, condition_id: str | None) -> None:
        if not condition_id:
            return
        for row in self._rows:
            if row.condition_id == condition_id:
                self.scroll.ensureWidgetVisible(row)
                row.value_edit.setFocus(Qt.FocusReason.OtherFocusReason)
                return

    def set_estimate(self, matched: int | None, active: int | None, status: str = "") -> None:
        if status:
            self.estimate_label.setText(status)
        elif matched is None or active is None:
            self.estimate_label.setText(self.tr("Set conditions to preview matches"))
        else:
            self.estimate_label.setText(
                self.tr("Estimated matches: {matched:,} / {active:,} structures").format(
                    matched=matched,
                    active=active,
                )
            )

    def set_error(self, error: StructureFilterValidationError) -> None:
        self.clear_error()
        self.error_label.setText(error.message)
        self.error_label.setVisible(True)
        if error.condition_id:
            for row in self._rows:
                if row.condition_id == error.condition_id:
                    row.set_error(error.message)
                    self.scroll.ensureWidgetVisible(row)
                    break
        self._update_content_height()

    def clear_error(self) -> None:
        self.error_label.setVisible(False)
        for row in self._rows:
            row.clear_error()
        self._update_content_height()


class StructureFilterBar(QFrame):
    """Single-line query summary with preview and cached-result actions."""

    specChanged = Signal(object)
    previewRequested = Signal()
    applyRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("structureFilterBar")
        self.setFixedHeight(38)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._spec = StructureFilterSpec()
        self._chips: list[FilterChip] = []
        self._chip_revision = 0
        self._overflow_button: PushButton | None = None
        self._match_count: int | None = None
        self._active_count: int | None = None
        self._elapsed_ms: float | None = None
        self._selection_count = 0
        self._result_current = False
        self._popup = StructureFilterEditorPopup(self)
        self._popup.specChanged.connect(self._on_popup_spec_changed)
        self._popup.previewRequested.connect(self.previewRequested.emit)
        self._build_ui()
        self._rebuild_chips()
        self._refresh()
        qconfig.themeChangedFinished.connect(self._refresh_theme)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        self.chip_view = QWidget(self)
        self.chip_view.setMinimumWidth(60)
        self.chip_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.chip_layout = QHBoxLayout(self.chip_view)
        self.chip_layout.setContentsMargins(0, 0, 0, 0)
        self.chip_layout.setSpacing(5)
        layout.addWidget(self.chip_view, 1)

        self.preview_button = PrimaryPushButton(self.tr("Preview"), self)
        self.preview_button.setFixedHeight(32)
        self.preview_button.setMinimumWidth(self.preview_button.sizeHint().width())
        self.preview_button.clicked.connect(lambda _checked=False: self.previewRequested.emit())
        layout.addWidget(self.preview_button)

        self.match_button = PushButton(self.tr("No preview"), self)
        self.match_button.setFixedHeight(32)
        self.match_button.setMinimumWidth(54)
        self.match_button.clicked.connect(self._show_result_details)
        layout.addWidget(self.match_button)

        self.apply_button = PushButton(self.tr("Apply result ▾"), self)
        self.apply_button.setFixedHeight(32)
        self.apply_button.setMinimumWidth(self.apply_button.sizeHint().width())
        self.apply_button.clicked.connect(self._show_apply_menu)
        layout.addWidget(self.apply_button)

    @property
    def spec(self) -> StructureFilterSpec:
        return self._spec

    @property
    def editor_is_open(self) -> bool:
        """Whether the anchored condition editor is currently visible."""
        return self._popup.isVisible()

    def set_spec(self, spec: StructureFilterSpec) -> None:
        self._apply_spec(spec, sync_popup=True)

    def _on_popup_spec_changed(self, spec: StructureFilterSpec) -> None:
        self._apply_spec(spec, sync_popup=False)

    def _apply_spec(self, spec: StructureFilterSpec, *, sync_popup: bool) -> None:
        if spec == self._spec:
            return
        self._spec = spec
        self._result_current = False
        if sync_popup:
            self._popup.set_spec(spec)
        self._rebuild_chips()
        self.specChanged.emit(spec)
        self._refresh()

    def set_selection_count(self, count: int) -> None:
        self._selection_count = max(0, int(count))

    def set_suggestions(self, suggestions: dict[SearchType, dict[str, int]]) -> None:
        """Apply dataset-backed completion candidates to every editor row."""
        self._popup.set_suggestions(suggestions)

    def set_running(self, running: bool) -> None:
        self.preview_button.setText(self.tr("Calculating…") if running else self.tr("Preview"))
        self.preview_button.setEnabled(not running and not self._spec.is_empty())
        if running:
            self.match_button.setToolTip("")
            self.match_button.setText("…")
            self._refresh_match_style("running")
            self._popup.set_estimate(None, None, self.tr("Calculating…"))

    def set_result(self, matched: int, active: int, elapsed_ms: float) -> None:
        self._match_count = int(matched)
        self._active_count = int(active)
        self._elapsed_ms = float(elapsed_ms)
        self._result_current = True
        self.match_button.setToolTip("")
        self._popup.clear_error()
        self._popup.set_estimate(self._match_count, self._active_count)
        self._refresh()

    def set_stale(self) -> None:
        if self._match_count is not None:
            self._result_current = False
            self.match_button.setText(self.tr("Expired"))
            self._refresh_match_style("stale")
        self._refresh_actions()

    def set_error(self, error: StructureFilterValidationError) -> None:
        self._result_current = False
        self.match_button.setText(self.tr("Error"))
        self.match_button.setToolTip(error.message)
        self._popup.set_error(error)
        self._refresh_match_style("error")
        self._refresh_actions()

    def clear_state(self) -> None:
        self._spec = StructureFilterSpec()
        self._match_count = None
        self._active_count = None
        self._elapsed_ms = None
        self._result_current = False
        self.match_button.setToolTip("")
        self._popup.set_spec(self._spec)
        self._rebuild_chips()
        self._refresh()

    def open_editor(self, condition_id: str | None = None, *, add_if_empty: bool = False) -> None:
        self._popup.set_spec(self._spec)
        self._popup.refresh_presets()
        if add_if_empty and not self._spec.conditions:
            self._popup.add_condition()
        width = max(self._popup.minimumWidth(), min(self._popup.maximumWidth(), self.width()))
        self._popup.resize(width, self._popup.sizeHint().height())
        global_pos = self.mapToGlobal(QPoint(0, self.height()))
        screen = self.screen().availableGeometry() if self.screen() is not None else None
        if screen is not None:
            global_pos.setX(max(screen.left(), min(global_pos.x(), screen.right() - width)))
            if global_pos.y() + self._popup.height() > screen.bottom():
                global_pos.setY(max(screen.top(), self.mapToGlobal(QPoint(0, 0)).y() - self._popup.height()))
        self._popup.move(global_pos)
        self._popup.show()
        self._popup.raise_()
        self._popup.focus_condition(condition_id)

    def _condition_summary(self, condition: StructureFilterCondition) -> tuple[str, str]:
        values = ",".join(condition.text_values)
        if condition.field == FilterField.CONFIG_TYPE:
            label = self.tr("Config type")
        elif condition.field == FilterField.FORMULA:
            label = self.tr("Formula")
        elif condition.field == FilterField.ELEMENT_REQUIRED:
            label = self.tr("Required")
        elif condition.field == FilterField.ELEMENT_EXCLUDED:
            label = self.tr("Excluded")
        elif condition.field == FilterField.ELEMENT_ALLOWED:
            label = self.tr("Allowed")
        else:
            label = self.tr("Expression")
        return label, values

    def _rebuild_chips(self) -> None:
        self._chip_revision += 1
        revision = self._chip_revision
        # The overflow button is part of this layout and is deleted below.  Drop
        # the Python reference before Qt processes deleteLater(), otherwise a
        # queued resize callback can access an already-destroyed C++ object.
        self._overflow_button = None
        while self.chip_layout.count():
            item = self.chip_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self._chips.clear()
        if not self._spec.conditions:
            filter_entry = PushButton(FluentIcon.SEARCH, self.tr("Filter conditions"), self.chip_view)
            filter_entry.setFixedHeight(32)
            filter_entry.clicked.connect(lambda: self.open_editor(None, add_if_empty=True))
            self.chip_layout.addWidget(filter_entry)
            self._empty_hint = None
            self.chip_layout.addStretch(1)
            self._refresh_theme()
            return
        self._empty_hint = None
        for condition in self._spec.conditions:
            kind, value = self._condition_summary(condition)
            chip = FilterChip(
                condition.condition_id,
                kind,
                value,
                condition.enabled,
                self.chip_view,
            )
            chip.editRequested.connect(lambda condition_id: self.open_editor(condition_id))
            chip.removeRequested.connect(self._remove_condition)
            self.chip_layout.addWidget(chip)
            self._chips.append(chip)
        self.chip_layout.addStretch(1)
        QTimer.singleShot(0, lambda: self._update_chip_overflow(revision))

    def _remove_condition(self, condition_id: str) -> None:
        self.set_spec(
            StructureFilterSpec(
                conditions=tuple(
                    condition for condition in self._spec.conditions if condition.condition_id != condition_id
                ),
                logic=self._spec.logic,
            )
        )

    def _update_chip_overflow(self, revision: int | None = None) -> None:
        if revision is not None and revision != self._chip_revision:
            return
        if not self._chips:
            return
        available = max(0, self.chip_view.width())
        widths = [chip.sizeHint().width() + self.chip_layout.spacing() for chip in self._chips]
        budget = available if sum(widths) <= available else max(0, available - 70)
        used = 0
        hidden = 0
        for chip, width in zip(self._chips, widths):
            visible = used + width <= budget
            chip.setVisible(visible)
            if visible:
                used += width
            else:
                hidden += 1
        overflow = self._overflow_button
        if overflow is not None:
            self.chip_layout.removeWidget(overflow)
            overflow.hide()
            overflow.deleteLater()
            self._overflow_button = None
        if hidden:
            overflow = PushButton(self.tr("+{count} conditions").format(count=hidden), self.chip_view)
            overflow.setFixedHeight(30)
            overflow.clicked.connect(lambda: self.open_editor(None))
            self.chip_layout.insertWidget(max(0, len(self._chips) - hidden), overflow)
            self._overflow_button = overflow

    def _show_result_details(self) -> None:
        if self._match_count is None or self._active_count is None:
            return
        ratio = 0.0 if self._active_count == 0 else 100.0 * self._match_count / self._active_count
        text = self.tr(
            "Matched structures: {matched:,}\nActive structures: {active:,}\nMatch ratio: {ratio:.2f}%\nElapsed: {elapsed:.1f} ms"
        ).format(
            matched=self._match_count,
            active=self._active_count,
            ratio=ratio,
            elapsed=self._elapsed_ms or 0.0,
        )
        QToolTip.showText(self.match_button.mapToGlobal(QPoint(0, self.match_button.height())), text, self.match_button)

    def _show_apply_menu(self) -> None:
        menu = RoundMenu(parent=self)
        count = self._match_count or 0
        actions = (
            (FluentIcon.SYNC, self.tr("Replace current selection ({count:,})").format(count=count), "replace"),
            (FluentIcon.ADD, self.tr("Add to current selection ({count:,})").format(count=count), "add"),
            (FluentIcon.REMOVE, self.tr("Remove from current selection ({count:,})").format(count=count), "remove"),
        )
        for icon, text, mode in actions:
            action = Action(icon, text, triggered=lambda checked=False, mode=mode: self.applyRequested.emit(mode))
            action.setEnabled(self._result_current)
            menu.addAction(action)
        menu.addSeparator()
        clear = Action(
            FluentIcon.DELETE,
            self.tr("Clear current selection ({count:,})").format(count=self._selection_count),
            triggered=lambda: self.applyRequested.emit("clear"),
        )
        clear.setEnabled(self._selection_count > 0)
        menu.addAction(clear)
        menu.exec(self.apply_button.mapToGlobal(QPoint(0, self.apply_button.height())))

    def _refresh(self) -> None:
        empty = self._spec.is_empty()
        self.preview_button.setEnabled(not empty)
        if self._match_count is None:
            self.match_button.setText("—")
            self._refresh_match_style("idle")
        elif self._result_current:
            self.match_button.setText(f"{self._match_count:,}")
            self._refresh_match_style("current")
        self._refresh_actions()

    def _refresh_actions(self) -> None:
        self.apply_button.setEnabled(self._result_current or self._selection_count > 0)

    def _refresh_match_style(self, state: str) -> None:
        dark = isDarkTheme()
        styles = {
            "current": ("#163c2b" if dark else "#f3fbf6", "#2d7650" if dark else "#b9d8c7", "#56c98c" if dark else "#18864f"),
            "stale": ("#44341f" if dark else "#fff8ec", "#8a6938" if dark else "#e6c98c", "#f1b85b" if dark else "#9a650c"),
            "error": ("#47282b" if dark else "#fff3f3", "#8d4c52" if dark else "#efb7bb", "#ff8a8a" if dark else "#b10e1e"),
            "running": ("#26384a" if dark else "#f2f7fc", "#496a89" if dark else "#bfd3e6", "#8fc5f4" if dark else "#326b9b"),
            "idle": ("#303030" if dark else "#ffffff", "#555555" if dark else "#d0d0d0", "#a6a6a6" if dark else "#707070"),
        }
        background, border, foreground = styles.get(state, styles["idle"])
        self.match_button.setStyleSheet(
            f"PushButton {{ background: {background}; border: 1px solid {border}; color: {foreground}; font-weight: 600; }}"
        )

    def _refresh_theme(self) -> None:
        """Refresh custom surfaces that are not styled by Fluent widgets."""
        _, _, _, muted = _surface_colors()
        if getattr(self, "_empty_hint", None) is not None:
            self._empty_hint.setStyleSheet(f"color: {muted}; border: none;")
        for chip in self._chips:
            chip._refresh_style()
        self._popup._refresh_style()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        revision = self._chip_revision
        QTimer.singleShot(0, lambda: self._update_chip_overflow(revision))

    def changeEvent(self, event):
        if event.type() in (QEvent.Type.LanguageChange, QEvent.Type.PaletteChange):
            self._refresh()
        super().changeEvent(event)
