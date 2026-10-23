#!/usr/bin/env python
"""Fluent dialogs for confirming LAMMPS dump atom types and column roles."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QPoint, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCompleter,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CheckBox,
    ComboBox,
    FluentIcon,
    LineEdit,
    MessageBoxBase,
    PushButton,
    SimpleCardWidget,
    StrongBodyLabel,
    TransparentToolButton,
)

from NepTrainKit.core.io.lammps_dump import (
    ColumnOverrides,
    DumpProbe,
    ElementSuggestion,
    build_plan,
    coordinate_triplets,
)
from NepTrainKit.core.structure import atomic_numbers

from .periodic_table import PeriodicTableDialog

_ELEMENT_SYMBOLS = tuple(sorted(atomic_numbers, key=atomic_numbers.__getitem__))
_ELEMENT_BY_SYMBOL = {symbol.lower(): symbol for symbol in _ELEMENT_SYMBOLS}
_ELEMENT_BY_NUMBER = {number: symbol for symbol, number in atomic_numbers.items()}


def _tr(text: str) -> str:
    """Translate with the dump-import context shared by both dialogs."""
    return QCoreApplication.translate("DumpImportDialog", text)


@dataclass
class _CustomRow:
    name: LineEdit
    columns: LineEdit
    widget: QWidget
    preview: CaptionLabel


class _WindowMaskDialog(MessageBoxBase):
    """A mask dialog that behaves as a real modal window.

    ``MessageBoxBase`` normally turns itself into a child overlay, which can
    end up behind page widgets or stay disabled after a previous import.  A
    frameless ``Qt.Dialog`` window over the parent window is independent of
    that stacking and stays clickable.
    """

    def __init__(self, parent=None):
        """Create the dialog and make it a frameless window."""
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)

    def showEvent(self, event) -> None:
        """Cover the parent window and raise the dialog above other windows."""
        parent = self.parentWidget()
        if parent is not None:
            origin = parent.mapToGlobal(QPoint(0, 0))
            self.setGeometry(origin.x(), origin.y(), parent.width(), parent.height())
        super().showEvent(event)
        self.raise_()


@dataclass(frozen=True)
class DumpImportOptions:
    """Choices made in :class:`DumpImportDialog`."""

    element_map: dict[int, str]
    overrides: ColumnOverrides | None = None
    unwrap: bool = False
    keep_extra: bool = True
    strict: bool = False
    remember: bool = False


def _canonical_element(text: str) -> str | None:
    """Accept an element symbol (any case) or an atomic number."""
    token = str(text or "").strip()
    if not token:
        return None
    if token.isdigit():
        return _ELEMENT_BY_NUMBER.get(int(token))
    return _ELEMENT_BY_SYMBOL.get(token.lower())


def _element_symbol(edit: LineEdit) -> str | None:
    return _canonical_element(edit.text())


def _find_data(combo: ComboBox, data) -> int:
    for index in range(combo.count()):
        if combo.itemData(index) == data:
            return index
    return -1


def _cell_text(probe: DumpProbe) -> str:
    if len(probe.lattice) != 9:
        return "-"
    cell = np.asarray(probe.lattice, dtype=np.float64).reshape(3, 3)
    lengths = np.linalg.norm(cell, axis=1)
    return " × ".join(f"{value:.3f}" for value in lengths) + " Å"


def _atom_text(probe: DumpProbe) -> str:
    counts = sorted({int(value) for value in probe.atom_counts})
    if not counts:
        return "-"
    if len(counts) == 1:
        return f"{counts[0]:,}"
    if len(counts) <= 3:
        return " / ".join(f"{value:,}" for value in counts)
    return f"{counts[0]:,} … {counts[-1]:,}"


def _describe_plan(
    probe: DumpProbe,
    translate,
    overrides: ColumnOverrides | None = None,
) -> str:
    """Summarise how the current dump columns are interpreted."""
    signature = probe.signatures[0] if probe.signatures else probe.columns
    if not signature:
        return ""
    plan = build_plan(signature, overrides)
    parts: list[str] = []
    if plan.coord_tokens:
        parts.append(
            translate("Coordinates") + ": " + " ".join(plan.signature[t] for t in plan.coord_tokens)
        )
    if plan.force_tokens:
        parts.append(
            translate("Forces") + ": " + " ".join(plan.signature[t] for t in plan.force_tokens)
        )
    if plan.spin_tokens:
        names = [plan.signature[t] for t in plan.spin_tokens]
        if plan.spin_mode == "magnitude_direction":
            spin = f"{names[0]} × ({', '.join(names[1:])})"
        else:
            spin = ", ".join(names)
        parts.append(translate("Spin") + ": " + spin)
    extras = [name.removeprefix("dump.") for name, _tokens in plan.extras]
    if extras:
        shown = ", ".join(extras[:6])
        if len(extras) > 6:
            shown += " …"
        parts.append(translate("Extras") + ": " + shown)
    if not parts:
        return ""
    return translate("Detected") + " · " + " · ".join(parts)


def _auto_overrides(columns: tuple[str, ...]) -> ColumnOverrides:
    plan = build_plan(columns)

    def names(tokens) -> tuple[str, ...]:
        return tuple(columns[token] for token in (tokens or ()))

    return ColumnOverrides(
        coordinate_names=names(plan.coord_tokens) or None,
        force_names=names(plan.force_tokens),
        spin_mode=plan.spin_mode or "off",
        spin_names=names(plan.spin_tokens),
        ignored_extras=frozenset(),
    )


class DumpFieldMappingDialog(_WindowMaskDialog):
    """Edit coordinates, forces, spin and extra columns for a dump."""

    def __init__(self, columns, overrides: ColumnOverrides | None, probe: DumpProbe, parent=None):
        """Build the field editor from automatic detection plus saved overrides."""
        super().__init__(parent)
        self._columns = tuple(str(name) for name in columns)
        self._probe = probe
        self._auto_plan = build_plan(self._columns)
        self._auto = _auto_overrides(self._columns)
        self._extra_checks: dict[str, CheckBox] = {}
        self._extra_edits: dict[str, LineEdit] = {}
        self._initial_ignored = frozenset(overrides.ignored_extras) if overrides else frozenset()
        self._initial_renames = dict(overrides.extra_names) if overrides else {}
        self._custom_rows: list[tuple[LineEdit, LineEdit, QWidget]] = []
        self._building_extras = False
        self._spin_combos: list[ComboBox] = []
        self._force_names = tuple(self._columns[t] for t in (self._auto_plan.force_tokens or ()))

        self.titleLabel = StrongBodyLabel(_tr("Field mapping"), self)
        self.subtitleLabel = CaptionLabel(_tr("Set how the columns are read."), self)
        self.subtitleLabel.setTextColor("#8a8a8a", "#9a9a9a")

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.subtitleLabel)
        self.viewLayout.addWidget(self._build_card(overrides))

        self.widget.setMinimumWidth(560)
        self.yesButton.setText(_tr("OK"))
        self.cancelButton.setText(_tr("Cancel"))
        self.resetButton = PushButton(_tr("Reset to automatic"), self)
        self.resetButton.clicked.connect(self._reset)
        self.buttonLayout.insertWidget(0, self.resetButton, 0, Qt.AlignmentFlag.AlignVCenter)

    # ------------------------------------------------------------------ layout

    def _build_card(self, overrides: ColumnOverrides | None) -> SimpleCardWidget:
        card = SimpleCardWidget(self)
        card.setBorderRadius(8)
        layout = QGridLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)
        layout.setColumnMinimumWidth(0, 78)
        layout.setColumnStretch(1, 1)

        effective = overrides if overrides is not None else self._auto

        layout.addWidget(CaptionLabel(_tr("Coordinates"), card), 0, 0, Qt.AlignmentFlag.AlignTop)
        self.coordCombo = ComboBox(card)
        for _mode, triplet in coordinate_triplets(self._columns):
            self.coordCombo.addItem(", ".join(triplet), userData=triplet)
        if effective.coordinate_names:
            index = _find_data(self.coordCombo, effective.coordinate_names)
            if index >= 0:
                self.coordCombo.setCurrentIndex(index)
        if self.coordCombo.count() == 0:
            self.coordCombo.addItem("-", userData=None)
            self.coordCombo.setEnabled(False)
        layout.addWidget(self.coordCombo, 0, 1)

        layout.addWidget(CaptionLabel(_tr("Forces"), card), 1, 0)
        self.forceCheck = CheckBox(_tr("Read forces (fx, fy, fz)"), card)
        self.forceCheck.setChecked(bool(effective.force_names))
        self.forceCheck.setEnabled(bool(self._force_names))
        layout.addWidget(self.forceCheck, 1, 1)

        layout.addWidget(CaptionLabel(_tr("Spin"), card), 2, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._build_spin_widget(card, effective, overrides), 2, 1)

        layout.addWidget(CaptionLabel(_tr("Extra columns"), card), 3, 0, Qt.AlignmentFlag.AlignTop)
        self.extrasHost = QWidget(card)
        self.extrasLayout = QGridLayout(self.extrasHost)
        self.extrasLayout.setContentsMargins(0, 0, 0, 0)
        self.extrasLayout.setHorizontalSpacing(18)
        self.extrasLayout.setVerticalSpacing(6)
        layout.addWidget(self.extrasHost, 3, 1)

        layout.addWidget(CaptionLabel(_tr("Custom fields"), card), 4, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._build_custom_fields(card, overrides), 4, 1)

        self.errorLabel = CaptionLabel("", card)
        self.errorLabel.setWordWrap(True)
        self.errorLabel.setTextColor("#c42b1c", "#ff99a4")
        self.errorLabel.setVisible(False)
        layout.addWidget(self.errorLabel, 5, 0, 1, 2)

        self.coordCombo.currentIndexChanged.connect(self._refresh_extras)
        self.forceCheck.stateChanged.connect(self._refresh_extras)
        self.spinModeCombo.currentIndexChanged.connect(self._sync_spin_widgets)
        for combo in self._spin_combos:
            combo.currentIndexChanged.connect(self._refresh_extras)
        self._sync_spin_widgets()
        return card

    def _build_custom_fields(self, parent: QWidget, overrides: ColumnOverrides | None) -> QWidget:
        host = QWidget(parent)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        hint = CaptionLabel(
            _tr("Multiple columns are merged into one multi-column field, in the order you list them."),
            host,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.customRowsLayout = QVBoxLayout()
        self.customRowsLayout.setContentsMargins(0, 0, 0, 0)
        self.customRowsLayout.setSpacing(6)
        layout.addLayout(self.customRowsLayout)

        self.addFieldButton = PushButton(FluentIcon.ADD, _tr("Add field"), host)
        self.addFieldButton.clicked.connect(self._on_add_field)
        layout.addWidget(self.addFieldButton)

        if overrides is not None:
            for name, columns in overrides.custom_fields:
                self._add_custom_row(name, columns)
        return host

    def _on_add_field(self) -> None:
        self._add_custom_row()
        self._validate()

    def _add_custom_row(self, name: str = "", columns=()) -> None:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        controls = QWidget(container)
        control_layout = QHBoxLayout(controls)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(6)

        name_edit = LineEdit(controls)
        name_edit.setPlaceholderText(_tr("Field name"))
        name_edit.setMaximumWidth(130)
        name_edit.setText(str(name))
        columns_edit = LineEdit(controls)
        columns_edit.setPlaceholderText(_tr("Columns (comma separated)"))
        columns_edit.setText(", ".join(str(column) for column in columns))
        remove = TransparentToolButton(FluentIcon.DELETE, controls)
        remove.setFixedSize(26, 26)
        remove.setToolTip(_tr("Remove field"))

        control_layout.addWidget(name_edit, 1)
        control_layout.addWidget(columns_edit, 2)
        control_layout.addWidget(remove)
        preview = CaptionLabel("", container)
        preview.setTextColor("#8a8a8a", "#9a9a9a")
        preview.setVisible(False)
        layout.addWidget(controls)
        layout.addWidget(preview)
        self.customRowsLayout.addWidget(container)

        entry = _CustomRow(name_edit, columns_edit, container, preview)
        self._custom_rows.append(entry)
        name_edit.textChanged.connect(self._validate)
        columns_edit.textChanged.connect(self._validate)
        name_edit.editingFinished.connect(self._refresh_extras)
        columns_edit.editingFinished.connect(self._refresh_extras)
        remove.clicked.connect(lambda _checked=False, item=entry: self._remove_custom_row(item))
        self._update_custom_preview(entry)

    def _remove_custom_row(self, entry: _CustomRow, *, refresh: bool = True) -> None:
        if entry in self._custom_rows:
            self._custom_rows.remove(entry)
        entry.widget.setParent(None)
        entry.widget.deleteLater()
        if refresh:
            self._refresh_extras()
            self._validate()

    def _custom_fields_override(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (
                row.name.text().strip(),
                tuple(part.strip() for part in row.columns.text().split(",") if part.strip()),
            )
            for row in self._custom_rows
            if row.name.text().strip() and row.columns.text().strip()
        )

    def _update_custom_preview(self, row: _CustomRow) -> None:
        name = row.name.text().strip()
        columns = [part.strip() for part in row.columns.text().split(",") if part.strip()]
        if name and columns and all(column in self._columns for column in columns):
            row.preview.setText(f"→ {name}:R:{len(columns)}")
            row.preview.setVisible(True)
        else:
            row.preview.setText("")
            row.preview.setVisible(False)

    def _build_spin_widget(
        self, parent: QWidget, effective: ColumnOverrides, overrides: ColumnOverrides | None
    ) -> QWidget:
        host = QWidget(parent)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.spinModeCombo = ComboBox(host)
        self.spinModeCombo.setFixedWidth(170)
        self.spinModeCombo.addItem(_tr("Automatic"), userData=None)
        self.spinModeCombo.addItem(_tr("Magnitude + direction"), userData="magnitude_direction")
        self.spinModeCombo.addItem(_tr("Vector (3 columns)"), userData="vector")
        self.spinModeCombo.addItem(_tr("Ignore spin"), userData="off")
        chosen = overrides.spin_mode if overrides is not None else None
        if chosen is not None:
            index = self.spinModeCombo.findData(chosen)
            if index >= 0:
                self.spinModeCombo.setCurrentIndex(index)
        layout.addWidget(self.spinModeCombo)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.spinLabels: list[CaptionLabel] = []
        for slot in range(4):
            label = CaptionLabel("", host)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            label.setFixedWidth(56)
            self.spinLabels.append(label)
            combo = ComboBox(host)
            for name in self._columns:
                combo.addItem(name, userData=name)
            if slot < len(effective.spin_names):
                index = _find_data(combo, effective.spin_names[slot])
                if index >= 0:
                    combo.setCurrentIndex(index)
            self._spin_combos.append(combo)
            row, column = divmod(slot, 2)
            grid.addWidget(label, row, column * 2)
            grid.addWidget(combo, row, column * 2 + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)

        self.spinHint = CaptionLabel(_tr("Detected automatically"), host)
        self.spinHint.setWordWrap(True)
        layout.addWidget(self.spinHint)
        return host

    # ----------------------------------------------------------------- behavior

    def _sync_spin_widgets(self, *_args) -> None:
        mode = self.spinModeCombo.currentData()
        magnitude = _tr("Magnitude")
        direction = (_tr("Direction X"), _tr("Direction Y"), _tr("Direction Z"))
        if mode is None:
            effective = self._auto_plan.spin_mode
            names = tuple(self._columns[t] for t in (self._auto_plan.spin_tokens or ()))
            for combo in self._spin_combos:
                combo.setEnabled(False)
            for combo, name in zip(self._spin_combos, names):
                index = _find_data(combo, name)
                if index >= 0:
                    combo.setCurrentIndex(index)
            if effective == "magnitude_direction":
                hint = _tr("Detected automatically") + " · " + _tr(
                    "4 columns (magnitude + direction) → 3-component spin vector"
                )
            elif effective == "vector":
                hint = _tr("Detected automatically") + " · " + _tr(
                    "3 columns → 3-component spin vector"
                )
            else:
                hint = _tr("Detected automatically") + " · " + _tr("No spin columns detected")
            self.spinHint.setText(hint)
            if effective == "magnitude_direction":
                labels = (magnitude, *direction)
            elif effective == "vector":
                labels = (*direction, "")
            else:
                labels = ()
        elif mode == "off":
            effective = None
            for combo in self._spin_combos:
                combo.setEnabled(False)
            self.spinHint.setText(_tr("Spin is ignored."))
            labels = ("", "", "", "")
        elif mode == "magnitude_direction":
            effective = mode
            for combo in self._spin_combos:
                combo.setEnabled(True)
            self.spinHint.setText(
                _tr("4 columns (magnitude + direction) → 3-component spin vector")
            )
            labels = (magnitude, *direction)
        else:
            effective = "vector"
            for position, combo in enumerate(self._spin_combos):
                combo.setEnabled(position < 3)
            self.spinHint.setText(_tr("3 columns → 3-component spin vector"))
            labels = (*direction, "")
        show_fourth = effective != "vector"
        for slot, label in enumerate(self.spinLabels):
            label.setText(labels[slot] if slot < len(labels) else "")
            label.setVisible(bool(label.text()) and (slot != 3 or show_fourth))
        self._spin_combos[3].setVisible(show_fourth)
        self._refresh_extras()

    def _current_overrides(self) -> ColumnOverrides:
        mode = self.spinModeCombo.currentData()
        if mode is None:
            spin_mode = self._auto_plan.spin_mode or "off"
            spin_names = tuple(self._columns[t] for t in (self._auto_plan.spin_tokens or ()))
        elif mode == "off":
            spin_mode = "off"
            spin_names = ()
        elif mode == "magnitude_direction":
            spin_mode = "magnitude_direction"
            spin_names = tuple(str(combo.currentData()) for combo in self._spin_combos)
        else:
            spin_mode = "vector"
            spin_names = tuple(str(combo.currentData()) for combo in self._spin_combos[:3])
        return ColumnOverrides(
            coordinate_names=self.coordCombo.currentData(),
            force_names=self._force_names if self.forceCheck.isChecked() else (),
            spin_mode=spin_mode,
            spin_names=spin_names,
            ignored_extras=frozenset(
                name for name, box in self._extra_checks.items() if not box.isChecked()
            ),
            extra_names=tuple(
                (name, edit.text().strip())
                for name, edit in self._extra_edits.items()
                if edit.text().strip() and edit.text().strip() != name
            ),
            custom_fields=self._custom_fields_override(),
        )

    def _refresh_extras(self, *_args) -> None:
        self._building_extras = True
        current = self._current_overrides()
        custom_names = {name for name, _columns in current.custom_fields}
        overrides = replace(current, ignored_extras=frozenset(), extra_names=())
        plan = build_plan(self._columns, overrides)
        plan_extras = [(name, tokens) for name, tokens in plan.extras if name not in custom_names]
        previous_checks = {name: box.isChecked() for name, box in self._extra_checks.items()}
        previous_names = dict(self._initial_renames)
        previous_names.update({name: edit.text() for name, edit in self._extra_edits.items()})
        while self.extrasLayout.count():
            item = self.extrasLayout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._extra_checks = {}
        self._extra_edits = {}
        if not plan_extras:
            self.extrasLayout.addWidget(CaptionLabel(_tr("No extra columns"), self.extrasHost), 0, 0)
        for index, (name, _tokens) in enumerate(plan_extras):
            cell = QWidget(self.extrasHost)
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(6)
            box = CheckBox("", cell)
            box.setToolTip(_tr("Keep this column"))
            box.setChecked(previous_checks.get(name, name not in self._initial_ignored))
            edit = LineEdit(cell)
            edit.setText(previous_names.get(name, name))
            edit.setPlaceholderText(_tr("Field name"))
            box.stateChanged.connect(self._validate)
            edit.textChanged.connect(self._validate)
            cell_layout.addWidget(box)
            cell_layout.addWidget(edit, 1)
            self._extra_checks[name] = box
            self._extra_edits[name] = edit
            self.extrasLayout.addWidget(cell, index // 2, index % 2)
        self._building_extras = False
        self._validate()

    def _extras_error(self) -> str | None:
        names: list[str] = []
        for auto, edit in self._extra_edits.items():
            if not self._extra_checks[auto].isChecked():
                continue
            text = edit.text().strip()
            if (
                not text
                or any(char.isspace() or char == ":" for char in text)
                or text in names
                or text in {"species", "pos", "forces", "spin"}
            ):
                return _tr(
                    "Extra column names must be unique, non-empty, free of spaces or ':', "
                    "and not reuse species/pos/forces/spin."
                )
            names.append(text)
        return None

    def _custom_error(self) -> str | None:
        base = replace(
            self._current_overrides(),
            custom_fields=(),
            ignored_extras=frozenset(),
            extra_names=(),
        )
        plan = build_plan(self._columns, base)
        taken: set[str] = set()
        for tokens in (plan.coord_tokens, plan.force_tokens, plan.spin_tokens, plan.image_tokens):
            for token in tokens or ():
                taken.add(self._columns[token])
        used: set[str] = set()
        names: list[str] = []
        for row in self._custom_rows:
            name = row.name.text().strip()
            columns = [part.strip() for part in row.columns.text().split(",") if part.strip()]
            if not name or not columns:
                return _tr(
                    "Custom fields need a unique name and at least one unused column."
                )
            if name in {"species", "pos", "forces", "spin"} or name in names:
                return _tr(
                    "Custom fields need a unique name and at least one unused column."
                )
            if any(column not in self._columns or column in used or column in taken for column in columns):
                return _tr(
                    "Custom fields need a unique name and at least one unused column."
                )
            used.update(columns)
            names.append(name)
        return None

    def _validate(self, *_args) -> None:
        if self._building_extras:
            return
        for row in self._custom_rows:
            self._update_custom_preview(row)
        error = self._extras_error() or self._custom_error()
        self.errorLabel.setText(error or "")
        self.errorLabel.setVisible(bool(error))
        self.yesButton.setEnabled(error is None)

    def _reset(self) -> None:
        if self.coordCombo.count():
            index = _find_data(self.coordCombo, self._auto.coordinate_names)
            if index >= 0:
                self.coordCombo.setCurrentIndex(index)
        self.forceCheck.setChecked(bool(self._auto.force_names))
        self.spinModeCombo.setCurrentIndex(0)
        for box in self._extra_checks.values():
            box.setChecked(True)
        for name, edit in self._extra_edits.items():
            edit.setText(name)
        while self._custom_rows:
            self._remove_custom_row(self._custom_rows[-1], refresh=False)
        self._sync_spin_widgets()
        self._validate()

    def result_overrides(self) -> ColumnOverrides | None:
        """Return the edited overrides, or ``None`` when they match auto-detection."""
        overrides = self._current_overrides()
        if overrides == self._auto:
            return None
        return overrides

    @classmethod
    def ask(
        cls,
        columns,
        overrides: ColumnOverrides | None,
        probe: DumpProbe,
        parent=None,
    ) -> tuple[bool, ColumnOverrides | None]:
        """Show the editor; return ``(accepted, overrides)``."""
        dialog = cls(columns, overrides, probe, parent)
        accepted = bool(dialog.exec())
        overrides_result = dialog.result_overrides() if accepted else None
        dialog.hide()
        dialog.deleteLater()
        return accepted, overrides_result


class DumpImportDialog(_WindowMaskDialog):
    """Preview a dump and confirm the element mapping for numeric atom types.

    Field roles are detected automatically; the "adjust field mapping" button
    opens :class:`DumpFieldMappingDialog` without resizing this window.
    """

    def __init__(self, suggestion: ElementSuggestion, parent=None, overrides=None):
        """Build the preview, element rows and import options."""
        super().__init__(parent)
        self._suggestion = suggestion
        self._probe = suggestion.probe
        self._types = tuple(int(value) for value in self._probe.types)
        self._columns = (
            tuple(str(name) for name in self._probe.signatures[0])
            if self._probe.signatures
            else tuple(str(name) for name in self._probe.columns)
        )
        self._overrides: ColumnOverrides | None = overrides
        self._edits: list[LineEdit] = []

        self.titleLabel = StrongBodyLabel(self.tr("LAMMPS dump: atom type mapping"), self)
        self.subtitleLabel = CaptionLabel(
            f"{self._probe.path.name} · {self.tr('Frames')} {self._probe.frame_count:,}",
            self,
        )
        self.subtitleLabel.setTextColor("#8a8a8a", "#9a9a9a")

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.subtitleLabel)
        self.viewLayout.addWidget(self._build_summary())
        self.viewLayout.addWidget(self._build_mapping_header())
        self.viewLayout.addWidget(self._build_element_card())

        self.fieldsButton = PushButton(
            FluentIcon.EDIT, self.tr("Adjust field mapping…"), self
        )
        self.fieldsButton.clicked.connect(self._edit_fields)
        self.viewLayout.addWidget(self.fieldsButton)

        self.viewLayout.addWidget(self._build_options())

        self.widget.setMinimumWidth(560)
        self.yesButton.setText(self.tr("Import"))
        self.cancelButton.setText(self.tr("Cancel"))
        self._sync_accept_state()

    # ------------------------------------------------------------------ layout

    def _build_summary(self) -> SimpleCardWidget:
        card = SimpleCardWidget(self)
        card.setBorderRadius(8)
        layout = QGridLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(8)
        layout.setColumnMinimumWidth(0, 108)
        layout.setColumnStretch(1, 1)

        style_names = {
            "ortho": self.tr("orthorhombic"),
            "triclinic": self.tr("triclinic"),
            "general": self.tr("general triclinic"),
        }
        style = " / ".join(style_names.get(item, item) for item in self._probe.cell_styles)
        cell_text = f"{_cell_text(self._probe)} · {style} · {self._probe.pbc}"
        all_columns = list(self._probe.columns)
        shown_columns = ", ".join(all_columns[:10]) + (" …" if len(all_columns) > 10 else "")

        rows = (
            (self.tr("Frames"), f"{self._probe.frame_count:,}"),
            (self.tr("Atoms per frame"), _atom_text(self._probe)),
            (self.tr("Cell"), cell_text),
            (self.tr("Columns"), shown_columns or "-"),
        )
        for row, (label, value) in enumerate(rows):
            key = CaptionLabel(label, card)
            value_label = BodyLabel(value, card)
            value_label.setWordWrap(True)
            if label == self.tr("Columns"):
                value_label.setToolTip(", ".join(all_columns))
            layout.addWidget(key, row, 0, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(value_label, row, 1)
        return card

    def _build_mapping_header(self) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(StrongBodyLabel(self.tr("Atom types"), container))
        self.detectedLabel = CaptionLabel(
            _describe_plan(self._probe, self.tr, self._overrides), container
        )
        self.detectedLabel.setWordWrap(True)
        self.detectedLabel.setTextColor("#8a8a8a", "#9a9a9a")
        layout.addWidget(self.detectedLabel)
        if self._suggestion.mapping:
            note = CaptionLabel(
                self.tr("Some elements were pre-filled from matching files in this folder."),
                container,
            )
            note.setWordWrap(True)
            layout.addWidget(note)
        return container

    def _build_element_card(self) -> SimpleCardWidget:
        card = SimpleCardWidget(self)
        card.setBorderRadius(8)
        outer = QVBoxLayout(card)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(0)

        host = QWidget(card)
        rows = QGridLayout(host)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setHorizontalSpacing(8)
        rows.setVerticalSpacing(8)
        for index, type_id in enumerate(self._types):
            rows.addWidget(self._build_element_row(host, type_id), index // 4, index % 4)
        rows.setColumnStretch(4, 1)

        scroll = QScrollArea(card)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        scroll.setWidget(host)
        visible = min(max(1, (len(self._types) + 3) // 4), 5)
        scroll.setFixedHeight(visible * 34 + (visible - 1) * 8 + 4)
        outer.addWidget(scroll)
        return card

    def _build_element_row(self, parent: QWidget, type_id: int) -> QWidget:
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        badge = CaptionLabel(str(type_id), row)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(22, 22)
        badge.setStyleSheet(
            "QLabel { border: 1px solid rgba(128, 128, 128, 0.35); border-radius: 7px; }"
        )
        layout.addWidget(badge)

        edit = LineEdit(row)
        edit.setPlaceholderText(self.tr("Element"))
        edit.setFixedWidth(60)
        edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        completer = QCompleter(list(_ELEMENT_SYMBOLS), edit)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setMaxVisibleItems(10)
        edit.setCompleter(completer)
        known = self._suggestion.mapping.get(type_id)
        if known:
            edit.setText(known)
        edit.textChanged.connect(self._sync_accept_state)
        edit.editingFinished.connect(lambda e=edit: self._normalise_element(e))
        self._edits.append(edit)
        layout.addWidget(edit)
        picker = TransparentToolButton(FluentIcon.EDIT, row)
        picker.setFixedSize(22, 22)
        picker.setToolTip(self.tr("Choose from the periodic table"))
        picker.clicked.connect(lambda _checked=False, e=edit: self._pick_element(e))
        layout.addWidget(picker)
        return row

    def _build_options(self) -> SimpleCardWidget:
        card = SimpleCardWidget(self)
        card.setBorderRadius(8)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(6)
        layout.addWidget(CaptionLabel(self.tr("Import options"), card))

        has_images = all(name in self._probe.columns for name in ("ix", "iy", "iz"))
        self.unwrapCheck = CheckBox(self.tr("Unwrap image flags (ix/iy/iz)"), card)
        self.unwrapCheck.setEnabled(has_images)
        if not has_images:
            self.unwrapCheck.setToolTip(self.tr("This dump has no image-flag columns"))
        self.keepExtraCheck = CheckBox(self.tr("Keep extra columns"), card)
        self.keepExtraCheck.setToolTip(
            self.tr("Velocity, charge, per-atom energy and similar columns.")
        )
        self.keepExtraCheck.setChecked(True)
        self.strictCheck = CheckBox(self.tr("Fail on damaged frames"), card)
        self.rememberCheck = CheckBox(
            self.tr("Remember this mapping"), card
        )
        self.rememberCheck.setChecked(True)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        for index, widget in enumerate(
            (self.unwrapCheck, self.keepExtraCheck, self.strictCheck, self.rememberCheck)
        ):
            grid.addWidget(widget, index // 2, index % 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        return card

    # ----------------------------------------------------------------- behavior

    def _missing_types(self) -> list[int]:
        return [type_id for type_id, edit in zip(self._types, self._edits) if not _element_symbol(edit)]

    def _normalise_element(self, edit: LineEdit) -> None:
        symbol = _element_symbol(edit)
        if symbol and edit.text() != symbol:
            edit.setText(symbol)

    def _pick_element(self, edit: LineEdit) -> None:
        picker = PeriodicTableDialog(self.window())
        current = _element_symbol(edit)
        if current:
            picker.set_selected_element(current)
        if picker.exec() and picker.selected_symbol:
            edit.setText(picker.selected_symbol)
            self._sync_accept_state()

    def _sync_accept_state(self, *_args) -> None:
        self.yesButton.setEnabled(not self._missing_types())

    def _edit_fields(self) -> None:
        accepted, overrides = DumpFieldMappingDialog.ask(
            self._columns, self._overrides, self._probe, self
        )
        if not accepted:
            return
        self._overrides = overrides
        self.detectedLabel.setText(_describe_plan(self._probe, self.tr, overrides))

    def validate(self) -> bool:
        return not self._missing_types()

    def options(self) -> DumpImportOptions:
        """Return the confirmed mapping, column overrides and import flags."""
        return DumpImportOptions(
            element_map={
                type_id: symbol
                for type_id, edit in zip(self._types, self._edits)
                if (symbol := _element_symbol(edit))
            },
            overrides=self._overrides,
            unwrap=self.unwrapCheck.isChecked(),
            keep_extra=self.keepExtraCheck.isChecked(),
            strict=self.strictCheck.isChecked(),
            remember=self.rememberCheck.isChecked(),
        )

    @classmethod
    def ask(
        cls,
        suggestion: ElementSuggestion,
        parent=None,
        overrides: ColumnOverrides | None = None,
    ) -> DumpImportOptions | None:
        """Show the dialog and return the chosen options, or ``None`` on cancel."""
        if parent is None:
            parent = QApplication.activeWindow()
        if parent is None:
            parent = next(
                (widget for widget in QApplication.topLevelWidgets() if widget.isVisible()),
                None,
            )
        if parent is None:
            # No window to mask over (for example a headless import): treat it
            # as a cancellation instead of crashing the loader.
            return None
        dialog = cls(suggestion, parent, overrides)
        accepted = bool(dialog.exec())
        options = dialog.options() if accepted else None
        dialog.hide()
        dialog.deleteLater()
        return options

    @classmethod
    def ask_on_gui_thread(
        cls,
        suggestion: ElementSuggestion,
        overrides: ColumnOverrides | None = None,
    ) -> DumpImportOptions | None:
        """Show the dialog on the GUI thread even when called from a worker."""
        app = QApplication.instance()
        if app is None or QThread.currentThread() is app.thread():
            return cls.ask(suggestion, overrides=overrides)
        bridge = _DialogBridge()
        bridge.moveToThread(app.thread())
        bridge.request.connect(bridge.show, Qt.ConnectionType.BlockingQueuedConnection)
        bridge.request.emit(suggestion, overrides)
        bridge.request.disconnect(bridge.show)
        result = bridge.result
        bridge.deleteLater()
        return result


class _DialogBridge(QObject):
    """Carry one dialog request from a worker thread to the GUI thread."""

    request = Signal(object, object)

    def __init__(self):
        """Create the bridge with no result yet."""
        super().__init__()
        self.result = None

    @Slot(object, object)
    def show(self, suggestion, overrides) -> None:
        """Run the modal dialog in the thread that owns this object."""
        self.result = DumpImportDialog.ask(suggestion, overrides=overrides)
