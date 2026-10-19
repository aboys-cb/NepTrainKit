"""Searchable multi-selection of saved versions for validation overlays."""

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QListWidgetItem, QVBoxLayout
from qfluentwidgets import CaptionLabel, FluentIcon, ListWidget, PushButton, SearchLineEdit, TransparentPushButton

from NepTrainKit import palette


class _ComparisonPopup(QFrame):
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            event.accept()
        else:
            super().keyPressEvent(event)


class ComparisonPicker(PushButton):
    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIcon(FluentIcon.CHEVRON_DOWN_MED)
        self._selected = []
        self._options = []
        self.popup = _ComparisonPopup(self, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.popup.setObjectName("comparisonPopup")
        self.popup.setStyleSheet(f"#comparisonPopup {{ background: {palette.SURFACE}; border: 1px solid {palette.BORDER}; border-radius: 8px; }}")
        layout = QVBoxLayout(self.popup)
        layout.setContentsMargins(12, 12, 12, 12)
        self.search = SearchLineEdit(self.popup)
        self.search.setPlaceholderText(self.tr("Search project or version"))
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)
        self.list = ListWidget(self.popup)
        self.list.itemChanged.connect(self._item_changed)
        layout.addWidget(self.list, 1)
        footer = QHBoxLayout()
        self.hint = CaptionLabel(self.tr("Select multiple versions; uncheck to remove."), self.popup)
        self.hint.setWordWrap(True)
        footer.addWidget(self.hint, 1)
        clear = TransparentPushButton(self.tr("Clear selection"), self.popup)
        clear.clicked.connect(lambda: self.set_selected_ids([]))
        footer.addWidget(clear)
        layout.addLayout(footer)
        self.clicked.connect(self.open_popup)
        self._update_text()

    def selected_ids(self):
        return list(self._selected)

    def set_options(self, options, selected_ids):
        """Options are (model id, full label, validation status), excluding current."""
        self._options = list(options)
        valid = {key for key, _label, _status in self._options}
        self._selected = list(dict.fromkeys(key for key in selected_ids if key in valid))
        self.list.blockSignals(True)
        self.list.clear()
        for key, label, status in self._options:
            item = QListWidgetItem(f"{label}\n{status}")
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if key in self._selected else Qt.CheckState.Unchecked)
            item.setSizeHint(QSize(0, 52))
            item.setToolTip(label + "\n" + status)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.setEnabled(bool(self._options))
        if not self._options:
            self.popup.hide()
        self._filter(self.search.text())
        self._update_text()

    def set_selected_ids(self, ids):
        previous = self._selected[:]
        self.set_options(self._options, ids)
        if previous != self._selected:
            self.selectionChanged.emit()

    def remove_id(self, key):
        self.set_selected_ids([item for item in self._selected if item != key])

    def _item_changed(self, item):
        key = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            if key not in self._selected:
                self._selected.append(key)
        elif key in self._selected:
            self._selected.remove(key)
        self._update_text()
        self.selectionChanged.emit()

    def _filter(self, text):
        query = text.strip().casefold()
        for index in range(self.list.count()):
            item = self.list.item(index)
            item.setHidden(query not in item.text().casefold())

    def _update_text(self):
        self.setText(self.tr("{count} comparison models selected").format(count=len(self._selected)) if self._selected else self.tr("Select comparison models…"))
        self.setToolTip("\n".join(label for key, label, _status in self._options if key in self._selected) or self.tr("Select multiple saved versions to compare."))

    def open_popup(self):
        self.search.clear()
        screen = self.screen().availableGeometry()
        self.popup.resize(min(max(400, self.width()), screen.width()), min(410, screen.height()))
        point = self.mapToGlobal(QPoint(0, self.height() + 4))
        x = max(screen.left(), min(point.x(), screen.right() - self.popup.width() + 1))
        y = point.y() if point.y() + self.popup.height() <= screen.bottom() + 1 else self.mapToGlobal(QPoint(0, 0)).y() - self.popup.height() - 4
        self.popup.move(x, max(screen.top(), y))
        self.popup.show()
        self.search.setFocus()
