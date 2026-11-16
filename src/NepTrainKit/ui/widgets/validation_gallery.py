"""Scrollable previews of saved validation results."""

from PySide6.QtCore import QSize, QTimer, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from NepTrainKit import palette
from NepTrainKit.ui.widgets.validation_presentation import presentation_for


class ValidationGallery(QListWidget):
    itemOpened = Signal(str)

    def __init__(self, render, parent=None):
        super().__init__(parent)
        self.render = render
        self.records = []
        self.loaded = set()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._load_visible)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setUniformItemSizes(True)
        self.setWordWrap(False)
        self.setMovement(QListWidget.Movement.Static)
        self.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSpacing(4)
        self.setStyleSheet(
            f"QListWidget {{ border: none; background: {palette.PLOT_BACKGROUND}; }}"
            f"QListWidget::item {{ background: {palette.PLOT_BACKGROUND}; border: 1px solid {palette.BORDER}; border-radius: 8px; }}"
            f"QListWidget::item:hover, QListWidget::item:selected {{ border-color: {palette.ACCENT}; color: {palette.TEXT_PRIMARY}; }}"
        )
        self.verticalScrollBar().valueChanged.connect(lambda _: self.timer.start(0))
        self.itemClicked.connect(self._open)
        self.itemActivated.connect(self._open)

    def _open(self, item):
        if self.isVisible():
            self.itemOpened.emit(self.records[self.row(item)]["id"])

    def set_records(self, records):
        scroll = self.verticalScrollBar().value() if records == self.records else 0
        self.clear()
        self.loaded.clear()
        self.records = records
        for record in records:
            view = presentation_for(record["protocol"])
            title = view.item_title(record) if view.group_title else record["name"].removeprefix(
                (record.get("group_name") or "") + " / ")
            caption = f"{view.item_badge(record)} · {title}"
            item = QListWidgetItem(caption, self)
            item.setData(Qt.ItemDataRole.UserRole, caption)
            item.setToolTip(record["name"])
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fit()
        self.doItemsLayout()
        self.verticalScrollBar().setValue(scroll)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "timer"):
            self._fit()

    def _fit(self):
        width = max(160, self.viewport().width() - 16)
        columns = min(3, max(1, width // 340))
        cell = min(500, width // columns)
        plot_width = cell - 12
        plot_height = max(224, round(plot_width * 0.53))
        self.setGridSize(QSize(cell, plot_height + 35))
        size = QSize(plot_width, plot_height)
        if size != self.iconSize():
            for index in self.loaded:
                self.item(index).setIcon(QIcon())
            self.loaded.clear()
            self.setIconSize(size)
        hint = QSize(cell - 8, plot_height + 25)
        for i in range(self.count()):
            if self.item(i).sizeHint() != hint:
                self.item(i).setSizeHint(hint)
        self.timer.start(0)

    def _load_visible(self):
        if not self.isVisible():
            return
        viewport = self.viewport().rect()
        visible = [i for i in range(self.count()) if self.visualItemRect(self.item(i)).intersects(viewport)]
        for i in self.loaded.difference(visible):
            self.item(i).setIcon(QIcon())
            self.loaded.remove(i)
        for i in visible:
            if i not in self.loaded:
                image, x_label, y_label = self.render(self.records[i], self.iconSize())
                item = self.item(i)
                title = item.data(Qt.ItemDataRole.UserRole)
                x_label = {"volume_ratio": "V/V₀", "moment_ratio": "M/M₀"}.get(x_label, x_label)
                axes = " → ".join(filter(None, (x_label, y_label)))
                item.setText(f"{title}   ·   {axes}" if axes else title)
                item.setToolTip("\n".join(filter(None, (self.records[i]["name"], axes))))
                item.setIcon(QIcon(image))
                self.loaded.add(i)
                self.timer.start(0)
                break
