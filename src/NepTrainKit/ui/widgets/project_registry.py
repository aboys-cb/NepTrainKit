"""Model identity, validation coverage and role controls for the project table."""

from PySide6.QtCore import QEvent, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QHeaderView, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from NepTrainKit import palette

RMSE_COLUMNS = {3: ("E", "meV/atom"), 4: ("F", "meV/Å"), 5: ("V", "meV/atom")}
ROLE_COLORS = {
    "current": (palette.INFO, palette.INFO_SOFT),
    "baseline": (palette.WARNING_STRONG, palette.WARNING_SOFT),
    "compare": (palette.ACCENT_PRESSED, palette.ACCENT_SOFT),
    "hidden": (palette.TEXT_SECONDARY, palette.SURFACE),
}


class RegistryHeader(QHeaderView):
    """One shared heading across the visible error columns, with per-column units."""

    def paintSection(self, painter, rect, logicalIndex):
        if logicalIndex not in RMSE_COLUMNS:
            return super().paintSection(painter, rect, logicalIndex)
        visible = [c for c in RMSE_COLUMNS if not self.isSectionHidden(c)]
        first = min(self.sectionViewportPosition(c) for c in visible)
        group = QRect(first, rect.top(), sum(self.sectionSize(c) for c in visible), rect.height() // 2)
        metric, unit = RMSE_COLUMNS[logicalIndex]
        painter.save()
        painter.setClipRect(rect, Qt.ClipOperation.IntersectClip)
        painter.fillRect(rect, QColor(palette.SURFACE_SUBTLE))
        font = QFont(self.font())
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor(palette.TEXT_PRIMARY))
        painter.drawText(group, Qt.AlignmentFlag.AlignCenter,
                         self.model().headerData(logicalIndex, self.orientation(), Qt.ItemDataRole.ToolTipRole))
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QColor(palette.TEXT_SECONDARY))
        if self.sortIndicatorSection() == logicalIndex:
            metric = ("↑ " if self.sortIndicatorOrder() == Qt.SortOrder.AscendingOrder else "↓ ") + metric
        painter.drawText(rect.adjusted(0, group.height(), -8, -3),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         f"{metric} · {unit}")
        painter.setPen(QColor(palette.BORDER))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        painter.restore()


class RegistryDelegate(QStyledItemDelegate):
    def __init__(self, registry):
        super().__init__(registry._view)
        self.registry = registry

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(size.width(), max(55, option.fontMetrics.height() * 2 + 12))

    def paint(self, painter, option, index):
        registry = self.registry
        column = index.column()
        if column not in (1, 2, 8, 9):
            return super().paint(painter, option, index)
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        rect = opt.rect.adjusted(10, 0, -8, 0)
        key = index.internalPointer().data(0)
        model = registry.model_item_dict[key]
        summary = registry.score_rows.get(key, {})
        painter.save()
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        font = QFont(opt.font)
        text_color = opt.palette.color(QPalette.ColorRole.Text)

        def label(box, text, color):
            painter.setPen(color)
            painter.drawText(box, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             painter.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, box.width()))

        if column == 9:
            role = registry.role(key)
            caption = registry.role_caption(role)
            ink, fill = ROLE_COLORS[role]
            button = rect.adjusted(0, 0, 0, 0)
            button.setTop(rect.center().y() - 14)
            button.setHeight(28)
            painter.setPen(QColor(palette.BORDER))
            painter.setBrush(QColor(fill))
            painter.drawRoundedRect(button, 6, 6)
            if role != "hidden":
                font.setWeight(QFont.Weight.DemiBold)
                painter.setFont(font)
            label(button.adjusted(9, 0, -20, 0), caption, QColor(ink))
            x, y = button.right()-12, button.center().y()
            painter.drawLine(x-10, y-6, x-10, y+6)
            painter.setPen(QColor(ink))
            painter.drawLine(x-3, y-1, x, y+2)
            painter.drawLine(x, y+2, x+3, y-1)
        else:
            if column == 1:
                first = model.name
                role = registry.role(key)
                if role != "hidden":
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(ROLE_COLORS[role][0]))
                    painter.drawRoundedRect(QRect(rect.left() - 8, rect.center().y() - 12, 3, 24), 1, 1)
                second = registry.project_names.get(model.project_id, "") or "—"
                if registry._view.isColumnHidden(2):
                    second += "  ·  " + (model.model_kind or model.model_type).upper()
                if summary.get("starred"):
                    first = "★ " + first
            elif column == 2:
                first = (model.model_kind or model.model_type).upper()
                second = registry.tr("{count} frames").format(count=model.data_size) if model.data_size is not None else ""
            else:
                first, second = registry.metric_text(key)
                cell = registry.metric_cell(key)
            font.setBold(column == 1 or column == 8 and cell.get("value") is not None)
            if column == 1:
                if font.pixelSize() > 0:
                    font.setPixelSize(font.pixelSize() + 1)
                else:
                    font.setPointSizeF(font.pointSizeF() + 1)
            painter.setFont(font)
            line = painter.fontMetrics().height()
            upper = rect.adjusted(0, 0, 0, 0)
            upper.setTop(rect.center().y()-line if second else rect.center().y()-line//2)
            upper.setHeight(line)
            label(upper, first, text_color if font.bold() else QColor(palette.TEXT_SECONDARY))
            if second:
                font.setBold(False)
                font.setPointSizeF(max(8., opt.font.pointSizeF()-1))
                painter.setFont(font)
                lower = upper.translated(0, line+1)
                ink = palette.TEXT_SECONDARY
                if column == 8:
                    ink = {"better": palette.SUCCESS, "worse": palette.WARNING_STRONG}.get(cell.get("verdict"), ink)
                label(lower, second, QColor(ink))
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if (index.column() == 9 and event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton):
            key = index.internalPointer().data(0)
            self.registry._view.setCurrentIndex(index)
            if event.position().x() >= option.rect.right()-32:
                self.registry.show_role_menu(key, self.registry._view.viewport().mapToGlobal(option.rect.bottomLeft()))
            else:
                self.registry.activate_role(key)
            return True
        return super().editorEvent(event, model, option, index)
