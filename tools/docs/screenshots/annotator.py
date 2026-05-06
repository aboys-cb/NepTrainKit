"""Qt-based screenshot annotation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from registry import Annotation


ACCENT = QColor(0, 151, 157, 255)
ACCENT_FILL = QColor(0, 151, 157, 26)
PANEL_BG = QColor(255, 255, 255, 238)
PANEL_BORDER = QColor(205, 220, 225, 255)
INK = QColor(35, 45, 55, 255)
WHITE = QColor(255, 255, 255, 255)
SHADOW = QColor(0, 0, 0, 65)


def _resolve_attr(root: object, path: str) -> object:
    current: object = root
    for part in path.split("."):
        if not part:
            continue
        if isinstance(current, Sequence) and not isinstance(current, str) and part.isdigit():
            current = current[int(part)]
            continue
        current = getattr(current, part)
    return current


def _widget_rect(target: QWidget, capture_widget: QWidget) -> QRect:
    top_left = target.mapTo(capture_widget, QPoint(0, 0))
    return QRect(top_left, target.size())


def _action_rect(window: QWidget, target: str, capture_widget: QWidget) -> QRect:
    toolbar_path, action_name = target.split(":", 1)
    toolbar = _resolve_attr(window, toolbar_path)
    if hasattr(toolbar, "_actions"):
        action = getattr(toolbar, "_actions")[action_name]
    else:
        matches = [
            action
            for action in toolbar.actions()
            if action.text() == action_name or action.objectName() == action_name
        ]
        if not matches:
            raise RuntimeError(f"Cannot resolve action '{action_name}' on '{toolbar_path}'")
        action = matches[0]
    if hasattr(toolbar, "widgetForAction"):
        action_widget = toolbar.widgetForAction(action)
    else:
        action_widget = None
        if hasattr(action, "associatedObjects"):
            widgets = [
                obj
                for obj in action.associatedObjects()
                if isinstance(obj, QWidget) and obj.isVisible() and obj.width() > 0 and obj.height() > 0
            ]
            if widgets:
                action_widget = min(widgets, key=lambda widget: widget.width() * widget.height())
    if action_widget is None:
        raise RuntimeError(f"Cannot resolve widget for action '{action_name}' on '{toolbar_path}'")
    return _widget_rect(action_widget, capture_widget)


def resolve_rect(window: QWidget, capture_widget: QWidget, target: str | tuple[int, int, int, int]) -> QRect:
    """Resolve a registry target to a rectangle in capture-widget coordinates."""
    if isinstance(target, tuple):
        x, y, width, height = target
        return QRect(x, y, width, height)
    if target.startswith("widget:"):
        widget = _resolve_attr(window, target.removeprefix("widget:"))
        if not isinstance(widget, QWidget):
            raise RuntimeError(f"Target is not a QWidget: {target}")
        return _widget_rect(widget, capture_widget)
    if target.startswith("action:"):
        return _action_rect(window, target.removeprefix("action:"), capture_widget)
    raise ValueError(f"Unsupported annotation target: {target}")


def _badge_point(rect: QRect, placement: str | tuple[int, int] | None, image_width: int, image_height: int) -> QPoint:
    if isinstance(placement, tuple):
        return QPoint(*placement)
    place = placement or "right"
    margin = 10
    if place == "right":
        return QPoint(min(rect.right() + margin, image_width - 24), rect.center().y())
    if place == "left":
        return QPoint(max(rect.left() - margin, 24), rect.center().y())
    if place == "top-right":
        return QPoint(min(rect.right() - 18, image_width - 24), max(rect.top() + 18, 24))
    if place == "top-left":
        return QPoint(max(rect.left() + 18, 24), max(rect.top() + 18, 24))
    if place == "bottom-right":
        return QPoint(min(rect.right() - 18, image_width - 24), min(rect.bottom() - 18, image_height - 24))
    if place == "bottom-left":
        return QPoint(max(rect.left() + 18, 24), min(rect.bottom() - 18, image_height - 24))
    raise ValueError(f"Unsupported badge placement: {placement}")


def _draw_arrow(painter: QPainter, start: QPoint, end: QPoint) -> None:
    painter.setPen(QPen(ACCENT, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawLine(start, end)
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    length = 12
    for delta in (2.55, -2.55):
        point = QPoint(
            int(end.x() - length * math.cos(angle + delta)),
            int(end.y() - length * math.sin(angle + delta)),
        )
        painter.drawLine(end, point)


def annotate(pixmap: QPixmap, capture_widget: QWidget, annotations: tuple[Annotation, ...], title: str) -> QPixmap:
    """Draw numbered callouts and a compact legend onto a screenshot pixmap."""
    if not annotations:
        return pixmap

    result = QPixmap(pixmap)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    title_font = QFont("Arial", 17, QFont.Weight.Bold)
    text_font = QFont("Arial", 14)
    number_font = QFont("Arial", 13, QFont.Weight.Bold)

    resolved: list[tuple[Annotation, QRect, QPoint]] = []
    for annotation in annotations:
        rect = resolve_rect(capture_widget, capture_widget, annotation.target)
        badge = _badge_point(rect, annotation.badge, result.width(), result.height())
        resolved.append((annotation, rect, badge))

        painter.setPen(QPen(ACCENT, 3))
        painter.setBrush(ACCENT_FILL)
        painter.drawRoundedRect(rect, 8, 8)

        if not rect.adjusted(-6, -6, 6, 6).contains(badge):
            _draw_arrow(painter, badge, rect.center())

        radius = 14
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(SHADOW)
        painter.drawEllipse(badge + QPoint(2, 2), radius, radius)
        painter.setBrush(ACCENT)
        painter.drawEllipse(badge, radius, radius)
        painter.setPen(QPen(WHITE, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(badge, radius, radius)

        painter.setFont(number_font)
        painter.setPen(WHITE)
        painter.drawText(QRect(badge.x() - radius, badge.y() - radius - 1, radius * 2, radius * 2), Qt.AlignmentFlag.AlignCenter, annotation.number)

    row_height = 25
    panel_width = 370
    panel_height = 46 + row_height * len(annotations)
    panel_x = max(24, result.width() - panel_width - 66)
    panel_y = max(24, result.height() - panel_height - 48)
    panel_rect = QRect(panel_x, panel_y, panel_width, panel_height)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(SHADOW)
    painter.drawRoundedRect(panel_rect.translated(2, 2), 12, 12)
    painter.setBrush(PANEL_BG)
    painter.setPen(QPen(PANEL_BORDER, 1))
    painter.drawRoundedRect(panel_rect, 12, 12)

    painter.setFont(title_font)
    painter.setPen(INK)
    painter.drawText(panel_x + 18, panel_y + 32, title)

    painter.setFont(text_font)
    y = panel_y + 58
    for annotation, _rect, _badge in resolved:
        center = QPoint(panel_x + 31, y - 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(ACCENT)
        painter.drawEllipse(center, 11, 11)
        painter.setFont(number_font)
        painter.setPen(WHITE)
        painter.drawText(QRect(center.x() - 11, center.y() - 12, 22, 22), Qt.AlignmentFlag.AlignCenter, annotation.number)
        painter.setFont(text_font)
        painter.setPen(INK)
        painter.drawText(panel_x + 54, y, annotation.label)
        y += row_height

    painter.end()
    return result
