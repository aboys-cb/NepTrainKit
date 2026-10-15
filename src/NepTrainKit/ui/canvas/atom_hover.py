"""Hover picking and the floating atom card shared by both structure viewers."""
from __future__ import annotations

from html import escape

import numpy as np
from ase.data import atomic_numbers
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QGuiApplication
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QLabel, QVBoxLayout, QWidget

from NepTrainKit import palette
from NepTrainKit.core.atom_inspection import AtomInspection, inspect_atom
from NepTrainKit.core.structure import table_info


def _tr(text: str) -> str:
    return QCoreApplication.translate("AtomInfoCard", text)


_NUMBER_FONT = "'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace"


def _element_color(element: str) -> str:
    try:
        return str(table_info[str(atomic_numbers[element])]["color"])
    except (KeyError, TypeError):
        return palette.TEXT_SECONDARY


def _number(value: float) -> str:
    value = float(value)
    magnitude = abs(value)
    if magnitude != 0.0 and (magnitude < 1e-3 or magnitude >= 1e4):
        return f"{value:.2e}"
    return f"{value:.3f}"


def _numeric_cell(text: str, *, bold: bool = False, color: str | None = None) -> str:
    style = f"font-family:{_NUMBER_FONT}; padding-left:12px;"
    if bold:
        style += " font-weight:600;"
    if color:
        style += f" color:{color};"
    return f"<td align='right' style=\"{style}\">{text}</td>"


def _label_cell(text: str) -> str:
    return (
        f"<td style='color:{palette.TEXT_SECONDARY}; padding-right:6px;'>"
        f"{escape(text)}</td>"
    )


def _vector_row(label: str, vector, magnitude: str = "", *, emphasize: bool = False) -> str:
    cells = "".join(_numeric_cell(_number(value)) for value in vector)
    magnitude_cell = _numeric_cell(
        magnitude,
        bold=emphasize,
        color=palette.PLOT_SELECTED if emphasize else palette.TEXT_SECONDARY,
    )
    return f"<tr>{_label_cell(label)}{cells}{magnitude_cell}</tr>"


def _text_row(label: str, text: str) -> str:
    return f"<tr>{_label_cell(label)}<td colspan='4' style='padding-left:12px;'>{text}</td></tr>"


def _section(title: str) -> str:
    return (
        "<tr><td colspan='5' style='padding-top:7px; padding-bottom:1px; "
        f"color:{palette.ACCENT}; font-size:11px; font-weight:600;'>{escape(title)}</td></tr>"
    )


def _extra_value(values: tuple[object, ...]) -> str:
    parts = []
    for value in values[:6]:
        if isinstance(value, float):
            parts.append(f"{value:.4g}")
        else:
            parts.append(escape(str(value)))
    if len(values) > 6:
        parts.append("…")
    return " ".join(parts)


def format_atom_info(info: AtomInspection) -> str:
    """Return the rich-text body of the hover card."""
    axis_header = "".join(
        f"<td align='right' style='color:{palette.TEXT_SECONDARY}; font-size:11px; "
        f"padding-left:12px;'>{axis}</td>"
        for axis in ("x", "y", "z")
    )
    rows = [f"<tr><td></td>{axis_header}<td></td></tr>", _section(_tr("Position"))]
    rows.append(_vector_row(_tr("Cartesian (Å)"), info.position))
    if info.fractional is not None:
        rows.append(_vector_row(_tr("Fractional"), info.fractional))

    if info.dft_force is not None or info.ml_force is not None:
        rows.append(_section(_tr("Force · eV/Å")))
        for label, vector in ((_tr("DFT"), info.dft_force), (_tr("Predicted"), info.ml_force)):
            if vector is not None:
                rows.append(
                    _vector_row(label, vector, f"|F| {_number(np.linalg.norm(vector))}")
                )
        if info.force_error is not None:
            rows.append(
                _vector_row(
                    _tr("Error"),
                    info.ml_force - info.dft_force,
                    f"|ΔF| {_number(info.force_error)}",
                    emphasize=True,
                )
            )

    if info.spin is not None or info.dft_mforce is not None or info.ml_mforce is not None:
        rows.append(_section(_tr("Magnetism")))
        if info.spin is not None:
            rows.append(
                _vector_row(
                    _tr("Spin (μB)"), info.spin, f"|m| {_number(np.linalg.norm(info.spin))}"
                )
            )
        for label, vector in (
            (_tr("DFT mag. force"), info.dft_mforce),
            (_tr("Predicted mag. force"), info.ml_mforce),
        ):
            if vector is not None:
                rows.append(_vector_row(label, vector))

    if info.nearest_distance is not None or info.extra:
        rows.append(_section(_tr("Environment")))
        if info.nearest_distance is not None:
            neighbor = f"{escape(info.nearest_element or '')} #{info.nearest_index}"
            if info.nearest_index == info.index:
                neighbor += " " + escape(_tr("(periodic image)"))
            rows.append(
                _text_row(
                    _tr("Nearest"),
                    f"<span style=\"font-family:{_NUMBER_FONT}\">"
                    f"{info.nearest_distance:.3f} Å</span>"
                    f"&nbsp;&nbsp;→&nbsp;&nbsp;{neighbor}",
                )
            )
        for key, values in info.extra:
            rows.append(
                _text_row(
                    key,
                    f"<span style=\"font-family:{_NUMBER_FONT}\">{_extra_value(values)}</span>",
                )
            )

    color = _element_color(info.element)
    title = (
        f"<span style='color:{color}; font-size:15px;'>●</span>&nbsp;"
        f"<span style='font-size:14px; font-weight:600;'>{escape(info.element)}</span>"
        f"&nbsp;&nbsp;<span style='color:{palette.TEXT_SECONDARY};'>#{info.index}</span>"
    )
    return (
        f"<div>{title}</div>"
        f"<table cellspacing='0' cellpadding='1'>{''.join(rows)}</table>"
    )


class AtomInfoCard(QWidget):
    """Tooltip-style window that floats above either OpenGL backend."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        # Transparent window so the rounded panel can cast a soft shadow.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 12)
        self.panel = QFrame(self)
        self.panel.setObjectName("atomInfoCard")
        self.panel.setStyleSheet(
            f"QFrame#atomInfoCard {{ background: {palette.SURFACE}; "
            f"border: 1px solid {palette.BORDER}; border-radius: 10px; }}"
            f"QLabel {{ color: {palette.TEXT_PRIMARY}; background: transparent; }}"
        )
        shadow = QGraphicsDropShadowEffect(self.panel)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(15, 23, 42, 45))
        self.panel.setGraphicsEffect(shadow)
        outer.addWidget(self.panel)
        layout = QVBoxLayout(self.panel)
        layout.setContentsMargins(12, 9, 12, 10)
        self.label = QLabel(self.panel)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.label)

    def show_info(self, info: AtomInspection, global_pos: QPoint) -> None:
        self.label.setText(format_atom_info(info))
        self.adjustSize()
        self.move_near(global_pos)
        self.show()

    def move_near(self, global_pos: QPoint) -> None:
        """Place the card beside the cursor while keeping it on screen."""
        x = global_pos.x() + 12
        y = global_pos.y() + 12
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            if x + self.width() > area.right():
                x = global_pos.x() - self.width() - 12
            if y + self.height() > area.bottom():
                y = global_pos.y() - self.height() - 12
            x = max(area.left(), x)
            y = max(area.top(), y)
        self.move(x, y)


class AtomHoverController(QObject):
    """Show the atom card after the pointer rests on an atom.

    ``view`` provides ``structure``, ``atom_index_at((x, y))`` in host-widget
    logical pixels, and ``set_hover_atom(index | None)`` for the highlight.
    """

    DELAY_MS = 80

    def __init__(self, host: QWidget, view):
        super().__init__(host)
        self._host = host
        self._view = view
        self._index: int | None = None
        self._pos: tuple[float, float] | None = None
        self.card = AtomInfoCard(host)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._pick)
        host.setMouseTracking(True)
        host.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
        # Qt may deliver Hide/Leave while cyclic GC has already cleared the
        # Python attributes, before removing this filter with its host.
        if watched is not getattr(self, "_host", None):
            return False
        kind = event.type()
        if kind == QEvent.Type.MouseMove:
            if event.buttons() != Qt.MouseButton.NoButton:
                # Rotating or panning: no picking until the drag ends.
                self.clear()
            else:
                point = event.position()
                self._pos = (float(point.x()), float(point.y()))
                self._timer.start(self.DELAY_MS)
        elif kind in (
            QEvent.Type.Leave,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.Wheel,
            QEvent.Type.Hide,
        ):
            self.clear()
        return False

    def _pick(self) -> None:
        structure = getattr(self._view, "structure", None)
        if structure is None or self._pos is None:
            self.clear()
            return
        try:
            index = self._view.atom_index_at(self._pos)
        except Exception:
            index = None
        if index is None:
            self.clear()
            return
        global_pos = QCursor.pos()
        if index == self._index and self.card.isVisible():
            self.card.move_near(global_pos)
            return
        info = inspect_atom(structure, index)
        if info is None:
            self.clear()
            return
        self._index = index
        self._view.set_hover_atom(index)
        self.card.show_info(info, global_pos)

    def clear(self) -> None:
        """Hide the card and highlight, e.g. when the structure changes."""
        self._timer.stop()
        if self._index is not None:
            self._index = None
            try:
                self._view.set_hover_atom(None)
            except Exception:
                pass
        self.card.hide()


def projected_atom_discs(
    clip: np.ndarray,
    clip_offsets: list[np.ndarray],
    to_screen,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert homogeneous atom centres to screen discs and depth.

    ``clip`` holds (N, 4) homogeneous centres; ``clip_offsets`` the same points
    moved by one radius along each world axis.  The largest offset in pixels
    approximates the projected sphere radius.
    """
    def divide(points: np.ndarray) -> np.ndarray:
        w = points[:, 3:4]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(w) > 1e-12, points[:, :3] / w, np.nan)

    centre = divide(clip)
    screen = to_screen(centre)
    radius = np.zeros(len(clip))
    for offset in clip_offsets:
        moved = to_screen(divide(offset))
        radius = np.maximum(radius, np.linalg.norm(moved - screen, axis=1))
    return screen, centre[:, 2], radius


__all__ = [
    "AtomHoverController",
    "AtomInfoCard",
    "format_atom_info",
    "projected_atom_discs",
]
