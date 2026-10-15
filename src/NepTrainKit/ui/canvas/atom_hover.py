"""Hover picking and the floating atom card shared by both structure viewers."""
from __future__ import annotations

from html import escape

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPoint, Qt, QTimer
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from NepTrainKit import palette
from NepTrainKit.core.atom_inspection import AtomInspection, inspect_atom


def _tr(text: str) -> str:
    return QCoreApplication.translate("AtomInfoCard", text)


def _vector_cells(vector: np.ndarray) -> str:
    return "".join(f"<td align='right'>{float(value):.3f}</td>" for value in vector)


def _row(label: str, cells: str, note: str = "") -> str:
    return (
        f"<tr><td style='color:{palette.TEXT_SECONDARY}; padding-right:10px'>"
        f"{escape(label)}</td>{cells}"
        f"<td style='padding-left:10px'>{note}</td></tr>"
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
    rows = [
        _row(_tr("Position (Å)"), _vector_cells(info.position)),
    ]
    if info.fractional is not None:
        rows.append(_row(_tr("Fractional"), _vector_cells(info.fractional)))
    if info.dft_force is not None:
        rows.append(
            _row(
                _tr("DFT force"),
                _vector_cells(info.dft_force),
                f"|F| {np.linalg.norm(info.dft_force):.3f} eV/Å",
            )
        )
    if info.ml_force is not None:
        error = info.force_error
        rows.append(
            _row(
                _tr("Predicted force"),
                _vector_cells(info.ml_force),
                "" if error is None else f"|ΔF| {error:.3f} eV/Å",
            )
        )
    if info.spin is not None:
        rows.append(
            _row(
                _tr("Spin"),
                _vector_cells(info.spin),
                f"|m| {np.linalg.norm(info.spin):.3f} μB",
            )
        )
    if info.dft_mforce is not None:
        rows.append(_row(_tr("DFT magnetic force"), _vector_cells(info.dft_mforce)))
    if info.ml_mforce is not None:
        rows.append(
            _row(_tr("Predicted magnetic force"), _vector_cells(info.ml_mforce))
        )
    if info.nearest_distance is not None:
        neighbor = f"{escape(info.nearest_element or '')} #{info.nearest_index}"
        if info.nearest_index == info.index:
            neighbor += " " + escape(_tr("(periodic image)"))
        rows.append(
            _row(
                _tr("Nearest"),
                f"<td colspan='3'>{info.nearest_distance:.3f} Å → {neighbor}</td>",
            )
        )
    for key, values in info.extra:
        rows.append(_row(key, f"<td colspan='3'>{_extra_value(values)}</td>"))
    title = f"<b>#{info.index}&nbsp;&nbsp;{escape(info.element)}</b>"
    return f"{title}<table cellspacing='0' cellpadding='1'>{''.join(rows)}</table>"


class AtomInfoCard(QFrame):
    """Tooltip-style window that floats above either OpenGL backend."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setObjectName("atomInfoCard")
        self.setStyleSheet(
            f"QFrame#atomInfoCard {{ background: {palette.SURFACE}; "
            f"border: 1px solid {palette.BORDER}; }}"
            f"QLabel {{ color: {palette.TEXT_PRIMARY}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        self.label = QLabel(self)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.label)

    def show_info(self, info: AtomInspection, global_pos: QPoint) -> None:
        self.label.setText(format_atom_info(info))
        self.adjustSize()
        self.move_near(global_pos)
        self.show()

    def move_near(self, global_pos: QPoint) -> None:
        """Place the card beside the cursor while keeping it on screen."""
        x = global_pos.x() + 16
        y = global_pos.y() + 16
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            if x + self.width() > area.right():
                x = global_pos.x() - self.width() - 16
            if y + self.height() > area.bottom():
                y = global_pos.y() - self.height() - 16
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
        self.card = AtomInfoCard()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._pick)
        host.setMouseTracking(True)
        host.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt override
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
