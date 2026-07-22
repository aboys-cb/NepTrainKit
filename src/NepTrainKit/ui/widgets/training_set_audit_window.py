"""Owned floating-window support for the Training Set Audit page."""
from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtGui import QCloseEvent, QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    FluentIcon,
    IconWidget,
    PrimaryPushButton,
    PushButton,
    SimpleCardWidget,
    SubtitleLabel,
)

from NepTrainKit.config import Config


_CONFIG_SECTION = "training_set_audit"
_GEOMETRY_OPTION = "floating_window_geometry"


class TrainingSetAuditHost(QWidget):
    """Navigation page that either contains the audit or explains where it is."""

    locateRequested = Signal()
    restoreRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TrainingSetAuditHost")
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self._content: QWidget | None = None
        self._page: QWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._layout = layout

        self.placeholder = SimpleCardWidget(self)
        self.placeholder.setObjectName("auditDetachedPlaceholder")
        self.placeholder.setMaximumWidth(560)
        self.placeholder.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        placeholder_layout = QVBoxLayout(self.placeholder)
        placeholder_layout.setContentsMargins(32, 28, 32, 28)
        placeholder_layout.setSpacing(10)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = IconWidget(FluentIcon.FULL_SCREEN, self.placeholder)
        icon.setFixedSize(36, 36)
        title = SubtitleLabel(
            self.tr("Training Set Check is open in a separate window"),
            self.placeholder,
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = BodyLabel(
            self.tr(
                "Keep Dataset Display on this screen and move the check window to another screen for linked review."
            ),
            self.placeholder,
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        locate_button = PushButton(
            FluentIcon.PIN,
            self.tr("Locate window"),
            self.placeholder,
        )
        locate_button.setAccessibleName(self.tr("Locate Training Set Check window"))
        locate_button.clicked.connect(self.locateRequested)
        restore_button = PrimaryPushButton(
            FluentIcon.BACK_TO_WINDOW,
            self.tr("Return to main window"),
            self.placeholder,
        )
        restore_button.setAccessibleName(
            self.tr("Return Training Set Check to main window")
        )
        restore_button.clicked.connect(self.restoreRequested)
        button_row.addWidget(locate_button)
        button_row.addWidget(restore_button)

        placeholder_layout.addWidget(icon, alignment=Qt.AlignmentFlag.AlignHCenter)
        placeholder_layout.addWidget(title)
        placeholder_layout.addWidget(hint)
        placeholder_layout.addSpacing(4)
        placeholder_layout.addLayout(button_row)
        layout.addWidget(
            self.placeholder,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        self.placeholder.hide()

    @property
    def content(self) -> QWidget | None:
        return self._content

    def attach(self, widget: QWidget) -> None:
        """Place the shared audit widget back into the navigation page."""
        if self._content is widget and widget.parentWidget() is self:
            return
        self._content = widget
        self._page = widget
        widget.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        widget.setParent(self)
        self._layout.insertWidget(0, widget)
        widget.show()
        self.placeholder.hide()

    def take(self) -> QWidget | None:
        """Release the shared widget while leaving a useful docked placeholder."""
        widget = self._content
        if widget is None:
            return None
        self._layout.removeWidget(widget)
        widget.setParent(None)
        self._content = None
        self.placeholder.show()
        return widget

    def open_file(self) -> None:
        """Preserve the audit page's global Open action while it is wrapped."""
        handler = getattr(self._page, "open_file", None)
        if callable(handler):
            handler()


class TrainingSetAuditWindow(QWidget):
    """Non-modal top-level child window that temporarily owns the audit page."""

    returnRequested = Signal()

    def __init__(self, owner: QWidget):
        super().__init__(owner, Qt.WindowType.Window)
        self.setObjectName("TrainingSetAuditWindow")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowTitle(self.tr("Training Set Check — NepTrainKit"))
        self.setWindowIcon(owner.windowIcon())
        self.setMinimumSize(840, 560)
        self._owner = owner
        self._content: QWidget | None = None
        self._positioned = False
        self._allow_close = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._layout = layout

    @property
    def content(self) -> QWidget | None:
        return self._content

    @property
    def is_detached(self) -> bool:
        return self._content is not None

    def attach(self, widget: QWidget) -> None:
        if self._content is widget and widget.parentWidget() is self:
            return
        self._content = widget
        widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        widget.setParent(self)
        self._layout.addWidget(widget)
        widget.show()

    def take(self) -> QWidget | None:
        widget = self._content
        if widget is None:
            return None
        self._layout.removeWidget(widget)
        widget.setParent(None)
        self._content = None
        return widget

    def show_owned(self) -> None:
        """Show on a useful screen, then bring the owned window to the front."""
        if not self._positioned:
            self._positioned = self._restore_or_choose_geometry()
        self.show()
        self.raise_()
        self.activateWindow()

    def remember_geometry(self) -> None:
        if not self._positioned:
            return
        encoded = bytes(self.saveGeometry().toBase64()).decode("ascii")
        try:
            Config.set(_CONFIG_SECTION, _GEOMETRY_OPTION, encoded)
        except Exception:
            # Window placement persistence must never block docking or shutdown.
            pass

    def shutdown(self) -> None:
        """Allow the owned window to close as part of application shutdown."""
        if self.is_detached:
            self.remember_geometry()
        self._allow_close = True
        self.close()

    def _restore_or_choose_geometry(self) -> bool:
        raw = Config.get(_CONFIG_SECTION, _GEOMETRY_OPTION)
        if isinstance(raw, str) and raw:
            geometry = QByteArray.fromBase64(raw.encode("ascii"))
            if not geometry.isEmpty() and self.restoreGeometry(geometry):
                frame = self.frameGeometry()
                if any(
                    frame.intersects(screen.availableGeometry())
                    for screen in QGuiApplication.screens()
                ):
                    return True

        screens = QGuiApplication.screens()
        owner_screen = self._owner.screen()
        target = next(
            (screen for screen in screens if screen is not owner_screen),
            owner_screen,
        )
        available = target.availableGeometry()
        width = min(max(960, self._owner.width()), int(available.width() * 0.92))
        height = min(max(640, self._owner.height()), int(available.height() * 0.90))
        self.resize(width, height)
        self.move(
            available.x() + (available.width() - width) // 2,
            available.y() + (available.height() - height) // 2,
        )
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        """Closing the owned window docks its content instead of destroying it."""
        if self._allow_close:
            event.accept()
            return
        event.ignore()
        self.returnRequested.emit()


__all__ = ["TrainingSetAuditHost", "TrainingSetAuditWindow"]
