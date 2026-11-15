"""Live model-role selection in the validation workspace."""

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QBoxLayout, QColorDialog, QDialog, QFrame, QHBoxLayout, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, PrimaryPushButton, SearchLineEdit, StrongBodyLabel

from NepTrainKit import palette
from NepTrainKit.core.validation.service import find_model_files


def model_role(key, current, baseline, comparisons):
    if key == current:
        return "current"
    if key == baseline:
        return "baseline"
    return "compare" if key in comparisons else "hidden"


class ModelColorButton(QPushButton):
    colorChanged = Signal(QColor)

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        self.setAutoDefault(False)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(self.tr("Choose model color"))
        self.setAccessibleName(self.toolTip())
        self.clicked.connect(self._choose)

    def set_color(self, color):
        self.color = QColor(color)
        self.setStyleSheet(f"""
            QPushButton {{ background: {self.color.name()}; margin: 2px; border: 1px solid {palette.BORDER_STRONG}; border-radius: 4px; }}
            QPushButton:hover, QPushButton:focus {{ border-color: {palette.TEXT_PRIMARY}; }}
        """)

    def _choose(self):
        color = QColorDialog.getColor(self.color, self.window(), self.toolTip())
        if color.isValid() and color != self.color:
            self.colorChanged.emit(color)


class ValidationModelPicker(QDialog):
    roleRequested = Signal(int, str)
    colorRequested = Signal(int, QColor)
    filesDropped = Signal(list)

    def __init__(self, parent):
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowCloseButtonHint)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAcceptDrops(True)
        self.setWindowTitle(self.tr("Choose model versions to compare"))
        self.setObjectName("validationModelPicker")
        self.setStyleSheet("#validationModelPicker { background: white; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        hint = CaptionLabel(self.tr("Choose one role per version. You can compare several versions."), self)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.search = SearchLineEdit(self)
        self.search.setPlaceholderText(self.tr("Search project or model name"))
        self.search.setAccessibleName(self.search.placeholderText())
        layout.addWidget(self.search)
        self.scroll = QScrollArea(self)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list_widget = QWidget()
        self.list_widget.setObjectName("versionList")
        self.list_widget.setStyleSheet("#versionList { background: white; }")
        self.rows = QVBoxLayout(self.list_widget)
        self.rows.setContentsMargins(0, 2, 2, 2)
        self.rows.setSpacing(7)
        self.rows.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.empty_label = BodyLabel(self.list_widget)
        self.rows.addWidget(self.empty_label)
        self.scroll.setWidget(self.list_widget)
        layout.addWidget(self.scroll, 1)
        footer = QHBoxLayout()
        tip = CaptionLabel(self.tr("Drop a NEP .txt file here to compare a temporary model."), self)
        tip.setWordWrap(True)
        footer.addWidget(tip, 1)
        self.done_button = PrimaryPushButton(self.tr("Done"), self)
        self.done_button.clicked.connect(self.accept)
        footer.addWidget(self.done_button)
        layout.addLayout(footer)
        self.buttons = {}
        self.color_buttons = {}
        self.row_widgets = []
        self._models = None
        self.search.textChanged.connect(self._filter_models)
        self.roles = {
            "current": (self.tr("Current model"), palette.INFO),
            "baseline": (self.tr("Baseline model"), palette.WARNING_STRONG),
            "compare": (self.tr("Comparison model"), palette.ACCENT_PRESSED),
            "hidden": (self.tr("Hidden"), palette.TEXT_SECONDARY),
        }

    def set_models(self, models):
        if self._models == models:
            return
        self._models = models
        for frame in self.row_widgets:
            self.rows.removeWidget(frame)
            frame.hide()
            frame.deleteLater()
        self.row_widgets.clear()
        self.buttons.clear()
        self.color_buttons.clear()
        for key, name, summary, color in models:
            card = QFrame(self.list_widget)
            card.setObjectName("roleRow")
            card.setStyleSheet(f"#roleRow {{ background: white; border: 1px solid {palette.BORDER}; border-radius: 8px; }}")
            layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, card)
            layout.setContentsMargins(11, 8, 9, 8)
            identity = QHBoxLayout()
            dot = ModelColorButton(card)
            dot.set_color(color)
            dot.setAccessibleName(f"{name} · {dot.toolTip()}")
            dot.colorChanged.connect(lambda value, model=key: self.colorRequested.emit(model, value))
            self.color_buttons[key] = dot
            identity.addWidget(dot)
            text = QVBoxLayout()
            text.setSpacing(1)
            label = StrongBodyLabel(name, card)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            text.addWidget(label)
            caption = CaptionLabel(summary, card)
            caption.setWordWrap(True)
            caption.setStyleSheet(f"color: {palette.TEXT_SECONDARY};")
            text.addWidget(caption)
            identity.addLayout(text, 1)
            layout.addLayout(identity, 1)
            actions = QHBoxLayout()
            actions.setSpacing(4)
            self.buttons[key] = {}
            for role, (title, accent) in self.roles.items():
                button = QPushButton(title, card)
                button.setAccessibleName(f"{name} · {title}")
                button.setCheckable(True)
                button.setAutoExclusive(True)
                button.setAutoDefault(False)
                button.setStyleSheet(f"""
                    QPushButton {{ padding: 4px 7px; color: {palette.TEXT_PRIMARY}; background: white;
                                  border: 1px solid {palette.BORDER_STRONG}; border-radius: 5px; }}
                    QPushButton:hover, QPushButton:focus {{ border-color: {accent}; }}
                    QPushButton:checked {{ color: white; background: {accent}; }}
                    QPushButton:disabled:!checked {{ color: {palette.TEXT_MUTED}; }}
                """)
                button.clicked.connect(lambda _, model=key, choice=role: self.roleRequested.emit(model, choice))
                self.buttons[key][role] = button
                actions.addWidget(button)
            layout.addLayout(actions)
            self.rows.addWidget(card)
            self.row_widgets.append(card)
        self._filter_models()
        self._layout_rows()

    def _filter_models(self):
        needle = self.search.text().strip().casefold()
        any_match = False
        for (_, name, _, _), card in zip(self._models or (), self.row_widgets):
            match = needle in name.casefold()
            card.setVisible(match)
            any_match |= match
        self.empty_label.setText(self.tr("No matching models.") if self._models else self.tr("No compatible models in this suite."))
        self.empty_label.setVisible(not any_match)

    def set_roles(self, current, baseline, comparisons, eligible, *, running):
        for key, buttons in self.buttons.items():
            active = model_role(key, current, baseline, comparisons)
            for role, button in buttons.items():
                button.setChecked(role == active)
                enabled = True
                tip = ""
                if role == "baseline" and key not in eligible:
                    enabled = False
                    tip = self.tr("Set a baseline for a suite using a saved project model.")
                elif running and (role == "baseline" or key == baseline) and role != active:
                    enabled = False
                    tip = self.tr("The baseline can be changed after evaluation finishes.")
                button.setEnabled(enabled)
                button.setToolTip(tip)

    def show_picker(self):
        if not self.isVisible():
            self.search.clear()
            parent = self.parentWidget()
            screen = parent.screen().availableGeometry()
            width = min(820, max(240, parent.width() - 32), screen.width())
            height = min(500, max(240, 136 + len(self.buttons) * (90 if width < 650 else 66)),
                         screen.height() - 40)
            self.resize(width, height)
            center = parent.mapToGlobal(parent.rect().center())
            self.move(QPoint(max(screen.left(), min(center.x() - width // 2, screen.right() - width + 1)),
                             max(screen.top() + 28, min(center.y() - height // 2, screen.bottom() - height + 1))))
            self.show()
            self.search.setFocus()
        self.raise_()
        self.activateWindow()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_rows()

    def _layout_rows(self):
        direction = QBoxLayout.Direction.TopToBottom if self.width() < 650 else QBoxLayout.Direction.LeftToRight
        for frame in self.row_widgets:
            if frame.layout():
                frame.layout().setDirection(direction)

    def dragEnterEvent(self, event):
        if any(url.isLocalFile() and find_model_files(url.toLocalFile()) for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = list(dict.fromkeys(path for url in event.mimeData().urls() if url.isLocalFile()
                                   for path in find_model_files(url.toLocalFile())))
        if paths:
            event.acceptProposedAction()
            self.filesDropped.emit(paths)
