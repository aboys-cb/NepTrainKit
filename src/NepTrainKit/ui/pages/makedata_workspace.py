"""In-page workflow drafts; each editor retains its input, results and selection."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget
from qfluentwidgets import TabBar, TabCloseButtonDisplayMode

from NepTrainKit.core import MessageManager

from .makedata import MakeDataWidget


class MakeDataWorkspace(QWidget):
    finalOutputRequestedSignal = Signal(list)

    def __init__(self, parent=None):
        """Keep the primary editor and validation drafts in separate tabs."""
        super().__init__(parent)
        self.setObjectName("MakeDataWorkspace")
        self._menu_owner = parent
        self._next_key = 0
        self.tabs = TabBar(self)
        self.tabs.setScrollable(True)
        self.tabs.addButton.hide()
        self.tabs.currentChanged.connect(self._select)
        self.tabs.tabCloseRequested.connect(self._close)
        self.stack = QStackedWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.tabs)
        layout.addWidget(self.stack, 1)
        self.primary_editor = self._add_editor(self.tr("Original workflow"))
        self.tabs.items[0].setCloseButtonDisplayMode(TabCloseButtonDisplayMode.NEVER)
        self.tabs.hide()

    def _add_editor(self, title):
        editor = MakeDataWidget(self._menu_owner)
        editor.finalOutputRequestedSignal.connect(self.finalOutputRequestedSignal)
        self.stack.addWidget(editor)
        self._next_key += 1
        self.tabs.addTab(str(self._next_key), title)
        self.tabs.items[-1].setToolTip(title)
        self.tabs.setCurrentIndex(self.stack.count() - 1)
        self.stack.setCurrentWidget(editor)
        return editor

    def _select(self, index):
        if 0 <= index < self.stack.count():
            self.stack.setCurrentIndex(index)

    def open_file(self):
        self.stack.currentWidget().open_file()

    def export_file(self):
        self.stack.currentWidget().export_file()

    def open_sampling(self, payload):
        name = payload["name"].rsplit(" / ", 1)[-1]
        editor = self._add_editor(self.tr("EOS sampling · {name}").format(name=name))
        try:
            editor.load_sampling_draft(payload)
        except Exception:
            index = self.stack.indexOf(editor)
            self.stack.removeWidget(editor)
            self.tabs.removeTab(index)
            editor.deleteLater()
            raise
        self.tabs.show()
        return editor

    def _close(self, index):
        editor = self.stack.widget(index)
        if editor is None or editor is self.primary_editor:
            return
        if editor._has_running_card():
            MessageManager.send_info_message(self.tr("Stop the running workflow before closing this draft."))
            return
        if not editor._confirm_discard_workflow_changes():
            return
        editor.hide()
        editor.workspace_card_widget.clear_cards()
        if editor._sampling_panel is not None:
            editor._sampling_panel.plot.close()
            editor._sampling_panel.plot.deleteLater()
        self.stack.removeWidget(editor)
        self.tabs.removeTab(index)
        editor.deleteLater()
        self.tabs.setVisible(self.stack.count() > 1)
        self._select(self.tabs.currentIndex())
