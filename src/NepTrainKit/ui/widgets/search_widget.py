"""Search widgets that combine config-type operations with completers."""

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import QCompleter
from qfluentwidgets import SearchLineEdit, ToolTipFilter, ToolTipPosition
from qfluentwidgets.components.widgets.line_edit import CompleterMenu, LineEditButton

from .completer import CompleterModel, JoinDelegate
from NepTrainKit.core.types import SearchType


class ConfigTypeSearchLineEdit(SearchLineEdit):
    """Search box that supports marking structures by configuration type."""

    searchSignal = Signal(str, SearchType)
    checkSignal = Signal(str, SearchType)
    uncheckSignal = Signal(str, SearchType)
    typeChangeSignal = Signal(str)

    def __init__(self, parent):
        """Initialize line-edit controls and completer support.

        Parameters
        ----------
        parent : QWidget
            Parent widget responsible for ownership.
        """
        super().__init__(parent)
        self.init()
        self.search_type: SearchType = SearchType.TAG

    def init(self):
        """Configure buttons, tooltips, and completer delegates."""
        self.searchButton.setToolTip("Searching for structures based on Config_type")
        self.searchButton.installEventFilter(ToolTipFilter(self.searchButton, 300, ToolTipPosition.TOP))

        self.checkButton = LineEditButton(":/images/src/images/check.svg", self)
        self.checkButton.installEventFilter(ToolTipFilter(self.checkButton, 300, ToolTipPosition.TOP))

        self.checkButton.setToolTip("Mark structure according to Config_type")
        self.uncheckButton = LineEditButton(":/images/src/images/uncheck.svg", self)
        self.uncheckButton.setToolTip("Unmark structure according to Config_type")
        self.uncheckButton.installEventFilter(ToolTipFilter(self.uncheckButton, 300, ToolTipPosition.TOP))

        self.searchButton.setIconSize(QSize(16, 16))
        self.checkButton.setIconSize(QSize(16, 16))
        self.uncheckButton.setIconSize(QSize(16, 16))

        self.hBoxLayout.addWidget(self.checkButton, 0, Qt.AlignmentFlag.AlignRight)
        self.hBoxLayout.addWidget(self.uncheckButton, 0, Qt.AlignmentFlag.AlignRight)

        self.checkButton.clicked.connect(self._checked)
        self.uncheckButton.clicked.connect(self._unchecked)

        self.setObjectName("search_lineEdit")
        self.set_search_type(SearchType.TAG)
        stands = []
        self.completer_model = CompleterModel(stands)

        completer = QCompleter(self.completer_model, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.setCompleter(completer)
        _completerMenu = CompleterMenu(self)
        self.setCompleterMenu(_completerMenu)
        self._delegate = JoinDelegate(self, {})
        _completerMenu.view.setItemDelegate(self._delegate)
        _completerMenu.view.setMaxVisibleItems(10)

    def search(self):
        """Emit the search signal when text is available."""
        text = self.text().strip()
        if text:
            self.searchSignal.emit(text, self.search_type)
        else:
            self.clearSignal.emit()

    def set_search_type(self, search_type: SearchType):
        """Update the active search type and refresh placeholders.

        Parameters
        ----------
        search_type : SearchType
            Category applied to subsequent search requests.
        """
        self.search_type = search_type
        self.setPlaceholderText(f"Mark structure according to {search_type}")
        self.typeChangeSignal.emit(search_type)

    def _checked(self):
        """Emit the check signal using the current text."""
        self.checkSignal.emit(self.text(), self.search_type)

    def _unchecked(self):
        """Emit the uncheck signal using the current text."""
        self.uncheckSignal.emit(self.text(), self.search_type)

    def mousePressEvent(self, event):
        """Open the completer popup while maintaining default behavior."""
        self._completer.setCompletionPrefix(self.text())
        self._completerMenu.setCompletion(self._completer.completionModel())
        self._completerMenu.popup()
        super().mousePressEvent(event)

    def setCompleterKeyWord(self, new_words):
        """Update the completer data source.

        Parameters
        ----------
        new_words : list[str] or dict[str, int]
            Word list or mapping used to populate the completer.
        """
        if isinstance(new_words, list):
            new_words = self.completer_model.parser_list(new_words)
        self._delegate.data = new_words
        self.completer_model.set_data(new_words)

