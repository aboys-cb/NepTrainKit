"""Composite card that sequences multiple data cards."""

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout
from shiboken6 import isValid

from NepTrainKit.ui.dialogs import call_path_dialog
from NepTrainKit.ui.threads import LoadingThread
from NepTrainKit.core import CardManager
from NepTrainKit.ui.widgets import MakeDataCardWidget, MakeDataCard, FilterDataCard


@CardManager.register_card
class CardGroup(MakeDataCardWidget):
    """Group container that executes child cards in sequence and aggregates outputs.
    
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget that owns the group card.
    
    Attributes
    ----------
    runFinishedSignal : Signal
        Emitted with the group index when execution finishes.
    filter_card : FilterDataCard or None
        Optional post-processing card applied to the aggregated dataset.
    """

    separator=True
    card_name= "Card Group"
    menu_icon=r":/images/src/images/group.svg"
    runFinishedSignal=Signal(int)
    def __init__(self, parent=None):
        """Initialise layouts, drag-and-drop targets, and default execution state.
        """
        super().__init__(parent)
        self.setTitle("Card Group")
        self.setAcceptDrops(True)
        self.index=0
        self.group_widget = QWidget(self)
        # self.setStyleSheet("CardGroup{boder: 2px solid #C0C0C0;}")
        self.viewLayout.addWidget(self.group_widget)
        self.group_layout = QVBoxLayout(self.group_widget)
        self.exportSignal.connect(self.export_data)
        self.windowStateChangedSignal.connect(self.show_card_setting)
        self.filter_widget = QWidget(self)
        self.filter_layout = QVBoxLayout(self.filter_widget)
        self.vBoxLayout.addWidget(self.filter_widget)
        self.run_card_num:int
        self.filter_card=None
        self.dataset:Any
        self.result_dataset=[]
        self.cards_to_run = []
        self.current_index = 0
        self.resize(400, 200)

    def set_filter_card(self,card):
        """Attach a filter card that refines results after the grouped cards run.
        
        Parameters
        ----------
        card : QWidget
            Filter card widget to embed beneath the grouped cards.
        """
        self.filter_card=card
        self.filter_layout.addWidget(card)

    def state_changed(self, state):
        """Update the enabled state of child cards to match the group checkbox.
        
        Parameters
        ----------
        state : bool
            Toggle state propagated from the group header.
        """
        super().state_changed(state)
        for card in self.card_list:
            card.state_checkbox.setChecked(state)

    @property
    def card_list(self)->list["MakeDataCard"]:
        """List the child card widgets currently managed by the group.
        
        Returns
        -------
        list of MakeDataCard
            Ordered collection of child cards.
        """
        return [self.group_layout.itemAt(i).widget() for i in range(self.group_layout.count()) ]  # pyright:ignore
    def show_card_setting(self):
        """Propagate window state changes to every child card.
        """
        for card in self.card_list:
            card.window_state = self.window_state
            card.windowStateChangedSignal.emit()
    def set_dataset(self,dataset):
        """Store the shared dataset reference and clear accumulated results.
        
        Parameters
        ----------
        dataset : Any
            Dataset that will be passed to each child card.
        """
        self.dataset =dataset
        self.result_dataset=[]

    def add_card(self, card):
        """Insert a card widget into the group layout.
        
        Parameters
        ----------
        card : QWidget
            Card widget to append.
        """
        self.group_layout.addWidget(card)

    def remove_card(self, card):
        """Remove a card widget from the group layout.
        
        Parameters
        ----------
        card : QWidget
            Card widget to detach.
        """
        self.group_layout.removeWidget(card)

    def clear_cards(self):
        """Remove every child card from the layout.
        """
        for card in self.card_list:
            self.group_layout.removeWidget(card)

    def closeEvent(self, event):
        """Close nested cards before destroying the group widget.
        
        Parameters
        ----------
        event : QCloseEvent
            Close event propagated from Qt.
        """
        for card in self.card_list:
            card.close()
        self.deleteLater()
        super().closeEvent(event)

    def dragEnterEvent(self, event):
        """Accept drag events from compatible card widgets.
        
        Parameters
        ----------
        event : QDragEnterEvent
            Drag event describing the incoming payload.
        """
        widget = event.source()

        if widget == self:
            return
        if isinstance(widget, (MakeDataCard,CardGroup)):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle dropped cards by inserting them or assigning the filter card.
        
        Parameters
        ----------
        event : QDropEvent
            Drop event containing the dragged widget.
        """
        widget = event.source()
        if widget == self:
            return
        if isinstance(widget, FilterDataCard):
            self.set_filter_card(widget)

        elif isinstance(widget, (MakeDataCard,CardGroup)):
            self.add_card(widget)
        event.acceptProposedAction()

    def on_card_finished(self, index):
        """Collect results from the finished card and start the next queued card.
        
        Parameters
        ----------
        index : int
            Index of the card that finished processing.
        """
        card = self.cards_to_run[self.current_index]
        card.runFinishedSignal.disconnect(self.on_card_finished)
        self.result_dataset.extend(card.result_dataset)
        self.current_index += 1
        self.run_card_num -= 1

        if self.current_index < len(self.cards_to_run):
            self.start_next_card()
        else:
            self.runFinishedSignal.emit(self.index)
            if self.filter_card and isValid(self.filter_card) and self.filter_card.check_state:
                self.filter_card.set_dataset(self.result_dataset)
                self.filter_card.run()

    def stop(self):
        """Stop execution across child cards and the optional filter card.
        """
        for card in self.card_list:
            try:
                card.runFinishedSignal.disconnect(self.on_card_finished)
            except Exception:
                pass
            card.stop()
        if self.filter_card:
            self.filter_card.stop()

    def run(self):
        """Execute selected child cards sequentially and emit when finished.
        """
