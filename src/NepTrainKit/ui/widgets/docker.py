"""Widgets for arranging data processing cards in a draggable workflow."""

from PySide6.QtWidgets import QWidget, QApplication, QScrollArea

from .layout import FlowLayout
from .card_widget import MakeDataCard, MakeDataCardWidget


class MakeWorkflowArea(QScrollArea):
    """Scrollable area that hosts draggable data-processing cards."""

    def __init__(self, parent=None):
        """Build the workflow area and prepare the container widgets.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget responsible for ownership.
        """
        super().__init__(parent)
        self._parent = parent
        self.setObjectName("MakeWorkflowArea")
        self.setWidgetResizable(True)

        self.setAcceptDrops(True)

        self.init_ui()

    @property
    def cards(self) -> list[MakeDataCard]:
        """Return the currently displayed cards."""
        return [item.widget() for item in self.flow_layout.itemList]  # type: ignore[attr-defined]

    def dragEnterEvent(self, event):
        """Accept drag events that originate from workflow cards.

        Parameters
        ----------
        event : QDragEnterEvent
            Drag event emitted by Qt during the enter phase.
        """
        if isinstance(event.source(), MakeDataCardWidget):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Reorder cards when a workflow card is dropped.

        Parameters
        ----------
        event : QDropEvent
            Drop event carrying the source widget and position.
        """
        if isinstance(event.source(), MakeDataCardWidget):
            dragged_widget = event.source()
            drag_start_index = self.flow_layout.findWidgetAt(dragged_widget)[0]
            drop_pos = event.position().toPoint()
            drop_index, _ = self.flow_layout.findItemAt(drop_pos)

            if drop_index == -1:
                drop_index = self.flow_layout.count() - 1

            drop_index = min(max(0, drop_index), self.flow_layout.count())

            if drag_start_index == -1:
                self.flow_layout.insertWidget(drop_index, dragged_widget)
            else:
                if drag_start_index != drop_index:
                    self.flow_layout.moveItem(drag_start_index, drop_index)

            self.flow_layout.update()

            event.acceptProposedAction()

    def init_ui(self):
        """Create the container and layout for hosting cards."""
        self.container = QWidget(self)
        self.flow_layout = FlowLayout(self.container)
        self.container.setLayout(self.flow_layout)
        self.setWidget(self.container)

    def add_card(self, card):
        """Append a card widget to the workflow.

        Parameters
        ----------
        card : MakeDataCardWidget
            Widget to add to the flow layout.
        """
        self.flow_layout.addWidget(card)

    def clear_cards(self):
        """Close and remove all cards from the workflow."""
        for item in self.cards:
            item.close()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    window = MakeWorkflowArea()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec())

