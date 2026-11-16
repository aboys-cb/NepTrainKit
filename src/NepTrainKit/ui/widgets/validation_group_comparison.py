"""Selection and marker preferences shared by grouped validation plots."""

from PySide6.QtCore import QObject, Signal

from NepTrainKit.config import Config
from NepTrainKit.ui.widgets.validation_model_picker import GROUP_MARKERS, ValidationModelPicker, default_group_marker


def remember_import_markers(library, assignments):
    """Keep preview shapes when only part of a source is imported."""
    section = f"validation_group_markers:{library}"
    saved = Config.get_section(section)
    for key, symbol, edited in assignments:
        if edited or key not in saved:
            Config.set(section, key, symbol)


class ValidationGroupComparison(QObject):
    currentRequested = Signal(str)
    changed = Signal()
    markerChanged = Signal(str, str)

    def __init__(self, parent):
        super().__init__(parent)
        self.current = None
        self.comparisons = set()
        self.focused = False
        self.candidates = []
        self.markers = {}
        self.dialog = None

    def load_markers(self, library, families):
        """Each family supplies reference IDs in source order, independent of navigation."""
        self.section = f"validation_group_markers:{library}"
        saved = Config.get_section(self.section)
        self.markers = {}
        for family in families:
            for index, key in enumerate(family):
                symbol = saved.get(key)
                self.markers[key] = symbol if symbol in GROUP_MARKERS else default_group_marker(index)

    def set_marker(self, key, symbol):
        if key in self.markers and symbol in GROUP_MARKERS and self.markers[key] != symbol:
            Config.set(self.section, key, symbol)
            self.markers[key] = symbol
            if self.dialog and key in self.dialog.marker_buttons:
                self.dialog.marker_buttons[key].set_marker(symbol)
            self.markerChanged.emit(key, symbol)

    def browse(self, key):
        if key in self.comparisons:
            self.comparisons.remove(key)
            if self.current:
                self.comparisons.add(self.current)
        self.current = key

    def set_candidates(self, current, candidates):
        self.browse(current)
        self.candidates = candidates
        self.comparisons.intersection_update(row[0] for row in candidates)
        if not self.comparisons:
            self.focused = False
        if self.dialog and self.dialog.isVisible():
            if len(candidates) > 1:
                self._sync_dialog()
            else:
                self.dialog.close()

    def choose_role(self, key, role):
        if key == self.current or key not in {row[0] for row in self.candidates}:
            return
        self.focused = False
        if role == "current":
            self.comparisons.add(key)
            self.currentRequested.emit(key)
        elif role in ("compare", "hidden"):
            if role == "compare":
                self.comparisons.add(key)
            else:
                self.comparisons.discard(key)
            self.changed.emit()

    def focus(self, key):
        self.focused = True
        self.currentRequested.emit(key)

    def expand(self):
        self.focused = False
        self.changed.emit()

    def open_picker(self):
        if self.dialog is None:
            self.dialog = ValidationModelPicker(self.parent().window(), groups=True)
            self.dialog.roleRequested.connect(self.choose_role)
            self.dialog.markerRequested.connect(self.set_marker)
        self._sync_dialog()
        self.dialog.show_picker()

    def _sync_dialog(self):
        self.dialog.set_models([(key, name, summary, self.markers.get(key, "o")) for key, name, summary in self.candidates])
        self.dialog.set_roles(self.current, None, self.comparisons, (), running=False)

    def close_picker(self):
        if self.dialog:
            self.dialog.close()
