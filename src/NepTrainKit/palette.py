"""Shared colour tokens for NepTrainKit widgets and plots.

Keep hard-coded colours out of widgets: reference these names instead so the
application keeps one accent, one set of neutrals, and one plot palette.
``src/qss/theme.qss`` cannot import Python, so it mirrors the UI tokens below.
"""

from __future__ import annotations

# --- UI -------------------------------------------------------------------
# qfluentwidgets' default theme colour; buttons, switches and links use it.
ACCENT = "#009FAA"
ACCENT_HOVER = "#008B94"
ACCENT_PRESSED = "#00757D"
ACCENT_SOFT = "#E0F4F5"
ACCENT_BORDER = "#8AD8DE"

# Marks state the user chose to keep, such as a starred model version.
FAVOURITE = "#E03131"

# Layered surfaces: App Canvas -> Card Surface -> Subtle Recessed/Header -> Hover
CANVAS_BG = "#EEF2F6"
SURFACE = "#F9FBFD"
SURFACE_SUBTLE = "#F2F5F8"
SURFACE_HOVER = "#E7EDF3"
SURFACE_ACTIVE = "#E0F2F4"

TEXT_PRIMARY = "#1B1F23"
# Captions, helper text and collapsed summaries (4.9:1 on white).
TEXT_SECONDARY = "#5C6870"
TEXT_MUTED = "#84919A"

BORDER = "#E3E9EF"
BORDER_SUBTLE = "#ECF0F4"
BORDER_STRONG = "#CFD8E0"

# Semantic status colours for badges, audit diagnostics, and validation cards.
SUCCESS = "#2B8A3E"
SUCCESS_SOFT = "#EBFBEE"
SUCCESS_BORDER = "#8CE99A"

WARNING = "#D97706"
WARNING_STRONG = "#946100"
WARNING_SOFT = "#FEF3C7"
WARNING_BORDER = "#FCD34D"

DANGER = "#E03131"
DANGER_SOFT = "#FFF5F5"
DANGER_BORDER = "#FFC9C9"

INFO = "#1971C2"
INFO_SOFT = "#E7F5FF"
INFO_BORDER = "#A5D8FF"

# --- Plots ------------------------------------------------------------------
# Defaults for the ``plot`` config section.  Users can still override every
# value from Settings; see ``migrate_plot_palette`` in ``core.types``.
PLOT_MARKER_EDGE = "#2F6FA8"
PLOT_MARKER_FACE = "#FFFFFF"
PLOT_LINE = "#9AA5AD"
PLOT_SELECTED = "#F76707"
# Distinct from base blue, selected orange, reject red and search green.
PLOT_CURRENT = "#AE3EC9"
PLOT_SHOW = "#2F9E44"
PLOT_REJECT = "#E03131"
PLOT_TRAINING_OVERLAY = "#A0A0A0"
PLOT_TRAINING_OVERLAY_EDGE = "#505050"
PLOT_LOADED_OVERLAY = "#1E78D7"
PLOT_LOADED_OVERLAY_EDGE = "#1450B4"
PLOT_AXIS = "#5F6B73"
PLOT_TITLE = "#1B1F23"
