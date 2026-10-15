"""Shared colour tokens for NepTrainKit widgets and plots.

Keep hard-coded colours out of widgets: reference these names instead so the
application keeps one accent, one set of neutrals, and one plot palette.
``src/qss/theme.qss`` cannot import Python, so it mirrors the UI tokens below.
"""

from __future__ import annotations

# --- UI -------------------------------------------------------------------
# qfluentwidgets' default theme colour; buttons, switches and links use it.
ACCENT = "#009FAA"
ACCENT_SOFT = "#E0F4F5"

TEXT_PRIMARY = "#1B1F23"
# Captions, helper text and collapsed summaries (4.9:1 on white).
TEXT_SECONDARY = "#66737B"
BORDER = "#D9E1E3"
SURFACE = "#FFFFFF"

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
