from __future__ import annotations

import pytest

from NepTrainKit import palette
from NepTrainKit.config import Config
from NepTrainKit.core import types

_OPTIONS = ("palette_version", "selected_color", "line_color", "reject_color", "current_color")


@pytest.fixture
def restore_plot_config():
    previous = {option: Config.get("plot", option) for option in _OPTIONS}
    yield
    for option, value in previous.items():
        if value is None:
            Config.delete("plot", option)
        else:
            Config.set("plot", option, value)


def test_migration_moves_untouched_legacy_defaults(restore_plot_config):
    Config.delete("plot", "palette_version")
    Config.set("plot", "selected_color", "#ff0000")
    Config.set("plot", "line_color", "#FF0000")

    types.migrate_plot_palette()

    assert Config.get("plot", "selected_color") == palette.PLOT_SELECTED
    assert Config.get("plot", "line_color") == palette.PLOT_LINE
    assert Config.get("plot", "palette_version") == types._PLOT_PALETTE_VERSION


def test_migration_keeps_user_picked_colours(restore_plot_config):
    Config.delete("plot", "palette_version")
    Config.set("plot", "reject_color", "#123456")

    types.migrate_plot_palette()

    assert Config.get("plot", "reject_color") == "#123456"


def test_migration_runs_only_once(restore_plot_config):
    Config.set("plot", "palette_version", types._PLOT_PALETTE_VERSION)
    Config.set("plot", "selected_color", "#FF0000")

    types.migrate_plot_palette()

    assert Config.get("plot", "selected_color") == "#FF0000"


def test_migration_moves_the_previous_dark_current_ring(restore_plot_config):
    Config.set("plot", "palette_version", "2")
    Config.set("plot", "current_color", "#1B1F23")

    types.migrate_plot_palette()

    assert Config.get("plot", "current_color") == palette.PLOT_CURRENT
