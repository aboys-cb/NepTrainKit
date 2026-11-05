"""Console entry point that dispatches CLI subcommands before importing Qt.

``NepTrainKit.main`` imports PySide6 at module scope, so the workflow
subcommand must be recognised before that import happens. Anything that is not
a known subcommand falls through to the graphical application unchanged.
"""

from __future__ import annotations

import sys

__all__ = ["main", "CLI_SUBCOMMANDS"]

#: First arguments handled by :mod:`NepTrainKit.cli` instead of the GUI.
CLI_SUBCOMMANDS = frozenset({"workflow"})


def main() -> None:
    """Run the workflow CLI, or launch the GUI when no subcommand is given."""
    argv = sys.argv[1:]
    if argv and argv[0] in CLI_SUBCOMMANDS:
        from NepTrainKit.cli import run_cli

        raise SystemExit(run_cli(argv))

    from NepTrainKit.main import main as run_gui

    run_gui()


if __name__ == "__main__":
    main()
