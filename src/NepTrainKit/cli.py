"""Command line interface for running exported Make Dataset workflows.

The graphical application remains the default entry point. Only when the first
argument names the ``workflow`` subcommand does this module take over, so
``NepTrainKit`` and ``nepkit`` keep launching the GUI exactly as before.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .core.cards.errors import CardOperationError
from .core.workflow import WorkflowError, WorkflowRun, load_workflow

__all__ = ["main", "run_cli"]

PROG = "nepkit workflow"
SUBCOMMANDS = frozenset({"workflow"})
OUTPUT_FORMAT = "extxyz"


def main(argv: list[str] | None = None) -> int:
    """Run the workflow CLI and return a process exit code."""
    return run_cli(argv)


def run_cli(argv: list[str] | None = None) -> int:
    """Dispatch ``workflow`` subcommands.

    ``argv`` may either start at the subcommand or repeat the ``workflow``
    token, so both ``nepkit workflow run ...`` and ``run_cli(["run", ...])``
    behave the same.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    while args and args[0] in SUBCOMMANDS:
        args.pop(0)
    parser = _build_parser()
    try:
        namespace = parser.parse_args(args)
    except SystemExit as exc:  # argparse already printed usage/help
        return int(exc.code or 0)

    _configure_logging(getattr(namespace, "verbose", False))

    try:
        if namespace.command == "preview":
            return _preview(namespace)
        if namespace.command == "run":
            return _run(namespace)
    except (WorkflowError, CardOperationError) as exc:
        _fail(str(exc))
        return 2
    except FileNotFoundError as exc:
        # A card may raise this for a missing model file with its own wording and
        # no ``filename``, while a missing input surfaces as a plain errno text.
        detail = str(exc)
        if exc.filename and str(exc.filename) not in detail:
            detail = f"file not found: {exc.filename}" if detail.startswith("[Errno") else (
                f"{detail}: {exc.filename}"
            )
        elif detail.startswith("[Errno"):
            detail = f"file not found: {detail.split(': ', 1)[-1]}"
        _fail(detail or "a required file is missing")
        return 2
    except OSError as exc:
        _fail(f"{exc.strerror or exc}")
        return 2

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Run or inspect a workflow exported from the Make Dataset workbench.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{preview,run}")

    preview = subparsers.add_parser(
        "preview",
        help="show what a workflow would do without running it",
    )
    preview.add_argument("workflow", help="exported workflow JSON file")
    preview.add_argument(
        "-i",
        "--input",
        help="input structure file; omit to see whether it is required",
    )

    run = subparsers.add_parser("run", help="execute a workflow and write structures")
    run.add_argument("workflow", help="exported workflow JSON file")
    run.add_argument("-i", "--input", required=True, help="input structure file")
    run.add_argument("-o", "--output", help="destination file (default: stdout summary only)")
    run.add_argument(
        "--card",
        action="append",
        default=None,
        metavar="CLASS",
        help="write only this card's output; repeat for several cards",
    )
    run.add_argument(
        "--format",
        default=OUTPUT_FORMAT,
        choices=[OUTPUT_FORMAT],
        help=f"output format (default: {OUTPUT_FORMAT})",
    )
    run.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only report errors",
    )

    for subparser in (preview, run):
        subparser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="also print library diagnostics (card timing, backend details)",
        )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Keep stderr usable in pipelines unless diagnostics were requested.

    The package initializes logging from the user's saved preference, which is
    often ``DEBUG`` while working in the GUI. A command line run should not
    inherit that: stdout carries the result summary and stderr should carry
    errors, so scripts can read both without filtering.

    ``NepTrainKit.config`` re-applies the saved level as an import side effect,
    so the first card that needs configuration would turn DEBUG back on. Pinning
    the level keeps the caller's choice in charge.
    """
    from .logging_config import pin_log_level

    pin_log_level("DEBUG" if verbose else "INFO")


def _preview(namespace: argparse.Namespace) -> int:
    run = load_workflow(Path(namespace.workflow))
    stages = run.preview()

    print(f"workflow: {namespace.workflow}")
    if run.plan.software_version:
        print(f"saved by: {run.plan.software_version} (schema {run.plan.schema})")
    print(f"cards: {len(stages)}")
    print()

    width = max((len(stage.card_class) for stage in stages), default=0)
    for stage in stages:
        print(f"  {stage.index:>2}. {stage.card_class:<{width}}  {stage.kind}")

    print()
    if run.requires_input():
        print("input: required")
        if namespace.input:
            structures = _read_structures(namespace.input)
            print(f"  {namespace.input}: {len(structures)} structure(s)")
    else:
        print("input: not required (this workflow generates its own structures)")

    print("no structures were produced; use 'run' to execute")
    return 0


def _run(namespace: argparse.Namespace) -> int:
    run: WorkflowRun = load_workflow(Path(namespace.workflow))
    structures = _read_structures(namespace.input)

    if not structures:
        _fail(f"{namespace.input} contains no structures")
        return 2

    # Cards pull in NepTrainKit.config, which re-applies the saved level.
    _configure_logging(getattr(namespace, "verbose", False))
    result = run.run(structures)

    if namespace.card:
        selected: list[Any] = []
        for card_class in namespace.card:
            selected.extend(result.outputs(card_class))
    else:
        selected = result.all()

    if namespace.output:
        _write_structures(namespace.output, selected)

    if not namespace.quiet:
        chain = " → ".join(
            f"{name} {len(result.outputs(name))}" for name in result.card_names()
        )
        print(f"{len(structures)} in → {len(selected)} out | {chain}")
        if namespace.output:
            print(f"written: {namespace.output}")
    return 0


def _read_structures(path: str) -> list[Any]:
    from ase.io import read

    return list(read(path, index=":"))


def _write_structures(path: str, structures: list[Any]) -> None:
    from ase.io import write

    from .core.magnetism import prepare_magnetic_extxyz_export

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write(
        str(destination),
        [prepare_magnetic_extxyz_export(atoms) for atoms in structures],
        format=OUTPUT_FORMAT,
    )


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
