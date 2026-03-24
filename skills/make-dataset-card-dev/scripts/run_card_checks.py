#!/usr/bin/env python
"""Run focused validation checks for Make Dataset card development."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """Return the nearest parent containing pyproject.toml."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root (pyproject.toml not found).")


def run_commands(commands: list[list[str]], cwd: Path) -> int:
    """Run commands sequentially and stop at first failure."""
    for cmd in commands:
        print(f"[run] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            print(f"[fail] exit code {result.returncode}")
            return result.returncode
    print("[ok] all checks passed")
    return 0


def build_commands(full: bool, with_docs: bool) -> list[list[str]]:
    """Build command list based on mode flags."""
    py = sys.executable
    if full:
        return [
            [py, "-m", "pytest", "tests/"],
            [py, "tools/docs/audit_card_docs.py"],
            [py, "-m", "sphinx", "-W", "-b", "html", "docs/source", "docs/build/html"],
        ]

    commands: list[list[str]] = [
        [py, "-m", "pytest", "tests/test_makedata_source_card.py", "tests/test_card.py", "-q"],
        [py, "tools/docs/audit_card_docs.py"],
    ]
    if with_docs:
        commands.append([py, "-m", "sphinx", "-W", "-b", "html", "docs/source", "docs/build/html"])
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run focused tests and doc audit (default mode if no flags provided).",
    )
    parser.add_argument(
        "--with-docs",
        action="store_true",
        help="Include Sphinx docs build in quick mode.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite plus docs audit and docs build.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.full and args.quick:
        print("Choose either --full or --quick, not both.")
        return 2

    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path.parent)
    commands = build_commands(full=args.full, with_docs=args.with_docs)

    if args.dry_run:
        print(f"[info] repo root: {repo_root}")
        for cmd in commands:
            print(f"[dry-run] {' '.join(cmd)}")
        return 0

    return run_commands(commands, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
