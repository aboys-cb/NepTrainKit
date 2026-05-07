#!/usr/bin/env python
"""Run focused validation checks for Make Dataset card development."""

from __future__ import annotations

import argparse
import os
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


def audit_operation_architecture(repo_root: Path) -> int:
    """Check Make Dataset cards follow the core operation architecture."""
    import ast

    errors: list[str] = []
    card_dir = repo_root / "src" / "NepTrainKit" / "ui" / "views" / "_card"
    core_cards_dir = repo_root / "src" / "NepTrainKit" / "core" / "cards"

    for path in sorted(card_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(base.attr)
            if not any(base in {"MakeDataCard", "FilterDataCard"} for base in bases):
                continue

            methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
            if "create_operation" not in methods:
                errors.append(f"{path.relative_to(repo_root)}:{node.name} missing create_operation()")
            if "run" in methods:
                errors.append(f"{path.relative_to(repo_root)}:{node.name} defines custom run()")

    forbidden_modules = ("PySide6", "qfluentwidgets")
    forbidden_names = {"MessageManager"}
    for path in sorted(core_cards_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".", 1)[0]
                    if root_name in forbidden_modules or alias.name in forbidden_names:
                        errors.append(f"{path.relative_to(repo_root)} imports UI dependency {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root_name = module.split(".", 1)[0]
                if root_name in forbidden_modules:
                    errors.append(f"{path.relative_to(repo_root)} imports UI dependency {module}")
                for alias in node.names:
                    if alias.name in forbidden_names:
                        source = module or "."
                        errors.append(f"{path.relative_to(repo_root)} imports UI dependency {source}.{alias.name}")

    if errors:
        print("[fail] operation architecture audit failed")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("[ok] operation architecture audit passed")
    return 0


def run_commands(commands: list[list[str]], cwd: Path) -> int:
    """Run commands sequentially and stop at first failure."""
    audit_result = audit_operation_architecture(cwd)
    if audit_result != 0:
        return audit_result
    env = os.environ.copy()
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    for cmd in commands:
        print(f"[run] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, env=env)
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
        [py, "-m", "pytest", "tests/test_makedata_source_card.py", "tests/cards", "-q"],
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
        print("[dry-run] operation architecture audit")
        for cmd in commands:
            print(f"[dry-run] {' '.join(cmd)}")
        return 0

    return run_commands(commands, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
