#!/usr/bin/env python
"""Reject macOS wheels that retain local package-manager library paths."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import zipfile


FORBIDDEN_PREFIXES = ("/opt/homebrew/", "/usr/local/opt/")


def verify_wheel(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="neptrainkit-wheel-check-") as tmp_dir:
        root = Path(tmp_dir)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)

        extensions = sorted((root / "NepTrainKit" / "_native").glob("*.so"))
        if not extensions:
            raise RuntimeError(f"{wheel.name} contains no macOS native extensions")

        failures: list[str] = []
        for extension in extensions:
            result = subprocess.run(
                ["otool", "-L", str(extension)],
                check=True,
                capture_output=True,
                text=True,
            )
            for dependency in result.stdout.splitlines()[1:]:
                dependency_path = dependency.strip().split(" ", 1)[0]
                if dependency_path.startswith(FORBIDDEN_PREFIXES):
                    failures.append(f"{extension.name}: {dependency_path}")

        if failures:
            joined = "\n".join(failures)
            raise RuntimeError(
                f"{wheel.name} retains non-portable package-manager paths:\n{joined}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    for wheel in args.wheels:
        verify_wheel(wheel)
        print(f"PASS: {wheel}")


if __name__ == "__main__":
    main()
