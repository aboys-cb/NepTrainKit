"""Compile Qt catalogs without importing the application or creating a GUI."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def compile_catalog(ts: Path, qm: Path | None = None, *, force: bool = False) -> Path:
    """Generate a missing/outdated catalog, replacing it only after success."""
    ts = Path(ts)
    qm = Path(qm) if qm is not None else ts.with_suffix(".qm")
    if not force and qm.is_file() and qm.stat().st_mtime_ns >= ts.stat().st_mtime_ns:
        return qm
    qm.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qm-build-", dir=qm.parent) as directory:
        output = Path(directory) / qm.name
        subprocess.run(
            [sys.executable, "-c", "from PySide6.scripts.pyside_tool import lrelease; lrelease()",
             str(ts), "-qm", str(output)],
            check=True,
            capture_output=True,
            text=True,
        )
        os.replace(output, qm)
    return qm


def ensure_source_catalog(qm: Path) -> None:
    """Refresh only a source checkout; installed/frozen packages ship their QM."""
    package = qm.parent.parent
    if package.parent.name == "src" and (package.parent.parent / "pyproject.toml").is_file():
        compile_catalog(qm.with_suffix(".ts"), qm)
