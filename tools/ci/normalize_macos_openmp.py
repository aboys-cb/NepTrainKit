#!/usr/bin/env python3
"""Give bundled macOS OpenMP runtimes one process-wide dyld identity.

Independent wheels normally get different ``@loader_path`` references from
delocate. If two such wheels use LLVM OpenMP in one process, both copies are
loaded and libomp aborts with OMP Error #15. This repair step keeps each wheel
self-contained while changing the dependency contract to:

* ``libomp.dylib`` ID: ``@rpath/libomp.dylib``
* native extension dependency: ``@rpath/libomp.dylib``
* native extension rpath: its wheel-local directory containing libomp

With the same load command in every cooperating wheel, dyld reuses the first
loaded runtime instead of initializing a second copy.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

OPENMP_ID = "@rpath/libomp.dylib"


def _run(*args: str | Path) -> str:
    result = subprocess.run(
        [str(arg) for arg in args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _dependencies(binary: Path) -> list[str]:
    return [
        line.strip().split(" ", 1)[0]
        for line in _run("otool", "-L", binary).splitlines()[1:]
        if line.strip()
    ]


def _dylib_id(binary: Path) -> str:
    lines = [line.strip() for line in _run("otool", "-D", binary).splitlines()]
    if len(lines) != 2:
        raise RuntimeError(f"{binary}: expected one dylib ID, got {lines[1:]!r}")
    return lines[1]


def _rpaths(binary: Path) -> set[str]:
    lines = _run("otool", "-l", binary).splitlines()
    paths: set[str] = set()
    for index, line in enumerate(lines):
        if line.strip() != "cmd LC_RPATH":
            continue
        for detail in lines[index + 1 : index + 5]:
            stripped = detail.strip()
            if stripped.startswith("path "):
                paths.add(stripped.split()[1])
                break
    return paths


def _loader_rpath(binary: Path, runtime: Path) -> str:
    relative = PurePosixPath(os.path.relpath(runtime.parent, binary.parent))
    if str(relative) == ".":
        return "@loader_path"
    return f"@loader_path/{relative}"


def _openmp_dependencies(binary: Path) -> list[str]:
    return [
        dependency
        for dependency in _dependencies(binary)
        if PurePosixPath(dependency).name == "libomp.dylib"
    ]


def verify_tree(root: Path) -> list[Path]:
    """Verify the shared macOS OpenMP identity in one unpacked wheel."""
    runtimes = sorted(root.rglob("libomp.dylib"))
    if len(runtimes) != 1:
        raise RuntimeError(
            f"{root}: expected exactly one bundled libomp.dylib, found {runtimes!r}"
        )
    runtime = runtimes[0]
    if _dylib_id(runtime) != OPENMP_ID:
        raise RuntimeError(f"{runtime}: dylib ID must be {OPENMP_ID}")

    consumers: list[Path] = []
    for binary in sorted(root.rglob("*.so")):
        dependencies = _openmp_dependencies(binary)
        if not dependencies:
            continue
        if dependencies != [OPENMP_ID]:
            raise RuntimeError(
                f"{binary}: OpenMP dependency must be {OPENMP_ID}, got {dependencies!r}"
            )
        expected_rpath = _loader_rpath(binary, runtime)
        if expected_rpath not in _rpaths(binary):
            raise RuntimeError(f"{binary}: missing rpath {expected_rpath}")
        _run("codesign", "--verify", "--strict", binary)
        consumers.append(binary)

    if not consumers:
        raise RuntimeError(f"{root}: no native extension links to libomp.dylib")
    _run("codesign", "--verify", "--strict", runtime)
    return consumers


def normalize_tree(root: Path) -> list[Path]:
    """Normalize one unpacked wheel and return its modified Mach-O files."""
    runtimes = sorted(root.rglob("libomp.dylib"))
    if len(runtimes) != 1:
        raise RuntimeError(
            f"{root}: expected exactly one bundled libomp.dylib, found {runtimes!r}"
        )
    runtime = runtimes[0]
    _run("install_name_tool", "-id", OPENMP_ID, runtime)

    modified = [runtime]
    consumers: list[tuple[Path, str]] = []
    for binary in sorted(root.rglob("*.so")):
        dependencies = _openmp_dependencies(binary)
        if not dependencies:
            continue
        if len(dependencies) != 1:
            raise RuntimeError(
                f"{binary}: expected one libomp dependency, got {dependencies!r}"
            )
        dependency = dependencies[0]
        if dependency != OPENMP_ID:
            _run("install_name_tool", "-change", dependency, OPENMP_ID, binary)
        expected_rpath = _loader_rpath(binary, runtime)
        if expected_rpath not in _rpaths(binary):
            _run("install_name_tool", "-add_rpath", expected_rpath, binary)
        consumers.append((binary, expected_rpath))
        modified.append(binary)

    if not consumers:
        raise RuntimeError(f"{root}: no native extension links to libomp.dylib")

    for binary in modified:
        _run("codesign", "--force", "--sign", "-", binary)

    verify_tree(root)
    return modified


def _record_bytes(root: Path, record: Path) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if path == record:
            writer.writerow((relative, "", ""))
            continue
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        writer.writerow((relative, f"sha256={digest.decode('ascii')}", len(data)))
    return output.getvalue().encode("utf-8")


def normalize_wheel(wheel: Path) -> None:
    """Normalize a repaired wheel in place and refresh its RECORD."""
    wheel = wheel.resolve()
    with tempfile.TemporaryDirectory(prefix="macos-openmp-wheel-") as tmp_dir:
        root = Path(tmp_dir) / "wheel"
        root.mkdir()
        with zipfile.ZipFile(wheel) as archive:
            original_info = {info.filename: info for info in archive.infolist()}
            archive.extractall(root)

        normalize_tree(root)
        records = sorted(root.glob("*.dist-info/RECORD"))
        if len(records) != 1:
            raise RuntimeError(f"{wheel.name}: expected one wheel RECORD")
        records[0].write_bytes(_record_bytes(root, records[0]))

        replacement = Path(tmp_dir) / wheel.name
        with zipfile.ZipFile(replacement, "w") as archive:
            for path in sorted(item for item in root.rglob("*") if item.is_file()):
                relative = path.relative_to(root).as_posix()
                info = original_info.get(relative)
                if info is None:
                    archive.write(path, relative, compress_type=zipfile.ZIP_DEFLATED)
                else:
                    archive.writestr(info, path.read_bytes())
        with zipfile.ZipFile(replacement) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError(f"{wheel.name}: corrupt member after repair: {corrupt}")
        os.replace(replacement, wheel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    for wheel in args.wheels:
        normalize_wheel(wheel)
        print(f"Normalized macOS OpenMP runtime: {wheel}")


if __name__ == "__main__":
    main()
