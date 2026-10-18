"""Locate the software OpenGL DLL belonging to the active PySide6 install."""

from pathlib import Path


def find_software_opengl(package_dir: Path, binaries_dir: Path) -> Path:
    # Windows wheels put Qt DLLs beside PySide6/__init__.py, even when
    # QLibraryInfo.BinariesPath reports the package's bin subdirectory.
    candidates = (
        package_dir / "opengl32sw.dll",
        binaries_dir / "opengl32sw.dll",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Qt software OpenGL DLL not found; checked: "
        + ", ".join(str(path) for path in candidates)
    )


if __name__ == "__main__":
    import PySide6
    from PySide6.QtCore import QLibraryInfo

    print(find_software_opengl(
        Path(PySide6.__file__).resolve().parent,
        Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.BinariesPath)),
    ))
