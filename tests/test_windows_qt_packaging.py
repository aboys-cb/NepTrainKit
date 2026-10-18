import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "find_qt_software_opengl",
    Path(__file__).resolve().parents[1] / "tools/ci/find_qt_software_opengl.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
find_software_opengl = module.find_software_opengl


def test_windows_wheel_dll_is_found_despite_missing_bin_directory(tmp_path):
    package = tmp_path / "site packages" / "PySide6"
    package.mkdir(parents=True)
    dll = package / "opengl32sw.dll"
    dll.write_bytes(b"wheel DLL")
    assert find_software_opengl(package, package / "bin") == dll.resolve()


def test_qt_binary_directory_remains_supported(tmp_path):
    package = tmp_path / "PySide6"
    binaries = package / "bin"
    binaries.mkdir(parents=True)
    dll = binaries / "opengl32sw.dll"
    dll.write_bytes(b"Qt DLL")
    assert find_software_opengl(package, binaries) == dll.resolve()


def test_missing_dll_reports_all_searched_paths(tmp_path):
    package = tmp_path / "PySide6"
    with pytest.raises(FileNotFoundError) as error:
        find_software_opengl(package, package / "bin")
    assert str(package / "opengl32sw.dll") in str(error.value)
    assert str(package / "bin" / "opengl32sw.dll") in str(error.value)
