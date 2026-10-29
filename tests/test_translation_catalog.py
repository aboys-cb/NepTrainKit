"""Exercise generated translation catalogs with the real Qt compiler."""

import os
from pathlib import Path
import subprocess

import pytest
from PySide6.QtCore import QTranslator

from NepTrainKit import translation_catalog as catalog


def _write_ts(path, translation="你好"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<TS version="2.1" language="zh_CN"><context><name>Example</name>'
        '<message><source>Hello</source>'
        f'<translation>{translation}</translation></message></context></TS>',
        encoding="utf-8",
    )


def _translation(qm):
    translator = QTranslator()
    assert translator.load(str(qm))
    return translator.translate("Example", "Hello")


def test_missing_catalog_is_compiled_and_unchanged_catalog_is_reused(tmp_path, monkeypatch):
    ts = tmp_path / "example.ts"
    _write_ts(ts)
    qm = catalog.compile_catalog(ts)
    assert _translation(qm) == "你好"
    timestamp = qm.stat().st_mtime_ns

    def unexpected_compile(*args, **kwargs):
        pytest.fail("An unchanged catalog must not start the compiler")

    monkeypatch.setattr(catalog.subprocess, "run", unexpected_compile)
    assert catalog.compile_catalog(ts) == qm
    assert qm.stat().st_mtime_ns == timestamp


def test_updated_ts_recompiles_catalog(tmp_path):
    ts = tmp_path / "example.ts"
    _write_ts(ts)
    qm = catalog.compile_catalog(ts)
    _write_ts(ts, "您好")
    # Avoid sleeps and filesystem timestamp resolution assumptions.
    os.utime(qm, ns=(1, 1))
    catalog.compile_catalog(ts)
    assert _translation(qm) == "您好"


def test_failed_compilation_keeps_previous_catalog_and_cleans_temporary_files(tmp_path, monkeypatch):
    ts = tmp_path / "example.ts"
    _write_ts(ts)
    qm = catalog.compile_catalog(ts)
    before = qm.read_bytes()

    def fail(command, **kwargs):
        Path(command[-1]).write_bytes(b"incomplete")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(catalog.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        catalog.compile_catalog(ts, force=True)
    assert qm.read_bytes() == before
    assert not list(tmp_path.glob(".qm-build-*"))


def test_source_checkout_compiles_but_installed_package_does_not(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").touch()
    ts = tmp_path / "src/NepTrainKit/translations/example.ts"
    _write_ts(ts)
    catalog.ensure_source_catalog(ts.with_suffix(".qm"))
    assert _translation(ts.with_suffix(".qm")) == "你好"

    def unexpected_compile(*args, **kwargs):
        pytest.fail("Installed packages must use their bundled catalog")

    monkeypatch.setattr(catalog, "compile_catalog", unexpected_compile)
    catalog.ensure_source_catalog(tmp_path / "site-packages/NepTrainKit/translations/example.qm")
    catalog.ensure_source_catalog(tmp_path / "standalone/translations/example.qm")


def test_install_translator_generates_source_catalog(tmp_path, monkeypatch):
    from NepTrainKit import i18n
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    (tmp_path / "pyproject.toml").touch()
    package = tmp_path / "src/NepTrainKit"
    ts = package / "translations/neptrainkit_zh_CN.ts"
    _write_ts(ts)
    monkeypatch.setattr(i18n, "module_path", package)
    try:
        i18n.install_translator(app, "zh_CN")
        assert app.translate("Example", "Hello") == "你好"
    finally:
        i18n.install_translator(app, "en_US")


def test_build_py_generates_catalog_in_build_output_without_source_qm(tmp_path, monkeypatch):
    import runpy
    from unittest.mock import patch
    from setuptools import Distribution

    root = Path(__file__).resolve().parents[1]
    with patch("setuptools.setup") as setup:
        runpy.run_path(str(root / "setup.py"))
    build_class = setup.call_args.kwargs["cmdclass"]["build_py"]
    package = tmp_path / "src/NepTrainKit"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    ts = package / "translations/example.ts"
    _write_ts(ts)
    monkeypatch.chdir(tmp_path)
    distribution = Distribution({
        "name": "catalog-test",
        "packages": ["NepTrainKit"],
        "package_dir": {"": "src"},
        "package_data": {"NepTrainKit": ["translations/*.ts", "translations/*.qm"]},
    })
    distribution.script_name = "setup.py"
    command = build_class(distribution)
    command.build_lib = str(tmp_path / "build")
    command.ensure_finalized()
    command.run()
    qm = tmp_path / "build/NepTrainKit/translations/example.qm"
    assert _translation(qm) == "你好"
    assert str(qm) in command.get_outputs()
    assert not ts.with_suffix(".qm").exists()
