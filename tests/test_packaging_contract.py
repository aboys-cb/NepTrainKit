from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_nep_adapters_runtime_dependency():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]

    assert data["project"]["requires-python"] == ">=3.10,<3.14"
    assert "Programming Language :: Python :: 3.13" in data["project"]["classifiers"]
    assert "nep-adapters>=1.0,<2" in dependencies
    assert "packaging>=24.0" in dependencies


def test_wheel_metadata_declares_license_and_third_party_notices():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["license"] == "GPL-3.0-or-later"
    assert data["project"]["license-files"] == [
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    ]
    assert "setuptools>=77.0.0" in data["build-system"]["requires"]
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()


def test_development_requirements_match_nep_adapters_contract():
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "nep-adapters>=1.0,<2" in requirements
    assert "packaging>=24.0" in requirements


def test_nuitka_build_keeps_nep_adapters_outside_the_executable():
    workflow = (
        ROOT / ".github" / "workflows" / "Build-with-Nuitka.yml"
    ).read_text(encoding="utf-8")

    assert "include-package: NepTrainKit,nep_adapters" not in workflow
    assert "runtime/nep-adapters/versions" in workflow
    assert "--neptrainkit-runtime-health-check" in workflow


def test_nuitka_build_copies_and_verifies_chinese_translation_catalog():
    workflow = (
        ROOT / ".github" / "workflows" / "Build-with-Nuitka.yml"
    ).read_text(encoding="utf-8")

    assert "src/NepTrainKit/translations" in workflow
    assert '$dist/translations/neptrainkit_zh_CN.qm' in workflow
    assert "Chinese translation catalog is missing" in workflow


def test_nuitka_build_validates_flat_standalone_layout():
    workflow = (
        ROOT / ".github" / "workflows" / "Build-with-Nuitka.yml"
    ).read_text(encoding="utf-8")

    assert (
        'Test-Path -LiteralPath "$dist/NepTrainKit.exe" -PathType Leaf'
        in workflow
    )
    assert 'Remove-Item -Path "$dist/NepTrainKit"' not in workflow


def test_windows_standalone_name_matches_verified_x86_64_architecture():
    workflow = (
        ROOT / ".github" / "workflows" / "Build-with-Nuitka.yml"
    ).read_text(encoding="utf-8")

    assert "struct.calcsize('P') * 8 == 64" in workflow
    assert "NepTrainKit.windows-x86_64.zip" in workflow
    assert "NepTrainKit.win32.zip" not in workflow


def test_nuitka_runtime_resources_are_resolved_from_binary_directory():
    package_init = (
        ROOT / "src" / "NepTrainKit" / "__init__.py"
    ).read_text(encoding="utf-8")
    i18n_source = (
        ROOT / "src" / "NepTrainKit" / "i18n.py"
    ).read_text(encoding="utf-8")

    assert "_nuitka_binary_dir = Path(str(__nuitka_binary_dir)).resolve()" in package_init
    assert "module_path = _nuitka_binary_dir" in package_init
    assert 'module_path / "translations"' in i18n_source


def test_macos_wheels_use_explicit_openmp_repair_and_portability_gate():
    setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
    workflow = (
        ROOT / ".github" / "workflows" / "python-publish.yml"
    ).read_text(encoding="utf-8")

    assert 'default_omp_mode = "0" if sys.platform == "darwin" else "auto"' in setup_source
    assert 'echo "NEPKIT_OPENMP=1"' in workflow
    assert "CIBW_REPAIR_WHEEL_COMMAND_MACOS" in workflow
    assert "delocate-wheel --require-archs" in workflow
    assert "verify_macos_wheel.py wheelhouse/*.whl" in workflow


def test_test_and_wheel_matrices_cover_documented_python_versions():
    test_workflow = (
        ROOT / ".github" / "workflows" / "pytest.yml"
    ).read_text(encoding="utf-8")
    publish_workflow = (
        ROOT / ".github" / "workflows" / "python-publish.yml"
    ).read_text(encoding="utf-8")
    quickstart = (ROOT / "docs" / "source" / "quickstart.md").read_text(
        encoding="utf-8"
    )

    for version in ("3.10", "3.11", "3.12", "3.13"):
        assert f'python: "{version}"' in test_workflow
    assert "Python 版本用 3.10 到 3.13" in quickstart
    for tag in ("cp310", "cp311", "cp312", "cp313"):
        assert tag in publish_workflow


def test_test_ci_installs_qt_runtime_and_uses_repository_import_path():
    workflow = (
        ROOT / ".github" / "workflows" / "pytest.yml"
    ).read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "sudo apt-get install -y libegl1" in workflow
    assert "python -m pytest tests/" in workflow
    assert any(
        requirement.startswith("scikit-learn")
        for requirement in project["project"]["optional-dependencies"]["test"]
    )
    assert all(
        not requirement.startswith("scikit-learn")
        for requirement in project["project"]["dependencies"]
    )


def test_runtime_delivery_ci_covers_pip_real_pypi_dialog_and_nuitka():
    workflow = (
        ROOT / ".github" / "workflows" / "pytest.yml"
    ).read_text(encoding="utf-8")
    probe = (
        ROOT / "tools" / "ci" / "runtime_package_requests_ui_e2e.py"
    ).read_text(encoding="utf-8")

    assert "runtime_package_e2e.py" in workflow
    assert "runtime_package_requests_ui_e2e.py" in workflow
    assert "requests-runtime-update-dialog.png" in workflow
    assert "python -m nuitka" in workflow
    assert "Get-ChildItem dist/*.whl" in workflow
    assert "json.dumps(result, ensure_ascii=True" in probe


def test_versioned_dataset_catalog_is_packaged():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]

    assert "catalogs/*.json" in package_data["NepTrainKit.core.dataset"]
    assert (
        ROOT
        / "src"
        / "NepTrainKit"
        / "core"
        / "dataset"
        / "catalogs"
        / "nep_data.v1.json"
    ).is_file()
