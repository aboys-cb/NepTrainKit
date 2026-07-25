from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_nep_adapters_runtime_dependency():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = data["project"]["dependencies"]

    assert "nep-adapters>=1.0,<2" in dependencies
    assert "packaging>=24.0" in dependencies


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


def test_runtime_delivery_ci_covers_pip_real_pypi_dialog_and_nuitka():
    workflow = (
        ROOT / ".github" / "workflows" / "pytest.yml"
    ).read_text(encoding="utf-8")

    assert "runtime_package_e2e.py" in workflow
    assert "runtime_package_requests_ui_e2e.py" in workflow
    assert "requests-runtime-update-dialog.png" in workflow
    assert "python -m nuitka" in workflow
    assert "Get-ChildItem dist/*.whl" in workflow


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
