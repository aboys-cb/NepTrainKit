from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from NepTrainKit.runtime_package import (
    NEP_ADAPTERS_SPEC,
    RuntimePackageError,
    RuntimePackageHealthError,
    RuntimePackageSpec,
    activate_runtime_package,
    check_runtime_package_update,
    install_runtime_package_update,
    rollback_runtime_package,
    seed_runtime_package,
)

PROBE_COMMAND = [
    sys.executable,
    "-c",
    (
        "import sys; "
        "from NepTrainKit.runtime_package import run_runtime_health_command; "
        "result = run_runtime_health_command(); "
        "raise SystemExit(result if result is not None else 99)"
    ),
]


def test_nep_adapters_runtime_constraint_allows_future_major_versions() -> None:
    constraint = SpecifierSet(NEP_ADAPTERS_SPEC.version_constraint)

    assert Version("1.0.0") in constraint
    assert Version("2.0.0") in constraint
    assert Version("99.0.0") in constraint


class _Response:
    def __init__(self, *, payload: Any = None, content: bytes = b""):
        self._payload = payload
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self._content


def _wheel_bytes(version: str, *, broken: bool = False) -> tuple[str, bytes, str]:
    filename = f"runtime_probe_pkg-{version}-py3-none-any.whl"
    module = "raise RuntimeError('broken candidate')\n" if broken else f"__version__ = {version!r}\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("runtime_probe_pkg/__init__.py", module)
        dist_info = f"runtime_probe_pkg-{version}.dist-info"
        archive.writestr(
            f"{dist_info}/METADATA",
            (f"Metadata-Version: 2.1\nName: runtime-probe-pkg\nVersion: {version}\n"),
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            ("Wheel-Version: 1.0\nGenerator: NepTrainKit test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    content = buffer.getvalue()
    return filename, content, hashlib.sha256(content).hexdigest()


def _index_payload(
    wheels: list[tuple[str, bytes, str]],
) -> dict[str, Any]:
    releases: dict[str, list[dict[str, Any]]] = {}
    for filename, _content, digest in wheels:
        version = filename.split("-")[1]
        releases[version] = [
            {
                "filename": filename,
                "packagetype": "bdist_wheel",
                "url": f"https://packages.invalid/{filename}",
                "digests": {"sha256": digest},
                "yanked": False,
            }
        ]
    return {"releases": releases}


def _get_factory(
    payload: dict[str, Any],
    wheel_map: dict[str, bytes],
):
    def get(url: str, **_kwargs: Any) -> _Response:
        if url.endswith("/json"):
            return _Response(payload=payload)
        return _Response(content=wheel_map[url.rsplit("/", 1)[-1]])

    return get


def _seed_version(
    root: Path,
    spec: RuntimePackageSpec,
    version: str,
) -> None:
    target = root / spec.key / "versions" / version
    target.mkdir(parents=True)
    target.joinpath("runtime_probe_pkg").mkdir()
    target.joinpath("runtime_probe_pkg", "__init__.py").write_text(
        f"__version__ = {version!r}\n",
        encoding="utf-8",
    )
    seed_runtime_package(spec, root, version)


def test_activation_recovers_to_previous_when_active_directory_is_missing(
    tmp_path: Path,
) -> None:
    spec = RuntimePackageSpec(
        "runtime-probe-pkg",
        "runtime_probe_pkg",
        ">=1,<4",
    )
    _seed_version(tmp_path, spec, "1.0.0")
    state_path = tmp_path / spec.key / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "active": "2.0.0",
                "previous": "1.0.0",
                "distribution": spec.distribution,
                "import_name": spec.import_name,
            }
        ),
        encoding="utf-8",
    )

    activation = activate_runtime_package(spec, tmp_path)
    try:
        assert activation.version == "1.0.0"
        assert activation.recovered_from == "2.0.0"
        assert json.loads(state_path.read_text(encoding="utf-8"))["active"] == "1.0.0"
    finally:
        sys.path.remove(str(activation.package_path.resolve()))


def test_update_install_rejection_and_rollback(tmp_path: Path) -> None:
    spec = RuntimePackageSpec(
        "runtime-probe-pkg",
        "runtime_probe_pkg",
        ">=1,<4",
        index_url="https://packages.invalid/json",
    )
    _seed_version(tmp_path, spec, "1.0.0")
    wheels = [
        _wheel_bytes("2.0.0"),
        _wheel_bytes("3.0.0", broken=True),
    ]
    wheel_map = {filename: content for filename, content, _digest in wheels}

    get_v2 = _get_factory(_index_payload(wheels[:1]), wheel_map)
    update = check_runtime_package_update(spec, tmp_path, get=get_v2)
    assert update.current_version == "1.0.0"
    assert update.latest_version == "2.0.0"
    assert update.update_available

    installed = install_runtime_package_update(
        spec,
        tmp_path,
        update,
        probe_command=PROBE_COMMAND,
        get=get_v2,
    )
    assert installed.version == "2.0.0"
    assert installed.previous_version == "1.0.0"
    assert installed.package_path.is_dir()

    get_v3 = _get_factory(_index_payload(wheels), wheel_map)
    broken_update = check_runtime_package_update(spec, tmp_path, get=get_v3)
    assert broken_update.latest_version == "3.0.0"
    with pytest.raises(RuntimePackageHealthError, match="broken candidate"):
        install_runtime_package_update(
            spec,
            tmp_path,
            broken_update,
            probe_command=PROBE_COMMAND,
            get=get_v3,
        )

    state_path = tmp_path / spec.key / "state.json"
    assert json.loads(state_path.read_text(encoding="utf-8"))["active"] == "2.0.0"

    rolled_back = rollback_runtime_package(
        spec,
        tmp_path,
        probe_command=PROBE_COMMAND,
    )
    assert rolled_back.version == "1.0.0"
    assert json.loads(state_path.read_text(encoding="utf-8"))["active"] == "1.0.0"


def test_hash_mismatch_does_not_change_active_version(tmp_path: Path) -> None:
    spec = RuntimePackageSpec(
        "runtime-probe-pkg",
        "runtime_probe_pkg",
        ">=1,<3",
        index_url="https://packages.invalid/json",
    )
    _seed_version(tmp_path, spec, "1.0.0")
    wheel = _wheel_bytes("2.0.0")
    get = _get_factory(
        _index_payload([wheel]),
        {wheel[0]: wheel[1]},
    )
    update = check_runtime_package_update(spec, tmp_path, get=get)
    assert update.artifact is not None
    corrupted = replace(
        update,
        artifact=replace(update.artifact, sha256="0" * 64),
    )

    with pytest.raises(RuntimePackageError, match="SHA256 mismatch"):
        install_runtime_package_update(
            spec,
            tmp_path,
            corrupted,
            probe_command=PROBE_COMMAND,
            get=get,
        )

    state_path = tmp_path / spec.key / "state.json"
    assert json.loads(state_path.read_text(encoding="utf-8"))["active"] == "1.0.0"


def test_update_check_rejects_wheel_for_another_python_version(
    tmp_path: Path,
) -> None:
    spec = RuntimePackageSpec(
        "runtime-probe-pkg",
        "runtime_probe_pkg",
        ">=1,<3",
        index_url="https://packages.invalid/json",
    )
    wheel = _wheel_bytes("2.0.0")
    payload = _index_payload([wheel])
    payload["releases"]["2.0.0"][0]["requires_python"] = ">=99"

    update = check_runtime_package_update(
        spec,
        tmp_path,
        get=_get_factory(payload, {wheel[0]: wheel[1]}),
    )

    assert update.latest_version is None
    assert not update.update_available
    assert update.artifact is None
