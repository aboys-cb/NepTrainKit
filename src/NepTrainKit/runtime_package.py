"""Managed sidecar packages for Python and standalone NepTrainKit installs.

The application keeps updateable native runtimes outside the compiled program.
An update is downloaded into a versioned directory, checked in a fresh process,
and only then made active for the next application start.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

PYPI_JSON_URL = "https://pypi.org/pypi/{distribution}/json"
HEALTH_CHECK_FLAG = "--neptrainkit-runtime-health-check"


class RuntimePackageError(RuntimeError):
    """Base error raised by the managed-runtime module."""


class RuntimePackageHealthError(RuntimePackageError):
    """Raised when a candidate package fails its isolated health check."""


@dataclass(frozen=True)
class RuntimePackageSpec:
    """Stable description of one updateable runtime package."""

    distribution: str
    import_name: str
    version_constraint: str
    health_kind: str = "import"
    index_url: str = PYPI_JSON_URL

    @property
    def key(self) -> str:
        """Return the normalized on-disk package key."""
        return canonicalize_name(self.distribution)


NEP_ADAPTERS_SPEC = RuntimePackageSpec(
    distribution="nep-adapters",
    import_name="nep_adapters",
    version_constraint=">=1.0",
    health_kind="nep_adapters_cpu",
)

MANAGED_RUNTIME_SPEC = NEP_ADAPTERS_SPEC


@dataclass(frozen=True)
class WheelArtifact:
    """One compatible wheel selected from a package index."""

    version: str
    filename: str
    url: str
    sha256: str


@dataclass(frozen=True)
class RuntimePackageUpdate:
    """Result of checking a runtime package index."""

    current_version: str | None
    latest_version: str | None
    update_available: bool
    artifact: WheelArtifact | None


@dataclass(frozen=True)
class RuntimePackageInstall:
    """Result of installing and activating a checked wheel."""

    version: str
    previous_version: str | None
    package_path: Path
    restart_required: bool = True


@dataclass(frozen=True)
class RuntimeActivation:
    """The sidecar path selected during application startup."""

    version: str | None
    package_path: Path | None
    recovered_from: str | None = None


def _is_nuitka_compiled() -> bool:
    if "__compiled__" in globals():
        return True
    try:
        return Path(sys.argv[0]).resolve() == Path(sys.executable).resolve()
    except (IndexError, OSError):
        return False


def default_runtime_root(*, compiled: bool | None = None) -> Path:
    """Return the writable runtime root for the current delivery mode."""
    override = os.environ.get("NEPTRAINKIT_RUNTIME_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    is_compiled = _is_nuitka_compiled() if compiled is None else compiled
    if is_compiled:
        return Path(sys.executable).resolve().parent / "runtime"

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", "").strip()
        if base:
            return Path(base) / "NepTrainKit" / "runtime"
        return Path.home() / "AppData" / "Local" / "NepTrainKit" / "runtime"
    return Path.home() / ".config" / "NepTrainKit" / "runtime"


def _package_root(runtime_root: Path, spec: RuntimePackageSpec) -> Path:
    return Path(runtime_root) / spec.key


def _versions_root(runtime_root: Path, spec: RuntimePackageSpec) -> Path:
    return _package_root(runtime_root, spec) / "versions"


def _state_path(runtime_root: Path, spec: RuntimePackageSpec) -> Path:
    return _package_root(runtime_root, spec) / "state.json"


def _read_state(runtime_root: Path, spec: RuntimePackageSpec) -> dict[str, Any]:
    path = _state_path(runtime_root, spec)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(
    runtime_root: Path,
    spec: RuntimePackageSpec,
    *,
    active: str,
    previous: str | None,
) -> None:
    path = _state_path(runtime_root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "distribution": spec.distribution,
        "import_name": spec.import_name,
        "active": str(active),
        "previous": str(previous) if previous else None,
    }
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _version_directory(
    runtime_root: Path,
    spec: RuntimePackageSpec,
    version: str,
) -> Path:
    return _versions_root(runtime_root, spec) / str(version)


def activate_runtime_package(
    spec: RuntimePackageSpec,
    runtime_root: Path,
) -> RuntimeActivation:
    """Prepend the active sidecar directory, recovering to ``previous`` if missing."""
    state = _read_state(runtime_root, spec)
    active = str(state.get("active") or "").strip()
    previous = str(state.get("previous") or "").strip()
    if not active:
        return RuntimeActivation(version=None, package_path=None)

    active_path = _version_directory(runtime_root, spec, active)
    recovered_from: str | None = None
    if not active_path.is_dir():
        previous_path = _version_directory(runtime_root, spec, previous)
        if not previous or not previous_path.is_dir():
            return RuntimeActivation(version=None, package_path=None)
        recovered_from = active
        active = previous
        active_path = previous_path
        _write_state(
            runtime_root,
            spec,
            active=active,
            previous=None,
        )

    active_text = str(active_path.resolve())
    if active_text in sys.path:
        sys.path.remove(active_text)
    sys.path.insert(0, active_text)
    return RuntimeActivation(
        version=active,
        package_path=active_path,
        recovered_from=recovered_from,
    )


def active_runtime_version(
    spec: RuntimePackageSpec,
    runtime_root: Path,
) -> str | None:
    """Return the active sidecar version, or the environment version."""
    state = _read_state(runtime_root, spec)
    active = str(state.get("active") or "").strip()
    if active and _version_directory(runtime_root, spec, active).is_dir():
        return active
    try:
        return metadata.version(spec.distribution)
    except metadata.PackageNotFoundError:
        return None


def _compatible_wheel(
    spec: RuntimePackageSpec,
    files: Sequence[dict[str, Any]],
    version: Version,
) -> WheelArtifact | None:
    supported = set(sys_tags())
    python_version = Version(
        ".".join(str(part) for part in sys.version_info[:3])
    )
    candidates: list[WheelArtifact] = []
    for item in files:
        if item.get("packagetype") != "bdist_wheel" or bool(item.get("yanked")):
            continue
        requires_python = str(item.get("requires_python") or "").strip()
        if requires_python:
            try:
                if python_version not in SpecifierSet(requires_python):
                    continue
            except InvalidSpecifier:
                continue
        filename = str(item.get("filename") or "").strip()
        url = str(item.get("url") or "").strip()
        digest = str((item.get("digests") or {}).get("sha256") or "").strip()
        if not filename or not url or not digest:
            continue
        try:
            name, wheel_version, _build, tags = parse_wheel_filename(filename)
        except (InvalidVersion, ValueError):
            continue
        if canonicalize_name(name) != spec.key or wheel_version != version or not supported.intersection(tags):
            continue
        candidates.append(
            WheelArtifact(
                version=str(version),
                filename=filename,
                url=url,
                sha256=digest,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.filename)
    return candidates[0]


def check_runtime_package_update(
    spec: RuntimePackageSpec,
    runtime_root: Path,
    *,
    timeout: tuple[float, float] = (2.0, 8.0),
    get: Callable[..., Any] | None = None,
) -> RuntimePackageUpdate:
    """Check the package index and select the newest compatible wheel."""
    if get is None:
        import requests

        get = requests.get

    index_url = spec.index_url.format(distribution=spec.distribution)
    response = get(
        index_url,
        headers={"User-Agent": "NepTrainKit-Runtime-Updater"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    releases = payload.get("releases") if isinstance(payload, dict) else None
    if not isinstance(releases, dict):
        raise RuntimePackageError("Unexpected package-index response.")

    constraint = SpecifierSet(spec.version_constraint)
    current_text = active_runtime_version(spec, runtime_root)
    try:
        current = Version(current_text) if current_text else None
    except InvalidVersion:
        current = None

    compatible: list[tuple[Version, WheelArtifact]] = []
    for version_text, files in releases.items():
        try:
            version = Version(str(version_text))
        except InvalidVersion:
            continue
        if version not in constraint or not isinstance(files, list):
            continue
        wheel = _compatible_wheel(spec, files, version)
        if wheel is not None:
            compatible.append((version, wheel))

    if not compatible:
        return RuntimePackageUpdate(
            current_version=current_text,
            latest_version=None,
            update_available=False,
            artifact=None,
        )

    compatible.sort(key=lambda item: item[0])
    latest, artifact = compatible[-1]
    return RuntimePackageUpdate(
        current_version=current_text,
        latest_version=str(latest),
        update_available=current is None or latest > current,
        artifact=artifact if current is None or latest > current else None,
    )


def _download_wheel(
    artifact: WheelArtifact,
    target: Path,
    *,
    timeout: tuple[float, float] = (3.0, 60.0),
    get: Callable[..., Any] | None = None,
) -> None:
    if get is None:
        import requests

        get = requests.get
    response = get(
        artifact.url,
        headers={"User-Agent": "NepTrainKit-Runtime-Updater"},
        timeout=timeout,
        stream=True,
    )
    response.raise_for_status()
    digest = hashlib.sha256()
    with target.open("wb") as stream:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                digest.update(chunk)
                stream.write(chunk)
    if digest.hexdigest().lower() != artifact.sha256.lower():
        raise RuntimePackageError(f"SHA256 mismatch for downloaded wheel {artifact.filename}.")


def _safe_extract_wheel(wheel: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in info.filename
                or (relative.parts and ":" in relative.parts[0])
            ):
                raise RuntimePackageError(f"Unsafe path in wheel: {info.filename}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise RuntimePackageError(f"Symbolic links are not allowed in runtime wheels: {info.filename}")
        archive.extractall(destination)


def _default_probe_command() -> list[str]:
    if _is_nuitka_compiled():
        return [sys.executable]
    return [sys.executable, "-m", "NepTrainKit.main"]


def _probe_candidate(
    spec: RuntimePackageSpec,
    candidate: Path,
    version: str,
    *,
    probe_command: Sequence[str] | None,
    timeout: float,
) -> None:
    command = list(probe_command or _default_probe_command())
    command.extend(
        [
            HEALTH_CHECK_FLAG,
            str(candidate),
            spec.import_name,
            spec.distribution,
            str(version),
            spec.health_kind,
        ]
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimePackageHealthError(detail or f"Runtime health check exited with {completed.returncode}.")


def _activate_version(
    spec: RuntimePackageSpec,
    runtime_root: Path,
    version: str,
) -> str | None:
    state = _read_state(runtime_root, spec)
    current = str(state.get("active") or "").strip() or None
    previous = current if current != version else state.get("previous")
    _write_state(
        runtime_root,
        spec,
        active=version,
        previous=str(previous) if previous else None,
    )
    return current


def _cleanup_old_versions(
    spec: RuntimePackageSpec,
    runtime_root: Path,
) -> None:
    state = _read_state(runtime_root, spec)
    keep = {str(value) for value in (state.get("active"), state.get("previous")) if value}
    versions = _versions_root(runtime_root, spec)
    if not versions.is_dir():
        return
    for child in versions.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child)


def install_runtime_package_update(
    spec: RuntimePackageSpec,
    runtime_root: Path,
    update: RuntimePackageUpdate,
    *,
    probe_command: Sequence[str] | None = None,
    health_timeout: float = 60.0,
    get: Callable[..., Any] | None = None,
) -> RuntimePackageInstall:
    """Download, verify, health-check, and activate one checked update."""
    artifact = update.artifact
    if not update.update_available or artifact is None:
        raise RuntimePackageError("No compatible runtime-package update is available.")

    package_root = _package_root(runtime_root, spec)
    package_root.mkdir(parents=True, exist_ok=True)
    versions_root = _versions_root(runtime_root, spec)
    versions_root.mkdir(parents=True, exist_ok=True)
    target = _version_directory(runtime_root, spec, artifact.version)

    with tempfile.TemporaryDirectory(
        prefix=".install-",
        dir=package_root,
    ) as temporary:
        temporary_path = Path(temporary)
        wheel = temporary_path / artifact.filename
        candidate = temporary_path / "candidate"
        _download_wheel(artifact, wheel, get=get)
        _safe_extract_wheel(wheel, candidate)
        _probe_candidate(
            spec,
            candidate,
            artifact.version,
            probe_command=probe_command,
            timeout=health_timeout,
        )

        if target.exists():
            shutil.rmtree(target)
        os.replace(candidate, target)

    previous = _activate_version(spec, runtime_root, artifact.version)
    _cleanup_old_versions(spec, runtime_root)
    return RuntimePackageInstall(
        version=artifact.version,
        previous_version=previous,
        package_path=target,
    )


def seed_runtime_package(
    spec: RuntimePackageSpec,
    runtime_root: Path,
    version: str,
) -> RuntimePackageInstall:
    """Activate a version directory populated by the release build."""
    target = _version_directory(runtime_root, spec, version)
    if not target.is_dir():
        raise RuntimePackageError(f"Seed package directory does not exist: {target}")
    previous = _activate_version(spec, runtime_root, version)
    _cleanup_old_versions(spec, runtime_root)
    return RuntimePackageInstall(
        version=version,
        previous_version=previous,
        package_path=target,
    )


def rollback_runtime_package(
    spec: RuntimePackageSpec,
    runtime_root: Path,
    *,
    probe_command: Sequence[str] | None = None,
    health_timeout: float = 60.0,
) -> RuntimePackageInstall:
    """Health-check and reactivate the previously active version."""
    state = _read_state(runtime_root, spec)
    active = str(state.get("active") or "").strip()
    previous = str(state.get("previous") or "").strip()
    if not previous:
        raise RuntimePackageError("No previous runtime-package version is available.")
    target = _version_directory(runtime_root, spec, previous)
    if not target.is_dir():
        raise RuntimePackageError(f"Previous runtime-package directory is missing: {target}")
    _probe_candidate(
        spec,
        target,
        previous,
        probe_command=probe_command,
        timeout=health_timeout,
    )
    _write_state(
        runtime_root,
        spec,
        active=previous,
        previous=active or None,
    )
    return RuntimePackageInstall(
        version=previous,
        previous_version=active or None,
        package_path=target,
    )


def run_runtime_health_command(argv: Sequence[str] | None = None) -> int | None:
    """Run the internal fresh-process health command when its flag is present."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] != HEALTH_CHECK_FLAG:
        return None
    if len(arguments) != 6:
        print("Invalid runtime health-check arguments.", file=sys.stderr)
        return 2

    _flag, candidate, import_name, distribution, expected_version, health_kind = arguments
    candidate_path = Path(candidate).resolve()
    if not candidate_path.is_dir():
        print(f"Candidate directory does not exist: {candidate_path}", file=sys.stderr)
        return 3

    for name in list(sys.modules):
        if name == import_name or name.startswith(f"{import_name}."):
            sys.modules.pop(name, None)
    candidate_text = str(candidate_path)
    if candidate_text in sys.path:
        sys.path.remove(candidate_text)
    sys.path.insert(0, candidate_text)
    importlib.invalidate_caches()

    try:
        module = importlib.import_module(import_name)
        try:
            loaded_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            loaded_version = str(getattr(module, "__version__", ""))
        if Version(loaded_version) != Version(expected_version):
            raise RuntimePackageHealthError(f"Expected {expected_version}, loaded {loaded_version or 'unknown'}.")
        if health_kind == "nep_adapters_cpu":
            status = module.backend_status("cpu")
            if not bool(getattr(status, "available", False)):
                detail = str(getattr(status, "detail", "") or "")
                raise RuntimePackageHealthError(detail or "The nep-adapters CPU backend is unavailable.")
        elif health_kind != "import":
            raise RuntimePackageHealthError(f"Unknown runtime health check: {health_kind}")
    except Exception as exc:  # noqa: BLE001 - this command reports loader failures
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    return 0


def _command_line(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed = subparsers.add_parser("seed", help="activate a pre-populated version")
    seed.add_argument("--runtime-root", type=Path, required=True)
    seed.add_argument("--distribution", required=True)
    seed.add_argument("--import-name", required=True)
    seed.add_argument("--constraint", default="")
    seed.add_argument("--health-kind", default="import")
    seed.add_argument("--version", required=True)
    args = parser.parse_args(argv)

    if args.command == "seed":
        spec = RuntimePackageSpec(
            distribution=args.distribution,
            import_name=args.import_name,
            version_constraint=args.constraint,
            health_kind=args.health_kind,
        )
        seed_runtime_package(spec, args.runtime_root, args.version)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_command_line())


__all__ = [
    "HEALTH_CHECK_FLAG",
    "MANAGED_RUNTIME_SPEC",
    "NEP_ADAPTERS_SPEC",
    "PYPI_JSON_URL",
    "RuntimeActivation",
    "RuntimePackageError",
    "RuntimePackageHealthError",
    "RuntimePackageInstall",
    "RuntimePackageSpec",
    "RuntimePackageUpdate",
    "WheelArtifact",
    "activate_runtime_package",
    "active_runtime_version",
    "check_runtime_package_update",
    "default_runtime_root",
    "install_runtime_package_update",
    "rollback_runtime_package",
    "run_runtime_health_command",
    "seed_runtime_package",
]
