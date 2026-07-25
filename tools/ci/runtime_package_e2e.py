#!/usr/bin/env python
"""End-to-end probe for managed package updates in Python and Nuitka modes."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import tempfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from NepTrainKit.runtime_package import (
    RuntimePackageHealthError,
    RuntimePackageSpec,
    activate_runtime_package,
    check_runtime_package_update,
    install_runtime_package_update,
    rollback_runtime_package,
    run_runtime_health_command,
)

health_exit = run_runtime_health_command()
if health_exit is not None:
    raise SystemExit(health_exit)


def _wheel(version: str, *, broken: bool = False) -> tuple[str, bytes, str]:
    filename = f"runtime_probe_pkg-{version}-py3-none-any.whl"
    module_source = "raise RuntimeError('intentional broken update')\n" if broken else f"__version__ = {version!r}\n"
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("runtime_probe_pkg/__init__.py", module_source)
        dist_info = f"runtime_probe_pkg-{version}.dist-info"
        archive.writestr(
            f"{dist_info}/METADATA",
            (f"Metadata-Version: 2.1\nName: runtime-probe-pkg\nVersion: {version}\n"),
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            ("Wheel-Version: 1.0\nGenerator: NepTrainKit CI probe\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    content = data.getvalue()
    return filename, content, hashlib.sha256(content).hexdigest()


class _PackageServer:
    def __init__(self) -> None:
        wheels = [
            _wheel("1.0.0"),
            _wheel("2.0.0"),
            _wheel("3.0.0", broken=True),
        ]
        self.files = {filename: content for filename, content, _digest in wheels}
        self.digests = {filename: digest for filename, _content, digest in wheels}
        self.visible_versions = ["1.0.0"]
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler interface
                if self.path == "/pypi/runtime-probe-pkg/json":
                    payload = owner._payload()
                    content = json.dumps(payload).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                filename = self.path.removeprefix("/files/")
                content = owner.files.get(filename)
                if content is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, _format: str, *_args: Any) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def _payload(self) -> dict[str, Any]:
        releases: dict[str, list[dict[str, Any]]] = {}
        for version in self.visible_versions:
            filename = f"runtime_probe_pkg-{version}-py3-none-any.whl"
            releases[version] = [
                {
                    "filename": filename,
                    "packagetype": "bdist_wheel",
                    "url": f"{self.base_url}/files/{filename}",
                    "digests": {"sha256": self.digests[filename]},
                    "yanked": False,
                }
            ]
        return {"releases": releases}

    def __enter__(self) -> _PackageServer:
        self.thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="neptrainkit-runtime-e2e-") as temporary:
        runtime_root = Path(temporary) / "runtime"
        with _PackageServer() as package_server:
            spec = RuntimePackageSpec(
                distribution="runtime-probe-pkg",
                import_name="runtime_probe_pkg",
                version_constraint=">=1,<4",
                health_kind="import",
                index_url=f"{package_server.base_url}/pypi/{{distribution}}/json",
            )

            first = check_runtime_package_update(spec, runtime_root)
            installed_v1 = install_runtime_package_update(spec, runtime_root, first)
            assert installed_v1.version == "1.0.0"

            package_server.visible_versions = ["1.0.0", "2.0.0"]
            second = check_runtime_package_update(spec, runtime_root)
            assert second.current_version == "1.0.0"
            assert second.latest_version == "2.0.0"
            installed_v2 = install_runtime_package_update(spec, runtime_root, second)
            assert installed_v2.previous_version == "1.0.0"

            package_server.visible_versions = ["1.0.0", "2.0.0", "3.0.0"]
            broken = check_runtime_package_update(spec, runtime_root)
            try:
                install_runtime_package_update(spec, runtime_root, broken)
            except RuntimePackageHealthError:
                pass
            else:
                raise AssertionError("The intentionally broken update was activated.")

            state_path = runtime_root / spec.key / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            assert state["active"] == "2.0.0"

            rolled_back = rollback_runtime_package(spec, runtime_root)
            assert rolled_back.version == "1.0.0"
            activation = activate_runtime_package(spec, runtime_root)
            assert activation.version == "1.0.0"
            importlib.invalidate_caches()
            module = importlib.import_module(spec.import_name)
            assert module.__version__ == "1.0.0"

            result = {
                "mode": "nuitka" if "__compiled__" in globals() else "python",
                "detected": ["1.0.0", "2.0.0", "3.0.0"],
                "rejected": "3.0.0",
                "active_after_rollback": activation.version,
            }
            print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
