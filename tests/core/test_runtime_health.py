from types import SimpleNamespace
from unittest.mock import patch

from NepTrainKit.core import runtime_health


def test_runtime_health_reports_native_and_adapter_capabilities():
    adapter = SimpleNamespace(
        backend_status=lambda backend: SimpleNamespace(
            available=backend == "cpu",
            reason="available" if backend == "cpu" else "module_missing",
            detail=f"{backend} detail",
        ),
        __version__="1.2.3",
    )

    def import_module(name):
        if name == "NepTrainKit._native._phase":
            raise ImportError("phase missing")
        if name == "nep_adapters":
            return adapter
        return object()

    with (
        patch.object(runtime_health.importlib, "import_module", side_effect=import_module),
        patch.object(
            runtime_health.metadata,
            "version",
            side_effect=runtime_health.metadata.PackageNotFoundError,
        ),
    ):
        report = runtime_health.inspect_runtime_health()

    assert report.native_available_count == len(runtime_health.NATIVE_HELPER_MODULES) - 1
    native_status = {capability.name: capability for capability in report.native}
    assert native_status["_sampling"].available
    assert not native_status["_phase"].available
    assert not report.native_complete
    assert report.adapters_version == "1.2.3"
    assert report.cpu.available
    assert not report.cuda.available
    assert report.cuda.reason == "module_missing"


def test_runtime_health_prefers_the_loaded_adapter_version():
    adapter = SimpleNamespace(
        backend_status=lambda backend: SimpleNamespace(
            available=True,
            reason="available",
            detail="",
        ),
        __version__="1.0.2",
    )

    def import_module(name):
        if name == "nep_adapters":
            return adapter
        return object()

    with (
        patch.object(runtime_health.importlib, "import_module", side_effect=import_module),
        patch.object(runtime_health.metadata, "version", return_value="1.0.1") as version,
    ):
        report = runtime_health.inspect_runtime_health()

    assert report.adapters_version == "1.0.2"
    version.assert_not_called()


def test_runtime_health_reports_missing_adapter_without_raising():
    def import_module(name):
        if name == "nep_adapters":
            raise ImportError("adapter missing")
        return object()

    with patch.object(
        runtime_health.importlib,
        "import_module",
        side_effect=import_module,
    ):
        report = runtime_health.inspect_runtime_health()

    assert report.native_complete
    assert report.adapters_version is None
    assert not report.cpu.available
    assert report.cpu.reason == "ImportError"
    assert not report.cuda.available
