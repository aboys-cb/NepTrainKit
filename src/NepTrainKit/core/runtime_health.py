"""Runtime capability inspection for packaged NepTrainKit features."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata


_NATIVE_MODULES = ("_io", "_audit", "_phase", "_magnetism")


@dataclass(frozen=True)
class RuntimeCapability:
    """One runtime feature and its current availability."""

    name: str
    available: bool
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class RuntimeHealth:
    """Snapshot of native helpers and external NEP backends."""

    native: tuple[RuntimeCapability, ...]
    adapters_version: str | None
    cpu: RuntimeCapability
    cuda: RuntimeCapability

    @property
    def native_available_count(self) -> int:
        return sum(item.available for item in self.native)

    @property
    def native_complete(self) -> bool:
        return self.native_available_count == len(self.native)


def _native_capability(module_name: str) -> RuntimeCapability:
    qualified_name = f"NepTrainKit._native.{module_name}"
    try:
        importlib.import_module(qualified_name)
    except Exception as exc:  # noqa: BLE001 - report all loader failures
        return RuntimeCapability(
            name=module_name,
            available=False,
            reason=type(exc).__name__,
            detail=str(exc),
        )
    return RuntimeCapability(
        name=module_name,
        available=True,
        reason="available",
    )


def _adapter_capability(adapter_module, backend: str) -> RuntimeCapability:
    try:
        status = adapter_module.backend_status(backend)
    except Exception as exc:  # noqa: BLE001 - report adapter diagnostics
        return RuntimeCapability(
            name=backend,
            available=False,
            reason=type(exc).__name__,
            detail=str(exc),
        )
    return RuntimeCapability(
        name=backend,
        available=bool(getattr(status, "available", False)),
        reason=str(getattr(status, "reason", "unknown")),
        detail=str(getattr(status, "detail", "")),
    )


def inspect_runtime_health() -> RuntimeHealth:
    """Return a side-effect-free snapshot of packaged runtime capabilities."""
    native = tuple(_native_capability(name) for name in _NATIVE_MODULES)
    try:
        adapters = importlib.import_module("nep_adapters")
    except Exception as exc:  # noqa: BLE001 - missing/broken package is report data
        unavailable = RuntimeCapability(
            name="nep-adapters",
            available=False,
            reason=type(exc).__name__,
            detail=str(exc),
        )
        return RuntimeHealth(
            native=native,
            adapters_version=None,
            cpu=RuntimeCapability(
                name="cpu",
                available=False,
                reason=unavailable.reason,
                detail=unavailable.detail,
            ),
            cuda=RuntimeCapability(
                name="cuda",
                available=False,
                reason=unavailable.reason,
                detail=unavailable.detail,
            ),
        )

    try:
        version = metadata.version("nep-adapters")
    except metadata.PackageNotFoundError:
        version = getattr(adapters, "__version__", None)

    return RuntimeHealth(
        native=native,
        adapters_version=str(version) if version else None,
        cpu=_adapter_capability(adapters, "cpu"),
        cuda=_adapter_capability(adapters, "cuda"),
    )


__all__ = [
    "RuntimeCapability",
    "RuntimeHealth",
    "inspect_runtime_health",
]
