"""Dynamic access to the external ``nep-adapters`` runtime.

Keeping the import behind this module prevents standalone builds from compiling
the backend into NepTrainKit while preserving one application-facing interface.
"""

from __future__ import annotations

import importlib

from NepTrainKit.runtime_package import NEP_ADAPTERS_SPEC

try:
    _adapter = importlib.import_module(NEP_ADAPTERS_SPEC.import_name)
except ImportError:
    _adapter = None


if _adapter is None:
    BackendStatus = None
    BackendUnavailableError = Exception
    ChargePrediction = None
    ModelInfo = None
    AdapterCalculator = None
    NepAdaptersError = Exception
    OutOfMemoryError = Exception
    Prediction = None
    SpinPrediction = None
    UnsupportedModelError = Exception
    nep_adapters_version = "0.0.0"

    def backend_status(backend: str | None = None):
        """Raise the standard missing-runtime error."""
        raise ImportError("nep-adapters is not installed")

    def inspect_model(model_path, **kwargs):
        """Raise the standard missing-runtime error."""
        raise ImportError("nep-adapters is not installed")

else:
    BackendStatus = _adapter.BackendStatus
    BackendUnavailableError = _adapter.BackendUnavailableError
    ChargePrediction = _adapter.ChargePrediction
    ModelInfo = _adapter.ModelInfo
    AdapterCalculator = _adapter.NEPCalculator
    NepAdaptersError = _adapter.NepAdaptersError
    OutOfMemoryError = _adapter.OutOfMemoryError
    Prediction = _adapter.Prediction
    SpinPrediction = _adapter.SpinPrediction
    UnsupportedModelError = _adapter.UnsupportedModelError
    backend_status = _adapter.backend_status
    inspect_model = _adapter.inspect_model
    nep_adapters_version = str(getattr(_adapter, "__version__", "0.0.0"))


__all__ = [
    "AdapterCalculator",
    "BackendStatus",
    "BackendUnavailableError",
    "ChargePrediction",
    "ModelInfo",
    "NepAdaptersError",
    "OutOfMemoryError",
    "Prediction",
    "SpinPrediction",
    "UnsupportedModelError",
    "backend_status",
    "inspect_model",
    "nep_adapters_version",
]
