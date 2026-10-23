"""Registry of validation protocols.

Adding a property (elastic, phonon, magnetic order ...) means writing one
``ValidationProtocol`` subclass and calling :func:`register` for it.
"""

from __future__ import annotations

from .base import (
    FIXED_GEOMETRY,
    MODEL_RELAXED,
    Evaluation,
    Issue,
    Metric,
    ModelCapabilities,
    ValidationProtocol,
    read_frames,
)
from .efv import EnergyForceVirialProtocol
from .eos import EquationOfStateProtocol
from .diagnostic import DiagnosticProtocol

_REGISTRY: dict[str, ValidationProtocol] = {}


def register(protocol: ValidationProtocol) -> ValidationProtocol:
    if not protocol.kind:
        raise ValueError("A protocol needs a kind.")
    _REGISTRY[protocol.kind] = protocol
    return protocol


def get_protocol(kind: str) -> ValidationProtocol:
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise ValueError(f"Unknown validation protocol: {kind}") from None


def protocols() -> list[ValidationProtocol]:
    return list(_REGISTRY.values())


def detect_protocol(frames) -> list[tuple[float, ValidationProtocol]]:
    """All protocols ranked by how well ``frames`` match them, best first."""
    ranked = [(protocol.detect(frames), protocol) for protocol in _REGISTRY.values()]
    return sorted((item for item in ranked if item[0] > 0), key=lambda item: -item[0])


register(EnergyForceVirialProtocol())
register(EquationOfStateProtocol())
register(DiagnosticProtocol())

__all__ = [
    "FIXED_GEOMETRY",
    "MODEL_RELAXED",
    "Evaluation",
    "Issue",
    "Metric",
    "ModelCapabilities",
    "ValidationProtocol",
    "detect_protocol",
    "get_protocol",
    "protocols",
    "read_frames",
    "register",
]
