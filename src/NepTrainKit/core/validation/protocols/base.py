"""Contract shared by every validation protocol (EOS, elastic, phonon, magnetic ...).

A protocol turns one sealed benchmark set plus one model into named metrics and
plot-ready details. Two kinds exist and the UI must label them:

* ``fixed-geometry``: the model predicts on exactly the reference structures.
* ``model-relaxed``: the model first relaxes or derives its own structures.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from typing import Protocol as TypingProtocol

import numpy as np
from ase import Atoms
from ase.io import read

FIXED_GEOMETRY = "fixed-geometry"
MODEL_RELAXED = "model-relaxed"


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    better: str = "lower"  # lower | higher | closer-to-zero | none (descriptive, no ranking)
    headline: bool = False
    # Differences up to this many ``unit`` count as unchanged; relative
    # tolerance alone turns noise into verdicts when the baseline is near zero.
    abs_tolerance: float = 0.0


@dataclass(frozen=True)
class Issue:
    level: str  # error | warning
    message: str


@dataclass
class Evaluation:
    """Metrics are flat numbers (``None`` when not applicable); details feed plots."""

    metrics: dict[str, float | None]
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class Predictor(TypingProtocol):
    """What a protocol may ask of a model; ``NepCalculator`` satisfies it."""

    def predict(self, structures: Sequence[Atoms]): ...


@dataclass(frozen=True)
class ModelCapabilities:
    model_kind: str
    elements: tuple[str, ...]
    is_spin: bool = False


class ValidationProtocol(ABC):
    kind: str = ""
    version: int = 1
    mode: str = FIXED_GEOMETRY
    title: str = ""
    metrics: tuple[Metric, ...] = ()
    supports_spin: bool = False

    def detect(self, frames: Sequence[Atoms]) -> float:
        """Confidence in [0, 1] that ``frames`` were prepared for this protocol."""
        return 0.0

    @abstractmethod
    def check(self, frames: Sequence[Atoms]) -> list[Issue]:
        """Validate a candidate reference set before it is sealed."""

    def summarize_reference(self, frames: Sequence[Atoms]) -> dict:
        """Reference-only data shown before any model is evaluated."""
        return {}

    def supports(self, model: ModelCapabilities) -> str | None:
        """Return a reason when the model cannot be evaluated, else ``None``."""
        if model.is_spin and not self.supports_spin:
            return "This protocol does not support spin models yet."
        return None

    @abstractmethod
    def evaluate(self, predictor: Predictor, frames: Sequence[Atoms]) -> Evaluation:
        """Compute metrics; must not mutate ``frames`` labels."""

    def metric(self, key: str) -> Metric:
        for metric in self.metrics:
            if metric.key == key:
                return metric
        raise KeyError(key)

    def headline_metrics(self) -> tuple[Metric, ...]:
        return tuple(m for m in self.metrics if m.headline)


# -- shared helpers -------------------------------------------------------------

def read_frames(data: bytes) -> list[Atoms]:
    return read(io.StringIO(data.decode("utf-8")), index=":", format="extxyz")


@dataclass
class ReferenceLabels:
    """DFT labels normalised to the units NEP reports."""

    atom_counts: np.ndarray
    energy_per_atom: np.ndarray            # eV/atom
    forces: np.ndarray | None              # (total_atoms, 3) eV/Å
    virial_per_atom: np.ndarray | None     # (frames, 9) eV/atom


def _virial_per_atom(atoms: Atoms) -> np.ndarray | None:
    raw = atoms.info.get("virial")
    if raw is None:
        raw = atoms.info.get("virials")
    if raw is not None:
        virial = np.asarray(raw, dtype=float).reshape(-1)
        if virial.size == 6:  # Voigt xx yy zz yz xz xy
            xx, yy, zz, yz, xz, xy = virial
            virial = np.array([xx, xy, xz, xy, yy, yz, xz, yz, zz])
        return virial.reshape(9) / len(atoms)
    try:
        stress = atoms.get_stress(voigt=False)
    except Exception:
        return None
    return (-np.asarray(stress) * atoms.get_volume()).reshape(9) / len(atoms)


def _energy(atoms: Atoms) -> float:
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        if "energy" in atoms.info:
            return float(atoms.info["energy"])
        raise ValueError("Every reference frame needs a total energy label in eV.") from None


def _forces(atoms: Atoms) -> np.ndarray | None:
    try:
        return np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        # NEP/GPUMD files name the column ``force``, which ASE keeps as a plain array.
        for key in ("forces", "force"):
            if key in atoms.arrays:
                return np.asarray(atoms.arrays[key], dtype=float)
    return None


def reference_labels(frames: Sequence[Atoms]) -> ReferenceLabels:
    counts = np.asarray([len(atoms) for atoms in frames], dtype=int)
    energies = [_energy(atoms) / len(atoms) for atoms in frames]
    forces = [_forces(atoms) for atoms in frames]
    if any(block is None for block in forces):
        forces = None
    virials = [_virial_per_atom(atoms) for atoms in frames]
    return ReferenceLabels(
        atom_counts=counts,
        energy_per_atom=np.asarray(energies, dtype=float),
        forces=np.concatenate(forces) if forces else None,
        virial_per_atom=None if any(v is None for v in virials) else np.vstack(virials),
    )


def unlabelled_copies(frames: Sequence[Atoms]) -> list[Atoms]:
    """Return copies without reference calculators, so labels never reach the model."""
    copies = []
    for atoms in frames:
        clone = atoms.copy()
        clone.calc = None
        copies.append(clone)
    return copies


def rmse(delta) -> float:
    delta = np.asarray(delta, dtype=float)
    return float(np.sqrt(np.mean(delta**2)))


def mae(delta) -> float:
    return float(np.mean(np.abs(np.asarray(delta, dtype=float))))
