"""Generic held-out test set: energy / force / virial errors on labelled frames."""

from __future__ import annotations

import numpy as np

from .base import (
    Evaluation,
    Issue,
    Metric,
    ValidationProtocol,
    mae,
    reference_labels,
    rmse,
    unlabelled_copies,
)


class EnergyForceVirialProtocol(ValidationProtocol):
    kind = "efv"
    version = 1
    title = "Energy / force / virial"
    supports_spin = True
    metrics = (
        Metric("energy_rmse", "Energy RMSE", "meV/atom", headline=True),
        Metric("force_rmse", "Force RMSE", "meV/Å", headline=True),
        Metric("virial_rmse", "Virial RMSE", "meV/atom"),
        Metric("energy_mae", "Energy MAE", "meV/atom"),
        Metric("force_mae", "Force MAE", "meV/Å"),
    )

    def detect(self, frames) -> float:
        # Any labelled set qualifies; specific protocols should outrank this one.
        return 0.1 if frames else 0.0

    def check(self, frames) -> list[Issue]:
        if not frames:
            return [Issue("error", "The file contains no structures.")]
        try:
            labels = reference_labels(frames)
        except ValueError as error:
            return [Issue("error", str(error))]
        issues = []
        if not np.isfinite(labels.energy_per_atom).all():
            issues.append(Issue("error", "Reference energies must be finite."))
        if labels.forces is None:
            issues.append(Issue("warning", "No force labels; only energies will be compared."))
        return issues

    def summarize_reference(self, frames) -> dict:
        labels = reference_labels(frames)
        return {
            "frames": len(frames),
            "atoms": int(labels.atom_counts.sum()),
            "has_forces": labels.forces is not None,
            "has_virials": labels.virial_per_atom is not None,
        }

    def evaluate(self, predictor, frames) -> Evaluation:
        labels = reference_labels(frames)
        prediction = predictor.predict(unlabelled_copies(frames))
        energy = np.asarray(prediction.energy, dtype=float) / labels.atom_counts
        de = (energy - labels.energy_per_atom) * 1000
        metrics: dict[str, float | None] = {
            "energy_rmse": rmse(de),
            "energy_mae": mae(de),
            "force_rmse": None,
            "force_mae": None,
            "virial_rmse": None,
        }
        details = {"energy": {"reference": labels.energy_per_atom.tolist(), "predicted": energy.tolist()}}
        if labels.forces is not None:
            df = (np.asarray(prediction.forces, dtype=float) - labels.forces) * 1000
            metrics["force_rmse"] = rmse(df)
            metrics["force_mae"] = mae(df)
        if labels.virial_per_atom is not None:
            predicted_virial = np.asarray(prediction.structure_virials, dtype=float).reshape(-1, 9)
            metrics["virial_rmse"] = rmse((predicted_virial - labels.virial_per_atom) * 1000)
        return Evaluation(metrics=metrics, details=details)
