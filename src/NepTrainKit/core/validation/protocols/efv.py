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
        # Check present labels even when another frame lacks that quantity.
        per_frame = [reference_labels([frame]) for frame in frames]
        if any(item.forces is not None and not np.isfinite(item.forces).all() for item in per_frame):
            issues.append(Issue("error", "Reference forces must be finite."))
        if any(item.virial_per_atom is not None and not np.isfinite(item.virial_per_atom).all() for item in per_frame):
            issues.append(Issue("error", "Reference virials must be finite."))
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
        errors = [issue.message for issue in self.check(frames) if issue.level == "error"]
        if errors:
            raise ValueError("; ".join(errors))
        labels = reference_labels(frames)
        prediction = predictor.predict(unlabelled_copies(frames))
        energy = np.asarray(prediction.energy, dtype=float)
        forces = np.asarray(prediction.forces, dtype=float)
        virials = np.asarray(prediction.structure_virials, dtype=float).reshape(-1, 9)
        if (energy.shape != labels.energy_per_atom.shape or
                forces.shape != (int(labels.atom_counts.sum()), 3) or virials.shape != (len(frames), 9)):
            raise ValueError("Prediction shapes do not match the reference structures.")
        if not all(np.isfinite(values).all() for values in (energy, forces, virials)):
            raise ValueError("Predicted energies, forces and virials must be finite.")
        energy = energy / labels.atom_counts
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
            df = (forces - labels.forces) * 1000
            metrics["force_rmse"] = rmse(df)
            metrics["force_mae"] = mae(df)
        if labels.virial_per_atom is not None:
            metrics["virial_rmse"] = rmse((virials - labels.virial_per_atom) * 1000)
        return Evaluation(metrics=metrics, details=details)
