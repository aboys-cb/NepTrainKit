"""Energy, force, virial and magnetic-force errors with independent label masks."""

from .base import Evaluation, Issue, Metric, ValidationProtocol, unlabelled_copies


class EnergyForceVirialProtocol(ValidationProtocol):
    kind = "efv"
    version = 2
    title = "Energy / force / virial"
    supports_spin = True
    metrics = (
        Metric("energy_rmse", "Energy RMSE", "meV/atom", headline=True),
        Metric("force_rmse", "Force RMSE", "meV/Å", headline=True),
        Metric("mforce_rmse", "Magnetic force RMSE", "meV/μB", headline=True),
        Metric("virial_rmse", "Virial RMSE", "meV/atom"),
        Metric("energy_mae", "Energy MAE", "meV/atom"),
        Metric("force_mae", "Force MAE", "meV/Å"),
        Metric("virial_mae", "Virial MAE", "meV/atom"),
        Metric("mforce_mae", "Magnetic force MAE", "meV/μB"),
    )

    def detect(self, frames):
        return 0.1 if frames else 0.0

    def check(self, frames):
        from ..accuracy import accuracy_labels
        if not frames:
            return [Issue("error", "The file contains no structures.")]
        try:
            labels = accuracy_labels(frames)
        except (ValueError, TypeError) as error:
            return [Issue("error", str(error))]
        if not any(count["components"] for count in labels.counts.values()):
            return [Issue("error", "At least one energy, force, virial or magnetic-force label is required.")]
        return []

    def summarize_reference(self, frames):
        from ..accuracy import accuracy_labels
        labels = accuracy_labels(frames)
        return {"frames": len(frames), "atoms": int(labels.atom_counts.sum()),
                "has_forces": bool(labels.masks["force"].any()),
                "has_virials": bool(labels.masks["virial"].any()), "label_counts": labels.counts}

    def evaluate(self, predictor, frames):
        from ..accuracy import accuracy_labels, accuracy_statistics, prediction_values
        labels = accuracy_labels(frames)
        prediction = predictor.predict(unlabelled_copies(frames))
        metrics, details = accuracy_statistics(labels, prediction_values(prediction, labels))
        details["prediction_source"] = {"kind": "recalculated"}
        warnings = []
        if labels.counts["mforce"]["components"] and getattr(prediction, "mforces", None) is None:
            warnings.append("Magnetic-force labels exist, but this model does not provide magnetic forces.")
        return Evaluation(metrics=metrics, details=details, warnings=warnings)
