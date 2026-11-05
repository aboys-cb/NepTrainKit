"""Fixed-geometry equation of state on one phase / magnetic branch.

Energies are compared per atom against volume per atom. Shape errors remove ONE
least-squares constant offset per curve; aligned curves must never be used to
rank different phases (a multi-phase protocol with a shared offset will).
"""

from __future__ import annotations

import warnings

import numpy as np
from ase.eos import EquationOfState
from ase.units import GPa

from .base import Evaluation, Issue, Metric, ValidationProtocol, reference_labels, unlabelled_copies

MIN_POINTS = 5


def fit_eos(volumes, energies) -> dict:
    """Fit only bracketed minima; never report an extrapolated equilibrium."""
    volumes = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if int(np.argmin(energies)) in (0, len(energies) - 1):
        return {"reason": "Minimum outside the sampled interior; fit unavailable."}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v0, e0, modulus = EquationOfState(volumes, energies, eos="birchmurnaghan").fit()
        if not np.isfinite([v0, e0, modulus]).all() or modulus <= 0 or not volumes[0] < v0 < volumes[-1]:
            raise ValueError("Unphysical or unbracketed fit")
        return {"v0": float(v0), "e0": float(e0), "b0_gpa": float(modulus / GPa)}
    except Exception as error:
        return {"reason": str(error)}


def summarize_curve(volumes, reference, predicted) -> dict:
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if predicted.shape != reference.shape or not np.isfinite(predicted).all():
        raise ValueError("Prediction must contain one finite energy per reference frame.")
    delta = predicted - reference
    offset = float(np.mean(delta))
    centered = delta - offset
    return {
        "offset_ev_atom": offset,
        "raw_rmse_mev": float(np.sqrt(np.mean(delta**2)) * 1000),
        "shape_rmse_mev": float(np.sqrt(np.mean(centered**2)) * 1000),
        "max_shape_error_mev": float(np.max(np.abs(centered)) * 1000),
        "fit": fit_eos(volumes, predicted),
    }


def _volume_scan(frames):
    volumes = np.asarray([atoms.get_volume() / len(atoms) for atoms in frames])
    order = np.argsort(volumes)
    return order, volumes[order]


class EquationOfStateProtocol(ValidationProtocol):
    kind = "eos"
    version = 1
    title = "Equation of state"
    metrics = (
        Metric("shape_rmse", "Curve RMSE (offset removed)", "meV/atom", headline=True),
        Metric("max_shape_error", "Max curve error", "meV/atom"),
        Metric("raw_rmse", "Raw RMSE", "meV/atom"),
        # Floors sit at the precision of a Birch-Murnaghan fit on a 5-10 point
        # scan: V₀ is pinned to ~0.1 %, while B₀ (a curvature) wobbles by ~1 %.
        Metric("v0_error", "V₀ error", "%", better="closer-to-zero", headline=True, abs_tolerance=0.1),
        Metric("b0_error", "B₀ error", "%", better="closer-to-zero", headline=True, abs_tolerance=1.0),
    )

    def detect(self, frames) -> float:
        if len(frames) < MIN_POINTS or self._structural_errors(frames):
            return 0.0
        # A uniform scan keeps fractional coordinates and cell shape fixed.
        first = frames[0]
        shape = first.cell.array / np.cbrt(abs(np.linalg.det(first.cell.array)))
        for atoms in frames[1:]:
            scaled = atoms.cell.array / np.cbrt(abs(np.linalg.det(atoms.cell.array)))
            if not np.allclose(scaled, shape, atol=1e-3):
                return 0.4
        return 0.9

    @staticmethod
    def _structural_errors(frames) -> list[str]:
        errors = []
        if len(frames) < MIN_POINTS:
            errors.append(f"EOS needs at least {MIN_POINTS} labelled volume points.")
            return errors
        symbols = frames[0].get_chemical_symbols()
        for atoms in frames:
            if not len(atoms) or atoms.get_chemical_symbols() != symbols:
                errors.append("EOS frames must have identical atom counts and species order.")
                break
            if not atoms.pbc.all() or not np.isfinite(atoms.cell.array).all() or not np.isfinite(atoms.positions).all():
                errors.append("EOS requires finite, fully periodic bulk structures.")
                break
        if not errors:
            _, volumes = _volume_scan(frames)
            if np.any(volumes <= 0):
                errors.append("EOS volume must be positive.")
            elif np.any(np.diff(volumes) < 1e-6):
                errors.append("Duplicate volumes: import each phase / magnetic branch as a separate EOS set.")
        return errors

    def check(self, frames) -> list[Issue]:
        issues = [Issue("error", message) for message in self._structural_errors(frames)]
        if issues:
            return issues
        try:
            energies = reference_labels(frames).energy_per_atom
        except ValueError as error:
            return [Issue("error", str(error))]
        if not np.isfinite(energies).all():
            return [Issue("error", "Reference energy is not finite.")]
        order, volumes = _volume_scan(frames)
        if "v0" not in fit_eos(volumes, energies[order]):
            issues.append(Issue("warning", "Reference minimum is not bracketed; V₀/B₀ will not be compared."))
        return issues

    def summarize_reference(self, frames) -> dict:
        order, volumes = _volume_scan(frames)
        energies = reference_labels(frames).energy_per_atom[order]
        return {
            "formula": frames[0].get_chemical_formula(),
            "volumes": volumes.tolist(),
            "energies": energies.tolist(),
            "fit": fit_eos(volumes, energies),
        }

    def evaluate(self, predictor, frames) -> Evaluation:
        order, volumes = _volume_scan(frames)
        labels = reference_labels(frames)
        reference = labels.energy_per_atom[order]
        prediction = predictor.predict(unlabelled_copies(frames))
        predicted = (np.asarray(prediction.energy, dtype=float) / labels.atom_counts)[order]
        curve = summarize_curve(volumes, reference, predicted)
        reference_fit = fit_eos(volumes, reference)
        model_fit = curve["fit"]
        metrics: dict[str, float | None] = {
            "shape_rmse": curve["shape_rmse_mev"],
            "max_shape_error": curve["max_shape_error_mev"],
            "raw_rmse": curve["raw_rmse_mev"],
            "v0_error": None,
            "b0_error": None,
        }
        warnings_ = []
        if "v0" in reference_fit and "v0" in model_fit:
            metrics["v0_error"] = (model_fit["v0"] / reference_fit["v0"] - 1) * 100
            metrics["b0_error"] = (model_fit["b0_gpa"] / reference_fit["b0_gpa"] - 1) * 100
        elif "v0" in reference_fit:
            warnings_.append(model_fit.get("reason", "Model fit unavailable."))
        details = {
            "volumes": volumes.tolist(),
            "reference": reference.tolist(),
            "predicted": predicted.tolist(),
            "offset_ev_atom": curve["offset_ev_atom"],
            "reference_fit": reference_fit,
            "model_fit": model_fit,
        }
        return Evaluation(metrics=metrics, details=details, warnings=warnings_)
