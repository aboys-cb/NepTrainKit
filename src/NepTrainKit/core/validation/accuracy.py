"""Reference masks and error statistics shared by calculated and cached predictions."""

from dataclasses import dataclass

import numpy as np

from .protocols.base import _energy, _forces, _virial_per_atom


VIRIAL_COMPONENTS = [0, 4, 8, 1, 5, 6]  # NEP xx, yy, zz, xy, yz, zx
QUANTITIES = ("energy", "force", "virial", "mforce")


@dataclass
class AccuracyLabels:
    atom_counts: np.ndarray
    values: dict
    masks: dict
    counts: dict


def accuracy_labels(frames) -> AccuracyLabels:
    """Missing properties are masked; malformed or non-finite supplied labels fail."""
    counts = np.asarray([len(frame) for frame in frames], dtype=int)
    if not len(counts) or np.any(counts <= 0):
        raise ValueError("Accuracy evaluation needs non-empty structures.")
    sizes = {"energy": (len(frames),), "force": (int(counts.sum()), 3),
             "virial": (len(frames), 6), "mforce": (int(counts.sum()), 3)}
    values = {key: np.full(shape, np.nan) for key, shape in sizes.items()}
    masks = {key: np.zeros(shape[0], dtype=bool) for key, shape in sizes.items()}
    labelled_frames = {key: 0 for key in QUANTITIES}
    offset = 0
    for index, atoms in enumerate(frames):
        results = getattr(atoms.calc, "results", {})
        energy = _energy(atoms) / len(atoms) if "energy" in atoms.info or "energy" in results else None
        forces = _forces(atoms)
        virial = _virial_per_atom(atoms)
        if virial is not None:
            if not np.isfinite(virial).all():
                raise ValueError("Reference virials must be finite.")
            virial = virial[VIRIAL_COMPONENTS]
        magnetic = [np.asarray(atoms.arrays[key], dtype=float)
                    for key in ("force_mag", "mforce") if key in atoms.arrays]
        if len(magnetic) == 2 and not np.array_equal(magnetic[0], magnetic[1]):
            raise ValueError("force_mag and mforce reference labels disagree.")
        mforces = magnetic[0] if magnetic else None
        for key, block in (("energy", energy), ("force", forces), ("virial", virial), ("mforce", mforces)):
            if block is None:
                continue
            block = np.asarray(block, dtype=float)
            expected = () if key == "energy" else (6,) if key == "virial" else (len(atoms), 3)
            if block.shape != expected:
                raise ValueError(f"Reference {key} labels have shape {block.shape}; expected {expected}.")
            if not np.isfinite(block).all():
                raise ValueError(f"Reference {key} labels must be finite.")
            selection = index if key in ("energy", "virial") else slice(offset, offset + len(atoms))
            values[key][selection] = block
            masks[key][selection] = True
            labelled_frames[key] += 1
        offset += len(atoms)
    label_counts = {}
    for key in QUANTITIES:
        rows = int(masks[key].sum())
        label_counts[key] = {
            "structures": labelled_frames[key],
            "atoms": int(counts[masks[key]].sum()) if key in ("energy", "virial") else rows,
            "components": rows * (1 if key == "energy" else sizes[key][1]),
            "missing_structures": len(frames) - labelled_frames[key],
        }
    return AccuracyLabels(counts, values, masks, label_counts)


def prediction_values(prediction, labels: AccuracyLabels) -> dict:
    """Convert NEPAdapters totals and tensors to the official comparison units."""
    n = len(labels.atom_counts)
    atoms = int(labels.atom_counts.sum())
    energy = np.asarray(prediction.energy, dtype=float)
    forces = np.asarray(prediction.forces, dtype=float)
    virials = np.asarray(prediction.structure_virials, dtype=float)
    if energy.shape != (n,) or forces.shape != (atoms, 3) or virials.shape not in ((n, 9), (n, 3, 3)):
        raise ValueError("Prediction shapes do not match the reference structures.")
    if not all(np.isfinite(value).all() for value in (energy, forces, virials)):
        raise ValueError("Predicted energies, forces and virials must be finite.")
    values = {"energy": energy / labels.atom_counts, "force": forces,
              "virial": virials.reshape(n, 9)[:, VIRIAL_COMPONENTS]}
    if getattr(prediction, "mforces", None) is not None:
        mforces = np.asarray(prediction.mforces, dtype=float)
        if mforces.shape != (atoms, 3) or not np.isfinite(mforces).all():
            raise ValueError("Predicted magnetic forces must be finite and have shape (atoms, 3).")
        values["mforce"] = mforces
    return values


def accuracy_statistics(labels: AccuracyLabels, predicted: dict) -> tuple[dict, dict]:
    """Compute component RMSE/MAE only where a reference label exists."""
    metrics, details = {}, {"label_counts": {}, "virial_components": ["xx", "yy", "zz", "xy", "yz", "zx"]}
    for key in QUANTITIES:
        mask = labels.masks[key]
        reference = labels.values[key][mask]
        available = predicted.get(key)
        count = dict(labels.counts[key], evaluated_components=0)
        metrics[f"{key}_rmse"] = None
        metrics[f"{key}_mae"] = None
        if available is not None:
            available = np.asarray(available, dtype=float)
            if available.shape != labels.values[key].shape or not np.isfinite(available).all():
                raise ValueError(f"Predicted {key} values must be finite and match the reference shape.")
            if reference.size:
                prediction = available[mask]
                delta = (prediction - reference) * 1000
                metrics[f"{key}_rmse"] = float(np.sqrt(np.mean(delta ** 2)))
                metrics[f"{key}_mae"] = float(np.mean(np.abs(delta)))
                count["evaluated_components"] = int(reference.size)
                if key == "energy":
                    details[key] = {"reference": reference.tolist(), "predicted": prediction.tolist(),
                                    "frame_indices": np.flatnonzero(mask).tolist()}
        details["label_counts"][key] = count
    if not any(value is not None for value in metrics.values()):
        raise ValueError("The model cannot evaluate any labelled quantity in this dataset.")
    return metrics, details
