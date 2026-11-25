"""Turn one EOS validation group into an editable, unlabelled strain scan.

Recommendations localise curve disagreement; they do not diagnose training coverage.
The existing CellStrain operation owns the geometry transformation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from NepTrainKit.core.cards.lattice import CellStrainOperation, CellStrainParams
from NepTrainKit.core.cards.operation import params_to_dict

from .protocols import get_protocol


@dataclass(frozen=True)
class EOSSamplingParams:
    start_volume: float
    stop_volume: float
    count: int = 5


def clean_sampling_parent(atoms: Atoms, provenance: dict) -> Atoms:
    """Allow only geometry and magnetic inputs, never copied calculation labels."""
    clean = Atoms(
        numbers=atoms.numbers.copy(), positions=atoms.positions.copy(), cell=atoms.cell.copy(), pbc=atoms.pbc.copy()
    )
    for key in ("initial_magmoms", "spin", "spins", "spin_active"):
        if key in atoms.arrays:
            clean.set_array(key, atoms.arrays[key].copy())
    clean.info.update(
        Config_type="EOS_sampling",
        sampling_status="needs_dft",
        sampling_source=dict(provenance),
        eos_group=provenance.get("group") or atoms.info.get("eos_group", "EOS_sampling"),
    )
    return clean


def strain_params(parent: Atoms, params: EOSSamplingParams) -> CellStrainParams:
    values = (params.start_volume, params.stop_volume)
    if not np.isfinite(values).all() or not 0 < values[0] < values[1]:
        raise ValueError("Sampling volumes must be finite, positive and increasing.")
    if isinstance(params.count, bool) or int(params.count) != params.count or not 2 <= params.count <= 100:
        raise ValueError("Choose between 2 and 100 sampling points.")
    if not len(parent) or not parent.pbc.all() or parent.get_volume() <= 0:
        raise ValueError("EOS sampling requires a periodic bulk parent.")
    volume = parent.get_volume() / len(parent)
    low, high = 100 * (np.cbrt(np.array(values) / volume) - 1)
    low, high, step = (round(float(value), 6) for value in (low, high, (high - low) / (params.count - 1)))
    high = round(low + step * (params.count - 1), 6)
    if low < -99 or high > 100 or step < 1e-6 or step > 199:
        raise ValueError("Sampling interval exceeds the Lattice Strain card range or precision.")
    # Match the editable card's six-decimal engineering-strain precision exactly.
    return CellStrainParams(axes="isotropic", x_range=(low, high, step))


def sampling_workflow(parent: Atoms, params: EOSSamplingParams) -> dict:
    return {
        "workflow_schema": 2,
        "cards": [
            {"class": "CellStrainCard", "check_state": True, "params": params_to_dict(strain_params(parent, params))}
        ],
    }


def sampling_volumes(parent: Atoms, params: CellStrainParams) -> np.ndarray:
    """Preview the existing card's strain grid without generating any structures."""
    if params.axes != "isotropic" or params.identify_organic:
        raise ValueError("EOS preview requires isotropic strain with molecular handling disabled.")
    summary = CellStrainOperation.sampling_summary(params)
    if not 2 <= summary["outputs_per_input"] <= 100:
        raise ValueError("Choose between 2 and 100 sampling points.")
    strains = summary["ranges"][0]
    return parent.get_volume() / len(parent) * (1 + strains / 100) ** 3


def sampling_payload(frames, benchmark, details=None, model=None, run=None) -> dict:
    """Prepare runtime context and a standard card configuration for a new draft."""
    suggestion = suggest_sampling(frames, details)
    parent = clean_sampling_parent(
        frames[suggestion["parent_index"]],
        {
            "set_id": benchmark["id"],
            "set_hash": benchmark["object_hash"],
            "group": benchmark["name"],
            "parent_index": suggestion["parent_index"],
            "model_hash": (model or {}).get("content_hash"),
            "run_id": (run or {}).get("id"),
            "reason": suggestion["reason"],
        },
    )
    return {
        "name": benchmark["name"],
        "structures": [parent],
        "workflow": sampling_workflow(parent, suggestion["params"]),
        "eos_context": {"benchmark": benchmark, "details": details, "suggestion": suggestion, "references": frames},
    }


def reference_overlaps(candidates, references) -> list[bool]:
    """Detect same ordered geometry and magnetic inputs, allowing PBC wrapping."""
    result = []
    for candidate in candidates:
        overlap = False
        for reference in references:
            if not np.array_equal(candidate.numbers, reference.numbers):
                continue
            if not np.allclose(candidate.cell, reference.cell, rtol=0, atol=1e-5):
                continue
            delta = candidate.get_scaled_positions() - reference.get_scaled_positions()
            if not np.allclose(delta - np.rint(delta), 0, rtol=0, atol=1e-6):
                continue
            if any(
                not np.array_equal(candidate.arrays.get(key), reference.arrays.get(key))
                for key in ("initial_magmoms", "spin", "spins")
            ):
                continue
            overlap = True
            break
        result.append(overlap)
    return result


def suggest_sampling(frames, details: dict | None = None) -> dict:
    """Suggest the neighbouring interval of the largest offset-free residual.

    Endpoints are inset from reference nodes to avoid recommending their exact copies.
    Missing/flat predictions and incomplete reference minima get explicit reason codes.
    """
    summary = get_protocol("eos").summarize_reference(frames)
    volumes = np.asarray(summary["volumes"])
    reference = np.asarray(summary["energies"])
    parent_volume = volumes[np.argmin(reference)]
    parent_index = int(np.argmin([abs(atoms.get_volume() / len(atoms) - parent_volume) for atoms in frames]))
    start, stop = float(volumes[0]), float(volumes[-1])
    reason, peak = "manual", None
    residual = None
    if details:
        predicted = np.asarray(details.get("predicted", []), dtype=float)
        run_volumes = np.asarray(details.get("volumes", []), dtype=float)
        run_reference = np.asarray(details.get("reference", []), dtype=float)
        if (
            predicted.shape != volumes.shape
            or run_volumes.shape != volumes.shape
            or run_reference.shape != reference.shape
            or not np.isfinite(predicted).all()
            or not np.allclose(run_volumes, volumes)
            or not np.allclose(run_reference, reference)
        ):
            raise ValueError("Validation curve does not match this reference group.")
        residual = predicted - reference
        residual -= np.mean(residual)
        peak = float(np.max(np.abs(residual)) * 1000)
        if peak <= 1e-3:
            reason = "offset_only"
        else:
            index = int(np.argmax(np.abs(residual)))
            start, stop = float(volumes[max(0, index - 1)]), float(volumes[min(len(volumes) - 1, index + 1)])
            reason = "shape_error"
    if int(np.argmin(reference)) in (0, len(reference) - 1):
        reason = "reference_incomplete"
        start, stop = float(volumes[0]), float(volumes[-1])
    # The scan samples lattice scale, not volume, uniformly, as the card does.
    low, high = np.cbrt([start, stop])
    for fraction in (0.1, 0.13, 0.17, 0.23):
        margin = (high - low) * fraction
        params = EOSSamplingParams(float((low + margin) ** 3), float((high - margin) ** 3))
        trial = np.linspace(low + margin, high - margin, params.count) ** 3
        if not np.any(np.isclose(trial[:, None], volumes[None, :], rtol=0, atol=1e-4)):
            break
    return dict(
        summary=summary, parent_index=parent_index, params=params, reason=reason, peak_mev=peak, residual=residual
    )


def magnetic_sampling_payload(frames, benchmark, branch, model=None, run=None):
    """Seed an editable scan from the measured branch, retaining its fixed coordinate."""
    from NepTrainKit.core.magnetic_response import LocalMagneticResponseParams
    from NepTrainKit.core.magnetism import existing_moment_vectors

    key, label, indices = branch
    summary = benchmark["summary"]
    rows = [summary["rows"][i] for i in indices]
    axis = summary["analysis"]["x_field"]
    candidates = [r for r in rows if r["x"] != 0] if axis == "moment_ratio" else rows
    nearest = min(candidates or rows, key=lambda r: abs(r["x"] - 1.0))
    index = nearest["terms"][0][0]
    parent = clean_sampling_parent(frames[index], {
        "set_id": benchmark["id"], "set_hash": benchmark["object_hash"], "group": label,
        "branch": key, "parent_index": index, "x_field": axis, "parent_coordinate": nearest["x"],
        "second_axis": summary["analysis"]["second_axis"], "fixed_coordinate": nearest["second_axis"],
        "model_hash": (model or {}).get("content_hash"), "run_id": (run or {}).get("id"),
    })
    parent.info["Config_type"] = "mEOS_sampling"
    parents = [frames[r["terms"][0][0]] for r in rows]
    if axis == "volume_ratio":
        volumes = [frame.get_volume() / len(frame) for frame in parents]
        low, high = min(volumes), max(volumes)
        if low == high:
            low, high = low * 0.99, high * 1.01
        margin = (high - low) * 0.1
        workflow = sampling_workflow(parent, EOSSamplingParams(low + margin, high - margin))
    else:
        spins = existing_moment_vectors(parent)
        if spins is None or not np.any(np.linalg.norm(spins, axis=1) > 1e-10):
            raise ValueError("Moment sampling requires non-zero input magnetic moments.")
        magnitudes = np.linalg.norm(spins, axis=1)
        targets = magnitudes > 1e-10
        if len(parents) > 1:
            varies = np.zeros(len(parent), dtype=bool)
            for frame in parents:
                moments = existing_moment_vectors(frame)
                if moments is None or not np.array_equal(frame.numbers, parent.numbers):
                    raise ValueError("Moment sampling needs consistent magnetic sites within the branch.")
                varies |= ~np.isclose(np.linalg.norm(moments, axis=1), magnitudes)
            targets &= varies
        if not np.any(targets):
            raise ValueError("No varying magnetic moments in this scan branch.")
        params = LocalMagneticResponseParams(response_kind="Moment magnitude", target_mode="Explicit indices",
                                             target_indices=",".join(str(i + 1) for i in np.flatnonzero(targets)),
                                             moment_scale_scan="0.98,1,1.02")
        workflow = {"workflow_schema": 2, "cards": [
            {"class": "LocalMagneticResponseCard", "check_state": True, "params": params_to_dict(params)}]}
    return {"name": f"{benchmark['name']} · {label}", "kind": "mEOS", "structures": [parent], "workflow": workflow}
