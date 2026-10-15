"""Offline, selector-independent coverage audit; see sampling_evaluation.md.

NPZ keys: scalar__NAME (N,), view__NAME (N,D), optional group (N,).
View coordinates must be frozen/scaled before comparing selections. JSON maps
method names to zero-based source row indices. No sampling code is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def validate_indices(indices, count):
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in "iu" or not len(values):
        raise ValueError("Selection must be a nonempty integer vector")
    if values.min() < 0 or values.max() >= count or len(np.unique(values)) != len(values):
        raise ValueError("Selection contains duplicate or out-of-range indices")
    return values


def scalar_report(values, indices):
    """Report extent, gaps and signed tails separately; missing is not zero.

    Distance uses the full finite population span, so it is unit invariant.
    Retained range alone cannot reveal an empty interior. Quantile tails use
    inclusive ties and their actual population count, not an assumed 1%/5%.
    """
    values = np.asarray(values, dtype=float)
    full = np.sort(values[np.isfinite(values)])
    chosen = np.sort(values[indices][np.isfinite(values[indices])])
    result = {"valid": len(full), "selected_valid": len(chosen),
              "missing": int(len(values) - len(full))}
    if not len(full):
        return result
    span = float(full[-1] - full[0])
    result["full_range"] = [float(full[0]), float(full[-1])]
    result["selected_range"] = chosen[[0, -1]].tolist() if len(chosen) else None
    if not len(chosen):
        result.update(range_fraction=0.0, gap_p95=None, gap_max=None)
    elif span == 0:
        result.update(range_fraction=1.0, gap_p95=0.0, gap_max=0.0)
    else:
        pos = np.searchsorted(chosen, full)
        gaps = np.minimum(abs(full - chosen[np.maximum(pos - 1, 0)]),
                          abs(full - chosen[np.minimum(pos, len(chosen) - 1)])) / span
        result.update(range_fraction=float(np.ptp(chosen) / span),
                      gap_p95=float(np.quantile(gaps, .95)), gap_max=float(gaps.max()))
    tails = {}
    if span > 0:
        for quantile in (.01, .05, .95, .99):
            threshold = float(np.quantile(full, quantile))
            pop = full <= threshold if quantile < .5 else full >= threshold
            sel = chosen <= threshold if quantile < .5 else chosen >= threshold
            expected = len(chosen) * int(pop.sum()) / len(full)
            tails[str(quantile)] = {
                "threshold": threshold, "population": int(pop.sum()),
                "selected": int(sel.sum()), "uniform_expected": expected,
                "relative_to_uniform": float(sel.sum() / expected) if expected else None,
            }
    result["tails"] = tails
    return result


def evaluate(observations, selections):
    """Global coverage may use any selected equivalent state, across sources.

    Within-source scalar retention is a diagnostic, never a sampling quota.
    Multidimensional view gaps expose missing combinations that marginals hide.
    """
    arrays = {k: np.asarray(v) for k, v in observations.items()}
    features = {k: v for k, v in arrays.items() if k.startswith(("scalar__", "view__"))}
    if not features:
        raise ValueError("No scalar__ or view__ observations")
    count = len(next(iter(features.values())))
    if any(len(v) != count for v in arrays.values()):
        raise ValueError("All observations must share source row order and length")
    for key, values in features.items():
        if key.startswith("scalar__") and values.ndim != 1:
            raise ValueError(f"{key} must have shape (N,)")
        if key.startswith("view__") and (values.ndim != 2 or values.shape[1] == 0
                                         or not np.isfinite(values).all()):
            raise ValueError(f"{key} must be a finite (N,D) matrix")
    groups = arrays.get("group", np.full(count, "all"))
    if groups.ndim != 1:
        raise ValueError("group must have shape (N,)")
    scopes = {str(g): np.flatnonzero(groups == g) for g in np.unique(groups)}
    output = {}
    for name, selection in selections.items():
        ids = validate_indices(selection, count)
        row = {"count": len(ids), "scalars": {}, "views": {}, "groups": {}}
        selected = np.zeros(count, dtype=bool)
        selected[ids] = True
        for key, values in features.items():
            if key.startswith("scalar__"):
                row["scalars"][key[8:]] = scalar_report(values, ids)
            else:
                distances = cKDTree(values[ids]).query(values, workers=1)[0]
                row["views"][key[6:]] = {
                    "p95": float(np.quantile(distances, .95)),
                    "p99": float(np.quantile(distances, .99)),
                    "max": float(distances.max()),
                    "groups": {g: {"p95": float(np.quantile(distances[q], .95)),
                                    "max": float(distances[q].max())}
                               for g, q in scopes.items()},
                }
        for group, q in scopes.items():
            local = np.flatnonzero(selected[q])
            row["groups"][group] = {
                "population": len(q), "selected": len(local),
                "within_source_scalars": {
                    key[8:]: scalar_report(values[q], local)
                    for key, values in features.items() if key.startswith("scalar__")},
            }
        output[name] = row
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("selections", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--random-seeds", type=int, default=20)
    args = parser.parse_args()
    if args.random_seeds < 0:
        parser.error("--random-seeds must be nonnegative")
    with np.load(args.observations, allow_pickle=False) as archive:
        observations = dict(archive)
    selections = json.loads(args.selections.read_text())
    if not selections or len({len(v) for v in selections.values()}) != 1:
        parser.error("Use nonempty selections with the same budget")
    count = len(next(iter(observations.values())))
    budget = len(next(iter(selections.values())))
    for ids in selections.values():
        validate_indices(ids, count)
    for seed in range(args.random_seeds):
        name = f"uniform_seed_{seed}"
        if name in selections:
            parser.error(f"Reserved control name: {name}")
        selections[name] = np.random.default_rng(seed).choice(count, budget, replace=False)
    result = {
        "schema_version": 1, "population": count, "budget": budget,
        "provenance": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (args.observations, args.selections, Path(__file__))},
        "random_seeds": list(range(args.random_seeds)),
        "results": evaluate(observations, selections),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
