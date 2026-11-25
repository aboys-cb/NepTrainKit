"""Explicit unit-spin bilinear projections from complete, ordered four-state groups."""

import numpy as np
from NepTrainKit.core.magnetism import existing_moment_vectors


def _pair(frame, spec):
    values = [frame.info[key] for key in spec.pair_fields]
    indices = tuple(int(value) for value in values)
    if any(isinstance(value, (bool, np.bool_)) or float(value) != index for value, index in zip(values, indices)):
        raise ValueError("Atom-pair indices must be integers.")
    return tuple(index - spec.index_base for index in indices)


def exchange_rows(frames, rows, spec):
    result = []
    for number, row in enumerate(rows):
        indices = [int(i) for i, _ in row["terms"]]
        group = [frames[i] for i in indices]
        pair = _pair(group[0], spec)
        if (
            pair[0] == pair[1]
            or any(not 0 <= site < len(group[0]) for site in pair)
            or any(_pair(a, spec) != pair for a in group)
        ):
            raise ValueError("Four-state exchange needs one consistent ordered pair of distinct atom indices.")
        moments = [existing_moment_vectors(a, lift_scalar=False) for a in group]
        if any(m is None or not np.isfinite(m).all() for m in moments):
            raise ValueError("Four-state exchange requires explicit finite vector moments.")
        norms = [np.linalg.norm(m, axis=1) for m in moments]
        if any(not np.allclose(n, norms[0], atol=1e-7, rtol=0) for n in norms[1:]) or any(
            norms[0][site] < 1e-10 for site in pair
        ):
            raise ValueError("Unit-spin exchange requires constant nonzero moment lengths on the selected pair.")
        others = [i for i in range(len(group[0])) if i not in pair]
        if any(not np.allclose(m[others], moments[0][others], rtol=0, atol=1e-7) for m in moments[1:]):
            raise ValueError("Four-state exchange requires all other input moments to remain fixed.")
        states = {tuple(str(int(a.info[key])) for key in spec.state_fields): m for a, m in zip(group, moments)}
        pp, mm, pm, mp = (states[key] for key in (("1", "1"), ("-1", "-1"), ("1", "-1"), ("-1", "1")))
        i, j = pair
        if (
            not np.allclose(pp[i], pm[i], atol=1e-7, rtol=0)
            or not np.allclose(mp[i], mm[i], atol=1e-7, rtol=0)
            or not np.allclose(pp[j], mp[j], atol=1e-7, rtol=0)
            or not np.allclose(pm[j], mm[j], atol=1e-7, rtol=0)
        ):
            raise ValueError("The two four-state perturbations must be independently controlled.")
        directions = [(pp[site] - mm[site]) / norms[0][site] for site in pair]
        lengths = [np.linalg.norm(v) for v in directions]
        if min(lengths) < 1e-10:
            raise ValueError("Four-state exchange needs nonzero perturbations on both atoms.")
        sign = 1
        for k, vector in enumerate(directions):
            vector = vector / lengths[k]
            if vector[np.argmax(np.abs(vector))] < 0:
                vector = -vector
                sign *= -1
            directions[k] = vector
        metadata = group[0].info
        parent = tuple((key, str(metadata[key])) for key in spec.group_fields if key not in spec.axis_fields)
        result.append(
            dict(
                row=number,
                ordered_pair=list(pair),
                parent=parent,
                direction_i=directions[0].tolist(),
                direction_j=directions[1].tolist(),
                factor=float(sign / (lengths[0] * lengths[1] * spec.multiplicity)),
            )
        )
    return result


def evaluate_exchange(projections, reference, predicted=None):
    rows = []
    for item in projections:
        row = dict(item, reference=float(reference[item["row"]] * item["factor"]))
        if predicted is not None:
            row.update(predicted=float(predicted[item["row"]] * item["factor"]))
            row["error"] = row["predicted"] - row["reference"]
        rows.append(row)
    pairs = []
    for i, item in enumerate(rows):
        u, v = np.array(item["direction_i"]), np.array(item["direction_j"])
        normal = np.cross(u, v)
        length = np.linalg.norm(normal)
        if length < 1e-7:
            continue
        candidates = [
            other
            for other in rows[i + 1 :]
            if other["parent"] == item["parent"]
            and other["ordered_pair"] == item["ordered_pair"]
            and np.allclose(other["direction_i"], v, atol=1e-7, rtol=0)
            and np.allclose(other["direction_j"], u, atol=1e-7, rtol=0)
        ]
        if len(candidates) != 1:
            continue
        other = candidates[0]
        result = dict(parent=item["parent"], ordered_pair=item["ordered_pair"], direction=(normal / length).tolist())
        for field in ("reference", "predicted") if predicted is not None else ("reference",):
            result[field + "_symmetric"] = (item[field] + other[field]) / 2
            result[field + "_d_projection"] = (item[field] - other[field]) / (2 * length)
        pairs.append(result)
    return dict(
        convention="H = sum e_i^T J_ij e_j; unit spins; ordered pair; positive bilinear sign",
        projections=rows,
        symmetric_chiral=pairs,
    )
