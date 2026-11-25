"""Recognise collinear collective spin rotations without changing input frames."""

from itertools import combinations

import numpy as np

from NepTrainKit.core.magnetism import existing_moment_vectors
from .grouping import _scalar


def spin_direction(frame):
    """Return a direction and sublattice pattern when all active spins are collinear."""
    moments = existing_moment_vectors(frame, lift_scalar=False)
    if moments is None or not np.isfinite(moments).all():
        return None
    norms = np.linalg.norm(moments, axis=1)
    active = norms > 1e-8
    if not active.any():
        return None
    unit = moments[active] / norms[active, None]
    direction = unit[0]
    signs = np.where(unit @ direction >= 0, 1, -1)
    direction = np.mean(unit * signs[:, None], axis=0)
    direction /= np.linalg.norm(direction)
    if np.max(np.linalg.norm(unit - signs[:, None] * direction, axis=1)) > 1e-3:
        return None
    pattern = np.zeros(len(frame), dtype=int)
    pattern[active] = signs
    return direction, pattern, norms


def rotation_metadata(frames, group_fields=()):
    """Persist group-local state numbers; state 0 is the first frame of that group."""
    groups = []
    metadata = {}
    for index, frame in enumerate(frames):
        spin = spin_direction(frame)
        if spin is None:
            return {}
        direction, pattern, norms = spin
        declared = tuple(_scalar(frame.info.get(name)) for name in group_fields)
        if any(value is None for value in declared):
            return {}
        group = next((g for g in groups if g["declared"] == declared and
                      np.array_equal(frame.numbers, g["frame"].numbers)
                      and np.array_equal(frame.pbc, g["frame"].pbc)
                      and np.allclose(frame.cell, g["frame"].cell, atol=1e-7, rtol=0)
                      and np.allclose(frame.positions, g["frame"].positions, atol=1e-7, rtol=0)
                      and np.array_equal(pattern, g["pattern"])
                      and np.allclose(norms, g["norms"], atol=1e-5, rtol=1e-3)), None)
        if group is None:
            active = pattern != 0
            order = "FM" if (pattern[active] == 1).all() else (
                "AFM" if abs(np.dot(pattern, norms)) < 1e-3 * norms.sum() else "FIM")
            name = " · ".join([*(str(value) for value in declared), f"{order} {len(groups) + 1}"])
            group = dict(frame=frame, pattern=pattern, norms=norms, directions=[], name=name, declared=declared)
            groups.append(group)
        metadata[str(index)] = {"rotation_group": group["name"], "rotation_state": str(len(group["directions"]))}
        group["directions"].append(direction)
    # Constant spin directions alone do not identify a rotation experiment.
    if not groups or any(len(g["directions"]) < 2 or
                         np.max(np.linalg.norm(np.asarray(g["directions"]) - g["directions"][0], axis=1)) < 1e-3
                         for g in groups):
        return {}
    return metadata


def rotation_series(frames, summary):
    """Use the chosen reference direction for zero and a Cartesian right-hand sign."""
    branches = []
    for group in summary["series"]:
        directions = {i: spin_direction(frames[i])[0] for i in group["frame_indices"]}
        by_frame = {summary["rows"][i]["terms"][0][0]: i for i in group["indices"]}
        reference = summary["rows"][group["indices"][0]]["terms"][1][0]
        normals = []
        for i, j in combinations(directions, 2):
            normal = np.cross(directions[i], directions[j])
            length = np.linalg.norm(normal)
            if length < 1e-3:
                continue
            normal /= length
            if normal[np.argmax(np.abs(normal))] < 0:
                normal = -normal
            if any(np.linalg.norm(normal - n) < 1e-3 for n in normals):
                continue
            normals.append(normal)
        remaining = set(directions)
        while remaining and normals:
            position = max(range(len(normals)), key=lambda index: (
                sum(abs(np.dot(directions[i], normals[index])) < 1e-3 for i in remaining),
                sum(abs(np.dot(d, normals[index])) < 1e-3 for d in directions.values())))
            normal = normals.pop(position)
            selected = [i for i, d in directions.items() if abs(np.dot(d, normal)) < 1e-3]
            if not remaining.intersection(selected):
                continue
            zero = reference if reference in selected else selected[0]
            u = directions[zero]
            v = np.cross(normal, u)
            points = [(round(float(np.degrees(np.arctan2(np.dot(directions[i], v),
                                                         np.dot(directions[i], u)))), 3),
                       by_frame.get(i)) for i in selected]
            points.sort(key=lambda point: point[0])
            normal_text = ", ".join(f"{x:.3g}" for x in normal)
            plane = ("YZ", "XZ", "XY")[int(np.argmax(np.abs(normal)))] if np.max(np.abs(normal)) > .999 else f"n=({normal_text})"
            name = group["name"].removeprefix("rotation_group=")
            branches.append(dict(name=name + " · " + plane, axis="angle_deg",
                                 x=[p[0] for p in points], indices=[p[1] for p in points], connected=True,
                                 reference_frame=reference, zero_frame=zero, normal=normal.tolist(),
                                 reference=str(reference), plane=plane))
            remaining.difference_update(selected)
        if remaining:
            indices = [by_frame[i] for i in directions if i in remaining and i in by_frame]
            if indices:
                branches.append(dict(name=group["name"].removeprefix("rotation_group="), axis="direction", indices=indices,
                                     x=list(range(1, len(indices) + 1)), connected=False,
                                     labels=[summary["rows"][i]["label"] for i in indices]))
    return branches
