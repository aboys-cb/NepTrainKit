"""Measured response coordinates, shared by reference previews and saved plots."""

from itertools import combinations
import re

import numpy as np

from NepTrainKit.core.magnetism import existing_moment_vectors


def is_mae(summary):
    spec = summary.get("analysis", {})
    return (spec.get("mode") == "difference" and spec.get("quantity") == "energy"
            and bool({"direction", "angle_deg", "rotation_state"} & set(spec.get("state_fields", ()))))


def _angular_series(frames, summary):
    """Resolve rotation planes from labelled endpoints and input spin directions."""
    branches = []
    for group in summary["series"]:
        indices = group["indices"]
        by_frame = {summary["rows"][i]["terms"][0][0]: i for i in indices}
        endpoints = {frames[i].info["direction"]: i for i in group["frame_indices"]
                     if len(str(frames[i].info.get("direction", ""))) == 1}
        reference = summary["rows"][indices[0]]["terms"][1][0]
        vectors = {}
        for name, i in endpoints.items():
            moments = existing_moment_vectors(frames[i], lift_scalar=False)
            if moments is not None and np.linalg.norm(moments) > 1e-10:
                vectors[name] = np.asarray(moments).ravel() / np.linalg.norm(moments)
        covered = set()
        for first, second in combinations(sorted(endpoints), 2):
            if first not in vectors or second not in vectors:
                continue
            u, v = vectors[first], vectors[second]
            if u.shape != v.shape or abs(np.dot(u, v)) > 1e-3:
                continue
            v = v - np.dot(u, v) * u
            v /= np.linalg.norm(v)
            plane = first + second
            points = []
            for i in group["frame_indices"]:
                info = frames[i].info
                if "angle_deg" in info:
                    if info.get("plane") != plane:
                        continue
                    angle = float(info["angle_deg"])
                else:
                    moments = existing_moment_vectors(frames[i], lift_scalar=False)
                    if moments is None or np.linalg.norm(moments) < 1e-10:
                        continue
                    direction = np.asarray(moments).ravel() / np.linalg.norm(moments)
                    if direction.shape != u.shape:
                        continue
                    a, b = np.dot(direction, u), np.dot(direction, v)
                    if np.linalg.norm(direction - a * u - b * v) > 1e-3:
                        continue
                    angle = round(float(np.degrees(np.arctan2(b, a))), 3)
                if not np.isfinite(angle) or (i not in by_frame and i != reference):
                    continue
                row = by_frame.get(i)
                points.append((angle, row))
            if len(points) > 1:
                points.sort(key=lambda point: point[0])
                covered.update(row for _, row in points if row is not None)
                branches.append(dict(name=group["name"] + " · " + plane, axis="angle_deg",
                                     x=[p[0] for p in points], indices=[p[1] for p in points],
                                     connected=True, reference=summary["analysis"]["reference"], plane=plane))
        # Explicit angle metadata also works without orthogonal endpoint states.
        planes = {}
        for i in indices:
            info = frames[summary["rows"][i]["terms"][0][0]].info
            if i not in covered and "angle_deg" in info:
                planes.setdefault(str(info.get("plane", "")), []).append((float(info["angle_deg"]), i))
        for plane, points in planes.items():
            if reference in group["frame_indices"] and frames[reference].info.get("angle_deg") is not None:
                points.append((float(frames[reference].info["angle_deg"]), None))
            points.sort(key=lambda point: point[0])
            branches.append(dict(name=group["name"] + (" · " + plane if plane else ""), axis="angle_deg",
                                 x=[p[0] for p in points], indices=[p[1] for p in points],
                                 connected=True, reference=summary["analysis"]["reference"], plane=plane))
            covered.update(i for _, i in points if i is not None)
        remaining = [i for i in indices if i not in covered]
        if remaining:
            branches.append(dict(name=group["name"], axis="direction", indices=remaining,
                                 x=list(range(1, len(remaining) + 1)), connected=False,
                                 labels=[summary["rows"][i]["label"] for i in remaining]))
    return branches


def response_plot_series(frames, summary):
    """Describe display coordinates without changing signed values or frame terms."""
    spec = summary["analysis"]
    if spec["mode"] == "four_state":
        records = []
        for row in summary["rows"]:
            info = frames[row["terms"][0][0]].info
            fields = tuple(spec.get("axis_fields", ("axis_i", "axis_j"))) + tuple(spec.get("pair_fields", ("site_i", "site_j")))
            if any(field not in info for field in fields):
                return []
            angle = str(info.get("theta_deg", ""))
            if not angle and "theta_i_deg" in info and "theta_j_deg" in info:
                angle = f"{info['theta_i_deg']}/{info['theta_j_deg']}"
            condition = (str(info[fields[2]]), str(info[fields[3]]), angle)
            structure = str(info.get("geometry", info.get("geometry_id", info.get("phase", ""))))
            if not structure and "pair_geometry" in info:
                structure = f"geometry {info['pair_geometry']}"
            if not structure:
                return []
            records.append((condition, (str(info[fields[0]]), str(info[fields[1]])), structure))
        structures = sorted({r[2] for r in records},
                            key=lambda name: [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)])
        positions = {name: i + 1 for i, name in enumerate(structures)}
        grouped = {}
        for i, (condition, axes, structure) in enumerate(records):
            grouped.setdefault((condition, axes), []).append((positions[structure], i, structure))
        result = []
        for (condition, axes), points in grouped.items():
            points.sort(key=lambda point: point[0])
            if len({p[0] for p in points}) != len(points):
                return []
            angle = f" · θ={condition[2]}°" if condition[2] else ""
            result.append(dict(name=f"{axes[0]} / {axes[1]} · i={condition[0]}, j={condition[1]}{angle}",
                               axis="structure", x=[p[0] for p in points], indices=[p[1] for p in points],
                               labels=[p[2] for p in points], connected=False, condition=list(condition)))
        return result
    if spec["mode"] == "difference" and "rotation_state" in spec["state_fields"]:
        from .spin_rotation import rotation_series

        return rotation_series(frames, summary)
    if is_mae(summary):
        return _angular_series(frames, summary)
    if spec["mode"] == "difference" and {"path", "image"} <= set(spec["group_fields"]):
        paths = {}
        for i, row in enumerate(summary["rows"]):
            info = frames[row["terms"][0][0]].info
            key = (info["path"], row["label"], info.get("gamma", 0))
            paths.setdefault(key, []).append((float(info["image"]), i))
        result = []
        for (path, state, shear), points in paths.items():
            points.sort()
            result.append(dict(name=f"{path} · {state} · γ={shear}", axis="image",
                               x=[p[0] for p in points], indices=[p[1] for p in points], connected=True,
                               reference=spec["reference"]))
        return result
    return []
