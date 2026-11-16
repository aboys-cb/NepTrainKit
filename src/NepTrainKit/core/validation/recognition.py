"""Conservative scan and four-state proposals from ordered atomic inputs."""

from itertools import combinations, product

import numpy as np

from NepTrainKit.core.magnetism import existing_moment_vectors

AUTO_EOS_GROUP = "__structure_scan__"


def _geometry_key(frame, *, scaled=False, cell=True):
    coordinates = frame.get_scaled_positions(wrap=True) if scaled else frame.positions
    coordinates = np.round(coordinates, 5 if scaled else 6)
    if scaled:
        coordinates[:, frame.pbc] %= 1
    lattice = frame.cell.array.copy() if cell else np.empty(0)
    if scaled and cell:
        lattice /= np.cbrt(abs(np.linalg.det(lattice)))
    lattice = np.round(lattice, 6)
    coordinates[coordinates == 0] = 0
    lattice[lattice == 0] = 0
    return frame.numbers.tobytes(), frame.pbc.tobytes(), coordinates.tobytes(), lattice.tobytes()


def same_geometry(first, other, *, scaled=False, cell=True):
    if (not np.array_equal(first.numbers, other.numbers)
            or not np.array_equal(first.pbc, other.pbc)):
        return False
    if cell:
        a, b = first.cell.array, other.cell.array
        if scaled:
            if min(abs(np.linalg.det(a)), abs(np.linalg.det(b))) < 1e-10:
                return False
            a, b = a / np.cbrt(abs(np.linalg.det(a))), b / np.cbrt(abs(np.linalg.det(b)))
        if not np.allclose(a, b, atol=1e-6, rtol=0):
            return False
    if not scaled:
        return np.allclose(first.positions, other.positions, atol=1e-7, rtol=0)
    if first.cell.rank != 3 or other.cell.rank != 3:
        return False
    delta = first.get_scaled_positions(wrap=False) - other.get_scaled_positions(wrap=False)
    delta[:, first.pbc] -= np.rint(delta[:, first.pbc])
    return np.allclose(delta, 0, atol=1e-5, rtol=0)


def _same_moments(first, other):
    a = existing_moment_vectors(first)
    b = existing_moment_vectors(other)
    return a is None and b is None or (
        a is not None and b is not None and np.allclose(a, b, atol=1e-7, rtol=0))


def eos_scan_groups(frames):
    """Split uniform volume scans by geometry and the complete input spin pattern."""
    from .protocols.eos import EquationOfStateProtocol

    groups, buckets = [], {}
    for i, frame in enumerate(frames):
        if not frame.pbc.all() or frame.cell.rank != 3:
            return {}
        candidates = buckets.setdefault(_geometry_key(frame, scaled=True), [])
        group = next((g for g in candidates if same_geometry(frames[g[0]], frame, scaled=True)
                      and _same_moments(frames[g[0]], frame)), None)
        if group is None:
            group = []
            groups.append(group)
            candidates.append(group)
        group.append(i)
    protocol = EquationOfStateProtocol()
    if not groups or any(any(issue.level == "error" for issue in protocol.check([frames[i] for i in g]))
                         for g in groups):
        return {}
    return {f"Scan {n + 1}": indices for n, indices in enumerate(groups)}


def magnitude_scan(frames):
    """Detect uniform moment scaling, including a unique zero-moment endpoint."""
    groups, zeros, buckets = [], [], {}
    for i, frame in enumerate(frames):
        if not frame.pbc.all() or frame.cell.rank != 3:
            return None
        moments = existing_moment_vectors(frame, lift_scalar=False)
        if moments is None or not np.isfinite(moments).all():
            return None
        norm = np.linalg.norm(moments)
        if norm < 1e-10:
            zeros.append(i)
            continue
        direction = moments / norm
        candidates = buckets.setdefault(_geometry_key(frame, scaled=True), [])
        group = next((g for g in candidates if same_geometry(frames[g[0][0]], frame, scaled=True)
                      and np.allclose(g[1], direction, atol=1e-5, rtol=0)), None)
        if group is None:
            group = [[], direction]
            groups.append(group)
            candidates.append(group)
        group[0].append(i)
    for i in zeros:
        matches = [g for g in groups if same_geometry(frames[g[0][0]], frames[i], scaled=True)]
        if len(matches) != 1:
            return None
        matches[0][0].append(i)
    if not groups:
        return None
    metadata, changed_volume, changed_moment = {}, [], []
    for n, (indices, _) in enumerate(groups):
        indices.sort()
        norms = np.array([np.linalg.norm(existing_moment_vectors(frames[i], lift_scalar=False)) for i in indices])
        volumes = np.array([frames[i].get_volume() for i in indices])
        ref = next(k for k, value in enumerate(norms) if value > 1e-10)
        v, m = volumes / volumes[ref], norms / norms[ref]
        if len(indices) < 2 or np.unique(np.round(np.column_stack((v, m)), 8), axis=0).shape[0] != len(indices):
            return None
        changed_volume.append(np.ptp(v) > 1e-6)
        changed_moment.append(np.ptp(m) > 1e-6)
        for k, i in enumerate(indices):
            metadata[str(i)] = dict(scan_group=f"Scan {n + 1}", volume_ratio=float(v[k]),
                                    moment_ratio=float(m[k]), scan_reference_frame=indices[ref])
    if not any(changed_moment):
        return None
    if any(changed_volume) and not all(changed_volume):
        return None
    return dict(mode="scan", group_fields=["scan_group"],
                x_field="volume_ratio" if any(changed_volume) else "moment_ratio",
                second_axis="moment_ratio" if any(changed_volume) else "", frame_metadata=metadata)


def shear_scan(frames):
    """Recognise a single engineering-shear coordinate relative to each group's first cell."""
    groups, buckets = [], {}
    for i, frame in enumerate(frames):
        if not frame.pbc.all() or frame.cell.rank != 3:
            return None
        candidates = buckets.setdefault(_geometry_key(frame, scaled=True, cell=False), [])
        group = next((g for g in candidates if same_geometry(frames[g[0]], frame, scaled=True, cell=False)
                      and _same_moments(frames[g[0]], frame)), None)
        if group is None:
            group = []
            groups.append(group)
            candidates.append(group)
        group.append(i)
    metadata, axes = {}, set()
    for n, indices in enumerate(groups):
        if len(indices) < 2:
            return None
        reference = frames[indices[0]].cell.array.T
        values = []
        for i in indices:
            deformation = frames[i].cell.array.T @ np.linalg.inv(reference)
            if abs(np.linalg.det(deformation) - 1) > 1e-3:
                return None
            strain = (deformation + deformation.T) / 2 - np.eye(3)
            values.append([strain[0, 0], strain[1, 1], strain[2, 2],
                           2 * strain[1, 2], 2 * strain[2, 0], 2 * strain[0, 1]])
        values = np.asarray(values)
        axis = 3 + int(np.argmax(np.ptp(values[:, 3:], axis=0)))
        amplitude = np.max(np.abs(values[:, axis]))
        if amplitude < 1e-7 or amplitude > .2:
            return None
        if np.max(np.abs(np.delete(values, axis, axis=1))) > max(1e-6, .03 * amplitude):
            return None
        if len(np.unique(np.round(values[:, axis], 10))) != len(indices):
            return None
        field = ("shear_yz", "shear_zx", "shear_xy")[axis - 3]
        axes.add(field)
        for k, i in enumerate(indices):
            metadata[str(i)] = {field: float(values[k, axis]), "shear_group": f"Shear {n + 1}",
                                "strain_reference_frame": indices[0]}
    if len(axes) != 1:
        return None
    return dict(mode="scan", x_field=axes.pop(), group_fields=["shear_group"], frame_metadata=metadata)


def four_state_candidates(frames):
    """Propose complete independent two-spin products with an explicit canonical sign."""
    if len(frames) < 4 or any(existing_moment_vectors(frame, lift_scalar=False) is None for frame in frames):
        return None
    groups, buckets = [], {}
    for i, frame in enumerate(frames):
        candidates = buckets.setdefault(_geometry_key(frame), [])
        group = next((g for g in candidates if same_geometry(frames[g[0]], frame)), None)
        if group is None:
            group = []
            groups.append(group)
            candidates.append(group)
        group.append(i)
    metadata = {}
    for indices in groups:
        moments = [existing_moment_vectors(frames[i], lift_scalar=False) for i in indices]
        if len(indices) < 4 or any(m is None or not np.isfinite(m).all() for m in moments):
            return None
        spins = np.asarray(moments)
        varied = np.flatnonzero(np.any(np.abs(spins - spins[0]) > 1e-7, axis=(0, 2)))
        if len(varied) != 2:
            return None
        sites = list(map(int, varied))
        others = [k for k in range(spins.shape[1]) if k not in sites and np.linalg.norm(spins[0, k]) > 1e-8]
        background = spins[0, others]
        normal = background[0] / np.linalg.norm(background[0]) if len(background) else None
        levels, codes, pairs = [], [], []
        for site in sites:
            distinct, encoded = [], []
            for vector in spins[:, site]:
                code = next((k for k, value in enumerate(distinct) if np.allclose(vector, value, atol=1e-7, rtol=0)), None)
                if code is None:
                    code = len(distinct)
                    distinct.append(vector)
                encoded.append(code)
            level_pairs = []
            for a, b in combinations(range(len(distinct)), 2):
                first, second = distinct[a], distinct[b]
                if np.linalg.norm(first) < 1e-8 or not np.isclose(np.linalg.norm(first), np.linalg.norm(second), atol=1e-7, rtol=0):
                    continue
                center = (first + second) / 2
                if len(distinct) > 2 and (normal is None or np.linalg.norm(center) < 1e-8
                                        or np.linalg.norm(np.cross(center / np.linalg.norm(center), normal)) > 1e-3):
                    continue
                difference = first - second
                positive = a if difference[np.argmax(np.abs(difference))] > 0 else b
                level_pairs.append((a, b, positive))
            levels.append(distinct)
            codes.append(encoded)
            pairs.append(level_pairs)
        found = []
        for left, right in product(*pairs):
            selected = [k for k in range(len(indices)) if codes[0][k] in left[:2] and codes[1][k] in right[:2]]
            states = {(codes[0][k], codes[1][k]) for k in selected}
            if len(selected) == 4 and len(states) == 4:
                found.append((selected, left, right))
        coverage = [sum(k in selected for selected, _, _ in found) for k in range(len(indices))]
        if any(count != 1 for count in coverage):
            return None
        for selected, left, right in found:
            axes, angles = [], []
            for values, pair in zip(levels, (left, right)):
                first, second = values[pair[0]], values[pair[1]]
                direction = first - second
                direction /= np.linalg.norm(direction)
                if direction[np.argmax(np.abs(direction))] < 0:
                    direction = -direction
                axis = "xyz"[int(np.argmax(np.abs(direction)))] if np.max(np.abs(direction)) > .999999 else (
                    ",".join(f"{value:.6g}" for value in direction))
                axes.append(axis)
                angles.append(round(float(np.degrees(np.arccos(np.clip(
                    np.linalg.norm((first + second) / 2) / np.linalg.norm(first), 0, 1)))), 3))
            for k in selected:
                metadata[str(indices[k])] = dict(
                    pair_group=f"{sites[0]}–{sites[1]} · {indices[0]} · {angles[0]}°/{angles[1]}°",
                    site_i=sites[0], site_j=sites[1],
                    pair_geometry=indices[0], theta_i_deg=angles[0], theta_j_deg=angles[1],
                    axis_i=axes[0], axis_j=axes[1],
                    sign_i=1 if codes[0][k] == left[2] else -1,
                    sign_j=1 if codes[1][k] == right[2] else -1)
    if not metadata:
        return None
    return dict(mode="four_state", group_fields=["pair_group", "axis_i", "axis_j"], state_fields=["sign_i", "sign_j"],
                normalization="total", extract_exchange=False, frame_metadata=metadata)
