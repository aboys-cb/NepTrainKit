"""Fixed-budget input-state and local-environment coverage.

Supported input regions (scans, smooth paths, separated joint levels, finite
state sets) become witness requirements; each physical stratum is a witness
too. A minimum exact cover of these witnesses seeds unweighted multi-view
farthest-point coverage of structures and actual atomic environments, which
spends the remaining budget. No frame is pinned by a planner, and no reference
labels, source names or material-specific rules are read.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .sampling_coverage import _prepare, _select_coverage
from .sampling_plan import PhysicsSamplingStratum

_MAX_COMPONENT_VARIABLES = 4096


def _parts(labels):
    order = np.argsort(labels, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(np.asarray(labels)[order])) + 1)


def _affine_rank(points):
    singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
    return int(np.count_nonzero(singular > max(singular.max(initial=0)*1e-5, 1e-10)))


def _factors(points):
    """Return a verified path/grid coordinate subset, or decline the fit.

    Repeated levels support factorization. Every combination must actually
    occur, and identify one distinct input state; integer labels from a single
    accidentally unique coordinate cannot certify a multidimensional cloud.
    """
    ranks, levels = [], []
    for column in points.T:
        values, inverse = np.unique(column, return_inverse=True)
        ranks.append(inverse)
        levels.append(len(values))
    encoded = np.asarray(ranks).T
    chosen, count = [], 1
    for j in sorted(range(len(levels)), key=lambda i: (levels[i], i)):
        if levels[j] < 2:
            continue
        current = len(np.unique(encoded[:, chosen + [j]], axis=0))
        if current > count:
            chosen.append(j)
            count = current
        if count == len(points):
            break
    for j in chosen.copy():
        remaining = [k for k in chosen if k != j]
        if remaining and len(np.unique(encoded[:, remaining], axis=0)) == len(points):
            chosen = remaining
    if len(chosen) == 1 and _affine_rank(points) <= 1:
        return chosen
    # Bound discrete-grid construction cost. Higher-dimensional and incomplete
    # products go through local-environment FPS rather than guessed axes.
    if 2 <= len(chosen) <= 3 and math.prod(levels[j] for j in chosen) == len(points):
        return chosen
    return None


def _repeated_levels(values):
    """Separated repeated levels, allowing within-level observation noise.

    Require two observations per level and a clear gap above its full width.
    Unique continuous values alone are not evidence of a discrete factor.
    """
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    limit = 2 * math.sqrt(len(values))
    if 3 <= len(unique) <= limit and counts.min() >= 2:
        return inverse
    if len(unique) < 4:
        return None
    gaps = np.diff(unique)
    sorted_gaps = np.sort(gaps)
    ratios = sorted_gaps[1:] / np.maximum(sorted_gaps[:-1], 1e-12)
    for split in np.argsort(ratios)[::-1]:
        if ratios[split] < 6:
            break
        boundaries = unique[1:][gaps > (sorted_gaps[split] + sorted_gaps[split+1])/2]
        if not 3 <= len(boundaries)+1 <= limit:
            continue
        labels = np.searchsorted(boundaries, values, side="right")
        if np.bincount(labels).min() < 2:
            continue
        width = max(np.ptp(values[rows]) for rows in _parts(labels))
        if width * 2 <= sorted_gaps[split+1]:
            return labels
    return None


def _joint_coordinates(points, factors):
    """Accept occupied joint cells only when their observation clouds separate."""
    from scipy.spatial import cKDTree

    coordinates = np.column_stack(factors)
    levels = coordinates.max(0)+1
    unique, labels = np.unique(coordinates, axis=0, return_inverse=True)
    if not len(unique) <= math.prod(levels) <= 2*len(unique):
        return None
    if len(unique) < len(points):
        groups = _parts(labels)
        centers = np.array([points[rows].mean(0) for rows in groups])
        radius = np.array([np.linalg.norm(points[rows]-center, axis=1).max()
                           for rows, center in zip(groups, centers)])
        separation = cKDTree(centers).query(centers, k=2, workers=1)[0][:, 1]
        # Disjoint balls in the full input space, not only the inferred axes.
        # A local motif with significant unexplained variation cannot vanish
        # by receiving the same marginal coordinate labels.
        if np.any(2*radius >= separation):
            return None
    return coordinates / (levels-1)


def _incomplete_factors(points):
    """Identify observed joint states without inventing missing grid sites.

    Distinct marginal partitions are deduplicated before the bounded 2/3-axis
    search. At least half the product must be observed; each observed state
    must remain distinguishable. Unexplained local variation declines the fit
    and therefore retains the atomic/descriptor fallback.
    """
    partitions = {}
    for values in points.T:
        labels = _repeated_levels(values)
        if labels is not None:
            key = min(tuple(labels), tuple(labels.max()-labels))
            partitions[key] = np.asarray(key)
    # Highly fragmented observations are not a low-dimensional grid. Avoid
    # combinatorial search on arbitrary large continuous populations.
    if not 1 <= len(partitions) <= 16:
        return None
    axes = sorted(partitions.values(), key=lambda x: (int(x.max()), tuple(x)))
    for dimension in (2, 3):
        for factors in itertools.combinations(axes, dimension):
            coordinates = _joint_coordinates(points, factors)
            if coordinates is not None:
                return coordinates
    # Remove a resolved factor's conditional center before looking for a
    # second repeated factor. This also handles independently perturbed
    # observations of one state rather than identical duplicated frames.
    for labels in axes:
        residual = points.copy()
        for rows in _parts(labels):
            residual[rows] -= np.median(residual[rows], axis=0)
        for values in residual.T:
            other = _repeated_levels(values)
            if other is None:
                continue
            coordinates = _joint_coordinates(points, (labels, other))
            if coordinates is not None:
                return coordinates
    # A noisy observation can erase the repeated levels of one coordinate.
    # Condition on an independently resolved factor and verify smooth paths
    # in every slice, with a shared monotonic observable for orientation.
    for labels in axes:
        slices = _parts(labels)
        if len({len(rows) for rows in slices}) != 1 or len(slices[0]) < 3:
            continue
        ordered = []
        monotonic = np.ones(points.shape[1], dtype=bool)
        for rows in slices:
            paths = _continuous_paths(points[rows], minimum=3)
            if len(paths) != 1 or len(paths[0][0]) != len(rows):
                break
            path = rows[paths[0][0]]
            delta = np.diff(points[path], axis=0)
            monotonic &= ((np.all(delta >= -1e-8, axis=0) | np.all(delta <= 1e-8, axis=0))
                          & (np.ptp(points[path], axis=0) > 1e-6))
            ordered.append(path)
        if len(ordered) != len(slices) or not monotonic.any():
            continue
        columns = np.flatnonzero(monotonic)
        span = np.array([np.ptp(points[path][:, columns], axis=0) for path in ordered])
        column = columns[np.argmax(span.min(0))]
        second = np.empty(len(points), dtype=float)
        for path in ordered:
            if points[path[-1], column] < points[path[0], column]:
                path = path[::-1]
            second[path] = np.linspace(0., 1., len(path))
        return np.column_stack((labels / labels.max(), second))
    return None


def _continuous_paths(points, *, minimum=9):
    """Recognize smooth open paths using a sparse neighborhood forest.

    A path needs two ends, no branches, and locally aligned tangents. Long
    edges relative to both endpoints' local spacing separate motif branches.
    A random cloud or a two-dimensional sheet is never certified by sorting
    one accidentally unique scalar. Memory is O(N k), not O(N squared).
    """
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
    from scipy.spatial import cKDTree

    count = len(points)
    if count < minimum:
        return []
    # Use a low-dimensional projection only to propose sparse neighbors.
    # Edge lengths, separation and tangents are checked in the full space.
    # This avoids an effectively quadratic high-dimensional tree query.
    centered = points-points.mean(0)
    if points.shape[1] > 8:
        _, vectors = np.linalg.eigh(centered.T @ centered)
        search = centered @ vectors[:, -8:]
    else:
        search = points
    proposals = cKDTree(search).query(search, k=min(33, count), workers=1)[1]
    # A rank-deficient projection can give many points identical search
    # coordinates. Explicitly include each center, exactly once.
    proposals = np.column_stack((np.arange(count), proposals))
    neighbor = np.empty((count, min(9, count)), dtype=int)
    distance = np.empty_like(neighbor, dtype=float)
    for begin in range(0, count, 1024):
        stop = min(count, begin+1024)
        proposed = proposals[begin:stop]
        lengths = np.linalg.norm(points[proposed]-points[begin:stop, None, :], axis=2)
        # The center must be first even for numerically degenerate proposals.
        lengths[proposed == np.arange(begin, stop)[:, None]] = np.inf
        lengths[:, 0] = -1.
        order = np.argsort(lengths, axis=1, kind="stable")[:, :neighbor.shape[1]]
        neighbor[begin:stop] = np.take_along_axis(proposed, order, axis=1)
        distance[begin:stop] = np.maximum(0., np.take_along_axis(lengths, order, axis=1))
    rows = np.repeat(np.arange(count), neighbor.shape[1]-1)
    columns = neighbor[:, 1:].ravel()
    lengths = distance[:, 1:].ravel()
    spacing = distance[:, 1]
    keep = lengths <= 3*np.maximum(spacing[rows], spacing[columns])
    graph = csr_matrix((lengths[keep], (rows[keep], columns[keep])), shape=(count, count))
    graph = graph.maximum(graph.T)
    _, labels = connected_components(graph, directed=False)
    result = []
    for members in _parts(labels):
        if len(members) < minimum:
            continue
        tree = minimum_spanning_tree(graph[members][:, members])
        tree = (tree + tree.T).tocsr()
        degree = np.diff(tree.indptr)
        ends = np.flatnonzero(degree == 1)
        if len(ends) != 2 or np.any(degree > 2):
            continue
        ordered, previous, row = [], -1, int(ends[0])
        while True:
            ordered.append(row)
            neighbors = tree.indices[tree.indptr[row]:tree.indptr[row+1]]
            following = neighbors[neighbors != previous]
            if not len(following):
                break
            previous, row = row, int(following[0])
        path = points[members[ordered]]
        steps = np.diff(path, axis=0)
        norms = np.linalg.norm(steps, axis=1)
        cosine = np.sum(steps[:-1]*steps[1:], axis=1)/(norms[:-1]*norms[1:])
        if np.min(cosine) < .5:
            continue
        # Prefer a physical observable which advances monotonically along the
        # verified path. Use its actual values, not density-biased rank order.
        monotonic = np.all(steps >= -1e-8, axis=0) | np.all(steps <= 1e-8, axis=0)
        span = np.ptp(path, axis=0)
        monotonic &= span > 1e-6
        if not monotonic.any():
            coordinate = np.r_[0., np.cumsum(norms)]
        else:
            # The least varying normalized slope with respect to arc length
            # is the least distorted monotonic observable available.
            slopes = np.abs(steps[:, monotonic])/norms[:, None]
            variation = np.std(slopes, axis=0)/np.maximum(slopes.mean(0), 1e-12)
            coordinate = path[:, np.flatnonzero(monotonic)[variation.argmin()]]
        coordinate = (coordinate-coordinate.min())/np.ptp(coordinate)
        result.append((members[np.asarray(ordered)], coordinate[:, None]))
    return result


def _nonideal_regions(points):
    coordinates = _incomplete_factors(points)
    if coordinates is not None:
        return [(np.arange(len(points)), coordinates)]
    return _continuous_paths(points)


def _regions(points, view_names, phases, *, partial=False):
    """Find supported paths/grids while holding other input views constant."""
    views = sorted(set(view_names))
    fixed_sets = {(), *(tuple([v]) for v in views),
                  *(tuple(w for w in views if w != v) for v in views)}
    possible = {}
    for fixed in sorted(fixed_sets):
        _, labels = np.unique(points[:, np.isin(view_names, fixed)], axis=0, return_inverse=True)
        for members in _parts(labels):
            if len(members) < 9:
                continue
            factors = _factors(points[members])
            if factors is not None:
                possible[tuple(members)] = factors
                continue
            # A physical phase may separate distinct one-dimensional branches.
            # Do not split a supported multidimensional strain grid merely
            # because a classifier changes phase along that continuous grid.
            _, local = np.unique(phases[members], return_inverse=True)
            if local.max(initial=0) == 0:
                continue
            pieces = [members[rows] for rows in _parts(local)]
            fits = [_factors(points[rows]) if len(rows) >= 9 else None for rows in pieces]
            if partial or all(fit is not None for fit in fits):
                possible.update((tuple(rows), fit) for rows, fit in zip(pieces, fits)
                                if fit is not None)
    used = np.zeros(len(points), dtype=bool)
    result = []
    for members in sorted(possible, key=lambda rows: (-len(rows), rows)):
        members_array = np.asarray(members, dtype=int)
        if used[members_array].any():
            continue
        used[members_array] = True
        result.append((members_array, possible[members]))
    return result


def _line_representatives(values, budget, *, require_endpoints=False, radius_limit=None):
    """Minimize finite path gaps; retain endpoints whenever equally feasible."""
    order = np.argsort(values, kind="stable")
    x = np.asarray(values)[order]
    x = (x-x[0]) / max(float(np.ptp(x)), 1e-12)

    def cover(radius, endpoints=False):
        picked = [0, len(x)-1] if endpoints and len(x) > 1 else []
        left, stop = 0, len(x)
        if picked:
            left = int(np.searchsorted(x, radius + 1e-10, side="right"))
            stop = int(np.searchsorted(x, 1-radius-1e-10, side="left"))
        while left < stop:
            center = int(np.searchsorted(x, x[left]+radius+1e-10, side="right"))-1
            picked.append(center)
            left = int(np.searchsorted(x, x[center]+radius+1e-10, side="right"))
        return list(dict.fromkeys(picked))

    require_endpoints = require_endpoints and budget >= 2
    low, high = 0., 1.
    for _ in range(40):
        middle = (low+high)/2
        if len(cover(middle, endpoints=require_endpoints)) <= budget:
            high = middle
        else:
            low = middle
    if radius_limit is not None:
        high = max(high, float(radius_limit))
    if require_endpoints:
        # Resolve equally good finite covers by a spread-out representation,
        # avoiding a systematic left-to-right greedy packing bias.
        spread = np.abs(x[:, None]-np.linspace(0., 1., budget)).argmin(0)
        gap = np.abs(x[:, None]-x[spread]).min(1).max()
        if len(np.unique(spread)) == budget and gap <= high+1e-6:
            return order[spread]
    chosen = cover(high, endpoints=True)
    if len(chosen) > budget:
        chosen = cover(high)
    nearest = np.full(len(x), np.inf)
    for row in chosen:
        nearest = np.minimum(nearest, np.abs(x-x[row]))
    while len(chosen) < min(budget, len(x)):
        nearest[chosen] = -1
        row = int(nearest.argmax())
        chosen.append(row)
        nearest = np.minimum(nearest, np.abs(x-x[row]))
    return order[chosen]


def _grid_shape(levels, budget):
    best, score = None, None
    # Enumerate at most the number of observed product states; no exponential
    # Cartesian array of unobserved cells is materialized.
    for prefix in itertools.product(*(range(1, min(n, budget)+1) for n in levels[:-1])):
        product = math.prod(prefix)
        if product > budget:
            continue
        shape = (*prefix, min(levels[-1], budget//product))
        retention = np.asarray(shape)/levels
        current = (math.prod(shape), -float(np.ptp(retention)))
        if score is None or current > score:
            best, score = shape, current
    return np.asarray(best, dtype=int)


def _grid_representatives(points, factors, budget):
    if len(factors) == 1:
        return _line_representatives(points[:, factors[0]], budget)
    ranks, levels = [], []
    for j in factors:
        values, inverse = np.unique(points[:, j], return_inverse=True)
        ranks.append(inverse)
        levels.append(len(values))
    levels = np.asarray(levels)
    coordinates = np.asarray(ranks).T / (levels-1)
    shape = _grid_shape(levels, budget)
    cells = np.minimum((coordinates*shape).astype(int), shape-1)
    _, labels = np.unique(cells, axis=0, return_inverse=True)
    chosen = []
    for members in _parts(labels):
        # Targets spread across the observed level range. A frame is selected
        # for its joint state, never for its previous selection index.
        target = cells[members[0]] / np.maximum(1, shape-1)
        if np.any(shape == 1):
            target[shape == 1] = .5
        chosen.append(int(members[np.argmin(((coordinates[members]-target)**2).sum(1))]))
    nearest = np.full(len(points), np.inf)
    for row in chosen:
        nearest = np.minimum(nearest, ((coordinates-coordinates[row])**2).sum(1))
    while len(chosen) < min(budget, len(points)):
        nearest[chosen] = -1
        row = int(nearest.argmax())
        chosen.append(row)
        nearest = np.minimum(nearest, ((coordinates-coordinates[row])**2).sum(1))
    return np.asarray(chosen, dtype=int)


def _state_seeds(candidate, plan, order, budget, metric, supports, finite_supports):
    """Detect supported input regions and their minimax reference frames.

    Region and cloud seats come from ``_allocate``. ``supports`` receives each
    region's frame rows, chart coordinates and grid shape; ``finite_supports``
    receives the distinct states of finite clouds that obtained enough seats.
    """
    represented = np.zeros(candidate.row_count, dtype=bool)
    names, values = [], []
    for name, value in zip(candidate.names, candidate.values):
        if name.startswith("input_"):
            names.extend([name.split(":")[0]]*value.shape[1])
            values.append(value)
    if not values:
        return (), 0, represented
    raw = np.hstack(values)
    element_groups, phases = {}, np.empty(candidate.row_count, dtype=object)
    for key, rows in plan.groups:
        element_groups.setdefault(key.element_set, []).extend(rows)
        phases[list(rows)] = key.phase
    pools, regions, remainders, cloud_rows = [], [], [], []
    for element in sorted(element_groups):
        rows = order[np.isin(order, element_groups[element])]
        local = raw[rows]
        span = np.ptp(local, axis=0)
        active = span > np.maximum(1., np.max(np.abs(local), axis=0))*1e-6
        if not active.any():
            # Constant input fingerprints do not establish equivalent atomic
            # environments. Leave this population to descriptor FPS.
            signatures = np.hstack([v[rows] for v in candidate.values])
            remainders.append(len(np.unique(signatures, axis=0)))
            cloud_rows.append(rows)
            continue
        normalized = (local[:, active]-local[:, active].min(0))/span[active]
        points, first, inverse = np.unique(np.round(normalized, 6), axis=0,
                                           return_index=True, return_inverse=True)
        representatives = rows[first]
        exact = _regions(points, np.asarray(names)[active], phases[representatives], partial=True)
        detected = [(members, points[members][:, factors], False) for members, factors in exact]
        remaining_rows = np.ones(len(points), dtype=bool)
        for members, _ in exact:
            remaining_rows[members] = False
        leftover = np.flatnonzero(remaining_rows)
        if len(leftover) >= 9:
            detected.extend((leftover[members], coordinates, True) for members, coordinates in
                            _nonideal_regions(points[leftover]))
        for members, _, _ in detected:
            represented[rows[np.isin(inverse, members)]] = True
        pools.append((points, representatives, rows, inverse))
        regions.extend((len(pools)-1, members, coordinates, nonideal)
                       for members, coordinates, nonideal in detected)
        remaining = len(points)-sum(len(members) for members, _, _ in detected)
        if remaining:
            remainders.append(remaining)
            cloud_rows.append(rows[~represented[rows]])
    if not regions or len(regions) > budget:
        return (), 0, np.zeros_like(represented)
    sizes = np.asarray([len(np.unique(coordinates, axis=0))
                        for _, _, coordinates, _ in regions], dtype=int)
    sizes = np.r_[sizes, [len(np.unique(metric[rows], axis=0)) for rows in cloud_rows]].astype(int)
    if len(sizes) > budget:
        return (), 0, np.zeros_like(represented)
    charts = [(coordinates, nonideal) for _, _, coordinates, nonideal in regions]
    quotas, _, allocated = _allocate(charts, [metric[rows] for rows in cloud_rows], sizes, budget)
    chosen = []
    for rows, quota in zip(cloud_rows, quotas[len(regions):]):
        unique, first, inverse = np.unique(metric[rows], axis=0, return_index=True, return_inverse=True)
        if quota >= len(unique):
            # Once a finite cloud has zero predicted gap, realize those
            # seats explicitly. Otherwise unrestricted FPS can spend its
            # quota back on a previously covered path and lose an island.
            chosen.extend(int(row) for row in rows[first])
            finite_supports.extend(rows[part] for part in _parts(inverse))
    for region_id, ((pool, members, coordinates, _), quota) in enumerate(zip(regions, quotas)):
        points, rows, frame_rows, inverse = pools[pool]
        # Multiple thermal observations can represent the same joint state.
        # Spend one state seat on its central observation; any extra budget
        # still goes through local-environment FPS below.
        _, state_labels = np.unique(coordinates, axis=0, return_inverse=True)
        representatives = []
        for rows_in_state in _parts(state_labels):
            local = points[members[rows_in_state]]
            center = np.median(local, axis=0)
            representatives.append(rows_in_state[np.sum((local-center)**2, axis=1).argmin()])
        representatives = np.asarray(representatives, dtype=int)
        chosen.extend(int(row) for row in rows[members[representatives[allocated[region_id]]]])
        if coordinates.shape[1] > 1:
            coordinates = np.column_stack([np.unique(v, return_inverse=True)[1] for v in coordinates.T])
        levels = np.array([len(np.unique(v)) for v in coordinates.T])
        coordinates = (coordinates-coordinates.min(0))/np.maximum(np.ptp(coordinates, axis=0), 1e-12)
        local = np.full(len(points), -1, dtype=int)
        local[members] = np.arange(len(members))
        keep = local[inverse] >= 0
        shape = _grid_shape(levels, int(quota)) if coordinates.shape[1] > 1 else None
        supports.append((frame_rows[keep], coordinates[local[inverse[keep]]], shape))
    return tuple(chosen), len(regions), represented


class _RangeRequirements:
    """Input extents of an unstructured pool, satisfied by any representative.

    A selected scan point outside an unstructured interval already satisfies
    that side of the range. Unlike witness IDs, these constraints do not pin
    whichever frame happened to supply an argmin/argmax.
    """

    def __init__(self, candidate, groups, structured):
        raw = np.hstack([v for name, v in zip(candidate.names, candidate.values)
                         if name.startswith("input_")])
        constraints = set()
        for rows in groups.values():
            rows = np.asarray(rows, dtype=int)
            remainder = rows[~structured[rows]]
            if not len(remainder):
                continue
            scale = np.maximum(1., np.max(np.abs(raw[rows]), axis=0))*1e-6
            low, high = raw[remainder].min(0), raw[remainder].max(0)
            for j in np.flatnonzero(high-low > scale):
                constraints.add(tuple(rows[raw[rows, j] <= low[j]+scale[j]]))
                constraints.add(tuple(rows[raw[rows, j] >= high[j]-scale[j]]))
        self.members = [np.asarray(rows, dtype=int) for rows in sorted(constraints)]
        self.covered = np.zeros(len(self.members), dtype=bool)
        self.count = np.zeros(candidate.row_count, dtype=int)
        for rows in self.members:
            self.count[rows] += 1
        if self.members:
            rows = np.concatenate(self.members)
            columns = np.repeat(np.arange(len(self.members)), [len(x) for x in self.members])
            self.incidence = csr_matrix((np.ones(len(rows), dtype=bool), (rows, columns)),
                                        shape=(candidate.row_count, len(self.members)))
        else:
            self.incidence = csr_matrix((candidate.row_count, 0), dtype=bool)

    def add(self, row):
        ids = self.incidence.indices[self.incidence.indptr[row]:self.incidence.indptr[row+1]]
        for constraint in ids[~self.covered[ids]]:
            self.count[self.members[constraint]] -= 1
            self.covered[constraint] = True

    def pending(self):
        return self.count > 0


def _cloud_curve(points, size, budget):
    """Full-population distances for a bounded 64-center pilot.

    Beyond the pilot, extrapolation is an allocation estimate only. Final
    protection always uses the actual structure and atomic coverage backend.
    """
    x = np.unique(np.asarray(points, dtype=float), axis=0)
    norms = np.einsum('ij,ij->i', x, x)
    center = np.mean(x, axis=0)
    row = int(np.argmin(norms-2*x@center))
    distance = np.full(len(x), np.inf)
    errors = [np.inf]
    for _ in range(min(size, budget, 64)):
        distance = np.minimum(distance, np.maximum(0., norms+norms[row]-2*x@x[row]))
        errors.append(float(np.sqrt(distance.max())))
        if errors[-1] <= max(1., errors[1])*1e-7:
            errors[-1] = 0.
            break
        row = int(distance.argmax())
    errors = np.asarray(errors)
    scale = max(errors[1], 1e-12)
    errors /= scale
    end = len(errors)-1
    middle = max(1, end//2)
    if errors[end] > 0 and end > middle:
        exponent = np.clip(np.log(max(errors[middle]/errors[end], 1.))/np.log(end/middle),
                           1/max(1, x.shape[1]), 1.)
    else:
        exponent = 1.

    def radius(count):
        if count >= size:
            return 0.
        if count <= end:
            return float(errors[count])
        return float(errors[end]*(end/count)**exponent)
    return radius


def _allocate(charts, clouds, sizes, budget):
    """Minimize the largest normalized remaining gap across populations.

    Each population's one-representative gap sets its fixed reference scale.
    This makes densifying one path different from adding independent states.
    It is a minimax allocation in the estimated curves, not a proof that the
    physical views have comparable information content.
    """
    curves, scales, plans, coordinates = [], [], [], []
    for points, nonideal in charts:
        points = np.unique(points, axis=0)
        if points.shape[1] > 1:
            points = np.column_stack([np.unique(v, return_inverse=True)[1] for v in points.T])
        points = (points-points.min(0))/np.maximum(np.ptp(points, axis=0), 1e-12)
        cache, choices = {}, {}

        def radius(count, x=points, endpoints=nonideal, values=cache, selected=choices):
            if count not in values:
                ids = (_line_representatives(x[:, 0], count, require_endpoints=endpoints)
                       if x.shape[1] == 1 else
                       _grid_representatives(x, list(range(x.shape[1])), count))
                gap = float(cKDTree(x[ids]).query(x, workers=1)[0].max())
                # A prime quota can collapse a grid to a thin factor shape.
                # Keep the previous cover plus a farthest point if it is
                # better; spending a seat must not enlarge the estimated gap.
                if count-1 in values and values[count-1] < gap-1e-10:
                    ids = selected[count-1].tolist()
                    distance = cKDTree(x[ids]).query(x, workers=1)[0]
                    distance[ids] = -1.
                    ids.append(int(distance.argmax()))
                    ids = np.asarray(ids)
                    gap = float(cKDTree(x[ids]).query(x, workers=1)[0].max())
                values[count] = gap
                selected[count] = ids
            return values[count]
        scale = max(radius(1), 1e-12)
        curves.append(lambda count, r=radius, s=scale: r(count)/s)
        scales.append(scale)
        plans.append(choices)
        coordinates.append((points, nonideal))
    for points, size in zip(clouds, sizes[len(charts):]):
        curves.append(_cloud_curve(points, int(size), budget))
    quotas = np.ones(len(sizes), dtype=int)
    errors = np.array([f(1) for f in curves])
    for _ in range(min(budget, int(np.sum(sizes)))-len(sizes)):
        eligible = quotas < sizes
        if not eligible.any():
            break
        row = int(np.argmax(np.where(eligible, errors, -1.)))
        quotas[row] += 1
        errors[row] = curves[row](int(quotas[row]))
    limits = np.asarray(scales)*float(errors.max(initial=0.))
    representatives = []
    for i, ((points, nonideal), choices) in enumerate(zip(coordinates, plans)):
        count = int(quotas[i])
        representatives.append(_line_representatives(points[:, 0], count,
                                                      require_endpoints=nonideal, radius_limit=limits[i])
                               if points.shape[1] == 1 else choices[count])
    return quotas, limits, representatives


def _input_requirements(candidate, groups, supports, selected):
    """Preserve input ranges, supported path gaps and occupied joint cells."""
    structured = np.zeros(candidate.row_count, dtype=bool)
    requirements = []
    selected = set(selected)
    for rows, points, shape in supports:
        structured[rows] = True
        chosen = np.array([i for i, row in enumerate(rows) if row in selected])
        if not len(chosen):
            raise RuntimeError("supported region has no initial representative")
        radius = cKDTree(points[chosen]).query(points, workers=1)[0].max()
        # Identical input coordinates can be carried by different atomic
        # environments; all such frames belong to the same requirement.
        unique = np.unique(points, axis=0)
        neighborhoods = cKDTree(points).query_ball_point(unique, radius+1e-10, workers=1)
        requirements.extend(rows[local] for local in neighborhoods)
        low, high = points[chosen].min(0), points[chosen].max(0)
        for j in range(points.shape[1]):
            requirements.append(rows[points[:, j] <= low[j]+1e-10])
            requirements.append(rows[points[:, j] >= high[j]-1e-10])
        if shape is not None:
            # A factor can increase or decrease with a physical coordinate.
            # Preserve occupied cells for both axis orientations so boundary
            # conventions do not make an exchange depend on that arbitrary
            # direction. At most eight orientations for supported 3-D grids.
            for reverse in itertools.product((False, True), repeat=points.shape[1]):
                oriented = np.where(reverse, 1-points, points)
                cells = np.minimum(np.floor(oriented*shape+1e-10).astype(int), shape-1)
                for cell in np.unique(cells[chosen], axis=0):
                    requirements.append(rows[np.all(cells == cell, axis=1)])
    ranges = _RangeRequirements(candidate, groups, structured) if any(
        name.startswith("input_") for name in candidate.names) else None
    if ranges is not None:
        # A budget can be too small to satisfy every range. Preserve achieved
        # requirements rather than pretending the initial set met all of them.
        requirements.extend(rows for rows in ranges.members if selected.intersection(rows))
    requirements.extend(np.asarray(rows) for rows in groups.values())
    requirements = sorted({tuple(sorted(rows)) for rows in requirements})
    if not requirements:
        return csr_matrix((candidate.row_count, 0), dtype=bool)
    rows = np.concatenate(requirements)
    columns = np.repeat(np.arange(len(requirements)), [len(x) for x in requirements])
    return csr_matrix((np.ones(len(rows), dtype=bool), (rows, columns)),
                      shape=(candidate.row_count, len(requirements)))


def _region_targets(candidate, plan, prepared, budget):
    supports, finite = [], []
    reference, _, structured = _state_seeds(candidate, plan, prepared[4], budget, prepared[1],
                                            supports, finite)
    groups = {}
    for key, rows in prepared[2].items():
        groups.setdefault(PhysicsSamplingStratum(key.element_set, "unresolved"), []).extend(rows)
    if not supports:
        return [], None, supports, groups
    reference = list(reference)
    order, points = prepared[4], prepared[1]
    for rows in groups.values():
        if set(reference).intersection(rows):
            continue
        ids = order[np.isin(order, rows)]
        center = np.mean(np.unique(points[ids], axis=0), axis=0)
        reference.append(int(ids[np.sum((points[ids]-center)**2, axis=1).argmin()]))
    ranges = _RangeRequirements(candidate, groups, structured)
    for row in reference:
        ranges.add(row)
    while np.any(ranges.pending()) and len(reference) < budget:
        row = int(order[np.argmax(ranges.count[order])])
        reference.append(row)
        ranges.add(row)
    incidence = _input_requirements(candidate, groups, supports, reference)
    if finite:
        rows = np.concatenate(finite)
        cols = np.repeat(np.arange(len(finite)), [len(rows) for rows in finite])
        extra = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                           shape=(candidate.row_count, len(finite)))
        incidence = hstack((incidence, extra), format="csr")
    return reference, incidence, supports, groups


def _reduce_cover(matrix):
    """Unit-cost cover reductions; all preserve a feasible minimum count."""
    matrix = matrix.astype(bool).tocsr()
    active = np.arange(matrix.shape[1])
    forced, rounds = [], []
    while True:
        lengths = np.diff(matrix.indptr)
        if np.any(lengths == 0):
            raise ValueError("coverage requirement has no candidate")
        fixed = np.unique(matrix.indices[matrix.indptr[np.flatnonzero(lengths == 1)]])
        forced.extend(active[fixed])
        satisfied = np.asarray(matrix[:, fixed].sum(1)).ravel() > 0
        matrix = matrix[~satisfied]
        live = np.flatnonzero(np.asarray(matrix.sum(0)).ravel())
        active, matrix = active[live], matrix[:, live].tocsr()
        columns = matrix.tocsc()
        degrees = np.diff(columns.indptr)
        drop = np.zeros(matrix.shape[1], dtype=bool)
        for col in np.argsort(degrees, kind="stable"):
            required = columns.indices[columns.indptr[col]:columns.indptr[col+1]]
            shortest = min(required, key=lambda row: matrix.indptr[row+1]-matrix.indptr[row])
            alternatives = matrix.indices[matrix.indptr[shortest]:matrix.indptr[shortest+1]]
            for other in alternatives:
                if (other == col or drop[other] or degrees[other] < degrees[col]
                        or (degrees[other] == degrees[col] and active[other] > active[col])):
                    continue
                offered = columns.indices[columns.indptr[other]:columns.indptr[other+1]]
                if len(np.intersect1d(required, offered, assume_unique=True)) == len(required):
                    drop[col] = True
                    break
        rounds.append(dict(forced=len(fixed), dominated=int(drop.sum())))
        if not drop.any() and not len(fixed):
            break
        active, matrix = active[~drop], matrix[:, ~drop].tocsr()
    # A broad requirement containing a stricter one adds no restriction.
    # Native certificates often have one/two carriers, making this cheap.
    pairs = {tuple(matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]])
             for i in range(matrix.shape[0]) if matrix.indptr[i+1]-matrix.indptr[i] <= 2}
    keep = np.ones(matrix.shape[0], dtype=bool)
    for row in range(matrix.shape[0]):
        ids = matrix.indices[matrix.indptr[row]:matrix.indptr[row+1]]
        if len(ids) > 2:
            pool = set(ids)
            if any(all(i in pool for i in pair) for pair in pairs):
                keep[row] = False
    return matrix[keep], active, np.asarray(forced, dtype=int), rounds


def _solve_cover(matrix, budget):
    """Solve independent residual components with a shared solver time budget.

    A time-limited incumbent is usable only after checking its actual integer
    coverage and total count. Failure is about this sparse certificate graph,
    not a proof that the full geometric problem is infeasible.
    """
    started = time.perf_counter()
    reduced, active, forced, rounds = _reduce_cover(matrix)
    stats = dict(reduction_rounds=rounds, forced=len(forced),
                 residual_variables=len(active), residual_constraints=reduced.shape[0], components=[])
    if len(forced) > budget:
        stats.update(status="certificate_budget_exceeded", seconds=time.perf_counter()-started)
        return None, stats
    chosen = forced.tolist()
    if reduced.shape[0]:
        rows, cols = reduced.nonzero()
        stars = reduced.indices[reduced.indptr[:-1]]
        adjacency = coo_matrix((np.ones(len(rows), dtype=bool), (cols, stars[rows])),
                               shape=(len(active), len(active))).tocsr()
        _, labels = connected_components(adjacency, directed=False)
        sizes = np.bincount(labels)
        remaining_seconds = 30.
        for group in np.argsort(-sizes, kind="stable"):
            local_cols = np.flatnonzero(labels == group)
            local_rows = np.flatnonzero(labels[stars] == group)
            if not len(local_rows):
                continue
            if len(local_cols) > _MAX_COMPONENT_VARIABLES:
                stats.update(status="component_resource_limit", seconds=time.perf_counter()-started)
                return None, stats
            if remaining_seconds <= 0:
                stats.update(status="solver_time_limit", seconds=time.perf_counter()-started)
                return None, stats
            local = reduced[local_rows][:, local_cols].astype(float).tocsc()
            start = time.perf_counter()
            result = milp(np.ones(len(local_cols)), integrality=np.ones(len(local_cols)),
                          bounds=Bounds(0, 1), constraints=LinearConstraint(local, 1, np.inf),
                          options={"time_limit": remaining_seconds, "mip_rel_gap": .005})
            elapsed = time.perf_counter()-start
            remaining_seconds -= elapsed
            stats["components"].append(dict(variables=len(local_cols), status=int(result.status),
                                               seconds=elapsed))
            if result.x is None or not np.all(local @ (result.x > .5) >= 1):
                stats.update(status="no_certified_solution", seconds=time.perf_counter()-started)
                return None, stats
            chosen.extend(active[local_cols[result.x > .5]])
    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) > budget:
        stats.update(status="certificate_budget_exceeded", count=len(chosen),
                      seconds=time.perf_counter()-started)
        return None, stats
    selected = np.zeros(matrix.shape[1], dtype=bool)
    selected[chosen] = True
    if len(set(chosen)) != len(chosen) or not np.all(matrix @ selected >= 1):
        raise RuntimeError("reduced cover failed the original constraints")
    stats.update(status="feasible", count=len(chosen), seconds=time.perf_counter()-started)
    return chosen, stats


class _WitnessRequirements:
    """Region witnesses satisfied by any member frame; none is pinned."""

    def __init__(self, incidence):
        self.rows = incidence.tocsr()
        self.columns = incidence.tocsc()
        self.covered = np.zeros(incidence.shape[1], dtype=bool)
        self.count = np.diff(self.rows.indptr).astype(np.int64)

    def add(self, row):
        ids = self.rows.indices[self.rows.indptr[row]:self.rows.indptr[row+1]]
        for column in ids[~self.covered[ids]]:
            self.count[self.columns.indices[self.columns.indptr[column]:self.columns.indptr[column+1]]] -= 1
            self.covered[column] = True

    def pending(self):
        return self.count > 0


def _select_state_coverage(candidate, plan, *, budget, policy="balanced", _prepared=None):
    """Witness cover first, then shared FPS; returns the selection and stats.

    If witness solving fails, FPS first reserves one frame per physical stratum,
    then targets pending witnesses. Unmet witnesses are reported.
    """
    if isinstance(budget, bool) or int(budget) != budget or budget <= 0:
        raise ValueError("sample budget must be a positive integer")
    started = time.perf_counter()
    budget = min(int(budget), candidate.row_count)
    if budget < plan.group_count:
        raise ValueError("sample budget is smaller than the number of uncovered physical strata "
                         f"({budget} < {plan.group_count})")
    if not any(name.startswith("input_") for name in candidate.names):
        # No input regions to detect: the shared FPS trajectory is the result.
        result = _select_coverage(candidate, plan, budget=budget, policy=policy, _prepared=_prepared)
        return result, dict(regions=0, witnesses=0, unmet_witnesses=0, cover=None,
                            total_seconds=time.perf_counter()-started)
    prepared = list(_prepare(candidate, plan, None, None) if _prepared is None else _prepared)
    _, inputs, supports, groups = _region_targets(candidate, plan, prepared, budget)
    n = candidate.row_count
    columns = [] if inputs is None else [inputs]
    if inputs is None:
        ranges = _RangeRequirements(candidate, groups, np.zeros(n, dtype=bool))
        columns.append(ranges.incidence)
    # Every observed physical stratum keeps at least one frame.
    rows = np.concatenate([np.asarray(r, dtype=int) for _, r in plan.groups])
    labels = np.repeat(np.arange(plan.group_count), [len(r) for _, r in plan.groups])
    columns.append(csr_matrix((np.ones(len(rows), dtype=bool), (rows, labels)),
                              shape=(n, plan.group_count)))
    incidence = hstack(columns, format="csr")
    requirements = _WitnessRequirements(incidence)
    # Content order, not input order, breaks ties between equivalent covers.
    order = prepared[4]
    chosen, cover = _solve_cover(incidence.T.tocsr()[:, order], budget)
    initial = () if chosen is None else tuple(int(i) for i in order[np.sort(chosen)])
    if chosen is not None:
        # The certified seeds already cover every physical stratum. Without
        # them, retain the original groups so FPS enforces that same floor.
        prepared[2] = groups
    result = _select_coverage(candidate, plan, budget=budget, policy=policy, _prepared=tuple(prepared),
                              _initial_indices=initial, _requirements=requirements)
    stats = dict(regions=len(supports), witnesses=incidence.shape[1],
                 unmet_witnesses=int(np.sum(~requirements.covered)),
                 cover={k: v for k, v in cover.items() if k in ("status", "count", "forced", "seconds")},
                 total_seconds=time.perf_counter()-started)
    return result, stats


def select_state_coverage(candidate, plan, *, budget, policy="balanced"):
    """Fixed-budget coverage used by the physics sampling path."""
    return _select_state_coverage(candidate, plan, budget=budget, policy=policy)[0]
