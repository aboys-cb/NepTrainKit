"""Atomistic sampling fixtures and a selector-independent coverage oracle.

The truth arrays never enter the XYZ or the sampler. These are declared
coverage tasks, not claims that a finite subset guarantees training accuracy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.io import write
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist


def make_cases(*, variant=0):
    """Original five tasks, or a prespecified changed-grid/material challenge.

    Variant 0 reproduces the original toy definitions. Variant 1 changes
    grid sizes, budgets, physical ranges and the nonmagnetic element before
    evaluating any selector; it is a transfer check, not a blind test set.
    """
    if variant not in (0, 1):
        raise ValueError("variant must be 0 or 1")
    cases = []
    symbol = "Cu" if variant == 0 else "Al"
    lattice_scale = 1.0 if variant == 0 else 1.13

    def cell(phase="fcc", strain=(1., 1., 1.), vacancy=False, magnetic=False):
        atoms = bulk("Fe" if magnetic else symbol, phase,
                     a=(3.6 if phase == "fcc" else 2.9) * lattice_scale,
                     cubic=True).repeat((2, 2, 2))
        atoms.arrays.pop("initial_magmoms", None)
        atoms.set_cell(atoms.cell.array @ np.diag(strain), scale_atoms=True)
        if vacancy:
            del atoms[0]
        return atoms

    def add(name, states, coordinates, cells, repeats, reference, criteria, magnetic=False):
        state_ids = np.repeat(np.arange(len(states)), repeats)
        state_ids = state_ids[np.random.default_rng(621 + variant).permutation(len(state_ids))]
        frames = [states[i].copy() for i in state_ids]
        name += "_transfer" if variant else ""
        truth = dict(name=name, magnetic=magnetic, variant=variant, count=len(frames),
                     unique_states=len(states), budget=len(reference),
                     state_id=state_ids.tolist(), coordinates=np.asarray(coordinates)[state_ids].tolist(),
                     required_cell=np.asarray(cells)[state_ids].tolist(),
                     oracle_indices=[int(np.flatnonzero(state_ids == i)[0]) for i in reference],
                     criteria=criteria)
        cases.append((frames, truth))

    size = 61 if not variant else 81
    x = np.linspace(0, 1, size)
    lower, span = (.9, .2) if not variant else (.84, .27)
    states = [cell(strain=(lower + span*t,) * 3) for t in x]
    reference = np.rint(np.linspace(0, size-1, 11 if not variant else 13)).astype(int)
    for name, repeats in (("path_balanced", np.ones(size, int)),
                          ("path_dense_middle", np.where((x >= .35) & (x <= .65), 25, 1))):
        add(name, states, x[:, None], np.minimum((x * len(reference)).astype(int),
                                                len(reference)-1).astype(str), repeats, reference,
            dict(require_endpoints=True, require_all_cells=False,
                 max_latent_gap=.05 if not variant else 3/(size-1)))

    states, coordinates, cells, repeats, reference = [], [], [], [], []
    size, bins = (21, 7) if not variant else (25, 9)
    branch_reference = list(range(1 if not variant else 0, size, 3))
    for branch in range(3):
        for i, t in enumerate(np.linspace(0, 1, size)):
            strain = (.92 + .16*t) if not variant else (.87 + .23*t)
            states.append(cell("bcc" if branch == 1 else "fcc", (strain,) * 3,
                               vacancy=branch == 2))
            coordinates.append([t])
            cells.append(f"{branch}:{min(bins-1, int(t*bins))}")
            repeats.append(20 if branch == 0 else 1)
            if i in branch_reference:
                reference.append(len(states)-1)
    add("rare_branches", states, coordinates, cells, repeats, reference,
        dict(require_all_cells=True, separate_branches=True, max_latent_gap=1/(size-1)))

    states, coordinates, cells, repeats, reference = [], [], [], [], []
    size = 7 if not variant else 9
    bins = (size+1)//2
    for i, u in enumerate(np.linspace(0, 1, size)):
        for j, v in enumerate(np.linspace(0, 1, size)):
            # Disjoint principal-stretch ranges avoid treating axis swaps as
            # distinct physical states. No cell basis coordinates go to truth.
            stretch = (.85+.1*u, 1.05+.1*v, 1.3) if not variant else (.8+.12*u, 1.03+.14*v, 1.37)
            states.append(cell(strain=stretch))
            coordinates.append([u, v])
            cells.append(f"{min(bins-1, int(u*bins))}:{min(bins-1, int(v*bins))}")
            repeats.append(20 if i == size//2 or j == size//2 else 1)
            if i % 2 == 0 and j % 2 == 0:
                reference.append(len(states)-1)
    add("joint_strain", states, coordinates, cells, repeats, reference,
        dict(require_all_cells=True, require_endpoints=True, max_latent_gap=np.sqrt(2)/(size-1)))

    states, coordinates, cells, repeats, reference = [], [], [], [], []
    amplitude_size, angle_size = (5, 7) if not variant else (7, 9)
    for i, u in enumerate(np.linspace(0, 1, amplitude_size)):
        for j, v in enumerate(np.linspace(0, 1, angle_size)):
            atoms = cell("bcc", magnetic=True)
            moment = .5+2*u if not variant else .7+1.4*u
            spin = np.tile([0., 0., moment], (len(atoms), 1))
            spin[1::2] = moment * np.array([np.sin(np.pi*v), 0, np.cos(np.pi*v)])
            atoms.new_array("spin", spin)
            states.append(atoms)
            coordinates.append([u, v])
            m, a = (amplitude_size+1)//2, (angle_size+1)//2
            cells.append(f"{min(m-1, int(u*m))}:{min(a-1, int(v*a))}")
            repeats.append(15 if i == amplitude_size//2 or j == angle_size//2 else 1)
            if i % 2 == 0 and j % 2 == 0:
                reference.append(len(states)-1)
    add("spin_amplitude_angle", states, coordinates, cells, repeats, reference,
        dict(require_all_cells=True, require_endpoints=True,
             max_latent_gap=np.hypot(1/(amplitude_size-1), 1/(angle_size-1))), True)
    return cases


def make_challenges(*, variant=0):
    """Frozen non-ideal input tasks; no selector output defines their truth.

    References/budgets inherit feasible latent-space tasks. Thermal disorder
    is a nuisance variable here, not a claim about its energy/force relevance.
    """
    from copy import deepcopy

    originals = make_cases(variant=variant)
    result = []
    frames, truth = deepcopy(originals[0])
    for atoms, coordinate in zip(frames, truth["coordinates"]):
        t = coordinate[0]
        lengths = ([7.2*(.86+.1*t), 7.2*(1.1+.06*np.sin(np.pi*t)), 9.36] if not variant else
                   [8.136*(.82+.15*t), 8.136*(1.16+.04*np.sin(2*np.pi*t)), 11.])
        atoms.set_cell(np.diag(lengths), scale_atoms=True)
    truth["name"] = "curved_path"
    result.append((frames, truth))

    frames, truth = deepcopy(originals[3])
    # Missing interior odd/odd grid sites, never the declared even/even
    # reference. Retain original task cells, radius and budget.
    states = np.asarray(truth["state_id"])
    size = 7 if not variant else 9
    keep = ~((states//size % 2 == 1) & (states % size % 2 == 1))
    mapping = np.cumsum(keep)-1
    truth["oracle_indices"] = [int(mapping[i]) for i in truth["oracle_indices"]]
    for key in ("state_id", "coordinates", "required_cell"):
        truth[key] = np.asarray(truth[key])[keep].tolist()
    frames = [f for f, retained in zip(frames, keep) if retained]
    truth.update(name="incomplete_grid", count=len(frames), unique_states=len(set(truth["state_id"])))
    result.append((frames, truth))

    frames, truth = deepcopy(originals[3])
    for atoms, state in zip(frames, truth["state_id"]):
        noise = np.random.default_rng(7321+state+1000*variant).normal(
            scale=.003 if not variant else .005, size=atoms.positions.shape)
        atoms.positions += noise-noise.mean(0)
    truth["name"] = "thermally_perturbed_grid"
    result.append((frames, truth))

    base = originals[0][0][0]
    frames, coordinates, cells, reference = [], [], [], []
    size = 41 if not variant else 51
    for branch in range(2):
        for i, t in enumerate(np.linspace(0, 1, size)):
            atoms = base.copy()
            atoms.set_cell(np.eye(3)*(7.2*(.9+.2*t) if not variant else 8.136*(.84+.27*t)),
                           scale_atoms=True)
            if branch:
                displacement = [.24, .08, -.12] if not variant else [.18, .14, -.08]
                atoms.positions[0] += displacement
                atoms.positions[1] -= displacement
            if i % 5 == 0:
                reference.append(len(frames))
            frames.append(atoms)
            coordinates.append([float(t)])
            bins = (size-1)//5+1
            cells.append(f"{branch}:{min(bins-1, int(t*bins))}")
    truth = dict(name="local_motif_same_cell", magnetic=False, count=len(frames), unique_states=len(frames),
                 budget=len(reference), state_id=list(range(len(frames))), coordinates=coordinates,
                 required_cell=cells, oracle_indices=reference,
                 criteria=dict(require_all_cells=True, separate_branches=True,
                               require_endpoints=True, max_latent_gap=2/(size-1)))
    result.append((frames, truth))
    # Independent perturbations for every repeated observation prevent an
    # algorithm from passing solely by exact deduplication of noisy states.
    frames, truth = deepcopy(originals[3])
    for row, atoms in enumerate(frames):
        noise = np.random.default_rng(510009+row+1000*variant).normal(
            scale=.003 if not variant else .005, size=atoms.positions.shape)
        atoms.positions += noise-noise.mean(0)
    truth["name"] = "thermal_independent_repeats"
    result.append((frames, truth))
    if variant:
        for _, truth in result:
            truth["name"] += "_transfer"
    return result


def make_mixed_cases(*, variant=0):
    """Frozen allocation tasks; truth is never attached to sampler inputs."""
    if variant not in (0, 1):
        raise ValueError("variant must be 0 or 1")
    symbol = "Cu" if not variant else "Al"
    suffix = "_transfer" if variant else ""
    cases = []

    def record(name, frames, coordinates, cells, reference, radius):
        count = len(frames)
        order = np.random.default_rng(811+variant).permutation(count)
        reverse = np.argsort(order)
        truth = dict(name=name+suffix, magnetic=False, count=count, unique_states=count,
                     budget=len(reference), state_id=order.tolist(),
                     coordinates=np.asarray(coordinates)[order].tolist(),
                     required_cell=np.asarray(cells)[order].tolist(),
                     oracle_indices=reverse[reference].tolist(),
                     criteria=dict(separate_branches=True, require_endpoints=True,
                                   require_all_cells=False, max_latent_gap=radius))
        cases.append(([frames[i] for i in order], truth))

    frames, coordinates, cells, reference = [], [], [], []
    for branch, (phase, size, seats) in enumerate(
            (("fcc", 121, 11), ("bcc", 21, 11)) if not variant else
            (("fcc", 201, 17), ("bcc", 31, 16))):
        base = bulk(symbol, phase, a=3.6 if phase == "fcc" else 2.9, cubic=True).repeat((2, 2, 2))
        picks = set(np.rint(np.linspace(0, size-1, seats)).astype(int))
        for i, t in enumerate(np.linspace(0., 1., size)):
            atoms = base.copy()
            atoms.set_cell(base.cell.array*(.9+.2*t), scale_atoms=True)
            if i in picks:
                reference.append(len(frames))
            frames.append(atoms)
            coordinates.append([t])
            cells.append(f"{branch}:{i}")
    record("mixed_path_density", frames, coordinates, cells, reference, .05 if not variant else .035)

    frames, coordinates, cells, reference = [], [], [], []
    size, seats, islands = (61, 11, 12) if not variant else (81, 13, 20)
    base = bulk(symbol, "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    picks = set(np.rint(np.linspace(0, size-1, seats)).astype(int))
    for i, t in enumerate(np.linspace(0., 1., size)):
        atoms = base.copy()
        atoms.set_cell(base.cell.array*(.9+.2*t), scale_atoms=True)
        if i in picks:
            reference.append(len(frames))
        frames.append(atoms)
        coordinates.append([t])
        cells.append(f"path:{i}")
    for i in range(islands):
        atoms = bulk(symbol, "bcc", a=2.9, cubic=True).repeat((2, 2, 2))
        noise = np.random.default_rng(89001+100*variant+i).normal(scale=.23, size=atoms.positions.shape)
        atoms.positions += noise-noise.mean(0)
        reference.append(len(frames))
        frames.append(atoms)
        coordinates.append([0.])
        # Each island is a distinct prescribed local state, not a trajectory
        # density multiplier. Missing any island fails the branch gap test.
        cells.append(f"island{i}:0")
    record("mixed_path_islands", frames, coordinates, cells, reference,
           .05 if not variant else .0375)
    return cases


def write_case(directory, frames, truth):
    """Write label-free XYZ and a separate ground-truth sidecar."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        if frame.info or set(frame.arrays) - {"numbers", "positions", "spin"} or frame.calc is not None:
            raise ValueError("toy frames must contain only structural inputs")
    path = directory / f"{truth['name']}.xyz"
    write(path, frames, format="extxyz", write_results=False)
    truth = dict(truth, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".oracle.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return truth


def _arrays(truth):
    points = np.asarray(truth["coordinates"], dtype=float)
    cells = np.asarray(truth["required_cell"])
    states = np.asarray(truth["state_id"])
    branches = (np.asarray([c.split(":")[0] for c in cells])
                if truth["criteria"].get("separate_branches") else np.full(len(points), "all"))
    return points, cells, states, branches


def evaluate(truth, indices):
    """Report distance, grid occupancy and endpoints independently."""
    points, cells, states, branches = _arrays(truth)
    ids = np.asarray(indices)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(points)):
        raise ValueError("selection must be nonempty, unique, integral and in range")
    distance = np.full(len(points), np.inf)
    for branch in np.unique(branches):
        rows = np.flatnonzero(branches == branch)
        chosen = ids[branches[ids] == branch]
        if len(chosen):
            distance[rows] = cdist(points[rows], points[chosen]).min(axis=1)
    missing = sorted(set(cells) - set(cells[ids]))
    endpoints = bool(np.allclose(points[ids].min(0), points.min(0), rtol=0, atol=1e-12)
                     and np.allclose(points[ids].max(0), points.max(0), rtol=0, atol=1e-12))
    gap = float(distance.max())
    criteria = truth["criteria"]
    passed = (len(ids) == truth["budget"] and (not criteria.get("require_all_cells") or not missing)
              and gap <= criteria["max_latent_gap"] + 1e-10
              and (not criteria.get("require_endpoints") or endpoints))
    return dict(count=len(ids), unique_states=len(set(states[ids])),
                cells_covered=len(set(cells))-len(missing), cells_total=len(set(cells)),
                missing_cells=missing, gap_max=gap if np.isfinite(gap) else None,
                endpoints_preserved=endpoints, passed=bool(passed),
                selected_state_ids=sorted(set(states[ids].tolist())))


def minimum_budget(truth, *, mandatory_states=()):
    """Certify toy feasibility; never used by a sampler or on real datasets.

    Solve finite binary set cover in the independent truth coordinates. A
    timeout is unknown, not proof of infeasibility. Mandatory state IDs allow
    diagnosing conflicts introduced by a selector's additional constraints.
    """
    points, cells, states, branches = _arrays(truth)
    unique_states, rows = np.unique(states, return_index=True)
    points, cells, branches = points[rows], cells[rows], branches[rows]
    distance = cdist(points, points)
    matrix = [(distance <= truth["criteria"]["max_latent_gap"] + 1e-10)
              & (branches[:, None] == branches[None, :])]
    if truth["criteria"].get("require_all_cells"):
        matrix.append(np.asarray([cells == c for c in np.unique(cells)]))
    if truth["criteria"].get("require_endpoints"):
        for j in range(points.shape[1]):
            matrix.append(np.asarray([np.isclose(points[:, j], extreme, rtol=0, atol=1e-12)
                                      for extreme in (points[:, j].min(), points[:, j].max())]))
    for state in mandatory_states:
        if state not in unique_states:
            raise ValueError("mandatory state absent from toy")
        matrix.append((unique_states == state)[None, :])
    matrix = np.vstack(matrix).astype(float)
    result = milp(np.ones(len(rows)), integrality=np.ones(len(rows)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(csr_matrix(matrix), np.ones(len(matrix)), np.inf),
                  options={"time_limit": 10})
    if not result.success:
        return dict(status="infeasible" if result.status == 2 else "unknown", message=result.message)
    chosen = np.flatnonzero(result.x > .5)
    # Check the integer witness independently of the optimizer's status.
    if not np.all(matrix[:, chosen].sum(axis=1) >= 1):
        raise RuntimeError("solver returned an invalid coverage certificate")
    return dict(status="optimal", minimum=len(chosen),
                state_ids=unique_states[chosen].tolist(), fits_budget=len(chosen) <= truth["budget"])
