#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Torsion-based coordinate perturbations with fast bonding guards.
This module provides topology construction (with/without PBC), efficient
rotatable torsion detection, and guarded perturbations suitable for generating
physically plausible local conformations.
Examples
--------
>>> # See process_single for the main entrypoint
"""
from __future__ import annotations
import sys
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
from collections import deque, defaultdict
COVALENT_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24,
    'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39,
    'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
    'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96,
    'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87,
    'Hf': 1.75, 'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32,
    'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.40, 'At': 1.50, 'Rn': 1.50,
    'Fr': 2.60, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96, 'Np': 1.90, 'Pu': 1.87, 'Am': 1.80, 'Cm': 1.69
}
@dataclass
class TorsionGuardParams:
    perturb_per_frame: int = 100
    torsion_range_deg: Tuple[float, float] = (-180.0, 180.0)
    max_torsions_per_conf: int = 5
    gaussian_sigma: float = 0.03
    pbc_mode: str = "auto"  # "auto" | "yes" | "no"
    local_mode_cutoff_atoms: int = 150
    local_torsion_max_subtree: int = 40
    bond_detect_factor: float = 1.15
    bond_keep_min_factor: float = 0.60
    bond_keep_max_factor: Optional[float] = None
    nonbond_min_factor: float = 0.80
    max_retries_per_frame: int = 12
    mult_bond_factor: float = 0.87
    nonpbc_box_size: float = 100.0
    # Pauling bond order parameters
    bo_c_const: float = 0.3
    bo_threshold: float = 0.2
# ---------- Geometry and PBC helpers ----------
def rotate_coords(coords: np.ndarray, atom_indices: Iterable[int], axis_point1: np.ndarray,
                  axis_point2: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.array(axis_point2) - np.array(axis_point1)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return coords
    axis = axis / norm
    angle = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ux, uy, uz = axis
    R = np.array([
        [cos_a + ux * ux * (1 - cos_a), ux * uy * (1 - cos_a) - uz * sin_a, ux * uz * (1 - cos_a) + uy * sin_a],
        [uy * ux * (1 - cos_a) + uz * sin_a, cos_a + uy * uy * (1 - cos_a), uy * uz * (1 - cos_a) - ux * sin_a],
        [uz * ux * (1 - cos_a) - uy * sin_a, uz * uy * (1 - cos_a) + ux * sin_a, cos_a + uz * uz * (1 - cos_a)]
    ])
    idx = list(atom_indices)
    shifted = coords[idx] - axis_point1
    coords[idx] = shifted.dot(R.T) + axis_point1
    return coords
def center_in_box(coords: np.ndarray, box_size: float) -> np.ndarray:
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0
    box_center = np.array([box_size / 2.0] * 3)
    shift = box_center - center
    return coords + shift
def wrap_to_cell(coords: np.ndarray, cell: np.ndarray, inv_cell_T: np.ndarray) -> np.ndarray:
    frac = coords @ inv_cell_T
    frac = frac - np.floor(frac)
    return frac @ cell
def mic_delta(delta: np.ndarray, cell: np.ndarray, inv_cell_T: np.ndarray) -> np.ndarray:
    dfrac = delta @ inv_cell_T
    dfrac -= np.round(dfrac)
    return dfrac @ cell
# ---------- Grid utils (non-PBC only) ----------
def _grid_key(p: np.ndarray, inv_h: float, origin: np.ndarray) -> Tuple[int, int, int]:
    return tuple(((p - origin) * inv_h).astype(np.int64))  # type: ignore[return-value]
# ---------- Topology (built once per frame) ----------
def build_adjacency_nonpbc(symbols: Sequence[str], coords: np.ndarray, bond_detect_factor: float,
                           c_const: float = 0.3, bo_threshold: float = 0.2):
    N = len(symbols)
    radii = np.array([COVALENT_RADII.get(s, 0.77) for s in symbols], dtype=float)
    if N == 0:
        return [set()], {}, radii
    max_r = float(np.max(radii))
    h = bond_detect_factor * 2.0 * max_r + 1e-6
    inv_h = 1.0 / h
    origin = coords.min(axis=0) - 1e-6
    buckets: dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    keys = np.empty((N, 3), dtype=np.int64)
    for i in range(N):
        k = _grid_key(coords[i], inv_h, origin)
        keys[i] = k
        buckets[k].append(i)
    adj: List[set[int]] = [set() for _ in range(N)]
    edge_len: dict[Tuple[int, int], float] = {}
    edge_order: dict[Tuple[int, int], int] = {}
    neighbor_offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i in range(N):
        ri = radii[i]
        pi = coords[i]
        ki = tuple(keys[i])
        for dx, dy, dz in neighbor_offsets:
            k2 = (ki[0] + dx, ki[1] + dy, ki[2] + dz)
            if k2 not in buckets:
                continue
            for j in buckets[k2]:
                if j <= i:
                    continue
                rj = radii[j]
                pj = coords[j]
                rij2 = float(np.dot(pj - pi, pj - pi))
                rx = math.sqrt(rij2)
                r0 = (ri + rj)
                # Linus Pauling bond order
                bo = math.exp((r0 - rx) / float(c_const))
                if bo <= bo_threshold:
                    continue
                # classify bond order
                rounded = int(math.floor(bo + 0.5))
                order = 1 if rounded <= 1 else min(3, rounded)
                adj[i].add(j)
                adj[j].add(i)
                edge_len[(i, j)] = rx
                edge_order[(i, j)] = order
    return adj, edge_len, radii, edge_order
def build_adjacency_pbc(symbols: Sequence[str], coords: np.ndarray, cell: np.ndarray, bond_detect_factor: float,
                        c_const: float = 0.3, bo_threshold: float = 0.2):
    N = len(symbols)
    radii = np.array([COVALENT_RADII.get(s, 0.77) for s in symbols], dtype=float)
    if N == 0:
        return [set()], {}, radii
    inv_cell_T = np.linalg.inv(cell).T
    adj: List[set[int]] = [set() for _ in range(N)]
    edge_len: dict[Tuple[int, int], float] = {}
    edge_order: dict[Tuple[int, int], int] = {}
    for i in range(N):
        ri = radii[i]
        for j in range(i + 1, N):
            rj = radii[j]
            dvec = coords[j] - coords[i]
            dvec = mic_delta(dvec, cell, inv_cell_T)
            d2 = float(np.dot(dvec, dvec))
            rx = math.sqrt(d2)
            r0 = (ri + rj)
            bo = math.exp((r0 - rx) / float(c_const))
            if bo <= bo_threshold:
                continue
            rounded = int(math.floor(bo + 0.5))
            order = 1 if rounded <= 1 else min(3, rounded)
            adj[i].add(j)
            adj[j].add(i)
            edge_len[(i, j)] = rx
            edge_order[(i, j)] = order
    return adj, edge_len, radii, edge_order
# ---------- Fast torsion detection ----------
def get_bridges(adj: Sequence[set[int]]):
    sys.setrecursionlimit(max(100000, 10 * len(adj) + 100))
    n = len(adj)
    timer = 0
    tin = [-1] * n
    low = [-1] * n
    bridges: set[Tuple[int, int]] = set()
    def dfs(v: int, p: int):
        nonlocal timer
        tin[v] = low[v] = timer
        timer += 1
        for to in adj[v]:
            if to == p:
                continue
            if tin[to] != -1:
                low[v] = min(low[v], tin[to])
            else:
                dfs(to, v)
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    a, b = (v, to) if v < to else (to, v)
                    bridges.add((a, b))
    for v in range(n):
        if tin[v] == -1:
            dfs(v, -1)
    return bridges
def _prefer_neighbor(a: int, exclude_b: int, adj: Sequence[set[int]], symbols: Sequence[str]) -> Optional[int]:
    heavy = [n for n in adj[a] if n != exclude_b and symbols[n] != 'H']
    if heavy:
        return heavy[0]
    for n in adj[a]:
        if n != exclude_b:
            return n
    return None
def get_rotatable_torsions_fast(adj: Sequence[set[int]], edge_len: dict[Tuple[int, int], float],
                                radii: np.ndarray, symbols: Sequence[str], mult_bond_factor: float,
                                edge_order: Optional[dict[Tuple[int, int], int]] = None):
    """
    Prefer bridge edges; if none, fall back to all internal bonds (deg>1) excluding suspected multiple bonds.
    Returns List[(n1,a,b,n2)]
    """
    N = len(adj)
    deg = [len(adj[i]) for i in range(N)]
    bridges = get_bridges(adj)
    def build_from_edges(edges: Iterable[Tuple[int, int]]):
        torsions: list[Tuple[int, int, int, int]] = []
        for (a, b) in edges:
            if a > b:
                a, b = b, a
            d = edge_len.get((a, b))
            if d is None:
                continue
            if deg[a] <= 1 or deg[b] <= 1:
                continue
            # Exclude multiple bonds: prefer bond order if available, otherwise fallback to distance heuristic
            if edge_order is not None:
                if edge_order.get((a, b), 1) >= 2:
                    continue
            else:
                if d < mult_bond_factor * (radii[a] + radii[b]):
                    continue
            n1 = _prefer_neighbor(a, b, adj, symbols)
            n2 = _prefer_neighbor(b, a, adj, symbols)
            if n1 is None or n2 is None:
                continue
            torsions.append((n1, a, b, n2))
        return torsions
    torsions = build_from_edges(bridges)
    if not torsions:
        torsions = build_from_edges(edge_len.keys())
    return torsions
def get_local_subtree_adj(adj: Sequence[set[int]], start_atom: int, exclude_atom: int,
                          max_subtree_atoms: Optional[int] = None) -> set[int]:
    visited: set[int] = set()
    queue: deque[int] = deque([start_atom])
    while queue:
        atom = queue.popleft()
        if atom in visited:
            continue
        visited.add(atom)
        for ni in adj[atom]:
            if ni == exclude_atom or ni in visited:
                continue
            if max_subtree_atoms is not None and len(visited) >= max_subtree_atoms:
                return visited
            queue.append(ni)
    return visited
# ---------- Guards ----------
def bonds_within_range_nonpbc(coords: np.ndarray, bond_pairs: Sequence[Tuple[int, int]], radii: np.ndarray,
                              min_factor: float, max_factor: Optional[float], detect_factor: float) -> bool:
    maxf = detect_factor if max_factor is None else float(max_factor)
    minf = float(min_factor) if (min_factor is not None) else 0.0
    for (i, j) in bond_pairs:
        d = float(np.linalg.norm(coords[i] - coords[j]))
        ref = radii[i] + radii[j]
        if d > maxf * ref:
            return False
        if minf > 0.0 and d < minf * ref:
            return False
    return True
def bonds_within_range_pbc(coords: np.ndarray, bond_pairs: Sequence[Tuple[int, int]], radii: np.ndarray,
                           min_factor: float, max_factor: Optional[float], detect_factor: float,
                           cell: np.ndarray, inv_cell_T: np.ndarray) -> bool:
    maxf = detect_factor if max_factor is None else float(max_factor)
    minf = float(min_factor) if (min_factor is not None) else 0.0
    for (i, j) in bond_pairs:
        dvec = mic_delta(coords[j] - coords[i], cell, inv_cell_T)
        d = float(np.linalg.norm(dvec))
        ref = radii[i] + radii[j]
        if d > maxf * ref:
            return False
        if minf > 0.0 and d < minf * ref:
            return False
    return True
def nonbond_clash_free_fast_nonpbc(coords: np.ndarray, radii: np.ndarray, bonded_set: set[Tuple[int, int]],
                                   min_factor: float) -> bool:
    N = coords.shape[0]
    if N == 0 or min_factor <= 0.0:
        return True
    max_r = float(np.max(radii))
    h = min_factor * 2.0 * max_r + 1e-6
    inv_h = 1.0 / h
    origin = coords.min(axis=0) - 1e-6
    buckets: dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    keys = np.empty((N, 3), dtype=np.int64)
    for i in range(N):
        k = _grid_key(coords[i], inv_h, origin)
        keys[i] = k
        buckets[k].append(i)
    neighbor_offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i in range(N):
        ri = radii[i]
        pi = coords[i]
        ki = tuple(keys[i])
        for dx, dy, dz in neighbor_offsets:
            k2 = (ki[0] + dx, ki[1] + dy, ki[2] + dz)
            if k2 not in buckets:
                continue
            for j in buckets[k2]:
                if j <= i:
                    continue
                if (i, j) in bonded_set:
                    continue
                rj = radii[j]
                pj = coords[j]
                rij2 = float(np.dot(pj - pi, pj - pi))
                cutoff = min_factor * (ri + rj)
                if rij2 < cutoff * cutoff:
                    return False
    return True
def nonbond_clash_free_fast_pbc(coords: np.ndarray, radii: np.ndarray, bonded_set: set[Tuple[int, int]],
                                min_factor: float, cell: np.ndarray, inv_cell_T: np.ndarray) -> bool:
    N = coords.shape[0]
    if N == 0 or min_factor <= 0.0:
        return True
    for i in range(N):
        ri = radii[i]
        for j in range(i + 1, N):
            if (i, j) in bonded_set:
                continue
            rj = radii[j]
            dvec = mic_delta(coords[j] - coords[i], cell, inv_cell_T)
            d2 = float(np.dot(dvec, dvec))
            cutoff = min_factor * (ri + rj)
            if d2 < cutoff * cutoff:
                return False
    return True
# ---------- Perturbation ----------
def perturb_coords_fast(coords: np.ndarray, adj: Sequence[set[int]], torsions: Sequence[Tuple[int, int, int, int]],
                        degrees_range: Tuple[float, float], num_torsions: int,
                        sigma: float, local_mode: bool, max_subtree_atoms: Optional[int],
                        pbc_active: bool, cell: Optional[np.ndarray], inv_cell_T: Optional[np.ndarray]) -> np.ndarray:
    if torsions:
        chosen = random.sample(list(torsions), min(len(torsions), num_torsions))
        for (n1, a, b, n2) in chosen:
            if local_mode:
                subtree_atoms = get_local_subtree_adj(adj, b, a, max_subtree_atoms)
            else:
                subtree_atoms = get_local_subtree_adj(adj, b, a, None)
            angle = random.uniform(*degrees_range)
            if subtree_atoms:
                if pbc_active and cell is not None and inv_cell_T is not None:
                    dvec = mic_delta(coords[b] - coords[a], cell, inv_cell_T)
                    axis_p1 = coords[a]
                    axis_p2 = coords[a] + dvec
                else:
                    axis_p1 = coords[a]
                    axis_p2 = coords[b]
                coords = rotate_coords(coords, subtree_atoms, axis_p1, axis_p2, angle)
    if sigma > 0:
        coords = coords + np.random.normal(0.0, sigma, size=coords.shape)
    return coords
# ---------- Public API ----------
def process_single(symbols: Sequence[str],
                   coords: np.ndarray,
                   cell: Optional[np.ndarray],
                   params: TorsionGuardParams) -> list[tuple]:
    # Decide PBC activation for this frame
    has_cell = cell is not None
    if params.pbc_mode == "yes":
        pbc_active = bool(has_cell)
        if not pbc_active:
            print("[WARN] pbc=yes but no Lattice found in frame; falling back to non-PBC.", file=sys.stderr)
    elif params.pbc_mode == "no":
        pbc_active = False
    else:  # "auto"
        pbc_active = bool(has_cell)
    inv_cell_T = np.linalg.inv(cell).T if (pbc_active and has_cell and cell is not None) else None
    # Build topology once (fixed for this frame). Multiple disconnected molecules are allowed.
    if pbc_active:
        assert cell is not None
        adj, edge_len, radii, edge_order = build_adjacency_pbc(
            symbols, coords, cell,
            bond_detect_factor=params.bond_detect_factor,
            c_const=float(params.bo_c_const), bo_threshold=float(params.bo_threshold)
        )
    else:
        adj, edge_len, radii, edge_order = build_adjacency_nonpbc(
            symbols, coords,
            bond_detect_factor=params.bond_detect_factor,
            c_const=float(params.bo_c_const), bo_threshold=float(params.bo_threshold)
        )
    bond_pairs = [(a, b) if a < b else (b, a) for (a, b) in edge_len.keys()]
    bonded_set = set(bond_pairs)
    # Torsion set (always fast method with fallback to internal bonds)
    torsions = get_rotatable_torsions_fast(adj, edge_len, radii, symbols, params.mult_bond_factor, edge_order=edge_order)
    local_mode_flag = len(symbols) > params.local_mode_cutoff_atoms
    results: list[tuple] = []
    for _ in range(int(params.perturb_per_frame)):
        new_coords = coords.copy()
        success = False
        for attempt in range(int(params.max_retries_per_frame) + 1):
            scale = 1.0 if attempt == 0 else (0.5 ** attempt)
            ang_lo, ang_hi = params.torsion_range_deg
            angle_range = (float(ang_lo) * scale, float(ang_hi) * scale)
            sigma_scaled = float(params.gaussian_sigma) * scale
            new_coords = coords.copy()
            new_coords = perturb_coords_fast(
                new_coords, adj, torsions,
                angle_range, int(params.max_torsions_per_conf),
                sigma_scaled, local_mode_flag, int(params.local_torsion_max_subtree),
                pbc_active, cell if cell is not None else None, inv_cell_T
            )
            # Guards: only original bonded pairs are enforced; non-bonded pairs kept apart
            if pbc_active and cell is not None and inv_cell_T is not None:
                if not bonds_within_range_pbc(
                    new_coords, bond_pairs, radii,
                    float(params.bond_keep_min_factor), params.bond_keep_max_factor, float(params.bond_detect_factor),
                    cell, inv_cell_T
                ):
                    continue
                if not nonbond_clash_free_fast_pbc(
                    new_coords, radii, bonded_set, float(params.nonbond_min_factor),
                    cell, inv_cell_T
                ):
                    continue
            else:
                if not bonds_within_range_nonpbc(
                    new_coords, bond_pairs, radii,
                    float(params.bond_keep_min_factor), params.bond_keep_max_factor, float(params.bond_detect_factor)
                ):
                    continue
                if not nonbond_clash_free_fast_nonpbc(
                    new_coords, radii, bonded_set, float(params.nonbond_min_factor)
                ):
                    continue
            success = True
            break
        if not success:
            new_coords = coords.copy()
        if pbc_active and cell is not None and inv_cell_T is not None:
            new_coords = wrap_to_cell(new_coords, cell, inv_cell_T)
        else:
            new_coords = center_in_box(new_coords, float(params.nonpbc_box_size))
        results.append((list(symbols), new_coords, cell, pbc_active))
    return results
