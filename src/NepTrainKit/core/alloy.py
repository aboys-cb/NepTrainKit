"""Utilities for building alloy datasets (compositions, occupancies, prototypes)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from ase import Atoms
from scipy.stats.qmc import Sobol


_ELEMENT_SPLIT_RE = re.compile(r"[,\s;]+")


def parse_element_list(text: str) -> list[str]:
    """Parse an element list like ``Co,Cr,Ni`` or ``Co Cr Ni`` into symbols."""
    if not text:
        return []
    raw = [t.strip() for t in _ELEMENT_SPLIT_RE.split(text.strip()) if t.strip()]
    out: list[str] = []
    for token in raw:
        if not token:
            continue
        symbol = token[0].upper() + token[1:].lower()
        if symbol not in out:
            out.append(symbol)
    return out


def parse_composition(text: str) -> dict[str, float]:
    """Parse a composition mapping from JSON or ``A:0.2,B:0.8`` syntax."""
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("{") and text.endswith("}"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError("Composition JSON must be an object mapping element->fraction.")
        return {str(k): float(v) for k, v in data.items()}
    parts = [p.strip() for p in text.split(",") if p.strip()]
    comp: dict[str, float] = {}
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid composition token: {part!r}")
        elem, val = part.split(":", 1)
        elem = elem.strip()
        if not elem:
            continue
        symbol = elem[0].upper() + elem[1:].lower()
        comp[symbol] = float(val)
    return comp


def normalize_composition(comp: Mapping[str, float]) -> dict[str, float]:
    """Return a normalized copy of ``comp`` whose values sum to 1."""
    items = [(k, float(v)) for k, v in comp.items() if float(v) > 0.0]
    if not items:
        return {}
    total = sum(v for _, v in items)
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in items}


def _is_near_rational_step(step: float, tol: float = 1e-9) -> tuple[bool, int]:
    if step <= 0:
        return False, 0
    n = int(round(1.0 / float(step)))
    if n <= 0:
        return False, 0
    if abs(step - 1.0 / n) <= tol:
        return True, n
    return False, n


def simplex_grid_points(
    order: int,
    step: float,
    *,
    include_endpoints: bool = True,
    min_fraction: float = 0.0,
) -> list[tuple[float, ...]]:
    """Generate simplex grid points for ``order`` components.

    Notes
    -----
    - For binary, this is a 1D grid over ``x`` with ``(x, 1-x)``.
    - For ternary, points satisfy ``x+y+z=1``.
    """
    if order < 2:
        raise ValueError("order must be >= 2")
    step = float(step)
    min_fraction = float(min_fraction)
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1].")
    if min_fraction < 0 or min_fraction > 1:
        raise ValueError("min_fraction must be in [0, 1].")

    near, n = _is_near_rational_step(step)
    points: list[tuple[float, ...]] = []

    if order == 2:
        if near:
            xs = [i / n for i in range(n + 1)]
        else:
            xs = list(np.arange(0.0, 1.0 + 1e-12, step, dtype=float))
        for x in xs:
            y = 1.0 - float(x)
            if not include_endpoints and (x <= 0.0 or y <= 0.0):
                continue
            if x + 1e-12 < min_fraction or y + 1e-12 < min_fraction:
                continue
            points.append((float(x), float(y)))
        return points

    if order == 3:
        if near:
            for i in range(n + 1):
                for j in range(n + 1 - i):
                    k = n - i - j
                    x, y, z = i / n, j / n, k / n
                    if not include_endpoints and (x <= 0.0 or y <= 0.0 or z <= 0.0):
                        continue
                    if x + 1e-12 < min_fraction or y + 1e-12 < min_fraction or z + 1e-12 < min_fraction:
                        continue
                    points.append((float(x), float(y), float(z)))
            return points

        vals = list(np.arange(0.0, 1.0 + 1e-12, step, dtype=float))
        for x in vals:
            for y in vals:
                z = 1.0 - float(x) - float(y)
                if z < -1e-10:
                    continue
                if z < 0.0:
                    z = 0.0
                if not include_endpoints and (x <= 0.0 or y <= 0.0 or z <= 0.0):
                    continue
                if x + 1e-12 < min_fraction or y + 1e-12 < min_fraction or z + 1e-12 < min_fraction:
                    continue
                s = float(x) + float(y) + float(z)
                if s <= 0:
                    continue
                points.append((float(x) / s, float(y) / s, float(z) / s))
        return points

    # For higher orders, support integer-grid steps only (step ~= 1/n).
    if not near:
        raise NotImplementedError("Grid simplex points for order>=4 require step ~= 1/n.")

    min_each = 0
    if not include_endpoints:
        min_each = 1
    if min_fraction > 0.0:
        min_each = max(min_each, int(np.ceil(min_fraction * n - 1e-12)))
    if min_each * order > n:
        return []

    def _compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
        if parts == 1:
            yield (total,)
            return
        start = min_each
        end = total - min_each * (parts - 1)
        for i in range(start, end + 1):
            for rest in _compositions(total - i, parts - 1):
                yield (i,) + rest

    for ints in _compositions(n, order):
        fracs = tuple(float(v) / float(n) for v in ints)
        # min_each already enforced; still guard floating errors.
        if any(f + 1e-12 < min_fraction for f in fracs):
            continue
        points.append(fracs)
    return points


def simplex_sobol_points(
    order: int,
    n_points: int,
    *,
    seed: int | None = None,
    min_fraction: float = 0.0,
) -> list[tuple[float, ...]]:
    """Generate low-discrepancy points on a simplex using Sobol + sorting map."""
    if order < 2:
        raise ValueError("order must be >= 2")
    if n_points <= 0:
        return []
    min_fraction = float(min_fraction)
    engine = Sobol(d=order - 1, scramble=True, seed=seed)
    u = engine.random(n_points)
    out: list[tuple[float, ...]] = []
    for row in u:
        cuts = sorted(float(x) for x in row.tolist())
        parts = []
        prev = 0.0
        for c in cuts:
            parts.append(c - prev)
            prev = c
        parts.append(1.0 - prev)
        fracs = tuple(float(x) for x in parts)
        if any(f + 1e-12 < min_fraction for f in fracs):
            continue
        s = float(sum(fracs))
        if s <= 0:
            continue
        out.append(tuple(f / s for f in fracs))
    return out


def fractions_to_counts_exact(fractions: Iterable[float], n_sites: int) -> np.ndarray:
    """Convert fractions to integer counts that sum to ``n_sites`` (largest remainder)."""
    fractions_arr = np.asarray(list(fractions), dtype=float)
    fractions_arr = np.clip(fractions_arr, 0.0, None)
    total = float(fractions_arr.sum())
    if total <= 0.0:
        raise ValueError("Fractions sum to zero.")
    fractions_arr = fractions_arr / total

    raw = fractions_arr * int(n_sites)
    counts = np.floor(raw).astype(int)
    remain = int(n_sites) - int(counts.sum())
    if remain > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remain]:
            counts[int(idx)] += 1
    elif remain < 0:
        order = np.argsort(raw - counts)  # smallest remainder first
        for idx in order[: (-remain)]:
            if counts[int(idx)] > 0:
                counts[int(idx)] -= 1
    if int(counts.sum()) != int(n_sites):
        raise RuntimeError("Failed to allocate exact counts.")
    return counts


def assign_random_occupancy(
    atoms: Atoms,
    composition: Mapping[str, float],
    *,
    indices: np.ndarray | None = None,
    mode: str = "Exact",
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Assign species to ``atoms`` for the selected indices according to composition."""
    if rng is None:
        rng = np.random.default_rng()
    comp = normalize_composition(composition)
    if not comp:
        return atoms

    symbols = list(comp.keys())
    fractions = np.array([comp[s] for s in symbols], dtype=float)

    if indices is None:
        indices = np.arange(len(atoms), dtype=int)
    indices = np.asarray(indices, dtype=int)
    n_sites = int(len(indices))
    if n_sites <= 0:
        return atoms

    if mode.lower() == "random":
        counts = rng.multinomial(n_sites, fractions / fractions.sum())
    else:
        counts = fractions_to_counts_exact(fractions, n_sites)

    pool = np.concatenate([np.repeat(sym, int(c)) for sym, c in zip(symbols, counts) if int(c) > 0])
    if len(pool) != n_sites:
        raise RuntimeError("Occupancy pool length mismatch.")
    rng.shuffle(pool)

    new_atoms = atoms.copy()
    chem = new_atoms.get_chemical_symbols()
    for idx, sym in zip(indices.tolist(), pool.tolist()):
        chem[int(idx)] = sym
    new_atoms.set_chemical_symbols(chem)
    return new_atoms


@dataclass(frozen=True)
class SupercellFactors:
    na: int
    nb: int
    nc: int


def best_supercell_factors_max_atoms(atoms: Atoms, max_atoms: int) -> SupercellFactors:
    """Choose (na, nb, nc) that maximizes atom count within ``max_atoms`` with a shape tie-breaker."""
    max_atoms = int(max_atoms)
    if max_atoms <= 0:
        return SupercellFactors(1, 1, 1)
    base_n = len(atoms)
    if base_n <= 0:
        return SupercellFactors(1, 1, 1)
    if base_n >= max_atoms:
        return SupercellFactors(1, 1, 1)

    max_n = max(int(max_atoms // base_n), 1)
    a_len, b_len, c_len = atoms.cell.lengths()
    base_lengths = np.array([a_len, b_len, c_len], dtype=float)

    best: tuple[int, float, tuple[int, int, int]] | None = None
    for na in range(1, max_n + 1):
        for nb in range(1, max_n + 1):
            for nc in range(1, max_n + 1):
                total = base_n * na * nb * nc
                if total > max_atoms:
                    continue
                lengths = base_lengths * np.array([na, nb, nc], dtype=float)
                aspect = float(lengths.max() / max(lengths.min(), 1e-12))
                score = (int(total), -aspect, (na, nb, nc))
                if best is None or score > best:
                    best = score
    if best is None:
        return SupercellFactors(1, 1, 1)
    _, _, (na, nb, nc) = best
    return SupercellFactors(int(na), int(nb), int(nc))
