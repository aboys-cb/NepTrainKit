"""Feature construction for representative whole-dataset sampling.

Preserve whole-structure features and the actual atomic environments. The
coverage engine fits shared scales and evaluates each view without allowing
one high-variance view to hide another.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from NepTrainKit.core.audit.magnetic_inventory import _spin_array, analyze_structure_magnetism
from NepTrainKit.core.geometry_cache import structure_cell_array, structure_pbc_flags

_SCALE_EPS = 1.0e-12


def _input_environment_reference(positions, cell, pbc, spins):
    """ASE fallback for the native, model-independent input fingerprint."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(positions=positions, cell=cell, pbc=pbc)
    n = len(atoms)
    cutoff = max(abs(np.linalg.det(cell)) / max(1, n), 1.0) ** (1 / 3) * 2
    for _ in range(16):
        centers, sources, vectors = neighbor_list("ijD", atoms, cutoff)
        distances = np.linalg.norm(vectors, axis=1)
        nonzero = distances > 1e-12
        centers, sources, vectors, distances = (
            value[nonzero] for value in (centers, sources, vectors, distances))
        counts = np.bincount(centers, minlength=n)
        required = 12 if np.any(pbc) else min(12, max(0, n - 1))
        if np.all(counts >= required):
            break
        cutoff *= 1.7
    output = np.zeros((n, 17 if spins is not None else 12))
    magnitudes = None if spins is None else np.linalg.norm(spins, axis=1)
    for i in range(n):
        edges = np.flatnonzero(centers == i)
        edges = edges[np.argsort(distances[edges])]
        if len(edges):
            boundary = distances[edges[min(11, len(edges) - 1)]]
            edges = edges[distances[edges] <= boundary + max(1.0, boundary) * 1e-7]
            output[i, :12] = distances[edges[np.minimum(np.arange(12), len(edges) - 1)]]
        if spins is None:
            continue
        output[i, 12] = magnitudes[i]
        if magnitudes[i] <= 1e-8 or not len(edges):
            continue
        direction = spins[i] / magnitudes[i]
        projection = (vectors[edges] @ direction) / distances[edges]
        output[i, 14:] = [np.mean(projection ** power) for power in (2, 4, 6)]
        other = sources[edges]
        unit = spins[other] / np.maximum(magnitudes[other, None], 1e-8)
        unit[magnitudes[other] <= 1e-8] = 0.0
        output[i, 13] = np.mean(unit @ direction)
    return output


def _input_environment(structure, spin_model):
    """Read only coordinates, cell/PBC, and canonical spin inputs."""
    from ase.cell import Cell

    positions = np.asarray(structure.positions, dtype=np.float64)
    pbc = np.asarray(structure_pbc_flags(structure), dtype=bool)
    cell = structure_cell_array(structure)
    if not np.all(pbc):
        # Vacuum vectors are not part of the periodic geometry and must not
        # determine the neighbor-search radius (or make a slab singular).
        cell = cell.copy()
        cell[~pbc] = 0.0
        cell = Cell(cell).complete().array
    spins = _spin_array(structure, len(structure)) if spin_model else None
    try:
        from NepTrainKit._native import _sampling
    except ImportError:
        _sampling = None
    if _sampling is not None and hasattr(_sampling, "input_environment_features"):
        return _sampling.input_environment_features(positions, cell, pbc, spins)
    return _input_environment_reference(positions, cell, pbc, spins)


def _input_feature_blocks(structures, spin_model):
    """Element-resolved distributions retain small sublattice/scan changes.

    Geometry uses the first twelve neighbor radii; spin inputs additionally
    retain magnitudes, relative alignment and orientation relative to bonds.
    Min/median/max summaries are atom-order and supercell invariant. Completing
    the outer neighbor shell avoids breaking spatial rotation invariance.
    """
    rows = []
    for structure in structures:
        symbols = np.asarray(structure.get_chemical_symbols()
                             if hasattr(structure, "get_chemical_symbols") else structure.elements)
        local = _input_environment(structure, spin_model)
        views = {"geometry": local[:, :12]}
        if spin_model:
            views["moment"] = local[:, 12:13] if local.shape[1] == 17 else np.zeros((len(local), 1))
            views["orientation"] = local[:, 13:] if local.shape[1] == 17 else np.zeros((len(local), 4))
        row = {}
        for symbol in sorted(set(symbols)):
            for name, values in views.items():
                row[f"input_{name}:{symbol}"] = np.quantile(
                    values[symbols == symbol], (0, .5, 1), axis=0).ravel()
        rows.append(row)
    names = sorted({name for row in rows for name in row})
    values = []
    for name in names:
        width = next(len(row[name]) for row in rows if name in row)
        values.append(np.vstack([row.get(name, np.zeros(width)) for row in rows]))
    return names, values


@dataclass(frozen=True)
class SamplingFeatureBlocks:
    """Named continuous feature blocks aligned one-to-one with structures."""

    names: tuple[str, ...]
    values: tuple[np.ndarray, ...]
    atomic_values: np.ndarray | None = None
    atom_counts: tuple[int, ...] = ()
    atom_species: tuple[str, ...] = ()

    @property
    def row_count(self) -> int:
        return 0 if not self.values else int(self.values[0].shape[0])



def _validated_structure_descriptors(
    descriptors: np.ndarray,
    structure_count: int,
) -> np.ndarray:
    values = np.ascontiguousarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != int(structure_count) or values.shape[1] < 1:
        raise ValueError("structure descriptors must align with structures")
    if not np.isfinite(values).all():
        raise ValueError("structure descriptors must be finite")
    return values


def _cell_feature_row(structure: Any) -> np.ndarray:
    cell = np.asarray(structure_cell_array(structure), dtype=np.float64).reshape(3, 3)
    pbc = np.asarray(structure_pbc_flags(structure), dtype=np.float64).reshape(3)
    atom_count = max(1, len(structure))
    volume = abs(float(np.linalg.det(cell)))
    volume_per_atom = np.log(max(volume / atom_count, _SCALE_EPS)) if np.all(pbc) else 0.0

    # Cell angles/shape depend on the chosen periodic basis and supercell.
    # Local descriptors carry strain; retain the basis-independent density here.
    return np.concatenate(([volume_per_atom], np.sort(pbc)))


def _magnetic_feature_row(structure: Any) -> np.ndarray:
    evidence = analyze_structure_magnetism(structure, source_index=0)
    if evidence is None:
        return np.zeros(9, dtype=np.float64)

    # A peak direction is undefined for uniform/weakly modulated spins and
    # depends on the reciprocal-cell convention. Do not treat that categorical
    # estimate as a continuous Euclidean coordinate. Spin-model descriptors
    # retain model-resolved directional (including SOC) information.
    return np.asarray((
        evidence.mean_moment, evidence.moment_std, evidence.net_moment_ratio,
        evidence.collinearity, evidence.coplanarity, evidence.neighbor_correlation,
        evidence.neighbor_abs_correlation, evidence.parallel_fraction,
        evidence.antiparallel_fraction,
    ), dtype=np.float64)


def build_sampling_feature_blocks(
    structures: Sequence[Any],
    structure_descriptors: np.ndarray,
    *,
    per_atom_descriptors: np.ndarray | None = None,
    spin_model: bool,
) -> SamplingFeatureBlocks:
    """Keep structure views plus aligned atomic descriptors and central species."""
    structure_list = tuple(structures)
    mean = _validated_structure_descriptors(
        structure_descriptors,
        len(structure_list),
    )
    names: list[str] = ["descriptor_mean"]
    blocks: list[np.ndarray] = [mean]
    if per_atom_descriptors is not None:
        atomic = np.asarray(per_atom_descriptors, dtype=np.float64)
        if atomic.shape != (sum(len(s) for s in structure_list), mean.shape[1]):
            raise ValueError("atomic descriptors must align with structures and model dimensions")
        if not np.isfinite(atomic).all():
            raise ValueError("atomic descriptors must be finite")
    lattice = np.vstack([_cell_feature_row(structure) for structure in structure_list])
    names.append("lattice")
    blocks.append(lattice)
    if spin_model:
        magnetic = np.vstack(
            [_magnetic_feature_row(structure) for structure in structure_list]
        )
        names.append("magnetism")
        blocks.append(magnetic)
    input_names, input_blocks = _input_feature_blocks(structure_list, spin_model)
    names.extend(input_names)
    blocks.extend(input_blocks)
    symbols = tuple(
        str(symbol)
        for structure in structure_list
        for symbol in (structure.get_chemical_symbols()
                       if hasattr(structure, "get_chemical_symbols") else structure.elements)
    )
    return SamplingFeatureBlocks(
        tuple(names), tuple(blocks),
        None if per_atom_descriptors is None else np.asarray(per_atom_descriptors, dtype=np.float64),
        tuple(len(s) for s in structure_list), symbols,
    )


def _normalize_block_sets(
    candidate: np.ndarray,
    existing: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    fit = candidate if existing is None else np.vstack((candidate, existing))
    # Duplicate trajectory frames must not redefine the metric. Fit one scale
    # across the comparable pool, never separately inside a phase/spin label.
    fit = np.unique(fit, axis=0)
    center = np.mean(fit, axis=0)
    standard = np.std(fit, axis=0)
    # Regularized standardization: bounded amplification of near-constant
    # coordinates, without dividing by a tiny IQR of a multimodal population.
    floor = max(float(np.max(standard, initial=0.0)) * 0.05, _SCALE_EPS)
    # Inputs may originate in float32. Do not turn roundoff from a rigid
    # rotation/rebased cell into an order-one coverage direction.
    active = standard > max(float(np.max(np.abs(fit), initial=0.0)) * 1e-6, _SCALE_EPS)
    scale = np.maximum(standard, floor)
    scale *= np.sqrt(max(1, int(np.count_nonzero(active))))
    candidate_normalized = (candidate - center) / scale
    candidate_normalized[:, ~active] = 0.0
    existing_normalized = None if existing is None else (existing - center) / scale
    if existing_normalized is not None:
        existing_normalized[:, ~active] = 0.0
    return candidate_normalized, existing_normalized


def _normalize_descriptor_sets(
    candidate: np.ndarray,
    existing: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Put descriptors on one RMS scale without changing their geometry.

    Unlike heterogeneous physical scalars, descriptor coordinates already
    belong to one model representation. Per-coordinate whitening changes its
    distance ordering and can amplify small directions. One scalar preserves
    all Euclidean distance ratios, including under orthogonal basis changes.
    """
    fit = candidate if existing is None else np.vstack((candidate, existing))
    fit = np.unique(fit, axis=0)
    center = np.mean(fit, axis=0)
    scale = max(float(np.sqrt(np.sum(np.var(fit, axis=0)))), _SCALE_EPS)
    return ((candidate - center) / scale,
            None if existing is None else (existing - center) / scale)
