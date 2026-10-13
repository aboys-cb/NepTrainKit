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

from NepTrainKit.core.audit.magnetic_inventory import analyze_structure_magnetism
from NepTrainKit.core.geometry_cache import structure_cell_array, structure_pbc_flags

_SCALE_EPS = 1.0e-12


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
    active = standard > max(float(np.max(np.abs(fit), initial=0.0)) * 1e-10, _SCALE_EPS)
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
