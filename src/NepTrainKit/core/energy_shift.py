"""Energy alignment utilities based on atomic reference baselines.

This module aligns per-structure energies against reference targets by optimizing
per-element reference energies (atomic baselines) using a simple NES optimizer.

Notes
-----
- Adapted conceptually from the GPUMD energy reference aligner:
  https://github.com/brucefan1983/GPUMD/tree/master/tools/Analysis_and_Processing/energy-reference-aligner

Examples
--------
>>> # Use shift_dataset_energy(...) to align a dataset in-place
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List
from collections import Counter

import numpy as np

from NepTrainKit.config import Config
from NepTrainKit.utils import timeit
from .structure import Structure


REF_GROUP_ALIGNMENT = "REF_GROUP"
ZERO_BASELINE_ALIGNMENT = "ZERO_BASELINE"
DFT_TO_NEP_ALIGNMENT = "DFT_TO_NEP"
BASELINE_PRESET_SECTION = "energy_baseline_preset"


@dataclass
class EnergyBaselinePreset:
    """Serializable container for per-group atomic reference baselines."""

    version: int = 1
    alignment_mode: str = REF_GROUP_ALIGNMENT
    elements: list[str] = field(default_factory=list)
    group_to_ref: dict[str, list[float]] = field(default_factory=dict)
    group_patterns: list[str] = field(default_factory=list)
    config_to_group: dict[str, str] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alignment_mode": self.alignment_mode,
            "elements": list(self.elements),
            "group_to_ref": {k: list(v) for k, v in self.group_to_ref.items()},
            "group_patterns": list(self.group_patterns),
            "config_to_group": dict(self.config_to_group),
            "optimizer": dict(self.optimizer),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnergyBaselinePreset":
        return cls(
            version=int(data.get("version", 1)),
            alignment_mode=data.get("alignment_mode", REF_GROUP_ALIGNMENT),
            elements=list(data.get("elements", [])),
            group_to_ref={k: list(v) for k, v in (data.get("group_to_ref", {}) or {}).items()},
            group_patterns=list(data.get("group_patterns", [])),
            config_to_group=dict(data.get("config_to_group", {}) or {}),
            optimizer=dict(data.get("optimizer", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )


def save_energy_baseline_preset(name: str, baseline: EnergyBaselinePreset) -> None:
    """Persist a baseline preset in the config database."""
    Config.set(BASELINE_PRESET_SECTION, name, json.dumps(baseline.to_dict()))


def load_energy_baseline_preset(name: str) -> EnergyBaselinePreset | None:
    """Load a baseline preset by name."""
    raw = Config.get(BASELINE_PRESET_SECTION, name)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return EnergyBaselinePreset.from_dict(data)
    except Exception:  # noqa: BLE001
        return None


def list_energy_baseline_preset_names() -> list[str]:
    """Return available baseline preset names."""
    return Config.list_options(BASELINE_PRESET_SECTION)


def apply_energy_baseline(structures: List[Structure], baseline: EnergyBaselinePreset) -> None:
    """Apply a precomputed baseline to structures in-place."""
    elements = list(baseline.elements)
    if not elements or not baseline.group_to_ref:
        return
    for structure in structures:
        if not getattr(structure, "has_energy", False):
            continue
        config_type = str(structure.tag)
        group = baseline.config_to_group.get(config_type, config_type)
        ref_vec = baseline.group_to_ref.get(group)
        if ref_vec is None:
            continue
        cnt = Counter(structure.elements)
        count_vec = np.array([cnt.get(e, 0) for e in elements], dtype=float)
        shift = float(np.dot(count_vec, np.asarray(ref_vec, dtype=float)))
        structure.energy = float(structure.energy) - shift

def longest_common_prefix(strs: List[str]) -> str:
    """Return the longest common prefix among strings.

    Parameters
    ----------
    strs : list[str]
        Collection of input strings.

    Returns
    -------
    str
        Longest shared prefix or an empty string if ``strs`` is empty.

    Examples
    --------
    >>> longest_common_prefix(["abc", "abd", "ab"])
    'ab'
    """
    if not strs:
        return ""
    s1, s2 = min(strs), max(strs)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


def suggest_group_patterns(config_types: List[str], min_group_size: int = 2, min_prefix_len: int = 3) -> List[str]:
    """Suggest regex patterns that group items by common prefixes.

    Parameters
    ----------
    config_types : list[str]
        Set of config type labels to group.
    min_group_size : int, default=2
        Minimum number of items required to form a grouped prefix.
    min_prefix_len : int, default=3
        Minimum prefix length to consider for grouping.

    Returns
    -------
    list[str]
        Regex-like patterns; ungrouped items are returned literally.

    Examples
    --------
    >>> suggest_group_patterns(["A001", "A002", "B1"])  # doctest: +ELLIPSIS
    ['A0.*', 'B1']
    """
    unused = set(config_types)
    patterns = []

    while unused:
        base = unused.pop()
        group = [base]
        to_remove = []

        for other in unused:
            prefix = longest_common_prefix([base, other])
            if len(prefix) >= min_prefix_len:
                group.append(other)
                to_remove.append(other)

        for item in to_remove:
            unused.remove(item)

        if len(group) >= min_group_size:
            prefix = longest_common_prefix(group)
            patterns.append(re.escape(prefix) + '.*')
        else:
            patterns.extend(re.escape(g) for g in group)

    return sorted(patterns)
def atomic_baseline_cost(param_population: np.ndarray,
                         energies: np.ndarray,
                         element_counts: np.ndarray,
                         target_energies: np.ndarray) -> np.ndarray:
    """Vectorized MSE cost for per-element baseline parameters.

    Parameters
    ----------
    param_population : numpy.ndarray
        Population matrix of shape ``(pop, n_elem)``.
    energies : numpy.ndarray
        Reference structure energies of shape ``(n_samples,)``.
    element_counts : numpy.ndarray
        Per-structure element counts of shape ``(n_samples, n_elem)``.
    target_energies : numpy.ndarray
        Target energy per structure of shape ``(n_samples,)``.

    Returns
    -------
    numpy.ndarray
        Column vector of costs with shape ``(pop, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> pop = np.zeros((2, 2)); e = np.array([1.0, 2.0])
    >>> cnt = np.array([[1, 1], [2, 0]]); t = np.array([0.0, 0.0])
    >>> atomic_baseline_cost(pop, e, cnt, t).shape
    (2, 1)
    """
    shifted = energies[None, :] - np.dot(param_population, element_counts.T)
    cost = np.mean((shifted - target_energies[None, :]) ** 2, axis=1)
    return cost.reshape(-1, 1)

@timeit
def nes_optimize_atomic_baseline(num_variables: int,
                                 max_generations: int,
                                 energies: np.ndarray,
                                 element_counts: np.ndarray,
                                 targets: np.ndarray,
                                 pop_size: int = 40,
                                 tol: float = 1e-8,
                                 seed: int = 42,
                                 print_every: int = 100) -> np.ndarray:
    """Optimize per-element reference energies using NES.

    Parameters
    ----------
    num_variables : int
        Number of element types (dimension of the baseline vector).
    max_generations : int
        Maximum NES iterations.
    energies : numpy.ndarray
        Structure energies, shape ``(n_samples,)``.
    element_counts : numpy.ndarray
        Per-structure element counts, shape ``(n_samples, n_elem)``.
    targets : numpy.ndarray
        Target energies for alignment, shape ``(n_samples,)``.
    pop_size : int, default=40
        Population size per iteration.
    tol : float, default=1e-8
        Early-stop tolerance on best fitness improvement.
    seed : int, default=42
        Random seed.
    print_every : int, default=100
        Unused placeholder retained for potential logging.

    Returns
    -------
    numpy.ndarray
        Optimized baseline vector with shape ``(n_elem,)``.

    Examples
    --------
    >>> import numpy as np
    >>> e = np.array([1.0, 2.0]); cnt = np.array([[1, 0], [0, 1]])
    >>> best = nes_optimize_atomic_baseline(2, 10, e, cnt, np.zeros_like(e))
    >>> best.shape
    (2,)
    """
    np.random.seed(seed)

    best_fitness = np.ones((max_generations, 1))
    elite = np.zeros((max_generations, num_variables))
    mean = -1 * np.random.rand(1, num_variables)
    stddev = 0.1 * np.ones((1, num_variables))
    lr_mean = 1.0
    lr_std = (3 + np.log(num_variables)) / (5 * np.sqrt(num_variables)) / 2
    weights = np.maximum(0, np.log(pop_size / 2 + 1) - np.log(np.arange(1, pop_size + 1)))
    weights = weights / np.sum(weights) - 1 / pop_size

    for gen in range(max_generations):
        z = np.random.randn(pop_size, num_variables)
        pop = mean + stddev * z
        fitness = atomic_baseline_cost(pop, energies, element_counts, targets)
        idx = np.argsort(fitness.flatten())
        fitness = fitness[idx]
        z = z[idx, :]
        pop = pop[idx, :]
        best_fitness[gen] = fitness[0]
        elite[gen, :] = pop[0, :]
        mean += lr_mean * stddev * (weights @ z)
        stddev *= np.exp(lr_std * (weights @ (z ** 2 - 1)))
        if gen > 0 and abs(best_fitness[gen] - best_fitness[gen - 1]) < tol:
            best_fitness = best_fitness[:gen + 1]
            elite = elite[:gen + 1]
            break
    return elite[-1]



def shift_dataset_energy(
        structures: List[Structure],
        reference_structures: List[Structure] | None,
        max_generations: int = 100000,
        population_size: int = 40,
        convergence_tol: float = 1e-8,
        random_seed: int = 42,
        group_patterns: List[str] | None = None,
        alignment_mode: str = REF_GROUP_ALIGNMENT,
        nep_energy_array: np.ndarray | None = None,
        precomputed_baseline: EnergyBaselinePreset | Dict[str, Any] | None = None,
        baseline_store: Dict[str, EnergyBaselinePreset] | None = None,
        source_summary: Dict[str, Any] | None = None):
    """Shift dataset energies using group-wise atomic baseline alignment.

    Parameters
    ----------
    structures : list[Structure]
        Structures whose energies are to be shifted in-place.
    reference_structures : list[Structure] or None
        Structures used to compute the reference mean energy when
        ``alignment_mode`` is ``REF_GROUP_ALIGNMENT``.
    max_generations : int, optional
        Maximum iterations for the NES optimizer.
    population_size : int, optional
        Population size for the NES optimizer.
    convergence_tol : float, optional
        Early-stop criterion for NES best-fitness improvements.
    random_seed : int, optional
        Seed forwarded to the NES optimizer for reproducibility.
    group_patterns : list[str] or None
        Optional regex patterns to group configurations; otherwise inferred by
        common-prefix detection.
    alignment_mode : {REF_GROUP_ALIGNMENT, ZERO_BASELINE_ALIGNMENT, DFT_TO_NEP_ALIGNMENT}, optional
        Alignment strategy controlling the target energies used by the
        optimizer.
    nep_energy_array : numpy.ndarray or None
        Per-structure NEP energies used when ``alignment_mode`` is
        ``DFT_TO_NEP_ALIGNMENT`` (units must match ``structures`` input).
    precomputed_baseline : EnergyBaselinePreset or dict or None
        When provided, apply the baseline directly without recomputing.
    baseline_store : dict or None
        Optional mutable container to receive the baseline used.
    source_summary : dict or None
        Optional metadata to embed in the stored baseline.

    Yields
    ------
    int
        Progress indicator (always ``1``) suitable for UI progress hooks.

    Notes
    -----
    The function updates ``Structure.energy`` in-place.
    """
    frames = []
    for s in structures:
        energy = float(s.energy)
        config_type = str(s.tag)
        elem_counts = Counter(s.elements)

        frames.append({"energy": energy, "config_type": config_type, "elem_counts": elem_counts})

    all_elements = sorted({e for f in frames for e in f["elem_counts"]})
    num_elements = len(all_elements)

    # Apply preset directly when supplied
    if precomputed_baseline is not None:
        if isinstance(precomputed_baseline, dict):
            baseline = EnergyBaselinePreset.from_dict(precomputed_baseline)
        else:
            baseline = precomputed_baseline
        apply_energy_baseline(structures, baseline)
        if baseline_store is not None:
            baseline_store["baseline"] = baseline
        return

    ref_mean = None
    if alignment_mode == REF_GROUP_ALIGNMENT:
        if not len(reference_structures):
            raise ValueError("reference_structures is required for REF_GROUP_ALIGNMENT")
        ref_energies = np.array([f.energy for f in reference_structures])
        ref_mean = np.mean(ref_energies)

    if alignment_mode == DFT_TO_NEP_ALIGNMENT:
        if nep_energy_array is None:
            raise ValueError("nep_energy_array is required for DFT_TO_NEP_ALIGNMENT")

        for f, e in zip(frames, nep_energy_array):
            f["nep_energy"] = e * f["elem_counts"].total()

    all_config_types = {f["config_type"] for f in frames}

    # build mapping from config_type to regex group name
    config_to_group: Dict[str, str] = {}
    if group_patterns:
        for pat in group_patterns:
            try:
                regex = re.compile(pat)
            except re.error:
                continue
            for ct in all_config_types:
                if ct not in config_to_group and regex.match(ct):
                    config_to_group[ct] = pat
    for ct in all_config_types:
        config_to_group.setdefault(ct, ct)

    shift_groups = sorted(set(config_to_group.values()))

    group_to_atomic_ref = {}
    for group in shift_groups:

        grp_frames = [f for f in frames if config_to_group[f["config_type"]] == group]

        if not grp_frames:
            continue
        energies = np.array([f["energy"] for f in grp_frames])
        counts = np.array([[f["elem_counts"].get(e, 0) for e in all_elements] for f in grp_frames], dtype=float)

        if alignment_mode == REF_GROUP_ALIGNMENT:
            targets = np.full_like(energies, ref_mean)
        elif alignment_mode == ZERO_BASELINE_ALIGNMENT:
            targets = np.zeros_like(energies)
        else:  # DFT_TO_NEP_ALIGNMENT
            targets = np.array([f["nep_energy"] for f in grp_frames])

        atomic_ref = nes_optimize_atomic_baseline(
            num_elements,
            max_generations,
            energies,
            counts,
            targets,
            pop_size=population_size,
            tol=convergence_tol,
            seed=random_seed,
            print_every=100,
        )
        group_to_atomic_ref[group] = atomic_ref
        # Update UI progress incrementally
        yield 1

    baseline = EnergyBaselinePreset(
        version=1,
        alignment_mode=alignment_mode,
        elements=all_elements,
        group_to_ref={k: v.tolist() for k, v in group_to_atomic_ref.items()},
        group_patterns=list(group_patterns) if group_patterns else [],
        config_to_group=config_to_group,
        optimizer={
            "max_generations": max_generations,
            "population_size": population_size,
            "convergence_tol": convergence_tol,
            "random_seed": random_seed,
        },
        metadata={
            "struct_count": len(structures),
            "ref_count": len(reference_structures) if reference_structures is not None else 0,
            "source_summary": source_summary or {},
        },
    )
    if baseline_store is not None:
        baseline_store["baseline"] = baseline

    apply_energy_baseline(structures, baseline)
