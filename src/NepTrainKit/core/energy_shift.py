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

import numpy as np
from collections import Counter
from typing import List, Dict
import re
from NepTrainKit.utils import timeit
from .structure import Structure


REF_GROUP_ALIGNMENT = "REF_GROUP"
ZERO_BASELINE_ALIGNMENT = "ZERO_BASELINE"
DFT_TO_NEP_ALIGNMENT = "DFT_TO_NEP"

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
        nep_energy_array: np.ndarray | None = None):
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

    Yields
    ------
    int
        Progress indicator (always ``1``) suitable for UI progress hooks.

    Notes
    -----
    The function updates ``Structure.energy`` in-place.

    Examples
    --------
    >>> # Example usage can be filled later
    """
    frames = []
    for s in structures:
        energy = float(s.energy)
        config_type = str(s.tag)
        elem_counts = Counter(s.elements)

        frames.append({"energy": energy, "config_type": config_type, "elem_counts": elem_counts})

    all_elements = sorted({e for f in frames for e in f["elem_counts"]})
    num_elements = len(all_elements)


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
    # apply shift
    for s, frame in zip(structures, frames):
        group = config_to_group[frame["config_type"]]
        if group in group_to_atomic_ref:
            count_vec = np.array([frame["elem_counts"].get(e, 0) for e in all_elements], dtype=float)
            shift = np.dot(count_vec, group_to_atomic_ref[group])
            new_energy = frame["energy"] - shift
            # print( frame["energy"],shift,new_energy)
            s.energy = new_energy
    # return group_to_atomic_ref
