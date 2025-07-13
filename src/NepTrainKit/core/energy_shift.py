"""Utilities for shifting structure energies using atomic baselines.
抄的陈博的代码(已允许)
url: https://github.com/brucefan1983/GPUMD/tree/master/tools/Analysis_and_Processing/energy-reference-aligner
Zherui Chen Email: <chenzherui0124@foxmail.com>
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from typing import List, Dict
import re
from NepTrainKit import utils
from .structure import Structure
from .calculator import NepCalculator

REF_GROUP_ALIGNMENT = "REF_GROUP"
ZERO_BASELINE_ALIGNMENT = "ZERO_BASELINE"
DFT_TO_NEP_ALIGNMENT = "DFT_TO_NEP"

def longest_common_prefix(strs: List[str]) -> str:
    if not strs:
        return ""
    s1, s2 = min(strs), max(strs)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


def suggest_group_patterns(config_types: List[str], min_group_size: int = 2, min_prefix_len: int = 3) -> List[str]:
    """Group strings by common prefix without relying on delimiters, and output regex patterns."""
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
    """Vectorized MSE cost for atomic reference baseline."""
    shifted = energies[None, :] - np.dot(param_population, element_counts.T)
    cost = np.mean((shifted - target_energies[None, :]) ** 2, axis=1)
    return cost.reshape(-1, 1)

@utils.timeit
def nes_optimize_atomic_baseline(num_variables: int,
                                 max_generations: int,
                                 energies: np.ndarray,
                                 element_counts: np.ndarray,
                                 targets: np.ndarray,
                                 pop_size: int = 40,
                                 tol: float = 1e-8,
                                 seed: int = 42,
                                 print_every: int = 100) -> np.ndarray:
    """NES optimizer for atomic reference energies."""
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
        nep_energy_array: np.array | None = None):
    """Shift structure energies using different alignment strategies.

    Parameters
    ----------
    structures
        Structures whose energies will be shifted.
    reference_structures
        Structures used to compute the reference mean energy when
        ``alignment_mode`` is ``REF_GROUP_ALIGNMENT``.
    alignment_mode
        One of ``REF_GROUP_ALIGNMENT``, ``ZERO_BASELINE_ALIGNMENT`` or
        ``DFT_TO_NEP_ALIGNMENT``.
    nep_energy_array
        nep energy array when ``alignment_mode`` is
        ``DFT_TO_NEP_ALIGNMENT``.
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
        #这里是为了更新ui信号
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
