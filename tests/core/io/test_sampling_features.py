from __future__ import annotations

import numpy as np
from ase.build import bulk

from NepTrainKit.core.io.sampling_features import (
    build_sampling_feature_blocks,
    representative_sampling_features,
)


def test_atomic_descriptor_spread_preserves_local_environment_novelty():
    structures = [bulk("Fe", "bcc", a=2.8), bulk("Fe", "bcc", a=2.8)]
    means = np.zeros((2, 2), dtype=np.float32)
    per_atom = np.asarray(
        [[0.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    structures = [structures[0], structures[1]]
    structures[0] *= (2, 1, 1)
    structures[1] *= (2, 1, 1)
    per_atom = np.asarray(
        [[-1.0, 0.0], [1.0, 0.0], [-3.0, 0.0], [3.0, 0.0]],
        dtype=np.float32,
    )

    blocks = build_sampling_feature_blocks(
        structures,
        means,
        per_atom_descriptors=per_atom,
        spin_model=False,
    )
    by_name = dict(zip(blocks.names, blocks.values))

    assert np.allclose(by_name["descriptor_mean"], 0.0)
    assert by_name["descriptor_std"][1, 0] > by_name["descriptor_std"][0, 0]
    assert by_name["descriptor_tail"][1, 0] > by_name["descriptor_tail"][0, 0]


def test_lattice_and_descriptor_blocks_are_normalized_and_balanced():
    structures = [
        bulk("Fe", "bcc", a=2.7),
        bulk("Fe", "bcc", a=2.9),
        bulk("Fe", "bcc", a=3.1),
    ]
    descriptors = np.asarray([[0.0, 0.0], [2.0, 0.0], [8.0, 0.0]])
    blocks = build_sampling_feature_blocks(
        structures,
        descriptors,
        spin_model=False,
    )

    values, existing = representative_sampling_features(blocks)

    assert existing is None
    assert values.shape == (3, descriptors.shape[1] + 10)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    assert not np.allclose(values[0], values[-1])


def test_existing_and_candidates_share_one_feature_scaling():
    candidate_structures = [bulk("Fe", "bcc", a=2.8), bulk("Fe", "bcc", a=3.0)]
    existing_structures = [bulk("Fe", "bcc", a=2.9)]
    candidate = build_sampling_feature_blocks(
        candidate_structures,
        np.asarray([[0.0], [2.0]]),
        spin_model=False,
    )
    existing = build_sampling_feature_blocks(
        existing_structures,
        np.asarray([[1.0]]),
        spin_model=False,
    )

    candidate_values, existing_values = representative_sampling_features(
        candidate,
        existing,
    )

    assert existing_values is not None
    assert candidate_values.shape[1] == existing_values.shape[1]
    assert existing_values[0, 0] == 0.0
