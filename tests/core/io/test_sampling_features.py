from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from NepTrainKit.core.io.sampling_features import (
    _input_environment,
    _input_environment_reference,
    _normalize_block_sets,
    _normalize_descriptor_sets,
    build_sampling_feature_blocks,
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
    np.testing.assert_array_equal(blocks.atomic_values, per_atom)
    assert blocks.atom_counts == (2, 2)
    assert blocks.atom_species == ("Fe",) * 4


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

    values, existing = _normalize_block_sets(np.hstack(blocks.values), None)

    assert existing is None
    assert values.shape == (3, sum(block.shape[1] for block in blocks.values))
    assert values.dtype == np.float64
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

    candidate_values, existing_values = _normalize_block_sets(
        np.hstack(candidate.values),
        np.hstack(existing.values),
    )

    assert existing_values is not None
    assert candidate_values.shape[1] == existing_values.shape[1]
    assert existing_values[0, 0] == 0.0


def test_descriptor_normalization_preserves_distance_ratios_and_basis():
    rng = np.random.default_rng(83)
    values = rng.normal(size=(30, 4)) * [10, 1, .1, .001]
    normalized, _ = _normalize_descriptor_sets(values, None)
    raw_distances = np.linalg.norm(values[1:] - values[0], axis=1)
    distances = np.linalg.norm(normalized[1:] - normalized[0], axis=1)
    np.testing.assert_allclose(distances / raw_distances, distances[0] / raw_distances[0])
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    rotated, _ = _normalize_descriptor_sets(values @ rotation, None)
    np.testing.assert_allclose(np.linalg.norm(rotated[1:] - rotated[0], axis=1), distances)


@pytest.mark.parametrize("seed", range(20))
def test_native_input_features_match_independent_periodic_neighbor_reference(seed):
    rng = np.random.default_rng(seed)
    cell = np.array([[3.1, 0, 0], [.4, 3.3, 0], [.3, .2, 3.5]])
    pbc = np.asarray([(seed % 3) != 0, (seed % 3) == 1, True])
    atoms = Atoms("Ni4", positions=rng.random((4, 3)) @ cell, cell=cell, pbc=pbc)
    atoms.arrays["spin"] = rng.normal(size=(4, 3)).astype(np.float32)
    native = _input_environment(atoms, True)
    reference = _input_environment_reference(atoms.positions, cell, pbc,
                                             atoms.arrays["spin"].astype(float))
    np.testing.assert_allclose(native, reference, atol=1e-8, rtol=1e-7)


def test_input_features_preserve_spatial_rotation_atom_order_and_supercell():
    original = bulk("Co", "fcc", a=3.5, cubic=True)
    original.arrays["spin"] = np.tile([.3, .7, 1.1], (len(original), 1))
    rotated = original.copy()
    rotation, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(3, 3)))
    rotated.positions[:] = original.positions @ rotation
    rotated.set_cell(original.cell.array @ rotation)
    rotated.arrays["spin"] = original.arrays["spin"] @ rotation
    reordered = original[::-1]
    enlarged = original.repeat((2, 1, 1))
    blocks = build_sampling_feature_blocks([original, rotated, reordered, enlarged],
                                           np.zeros((4, 1)), spin_model=True)
    for name, block in zip(blocks.names, blocks.values):
        if name.startswith("input_"):
            np.testing.assert_allclose(block, np.tile(block[:1], (4, 1)), atol=2e-7)


def test_spin_rotation_relative_to_lattice_is_visible_without_model_or_labels():
    structures = [bulk("Co", "fcc", a=3.5, cubic=True) for _ in range(3)]
    for s, direction in zip(structures, ([0, 0, 1], [1, 1, 0], [1, 1, 1])):
        unit = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        s.arrays["spin"] = np.tile(unit, (len(s), 1))
    blocks = build_sampling_feature_blocks(structures, np.zeros((3, 1)), spin_model=True)
    orientation = blocks.values[blocks.names.index("input_orientation:Co")]
    assert np.linalg.norm(orientation[0] - orientation[1]) > .01
    assert np.linalg.norm(orientation[1] - orientation[2]) > .01


def test_inputs_ignore_all_reference_labels_and_source_names():
    original = bulk("Ni", "fcc", a=3.5, cubic=True)
    changed = original.copy()
    changed.info.update(energy=np.nan, virial=np.full(9, np.inf), Config_type="anything")
    for field in ("force", "forces", "mforce", "force_mag"):
        changed.arrays[field] = np.full((len(changed), 3), np.nan)
    blocks = build_sampling_feature_blocks([original, changed], np.zeros((2, 1)), spin_model=False)
    assert all("moment" not in name and "orientation" not in name for name in blocks.names)
    for value in blocks.values:
        np.testing.assert_array_equal(value[0], value[1])


def test_input_geometry_handles_isolated_atom_and_partial_periodic_zero_axis():
    for s in [Atoms("C"), Atoms("C", cell=[3, 3, 0], pbc=[True, True, False])]:
        values = _input_environment(s, False)
        assert values.shape == (1, 12)
        assert np.isfinite(values).all()
    np.testing.assert_array_equal(_input_environment(Atoms("C"), False), np.zeros((1, 12)))
    thin = Atoms("C", cell=[3, 3, 1], pbc=[True, True, False])
    vacuum = Atoms("C", cell=[3, 3, 10000], pbc=[True, True, False])
    np.testing.assert_array_equal(_input_environment(thin, False), _input_environment(vacuum, False))


def test_roundoff_filter_does_not_drop_a_small_but_resolved_input_scale():
    raw = np.array([[1e-7], [2e-7], [3e-7]])
    values, _ = _normalize_block_sets(raw, None)
    assert np.ptp(values[:, 0]) > 1
