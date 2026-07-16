from itertools import product

import numpy as np
import pytest
from ase.build import bulk

from NepTrainKit.core.audit import phase_sketch as phase_sketch_module
from NepTrainKit.core.audit.phase_sketch import (
    PrototypeBank,
    periodic_knn_vectors,
    phase_sketch,
    summarize_phase_sketch,
)


def test_periodic_knn_matches_bruteforce_for_skewed_cell():
    cell = np.asarray(
        (
            (4.2, 0.0, 0.0),
            (1.3, 3.8, 0.0),
            (-0.7, 0.9, 4.5),
        )
    )
    fractional = np.asarray(
        (
            (0.05, 0.10, 0.15),
            (0.52, 0.23, 0.77),
            (0.81, 0.64, 0.31),
            (0.33, 0.88, 0.57),
        )
    )
    positions = fractional @ cell
    vectors, _, valid = periodic_knn_vectors(positions, cell, (True, True, True), neighbors=8)
    native_result = None
    if phase_sketch_module._native_phase is not None:
        native_result = phase_sketch_module._native_phase.periodic_knn_vectors(
            positions, cell, (True, True, True), 8
        )

    shifts = np.asarray(tuple(product(range(-2, 3), repeat=3)))
    for center in range(len(positions)):
        candidates = []
        for shift in shifts:
            for source in range(len(positions)):
                if source == center and np.all(shift == 0):
                    continue
                candidates.append(positions[source] + shift @ cell - positions[center])
        expected = np.sort(np.linalg.norm(candidates, axis=1))[:8]
        actual = np.sort(np.linalg.norm(vectors[center, valid[center]], axis=1))
        np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

        if native_result is not None:
            native_vectors, _, native_valid = native_result
            native_distances = np.sort(
                np.linalg.norm(native_vectors[center, native_valid[center]], axis=1)
            )
            np.testing.assert_allclose(native_distances, expected, rtol=2.0e-6, atol=2.0e-6)


def test_periodic_knn_does_not_return_self_for_skewed_boundary_atom():
    cell = np.asarray(
        (
            (10.53, 10.53, 0.0),
            (10.53, 0.0, 10.53),
            (0.0, 10.53, 10.53),
        )
    )
    fractional = np.asarray(
        (
            (0.0, 0.0, 1.0 / 6.0),
            (0.25, 0.25, 0.25),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
        )
    )
    positions = fractional @ cell

    implementations = [
        periodic_knn_vectors(positions, cell, (True, True, True), neighbors=8)
    ]
    if phase_sketch_module._native_phase is not None:
        implementations.append(
            phase_sketch_module._native_phase.periodic_knn_vectors(
                positions, cell, (True, True, True), 8
            )
        )

    for vectors, indices, valid in implementations:
        assert np.all(np.linalg.norm(vectors[valid], axis=1) > 1.0e-7)
        for center in range(len(positions)):
            zero_distance_self = (
                (indices[center] == center)
                & valid[center]
                & (np.linalg.norm(vectors[center], axis=1) <= 1.0e-7)
            )
            assert not np.any(zero_distance_self)


@pytest.mark.parametrize(
    "pbc",
    ((True, True, True), (True, False, True), (False, False, False)),
)
def test_native_large_frame_knn_matches_python_oracle_for_skewed_cell(pbc, monkeypatch):
    native = phase_sketch_module._native_phase
    if native is None:
        pytest.skip("optional native phase extension is not built")

    rng = np.random.default_rng(20260716)
    cell = np.asarray(
        (
            (11.2, 0.0, 0.0),
            (3.1, 10.4, 0.0),
            (-2.2, 1.7, 12.3),
        )
    )
    positions = rng.uniform(-0.4, 1.4, size=(257, 3)) @ cell
    expected_vectors, expected_indices, expected_valid = periodic_knn_vectors(
        positions, cell, pbc, neighbors=20
    )
    monkeypatch.setattr(
        phase_sketch_module,
        "periodic_knn_vectors",
        lambda *args, **kwargs: pytest.fail("large frame unexpectedly used Python KNN"),
    )
    actual_vectors, actual_indices, actual_valid = (
        phase_sketch_module.accelerated_periodic_knn_vectors(
            positions, cell, pbc, neighbors=20
        )
    )

    np.testing.assert_array_equal(actual_valid, expected_valid)
    np.testing.assert_array_equal(actual_indices[actual_valid], expected_indices[expected_valid])
    np.testing.assert_allclose(
        np.linalg.norm(actual_vectors[actual_valid], axis=1),
        np.linalg.norm(expected_vectors[expected_valid], axis=1),
        rtol=3.0e-5,
        atol=3.0e-5,
    )


def test_chemistry_sketch_is_invariant_to_element_ids_and_atom_order():
    atoms = bulk("Cu", "fcc", a=3.62, cubic=True).repeat((2, 2, 2))
    types = np.where(np.arange(len(atoms)) % 2, 28, 13)
    reference = phase_sketch(atoms.positions, atoms.cell.array, atoms.pbc, types)

    rng = np.random.default_rng(11)
    permutation = rng.permutation(len(atoms))
    relabeled = np.where(types[permutation] == 28, 101, 7)
    permuted = phase_sketch(
        atoms.positions[permutation],
        atoms.cell.array,
        atoms.pbc,
        relabeled,
    )

    np.testing.assert_allclose(
        summarize_phase_sketch(permuted.chemistry),
        summarize_phase_sketch(reference.chemistry),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_prototype_bank_rejects_far_out_of_reference_values():
    values = np.asarray(((0.0, 0.0), (0.1, -0.1), (3.0, 3.1), (2.9, 3.0)))
    bank = PrototypeBank(
        prototypes_per_class=1,
        samples_per_prototype=2,
        rejection_scale=2.0,
        minimum_margin=1.0,
    ).fit(values, ("alpha", "alpha", "beta", "beta"))

    prediction = bank.predict(np.asarray(((0.05, -0.02), (20.0, -20.0))))
    assert prediction.labels.tolist() == ["alpha", "unknown"]


def test_native_features_match_python_reference(monkeypatch):
    native = phase_sketch_module._native_phase
    if native is None:
        pytest.skip("optional native phase extension is not built")

    atoms = bulk("Fe", "bcc", a=2.88, cubic=True).repeat((3, 3, 3))
    rng = np.random.default_rng(19)
    atoms.positions += rng.normal(scale=0.04, size=atoms.positions.shape)
    types = np.where(np.arange(len(atoms)) % 3, 26, 13)
    vectors, indices, valid = periodic_knn_vectors(
        atoms.positions,
        atoms.cell.array,
        atoms.pbc,
        neighbors=24,
    )
    monkeypatch.setattr(
        phase_sketch_module,
        "periodic_knn_vectors",
        lambda *args, **kwargs: (vectors, indices, valid),
    )

    monkeypatch.setattr(phase_sketch_module, "_native_phase", None)
    reference = phase_sketch(
        atoms.positions, atoms.cell.array, atoms.pbc, types
    )
    monkeypatch.setattr(phase_sketch_module, "_native_phase", native)
    accelerated = phase_sketch(
        atoms.positions, atoms.cell.array, atoms.pbc, types
    )

    np.testing.assert_allclose(accelerated.geometry, reference.geometry, rtol=3.0e-4, atol=1.0e-4)
    np.testing.assert_allclose(accelerated.chemistry, reference.chemistry, rtol=3.0e-4, atol=1.0e-4)
