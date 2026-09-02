from itertools import product

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from NepTrainKit.core.audit import phase_sketch as phase_sketch_module
from NepTrainKit.core.audit.phase_sketch import (
    PrototypeBank,
    _adaptive_cna_reference,
    _translational_order_reference,
    accelerated_periodic_knn_vectors,
    adaptive_cna_labels,
    periodic_knn_vectors,
    phase_sketch,
    summarize_phase_sketch,
    translational_order_evidence,
)


def _stacked_close_packed(sequence: str, size: int = 3) -> Atoms:
    distance = 2.55
    first = np.asarray((distance, 0.0, 0.0))
    second = np.asarray((0.5 * distance, np.sqrt(3.0) * 0.5 * distance, 0.0))
    layer_height = np.sqrt(2.0 / 3.0) * distance
    offsets = {
        "A": np.zeros(3),
        "B": (first + second) / 3.0,
        "C": 2.0 * (first + second) / 3.0,
    }
    positions = [
        row * first + column * second + offsets[layer] + (0.0, 0.0, depth * layer_height)
        for depth, layer in enumerate(sequence)
        for row in range(size)
        for column in range(size)
    ]
    return Atoms(
        f"Cu{len(positions)}",
        positions=positions,
        cell=(size * first, size * second, (0.0, 0.0, len(sequence) * layer_height)),
        pbc=True,
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


def test_translational_order_matches_reference_for_skewed_crystal():
    atoms = bulk("Cu", "fcc", a=3.62, cubic=True).repeat((3, 3, 3))
    deformation = np.asarray(
        ((1.0, 0.12, -0.04), (0.03, 0.96, 0.08), (-0.07, 0.05, 1.04))
    )
    atoms.set_cell(atoms.cell.array @ deformation, scale_atoms=True)

    actual = translational_order_evidence(
        atoms.positions, atoms.cell.array, atoms.pbc
    )
    expected = _translational_order_reference(
        atoms.positions, atoms.cell.array
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_translational_order_separates_crystal_from_uniform_liquid():
    crystal = bulk("Cu", "fcc", a=3.62, cubic=True).repeat((5, 5, 5))
    crystal_score, crystal_limit = translational_order_evidence(
        crystal.positions, crystal.cell.array, crystal.pbc
    )

    rng = np.random.default_rng(20260716)
    liquid_positions = rng.random((len(crystal), 3)) @ crystal.cell.array
    liquid_score, liquid_limit = translational_order_evidence(
        liquid_positions, crystal.cell.array, crystal.pbc
    )

    assert crystal_score > crystal_limit
    assert liquid_score < liquid_limit


@pytest.mark.parametrize(
    ("sequence", "expected"),
    (("ABC" * 4, 1), ("AB" * 6, 2)),
)
def test_adaptive_cna_identifies_fcc_and_hcp(sequence, expected):
    atoms = _stacked_close_packed(sequence)
    vectors, indices, valid = accelerated_periodic_knn_vectors(
        atoms.positions, atoms.cell.array, atoms.pbc, neighbors=24
    )

    actual = adaptive_cna_labels(vectors, indices, valid)
    reference = _adaptive_cna_reference(vectors, valid)

    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(actual, np.full(len(atoms), expected))


def test_adaptive_cna_identifies_bcc_without_mislabeling_common_controls():
    structures = (
        (bulk("Fe", "bcc", a=2.87, cubic=True).repeat((3, 3, 3)), 3),
        (bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3)), 0),
        (bulk("Po", "sc", a=3.35, cubic=True).repeat((3, 3, 3)), 0),
    )
    for atoms, expected in structures:
        vectors, indices, valid = accelerated_periodic_knn_vectors(
            atoms.positions, atoms.cell.array, atoms.pbc, neighbors=24
        )
        actual = adaptive_cna_labels(vectors, indices, valid)
        reference = _adaptive_cna_reference(vectors, valid)

        np.testing.assert_array_equal(actual, reference)
        np.testing.assert_array_equal(actual, np.full(len(atoms), expected))


def test_native_phase_partition_primitives_match_separate_native_calls():
    native = phase_sketch_module._native_phase
    if native is None or not hasattr(native, "phase_partition_primitives"):
        pytest.skip("native phase-partition primitive is not built")
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True).repeat((2, 2, 2))

    vectors, indices, valid, labels = native.phase_partition_primitives(
        atoms.positions,
        atoms.cell.array,
        atoms.pbc,
        32,
    )
    expected_vectors, expected_indices, expected_valid = (
        native.periodic_knn_vectors(
            atoms.positions,
            atoms.cell.array,
            atoms.pbc,
            32,
        )
    )
    expected_labels = native.adaptive_cna_labels(
        expected_vectors,
        expected_indices,
        expected_valid,
    )

    np.testing.assert_allclose(vectors, expected_vectors, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(labels, expected_labels)


def test_phase_partition_primitives_keeps_python_fallback(monkeypatch):
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True).repeat((2, 2, 2))
    expected_vectors, expected_indices, expected_valid = periodic_knn_vectors(
        atoms.positions,
        atoms.cell.array,
        atoms.pbc,
        neighbors=32,
    )
    expected_labels = _adaptive_cna_reference(
        expected_vectors,
        expected_valid,
    )
    monkeypatch.setattr(phase_sketch_module, "_native_phase", None)

    vectors, indices, valid, labels = (
        phase_sketch_module.phase_partition_primitives(
            atoms.positions,
            atoms.cell.array,
            atoms.pbc,
            neighbors=32,
        )
    )

    np.testing.assert_array_equal(vectors, expected_vectors)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(labels, expected_labels)


def test_adaptive_cna_localizes_intrinsic_stacking_fault():
    sequence = "ABCABCABABCABC"
    atoms = _stacked_close_packed(sequence)
    vectors, indices, valid = accelerated_periodic_knn_vectors(
        atoms.positions, atoms.cell.array, atoms.pbc, neighbors=24
    )

    labels = adaptive_cna_labels(vectors, indices, valid)

    atoms_per_layer = len(atoms) // len(sequence)
    assert np.count_nonzero(labels == 2) == 2 * atoms_per_layer
    assert np.count_nonzero(labels == 1) == len(atoms) - 2 * atoms_per_layer
