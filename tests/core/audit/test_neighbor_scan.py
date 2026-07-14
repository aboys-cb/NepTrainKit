import numpy as np
import pytest

from NepTrainKit.core.audit import neighbor_scan
from NepTrainKit.core.audit.neighbor_scan import (
    cutoff_neighbor_pairs_batch,
    find_short_distance_structure_rows,
    periodic_cell_statuses,
)


def test_batch_scan_handles_nonperiodic_and_orthorhombic_periodic_pairs():
    rows = find_short_distance_structure_rows(
        positions_by_structure=[
            np.asarray([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]),
            np.asarray([[0.1, 1.0, 1.0], [4.9, 1.0, 1.0]]),
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ],
        cells=[np.eye(3) * 5.0] * 3,
        pbc_flags=[
            np.asarray([False, False, False]),
            np.asarray([True, True, True]),
            np.asarray([False, False, False]),
        ],
        cutoff=0.5,
    )

    assert rows == (0, 1)


def test_batch_scan_handles_triclinic_minimum_image():
    cell = np.asarray(
        [
            [4.0, 0.0, 0.0],
            [1.2, 3.7, 0.0],
            [0.4, 0.6, 4.2],
        ]
    )
    first_fractional = np.asarray([0.01, 0.01, 0.01])
    second_fractional = np.asarray([0.99, 0.99, 0.99])

    rows = find_short_distance_structure_rows(
        [np.stack([first_fractional @ cell, second_fractional @ cell])],
        [cell],
        [np.asarray([True, True, True])],
        cutoff=0.5,
    )

    assert rows == (0,)


def test_batch_scan_falls_back_for_valid_singular_partial_periodic_cell():
    cell = np.asarray(
        [
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    rows = find_short_distance_structure_rows(
        [np.asarray([[0.1, 0.1, 2.0], [4.9, 4.9, 2.0]])],
        [cell],
        [np.asarray([True, True, False])],
        cutoff=0.5,
    )

    assert rows == (0,)


def test_cell_validation_supports_orthogonal_triclinic_and_partial_periodic_cells():
    cells = [
        np.eye(3) * 5.0,
        np.asarray([[4.0, 0.0, 0.0], [1.1, 3.8, 0.0], [0.4, 0.7, 4.2]]),
        np.asarray([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]]),
        np.asarray([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
    ]
    pbc = [
        np.asarray([True, True, True]),
        np.asarray([True, True, True]),
        np.asarray([True, True, False]),
        np.asarray([True, True, True]),
    ]

    statuses = periodic_cell_statuses(cells, pbc)

    assert tuple(bool(status & 1) for status in statuses) == (True, True, True, False)


def test_native_and_reference_scans_match_for_random_triclinic_cells(monkeypatch):
    if neighbor_scan._native_scan is None:
        pytest.skip("native audit extension is not built")
    rng = np.random.default_rng(20260712)
    cells = []
    positions = []
    pbc = []
    for row in range(96):
        cell = np.asarray(
            [
                [rng.uniform(3.0, 6.0), 0.0, 0.0],
                [rng.uniform(-1.5, 1.5), rng.uniform(3.0, 6.0), 0.0],
                [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(3.0, 6.0)],
            ]
        )
        fractional = rng.random((8, 3))
        if row % 3 == 0:
            fractional[1] = np.mod(fractional[0] + np.asarray([0.01, -0.01, 0.01]), 1.0)
        cells.append(cell)
        positions.append(fractional @ cell)
        pbc.append(np.asarray([True, True, True]))

    native_rows = find_short_distance_structure_rows(positions, cells, pbc, cutoff=0.5)
    monkeypatch.setattr(neighbor_scan, "_native_scan", None)
    reference_rows = find_short_distance_structure_rows(positions, cells, pbc, cutoff=0.5)

    assert native_rows == reference_rows


def test_native_cutoff_pairs_match_reference_for_orthogonal_and_triclinic_cells(monkeypatch):
    if neighbor_scan._native_scan is None or not hasattr(
        neighbor_scan._native_scan,
        "cutoff_neighbor_pairs",
    ):
        pytest.skip("native audit extension is not built")
    cells = [
        np.eye(3) * 4.0,
        np.asarray([[4.0, 0.0, 0.0], [1.3, 3.8, 0.0], [0.4, 0.7, 4.1]]),
    ]
    fractional = [
        np.asarray([[0.05, 0.05, 0.05], [0.95, 0.05, 0.05], [0.5, 0.5, 0.5]]),
        np.asarray([[0.03, 0.04, 0.05], [0.97, 0.96, 0.95], [0.4, 0.6, 0.3]]),
    ]
    positions = [values @ cell for values, cell in zip(fractional, cells)]
    pbc = [np.asarray([True, True, True])] * 2

    native = cutoff_neighbor_pairs_batch(positions, cells, pbc, cutoff=2.0)
    monkeypatch.setattr(neighbor_scan, "_native_scan", None)
    reference = cutoff_neighbor_pairs_batch(positions, cells, pbc, cutoff=2.0)

    assert np.array_equal(native[0], reference[0])
    for row in range(2):
        native_slice = slice(int(native[0][row]), int(native[0][row + 1]))
        reference_slice = slice(int(reference[0][row]), int(reference[0][row + 1]))
        native_pairs = sorted(
            zip(
                native[1][native_slice].tolist(),
                native[2][native_slice].tolist(),
                np.round(native[3][native_slice], 10).tolist(),
            )
        )
        reference_pairs = sorted(
            zip(
                reference[1][reference_slice].tolist(),
                reference[2][reference_slice].tolist(),
                np.round(reference[3][reference_slice], 10).tolist(),
            )
        )
        assert native_pairs == reference_pairs
