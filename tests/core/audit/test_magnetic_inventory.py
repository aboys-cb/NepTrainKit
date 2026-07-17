from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from NepTrainKit.core.audit.magnetic_inventory import (
    build_magnetic_inventory,
    summarize_magnetic_inventory,
)
from NepTrainKit.core.audit.result import CompositionPoint, DatasetInventory
from NepTrainKit.core.geometry_cache import GeometrySnapshot


def _grid(size: int = 4, cell: np.ndarray | None = None):
    cell = np.asarray(cell if cell is not None else np.eye(3) * size, dtype=np.float32)
    integer = np.asarray(
        [(i, j, k) for i in range(size) for j in range(size) for k in range(size)],
        dtype=np.float32,
    )
    positions = (integer / size) @ cell
    return integer, np.ascontiguousarray(positions), cell


def _native_evidence(spins: np.ndarray, *, cell: np.ndarray | None = None):
    from NepTrainKit._native import _magnetism

    _, positions, cell = _grid(cell=cell)
    return _magnetism.magnetic_order_evidence(
        positions,
        cell,
        np.ones(3, dtype=bool),
        np.ascontiguousarray(spins, dtype=np.float32),
        12,
        3,
    )


def _reference_spins() -> dict[str, np.ndarray]:
    integer, _, _ = _grid()
    x = integer[:, 0].astype(int)
    phase = 2.0 * np.pi * integer[:, 0] / 4.0
    rng = np.random.default_rng(42)
    random = rng.normal(size=(len(integer), 3))
    random /= np.linalg.norm(random, axis=1)[:, None]
    return {
        "fm": np.tile([0.0, 0.0, 1.0], (len(integer), 1)),
        "afm": np.column_stack(
            [np.zeros(len(x)), np.zeros(len(x)), np.where(x % 2, -1.0, 1.0)]
        ),
        "ferrimagnetic": np.column_stack(
            [np.zeros(len(x)), np.zeros(len(x)), np.where(x % 2, -0.5, 1.0)]
        ),
        "spin_spiral": np.column_stack(
            [np.cos(phase), np.sin(phase), np.zeros(len(phase))]
        ),
        "spin_disordered": random,
        "low_moment": np.zeros((len(integer), 3)),
    }


@pytest.mark.parametrize("expected", tuple(_reference_spins()))
def test_native_classifies_reference_spin_patterns(expected: str):
    values = _native_evidence(_reference_spins()[expected])

    assert values[14] == expected
    assert values[15] == "strong"


def test_native_evidence_is_rotation_scale_and_cell_shape_invariant():
    spins = _reference_spins()["spin_spiral"]
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    triclinic = np.asarray(
        [[4.0, 0.0, 0.0], [1.1, 3.8, 0.0], [0.5, 0.7, 4.2]],
        dtype=np.float32,
    )
    baseline = _native_evidence(spins)
    transformed = _native_evidence(2.5 * spins @ rotation, cell=triclinic)

    assert transformed[14] == baseline[14] == "spin_spiral"
    np.testing.assert_allclose(transformed[3:11], baseline[3:11], atol=2.0e-6)
    assert transformed[11:14] == baseline[11:14]


def test_native_resolves_element_local_order_and_pair_coupling():
    from NepTrainKit._native import _magnetism

    integer, positions, cell = _grid()
    x = integer[:, 0].astype(int)
    atomic_numbers = np.where(integer[:, 1].astype(int) % 2, 28, 26).astype(np.int16)
    spins = np.zeros((len(positions), 3), dtype=np.float32)
    fe = atomic_numbers == 26
    ni = atomic_numbers == 28
    spins[fe, 2] = np.where(x[fe] % 2, -1.0, 1.0)
    spins[ni, 2] = 1.0

    element_rows, pair_rows = _magnetism.element_magnetic_evidence(
        positions, cell, np.ones(3, dtype=bool), spins, atomic_numbers, 12, 3
    )
    by_number = {int(row[0]): row for row in element_rows}

    assert by_number[26][12] == "compensated"
    assert by_number[26][4] == pytest.approx(0.0, abs=1.0e-7)
    assert by_number[26][8] == pytest.approx(1.0, abs=1.0e-6)
    assert by_number[28][12] == "aligned"
    assert by_number[28][4] == pytest.approx(1.0, abs=1.0e-7)
    assert pair_rows[0][4] == "mixed"

    spins[fe, 2] = 1.0
    spins[ni, 2] = -1.0
    _, antiparallel_pairs = _magnetism.element_magnetic_evidence(
        positions, cell, np.ones(3, dtype=bool), spins, atomic_numbers, 12, 3
    )
    assert antiparallel_pairs[0][3] == pytest.approx(-1.0, abs=1.0e-7)
    assert antiparallel_pairs[0][4] == "antiparallel"

    triclinic = np.asarray(
        [[4.0, 0.0, 0.0], [1.1, 3.8, 0.0], [0.5, 0.7, 4.2]],
        dtype=np.float32,
    )
    _, triclinic_positions, _ = _grid(cell=triclinic)
    transformed_elements, transformed_pairs = _magnetism.element_magnetic_evidence(
        triclinic_positions, triclinic, np.ones(3, dtype=bool),
        spins, atomic_numbers, 12, 3,
    )
    assert {int(row[0]): row[12] for row in transformed_elements} == {
        26: "aligned", 28: "aligned",
    }
    assert transformed_pairs[0][3] == pytest.approx(-1.0, abs=1.0e-7)
    assert transformed_pairs[0][4] == "antiparallel"


def test_single_atom_element_is_insufficient_not_low_moment():
    from NepTrainKit._native import _magnetism

    _, positions, cell = _grid()
    atomic_numbers = np.full(len(positions), 28, dtype=np.int16)
    atomic_numbers[0] = 26
    spins = np.tile(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), (len(positions), 1))

    element_rows, _ = _magnetism.element_magnetic_evidence(
        positions, cell, np.ones(3, dtype=bool), spins, atomic_numbers, 12, 3
    )
    by_number = {int(row[0]): row for row in element_rows}

    assert by_number[26][3] == pytest.approx(1.0)
    assert by_number[26][12] == "insufficient"
    assert by_number[28][12] == "aligned"


def test_nonperiodic_element_compensation_uses_local_spin_correlation():
    from NepTrainKit._native import _magnetism

    positions = np.column_stack((
        np.arange(8, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
    ))
    cell = np.eye(3, dtype=np.float32) * 12.0
    spins = np.column_stack((
        np.zeros(8, dtype=np.float32),
        np.zeros(8, dtype=np.float32),
        np.where(np.arange(8) % 2, -1.0, 1.0).astype(np.float32),
    ))
    element_rows, _ = _magnetism.element_magnetic_evidence(
        positions, cell, np.zeros(3, dtype=bool), spins,
        np.full(8, 26, dtype=np.int16), 2, 3,
    )

    assert element_rows[0][8] == pytest.approx(0.0)
    assert element_rows[0][6] < -0.25
    assert element_rows[0][12] == "compensated"


def test_inventory_uses_only_spin_and_caches_complete_result():
    integer, positions, cell = _grid()
    spins = _reference_spins()["fm"]
    geometry = GeometrySnapshot(
        source_indices=np.asarray([0, 1], dtype=np.int64),
        positions=np.concatenate((positions, positions), axis=0),
        atom_offsets=np.asarray([0, len(positions), 2 * len(positions)], dtype=np.int64),
        cells=np.stack((cell, cell)),
        pbc=np.ones((2, 3), dtype=np.uint8),
        atomic_numbers=np.full(2 * len(positions), 26, dtype=np.int16),
    )
    point = CompositionPoint(
        reduced_counts=(1,),
        fractions=(1.0,),
        structure_count=2,
        share=1.0,
        structure_indices=(0, 1),
    )
    inventory = DatasetInventory(
        structure_count=2,
        elements=("Fe",),
        composition_points=(point,),
    )
    structures = (
        SimpleNamespace(atomic_properties={"spin": spins}),
        SimpleNamespace(atomic_properties={"force_mag": spins, "mforce": spins}),
    )

    class Cache:
        value = None

        def cached_geometry_analysis(self, _namespace, _key, build):
            if self.value is None:
                self.value = build()
                return self.value, False
            return self.value, True

    cache = Cache()
    result, hit = build_magnetic_inventory(
        geometry, inventory, structures, cache_owner=cache
    )
    cached, cached_hit = build_magnetic_inventory(
        geometry, inventory, structures, cache_owner=cache
    )

    assert hit is False
    assert cached_hit is True
    assert cached is result
    assert result.source_structure_count == 2
    assert result.analyzed_structure_count == 1
    assert result.missing_spin_count == 1
    assert result.composition_points[0].order_fractions == (("fm", 1.0),)
    element_summary = result.composition_points[0].element_summaries[0]
    assert element_summary.element == "Fe"
    assert element_summary.order_fractions == (("aligned", 1.0),)
    summary = summarize_magnetic_inventory(result)
    assert summary is not None
    assert summary.order_fractions == (("fm", 1.0),)
    assert summary.missing_spin_count == 1
