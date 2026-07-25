from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase.data import chemical_symbols

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

    size = round(len(spins) ** (1.0 / 3.0))
    _, positions, cell = _grid(size=size, cell=cell)
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
        "noncollinear": np.column_stack(
            [np.cos(phase), np.sin(phase), np.zeros(len(phase))]
        ),
        "pm_like": random,
        "low_moment": np.zeros((len(integer), 3)),
    }


@pytest.mark.parametrize("expected", tuple(_reference_spins()))
def test_native_classifies_reference_spin_patterns(expected: str):
    values = _native_evidence(_reference_spins()[expected])

    assert values[14] == expected
    assert values[15] == "strong"


def test_native_evidence_is_rotation_scale_and_cell_shape_invariant():
    spins = _reference_spins()["noncollinear"]
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

    assert transformed[14] == baseline[14] == "noncollinear"
    np.testing.assert_allclose(transformed[3:11], baseline[3:11], atol=2.0e-6)
    assert transformed[11:14] == baseline[11:14]


def _inventory_structure(
    spins: np.ndarray,
    *,
    atomic_numbers: np.ndarray | None = None,
    cell: np.ndarray | None = None,
    positions: np.ndarray | None = None,
):
    size = round(len(spins) ** (1.0 / 3.0))
    _, default_positions, cell = _grid(size=size, cell=cell)
    positions = default_positions if positions is None else np.asarray(
        positions, dtype=np.float32
    )
    numbers = (
        np.full(len(positions), 26, dtype=np.int16)
        if atomic_numbers is None
        else np.asarray(atomic_numbers, dtype=np.int16)
    )
    geometry = GeometrySnapshot(
        source_indices=np.asarray([0], dtype=np.int64),
        positions=positions,
        atom_offsets=np.asarray([0, len(positions)], dtype=np.int64),
        cells=np.asarray([cell], dtype=np.float32),
        pbc=np.ones((1, 3), dtype=np.uint8),
        atomic_numbers=numbers,
    )
    unique_numbers = tuple(sorted(int(number) for number in np.unique(numbers)))
    counts = tuple(int(np.count_nonzero(numbers == number)) for number in unique_numbers)
    divisor = int(np.gcd.reduce(counts))
    point = CompositionPoint(
        reduced_counts=tuple(count // divisor for count in counts),
        fractions=tuple(count / len(numbers) for count in counts),
        structure_count=1,
        share=1.0,
        structure_indices=(0,),
    )
    inventory = DatasetInventory(
        structure_count=1,
        elements=tuple(chemical_symbols[number] for number in unique_numbers),
        composition_points=(point,),
    )
    result, _ = build_magnetic_inventory(
        geometry,
        inventory,
        (SimpleNamespace(atomic_properties={"spin": spins}),),
    )
    return result.composition_points[0].structures[0]


def test_one_periodic_layer_sequence_identifies_afm_subtypes():
    integer2, _, _ = _grid(size=2)
    x2 = integer2[:, 0].astype(int)
    one_layered_period = np.column_stack(
        (np.zeros(len(x2)), np.zeros(len(x2)), np.where(x2 % 2, -1.0, 1.0))
    )
    integer4, _, _ = _grid(size=4)
    x4 = integer4[:, 0].astype(int)
    one_double_period = np.column_stack(
        (
            np.zeros(len(x4)),
            np.zeros(len(x4)),
            np.where((x4 // 2) % 2, -1.0, 1.0),
        )
    )
    integer8, _, _ = _grid(size=8)
    x8 = integer8[:, 0].astype(int)
    repeated_double = np.column_stack(
        (
            np.zeros(len(x8)),
            np.zeros(len(x8)),
            np.where((x8 // 2) % 2, -1.0, 1.0),
        )
    )

    layered_result = _inventory_structure(one_layered_period)
    one_double_result = _inventory_structure(one_double_period)
    repeated_double_result = _inventory_structure(repeated_double)

    assert layered_result.order_label == "afm"
    assert layered_result.order_subtype == "layered"
    assert one_double_result.order_label == "afm"
    assert one_double_result.order_subtype == "double_layered"
    assert repeated_double_result.order_label == "afm"
    assert repeated_double_result.order_subtype == "double_layered"


def test_double_layer_afm_subtype_survives_spin_rotation_and_small_distortion():
    integer, positions, _ = _grid(size=8)
    triclinic = np.asarray(
        [[8.0, 0.0, 0.0], [1.1, 7.7, 0.0], [0.6, 0.9, 8.2]],
        dtype=np.float32,
    )
    fractional = integer / 8.0
    rng = np.random.default_rng(7)
    fractional += rng.uniform(-0.003, 0.003, size=fractional.shape)
    positions = np.ascontiguousarray(fractional @ triclinic, dtype=np.float32)
    layer = integer[:, 2].astype(int)
    direction = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    spins = np.where(((layer // 2) % 2)[:, None], -direction, direction)

    structure = _inventory_structure(
        spins,
        cell=triclinic,
        positions=positions,
    )

    assert structure.order_label == "afm"
    assert structure.order_subtype == "double_layered"


def test_zero_moment_element_does_not_hide_magnetic_sublattice_order():
    integer, _, _ = _grid()
    silicon = integer[:, 1].astype(int) % 2 == 1
    atomic_numbers = np.where(silicon, 14, 26).astype(np.int16)
    spins = np.zeros((len(integer), 3), dtype=np.float32)
    spins[~silicon, 2] = 2.0

    structure = _inventory_structure(spins, atomic_numbers=atomic_numbers)

    assert structure.order_label == "fm"
    by_element = {item.element: item for item in structure.element_evidence}
    assert by_element["Fe"].order_label == "aligned"
    assert by_element["Si"].order_label == "low_moment"


def test_single_active_spin_is_unresolved_not_nonmagnetic():
    spins = np.zeros((64, 3), dtype=np.float32)
    spins[0, 2] = 1.0

    values = _native_evidence(spins)

    assert values[14] == "unresolved"
    assert values[15] == "unresolved"


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
