from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, make_supercell

from NepTrainKit.core.audit import prototype_registry
from NepTrainKit.core.audit.phase_inventory import analyze_structure_phase
from NepTrainKit.core.audit.prototype_registry import (
    match_common_prototype,
    reference_crystallography,
)


def _fcc_sites(offset=(0.0, 0.0, 0.0)) -> np.ndarray:
    sites = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
        ),
        dtype=float,
    )
    return np.mod(sites + np.asarray(offset, dtype=float), 1.0)


def _cubic_sublattices(
    sublattices: tuple[tuple[str, np.ndarray], ...],
    *,
    a: float,
) -> Atoms:
    return Atoms(
        symbols=[
            symbol
            for symbol, positions in sublattices
            for _ in range(len(positions))
        ],
        scaled_positions=np.concatenate(
            [positions for _symbol, positions in sublattices],
            axis=0,
        ),
        cell=np.eye(3) * a,
        pbc=True,
    )


def _l10() -> Atoms:
    return Atoms(
        symbols=("Au", "Au", "Cu", "Cu"),
        scaled_positions=(
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ),
        cell=np.diag((3.82, 3.82, 3.70)),
        pbc=True,
    ).repeat((2, 2, 2))


def _b2() -> Atoms:
    return Atoms(
        symbols=("Fe", "Al"),
        scaled_positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
        cell=np.eye(3) * 4.12,
        pbc=True,
    ).repeat((3, 3, 3))


def _nias() -> Atoms:
    a, c = 3.62, 3.62 * 1.39
    cell = np.asarray(
        (
            (0.5 * a, -0.5 * np.sqrt(3.0) * a, 0.0),
            (0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0),
            (0.0, 0.0, c),
        )
    )
    return Atoms(
        symbols=("Ni", "Ni", "As", "As"),
        scaled_positions=(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5),
            (1.0 / 3.0, 2.0 / 3.0, 0.25),
            (2.0 / 3.0, 1.0 / 3.0, 0.75),
        ),
        cell=cell,
        pbc=True,
    ).repeat((2, 2, 2))


def _d03() -> Atoms:
    return _cubic_sublattices(
        (
            ("Al", _fcc_sites()),
            ("Fe", _fcc_sites((0.5, 0.5, 0.5))),
            ("Fe", _fcc_sites((0.25, 0.25, 0.25))),
            ("Fe", _fcc_sites((0.75, 0.75, 0.75))),
        ),
        a=5.78,
    ).repeat((2, 2, 2))


def _l21() -> Atoms:
    return _cubic_sublattices(
        (
            ("Al", _fcc_sites()),
            ("Mn", _fcc_sites((0.5, 0.5, 0.5))),
            ("Co", _fcc_sites((0.25, 0.25, 0.25))),
            ("Co", _fcc_sites((0.75, 0.75, 0.75))),
        ),
        a=5.95,
    ).repeat((2, 2, 2))


def _c1b() -> Atoms:
    return _cubic_sublattices(
        (
            ("Ni", _fcc_sites()),
            ("Mn", _fcc_sites((0.25, 0.25, 0.25))),
            ("Sb", _fcc_sites((0.5, 0.5, 0.5))),
        ),
        a=5.90,
    ).repeat((2, 2, 2))


def _d019() -> Atoms:
    a, c = 5.30, 5.30 * 0.81
    cell = np.asarray(
        (
            (0.5 * a, -0.5 * np.sqrt(3.0) * a, 0.0),
            (0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0),
            (0.0, 0.0, c),
        )
    )
    x = 5.0 / 6.0
    positions = np.mod(
        np.asarray(
            (
                (x, 2 * x, 0.25),
                (-2 * x, -x, 0.25),
                (x, -x, 0.25),
                (-x, -2 * x, 0.75),
                (2 * x, x, 0.75),
                (-x, x, 0.75),
                (1.0 / 3.0, 2.0 / 3.0, 0.25),
                (2.0 / 3.0, 1.0 / 3.0, 0.75),
            )
        ),
        1.0,
    )
    return Atoms(
        symbols=("Ti",) * 6 + ("Al",) * 2,
        scaled_positions=positions,
        cell=cell,
        pbc=True,
    ).repeat((2, 2, 2))


_FACTORIES: dict[str, Callable[[], Atoms]] = {
    "diamond": lambda: bulk("Si", "diamond", a=5.43, cubic=True).repeat(
        (2, 2, 2)
    ),
    "l10": _l10,
    "b1": lambda: bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat(
        (2, 2, 2)
    ),
    "b2": _b2,
    "b3": lambda: bulk("ZnS", "zincblende", a=5.41, cubic=True).repeat(
        (2, 2, 2)
    ),
    "b4": lambda: bulk(
        "ZnS",
        "wurtzite",
        a=3.82,
        c=6.238,
        u=0.375,
    ).repeat((2, 2, 2)),
    "fluorite": lambda: bulk(
        "CaF2", "fluorite", a=5.46, cubic=True
    ).repeat((2, 2, 2)),
    "nias": _nias,
    "d03": _d03,
    "l21": _l21,
    "c1b": _c1b,
    "d019": _d019,
}


def _match(atoms: Atoms):
    return match_common_prototype(
        atoms.positions,
        atoms.cell.array,
        atoms.pbc,
        atoms.numbers,
    )


def _nearest_distance(atoms: Atoms) -> float:
    distances = atoms.get_all_distances(mic=True)
    return float(np.min(distances[distances > 1.0e-8]))


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_independent_reference_prototypes_are_confirmed(label, factory):
    result = _match(factory())

    assert result.confirmed is True
    assert result.label == label
    assert result.geometry_match_fraction >= 0.82
    assert result.chemistry_match_fraction >= 0.80
    assert result.joint_match_fraction >= 0.80
    assert result.mean_shape_rms is not None


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_phase_inventory_uses_the_same_confirmed_prototype_contract(label, factory):
    result = analyze_structure_phase(factory())

    assert result.phase_label == label
    assert result.confidence_state == "strong"


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
@pytest.mark.parametrize("seed", (17, 4312, 90817))
def test_thermal_scale_distortion_keeps_the_correct_label(label, factory, seed):
    atoms = factory()
    displacement = 0.02 * _nearest_distance(atoms)
    atoms.positions += np.random.default_rng(seed).normal(
        0.0,
        displacement,
        size=(len(atoms), 3),
    )

    result = _match(atoms)

    assert result.confirmed is True
    assert result.label == label


@pytest.mark.parametrize(("label", "factory"), _FACTORIES.items())
def test_large_distortion_never_changes_one_supported_label_into_another(
    label,
    factory,
):
    atoms = factory()
    displacement = 0.12 * _nearest_distance(atoms)
    atoms.positions += np.random.default_rng(773).normal(
        0.0,
        displacement,
        size=(len(atoms), 3),
    )

    result = _match(atoms)

    assert result.label in {label, "unresolved"}


@pytest.mark.parametrize(
    "atoms",
    (
        bulk("Po", "sc", a=3.35, cubic=True).repeat((4, 4, 4)),
        Atoms(
            symbols=("Sr", "Ti", "O", "O", "O"),
            scaled_positions=(
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.0),
                (0.5, 0.0, 0.5),
                (0.0, 0.5, 0.5),
            ),
            cell=np.eye(3) * 3.9,
            pbc=True,
        ).repeat((3, 3, 3)),
        Atoms(
            symbols=("Si", "Si", "Cr", "Cr", "Cr", "Cr", "Cr", "Cr"),
            scaled_positions=(
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.5),
                (0.0, 0.5, 0.25),
                (0.0, 0.5, 0.75),
                (0.5, 0.25, 0.0),
                (0.5, 0.75, 0.0),
                (0.25, 0.0, 0.5),
                (0.75, 0.0, 0.5),
            ),
            cell=np.eye(3) * 4.56,
            pbc=True,
        ).repeat((3, 3, 3)),
    ),
    ids=("simple-cubic", "perovskite", "a15"),
)
def test_unsupported_competing_prototypes_fail_closed(atoms):
    result = _match(atoms)

    assert result.confirmed is False
    assert result.label == "unresolved"


@pytest.mark.parametrize("crystal", ("fcc", "bcc", "hcp"))
def test_random_binary_decoration_is_not_reported_as_ordered(crystal):
    atoms = bulk(
        "Cu",
        crystal,
        a=3.62,
        c=5.91 if crystal == "hcp" else None,
        cubic=crystal != "hcp",
    ).repeat((3, 3, 3))
    numbers = np.asarray(
        [29] * (len(atoms) // 2) + [79] * (len(atoms) - len(atoms) // 2)
    )
    np.random.default_rng(1181).shuffle(numbers)
    atoms.numbers = numbers

    result = _match(atoms)

    assert result.confirmed is False
    assert result.label == "unresolved"


@pytest.mark.parametrize(
    ("label", "factory"),
    (
        ("b1", _FACTORIES["b1"]),
        ("b2", _FACTORIES["b2"]),
        ("b3", _FACTORIES["b3"]),
        ("l21", _FACTORIES["l21"]),
    ),
)
def test_heavy_antisite_disorder_fails_closed(label, factory):
    atoms = factory()
    rng = np.random.default_rng(667)
    present = np.unique(atoms.numbers)
    first = np.flatnonzero(atoms.numbers == present[0])
    second = np.flatnonzero(atoms.numbers == present[1])
    count = max(2, int(np.ceil(0.20 * min(len(first), len(second)))))
    first_selected = rng.choice(first, count, replace=False)
    second_selected = rng.choice(second, count, replace=False)
    atoms.numbers[first_selected] = present[1]
    atoms.numbers[second_selected] = present[0]

    result = _match(atoms)

    assert result.label in {label, "unresolved"}
    assert result.label == "unresolved"


@pytest.mark.parametrize(
    ("label", "factory"),
    (("b1", _FACTORIES["b1"]), ("l21", _FACTORIES["l21"])),
)
def test_matching_is_invariant_to_order_origin_species_ids_and_cell_basis(
    label,
    factory,
):
    atoms = factory()
    rng = np.random.default_rng(928)
    permutation = rng.permutation(len(atoms))
    transformed = atoms[permutation]
    transformed.positions += np.asarray((1.31, -0.87, 2.04))
    remap = {
        number: replacement
        for number, replacement in zip(
            np.unique(transformed.numbers),
            (6, 14, 26),
        )
    }
    transformed.numbers = np.asarray(
        [remap[number] for number in transformed.numbers]
    )
    transformed = make_supercell(
        transformed,
        np.asarray(((1, 1, 0), (0, 1, 0), (0, 0, 1))),
        wrap=True,
    )

    result = _match(transformed)

    assert result.confirmed is True
    assert result.label == label


def test_partial_periodicity_fails_closed():
    atoms = _FACTORIES["b1"]()
    atoms.pbc = (True, True, False)

    result = _match(atoms)

    assert result.confirmed is False
    assert result.label == "unresolved"
    assert "three-dimensional periodicity" in result.reason


def test_phase_inventory_does_not_confirm_ordering_for_partial_periodicity():
    atoms = _cubic_sublattices(
        (
            ("Au", np.asarray(((0.0, 0.0, 0.0),))),
            (
                "Cu",
                np.asarray(
                    (
                        (0.0, 0.5, 0.5),
                        (0.5, 0.0, 0.5),
                        (0.5, 0.5, 0.0),
                    )
                ),
            ),
        ),
        a=3.75,
    ).repeat((4, 4, 4))
    atoms.pbc = (True, True, False)
    atoms.additional_fields = {"pbc": "T T F"}

    result = analyze_structure_phase(atoms)

    assert result.phase_label not in {
        "l12",
        "c14",
        "c15",
        *tuple(_FACTORIES),
    }


def test_distorted_un_example_is_confirmed_as_b1():
    atoms = Atoms(
        symbols=("U",) * 4 + ("N",) * 4,
        positions=np.asarray(
            (
                (4.79906556, 0.09870071, 0.02833032),
                (0.02872777, 2.33728343, 2.12431060),
                (2.37496850, 4.57417236, 2.45791805),
                (2.41401613, 2.38676186, 0.03652165),
                (2.59038323, 0.03956166, 0.19305613),
                (0.12145907, 2.26471201, 4.65278409),
                (4.73730682, 0.03911880, 2.36907009),
                (2.32380429, 2.25533789, 2.21070554),
            )
        ),
        cell=np.eye(3) * 4.8114,
        pbc=True,
    )

    result = _match(atoms)

    assert result.confirmed is True
    assert result.label == "b1"
    assert result.joint_match_fraction == pytest.approx(1.0)
    assert result.mean_shape_rms == pytest.approx(0.0534689347)

    phase = analyze_structure_phase(atoms, source_index=23)

    assert phase.source_index == 23
    assert phase.phase_label == "b1"
    assert phase.confidence_state == "strong"
    assert dict(phase.local_phase_fractions) == {
        "fcc": 0.0,
        "hcp": 0.0,
        "bcc": 0.0,
        "unresolved": 1.0,
    }


@pytest.mark.parametrize(
    ("label", "pearson", "space_group", "number"),
    (
        ("fcc", "cF4", "Fm-3m", 225),
        ("bcc", "cI2", "Im-3m", 229),
        ("hcp", "hP2", "P6₃/mmc", 194),
        ("diamond", "cF8", "Fd-3m", 227),
        ("l10", "tP2", "P4/mmm", 123),
        ("l12", "cP4", "Pm-3m", 221),
        ("b1", "cF8", "Fm-3m", 225),
        ("b2", "cP2", "Pm-3m", 221),
        ("b3", "cF8", "F-43m", 216),
        ("b4", "hP4", "P6₃mc", 186),
        ("fluorite", "cF12", "Fm-3m", 225),
        ("nias", "hP4", "P6₃/mmc", 194),
        ("d03", "cF16", "Fm-3m", 225),
        ("l21", "cF16", "Fm-3m", 225),
        ("c1b", "cF12", "F-43m", 216),
        ("d019", "hP8", "P6₃/mmc", 194),
        ("c14", "hP12", "P6₃/mmc", 194),
        ("c15", "cF24", "Fd-3m", 227),
    ),
)
def test_reference_crystallography_is_exact(
    label,
    pearson,
    space_group,
    number,
):
    reference = reference_crystallography(label)

    assert reference is not None
    assert reference.pearson == pearson
    assert reference.space_group == space_group
    assert reference.space_group_number == number


def test_unknown_reference_crystallography_returns_none():
    assert reference_crystallography("not-a-prototype") is None


def test_candidate_restriction_never_returns_a_disallowed_label():
    atoms = _FACTORIES["b1"]()

    excluded = match_common_prototype(
        atoms.positions,
        atoms.cell.array,
        atoms.pbc,
        atoms.numbers,
        candidate_labels=("b2",),
    )
    included = match_common_prototype(
        atoms.positions,
        atoms.cell.array,
        atoms.pbc,
        atoms.numbers,
        candidate_labels=("b1",),
    )

    assert excluded.confirmed is False
    assert excluded.label == "unresolved"
    assert included.confirmed is True
    assert included.label == "b1"


def test_empty_candidate_set_skips_neighbor_search(monkeypatch):
    atoms = _FACTORIES["b1"]()

    def unexpected_neighbor_search(*_args, **_kwargs):
        raise AssertionError("empty candidate set must not calculate neighbors")

    monkeypatch.setattr(
        prototype_registry._phase_sketch,
        "accelerated_periodic_knn_vectors",
        unexpected_neighbor_search,
    )

    result = match_common_prototype(
        atoms.positions,
        atoms.cell.array,
        atoms.pbc,
        atoms.numbers,
        candidate_labels=(),
    )

    assert result.confirmed is False
    assert result.label == "unresolved"


def test_large_structure_shape_descriptors_are_evaluated_in_bounded_batches(
    monkeypatch,
):
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((9, 9, 9))
    observed_batch_sizes = []
    original = prototype_registry._batch_shape_descriptors

    def record_batch(vectors):
        observed_batch_sizes.append(len(vectors))
        return original(vectors)

    monkeypatch.setattr(
        prototype_registry,
        "_batch_shape_descriptors",
        record_batch,
    )

    result = match_common_prototype(
        atoms.positions,
        atoms.cell.array,
        atoms.pbc,
        atoms.numbers,
        candidate_labels=("b1",),
    )

    assert result.confirmed is True
    assert result.label == "b1"
    assert len(atoms) > prototype_registry._DESCRIPTOR_BATCH_SIZE
    assert observed_batch_sizes
    assert max(observed_batch_sizes) <= prototype_registry._DESCRIPTOR_BATCH_SIZE
