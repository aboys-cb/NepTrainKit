import importlib

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.spacegroup import crystal

from NepTrainKit.core.audit.phase_refinement import (
    _c14_prototype,
    _c15_prototype,
    _c36_prototype,
    _fcc_prototype,
    _repeat_crystal,
    refine_l12,
    refine_laves,
)
from NepTrainKit.core.audit.phase_sketch import periodic_knn_vectors


phase_sketch_module = importlib.import_module("NepTrainKit.core.audit.phase_sketch")
phase_refinement_module = importlib.import_module(
    "NepTrainKit.core.audit.phase_refinement"
)


def _distort(
    positions,
    cell,
    seed,
    *,
    strain=0.0,
    noise=0.0,
    shear=0.0,
):
    rng = np.random.default_rng(seed)
    random_matrix = rng.normal(size=(3, 3))
    symmetric = 0.5 * (random_matrix + random_matrix.T)
    symmetric /= max(float(np.linalg.norm(symmetric, ord=2)), 1.0)
    deformation = np.eye(3) + strain * symmetric
    deformation += np.triu(rng.uniform(-shear, shear, size=(3, 3)), k=1)
    distorted_cell = np.asarray(cell) @ deformation.T
    distorted_positions = np.asarray(positions) @ deformation.T
    vectors, _, valid = periodic_knn_vectors(
        distorted_positions, distorted_cell, (True, True, True), neighbors=1
    )
    nearest = float(np.median(np.linalg.norm(vectors[valid], axis=1)))
    distorted_positions += rng.normal(
        scale=noise * nearest, size=distorted_positions.shape
    )
    return distorted_positions, distorted_cell


def _random_groups(atom_count, a_fraction, seed):
    a_count = int(round(atom_count * a_fraction))
    types = np.ones(atom_count, dtype=np.int32)
    types[np.random.default_rng(seed).choice(atom_count, a_count, replace=False)] = 0
    return types


def _a15_positions():
    fractional = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.0, 0.5, 0.25),
            (0.0, 0.5, 0.75),
            (0.5, 0.25, 0.0),
            (0.5, 0.75, 0.0),
            (0.25, 0.0, 0.5),
            (0.75, 0.0, 0.5),
        )
    )
    atoms = Atoms(
        symbols="Cr8",
        scaled_positions=fractional,
        cell=np.eye(3) * 4.56,
        pbc=True,
    ).repeat((3, 3, 3))
    return atoms.positions, atoms.cell.array


def _sigma_positions():
    a, c_over_a = 8.7966, 0.518177
    x2, x3, y3 = 0.39864, 0.46349, 0.13122
    x4, y4, x5, z5 = 0.73933, 0.06609, 0.18267, 0.25202
    fractional = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (x2, x2, 0.0),
            (-x2, -x2, 0.0),
            (0.5 - x2, 0.5 + x2, 0.5),
            (0.5 + x2, 0.5 - x2, 0.5),
            (x3, y3, 0.0),
            (-x3, -y3, 0.0),
            (0.5 - y3, 0.5 + x3, 0.5),
            (0.5 + y3, 0.5 - x3, 0.5),
            (0.5 - x3, 0.5 + y3, 0.5),
            (0.5 + x3, 0.5 - y3, 0.5),
            (y3, x3, 0.0),
            (-y3, -x3, 0.0),
            (x4, y4, 0.0),
            (-x4, -y4, 0.0),
            (0.5 - y4, 0.5 + x4, 0.5),
            (0.5 + y4, 0.5 - x4, 0.5),
            (0.5 - x4, 0.5 + y4, 0.5),
            (0.5 + x4, 0.5 - y4, 0.5),
            (y4, x4, 0.0),
            (-y4, -x4, 0.0),
            (x5, x5, z5),
            (-x5, -x5, z5),
            (0.5 - x5, 0.5 + x5, 0.5 + z5),
            (0.5 + x5, 0.5 - x5, 0.5 + z5),
            (0.5 - x5, 0.5 + x5, 0.5 - z5),
            (0.5 + x5, 0.5 - x5, 0.5 - z5),
            (x5, x5, -z5),
            (-x5, -x5, -z5),
        )
    )
    cell = np.diag((a, a, a * c_over_a))
    return np.mod(fractional, 1.0) @ cell, cell


def _body_centered_tetragonal_cell(a, c_over_a):
    c = a * c_over_a
    return np.asarray(
        (
            (-0.5 * a, 0.5 * a, 0.5 * c),
            (0.5 * a, -0.5 * a, 0.5 * c),
            (0.5 * a, 0.5 * a, -0.5 * c),
        )
    )


def _d022_prototype():
    cell = _body_centered_tetragonal_cell(3.8537, 2.22744)
    fractional = np.asarray(
        ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.75, 0.25, 0.5), (0.25, 0.75, 0.5))
    )
    types = np.asarray((1, 0, 1, 1), dtype=np.int32)
    return _repeat_crystal(cell, fractional, types, (3, 3, 3))


def _d023_prototype():
    z3, z4 = 0.37498, 0.11886
    cell = _body_centered_tetragonal_cell(3.9993, 4.32151)
    fractional = np.asarray(
        (
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
            (0.75, 0.25, 0.5),
            (0.25, 0.75, 0.5),
            (z3, z3, 0.0),
            (-z3, -z3, 0.0),
            (z4, z4, 0.0),
            (-z4, -z4, 0.0),
        )
    )
    types = np.asarray((1, 1, 1, 1, 1, 1, 0, 0), dtype=np.int32)
    return _repeat_crystal(cell, np.mod(fractional, 1.0), types, (3, 3, 3))


@pytest.mark.parametrize(
    ("builder", "expected_label", "expected_confirmed"),
    (
        (_c14_prototype, "c14", True),
        (_c15_prototype, "c15", True),
        (_c36_prototype, "c36_like_or_mixed", False),
    ),
)
def test_laves_standard_prototypes_are_classified_conservatively(
    builder, expected_label, expected_confirmed
):
    positions, cell, types = builder()
    result = refine_laves(positions, cell, (True, True, True), types)

    assert result.label == expected_label
    assert result.confirmed is expected_confirmed
    assert result.geometry_match_fraction == pytest.approx(1.0)
    assert result.joint_match_fraction == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("builder", "expected_label"),
    ((_c14_prototype, "c14"), (_c15_prototype, "c15")),
)
def test_laves_moderate_distortion_passes_but_extreme_noise_fails_closed(
    builder, expected_label
):
    positions, cell, types = builder()
    for seed in range(6):
        moderate_positions, moderate_cell = _distort(
            positions, cell, 31 + seed, strain=0.06, noise=0.05, shear=0.08
        )
        extreme_positions, extreme_cell = _distort(
            positions, cell, 131 + seed, strain=0.15, noise=0.15, shear=0.20
        )

        moderate = refine_laves(
            moderate_positions, moderate_cell, (True, True, True), types
        )
        extreme = refine_laves(
            extreme_positions, extreme_cell, (True, True, True), types
        )

        assert moderate.label == expected_label
        assert moderate.confirmed
        assert extreme.label == "unknown"
        assert not extreme.confirmed


def test_laves_rejects_common_crystals_with_random_ab2_decoration():
    controls = []
    for atoms in (
        bulk("Cu", "fcc", a=3.62, cubic=True).repeat((3, 3, 3)),
        bulk("Fe", "bcc", a=2.88, cubic=True).repeat((3, 3, 3)),
        bulk("Ti", "hcp", a=2.95, c=4.68, orthorhombic=True).repeat((3, 3, 3)),
        bulk("Si", "diamond", a=5.43, cubic=True).repeat((3, 3, 3)),
        bulk("Po", "sc", a=3.35, cubic=True).repeat((3, 3, 3)),
    ):
        controls.append((atoms.positions, atoms.cell.array))
    controls.append(_a15_positions())
    controls.append(_sigma_positions())
    fluorite = bulk("CaF2", "fluorite", a=5.46, cubic=True).repeat((2, 2, 2))
    controls.append((fluorite.positions, fluorite.cell.array))

    maximum_joint_match = 0.0
    for control_index, (positions, cell) in enumerate(controls):
        for seed in range(8):
            types = _random_groups(len(positions), 1.0 / 3.0, 100 * control_index + seed)
            distorted_positions, distorted_cell = _distort(
                positions,
                cell,
                1000 + 100 * control_index + seed,
                strain=0.04,
                noise=0.03,
                shear=0.05,
            )
            result = refine_laves(
                distorted_positions,
                distorted_cell,
                (True, True, True),
                types,
            )
            maximum_joint_match = max(maximum_joint_match, result.joint_match_fraction)
            assert not result.confirmed
            assert result.label == "unknown"

    assert maximum_joint_match < 0.30


def test_laves_rejects_ordered_non_laves_ab2_compounds():
    fluorite = bulk("CaF2", "fluorite", a=5.46, cubic=True).repeat((2, 2, 2))
    rutile = crystal(
        ("Ti", "O"),
        basis=((0.0, 0.0, 0.0), (0.305, 0.305, 0.0)),
        spacegroup=136,
        cellpar=(4.594, 4.594, 2.959, 90, 90, 90),
    ).repeat((3, 3, 3))
    c11b = crystal(
        ("Mo", "Si"),
        basis=((0.0, 0.0, 0.0), (0.0, 0.0, 0.335)),
        spacegroup=139,
        cellpar=(3.20, 3.20, 7.85, 90, 90, 90),
    ).repeat((3, 3, 3))

    for atoms in (fluorite, rutile, c11b):
        result = refine_laves(
            atoms.positions, atoms.cell.array, atoms.pbc, atoms.numbers
        )
        assert result.label == "unknown"
        assert not result.confirmed


def test_laves_is_invariant_to_order_ids_translation_and_cell_basis():
    positions, cell, types = _c15_prototype()
    reference = refine_laves(positions, cell, (True, True, True), types)
    rng = np.random.default_rng(73)
    permutation = rng.permutation(len(positions))
    relabeled = np.where(types[permutation] == 0, 101, 7)
    translated = positions[permutation] + np.asarray((7.3, -4.1, 2.7))
    unimodular = np.asarray(((1, 1, 0), (0, 1, 0), (0, 0, 1)), dtype=float)

    transformed = refine_laves(
        translated,
        unimodular @ cell,
        (True, True, True),
        relabeled,
    )

    assert transformed.label == reference.label == "c15"
    assert transformed.confirmed and reference.confirmed
    assert transformed.joint_match_fraction == pytest.approx(
        reference.joint_match_fraction, abs=1.0e-12
    )
    assert transformed.b2_fraction == pytest.approx(reference.b2_fraction, abs=1.0e-12)


def test_multicomponent_laves_requires_explicit_unambiguous_roles():
    positions, cell, types = _c14_prototype()
    split_types = types.copy()
    b_sites = np.flatnonzero(split_types == 1)
    split_types[b_sites[::2]] = 2

    automatic = refine_laves(positions, cell, (True, True, True), split_types)
    explicit = refine_laves(
        positions,
        cell,
        (True, True, True),
        split_types,
        a_types=(0,),
        b_types=(1, 2),
    )

    assert automatic.label == "unknown"
    assert "ambiguous" in automatic.reason
    assert explicit.label == "c14"
    assert explicit.confirmed


def test_laves_keeps_a_single_point_defect_but_exposes_unmatched_environments():
    positions, cell, types = _c14_prototype()
    defective_positions = np.delete(positions, 0, axis=0)
    defective_types = np.delete(types, 0)

    result = refine_laves(
        defective_positions, cell, (True, True, True), defective_types
    )

    assert result.label == "c14"
    assert result.confirmed
    assert 0.0 < result.defect_fraction < 0.15


def test_l12_standard_and_moderately_distorted_structures_are_confirmed():
    positions, cell, types = _fcc_prototype()
    ideal = refine_l12(positions, cell, (True, True, True), types)
    assert ideal.label == "l12"
    assert ideal.confirmed
    for seed in range(6):
        distorted_positions, distorted_cell = _distort(
            positions, cell, 41 + seed, strain=0.06, noise=0.05, shear=0.08
        )
        extreme_positions, extreme_cell = _distort(
            positions, cell, 141 + seed, strain=0.15, noise=0.15, shear=0.20
        )
        distorted = refine_l12(
            distorted_positions, distorted_cell, (True, True, True), types
        )
        extreme = refine_l12(
            extreme_positions, extreme_cell, (True, True, True), types
        )
        assert distorted.label == "l12"
        assert distorted.confirmed
        assert extreme.label == "unknown"
        assert not extreme.confirmed


def test_l12_antisites_reduce_order_without_false_confirmation():
    positions, cell, types = _fcc_prototype()
    rng = np.random.default_rng(47)
    a_sites = np.flatnonzero(types == 0)
    b_sites = np.flatnonzero(types == 1)

    one_pair = types.copy()
    one_a = rng.choice(a_sites, 1, replace=False)
    one_b = rng.choice(b_sites, 1, replace=False)
    one_pair[one_a] = 1
    one_pair[one_b] = 0

    six_pairs = types.copy()
    six_a = rng.choice(a_sites, 6, replace=False)
    six_b = rng.choice(b_sites, 6, replace=False)
    six_pairs[six_a] = 1
    six_pairs[six_b] = 0

    slightly_disordered = refine_l12(
        positions, cell, (True, True, True), one_pair
    )
    disordered = refine_l12(positions, cell, (True, True, True), six_pairs)

    assert slightly_disordered.label == "l12_partial"
    assert not slightly_disordered.confirmed
    assert disordered.label == "not_l12"
    assert not disordered.confirmed


def test_l12_rejects_random_ab3_on_fcc_bcc_hcp_and_a15():
    fcc_positions, fcc_cell, _ = _fcc_prototype()
    controls = [(fcc_positions, fcc_cell)]
    for atoms in (
        bulk("Fe", "bcc", a=2.88, cubic=True).repeat((3, 3, 3)),
        bulk("Ti", "hcp", a=2.95, c=4.68, orthorhombic=True).repeat((3, 3, 3)),
    ):
        controls.append((atoms.positions, atoms.cell.array))
    controls.append(_a15_positions())

    maximum_order_match = 0.0
    for control_index, (positions, cell) in enumerate(controls):
        for seed in range(12):
            types = _random_groups(len(positions), 0.25, 300 * control_index + seed)
            result = refine_l12(positions, cell, (True, True, True), types)
            maximum_order_match = max(
                maximum_order_match, result.chemistry_match_fraction
            )
            assert not result.confirmed
            assert result.label in {"not_l12", "unknown"}

    assert maximum_order_match < 0.40


@pytest.mark.parametrize("builder", (_d022_prototype, _d023_prototype))
def test_l12_does_not_confirm_tetragonal_a3b_competitors(builder):
    positions, cell, types = builder()

    result = refine_l12(positions, cell, (True, True, True), types)

    assert not result.confirmed
    assert result.label != "l12"


def test_l12_is_invariant_to_order_ids_translation_and_cell_basis():
    positions, cell, types = _fcc_prototype()
    reference = refine_l12(positions, cell, (True, True, True), types)
    rng = np.random.default_rng(89)
    permutation = rng.permutation(len(positions))
    relabeled = np.where(types[permutation] == 0, 41, 99)
    unimodular = np.asarray(((1, 0, 1), (0, 1, 0), (0, 0, 1)), dtype=float)

    transformed = refine_l12(
        positions[permutation] + np.asarray((-5.2, 3.1, 8.4)),
        unimodular @ cell,
        (True, True, True),
        relabeled,
    )

    assert transformed.label == reference.label == "l12"
    assert transformed.confirmed and reference.confirmed
    assert transformed.chemistry_match_fraction == pytest.approx(
        reference.chemistry_match_fraction, abs=1.0e-12
    )


def test_multicomponent_l12_requires_explicit_unambiguous_roles():
    positions, cell, types = _fcc_prototype()
    split_types = types.copy()
    b_sites = np.flatnonzero(split_types == 1)
    split_types[b_sites[::2]] = 2

    automatic = refine_l12(positions, cell, (True, True, True), split_types)
    explicit = refine_l12(
        positions,
        cell,
        (True, True, True),
        split_types,
        a_types=(0,),
        b_types=(1, 2),
    )

    assert automatic.label == "unknown"
    assert "ambiguous" in automatic.reason
    assert explicit.label == "l12"
    assert explicit.confirmed


def test_candidate_refinement_fails_closed_for_small_nonperiodic_cluster():
    rng = np.random.default_rng(97)
    positions = rng.uniform(0.0, 8.0, size=(10, 3))
    cell = np.eye(3) * 10.0

    l12 = refine_l12(
        positions,
        cell,
        (False, False, False),
        _random_groups(len(positions), 0.2, 1),
        a_types=(0,),
        b_types=(1,),
    )
    laves = refine_laves(
        positions,
        cell,
        (False, False, False),
        _random_groups(len(positions), 0.3, 2),
        a_types=(0,),
        b_types=(1,),
    )

    assert l12.label == "unknown"
    assert not l12.confirmed
    assert laves.label == "unknown"
    assert not laves.confirmed


def test_native_and_python_refinement_paths_make_the_same_decisions(monkeypatch):
    native = phase_sketch_module._native_phase
    if native is None:
        pytest.skip("optional native phase extension is not built")
    l12_positions, l12_cell, l12_types = _fcc_prototype()
    c14_positions, c14_cell, c14_types = _c14_prototype()
    c15_positions, c15_cell, c15_types = _c15_prototype()
    c36_positions, c36_cell, c36_types = _c36_prototype()
    l12_positions, l12_cell = _distort(
        l12_positions, l12_cell, 211, strain=0.04, noise=0.03, shear=0.05
    )
    c14_positions, c14_cell = _distort(
        c14_positions, c14_cell, 212, strain=0.04, noise=0.03, shear=0.05
    )
    c15_positions, c15_cell = _distort(
        c15_positions, c15_cell, 213, strain=0.04, noise=0.03, shear=0.05
    )
    c36_positions, c36_cell = _distort(
        c36_positions, c36_cell, 214, strain=0.04, noise=0.03, shear=0.05
    )
    cases = (
        (refine_l12, l12_positions, l12_cell, l12_types),
        (refine_laves, c14_positions, c14_cell, c14_types),
        (refine_laves, c15_positions, c15_cell, c15_types),
        (refine_laves, c36_positions, c36_cell, c36_types),
    )

    monkeypatch.setattr(phase_sketch_module, "_native_phase", None)
    phase_refinement_module._reference_templates.cache_clear()
    python_results = tuple(
        refine(positions, cell, (True, True, True), types)
        for refine, positions, cell, types in cases
    )

    monkeypatch.setattr(phase_sketch_module, "_native_phase", native)
    phase_refinement_module._reference_templates.cache_clear()
    native_results = tuple(
        refine(positions, cell, (True, True, True), types)
        for refine, positions, cell, types in cases
    )

    for python_result, native_result in zip(python_results, native_results):
        assert native_result.label == python_result.label
        assert native_result.confirmed is python_result.confirmed
        assert native_result.geometry_match_fraction == pytest.approx(
            python_result.geometry_match_fraction, abs=1.0e-12
        )
        assert native_result.chemistry_match_fraction == pytest.approx(
            python_result.chemistry_match_fraction, abs=1.0e-12
        )
        assert native_result.joint_match_fraction == pytest.approx(
            python_result.joint_match_fraction, abs=1.0e-12
        )
        assert native_result.defect_fraction == pytest.approx(
            python_result.defect_fraction, abs=1.0e-12
        )
        if python_result.b2_fraction is None:
            assert native_result.b2_fraction is None
        else:
            assert native_result.b2_fraction == pytest.approx(
                python_result.b2_fraction, abs=1.0e-12
            )
