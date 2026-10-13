from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

import NepTrainKit.core.io.sampling_coverage as coverage
from NepTrainKit.core.io.sampling_features import SamplingFeatureBlocks, build_sampling_feature_blocks
from NepTrainKit.core.io.sampling_plan import PhysicsSamplingPlan, PhysicsSamplingStratum
from NepTrainKit.core.io.sampling_recommendation import (
    recommend_physics_sampling_from_blocks,
    select_physics_budget,
)


def plan(n, labels=None):
    labels = ["fm"] * n if labels is None else labels
    groups = tuple((PhysicsSamplingStratum(("Fe",), "bcc", name),
                    tuple(i for i, label in enumerate(labels) if label == name))
                   for name in sorted(set(labels)))
    return PhysicsSamplingPlan(groups, True, (), (), (), tuple(range(n)))


def test_narrow_iqr_magnetic_coordinate_cannot_hide_descriptor_manifold():
    x = np.linspace(-1, 1, 401)
    # Regression for FeRh: a nearly binary magnetic coordinate, with tiny IQR,
    # formerly explained >99% of total variance after robust standardization.
    magnetic = np.where(x < .8, -.5 + x * 1e-7, 0)[:, None]
    blocks = SamplingFeatureBlocks(("descriptor_mean", "magnetism"), (x[:, None], magnetic))
    result = recommend_physics_sampling_from_blocks(blocks, plan(len(x)))
    selected = x[list(result.selected_indices)]
    assert len(selected) > 5
    assert np.max(np.min(abs(x[:, None] - selected), axis=1)) < .16
    assert not result.is_lower_bound
    assert not result.evidence.uncovered_indices


def test_structure_views_need_one_joint_representative():
    candidate = SamplingFeatureBlocks(("descriptor_mean", "magnetism"),
                                      (np.array([[0.]]), np.array([[0.]])))
    existing = SamplingFeatureBlocks(candidate.names, (np.array([[0.], [10.]]),
                                                        np.array([[10.], [0.]])))
    result = recommend_physics_sampling_from_blocks(candidate, plan(1), existing=existing,
                                                    existing_plan=plan(2))
    # Neither existing structure represents the combination, although each
    # coordinate independently has an exact match in the training set.
    assert result.selected_indices == (0,)


def test_budget_and_automatic_are_the_same_selection_prefix():
    rng = np.random.default_rng(5)
    blocks = SamplingFeatureBlocks(("descriptor_mean", "magnetism"),
                                  (rng.normal(size=(160, 3)), rng.normal(size=(160, 2))))
    automatic = recommend_physics_sampling_from_blocks(blocks, plan(160))
    fixed = select_physics_budget(blocks, plan(160), n_samples=automatic.recommended_count)
    assert fixed.selected_indices == automatic.selected_indices
    assert automatic.evidence.stop_reason == "resolution"
    assert max(automatic.evidence.max_distances) <= automatic.evidence.resolution


def test_duplicate_frames_do_not_change_the_metric_or_representative_values():
    points = np.array([[0.], [.1], [.3], [1.], [2.], [5.], [8.]])
    blocks = SamplingFeatureBlocks(("descriptor_mean",), (points,))
    first = select_physics_budget(blocks, plan(len(points)), n_samples=4)
    repeated = np.vstack((points, np.repeat(points[2:3], 1000, axis=0)))
    second = select_physics_budget(SamplingFeatureBlocks(blocks.names, (repeated,)),
                                   plan(len(repeated)), n_samples=4)
    np.testing.assert_array_equal(np.sort(points[list(first.selected_indices), 0]),
                                  np.sort(repeated[list(second.selected_indices), 0]))


def test_input_order_and_atom_order_do_not_change_selected_structure_content():
    rng = np.random.default_rng(101)
    structures = [Atoms("Fe3", cell=np.eye(3) * 3, pbc=True) for _ in range(20)]
    atoms = rng.normal(size=(60, 4))
    means = atoms.reshape(20, 3, 4).mean(1)
    blocks = build_sampling_feature_blocks(structures, means, per_atom_descriptors=atoms,
                                          spin_model=False)
    first = select_physics_budget(blocks, plan(20), n_samples=8)
    order = rng.permutation(20)
    permuted_atoms = atoms.reshape(20, 3, 4)[order, ::-1].reshape(60, 4)
    second_blocks = build_sampling_feature_blocks(structures, means[order],
                                                  per_atom_descriptors=permuted_atoms,
                                                  spin_model=False)
    second = select_physics_budget(second_blocks, plan(20), n_samples=8)
    assert set(first.selected_indices) == set(order[list(second.selected_indices)])


def test_same_species_local_environments_are_not_lost_in_structure_average():
    structures = [Atoms("Fe2", cell=np.eye(3)*3, pbc=True) for _ in range(3)]
    atoms = np.array([[-1.], [1.], [-1.], [1.], [-4.], [4.]])
    blocks = build_sampling_feature_blocks(structures, np.zeros((3, 1)),
                                          per_atom_descriptors=atoms, spin_model=False)
    result = recommend_physics_sampling_from_blocks(blocks, plan(3))
    assert 2 in result.selected_indices
    assert len(result.selected_indices) == 2
    assert result.evidence.atomic_available


def test_atomic_coverage_compares_central_species_separately():
    candidate = SamplingFeatureBlocks(("descriptor_mean",), (np.zeros((1, 1)),),
                                      np.array([[0.], [10.]]), (2,), ("Fe", "Rh"))
    warm = SamplingFeatureBlocks(candidate.names, candidate.values,
                                 np.array([[10.], [0.]]), (2,), ("Fe", "Rh"))
    result = recommend_physics_sampling_from_blocks(candidate, plan(1), existing=warm,
                                                    existing_plan=plan(1))
    assert result.selected_indices == (0,)


def test_covered_training_environments_need_no_new_structure():
    structures = [bulk("Fe", "bcc", a=2.9, cubic=True)] * 4
    atoms = np.arange(8, dtype=float)[:, None]
    blocks = build_sampling_feature_blocks(structures, atoms.reshape(4, 2, 1).mean(1),
                                          per_atom_descriptors=atoms, spin_model=False)
    result = recommend_physics_sampling_from_blocks(blocks, plan(4), existing=blocks,
                                                    existing_plan=plan(4))
    assert result.recommended_count == 0
    assert result.evidence.uncovered_indices == ()


def test_cell_basis_and_supercell_do_not_create_artificial_lattice_novelty():
    original = bulk("Fe", "bcc", a=2.9, cubic=True)
    rebased = original.copy()
    transform = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    rebased.set_cell(transform @ original.cell.array, scale_atoms=False)
    enlarged = original.repeat((2, 1, 1))
    structures = [original, rebased, enlarged]
    blocks = build_sampling_feature_blocks(structures, np.ones((3, 2)),
                                          per_atom_descriptors=np.ones((8, 2)), spin_model=False)
    result = recommend_physics_sampling_from_blocks(blocks, plan(3))
    assert result.recommended_count == 1


def test_budget_limit_and_missing_atomic_evidence_are_reported():
    points = np.random.default_rng(1).normal(size=(100, 4))
    blocks = SamplingFeatureBlocks(("descriptor_mean",), (points,))
    result = select_physics_budget(blocks, plan(100), n_samples=2)
    assert result.evidence.stop_reason == "budget"
    assert result.evidence.uncovered_indices
    assert not result.evidence.atomic_available
    capped = recommend_physics_sampling_from_blocks(blocks, plan(100), sample_cap_per_stratum=2)
    assert capped.is_lower_bound
    assert capped.evidence.stop_reason == "resource_limit"


def test_native_and_reference_local_coverage_agree(monkeypatch):
    if coverage._native_sampling is None or not hasattr(coverage._native_sampling, "CoverageState"):
        pytest.skip("native coverage helper is not built")
    structures = [Atoms("Fe2Rh", cell=np.eye(3)*3, pbc=True) for _ in range(28)]
    atoms = np.random.default_rng(45).normal(size=(84, 6))
    blocks = build_sampling_feature_blocks(structures, atoms.reshape(28, 3, 6).mean(1),
                                          per_atom_descriptors=atoms, spin_model=False)
    native = coverage.select_coverage(blocks, plan(28), budget=15)
    monkeypatch.setattr(coverage, "_native_sampling", None)
    reference = coverage.select_coverage(blocks, plan(28), budget=15)
    assert native.indices == reference.indices
    np.testing.assert_allclose(native.distances, reference.distances, atol=1e-12)


def test_atomic_distance_certificate_is_an_upper_bound():
    rng = np.random.default_rng(12)
    points = rng.normal(size=(100, 3))
    atoms = rng.normal(size=(200, 5))
    args = (points, np.array([3]), np.zeros(100, dtype=int), atoms,
            np.arange(0, 202, 2), np.zeros(200, dtype=int), 100)
    state = coverage._NumpyCoverageState(*args)
    chosen = [2, 4, 8, 10, 11, 17, 27]
    for i in chosen:
        state.add(i)
    centers = atoms[np.concatenate([np.arange(2*i, 2*i+2) for i in chosen])]
    exact = np.sqrt(np.min(np.sum((atoms[:, None] - centers)**2, axis=-1), axis=1))
    bound = state.distances()[:, -1]
    assert np.all(bound + 1e-12 >= exact.reshape(100, 2).max(1))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_features_fail_before_sampling(bad):
    with pytest.raises(ValueError, match="finite"):
        select_physics_budget(SamplingFeatureBlocks(("descriptor_mean",),
                                                    (np.array([[0.], [bad]]),)),
                              plan(2), n_samples=1)


def test_card_and_main_backend_return_identical_physics_subset(monkeypatch, tmp_path):
    from NepTrainKit.core.cards.filter import FPSFilterOperation, FPSFilterParams
    from NepTrainKit.core.io.sampling_plan import build_physics_sampling_plan

    structures = [bulk("Fe", "bcc", a=a, cubic=True) for a in np.linspace(2.6, 3.1, 12)]
    atoms = np.random.default_rng(32).normal(size=(24, 4))
    model = tmp_path / "nep.txt"
    model.write_text("descriptor stub")

    class Calculator:
        is_spin_model = False

        def __init__(self, **kwargs):
            pass

        def descriptors(self, structures, *, mean):
            assert not mean
            return atoms

    monkeypatch.setattr("NepTrainKit.core.cards.filter.NepCalculator", Calculator)
    operation = FPSFilterOperation()
    output = operation.run_dataset(structures, FPSFilterParams(str(model), n_samples=5,
                                                               strategy="physics"))
    blocks = build_sampling_feature_blocks(structures, atoms.reshape(12, 2, 4).mean(1),
                                          per_atom_descriptors=atoms, spin_model=False)
    main = select_physics_budget(blocks, build_physics_sampling_plan(structures, spin_model=False),
                                 n_samples=5)
    assert [id(s) for s in output] == [id(structures[i]) for i in main.selected_indices]
    assert operation.last_coverage_evidence == main.evidence


def test_nonperiodic_vacuum_is_not_a_density_coverage_axis():
    structures = [Atoms("Fe", cell=np.eye(3)*a, pbc=False) for a in (5, 50)]
    blocks = build_sampling_feature_blocks(structures, np.zeros((2, 1)), spin_model=False)
    result = recommend_physics_sampling_from_blocks(blocks, plan(2))
    assert result.recommended_count == 1


def test_report_distinguishes_incomplete_budget_from_completed_coverage(monkeypatch):
    from NepTrainKit.ui.messages import format_sampling_coverage

    monkeypatch.setattr("NepTrainKit.ui.messages._tr", lambda text: text)
    blocks = SamplingFeatureBlocks(("descriptor_mean",), (np.arange(100)[:, None],))
    result = select_physics_budget(blocks, plan(100), n_samples=2)
    text = format_sampling_coverage(result.evidence)
    assert "Sample budget reached" in text
    assert f"Uncovered structures: {len(result.evidence.uncovered_indices)}" in text
    assert "only structure coverage" in text


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("cutoff, expected", [(0.0, 5), (0.0001, 5), (0.01, 2)])
def test_manual_sampling_can_resolve_atoms_below_the_automatic_floor(monkeypatch, native, cutoff, expected):
    if not native:
        monkeypatch.setattr(coverage, "_native_sampling", None)
    elif coverage._native_sampling is None or not hasattr(coverage._native_sampling.CoverageState, "refine_atoms"):
        pytest.skip("native coverage refinement is not built")
    values = np.array([1., 1.01, 1.02, 1.03, 100., 1.01])
    atoms = np.column_stack((-values, values)).reshape(-1, 1)
    blocks = SamplingFeatureBlocks(("descriptor_mean",), (np.zeros((6, 1)),),
                                   atoms, (2,) * 6, ("Cu",) * 12)
    groups = PhysicsSamplingPlan(((PhysicsSamplingStratum(("Cu",), "unresolved"), tuple(range(6))),),
                                  False, (), (), ())
    result = coverage.select_coverage(blocks, groups, budget=6, min_distance=cutoff)
    assert len(result.indices) == expected
    if cutoff == 0:
        assert result.evidence.stop_reason == "exhausted"
        np.testing.assert_allclose(result.distances, 0., atol=1e-12)


def test_fine_manual_warm_start_revisits_retired_atoms():
    def blocks(values):
        values = np.array(values)
        return SamplingFeatureBlocks(("descriptor_mean",), (np.zeros((len(values), 1)),),
                                     np.column_stack((-values, values)).reshape(-1, 1),
                                     (2,) * len(values), ("Cu",) * (2 * len(values)))

    result = coverage.select_coverage(blocks([1., 1.01, 1.02]), plan(3),
                                      existing=blocks([1., 100.]), existing_plan=plan(2), budget=3)
    assert set(result.indices) == {1, 2}
    assert result.evidence.stop_reason == "exhausted"


def test_nonmagnetic_model_never_requests_magnetic_features(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary NEP must not analyze magnetism")

    monkeypatch.setattr("NepTrainKit.core.io.sampling_features.analyze_structure_magnetism", forbidden)
    monkeypatch.setattr("NepTrainKit.core.io.sampling_plan.analyze_structure_magnetism", forbidden)
    from NepTrainKit.core.io.sampling_plan import build_physics_sampling_plan

    structures = [bulk("Cu", "fcc", a=a) for a in (3.5, 3.6, 3.7)]
    blocks = build_sampling_feature_blocks(structures, np.arange(3.)[:, None],
                                          per_atom_descriptors=np.arange(3.)[:, None], spin_model=False)
    result = coverage.select_coverage(blocks, build_physics_sampling_plan(structures, spin_model=False))
    assert "magnetism" not in result.evidence.view_names
    assert result.evidence.atomic_available
    assert result.evidence.uncovered_indices == ()


def test_real_nonmagnetic_nep_card_warm_start_needs_no_spin_or_duplicate_output():
    from NepTrainKit.core.cards.filter import FPSFilterOperation, FPSFilterParams
    from NepTrainKit.core.structure import Structure

    fixture = Path(__file__).resolve().parents[2] / "data" / "nep"
    structures = Structure.read_multiple(str(fixture / "train.xyz"))[:5]
    operation = FPSFilterOperation()
    result = operation.run_dataset(structures, FPSFilterParams(
        str(fixture / "nep.txt"), n_samples=5, strategy="physics", backend="cpu",
        existing_dataset_path=str(fixture / "train.xyz"),
    ))
    assert result == []
    assert not operation.last_physics_plan_report.spin_model
    assert operation.last_coverage_evidence.atomic_available
    assert "magnetism" not in operation.last_coverage_evidence.view_names
    assert operation.last_coverage_evidence.uncovered_indices == ()


@pytest.mark.parametrize("radius", [0.35, 0.25, 0.12, 0.005])
def test_custom_radius_stops_at_requested_prefix_and_certifies_actual_atoms(radius):
    values = np.linspace(1., 3., 180)
    blocks = SamplingFeatureBlocks(("descriptor_mean",), (np.zeros((180, 1)),),
                                   np.column_stack((-values, values)).reshape(-1, 1),
                                   (2,) * 180, ("Cu",) * 360)
    result = coverage.select_coverage(blocks, plan(180), resolution=radius)
    assert result.evidence.resolution == radius
    assert result.evidence.stop_reason == "resolution"
    assert max(result.evidence.max_distances) <= radius + 1e-12
    assert result.trajectory == result.indices  # No unused tighter envelope.
    fixed = coverage.select_coverage(blocks, plan(180), budget=len(result.indices))
    assert fixed.indices == result.indices
    chosen = np.column_stack((-values[list(result.indices)], values[list(result.indices)])).ravel()
    raw = blocks.atomic_values.ravel()
    scale = np.std(np.unique(raw))
    exact = abs(raw[:, None] - chosen).min(axis=1) / scale
    assert exact.max() <= radius + 1e-12


@pytest.mark.parametrize("radius", [0., -1., np.nan, np.inf])
def test_invalid_automatic_radius_is_rejected(radius):
    with pytest.raises(ValueError, match="coverage radius"):
        coverage.select_coverage(SamplingFeatureBlocks(("descriptor_mean",), (np.zeros((2, 1)),)),
                                 plan(2), resolution=radius)


def test_custom_recommendation_does_not_claim_an_uncomputed_envelope():
    blocks = SamplingFeatureBlocks(("descriptor_mean",), (np.linspace(0, 1, 200)[:, None],))
    result = recommend_physics_sampling_from_blocks(blocks, plan(200), resolution=.4)
    assert result.evidence.resolution == .4
    assert not result.is_lower_bound
    assert result.conservative_count is None


@pytest.mark.parametrize("seed", range(20))
def test_native_tree_matches_independent_reference_through_warm_start_and_refinement(seed):
    native = coverage._native_sampling
    if native is None or getattr(native, "coverage_api_version", 0) < 2:
        pytest.skip("native spatial coverage index is not built")
    rng = np.random.default_rng(seed)
    count, warm, per_frame, dims = 100, 4, (130 if seed == 0 else 8), 12
    points = rng.normal(size=(count + warm, 5))
    atoms = rng.normal(size=((count + warm) * per_frame, dims))
    atoms += np.repeat(rng.integers(0, 5, count + warm), per_frame)[:, None] * 4
    if seed % 2 == 0:
        atoms[:, :8] = 0.  # Index axes cannot see novelty in the remaining axes.
    species = np.tile(np.arange(per_frame) % 2, count + warm)
    args = (points, np.array([3, 5]), np.zeros(count + warm, dtype=int), atoms,
            np.arange(0, (count + warm + 1) * per_frame, per_frame), species, count)
    actual = native.CoverageState(*args)
    expected = coverage._NumpyCoverageState(*args)
    centers = [*range(count, count + warm), *rng.choice(count, 30, replace=False)]
    for center in centers:
        actual.add(int(center))
        expected.add(int(center))
    np.testing.assert_allclose(actual.scores(), expected.scores(), atol=1e-12)
    np.testing.assert_allclose(actual.distances(), expected.distances(), atol=1e-12)
    actual.refine_atoms(.02)
    expected.refine_atoms(.02)
    for center in centers:
        actual.add(int(center))
        expected.add(int(center))
    np.testing.assert_allclose(actual.distances(), expected.distances(), atol=1e-12)
