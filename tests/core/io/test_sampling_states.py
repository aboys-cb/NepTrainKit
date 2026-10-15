import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from NepTrainKit.core.io import sampling_coverage
from NepTrainKit.core.io.sampling_coverage import _prepare, _select_coverage
from NepTrainKit.core.io.sampling_features import SamplingFeatureBlocks
from NepTrainKit.core.io.sampling_plan import PhysicsSamplingPlan, PhysicsSamplingStratum
from NepTrainKit.core.io.sampling_states import (
    _allocate,
    _cloud_curve,
    _factors,
    _grid_representatives,
    _input_requirements,
    _line_representatives,
    _RangeRequirements,
    _reduce_cover,
    _select_state_coverage,
    _solve_cover,
    _state_seeds,
    _WitnessRequirements,
    select_state_coverage,
)


def plan(n):
    return PhysicsSamplingPlan(((PhysicsSamplingStratum(("C",), "unresolved"), tuple(range(n))),),
                               False, (), (), ())


def seeds(blocks, p, budget, supports=None):
    metric = _prepare(blocks, p, None, None)[1]
    return _state_seeds(blocks, p, np.arange(blocks.row_count), budget, metric,
                        [] if supports is None else supports, [])


@pytest.mark.parametrize("seed", range(5))
def test_path_cover_matches_independent_finite_minimax(seed):
    x = np.sort(np.random.default_rng(seed).uniform(size=10))
    for budget in (1, 2, 3, 5):
        ids = _line_representatives(x, budget)
        actual = abs(x[:, None]-x[ids]).min(1).max()
        optimum = min(abs(x[:, None]-x[list(s)]).min(1).max()
                      for s in itertools.combinations(range(len(x)), budget))
        assert actual == pytest.approx(optimum, abs=1e-9)
        assert len(ids) == len(set(ids)) == budget


def test_range_is_retained_when_it_does_not_conflict_with_gap_budget():
    x = np.arange(61.)
    ids = _line_representatives(x, 11)
    assert {0, 60}.issubset(ids)
    assert abs(x[:, None]-x[ids]).min(1).max() == 3
    # Twenty-one path states need all seven interior centers to reach gap 1.
    ids = _line_representatives(np.arange(21.), 7)
    np.testing.assert_array_equal(ids, np.arange(1, 21, 3))


def test_accidentally_unique_scalar_does_not_turn_cloud_into_path():
    x = np.random.default_rng(87).normal(size=(100, 8))
    assert _factors(x) is None
    # An incomplete two-dimensional product is not certified either.
    x = np.array(list(itertools.product(range(5), range(7))), float)
    assert _factors(x[:-1]) is None


def test_grid_factors_discard_dependent_features_and_cover_joint_cells():
    x = np.array(list(itertools.product(range(5), range(7))), float)
    # A dependent folded coordinate is insufficient to distinguish states.
    features = np.column_stack((x[:, 0], abs(x[:, 1]-3), x[:, 1]))
    factors = _factors(features)
    assert factors == [0, 2]
    ids = _grid_representatives(features, factors, 12)
    cells = np.minimum((x[ids]/[4, 6]*[3, 4]).astype(int), [2, 3])
    assert len(np.unique(cells, axis=0)) == 12
    np.testing.assert_array_equal(x[ids].min(0), [0, 0])
    np.testing.assert_array_equal(x[ids].max(0), [4, 6])


def test_three_factor_grid_stays_inside_fixed_budget():
    x = np.array(list(itertools.product(range(4), range(5), range(6))), float)
    factors = _factors(x)
    ids = _grid_representatives(x, factors, 24)
    assert len(ids) == len(set(ids)) == 24
    assert np.all((ids >= 0) & (ids < len(x)))


def test_new_selection_is_invariant_to_duplicates_and_row_order():
    x = np.array(list(itertools.product(range(5), range(7))), float)

    def select(values):
        blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"), (values, values))
        return {tuple(row) for row in values[list(select_state_coverage(blocks, plan(len(values)), budget=12).indices)]}

    expected = select(x)
    repeated = np.vstack((x, np.repeat(x[7:8], 100, axis=0)))
    np.random.default_rng(982).shuffle(repeated)
    assert select(repeated) == expected


def test_unsupported_cloud_does_not_receive_invented_scan_seeds():
    x = np.random.default_rng(33).normal(size=(75, 5))
    blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"), (x, x))
    chosen, count, mask = seeds(blocks, plan(len(x)), 12)
    assert not chosen and count == 0 and not mask.any()
    actual = select_state_coverage(blocks, plan(len(x)), budget=12)
    assert len(actual.indices) == len(set(actual.indices)) == 12
    assert np.isfinite(actual.distances).all()


def test_seeded_backend_keeps_total_budget_and_rejects_bad_seeds():
    x = np.arange(30.)[:, None]
    blocks = SamplingFeatureBlocks(("descriptor_mean",), (x,))
    result = _select_coverage(blocks, plan(len(x)), budget=5, _initial_indices=[0, 29])
    assert len(result.indices) == len(set(result.indices)) == 5
    assert {0, 29}.issubset(result.indices)
    for seeds in ([0, 0], [-1], [30], list(range(6))):
        with pytest.raises(ValueError, match="initial coverage"):
            _select_coverage(blocks, plan(len(x)), budget=5, _initial_indices=seeds)


def test_range_requirements_accept_alternate_and_outside_representatives():
    x = np.array([[-2.], [0.], [0.], [1.], [2.], [4.]])
    blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"), (x, x))
    groups = dict(plan(len(x)).groups)
    structured = np.array([True, False, False, False, False, True])
    ranges = _RangeRequirements(blocks, groups, structured)
    assert ranges.pending()[1] and ranges.pending()[2]
    ranges.add(2)  # Either of the identical minima can satisfy this side.
    assert not ranges.pending()[1]
    ranges.add(5)  # A scan representative beyond the other end also suffices.
    assert not ranges.pending().any()
    ranges.add(5)
    assert np.all(ranges.count == 0)


def test_small_budget_reserves_unstructured_element_pool():
    x = np.arange(24.)[:, None]
    geometry = np.r_[np.arange(21.), [2., 2., 2.]][:, None]
    blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"), (x, geometry))
    groups = ((PhysicsSamplingStratum(("C",), "unresolved"), tuple(range(21))),
              (PhysicsSamplingStratum(("O",), "unresolved"), (21, 22, 23)))
    p = PhysicsSamplingPlan(groups, False, (), (), ())
    selected = select_state_coverage(blocks, p, budget=2).indices
    assert len(selected) == 2
    assert sum(i < 21 for i in selected) == 1


@pytest.mark.parametrize("budget", [0, -1, .5, True])
def test_invalid_budget_rejected(budget):
    x = np.arange(12.)[:, None]
    blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"), (x, x))
    with pytest.raises(ValueError, match="positive integer"):
        select_state_coverage(blocks, plan(len(x)), budget=budget)


@pytest.mark.parametrize("seed", range(10))
def test_nonideal_recognition_declines_unstructured_clouds(seed):
    from NepTrainKit.core.io.sampling_states import _nonideal_regions
    points = np.random.default_rng(seed).uniform(size=(80, 6))
    assert _nonideal_regions(points) == []


def test_sparse_path_forest_separates_local_branches_without_phase_labels():
    from NepTrainKit.core.io.sampling_states import _continuous_paths
    t = np.linspace(0., 1., 41)
    x = np.column_stack((t, .1*np.sin(t), np.zeros(len(t))))
    x = np.vstack((x, x + [0., 0., .3]))
    paths = _continuous_paths(x)
    assert len(paths) == 2
    assert {frozenset(rows) for rows, _ in paths} == {frozenset(range(41)), frozenset(range(41, 82))}


def test_curve_is_recognized_but_branched_tree_is_not_a_single_path():
    from NepTrainKit.core.io.sampling_states import _continuous_paths
    t = np.linspace(0., 1., 61)
    assert len(_continuous_paths(np.column_stack((t, np.sin(np.pi*t))))) == 1
    branch = np.vstack((np.column_stack((t, np.zeros(len(t)))),
                        np.column_stack((np.full(30, .5), t[1:31]))))
    assert not _continuous_paths(branch)


def test_noisy_joint_levels_and_missing_cells_preserve_observed_states():
    from NepTrainKit.core.io.sampling_states import _incomplete_factors
    x = np.array(list(itertools.product(range(7), range(7))), dtype=float)
    keep = ~((x[:, 0] % 2 == 1) & (x[:, 1] % 2 == 1))
    x = x[keep]
    noisy = x + np.random.default_rng(183).normal(scale=.015, size=x.shape)
    c = _incomplete_factors(noisy)
    assert c is not None
    assert len(np.unique(c, axis=0)) == len(x)
    # Recognition must recover the actual joint assignment, up to axis order
    # and sign; it cannot create fictitious representatives for missing cells.
    for column in c.T:
        assert any(np.allclose(column, v/6) or np.allclose(column, 1-v/6) for v in x.T)


def test_endpoint_constrained_path_is_minimax_and_uses_actual_spacing():
    x = np.array([0., .01, .03, .1, .2, .43, .7, .82, .99, 1.])
    ids = _line_representatives(x, 4, require_endpoints=True)
    assert {0, 9} <= set(ids)
    gap = abs(x[:, None]-x[ids]).min(1).max()
    optimum = min(abs(x[:, None]-x[[0, *middle, 9]]).min(1).max()
                  for middle in itertools.combinations(range(1, 9), 2))
    assert gap == pytest.approx(optimum)


def test_independent_thermal_repeats_are_states_not_extra_budget_seats():
    from NepTrainKit.core.io.sampling_states import _incomplete_factors
    grid = np.array(list(itertools.product(range(7), range(7))), dtype=float)
    repeated = np.repeat(grid, 4, axis=0)
    noisy = repeated + np.random.default_rng(8102).normal(scale=.01, size=repeated.shape)
    coordinates = _incomplete_factors(noisy)
    assert coordinates is not None
    assert len(np.unique(coordinates, axis=0)) == 49
    blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"), (noisy, noisy))
    chosen, _, _ = seeds(blocks, plan(len(noisy)), 60)
    assert len(chosen) == 49
    result = select_state_coverage(blocks, plan(len(noisy)), budget=60)
    assert len(result.indices) == len(set(result.indices)) == 60


def test_joint_cells_do_not_hide_an_unexplained_local_motif():
    from NepTrainKit.core.io.sampling_states import _joint_coordinates
    grid = np.array(list(itertools.product(range(5), range(5))), dtype=float)
    repeated = np.repeat(grid, 2, axis=0)
    inputs = np.column_stack((repeated, np.tile([0., 3.], len(grid))))
    assert _joint_coordinates(inputs, repeated.T.astype(int)) is None


def test_state_supports_keep_frame_mapping_with_repeated_observations():
    x = np.repeat(np.array(list(itertools.product(range(7), range(7))), float), 3, axis=0)
    x += np.random.default_rng(47).normal(scale=.005, size=x.shape)
    blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"), (x, x))
    supports = []
    seeds(blocks, plan(len(x)), 16, supports)
    assert supports
    rows, coordinates, _ = supports[0]
    assert len(rows) == len(coordinates) == len(x)
    np.testing.assert_array_equal(np.sort(rows), np.arange(len(x)))
    assert len(select_state_coverage(blocks, plan(len(x)), budget=16).indices) == 16


def test_cloud_estimate_is_duplicate_and_order_independent():
    points = np.random.default_rng(789).normal(size=(90, 4))
    repeated = np.vstack((points, np.repeat(points[:1], 200, axis=0)))[::-1]
    first = _cloud_curve(points, len(points), 80)
    second = _cloud_curve(repeated, len(points), 80)
    np.testing.assert_allclose([first(k) for k in range(1, 91)],
                               [second(k) for k in range(1, 91)])
    assert first(90) == 0.


def test_path_allocation_responds_to_gap_not_observation_density():
    dense = np.linspace(0, 1, 121)[:, None]
    sparse = np.linspace(0, 1, 21)[:, None]
    quotas, _, chosen = _allocate([(dense, False), (sparse, False)], [], np.array([121, 21]), 22)
    assert quotas.sum() == 22
    assert min(quotas) >= 9
    for x, ids in zip((dense, sparse), chosen):
        assert cKDTree(x[ids]).query(x)[0].max() <= .05+1e-10
        assert {0, len(x)-1}.issubset(ids)


def test_joint_grid_curve_does_not_collapse_at_prime_budget():
    grid = np.array(list(np.ndindex((7, 7))), float)/6
    gap = []
    for budget in range(1, 20):
        quotas, _, chosen = _allocate([(grid, False)], [], np.array([len(grid)]), budget)
        assert quotas.sum() == budget
        assert len(set(chosen[0])) == budget
        gap.append(cKDTree(grid[chosen[0]]).query(grid)[0].max())
    assert np.all(np.diff(gap) <= 1e-10)


@pytest.mark.parametrize("shape, budget, cells", [((7, 7), 16, (4, 4)), ((5, 7), 12, (3, 4))])
def test_equal_radius_prefers_complete_joint_grid_over_roundoff_gain(shape, budget, cells):
    grid = np.array(list(np.ndindex(shape)), float)/(np.array(shape)-1)
    _, _, chosen = _allocate([(grid, False)], [], np.array([len(grid)]), budget)
    occupied = np.minimum(np.floor(grid[chosen[0]]*cells+1e-10).astype(int), np.array(cells)-1)
    assert len(np.unique(occupied, axis=0)) == budget


def test_joint_requirements_do_not_depend_on_axis_orientation():
    points = np.array([(i/4, j/6) for i in range(5) for j in range(7)])
    chosen = [i*7+j for i in range(0, 5, 2) for j in range(0, 7, 2)]
    features = SamplingFeatureBlocks(("input_geometry:C",), (points,))
    matrices = []
    for oriented in (points, 1-points):
        support = [(np.arange(len(points)), oriented, np.array([3, 4]))]
        matrix = _input_requirements(features, {("C",): list(range(len(points)))}, support, chosen)
        assert np.all(np.asarray(matrix[chosen].sum(0)) > 0)
        # Equal maximum distance can hide loss of a joint cell at a boundary.
        trial = [i for i in chosen if i != 2]+[10]
        assert np.any(np.asarray(matrix[trial].sum(0)) == 0)
        matrices.append(matrix)
    np.testing.assert_array_equal(matrices[0].toarray(), matrices[1].toarray())


@pytest.mark.parametrize("seed", range(20))
def test_reductions_and_component_solve_match_independent_exhaustive_cover(seed):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(size=(7, 10)) < .3
    matrix[np.arange(7), np.arange(7)] = True
    optimum = next(k for k in range(1, 8) if any(
        matrix[:, list(ids)].any(1).all() for ids in itertools.combinations(range(10), k)))
    selected, stats = _solve_cover(csr_matrix(matrix), optimum)
    assert selected is not None and len(selected) == optimum
    assert matrix[:, selected].any(1).all()
    assert stats["status"] == "feasible"
    rejected, _ = _solve_cover(csr_matrix(matrix), optimum-1)
    assert rejected is None


def test_forced_and_dominated_candidates_are_eliminated_without_pinning_a_frame():
    matrix = csr_matrix([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool)
    reduced, _, forced, _ = _reduce_cover(matrix)
    assert reduced.shape[0] == 0
    assert set(forced) == {0, 2}
    chosen, stats = _solve_cover(csr_matrix([[1, 1], [1, 1]], dtype=bool), 1)
    assert len(chosen) == 1 and stats["status"] == "feasible"


def test_impossible_empty_requirement_is_not_silently_dropped():
    with pytest.raises(ValueError, match="no candidate"):
        _solve_cover(csr_matrix([[1, 0], [0, 0]], dtype=bool), 2)


def test_redundant_broad_requirement_does_not_join_independent_components():
    matrix = csr_matrix([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=bool)
    chosen, _ = _solve_cover(matrix, 2)
    assert len(chosen) == 2 and np.all(matrix[:, chosen].sum(1) >= 1)


@pytest.mark.parametrize("incumbent,accepted", [(None, False), ([1., 1., 0.], True), ([1., 0., 0.], False)])
def test_time_limited_incumbent_must_cover_actual_constraints(monkeypatch, incumbent, accepted):
    from types import SimpleNamespace

    from NepTrainKit.core.io import sampling_states
    # Odd cycle has no dominated columns and reaches the integer solver.
    matrix = csr_matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool)
    monkeypatch.setattr(sampling_states, "milp", lambda *a, **k: SimpleNamespace(
        status=1, x=None if incumbent is None else np.asarray(incumbent)))
    chosen, _ = _solve_cover(matrix, 3 if incumbent is None else 2)
    assert (chosen is not None) == accepted
    if accepted:
        assert np.all(matrix[:, chosen].sum(1) >= 1)


def test_large_irreducible_component_is_declined_before_solver(monkeypatch):
    from NepTrainKit.core.io import sampling_states
    matrix = csr_matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool)
    monkeypatch.setattr(sampling_states, "_MAX_COMPONENT_VARIABLES", 2)
    monkeypatch.setattr(sampling_states, "milp", lambda *a, **k: pytest.fail("solver must not run"))
    chosen, stats = _solve_cover(matrix, 3)
    assert chosen is None and stats["status"] == "component_resource_limit"


def test_witness_requirements_release_every_member_once_satisfied():
    incidence = csr_matrix([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)
    requirements = _WitnessRequirements(incidence)
    assert requirements.pending().tolist() == [True, True, True, False]
    requirements.add(0)
    assert requirements.pending().tolist() == [False, True, True, False]
    requirements.add(2)
    assert not requirements.pending().any()


def test_state_coverage_matches_numpy_backend_and_meets_witnesses(monkeypatch):
    x = np.arange(21.)[:, None]
    atoms = np.column_stack((x, np.sin(x)))
    blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"),
                                    (atoms, x), atoms, (1,)*len(x), ("C",)*len(x))
    native, stats = _select_state_coverage(blocks, plan(len(x)), budget=7)
    assert len(native.indices) == len(set(native.indices)) == 7
    assert stats["unmet_witnesses"] == 0
    assert abs(x-x[list(native.indices)].T).min(1).max() <= 1
    monkeypatch.setattr(sampling_coverage, "_native_sampling", None)
    reference, _ = _select_state_coverage(blocks, plan(len(x)), budget=7)
    assert reference.indices == native.indices
    np.testing.assert_allclose(native.distances, reference.distances, atol=1e-12)


def test_every_physical_stratum_keeps_a_frame_and_too_small_budget_is_rejected():
    x = np.arange(30.)[:, None]
    blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"), (x, x))
    groups = ((PhysicsSamplingStratum(("C",), "bcc"), tuple(range(27))),
              (PhysicsSamplingStratum(("C",), "fcc"), (27, 28, 29)))
    p = PhysicsSamplingPlan(groups, False, (), (), ())
    selected = select_state_coverage(blocks, p, budget=3).indices
    assert any(i >= 27 for i in selected) and any(i < 27 for i in selected)
    with pytest.raises(ValueError, match="physical strata"):
        select_state_coverage(blocks, p, budget=1)


@pytest.mark.parametrize("failure", [None, "solver_time_limit", "component_resource_limit",
                                    "no_certified_solution"])
@pytest.mark.parametrize("rare", [0, 1, 3, 5])
def test_infeasible_or_limited_cover_still_reserves_every_physical_stratum(monkeypatch, failure, rare):
    from NepTrainKit.core.io import sampling_states

    x = np.arange(9.)[:, None]
    blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"), (x, x))
    groups = ((PhysicsSamplingStratum(("C",), "common"), tuple(i for i in range(9) if i != rare)),
              (PhysicsSamplingStratum(("C",), "rare"), (rare,)))
    p = PhysicsSamplingPlan(groups, False, (), (), ())
    if failure is not None:
        monkeypatch.setattr(sampling_states, "_solve_cover",
                            lambda *args: (None, {"status": failure}))
    selected, stats = _select_state_coverage(blocks, p, budget=2)
    assert len(selected.indices) == len(set(selected.indices)) == 2
    assert all(set(rows).intersection(selected.indices) for _, rows in groups)
    assert stats["cover"]["status"] == (failure or "certificate_budget_exceeded")
    # Range/continuity deficits remain explicit, rather than being reported
    # as fully covered merely because the categorical floor was met.
    assert stats["unmet_witnesses"] > 0
    assert selected.evidence.uncovered_indices


def test_frame_density_and_duplicates_do_not_buy_path_seats():
    rng = np.random.default_rng(3)
    cloud = rng.normal(size=(300, 3))

    def path_seats(density, copies):
        t = np.r_[np.linspace(0, .5, 20*density, endpoint=False), np.linspace(.5, 1, 20)]
        path = np.c_[2*t, np.zeros((len(t), 2))] + [6, 0, 0]
        pool = np.repeat(cloud, copies, axis=0)
        x = np.vstack((pool, path))
        blocks = SamplingFeatureBlocks(("descriptor_mean", "input_geometry:C"), (x, x))
        selected = np.asarray(select_state_coverage(blocks, plan(len(x)), budget=30).indices)
        return int(np.sum(selected >= len(pool)))

    reference = path_seats(1, 1)
    assert path_seats(10, 1) == reference
    assert path_seats(1, 5) == reference
