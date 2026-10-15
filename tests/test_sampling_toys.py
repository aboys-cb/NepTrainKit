import copy
import json

import numpy as np
import pytest
from ase.io import read

from tools.sampling_toy_data import evaluate, make_cases, make_challenges, make_mixed_cases, minimum_budget, write_case


@pytest.mark.parametrize("variant", [0, 1])
def test_mixed_tasks_have_frozen_feasible_label_free_references(tmp_path, variant):
    cases = make_mixed_cases(variant=variant)
    assert len(cases) == 2
    for frames, truth in cases:
        assert evaluate(truth, truth["oracle_indices"])["passed"]
        assert minimum_budget(truth)["fits_budget"]
        write_case(tmp_path, frames, truth)
        actual = read(tmp_path / f"{truth['name']}.xyz", index=":")
        assert len(actual) == truth["count"]
        assert all(not frame.info and frame.calc is None for frame in actual)
        assert all(set(frame.arrays) <= {"numbers", "positions"} for frame in actual)


@pytest.mark.parametrize("variant", [0, 1])
def test_nonideal_challenges_have_feasible_frozen_references_and_no_labels(tmp_path, variant):
    cases = make_challenges(variant=variant)
    assert len(cases) == 5
    for frames, truth in cases:
        assert evaluate(truth, truth["oracle_indices"])["passed"]
        exported = write_case(tmp_path, frames, truth)
        assert exported["sha256"]


@pytest.fixture(scope="module", params=[0, 1])
def cases(request):
    return make_cases(variant=request.param)


def test_references_are_feasible_and_xyz_contains_only_inputs(cases, tmp_path):
    for frames, truth in cases:
        assert evaluate(truth, truth["oracle_indices"])["passed"]
        assert minimum_budget(truth)["fits_budget"]
        saved = write_case(tmp_path, frames, truth)
        sidecar = json.loads((tmp_path / f"{truth['name']}.oracle.json").read_text())
        assert sidecar == saved
        actual = read(tmp_path / f"{truth['name']}.xyz", index=":")
        assert len(actual) == truth["count"]
        for frame in actual:
            assert not frame.info and frame.calc is None
            assert set(frame.arrays) <= {"numbers", "positions", "spin"}


def test_center_only_negative_fails_even_at_full_budget(cases):
    for _, truth in cases:
        x = np.asarray(truth["coordinates"])
        rows = np.argsort(np.linalg.norm(x - .5, axis=1))[:truth["budget"]]
        assert not evaluate(truth, rows)["passed"]


def test_evaluation_is_row_order_invariant(cases):
    for _, truth in cases:
        order = np.random.default_rng(873).permutation(truth["count"])
        inverse = np.argsort(order)
        shuffled = copy.deepcopy(truth)
        for key in ("coordinates", "required_cell", "state_id"):
            shuffled[key] = np.asarray(truth[key])[order].tolist()
        assert evaluate(truth, truth["oracle_indices"]) == evaluate(
            shuffled, inverse[truth["oracle_indices"]])


def test_coordinates_can_be_recovered_from_atomistic_inputs(cases):
    for frames, truth in cases:
        name = truth["name"]
        target = np.asarray(truth["coordinates"])
        if name.startswith("path_") or name.startswith("rare_"):
            volumes = np.asarray([s.get_volume()/len(s) for s in frames]) ** (1/3)
            branches = ([c.split(":")[0] for c in truth["required_cell"]]
                        if name.startswith("rare_") else np.zeros(len(frames)))
            recovered = np.empty(len(frames))
            for branch in np.unique(branches):
                rows = np.flatnonzero(np.asarray(branches) == branch)
                v = volumes[rows]
                recovered[rows] = (v-v.min()) / np.ptp(v)
            np.testing.assert_allclose(recovered, target[:, 0], atol=1e-12)
        elif name.startswith("joint_strain"):
            principal = np.asarray([np.sort(np.linalg.svd(s.cell.array, compute_uv=False))[:2]
                                    for s in frames])
            np.testing.assert_allclose((principal-principal.min(0))/np.ptp(principal, axis=0),
                                       target, atol=1e-12)
        else:
            moments, angles = [], []
            for frame in frames:
                a, b = frame.arrays["spin"][:2]
                moments.append(np.linalg.norm(a))
                angles.append(np.arccos(np.clip(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))/np.pi)
            moments = np.asarray(moments)
            np.testing.assert_allclose((moments-moments.min())/np.ptp(moments), target[:, 0], atol=1e-12)
            np.testing.assert_allclose(angles, target[:, 1], atol=1e-12)


def test_extra_boundary_pins_can_make_an_otherwise_feasible_budget_impossible():
    cases = {truth["name"]: truth for _, truth in make_cases()}
    spin = cases["spin_amplitude_angle"]
    assert minimum_budget(spin)["minimum"] == 12
    # Two distinct states in the same one of twelve required cells consume
    # an extra slot. This is a constraint conflict, not a search failure.
    assert minimum_budget(spin, mandatory_states=[5, 6])["minimum"] == 13
    rare = cases["rare_branches"]
    assert minimum_budget(rare)["minimum"] == 21
    assert minimum_budget(rare, mandatory_states=[0, 20, 21, 41, 62])["minimum"] == 24
    assert minimum_budget(cases["path_balanced"])["minimum"] == 10


@pytest.mark.parametrize("ids", [[], [0, 0], [-1], [10000], [1.2], [[0]]])
def test_invalid_selection_rejected(ids):
    truth = make_cases()[0][1]
    with pytest.raises(ValueError):
        evaluate(truth, ids)


def test_labelled_frame_cannot_leak_into_export(tmp_path):
    frames, truth = make_cases()[0]
    frames[0].info["energy"] = -10
    with pytest.raises(ValueError, match="only structural inputs"):
        write_case(tmp_path, frames, truth)
