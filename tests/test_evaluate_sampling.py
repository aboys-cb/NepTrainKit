import numpy as np
import pytest

from tools.evaluate_sampling import evaluate, scalar_report, validate_indices


def test_endpoints_alone_do_not_pass_interior_coverage():
    report = scalar_report(np.arange(101), [0, 100])
    assert report["range_fraction"] == 1
    assert report["gap_max"] == .5
    assert report["gap_p95"] > .4


def test_joint_gap_detects_missing_combinations_despite_full_marginals():
    points = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
    result = evaluate({"scalar__x": points[:, 0], "scalar__y": points[:, 1],
                       "view__joint": points}, {"diagonal": [0, 1]})["diagonal"]
    assert result["scalars"]["x"]["gap_max"] == 0
    assert result["scalars"]["y"]["gap_max"] == 0
    assert result["views"]["joint"]["max"] == 1


def test_equivalent_selected_state_can_cover_another_source():
    result = evaluate({"view__x": np.array([[0], [0], [1]]),
                       "group": np.array(["a", "b", "b"])}, {"method": [0, 2]})
    assert result["method"]["views"]["x"]["groups"]["b"]["max"] == 0


def test_missing_constant_tied_and_signed_values():
    report = scalar_report([-5, -5, 0, 5, 5, np.nan], [0, 2])
    assert report["missing"] == 1
    assert report["tails"]["0.01"]["population"] == 2
    assert report["tails"]["0.99"]["selected"] == 0
    assert scalar_report([1, 1], [0])["gap_max"] == 0
    assert scalar_report([1, np.nan], [1])["gap_max"] is None
    assert "full_range" not in scalar_report([np.nan], [0])


def test_units_and_order_do_not_change_coverage():
    values = np.array([1., 2., 3., 7., 9.])
    a = scalar_report(values, [0, 2, 4])
    b = scalar_report(values * 100 + 3000, [4, 2, 0])
    assert a["gap_max"] == pytest.approx(b["gap_max"])
    assert a["range_fraction"] == pytest.approx(b["range_fraction"])


@pytest.mark.parametrize("ids", [[0, 0], [-1], [3], [1.5], [], [[0]]])
def test_invalid_indices_rejected(ids):
    with pytest.raises(ValueError):
        validate_indices(ids, 3)


def test_invalid_view_and_row_alignment_rejected():
    with pytest.raises(ValueError):
        evaluate({"view__x": np.array([[np.nan]])}, {"a": [0]})
    with pytest.raises(ValueError):
        evaluate({"scalar__x": np.arange(3), "group": ["a"]}, {"a": [0]})
