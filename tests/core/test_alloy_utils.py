from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from NepTrainKit.core.alloy import assign_random_occupancy, parse_composition, simplex_grid_points


def test_parse_composition_accepts_text_and_json_forms():
    assert parse_composition("Fe:2, ni=1, O") == {"Fe": 2.0, "Ni": 1.0, "O": 1.0}
    assert parse_composition('{"fe": 2, "O": 0}') == {"Fe": 2.0, "O": 0.0}


@pytest.mark.parametrize("text", ["Fe:-1", "Fe:nan", '{"Fe": -0.1}', "[1, 2]"])
def test_parse_composition_rejects_invalid_ratios(text):
    with pytest.raises((TypeError, ValueError)):
        parse_composition(text)


def test_simplex_grid_points_binary_and_ternary_are_normalized():
    assert simplex_grid_points(2, 0.5) == [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]

    ternary = simplex_grid_points(3, 0.5, include_endpoints=False)
    assert ternary == []

    ternary_with_endpoints = simplex_grid_points(3, 0.5)
    assert all(abs(sum(point) - 1.0) < 1e-12 for point in ternary_with_endpoints)


def test_simplex_grid_points_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        simplex_grid_points(1, 0.5)
    with pytest.raises(ValueError):
        simplex_grid_points(2, 0.0)
    with pytest.raises(ValueError):
        simplex_grid_points(2, 0.5, min_fraction=-0.1)


def test_assign_random_occupancy_exact_preserves_requested_counts():
    atoms = Atoms("H4", positions=np.zeros((4, 3)))

    result = assign_random_occupancy(
        atoms,
        {"Fe": 0.5, "Ni": 0.5},
        mode="Exact",
        rng=np.random.default_rng(0),
    )

    assert sorted(result.get_chemical_symbols()) == ["Fe", "Fe", "Ni", "Ni"]
    assert atoms.get_chemical_symbols() == ["H", "H", "H", "H"]


def test_assign_random_occupancy_rejects_negative_direct_composition():
    atoms = Atoms("H2", positions=np.zeros((2, 3)))

    with pytest.raises(ValueError):
        assign_random_occupancy(atoms, {"Fe": 1.0, "Ni": -0.1})
