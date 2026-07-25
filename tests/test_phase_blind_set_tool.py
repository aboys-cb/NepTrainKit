import json

import numpy as np

from tools.validate_phase_blind_set import BlindSource, _distort, _load_source


def test_blind_set_loads_cached_optimade_structure_without_network(tmp_path):
    source = BlindSource(
        "fixture",
        "test",
        "https://invalid.example/fixture",
        "optimade",
        7,
        "AlNi3",
        "l12",
        "l12",
        True,
        (2, 1, 1),
    )
    payload = {
        "data": {
            "attributes": {
                "species_at_sites": ["Al", "Ni", "Ni", "Ni"],
                "cartesian_site_positions": [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.75, 1.75],
                    [1.75, 0.0, 1.75],
                    [1.75, 1.75, 0.0],
                ],
                "lattice_vectors": [
                    [3.5, 0.0, 0.0],
                    [0.0, 3.5, 0.0],
                    [0.0, 0.0, 3.5],
                ],
            }
        }
    }
    (tmp_path / "fixture.json").write_text(json.dumps(payload), encoding="utf-8")

    atoms = _load_source(source, tmp_path)
    clean = _distort(atoms, 7, strain=0.0, noise=0.0, shear=0.0)

    assert len(atoms) == 8
    np.testing.assert_allclose(atoms.cell.lengths(), (7.0, 3.5, 3.5))
    np.testing.assert_allclose(clean.positions, atoms.positions)
    assert clean is not atoms
