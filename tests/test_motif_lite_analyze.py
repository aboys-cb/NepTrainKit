import sys
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.motif_lite_analyze import _distance_matrix, analyze_structures, structure_signatures


def test_distance_matrix_matches_ase_mic_for_triclinic_cell():
    atoms = Atoms(
        symbols=["Ni", "Co"],
        scaled_positions=[(0.98, 0.97, 0.96), (0.02, 0.03, 0.04)],
        cell=[(3.2, 0.0, 0.0), (1.1, 2.9, 0.0), (0.4, 0.7, 3.4)],
        pbc=True,
    )

    np.testing.assert_allclose(
        _distance_matrix(atoms),
        atoms.get_all_distances(mic=True),
        atol=1e-12,
    )


def test_structure_signatures_count_first_shell_elements():
    atoms = Atoms(
        symbols=["Ni", "Cr", "Co"],
        positions=[(0, 0, 0), (1, 0, 0), (-1, 0, 0)],
        cell=[8, 8, 8],
        pbc=False,
    )

    signatures = structure_signatures(atoms, cutoff=1.1)

    assert "Ni | NN: Co1 Cr1 | cn=2" in signatures
    assert "Cr | NN: Ni1 | cn=1" in signatures
    assert "Co | NN: Ni1 | cn=1" in signatures


def test_analyze_structures_reports_rare_signature_contribution():
    common = Atoms(
        symbols=["Ni", "Cr"],
        positions=[(0, 0, 0), (1, 0, 0)],
        cell=[8, 8, 8],
        pbc=False,
    )
    rare = Atoms(
        symbols=["Ni", "Co"],
        positions=[(0, 0, 0), (1, 0, 0)],
        cell=[8, 8, 8],
        pbc=False,
    )

    report = analyze_structures([common, common.copy(), rare], cutoff=1.1, rare_max_count=1)

    assert report["total_structures"] == 3
    assert report["unique_signature_count"] == 4
    assert report["structures"][0]["rare_environment_count"] == 0
    assert report["structures"][2]["rare_environment_count"] == 2


def test_pair_mode_distinguishes_shell_arrangements_with_same_counts():
    clustered = Atoms(
        symbols=["Ni", "Co", "Co", "Cr", "Cr"],
        positions=[(0, 0, 0), (1, 0, 0), (1, 0.6, 0), (-1, 0, 0), (-1, -0.6, 0)],
        cell=[8, 8, 8],
        pbc=False,
    )
    mixed = Atoms(
        symbols=["Ni", "Co", "Cr", "Co", "Cr"],
        positions=[(0, 0, 0), (1, 0, 0), (1, 0.6, 0), (-1, 0, 0), (-1, -0.6, 0)],
        cell=[8, 8, 8],
        pbc=False,
    )

    clustered_count = structure_signatures(clustered, cutoff=1.3, mode="count")[0]
    mixed_count = structure_signatures(mixed, cutoff=1.3, mode="count")[0]
    clustered_pair = structure_signatures(clustered, cutoff=1.3, mode="pair")[0]
    mixed_pair = structure_signatures(mixed, cutoff=1.3, mode="pair")[0]

    assert clustered_count == mixed_count
    assert clustered_pair != mixed_pair
    assert "NN-pairs:" in clustered_pair


def test_knn_shell_uses_fixed_neighbor_count():
    atoms = Atoms(
        symbols=["Ni", "Fe", "Fe", "Fe"],
        positions=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],
        cell=[10, 10, 10],
        pbc=False,
    )

    signature = structure_signatures(atoms, shell_method="knn", shell_k=2)[0]

    assert signature == "Ni | NN: Fe2 | cn=2"


def test_adaptive_gap_shell_stops_before_large_distance_gap():
    atoms = Atoms(
        symbols=["Ni", "Fe", "Fe", "Fe"],
        positions=[(0, 0, 0), (1.0, 0, 0), (1.1, 0, 0), (4.0, 0, 0)],
        cell=[10, 10, 10],
        pbc=False,
    )

    signature = structure_signatures(
        atoms,
        shell_method="adaptive-gap",
        adaptive_min_neighbors=1,
        adaptive_max_neighbors=3,
    )[0]

    assert signature == "Ni | NN: Fe2 | cn=2"
