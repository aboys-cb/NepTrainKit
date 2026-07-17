import pytest
from ase import Atoms
from ase.io import write

from NepTrainKit.core.audit.phase_refinement import _c14_prototype, _fcc_prototype
from tools.evaluate_phase_atlas import evaluate_dataset


pytest.importorskip("ovito")


def _prototype_atoms(builder, symbols):
    positions, cell, atom_types = builder()
    return Atoms(
        symbols=[symbols[int(value)] for value in atom_types],
        positions=positions,
        cell=cell,
        pbc=True,
    )


def test_phase_atlas_aggregates_geometry_and_candidate_local_fractions(tmp_path):
    dataset = tmp_path / "phase_candidates.xyz"
    write(
        dataset,
        [
            _prototype_atoms(_fcc_prototype, ("Al", "Ni")),
            _prototype_atoms(_c14_prototype, ("Mg", "Zn")),
        ],
        format="extxyz",
    )

    report = evaluate_dataset(dataset, sample_per_composition=1)
    by_counts = {
        tuple(sorted(row["counts"].items())): row
        for row in report["by_composition"]
    }
    l12 = by_counts[(("Al", 27), ("Ni", 81))]
    c14 = by_counts[(("Mg", 72), ("Zn", 144))]

    assert l12["cna_local_fractions"]["fcc"] == 1.0
    assert l12["candidate_local_phases"]["l12"] == {
        "eligible_structures": 1,
        "confirmed_structures": 1,
        "labels": {"l12": 1},
        "eligible_atoms": 108,
        "local_match_fraction": 1.0,
    }
    assert c14["cna_local_fractions"]["other_or_unresolved"] == 1.0
    assert c14["candidate_local_phases"]["laves"] == {
        "eligible_structures": 1,
        "confirmed_structures": 1,
        "labels": {"c14": 1},
        "eligible_atoms": 216,
        "local_match_fraction": 1.0,
    }
