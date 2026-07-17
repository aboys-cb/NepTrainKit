import sys
from pathlib import Path

from ase import Atoms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.motif_lite_reference import (
    compare_to_reference,
    enumerate_reference,
    enumerate_signatures,
    expected_probabilities,
)


def test_enumerate_signatures_counts_compositions():
    signatures = enumerate_signatures(["Co", "Ni"], ["Co", "Ni"], [2])

    assert len(signatures) == 6
    assert "Co | NN: Co1 Ni1 | cn=2" in signatures
    assert "Ni | NN: Ni2 | cn=2" in signatures


def test_compare_to_reference_reports_missing_and_outside():
    atoms = Atoms(
        symbols=["Ni", "Co", "Cr"],
        positions=[(0, 0, 0), (1, 0, 0), (-1, 0, 0)],
        cell=[8, 8, 8],
        pbc=False,
    )
    reference = enumerate_signatures(["Co", "Cr"], ["Ni"], [2])

    report = compare_to_reference([atoms], reference, cutoff=1.1, mult=1.2, top=10)

    assert report["reference_signature_count"] == 3
    assert report["observed_reference_signature_count"] == 1
    assert report["coverage"] == 1 / 3
    assert report["outside_reference_signature_count"] == 2


def test_expected_probabilities_weight_common_compositions_more():
    reference = enumerate_reference(["Co", "Ni"], ["Co", "Ni"], [2])
    expected = expected_probabilities(reference, {"Co": 0.5, "Ni": 0.5})

    assert sum(expected.values()) == 1.0
    assert expected["Co | NN: Co1 Ni1 | cn=2"] > expected["Co | NN: Co2 | cn=2"]


def test_compare_to_reference_reports_expected_mass_covered():
    atoms = Atoms(
        symbols=["Ni", "Co", "Cr"],
        positions=[(0, 0, 0), (1, 0, 0), (-1, 0, 0)],
        cell=[8, 8, 8],
        pbc=False,
    )
    reference_entries = enumerate_reference(["Co", "Cr"], ["Ni"], [2])
    expected = expected_probabilities(reference_entries, {"Ni": 1.0, "Co": 0.5, "Cr": 0.5})

    report = compare_to_reference(
        [atoms],
        [entry["signature"] for entry in reference_entries],
        cutoff=1.1,
        mult=1.2,
        top=10,
        expected=expected,
    )

    assert report["expected_probability_mass_covered"] == expected["Ni | NN: Co1 Cr1 | cn=2"]
    assert report["missing_expected_probability_mass"] > 0
