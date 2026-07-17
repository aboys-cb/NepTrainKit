import sys
from pathlib import Path

from ase import Atoms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.motif_lite_shell_sensitivity import analyze_shell_sensitivity


def test_shell_sensitivity_reports_changed_atoms():
    atoms = Atoms(
        symbols=["Ni", "Fe", "Fe", "Fe", "Fe"],
        positions=[(0, 0, 0), (1.0, 0, 0), (1.1, 0, 0), (1.2, 0, 0), (4.0, 0, 0)],
        cell=[10, 10, 10],
        pbc=False,
    )

    report, rows = analyze_shell_sensitivity([atoms], top=5)

    assert report["total_structures"] == 1
    assert report["total_atoms"] == 5
    assert report["stable_atom_fraction"] < 1.0
    assert "natural-cutoff__knn12" in report["top_overlap"]
    assert rows[0]["changed_atoms"] > 0
