import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.motif_lite_compare_cmi import _contingency


def test_contingency_reports_bidirectional_purity():
    report = _contingency(
        ["Ni | NN: Cr1 | cn=1", "Ni | NN: Cr1 | cn=1", "Cr | NN: Ni1 | cn=1"],
        ["Ni|7", "Ni|7", "Cr|9"],
    )

    assert report["matched_environments"] == 3
    assert report["motif_lite_unique"] == 2
    assert report["cmi_unique"] == 2
    assert report["motif_lite_to_cmi_purity"] == 1.0
    assert report["cmi_to_motif_lite_purity"] == 1.0
