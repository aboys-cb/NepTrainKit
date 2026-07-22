from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from NepTrainKit.core.audit import StructurePhaseEvidence
from NepTrainKit.core.structure import Structure
from NepTrainKit.core.structure_inspection import StructureInspection, inspect_structure
from NepTrainKit.ui.pages.show_nep import ShowNepWidget
from NepTrainKit.ui.views.structure import StructureInfoWidget


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _structure(distance: float = 0.74) -> Structure:
    return Structure(
        lattice=np.diag([4.0, 4.0, 4.0]),
        atomic_properties={
            "species": np.asarray(["H", "H"]),
            "pos": np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
            "force": np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        },
        properties=[
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
            {"name": "force", "type": "R", "count": 3},
        ],
        additional_fields={"Config_type": "molecule", "energy": -2.0, "pbc": "T T T"},
    )


def _inspection() -> StructureInspection:
    return StructureInspection(
        volume=64.0,
        mass_density=0.052,
        energy=-2.0,
        per_atom_energy=-1.0,
        maximum_force=1.0,
        rms_force=1.0,
        net_force=0.0,
        shortest_distance=0.74,
        shortest_pair=("H", "H"),
    )


def _phase() -> StructurePhaseEvidence:
    return StructurePhaseEvidence(
        source_index=0,
        atom_count=2,
        phase_label="fcc",
        confidence_state="strong",
        local_phase_fractions=(
            ("fcc", 0.875),
            ("hcp", 0.0625),
            ("bcc", 0.0),
            ("unresolved", 0.0625),
        ),
    )


def test_inspect_structure_reports_scannable_frame_metrics():
    result = inspect_structure(_structure())

    assert result.volume == pytest.approx(64.0)
    assert result.energy == pytest.approx(-2.0)
    assert result.per_atom_energy == pytest.approx(-1.0)
    assert result.maximum_force == pytest.approx(1.0)
    assert result.rms_force == pytest.approx(1.0)
    assert result.net_force == pytest.approx(0.0)
    assert result.shortest_distance == pytest.approx(0.74)
    assert result.shortest_pair == ("H", "H")
    assert result.short_contacts == ()


def test_inspect_structure_preserves_short_contact_warning():
    result = inspect_structure(_structure(distance=0.2))

    assert result.short_contacts == ((('H', 'H'), pytest.approx(0.2)),)


def test_structure_info_widget_prioritizes_phase_and_quality(app):
    widget = StructureInfoWidget()
    widget.show_structure_info(_structure())
    widget.show_analysis(_inspection(), _phase())

    assert widget.phase_badge.text() == "FCC · Strong evidence"
    assert widget.phase_summary_label.text() == (
        "Local topology: FCC 87.5% · HCP 6.2% · Unresolved 6.2%"
    )
    assert widget.formula_text.text() == "H<sub>2</sub>"
    assert widget.atom_num_text.text() == "2"
    assert widget.shortest_text.text() == "0.740 Å  ·  H–H"
    assert widget.maximum_force_text.text() == "1.000 eV/Å"
    assert widget.net_force_text.text() == "0.000 eV/Å"
    assert widget.contact_badge.text() == "Within threshold"


def test_stale_structure_analysis_result_is_cached_but_not_displayed():
    widget = ShowNepWidget.__new__(ShowNepWidget)
    dataset = object()
    widget.nep_result_data = dataset
    widget._structure_analysis_job_id = 2
    widget._structure_analysis_cache = {}
    widget._phase_evidence_dataset_id = None
    widget._phase_evidence_lookup = {}
    widget.struct_index_spinbox = SimpleNamespace(value=lambda: 8)
    widget.struct_info_widget = SimpleNamespace(
        show_analysis=MagicMock(),
        show_analysis_unavailable=MagicMock(),
    )
    widget._track_worker_thread = MagicMock()
    callbacks = {}

    def fake_run_in_thread(_parent, _func, **kwargs):
        callbacks.update(kwargs)
        return object()

    with patch("NepTrainKit.ui.pages.show_nep.run_in_thread", fake_run_in_thread):
        widget._start_structure_analysis(_structure(), 7, id(dataset), 2)

    widget._structure_analysis_job_id = 3
    callbacks["on_finished"]((_inspection(), _phase()))

    assert (id(dataset), 7) in widget._structure_analysis_cache
    widget.struct_info_widget.show_analysis.assert_not_called()
