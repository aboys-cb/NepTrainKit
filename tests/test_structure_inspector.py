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


def test_structure_info_widget_starts_idle_until_structure_is_loaded(app):
    widget = StructureInfoWidget()

    assert widget.phase_badge.text() == "Not analyzed"
    assert widget.contact_badge.text() == "Not analyzed"
    assert all(value.text() == "—" for value in widget._metric_values.values())


def test_structure_info_widget_prioritizes_phase_and_quality(app):
    widget = StructureInfoWidget()
    widget.show_structure_info(_structure())
    widget.show_analysis(_inspection(), _phase())

    assert widget.phase_badge.text() == "FCC · Strong evidence"
    assert widget.phase_summary_label.text() == (
        "a-CNA local environments (FCC/HCP/BCC only): "
        "FCC 87.5% · HCP 6.2% · Other / unresolved 6.2%"
    )
    assert widget.crystallography_label.text() == (
        "Reference crystallography (ideal prototype): "
        "cF4 · Fm-3m (No. 225) · Face-centered cubic Bravais lattice"
    )
    assert widget.crystallography_label.isVisibleTo(widget)
    assert widget.formula_text.text() == "H<sub>2</sub>"
    assert widget.atom_num_text.text() == "2"
    assert widget.shortest_text.text() == "0.740 Å  ·  H–H"
    assert widget.maximum_force_text.text() == "1.000 eV/Å"
    assert widget.net_force_text.text() == "0.000 eV/Å"
    assert widget.contact_badge.text() == "Within threshold"


def test_structure_info_widget_explains_prototype_vs_acna_evidence(app):
    widget = StructureInfoWidget()
    phase = StructurePhaseEvidence(
        source_index=0,
        atom_count=24,
        phase_label="c15",
        confidence_state="strong",
        local_phase_fractions=(
            ("fcc", 0.0),
            ("hcp", 0.0),
            ("bcc", 0.0),
            ("unresolved", 1.0),
        ),
    )

    widget.show_structure_info(_structure())
    widget.show_analysis(_inspection(), phase)

    assert widget.phase_badge.text() == "C15 Laves · Confirmed prototype"
    assert widget.phase_summary_label.text() == (
        "a-CNA local environments (FCC/HCP/BCC only): Other / unresolved 100.0%"
    )
    assert widget.crystallography_label.text() == (
        "Reference crystallography (ideal prototype): "
        "cF24 · Fd-3m (No. 227) · Face-centered cubic Bravais lattice"
    )
    assert widget.crystallography_label.isVisibleTo(widget)
    assert "separate geometry and species-ordering checks" in (
        widget.phase_summary_label.toolTip()
    )


@pytest.mark.parametrize(
    ("phase_label", "reference"),
    (
        ("fcc", "cF4 · Fm-3m (No. 225) · Face-centered cubic Bravais lattice"),
        ("bcc", "cI2 · Im-3m (No. 229) · Body-centered cubic Bravais lattice"),
        ("hcp", "hP2 · P6₃/mmc (No. 194) · Primitive hexagonal Bravais lattice"),
        ("diamond", "cF8 · Fd-3m (No. 227) · Face-centered cubic Bravais lattice"),
        (
            "l10",
            "tP2 · P4/mmm (No. 123) · "
            "Primitive tetragonal Bravais lattice; FCC-derived ordering",
        ),
        (
            "l12",
            "cP4 · Pm-3m (No. 221) · "
            "Primitive cubic Bravais lattice; FCC-derived ordering",
        ),
        ("b1", "cF8 · Fm-3m (No. 225) · Face-centered cubic Bravais lattice"),
        (
            "b2",
            "cP2 · Pm-3m (No. 221) · "
            "Primitive cubic Bravais lattice; BCC-derived ordering",
        ),
        ("b3", "cF8 · F-43m (No. 216) · Face-centered cubic Bravais lattice"),
        ("b4", "hP4 · P6₃mc (No. 186) · Primitive hexagonal Bravais lattice"),
        ("fluorite", "cF12 · Fm-3m (No. 225) · Face-centered cubic Bravais lattice"),
        ("nias", "hP4 · P6₃/mmc (No. 194) · Primitive hexagonal Bravais lattice"),
        (
            "d03",
            "cF16 · Fm-3m (No. 225) · "
            "Face-centered cubic Bravais lattice; BCC-derived ordering",
        ),
        ("l21", "cF16 · Fm-3m (No. 225) · Face-centered cubic Bravais lattice"),
        ("c1b", "cF12 · F-43m (No. 216) · Face-centered cubic Bravais lattice"),
        ("d019", "hP8 · P6₃/mmc (No. 194) · Primitive hexagonal Bravais lattice"),
        ("c14", "hP12 · P6₃/mmc (No. 194) · Primitive hexagonal Bravais lattice"),
        ("c15", "cF24 · Fd-3m (No. 227) · Face-centered cubic Bravais lattice"),
    ),
)
def test_structure_info_widget_shows_reference_crystallography_for_strong_phases(
    app,
    phase_label,
    reference,
):
    widget = StructureInfoWidget()
    widget.show_phase_evidence(
        StructurePhaseEvidence(
            source_index=0,
            atom_count=24,
            phase_label=phase_label,
            confidence_state="strong",
            local_phase_fractions=(("unresolved", 1.0),),
        )
    )

    assert widget.crystallography_label.text() == (
        f"Reference crystallography (ideal prototype): {reference}"
    )
    assert widget.crystallography_label.isVisibleTo(widget)


def test_structure_info_widget_hides_reference_crystallography_for_mixed_evidence(app):
    widget = StructureInfoWidget()
    widget.show_phase_evidence(
        StructurePhaseEvidence(
            source_index=0,
            atom_count=24,
            phase_label="fcc",
            confidence_state="mixed",
            local_phase_fractions=(("fcc", 0.5), ("hcp", 0.5)),
        )
    )

    assert widget.crystallography_label.isHidden()


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
