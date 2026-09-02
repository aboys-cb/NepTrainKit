from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import NepTrainKit.core.audit.phase_sketch as phase_sketch_module
from NepTrainKit.core.audit.phase_inventory import (
    analyze_structure_phase,
    build_phase_inventory,
    phase_partition_label,
    summarize_phase_inventory,
)
from NepTrainKit.core.audit.result import (
    CompositionPhaseEvidence,
    CompositionPoint,
    DatasetInventory,
    PhaseInventory,
    StructurePhaseEvidence,
)
from NepTrainKit.core.geometry_cache import GeometrySnapshot


class _CacheOwner:
    def __init__(self) -> None:
        self.values = {}

    def cached_geometry_analysis(self, namespace, key, build):
        cache_key = (namespace, key)
        if cache_key in self.values:
            return self.values[cache_key], True
        value = build()
        self.values[cache_key] = value
        return value, False


class _CellWithExplicitStorage:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def __array__(self, *args, **kwargs):
        raise AssertionError("ASE-style cell array protocol must not be invoked")


def _geometry() -> GeometrySnapshot:
    return GeometrySnapshot(
        source_indices=np.asarray((0, 1, 2), dtype=np.int64),
        positions=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        ),
        atom_offsets=np.asarray((0, 2, 4, 6), dtype=np.int64),
        cells=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 3, axis=0),
        pbc=np.ones((3, 3), dtype=np.uint8),
        atomic_numbers=np.asarray((28, 27, 28, 27, 28, 27), dtype=np.int16),
    )


def _inventory() -> DatasetInventory:
    return DatasetInventory(
        structure_count=3,
        elements=("Co", "Ni"),
        composition_points=(
            CompositionPoint(
                reduced_counts=(1, 1),
                fractions=(0.5, 0.5),
                structure_count=3,
                share=1.0,
                structure_indices=(0, 1, 2),
            ),
        ),
    )


def test_phase_inventory_analyzes_every_structure_and_is_dataset_cached():
    owner = _CacheOwner()
    local_counts = (
        {"fcc": 2},
        {"bcc": 2},
        {"unresolved": 2},
    )
    progress = []
    with (
        patch(
            "NepTrainKit.core.audit.phase_inventory._local_phase_counts",
            side_effect=local_counts,
        ) as local_mock,
        patch(
            "NepTrainKit.core.audit.phase_inventory._confirmed_ordering",
            side_effect=(None, "l12", None),
        ),
    ):
        first, first_hit = build_phase_inventory(
            _geometry(),
            _inventory(),
            cache_owner=owner,
            progress=lambda completed, total: progress.append((completed, total)),
        )
        second, second_hit = build_phase_inventory(
            _geometry(),
            _inventory(),
            cache_owner=owner,
            progress=lambda completed, total: progress.append((completed, total)),
        )

    assert first_hit is False
    assert second_hit is True
    assert second is first
    assert local_mock.call_count == 3
    assert first.schema_version == "phase-inventory-v2"
    assert first.method_id == "adaptive-cna-prototype-v2"
    assert first.reference_bank_id == "aflow-common-prototypes-v2"
    assert first.analysis_strategy == "all-structures-v1"
    assert first.analyzed_structure_count == 3
    assert progress == [(3, 3), (3, 3)]
    point = first.composition_points[0]
    assert point.analyzed_structure_count == 3
    assert dict(point.local_phase_fractions) == {
        "fcc": 1 / 3,
        "hcp": 0.0,
        "bcc": 1 / 3,
        "unresolved": 1 / 3,
    }
    assert dict(point.structure_phase_fractions) == {
        "fcc": 1 / 3,
        "l12": 1 / 3,
        "unresolved": 1 / 3,
    }
    assert point.confirmed_candidates == (("l12", 1),)
    assert tuple(item.source_index for item in point.structures) == (0, 1, 2)
    assert tuple(item.phase_label for item in point.structures) == (
        "fcc",
        "l12",
        "unresolved",
    )


def test_single_structure_phase_api_preserves_source_index_and_local_fractions():
    structure = SimpleNamespace(
        positions=np.zeros((4, 3), dtype=np.float32),
        cell=_CellWithExplicitStorage(np.eye(3, dtype=np.float64)),
        additional_fields={"pbc": "T T T"},
        numbers=[28, 28, 28, 28],
    )
    with (
        patch(
            "NepTrainKit.core.audit.phase_inventory._local_phase_counts",
            return_value={"fcc": 3, "unresolved": 1},
        ),
        patch(
            "NepTrainKit.core.audit.phase_inventory._confirmed_ordering",
            return_value=None,
        ),
    ):
        result = analyze_structure_phase(structure, source_index=19)

    assert result.source_index == 19
    assert result.phase_label == "fcc"
    assert result.confidence_state == "mixed"
    assert dict(result.local_phase_fractions) == {
        "fcc": 0.75,
        "hcp": 0.0,
        "bcc": 0.0,
        "unresolved": 0.25,
    }


def test_phase_classification_reuses_one_native_neighbor_field():
    structure = SimpleNamespace(
        positions=np.zeros((4, 3), dtype=np.float32),
        cell=_CellWithExplicitStorage(np.eye(3, dtype=np.float64)),
        additional_fields={"pbc": "T T T"},
        numbers=[28, 28, 28, 28],
    )
    vectors = np.zeros((4, 32, 3), dtype=np.float32)
    indices = np.zeros((4, 32), dtype=np.int32)
    valid = np.zeros((4, 32), dtype=bool)
    labels = np.asarray((1, 1, 1, 1), dtype=np.int8)
    with (
        patch.object(
            phase_sketch_module,
            "phase_partition_primitives",
            return_value=(vectors, indices, valid, labels),
        ) as primitive_mock,
        patch(
            "NepTrainKit.core.audit.phase_inventory._local_phase_counts",
            return_value={"fcc": 4},
        ) as local_mock,
        patch(
            "NepTrainKit.core.audit.phase_inventory._confirmed_ordering",
            return_value=None,
        ) as ordering_mock,
    ):
        result = analyze_structure_phase(structure, source_index=3)

    primitive_mock.assert_called_once()
    assert local_mock.call_args.kwargs["cna_labels"] is labels
    neighbor_data = ordering_mock.call_args.kwargs["neighbor_data"]
    assert neighbor_data[0] is vectors
    assert neighbor_data[1] is indices
    assert neighbor_data[2] is valid
    assert result.phase_label == "fcc"


def test_phase_summary_weights_exact_compositions_by_analyzed_atoms():
    def point(reduced_counts, analyzed_atoms, phase):
        return CompositionPhaseEvidence(
            reduced_counts=reduced_counts,
            source_structure_count=1,
            analyzed_structure_count=1,
            analyzed_atom_count=analyzed_atoms,
            local_phase_fractions=tuple(
                (label, 1.0 if label == phase else 0.0)
                for label in ("fcc", "hcp", "bcc", "unresolved")
            ),
            structure_phase_fractions=((phase, 1.0),),
            confidence_counts=(("strong", 1),),
        )

    inventory = PhaseInventory(
        schema_version="phase-inventory-v2",
        method_id="adaptive-cna-ordering-v1",
        reference_bank_id="aflow-l12-laves-v1",
        analysis_strategy="all-structures-v1",
        source_structure_count=2,
        analyzed_structure_count=2,
        analyzed_atom_count=100,
        composition_points=(
            point((1, 0), 90, "fcc"),
            point((1, 1), 10, "bcc"),
        ),
    )

    summary = summarize_phase_inventory(inventory)

    assert summary is not None
    assert dict(summary.local_phase_fractions) == {
        "fcc": 0.9,
        "hcp": 0.0,
        "bcc": 0.1,
        "unresolved": 0.0,
    }


def test_mixed_structure_evidence_is_not_counted_as_a_hard_phase_label():
    structure = StructurePhaseEvidence(
        source_index=7,
        atom_count=32,
        phase_label="bcc",
        confidence_state="mixed",
        local_phase_fractions=(("bcc", 0.5), ("fcc", 0.5)),
    )

    assert phase_partition_label(structure) == "mixed"
