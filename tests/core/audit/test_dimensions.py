from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list as ase_neighbor_list

from NepTrainKit.core.audit import local_chemistry
from NepTrainKit.core.audit.composition import audit_composition
from NepTrainKit.core.audit.config_types import audit_config_types
from NepTrainKit.core.audit.engine import build_training_set_audit
from NepTrainKit.core.audit.extract import StructureAuditRecord, indexed_structures_from_result_data
from NepTrainKit.core.audit.label_ranges import audit_label_ranges
from NepTrainKit.core.audit.local_chemistry import audit_local_chemistry
from NepTrainKit.core.audit.nep_cutoff import NepCutoffProfile
from NepTrainKit.core.audit.pair_contacts import PairContactCollector
from NepTrainKit.core.audit.result import AuditBiasType, AuditSeverity, AuditStatus
from NepTrainKit.core.structure import Structure


def _record(index, composition, config_type, energy=None, force=None, virial=None):
    return StructureAuditRecord(
        index=index,
        formula="X",
        num_atoms=10,
        composition=composition,
        config_type=config_type,
        energy_per_atom=energy,
        max_force=force,
        virial_norm=virial,
    )


def _structure(elements, positions, *, cell=10.0, pbc="T T T", config_type="bulk"):
    return Structure(
        np.eye(3) * cell,
        {
            "species": np.asarray(elements),
            "pos": np.asarray(positions, dtype=np.float64).reshape(-1, 3),
        },
        [
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
        ],
        {"pbc": pbc, "Config_type": config_type},
    )


def _plotted_indices(plot):
    return {
        structure_index
        for series in plot["series"]
        for group in series["structure_indices"]
        for structure_index in group
    }


def test_composition_flags_sparse_endpoint_bin():
    records = [_record(0, {"Fe": 0.0, "Ni": 1.0}, "bulk")]
    records.extend(_record(i, {"Fe": 0.5, "Ni": 0.5}, "bulk") for i in range(1, 19))
    records.append(_record(19, {"Fe": 1.0, "Ni": 0.0}, "bulk"))

    dimension, slices, overview = audit_composition(records)

    assert dimension.status == AuditStatus.AVAILABLE
    assert overview["element_count"] == 2
    sparse = [item for item in slices if item.bias_type == AuditBiasType.SPARSITY]
    assert sparse
    assert all(item.observed for item in sparse)
    assert all(item.interpretation for item in sparse)
    assert all(item.limit for item in sparse)


def test_composition_plot_emits_histogram_payload():
    records = [
        _record(0, {"Fe": 0.0, "H": 1.0}, "bulk"),
        _record(1, {"Fe": 0.5, "H": 0.5}, "bulk"),
        _record(2, {"Fe": 1.0, "H": 0.0}, "defect"),
    ]

    dimension, _, _ = audit_composition(records)

    plot = next(plot for plot in dimension.plots if plot["id"] == "composition:H")
    assert plot["kind"] == "histogram"
    assert plot["x_label"] == "Atomic fraction"
    assert sum(plot["series"][0]["counts"]) == len(records)
    assert len(plot["series"][0]["bin_edges"]) == len(plot["series"][0]["counts"]) + 1
    assert all(isinstance(group, tuple) for group in plot["series"][0]["structure_indices"])


def test_config_type_distribution_is_evidence_not_a_rarity_alarm():
    records = [
        _record(4, {"Fe": 1.0}, " bulk "),
        _record(8, {"Fe": 1.0}, "bulk"),
        _record(11, {"Fe": 1.0}, "defect"),
        _record(15, {"Fe": 1.0}, ""),
    ]

    dimension, slices, overview = audit_config_types(records)

    assert dimension.status is AuditStatus.PARTIAL
    assert slices == ()
    assert overview == {
        "group_count": 2,
        "labeled_count": 3,
        "missing_count": 1,
        "groups": {"bulk": 2, "defect": 1},
    }
    plot = dimension.plots[0]
    series = plot["series"][0]
    assert series["labels"] == ("bulk", "defect")
    assert series["counts"] == (2, 1)
    assert series["structure_indices"] == ((4, 8), (11,))
    assert plot["missing_structure_indices"] == (15,)


def test_composition_uses_exact_nonoverlapping_labels_and_fixed_edge_membership():
    fractions = (
        0.0,
        0.049,
        0.05,
        0.199,
        0.20,
        0.399,
        0.40,
        0.599,
        0.60,
        0.799,
        0.80,
        0.949,
        0.95,
        1.0,
    )
    records = [
        _record(100 + index, {"Fe": fraction, "Ni": 1.0 - fraction}, "bulk")
        for index, fraction in enumerate(fractions)
    ]

    dimension, _, _ = audit_composition(records)

    plot = next(plot for plot in dimension.plots if plot["id"] == "composition:Fe")
    series = plot["series"][0]
    assert series["bin_labels"] == (
        "0-5%",
        "5-20%",
        "20-40%",
        "40-60%",
        "60-80%",
        "80-95%",
        "95-100%",
    )
    assert series["structure_indices"] == tuple(
        (100 + 2 * bin_index, 101 + 2 * bin_index) for bin_index in range(7)
    )


def test_composition_findings_reuse_plot_bins_without_duplicating_element_text():
    fractions = [0.049, 0.05, 0.95] + [0.5] * 27
    records = [
        _record(200 + index, {"Fe": fraction, "Ni": 1.0 - fraction}, "bulk")
        for index, fraction in enumerate(fractions)
    ]

    dimension, slices, _ = audit_composition(records)

    plot = next(plot for plot in dimension.plots if plot["id"] == "composition:Fe")
    series = plot["series"][0]
    groups_by_label = dict(zip(series["bin_labels"], series["structure_indices"]))
    fe_slices = [item for item in slices if item.id.startswith("composition:Fe:")]
    assert {item.title for item in fe_slices} == {
        "Sparse composition bin: Fe 0-5%",
        "Sparse composition bin: Fe 5-20%",
        "Sparse composition bin: Fe 95-100%",
    }
    for item in fe_slices:
        label = item.id.rsplit(":", 1)[-1]
        assert item.structure_indices == groups_by_label[label]
        assert f"Fe Fe" not in item.title
        assert item.observed.startswith(f"Fe {label} contains")


def test_label_ranges_flags_high_force_tail():
    records = [_record(i, {"Fe": 1.0}, "bulk", energy=-1.0, force=0.2) for i in range(8)]
    records.extend(
        [
            _record(8, {"Fe": 1.0}, "defect", energy=-0.4, force=5.0),
            _record(9, {"Fe": 1.0}, "defect", energy=-0.3, force=6.0),
        ]
    )

    dimension, slices, overview = audit_label_ranges(records)

    assert dimension.status == AuditStatus.AVAILABLE
    assert overview["has_force"] is True
    force_slices = [item for item in slices if "force" in item.id]
    assert force_slices
    assert force_slices[0].structure_indices == (9,)
    assert force_slices[0].bias_type == AuditBiasType.RISK_CONCENTRATION


def test_label_range_tails_do_not_flag_constant_or_tied_plateaus():
    constant_records = [
        _record(i, {"Fe": 1.0}, "bulk", energy=-1.0, force=3.0)
        for i in range(20)
    ]
    tied_records = [
        _record(
            100 + i,
            {"Fe": 1.0},
            "bulk",
            energy=-1.0 if i < 18 else 5.0,
            force=0.0 if i < 16 else 5.0,
        )
        for i in range(20)
    ]

    _, constant_slices, _ = audit_label_ranges(constant_records)
    _, tied_slices, _ = audit_label_ranges(tied_records)

    assert not [item for item in constant_slices if item.id.startswith("label_ranges:")]
    assert not [item for item in tied_slices if item.id.startswith("label_ranges:")]


def test_label_range_strict_quantile_tails_preserve_force_and_energy_indices():
    records = [
        _record(
            1000 + 7 * i,
            {"Fe": 1.0},
            "bulk",
            energy=float(i),
            force=float(i),
        )
        for i in range(20)
    ]

    _, slices, _ = audit_label_ranges(records)

    by_id = {item.id: item for item in slices}
    assert by_id["label_ranges:force_high_tail"].structure_indices == (1126, 1133)
    assert by_id["label_ranges:energy_high_tail"].structure_indices == (1133,)
    force_metrics = {
        metric.name: metric for metric in by_id["label_ranges:force_high_tail"].metrics
    }
    energy_metrics = {
        metric.name: metric for metric in by_id["label_ranges:energy_high_tail"].metrics
    }
    assert force_metrics["threshold"].value == pytest.approx(17.1)
    assert force_metrics["threshold"].unit == "eV/angstrom"
    assert energy_metrics["threshold"].value == pytest.approx(18.05)
    assert energy_metrics["threshold"].unit == "eV/atom"


def test_label_plots_include_only_available_metrics_and_labeled_records():
    records = [
        _record(0, {"Fe": 1.0}, "bulk", energy=-1.0, force=0.2, virial=None),
        _record(1, {"Fe": 1.0}, "bulk", energy=-0.8, force=None, virial=None),
        _record(2, {"Fe": 1.0}, "defect", energy=None, force=0.4, virial=float("nan")),
    ]

    dimension, _, _ = audit_label_ranges(records)

    energy_labeled_count = 2
    energy_plot = next(plot for plot in dimension.plots if plot["id"] == "label_ranges:energy_per_atom")
    assert sum(energy_plot["series"][0]["counts"]) == energy_labeled_count
    assert energy_plot["labeled_count"] == energy_labeled_count
    assert energy_plot["total_count"] == len(records)
    assert {plot["id"] for plot in dimension.plots} == {
        "label_ranges:energy_per_atom",
        "label_ranges:max_force",
    }


def test_label_ranges_marks_partial_when_only_subset_has_force_labels():
    records = [_record(i, {"Fe": 1.0}, "bulk", energy=-1.0, force=0.2) for i in range(8)]
    records.extend(
        [
            _record(8, {"Fe": 1.0}, "defect", energy=-0.4, force=5.0),
            _record(9, {"Fe": 1.0}, "defect", energy=-0.3, force=6.0),
            _record(10, {"Fe": 1.0}, "defect", energy=-0.2, force=None),
            _record(11, {"Fe": 1.0}, "defect", energy=-0.1, force=None),
        ]
    )

    dimension, slices, overview = audit_label_ranges(records)

    assert dimension.status == AuditStatus.PARTIAL
    assert overview["force_labeled_count"] == 10
    assert overview["force_total_count"] == 12
    force_slices = [item for item in slices if item.id == "label_ranges:force_high_tail"]
    assert force_slices
    assert "labeled subset" in force_slices[0].observed
    assert force_slices[0].severity == AuditSeverity.MEDIUM


def test_label_tail_excludes_nonfinite_values_from_selected_indices():
    records = [
        _record(101, {"Fe": 1.0}, "bulk", force=0.1),
        _record(203, {"Fe": 1.0}, "bulk", force=0.2),
        _record(307, {"Fe": 1.0}, "bulk", force=1.0),
        _record(401, {"Fe": 1.0}, "bulk", force=2.0),
        _record(509, {"Fe": 1.0}, "bulk", force=float("inf")),
        _record(613, {"Fe": 1.0}, "bulk", force=float("nan")),
    ]

    _, slices, overview = audit_label_ranges(records)

    force_tail = next(item for item in slices if item.id == "label_ranges:force_high_tail")
    assert force_tail.structure_indices == (401,)
    assert overview["force_labeled_count"] == 4


def test_audit_plots_preserve_noncontiguous_original_structure_indices():
    records = [
        _record(101, {"Fe": 1.0}, "bulk", energy=-1.0, force=0.1, virial=1.0),
        _record(203, {"Fe": 0.5, "Ni": 0.5}, " defect ", energy=-0.5, force=0.2, virial=None),
        _record(307, {"Ni": 1.0}, "   ", energy=float("nan"), force=float("inf"), virial=2.0),
        _record(401, {"Fe": 1.0}, "bulk", energy=0.5, force=2.0, virial=float("nan")),
    ]

    composition_dimension, _, _ = audit_composition(records)
    label_dimension, _, _ = audit_label_ranges(records)

    def plotted_indices(plot):
        return {
            structure_index
            for group in plot["series"][0]["structure_indices"]
            for structure_index in group
        }

    assert all(plotted_indices(plot) == {101, 203, 307, 401} for plot in composition_dimension.plots)
    assert {plot["id"]: plotted_indices(plot) for plot in label_dimension.plots} == {
        "label_ranges:energy_per_atom": {101, 203, 401},
        "label_ranges:max_force": {101, 203, 401},
        "label_ranges:virial_norm": {101, 307},
    }


def test_indexed_structure_extraction_preserves_active_frame_indices():
    removed = _structure(["Fe"], [[1.0, 1.0, 1.0]], config_type="removed")
    first = _structure(["Fe"], [[2.0, 2.0, 2.0]], config_type="first")
    second = _structure(["Ni"], [[3.0, 3.0, 3.0]], config_type="second")
    all_data = np.asarray([removed] * 12, dtype=object)
    all_data[4] = first
    all_data[11] = second
    result_data = SimpleNamespace(
        structure=SimpleNamespace(all_data=all_data, now_indices=np.asarray([4, 11], dtype=np.int32))
    )

    indexed = indexed_structures_from_result_data(result_data)

    assert indexed == [(4, first), (11, second)]


def test_local_chemistry_uses_nep_scopes_pbc_and_original_indices():
    angular_outside = _structure(
        ["Fe", "Ni"],
        [[2.0, 2.0, 2.0], [3.5, 2.0, 2.0]],
    )
    across_boundary = _structure(
        ["Fe", "Ni"],
        [[0.2, 2.0, 2.0], [9.2, 2.0, 2.0]],
    )
    profile = NepCutoffProfile(
        elements=("Fe", "Ni"),
        radial_cutoffs=(2.0, 2.0),
        angular_cutoffs=(1.1, 1.1),
    )

    dimension, _, overview = audit_local_chemistry([(4, angular_outside), (11, across_boundary)], profile)

    assert dimension.status is AuditStatus.AVAILABLE
    assert {plot["scope"] for plot in dimension.plots} == {"angular", "radial"}
    assert {plot["title"].split(":", 1)[0] for plot in dimension.plots} == {
        "Angular core",
        "Radial context",
    }
    assert all(_plotted_indices(plot) <= {4, 11} for plot in dimension.plots)
    assert overview["available_scopes"] == ("angular", "radial")
    assert overview["center_element_count"] == 2

    angular_count = next(
        plot for plot in dimension.plots if plot["id"] == "local_chemistry:angular:Fe:neighbor_count"
    )
    radial_count = next(
        plot for plot in dimension.plots if plot["id"] == "local_chemistry:radial:Fe:neighbor_count"
    )
    angular_series = angular_count["series"][0]
    radial_series = radial_count["series"][0]
    angular_membership = dict(zip(angular_series["bin_labels"], angular_series["structure_indices"]))
    assert angular_membership["0"] == (4,)
    assert angular_membership["1"] == (11,)
    assert sum(int(label) * count for label, count in zip(angular_series["bin_labels"], angular_series["counts"])) == 1
    assert sum(int(label) * count for label, count in zip(radial_series["bin_labels"], radial_series["counts"])) == 2


def test_local_chemistry_uses_compiled_neighbor_search_instead_of_ase_neighbor_list():
    structure = _structure(
        ["Fe", "Ni"],
        [[2.0, 2.0, 2.0], [3.5, 2.0, 2.0]],
    )
    profile = NepCutoffProfile(
        elements=("Fe", "Ni"),
        radial_cutoffs=(2.0, 2.0),
        angular_cutoffs=(1.1, 1.1),
    )

    with patch.object(local_chemistry, "neighbor_list", wraps=local_chemistry.neighbor_list) as neighbor_list_spy:
        audit_local_chemistry([(4, structure)], profile)

    assert neighbor_list_spy.call_count == 0


def test_local_chemistry_compiled_neighbor_search_matches_ase_for_periodic_structure():
    structure = _structure(
        ["Fe", "Ni", "Fe"],
        [[0.2, 2.0, 2.0], [9.2, 2.0, 2.0], [5.0, 2.0, 2.0]],
    )
    profile = NepCutoffProfile(
        elements=("Fe", "Ni"),
        radial_cutoffs=(2.0, 2.0),
        angular_cutoffs=(1.1, 1.1),
    )
    atoms, _ = local_chemistry._as_atoms(structure, set(profile.elements))
    pair_cutoffs = local_chemistry._pair_cutoffs(profile, profile.elements, "radial")
    cutoff_matrix = local_chemistry._cutoff_matrix(profile, profile.elements, "radial")
    element_indices = {atomic_numbers[element]: index for index, element in enumerate(profile.elements)}

    expected = ase_neighbor_list("ijd", atoms, pair_cutoffs, self_interaction=False)
    actual = local_chemistry._compiled_neighbor_pairs(atoms, pair_cutoffs, cutoff_matrix, element_indices)

    expected_rows = sorted(zip(expected[0], expected[1], np.round(expected[2], 12)))
    actual_rows = sorted(zip(actual[0], actual[1], np.round(actual[2], 12)))
    assert actual_rows == expected_rows


def test_native_local_chemistry_aggregates_match_python_policy_fallback():
    profile = NepCutoffProfile(("Fe", "Ni"), (2.4, 2.2), (1.5, 1.3))
    structures = [
        _structure(["Fe", "Ni"], [[0.2, 0.2, 0.2], [4.8, 0.2, 0.2]]),
        _structure(
            ["Fe", "Fe", "Ni"],
            [[0.1, 0.1, 0.1], [1.2, 0.2, 0.1], [4.7, 4.8, 0.2]],
            cell=np.asarray([[5.0, 0.0, 0.0], [1.1, 4.8, 0.0], [0.3, 0.5, 5.2]]),
        ),
    ]
    indexed = list(enumerate(structures))

    native_collector = PairContactCollector(profile)
    native_local = audit_local_chemistry(
        indexed,
        profile,
        pair_contact_collector=native_collector,
    )
    native_pair = native_collector.finalize()
    reference_collector = PairContactCollector(profile)
    with patch.object(local_chemistry, "typed_neighbor_counts", return_value=None), patch.object(
        local_chemistry,
        "typed_contact_summary",
        return_value=None,
    ):
        reference_local = audit_local_chemistry(
            indexed,
            profile,
            pair_contact_collector=reference_collector,
        )
    reference_pair = reference_collector.finalize()

    assert native_local == reference_local
    assert native_pair == reference_pair


def test_local_chemistry_batches_neighbor_recovery_without_ase_conversion():
    indexed = [
        (
            index,
            _structure(["Fe", "Ni"], [[2.0, 2.0, 2.0], [3.0 + index, 2.0, 2.0]]),
        )
        for index in range(3)
    ]
    profile = NepCutoffProfile(("Fe", "Ni"), (2.0, 2.0), (1.1, 1.1))
    original_batch = local_chemistry.cutoff_neighbor_pairs_batch
    batch_count = 0

    def track_batch(*args, **kwargs):
        nonlocal batch_count
        batch_count += 1
        return original_batch(*args, **kwargs)

    with patch.object(local_chemistry, "_as_atoms", side_effect=AssertionError("ASE conversion is not expected")), patch.object(
        local_chemistry, "cutoff_neighbor_pairs_batch", side_effect=track_batch
    ):
        audit_local_chemistry(indexed, profile)

    assert batch_count == 1


def test_local_chemistry_flags_populated_bins_below_ten_percent():
    indexed = []
    frame_indices = tuple(range(20, 42, 2))
    for offset, frame_index in enumerate(frame_indices):
        distance = 8.0 if offset == 0 else 1.0
        indexed.append(
            (
                frame_index,
                _structure(["Fe", "Ni"], [[2.0, 2.0, 2.0], [2.0 + distance, 2.0, 2.0]], cell=20.0),
            )
        )
    profile = NepCutoffProfile(("Fe", "Ni"), (2.0, 2.0), (1.5, 1.5))

    _, slices, overview = audit_local_chemistry(indexed, profile)

    sparse_count = next(
        item for item in slices if item.id.startswith("local_chemistry:radial:Fe:neighbor_count:0")
    )
    assert sparse_count.bias_type is AuditBiasType.SPARSITY
    assert sparse_count.structure_indices == (20,)
    assert overview["sparse_bin_count"] == len(slices)


def test_engine_appends_unavailable_local_chemistry_without_breaking_mvp_dimensions(tmp_path: Path):
    structure = _structure(["Fe"], [[2.0, 2.0, 2.0]])
    malformed_model = tmp_path / "nep.txt"
    malformed_model.write_text("nep4 1 Fe\ncutoff 6 3 8\n", encoding="utf-8")
    result_data = SimpleNamespace(
        structure=SimpleNamespace(
            all_data=np.asarray([structure], dtype=object),
            now_indices=np.asarray([0], dtype=np.int32),
        ),
        nep_txt_path=malformed_model,
    )

    result = build_training_set_audit(result_data)

    assert tuple(dimension.id for dimension in result.dimensions) == (
        "data_quality",
        "composition",
        "config_types",
        "label_ranges",
        "local_chemistry",
        "pair_contacts",
    )
    assert result.dimensions[0].status is AuditStatus.AVAILABLE
    assert result.dimensions[-1].status is AuditStatus.UNAVAILABLE
    assert result.dimensions[-1].plots == ()
    assert result.overview_metrics["local_chemistry"]["available_scopes"] == ()
    assert result.overview_metrics["pair_contacts"]["pair_count"] == 0


def test_engine_uses_active_nep_model_for_local_chemistry(tmp_path: Path):
    structure = _structure(["Fe", "Ni"], [[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]])
    model = tmp_path / "nep.txt"
    model.write_text("nep4 2 Fe Ni\ncutoff 2 1.5 8 4\n", encoding="utf-8")
    result_data = SimpleNamespace(
        structure=SimpleNamespace(
            all_data=np.asarray([structure], dtype=object),
            now_indices=np.asarray([0], dtype=np.int32),
        ),
        nep_txt_path=model,
    )

    result = build_training_set_audit(result_data)

    local_dimension = result.dimensions[-2]
    assert local_dimension.id == "local_chemistry"
    assert local_dimension.status is AuditStatus.AVAILABLE
    assert result.overview_metrics["local_chemistry"]["available_scopes"] == ("angular", "radial")
    pair_dimension = result.dimensions[-1]
    assert pair_dimension.id == "pair_contacts"
    assert pair_dimension.status is AuditStatus.AVAILABLE


def test_pair_contacts_distinguish_co_sampling_from_local_contact(tmp_path: Path):
    structures = np.asarray(
        [
            _structure(["Fe"], [[2.0, 2.0, 2.0]]),
            _structure(["Fe", "Ni"], [[2.0, 2.0, 2.0], [8.0, 2.0, 2.0]], cell=20.0),
            _structure(["Fe", "Ni"], [[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]], cell=20.0),
        ],
        dtype=object,
    )
    model = tmp_path / "nep.txt"
    model.write_text("nep4 2 Fe Ni\ncutoff 2 1.5 8 4\n", encoding="utf-8")
    result = build_training_set_audit(
        SimpleNamespace(
            structure=SimpleNamespace(all_data=structures, now_indices=np.asarray([0, 1, 2], dtype=np.int32)),
            nep_txt_path=model,
        )
    )

    pair_dimension = result.dimensions[-1]
    radial_plot = next(plot for plot in pair_dimension.plots if plot["id"] == "pair_contacts:radial")
    labels = radial_plot["series"][0]["labels"]
    counts = radial_plot["series"][0]["counts"]
    assert counts[labels.index("Fe-Ni")] == 2
    pair_slice = next(item for item in result.slices if item.id == "pair_contacts:radial:Fe:Ni")
    metrics = {metric.name: metric.value for metric in pair_slice.metrics}
    assert metrics["co_sampled_structures"] == 2
    assert metrics["contact_structures"] == 1


def test_engine_adds_dashboard_overview_counts():
    def labeled(element, config_type, energy, force, virial):
        structure = _structure([element], [[1.0, 1.0, 1.0]], config_type=config_type)
        if energy is not None:
            structure.energy = energy
        if force is not None:
            structure.forces = np.asarray([[force, 0.0, 0.0]])
        if virial is not None:
            structure.virial = np.asarray([virial, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return structure

    structures = np.asarray(
        [
            labeled("Fe", "bulk", -1.0, 0.2, 1.0),
            labeled("Fe", "   ", float("nan"), float("nan"), None),
            labeled("Ni", " defect ", -0.8, float("inf"), float("nan")),
            labeled("Ni", "bulk", float("inf"), None, float("inf")),
        ],
        dtype=object,
    )
    result_data = SimpleNamespace(
        structure=SimpleNamespace(
            all_data=structures,
            now_indices=np.arange(len(structures), dtype=np.int32),
        ),
        nep_txt_path=None,
    )

    result = build_training_set_audit(result_data)

    assert result.overview_metrics["finding_count"] == len(result.slices)
    assert result.overview_metrics["severity_counts"] == {
        severity.value: sum(item.severity == severity for item in result.slices)
        for severity in set(item.severity for item in result.slices)
    }
    assert result.overview_metrics["label_counts"] == {"energy": 2, "force": 1, "virial": 1}
    assert "config_type_labeled_count" not in result.overview_metrics
    assert "source_group_count" not in result.overview_metrics
