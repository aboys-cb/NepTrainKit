from types import SimpleNamespace

from NepTrainKit.core.audit.evidence_cache import TrainingSetEvidenceCache
from NepTrainKit.core.audit.result import (
    AuditFingerprints,
    AuditResult,
    CompositionMagneticEvidence,
    CompositionPhaseEvidence,
    ElementMagneticEvidence,
    ElementMagneticSummary,
    ElementPairMagneticEvidence,
    ElementPairMagneticSummary,
    MagneticInventory,
    PhaseInventory,
    StructureMagneticEvidence,
    StructurePhaseEvidence,
)


def _phase_inventory() -> PhaseInventory:
    structure = StructurePhaseEvidence(
        source_index=7,
        atom_count=4,
        phase_label="fcc",
        confidence_state="strong",
        local_phase_fractions=(("fcc", 1.0), ("unresolved", 0.0)),
    )
    point = CompositionPhaseEvidence(
        reduced_counts=(1, 1),
        source_structure_count=1,
        analyzed_structure_count=1,
        analyzed_atom_count=4,
        local_phase_fractions=(("fcc", 1.0),),
        structure_phase_fractions=(("fcc", 1.0),),
        confidence_counts=(("strong", 1),),
        confirmed_candidates=(("l12", 0),),
        structures=(structure,),
    )
    return PhaseInventory(
        schema_version="phase-inventory-v2",
        method_id="adaptive-cna-ordering-v1",
        reference_bank_id="aflow-l12-laves-v1",
        analysis_strategy="all-structures-v1",
        source_structure_count=1,
        analyzed_structure_count=1,
        analyzed_atom_count=4,
        composition_points=(point,),
    )


def _magnetic_inventory() -> MagneticInventory:
    element = ElementMagneticEvidence(
        element="Fe",
        atom_count=4,
        spin_atom_count=4,
        order_label="aligned",
        mean_moment=2.1,
        net_moment_ratio=0.98,
        collinearity=1.0,
        intra_element_correlation=0.95,
        intra_element_pair_count=12,
        q_peak_strength=0.1,
        q_vector=(1, 0, 0),
    )
    pair = ElementPairMagneticEvidence(
        element_a="Fe",
        element_b="Ni",
        pair_count=8,
        correlation=-0.6,
        coupling_label="antiparallel",
    )
    structure = StructureMagneticEvidence(
        source_index=7,
        atom_count=4,
        spin_atom_count=4,
        order_label="fm",
        confidence_state="strong",
        mean_moment=2.1,
        moment_std=0.1,
        net_moment_ratio=0.98,
        collinearity=1.0,
        coplanarity=1.0,
        neighbor_correlation=0.95,
        neighbor_abs_correlation=0.95,
        parallel_fraction=1.0,
        antiparallel_fraction=0.0,
        q_peak_strength=0.1,
        q_vector=(1, 0, 0),
        element_evidence=(element,),
        element_pair_evidence=(pair,),
        order_subtype="",
    )
    point = CompositionMagneticEvidence(
        reduced_counts=(1, 1),
        source_structure_count=1,
        analyzed_structure_count=1,
        missing_spin_count=0,
        order_fractions=(("fm", 1.0),),
        confidence_counts=(("strong", 1),),
        mean_net_moment_ratio=0.98,
        mean_collinearity=1.0,
        mean_q_peak_strength=0.1,
        element_summaries=(
            ElementMagneticSummary(
                element="Fe",
                structure_count=1,
                order_fractions=(("aligned", 1.0),),
                mean_moment=2.1,
                mean_net_moment_ratio=0.98,
                mean_collinearity=1.0,
                mean_intra_element_correlation=0.95,
                mean_q_peak_strength=0.1,
            ),
        ),
        element_pair_summaries=(
            ElementPairMagneticSummary(
                element_a="Fe",
                element_b="Ni",
                structure_count=1,
                coupling_fractions=(("antiparallel", 1.0),),
                mean_correlation=-0.6,
            ),
        ),
        structures=(structure,),
    )
    return MagneticInventory(
        schema_version="magnetic-inventory-v3",
        method_id="spin-order-layer-afm-v3",
        analysis_strategy="all-spin-structures-v1",
        source_structure_count=1,
        analyzed_structure_count=1,
        missing_spin_count=0,
        composition_points=(point,),
    )


def test_evidence_cache_round_trips_phase_and_magnetic_inventories(tmp_path):
    cache = TrainingSetEvidenceCache(tmp_path, "train.xyz", "dataset-fp", "scope-fp")
    phase = _phase_inventory()
    magnetic = _magnetic_inventory()

    assert cache.save_phase(phase)
    assert cache.save_magnetic(magnetic)
    assert cache.load_phase(
        schema_version=phase.schema_version,
        method_id=phase.method_id,
        reference_bank_id=phase.reference_bank_id,
        analysis_strategy=phase.analysis_strategy,
    ) == phase
    assert cache.load_magnetic(
        schema_version=magnetic.schema_version,
        method_id=magnetic.method_id,
        analysis_strategy=magnetic.analysis_strategy,
    ) == magnetic
    assert not list(cache.directory.glob("*.tmp"))


def test_evidence_cache_fails_closed_for_changed_identity_or_method(tmp_path):
    phase = _phase_inventory()
    cache = TrainingSetEvidenceCache(tmp_path, "train", "dataset-fp", "scope-fp")
    assert cache.save_phase(phase)

    changed = TrainingSetEvidenceCache(tmp_path, "train", "changed", "scope-fp")
    assert changed.load_phase(
        schema_version=phase.schema_version,
        method_id=phase.method_id,
        reference_bank_id=phase.reference_bank_id,
        analysis_strategy=phase.analysis_strategy,
    ) is None
    assert cache.load_phase(
        schema_version=phase.schema_version,
        method_id="new-method",
        reference_bank_id=phase.reference_bank_id,
        analysis_strategy=phase.analysis_strategy,
    ) is None


def test_evidence_cache_uses_result_directory_and_respects_output_setting(tmp_path):
    descriptor = tmp_path / "results" / "descriptor.out"
    result_data = SimpleNamespace(
        descriptor_path=descriptor,
        data_xyz_path=tmp_path / "train.xyz",
        cache_outputs_enabled=lambda: True,
    )
    audit_result = AuditResult(
        dataset_id="train",
        generated_at="now",
        inputs={},
        fingerprints=AuditFingerprints(dataset="dataset-fp", scope="scope-fp"),
    )

    cache = TrainingSetEvidenceCache.from_result_data(result_data, audit_result)

    assert cache is not None
    assert cache.directory == descriptor.parent / ".neptrainkit-cache"
    result_data.cache_outputs_enabled = lambda: False
    assert TrainingSetEvidenceCache.from_result_data(result_data, audit_result) is None
