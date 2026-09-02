from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

import NepTrainKit.core.io.sampling_plan as sampling_plan_module
from NepTrainKit.core.audit import phase_sketch as phase_sketch_module
from NepTrainKit.core.audit.context import build_fingerprints, resolve_audit_scope
from NepTrainKit.core.audit.evidence_cache import (
    PHYSICS_SAMPLING_CACHE_KIND,
    TrainingSetEvidenceCache,
)
from NepTrainKit.core.audit.magnetic_inventory import (
    MAGNETIC_ANALYSIS_STRATEGY,
    MAGNETIC_METHOD_ID,
    MAGNETIC_SCHEMA_VERSION,
)
from NepTrainKit.core.audit.phase_inventory import (
    PHASE_ANALYSIS_STRATEGY,
    PHASE_METHOD_ID,
    PHASE_REFERENCE_BANK_ID,
    PHASE_SCHEMA_VERSION,
)
from NepTrainKit.core.audit.result import (
    AuditScopeKind,
    CompositionMagneticEvidence,
    CompositionPhaseEvidence,
    MagneticInventory,
    PhaseInventory,
    StructureMagneticEvidence,
    StructurePhaseEvidence,
)
from NepTrainKit.core.io.sampling_plan import (
    PhysicsSamplingStratum,
    allocate_physics_quotas,
    build_physics_sampling_plan,
    build_result_physics_sampling_plan,
    element_set_key,
    reduced_composition_key,
)


def _with_spin(atoms: Atoms, spins) -> Atoms:
    result = atoms.copy()
    result.arrays["spin"] = np.asarray(spins, dtype=float).reshape(len(result), 3)
    return result


def _phase_inventory_for(labels: tuple[str, ...], atom_counts: tuple[int, ...]):
    structures = tuple(
        StructurePhaseEvidence(
            source_index=index,
            atom_count=atom_count,
            phase_label=label,
            confidence_state="strong",
            local_phase_fractions=((label, 1.0),),
        )
        for index, (label, atom_count) in enumerate(zip(labels, atom_counts))
    )
    point = CompositionPhaseEvidence(
        reduced_counts=(1,),
        source_structure_count=len(structures),
        analyzed_structure_count=len(structures),
        analyzed_atom_count=sum(atom_counts),
        local_phase_fractions=(),
        structure_phase_fractions=(),
        confidence_counts=(("strong", len(structures)),),
        structures=structures,
    )
    return PhaseInventory(
        schema_version=PHASE_SCHEMA_VERSION,
        method_id=PHASE_METHOD_ID,
        reference_bank_id=PHASE_REFERENCE_BANK_ID,
        analysis_strategy=PHASE_ANALYSIS_STRATEGY,
        source_structure_count=len(structures),
        analyzed_structure_count=len(structures),
        analyzed_atom_count=sum(atom_counts),
        composition_points=(point,),
    )


def _magnetic_inventory_for(atom_counts: tuple[int, ...]):
    structures = tuple(
        StructureMagneticEvidence(
            source_index=index,
            atom_count=atom_count,
            spin_atom_count=atom_count,
            order_label="fm",
            confidence_state="strong",
            mean_moment=2.0,
            moment_std=0.0,
            net_moment_ratio=1.0,
            collinearity=1.0,
            coplanarity=1.0,
            neighbor_correlation=1.0,
            neighbor_abs_correlation=1.0,
            parallel_fraction=1.0,
            antiparallel_fraction=0.0,
            q_peak_strength=0.0,
            q_vector=(0, 0, 0),
        )
        for index, atom_count in enumerate(atom_counts)
    )
    point = CompositionMagneticEvidence(
        reduced_counts=(1,),
        source_structure_count=len(structures),
        analyzed_structure_count=len(structures),
        missing_spin_count=0,
        order_fractions=(("fm", 1.0),),
        confidence_counts=(("strong", len(structures)),),
        mean_net_moment_ratio=1.0,
        mean_collinearity=1.0,
        mean_q_peak_strength=0.0,
        structures=structures,
    )
    return MagneticInventory(
        schema_version=MAGNETIC_SCHEMA_VERSION,
        method_id=MAGNETIC_METHOD_ID,
        analysis_strategy=MAGNETIC_ANALYSIS_STRATEGY,
        source_structure_count=len(structures),
        analyzed_structure_count=len(structures),
        missing_spin_count=0,
        composition_points=(point,),
    )


def test_reduced_composition_key_matches_compatible_supercells():
    assert reduced_composition_key(Atoms("FeNi")) == (("Fe", 1), ("Ni", 1))
    assert reduced_composition_key(Atoms("Fe2Ni2")) == (("Fe", 1), ("Ni", 1))
    assert reduced_composition_key(Atoms("Fe3Ni")) == (("Fe", 3), ("Ni", 1))


def test_element_set_key_ignores_concentration():
    assert element_set_key(Atoms("FeNi")) == ("Fe", "Ni")
    assert element_set_key(Atoms("Fe3Ni")) == ("Fe", "Ni")


def test_physics_plan_keeps_different_concentrations_in_one_physical_stratum(
    monkeypatch,
):
    first = bulk("Fe", "fcc", a=3.55, cubic=True)
    first.set_chemical_symbols(["Fe", "Fe", "Ni", "Ni"])
    second = first.copy()
    second.set_chemical_symbols(["Fe", "Fe", "Fe", "Ni"])
    monkeypatch.setattr(
        sampling_plan_module,
        "analyze_structure_phase",
        lambda *_args, **_kwargs: SimpleNamespace(
            phase_label="fcc",
            confidence_state="strong",
        ),
    )

    plan = build_physics_sampling_plan([first, second], spin_model=False)

    assert plan.group_count == 1
    assert plan.groups[0][0].element_set == ("Fe", "Ni")
    assert plan.groups[0][1] == (0, 1)


def test_physics_plan_separates_bcc_fcc_and_magnetic_order():
    bcc = bulk("Fe", "bcc", a=2.86, cubic=True)
    bcc = _with_spin(bcc, np.tile([0.0, 0.0, 2.2], (len(bcc), 1)))
    fcc = bulk("Fe", "fcc", a=3.55, cubic=True)
    fcc = _with_spin(
        fcc,
        np.asarray(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, -2.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, -2.0],
            ]
        ),
    )

    plan = build_physics_sampling_plan([bcc, fcc], spin_model=True)

    keys = tuple(key for key, _indices in plan.groups)
    assert {key.phase for key in keys} == {"bcc", "fcc"}
    assert {key.magnetic_order for key in keys} == {"fm", "afm_layered"}
    assert plan.element_set_count == 1
    assert plan.missing_spin_indices == ()


def test_physics_plan_keeps_working_without_native_phase_module(monkeypatch):
    bcc = bulk("Fe", "bcc", a=2.86, cubic=True)
    bcc = _with_spin(bcc, np.tile([0.0, 0.0, 2.2], (len(bcc), 1)))
    fcc = bulk("Fe", "fcc", a=3.55, cubic=True)
    fcc = _with_spin(fcc, np.tile([0.0, 0.0, 2.0], (len(fcc), 1)))
    monkeypatch.setattr(phase_sketch_module, "_native_phase", None)

    plan = build_physics_sampling_plan([bcc, fcc], spin_model=True)

    assert plan.phase_counts == (("bcc", 1), ("fcc", 1))
    assert plan.missing_spin_indices == ()


def test_parallel_physics_classification_matches_serial_plan(monkeypatch):
    bcc = bulk("Fe", "bcc", a=2.86, cubic=True)
    bcc = _with_spin(bcc, np.tile([0.0, 0.0, 2.2], (len(bcc), 1)))
    fcc = bulk("Fe", "fcc", a=3.55, cubic=True)
    fcc = _with_spin(fcc, np.tile([0.0, 0.0, 2.0], (len(fcc), 1)))
    structures = [bcc, fcc, bcc.copy(), fcc.copy()]

    monkeypatch.setattr(
        sampling_plan_module,
        "PHYSICS_CLASSIFICATION_PARALLEL_THRESHOLD",
        len(structures) + 1,
    )
    serial = build_physics_sampling_plan(structures, spin_model=True)
    monkeypatch.setattr(
        sampling_plan_module,
        "PHYSICS_CLASSIFICATION_PARALLEL_THRESHOLD",
        1,
    )
    parallel = build_physics_sampling_plan(structures, spin_model=True)

    assert parallel == serial


def test_result_plan_migrates_old_evidence_cache_and_reloads_partition_cache(
    tmp_path,
    monkeypatch,
):
    bcc = bulk("Fe", "bcc", a=2.86, cubic=True)
    bcc = _with_spin(bcc, np.tile([0.0, 0.0, 2.2], (len(bcc), 1)))
    fcc = bulk("Fe", "fcc", a=3.55, cubic=True)
    fcc = _with_spin(fcc, np.tile([0.0, 0.0, 2.0], (len(fcc), 1)))
    structures = [bcc, fcc]
    data_path = tmp_path / "train.xyz"
    data_path.write_text("stable cache identity\n", encoding="utf8")

    def result_data():
        return SimpleNamespace(
            structure=SimpleNamespace(
                all_data=structures,
                now_indices=np.arange(len(structures), dtype=np.int64),
                data=SimpleNamespace(version=0),
            ),
            data_xyz_path=data_path,
            descriptor_path=tmp_path / "descriptor.out",
            cache_outputs_enabled=lambda: True,
        )

    first_result = result_data()
    scope, _indexed = resolve_audit_scope(first_result, AuditScopeKind.ACTIVE)
    fingerprints = build_fingerprints(first_result, scope)
    cache = TrainingSetEvidenceCache.from_fingerprints(
        first_result,
        dataset_fingerprint=fingerprints.dataset,
        scope_fingerprint=fingerprints.scope,
    )
    assert cache is not None
    atom_counts = tuple(len(structure) for structure in structures)
    assert cache.save_phase(_phase_inventory_for(("bcc", "fcc"), atom_counts))
    assert cache.save_magnetic(_magnetic_inventory_for(atom_counts))
    assert not cache.path_for(PHYSICS_SAMPLING_CACHE_KIND).exists()

    monkeypatch.setattr(
        sampling_plan_module,
        "build_physics_sampling_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy evidence should avoid reclassification")
        ),
    )
    migrated = build_result_physics_sampling_plan(
        first_result,
        (0, 1),
        spin_model=True,
    )

    assert migrated is not None
    assert {key.phase for key, _indices in migrated.groups} == {"bcc", "fcc"}
    assert cache.path_for(PHYSICS_SAMPLING_CACHE_KIND).is_file()

    cache.path_for("phase").unlink()
    cache.path_for("magnetic").unlink()
    reloaded = build_result_physics_sampling_plan(
        result_data(),
        (1,),
        spin_model=True,
    )

    assert reloaded is not None
    assert reloaded.source_indices == (1,)
    assert tuple(
        (key.phase, indices)
        for key, indices in reloaded.groups
    ) == (("fcc", (0,)),)


def test_result_plan_loads_v1_exact_composition_cache_and_rewrites_element_sets(
    tmp_path,
    monkeypatch,
):
    first = bulk("Fe", "fcc", a=3.55, cubic=True)
    first.set_chemical_symbols(["Fe", "Fe", "Ni", "Ni"])
    second = first.copy()
    second.set_chemical_symbols(["Fe", "Fe", "Fe", "Ni"])
    structures = [first, second]
    data_path = tmp_path / "train.xyz"
    data_path.write_text("stable cache identity\n", encoding="utf8")
    result = SimpleNamespace(
        structure=SimpleNamespace(
            all_data=structures,
            now_indices=np.arange(len(structures), dtype=np.int64),
            data=SimpleNamespace(version=0),
        ),
        data_xyz_path=data_path,
        descriptor_path=tmp_path / "descriptor.out",
        cache_outputs_enabled=lambda: True,
    )
    scope, _indexed = resolve_audit_scope(result, AuditScopeKind.ACTIVE)
    fingerprints = build_fingerprints(result, scope)
    cache = TrainingSetEvidenceCache.from_fingerprints(
        result,
        dataset_fingerprint=fingerprints.dataset,
        scope_fingerprint=fingerprints.scope,
    )
    assert cache is not None
    current_identity = sampling_plan_module._sampling_cache_identity(
        spin_model=False,
        source_structure_count=2,
    )
    legacy_identity = {
        **current_identity,
        "sampling_schema_version": (
            sampling_plan_module.LEGACY_PHYSICS_SAMPLING_SCHEMA_VERSION
        ),
    }
    assert cache.save_sampling_partitions(
        (
            {
                "source_index": 0,
                "composition": [["Fe", 1], ["Ni", 1]],
                "phase": "fcc",
                "magnetic_order": "not_applicable",
                "missing_spin": False,
            },
            {
                "source_index": 1,
                "composition": [["Fe", 3], ["Ni", 1]],
                "phase": "fcc",
                "magnetic_order": "not_applicable",
                "missing_spin": False,
            },
        ),
        identity=legacy_identity,
    )
    monkeypatch.setattr(
        sampling_plan_module,
        "build_physics_sampling_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the v1 partition cache should be migrated")
        ),
    )

    plan = build_result_physics_sampling_plan(result, (0, 1), spin_model=False)

    assert plan is not None
    assert plan.group_count == 1
    assert plan.groups[0][0].element_set == ("Fe", "Ni")
    migrated = cache.load_sampling_partitions(identity=current_identity)
    assert migrated is not None
    assert all(record["element_set"] == ["Fe", "Ni"] for record in migrated)
    assert all("composition" not in record for record in migrated)


def test_spin_plan_reports_missing_canonical_spin_without_using_force_labels():
    bcc = bulk("Fe", "bcc", a=2.86, cubic=True)
    bcc.arrays["force_mag"] = np.ones((len(bcc), 3))

    plan = build_physics_sampling_plan([bcc], spin_model=True)

    assert plan.missing_spin_indices == (0,)
    assert plan.magnetic_order_counts == (("no_spin", 1),)


def test_physics_quota_balances_element_set_then_phase_then_magnetic_order():
    fe = ("Fe",)
    feni = ("Fe", "Ni")
    sizes = {
        PhysicsSamplingStratum(fe, "bcc", "fm"): 100,
        PhysicsSamplingStratum(fe, "bcc", "pm_like"): 100,
        PhysicsSamplingStratum(fe, "fcc", "fm"): 100,
        PhysicsSamplingStratum(feni, "bcc", "fm"): 100,
    }

    quotas = allocate_physics_quotas(sizes, 40)

    assert sum(quotas.values()) == 40
    assert all(value >= 1 for value in quotas.values())
    per_element_set = {
        element_set: sum(
            value for key, value in quotas.items() if key.element_set == element_set
        )
        for element_set in (fe, feni)
    }
    assert abs(per_element_set[fe] - per_element_set[feni]) <= 2
    assert sum(value for key, value in quotas.items() if key.phase == "fcc") > 1


def test_physics_quota_rejects_budget_that_cannot_cover_every_stratum():
    keys = {
        PhysicsSamplingStratum(("Fe",), "bcc", "fm"): 10,
        PhysicsSamplingStratum(("Fe",), "fcc", "fm"): 10,
    }

    with pytest.raises(ValueError, match="2 element-set/phase/magnetic-order strata"):
        allocate_physics_quotas(keys, 1)
