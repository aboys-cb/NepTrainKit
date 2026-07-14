from types import SimpleNamespace

import numpy as np

from NepTrainKit.core.audit import (
    AuditContext,
    AuditScopeKind,
    build_audit,
    build_fingerprints,
    resolve_audit_scope,
)
from NepTrainKit.core.structure import Structure


def _structure(x: float, element: str = "Fe") -> Structure:
    return Structure(
        np.eye(3) * 5.0,
        {
            "species": np.asarray([element]),
            "pos": np.asarray([[x, 0.0, 0.0]], dtype=np.float64),
        },
        [
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
        ],
        {"pbc": "T T T", "Config_type": "bulk"},
    )


def _dataset(tmp_path=None):
    structures = np.asarray(
        [
            _structure(0.5),
            _structure(1.0),
            _structure(1.5, "Ni"),
            _structure(2.0),
            _structure(2.5),
        ],
        dtype=object,
    )
    for index, structure in enumerate(structures):
        structure.energy = -float(index + 1)
    model = None
    if tmp_path is not None:
        model = tmp_path / "nep.txt"
        model.write_text("nep4_zbl 2 Fe Ni\ncutoff 6 4\n", encoding="utf-8")
    return SimpleNamespace(
        structure=SimpleNamespace(
            all_data=structures,
            now_indices=np.asarray([0, 2, 3, 4], dtype=np.int32),
        ),
        select_index={1, 2},
        nep_txt_path=model,
    )


def test_scope_resolution_preserves_original_indices():
    dataset = _dataset()

    all_scope, _ = resolve_audit_scope(dataset, AuditScopeKind.ALL)
    active_scope, _ = resolve_audit_scope(dataset, AuditScopeKind.ACTIVE)
    selected_scope, _ = resolve_audit_scope(dataset, AuditScopeKind.SELECTED)
    custom_scope, _ = resolve_audit_scope(dataset, AuditScopeKind.CUSTOM, (1, 99))

    assert all_scope.indices == (0, 1, 2, 3, 4)
    assert active_scope.indices == (0, 2, 3, 4)
    assert selected_scope.indices == (2,)
    assert custom_scope.indices == (1,)
    assert all_scope.source_count == active_scope.source_count == 5


def test_build_audit_records_scope_ruleset_and_canonical_findings():
    run = build_audit(
        AuditContext(
            dataset=_dataset(),
            dataset_id="fixture.xyz",
            scope_kind=AuditScopeKind.ACTIVE,
            ruleset_version="test-rules-v1",
        )
    )

    assert run.scope is not None
    assert run.scope.kind is AuditScopeKind.ACTIVE
    assert run.scope.indices == (0, 2, 3, 4)
    assert run.inputs["source_structure_count"] == 5
    assert run.ruleset_version == "test-rules-v1"
    assert run.fingerprints.dataset
    assert run.fingerprints.scope
    assert run.findings
    assert "label_ranges:energy_high_tail" in {finding.id for finding in run.findings}
    assert all(finding.evidence_ids for finding in run.findings)


def test_fingerprints_change_with_scope_data_and_model(tmp_path):
    dataset = _dataset(tmp_path)
    active_scope, _ = resolve_audit_scope(dataset, AuditScopeKind.ACTIVE)
    all_scope, _ = resolve_audit_scope(dataset, AuditScopeKind.ALL)

    initial = build_fingerprints(dataset, active_scope)
    assert initial.scope != build_fingerprints(dataset, all_scope).scope

    dataset.structure.all_data[0].atomic_properties["pos"][0, 0] = 0.75
    changed_data = build_fingerprints(dataset, active_scope)
    assert changed_data.dataset != initial.dataset

    dataset.nep_txt_path.write_text("nep4_zbl 2 Fe Ni\ncutoff 7 5\n", encoding="utf-8")
    changed_model = build_fingerprints(dataset, active_scope)
    assert changed_model.model != changed_data.model


def test_file_backed_versioned_dataset_fingerprint_tracks_source_and_version(tmp_path):
    dataset = _dataset(tmp_path)
    source = tmp_path / "train.xyz"
    source.write_text("source-v1\n", encoding="utf-8")
    dataset.data_xyz_path = source
    dataset.structure.data = SimpleNamespace(version=0)
    scope, _ = resolve_audit_scope(dataset, AuditScopeKind.ACTIVE)

    initial = build_fingerprints(dataset, scope)
    dataset.structure.data.version = 1
    version_changed = build_fingerprints(dataset, scope)
    assert version_changed.dataset != initial.dataset

    source.write_text("source-v2\n", encoding="utf-8")
    source_changed = build_fingerprints(dataset, scope)
    assert source_changed.dataset != version_changed.dataset
