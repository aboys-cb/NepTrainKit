"""Resolve audit scopes and stable input fingerprints."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from NepTrainKit.core.structure import Structure

from .result import AuditFingerprints, AuditScope, AuditScopeKind


def _structure_container(dataset: Any) -> Any:
    return getattr(dataset, "structure", dataset)


def _all_structures(dataset: Any) -> list[Structure]:
    container = _structure_container(dataset)
    all_data = getattr(container, "all_data", None)
    if all_data is not None:
        return list(all_data)
    now_data = getattr(container, "now_data", None)
    if now_data is not None:
        return list(now_data)
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        return list(container)
    return []


def resolve_audit_scope(
    dataset: Any,
    kind: AuditScopeKind = AuditScopeKind.ACTIVE,
    indices: Sequence[int] = (),
) -> tuple[AuditScope, list[tuple[int, Structure]]]:
    """Return an explicit scope plus structures indexed in the source dataset."""
    kind = AuditScopeKind(kind)
    structures = _all_structures(dataset)
    source_count = len(structures)
    valid = set(range(source_count))
    container = _structure_container(dataset)

    if kind == AuditScopeKind.ALL:
        chosen = tuple(range(source_count))
    elif kind == AuditScopeKind.SELECTED:
        active = {
            int(index)
            for index in getattr(container, "now_indices", range(source_count))
            if int(index) in valid
        }
        selected = {
            int(index)
            for index in getattr(dataset, "select_index", ())
            if int(index) in valid
        }
        chosen = tuple(sorted(active.intersection(selected)))
    elif kind == AuditScopeKind.CUSTOM:
        chosen = tuple(sorted({int(index) for index in indices if int(index) in valid}))
    else:
        chosen = tuple(
            int(index)
            for index in getattr(container, "now_indices", range(source_count))
            if int(index) in valid
        )

    scope = AuditScope(kind=kind, indices=chosen, source_count=source_count)
    return scope, [(index, structures[index]) for index in chosen]


def _update_hash(digest: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        if array.dtype.kind in {"O", "U", "S"}:
            for item in array.reshape(-1):
                _update_hash(digest, str(item))
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            _update_hash(digest, str(key))
            _update_hash(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _update_hash(digest, item)
        return
    if isinstance(value, (np.generic,)):
        value = value.item()
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(str(value).encode("utf-8", errors="replace"))


def fingerprint_structures(structures: Sequence[Structure]) -> str:
    """Hash structure content used by data-quality and staleness checks."""
    digest = hashlib.sha256()
    for structure in structures:
        _update_hash(digest, getattr(structure, "lattice", ()))
        _update_hash(digest, getattr(structure, "atomic_properties", {}))
        _update_hash(digest, getattr(structure, "additional_fields", {}))
    return digest.hexdigest()


def _fingerprint_versioned_source(dataset: Any) -> str | None:
    """Hash a file-backed dataset without walking every materialized Structure.

    NepTrainKit's StructureData owns a monotonic mutation version.  Combining
    that version with the source-file content preserves the staleness boundary
    while avoiding a second Python traversal of large datasets.  Generic audit
    callers without this contract continue to use ``fingerprint_structures``.
    """
    container = _structure_container(dataset)
    version = getattr(getattr(container, "data", None), "version", None)
    source_path = getattr(dataset, "data_xyz_path", None)
    if version is None or source_path is None or str(source_path).strip() == "":
        return None
    target = Path(source_path)
    if not target.is_file():
        return None

    digest = hashlib.sha256()
    _update_hash(digest, "versioned-source-v1")
    _update_hash(digest, fingerprint_file(target))
    _update_hash(digest, int(version))
    return digest.hexdigest()


def fingerprint_scope(scope: AuditScope) -> str:
    digest = hashlib.sha256()
    _update_hash(digest, scope.kind.value)
    _update_hash(digest, scope.source_count)
    _update_hash(digest, scope.indices)
    return digest.hexdigest()


def fingerprint_file(path: Any) -> str:
    if path is None or str(path).strip() == "":
        return ""
    target = Path(path)
    if not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_fingerprints(dataset: Any, scope: AuditScope) -> AuditFingerprints:
    all_structures = _all_structures(dataset)
    return AuditFingerprints(
        dataset=_fingerprint_versioned_source(dataset) or fingerprint_structures(all_structures),
        scope=fingerprint_scope(scope),
        model=fingerprint_file(getattr(dataset, "nep_txt_path", None)),
    )
