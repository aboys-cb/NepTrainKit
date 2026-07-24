#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Versioned, data-only catalog boundary for built-in and external assets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from .services import ModelService, ProjectService


CATALOG_SCHEMA_VERSION = 1
DEFAULT_CATALOG_RESOURCE = "catalogs/nep_data.v1.json"


class CatalogValidationError(ValueError):
    """Raised when catalog data does not satisfy the supported schema."""


@dataclass(frozen=True)
class CatalogModel:
    catalog_id: int
    name: str
    model_type: str
    model_path: str
    data_size: int
    energy: float
    force: float
    virial: float
    tags: tuple[str, ...] = ()
    notes: str = ""
    parent_id: int | None = None


@dataclass(frozen=True)
class DatasetCatalog:
    schema_version: int
    catalog_id: str
    catalog_version: str
    source: Mapping[str, str]
    project_name: str
    project_notes: str
    models: tuple[CatalogModel, ...]


@dataclass(frozen=True)
class CatalogImportResult:
    project_id: int
    catalog_id: str
    catalog_version: str
    imported_models: int


def _required_text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = str(payload.get(key, "") or "").strip()
    if not value:
        raise CatalogValidationError(f"{location}.{key} must be a non-empty string")
    return value


def parse_dataset_catalog(payload: Mapping[str, Any]) -> DatasetCatalog:
    """Validate an untrusted JSON mapping and return an immutable catalog."""
    if not isinstance(payload, Mapping):
        raise CatalogValidationError("catalog root must be a JSON object")
    schema_version = int(payload.get("schema_version", -1))
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise CatalogValidationError(
            f"unsupported catalog schema_version={schema_version}; "
            f"expected {CATALOG_SCHEMA_VERSION}"
        )

    project = payload.get("project")
    source = payload.get("source")
    raw_models = payload.get("models")
    if not isinstance(project, Mapping):
        raise CatalogValidationError("project must be a JSON object")
    if not isinstance(source, Mapping):
        raise CatalogValidationError("source must be a JSON object")
    if not isinstance(raw_models, list):
        raise CatalogValidationError("models must be a JSON array")

    models: list[CatalogModel] = []
    seen_ids: set[int] = set()
    for index, raw_model in enumerate(raw_models):
        location = f"models[{index}]"
        if not isinstance(raw_model, Mapping):
            raise CatalogValidationError(f"{location} must be a JSON object")
        catalog_model_id = int(raw_model.get("id", -1))
        if catalog_model_id < 0 or catalog_model_id in seen_ids:
            raise CatalogValidationError(
                f"{location}.id must be a unique non-negative integer"
            )
        seen_ids.add(catalog_model_id)
        parent_raw = raw_model.get("parent_id")
        parent_id = None if parent_raw is None else int(parent_raw)
        raw_tags = raw_model.get("tags", [])
        if not isinstance(raw_tags, list):
            raise CatalogValidationError(f"{location}.tags must be a JSON array")
        models.append(
            CatalogModel(
                catalog_id=catalog_model_id,
                name=_required_text(raw_model, "name", location),
                model_type=_required_text(raw_model, "model_type", location),
                model_path=_required_text(raw_model, "model_path", location),
                data_size=int(raw_model.get("data_size", 0)),
                energy=float(raw_model.get("energy", 0.0)),
                force=float(raw_model.get("force", 0.0)),
                virial=float(raw_model.get("virial", 0.0)),
                tags=tuple(str(tag) for tag in raw_tags if str(tag)),
                notes=str(raw_model.get("notes", "") or ""),
                parent_id=parent_id,
            )
        )

    for model in models:
        if model.parent_id is not None and model.parent_id not in seen_ids:
            raise CatalogValidationError(
                f"model {model.catalog_id} references missing parent {model.parent_id}"
            )
        if model.parent_id == model.catalog_id:
            raise CatalogValidationError(
                f"model {model.catalog_id} cannot be its own parent"
            )

    return DatasetCatalog(
        schema_version=schema_version,
        catalog_id=_required_text(payload, "catalog_id", "catalog"),
        catalog_version=_required_text(payload, "catalog_version", "catalog"),
        source={str(key): str(value) for key, value in source.items()},
        project_name=_required_text(project, "name", "project"),
        project_notes=str(project.get("notes", "") or ""),
        models=tuple(models),
    )


def load_dataset_catalog(path: str | Path | None = None) -> DatasetCatalog:
    """Load the packaged catalog or an explicitly selected external JSON file."""
    if path is None:
        resource = files("NepTrainKit.core.dataset").joinpath(DEFAULT_CATALOG_RESOURCE)
        text = resource.read_text(encoding="utf-8")
    else:
        text = Path(path).expanduser().read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogValidationError(f"invalid catalog JSON: {exc}") from exc
    return parse_dataset_catalog(payload)


def import_dataset_catalog(
    catalog: DatasetCatalog,
    project_service: ProjectService,
    model_service: ModelService,
    *,
    preserve_catalog_ids: bool = False,
) -> CatalogImportResult:
    """Import a validated catalog while remapping its parent references."""
    provenance = (
        f"Catalog {catalog.catalog_id}@{catalog.catalog_version}. "
        f"{catalog.project_notes}"
    ).strip()
    project = project_service.create_project(catalog.project_name, provenance)
    if project is None:
        raise ValueError(f"Project already exists or could not be created: {catalog.project_name}")

    pending = list(catalog.models)
    imported_ids: dict[int, int] = {}
    while pending:
        next_pending: list[CatalogModel] = []
        made_progress = False
        for model in pending:
            if model.parent_id is not None and model.parent_id not in imported_ids:
                next_pending.append(model)
                continue
            created = model_service.add_version(
                project_id=project.id,
                name=model.name,
                model_type=model.model_type,
                model_path=model.model_path,
                data_size=model.data_size,
                energy=model.energy,
                force=model.force,
                virial=model.virial,
                tags=list(model.tags),
                notes=model.notes,
                parent_id=(
                    imported_ids[model.parent_id]
                    if model.parent_id is not None
                    else None
                ),
                id=model.catalog_id if preserve_catalog_ids else None,
            )
            imported_ids[model.catalog_id] = int(created.id)
            made_progress = True
        if not made_progress:
            unresolved = ", ".join(str(model.catalog_id) for model in next_pending)
            raise CatalogValidationError(
                f"catalog parent graph contains a cycle: {unresolved}"
            )
        pending = next_pending

    return CatalogImportResult(
        project_id=int(project.id),
        catalog_id=catalog.catalog_id,
        catalog_version=catalog.catalog_version,
        imported_models=len(imported_ids),
    )
