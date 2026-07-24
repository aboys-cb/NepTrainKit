from __future__ import annotations

import json

import pytest
from sqlalchemy import func, select

from NepTrainKit.core.dataset.catalog import (
    CatalogValidationError,
    import_dataset_catalog,
    load_dataset_catalog,
    parse_dataset_catalog,
)
from NepTrainKit.core.dataset.database import Database
from NepTrainKit.core.dataset.models import ModelVersion
from NepTrainKit.core.dataset.services import ModelService, ProjectService


def test_packaged_catalog_is_versioned_and_parent_complete():
    catalog = load_dataset_catalog()

    assert catalog.schema_version == 1
    assert catalog.catalog_id == "nep-data"
    assert catalog.catalog_version
    assert len(catalog.models) == 55
    model_ids = {model.catalog_id for model in catalog.models}
    assert all(
        model.parent_id is None or model.parent_id in model_ids
        for model in catalog.models
    )


def test_external_catalog_rejects_unsupported_schema(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({"schema_version": 99, "models": []}),
        encoding="utf-8",
    )

    with pytest.raises(CatalogValidationError, match="schema_version=99"):
        load_dataset_catalog(path)


def test_catalog_import_remaps_parent_ids(tmp_path):
    catalog = parse_dataset_catalog(
        {
            "schema_version": 1,
            "catalog_id": "test-assets",
            "catalog_version": "1.0.0",
            "source": {"url": "https://example.invalid/catalog"},
            "project": {"name": "Imported assets", "notes": "test"},
            "models": [
                {
                    "id": 10,
                    "name": "parent",
                    "model_type": "NEP",
                    "model_path": "https://example.invalid/parent",
                    "data_size": 1,
                    "energy": 0,
                    "force": 0,
                    "virial": 0,
                },
                {
                    "id": 20,
                    "parent_id": 10,
                    "name": "child",
                    "model_type": "NEP",
                    "model_path": "https://example.invalid/child",
                    "data_size": 1,
                    "energy": 0,
                    "force": 0,
                    "virial": 0,
                },
            ],
        }
    )
    database = Database(tmp_path / "catalog.db")
    result = import_dataset_catalog(
        catalog,
        ProjectService(database),
        ModelService(database),
    )

    assert result.imported_models == 2
    with database.session() as session:
        models = session.scalars(
            select(ModelVersion).order_by(ModelVersion.name)
        ).all()
    by_name = {model.name: model for model in models}
    assert by_name["child"].parent_id == by_name["parent"].id
    assert by_name["parent"].id not in {10, 20}


def test_packaged_catalog_seeds_all_models(tmp_path):
    database = Database(tmp_path / "seed.db")
    catalog = load_dataset_catalog()
    result = import_dataset_catalog(
        catalog,
        ProjectService(database),
        ModelService(database),
        preserve_catalog_ids=True,
    )

    with database.session() as session:
        model_count = session.scalar(select(func.count(ModelVersion.id)))
    assert result.imported_models == len(catalog.models)
    assert model_count == len(catalog.models)
