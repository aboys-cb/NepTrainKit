"""Seed the packaged, versioned NEP-Data catalog."""

from .catalog import import_dataset_catalog, load_dataset_catalog
from .database import Database
from .services import ModelService, ProjectService


def seed_nep_data_git(
    db: Database,
    project_service: ProjectService,
    model_service: ModelService,
) -> None:
    """Import the packaged catalog only when creating a fresh database."""
    if not db.first:
        return
    catalog = load_dataset_catalog()
    import_dataset_catalog(
        catalog,
        project_service,
        model_service,
        preserve_catalog_ids=True,
    )
