from NepTrainKit.core.dataset.database import Database
from NepTrainKit.core.dataset.services import ModelService, ProjectService


def test_model_service_preserves_large_virial_values(tmp_path):
    database = Database(tmp_path / "models.db")
    project = ProjectService(database).create_project("audit")
    assert project is not None
    service = ModelService(database)

    large = service.add_version(
        project_id=project.id,
        name="large",
        model_type="NEP",
        model_path=tmp_path,
        data_size=1,
        energy=1.0,
        force=2.0,
        virial=12345.0,
    )
    models = {
        item.model_id: item
        for item in service.search_models_advanced(project_id=project.id)
    }
    assert models[large.id].virial == 12345.0
