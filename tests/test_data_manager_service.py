import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
from NepTrainKit.core.dataset.database import Database
from NepTrainKit.core.dataset.services import ProjectService,ModelService

 


def test_dataset_and_model_services(tmp_path):
    db_path = tmp_path / "mlpman.db"
    db = Database(db_path)
    ds_service = DatasetService(db)
    model_service = ModelService(db)
    lineage = LineageService(db)

    data_file = tmp_path / "data.txt"
    data_file.write_text("sample data")

    dataset = ds_service.create_dataset("dataset1", "")
    version = ds_service.register_data_file(dataset.id, str(data_file))
    assert version.id is not None
    versions = ds_service.get_versions(dataset.id)
    assert len(versions) == 1

    model_file = tmp_path / "model.txt"
    model_file.write_text("model")

    family = model_service.create_family("family1", "NEP")
    model_version = model_service.register_model_file(
        family.id, str(model_file), [(version.id, "train")]
    )
    assert model_version.id is not None

    data_chain = lineage.data_lineage(version.id)
    assert data_chain and data_chain[0].id == version.id

    model_chain = lineage.model_lineage(model_version.id)
    assert model_chain and model_chain[0].id == model_version.id
