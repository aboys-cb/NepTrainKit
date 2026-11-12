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
import pytest



@pytest.mark.parametrize("parent", ["self", "child", "grandchild", "missing"])
def test_project_rejects_invalid_parent_without_changing_tree(tmp_path, parent):
    service = ProjectService(Database(tmp_path / "projects.db"))
    root = service.create_project("root")
    child = service.create_project("child", parent_id=root.id)
    grandchild = service.create_project("grandchild", parent_id=child.id)
    parent_id = {"self": root.id, "child": child.id, "grandchild": grandchild.id, "missing": 999}[parent]
    with pytest.raises(ValueError):
        service.modify_project(root.id, "changed", parent_id=parent_id)
    tree = service.search_projects(parent_id=None)
    assert [(node.project_id, node.name) for node in tree] == [(root.id, "root")]
    assert tree[0].children[0].children[0].project_id == grandchild.id


def test_project_can_move_to_another_branch_and_back_to_root(tmp_path):
    service = ProjectService(Database(tmp_path / "projects.db"))
    first = service.create_project("first")
    second = service.create_project("second")
    child = service.create_project("child", parent_id=first.id)
    service.modify_project(child.id, "moved", parent_id=second.id)
    assert service.get_project_by_id(first.id).children == []
    assert service.get_project_by_id(second.id).children[0].name == "moved"
    service.modify_project(child.id, "moved")
    assert len(service.search_projects(parent_id=None)) == 3
