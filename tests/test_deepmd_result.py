from pathlib import Path

import numpy as np

from NepTrainKit.core.io import StructureData
from NepTrainKit.core.io.deepmd import DeepmdResultData
from NepTrainKit.core.structure import Structure


def _write_deepmd_outputs(root: Path) -> None:
    (root / "data" / "case").mkdir(parents=True)
    (root / "data" / "case" / "type.raw").write_text("0\n", encoding="utf-8")
    for name in (
        "test_result.e_peratom.out",
        "test_result.fr.out",
        "test_result.f.out",
        "test_result.v_peratom.out",
        "test_result.fm.out",
    ):
        (root / name).write_text("0 0\n", encoding="utf-8")


def test_deepmd_root_uses_existing_prediction_outputs(tmp_path: Path):
    _write_deepmd_outputs(tmp_path)

    result = DeepmdResultData.from_path(tmp_path)

    assert result._cached_outputs_only is True
    assert result.energy_out_path == tmp_path / "test_result.e_peratom.out"
    assert result.force_out_path == tmp_path / "test_result.fr.out"
    assert result.virial_out_path == tmp_path / "test_result.v_peratom.out"
    assert result.spin_out_path == tmp_path / "test_result.fm.out"
    assert not (tmp_path / "data_nep89").exists()


def test_deepmd_data_subdir_uses_parent_prediction_outputs(tmp_path: Path):
    _write_deepmd_outputs(tmp_path)

    result = DeepmdResultData.from_path(tmp_path / "data")

    assert result._cached_outputs_only is True
    assert result.energy_out_path == tmp_path / "test_result.e_peratom.out"
    assert result.force_out_path == tmp_path / "test_result.fr.out"
    assert result.virial_out_path == tmp_path / "test_result.v_peratom.out"
    assert result.spin_out_path == tmp_path / "test_result.fm.out"
    assert not (tmp_path / "data_nep89").exists()


def _structure(natoms: int, force_mag: np.ndarray | None = None) -> Structure:
    props = {
        "species": np.array(["Fe"] * natoms, dtype=object),
        "pos": np.zeros((natoms, 3), dtype=np.float32),
    }
    properties = [
        {"name": "species", "type": "S", "count": 1},
        {"name": "pos", "type": "R", "count": 3},
    ]
    if force_mag is not None:
        props["force_mag"] = np.asarray(force_mag, dtype=np.float32).reshape(natoms, 3)
        properties.append({"name": "force_mag", "type": "R", "count": 3})
    return Structure(
        lattice=np.eye(3, dtype=np.float32),
        atomic_properties=props,
        properties=properties,
        additional_fields={"energy": 0.0},
    )


def test_deepmd_fm_output_falls_back_to_atom_grouping_without_force_mag(tmp_path: Path):
    for name, text in {
        "test_result.e_peratom.out": "0 0\n",
        "test_result.fr.out": "0 0 0 0 0 0\n0 0 0 0 0 0\n",
        "test_result.v_peratom.out": "0 0 0 0 0 0 0 0 0 0 0 0\n",
        "test_result.fm.out": "0 0 0 0 0 0\n0 0 0 0 0 0\n",
    }.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    result = DeepmdResultData(
        tmp_path / "nep.txt",
        tmp_path,
        tmp_path / "test_result.e_peratom.out",
        tmp_path / "test_result.fr.out",
        tmp_path / "test_result.v_peratom.out",
        tmp_path / "descriptor.out",
        spin_out_path=tmp_path / "test_result.fm.out",
        cached_outputs_only=True,
    )
    result._atoms_dataset = StructureData([_structure(2)])
    result.atoms_num_list = np.array([2], dtype=np.int64)

    result._load_dataset()

    assert result.mforce.title == "mforce"
    assert result.mforce.group_array.all_data.tolist() == [0, 0]


def test_deepmd_cached_outputs_can_generate_missing_descriptor(tmp_path: Path):
    result = DeepmdResultData(
        tmp_path / "nep.txt",
        tmp_path,
        tmp_path / "test_result.e_peratom.out",
        tmp_path / "test_result.fr.out",
        tmp_path / "test_result.v_peratom.out",
        tmp_path / "descriptor.out",
        cached_outputs_only=True,
    )
    result._atoms_dataset = StructureData([_structure(1)])
    result.atoms_num_list = np.array([1], dtype=np.int64)

    class FakeCalculator:
        def get_structures_descriptor(self, structures):
            assert len(structures) == 1
            return np.array([[1.0, 2.0]], dtype=np.float32)

    result.nep_calc = FakeCalculator()

    result._load_descriptors()

    np.testing.assert_allclose(result.descriptor.now_data, np.array([[1.0, 2.0]], dtype=np.float32))


def test_deepmd_cached_outputs_read_predictions_while_generating_descriptor(tmp_path: Path):
    for name, text in {
        "test_result.e_peratom.out": "2.0 20.0\n",
        "test_result.fr.out": "1 2 3 10 20 30\n",
        "test_result.v_peratom.out": "1 2 3 4 5 6 10 20 30 40 50 60\n",
    }.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    result = DeepmdResultData(
        tmp_path / "nep.txt",
        tmp_path,
        tmp_path / "test_result.e_peratom.out",
        tmp_path / "test_result.fr.out",
        tmp_path / "test_result.v_peratom.out",
        tmp_path / "descriptor.out",
        cached_outputs_only=True,
    )
    result.set_structures([_structure(1)])
    result.load_structures()

    class FakeCalculator:
        def calculate(self, _structures):
            raise AssertionError("cached DeepMD outputs must not be recalculated")

        def get_structures_descriptor(self, structures):
            assert len(structures) == 1
            return np.array([[7.0, 8.0]], dtype=np.float32)

    result.nep_calc = FakeCalculator()

    result._load_descriptors()
    result._load_dataset()

    np.testing.assert_allclose(result.descriptor.all_data, [[7.0, 8.0]])
    np.testing.assert_allclose(result.energy.all_data, [[2.0, 20.0]])
    np.testing.assert_allclose(result.force.all_data, [[1, 2, 3, 10, 20, 30]])
    np.testing.assert_allclose(result.virial.all_data, [[1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60]])
