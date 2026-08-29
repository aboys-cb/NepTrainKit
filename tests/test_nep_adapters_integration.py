import json
import shutil
from pathlib import Path

import numpy as np
from ase.io import read as ase_read

from NepTrainKit.core import MessageManager
from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.io.importers import ase_atoms_to_structure
from NepTrainKit.core.io.nep import NepTrainResultData
from NepTrainKit.core.structure import Structure


def _labeled_structure(energy: float, forces: list[list[float]]) -> Structure:
    natoms = len(forces)
    return Structure(
        lattice=np.eye(3),
        atomic_properties={
            "species": ["Fe"] * natoms,
            "pos": np.zeros((natoms, 3)),
            "forces": np.asarray(forces, dtype=np.float64),
        },
        properties=[
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
            {"name": "forces", "type": "R", "count": 3},
        ],
        additional_fields={"energy": energy, "pbc": "T T T"},
    )


def _alignment_result(tmp_path: Path, structures: list[Structure]) -> NepTrainResultData:
    result = NepTrainResultData(
        tmp_path / "nep.txt",
        tmp_path / "train.xyz",
        tmp_path / "energy_train.out",
        tmp_path / "force_train.out",
        tmp_path / "stress_train.out",
        tmp_path / "virial_train.out",
        tmp_path / "descriptor.out",
        charge_model=False,
        spin_model=False,
    )
    result.set_structures(structures)
    result.load_structures()
    result.nep_calc = None
    return result


def test_cached_output_reference_check_allows_normal_text_precision_loss(tmp_path: Path):
    structure = _labeled_structure(
        -3408.80019046875,
        [[0.123456789, -0.00000012, 2.34567891]],
    )
    result = _alignment_result(tmp_path, [structure])
    energy = np.array([[-8.15, -3408.80029297]])
    force = np.array([[9.0, 8.0, 7.0, 0.12345679, -0.0000001, 2.3456789]])

    assert result._cached_output_alignment_error(energy, force) is None


def test_cached_output_reference_check_detects_same_size_reordering(tmp_path: Path):
    structures = [
        _labeled_structure(1.0, [[1.0, 2.0, 3.0]]),
        _labeled_structure(2.0, [[4.0, 5.0, 6.0]]),
    ]
    result = _alignment_result(tmp_path, structures)
    force = np.array(
        [
            [10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
            [40.0, 50.0, 60.0, 4.0, 5.0, 6.0],
        ]
    )

    energy_error = result._cached_output_alignment_error(
        np.array([[0.0, 2.0], [0.0, 1.0]]),
        force,
    )
    force_error = result._cached_output_alignment_error(
        np.array([[0.0, 1.0], [0.0, 2.0]]),
        force[::-1],
    )

    assert energy_error is not None and "energy_train.out" in energy_error
    assert force_error is not None and "force_train.out" in force_error


def test_cached_output_mismatch_activates_recalculation(tmp_path: Path, monkeypatch):
    structures = [
        _labeled_structure(1.0, [[1.0, 2.0, 3.0]]),
        _labeled_structure(2.0, [[4.0, 5.0, 6.0]]),
    ]
    result = _alignment_result(tmp_path, structures)
    result.energy_out_path.write_text("0 2\n0 1\n", encoding="utf-8")
    result.force_out_path.write_text(
        "0 0 0 1 2 3\n0 0 0 4 5 6\n", encoding="utf-8"
    )
    result.nep_txt_path.write_text("nep4 1 Fe\n", encoding="utf-8")
    fake_calculator = object()
    warnings: list[str] = []
    monkeypatch.setattr("NepTrainKit.core.io.nep.NepCalculator", lambda **_kwargs: fake_calculator)
    monkeypatch.setattr(MessageManager, "send_warning_message", warnings.append)

    result._prepare_cached_output_alignment()

    assert result._force_recalculate_outputs is True
    assert result._should_recalculate({}) is True
    assert result.nep_calc is fake_calculator
    assert result._cached_descriptors_are_usable() is False
    assert warnings and "will recalculate" in warnings[0]


def test_prediction_manifest_detects_same_dataset_with_different_model(tmp_path: Path):
    structure = _labeled_structure(1.0, [[1.0, 2.0, 3.0]])
    result = _alignment_result(tmp_path, [structure])
    result.data_xyz_path.write_text("dataset", encoding="utf-8")
    result.nep_txt_path.write_text("current model", encoding="utf-8")
    result.prediction_meta_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictions": {
                    "train.xyz": {
                        "dataset": {
                            "sha256": result._sha256(result.data_xyz_path),
                            "structures": 1,
                            "atoms": 1,
                        },
                        "model": {"sha256": "another-model"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        result._prediction_manifest_alignment_error()
        == "prediction.meta.json records a different NEP model"
    )


def test_result_writes_official_outputs_and_prediction_manifest(
    tmp_path: Path, monkeypatch
):
    fixture = Path(__file__).parent / "data" / "nep"
    model = tmp_path / "nep.txt"
    dataset = tmp_path / "train.xyz"
    shutil.copy2(fixture / "nep.txt", model)
    shutil.copy2(fixture / "train.xyz", dataset)

    result = NepTrainResultData.from_path(dataset, nep_txt_path=model)
    result.load_structures()
    result.nep_calc = NepCalculator(model, backend="cpu", chunk_max_atoms=500)
    monkeypatch.setattr(result, "cache_outputs_enabled", lambda: True)
    statuses = []
    result.predictionStatusSignal.connect(statuses.append)

    calls = {"combined": 0, "predict": 0}
    combined = result.nep_calc.predict_with_descriptors
    predict = result.nep_calc.predict

    def counted_combined(*args, **kwargs):
        calls["combined"] += 1
        return combined(*args, **kwargs)

    def counted_predict(*args, **kwargs):
        calls["predict"] += 1
        return predict(*args, **kwargs)

    monkeypatch.setattr(result.nep_calc, "predict_with_descriptors", counted_combined)
    monkeypatch.setattr(result.nep_calc, "predict", counted_predict)

    result._load_descriptors()
    result._load_dataset()

    assert calls == {"combined": 1, "predict": 0}
    assert any("together" in status for status in statuses)

    for name in (
        "energy_train.out",
        "force_train.out",
        "stress_train.out",
        "virial_train.out",
        "descriptor.out",
    ):
        assert (tmp_path / name).is_file()
    metadata = json.loads((tmp_path / "prediction.meta.json").read_text())
    assert metadata["producer"] == "NepTrainKit"
    record = metadata["predictions"]["train.xyz"]
    assert record["model"]["sha256"] == result.nep_calc.model_info.sha256
    assert record["backend"] == {
        "requested": "cpu",
        "resolved": "cpu",
        "reason": "cpu_requested",
    }
    assert record["chunk_max_atoms"] == 500
    assert record["dataset"]["structures"] == len(result.atoms_num_list)


def test_transient_result_keeps_predictions_in_memory_without_cache_files(
    tmp_path: Path,
):
    fixture = Path(__file__).parent / "data" / "nep"
    model = tmp_path / "nep89.txt"
    dataset = tmp_path / "make_dataset.xyz"
    shutil.copy2(fixture / "nep.txt", model)
    structures = [
        ase_atoms_to_structure(atoms)
        for atoms in ase_read(fixture / "train.xyz", index=":")
    ]

    result = NepTrainResultData.from_path(
        dataset,
        structures=structures,
        nep_txt_path=model,
        cache_outputs=False,
    )
    assert not dataset.exists()
    result.load_structures()
    result.nep_calc = NepCalculator(model, backend="cpu", chunk_max_atoms=500)

    result._load_descriptors()
    result._load_dataset()

    assert result.energy.now_data.size > 0
    assert result.force.now_data.size > 0
    output_directory = tmp_path / "make_dataset_nep89"
    assert not output_directory.exists()


def test_partial_external_outputs_are_not_overwritten(tmp_path: Path, monkeypatch):
    result = NepTrainResultData(
        tmp_path / "nep.txt",
        tmp_path / "train.xyz",
        tmp_path / "energy_train.out",
        tmp_path / "force_train.out",
        tmp_path / "stress_train.out",
        tmp_path / "virial_train.out",
        tmp_path / "descriptor.out",
        charge_model=False,
        spin_model=False,
    )
    result.energy_out_path.write_text("1 1\n", encoding="utf-8")
    monkeypatch.setattr(result, "cache_outputs_enabled", lambda: True)

    try:
        result._should_recalculate({})
    except FileExistsError as error:
        assert "will not overwrite" in str(error)
    else:
        raise AssertionError("partial external outputs must be protected")
