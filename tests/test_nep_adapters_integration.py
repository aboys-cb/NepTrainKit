import json
import shutil
from pathlib import Path

from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.io.nep import NepTrainResultData


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
