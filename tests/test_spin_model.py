from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.io import NepPlotData, NepTrainResultData
from NepTrainKit.core.utils import is_spin_model


@pytest.fixture()
def local_tmp_path() -> Path:
    base_tmp = Path(__file__).resolve().parents[1] / ".tmp_localappdata" / "spin_model_tmp"
    base_tmp.mkdir(parents=True, exist_ok=True)
    tmp_path = base_tmp / f"spin_model_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=False)
    try:
        yield tmp_path
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_is_spin_model_header(local_tmp_path: Path):
    p = local_tmp_path / "nep.txt"
    p.write_text("nep4_spin 1 Fe\ncutoff 8 4 10 10\n", encoding="utf-8")
    assert is_spin_model(p) is True


def test_is_spin_model_spin_mode_line(local_tmp_path: Path):
    p = local_tmp_path / "nep.txt"
    p.write_text("nep4 1 Fe\nspin_mode 1\ncutoff 8 4 10 10\n", encoding="utf-8")
    assert is_spin_model(p) is True


def test_is_spin_model_false(local_tmp_path: Path):
    p = local_tmp_path / "nep.txt"
    p.write_text("nep4 1 Fe\ncutoff 8 4 10 10\n", encoding="utf-8")
    assert is_spin_model(p) is False


def test_spin_result_keeps_detected_model_and_does_not_require_stress(local_tmp_path: Path):
    train = local_tmp_path / "train.xyz"
    train.write_text("", encoding="utf-8")
    model = local_tmp_path / "nep.txt"
    model.write_text("nep4_spin 1 Fe\nspin_mode 1\n", encoding="utf-8")
    for name, text in {
        "energy_train.out": "1.0 1.0\n",
        "force_train.out": "0.0 0.0 0.0 0.0 0.0 0.0\n",
        "virial_train.out": "0 0 0 0 0 0 0 0 0 0 0 0\n",
        "mforce_train.out": "0.0 0.0 0.0 0.0 0.0 0.0\n",
    }.items():
        (local_tmp_path / name).write_text(text, encoding="utf-8")

    result = NepTrainResultData.from_path(train, model_type=9)
    result.atoms_num_list = np.array([1])

    assert result.nep_txt_path == model
    assert result.is_spin_model is True
    assert result.spin_force_out_path == local_tmp_path / "mforce_train.out"
    assert result._should_recalculate({"prediction": "1"}) is False


def test_spin_result_datasets_hide_stress(local_tmp_path: Path):
    result = NepTrainResultData(
        local_tmp_path / "nep.txt",
        local_tmp_path / "train.xyz",
        local_tmp_path / "energy_train.out",
        local_tmp_path / "force_train.out",
        local_tmp_path / "stress_train.out",
        local_tmp_path / "virial_train.out",
        local_tmp_path / "descriptor.out",
        spin_model=True,
    )
    result._energy_dataset = NepPlotData([[0.0, 0.0]], title="energy")
    result._force_dataset = NepPlotData([[0.0, 0.0]], title="force")
    result._stress_dataset = NepPlotData([[0.0, 0.0]], title="stress")
    result._virial_dataset = NepPlotData([[0.0, 0.0]], title="virial")
    result._spin_force_dataset = NepPlotData([[0.0, 0.0]], title="mforce")
    result._descriptor_dataset = NepPlotData([[0.0]], title="descriptor")

    assert [dataset.title for dataset in result.datasets] == [
        "energy",
        "force",
        "virial",
        "mforce",
        "descriptor",
    ]


def test_spin_calculator_does_not_load_native_backend(local_tmp_path: Path, monkeypatch):
    model = local_tmp_path / "nep.txt"
    model.write_text("nep4_spin 1 Fe\nspin_mode 1\n", encoding="utf-8")

    def fail_load_nep(_self):
        raise AssertionError("spin calculator must not load native NEP backend")

    monkeypatch.setattr(NepCalculator, "load_nep", fail_load_nep)

    calc = NepCalculator(model)

    assert calc.is_spin_model is True
    assert calc.initialized is False
    assert calc.nep3 is None


def test_spin_calculator_descriptor_returns_empty(local_tmp_path: Path):
    model = local_tmp_path / "nep.txt"
    model.write_text("nep4_spin 1 Fe\nspin_mode 1\n", encoding="utf-8")

    calc = NepCalculator(model)

    assert calc.get_structures_descriptor([]).size == 0
