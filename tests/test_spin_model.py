from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
from nep_adapters import ModelInfo, SpinPrediction
from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.io import NepPlotData, NepTrainResultData
from NepTrainKit.core.structure import Structure


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


def test_spin_result_keeps_detected_model_and_does_not_require_stress(
    local_tmp_path: Path, monkeypatch
):
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
    monkeypatch.setattr(
        "NepTrainKit.core.io.nep.inspect_model",
        lambda _path: type("Info", (), {"model_type": "spin"})(),
    )
    result = NepTrainResultData.from_path(train, model_type=9)
    result.atoms_num_list = np.array([1])

    assert result.nep_txt_path == model
    assert result.is_spin_model is True
    assert result.spin_force_out_path == local_tmp_path / "mforce_train.out"
    assert result._should_recalculate({"prediction": "1"}) is False


def test_complete_official_spin_outputs_load_without_supported_model(
    local_tmp_path: Path, monkeypatch
):
    train = local_tmp_path / "train.xyz"
    train.write_text("", encoding="utf-8")
    model = local_tmp_path / "nep.txt"
    model.write_text(
        "nep4_spin 1 Fe\nspin_header_lines 12\nspin_density_order 2\n",
        encoding="utf-8",
    )
    for name, text in {
        "energy_train.out": "1.0 0.0\n",
        "force_train.out": "1 2 3 10 20 30\n",
        "virial_train.out": "1 2 3 4 5 6 10 20 30 40 50 60\n",
        "stress_train.out": "1 2 3 4 5 6 10 20 30 40 50 60\n",
        "mforce_train.out": "4 5 6 40 50 60\n",
    }.items():
        (local_tmp_path / name).write_text(text, encoding="utf-8")
    (local_tmp_path / "prediction.meta.json").write_text(
        '{"schema_version": 1, "predictions": {"train.xyz": {}}}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "NepTrainKit.core.io.nep.inspect_model",
        lambda _path: (_ for _ in ()).throw(AssertionError("model must not open")),
    )
    result = NepTrainResultData.from_path(train)
    result.set_structures([_structure(1)])

    warnings = []
    statuses = []
    result.predictionStatusSignal.connect(statuses.append)
    monkeypatch.setattr(
        "NepTrainKit.core.io.base.MessageManager.send_warning_message",
        warnings.append,
    )
    monkeypatch.setattr(
        "NepTrainKit.core.io.base.NepCalculator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("calculator must not be created")
        ),
    )

    result.load()

    assert result.load_flag is True
    assert result.nep_calc is None
    assert result.is_spin_model is True
    assert result.descriptor.num == 0
    np.testing.assert_allclose(result.energy.all_data, [[1.0, 0.0]])
    np.testing.assert_allclose(result.spin_force.all_data, [[4, 5, 6, 40, 50, 60]])
    assert len(warnings) == 1
    assert "official NEP .out files" in warnings[0]
    assert "descriptor.out is missing" in warnings[0]
    assert statuses == warnings


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


def _structure(natoms: int) -> Structure:
    props = {
        "species": np.array(["Fe"] * natoms, dtype=object),
        "pos": np.zeros((natoms, 3), dtype=np.float32),
    }
    properties = [
        {"name": "species", "type": "S", "count": 1},
        {"name": "pos", "type": "R", "count": 3},
    ]
    return Structure(
        lattice=np.eye(3, dtype=np.float32),
        atomic_properties=props,
        properties=properties,
        additional_fields={"energy": 0.0},
    )


def test_spin_cached_outputs_load_mforce_without_stress_while_generating_descriptor(tmp_path: Path):
    for name, text in {
        "energy_train.out": "1.0 10.0\n",
        "force_train.out": "1 2 3 10 20 30\n",
        "virial_train.out": "1 2 3 4 5 6 10 20 30 40 50 60\n",
        "mforce_train.out": "4 5 6 40 50 60\n",
    }.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    model = tmp_path / "nep.txt"
    model.write_text("nep4_spin 1 Fe\nspin_mode 1\n", encoding="utf-8")

    result = NepTrainResultData(
        model,
        tmp_path / "train.xyz",
        tmp_path / "energy_train.out",
        tmp_path / "force_train.out",
        tmp_path / "stress_train.out",
        tmp_path / "virial_train.out",
        tmp_path / "descriptor.out",
        spin_force_out_path=tmp_path / "mforce_train.out",
        spin_model=True,
    )
    result.set_structures([_structure(1)])
    result.load_structures()

    class FakeCalculator:
        def predict(self, _structures):
            raise AssertionError("spin cached outputs must not be recalculated")

        def descriptors(self, structures, **_kwargs):
            assert len(structures) == 1
            return np.array([[7.0, 8.0]], dtype=np.float32)

    result.nep_calc = FakeCalculator()

    result._load_descriptors()
    result._load_dataset()

    assert not result.stress_out_path.exists()
    assert [dataset.title for dataset in result.datasets] == ["energy", "force", "virial", "mforce", "descriptor"]
    np.testing.assert_allclose(result.descriptor.all_data, [[7.0, 8.0]])
    np.testing.assert_allclose(result.energy.all_data, [[1.0, 10.0]])
    np.testing.assert_allclose(result.force.all_data, [[1, 2, 3, 10, 20, 30]])
    np.testing.assert_allclose(result.virial.all_data, [[1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60]])
    np.testing.assert_allclose(result.spin_force.all_data, [[4, 5, 6, 40, 50, 60]])


def test_spin_calculator_loads_adapter_backend(local_tmp_path: Path, monkeypatch):
    model = local_tmp_path / "nep.txt"
    model.write_text("nep4_spin 1 Fe\nspin_mode 1\n", encoding="utf-8")

    model_info = ModelInfo(
        model_type="spin",
        elements=("Fe",),
        num_types=1,
        descriptor_dim=4,
        cutoff_radial=6.0,
        cutoff_angular=4.0,
        cutoff_max=6.0,
        capabilities=0,
        capability_names=frozenset(),
        sha256="test",
        backend="cpu",
    )

    class FakeAdapter:
        def __init__(self, _model, backend):
            assert backend == "cpu"
            self.model_info = model_info

        def close(self):
            pass

        def cancel(self):
            pass

        def reset_cancel(self):
            pass

    monkeypatch.setattr("NepTrainKit.core.calculator.AdapterCalculator", FakeAdapter)

    calc = NepCalculator(model, backend="cpu")

    assert calc.is_spin_model is True
    assert calc.initialized is True
    assert calc.backend.value == "cpu"


def test_spin_prediction_preserves_mforces(local_tmp_path: Path, monkeypatch):
    model = local_tmp_path / "nep.txt"
    model.write_text("nep4_spin 1 Fe\nspin_mode 1\n", encoding="utf-8")

    model_info = ModelInfo(
        model_type="spin",
        elements=("Fe",),
        num_types=1,
        descriptor_dim=4,
        cutoff_radial=6.0,
        cutoff_angular=4.0,
        cutoff_max=6.0,
        capabilities=0,
        capability_names=frozenset(),
        sha256="test",
        backend="cpu",
    )

    class FakeAdapter:
        def __init__(self, _model, backend):
            self.model_info = model_info

        def predict_spin_structures(self, structures):
            counts = np.asarray([len(item) for item in structures], dtype=np.int32)
            atoms = int(counts.sum())
            return SpinPrediction(
                energy=np.zeros(len(structures)),
                potential=np.zeros(atoms),
                forces=np.zeros((atoms, 3)),
                virials=np.zeros((atoms, 9)),
                structure_virials=np.zeros((len(structures), 9)),
                atom_counts=counts,
                mforces=np.ones((atoms, 3)),
            )

        def close(self):
            pass

        def cancel(self):
            pass

        def reset_cancel(self):
            pass

    monkeypatch.setattr("NepTrainKit.core.calculator.AdapterCalculator", FakeAdapter)
    calc = NepCalculator(model, backend="cpu")
    prediction = calc.predict([_structure(1), _structure(2)])

    assert isinstance(prediction, SpinPrediction)
    np.testing.assert_allclose(prediction.mforces, np.ones((3, 3)))
