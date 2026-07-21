from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from loguru import logger
from nep_adapters import (
    BackendStatus,
    BackendUnavailableError,
    ModelInfo,
    OutOfMemoryError,
    Prediction,
    UnsupportedModelError,
)

from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.io.nep import NepDipoleResultData, NepPolarizabilityResultData
from NepTrainKit.core.types import NepBackend


MODEL_INFO = ModelInfo(
    model_type="ordinary",
    elements=("H",),
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


def _status(available: bool) -> BackendStatus:
    return BackendStatus(
        backend="cuda",
        installed=available,
        available=available,
        reason="available" if available else "module_missing",
        detail="CUDA ready" if available else "CUDA module is not installed",
    )


def _prediction(structures) -> Prediction:
    counts = np.asarray([len(item) for item in structures], dtype=np.int32)
    atoms = int(counts.sum())
    return Prediction(
        energy=np.arange(len(structures), dtype=np.float64),
        potential=np.zeros(atoms),
        forces=np.zeros((atoms, 3)),
        virials=np.zeros((atoms, 9)),
        structure_virials=np.zeros((len(structures), 9)),
        atom_counts=counts,
    )


def test_explicit_cuda_fails_before_model_load_when_unavailable(monkeypatch):
    monkeypatch.setattr("NepTrainKit.core.calculator.backend_status", lambda _name: _status(False))
    monkeypatch.setattr(
        "NepTrainKit.core.calculator.AdapterCalculator",
        lambda *_args, **_kwargs: pytest.fail("unavailable CUDA must not load a model"),
    )

    with pytest.raises(BackendUnavailableError, match="Select CPU"):
        NepCalculator(Path("nep.txt"), backend="cuda")


def test_auto_falls_back_only_during_backend_selection(monkeypatch):
    loaded = []

    class FakeAdapter:
        def __init__(self, _model, backend):
            loaded.append(backend)
            if backend == "cuda":
                raise UnsupportedModelError("model is not supported", backend="cuda")
            self.model_info = MODEL_INFO

    monkeypatch.setattr("NepTrainKit.core.calculator.backend_status", lambda _name: _status(True))
    monkeypatch.setattr("NepTrainKit.core.calculator.AdapterCalculator", FakeAdapter)

    calculator = NepCalculator(Path("nep.txt"), backend="auto")

    assert loaded == ["cuda", "cpu"]
    assert calculator.selection.resolved.value == "cpu"
    assert calculator.selection.reason.startswith("cuda_model_unavailable")


def test_cuda_oom_splits_chunks_without_switching_to_cpu(monkeypatch):
    loaded = []

    class FakeAdapter:
        def __init__(self, _model, backend):
            loaded.append(backend)
            self.backend = backend
            self.model_info = MODEL_INFO

        def recommend_max_atoms(self):
            return None

        def predict_structures(self, structures):
            if len(structures) > 1:
                raise OutOfMemoryError("out of memory", backend=self.backend)
            return _prediction(structures)

    monkeypatch.setattr("NepTrainKit.core.calculator.backend_status", lambda _name: _status(True))
    monkeypatch.setattr("NepTrainKit.core.calculator.AdapterCalculator", FakeAdapter)
    calculator = NepCalculator(Path("nep.txt"), backend="auto", chunk_max_atoms=100)

    prediction = calculator.predict([Atoms("H"), Atoms("H"), Atoms("H")])

    assert loaded == ["cuda"]
    assert calculator.backend.value == "cuda"
    assert prediction.atom_counts.tolist() == [1, 1, 1]


def test_public_calculation_methods_keep_debug_timing(monkeypatch):
    class FakeAdapter:
        def __init__(self, _model, backend):
            self.backend = backend
            self.model_info = MODEL_INFO

        def predict_structures(self, structures):
            return _prediction(structures)

    monkeypatch.setattr(
        "NepTrainKit.core.calculator.backend_status", lambda _name: _status(False)
    )
    monkeypatch.setattr(
        "NepTrainKit.core.calculator.AdapterCalculator", FakeAdapter
    )
    calculator = NepCalculator(Path("nep.txt"), backend="cpu")

    messages = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        calculator.predict([Atoms("H")])
    finally:
        logger.remove(sink_id)

    assert any("Function 'predict' executed in" in str(message) for message in messages)
    for method_name in (
        "predict",
        "predict_with_descriptors",
        "descriptors",
        "polarizabilities",
        "predict_dftd3",
        "predict_with_dftd3",
    ):
        assert hasattr(getattr(NepCalculator, method_name), "__wrapped__")


def test_combined_prediction_uses_fused_adapter_capability(monkeypatch):
    calls = []
    fused_info = replace(
        MODEL_INFO,
        capabilities=1 << 11,
        capability_names=frozenset({"evaluate_with_descriptors"}),
    )

    class FakeAdapter:
        def __init__(self, _model, backend):
            self.backend = backend
            self.model_info = fused_info

        def predict_with_descriptors_structures(self, structures):
            calls.append(len(structures))
            return (
                _prediction(structures),
                np.zeros((sum(map(len, structures)), 4), dtype=np.float64),
            )

    monkeypatch.setattr(
        "NepTrainKit.core.calculator.backend_status", lambda _name: _status(False)
    )
    monkeypatch.setattr(
        "NepTrainKit.core.calculator.AdapterCalculator", FakeAdapter
    )
    calculator = NepCalculator(Path("nep.txt"), backend="cpu")

    prediction, descriptors = calculator.predict_with_descriptors(
        [Atoms("H"), Atoms("HH")]
    )

    assert calls == [2]
    assert prediction.atom_counts.tolist() == [1, 2]
    assert descriptors.shape == (2, 4)


def test_dipole_and_polarizability_results_force_cpu(tmp_path):
    dipole = NepDipoleResultData(
        tmp_path / "nep.txt",
        tmp_path / "train.xyz",
        tmp_path / "dipole_train.out",
        tmp_path / "descriptor.out",
    )
    polar = NepPolarizabilityResultData(
        tmp_path / "nep.txt",
        tmp_path / "train.xyz",
        tmp_path / "polarizability_train.out",
        tmp_path / "descriptor.out",
    )

    assert dipole._calculation_backend() is NepBackend.CPU
    assert polar._calculation_backend() is NepBackend.CPU


def test_dftd3_uses_lazy_cpu_engine_when_main_backend_is_cuda(monkeypatch):
    calls = []

    class FakeAdapter:
        def __init__(self, _model, backend):
            self.backend = backend
            self.model_info = MODEL_INFO
            calls.append((backend, "load"))

        def predict_dftd3_structures(
            self, structures, _functional, _cutoff, _cutoff_cn
        ):
            calls.append((self.backend, "dftd3"))
            return _prediction(structures)

        def predict_with_dftd3_structures(
            self, structures, _functional, _cutoff, _cutoff_cn
        ):
            calls.append((self.backend, "with_dftd3"))
            return _prediction(structures)

    monkeypatch.setattr(
        "NepTrainKit.core.calculator.backend_status", lambda _name: _status(True)
    )
    monkeypatch.setattr(
        "NepTrainKit.core.calculator.AdapterCalculator", FakeAdapter
    )
    calculator = NepCalculator(Path("nep.txt"), backend="cuda")

    calculator.predict_dftd3(
        [Atoms("H")], functional="pbe", cutoff=12.0, cutoff_cn=10.0
    )
    calculator.predict_with_dftd3(
        [Atoms("H")], functional="pbe", cutoff=12.0, cutoff_cn=10.0
    )

    assert calls == [
        ("cuda", "load"),
        ("cpu", "load"),
        ("cpu", "dftd3"),
        ("cpu", "with_dftd3"),
    ]
