import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.io import read as ase_read

from NepTrainKit.core import MessageManager
from NepTrainKit.core.calculator import NepCalculator
from NepTrainKit.core.io.importers import ase_atoms_to_structure
from NepTrainKit.core.io.nep import (
    NepDipoleResultData,
    NepPolarizabilityResultData,
    NepTrainResultData,
)
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


def _response_structures() -> list[Structure]:
    structures = []
    for dipole, diagonal in (
        ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
        ((4.0, 5.0, 6.0), (4.0, 5.0, 6.0)),
    ):
        structures.append(
            Structure(
                lattice=np.eye(3),
                atomic_properties={
                    "species": ["Fe"],
                    "pos": np.zeros((1, 3)),
                },
                properties=[
                    {"name": "species", "type": "S", "count": 1},
                    {"name": "pos", "type": "R", "count": 3},
                ],
                additional_fields={
                    "dipole": " ".join(map(str, dipole)),
                    "pol": " ".join(
                        map(
                            str,
                            (
                                diagonal[0], 0.0, 0.0,
                                0.0, diagonal[1], 0.0,
                                0.0, 0.0, diagonal[2],
                            ),
                        )
                    ),
                    "pbc": "T T T",
                },
            )
        )
    return structures


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


@pytest.mark.parametrize("managed", [False, True])
def test_cached_prediction_missing_labels_require_verified_neptrainkit_provenance(tmp_path: Path, managed: bool):
    labelled = _labeled_structure(1.0, [[1.0, 2.0, 3.0]])
    unlabelled = _labeled_structure(2.0, [[4.0, 5.0, 6.0]])
    unlabelled.additional_fields.pop("energy")
    unlabelled.atomic_properties.pop("forces")
    result = _alignment_result(tmp_path, [labelled, unlabelled])
    result.data_xyz_path.write_text("source dataset", encoding="utf-8")
    np.savetxt(result.energy_out_path, [[1.1, 1.0], [2.0, 2.0]])
    np.savetxt(result.force_out_path, [[2, 3, 4, 1, 2, 3], [5, 6, 7, 5, 6, 7]])
    for path in (result.stress_out_path, result.virial_out_path):
        np.savetxt(path, np.ones((2, 12)))
    paths = (result.energy_out_path, result.force_out_path, result.virial_out_path, result.stress_out_path)
    before = [path.read_bytes() for path in paths]
    if managed:
        result.prediction_meta_path.write_text(json.dumps({
            "producer": "NepTrainKit",
            "schema_version": 1,
            "predictions": {"train.xyz": {
                "dataset": {"sha256": result._sha256(result.data_xyz_path)},
                "outputs": [{"name": path.name, "sha256": result._sha256(path)} for path in paths],
            }},
        }), encoding="utf-8")

    result._load_dataset()

    assert result.energy.get_rmse() == pytest.approx(0.1 if managed else 0.1 / np.sqrt(2), rel=1e-5)
    assert result.force.get_rmse() == pytest.approx(1.0 if managed else 1 / np.sqrt(2))
    if managed:
        assert np.isnan(result.virial.get_rmse())
    else:
        assert result.virial.get_rmse() == 0
    assert [path.read_bytes() for path in paths] == before


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


def test_polarizability_mismatch_activates_recalculation(tmp_path: Path, monkeypatch):
    structures = _response_structures()
    output_path = tmp_path / "polarizability_train.out"
    result = NepPolarizabilityResultData(
        tmp_path / "nep.txt",
        tmp_path / "train.xyz",
        output_path,
        tmp_path / "descriptor.out",
    )
    result.set_structures(structures)
    result.load_structures()
    references = np.vstack([structure.nep_polarizability for structure in structures])
    np.savetxt(output_path, np.column_stack([np.zeros_like(references), references[::-1]]))
    monkeypatch.setattr(MessageManager, "send_warning_message", lambda _message: None)

    assert result._read_cached_response() is None

    assert result._force_recalculate_outputs is True
    assert result._should_recalculate() is True
    assert result._cached_descriptors_are_usable() is False


def test_dipole_mismatch_activates_recalculation(tmp_path: Path, monkeypatch):
    structures = _response_structures()
    output_path = tmp_path / "dipole_train.out"
    result = NepDipoleResultData(
        tmp_path / "nep.txt",
        tmp_path / "train.xyz",
        output_path,
        tmp_path / "descriptor.out",
    )
    result.set_structures(structures)
    result.load_structures()
    references = np.vstack([structure.nep_dipole for structure in structures])
    np.savetxt(output_path, np.column_stack([np.zeros_like(references), references[::-1]]))
    monkeypatch.setattr(MessageManager, "send_warning_message", lambda _message: None)

    assert result._read_cached_response() is None

    assert result._force_recalculate_outputs is True
    assert result._should_recalculate() is True
    assert result._cached_descriptors_are_usable() is False


@pytest.mark.parametrize(
    "loader, quantity, method, label, width, plots",
    [
        (NepDipoleResultData, "dipole", "dipoles", "dipole", 3, ("dipole",)),
        (
            NepPolarizabilityResultData, "polarizability", "polarizabilities", "pol", 6,
            ("polarizability_diagonal", "polarizability_no_diagonal"),
        ),
    ],
)
@pytest.mark.parametrize("reference_rows", [0, 1, 2])
def test_response_output_keeps_missing_references_separate(
    tmp_path, loader, quantity, method, label, width, plots, reference_rows,
):
    structures = [_labeled_structure(0, [[0, 0, 0]] * natoms) for natoms in (2, 4)]
    for structure in structures[:reference_rows]:
        structure.additional_fields[label] = " ".join(["0"] * (3 if label == "dipole" else 9))
    output = tmp_path / f"{quantity}_train.out"
    np.savetxt(output, np.full((2, width * 2), 99.0))

    def load(predict):
        result = loader.from_path(tmp_path / "train.xyz", structures=structures)
        result.set_cache_outputs_override(True)
        result.load_structures()
        result.nep_calc = SimpleNamespace(**{method: predict})
        result._load_dataset()
        return result

    result = load(lambda _: np.repeat([[2.0], [4.0]], width, axis=1))
    data = np.loadtxt(output, ndmin=2)
    np.testing.assert_equal(data[:, :width], np.ones((2, width)))
    np.testing.assert_equal(data[:reference_rows, width:], np.zeros((reference_rows, width)))
    assert np.isnan(data[reference_rows:, width:]).all()

    if reference_rows == len(structures):
        reopened = load(lambda _: pytest.fail("The verified response cache should be reused"))
    else:
        reopened = load(lambda _: np.repeat([[6.0], [8.0]], width, axis=1))
    for name in plots:
        plot = getattr(result, name)
        np.testing.assert_equal(plot.y, np.ones(6))
        if reference_rows == len(structures):
            np.testing.assert_equal(getattr(reopened, name).all_data, plot.all_data)
        else:
            np.testing.assert_equal(getattr(reopened, name).y, [3, 3, 3, 2, 2, 2])
        if reference_rows:
            assert plot.get_rmse() == 1.0
            assert plot.get_max_error_index(2) == list(range(reference_rows))
        else:
            assert np.isnan(plot.get_rmse())
            assert plot.get_formart_rmse() == "—"
            assert plot.get_max_error_index(2) == []


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
    for output in record["outputs"]:
        assert output["sha256"] == result._sha256(tmp_path / output["name"])
    force_path = tmp_path / "force_train.out"
    content = force_path.read_bytes()
    force_path.write_bytes(content.replace(b"0", b"1", 1))
    assert result._should_recalculate({"prediction": "1"}) is True


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


@pytest.mark.parametrize("origin", ["external", "edited_after_manifest", "managed", "aligned_external"])
def test_nep_load_only_replaces_verified_managed_files(tmp_path: Path, origin: str):
    fixture = Path(__file__).parent / "data/nep"
    for name in ("train.xyz", "nep.txt"):
        shutil.copy2(fixture / name, tmp_path / name)
    result = NepTrainResultData.from_path(tmp_path / "train.xyz", cache_outputs=True)
    result.load_structures()
    frames, atoms = len(result.atoms_num_list), int(result.atoms_num_list.sum())
    paths = (result.energy_out_path, result.force_out_path, result.virial_out_path, result.stress_out_path)
    for path, shape in zip(paths, ((frames, 2), (atoms, 6), (frames, 12), (frames, 12))):
        np.savetxt(path, np.full(shape, 123.456))
    result.descriptor_path.write_text("9 9\n", encoding="utf-8")
    if origin == "aligned_external":
        structures = result.structure.now_data
        energy = np.array([structure.per_atom_energy for structure in structures])
        force = np.vstack([structure.forces for structure in structures])
        np.savetxt(result.energy_out_path, np.column_stack((np.full(frames, 123.456), energy)))
        np.savetxt(result.force_out_path, np.column_stack((np.full_like(force, 123.456), force)))
    elif origin != "external":
        outputs = [{"name": path.name, "sha256": result._sha256(path)} for path in paths]
        result.prediction_meta_path.write_text(json.dumps({
            "producer": "NepTrainKit",
            "schema_version": 1,
            "predictions": {"train.xyz": {"dataset": {"sha256": "old-dataset"}, "outputs": outputs}},
        }), encoding="utf-8")
        if origin == "edited_after_manifest":
            with result.force_out_path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
    before = {path: path.read_bytes() for path in (*paths, result.descriptor_path, result.prediction_meta_path)
              if path.is_file()}

    result.load()

    assert result.load_flag
    assert np.isfinite(result.energy.y).all()
    if origin == "aligned_external":
        assert result.nep_calc is None
        np.testing.assert_allclose(result.energy.y, 123.456)
    else:
        assert not np.allclose(result.energy.y, 123.456)
    if origin == "managed":
        assert all(path.read_bytes() != before[path] for path in paths)
        assert result.prediction_meta_path.read_bytes() != before[result.prediction_meta_path]
    else:
        assert all(path.read_bytes() == old for path, old in before.items())
        assert result.prediction_meta_path.exists() == (origin == "edited_after_manifest")


def test_automatic_preference_checks_all_frames_before_opening_calculator(tmp_path, monkeypatch):
    from NepTrainKit.core.io import base
    from NepTrainKit.paths import get_bundled_nep89_path

    preferred = tmp_path / "candidate.txt"
    shutil.copyfile(Path(__file__).parent / "data" / "nep" / "nep.txt", preferred)
    path = tmp_path / "custom.xyz"
    opened = []

    def stop_after_model_selection(*, model_file, **kwargs):
        opened.append(Path(model_file))
        raise ValueError("probe ends before prediction")

    monkeypatch.setattr(base, "NepCalculator", stop_after_model_selection)
    for symbols, expected in ((("Te", "Pb"), preferred), (("Te", "Fe"), get_bundled_nep89_path())):
        path.write_text("".join(
            f'1\nLattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3\n{symbol} 0 0 0\n'
            for symbol in symbols
        ))
        result = NepTrainResultData.from_path(path, default_nep_path=preferred, cache_outputs=False)
        result.load()
        assert opened.pop() == expected
        assert result.nep_txt_path == expected
        assert result.energy_out_path.parent.name == f"custom_{expected.stem}"
        assert result.structure.num == 2
    assert not list(tmp_path.glob("custom_*/"))


def test_view_data_only_loads_partial_outputs_without_a_calculator(tmp_path, monkeypatch):
    from NepTrainKit.core.io import base

    def unexpected_calculator(**kwargs):
        raise AssertionError("View data only must not create a calculator")

    monkeypatch.setattr(base, "NepCalculator", unexpected_calculator)
    structures = [_labeled_structure(-2, [[0, 0, 0]])]
    path = tmp_path / "train.xyz"
    energy = tmp_path / "energy_train.out"
    energy.write_text("-2 -2\n")
    bad_forces = tmp_path / "force_train.out"
    bad_forces.write_text("1 2 3\n")
    (tmp_path / "nep.txt").write_text("a deliberately unreadable potential\n")
    result = NepTrainResultData.from_path(path, structures=structures, outputs_only=True)
    result.load()

    assert result.load_flag
    assert result.nep_calc is None
    assert result.structure.num == 1
    assert result.energy.num == 1
    assert result.force.num == result.virial.num == result.stress.num == result.descriptor.num == 0
    assert all(dataset.x.size == dataset.structure_index.size for dataset in result.datasets)
    assert not result.force.parity_mode and not result.force.show_rmse
    assert energy.read_text() == "-2 -2\n"
    assert bad_forces.read_text() == "1 2 3\n"
    assert not (tmp_path / "descriptor.out").exists()
