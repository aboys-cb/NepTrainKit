"""Validation library, schema migration, protocols, service and CLI."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from NepTrainKit.core.dataset.database import SCHEMA_VERSION, Database
from NepTrainKit.core.dataset.models import ModelVersion
from NepTrainKit.core.dataset.services import ModelService
from NepTrainKit.core.validation import (
    FileRef,
    Library,
    SetRejectedError,
    SetTooLargeError,
    ValidationService,
    parse_model_header,
)
from NepTrainKit.core.validation import library as library_module
from NepTrainKit.core.validation.cli import main as cli_main
from NepTrainKit.core.validation.protocols import detect_protocol, get_protocol
from NepTrainKit.core.validation.protocols.eos import fit_eos, summarize_curve
from NepTrainKit.core.validation.service import compare

DATA = Path(__file__).parent / "data" / "nep"
MODEL_HEADER = "nep4 2 Te Pb\ncutoff 8 4 74 10\n"


# -- fixtures -------------------------------------------------------------------

@dataclass
class FakePrediction:
    energy: np.ndarray
    forces: np.ndarray
    structure_virials: np.ndarray


class FakePredictor:
    """Reference labels distorted in a controlled way, without a native model."""

    calls = 0

    def __init__(self, scale=1.0, offset=0.0):
        self.scale, self.offset = scale, offset
        self.closed = False

    def predict(self, frames):
        FakePredictor.calls += 1
        assert all(atoms.calc is None for atoms in frames), "labels must not reach the model"
        energy = np.asarray([atoms.info["_ref_energy"] for atoms in frames]) * self.scale
        energy += self.offset * np.asarray([len(atoms) for atoms in frames])
        forces = np.concatenate([atoms.arrays["_ref_forces"] for atoms in frames]) * self.scale
        return FakePrediction(energy, forces, np.zeros((len(frames), 9)))

    def close(self):
        self.closed = True


def _label(atoms, energy, forces=None):
    forces = np.zeros((len(atoms), 3)) if forces is None else forces
    atoms.info["_ref_energy"] = energy
    atoms.arrays["_ref_forces"] = forces
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    return atoms


def eos_frames(curvature=0.5, minimum=1.0, points=7):
    frames = []
    v0 = 6.46**3 / 8 * minimum
    for scale in np.linspace(0.94, 1.06, points):
        atoms = bulk("PbTe", "rocksalt", a=6.46 * scale)
        volume = atoms.get_volume() / len(atoms)
        frames.append(_label(atoms, len(atoms) * (-4.0 + curvature * (volume - v0) ** 2 / v0)))
    return frames


def efv_frames(count=4):
    rng = np.random.default_rng(0)
    frames = []
    for index in range(count):
        atoms = bulk("PbTe", "rocksalt", a=6.46).repeat((1, 1, 2))
        atoms.rattle(0.02, seed=index)
        frames.append(_label(atoms, -4.0 * len(atoms) + 0.01 * index, rng.normal(0, 0.1, (len(atoms), 3))))
    return frames


def write_xyz(path, frames):
    clean = []
    for atoms in frames:
        copy = atoms.copy()
        copy.calc = SinglePointCalculator(copy, energy=atoms.info["_ref_energy"], forces=atoms.arrays["_ref_forces"])
        clean.append(copy)
    write(path, clean, format="extxyz")
    return path


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "run1" / "nep.txt"
    path.parent.mkdir()
    path.write_text(MODEL_HEADER)
    return path


@pytest.fixture
def make_service(tmp_path):
    predictors = {}

    def factory(model_path, backend):
        # Distortion keyed by model content so different versions score differently.
        scale, offset = predictors.get(Path(model_path).name, (1.0, 0.0))
        return FakePredictor(scale, offset)

    def build(**distortions):
        predictors.update(distortions)
        return ValidationService(Library(tmp_path / "lib"), predictor_factory=factory)

    build.predictors = predictors
    return build


def _import(service, tmp_path, frames, name, **kwargs):
    path = write_xyz(tmp_path / f"{name}.xyz", frames)
    return service.import_set(path, name=name, provenance={"source_kind": "DFT", "conditions": "PBE"}, **kwargs)


# -- library --------------------------------------------------------------------

def test_objects_are_content_addressed_read_only_and_verified(tmp_path):
    library = Library(tmp_path / "lib")
    digest = library.put_bytes(b"model")
    assert library.put_bytes(b"model") == digest
    path = library.open_object(digest)
    assert path.read_bytes() == b"model" and not (path.stat().st_mode & 0o200)
    path.chmod(0o600)
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="corrupt"):
        library.open_object(digest)
    with pytest.raises(ValueError):
        library.object_path("../../etc/passwd")


def test_library_root_resolution_and_newer_layout_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(library_module.ENV_VAR, str(tmp_path / "env"))
    assert library_module.resolve_library_root() == tmp_path / "env"
    assert library_module.resolve_library_root(tmp_path / "x") == tmp_path / "x"
    root = tmp_path / "future"
    root.mkdir()
    (root / "library.json").write_text(json.dumps({"layout": 99}))
    with pytest.raises(RuntimeError, match="newer"):
        Library(root)
    with pytest.raises(FileNotFoundError):
        Library(tmp_path / "missing", create=False)


def test_file_ref_detects_changes_and_relocates(tmp_path):
    original = tmp_path / "a" / "train.xyz"
    original.parent.mkdir()
    original.write_text("frames")
    ref = FileRef.of(original)
    assert FileRef.from_dict(ref.to_dict()) == ref and ref.status() == "ok"
    moved = tmp_path / "b" / "deep" / "train.xyz"
    moved.parent.mkdir(parents=True)
    original.rename(moved)
    assert ref.status() == "missing"
    assert ref.relocate(tmp_path / "b").path == str(moved.resolve())
    moved.write_text("other!")
    assert FileRef.of(moved).digest != ref.digest
    assert ref.relocate(tmp_path / "b") is None


def test_default_library_adopts_legacy_database_and_cas(tmp_path, monkeypatch):
    monkeypatch.delenv(library_module.ENV_VAR, raising=False)
    monkeypatch.setattr(library_module, "get_user_config_path", lambda: tmp_path)
    Database(tmp_path / "mlpman.db")
    blob = b"legacy"
    (tmp_path / "cas").mkdir()
    digest = library_module.sha256_bytes(blob)
    (tmp_path / "cas" / digest).write_bytes(blob)
    library = Library()
    assert library.root == (tmp_path / "library").resolve()
    assert library.db_path.exists() and (tmp_path / "mlpman.db").exists()  # copied, not moved
    assert library.open_object(digest).read_bytes() == blob


# -- schema migration -------------------------------------------------------------

LEGACY_SCHEMA = """
CREATE TABLE tags (id INTEGER NOT NULL, name VARCHAR(64) COLLATE "NOCASE" NOT NULL, color VARCHAR(9) NOT NULL,
    notes TEXT NOT NULL, created_at DATETIME NOT NULL, PRIMARY KEY (id));
CREATE UNIQUE INDEX ix_tags_name ON tags (name);
CREATE TABLE projects (id INTEGER NOT NULL, name VARCHAR(200) NOT NULL, notes TEXT NOT NULL,
    created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, parent_id INTEGER, PRIMARY KEY (id), UNIQUE (name),
    FOREIGN KEY(parent_id) REFERENCES projects (id) ON DELETE CASCADE);
CREATE TABLE model_versions (id INTEGER NOT NULL, name VARCHAR(80) NOT NULL, project_id INTEGER NOT NULL,
    model_type VARCHAR(20) NOT NULL, data_size INTEGER NOT NULL, energy FLOAT NOT NULL, force FLOAT NOT NULL,
    virial FLOAT NOT NULL, model_path TEXT NOT NULL, notes TEXT NOT NULL, parent_id INTEGER,
    created_at DATETIME NOT NULL, PRIMARY KEY (id),
    FOREIGN KEY(project_id) REFERENCES projects (id) ON DELETE CASCADE,
    FOREIGN KEY(parent_id) REFERENCES model_versions (id) ON DELETE CASCADE);
CREATE INDEX ix_model_versions_project_id ON model_versions (project_id);
CREATE INDEX ix_model_versions_parent_id ON model_versions (parent_id);
CREATE TABLE model_version_tags (model_version_id INTEGER NOT NULL, tag_id INTEGER NOT NULL,
    created_at DATETIME NOT NULL, PRIMARY KEY (model_version_id, tag_id),
    FOREIGN KEY(model_version_id) REFERENCES model_versions (id) ON DELETE CASCADE,
    FOREIGN KEY(tag_id) REFERENCES tags (id) ON DELETE CASCADE);
INSERT INTO projects VALUES (1, 'FeNi', '', '2025-01-01', '2025-01-01', NULL);
INSERT INTO tags VALUES (1, 'good', '#fff', '', '2025-01-01');
INSERT INTO model_versions VALUES
    (1, 'v1', 1, 'NEP', 100, 1.5, 50.0, 10.0, '/runs/v1', 'first', NULL, '2025-01-01 00:00:00');
INSERT INTO model_versions VALUES (2, 'v2', 1, 'NEP', 120, 1.2, 45.0, 9.0, '/runs/v2', '', 1, '2025-02-01 00:00:00');
INSERT INTO model_version_tags VALUES (2, 1, '2025-02-01');
"""


def test_legacy_database_is_migrated_without_losing_rows(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(LEGACY_SCHEMA)
    db = Database(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"benchmark_sets", "benchmark_runs", "benchmark_suites", "model_versions"} <= tables
    assert "model_versions_new" not in tables
    models = ModelService(db).get_models_by_project_id(1)
    assert [m.name for m in models] == ["v1"] and [c.name for c in models[0].children] == ["v2"]
    assert [t.name for t in models[0].children[0].tags] == ["good"]
    with db.session() as session:
        session.add(ModelVersion(name="captured", project_id=1, model_type="NEP", model_path="/x"))
        session.commit()  # training metrics may now be NULL
    Database(path)  # idempotent


def test_new_database_starts_at_latest_schema(tmp_path):
    Database(tmp_path / "fresh.db")
    with sqlite3.connect(tmp_path / "fresh.db") as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


# -- protocols --------------------------------------------------------------------

def test_eos_fit_and_curve_metrics():
    frames = eos_frames()
    volumes = np.asarray([a.get_volume() / len(a) for a in frames])
    energies = np.asarray([a.info["_ref_energy"] / len(a) for a in frames])
    assert "v0" in fit_eos(volumes, energies)
    assert "reason" in fit_eos(volumes, volumes)  # monotonic: minimum on the boundary
    summary = summarize_curve(volumes, energies, energies + 0.01)
    assert summary["offset_ev_atom"] == pytest.approx(0.01)
    assert summary["shape_rmse_mev"] == pytest.approx(0, abs=1e-9)
    assert summary["raw_rmse_mev"] == pytest.approx(10)
    with pytest.raises(ValueError):
        summarize_curve(volumes, energies, energies[:-1])


def test_protocol_detection_prefers_specific_protocols():
    assert detect_protocol(eos_frames())[0][1].kind == "eos"
    assert detect_protocol(efv_frames())[0][1].kind == "efv"
    eos = get_protocol("eos")
    mixed = eos_frames()[:4] + [_label(bulk("Pb", "fcc", a=5.0), -3.0)]
    assert any(issue.level == "error" for issue in eos.check(mixed))
    assert any("bracketed" in issue.message for issue in eos.check(eos_frames(minimum=0.85)))


def test_efv_reads_nep_style_labels_and_virials():
    atoms = bulk("PbTe", "rocksalt", a=6.46)
    atoms.info["energy"] = -8.0
    atoms.info["virial"] = np.arange(9.0)
    atoms.arrays["force"] = np.ones((len(atoms), 3))
    from NepTrainKit.core.validation.protocols.base import reference_labels

    labels = reference_labels([atoms])
    assert labels.energy_per_atom[0] == pytest.approx(-4.0)
    assert labels.forces.shape == (2, 3)
    assert labels.virial_per_atom[0] == pytest.approx(np.arange(9.0) / 2)


def test_compare_respects_metric_direction():
    assert compare(1.0, 2.0, "lower", 0.05) == "better"
    assert compare(3.0, 2.0, "lower", 0.05) == "worse"
    assert compare(2.05, 2.0, "lower", 0.05) == "same"
    assert compare(-1.0, 4.0, "closer-to-zero", 0.05) == "better"
    assert compare(None, 1.0, "lower", 0.05) is None


# -- service ----------------------------------------------------------------------

def test_header_parsing():
    assert parse_model_header(b"nep4_zbl 2 Fe Ni\n") == ("nep4_zbl", ("Fe", "Ni"))
    for bad in (b"", b"deepmd\n", b"nep4 3 Fe Ni\n"):
        with pytest.raises(ValueError):
            parse_model_header(bad)


def test_capture_is_deduplicated_and_retraining_makes_a_new_version(make_service, model_file):
    service = make_service()
    first = service.capture_model(model_file)
    assert first["name"] == "run1" and first["auto_captured"] and first["elements"] == ["Te", "Pb"]
    assert service.capture_model(model_file)["id"] == first["id"]
    assert service.find_model_by_file(model_file)["id"] == first["id"]
    model_file.write_text(MODEL_HEADER + "retrained\n")
    second = service.capture_model(model_file, auto=False)
    assert second["id"] != first["id"] and not second["auto_captured"]
    model_file.unlink()  # the snapshot survives the user's file
    assert service.model_file(first["id"]).read_text() == MODEL_HEADER
    assert [m["id"] for m in service.models()] == [second["id"], first["id"]]


def test_import_requires_declared_source_and_rejects_bad_sets(make_service, tmp_path):
    service = make_service()
    path = write_xyz(tmp_path / "eos.xyz", eos_frames())
    with pytest.raises(ValueError, match="source kind"):
        service.import_set(path, name="x", provenance={})
    with pytest.raises(SetTooLargeError):
        service.import_set(path, name="x", provenance={"source_kind": "DFT"}, max_bytes=10)
    with pytest.raises(SetRejectedError):
        _import(service, tmp_path, eos_frames()[:3], "few", protocol="eos")
    sealed = _import(service, tmp_path, eos_frames(), "eos")
    assert sealed["protocol"] == "eos" and sealed["elements"] == ["Pb", "Te"]
    assert _import(service, tmp_path, eos_frames(), "eos")["id"] == sealed["id"]
    path.unlink()
    assert len(service.set_frames(sealed["id"])) == 7


def test_evaluation_is_cached_per_model_and_failures_are_kept(make_service, model_file, tmp_path):
    service = make_service()
    model = service.capture_model(model_file)
    benchmark = _import(service, tmp_path, efv_frames(), "efv")
    FakePredictor.calls = 0
    run = service.evaluate(model["id"], benchmark["id"])
    assert run["status"] == "succeeded" and run["metrics"]["energy_rmse"] == pytest.approx(0)
    assert service.run_details(run)["details"]["energy"]["predicted"]
    assert service.evaluate(model["id"], benchmark["id"])["id"] == run["id"] and FakePredictor.calls == 1
    assert service.evaluate(model["id"], benchmark["id"], force=True)["id"] != run["id"]

    def broken(model_path, backend):
        raise RuntimeError("native crash")

    service.predictor_factory = broken
    failed = service.evaluate(model["id"], benchmark["id"], force=True)
    assert failed["status"] == "failed" and "native crash" in failed["error"]


def test_unsupported_models_are_reported_not_evaluated(make_service, tmp_path):
    service = make_service()
    other = tmp_path / "fe" / "nep.txt"
    other.parent.mkdir()
    other.write_text("nep4 1 Fe\n")
    spin = tmp_path / "spin" / "nep.txt"
    spin.parent.mkdir()
    spin.write_text("nep4_spin 2 Te Pb\n")
    eos = _import(service, tmp_path, eos_frames(), "eos")
    assert "element" in service.evaluate(service.capture_model(other)["id"], eos["id"])["error"]
    assert service.evaluate(service.capture_model(spin)["id"], eos["id"])["status"] == "unsupported"
    assert service.matching_sets(service.capture_model(other)["id"]) == []


def test_scorecard_compares_with_baseline(make_service, tmp_path):
    service = make_service()
    paths = []
    for index, text in enumerate(("a", "b")):
        path = tmp_path / f"m{index}" / "nep.txt"
        path.parent.mkdir()
        path.write_text(MODEL_HEADER + text)
        paths.append(path)
    old, new = (service.capture_model(path) for path in paths)
    old_hash, new_hash = old["content_hash"], new["content_hash"]
    make_service(**{old_hash: (1.10, 0.0), new_hash: (1.02, 0.0)})
    efv = _import(service, tmp_path, efv_frames(), "efv")
    eos = _import(service, tmp_path, eos_frames(), "eos")
    suite = service.create_suite("PbTe", [eos["id"], efv["id"]])
    for model in (old, new):
        assert all(run["status"] == "succeeded" for run in service.run_suite(model["id"], suite["id"]))
    service.set_baseline(suite["id"], old["id"], "first production model")
    card = service.scorecard(suite["id"])
    assert [c["set_name"] for c in card["columns"]][:1] == ["eos"]
    rows = {row["model_id"]: row for row in card["rows"]}
    assert rows[old["id"]]["is_baseline"] and rows[old["id"]]["regressions"] == 0
    verdicts = {c["metric"]: cell["verdict"] for c, cell in zip(card["columns"], rows[new["id"]]["cells"])}
    assert verdicts["energy_rmse"] == "better" and verdicts["force_rmse"] == "better"
    assert service.get_model(old["id"])["auto_captured"] is False  # baselines are protected


def test_garbage_collection_keeps_starred_and_baseline_models(make_service, tmp_path):
    service = make_service()
    paths = {}
    for name in ("stale", "starred", "fresh"):
        path = tmp_path / name / "nep.txt"
        path.parent.mkdir()
        path.write_text(MODEL_HEADER + name)
        paths[name] = service.capture_model(path)
    benchmark = _import(service, tmp_path, efv_frames(), "efv")
    service.evaluate(paths["stale"]["id"], benchmark["id"])
    service.star_model(paths["starred"]["id"])
    old = dt.datetime.utcnow() - dt.timedelta(days=90)
    with service.db.session() as session:
        for name in ("stale", "starred"):
            session.get(ModelVersion, paths[name]["id"]).last_seen_at = old
        session.commit()
    orphan = service.library.put_bytes(b"orphan")
    plan = service.collect_garbage(older_than_days=30)
    assert plan == {"models": 1, "objects": 2, "results": 1, "bytes": plan["bytes"], "dry_run": True}
    assert service.library.has_object(orphan)
    service.collect_garbage(older_than_days=30, dry_run=False)
    remaining = {model["name"] for model in service.models()}
    assert remaining == {"starred", "fresh"}
    assert not service.library.has_object(orphan) and service.library.has_object(benchmark["object_hash"])
    assert list(service.library.results_dir.glob("*.json")) == []


def test_real_nep_model_on_eos_set(tmp_path):
    pytest.importorskip("nep_adapters")
    service = ValidationService(Library(tmp_path / "lib"), backend="cpu")
    model = service.capture_model(DATA / "nep.txt", dataset_path=DATA / "train.xyz")
    assert model["dataset_ref"]["path"].endswith("train.xyz")
    benchmark = _import(service, tmp_path, eos_frames(), "eos")
    run = service.evaluate(model["id"], benchmark["id"])
    assert run["status"] == "succeeded", run["error"]
    assert np.isfinite(run["metrics"]["shape_rmse"]) and run["environment"]["backend"]


# -- integration ------------------------------------------------------------------

def test_cli_round_trip(tmp_path, model_file, capsys, monkeypatch):
    library = tmp_path / "cli-lib"
    created = []

    def factory(model_path, backend):
        created.append(backend)
        return FakePredictor()

    monkeypatch.setattr("NepTrainKit.core.validation.service._nep_predictor", factory)
    base = ["--library", str(library), "--backend", "cpu", "--json"]
    assert cli_main(base + ["capture", str(model_file), "--name", "v1", "--star"]) == 0
    assert json.loads(capsys.readouterr().out)["starred"] is True
    xyz = write_xyz(tmp_path / "efv.xyz", efv_frames())
    assert cli_main(base + ["import", str(xyz), "--name", "test", "--source", "DFT"]) == 0
    set_id = json.loads(capsys.readouterr().out)["id"]
    assert cli_main(base + ["suite", "create", "PbTe", set_id]) == 0
    capsys.readouterr()
    assert cli_main(base + ["run", str(model_file), "--suite", "PbTe"]) == 0
    card = json.loads(capsys.readouterr().out)
    assert card["rows"][0]["cells"][0]["value"] == pytest.approx(0) and created == ["cpu"]
    assert cli_main(base + ["library"]) == 0
    assert json.loads(capsys.readouterr().out)["root"] == str(library.resolve())


def test_viewer_captures_opened_model_and_never_raises(tmp_path, model_file, monkeypatch):
    from NepTrainKit.config import Config
    from NepTrainKit.ui.pages.show_nep import ShowNepWidget

    monkeypatch.setenv(library_module.ENV_VAR, str(tmp_path / "viewer-lib"))
    monkeypatch.setattr(Config, "getboolean", classmethod(lambda cls, s, o, fallback=None: True))
    viewer = SimpleNamespace(
        nep_result_data=SimpleNamespace(nep_txt_path=model_file, data_xyz_path=tmp_path / "none"),
        refresh_validation_chip=lambda: None,
    )
    ShowNepWidget._capture_model_version(viewer)
    assert ValidationService(Library(tmp_path / "viewer-lib")).find_model_by_file(model_file) is not None
    viewer.nep_result_data.nep_txt_path = tmp_path / "not-nep.txt"
    viewer.nep_result_data.nep_txt_path.write_text("deepmd\n")
    ShowNepWidget._capture_model_version(viewer)  # logged, not raised
    monkeypatch.setattr(Config, "getboolean", classmethod(lambda cls, s, o, fallback=None: False))
    model_file.write_text(MODEL_HEADER + "disabled")
    viewer.nep_result_data.nep_txt_path = model_file
    ShowNepWidget._capture_model_version(viewer)
    assert ValidationService(Library(tmp_path / "viewer-lib")).find_model_by_file(model_file) is None
