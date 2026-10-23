"""Model capture, benchmark sets, cached evaluation and the scorecard.

Everything here is UI-free so the GUI, the command line and tests share it.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sqlalchemy import delete, select

from NepTrainKit.core.dataset.models import (
    BenchmarkRun,
    BenchmarkSet,
    BenchmarkSuite,
    BenchmarkSuiteSet,
    ModelVersion,
    Project,
    StorageRef,
)

from .grouping import eos_frame_groups
from .library import FileRef, Library, get_library, sha256_bytes
from .protocols import ModelCapabilities, detect_protocol, get_protocol, read_frames

INBOX_PROJECT = "Inbox"
SOURCE_KINDS = ("DFT", "Other", "Demo")
DEFAULT_MAX_SET_BYTES = 50 * 1024 * 1024
DEFAULT_TOLERANCE = 0.05


class SetTooLargeError(ValueError):
    """Raised so the UI can ask whether to copy a large file anyway."""

    def __init__(self, size: int, limit: int):
        super().__init__(f"Benchmark file is {size / 1e6:.1f} MB (limit {limit / 1e6:.0f} MB).")
        self.size = size
        self.limit = limit


class SetRejectedError(ValueError):
    def __init__(self, issues):
        self.issues = list(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


def parse_model_header(data: bytes) -> tuple[str, tuple[str, ...]]:
    """Return (model kind, elements) from the first line of a NEP ``nep.txt``."""
    line = data.split(b"\n", 1)[0].decode("utf-8", "replace").split()
    if not line or not line[0].startswith("nep"):
        raise ValueError("Not a NEP model file (the first line must start with 'nep').")
    try:
        count = int(line[1])
        elements = tuple(line[2 : 2 + count])
    except (IndexError, ValueError):
        raise ValueError("The NEP model header does not list its elements.") from None
    if len(elements) != count:
        raise ValueError("The NEP model header does not list its elements.")
    return line[0], elements


def _record(row) -> dict:
    result = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    for key, value in result.items():
        if isinstance(value, dt.datetime):
            result[key] = value.isoformat(timespec="seconds") + "Z"
    return result


def _now() -> dt.datetime:
    return dt.datetime.utcnow()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def _nep_predictor(model_path: Path, backend: str):
    from NepTrainKit.core.calculator import NepCalculator

    return NepCalculator(model_path, backend=backend)


def compare(
    value: float | None, baseline: float | None, better: str, tolerance: float, abs_tolerance: float = 0.0
) -> str | None:
    """``better`` / ``worse`` / ``same`` relative to the baseline value.

    Differences within ``tolerance`` of the baseline, or within ``abs_tolerance``
    in the metric's own unit, are ``same``.
    """
    if better == "none" or value is None or baseline is None or not math.isfinite(value) or not math.isfinite(baseline):
        return None
    if better == "closer-to-zero":
        value, baseline, better = abs(value), abs(baseline), "lower"
    scale = max(abs(baseline), 1e-12)
    if abs(value - baseline) <= max(tolerance * scale, abs_tolerance):
        return "same"
    improved = value < baseline if better == "lower" else value > baseline
    return "better" if improved else "worse"


def _finite_metrics(metrics: dict) -> bool:
    return all(value is None or math.isfinite(value) for value in metrics.values())


@dataclass(frozen=True)
class ModelDraft:
    """Opened model bytes, kept in memory until Save or Run is requested."""

    data: bytes
    source_path: str

    @classmethod
    def from_file(cls, path: str | Path):
        path = Path(path).expanduser().resolve()
        data = path.read_bytes()
        parse_model_header(data)
        return cls(data, str(path))

    @property
    def content_hash(self) -> str:
        return sha256_bytes(self.data)

    @property
    def name(self) -> str:
        path = Path(self.source_path)
        return (path.parent.name if path.stem == "nep" else path.stem)[:80]


@dataclass
class ValidationService:
    library: Library | None = None
    predictor_factory: Callable[[Path, str], object] | None = None
    backend: str = "auto"

    def __post_init__(self):
        self.library = self.library or get_library()
        self.db = self.library.database
        self.predictor_factory = self.predictor_factory or _nep_predictor
        self._temporary_suites = {}
        self._temporary_sets = {}
        self._temporary_data = {}
        self._temporary_runs = {}
        self._temporary_results = {}

    # -- models ---------------------------------------------------------------
    def _inbox_project_id(self, session) -> int:
        project = session.scalars(select(Project).where(Project.name == INBOX_PROJECT)).first()
        if project is None:
            project = Project(name=INBOX_PROJECT, notes="Models captured without an explicit project.")
            session.add(project)
            session.flush()
        return project.id

    def capture_model(
        self,
        path: str | Path | ModelDraft,
        *,
        name: str = "",
        dataset_path: str | Path | None = None,
        auto: bool = True,
        project_id: int | None = None,
        notes: str = "",
        temporary: bool = False,
    ) -> dict:
        """Deduplicate bytes globally, version records within the chosen project.

        Temporary validation has no project and can reuse any saved association.
        """
        draft = path if isinstance(path, ModelDraft) else ModelDraft.from_file(path)
        path = Path(draft.source_path)
        data = draft.data
        kind, elements = parse_model_header(data)
        digest = self.library.put_bytes(data)
        dataset_ref = {}
        if dataset_path and Path(dataset_path).is_file():
            dataset_ref = FileRef.of(dataset_path).to_dict()
        with self.db.session() as session:
            if project_id is not None and session.get(Project, project_id) is None:
                raise ValueError(f"Project {project_id} not found")
            target_project = None if temporary else (project_id or self._inbox_project_id(session))
            query = select(ModelVersion).where(ModelVersion.content_hash == digest)
            if not temporary:
                query = query.where(ModelVersion.project_id == target_project)
            row = session.scalars(query.order_by(ModelVersion.project_id.is_(None), ModelVersion.id)).first()
            if row is None and not temporary:
                # Promote the temporary record in place so existing runs keep their identity.
                row = session.scalars(select(ModelVersion).where(
                    ModelVersion.content_hash == digest, ModelVersion.project_id.is_(None)
                )).first()
                if row is not None:
                    row.project_id = target_project
            if row is None:
                default_name = path.parent.name if path.stem == "nep" else path.stem
                row = ModelVersion(
                    name=(name.strip() or default_name or path.name)[:80],
                    project_id=target_project,
                    model_type="NEP",
                    model_path=str(path),
                    notes=notes,
                    content_hash=digest,
                    model_kind=kind,
                    elements=list(elements),
                    source_path=str(path),
                    dataset_ref=dataset_ref,
                    auto_captured=True if temporary else auto,
                )
                session.add(row)
            else:
                row.last_seen_at = _now()
                if not auto and not temporary:
                    row.auto_captured = False
                if name.strip() and not temporary:
                    row.name = name.strip()[:80]
                if dataset_ref and not row.dataset_ref:
                    row.dataset_ref = dataset_ref
            session.commit()
            return _record(row)

    def save_model(self, draft: ModelDraft, *, project_name: str, name: str) -> dict:
        """Explicitly save to a chosen project; other project associations stay unchanged."""
        project_name, name = project_name.strip(), name.strip()
        if not project_name or not name or len(project_name) > 200 or len(name) > 80:
            raise ValueError("Choose a project and a version name (up to 80 characters).")
        with self.db.session() as session:
            project = session.scalars(select(Project).where(Project.name == project_name)).first()
            if project is None:
                project = Project(name=project_name)
                session.add(project)
                session.commit()
            project_id = project.id
        return self.capture_model(draft, project_id=project_id, name=name, auto=False)

    def projects(self) -> list[dict]:
        with self.db.session() as session:
            return [_record(row) for row in session.scalars(select(Project).order_by(Project.name))]

    def find_model_by_hash(self, digest: str) -> dict | None:
        with self.db.session() as session:
            row = session.scalars(select(ModelVersion).where(ModelVersion.content_hash == digest)
                                  .order_by(ModelVersion.project_id.is_(None), ModelVersion.id)).first()
            return _record(row) if row is not None else None

    def find_model_by_file(self, path: str | Path) -> dict | None:
        """Return the captured version whose bytes equal ``path`` right now, if any."""
        try:
            digest = sha256_bytes(Path(path).read_bytes())
        except OSError:
            return None
        return self.find_model_by_hash(digest)

    def get_model(self, model_id: int) -> dict:
        with self.db.session() as session:
            row = session.get(ModelVersion, model_id)
            if row is None:
                raise ValueError(f"Unknown model version: {model_id}")
            return _record(row)

    def models(self, *, starred_only: bool = False) -> list[dict]:
        with self.db.session() as session:
            query = select(ModelVersion).where(ModelVersion.content_hash.is_not(None))
            if starred_only:
                query = query.where(ModelVersion.starred.is_(True))
            query = query.order_by(ModelVersion.last_seen_at.desc(), ModelVersion.id.desc())
            return [_record(row) for row in session.scalars(query)]

    def snapshot_registered_models(self) -> dict[int, str]:
        """Attach snapshots to explicitly registered local NEP entries in place.

        Existing snapshots remain immutable. A missing/remote/invalid path remains
        a project entry, with a reason it cannot yet appear in validation.
        """
        unavailable = {}
        with self.db.session() as session:
            rows = session.scalars(select(ModelVersion).where(
                ModelVersion.content_hash.is_(None), ModelVersion.project_id.is_not(None)
            )).all()
            for row in rows:
                try:
                    if row.model_type.upper() != "NEP":
                        raise ValueError("验证目前仅支持 NEP 模型。")
                    source = row.model_path or ""
                    if not source or "://" in source:
                        raise ValueError("请提供本地 NEP 文件或包含 nep.txt 的目录。")
                    path = Path(source).expanduser()
                    if path.is_dir():
                        path = path / "nep.txt"
                    draft = ModelDraft.from_file(path)
                    kind, elements = parse_model_header(draft.data)
                    digest = self.library.put_bytes(draft.data)
                except (OSError, ValueError) as error:
                    unavailable[row.id] = str(error)
                    continue
                row.content_hash = digest
                row.model_kind = kind
                row.elements = list(elements)
                row.source_path = str(path.resolve())
            session.commit()
        return unavailable

    def star_model(self, model_id: int, starred: bool = True) -> None:
        with self.db.session() as session:
            row = session.get(ModelVersion, model_id)
            if row is None:
                raise ValueError(f"Unknown model version: {model_id}")
            if row.project_id is None:
                raise ValueError("Save this temporary model to a project first.")
            row.starred = starred
            if starred:
                row.auto_captured = False
            session.commit()

    def model_file(self, model_id: int) -> Path:
        model = self.get_model(model_id)
        if not model["content_hash"]:
            raise ValueError("This model version has no snapshot; capture its nep.txt first.")
        return self.library.open_object(model["content_hash"])

    @staticmethod
    def capabilities(model: dict) -> ModelCapabilities:
        kind = model.get("model_kind") or ""
        return ModelCapabilities(kind, tuple(model.get("elements") or ()), is_spin="spin" in kind)

    # -- benchmark sets -------------------------------------------------------
    def import_set(
        self,
        path: str | Path,
        *,
        name: str,
        provenance: dict,
        protocol: str | None = None,
        max_bytes: int = DEFAULT_MAX_SET_BYTES,
        allow_large: bool = False,
    ) -> dict:
        """Seal an EXTXYZ with optional provenance; never infer that labels are DFT."""
        return self._import_sets(
            path, name=name, provenance=provenance, protocol=protocol,
            max_bytes=max_bytes, allow_large=allow_large,
        )[0]

    def import_eos_groups(
        self, path: str | Path, *, name: str, provenance: dict,
        group_field: str, group_names: list[str],
        max_bytes: int = DEFAULT_MAX_SET_BYTES, allow_large: bool = False,
        expected_hash: str | None = None,
    ) -> list[dict]:
        """Seal selected scans atomically, keeping the original file and frame indices."""
        if not group_field or not group_names or len(set(group_names)) != len(group_names):
            raise ValueError("Choose a grouping field and at least one distinct EOS group.")
        return self._import_sets(
            path, name=name, provenance=provenance, protocol="eos",
            max_bytes=max_bytes, allow_large=allow_large,
            group_field=group_field, group_names=group_names, expected_hash=expected_hash,
        )

    def _import_sets(
        self, path, *, name, provenance, protocol, max_bytes, allow_large,
        group_field=None, group_names=None, expected_hash=None,
    ) -> list[dict]:
        if not name.strip():
            raise ValueError("A benchmark set needs a name.")
        if provenance.get("source_kind") and provenance["source_kind"] not in SOURCE_KINDS:
            raise ValueError(f"Declare the reference source kind: one of {', '.join(SOURCE_KINDS)}.")
        path = Path(path).expanduser().resolve()
        size = path.stat().st_size
        if size > max_bytes and not allow_large:
            raise SetTooLargeError(size, max_bytes)
        data = path.read_bytes()
        if expected_hash is not None and sha256_bytes(data) != expected_hash:
            raise ValueError("The reference file changed. Load it again and review the new curve.")
        frames = read_frames(data)
        if protocol is None:
            ranked = detect_protocol(frames)
            if not ranked:
                raise ValueError("No validation protocol recognises this file.")
            chosen = ranked[0][1]
        else:
            chosen = get_protocol(protocol)
        if group_field is None:
            selections = [(name.strip(), frames, None)]
        else:
            groups = eos_frame_groups(frames, group_field)
            if any(group not in groups for group in group_names):
                raise ValueError("A selected EOS group is no longer present in the file.")
            selections = [
                (f"{name.strip()} / {group}", [frames[i] for i in groups[group]],
                 {"field": group_field, "value": group, "indices": groups[group]})
                for group in group_names
            ]
        # Validate every selected group before writing any benchmark records.
        prepared = []
        for set_name, subset, selection in selections:
            issues = chosen.check(subset)
            if any(issue.level == "error" for issue in issues):
                raise SetRejectedError([issue for issue in issues if issue.level == "error"])
            summary = chosen.summarize_reference(subset)
            summary["warnings"] = [issue.message for issue in issues if issue.level == "warning"]
            elements = sorted({symbol for atoms in subset for symbol in atoms.get_chemical_symbols()})
            prepared.append((set_name, summary, elements, selection))
        object_hash = self.library.put_bytes(data)
        source_ref = FileRef.of(path).to_dict()
        clean_provenance = {key: value for key, value in sorted(provenance.items()) if value not in ("", None)}
        with self.db.session() as session:
            records = []
            for set_name, summary, elements, selection in prepared:
                identity = [object_hash, chosen.kind, chosen.version, clean_provenance]
                reference = dict(source_ref)
                if selection is not None:
                    identity.append(selection)
                    reference["eos_group"] = selection
                key = sha256_bytes(json.dumps(identity, sort_keys=True).encode())
                row = session.get(BenchmarkSet, key)
                if row is None:
                    row = BenchmarkSet(
                        id=key, name=set_name, protocol=chosen.kind,
                        protocol_version=chosen.version, object_hash=object_hash,
                        source_ref=reference, elements=elements,
                        provenance=clean_provenance, summary=summary,
                    )
                    session.add(row)
                records.append(row)
            session.commit()
            return [_record(row) for row in records]

    def get_set(self, set_id: str) -> dict:
        if set_id in self._temporary_sets:
            return dict(self._temporary_sets[set_id])
        with self.db.session() as session:
            row = session.get(BenchmarkSet, set_id)
            if row is None:
                raise ValueError(f"Unknown benchmark set: {set_id}")
            return _record(row)

    def sets(self, *, protocol: str | None = None, include_archived: bool = False) -> list[dict]:
        with self.db.session() as session:
            query = select(BenchmarkSet)
            if protocol:
                query = query.where(BenchmarkSet.protocol == protocol)
            if not include_archived:
                query = query.where(BenchmarkSet.archived.is_(False))
            return [_record(row) for row in session.scalars(query.order_by(BenchmarkSet.created_at.desc()))]

    def set_frames(self, set_id: str):
        if set_id in self._temporary_data:
            return read_frames(self._temporary_data[set_id])
        benchmark = self.get_set(set_id)
        frames = read_frames(self.library.open_object(benchmark["object_hash"]).read_bytes())
        selection = (benchmark.get("source_ref") or {}).get("eos_group")
        if selection is not None:
            frames = [frames[index] for index in selection["indices"]]
        return frames

    def matching_sets(self, model_id: int) -> list[dict]:
        """Return the sets whose elements the model covers."""
        elements = set(self.get_model(model_id)["elements"] or ())
        return [item for item in self.sets() if set(item["elements"]) <= elements]

    def create_diagnostic_plan(self, name, experiments, *, temporary=False):
        """Create a saved plan by default; session-only trials require explicit opt-in."""
        from .protocols.diagnostic import encode_experiment
        if not name.strip() or not experiments:
            raise ValueError("请填写方案名称并选择至少一项实验。")
        prepared = []
        protocol = get_protocol("diagnostic")
        for seed, spec in experiments:
            data = encode_experiment(seed, spec)
            frames = read_frames(data)
            digest = sha256_bytes(data)
            summary = protocol.summarize_reference(frames)
            key = "temporary:" + digest
            record = dict(id=key, name=summary["formula"] + " / " + summary["title"],
                          protocol="diagnostic", protocol_version=protocol.version,
                          object_hash=digest, source_ref={}, provenance={}, summary=summary,
                          elements=sorted(set(seed.get_chemical_symbols())), archived=False)
            prepared.append((record, data))
        suite_id = min([0, *self._temporary_suites]) - 1
        suite = dict(id=suite_id, name=name.strip() + " · 临时", notes="无参考诊断", baseline_model_id=None,
                     baseline_note="", temporary=True, set_ids=list(dict.fromkeys(r["id"] for r, _ in prepared)))
        existing_keys = set(self._temporary_sets)
        for record, data in prepared:
            self._temporary_sets[record["id"]] = record
            self._temporary_data[record["id"]] = data
        self._temporary_suites[suite_id] = suite
        if temporary:
            return dict(suite)
        try:
            return self.save_diagnostic_plan(suite_id, name)
        finally:
            # Failed saves leave the editor available to retry, not a ghost draft
            # in the selector. Shared data belonging to other trials is untouched.
            self._temporary_suites.pop(suite_id, None)
            for key in suite["set_ids"]:
                if key not in existing_keys:
                    self._temporary_sets.pop(key, None)
                    self._temporary_data.pop(key, None)

    def save_diagnostic_plan(self, suite_id, name):
        """Save as a new plan atomically; existing plans and baselines are never overwritten."""
        if suite_id not in self._temporary_suites or not name.strip():
            raise ValueError("请选择临时诊断方案并填写新名称。")
        records = self.suite_sets(suite_id)
        with self.db.session() as session:
            if session.scalars(select(BenchmarkSuite).where(BenchmarkSuite.name == name.strip())).first():
                raise ValueError("已有同名方案，请使用新名称。")
            baseline_id = self._temporary_suites[suite_id]["baseline_model_id"]
            suite = BenchmarkSuite(name=name.strip(), notes="无参考诊断", baseline_model_id=baseline_id,
                                   baseline_note=self._temporary_suites[suite_id]["baseline_note"])
            if baseline_id is not None:
                session.get(ModelVersion, baseline_id).auto_captured = False
            session.add(suite)
            session.flush()
            for index, item in enumerate(records):
                digest = self.library.put_bytes(self._temporary_data[item["id"]])
                key = sha256_bytes(json.dumps([digest, "diagnostic", item["protocol_version"]]).encode())
                if session.get(BenchmarkSet, key) is None:
                    session.add(BenchmarkSet(id=key, name=item["name"], protocol="diagnostic",
                        protocol_version=item["protocol_version"], object_hash=digest, source_ref={},
                        provenance={}, summary=item["summary"], elements=item["elements"]))
                    session.flush()
                session.add(BenchmarkSuiteSet(suite_id=suite.id, set_id=key, position=index))
                # Keep already computed results when an explicitly saved experiment has the same bytes.
                for (model_id, set_id), run in self._temporary_runs.items():
                    if set_id != item["id"]:
                        continue
                    run_id = uuid.uuid4().hex
                    result_path = self.library.write_result(run_id, self.run_details(run)) if run["status"] == "succeeded" else ""
                    session.add(BenchmarkRun(id=run_id, model_id=model_id, set_id=key,
                        protocol_version=run["protocol_version"], status=run["status"], metrics=run["metrics"],
                        error=run["error"], environment=run["environment"], result_path=result_path,
                        elapsed_seconds=run["elapsed_seconds"]))
            session.commit()
            return _record(suite)

    # -- suites ---------------------------------------------------------------
    def create_suite(self, name: str, set_ids: Iterable[str] = (), *, notes: str = "") -> dict:
        with self.db.session() as session:
            suite = BenchmarkSuite(name=name.strip(), notes=notes)
            session.add(suite)
            session.flush()
            suite_id = suite.id
            session.commit()
        self.add_to_suite(suite_id, set_ids)
        return self.get_suite(suite_id)

    def add_to_suite(self, suite_id: int, set_ids: Iterable[str]) -> None:
        with self.db.session() as session:
            existing = {
                row.set_id: row
                for row in session.scalars(select(BenchmarkSuiteSet).where(BenchmarkSuiteSet.suite_id == suite_id))
            }
            position = len(existing)
            for set_id in set_ids:
                if set_id in existing:
                    continue
                if session.get(BenchmarkSet, set_id) is None:
                    raise ValueError(f"Unknown benchmark set: {set_id}")
                session.add(BenchmarkSuiteSet(suite_id=suite_id, set_id=set_id, position=position))
                position += 1
            session.commit()

    def get_suite(self, suite_id: int) -> dict:
        if suite_id in self._temporary_suites:
            return dict(self._temporary_suites[suite_id])
        with self.db.session() as session:
            suite = session.get(BenchmarkSuite, suite_id)
            if suite is None:
                raise ValueError(f"Unknown benchmark suite: {suite_id}")
            return _record(suite)

    def suites(self) -> list[dict]:
        with self.db.session() as session:
            return [*map(dict, self._temporary_suites.values()), *[_record(row) for row in session.scalars(select(BenchmarkSuite).order_by(BenchmarkSuite.name))]]

    def suite_sets(self, suite_id: int) -> list[dict]:
        if suite_id in self._temporary_suites:
            return [self.get_set(key) for key in self._temporary_suites[suite_id]["set_ids"]]
        with self.db.session() as session:
            rows = session.execute(
                select(BenchmarkSet)
                .join(BenchmarkSuiteSet, BenchmarkSuiteSet.set_id == BenchmarkSet.id)
                .where(BenchmarkSuiteSet.suite_id == suite_id)
                .order_by(BenchmarkSuiteSet.position)
            ).scalars()
            return [_record(row) for row in rows]

    def set_baseline(self, suite_id: int, model_id: int | None, note: str = "") -> None:
        if suite_id in self._temporary_suites:
            if model_id is not None and self.get_model(model_id)["project_id"] is None:
                raise ValueError("Save this temporary model to a project first.")
            self._temporary_suites[suite_id].update(baseline_model_id=model_id, baseline_note=note.strip())
            return
        with self.db.session() as session:
            suite = session.get(BenchmarkSuite, suite_id)
            if suite is None:
                raise ValueError(f"Unknown benchmark suite: {suite_id}")
            if model_id is not None and session.get(ModelVersion, model_id) is None:
                raise ValueError(f"Unknown model version: {model_id}")
            if model_id is not None and session.get(ModelVersion, model_id).project_id is None:
                raise ValueError("Save this temporary model to a project first.")
            suite.baseline_model_id = model_id
            suite.baseline_note = note.strip()
            if model_id is not None:
                session.get(ModelVersion, model_id).auto_captured = False  # never garbage-collect a baseline
            session.commit()

    # -- evaluation -----------------------------------------------------------
    def latest_run(self, model_id: int, set_id: str, *, protocol_version: int | None = None) -> dict | None:
        if set_id in self._temporary_sets:
            run = self._temporary_runs.get((model_id, set_id))
            if run and (protocol_version is None or run["protocol_version"] == protocol_version):
                return dict(run)
            return None
        with self.db.session() as session:
            model = session.get(ModelVersion, model_id)
            query = select(BenchmarkRun).join(ModelVersion, BenchmarkRun.model_id == ModelVersion.id).where(
                BenchmarkRun.set_id == set_id,
                ModelVersion.content_hash == model.content_hash if model and model.content_hash else BenchmarkRun.model_id == model_id,
            )
            if protocol_version is not None:
                query = query.where(BenchmarkRun.protocol_version == protocol_version)
            row = session.scalars(query.order_by(BenchmarkRun.created_at.desc())).first()
            return _record(row) if row is not None else None

    def evaluate(self, model_id: int, set_id: str, *, force: bool = False) -> dict:
        """Evaluate once per (model bytes, set, protocol version); failures are recorded."""
        model = self.get_model(model_id)
        benchmark = self.get_set(set_id)
        protocol = get_protocol(benchmark["protocol"])
        cached = None
        if not force:
            cached = self.latest_run(model_id, set_id, protocol_version=protocol.version)
            if cached is not None and protocol.kind != "diagnostic" and (cached["status"] == "unsupported" or
                                       cached["status"] == "succeeded" and _finite_metrics(cached["metrics"])):
                return cached
        run_id = uuid.uuid4().hex
        start = time.monotonic()
        environment = {
            "protocol": protocol.kind,
            "requested_backend": self.backend,
            "mode": protocol.mode,
            "neptrainkit": _package_version("NepTrainKit"),
            "nep_adapters": _package_version("nep-adapters"),
            "ase": _package_version("ase"),
            "numpy": _package_version("numpy"),
        }
        status, metrics, error, result_path = "failed", {}, "", ""
        reason = protocol.supports(self.capabilities(model))
        if reason is None and not set(benchmark["elements"]) <= set(model["elements"] or ()):
            reason = "The model does not cover every element of this benchmark set."
        if reason is not None:
            status, error = "unsupported", reason
        else:
            predictor = None
            try:
                model_path = self.model_file(model_id)
                frames = self.set_frames(set_id)
                predictor = self.predictor_factory(model_path, self.backend)
                environment["backend"] = str(getattr(predictor, "backend", self.backend))
                if protocol.kind == "diagnostic":
                    environment["model_capabilities"] = sorted(getattr(getattr(predictor, "model_info", None), "capability_names", ()))
                    # Resolve Auto before cache reuse: CPU and CUDA are different experiments.
                    if cached and cached["status"] == "succeeded" and _finite_metrics(cached["metrics"]) and cached["environment"] == environment:
                        return cached
                evaluation = protocol.evaluate(predictor, frames)
                if not _finite_metrics(evaluation.metrics):
                    raise ValueError("Validation metrics must be finite.")
                metrics = evaluation.metrics
                result = {"details": evaluation.details, "warnings": evaluation.warnings}
                if set_id in self._temporary_sets:
                    self._temporary_results[run_id] = result
                else:
                    result_path = self.library.write_result(run_id, result)
                status = "succeeded"
            except Exception as exc:
                from .protocols.diagnostic import UnsupportedDiagnostic
                if isinstance(exc, UnsupportedDiagnostic):
                    status = "unsupported"
                error = str(exc) or type(exc).__name__
            finally:
                close = getattr(predictor, "close", None)
                if callable(close):
                    close()
        if set_id in self._temporary_sets:
            run = dict(id=run_id, model_id=model_id, set_id=set_id, protocol_version=protocol.version,
                       status=status, metrics=metrics, result_path="", error=error, environment=environment,
                       elapsed_seconds=time.monotonic() - start)
            self._temporary_runs[(model_id, set_id)] = run
            return dict(run)
        with self.db.session() as session:
            row = BenchmarkRun(
                id=run_id,
                model_id=model_id,
                set_id=set_id,
                protocol_version=protocol.version,
                status=status,
                metrics=metrics,
                result_path=result_path,
                error=error,
                environment=environment,
                elapsed_seconds=time.monotonic() - start,
            )
            session.add(row)
            session.commit()
            return _record(row)

    def run_suite(self, model_id: int, suite_id: int, *, force: bool = False, progress=None) -> list[dict]:
        sets = self.suite_sets(suite_id)
        runs = []
        for index, benchmark in enumerate(sets):
            if progress is not None:
                progress(index, len(sets), benchmark)
            runs.append(self.evaluate(model_id, benchmark["id"], force=force))
        return runs

    def run_details(self, run: dict) -> dict:
        if run["id"] in self._temporary_results:
            return self._temporary_results[run["id"]]
        return self.library.read_result(run.get("result_path", ""))

    # -- scorecard ------------------------------------------------------------
    def scorecard(
        self, suite_id: int | None, model_ids: Iterable[int] | None = None, *, tolerance: float = DEFAULT_TOLERANCE,
        draft: ModelDraft | None = None,
    ) -> dict:
        """Headline metrics of each model on each set, compared with the suite baseline.

        Only cached runs are read; call :meth:`run_suite` first to fill gaps.
        """
        suite = self.get_suite(suite_id) if suite_id is not None else None
        sets = self.suite_sets(suite_id) if suite_id is not None else []
        columns = []
        for benchmark in sets:
            protocol = get_protocol(benchmark["protocol"])
            for metric in protocol.headline_metrics():
                columns.append(
                    {
                        "set_id": benchmark["id"],
                        "set_name": benchmark["name"],
                        "protocol": protocol.kind,
                        "protocol_version": protocol.version,
                        "metric": metric.key,
                        "label": metric.label,
                        "unit": metric.unit,
                        "better": metric.better,
                        "abs_tolerance": metric.abs_tolerance,
                    }
                )
        baseline_id = suite["baseline_model_id"] if suite else None
        if model_ids is None:
            with self.db.session() as session:
                set_ids = [benchmark["id"] for benchmark in sets]
                model_ids = list(
                    session.scalars(
                        select(BenchmarkRun.model_id).where(BenchmarkRun.set_id.in_(set_ids)).distinct()
                    )
                )
            # Show compatible snapshots before their first run, including newly opened models.
            model_ids.extend(
                model["id"] for model in self.models()
                if not sets or any(set(benchmark["elements"]) <= set(model["elements"] or ()) for benchmark in sets)
            )
        model_ids = list(dict.fromkeys([*(model_ids or [])] + ([baseline_id] if baseline_id else [])))

        def cells_for(model_id):
            runs = {}
            cells = []
            for column in columns:
                key = column["set_id"]
                if key not in runs:
                    runs[key] = self.latest_run(model_id, key, protocol_version=column["protocol_version"])
                    run = runs[key]
                    if run and run["status"] == "succeeded" and not _finite_metrics(run["metrics"]):
                        # Old libraries may contain successful runs with NaN/Inf metrics.
                        runs[key] = dict(run, status="failed", error="Validation metrics must be finite.")
                run = runs[key]
                value = None
                if run is not None and run["status"] == "succeeded":
                    value = run["metrics"].get(column["metric"])
                cells.append(
                    {
                        "value": value,
                        "status": run["status"] if run else "missing",
                        "error": run["error"] if run else "",
                        "run_id": run["id"] if run else None,
                    }
                )
            return cells

        baseline_cells = cells_for(baseline_id) if baseline_id else None
        project_names = {project["id"]: project["name"] for project in self.projects()}
        rows = []
        for model_id in model_ids:
            model = self.get_model(model_id)
            cells = cells_for(model_id) if model_id != baseline_id else baseline_cells
            for index, cell in enumerate(cells):
                cell = dict(cell)
                base = baseline_cells[index]["value"] if baseline_cells else None
                cell["verdict"] = None if model_id == baseline_id else compare(
                    cell["value"], base, columns[index]["better"], tolerance, columns[index]["abs_tolerance"]
                )
                cells[index] = cell
            set_statuses = {column["set_id"]: cell["status"] for column, cell in zip(columns, cells)}
            complete = bool(sets) and all(status == "succeeded" for status in set_statuses.values())
            comparable = complete and baseline_cells is not None and all(
                base["status"] == "succeeded" and (
                    model_id == baseline_id or cell["verdict"] is not None or
                    cell["value"] is None and base["value"] is None
                )
                for cell, base in zip(cells, baseline_cells or [])
            )
            rows.append(
                {
                    "model_id": model_id,
                    "name": model["name"],
                    "project_name": project_names.get(model["project_id"], ""),
                    "temporary": model["project_id"] is None,
                    "is_baseline": model_id == baseline_id,
                    "starred": model["starred"],
                    "cells": cells,
                    "regressions": sum(1 for cell in cells if cell.get("verdict") == "worse"),
                    "total_sets": len(sets),
                    "completed_sets": sum(status == "succeeded" for status in set_statuses.values()),
                    "failed_sets": sum(status == "failed" for status in set_statuses.values()),
                    "complete": complete,
                    "comparison_complete": comparable,
                }
            )
        if draft is not None:
            rows.append({
                "model_id": 0, "name": draft.name, "project_name": "", "temporary": True,
                "is_baseline": False, "starred": False, "regressions": 0,
                "cells": [{"value": None, "status": "missing", "error": "", "run_id": None, "verdict": None} for _ in columns],
                "total_sets": len(sets), "completed_sets": 0, "failed_sets": 0,
                "complete": False, "comparison_complete": False,
            })
        return {"suite": suite, "columns": columns, "rows": rows, "tolerance": tolerance}

    # -- housekeeping ---------------------------------------------------------
    def collect_garbage(self, *, older_than_days: int = 30, dry_run: bool = True) -> dict:
        """Forget stale auto-captured versions and delete unreferenced library files.

        Starred models, baselines and manually captured models are never removed.
        """
        cutoff = _now() - dt.timedelta(days=older_than_days)
        with self.db.session() as session:
            baselines = set(
                session.scalars(
                    select(BenchmarkSuite.baseline_model_id).where(BenchmarkSuite.baseline_model_id.is_not(None))
                )
            )
            stale = [
                row
                for row in session.scalars(
                    select(ModelVersion).where(
                        ModelVersion.auto_captured.is_(True),
                        ModelVersion.starred.is_(False),
                        ModelVersion.last_seen_at < cutoff,
                    )
                )
                if row.id not in baselines and not row.children
            ]
            stale_ids = [row.id for row in stale]
            keep_models = set(
                session.scalars(
                    select(ModelVersion.content_hash).where(
                        ModelVersion.content_hash.is_not(None), ModelVersion.id.not_in(stale_ids)
                    )
                )
            )
            keep_sets = set(session.scalars(select(BenchmarkSet.object_hash)))
            keep_sets |= set(
                session.scalars(select(StorageRef.content_hash).where(StorageRef.scheme == "cas://"))
            )
            keep_results = set(
                session.scalars(
                    select(BenchmarkRun.result_path).where(BenchmarkRun.model_id.not_in(stale_ids))
                )
            )
            if not dry_run and stale_ids:
                session.execute(delete(BenchmarkRun).where(BenchmarkRun.model_id.in_(stale_ids)))
                for row in stale:
                    session.delete(row)
                session.commit()
        keep = keep_models | keep_sets
        orphan_objects = [(digest, path) for digest, path in self.library.iter_objects() if digest not in keep]
        orphan_results = [
            path
            for path in self.library.results_dir.glob("*.json")
            if f"results/{path.name}" not in keep_results
        ]
        freed = sum(path.stat().st_size for _, path in orphan_objects) + sum(p.stat().st_size for p in orphan_results)
        if not dry_run:
            for digest, _ in orphan_objects:
                self.library.delete_object(digest)
            for path in orphan_results:
                path.unlink(missing_ok=True)
        return {
            "models": len(stale_ids),
            "objects": len(orphan_objects),
            "results": len(orphan_results),
            "bytes": freed,
            "dry_run": dry_run,
        }
