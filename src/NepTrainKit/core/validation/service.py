"""Model capture, benchmark sets, cached evaluation and the scorecard.

Everything here is UI-free so the GUI, the command line and tests share it.
"""

from __future__ import annotations

import datetime as dt
import json
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
    if value is None or baseline is None:
        return None
    if better == "closer-to-zero":
        value, baseline, better = abs(value), abs(baseline), "lower"
    scale = max(abs(baseline), 1e-12)
    if abs(value - baseline) <= max(tolerance * scale, abs_tolerance):
        return "same"
    improved = value < baseline if better == "lower" else value > baseline
    return "better" if improved else "worse"


@dataclass
class ValidationService:
    library: Library | None = None
    predictor_factory: Callable[[Path, str], object] | None = None
    backend: str = "auto"

    def __post_init__(self):
        self.library = self.library or get_library()
        self.db = self.library.database
        self.predictor_factory = self.predictor_factory or _nep_predictor

    # -- models ---------------------------------------------------------------
    def _inbox_project_id(self, session) -> int:
        project = session.scalars(select(Project).where(Project.name == INBOX_PROJECT)).first()
        if project is None:
            project = Project(name=INBOX_PROJECT, notes="Models captured automatically from the viewer.")
            session.add(project)
            session.flush()
        return project.id

    def capture_model(
        self,
        path: str | Path,
        *,
        name: str = "",
        dataset_path: str | Path | None = None,
        auto: bool = True,
        project_id: int | None = None,
        notes: str = "",
    ) -> dict:
        """Snapshot a model file; capturing identical bytes again returns the same version."""
        path = Path(path).expanduser().resolve()
        data = path.read_bytes()
        kind, elements = parse_model_header(data)
        digest = self.library.put_bytes(data)
        dataset_ref = {}
        if dataset_path and Path(dataset_path).is_file():
            dataset_ref = FileRef.of(dataset_path).to_dict()
        with self.db.session() as session:
            row = session.scalars(
                select(ModelVersion).where(ModelVersion.content_hash == digest).order_by(ModelVersion.id)
            ).first()
            if row is None:
                default_name = path.parent.name if path.stem == "nep" else path.stem
                row = ModelVersion(
                    name=(name.strip() or default_name or path.name)[:80],
                    project_id=project_id or self._inbox_project_id(session),
                    model_type="NEP",
                    model_path=str(path),
                    notes=notes,
                    content_hash=digest,
                    model_kind=kind,
                    elements=list(elements),
                    source_path=str(path),
                    dataset_ref=dataset_ref,
                    auto_captured=auto,
                )
                session.add(row)
            else:
                row.last_seen_at = _now()
                if not auto:
                    row.auto_captured = False
                if name.strip():
                    row.name = name.strip()[:80]
                if dataset_ref and not row.dataset_ref:
                    row.dataset_ref = dataset_ref
            session.commit()
            return _record(row)

    def find_model_by_file(self, path: str | Path) -> dict | None:
        """Return the captured version whose bytes equal ``path`` right now, if any."""
        try:
            digest = sha256_bytes(Path(path).read_bytes())
        except OSError:
            return None
        with self.db.session() as session:
            row = session.scalars(select(ModelVersion).where(ModelVersion.content_hash == digest)).first()
            return _record(row) if row is not None else None

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

    def star_model(self, model_id: int, starred: bool = True) -> None:
        with self.db.session() as session:
            row = session.get(ModelVersion, model_id)
            if row is None:
                raise ValueError(f"Unknown model version: {model_id}")
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
        """Seal an EXTXYZ and its declared conditions; never infer that labels are DFT."""
        if not name.strip():
            raise ValueError("A benchmark set needs a name.")
        if provenance.get("source_kind") not in SOURCE_KINDS:
            raise ValueError(f"Declare the reference source kind: one of {', '.join(SOURCE_KINDS)}.")
        path = Path(path).expanduser().resolve()
        size = path.stat().st_size
        if size > max_bytes and not allow_large:
            raise SetTooLargeError(size, max_bytes)
        data = path.read_bytes()
        frames = read_frames(data)
        if protocol is None:
            ranked = detect_protocol(frames)
            if not ranked:
                raise ValueError("No validation protocol recognises this file.")
            chosen = ranked[0][1]
        else:
            chosen = get_protocol(protocol)
        issues = chosen.check(frames)
        if any(issue.level == "error" for issue in issues):
            raise SetRejectedError([issue for issue in issues if issue.level == "error"])
        object_hash = self.library.put_bytes(data)
        clean_provenance = {key: value for key, value in sorted(provenance.items()) if value not in ("", None)}
        key = sha256_bytes(
            json.dumps(
                [object_hash, chosen.kind, chosen.version, clean_provenance], sort_keys=True
            ).encode()
        )
        summary = chosen.summarize_reference(frames)
        summary["warnings"] = [issue.message for issue in issues if issue.level == "warning"]
        elements = sorted({symbol for atoms in frames for symbol in atoms.get_chemical_symbols()})
        with self.db.session() as session:
            row = session.get(BenchmarkSet, key)
            if row is None:
                row = BenchmarkSet(
                    id=key,
                    name=name.strip(),
                    protocol=chosen.kind,
                    protocol_version=chosen.version,
                    object_hash=object_hash,
                    source_ref=FileRef.of(path).to_dict(),
                    elements=elements,
                    provenance=clean_provenance,
                    summary=summary,
                )
                session.add(row)
                session.commit()
            return _record(row)

    def get_set(self, set_id: str) -> dict:
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
        benchmark = self.get_set(set_id)
        return read_frames(self.library.open_object(benchmark["object_hash"]).read_bytes())

    def matching_sets(self, model_id: int) -> list[dict]:
        """Return the sets whose elements the model covers."""
        elements = set(self.get_model(model_id)["elements"] or ())
        return [item for item in self.sets() if set(item["elements"]) <= elements]

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
        with self.db.session() as session:
            suite = session.get(BenchmarkSuite, suite_id)
            if suite is None:
                raise ValueError(f"Unknown benchmark suite: {suite_id}")
            return _record(suite)

    def suites(self) -> list[dict]:
        with self.db.session() as session:
            return [_record(row) for row in session.scalars(select(BenchmarkSuite).order_by(BenchmarkSuite.name))]

    def suite_sets(self, suite_id: int) -> list[dict]:
        with self.db.session() as session:
            rows = session.execute(
                select(BenchmarkSet)
                .join(BenchmarkSuiteSet, BenchmarkSuiteSet.set_id == BenchmarkSet.id)
                .where(BenchmarkSuiteSet.suite_id == suite_id)
                .order_by(BenchmarkSuiteSet.position)
            ).scalars()
            return [_record(row) for row in rows]

    def set_baseline(self, suite_id: int, model_id: int | None, note: str = "") -> None:
        with self.db.session() as session:
            suite = session.get(BenchmarkSuite, suite_id)
            if suite is None:
                raise ValueError(f"Unknown benchmark suite: {suite_id}")
            if model_id is not None and session.get(ModelVersion, model_id) is None:
                raise ValueError(f"Unknown model version: {model_id}")
            suite.baseline_model_id = model_id
            suite.baseline_note = note.strip()
            if model_id is not None:
                session.get(ModelVersion, model_id).auto_captured = False  # never garbage-collect a baseline
            session.commit()

    # -- evaluation -----------------------------------------------------------
    def latest_run(self, model_id: int, set_id: str, *, protocol_version: int | None = None) -> dict | None:
        with self.db.session() as session:
            query = select(BenchmarkRun).where(BenchmarkRun.model_id == model_id, BenchmarkRun.set_id == set_id)
            if protocol_version is not None:
                query = query.where(BenchmarkRun.protocol_version == protocol_version)
            row = session.scalars(query.order_by(BenchmarkRun.created_at.desc())).first()
            return _record(row) if row is not None else None

    def evaluate(self, model_id: int, set_id: str, *, force: bool = False) -> dict:
        """Evaluate once per (model bytes, set, protocol version); failures are recorded."""
        model = self.get_model(model_id)
        benchmark = self.get_set(set_id)
        protocol = get_protocol(benchmark["protocol"])
        if not force:
            cached = self.latest_run(model_id, set_id, protocol_version=protocol.version)
            if cached is not None and cached["status"] in ("succeeded", "unsupported"):
                return cached
        run_id = uuid.uuid4().hex
        start = time.monotonic()
        environment = {
            "protocol": protocol.kind,
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
                evaluation = protocol.evaluate(predictor, frames)
                metrics = evaluation.metrics
                result_path = self.library.write_result(
                    run_id, {"details": evaluation.details, "warnings": evaluation.warnings}
                )
                status = "succeeded"
            except Exception as exc:
                error = str(exc) or type(exc).__name__
            finally:
                close = getattr(predictor, "close", None)
                if callable(close):
                    close()
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
        return self.library.read_result(run.get("result_path", ""))

    # -- scorecard ------------------------------------------------------------
    def scorecard(
        self, suite_id: int, model_ids: Iterable[int] | None = None, *, tolerance: float = DEFAULT_TOLERANCE
    ) -> dict:
        """Headline metrics of each model on each set, compared with the suite baseline.

        Only cached runs are read; call :meth:`run_suite` first to fill gaps.
        """
        suite = self.get_suite(suite_id)
        sets = self.suite_sets(suite_id)
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
        baseline_id = suite["baseline_model_id"]
        if model_ids is None:
            with self.db.session() as session:
                set_ids = [benchmark["id"] for benchmark in sets]
                model_ids = list(
                    session.scalars(
                        select(BenchmarkRun.model_id).where(BenchmarkRun.set_id.in_(set_ids)).distinct()
                    )
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
            rows.append(
                {
                    "model_id": model_id,
                    "name": model["name"],
                    "is_baseline": model_id == baseline_id,
                    "starred": model["starred"],
                    "cells": cells,
                    "regressions": sum(1 for cell in cells if cell.get("verdict") == "worse"),
                }
            )
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
