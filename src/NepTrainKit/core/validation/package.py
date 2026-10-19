"""Portable validation suites. Preview is read-only; import creates a new suite.

Version 1 is a ZIP with a JSON manifest and hash-addressed objects. Reference
frames retain their original EXTXYZ text; model predictions are never imported.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile

from sqlalchemy import select

from NepTrainKit.core.dataset.models import BenchmarkSet, BenchmarkSuite, BenchmarkSuiteSet, ModelVersion, Project
from .library import sha256_bytes
from .protocols import get_protocol, read_frames
from .service import parse_model_header

FORMAT = "neptrainkit-validation"
VERSION = 1
MAX_BYTES = 512 * 1024 * 1024
MAX_MANIFEST = 2 * 1024 * 1024


@dataclass(frozen=True)
class PackagePreview:
    data: bytes
    manifest: dict
    references: tuple[dict, ...]
    warnings: tuple[str, ...]


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _text(value, label, limit=10000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"Invalid package {label}.")
    return value


def _selected_frames(data, indices):
    """Select losslessly, retaining magnetic/custom columns and numeric precision."""
    lines = data.splitlines(keepends=True)
    blocks, offset = [], 0
    while offset < len(lines):
        if not lines[offset].strip():
            offset += 1
            continue
        count = int(lines[offset].strip())
        end = offset + count + 2
        if count < 1 or end > len(lines):
            raise ValueError("Invalid EXTXYZ frame.")
        blocks.append(b"".join(lines[offset:end]).rstrip(b"\r\n") + b"\n")
        offset = end
    return b"".join(blocks[index] for index in indices)


def build_package(service, suite_id: int, model_ids=()) -> PackagePreview:
    suite = service.get_suite(suite_id)
    refs, models, objects = [], [], {}
    baseline = None
    for row in service.suite_sets(suite_id):
        data = service.library.open_object(row["object_hash"]).read_bytes()
        group = row["source_ref"].get("eos_group")
        if group:
            data = _selected_frames(data, group["indices"])
        digest = sha256_bytes(data)
        objects[digest] = data
        refs.append(dict(name=row["name"], protocol=row["protocol"],
                         protocol_version=row["protocol_version"], object=digest,
                         provenance=row["provenance"],
                         group={k: group[k] for k in ("field", "value")} if group else None))
    for model_id in dict.fromkeys(model_ids):
        row = service.get_model(model_id)
        data = service.model_file(model_id).read_bytes()
        digest = sha256_bytes(data)
        objects[digest] = data
        if not any(m["object"] == digest for m in models):
            models.append(dict(name=row["name"], object=digest))
        if model_id == suite["baseline_model_id"]:
            baseline = digest
    manifest = dict(format=FORMAT, version=VERSION,
                    suite=dict(name=suite["name"], notes=suite["notes"], baseline=baseline),
                    references=refs, models=models)
    if sum(len(data) for data in objects.values()) + len(_json(manifest)) > MAX_BYTES:
        raise ValueError("Unpacked validation package exceeds 512 MiB.")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", _json(manifest))
        for digest, data in objects.items():
            archive.writestr(f"objects/{digest}", data)
    return _inspect(buffer.getvalue())


def export_package(preview: PackagePreview, path):
    target = Path(path)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_bytes(preview.data)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_package(path) -> PackagePreview:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Validation package exceeds 512 MiB.")
    return _inspect(path.read_bytes())


def _inspect(data):
    try:
        return _inspect_checked(data)
    except (KeyError, TypeError, AttributeError, UnicodeError, zipfile.BadZipFile, RuntimeError, EOFError, IndexError, AssertionError, NotImplementedError) as error:
        raise ValueError(f"Invalid validation package: {error}") from error


def _inspect_checked(data):
    if len(data) > MAX_BYTES:
        raise ValueError("Validation package exceeds 512 MiB.")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or len(names) > 10000:
            raise ValueError("Duplicate or excessive package entries.")
        if sum(entry.file_size for entry in entries) > MAX_BYTES:
            raise ValueError("Unpacked validation package exceeds 512 MiB.")
        if archive.getinfo("manifest.json").file_size > MAX_MANIFEST:
            raise ValueError("Package manifest is too large.")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["format"] != FORMAT or manifest["version"] != VERSION:
            raise ValueError("Unsupported validation package format/version.")
        suite = manifest["suite"]
        _text(suite["name"], "suite name", 200)
        if not isinstance(suite.get("notes", ""), str):
            raise ValueError("Invalid suite notes.")
        refs, models = manifest["references"], manifest["models"]
        if not isinstance(refs, list) or not refs or not isinstance(models, list):
            raise ValueError("A package needs reference datasets.")
        objects = {}
        for row in refs + models:
            digest = row["object"]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid object checksum.")
            if digest not in objects:
                payload = archive.read(f"objects/{digest}")
                if sha256_bytes(payload) != digest:
                    raise ValueError("Package object checksum mismatch.")
                objects[digest] = payload
        if set(names) != {"manifest.json", *(f"objects/{digest}" for digest in objects)}:
            raise ValueError("Unexpected files in validation package.")
        if len({_json(row) for row in refs}) != len(refs):
            raise ValueError("Duplicate reference dataset in package.")
        summaries, warnings = [], []
        for row in refs:
            _text(row["name"], "reference name")
            protocol = get_protocol(row["protocol"])
            if row["protocol_version"] != protocol.version:
                raise ValueError(f"Unsupported protocol version: {protocol.kind} {row['protocol_version']}")
            if not isinstance(row.get("provenance", {}), dict):
                raise ValueError("Invalid reference metadata.")
            group = row.get("group")
            if group is not None:
                _text(group["field"], "group field")
                _text(group["value"], "group value")
            frames = read_frames(objects[row["object"]])
            issues = protocol.check(frames)
            errors = [issue.message for issue in issues if issue.level == "error"]
            if errors:
                raise ValueError(f"{row['name']}: {'; '.join(errors)}")
            messages = [issue.message for issue in issues if issue.level == "warning"]
            warnings.extend(f"{row['name']}: {message}" for message in messages)
            summary = protocol.summarize_reference(frames)
            summary["warnings"] = messages
            summaries.append(dict(summary=summary, frame_count=len(frames),
                                  elements=sorted({symbol for frame in frames for symbol in frame.get_chemical_symbols()})))
        hashes = []
        for row in models:
            _text(row["name"], "model name", 80)
            parse_model_header(objects[row["object"]])
            hashes.append(row["object"])
        if len(hashes) != len(set(hashes)):
            raise ValueError("Duplicate model in package.")
        if suite.get("baseline") is not None and suite["baseline"] not in hashes:
            raise ValueError("Package baseline model is missing.")
    return PackagePreview(data, manifest, tuple(summaries), tuple(warnings))


def available_suite_name(service, name):
    names = {suite["name"] for suite in service.suites()}
    candidate, index = name[:200], 2
    while candidate in names:
        suffix = f" ({index})"
        candidate = name[:200-len(suffix)] + suffix
        index += 1
    return candidate


def import_package(service, preview: PackagePreview, *, name: str, project_name=""):
    # Revalidate the immutable bytes, not mutable UI metadata or a replaced file.
    preview = _inspect(preview.data)
    name = _text(name.strip(), "suite name", 200)
    manifest = preview.manifest
    if manifest["models"]:
        project_name = _text(project_name.strip(), "project name", 200)
    with service.db.session() as session, zipfile.ZipFile(io.BytesIO(preview.data)) as archive:
        models, project = {}, None
        if manifest["models"]:
            project = session.scalar(select(Project).where(Project.name == project_name))
            if project is None:
                project = Project(name=project_name)
                session.add(project)
                session.flush()
        for row in manifest["models"]:
            digest = row["object"]
            payload = archive.read(f"objects/{digest}")
            service.library.put_bytes(payload)
            model = session.scalar(select(ModelVersion).where(
                ModelVersion.project_id == project.id, ModelVersion.content_hash == digest))
            if model is None:
                kind, elements = parse_model_header(payload)
                model = ModelVersion(name=row["name"], project_id=project.id, model_type="NEP",
                    model_path=str(service.library.object_path(digest)), content_hash=digest,
                    model_kind=kind, elements=elements)
                session.add(model)
                session.flush()
            models[digest] = model.id
        # Only this newly inserted suite may receive the shared baseline.
        suite = BenchmarkSuite(name=available_suite_name(service, name), notes=manifest["suite"].get("notes", ""),
                               baseline_model_id=models.get(manifest["suite"].get("baseline")))
        session.add(suite)
        session.flush()
        added = set()
        for position, (row, checked) in enumerate(zip(manifest["references"], preview.references)):
            digest = row["object"]
            service.library.put_bytes(archive.read(f"objects/{digest}"))
            identity = sha256_bytes(_json(["validation-package-v1", row]))
            benchmark = session.get(BenchmarkSet, identity)
            if benchmark is None:
                reference = {}
                if row.get("group"):
                    reference["eos_group"] = dict(row["group"], indices=list(range(checked["frame_count"])))
                benchmark = BenchmarkSet(id=identity, name=row["name"], protocol=row["protocol"],
                    protocol_version=row["protocol_version"], object_hash=digest, source_ref=reference,
                    provenance=row.get("provenance", {}), summary=checked["summary"], elements=checked["elements"])
                session.add(benchmark)
                session.flush()
            if identity not in added:
                session.add(BenchmarkSuiteSet(suite_id=suite.id, set_id=identity, position=position))
                added.add(identity)
        suite_id = suite.id
        session.commit()
    return service.get_suite(suite_id)
