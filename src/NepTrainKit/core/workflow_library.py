"""Local reusable workflow definitions for the Make Dataset workbench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile
from typing import Any, Literal
from uuid import uuid4

from NepTrainKit.paths import ensure_directory, get_user_config_path


WorkflowKind = Literal["workflow", "template"]
_RUNTIME_KEYS = {
    "dataset",
    "result_dataset",
    "run_outcome",
    "runtime_state",
    "last_elapsed_seconds",
}


@dataclass(frozen=True)
class WorkflowEntry:
    """Metadata and configuration for one reusable workflow definition."""

    workflow_id: str
    name: str
    kind: WorkflowKind
    created_at: str
    updated_at: str
    workflow: dict[str, Any]

    @property
    def card_count(self) -> int:
        cards = self.workflow.get("cards", [])
        return len(cards) if isinstance(cards, list) else 0


class WorkflowLibrary:
    """Persist workflow configuration without datasets or runtime results."""

    schema = 1

    def __init__(self, root: Path | None = None):
        self.root = ensure_directory(
            root if root is not None else get_user_config_path() / "workflows"
        )
        self.workflows_dir = ensure_directory(self.root / "saved")
        self.templates_dir = ensure_directory(self.root / "templates")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _sanitise(value):
        if isinstance(value, dict):
            return {
                str(key): WorkflowLibrary._sanitise(item)
                for key, item in value.items()
                if str(key) not in _RUNTIME_KEYS
            }
        if isinstance(value, list):
            return [WorkflowLibrary._sanitise(item) for item in value]
        return value

    @classmethod
    def normalise_workflow(cls, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Workflow configuration must be a JSON object.")
        cards = payload.get("cards")
        if not isinstance(cards, list):
            raise ValueError("Workflow configuration must contain a card list.")
        return {
            "software_version": str(payload.get("software_version", "")),
            "workflow_schema": int(payload.get("workflow_schema", 2)),
            "cards": cls._sanitise(cards),
        }

    def _directory(self, kind: WorkflowKind) -> Path:
        if kind == "workflow":
            return self.workflows_dir
        if kind == "template":
            return self.templates_dir
        raise ValueError(f"Unknown workflow kind: {kind}")

    def _path(self, workflow_id: str, kind: WorkflowKind) -> Path:
        if not workflow_id or any(char not in "0123456789abcdef" for char in workflow_id):
            raise ValueError("Invalid workflow identifier.")
        return self._directory(kind) / f"{workflow_id}.json"

    @classmethod
    def _entry_from_record(cls, record: dict[str, Any]) -> WorkflowEntry:
        workflow = cls.normalise_workflow(record.get("workflow", {}))
        kind = str(record.get("kind", "workflow"))
        if kind not in ("workflow", "template"):
            raise ValueError("Invalid workflow kind.")
        return WorkflowEntry(
            workflow_id=str(record["id"]),
            name=str(record["name"]),
            kind=kind,
            created_at=str(record["created_at"]),
            updated_at=str(record["updated_at"]),
            workflow=workflow,
        )

    def _write_entry(self, entry: WorkflowEntry) -> None:
        record = {
            "library_schema": self.schema,
            "id": entry.workflow_id,
            "name": entry.name,
            "kind": entry.kind,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "workflow": entry.workflow,
        }
        path = self._path(entry.workflow_id, entry.kind)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def list(self, kind: WorkflowKind) -> list[WorkflowEntry]:
        entries = []
        for path in self._directory(kind).glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                entry = self._entry_from_record(record)
                if entry.kind == kind:
                    entries.append(entry)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return sorted(entries, key=lambda entry: entry.updated_at, reverse=True)

    def get(self, workflow_id: str, kind: WorkflowKind) -> WorkflowEntry:
        path = self._path(workflow_id, kind)
        record = json.loads(path.read_text(encoding="utf-8"))
        return self._entry_from_record(record)

    def save(
        self,
        name: str,
        workflow: dict[str, Any],
        *,
        kind: WorkflowKind = "workflow",
        workflow_id: str | None = None,
    ) -> WorkflowEntry:
        name = str(name).strip()
        if not name:
            raise ValueError("Workflow name cannot be empty.")
        normalised = self.normalise_workflow(workflow)
        now = self._now()
        if workflow_id is None:
            workflow_id = uuid4().hex
            created_at = now
        else:
            created_at = self.get(workflow_id, kind).created_at
        entry = WorkflowEntry(workflow_id, name, kind, created_at, now, normalised)
        self._write_entry(entry)
        return entry

    def rename(self, workflow_id: str, kind: WorkflowKind, name: str) -> WorkflowEntry:
        entry = self.get(workflow_id, kind)
        return self.save(name, entry.workflow, kind=kind, workflow_id=workflow_id)

    def duplicate(
        self,
        workflow_id: str,
        kind: WorkflowKind,
        *,
        name: str,
        target_kind: WorkflowKind | None = None,
    ) -> WorkflowEntry:
        entry = self.get(workflow_id, kind)
        return self.save(name, entry.workflow, kind=target_kind or kind)

    def delete(self, workflow_id: str, kind: WorkflowKind) -> None:
        self._path(workflow_id, kind).unlink()

    def import_file(
        self,
        path: Path,
        *,
        kind: WorkflowKind = "workflow",
        name: str | None = None,
    ) -> WorkflowEntry:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        workflow = record.get("workflow", record) if isinstance(record, dict) else record
        return self.save(name or Path(path).stem, workflow, kind=kind)

    def export_file(self, workflow_id: str, kind: WorkflowKind, path: Path) -> None:
        source = self._path(workflow_id, kind)
        copyfile(source, Path(path))


__all__ = ["WorkflowEntry", "WorkflowKind", "WorkflowLibrary"]
