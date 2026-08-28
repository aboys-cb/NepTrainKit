"""Local reusable workflow definitions for the Make Dataset workbench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from NepTrainKit import module_path
from NepTrainKit.paths import ensure_directory, get_user_config_path

WorkflowKind = Literal["workflow", "template"]
WorkflowOrigin = Literal["user", "builtin"]
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
    origin: WorkflowOrigin = "user"
    category: str = ""
    description: str = ""
    input_requirement: str = ""
    template_version: int = 1

    @property
    def card_count(self) -> int:
        cards = self.workflow.get("cards", [])
        return len(cards) if isinstance(cards, list) else 0

    @property
    def read_only(self) -> bool:
        """Return whether the entry is shipped with the application."""
        return self.origin == "builtin"


class WorkflowLibrary:
    """Persist workflow configuration without datasets or runtime results."""

    schema = 1
    builtin_package = "NepTrainKit.workflow_templates"

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
    def _entry_from_record(
        cls,
        record: dict[str, Any],
        *,
        origin: WorkflowOrigin = "user",
    ) -> WorkflowEntry:
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
            origin=origin,
            category=str(record.get("category", "")),
            description=str(record.get("description", "")),
            input_requirement=str(record.get("input_requirement", "")),
            template_version=int(record.get("template_version", 1)),
        )

    @classmethod
    def _record_for_entry(cls, entry: WorkflowEntry) -> dict[str, Any]:
        record = {
            "library_schema": cls.schema,
            "id": entry.workflow_id,
            "name": entry.name,
            "kind": entry.kind,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "workflow": entry.workflow,
        }
        if entry.category:
            record["category"] = entry.category
        if entry.description:
            record["description"] = entry.description
        if entry.input_requirement:
            record["input_requirement"] = entry.input_requirement
        if entry.kind == "template":
            record["template_version"] = entry.template_version
        return record

    @classmethod
    def _builtin_templates(cls) -> list[WorkflowEntry]:
        template_roots = []
        try:
            template_roots.append(resources.files(cls.builtin_package))
        except (ModuleNotFoundError, OSError):
            pass
        standalone_root = module_path / "workflow_templates"
        if standalone_root not in template_roots:
            template_roots.append(standalone_root)

        for template_root in template_roots:
            try:
                template_files = sorted(
                    (
                        item
                        for item in template_root.iterdir()
                        if item.name.endswith(".json")
                    ),
                    key=lambda item: item.name,
                )
            except OSError:
                continue
            entries = []
            for template_file in template_files:
                try:
                    record = json.loads(template_file.read_text(encoding="utf-8"))
                    entry = cls._entry_from_record(record, origin="builtin")
                    if entry.kind == "template":
                        entries.append(entry)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
            if entries:
                return entries
        return []

    def _write_entry(self, entry: WorkflowEntry) -> None:
        if entry.read_only:
            raise ValueError("Built-in workflow templates cannot be modified.")
        record = self._record_for_entry(entry)
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
        entries = sorted(entries, key=lambda entry: entry.updated_at, reverse=True)
        if kind == "template":
            return [*self._builtin_templates(), *entries]
        return entries

    def get(self, workflow_id: str, kind: WorkflowKind) -> WorkflowEntry:
        if kind == "template":
            for entry in self._builtin_templates():
                if entry.workflow_id == workflow_id:
                    return entry
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
            existing = self.get(workflow_id, kind)
            if existing.read_only:
                raise ValueError("Built-in workflow templates cannot be modified.")
            created_at = existing.created_at
        entry = WorkflowEntry(workflow_id, name, kind, created_at, now, normalised)
        self._write_entry(entry)
        return entry

    def rename(self, workflow_id: str, kind: WorkflowKind, name: str) -> WorkflowEntry:
        entry = self.get(workflow_id, kind)
        if entry.read_only:
            raise ValueError("Built-in workflow templates cannot be renamed.")
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
        entry = self.get(workflow_id, kind)
        if entry.read_only:
            raise ValueError("Built-in workflow templates cannot be deleted.")
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
        entry = self.get(workflow_id, kind)
        Path(path).write_text(
            json.dumps(self._record_for_entry(entry), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


__all__ = ["WorkflowEntry", "WorkflowKind", "WorkflowLibrary", "WorkflowOrigin"]
