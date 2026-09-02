"""Persistent, versioned cache for expensive structure-evidence inventories."""
from __future__ import annotations

import gzip
import json
import re
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from .result import (
    CompositionMagneticEvidence,
    CompositionPhaseEvidence,
    ElementMagneticEvidence,
    ElementMagneticSummary,
    ElementPairMagneticEvidence,
    ElementPairMagneticSummary,
    MagneticInventory,
    PhaseInventory,
    StructureMagneticEvidence,
    StructurePhaseEvidence,
)

EVIDENCE_CACHE_FORMAT_VERSION = 1
_LEGACY_CACHE_DIRECTORY = ".neptrainkit-cache"
PHYSICS_SAMPLING_CACHE_KIND = "phase-sampling"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _without_structures(value: Any) -> dict[str, Any]:
    return {
        field.name: _jsonable(getattr(value, field.name))
        for field in fields(value)
        if field.name != "structures"
    }


def _pairs(value: Any, second_type: type = float) -> tuple[tuple[Any, Any], ...]:
    return tuple((item[0], second_type(item[1])) for item in value)


def _phase_structure(payload: dict[str, Any]) -> StructurePhaseEvidence:
    return StructurePhaseEvidence(
        source_index=int(payload["source_index"]),
        atom_count=int(payload["atom_count"]),
        phase_label=str(payload["phase_label"]),
        confidence_state=str(payload["confidence_state"]),
        local_phase_fractions=_pairs(payload["local_phase_fractions"]),
    )


def _phase_point(
    payload: dict[str, Any],
    structures: tuple[StructurePhaseEvidence, ...],
) -> CompositionPhaseEvidence:
    return CompositionPhaseEvidence(
        reduced_counts=tuple(int(value) for value in payload["reduced_counts"]),
        source_structure_count=int(payload["source_structure_count"]),
        analyzed_structure_count=int(payload["analyzed_structure_count"]),
        analyzed_atom_count=int(payload["analyzed_atom_count"]),
        local_phase_fractions=_pairs(payload["local_phase_fractions"]),
        structure_phase_fractions=_pairs(payload["structure_phase_fractions"]),
        confidence_counts=_pairs(payload["confidence_counts"], int),
        confirmed_candidates=_pairs(payload.get("confirmed_candidates", ()), int),
        structures=structures,
    )


def _element_evidence(payload: dict[str, Any]) -> ElementMagneticEvidence:
    return ElementMagneticEvidence(
        element=str(payload["element"]),
        atom_count=int(payload["atom_count"]),
        spin_atom_count=int(payload["spin_atom_count"]),
        order_label=str(payload["order_label"]),
        mean_moment=float(payload["mean_moment"]),
        net_moment_ratio=float(payload["net_moment_ratio"]),
        collinearity=float(payload["collinearity"]),
        intra_element_correlation=float(payload["intra_element_correlation"]),
        intra_element_pair_count=int(payload["intra_element_pair_count"]),
        q_peak_strength=float(payload["q_peak_strength"]),
        q_vector=tuple(int(value) for value in payload["q_vector"]),
    )


def _element_pair_evidence(payload: dict[str, Any]) -> ElementPairMagneticEvidence:
    return ElementPairMagneticEvidence(
        element_a=str(payload["element_a"]),
        element_b=str(payload["element_b"]),
        pair_count=int(payload["pair_count"]),
        correlation=float(payload["correlation"]),
        coupling_label=str(payload["coupling_label"]),
    )


def _element_summary(payload: dict[str, Any]) -> ElementMagneticSummary:
    return ElementMagneticSummary(
        element=str(payload["element"]),
        structure_count=int(payload["structure_count"]),
        order_fractions=_pairs(payload["order_fractions"]),
        mean_moment=float(payload["mean_moment"]),
        mean_net_moment_ratio=float(payload["mean_net_moment_ratio"]),
        mean_collinearity=float(payload["mean_collinearity"]),
        mean_intra_element_correlation=float(
            payload["mean_intra_element_correlation"]
        ),
        mean_q_peak_strength=float(payload["mean_q_peak_strength"]),
    )


def _element_pair_summary(payload: dict[str, Any]) -> ElementPairMagneticSummary:
    return ElementPairMagneticSummary(
        element_a=str(payload["element_a"]),
        element_b=str(payload["element_b"]),
        structure_count=int(payload["structure_count"]),
        coupling_fractions=_pairs(payload["coupling_fractions"]),
        mean_correlation=float(payload["mean_correlation"]),
    )


def _magnetic_structure(payload: dict[str, Any]) -> StructureMagneticEvidence:
    return StructureMagneticEvidence(
        source_index=int(payload["source_index"]),
        atom_count=int(payload["atom_count"]),
        spin_atom_count=int(payload["spin_atom_count"]),
        order_label=str(payload["order_label"]),
        confidence_state=str(payload["confidence_state"]),
        mean_moment=float(payload["mean_moment"]),
        moment_std=float(payload["moment_std"]),
        net_moment_ratio=float(payload["net_moment_ratio"]),
        collinearity=float(payload["collinearity"]),
        coplanarity=float(payload["coplanarity"]),
        neighbor_correlation=float(payload["neighbor_correlation"]),
        neighbor_abs_correlation=float(payload["neighbor_abs_correlation"]),
        parallel_fraction=float(payload["parallel_fraction"]),
        antiparallel_fraction=float(payload["antiparallel_fraction"]),
        q_peak_strength=float(payload["q_peak_strength"]),
        q_vector=tuple(int(value) for value in payload["q_vector"]),
        element_evidence=tuple(
            _element_evidence(item) for item in payload.get("element_evidence", ())
        ),
        element_pair_evidence=tuple(
            _element_pair_evidence(item)
            for item in payload.get("element_pair_evidence", ())
        ),
        order_subtype=str(payload.get("order_subtype", "")),
    )


def _magnetic_point(
    payload: dict[str, Any],
    structures: tuple[StructureMagneticEvidence, ...],
) -> CompositionMagneticEvidence:
    return CompositionMagneticEvidence(
        reduced_counts=tuple(int(value) for value in payload["reduced_counts"]),
        source_structure_count=int(payload["source_structure_count"]),
        analyzed_structure_count=int(payload["analyzed_structure_count"]),
        missing_spin_count=int(payload["missing_spin_count"]),
        order_fractions=_pairs(payload["order_fractions"]),
        confidence_counts=_pairs(payload["confidence_counts"], int),
        mean_net_moment_ratio=float(payload["mean_net_moment_ratio"]),
        mean_collinearity=float(payload["mean_collinearity"]),
        mean_q_peak_strength=float(payload["mean_q_peak_strength"]),
        element_summaries=tuple(
            _element_summary(item) for item in payload.get("element_summaries", ())
        ),
        element_pair_summaries=tuple(
            _element_pair_summary(item)
            for item in payload.get("element_pair_summaries", ())
        ),
        structures=structures,
    )


class TrainingSetEvidenceCache:
    """Stream large audit inventories to a safe JSONL+gzip cache."""

    def __init__(
        self,
        output_directory: Path,
        dataset_name: str,
        dataset_fingerprint: str,
        scope_fingerprint: str,
    ) -> None:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name).strip("._")
        self.directory = Path(output_directory)
        self._legacy_directory = self.directory / _LEGACY_CACHE_DIRECTORY
        self.dataset_name = safe_name or "dataset"
        self.dataset_fingerprint = dataset_fingerprint
        self.scope_fingerprint = scope_fingerprint

    @classmethod
    def from_result_data(
        cls,
        result_data: Any,
        audit_result: Any,
    ) -> TrainingSetEvidenceCache | None:
        fingerprints = getattr(audit_result, "fingerprints", None)
        dataset_fingerprint = str(getattr(fingerprints, "dataset", "") or "")
        scope_fingerprint = str(getattr(fingerprints, "scope", "") or "")
        return cls.from_fingerprints(
            result_data,
            dataset_fingerprint=dataset_fingerprint,
            scope_fingerprint=scope_fingerprint,
        )

    @classmethod
    def from_fingerprints(
        cls,
        result_data: Any,
        *,
        dataset_fingerprint: str,
        scope_fingerprint: str,
    ) -> TrainingSetEvidenceCache | None:
        """Resolve the persistent evidence cache for an explicit data scope."""
        dataset_fingerprint = str(dataset_fingerprint or "")
        scope_fingerprint = str(scope_fingerprint or "")
        if not dataset_fingerprint or not scope_fingerprint:
            return None
        cache_enabled = getattr(result_data, "cache_outputs_enabled", None)
        if callable(cache_enabled) and not cache_enabled():
            return None
        descriptor_path = getattr(result_data, "descriptor_path", None)
        dataset_path = getattr(result_data, "data_xyz_path", None)
        if descriptor_path is None or dataset_path is None:
            return None
        return cls(
            Path(descriptor_path).parent,
            Path(dataset_path).stem,
            dataset_fingerprint,
            scope_fingerprint,
        )

    def _path(self, kind: str) -> Path:
        scope_tag = self.scope_fingerprint[:16]
        return self.directory / f"{self.dataset_name}.{scope_tag}.{kind}.jsonl.gz"

    def _load_path(self, kind: str) -> Path:
        path = self._path(kind)
        if path.is_file():
            return path
        legacy_path = self._legacy_directory / path.name
        return legacy_path if legacy_path.is_file() else path

    def _promote_legacy_path(self, path: Path, kind: str) -> None:
        target = self._path(kind)
        if path == target or target.exists():
            return
        try:
            path.replace(target)
        except OSError as error:
            logger.warning(
                "Could not move legacy {} evidence cache {} to {}: {}",
                kind,
                path,
                target,
                error,
            )
            return
        try:
            self._legacy_directory.rmdir()
        except OSError:
            pass

    def _header(self, kind: str, inventory: Any) -> dict[str, Any]:
        header = {
            "format_version": EVIDENCE_CACHE_FORMAT_VERSION,
            "kind": kind,
            "dataset_fingerprint": self.dataset_fingerprint,
            "scope_fingerprint": self.scope_fingerprint,
            "schema_version": inventory.schema_version,
            "method_id": inventory.method_id,
            "analysis_strategy": inventory.analysis_strategy,
            "source_structure_count": inventory.source_structure_count,
            "analyzed_structure_count": inventory.analyzed_structure_count,
            "composition_point_count": len(inventory.composition_points),
        }
        if kind == "phase":
            header.update(
                {
                    "reference_bank_id": inventory.reference_bank_id,
                    "analyzed_atom_count": inventory.analyzed_atom_count,
                }
            )
        else:
            header["missing_spin_count"] = inventory.missing_spin_count
        return header

    def _matches(self, header: dict[str, Any], kind: str) -> bool:
        return (
            header.get("format_version") == EVIDENCE_CACHE_FORMAT_VERSION
            and header.get("kind") == kind
            and header.get("dataset_fingerprint") == self.dataset_fingerprint
            and header.get("scope_fingerprint") == self.scope_fingerprint
        )

    @staticmethod
    def _read_line(handle: Any) -> dict[str, Any]:
        line = handle.readline()
        if not line:
            raise EOFError("The evidence cache ended before all records were read.")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError("An evidence cache record must be a JSON object.")
        return payload

    @staticmethod
    def _write_line(handle: Any, payload: dict[str, Any]) -> None:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        handle.write(b"\n")

    def load_phase(
        self,
        *,
        schema_version: str,
        method_id: str,
        reference_bank_id: str,
        analysis_strategy: str,
    ) -> PhaseInventory | None:
        path = self._load_path("phase")
        if not path.is_file():
            return None
        try:
            with gzip.open(path, "rb") as handle:
                header = self._read_line(handle)
                if not self._matches(header, "phase") or (
                    header.get("schema_version"),
                    header.get("method_id"),
                    header.get("reference_bank_id"),
                    header.get("analysis_strategy"),
                ) != (
                    schema_version,
                    method_id,
                    reference_bank_id,
                    analysis_strategy,
                ):
                    return None
                points = []
                for _ in range(int(header["composition_point_count"])):
                    record = self._read_line(handle)
                    structure_count = int(record["structure_record_count"])
                    structures = tuple(
                        _phase_structure(self._read_line(handle))
                        for _ in range(structure_count)
                    )
                    points.append(_phase_point(record["point"], structures))
                if handle.readline().strip():
                    raise ValueError("The evidence cache contains unexpected trailing records.")
            inventory = PhaseInventory(
                schema_version=str(header["schema_version"]),
                method_id=str(header["method_id"]),
                reference_bank_id=str(header["reference_bank_id"]),
                analysis_strategy=str(header["analysis_strategy"]),
                source_structure_count=int(header["source_structure_count"]),
                analyzed_structure_count=int(header["analyzed_structure_count"]),
                analyzed_atom_count=int(header["analyzed_atom_count"]),
                composition_points=tuple(points),
            )
            if (
                inventory.analyzed_structure_count
                != sum(point.analyzed_structure_count for point in points)
            ):
                raise ValueError("The phase cache structure count is inconsistent.")
            self._promote_legacy_path(path, "phase")
            logger.info("Loaded phase evidence cache: {}", self._path("phase"))
            return inventory
        except (EOFError, OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            logger.warning("Ignoring invalid phase evidence cache {}: {}", path, error)
            return None

    def load_magnetic(
        self,
        *,
        schema_version: str,
        method_id: str,
        analysis_strategy: str,
    ) -> MagneticInventory | None:
        path = self._load_path("magnetic")
        if not path.is_file():
            return None
        try:
            with gzip.open(path, "rb") as handle:
                header = self._read_line(handle)
                if not self._matches(header, "magnetic") or (
                    header.get("schema_version"),
                    header.get("method_id"),
                    header.get("analysis_strategy"),
                ) != (schema_version, method_id, analysis_strategy):
                    return None
                points = []
                for _ in range(int(header["composition_point_count"])):
                    record = self._read_line(handle)
                    structure_count = int(record["structure_record_count"])
                    structures = tuple(
                        _magnetic_structure(self._read_line(handle))
                        for _ in range(structure_count)
                    )
                    points.append(_magnetic_point(record["point"], structures))
                if handle.readline().strip():
                    raise ValueError("The evidence cache contains unexpected trailing records.")
            inventory = MagneticInventory(
                schema_version=str(header["schema_version"]),
                method_id=str(header["method_id"]),
                analysis_strategy=str(header["analysis_strategy"]),
                source_structure_count=int(header["source_structure_count"]),
                analyzed_structure_count=int(header["analyzed_structure_count"]),
                missing_spin_count=int(header["missing_spin_count"]),
                composition_points=tuple(points),
            )
            if (
                inventory.analyzed_structure_count
                != sum(point.analyzed_structure_count for point in points)
            ):
                raise ValueError("The magnetic cache structure count is inconsistent.")
            self._promote_legacy_path(path, "magnetic")
            logger.info("Loaded magnetic evidence cache: {}", self._path("magnetic"))
            return inventory
        except (EOFError, OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            logger.warning("Ignoring invalid magnetic evidence cache {}: {}", path, error)
            return None

    def load_sampling_partitions(
        self,
        *,
        identity: dict[str, Any],
    ) -> tuple[dict[str, Any], ...] | None:
        """Load optional per-structure physical partitions for FPS reuse."""
        path = self._load_path(PHYSICS_SAMPLING_CACHE_KIND)
        if not path.is_file():
            return None
        try:
            with gzip.open(path, "rb") as handle:
                header = self._read_line(handle)
                if not self._matches(header, PHYSICS_SAMPLING_CACHE_KIND):
                    return None
                if any(header.get(key) != value for key, value in identity.items()):
                    return None
                records = tuple(
                    self._read_line(handle)
                    for _ in range(int(header["assignment_count"]))
                )
                if handle.readline().strip():
                    raise ValueError(
                        "The phase-sampling cache contains unexpected trailing records."
                    )
            self._promote_legacy_path(path, PHYSICS_SAMPLING_CACHE_KIND)
            logger.info(
                "Loaded phase-sampling cache: {}",
                self._path(PHYSICS_SAMPLING_CACHE_KIND),
            )
            return records
        except (
            EOFError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as error:
            logger.warning(
                "Ignoring invalid phase-sampling cache {}: {}",
                path,
                error,
            )
            return None

    def _save(self, kind: str, inventory: Any) -> bool:
        path = self._path(kind)
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with gzip.open(temporary, "wb", compresslevel=6) as handle:
                self._write_line(handle, self._header(kind, inventory))
                for point in inventory.composition_points:
                    self._write_line(
                        handle,
                        {
                            "point": _without_structures(point),
                            "structure_record_count": len(point.structures),
                        },
                    )
                    for structure in point.structures:
                        self._write_line(handle, _jsonable(structure))
            temporary.replace(path)
            logger.info("Saved {} evidence cache: {}", kind, path)
            return True
        except (OSError, TypeError, ValueError) as error:
            logger.warning("Could not write {} evidence cache {}: {}", kind, path, error)
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def save_phase(self, inventory: PhaseInventory) -> bool:
        return self._save("phase", inventory)

    def save_magnetic(self, inventory: MagneticInventory) -> bool:
        return self._save("magnetic", inventory)

    def save_sampling_partitions(
        self,
        records: Sequence[dict[str, Any]],
        *,
        identity: dict[str, Any],
    ) -> bool:
        """Atomically persist compact physical partitions beside phase evidence."""
        path = self._path(PHYSICS_SAMPLING_CACHE_KIND)
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            header = {
                "format_version": EVIDENCE_CACHE_FORMAT_VERSION,
                "kind": PHYSICS_SAMPLING_CACHE_KIND,
                "dataset_fingerprint": self.dataset_fingerprint,
                "scope_fingerprint": self.scope_fingerprint,
                "assignment_count": len(records),
                **identity,
            }
            with gzip.open(temporary, "wb", compresslevel=6) as handle:
                self._write_line(handle, header)
                for record in records:
                    self._write_line(handle, dict(record))
            temporary.replace(path)
            logger.info("Saved phase-sampling cache: {}", path)
            return True
        except (OSError, TypeError, ValueError) as error:
            logger.warning(
                "Could not write phase-sampling cache {}: {}",
                path,
                error,
            )
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def path_for(self, kind: str) -> Path:
        """Expose the resolved path for diagnostics and focused tests."""
        if kind not in {"phase", "magnetic", PHYSICS_SAMPLING_CACHE_KIND}:
            raise ValueError(f"Unknown evidence cache kind: {kind}")
        return self._path(kind)
