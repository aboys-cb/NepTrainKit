"""Controlled magnetic-response scans, lineage, and dataset-level auditing.

The public operations in this module return ordinary ASE ``Atoms`` objects.
Response bookkeeping is deliberately centralized here so UI cards and DFT
adapters do not need to reproduce rotation, grouping, or manifest rules.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from ase import Atoms

from NepTrainKit.core.cards.errors import CardOperationError
from NepTrainKit.core.cards.lattice import BainPathOperation, BainPathParams
from NepTrainKit.core.cards.magnetism import (
    SmallAngleSpinTiltOperation,
    SmallAngleSpinTiltParams,
)
from NepTrainKit.core.cards.operation import StructureOperation
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.core.magnetism import (
    existing_moment_vectors,
    normalize_vector,
    set_initial_magmoms_safe,
)


RESPONSE_SCHEMA = "magnetic-response-v1"
def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _stable_digest(payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def structure_fingerprint(atoms: Atoms) -> str:
    """Return a stable fingerprint of geometry, species, PBC, and input spins."""
    spins = existing_moment_vectors(atoms, lift_scalar=True)
    payload = {
        "numbers": atoms.numbers.tolist(),
        "positions": np.round(atoms.positions, 12).tolist(),
        "cell": np.round(atoms.cell.array, 12).tolist(),
        "pbc": atoms.pbc.tolist(),
        "spin": None if spins is None else np.round(spins, 12).tolist(),
    }
    return _stable_digest(payload, length=24)


def _rotation_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    axis_hat = normalize_vector(np.asarray(axis, dtype=float))
    x, y, z = axis_hat
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def _parse_scan(text: str, *, minimum_points: int = 1) -> list[float]:
    values: list[float] = []
    for token in re.split(r"[\s,;]+", str(text or "")):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"invalid response coordinate {token!r}") from exc
    if len(values) < minimum_points:
        raise ValueError(f"response scan requires at least {minimum_points} distinct coordinates")
    if not np.isfinite(values).all():
        raise ValueError("response scan coordinates must be finite")
    unique = sorted(set(float(value) for value in values))
    if len(unique) != len(values):
        raise ValueError("response scan coordinates must be unique")
    return unique


def _branch(coordinate: float, *, atol: float = 1.0e-14) -> str:
    if abs(float(coordinate)) <= atol:
        return "reference"
    return "plus" if coordinate > 0.0 else "minus"


@dataclass(frozen=True)
class ResponseManifestRecord:
    task_id: str
    response_schema: str
    response_parent: str
    response_group: str
    response_probe: str
    response_kind: str
    response_coordinate: float
    response_branch: str
    source_structure_id: str
    target_indices: tuple[int, ...] = ()
    rotation_axis: tuple[float, float, float] | None = None
    rotation_plane_normal: tuple[float, float, float] | None = None
    pair_shell: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    input_spin_source: str = "spin_or_initial_magmoms"
    output_spin_source: str = "generated_constraint"
    mforce_convention: str = "-dE/dspin"
    structure_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = _jsonable(asdict(self))
        result["target_indices"] = list(self.target_indices)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResponseManifestRecord":
        payload = dict(data)
        payload["target_indices"] = tuple(int(v) for v in payload.get("target_indices", ()))
        for key in ("rotation_axis", "rotation_plane_normal"):
            if payload.get(key) is not None:
                payload[key] = tuple(float(v) for v in payload[key])
        return cls(**payload)


class ResponseManifest:
    """One-to-one task manifest used across DFT adapters and label collection."""

    def __init__(self, records: Iterable[ResponseManifestRecord] = ()):
        self.records = list(records)
        task_ids = [record.task_id for record in self.records]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("manifest task_id values must be unique")

    def to_dict(self) -> dict[str, Any]:
        records = [record.to_dict() for record in self.records]
        return {
            "response_schema": RESPONSE_SCHEMA,
            "records": records,
            "manifest_hash": _stable_digest(records, length=64),
        }

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_dataset(cls, dataset: Iterable[Atoms]) -> "ResponseManifest":
        records = []
        for atoms in dataset:
            raw = atoms.info.get("_response_manifest_record")
            if raw is not None:
                records.append(ResponseManifestRecord.from_dict(dict(raw)))
        if not records:
            raise ValueError("dataset contains no embedded response manifest seam")
        return cls(records)

    @classmethod
    def read(cls, path: str | Path) -> "ResponseManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records = [ResponseManifestRecord.from_dict(item) for item in data.get("records", [])]
        manifest = cls(records)
        if data.get("manifest_hash") != manifest.to_dict()["manifest_hash"]:
            raise ValueError("response manifest hash mismatch")
        return manifest

    def reattach(self, atoms: Atoms, task_id: str, *, spin_source: str = "dft_final") -> Atoms:
        matches = [record for record in self.records if record.task_id == str(task_id)]
        if len(matches) != 1:
            raise ValueError(f"task_id {task_id!r} does not map to exactly one manifest record")
        expected_numbers = matches[0].metadata.get("atomic_numbers")
        if expected_numbers is not None and list(map(int, atoms.numbers)) != list(map(int, expected_numbers)):
            raise ValueError(
                f"task_id {task_id!r} atom identity/order does not match its manifest record"
            )
        output = atoms.copy()
        _attach_minimal_metadata(output, matches[0])
        output.info["response_spin_provenance"] = str(spin_source)
        output.info["response_manifest_hash"] = self.to_dict()["manifest_hash"]
        return output


def _attach_minimal_metadata(atoms: Atoms, record: ResponseManifestRecord) -> None:
    atoms.info.update(
        response_schema=record.response_schema,
        response_parent=record.response_parent,
        response_group=record.response_group,
        response_probe=record.response_probe,
        response_kind=record.response_kind,
        response_coordinate=float(record.response_coordinate),
        response_branch=record.response_branch,
        response_task_id=record.task_id,
        response_source_structure_id=record.source_structure_id,
        response_spin_provenance=record.output_spin_source,
        response_mforce_convention=record.mforce_convention,
    )


@dataclass(frozen=True)
class LocalMagneticResponseParams:
    response_kind: str = "Atom pair canting"
    coordinate_scan_deg: str = "-2,-1,0,1,2"
    target_mode: str = "First eligible atom"
    target_indices: str = ""
    pair_source: str = "Manual indices"
    pair_left_indices: str = "1"
    pair_right_indices: str = "2"
    pair_shell: int = 1
    pair_shell_tolerance: float = 0.05
    pair_element_filter: str = ""
    pair_group_filter: str = ""
    bond_filter_mode: str = "Any"
    bond_filter_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    bond_filter_tolerance: float = 20.0
    group_a: str = "A"
    group_b: str = "B"
    rotation_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    apply_elements: str = ""
    moment_scale_scan: str = "0.8,0.9,1.0,1.1,1.2"
    max_outputs: int = 100


@dataclass(frozen=True)
class TextureMagneticResponseParams:
    response_kind: str = "Global anisotropy"
    coordinate_scan: str = "-2,-1,0,1,2"
    rotation_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    q_definition: str | None = None
    q_reciprocal_index: tuple[int, int, int] = (1, 0, 0)
    q_vector_cart: tuple[float, float, float] = (0.0, 0.0, 0.1)
    plane_normal: tuple[float, float, float] = (0.0, 1.0, 0.0)
    surface_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    cone_component: float = 0.0
    phase_deg: float = 0.0
    include_time_reversal: bool = False
    require_commensurate: bool = True
    max_outputs: int = 100


@dataclass(frozen=True)
class MagnetoelasticResponseParams:
    structural_mode: str = "Isotropic volume"
    structural_scan: str = "-0.02,-0.01,0,0.01,0.02"
    spin_scan_deg: str = "-2,0,2"
    rotation_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    target_indices: str = "1"
    strain_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    shear_direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    bain_axis: str = "c"
    max_outputs: int = 100


class MagneticResponseScanOperation(StructureOperation):
    """Unified production interface for complete controlled-response groups."""

    def __init__(self):
        self.last_manifest = ResponseManifest()
        self._parent_occurrences: dict[str, int] = {}

    def run_structure(self, structure: Atoms, params: Any) -> list[Atoms]:
        if isinstance(params, LocalMagneticResponseParams):
            return self.generate_local(structure, params)
        if isinstance(params, TextureMagneticResponseParams):
            return self.generate_texture(structure, params)
        if isinstance(params, MagnetoelasticResponseParams):
            return self.generate_magnetoelastic(structure, params)
        raise TypeError(f"unsupported magnetic response params: {type(params).__name__}")

    def _base(self, structure: Atoms) -> tuple[np.ndarray, str, str]:
        moments = existing_moment_vectors(structure, lift_scalar=True)
        if moments is None or moments.shape != (len(structure), 3):
            raise CardOperationError(
                "magnetic_response_missing_moments",
                "Magnetic response needs vector spin or initial magnetic moments on the input structure.",
            )
        if not np.isfinite(moments).all() or not np.any(np.linalg.norm(moments, axis=1) > 1.0e-12):
            raise CardOperationError(
                "magnetic_response_invalid_moments",
                "Magnetic response needs finite magnetic moments with at least one non-zero vector.",
            )
        fingerprint = structure_fingerprint(structure)
        explicit_id = next(
            (str(structure.info[key]) for key in ("source_structure_id", "structure_id", "Config_id") if structure.info.get(key)),
            "",
        )
        source_id = explicit_id or fingerprint
        parent_base = f"mrp-{_stable_digest([source_id, fingerprint], length=24)}"
        occurrence = self._parent_occurrences.get(parent_base, 0)
        self._parent_occurrences[parent_base] = occurrence + 1
        parent = parent_base if occurrence == 0 else f"{parent_base}-{occurrence + 1}"
        return moments, source_id, parent

    def _emit_group(
        self,
        structure: Atoms,
        *,
        source_id: str,
        parent: str,
        group_key: Any,
        probe: str,
        kind: str,
        frames: Sequence[tuple[float, np.ndarray, Atoms]],
        max_outputs: int,
        records: list[ResponseManifestRecord],
        outputs: list[Atoms],
        target_indices: Sequence[int] = (),
        rotation_axis: Sequence[float] | None = None,
        plane_normal: Sequence[float] | None = None,
        pair_shell: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if len(outputs) + len(frames) > int(max_outputs):
            return False
        group = f"mrg-{_stable_digest([parent, kind, group_key], length=20)}"
        for coordinate, spins, geometry in frames:
            atoms = geometry.copy()
            set_initial_magmoms_safe(atoms, spins)
            branch = _branch(coordinate)
            task_id = f"mrt-{_stable_digest([group, float(coordinate), branch], length=24)}"
            record = ResponseManifestRecord(
                task_id=task_id,
                response_schema=RESPONSE_SCHEMA,
                response_parent=parent,
                response_group=group,
                response_probe=probe,
                response_kind=kind,
                response_coordinate=float(coordinate),
                response_branch=branch,
                source_structure_id=source_id,
                target_indices=tuple(int(i) for i in target_indices),
                rotation_axis=None if rotation_axis is None else tuple(float(v) for v in normalize_vector(np.asarray(rotation_axis))),
                rotation_plane_normal=None if plane_normal is None else tuple(float(v) for v in normalize_vector(np.asarray(plane_normal))),
                pair_shell=pair_shell,
                metadata=_jsonable({"atomic_numbers": structure.numbers.tolist(), **(metadata or {})}),
                structure_hash=structure_fingerprint(atoms),
            )
            _attach_minimal_metadata(atoms, record)
            # In-memory only seam for workflow/card handoff.  Production
            # EXTXYZ export strips this richer record and keeps only the small
            # response_* header contract.
            atoms.info["_response_manifest_record"] = record.to_dict()
            append_config_tag(atoms, f"MagResponse({kind},{branch})")
            outputs.append(atoms)
            records.append(record)
        return True

    def _finish(self, outputs: list[Atoms], records: list[ResponseManifestRecord]) -> list[Atoms]:
        if not outputs:
            raise CardOperationError(
                "magnetic_response_budget_too_small",
                "Maximum structures is smaller than the coordinate count of one complete response group.",
            )
        self.last_manifest = ResponseManifest([*self.last_manifest.records, *records])
        report = audit_response_groups(outputs)
        if report["invalid_groups"]:
            reasons = "; ".join(
                f"{group}: {', '.join(items)}" for group, items in report["invalid_groups"].items()
            )
            raise ValueError(f"generated invalid magnetic response groups: {reasons}")
        return outputs

    def generate_local(self, structure: Atoms, params: LocalMagneticResponseParams) -> list[Atoms]:
        moments, source_id, parent = self._base(structure)
        outputs: list[Atoms] = []
        records: list[ResponseManifestRecord] = []
        helper_params = SmallAngleSpinTiltParams(
            canting_mode=params.response_kind,
            target_mode=params.target_mode,
            target_indices=params.target_indices,
            pair_source=params.pair_source,
            pair_left_indices=params.pair_left_indices,
            pair_right_indices=params.pair_right_indices,
            pair_shell=params.pair_shell,
            pair_shell_tolerance=params.pair_shell_tolerance,
            pair_element_filter=params.pair_element_filter,
            pair_group_filter=params.pair_group_filter,
            bond_filter_mode=params.bond_filter_mode,
            bond_filter_axis=params.bond_filter_axis,
            bond_filter_tolerance=params.bond_filter_tolerance,
            group_a=params.group_a,
            group_b=params.group_b,
            axis=params.rotation_axis,
            reference_direction=(1.0, 0.0, 0.0),
            apply_elements=params.apply_elements,
        )
        helper = SmallAngleSpinTiltOperation()
        if params.response_kind == "Moment magnitude":
            scales = _parse_scan(params.moment_scale_scan, minimum_points=3)
            if any(scale < 0.0 for scale in scales):
                raise ValueError("moment scale factors must be non-negative")
            if 1.0 not in scales:
                raise ValueError("moment magnitude response must include scale factor 1.0")
            targets = helper.candidate_indices(structure, moments, helper_params)
            if not targets:
                raise CardOperationError(
                    "local_response_no_target_moments",
                    "No non-zero magnetic moments match the selected atoms and elements.",
                )
            frames = []
            for scale in scales:
                spins = moments.copy()
                spins[targets] *= float(scale)
                frames.append((float(scale) - 1.0, spins, structure))
            self._emit_group(
                structure, source_id=source_id, parent=parent, group_key=["moment", targets],
                probe="moment_scale", kind="moment_magnitude", frames=frames,
                max_outputs=params.max_outputs, records=records, outputs=outputs,
                target_indices=targets, metadata={"scale_factors": scales},
            )
            return self._finish(outputs, records)

        angles_deg = _parse_scan(params.coordinate_scan_deg, minimum_points=3)
        if 0.0 not in angles_deg:
            raise ValueError("rotation response scan must include a zero-coordinate reference")
        if params.response_kind == "Single-spin tilt":
            target_sets = [([idx], []) for idx in helper.candidate_indices(structure, moments, helper_params)]
            kind = "single_spin_rotation"
        elif params.response_kind == "Atom pair canting":
            target_sets = [([left], [right]) for left, right in helper.pair_targets(structure, moments, helper_params)]
            kind = "pair_canting"
        elif params.response_kind == "Group pair canting":
            left, right = helper.group_targets(structure, moments, helper_params)
            target_sets = [(left, right)] if left and right else []
            kind = "group_pair_canting"
        else:
            raise ValueError(f"unsupported local response kind: {params.response_kind}")
        if not target_sets:
            if params.response_kind == "Group pair canting":
                raise CardOperationError(
                    "local_response_no_group_pair",
                    "The input needs a non-zero magnetic moment in both group '{group_a}' and group "
                    "'{group_b}'. Check the group labels or add Group Label upstream.",
                    group_a=params.group_a,
                    group_b=params.group_b,
                )
            if params.response_kind == "Atom pair canting" and params.pair_source == "Auto by neighbor shell":
                raise CardOperationError(
                    "local_response_no_auto_pair",
                    "No atom pairs match the selected neighbor shell and automatic-pair filters.",
                )
            if params.response_kind == "Atom pair canting":
                raise CardOperationError(
                    "local_response_no_manual_pair",
                    "No valid magnetic atom pairs match the left and right indices.",
                )
            raise CardOperationError(
                "local_response_no_target_moments",
                "No non-zero magnetic moments match the selected atoms and elements.",
            )
        for target_no, (left, right) in enumerate(target_sets):
            frames = []
            for angle_deg in angles_deg:
                spins = moments.copy()
                if right:
                    left_rotation = _rotation_matrix(
                        params.rotation_axis, math.radians(angle_deg) * 0.5
                    )
                    right_rotation = _rotation_matrix(
                        params.rotation_axis, -math.radians(angle_deg) * 0.5
                    )
                    spins[left] = spins[left] @ left_rotation.T
                    spins[right] = spins[right] @ right_rotation.T
                else:
                    rotation = _rotation_matrix(params.rotation_axis, math.radians(angle_deg))
                    spins[left] = spins[left] @ rotation.T
                frames.append((math.radians(angle_deg), spins, structure))
            emitted = self._emit_group(
                structure, source_id=source_id, parent=parent,
                group_key=[kind, target_no, left, right], probe="rotation", kind=kind,
                frames=frames, max_outputs=params.max_outputs, records=records, outputs=outputs,
                target_indices=[*left, *right], rotation_axis=params.rotation_axis,
                plane_normal=params.rotation_axis, pair_shell=params.pair_shell if params.pair_source == "Auto by neighbor shell" else None,
            )
            if not emitted:
                break
        return self._finish(outputs, records)

    def generate_texture(self, structure: Atoms, params: TextureMagneticResponseParams) -> list[Atoms]:
        moments, source_id, parent = self._base(structure)
        outputs: list[Atoms] = []
        records: list[ResponseManifestRecord] = []
        if params.response_kind == "Global anisotropy":
            angles_deg = _parse_scan(params.coordinate_scan, minimum_points=3)
            if 0.0 not in angles_deg:
                raise ValueError("global anisotropy scan must include a zero-coordinate reference")
            axis = normalize_vector(np.asarray(params.rotation_axis, dtype=float))
            frames = [
                (math.radians(angle), moments @ _rotation_matrix(axis, math.radians(angle)).T, structure)
                for angle in angles_deg
            ]
            required = len(frames) * (2 if params.include_time_reversal else 1)
            if int(params.max_outputs) < required:
                raise CardOperationError(
                    "texture_response_budget_too_small",
                    "Maximum structures must be at least {required} for the selected texture response path.",
                    required=required,
                )
            self._emit_group(
                structure, source_id=source_id, parent=parent, group_key=["anisotropy", axis.tolist()],
                probe="rotation", kind="global_anisotropy", frames=frames,
                max_outputs=params.max_outputs, records=records, outputs=outputs,
                target_indices=range(len(structure)), rotation_axis=axis,
            )
            if params.include_time_reversal:
                tr_frames = [(coordinate, -spins, geometry) for coordinate, spins, geometry in frames]
                self._emit_group(
                    structure, source_id=source_id, parent=parent, group_key=["anisotropy-tr", axis.tolist()],
                    probe="rotation", kind="global_anisotropy_time_reversed", frames=tr_frames,
                    max_outputs=params.max_outputs, records=records, outputs=outputs,
                    target_indices=range(len(structure)), rotation_axis=axis,
                    metadata={"negative_control": "global_time_reversal"},
                )
            return self._finish(outputs, records)

        kind_map = {
            "Bulk / Bloch": "bulk_bloch",
            "Interfacial / Cycloidal": "interfacial_cycloidal",
            "General spiral": "general_spiral",
        }
        if params.response_kind not in kind_map:
            raise ValueError(f"unsupported texture response kind: {params.response_kind}")
        multipliers = _parse_scan(params.coordinate_scan, minimum_points=3)
        if 0.0 not in multipliers:
            raise ValueError("q response scan must include q=0")
        q_definition = params.q_definition or "Cartesian vector"
        if q_definition == "Cell reciprocal vector":
            cell = np.asarray(structure.cell.array, dtype=float)
            if cell.shape != (3, 3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell)) <= 1.0e-14:
                raise CardOperationError(
                    "texture_response_invalid_cell",
                    "Cell-reciprocal q needs a finite, non-singular 3D cell.",
                )
            raw_index = np.asarray(params.q_reciprocal_index, dtype=float)
            if raw_index.shape != (3,) or not np.isfinite(raw_index).all():
                raise CardOperationError(
                    "texture_response_invalid_reciprocal_index",
                    "The reciprocal-cell index must contain three finite integers.",
                )
            reciprocal_index = np.rint(raw_index)
            if not np.allclose(raw_index, reciprocal_index, atol=1.0e-12):
                raise CardOperationError(
                    "texture_response_invalid_reciprocal_index",
                    "The reciprocal-cell index must contain three finite integers.",
                )
            if not np.any(reciprocal_index):
                raise CardOperationError(
                    "texture_response_zero_reciprocal_index",
                    "The reciprocal-cell index cannot be (0, 0, 0) for a spiral response.",
                )
            reciprocal = 2.0 * math.pi * np.linalg.inv(cell).T
            q0 = reciprocal.T @ reciprocal_index
        elif q_definition == "Cartesian vector":
            reciprocal_index = None
            q0 = np.asarray(params.q_vector_cart, dtype=float)
        else:
            raise CardOperationError(
                "texture_response_invalid_q_definition",
                "q definition must be Cell reciprocal vector or Cartesian vector.",
            )
        q0_norm = float(np.linalg.norm(q0))
        if q0_norm <= 1.0e-12:
            raise CardOperationError(
                "texture_response_zero_q",
                "The Cartesian base q vector must be non-zero for a spiral response.",
            )
        q_hat = q0 / q0_norm
        if params.response_kind == "Bulk / Bloch":
            plane_normal = q_hat
            e1 = normalize_vector(np.cross(q_hat, [1.0, 0.0, 0.0]) if abs(q_hat[0]) < 0.9 else np.cross(q_hat, [0.0, 1.0, 0.0]))
        elif params.response_kind == "Interfacial / Cycloidal":
            surface = normalize_vector(np.asarray(params.surface_normal, dtype=float))
            if abs(float(np.dot(surface, q_hat))) > 1.0 - 1.0e-8:
                raise ValueError("surface normal must not be parallel to q for a cycloidal spiral")
            plane_normal = normalize_vector(np.cross(q_hat, surface))
            e1 = q_hat
        else:
            plane_normal = normalize_vector(np.asarray(params.plane_normal, dtype=float))
            if abs(float(np.dot(plane_normal, q_hat))) > 1.0 - 1.0e-8:
                e1 = normalize_vector(np.cross(plane_normal, [1.0, 0.0, 0.0]) if abs(plane_normal[0]) < 0.9 else np.cross(plane_normal, [0.0, 1.0, 0.0]))
            else:
                e1 = normalize_vector(q_hat - np.dot(q_hat, plane_normal) * plane_normal)
        e2 = normalize_vector(np.cross(plane_normal, e1))
        cone = float(params.cone_component)
        if not -1.0 <= cone <= 1.0:
            raise ValueError("cone_component must be in [-1, 1]")
        radial = math.sqrt(max(0.0, 1.0 - cone * cone))
        magnitudes = np.linalg.norm(moments, axis=1)
        phase0 = math.radians(float(params.phase_deg))
        frames = []
        q_fractional_records = []
        reciprocal = 2.0 * math.pi * np.linalg.inv(structure.cell.array).T if abs(np.linalg.det(structure.cell.array)) > 1.0e-14 else None
        for multiplier in multipliers:
            q = float(multiplier) * q0
            if params.require_commensurate:
                for index, periodic in enumerate(structure.pbc):
                    if periodic:
                        turns = float(np.dot(q, structure.cell.array[index]) / (2.0 * math.pi))
                        if abs(turns - round(turns)) > 1.0e-7:
                            raise CardOperationError(
                                "texture_response_incommensurate_q",
                                "q does not close across periodic cell vector {index}. Use the cell-reciprocal q mode, or change q and the supercell together.",
                                index=index + 1,
                            )
            phases = structure.positions @ q + phase0
            unit = radial * (np.cos(phases)[:, None] * e1 + np.sin(phases)[:, None] * e2) + cone * plane_normal
            frames.append((float(np.sign(multiplier) * np.linalg.norm(q)), magnitudes[:, None] * unit, structure))
            q_fractional_records.append(
                None if reciprocal is None else np.linalg.solve(reciprocal.T, q).tolist()
            )
        q0_period = 2.0 * math.pi / q0_norm
        self._emit_group(
            structure, source_id=source_id, parent=parent, group_key=[kind_map[params.response_kind], q0.tolist(), phase0],
            probe="chirality", kind=kind_map[params.response_kind], frames=frames,
            max_outputs=params.max_outputs, records=records, outputs=outputs,
            target_indices=range(len(structure)), plane_normal=plane_normal,
            metadata={
                "q_cartesian_1_per_angstrom": [np.asarray(multiplier * q0).tolist() for multiplier in multipliers],
                "q_reciprocal_fractional": q_fractional_records,
                "q_definition": q_definition,
                "q_reciprocal_index": None if reciprocal_index is None else reciprocal_index.astype(int).tolist(),
                "period_angstrom": q0_period,
                "chirality": [int(np.sign(value)) for value in multipliers],
                "phase_radian": phase0,
                "plane_normal": plane_normal.tolist(),
                "cone_component": cone,
                "supercell": [1, 1, 1],
                "commensurate": bool(params.require_commensurate),
            },
        )
        return self._finish(outputs, records)

    def generate_magnetoelastic(self, structure: Atoms, params: MagnetoelasticResponseParams) -> list[Atoms]:
        moments, source_id, parent = self._base(structure)
        strains = _parse_scan(params.structural_scan, minimum_points=3)
        angles = _parse_scan(params.spin_scan_deg, minimum_points=3)
        if 0.0 not in strains or 0.0 not in angles:
            raise ValueError("magnetoelastic structural and spin scans must include zero")
        target_helper = SmallAngleSpinTiltOperation()
        tilt_params = SmallAngleSpinTiltParams(
            canting_mode="Single-spin tilt", target_mode="Explicit indices",
            target_indices=params.target_indices, axis=params.rotation_axis,
            reference_direction=(1.0, 0.0, 0.0),
        )
        targets = target_helper.candidate_indices(structure, moments, tilt_params)
        if not targets:
            raise ValueError("magnetoelastic response matched no target atoms")
        outputs: list[Atoms] = []
        records: list[ResponseManifestRecord] = []
        rotation_axis = normalize_vector(np.asarray(params.rotation_axis, dtype=float))
        strain_axis = None
        shear_direction = None
        if params.structural_mode in {"Uniaxial strain", "Biaxial strain", "Symmetric shear"}:
            strain_axis = normalize_vector(np.asarray(params.strain_axis, dtype=float))
        if params.structural_mode == "Symmetric shear":
            shear_direction = normalize_vector(np.asarray(params.shear_direction, dtype=float))
            if abs(float(np.dot(strain_axis, shear_direction))) > 1.0e-7:
                raise CardOperationError(
                    "magnetoelastic_nonorthogonal_shear_directions",
                    "The two symmetric-shear directions must be perpendicular Cartesian vectors.",
                )
        bain_axis = str(params.bain_axis).strip().lower()
        if params.structural_mode == "Bain / tetragonal" and bain_axis not in {"a", "b", "c"}:
            raise CardOperationError(
                "magnetoelastic_invalid_bain_axis",
                "The Bain lattice axis must be a, b, or c.",
            )
        for strain in strains:
            geometry = structure.copy()
            if params.structural_mode == "Isotropic volume":
                deformation = np.eye(3) * (1.0 + float(strain)) ** (1.0 / 3.0)
                structural_probe = "volume"
            elif params.structural_mode == "Uniaxial strain":
                deformation = np.eye(3) + float(strain) * np.outer(strain_axis, strain_axis)
                structural_probe = "strain"
            elif params.structural_mode == "Biaxial strain":
                deformation = np.eye(3) + float(strain) * (
                    np.eye(3) - np.outer(strain_axis, strain_axis)
                )
                structural_probe = "strain"
            elif params.structural_mode == "Symmetric shear":
                deformation = np.eye(3) + 0.5 * float(strain) * (
                    np.outer(strain_axis, shear_direction)
                    + np.outer(shear_direction, strain_axis)
                )
                structural_probe = "strain"
            elif params.structural_mode == "Bain / tetragonal":
                result = BainPathOperation().run_structure(
                    structure,
                    BainPathParams(
                        axis={"a": "x", "b": "y", "c": "z"}[bain_axis],
                        ca_range=(1.0 + float(strain), 1.0 + float(strain), 1.0),
                        mode="constant_volume",
                    ),
                )
                geometry = result[0]
                deformation = geometry.cell.array.T @ np.linalg.inv(structure.cell.array.T)
                structural_probe = "strain"
            else:
                raise ValueError(f"unsupported structural response mode: {params.structural_mode}")
            if params.structural_mode != "Bain / tetragonal":
                geometry.set_cell(structure.cell.array @ deformation.T, scale_atoms=True)
            frames = []
            for angle in angles:
                spins = moments.copy()
                rotation = _rotation_matrix(rotation_axis, math.radians(float(angle)))
                spins[targets] = spins[targets] @ rotation.T
                frames.append((math.radians(angle), spins, geometry))
            emitted = self._emit_group(
                structure, source_id=source_id, parent=parent,
                group_key=[
                    params.structural_mode,
                    float(strain),
                    targets,
                    rotation_axis.tolist(),
                    None if strain_axis is None else strain_axis.tolist(),
                    None if shear_direction is None else shear_direction.tolist(),
                    bain_axis if params.structural_mode == "Bain / tetragonal" else None,
                ], probe="rotation",
                kind="magnetoelastic_spin_probe", frames=frames,
                max_outputs=params.max_outputs, records=records, outputs=outputs,
                target_indices=targets, rotation_axis=rotation_axis,
                metadata={
                    "structural_probe": structural_probe,
                    "structural_coordinate": float(strain),
                    "deformation_gradient": deformation.tolist(),
                    "strain_tensor": (0.5 * (deformation + deformation.T) - np.eye(3)).tolist(),
                    "position_convention": "fractional coordinates fixed under cell deformation",
                    "strain_axis_cartesian": None if strain_axis is None else strain_axis.tolist(),
                    "shear_direction_cartesian": None if shear_direction is None else shear_direction.tolist(),
                    "bain_lattice_axis": bain_axis if params.structural_mode == "Bain / tetragonal" else None,
                },
            )
            if not emitted:
                break
        return self._finish(outputs, records)


def derived_spin_tangent(group: Sequence[Atoms]) -> np.ndarray:
    """Derive dS/dcoordinate from sorted grouped spins without persisting it."""
    if len(group) < 3:
        raise ValueError("at least three response coordinates are required for a tangent")
    coordinates = np.asarray([float(atoms.info["response_coordinate"]) for atoms in group])
    if len(np.unique(coordinates)) != len(coordinates):
        raise ValueError("response coordinates must be unique")
    order = np.argsort(coordinates)
    coordinates = coordinates[order]
    spins = np.asarray([np.asarray(group[index].arrays["spin"], dtype=float) for index in order])
    return np.gradient(spins, coordinates, axis=0, edge_order=2)


def audit_response_groups(dataset: Sequence[Atoms]) -> dict[str, Any]:
    """Audit completeness and physical invariants of response groups."""
    groups: dict[str, list[Atoms]] = {}
    ungrouped = 0
    for atoms in dataset:
        group = atoms.info.get("response_group")
        if group is None:
            ungrouped += 1
            continue
        groups.setdefault(str(group), []).append(atoms)
    invalid: dict[str, list[str]] = {}
    summaries = []
    for group_id, frames in groups.items():
        reasons: list[str] = []
        coordinates = [float(frame.info.get("response_coordinate", np.nan)) for frame in frames]
        if len(frames) < 3:
            reasons.append("fewer than three coordinates")
        if not np.isfinite(coordinates).all():
            reasons.append("missing or non-finite coordinate")
        if len(set(coordinates)) != len(coordinates):
            reasons.append("duplicate coordinate")
        if not any(abs(value) <= 1.0e-14 for value in coordinates):
            reasons.append("missing reference coordinate")
        positives = sorted(value for value in coordinates if value > 1.0e-14)
        negatives = sorted(-value for value in coordinates if value < -1.0e-14)
        if len(positives) != len(negatives) or not np.allclose(positives, negatives, atol=1.0e-12, rtol=0.0):
            reasons.append("unpaired plus/minus branches")
        reference = frames[0]
        numbers = reference.numbers
        positions = reference.positions
        cell = reference.cell.array
        for frame in frames[1:]:
            if len(frame) != len(reference):
                reasons.append("mixed atom counts")
                break
            if not np.array_equal(frame.numbers, numbers):
                reasons.append("mixed species/order")
                break
            if not np.array_equal(frame.positions, positions) or not np.array_equal(frame.cell.array, cell):
                reasons.append("mixed geometry/cell")
                break
        probe = str(reference.info.get("response_probe", ""))
        kind = str(reference.info.get("response_kind", ""))
        if probe in {"rotation", "chirality"}:
            base_norms = np.linalg.norm(np.asarray(reference.arrays.get("spin"), dtype=float), axis=1)
            final_spins = any(frame.info.get("response_spin_provenance") == "dft_final" for frame in frames)
            if final_spins:
                ordered = sorted(frames, key=lambda item: float(item.info["response_coordinate"]))
                spin_path = np.asarray([frame.arrays.get("spin") for frame in ordered], dtype=float)
                norms = np.linalg.norm(spin_path, axis=2)
                reference_scale = np.maximum(np.median(norms, axis=0), 1.0e-12)
                if np.any(norms < 0.2 * reference_scale[None, :]):
                    reasons.append("SCF spin magnitude collapsed")
                unit = np.divide(spin_path, norms[:, :, None], out=np.zeros_like(spin_path), where=norms[:, :, None] > 1.0e-12)
                if len(unit) > 1 and np.any(np.sum(unit[1:] * unit[:-1], axis=2) < 0.0):
                    reasons.append("SCF spin path flipped or became discontinuous")
            else:
                for frame in frames[1:]:
                    norms = np.linalg.norm(np.asarray(frame.arrays.get("spin"), dtype=float), axis=1)
                    if not np.allclose(norms, base_norms, atol=1.0e-10, rtol=1.0e-10):
                        reasons.append("spin magnitudes changed within angular group")
                        break
        if kind.startswith("global_anisotropy"):
            base = np.asarray(reference.arrays["spin"], dtype=float)
            base_dots = base @ base.T
            for frame in frames[1:]:
                spin = np.asarray(frame.arrays["spin"], dtype=float)
                if not np.allclose(spin @ spin.T, base_dots, atol=1.0e-10, rtol=1.0e-10):
                    reasons.append("global rotation changed pair spin dot products")
                    break
        if reasons:
            invalid[group_id] = sorted(set(reasons))
        summaries.append({"response_group": group_id, "count": len(frames), "valid": not reasons})
    return {
        "response_schema": RESPONSE_SCHEMA,
        "group_count": len(groups),
        "valid_group_count": len(groups) - len(invalid),
        "invalid_groups": invalid,
        "ungrouped_count": ungrouped,
        "groups": summaries,
    }


def write_response_audit(
    dataset: Sequence[Atoms], output_dir: str | Path, *, prefix: str = "magnetic-response"
) -> dict[str, Path]:
    """Write JSON/CSV/PNG audit artifacts, including labelled response curves."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = audit_response_groups(dataset)
    json_path = output / f"{prefix}-audit.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output / f"{prefix}-curves.csv"
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[Atoms]] = {}
    for atoms in dataset:
        if "response_group" in atoms.info:
            groups.setdefault(str(atoms.info["response_group"]), []).append(atoms)
    for group_id, frames in groups.items():
        ordered = sorted(frames, key=lambda item: float(item.info["response_coordinate"]))
        tangents = None
        if len(ordered) >= 3 and all("spin" in frame.arrays for frame in ordered):
            tangents = derived_spin_tangent(ordered)
        for index, atoms in enumerate(ordered):
            row = {
                "response_group": group_id,
                "response_kind": atoms.info.get("response_kind", ""),
                "coordinate": float(atoms.info["response_coordinate"]),
                "branch": atoms.info.get("response_branch", ""),
                "energy": atoms.info.get("energy", np.nan),
                "g": np.nan,
                "energy_even": np.nan,
                "energy_odd": np.nan,
                "g_even": np.nan,
                "g_odd": np.nan,
            }
            if tangents is not None and "mforce" in atoms.arrays:
                row["g"] = float(np.sum(np.asarray(atoms.arrays["mforce"], dtype=float) * tangents[index]))
            rows.append(row)
    for group_rows in _rows_by_group(rows).values():
        by_coordinate = {round(float(row["coordinate"]), 14): row for row in group_rows}
        for row in group_rows:
            partner = by_coordinate.get(round(-float(row["coordinate"]), 14))
            if partner is None:
                continue
            for key in ("energy", "g"):
                left = float(row[key])
                right = float(partner[key])
                if np.isfinite(left) and np.isfinite(right):
                    row[f"{key}_even"] = 0.5 * (left + right)
                    row[f"{key}_odd"] = 0.5 * (left - right)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "response_group", "response_kind", "coordinate", "branch", "energy", "g",
            "energy_even", "energy_odd", "g_even", "g_odd",
        ])
        writer.writeheader()
        writer.writerows(rows)
    png_path = output / f"{prefix}-curves.png"
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for group_id, group_rows in _rows_by_group(rows).items():
            x = [row["coordinate"] for row in group_rows]
            energy = np.asarray([row["energy"] for row in group_rows], dtype=float)
            g_values = np.asarray([row["g"] for row in group_rows], dtype=float)
            if np.isfinite(energy).any():
                axes[0].plot(x, energy, marker="o", label=group_id[-8:])
            if np.isfinite(g_values).any():
                axes[1].plot(x, g_values, marker="o", label=group_id[-8:])
        axes[0].set(xlabel="response coordinate", ylabel="E", title="Response energy")
        axes[1].set(xlabel="response coordinate", ylabel="g = sum mforce dot dS/dx", title="Conservative response")
        for axis_plot in axes:
            if axis_plot.lines:
                axis_plot.legend(fontsize=7)
            axis_plot.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(png_path, dpi=160)
        plt.close(figure)
    except ImportError:
        png_path = Path()
    return {"json": json_path, "csv": csv_path, "png": png_path}


def _rows_by_group(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["response_group"]), []).append(row)
    return result


__all__ = [
    "RESPONSE_SCHEMA",
    "LocalMagneticResponseParams",
    "TextureMagneticResponseParams",
    "MagnetoelasticResponseParams",
    "MagneticResponseScanOperation",
    "ResponseManifest",
    "ResponseManifestRecord",
    "audit_response_groups",
    "derived_spin_tangent",
    "structure_fingerprint",
    "write_response_audit",
]
