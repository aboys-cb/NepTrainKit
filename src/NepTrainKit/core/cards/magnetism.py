"""UI-independent magnetism Make Dataset operations."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np
from ase.data import atomic_numbers
from ase.geometry import get_distances

from NepTrainKit.core.config_type import append_config_tag, sanitize_config_tag, stable_config_id
from NepTrainKit.core.magnetism import (
    element_mask,
    existing_moment_magnitudes,
    existing_moment_scalars,
    existing_moment_vectors,
    existing_moments,
    kvec_signs,
    mapped_moment_magnitudes,
    mapped_moment_vectors,
    normalize_vector,
    orthonormal_frame,
    parse_element_set,
    parse_kvec as parse_layer_kvec,
    parse_magmom_map_any,
    random_signs,
    random_vector_moments,
    set_initial_magmoms_safe,
    spiral_unit_vectors,
)

from .errors import CardOperationError
from .operation import StructureOperation
from .sampling import derived_structure_seed


def parse_angle_list(text: str) -> list[float]:
    """Parse a comma-separated list of positive tilt angles in degrees."""
    values: list[float] = []
    seen: set[float] = set()
    for token in re.split(r"[\s,;]+", text or ""):
        if not token.strip():
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if value <= 0:
            continue
        rounded = float(np.round(value, 12))
        if rounded in seen:
            continue
        seen.add(rounded)
        values.append(rounded)
    return values


def parse_atom_indices(text: str, natoms: int) -> list[int]:
    """Parse 1-based atom indices from tokens such as ``1,3-5``."""
    indices: list[int] = []
    seen: set[int] = set()
    for token in re.split(r"[\s,;]+", text or ""):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            raw_values = range(start, end + 1)
        else:
            try:
                raw_values = [int(token)]
            except ValueError:
                continue

        for raw in raw_values:
            idx = raw - 1
            if idx < 0 or idx >= natoms or idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
    return indices


def parse_pair_filter(text: str, *, normalize_case: bool = False) -> set[tuple[str, str]]:
    """Parse pair filters such as ``Fe-Co,Fe-Fe`` into canonical tuple pairs."""
    pairs: set[tuple[str, str]] = set()
    for token in re.split(r"[\s,;]+", text or ""):
        token = token.strip()
        if not token:
            continue
        if token.count("-") != 1:
            raise ValueError(f"Invalid pair filter '{token}', expected A-B.")
        left, right = [part.strip() for part in token.split("-", 1)]
        if not left or not right:
            raise ValueError(f"Invalid pair filter '{token}', expected A-B.")
        if normalize_case:
            left = left[0].upper() + left[1:].lower()
            right = right[0].upper() + right[1:].lower()
        pairs.add(tuple(sorted((left, right))))
    return pairs


def range_values(values: list[float], *, minimum: float | None = None) -> list[float]:
    """Expand a [start, stop, step] scan triplet into stable float values."""
    if len(values) != 3:
        raise ValueError("Scan range must contain exactly three values: start, stop, step.")
    start, stop, step = [float(v) for v in values]
    if not np.all(np.isfinite([start, stop, step])):
        raise ValueError("Scan range values must be finite.")
    if step <= 0:
        raise ValueError("Scan range step must be positive.")
    if minimum is not None:
        start = max(start, minimum)
        stop = max(stop, minimum)
    if stop < start:
        start, stop = stop, start
    if abs(stop - start) <= 1e-12:
        return [start]
    raw = list(np.arange(start, stop + abs(step) * 0.5, abs(step), dtype=float))
    if not raw:
        raw = [start]
    ordered: list[float] = []
    for value in raw:
        rounded = float(np.round(value, 12))
        if not ordered or abs(rounded - ordered[-1]) > 1e-10:
            ordered.append(rounded)
    return ordered


def int_range_values(values: list[int], *, minimum: int = 1) -> list[int]:
    """Expand a [start, stop, step] integer scan triplet."""
    if len(values) != 3:
        raise ValueError("Integer scan range must contain exactly three values: start, stop, step.")
    start, stop, step = [int(v) for v in values]
    if step <= 0:
        raise ValueError("Integer scan range step must be positive.")
    start = max(start, minimum)
    stop = max(stop, minimum)
    if stop < start:
        start, stop = stop, start
    if stop == start:
        return [start]
    return list(range(start, stop + 1, step))


def axis_tag(axis: np.ndarray, *, precision: int = 6) -> str:
    """Return an EXTXYZ-friendly axis tag without quotes or spaces."""
    v = np.asarray(axis, dtype=float).reshape(3)
    basis = [
        (np.array([1.0, 0.0, 0.0]), "100"),
        (np.array([0.0, 1.0, 0.0]), "010"),
        (np.array([0.0, 0.0, 1.0]), "001"),
    ]
    for ref, tag in basis:
        if np.allclose(v, ref, atol=1e-8, rtol=0.0):
            return tag
        if np.allclose(v, -ref, atol=1e-8, rtol=0.0):
            return f"-{tag}"
    fmt = f".{precision}g"
    return f"{v[0]:{fmt}},{v[1]:{fmt}},{v[2]:{fmt}}"


def coerce_scan_triplet(values, *, default_step: float) -> list[float]:
    """Normalize legacy scalar/list scan values to a [start, stop, step] triplet."""
    if isinstance(values, (int, float)):
        scalar = float(values)
        return [scalar, scalar, float(default_step)]
    if not isinstance(values, (list, tuple)):
        return [0.0, 0.0, float(default_step)]
    if len(values) >= 3:
        return [float(values[0]), float(values[1]), float(values[2])]
    if len(values) == 2:
        return [float(values[0]), float(values[1]), float(default_step)]
    if len(values) == 1:
        scalar = float(values[0])
        return [scalar, scalar, float(default_step)]
    return [0.0, 0.0, float(default_step)]


@dataclass(frozen=True)
class MagneticMomentRotationParams:
    elements: str = ""
    max_angle: float = 10.0
    num_structures: int = 5
    lift_scalar: bool = True
    axis: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0)
    disturb_magnitude: bool = True
    magnitude_factor: list[float] | tuple[float, float] = (0.95, 1.05)
    use_seed: bool = False
    seed: int = 0


class MagneticMomentRotationOperation(StructureOperation):
    """Sample random spin directions in spherical caps and optionally rescale them."""

    @staticmethod
    def sample_spherical_cap(
        vectors: np.ndarray,
        max_angle_deg: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample directions uniformly in solid angle around each input vector."""
        vec = np.asarray(vectors, dtype=float)
        if vec.size == 0:
            return vec.copy(), np.empty(0, dtype=float)

        lengths = np.linalg.norm(vec, axis=1)
        base = vec / lengths[:, None]
        reference = np.zeros_like(base)
        use_x = np.abs(base[:, 2]) > 0.9
        reference[~use_x, 2] = 1.0
        reference[use_x, 0] = 1.0
        tangent_x = np.cross(reference, base)
        tangent_x /= np.linalg.norm(tangent_x, axis=1)[:, None]
        tangent_y = np.cross(base, tangent_x)

        cos_min = math.cos(math.radians(float(max_angle_deg)))
        cos_theta = rng.uniform(cos_min, 1.0, size=len(vec))
        sin_theta = np.sqrt(np.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
        phi = rng.uniform(0.0, 2.0 * math.pi, size=len(vec))
        radial = np.cos(phi)[:, None] * tangent_x + np.sin(phi)[:, None] * tangent_y
        directions = cos_theta[:, None] * base + sin_theta[:, None] * radial
        angles = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return directions * lengths[:, None], angles

    @staticmethod
    def rescale_vectors(vectors: np.ndarray, target_lengths: np.ndarray) -> np.ndarray:
        vec = np.asarray(vectors, dtype=float)
        targets = np.asarray(target_lengths, dtype=float)
        current = np.linalg.norm(vec, axis=1)
        out = np.zeros_like(vec)
        mask = (current > 1e-12) & (targets != 0.0)
        out[mask] = vec[mask] / current[mask, None] * targets[mask, None]
        return out

    def run_structure(self, structure, params: MagneticMomentRotationParams) -> list:
        num_structures_value = float(params.num_structures)
        if not np.isfinite(num_structures_value) or not num_structures_value.is_integer():
            raise CardOperationError(
                "magmom-rotation-output-integer",
                "Structures per input must be an integer.",
            )
        num_structures = int(num_structures_value)
        if num_structures <= 0:
            raise CardOperationError(
                "magmom-rotation-output-count",
                "Structures per input must be at least 1.",
            )

        max_angle = float(params.max_angle)
        if not np.isfinite(max_angle) or not 0.0 <= max_angle <= 180.0:
            raise CardOperationError(
                "magmom-rotation-angle",
                "Maximum perturbation angle must be between 0 and 180 degrees.",
            )

        raw_magmoms = existing_moments(structure)
        if raw_magmoms is None:
            raise CardOperationError(
                "magmom-rotation-missing-moments",
                "Spin Perturbation requires spin or initial magnetic moments on every input structure.",
            )

        is_vector = raw_magmoms.ndim == 2 and raw_magmoms.shape == (len(structure), 3)
        if not is_vector and not params.lift_scalar:
            raise CardOperationError(
                "magmom-rotation-scalar-disabled",
                "Scalar magnetic moments must be lifted to vectors before they can be rotated.",
            )
        axis = np.array([0.0, 0.0, 1.0])
        if not is_vector:
            axis_values = np.asarray(params.axis, dtype=float)
            if axis_values.shape != (3,) or not np.isfinite(axis_values).all():
                raise CardOperationError(
                    "magmom-rotation-axis-finite",
                    "The scalar lift direction must contain three finite Cartesian components.",
                )
            if np.linalg.norm(axis_values) <= 1e-12:
                raise CardOperationError(
                    "magmom-rotation-axis-zero",
                    "The scalar lift direction must be non-zero.",
                )
            axis = normalize_vector(axis_values)
        base_vectors = existing_moment_vectors(
            structure,
            axis=axis,
            lift_scalar=bool(params.lift_scalar),
        )
        if base_vectors is None:
            raise CardOperationError(
                "magmom-rotation-unusable-moments",
                "Spin Perturbation could not convert the input magnetic moments to vectors.",
            )

        min_factor, max_factor = [float(v) for v in params.magnitude_factor]
        if params.disturb_magnitude:
            if (
                not np.isfinite([min_factor, max_factor]).all()
                or min_factor < 0.0
                or max_factor < 0.0
            ):
                raise CardOperationError(
                    "magmom-rotation-magnitude-range",
                    "Magnitude scale bounds must be finite and non-negative.",
                )
            if min_factor > max_factor:
                raise CardOperationError(
                    "magmom-rotation-magnitude-order",
                    "Magnitude scale minimum must not exceed the maximum.",
                )
        else:
            min_factor = max_factor = 1.0
        if max_angle == 0.0 and (
            not params.disturb_magnitude
            or (min_factor == 1.0 and max_factor == 1.0)
        ):
            raise CardOperationError(
                "magmom-rotation-no-change",
                "Increase the perturbation angle or enable a magnitude scale range that changes the moments.",
            )

        base_seed = (
            derived_structure_seed(int(params.seed), structure)
            if params.use_seed
            else None
        )
        rng = np.random.default_rng(base_seed)
        elements = parse_element_set(params.elements)
        invalid_elements = sorted(symbol for symbol in elements if atomic_numbers.get(symbol, 0) <= 0)
        if invalid_elements:
            raise CardOperationError(
                "magmom-rotation-invalid-elements",
                "Invalid element symbols in the target list: {elements}.",
                elements=", ".join(invalid_elements),
            )

        results = []
        symbols = structure.get_chemical_symbols()
        if not elements:
            elements = set(symbols)
        selected_mask = np.array([symbol in elements for symbol in symbols], dtype=bool)
        base_lengths = np.linalg.norm(base_vectors, axis=1)
        selected_mask &= base_lengths > 1e-12
        selected_indices = np.nonzero(selected_mask)[0]
        if selected_indices.size == 0:
            raise CardOperationError(
                "magmom-rotation-no-targets",
                "No non-zero magnetic moments match the selected elements.",
            )

        can_rotate = max_angle > 0.0
        for sample_index in range(num_structures):
            new_structure = structure.copy()
            moment_array = np.array(base_vectors, copy=True)
            scales = np.ones(selected_indices.size, dtype=float)
            realized_angles = np.zeros(selected_indices.size, dtype=float)
            if can_rotate:
                rotated, realized_angles = self.sample_spherical_cap(
                    base_vectors[selected_indices], max_angle, rng
                )
                if params.disturb_magnitude:
                    scales = rng.uniform(min_factor, max_factor, size=selected_indices.size)
                    rotated = self.rescale_vectors(rotated, base_lengths[selected_indices] * scales)
                moment_array[selected_indices] = rotated
            elif params.disturb_magnitude:
                scales = rng.uniform(min_factor, max_factor, size=selected_indices.size)
                moment_array[selected_indices] *= scales[:, None]
            set_initial_magmoms_safe(new_structure, moment_array)
            label = "SpinPert" if can_rotate else "SpinScale"
            details = []
            if can_rotate:
                details.append(f"a={float(params.max_angle):.1f}")
            if params.disturb_magnitude:
                details.append(f"s={min_factor:.2f}-{max_factor:.2f}")
            append_config_tag(new_structure, label + (("(" + ",".join(details) + ")") if details else ""))
            new_structure.info["spin_perturbation"] = json.dumps(
                {
                    "direction_distribution": "uniform_spherical_cap",
                    "elements": sorted(elements),
                    "max_angle": max_angle,
                    "num_structures": num_structures,
                    "lifted_scalar": not is_vector,
                    "scalar_lift_direction": axis.tolist() if not is_vector else None,
                    "disturb_magnitude": bool(params.disturb_magnitude),
                    "magnitude_factor": [min_factor, max_factor],
                    "seed": int(params.seed) if params.use_seed else None,
                    "derived_seed": base_seed,
                    "sample_index": sample_index,
                    "selected_count": int(selected_indices.size),
                    "realized_angle_max": float(np.max(realized_angles)),
                    "realized_angle_mean": float(np.mean(realized_angles)),
                    "sampled_scale_min": float(np.min(scales)),
                    "sampled_scale_max": float(np.max(scales)),
                },
                separators=(",", ":"),
            )
            results.append(new_structure)
        return results


@dataclass(frozen=True)
class SetMagneticMomentsParams:
    source: str = "Map/default magnitude"
    format: str = "Non-collinear (vector)"
    axis: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0)
    magmom_map: str = ""
    use_element_dirs: bool = False
    default_moment: float = 0.0
    constant_moment: float = 2.0
    lift_scalar: bool = True
    apply_elements: str = ""


class SetMagneticMomentsOperation(StructureOperation):
    """Set magnetic moments from existing data, maps, or constants."""

    def run_structure(self, structure, params: SetMagneticMomentsParams) -> list:
        vector_output = params.format == "Non-collinear (vector)"
        moments = self._vector_moments(structure, params) if vector_output else self._scalar_moments(structure, params)
        if moments is None:
            return [structure.copy()]

        atoms = structure.copy()
        set_initial_magmoms_safe(
            atoms,
            np.asarray(moments, dtype=float),
            axis=np.asarray(params.axis, dtype=float),
        )
        source_map = {
            "Existing initial magmoms": "existing",
            "Map/default magnitude": "map",
            "Constant magnitude": "const",
        }
        fmt_tag = "vec" if vector_output else "sca"
        append_config_tag(atoms, f"MagSet({source_map.get(params.source, 'map')},{fmt_tag})")
        return [atoms]

    def _constant_magnitudes(self, structure, params: SetMagneticMomentsParams) -> np.ndarray:
        magnitude = abs(float(params.constant_moment))
        selected = parse_element_set(params.apply_elements)
        mask = (
            np.array([sym in selected for sym in structure.get_chemical_symbols()], dtype=bool)
            if selected
            else np.ones(len(structure), dtype=bool)
        )
        values = np.zeros(len(structure), dtype=float)
        values[mask] = magnitude
        return values

    def _scalar_moments(self, structure, params: SetMagneticMomentsParams) -> np.ndarray | None:
        selected = parse_element_set(params.apply_elements)
        axis = normalize_vector(np.array(params.axis, dtype=float))
        if params.source == "Existing initial magmoms":
            values = existing_moment_scalars(structure, axis=axis)
            if values is None:
                return None
            if selected:
                mask = np.array([sym in selected for sym in structure.get_chemical_symbols()], dtype=bool)
                values = np.where(mask, values, 0.0)
            return values
        if params.source == "Constant magnitude":
            return self._constant_magnitudes(structure, params)
        return mapped_moment_magnitudes(
            structure,
            parse_magmom_map_any(params.magmom_map),
            default_moment=float(params.default_moment),
            apply_elements=selected,
        )

    def _vector_moments(self, structure, params: SetMagneticMomentsParams) -> np.ndarray | None:
        selected = parse_element_set(params.apply_elements)
        axis = normalize_vector(np.array(params.axis, dtype=float))
        if params.source == "Existing initial magmoms":
            values = existing_moment_vectors(structure, axis=axis, lift_scalar=params.lift_scalar)
            if values is None:
                return None
            if selected:
                mask = np.array([sym in selected for sym in structure.get_chemical_symbols()], dtype=bool)
                values = np.where(mask[:, None], values, 0.0)
            return values
        if params.source == "Constant magnitude":
            return self._constant_magnitudes(structure, params).reshape(-1, 1) * axis.reshape(1, 3)
        return mapped_moment_vectors(
            structure,
            parse_magmom_map_any(params.magmom_map),
            default_moment=float(params.default_moment),
            axis=axis,
            apply_elements=selected,
            use_element_directions=params.use_element_dirs,
        )


@dataclass(frozen=True)
class MagneticOrderParams:
    format: str = "collinear"
    axis: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0)
    magmom_map: str = ""
    use_element_dirs: bool = False
    default_moment: float = 0.0
    apply_elements: str = ""
    gen_fm: bool = True
    gen_afm: bool = False
    afm_mode: str = "k_vector"
    afm_kvec: str = "111"
    afm_group_a: str = "A"
    afm_group_b: str = "B"
    afm_zero_unknown: bool = True
    gen_pm: bool = False
    pm_count: int = 10
    pm_direction: str = "sphere"
    pm_cone_angle: float = 30.0
    pm_balanced: bool = True
    use_seed: bool = False
    seed: int = 0
    max_outputs: int = 100


@dataclass(frozen=True)
class MagneticOrderPreview:
    """Compact first-input summary for the Magnetic Order card."""

    output_count: int
    magnetic_atoms: int
    afm_positive: int = 0
    afm_negative: int = 0
    afm_zero: int = 0


@dataclass(frozen=True)
class _MagneticOrderState:
    spin_model: str
    afm_mode: str
    pm_direction: str
    axis: np.ndarray
    magnitudes: np.ndarray
    directions: np.ndarray
    afm_signs: np.ndarray | None
    output_count: int


class MagneticOrderOperation(StructureOperation):
    """Assign FM, AFM, and PM magnetic order patterns."""

    _FORMAT_ALIASES = {
        "collinear": "collinear",
        "Collinear (scalar)": "collinear",
        "noncollinear": "noncollinear",
        "Non-collinear (vector)": "noncollinear",
    }
    _AFM_MODE_ALIASES = {
        "k_vector": "k_vector",
        "k-vector": "k_vector",
        "group_ab": "group_ab",
        "group A/B": "group_ab",
    }
    _PM_DIRECTIONS = {"sphere", "cone", "plane", "axis"}

    @staticmethod
    def parse_kvec(text: str) -> tuple[int, int, int]:
        return parse_layer_kvec(text)

    @classmethod
    def normalize_format(cls, value: str) -> str:
        return cls._normalize_choice(value, cls._FORMAT_ALIASES, "spin model")

    @classmethod
    def normalize_afm_mode(cls, value: str) -> str:
        return cls._normalize_choice(value, cls._AFM_MODE_ALIASES, "AFM mode")

    def run_structure(self, structure, params: MagneticOrderParams) -> list:
        state = self._prepare(structure, params)
        outputs = []

        base_seed = int(params.seed) if params.use_seed else None
        cfg_id = stable_config_id(structure)
        noncollinear = state.spin_model == "noncollinear"

        if params.gen_fm:
            signs = np.ones(len(structure), dtype=float)
            moms = self._moments_from_signs(state, signs)
            atoms = structure.copy()
            set_initial_magmoms_safe(
                atoms,
                moms,
                axis=state.axis,
            )
            append_config_tag(atoms, "MagFMnc" if noncollinear else "MagFM")
            outputs.append(atoms)

        if params.gen_afm:
            signs = np.asarray(state.afm_signs, dtype=float)
            moms = self._moments_from_signs(state, signs)
            atoms = structure.copy()
            set_initial_magmoms_safe(
                atoms,
                moms,
                axis=state.axis,
            )
            if state.afm_mode == "k_vector":
                k = self.parse_kvec(params.afm_kvec)
                append_config_tag(atoms, f"MagAFM{k[0]}{k[1]}{k[2]}" + ("nc" if noncollinear else ""))
            else:
                append_config_tag(atoms, "MagAFMg" + ("nc" if noncollinear else ""))
            outputs.append(atoms)

        if params.gen_pm:
            for i in range(max(int(params.pm_count), 1)):
                if base_seed is None:
                    rng = np.random.default_rng()
                    seed_note = ""
                else:
                    derived_seed = int(base_seed + cfg_id * 1000003 + i)
                    rng = np.random.default_rng(derived_seed)
                    seed_note = f"s{derived_seed}"
                if noncollinear:
                    moms = random_vector_moments(
                        state.magnitudes,
                        rng=rng,
                        direction_mode=state.pm_direction,
                        axis=state.axis,
                        max_angle_deg=float(params.pm_cone_angle),
                        balanced=params.pm_balanced,
                    )
                else:
                    signs = random_signs(len(structure), rng=rng, balanced=params.pm_balanced)
                    moms = signs * state.magnitudes
                atoms = structure.copy()
                set_initial_magmoms_safe(
                    atoms,
                    moms,
                    axis=state.axis,
                )
                base = "MagPM" + ("nc" if noncollinear else "")
                append_config_tag(atoms, base + (f"_{seed_note}" if seed_note else ""))
                outputs.append(atoms)

        return outputs

    def preview(self, structure, params: MagneticOrderParams) -> MagneticOrderPreview:
        """Validate parameters and summarize the first input without generating outputs."""
        state = self._prepare(structure, params)
        positive = negative = zero = 0
        if state.afm_signs is not None:
            magnetic = state.magnitudes > 1.0e-12
            positive = int(np.count_nonzero((state.afm_signs > 0.0) & magnetic))
            negative = int(np.count_nonzero((state.afm_signs < 0.0) & magnetic))
            zero = int(np.count_nonzero((state.afm_signs == 0.0) & magnetic))
        return MagneticOrderPreview(
            output_count=state.output_count,
            magnetic_atoms=int(np.count_nonzero(state.magnitudes > 1.0e-12)),
            afm_positive=positive,
            afm_negative=negative,
            afm_zero=zero,
        )

    def _prepare(self, structure, params: MagneticOrderParams) -> _MagneticOrderState:
        spin_model = self.normalize_format(params.format)
        if not (params.gen_fm or params.gen_afm or params.gen_pm):
            raise ValueError("MagneticOrder: select at least one of FM, AFM, or PM.")

        axis = np.asarray(params.axis, dtype=float)
        if (
            axis.shape != (3,)
            or not np.isfinite(axis).all()
            or float(np.linalg.norm(axis)) <= 1.0e-12
        ):
            raise ValueError("MagneticOrder: reference axis must be a finite nonzero 3-vector.")
        axis = normalize_vector(axis)

        magnitudes, directions = self._per_atom_mags_and_dirs(structure, params)
        magnetic_mask = magnitudes > 1.0e-12
        if not np.any(magnetic_mask):
            raise ValueError(
                "MagneticOrder: no nonzero magnetic moments; set an element moment "
                "or a nonzero unlisted-element moment."
            )

        output_count = int(bool(params.gen_fm)) + int(bool(params.gen_afm))
        pm_direction = "axis" if spin_model == "collinear" else "sphere"
        if params.gen_pm:
            if spin_model == "noncollinear":
                pm_direction = str(params.pm_direction or "").strip().lower()
                if pm_direction not in self._PM_DIRECTIONS:
                    raise ValueError(
                        f"MagneticOrder: unsupported PM direction {params.pm_direction!r}."
                    )
            pm_count = int(params.pm_count)
            if pm_count < 1:
                raise ValueError("MagneticOrder: PM structures must be at least 1.")
            if (
                spin_model == "noncollinear"
                and pm_direction == "cone"
                and (
                    not np.isfinite(float(params.pm_cone_angle))
                    or not 0.0 <= float(params.pm_cone_angle) <= 180.0
                )
            ):
                raise ValueError("MagneticOrder: PM cone angle must be between 0 and 180 degrees.")
            if params.use_seed and int(params.seed) < 0:
                raise ValueError("MagneticOrder: seed must be nonnegative.")
            output_count += pm_count

        if params.gen_pm:
            max_outputs = int(params.max_outputs)
            if max_outputs < 1:
                raise ValueError("MagneticOrder: max outputs must be at least 1.")
            if output_count > max_outputs:
                raise ValueError(
                    f"MagneticOrder: requested {output_count} outputs per input, "
                    f"above the limit of {max_outputs}."
                )

        afm_mode = "k_vector"
        afm_signs = None
        if params.gen_afm:
            afm_mode = self.normalize_afm_mode(params.afm_mode)
            if afm_mode == "group_ab":
                afm_signs = self._group_signs(structure, params, magnetic_mask)
            else:
                afm_signs = kvec_signs(
                    structure,
                    self.parse_kvec(params.afm_kvec),
                )
                positive = np.count_nonzero((afm_signs > 0.0) & magnetic_mask)
                negative = np.count_nonzero((afm_signs < 0.0) & magnetic_mask)
                if positive == 0 or negative == 0:
                    raise ValueError(
                        "MagneticOrder: the selected k-vector produces only one "
                        "AFM sign on magnetic atoms; expand the cell or choose another vector."
                    )

        return _MagneticOrderState(
            spin_model=spin_model,
            afm_mode=afm_mode,
            pm_direction=pm_direction,
            axis=axis,
            magnitudes=magnitudes,
            directions=directions,
            afm_signs=afm_signs,
            output_count=output_count,
        )

    @staticmethod
    def _normalize_choice(value: str, aliases: dict[str, str], label: str) -> str:
        text = str(value or "").strip()
        normalized = aliases.get(text)
        if normalized is None:
            raise ValueError(f"MagneticOrder: unsupported {label} {text!r}.")
        return normalized

    @staticmethod
    def _group_signs(
        structure,
        params: MagneticOrderParams,
        magnetic_mask: np.ndarray,
    ) -> np.ndarray:
        if "group" not in structure.arrays:
            raise ValueError(
                "MagneticOrder: AFM group mode requires atoms.arrays['group']; "
                "add Group Label upstream or choose k-vector mode."
            )
        group_a = str(params.afm_group_a or "").strip()
        group_b = str(params.afm_group_b or "").strip()
        if not group_a or not group_b:
            raise ValueError("MagneticOrder: AFM group labels must be non-empty.")
        if group_a == group_b:
            raise ValueError("MagneticOrder: AFM positive and negative group labels must differ.")

        groups = np.asarray(structure.arrays["group"], dtype=str)
        mask_a = (groups == group_a) & magnetic_mask
        mask_b = (groups == group_b) & magnetic_mask
        if not np.any(mask_a) or not np.any(mask_b):
            raise ValueError(
                "MagneticOrder: AFM group mode needs at least one magnetic atom "
                "in both the positive and negative groups."
            )

        signs = np.zeros(len(structure), dtype=float)
        signs[groups == group_a] = 1.0
        signs[groups == group_b] = -1.0
        if not params.afm_zero_unknown:
            signs[(groups != group_a) & (groups != group_b)] = 1.0
        return signs

    @staticmethod
    def _moments_from_signs(
        state: _MagneticOrderState,
        signs: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(signs, dtype=float).reshape(-1)
        if values.shape != state.magnitudes.shape:
            raise ValueError("MagneticOrder: sign array length does not match atoms.")
        if state.spin_model == "noncollinear":
            return (
                values[:, None]
                * state.magnitudes[:, None]
                * state.directions
            )
        return values * state.magnitudes

    def _per_atom_mags_and_dirs(self, structure, params: MagneticOrderParams) -> tuple[np.ndarray, np.ndarray]:
        magmom_map = parse_magmom_map_any(params.magmom_map)
        axis = normalize_vector(np.array(params.axis, dtype=float))
        only = parse_element_set(params.apply_elements)
        mags = mapped_moment_magnitudes(
            structure,
            magmom_map,
            default_moment=float(params.default_moment),
            apply_elements=only,
        )
        dirs = np.repeat(axis[None, :], len(structure), axis=0)
        symbols = structure.get_chemical_symbols()
        for i, sym in enumerate(symbols):
            val = magmom_map.get(sym, float(params.default_moment))
            if isinstance(val, np.ndarray) and params.use_element_dirs and mags[i] > 0:
                dirs[i] = normalize_vector(val)
        if only:
            mask = element_mask(symbols, only)
            dirs = np.where(mask[:, None], dirs, axis[None, :])
        return mags, dirs

    def _make_collinear(self, structure, params: MagneticOrderParams, *, signs: np.ndarray) -> np.ndarray:
        mags, _dirs = self._per_atom_mags_and_dirs(structure, params)
        signs = np.asarray(signs, dtype=float).reshape(-1)
        if signs.shape[0] != mags.shape[0]:
            raise ValueError("signs shape mismatch")
        return signs * mags

    def _make_noncollinear_axis(self, structure, params: MagneticOrderParams, *, signs: np.ndarray) -> np.ndarray:
        mags, dirs = self._per_atom_mags_and_dirs(structure, params)
        signs = np.asarray(signs, dtype=float).reshape(-1)
        if signs.shape[0] != mags.shape[0]:
            raise ValueError("signs shape mismatch")
        return (signs[:, None] * mags[:, None]) * dirs


@dataclass(frozen=True)
class SpinSpiralParams:
    axis: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0)
    spiral_parameter_mode: str = "Period (L_D)"
    period_range: list[float] | tuple[float, float, float] = (20.0, 40.0, 10.0)
    angle_gradient_range: list[float] | tuple[float, float, float] = (18.0, 18.0, 1.0)
    phase_range: list[float] | tuple[float, float, float] = (0.0, 0.0, 15.0)
    mz: list[float] | tuple[float, float, float] = (0.0, 0.0, 0.1)
    chirality: str = "Both"
    phase_mode: str = "Continuous by position"
    layer_tolerance: float = 0.05
    only_commensurate_periods: bool = False
    magnitude_source: str = "Existing initial magmoms"
    magmom_map: str = ""
    default_moment: float = 0.0
    apply_elements: str = ""
    max_outputs: int = 100


class SpinSpiralOperation(StructureOperation):
    """Generate helical and conical spin-spiral magnetic moments."""

    def run_structure(self, structure, params: SpinSpiralParams) -> list:
        periods = self.period_values(params)
        original_periods = list(periods)
        period_bounds = self.period_bounds(params)
        phases = range_values(list(params.phase_range))
        mz_values = self.mz_values(params)
        max_outputs = int(params.max_outputs)
        axis = normalize_vector(np.array(params.axis, dtype=float))

        if params.only_commensurate_periods:
            discovered_periods = discover_commensurate_periods_in_bounds(period_bounds, structure=structure, axis=axis)
            periods = periods if discovered_periods is None else discovered_periods
            if not periods:
                suggest_supercell_multipliers(original_periods, structure=structure, axis=axis)
                return [structure.copy()]

        mags = self.magnitudes(structure, params)
        if mags.shape[0] != len(structure) or not np.any(mags > 0):
            return [structure.copy()]

        positions = np.asarray(structure.get_positions(), dtype=float)
        phase_positions = self.phase_positions(positions, axis, params)
        outputs = []
        reached_limit = False
        for period in periods:
            for phase_deg in phases:
                for mz in mz_values:
                    for chirality_tag, chirality_sign in self.chirality_values(params):
                        atoms = structure.copy()
                        unit_vectors = spiral_unit_vectors(
                            phase_positions,
                            axis=axis,
                            period=float(period),
                            mz=float(mz),
                            phase_deg=float(phase_deg),
                            chirality=chirality_sign,
                        )
                        set_initial_magmoms_safe(atoms, mags[:, None] * unit_vectors)
                        kind = "Helix" if abs(mz) <= 1e-10 else "Spiral"
                        extra_tag = ""
                        if params.phase_mode == "Layer-locked":
                            extra_tag = f",pm=layer,ltol={float(params.layer_tolerance):.4g}"
                        append_config_tag(
                            atoms,
                            f"{kind}(L={float(period):.6g},ph={float(phase_deg):.6g},mz={float(mz):.6g},"
                            f"chi={chirality_tag},ax={axis_tag(axis, precision=3)}{extra_tag})",
                        )
                        outputs.append(atoms)
                        if len(outputs) >= max_outputs:
                            reached_limit = True
                            break
                    if reached_limit:
                        break
                if reached_limit:
                    break
            if reached_limit:
                break
        return outputs or [structure.copy()]

    @staticmethod
    def period_values(params: SpinSpiralParams) -> list[float]:
        if params.spiral_parameter_mode == "Angle gradient (deg/A)":
            gradients = range_values(list(params.angle_gradient_range), minimum=0.001)
            return [float(360.0 / g) for g in gradients if g > 0]
        return range_values(list(params.period_range), minimum=0.001)

    @staticmethod
    def period_bounds(params: SpinSpiralParams) -> tuple[float, float]:
        if params.spiral_parameter_mode == "Angle gradient (deg/A)":
            start, stop, _step = [float(v) for v in params.angle_gradient_range]
            start = max(start, 0.001)
            stop = max(stop, 0.001)
            if stop < start:
                start, stop = stop, start
            period_min = float(360.0 / stop)
            period_max = float(360.0 / start)
            return (min(period_min, period_max), max(period_min, period_max))
        start, stop, _step = [float(v) for v in params.period_range]
        start = max(start, 0.001)
        stop = max(stop, 0.001)
        if stop < start:
            start, stop = stop, start
        return (float(start), float(stop))

    @staticmethod
    def mz_values(params: SpinSpiralParams) -> list[float]:
        raw = coerce_scan_triplet(params.mz, default_step=0.1)
        values: list[float] = []
        for value in range_values(raw, minimum=-1.0):
            clipped = float(np.clip(value, -1.0, 1.0))
            if not values or abs(clipped - values[-1]) > 1e-10:
                values.append(clipped)
        return values or [0.0]

    @staticmethod
    def layer_ids(positions: np.ndarray, axis: np.ndarray, tolerance: float) -> np.ndarray:
        projections = np.asarray(positions, dtype=float) @ normalize_vector(axis)
        if projections.size == 0:
            return np.zeros(0, dtype=int)
        order = np.argsort(projections, kind="stable")
        layer_ids = np.zeros(len(projections), dtype=int)
        tol = max(float(tolerance), 1e-8)
        current_layer = 0
        center = float(projections[order[0]])
        count = 1
        layer_ids[order[0]] = current_layer
        for idx in order[1:]:
            value = float(projections[idx])
            if abs(value - center) <= tol:
                count += 1
                center += (value - center) / float(count)
                layer_ids[idx] = current_layer
                continue
            current_layer += 1
            center = value
            count = 1
            layer_ids[idx] = current_layer
        return layer_ids

    @classmethod
    def layer_locked_positions(cls, positions: np.ndarray, axis: np.ndarray, tolerance: float) -> np.ndarray:
        pos = np.asarray(positions, dtype=float)
        if pos.size == 0:
            return np.zeros((0, 3), dtype=float)
        axis_hat = normalize_vector(axis)
        projections = pos @ axis_hat
        layer_ids = cls.layer_ids(pos, axis_hat, tolerance)
        shared_projections = np.array(projections, copy=True)
        for layer_id in np.unique(layer_ids):
            mask = layer_ids == layer_id
            shared_projections[mask] = float(np.mean(projections[mask]))
        return pos + (shared_projections - projections)[:, None] * axis_hat[None, :]

    @classmethod
    def phase_positions(cls, positions: np.ndarray, axis: np.ndarray, params: SpinSpiralParams) -> np.ndarray:
        if params.phase_mode != "Layer-locked":
            return np.asarray(positions, dtype=float)
        return cls.layer_locked_positions(positions, axis, float(params.layer_tolerance))

    @staticmethod
    def chirality_values(params: SpinSpiralParams) -> list[tuple[str, int]]:
        if params.chirality == "Clockwise":
            return [("cw", -1)]
        if params.chirality == "Counterclockwise":
            return [("ccw", 1)]
        return [("cw", -1), ("ccw", 1)]

    @staticmethod
    def magnitudes(structure, params: SpinSpiralParams) -> np.ndarray:
        if params.magnitude_source == "Existing initial magmoms":
            mags = existing_moment_magnitudes(structure)
            if mags is not None:
                return mags
        return mapped_moment_magnitudes(
            structure,
            parse_magmom_map_any(params.magmom_map),
            default_moment=float(params.default_moment),
            apply_elements=parse_element_set(params.apply_elements),
        )


def filter_commensurate_periods(periods: list[float], *, structure, axis: np.ndarray, tolerance: float = 1e-6) -> list[float]:
    cell = np.asarray(structure.get_cell(), dtype=float).reshape(3, 3)
    pbc = np.asarray(structure.get_pbc(), dtype=bool).reshape(3)
    periodic_vectors = [cell[idx] for idx in range(3) if pbc[idx]]
    if not periodic_vectors:
        return periods
    compatible: list[float] = []
    for period in periods:
        period_ok = True
        for vector in periodic_vectors:
            turns = float(np.dot(vector, axis) / float(period))
            if abs(turns) <= tolerance:
                continue
            if abs(turns - round(turns)) > tolerance:
                period_ok = False
                break
        if period_ok:
            compatible.append(float(period))
    return compatible


def discover_commensurate_periods_in_bounds(
    period_bounds: tuple[float, float],
    *,
    structure,
    axis: np.ndarray,
    tolerance: float = 1e-6,
) -> list[float] | None:
    period_min, period_max = [float(v) for v in period_bounds]
    if period_max < period_min:
        period_min, period_max = period_max, period_min
    cell = np.asarray(structure.get_cell(), dtype=float).reshape(3, 3)
    pbc = np.asarray(structure.get_pbc(), dtype=bool).reshape(3)
    projections = [abs(float(np.dot(cell[idx], axis))) for idx in range(3) if pbc[idx]]
    projections = [proj for proj in projections if proj > tolerance]
    if not projections:
        return None
    ref_projection = max(projections)
    n_min = max(1, int(np.ceil(ref_projection / period_max - tolerance)))
    n_max = max(1, int(np.floor(ref_projection / period_min + tolerance)))
    discovered: list[float] = []
    for turns_ref in range(n_min, n_max + 1):
        period = ref_projection / float(turns_ref)
        if period < period_min - tolerance or period > period_max + tolerance:
            continue
        period_ok = True
        for projection in projections:
            turns = projection / period
            if abs(turns - round(turns)) > tolerance:
                period_ok = False
                break
        if not period_ok:
            continue
        rounded = float(np.round(period, 12))
        if not discovered or abs(rounded - discovered[-1]) > 1e-10:
            discovered.append(rounded)
    return discovered


def suggest_supercell_multipliers(
    periods: list[float],
    *,
    structure,
    axis: np.ndarray,
    tolerance: float = 1e-6,
    max_multiplier: int = 24,
) -> tuple[float, list[int]] | None:
    cell = np.asarray(structure.get_cell(), dtype=float).reshape(3, 3)
    pbc = np.asarray(structure.get_pbc(), dtype=bool).reshape(3)
    periodic_indices = [idx for idx in range(3) if pbc[idx]]
    if not periodic_indices:
        return None
    best: tuple[int, int, float, list[int]] | None = None
    for period in periods:
        multipliers = [1, 1, 1]
        feasible = True
        for idx in periodic_indices:
            turns = float(np.dot(cell[idx], axis) / float(period))
            if abs(turns) <= tolerance:
                continue
            found = None
            for multiplier in range(1, max_multiplier + 1):
                if abs(multiplier * turns - round(multiplier * turns)) <= tolerance:
                    found = multiplier
                    break
            if found is None:
                feasible = False
                break
            multipliers[idx] = found
        if not feasible:
            continue
        volume_factor = int(np.prod([multipliers[idx] for idx in periodic_indices], dtype=int))
        max_factor = max(multipliers[idx] for idx in periodic_indices)
        candidate = (volume_factor, max_factor, float(period), multipliers)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return best[2], best[3]


@dataclass(frozen=True)
class SmallAngleSpinTiltParams:
    canting_mode: str = "Single-spin tilt"
    target_mode: str = "First eligible atom"
    target_indices: str = ""
    pair_left_indices: str = ""
    pair_right_indices: str = ""
    pair_source: str = "Manual indices"
    pair_shell: int = 1
    pair_shell_tolerance: float = 0.05
    pair_element_filter: str = ""
    pair_group_filter: str = ""
    bond_filter_mode: str = "Any"
    bond_filter_axis: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0)
    bond_filter_tolerance: float = 20.0
    group_a: str = "A"
    group_b: str = "B"
    angle_list: str = "1,2,5,10"
    tilt_signs: str = "Positive only"
    include_reference: bool = True
    magnitude_source: str = "Existing initial magmoms"
    magmom_map: str = ""
    default_moment: float = 0.0
    lift_scalar: bool = True
    axis: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0)
    reference_direction: list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0)
    apply_elements: str = ""
    max_outputs: int = 100


class SmallAngleSpinTiltOperation(StructureOperation):
    """Generate deterministic single-spin and pair-canting small-angle states."""

    def run_structure(self, structure, params: SmallAngleSpinTiltParams) -> list:
        supported_modes = {
            "Global tilt",
            "Single-spin tilt",
            "Atom pair canting",
            "Group pair canting",
        }
        if params.canting_mode not in supported_modes:
            raise ValueError(
                "SmallAngleSpinTilt: canting_mode must be Global tilt, "
                "Single-spin tilt, Atom pair canting, or Group pair canting."
            )
        max_outputs = int(params.max_outputs)
        if max_outputs <= 0:
            raise ValueError("SmallAngleSpinTilt: max_outputs must be >= 1.")
        if params.include_reference and max_outputs < 2:
            raise ValueError(
                "SmallAngleSpinTilt: max_outputs must be >= 2 when include_reference is enabled."
            )

        base_moments = self.vector_moments(structure, params)
        if base_moments is None or base_moments.shape != (len(structure), 3):
            raise ValueError(
                "SmallAngleSpinTilt requires usable initial magnetic moments; "
                "provide initial_magmoms or select Map/default magnitude."
            )
        if not np.any(np.linalg.norm(base_moments, axis=1) > 1e-10):
            raise ValueError(
                "SmallAngleSpinTilt found no nonzero magnetic moments to tilt."
            )

        angles = parse_angle_list(params.angle_list) or [1.0, 2.0, 5.0, 10.0]
        outputs = []
        reached_limit = False

        if params.include_reference:
            reference = structure.copy()
            set_initial_magmoms_safe(reference, base_moments)
            append_config_tag(reference, "SpinTiltRef")
            outputs.append(reference)

        if params.canting_mode == "Global tilt":
            target_indices = self.all_eligible_indices(structure, base_moments, params)
            if not target_indices:
                raise ValueError(
                    "SmallAngleSpinTilt global mode found no eligible magnetic atoms; "
                    "check apply_elements and magnetic moments."
                )
            for angle_deg in angles:
                for sign_tag, sign_value in self.signs(params):
                    tilted = structure.copy()
                    moment_array = np.array(base_moments, copy=True)
                    for idx in target_indices:
                        moment_array[idx] = self.tilted_vector(moment_array[idx], angle_deg, params, sign=sign_value)
                    set_initial_magmoms_safe(tilted, moment_array)
                    append_config_tag(tilted, f"SpinTiltG(a={float(angle_deg):.6g},sg={sign_tag})")
                    outputs.append(tilted)
                    if len(outputs) >= max_outputs:
                        reached_limit = True
                        break
                if reached_limit:
                    break
        elif params.canting_mode == "Single-spin tilt":
            target_indices = self.candidate_indices(structure, base_moments, params)
            if not target_indices:
                raise ValueError(
                    "SmallAngleSpinTilt single-spin mode matched no target atoms; "
                    "check target_mode, target_indices, apply_elements, and magnetic moments."
                )
            for atom_index in target_indices:
                for angle_deg in angles:
                    for sign_tag, sign_value in self.signs(params):
                        tilted = structure.copy()
                        moment_array = np.array(base_moments, copy=True)
                        moment_array[atom_index] = self.tilted_vector(moment_array[atom_index], angle_deg, params, sign=sign_value)
                        set_initial_magmoms_safe(tilted, moment_array)
                        append_config_tag(tilted, f"SpinTilt(i={atom_index + 1},a={float(angle_deg):.6g},sg={sign_tag})")
                        outputs.append(tilted)
                        if len(outputs) >= max_outputs:
                            reached_limit = True
                            break
                    if reached_limit:
                        break
                if reached_limit:
                    break
        elif params.canting_mode == "Atom pair canting":
            if (
                str(params.pair_group_filter).strip()
                and "group" not in structure.arrays
            ):
                raise ValueError(
                    "SmallAngleSpinTilt pair_group_filter requires atoms.arrays['group']."
                )
            pairs = self.pair_targets(structure, base_moments, params)
            if not pairs:
                raise ValueError(
                    "SmallAngleSpinTilt atom-pair mode matched no valid pairs; "
                    "check pair source, indices, shell, element/group filters, and magnetic moments."
                )
            for left_idx, right_idx in pairs:
                for angle_deg in angles:
                    for sign_tag, sign_value in self.signs(params):
                        tilted = structure.copy()
                        moment_array = self.apply_pair_canting(base_moments, [left_idx], [right_idx], angle_deg, params, sign=sign_value)
                        set_initial_magmoms_safe(tilted, moment_array)
                        append_config_tag(tilted, f"SpinPair(i={left_idx + 1},j={right_idx + 1},a={float(angle_deg):.6g},sg={sign_tag})")
                        outputs.append(tilted)
                        if len(outputs) >= max_outputs:
                            reached_limit = True
                            break
                    if reached_limit:
                        break
                if reached_limit:
                    break
        else:
            if "group" not in structure.arrays:
                raise ValueError(
                    "SmallAngleSpinTilt group-pair mode requires atoms.arrays['group']."
                )
            left_group, right_group = self.group_targets(structure, base_moments, params)
            if not left_group or not right_group:
                group_a = (params.group_a or "A").strip()
                group_b = (params.group_b or "B").strip()
                raise ValueError(
                    "SmallAngleSpinTilt group-pair mode requires nonzero magnetic atoms "
                    f"in both groups '{group_a}' and '{group_b}'."
                )
            group_a = (params.group_a or "A").strip()
            group_b = (params.group_b or "B").strip()
            for angle_deg in angles:
                for sign_tag, sign_value in self.signs(params):
                    tilted = structure.copy()
                    moment_array = self.apply_pair_canting(base_moments, left_group, right_group, angle_deg, params, sign=sign_value)
                    set_initial_magmoms_safe(tilted, moment_array)
                    append_config_tag(tilted, f"SpinPairG(A={group_a},B={group_b},a={float(angle_deg):.6g},sg={sign_tag})")
                    outputs.append(tilted)
                    if len(outputs) >= max_outputs:
                        reached_limit = True
                        break
                if reached_limit:
                    break

        if not outputs:
            raise ValueError(
                "SmallAngleSpinTilt produced no tilted structures from the selected targets."
            )
        return outputs

    @staticmethod
    def signs(params: SmallAngleSpinTiltParams) -> list[tuple[str, float]]:
        if params.tilt_signs == "Negative only":
            return [("neg", -1.0)]
        if params.tilt_signs == "Both (+/- pair)":
            return [("pos", 1.0), ("neg", -1.0)]
        return [("pos", 1.0)]

    @staticmethod
    def reference_direction(params: SmallAngleSpinTiltParams) -> np.ndarray:
        return normalize_vector(np.array(params.reference_direction, dtype=float), default=np.array([1.0, 0.0, 0.0], dtype=float))

    def vector_moments(self, structure, params: SmallAngleSpinTiltParams) -> np.ndarray | None:
        axis = normalize_vector(np.array(params.axis, dtype=float))
        if params.magnitude_source == "Existing initial magmoms":
            values = existing_moment_vectors(structure, axis=axis, lift_scalar=params.lift_scalar)
            if values is not None:
                return values
            return None
        return mapped_moment_vectors(
            structure,
            parse_magmom_map_any(params.magmom_map),
            default_moment=float(params.default_moment),
            axis=axis,
            apply_elements=parse_element_set(params.apply_elements),
            use_element_directions=False,
        )

    @staticmethod
    def candidate_indices(structure, base_moments: np.ndarray, params: SmallAngleSpinTiltParams) -> list[int]:
        eligible = SmallAngleSpinTiltOperation.all_eligible_indices(structure, base_moments, params)
        if not eligible:
            return []
        if params.target_mode == "First eligible atom":
            return [eligible[0]]
        if params.target_mode == "All eligible atoms":
            return eligible
        explicit = parse_atom_indices(params.target_indices, len(structure))
        return [idx for idx in explicit if idx in set(eligible)]

    @staticmethod
    def all_eligible_indices(structure, base_moments: np.ndarray, params: SmallAngleSpinTiltParams) -> list[int]:
        norms = np.linalg.norm(base_moments, axis=1)
        apply_elements = parse_element_set(params.apply_elements)
        return [
            idx
            for idx, (sym, mag) in enumerate(zip(structure.get_chemical_symbols(), norms))
            if mag > 1e-10 and (not apply_elements or sym in apply_elements)
        ]

    def pair_targets(self, structure, base_moments: np.ndarray, params: SmallAngleSpinTiltParams) -> list[tuple[int, int]]:
        if params.pair_source == "Auto by neighbor shell":
            return self.auto_pair_targets(structure, base_moments, params)
        left = parse_atom_indices(params.pair_left_indices, len(structure))
        right = parse_atom_indices(params.pair_right_indices, len(structure))
        if not left or not right:
            return []
        if len(left) != len(right):
            raise ValueError("Manual atom pair canting requires the same number of left and right indices.")
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        norms = np.linalg.norm(base_moments, axis=1)
        for left_idx, right_idx in zip(left, right):
            if left_idx == right_idx:
                continue
            if norms[left_idx] <= 1e-10 or norms[right_idx] <= 1e-10:
                continue
            pair = (left_idx, right_idx)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
        return pairs

    def auto_pair_targets(self, structure, base_moments: np.ndarray, params: SmallAngleSpinTiltParams) -> list[tuple[int, int]]:
        norms = np.linalg.norm(base_moments, axis=1)
        apply_elements = parse_element_set(params.apply_elements)
        eligible = [
            idx
            for idx, (sym, mag) in enumerate(zip(structure.get_chemical_symbols(), norms))
            if mag > 1e-10 and (not apply_elements or sym in apply_elements)
        ]
        if len(eligible) < 2:
            return []

        positions = np.asarray(structure.get_positions(), dtype=float)
        vec_matrix, dist_matrix = self.pair_distance_matrix(
            positions[eligible],
            cell=np.asarray(structure.cell.array, dtype=float),
            pbc=np.asarray(structure.pbc, dtype=bool),
        )
        tol = float(params.pair_shell_tolerance)
        shell_index = int(params.pair_shell) - 1
        rows, cols = np.triu_indices(len(eligible), k=1)
        distances = np.asarray(dist_matrix[rows, cols], dtype=float)
        valid = distances > 1e-12
        if not np.any(valid):
            return []

        shells: list[float] = []
        for dist in np.sort(distances[valid]):
            if not shells or abs(dist - shells[-1]) > tol:
                shells.append(float(dist))
        if shell_index < 0 or shell_index >= len(shells):
            return []
        target_distance = shells[shell_index]
        shell_mask = valid & (np.abs(distances - target_distance) <= tol)
        shell_rows = rows[shell_mask]
        shell_cols = cols[shell_mask]
        left = np.asarray(eligible, dtype=int)[shell_rows]
        right = np.asarray(eligible, dtype=int)[shell_cols]

        has_pair_filters = (
            bool(parse_pair_filter(params.pair_element_filter, normalize_case=True))
            or bool(parse_pair_filter(params.pair_group_filter, normalize_case=False))
            or params.bond_filter_mode != "Any"
        )
        if not has_pair_filters:
            return [(int(i), int(j)) for i, j in zip(left, right)]

        return [
            (int(i), int(j))
            for i, j, row, col in zip(left, right, shell_rows, shell_cols)
            if self.passes_pair_filters(structure, int(i), int(j), np.asarray(vec_matrix[row, col], dtype=float), params)
        ]

    @staticmethod
    def pair_distance_matrix(positions: np.ndarray, *, cell: np.ndarray, pbc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cell_arr = np.asarray(cell, dtype=float)
        pbc_arr = np.asarray(pbc, dtype=bool)
        positions_arr = np.asarray(positions, dtype=float)
        if positions_arr.size == 0:
            return np.empty((0, 0, 3), dtype=float), np.empty((0, 0), dtype=float)

        offdiag = cell_arr.copy()
        np.fill_diagonal(offdiag, 0.0)
        if cell_arr.shape == (3, 3) and np.all(np.isfinite(cell_arr)) and np.allclose(offdiag, 0.0, atol=1e-12):
            lengths = np.diag(cell_arr)
            if np.all(np.abs(lengths[pbc_arr]) > 1e-12):
                vec = positions_arr[None, :, :] - positions_arr[:, None, :]
                for axis in range(3):
                    if pbc_arr[axis]:
                        length = float(lengths[axis])
                        vec[..., axis] -= np.rint(vec[..., axis] / length) * length
                return vec, np.linalg.norm(vec, axis=2)

        return get_distances(positions_arr, positions_arr, cell=cell_arr, pbc=pbc_arr)

    @staticmethod
    def passes_pair_filters(structure, left_idx: int, right_idx: int, bond_vector: np.ndarray, params: SmallAngleSpinTiltParams) -> bool:
        element_pairs = parse_pair_filter(params.pair_element_filter, normalize_case=True)
        if element_pairs:
            syms = structure.get_chemical_symbols()
            if tuple(sorted((syms[left_idx], syms[right_idx]))) not in element_pairs:
                return False
        group_pairs = parse_pair_filter(params.pair_group_filter, normalize_case=False)
        if group_pairs:
            if "group" not in structure.arrays:
                return False
            groups = [str(g) for g in np.asarray(structure.arrays["group"])]
            if tuple(sorted((groups[left_idx], groups[right_idx]))) not in group_pairs:
                return False
        if params.bond_filter_mode == "Any":
            return True
        if params.bond_filter_mode not in {"Near axis", "Near plane"}:
            raise ValueError(f"Unsupported bond_filter_mode: {params.bond_filter_mode}")
        reference = normalize_vector(np.array(params.bond_filter_axis, dtype=float))
        bond_hat = normalize_vector(np.asarray(bond_vector, dtype=float), default=reference)
        cos_angle = float(np.clip(abs(np.dot(bond_hat, reference)), 0.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_angle)))
        tolerance = float(params.bond_filter_tolerance)
        if params.bond_filter_mode == "Near axis":
            return angle <= tolerance
        return abs(90.0 - angle) <= tolerance

    @staticmethod
    def group_targets(structure, base_moments: np.ndarray, params: SmallAngleSpinTiltParams) -> tuple[list[int], list[int]]:
        if "group" not in structure.arrays:
            return [], []
        group_values = [str(g) for g in np.asarray(structure.arrays["group"])]
        group_a = (params.group_a or "A").strip()
        group_b = (params.group_b or "B").strip()
        norms = np.linalg.norm(base_moments, axis=1)
        left = [idx for idx, (g, mag) in enumerate(zip(group_values, norms)) if g == group_a and mag > 1e-10]
        right = [idx for idx, (g, mag) in enumerate(zip(group_values, norms)) if g == group_b and mag > 1e-10]
        return left, right

    def tilt_direction(self, base_direction: np.ndarray, params: SmallAngleSpinTiltParams) -> np.ndarray:
        base_hat = normalize_vector(base_direction)
        preferred = self.reference_direction(params)
        preferred = preferred - float(np.dot(preferred, base_hat)) * base_hat
        if np.linalg.norm(preferred) <= 1e-10:
            e1, _, _ = orthonormal_frame(base_hat)
            return e1
        return normalize_vector(preferred)

    def tilted_vector(self, vector: np.ndarray, angle_deg: float, params: SmallAngleSpinTiltParams, *, sign: float = 1.0) -> np.ndarray:
        vec = np.asarray(vector, dtype=float).reshape(3)
        magnitude = float(np.linalg.norm(vec))
        if magnitude <= 0.0 or angle_deg <= 0.0:
            return vec.copy()
        base_hat = vec / magnitude
        tilt_hat = self.tilt_direction(base_hat, params)
        theta = math.radians(float(angle_deg) * float(sign))
        direction = (math.cos(theta) * base_hat) + (math.sin(theta) * tilt_hat)
        direction = normalize_vector(direction, default=base_hat)
        return magnitude * direction

    def apply_pair_canting(
        self,
        base_moments: np.ndarray,
        left_indices: list[int],
        right_indices: list[int],
        angle_deg: float,
        params: SmallAngleSpinTiltParams,
        *,
        sign: float,
    ) -> np.ndarray:
        moment_array = np.array(base_moments, copy=True)
        half_angle = float(angle_deg) * 0.5
        for idx in left_indices:
            moment_array[idx] = self.tilted_vector(moment_array[idx], half_angle, params, sign=sign)
        for idx in right_indices:
            moment_array[idx] = self.tilted_vector(moment_array[idx], half_angle, params, sign=-sign)
        return moment_array


@dataclass(frozen=True)
class SpinDisorderParams:
    mode: str = "Flip fraction"
    fractions: str = "0.1,0.3,0.5,0.7"
    samples_per_fraction: int = 1
    cone_angle: float = 30.0
    magnitude_source: str = "Existing initial magmoms"
    magmom_map: str = ""
    default_moment: float = 0.0
    lift_scalar: bool = True
    axis: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0)
    apply_elements: str = ""
    use_seed: bool = False
    seed: int = 0
    max_outputs: int = 100


class SpinDisorderOperation(StructureOperation):
    """Generate controlled spin-disorder states between ordered and PM limits."""

    def run_structure(self, structure, params: SpinDisorderParams) -> list:
        valid_modes = {
            "Flip fraction",
            "Cone disorder",
            "Randomize fraction",
        }
        if params.mode not in valid_modes:
            raise ValueError(
                "Spin Disorder mode must be Flip fraction, Cone disorder, "
                "or Randomize fraction."
            )
        samples_per_fraction = int(params.samples_per_fraction)
        if samples_per_fraction <= 0:
            raise ValueError("Spin Disorder samples_per_fraction must be >= 1.")
        max_outputs = int(params.max_outputs)
        if max_outputs <= 0:
            raise ValueError("Spin Disorder max_outputs must be >= 1.")
        if params.magnitude_source not in {
            "Existing initial magmoms",
            "Map/default magnitude",
        }:
            raise ValueError(
                "Spin Disorder magnitude_source must be Existing initial magmoms "
                "or Map/default magnitude."
            )
        seed = int(params.seed)
        if params.use_seed and seed < 0:
            raise ValueError("Spin Disorder seed must be >= 0.")
        fractions = self.fraction_values(params.fractions)
        if params.mode == "Cone disorder":
            cone_angle = float(params.cone_angle)
            if not np.isfinite(cone_angle) or not 0.0 <= cone_angle <= 180.0:
                raise ValueError(
                    "Spin Disorder cone_angle must be within [0, 180] degrees."
                )
        base_moments = self.vector_moments(structure, params)
        if base_moments is None or base_moments.shape != (len(structure), 3):
            raise ValueError("Spin Disorder requires vector magnetic moments or liftable scalar magmoms.")

        eligible = self.eligible_indices(structure, base_moments, params)
        if eligible.size == 0:
            raise ValueError("Spin Disorder found no eligible nonzero magnetic moments.")

        base_seed = seed if params.use_seed else None
        cfg_id = stable_config_id(structure)
        outputs = []
        for frac_idx, fraction in enumerate(fractions):
            n_change = self.count_for_fraction(len(eligible), fraction)
            if n_change <= 0:
                continue
            for sample_idx in range(samples_per_fraction):
                if base_seed is None:
                    rng = np.random.default_rng()
                    seed_tag = ""
                else:
                    derived_seed = int(base_seed + cfg_id * 1000003 + frac_idx * 1009 + sample_idx)
                    rng = np.random.default_rng(derived_seed)
                    seed_tag = f",s={derived_seed}"
                selected = rng.choice(eligible, size=n_change, replace=False)
                moments = np.array(base_moments, copy=True)
                if params.mode == "Flip fraction":
                    moments[selected] *= -1.0
                    label = "flip"
                elif params.mode == "Cone disorder":
                    label = "cone"
                    for idx in selected:
                        moments[idx] = self.random_cone_vector(base_moments[idx], float(params.cone_angle), rng)
                else:
                    label = "rand"
                    for idx in selected:
                        magnitude = float(np.linalg.norm(base_moments[idx]))
                        moments[idx] = magnitude * self.random_unit_vector(rng)

                atoms = structure.copy()
                set_initial_magmoms_safe(atoms, moments)
                detail = f"f={float(fraction):.6g},n={int(n_change)},mode={label}{seed_tag}"
                if label == "cone":
                    detail += f",a={float(params.cone_angle):.6g}"
                append_config_tag(atoms, f"SpinDis({detail})")
                outputs.append(atoms)
                if len(outputs) >= max_outputs:
                    return outputs
        if not outputs:
            raise ValueError("Spin Disorder did not generate any structures.")
        return outputs

    @staticmethod
    def fraction_values(text: str) -> list[float]:
        values = []
        seen = set()
        for token in re.split(r"[\s,;]+", text or ""):
            if not token.strip():
                continue
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(
                    f"Spin Disorder fraction '{token}' is not a number."
                ) from exc
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(
                    "Spin Disorder fractions must be finite values within (0, 1]."
                )
            rounded = float(np.round(value, 12))
            if rounded in seen:
                continue
            seen.add(rounded)
            values.append(rounded)
        if not values:
            raise ValueError(
                "Spin Disorder requires at least one fraction within (0, 1]."
            )
        return values

    @staticmethod
    def count_for_fraction(n_items: int, fraction: float) -> int:
        if n_items <= 0 or fraction <= 0.0:
            return 0
        return min(n_items, max(1, int(round(float(n_items) * float(fraction)))))

    @staticmethod
    def eligible_indices(structure, base_moments: np.ndarray, params: SpinDisorderParams) -> np.ndarray:
        apply_elements = parse_element_set(params.apply_elements)
        norms = np.linalg.norm(base_moments, axis=1)
        indices = [
            idx
            for idx, (symbol, norm) in enumerate(zip(structure.get_chemical_symbols(), norms))
            if norm > 1e-10 and (not apply_elements or symbol in apply_elements)
        ]
        return np.asarray(indices, dtype=int)

    @staticmethod
    def vector_moments(structure, params: SpinDisorderParams) -> np.ndarray | None:
        axis = normalize_vector(np.array(params.axis, dtype=float))
        if params.magnitude_source == "Existing initial magmoms":
            values = existing_moment_vectors(structure, axis=axis, lift_scalar=params.lift_scalar)
            if values is not None:
                return values
            if len(np.asarray(structure.get_initial_magnetic_moments(), dtype=float).shape) == 1 and not params.lift_scalar:
                return None
        return mapped_moment_vectors(
            structure,
            parse_magmom_map_any(params.magmom_map),
            default_moment=float(params.default_moment),
            axis=axis,
            apply_elements=parse_element_set(params.apply_elements),
            use_element_directions=False,
        )

    @staticmethod
    def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
        vector = rng.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return vector / norm

    def random_cone_vector(self, vector: np.ndarray, cone_angle: float, rng: np.random.Generator) -> np.ndarray:
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= 1e-12:
            return np.zeros(3, dtype=float)
        axis = normalize_vector(np.asarray(vector, dtype=float))
        e1, e2, axis = orthonormal_frame(axis)
        max_angle = max(0.0, min(float(cone_angle), 180.0))
        cos_min = math.cos(math.radians(max_angle))
        cos_theta = float(rng.uniform(cos_min, 1.0))
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        direction = cos_theta * axis + sin_theta * (math.cos(phi) * e1 + math.sin(phi) * e2)
        return magnitude * normalize_vector(direction, default=axis)


@dataclass(frozen=True)
class CorrelatedRandomSpinParams:
    mode: str = "Cone around reference"
    correlation_kernel: str = "exponential"
    correlation_length: float = 3.0
    samples: int = 1
    cone_angle: float = 30.0
    magnitude_source: str = "Existing initial magmoms"
    magmom_map: str = ""
    default_moment: float = 0.0
    lift_scalar: bool = True
    axis: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0)
    apply_elements: str = ""
    max_atoms_for_full: int = 200
    use_seed: bool = False
    seed: int = 0


class CorrelatedRandomSpinOperation(StructureOperation):
    """Generate non-collinear magnetic moments from an exact correlated random field."""

    def run_structure(self, structure, params: CorrelatedRandomSpinParams) -> list:
        if params.mode not in {
            "Cone around reference",
            "Full random directions",
        }:
            raise ValueError(
                f"Correlated Random Spin: unsupported mode '{params.mode}'."
            )
        samples = int(params.samples)
        if samples <= 0:
            raise ValueError("Correlated Random Spin: samples must be >= 1.")

        xi = float(params.correlation_length)
        if xi <= 0.0:
            raise ValueError("Correlated Random Spin: correlation_length must be positive.")

        max_atoms = int(params.max_atoms_for_full)
        if max_atoms <= 0:
            raise ValueError("Correlated Random Spin: max_atoms_for_full must be >= 1.")
        if params.magnitude_source not in {
            "Existing initial magmoms",
            "Map/default magnitude",
        }:
            raise ValueError(
                "Correlated Random Spin: magnitude_source must be Existing "
                "initial magmoms or Map/default magnitude."
            )
        seed = int(params.seed)
        if params.use_seed and seed < 0:
            raise ValueError("Correlated Random Spin: seed must be >= 0.")
        kernel = self.kernel_name(params.correlation_kernel)
        if params.mode == "Cone around reference":
            cone_angle = float(params.cone_angle)
            if not np.isfinite(cone_angle) or not 0.0 <= cone_angle <= 180.0:
                raise ValueError(
                    "Correlated Random Spin: cone_angle must be within "
                    "[0, 180] degrees."
                )

        base_moments = self.vector_moments(structure, params)
        if base_moments is None or base_moments.shape != (len(structure), 3):
            raise ValueError("Correlated Random Spin requires vector magnetic moments or liftable scalar magmoms.")

        selected = self.eligible_indices(structure, base_moments, params)
        if selected.size == 0:
            raise ValueError("Correlated Random Spin found no eligible nonzero magnetic moments.")
        if selected.size > max_atoms:
            raise ValueError(
                "Correlated Random Spin exact full covariance is limited to "
                f"{max_atoms} eligible atoms; got {selected.size}. Reduce the selection or use a smaller structure."
            )

        positions = np.asarray(structure.get_positions(), dtype=float)[selected]
        _vec_matrix, distances = SmallAngleSpinTiltOperation.pair_distance_matrix(
            positions,
            cell=np.asarray(structure.cell.array, dtype=float),
            pbc=np.asarray(structure.pbc, dtype=bool),
        )
        covariance = self.covariance_matrix(distances, xi=xi, kernel=kernel)
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Correlated Random Spin covariance is not positive definite for this structure/kernel.") from exc

        base_seed = seed if params.use_seed else None
        cfg_id = stable_config_id(structure)
        outputs = []
        for sample_idx in range(samples):
            if base_seed is None:
                rng = np.random.default_rng()
                seed_tag = ""
            else:
                derived_seed = int(base_seed + cfg_id * 1000003 + sample_idx)
                rng = np.random.default_rng(derived_seed)
                seed_tag = f",s={derived_seed}"
            field = chol @ rng.normal(size=(selected.size, 3))
            moments = np.array(base_moments, copy=True)
            selected_moments = base_moments[selected]
            magnitudes = np.linalg.norm(selected_moments, axis=1)
            if params.mode == "Full random directions":
                dirs = self.normalize_rows(field, fallback=self.normalize_rows(selected_moments))
                mode_tag = "full"
            elif params.mode == "Cone around reference":
                dirs = self.cone_directions(selected_moments, field, float(params.cone_angle), rng)
                mode_tag = "cone"
            moments[selected] = magnitudes[:, None] * dirs
            atoms = structure.copy()
            set_initial_magmoms_safe(atoms, moments)
            detail = f"xi={xi:.6g},ker={kernel},mode={mode_tag},n={selected.size}{seed_tag}"
            if mode_tag == "cone":
                detail += f",a={float(params.cone_angle):.6g}"
            append_config_tag(atoms, f"CorrSpin({detail})")
            outputs.append(atoms)
        return outputs

    @staticmethod
    def kernel_name(value: str) -> str:
        normalized = (value or "exponential").strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in {"exponential", "exp"}:
            return "exponential"
        if normalized in {"squared_exponential", "squared", "gaussian"}:
            return "squared_exponential"
        raise ValueError(f"Correlated Random Spin: unsupported correlation_kernel '{value}'.")

    @staticmethod
    def covariance_matrix(distances: np.ndarray, *, xi: float, kernel: str) -> np.ndarray:
        dist = np.asarray(distances, dtype=float)
        if kernel == "squared_exponential":
            cov = np.exp(-0.5 * (dist / float(xi)) ** 2)
        else:
            cov = np.exp(-dist / float(xi))
        cov = 0.5 * (cov + cov.T)
        cov[np.diag_indices_from(cov)] = 1.0 + 1e-10
        return cov

    @staticmethod
    def eligible_indices(structure, base_moments: np.ndarray, params: CorrelatedRandomSpinParams) -> np.ndarray:
        apply_elements = parse_element_set(params.apply_elements)
        norms = np.linalg.norm(base_moments, axis=1)
        indices = [
            idx
            for idx, (symbol, norm) in enumerate(zip(structure.get_chemical_symbols(), norms))
            if norm > 1e-10 and (not apply_elements or symbol in apply_elements)
        ]
        return np.asarray(indices, dtype=int)

    @staticmethod
    def vector_moments(structure, params: CorrelatedRandomSpinParams) -> np.ndarray | None:
        axis = normalize_vector(np.array(params.axis, dtype=float))
        if params.magnitude_source == "Existing initial magmoms":
            return existing_moment_vectors(structure, axis=axis, lift_scalar=params.lift_scalar)
        return mapped_moment_vectors(
            structure,
            parse_magmom_map_any(params.magmom_map),
            default_moment=float(params.default_moment),
            axis=axis,
            apply_elements=parse_element_set(params.apply_elements),
            use_element_directions=False,
        )

    @staticmethod
    def normalize_rows(vectors: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        vec = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(vec, axis=1)
        out = np.zeros_like(vec)
        mask = norms > 1e-12
        out[mask] = vec[mask] / norms[mask, None]
        if np.any(~mask):
            if fallback is None:
                out[~mask] = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                fb = np.asarray(fallback, dtype=float)
                fb_norm = np.linalg.norm(fb, axis=1)
                fb_out = np.zeros_like(fb)
                fb_mask = fb_norm > 1e-12
                fb_out[fb_mask] = fb[fb_mask] / fb_norm[fb_mask, None]
                fb_out[~fb_mask] = np.array([0.0, 0.0, 1.0], dtype=float)
                out[~mask] = fb_out[~mask]
        return out

    @classmethod
    def cone_directions(
        cls,
        base_moments: np.ndarray,
        field: np.ndarray,
        cone_angle: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        base_hat = cls.normalize_rows(base_moments)
        transverse = np.asarray(field, dtype=float) - np.sum(field * base_hat, axis=1)[:, None] * base_hat
        transverse = cls.normalize_rows(transverse, fallback=np.asarray([orthonormal_frame(axis)[0] for axis in base_hat], dtype=float))
        max_angle = max(0.0, min(float(cone_angle), 180.0))
        cos_min = math.cos(math.radians(max_angle))
        cos_theta = rng.uniform(cos_min, 1.0, size=len(base_hat))
        sin_theta = np.sqrt(np.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
        dirs = cos_theta[:, None] * base_hat + sin_theta[:, None] * transverse
        return cls.normalize_rows(dirs, fallback=base_hat)


@dataclass(frozen=True)
class FoldedHelixParams:
    layer_axis: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0)
    plane_normal: list[float] | tuple[float, float, float] = (0.0, 0.0, 1.0)
    layer_tolerance: float = 0.05
    half_period_mode: str = "Auto from layer count"
    half_period_layers: list[int] | tuple[int, int, int] = (2, 4, 1)
    angle_step_range: list[float] | tuple[float, float, float] = (15.0, 45.0, 15.0)
    phase_range: list[float] | tuple[float, float, float] = (0.0, 0.0, 15.0)
    sequence_mode: str = "Clockwise then counterclockwise"
    magnitude_source: str = "Existing initial magmoms"
    magmom_map: str = ""
    default_moment: float = 0.0
    apply_elements: str = ""
    max_outputs: int = 100


class FoldedHelixOperation(StructureOperation):
    """Generate symmetric clockwise-then-counterclockwise layered helix moments."""

    def run_structure(self, structure, params: FoldedHelixParams) -> list:
        angle_steps = range_values(list(params.angle_step_range), minimum=0.0)
        phases = range_values(list(params.phase_range))
        max_outputs = int(params.max_outputs)

        mags = self.magnitudes(structure, params)
        if mags.shape[0] != len(structure) or not np.any(mags > 0):
            return [structure.copy()]

        layer_axis = normalize_vector(np.array(params.layer_axis, dtype=float))
        plane_normal = normalize_vector(np.array(params.plane_normal, dtype=float))
        e1, e2, plane_hat = orthonormal_frame(plane_normal)
        layer_ids = SpinSpiralOperation.layer_ids(
            np.asarray(structure.get_positions(), dtype=float),
            layer_axis,
            float(params.layer_tolerance),
        )
        layer_count = int(layer_ids.max()) + 1 if layer_ids.size else 0
        half_periods = self.half_period_values(params, layer_count=layer_count)
        auto_mode = params.half_period_mode == "Auto from layer count"

        outputs = []
        reached_limit = False
        for half_period in half_periods:
            if auto_mode:
                folded_steps = self.auto_folded_steps(layer_ids, layer_count=layer_count)
            else:
                period_layers = max(2, 2 * int(half_period))
                local_layer = np.mod(layer_ids, period_layers)
                folded_steps = np.where(local_layer <= half_period, local_layer, period_layers - local_layer).astype(float)

            for angle_step in angle_steps:
                for phase_deg in phases:
                    for seq_tag, seq_sign in self.sequence_values(params):
                        atoms = structure.copy()
                        phase_rad = np.deg2rad(float(phase_deg) + seq_sign * folded_steps * float(angle_step))
                        unit_vectors = (
                            np.cos(phase_rad)[:, None] * e1[None, :]
                            + np.sin(phase_rad)[:, None] * e2[None, :]
                        )
                        set_initial_magmoms_safe(atoms, mags[:, None] * unit_vectors)
                        append_config_tag(
                            atoms,
                            (
                                f"FoldedHelix(h={half_period},da={float(angle_step):.6g},"
                                f"ph={float(phase_deg):.6g},seq={seq_tag},"
                                f"ax={axis_tag(layer_axis, precision=3)},pn={axis_tag(plane_hat, precision=3)})"
                            ),
                        )
                        outputs.append(atoms)
                        if len(outputs) >= max_outputs:
                            reached_limit = True
                            break
                    if reached_limit:
                        break
                if reached_limit:
                    break
            if reached_limit:
                break

        return outputs or [structure.copy()]

    @staticmethod
    def sequence_values(params: FoldedHelixParams) -> list[tuple[str, int]]:
        if params.sequence_mode == "Counterclockwise then clockwise":
            return [("ccw-cw", 1)]
        if params.sequence_mode == "Both":
            return [("cw-ccw", -1), ("ccw-cw", 1)]
        return [("cw-ccw", -1)]

    @staticmethod
    def half_period_values(params: FoldedHelixParams, *, layer_count: int) -> list[int]:
        if params.half_period_mode == "Manual":
            return int_range_values(list(params.half_period_layers), minimum=1)
        derived = max(1, (int(layer_count) - 1) // 2)
        return [derived]

    @staticmethod
    def auto_folded_steps(layer_ids: np.ndarray, *, layer_count: int) -> np.ndarray:
        if layer_ids.size == 0 or layer_count <= 1:
            return np.zeros_like(layer_ids, dtype=float)
        top_index = float(layer_count - 1)
        half_span = top_index / 2.0
        layer_pos = layer_ids.astype(float)
        return half_span - np.abs(layer_pos - half_span)

    @staticmethod
    def magnitudes(structure, params: FoldedHelixParams) -> np.ndarray:
        apply_elements = parse_element_set(params.apply_elements)
        if params.magnitude_source == "Existing initial magmoms":
            mags = existing_moment_magnitudes(structure)
            if mags is not None:
                mask = element_mask(structure.get_chemical_symbols(), apply_elements)
                return np.where(mask, mags, 0.0)
        return mapped_moment_magnitudes(
            structure,
            parse_magmom_map_any(params.magmom_map),
            default_moment=float(params.default_moment),
            apply_elements=apply_elements,
        )
