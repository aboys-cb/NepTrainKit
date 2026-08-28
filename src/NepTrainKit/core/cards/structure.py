"""UI-independent structure-generation Make Dataset operations."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import bulk, fcc111
from ase.data import atomic_numbers
from ase.geometry import get_distances
from loguru import logger

from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.core.config_type import stable_config_id
from NepTrainKit.core.magnetism import parse_kvec
from NepTrainKit.core.structure import get_vibration_modes
from NepTrainKit.core.torsion_guard_pbc import (
    TorsionGuardParams,
    build_adjacency_nonpbc,
    build_adjacency_pbc,
    get_rotatable_torsions_fast,
    process_single as tg_process_single,
)

from .geometry import scaled_positions, wrapped_positions as fast_wrapped_positions
from .errors import CardOperationError
from .operation import GeneratorOperation, StructureOperation
from .sampling import derived_structure_seed


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_ALLOWED_FUNCS: dict[str, Any] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "where": np.where,
    "clip": np.clip,
    "min": np.minimum,
    "max": np.maximum,
}


def validate_dz_expr(expr: str, allowed_names: set[str]) -> ast.Expression:
    """Validate a dz expression and return its AST."""
    expr = expr.strip()
    if not expr:
        raise ValueError("dz expression is empty")
    tree = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Compare,
        ast.BoolOp,
    )
    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor)
    allowed_unaryops = (ast.UAdd, ast.USub)
    allowed_boolops = (ast.And, ast.Or)
    allowed_cmpops = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    for node in ast.walk(tree):
        if isinstance(node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
            continue
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.BinOp) and not isinstance(node.op, allowed_binops):
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        if isinstance(node, ast.UnaryOp) and not isinstance(node.op, allowed_unaryops):
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        if isinstance(node, ast.BoolOp) and not isinstance(node.op, allowed_boolops):
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")
        if isinstance(node, ast.Compare) and not all(isinstance(op, allowed_cmpops) for op in node.ops):
            raise ValueError("Unsupported comparison operator")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed (e.g. sin(x))")
            func_name = node.func.id
            if func_name not in _ALLOWED_FUNCS:
                raise ValueError(f"Function '{func_name}' is not allowed")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unknown name '{node.id}'")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            raise ValueError("String constants are not allowed")
    return tree  # pyright: ignore[reportReturnType]


def parse_dz_params(text: str) -> dict[str, float]:
    """Parse scalar expression parameters from ``name=value`` chunks."""
    params: dict[str, float] = {}
    chunks = [chunk.strip() for chunk in re.split(r"[,\n;]+", text or "") if chunk.strip()]
    for chunk in chunks:
        if "=" not in chunk:
            raise ValueError(f"Invalid param '{chunk}', expected name=value")
        name, value_expr = chunk.split("=", 1)
        name = name.strip()
        value_expr = value_expr.strip()
        if not _NAME_RE.match(name):
            raise ValueError(f"Invalid parameter name '{name}'")
        allowed_names = set(_ALLOWED_FUNCS) | {"pi", "e"} | set(params)
        tree = validate_dz_expr(value_expr, allowed_names=allowed_names)
        code = compile(tree, "<param>", "eval")
        env: dict[str, Any] = dict(_ALLOWED_FUNCS)
        env.update(params)
        env["pi"] = math.pi
        env["e"] = math.e
        val = eval(code, {"__builtins__": {}}, env)  # noqa: S307
        val = float(np.asarray(val).reshape(-1)[0])
        if not np.isfinite(val):
            raise ValueError(f"Parameter '{name}' is not finite")
        params[name] = val
    return params


def evaluate_dz_expression(expr: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Evaluate a validated dz expression over selected coordinates."""
    allowed_names = set(_ALLOWED_FUNCS) | {"x", "y", "z", "pi", "e"} | set(params)
    tree = validate_dz_expr(expr, allowed_names=allowed_names)
    code = compile(tree, "<dz_expr>", "eval")
    env: dict[str, Any] = dict(_ALLOWED_FUNCS)
    env.update(params)
    env["x"] = x
    env["y"] = y
    env["z"] = z
    env["pi"] = math.pi
    env["e"] = math.e
    out = eval(code, {"__builtins__": {}}, env)  # noqa: S307
    out_arr = np.asarray(out, dtype=float)
    if out_arr.ndim == 0:
        out_arr = np.full_like(x, float(out_arr))
    if out_arr.shape != x.shape:
        raise ValueError(f"dz expression returned shape {out_arr.shape}, expected {x.shape}")
    if not np.all(np.isfinite(out_arr)):
        raise ValueError("dz expression produced NaN/Inf values")
    return out_arr


def build_layers(base_positions: np.ndarray, num_layers: int, layer_distance: float) -> list[np.ndarray]:
    """Stack copies of positions by a Cartesian-z translation."""
    offsets = np.arange(num_layers, dtype=float) * float(layer_distance)
    layers = []
    for offset in offsets:
        shifted = base_positions.copy()
        shifted[:, 2] = shifted[:, 2] + offset
        layers.append(shifted)
    return layers


@dataclass(frozen=True)
class LayerCopyParams:
    """Parameters for surface warp and layer-copy generation."""

    preset_index: int = 1
    dz_expr: str = "0"
    expression_params: str = ""
    apply_mode: int = 0
    elements: str = ""
    z_range: tuple[float, float] = (-1000000.0, 1000000.0)
    wrap: bool = False
    extend_cell_z: bool = True
    extra_vacuum: float = 0.0
    layers: int = 2
    distance_mode: str = "surface_gap"
    distance: float = 3.35
    max_output_atoms: int = 100_000


class LayerCopyOperation(StructureOperation):
    """Warp selected atoms by dz=f(x,y,z) and copy into stacked layers."""

    @staticmethod
    def _integer(value: object, name: str, *, minimum: int) -> int:
        templates = {
            "layers": (
                "layer_copy.invalid_layers",
                "Total layers must be an integer of at least {minimum}.",
            ),
            "apply_mode": (
                "layer_copy.invalid_apply_mode",
                "Warp selection must be All atoms, Selected elements, or Cartesian z range.",
            ),
            "max_output_atoms": (
                "layer_copy.invalid_atom_budget",
                "Atom budget per output must be an integer of at least {minimum}.",
            ),
        }
        code, template = templates[name]
        if isinstance(value, bool):
            raise CardOperationError(code, template, minimum=minimum)
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise CardOperationError(code, template, minimum=minimum) from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise CardOperationError(code, template, minimum=minimum)
        result = int(numeric)
        if result < minimum:
            raise CardOperationError(code, template, minimum=minimum)
        return result

    @staticmethod
    def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
        templates = {
            "layer spacing": (
                "layer_copy.invalid_spacing",
                "Layer spacing must be a finite non-negative distance.",
            ),
            "extra vacuum": (
                "layer_copy.invalid_vacuum",
                "Additional top vacuum must be a finite non-negative distance.",
            ),
            "z_range": (
                "layer_copy.invalid_z_range",
                "Cartesian z range must contain two finite distances.",
            ),
        }
        code, template = templates[name]
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise CardOperationError(code, template) from exc
        if not np.isfinite(result):
            raise CardOperationError(code, template)
        if minimum is not None and result < minimum:
            raise CardOperationError(code, template)
        return result

    @classmethod
    def _validated_settings(cls, structure, params: LayerCopyParams) -> dict[str, Any]:
        if len(structure) < 1:
            raise ValueError("LayerCopy requires at least one atom.")
        positions = np.asarray(structure.get_positions(), dtype=float)
        if positions.shape != (len(structure), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("LayerCopy requires finite Cartesian atom positions.")

        expr = str(params.dz_expr).strip()
        if not expr:
            raise ValueError("LayerCopy: dz expression is empty.")
        layers = cls._integer(params.layers, "layers", minimum=1)
        distance_mode = str(params.distance_mode or "").strip()
        if distance_mode not in {"surface_gap", "translation"}:
            raise CardOperationError(
                "layer_copy.invalid_distance_mode",
                "Layer spacing must use Surface gap or Copy translation.",
            )
        distance = cls._finite(params.distance, "layer spacing", minimum=0.0)
        if layers > 1 and distance_mode == "translation" and distance <= 0.0:
            raise CardOperationError(
                "layer_copy.nonpositive_translation",
                "Copy translation must be positive when total layers is greater than 1.",
            )
        max_output_atoms = cls._integer(
            params.max_output_atoms,
            "max_output_atoms",
            minimum=1,
        )
        output_atoms = len(structure) * layers
        if output_atoms > max_output_atoms:
            raise CardOperationError(
                "layer_copy.atom_budget",
                "Layer Stack would create {actual} atoms per output, above the {limit}-atom budget. "
                "Reduce the layer count or increase the budget.",
                actual=output_atoms,
                limit=max_output_atoms,
            )
        extra_vacuum = cls._finite(
            params.extra_vacuum,
            "extra vacuum",
            minimum=0.0,
        )

        mode = cls._integer(params.apply_mode, "apply_mode", minimum=0)
        if mode not in {0, 1, 2}:
            raise CardOperationError(
                "layer_copy.invalid_apply_mode",
                "Warp selection must be All atoms, Selected elements, or Cartesian z range.",
            )
        z_range = tuple(params.z_range)
        if len(z_range) != 2:
            raise CardOperationError(
                "layer_copy.invalid_z_range",
                "Cartesian z range must contain two finite distances.",
            )
        z_min = cls._finite(z_range[0], "z_range")
        z_max = cls._finite(z_range[1], "z_range")
        if z_min > z_max:
            z_min, z_max = z_max, z_min

        normalized_params = LayerCopyParams(
            preset_index=int(params.preset_index),
            dz_expr=expr,
            expression_params=str(params.expression_params),
            apply_mode=mode,
            elements=str(params.elements),
            z_range=(z_min, z_max),
            wrap=bool(params.wrap),
            extend_cell_z=bool(params.extend_cell_z),
            extra_vacuum=extra_vacuum,
            layers=layers,
            distance_mode=distance_mode,
            distance=distance,
            max_output_atoms=max_output_atoms,
        )
        mask = cls.apply_mask(structure, normalized_params)
        if not np.any(mask):
            raise ValueError("LayerCopy: no atoms selected by apply settings.")
        expr_params = parse_dz_params(normalized_params.expression_params)
        dz = evaluate_dz_expression(
            expr,
            x=positions[mask, 0],
            y=positions[mask, 1],
            z=positions[mask, 2],
            params=expr_params,
        )
        warped_z = positions[:, 2].copy()
        warped_z[mask] += dz
        slab_thickness = float(np.ptp(warped_z))
        if distance_mode == "surface_gap":
            surface_gap = distance
            translation = slab_thickness + surface_gap
        else:
            translation = distance
            surface_gap = translation - slab_thickness
            if layers > 1 and surface_gap < -1.0e-10:
                raise CardOperationError(
                    "layer_copy.overlap",
                    "Copy translation {translation} Å is smaller than the warped slab thickness "
                    "{thickness} Å, giving a negative surface gap {gap} Å.",
                    translation=f"{translation:.6g}",
                    thickness=f"{slab_thickness:.6g}",
                    gap=f"{surface_gap:.6g}",
                )

        cell = np.asarray(structure.cell.array, dtype=float)
        input_det = float(np.linalg.det(cell)) if cell.shape == (3, 3) else 0.0
        if (
            cell.shape != (3, 3)
            or not np.all(np.isfinite(cell))
            or abs(input_det) <= 1.0e-12
        ):
            raise CardOperationError(
                "layer_copy.invalid_cell",
                "Layer Stack needs a finite, non-singular 3D cell.",
            )
        final_cell = cell.copy()
        if params.extend_cell_z:
            if float(cell[2, 2]) <= 0.0:
                raise CardOperationError(
                    "layer_copy.cell_direction",
                    "Extending the cell requires lattice vector c to have a positive Cartesian z component.",
                )
            final_cell[2, 2] += translation * (layers - 1) + extra_vacuum
        final_det = float(np.linalg.det(final_cell))
        if (
            not np.all(np.isfinite(final_cell))
            or abs(final_det) <= 1.0e-12
            or np.sign(final_det) != np.sign(input_det)
        ):
            raise CardOperationError(
                "layer_copy.invalid_final_cell",
                "Layer Stack would create a singular or inverted final cell.",
            )
        return {
            "positions": positions,
            "mask": mask,
            "dz": dz,
            "params": normalized_params,
            "cell": cell,
            "final_cell": final_cell,
            "slab_thickness": slab_thickness,
            "surface_gap": surface_gap,
            "translation": translation,
            "output_atoms": output_atoms,
        }

    @classmethod
    def geometry_summary(cls, structure, params: LayerCopyParams) -> dict[str, Any]:
        """Return the resolved warp, stack, and output geometry."""
        settings = cls._validated_settings(structure, params)
        normalized = settings["params"]
        c_length = float(np.linalg.norm(settings["cell"][2]))
        final_c_length = float(np.linalg.norm(settings["final_cell"][2]))
        return {
            "input_atoms": len(structure),
            "selected_atoms": int(np.count_nonzero(settings["mask"])),
            "layers": normalized.layers,
            "translation": settings["translation"],
            "surface_gap": settings["surface_gap"],
            "slab_thickness": settings["slab_thickness"],
            "distance_mode": normalized.distance_mode,
            "output_atoms": settings["output_atoms"],
            "max_output_atoms": normalized.max_output_atoms,
            "dz_min": float(np.min(settings["dz"])),
            "dz_max": float(np.max(settings["dz"])),
            "cell_c_before": c_length,
            "cell_c_after": final_c_length,
            "extend_cell": normalized.extend_cell_z,
            "extra_vacuum": normalized.extra_vacuum,
            "wrap": normalized.wrap,
        }

    def run_structure(self, structure, params: LayerCopyParams) -> list:
        settings = self._validated_settings(structure, params)
        normalized = settings["params"]

        base = structure.copy()
        positions = settings["positions"]
        mask = settings["mask"]
        warped_positions = positions.copy()
        warped_positions[mask, 2] = warped_positions[mask, 2] + settings["dz"]

        layer_positions = build_layers(
            warped_positions,
            num_layers=normalized.layers,
            layer_distance=settings["translation"],
        )
        combined = base.copy()
        combined.set_positions(layer_positions[0])
        for layer_pos in layer_positions[1:]:
            layer_struct = base.copy()
            layer_struct.set_positions(layer_pos)
            combined += layer_struct

        if normalized.extend_cell_z:
            dz_total = (
                settings["translation"] * (normalized.layers - 1)
                + normalized.extra_vacuum
            )
            if dz_total > 0.0:
                base_cell = settings["cell"].copy()
                base_cell[2, 2] += dz_total
                combined.set_cell(base_cell, scale_atoms=False)

        if normalized.wrap:
            combined.set_positions(fast_wrapped_positions(combined, combined.positions))

        append_config_tag(
            combined,
            (
                f"LayerStack(L={normalized.layers},gap={settings['surface_gap']:g},"
                f"step={settings['translation']:g})"
            ),
        )
        return [combined]

    @staticmethod
    def apply_mask(structure, params: LayerCopyParams) -> np.ndarray:
        n_atoms = len(structure)
        mode = int(params.apply_mode)
        if mode == 0:
            return np.ones(n_atoms, dtype=bool)
        if mode == 1:
            elems = [token.strip() for token in re.split(r"[,\s]+", params.elements) if token.strip()]
            if not elems:
                return np.zeros(n_atoms, dtype=bool)
            symbols = np.asarray(structure.get_chemical_symbols(), dtype=object)
            return np.isin(symbols, np.asarray(elems, dtype=object))
        if mode != 2:
            raise ValueError(f"LayerCopy: unsupported apply_mode {mode}.")
        z_min, z_max = [float(value) for value in params.z_range]
        if z_min > z_max:
            z_min, z_max = z_max, z_min
        z = structure.get_positions()[:, 2]
        return (z >= z_min) & (z <= z_max)


@dataclass(frozen=True)
class VibrationModePerturbParams:
    """Parameters for vibrational-mode perturbations."""

    distribution: int = 0
    amplitude: float = 0.05
    modes_per_sample: int = 2
    min_frequency: float = 10.0
    max_num: int = 32
    scale_by_frequency: bool = True
    exclude_near_zero: bool = True
    use_seed: bool = False
    seed: int = 0


class VibrationModePerturbOperation(StructureOperation):
    """Generate perturbations along precomputed vibrational modes."""

    @staticmethod
    def wrapped_positions(structure, positions: np.ndarray) -> np.ndarray:
        return fast_wrapped_positions(structure, positions)

    def run_structure(self, structure, params: VibrationModePerturbParams) -> list:
        amplitude = float(params.amplitude)
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            raise CardOperationError(
                "vibration-mode-perturb-amplitude",
                "The mode coefficient scale must be a positive finite number.",
            )

        modes_per_sample = int(params.modes_per_sample)
        if modes_per_sample <= 0:
            raise CardOperationError(
                "vibration-mode-perturb-mode-count",
                "Modes combined per sample must be at least 1.",
            )
        max_num_value = float(params.max_num)
        if not np.isfinite(max_num_value) or not max_num_value.is_integer():
            raise CardOperationError(
                "vibration-mode-perturb-output-integer",
                "Structures per input must be an integer.",
            )
        max_num = int(max_num_value)
        if max_num <= 0:
            raise CardOperationError(
                "vibration-mode-perturb-output-count",
                "Structures per input must be at least 1.",
            )
        distribution = int(params.distribution)
        if distribution not in {0, 1}:
            raise CardOperationError(
                "vibration-mode-perturb-distribution",
                "Coefficient distribution must be Normal or Uniform.",
            )

        min_frequency = float(params.min_frequency) if params.exclude_near_zero else 0.0
        if not np.isfinite(min_frequency) or min_frequency < 0.0:
            raise CardOperationError(
                "vibration-mode-perturb-frequency-cutoff",
                "The absolute frequency cutoff must be a finite non-negative number.",
            )
        frequencies, modes = get_vibration_modes(structure, min_frequency=min_frequency)
        if modes.size == 0:
            raise CardOperationError(
                "vibration-mode-perturb-no-modes",
                "Vibrational perturbation needs at least one usable mode on every input structure.",
            )

        needs_frequencies = bool(params.scale_by_frequency or params.exclude_near_zero)
        if needs_frequencies and not np.all(np.isfinite(frequencies)):
            raise CardOperationError(
                "vibration-mode-perturb-missing-frequencies",
                "Finite frequencies are required when frequency filtering or scaling is enabled.",
            )
        if params.scale_by_frequency and np.any(np.abs(frequencies) <= 0.0):
            raise CardOperationError(
                "vibration-mode-perturb-zero-frequency",
                "Frequency weighting requires non-zero frequencies for every usable mode.",
            )
        if modes_per_sample > modes.shape[0]:
            raise CardOperationError(
                "vibration-mode-perturb-too-many-modes",
                "Modes per sample is {requested}, but only {available} usable modes are available.",
                requested=modes_per_sample,
                available=modes.shape[0],
            )

        base_seed = (
            derived_structure_seed(int(params.seed), structure)
            if params.use_seed
            else None
        )
        rng = np.random.default_rng(base_seed)
        freq_for_scaling = np.abs(frequencies)
        freq_for_scaling[~np.isfinite(freq_for_scaling)] = 0.0
        orig_positions = structure.get_positions()

        generated = []
        for sample_index in range(max_num):
            indices = rng.choice(modes.shape[0], size=modes_per_sample, replace=False)
            if distribution == 0:
                coeffs = rng.normal(loc=0.0, scale=1.0, size=modes_per_sample)
            else:
                coeffs = rng.uniform(-1.0, 1.0, size=modes_per_sample)

            if params.scale_by_frequency:
                denominators = np.sqrt(np.clip(freq_for_scaling[indices], a_min=1e-12, a_max=None))
                denominators[denominators == 0.0] = 1.0
                coeffs = coeffs / denominators

            displacement = np.sum(coeffs[:, None, None] * modes[indices], axis=0)
            new_structure = structure.copy()
            new_positions = orig_positions + amplitude * displacement
            new_structure.set_positions(self.wrapped_positions(structure, new_positions))
            append_config_tag(new_structure, f"Vib(a={amplitude:.3f},m={modes_per_sample})")
            new_structure.info["vibration_mode_perturb"] = json.dumps(
                {
                    "distribution": "normal" if distribution == 0 else "uniform",
                    "amplitude": amplitude,
                    "modes_per_sample": modes_per_sample,
                    "min_frequency": min_frequency,
                    "scale_by_frequency": bool(params.scale_by_frequency),
                    "exclude_near_zero": bool(params.exclude_near_zero),
                    "seed": int(params.seed) if params.use_seed else None,
                    "derived_seed": base_seed,
                    "sample_index": sample_index,
                    "candidate_mode_count": int(modes.shape[0]),
                    "selected_mode_indices": [int(index) for index in indices],
                    "selected_frequencies": [
                        float(frequencies[index]) if np.isfinite(frequencies[index]) else None
                        for index in indices
                    ],
                },
                separators=(",", ":"),
            )
            generated.append(new_structure)
        return generated


@dataclass(frozen=True)
class GroupLabelParams:
    """Parameters for assigning alternating atomic-layer group labels."""

    miller_index: str = "111"
    layer_tolerance: float = 0.05
    group_a: str = "A"
    group_b: str = "B"
    overwrite: bool = False


class GroupLabelOperation(StructureOperation):
    """Detect atomic planes normal to ``(hkl)`` and label them A/B in order."""

    def run_structure(self, structure, params: GroupLabelParams) -> list:
        a_label, b_label = self._validated_labels(params.group_a, params.group_b)
        if (not params.overwrite) and "group" in structure.arrays:
            return [structure.copy()]

        atoms = structure.copy()
        layer_ids = self.layer_ids(
            atoms,
            params.miller_index,
            params.layer_tolerance,
        )
        layer_count = int(layer_ids.max()) + 1 if layer_ids.size else 0
        if layer_count < 2:
            raise CardOperationError(
                "group_label_too_few_layers",
                "Layer Groups needs at least two detected atomic layers; the current settings detect {actual}. "
                "Expand the cell, choose another plane, or reduce the layer tolerance.",
                actual=layer_count,
            )

        atoms.arrays["group"] = np.where((layer_ids % 2) == 0, a_label, b_label).astype(object)
        append_config_tag(
            atoms,
            f"Grp(hkl{params.miller_index},tol={float(params.layer_tolerance):.4g},{a_label}/{b_label})",
        )
        return [atoms]

    @staticmethod
    def _validated_labels(group_a: str, group_b: str) -> tuple[str, str]:
        a_label = str(group_a or "").strip()
        b_label = str(group_b or "").strip()
        if not a_label or not b_label:
            raise CardOperationError(
                "group_label_empty_labels",
                "Layer group labels must be non-empty.",
            )
        if a_label == b_label:
            raise CardOperationError(
                "group_label_duplicate_labels",
                "Layer group A and B labels must be different.",
            )
        return a_label, b_label

    @staticmethod
    def layer_ids(atoms, miller_index: str, tolerance: float) -> np.ndarray:
        """Return ordered atomic-plane indices along the reciprocal ``(hkl)`` normal."""
        cell = np.asarray(atoms.cell.array, dtype=float)
        if cell.shape != (3, 3) or not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) <= 1e-12:
            raise CardOperationError(
                "group_label_invalid_cell",
                "Layer Groups needs a finite, non-singular 3D cell.",
            )
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise CardOperationError(
                "group_label_invalid_tolerance",
                "Layer tolerance must be a positive finite distance.",
            )
        try:
            hkl = np.asarray(parse_kvec(miller_index), dtype=float)
        except ValueError as exc:
            raise CardOperationError(
                "group_label_invalid_miller_index",
                "Plane index must be 100, 010, 001, 110, or 111.",
            ) from exc

        reciprocal_normal = np.linalg.solve(cell, hkl)
        reciprocal_norm = float(np.linalg.norm(reciprocal_normal))
        if not np.isfinite(reciprocal_norm) or reciprocal_norm <= 1e-12:
            raise CardOperationError(
                "group_label_invalid_miller_index",
                "Plane index must be 100, 010, 001, 110, or 111.",
            )

        scaled = scaled_positions(atoms, wrap=True)
        periodic_axes = np.asarray(atoms.pbc, dtype=bool)
        scaled[:, periodic_axes] = np.where(
            np.isclose(scaled[:, periodic_axes], 1.0, atol=1e-10, rtol=0.0),
            0.0,
            scaled[:, periodic_axes],
        )
        phase = scaled @ hkl
        periodic_phase = bool(np.any(periodic_axes & (hkl != 0.0)))
        if periodic_phase:
            phase %= 1.0
            phase = np.where(
                np.isclose(phase, 1.0, atol=1e-10, rtol=0.0),
                0.0,
                phase,
            )
        projections = phase / reciprocal_norm
        if projections.size == 0:
            return np.zeros(0, dtype=int)

        order = np.argsort(projections, kind="stable")
        sorted_projections = projections[order]
        layer_ids = np.zeros(len(projections), dtype=int)
        current_layer = 0
        start = 0
        bounds: list[tuple[int, int]] = []
        while start < len(order):
            upper = float(sorted_projections[start]) + tolerance
            stop = int(np.searchsorted(sorted_projections, upper, side="right"))
            layer_ids[order[start:stop]] = current_layer
            bounds.append((start, stop))
            current_layer += 1
            start = stop

        if periodic_phase and len(bounds) > 1:
            first_start, first_stop = bounds[0]
            last_start, last_stop = bounds[-1]
            period = 1.0 / reciprocal_norm
            boundary_span = (
                float(sorted_projections[first_stop - 1])
                + period
                - float(sorted_projections[last_start])
            )
            if boundary_span <= tolerance:
                layer_ids[order[last_start:last_stop]] = 0
        return layer_ids

    @staticmethod
    def has_periodic_layer_axis(atoms, miller_index: str) -> bool:
        hkl = np.asarray(parse_kvec(miller_index), dtype=int)
        return bool(np.any(np.asarray(atoms.pbc, dtype=bool) & (hkl != 0)))


@dataclass(frozen=True)
class OrganicMolConfigPBCParams:
    """Parameters for torsion-guard organic conformer generation."""

    perturb_per_frame: int = 100
    torsion_range_deg: tuple[float, float] = (-180.0, 180.0)
    max_torsions_per_conf: int = 5
    gaussian_sigma: float = 0.03
    pbc_mode: str = "auto"
    local_cutoff: int = 150
    local_subtree: int = 40
    bond_detect_factor: float = 1.15
    bond_keep_min_factor: float = 0.60
    bond_keep_max_factor: float = 1.15
    bond_keep_max_enable: bool = False
    nonbond_min_factor: float = 0.80
    max_retries: int = 12
    mult_bond_factor: float = 0.87
    nonpbc_box_size: float = 100.0
    bo_c_const: float = 0.3
    bo_threshold: float = 0.2
    use_seed: bool = False
    seed: int = 0


@dataclass(frozen=True)
class OrganicMolConfigTopologySummary:
    """Resolved topology facts used by the card preview."""

    atom_count: int
    bond_count: int
    component_count: int
    torsion_count: int
    torsion_active: bool
    pbc_active: bool
    local_mode: bool
    requested_outputs: int
    gaussian_sigma: float


class OrganicMolConfigPBCOperation(StructureOperation):
    """Generate torsion-driven molecular conformers using TorsionGuard PBC."""

    @staticmethod
    def _integer(value: Any, key: str, field: str, *, minimum: int) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise CardOperationError(
                f"organic-{key}-integer",
                "{field} must be an integer.",
                field=field,
            ) from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise CardOperationError(
                f"organic-{key}-integer",
                "{field} must be an integer.",
                field=field,
            )
        integer = int(numeric)
        if integer < minimum:
            raise CardOperationError(
                f"organic-{key}-minimum",
                "{field} must be at least {minimum}.",
                field=field,
                minimum=minimum,
            )
        return integer

    @staticmethod
    def _finite_float(
        value: Any,
        key: str,
        field: str,
        *,
        minimum: float | None = None,
        strictly_positive: bool = False,
    ) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise CardOperationError(
                f"organic-{key}-finite",
                "{field} must be a finite number.",
                field=field,
            ) from exc
        if not np.isfinite(numeric):
            raise CardOperationError(
                f"organic-{key}-finite",
                "{field} must be a finite number.",
                field=field,
            )
        if strictly_positive and numeric <= 0.0:
            raise CardOperationError(
                f"organic-{key}-positive",
                "{field} must be positive.",
                field=field,
            )
        if minimum is not None and numeric < minimum:
            raise CardOperationError(
                f"organic-{key}-minimum",
                "{field} must be at least {minimum}.",
                field=field,
                minimum=f"{minimum:g}",
            )
        return numeric

    @classmethod
    def _validated_settings(
        cls,
        structure,
        params: OrganicMolConfigPBCParams,
    ) -> dict[str, Any]:
        symbols = structure.get_chemical_symbols()
        if not symbols:
            raise CardOperationError(
                "organic-empty-input",
                "Molecular Conformers requires at least one atom.",
            )
        coords = np.asarray(structure.get_positions(), dtype=float)
        if coords.shape != (len(symbols), 3) or not np.all(np.isfinite(coords)):
            raise CardOperationError(
                "organic-invalid-positions",
                "Molecular Conformers requires finite Cartesian atom positions.",
            )

        pbc_mode = str(params.pbc_mode).strip().lower()
        if pbc_mode not in {"auto", "yes", "no"}:
            raise CardOperationError(
                "organic-invalid-boundary",
                "Output boundary must be Follow input, 3D periodic, or Nonperiodic.",
            )
        pbc_flags = np.asarray(structure.pbc, dtype=bool)
        if pbc_mode == "auto" and np.any(pbc_flags) and not np.all(pbc_flags):
            raise CardOperationError(
                "organic-mixed-pbc",
                "Follow input does not support mixed periodic boundaries; choose Nonperiodic or provide full 3D PBC.",
            )
        pbc_active = pbc_mode == "yes" or (
            pbc_mode == "auto" and bool(np.all(pbc_flags))
        )
        cell_mat = None
        if pbc_active:
            cell_mat = np.asarray(structure.cell.array, dtype=float)
            if (
                cell_mat.shape != (3, 3)
                or not np.all(np.isfinite(cell_mat))
                or abs(float(np.linalg.det(cell_mat))) <= 1e-12
            ):
                raise CardOperationError(
                    "organic-invalid-periodic-cell",
                    "3D periodic mode requires a finite, nonsingular 3×3 cell.",
                )

        perturb_per_frame = cls._integer(
            params.perturb_per_frame,
            "outputs",
            "Maximum outputs per input",
            minimum=1,
        )
        max_torsions = cls._integer(
            params.max_torsions_per_conf,
            "torsions-per-output",
            "Bonds rotated per output",
            minimum=0,
        )
        local_cutoff = cls._integer(
            params.local_cutoff,
            "large-molecule-threshold",
            "Large-molecule threshold",
            minimum=0,
        )
        local_subtree = cls._integer(
            params.local_subtree,
            "local-subtree-cap",
            "Local subtree cap",
            minimum=1,
        )
        max_retries = cls._integer(
            params.max_retries,
            "retries",
            "Retries per output",
            minimum=0,
        )
        gaussian_sigma = cls._finite_float(
            params.gaussian_sigma,
            "coordinate-noise",
            "Coordinate noise",
            minimum=0.0,
        )
        if len(params.torsion_range_deg) != 2:
            raise CardOperationError(
                "organic-torsion-range-size",
                "Torsion increment range must contain a minimum and maximum.",
            )
        torsion_range = tuple(
            cls._finite_float(
                value,
                "torsion-range",
                "Torsion increment range",
            )
            for value in params.torsion_range_deg
        )
        if torsion_range[0] > torsion_range[1]:
            raise CardOperationError(
                "organic-torsion-range-order",
                "Torsion increment minimum must not exceed its maximum.",
            )
        bond_detect = cls._finite_float(
            params.bond_detect_factor,
            "bond-detection-radius",
            "Bond detection radius",
            strictly_positive=True,
        )
        bond_min = cls._finite_float(
            params.bond_keep_min_factor,
            "minimum-bond-length",
            "Minimum bond length",
            minimum=0.0,
        )
        bond_max = None
        if params.bond_keep_max_enable:
            bond_max = cls._finite_float(
                params.bond_keep_max_factor,
                "maximum-bond-length",
                "Maximum bond length",
                strictly_positive=True,
            )
            if bond_max < bond_min:
                raise CardOperationError(
                    "organic-bond-length-order",
                    "Maximum bond length must not be smaller than minimum bond length.",
                )
        nonbond_min = cls._finite_float(
            params.nonbond_min_factor,
            "minimum-nonbonded-distance",
            "Minimum nonbonded distance",
            minimum=0.0,
        )
        mult_bond = cls._finite_float(
            params.mult_bond_factor,
            "short-bond-cutoff",
            "Short-bond rotation cutoff",
            minimum=0.0,
        )
        box_size = cls._finite_float(
            params.nonpbc_box_size,
            "nonperiodic-box",
            "Nonperiodic display box",
            strictly_positive=True,
        )
        bo_c = cls._finite_float(
            params.bo_c_const,
            "pauling-decay-length",
            "Pauling decay length",
            strictly_positive=True,
        )
        bo_threshold = cls._finite_float(
            params.bo_threshold,
            "bond-order-threshold",
            "Bond-order threshold",
            minimum=0.0,
        )
        if bo_threshold > 1.0:
            raise CardOperationError(
                "organic-bond-order-range",
                "Bond-order threshold must be between 0 and 1.",
            )
        seed = cls._integer(params.seed, "seed", "Random seed", minimum=0)

        if pbc_active:
            assert cell_mat is not None
            adj, edge_len, radii, edge_order = build_adjacency_pbc(
                symbols,
                coords,
                cell_mat,
                bond_detect,
                c_const=bo_c,
                bo_threshold=bo_threshold,
            )
        else:
            adj, edge_len, radii, edge_order = build_adjacency_nonpbc(
                symbols,
                coords,
                bond_detect,
                c_const=bo_c,
                bo_threshold=bo_threshold,
            )
        torsions = get_rotatable_torsions_fast(
            adj,
            edge_len,
            radii,
            symbols,
            mult_bond,
            edge_order=edge_order,
        )
        torsion_active = bool(
            torsions
            and max_torsions > 0
            and (torsion_range[0] != 0.0 or torsion_range[1] != 0.0)
        )
        if not torsion_active and gaussian_sigma == 0.0:
            raise CardOperationError(
                "organic-no-coordinate-change",
                "The current settings cannot change coordinates; add coordinate noise or provide an active rotatable bond.",
            )

        components = 0
        visited: set[int] = set()
        for root in range(len(adj)):
            if root in visited:
                continue
            components += 1
            stack = [root]
            visited.add(root)
            while stack:
                atom = stack.pop()
                for neighbor in adj[atom]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        return {
            "symbols": symbols,
            "coords": coords,
            "cell_mat": cell_mat,
            "pbc_mode": pbc_mode,
            "pbc_active": pbc_active,
            "perturb_per_frame": perturb_per_frame,
            "torsion_range": torsion_range,
            "max_torsions": max_torsions,
            "gaussian_sigma": gaussian_sigma,
            "local_cutoff": local_cutoff,
            "local_subtree": local_subtree,
            "bond_detect": bond_detect,
            "bond_min": bond_min,
            "bond_max": bond_max,
            "nonbond_min": nonbond_min,
            "max_retries": max_retries,
            "mult_bond": mult_bond,
            "box_size": box_size,
            "bo_c": bo_c,
            "bo_threshold": bo_threshold,
            "seed": seed,
            "bond_count": len(edge_len),
            "torsion_count": len(torsions),
            "component_count": components,
            "torsion_active": torsion_active,
        }

    @classmethod
    def topology_summary(
        cls,
        structure,
        params: OrganicMolConfigPBCParams,
    ) -> OrganicMolConfigTopologySummary:
        """Return resolved topology and sampling information for the UI."""
        settings = cls._validated_settings(structure, params)
        return OrganicMolConfigTopologySummary(
            atom_count=len(settings["symbols"]),
            bond_count=settings["bond_count"],
            component_count=settings["component_count"],
            torsion_count=settings["torsion_count"],
            torsion_active=settings["torsion_active"],
            pbc_active=settings["pbc_active"],
            local_mode=len(settings["symbols"]) > settings["local_cutoff"],
            requested_outputs=settings["perturb_per_frame"],
            gaussian_sigma=settings["gaussian_sigma"],
        )

    def run_structure(self, structure, params: OrganicMolConfigPBCParams) -> list:
        settings = self._validated_settings(structure, params)
        tg_params = TorsionGuardParams(
            perturb_per_frame=settings["perturb_per_frame"],
            torsion_range_deg=settings["torsion_range"],
            max_torsions_per_conf=settings["max_torsions"],
            gaussian_sigma=settings["gaussian_sigma"],
            pbc_mode="yes" if settings["pbc_active"] else "no",
            local_mode_cutoff_atoms=settings["local_cutoff"],
            local_torsion_max_subtree=settings["local_subtree"],
            bond_detect_factor=settings["bond_detect"],
            bond_keep_min_factor=settings["bond_min"],
            bond_keep_max_factor=settings["bond_max"],
            nonbond_min_factor=settings["nonbond_min"],
            max_retries_per_frame=settings["max_retries"],
            mult_bond_factor=settings["mult_bond"],
            nonpbc_box_size=settings["box_size"],
            bo_c_const=settings["bo_c"],
            bo_threshold=settings["bo_threshold"],
            seed=(
                settings["seed"] + stable_config_id(structure) * 1000003
                if params.use_seed
                else None
            ),
        )
        result_list = tg_process_single(
            settings["symbols"],
            settings["coords"],
            settings["cell_mat"],
            tg_params,
        )
        if not result_list:
            raise CardOperationError(
                "organic-all-guards-failed",
                "All requested conformers failed the geometry guards; narrow the torsion range, reduce coordinate noise, or inspect the distance limits.",
            )

        structures_out = []
        for _symbols, new_coords, cell, pbc_active in result_list:
            new_atoms = structure.copy()
            new_atoms.set_positions(np.array(new_coords, dtype=float))
            if pbc_active and cell is not None:
                new_atoms.set_cell(np.array(cell, dtype=float))
                new_atoms.set_pbc(True)
                try:
                    new_atoms.set_positions(fast_wrapped_positions(new_atoms, new_atoms.positions))
                except Exception:
                    pass
            else:
                box = settings["box_size"]
                new_atoms.set_cell(np.diag([box, box, box]))
                new_atoms.set_pbc(False)
            append_config_tag(
                new_atoms,
                f"TG(req={settings['perturb_per_frame']},ok={len(result_list)},"
                f"sig={settings['gaussian_sigma']:g},pbc={settings['pbc_mode']})",
            )
            structures_out.append(new_atoms)
        return structures_out


@dataclass(frozen=True)
class RandomPackingParams:
    """Parameters for random atomic packing inside an existing cell."""

    structures: int = 1
    composition: str = ""
    min_distance: float = 1.5
    pair_min_distances: str = ""
    max_attempts_per_atom: int = 500
    strict_mode: bool = True
    use_seed: bool = False
    seed: int = 0


class RandomPackingOperation(StructureOperation):
    """Randomly repack atoms in the input cell under explicit distance constraints."""

    def run_structure(self, structure, params: RandomPackingParams) -> list:
        n_structures = int(params.structures)
        if n_structures <= 0:
            raise ValueError("Random Packing: structures must be >= 1.")

        min_distance = float(params.min_distance)
        if min_distance <= 0.0:
            raise ValueError("Random Packing: min_distance must be positive.")

        max_attempts = int(params.max_attempts_per_atom)
        if max_attempts <= 0:
            raise ValueError("Random Packing: max_attempts_per_atom must be >= 1.")

        cell = np.asarray(structure.cell.array, dtype=float)
        if cell.shape != (3, 3) or abs(float(np.linalg.det(cell))) <= 1e-12:
            raise ValueError("Random Packing requires a non-singular input cell.")

        pbc = np.asarray(structure.pbc, dtype=bool)
        symbols = self.symbols_from_params(structure, params.composition)
        pair_rules = self.parse_pair_min_distances(params.pair_min_distances)
        order = self.placement_order(symbols, min_distance, pair_rules)
        ortho_lengths = self.orthorhombic_lengths(cell, pbc)

        base_seed = int(params.seed) if params.use_seed else None
        cfg_id = stable_config_id(structure)
        outputs = []
        failures = 0
        for sample_idx in range(n_structures):
            rng = np.random.default_rng(None if base_seed is None else int(base_seed + cfg_id * 1000003 + sample_idx))
            try:
                atoms = self.pack_one(
                    structure,
                    symbols=symbols,
                    order=order,
                    min_distance=min_distance,
                    pair_rules=pair_rules,
                    max_attempts=max_attempts,
                    rng=rng,
                    cell=cell,
                    pbc=pbc,
                    ortho_lengths=ortho_lengths,
                )
            except ValueError:
                failures += 1
                if params.strict_mode:
                    raise
                logger.warning("RandomPackingOperation: skipped failed sample {} for {}", sample_idx + 1, structure.info.get("Config_type", "structure"))
                continue
            seed_tag = f",s={int(base_seed + cfg_id * 1000003 + sample_idx)}" if base_seed is not None else ""
            append_config_tag(atoms, f"RandPack(n={len(symbols)},d={min_distance:.6g}{seed_tag})")
            outputs.append(atoms)

        if not outputs:
            raise ValueError(f"Random Packing failed to generate any structures ({failures} failures).")
        return outputs

    @staticmethod
    def symbols_from_params(structure, composition: str) -> list[str]:
        text = (composition or "").strip()
        if not text:
            return list(structure.get_chemical_symbols())

        chunks = [chunk.strip() for chunk in re.split(r"[,;\n]+", text) if chunk.strip()]
        if not chunks:
            raise ValueError("Random Packing: composition is empty.")

        symbols: list[str] = []
        for chunk in chunks:
            if ":" in chunk:
                raw_symbol, raw_count = chunk.split(":", 1)
            elif "=" in chunk:
                raw_symbol, raw_count = chunk.split("=", 1)
            else:
                raise ValueError(f"Random Packing: invalid composition item '{chunk}', expected Element:count.")
            symbol = raw_symbol.strip()
            if not symbol:
                raise ValueError(f"Random Packing: invalid composition item '{chunk}'.")
            symbol = symbol[0].upper() + symbol[1:].lower()
            count_value = float(raw_count)
            count = int(round(count_value))
            if count <= 0 or abs(count_value - count) > 1e-9:
                raise ValueError(f"Random Packing: composition count for {symbol} must be a positive integer.")
            symbols.extend([symbol] * count)

        if not symbols:
            raise ValueError("Random Packing: composition produced no atoms.")
        return symbols

    @staticmethod
    def parse_pair_min_distances(text: str) -> dict[tuple[str, str], float]:
        rules: dict[tuple[str, str], float] = {}
        for chunk in re.split(r"[,;\n]+", text or ""):
            item = chunk.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Random Packing: invalid pair distance '{item}', expected A-B:value.")
            pair_text, value_text = item.split(":", 1)
            if "-" not in pair_text:
                raise ValueError(f"Random Packing: invalid pair '{pair_text}', expected A-B.")
            left, right = [part.strip() for part in pair_text.split("-", 1)]
            if not left or not right:
                raise ValueError(f"Random Packing: invalid pair '{pair_text}'.")
            left = left[0].upper() + left[1:].lower()
            right = right[0].upper() + right[1:].lower()
            value = float(value_text)
            if value <= 0.0:
                raise ValueError(f"Random Packing: pair distance for {left}-{right} must be positive.")
            rules[tuple(sorted((left, right)))] = value
        return rules

    @staticmethod
    def min_distance_for_pair(left: str, right: str, default: float, pair_rules: dict[tuple[str, str], float]) -> float:
        return float(pair_rules.get(tuple(sorted((left, right))), default))

    @classmethod
    def placement_order(cls, symbols: list[str], min_distance: float, pair_rules: dict[tuple[str, str], float]) -> np.ndarray:
        per_symbol = {
            symbol: max(cls.min_distance_for_pair(symbol, other, min_distance, pair_rules) for other in set(symbols))
            for symbol in set(symbols)
        }
        return np.asarray(sorted(range(len(symbols)), key=lambda idx: (-per_symbol[symbols[idx]], symbols[idx], idx)), dtype=int)

    @classmethod
    def pack_one(
        cls,
        structure,
        *,
        symbols: list[str],
        order: np.ndarray,
        min_distance: float,
        pair_rules: dict[tuple[str, str], float],
        max_attempts: int,
        rng: np.random.Generator,
        cell: np.ndarray,
        pbc: np.ndarray,
        ortho_lengths: np.ndarray | None,
    ) -> Atoms:
        placed_positions: list[np.ndarray] = []
        placed_symbols: list[str] = []
        positions_by_original = np.zeros((len(symbols), 3), dtype=float)

        for original_idx in rng.permutation(order):
            symbol = symbols[int(original_idx)]
            placed = False
            for _attempt in range(max_attempts):
                frac = rng.random(3)
                candidate = frac @ cell
                if cls.candidate_is_valid(
                    candidate,
                    symbol,
                    placed_positions=placed_positions,
                    placed_symbols=placed_symbols,
                    min_distance=min_distance,
                    pair_rules=pair_rules,
                    cell=cell,
                    pbc=pbc,
                    ortho_lengths=ortho_lengths,
                ):
                    positions_by_original[int(original_idx)] = candidate
                    placed_positions.append(candidate)
                    placed_symbols.append(symbol)
                    placed = True
                    break
            if not placed:
                raise ValueError(
                    "Random Packing could not place "
                    f"{symbol} after {int(max_attempts)} attempts. Reduce min distances, enlarge the cell, or lower atom count."
                )

        atoms = Atoms(symbols=symbols, positions=positions_by_original, cell=cell, pbc=pbc)
        atoms.info.update(dict(structure.info))
        return atoms

    @classmethod
    def candidate_is_valid(
        cls,
        candidate: np.ndarray,
        symbol: str,
        *,
        placed_positions: list[np.ndarray],
        placed_symbols: list[str],
        min_distance: float,
        pair_rules: dict[tuple[str, str], float],
        cell: np.ndarray,
        pbc: np.ndarray,
        ortho_lengths: np.ndarray | None,
    ) -> bool:
        if not placed_positions:
            return True
        positions = np.asarray(placed_positions, dtype=float)
        distances = cls.candidate_distances(candidate, positions, cell=cell, pbc=pbc, ortho_lengths=ortho_lengths)
        if not pair_rules:
            return bool(np.all(distances + 1e-12 >= float(min_distance)))
        thresholds = np.asarray(
            [cls.min_distance_for_pair(symbol, other, min_distance, pair_rules) for other in placed_symbols],
            dtype=float,
        )
        return bool(np.all(distances + 1e-12 >= thresholds))

    @staticmethod
    def orthorhombic_lengths(cell: np.ndarray, pbc: np.ndarray) -> np.ndarray | None:
        cell_arr = np.asarray(cell, dtype=float)
        pbc_arr = np.asarray(pbc, dtype=bool)
        if cell_arr.shape != (3, 3):
            return None
        offdiag = cell_arr.copy()
        np.fill_diagonal(offdiag, 0.0)
        if not np.allclose(offdiag, 0.0, atol=1e-12):
            return None
        lengths = np.diag(cell_arr)
        if not np.all(np.abs(lengths[pbc_arr]) > 1e-12):
            return None
        return lengths

    @staticmethod
    def candidate_distances(candidate: np.ndarray, positions: np.ndarray, *, cell: np.ndarray, pbc: np.ndarray, ortho_lengths: np.ndarray | None = None) -> np.ndarray:
        cell_arr = np.asarray(cell, dtype=float)
        pbc_arr = np.asarray(pbc, dtype=bool)
        positions_arr = np.asarray(positions, dtype=float)
        if ortho_lengths is not None:
            vec = positions_arr - np.asarray(candidate, dtype=float).reshape(1, 3)
            for axis in range(3):
                if pbc_arr[axis]:
                    length = float(ortho_lengths[axis])
                    vec[:, axis] -= np.rint(vec[:, axis] / length) * length
            return np.linalg.norm(vec, axis=1)

        _vec, distances = get_distances(
            np.asarray(candidate, dtype=float).reshape(1, 3),
            positions_arr,
            cell=cell_arr,
            pbc=pbc_arr,
        )
        return np.asarray(distances, dtype=float).reshape(-1)


@dataclass(frozen=True)
class CrystalPrototypeBuilderParams:
    """Parameters for generating simple crystal prototypes."""

    lattice: str = "fcc"
    element: str = "Cu"
    a_range: tuple[float, float, float] = (3.6, 3.6, 0.1)
    covera: float = 1.633
    max_outputs: int = 200


@dataclass(frozen=True)
class CrystalPrototypePlan:
    """Exact output plan for a crystal-prototype request."""

    a_values: tuple[float, ...]
    atoms_per_output: int
    cell_lengths: tuple[float, float, float]
    truncated: bool


class CrystalPrototypeBuilderOperation(GeneratorOperation):
    """Generate fcc/bcc/hcp prototype structures without input data."""

    def generate(self, params: CrystalPrototypeBuilderParams) -> list:
        plan = self.plan(params)
        element = self._canonical_element(params.element)
        lattice = params.lattice.strip().lower()
        out = []
        for a in plan.a_values[: int(params.max_outputs)]:
            atoms = self._build_base(element, lattice, float(a), float(params.covera))
            atoms.pbc = True
            atoms.set_positions(fast_wrapped_positions(atoms, atoms.positions))
            append_config_tag(atoms, f"Proto({lattice},a={float(a):.6g})")
            out.append(atoms)
            if len(out) >= int(params.max_outputs):
                break
        return out

    def plan(self, params: CrystalPrototypeBuilderParams) -> CrystalPrototypePlan:
        """Validate parameters and return the exact base-cell/count preview."""
        element = self._canonical_element(params.element)
        lattice = params.lattice.strip().lower()
        if lattice not in {"fcc", "bcc", "hcp", "fcc111"}:
            raise CardOperationError(
                "crystal-prototype-lattice",
                "Unsupported crystal prototype: {lattice}.",
                lattice=params.lattice,
            )
        a_values = tuple(self._a_values(params.a_range))
        max_outputs = int(params.max_outputs)
        if max_outputs < 1:
            raise CardOperationError(
                "crystal-prototype-output-limit",
                "Maximum outputs must be at least 1.",
            )
        if lattice == "hcp" and (not math.isfinite(float(params.covera)) or float(params.covera) <= 0):
            raise CardOperationError(
                "crystal-prototype-covera",
                "The hcp c/a ratio must be a positive finite number.",
            )

        base = self._build_base(element, lattice, a_values[0], float(params.covera))
        return CrystalPrototypePlan(
            a_values=a_values,
            atoms_per_output=len(base),
            cell_lengths=tuple(float(value) for value in base.cell.lengths()),
            truncated=len(a_values) > max_outputs,
        )

    @staticmethod
    def _canonical_element(element: str) -> str:
        raw = element.strip()
        canonical = raw[0].upper() + raw[1:].lower() if raw else ""
        if not canonical or atomic_numbers.get(canonical, 0) <= 0:
            raise CardOperationError(
                "crystal-prototype-element",
                "Enter one valid chemical element symbol, for example Cu, Fe, or Mg.",
            )
        return canonical

    @staticmethod
    def _build_base(element: str, lattice: str, a: float, covera: float):
        if lattice == "fcc111":
            return fcc111(element, size=(1, 2, 3), a=float(a), vacuum=None, periodic=True, orthogonal=True)
        if lattice == "hcp":
            return bulk(element, "hcp", a=float(a), covera=float(covera))
        return bulk(element, lattice, a=float(a), cubic=True)

    @staticmethod
    def _a_values(values: tuple[float, float, float]) -> list[float]:
        a_min, a_max, a_step = [float(value) for value in values]
        if not all(math.isfinite(value) for value in (a_min, a_max, a_step)):
            raise CardOperationError(
                "crystal-prototype-a-finite",
                "The lattice-constant range must contain finite numbers.",
            )
        if a_min <= 0 or a_max <= 0:
            raise CardOperationError(
                "crystal-prototype-a-positive",
                "Lattice constants must be positive.",
            )
        if a_step <= 0:
            raise CardOperationError(
                "crystal-prototype-a-step",
                "The lattice-constant step must be positive.",
            )
        if a_max < a_min:
            a_min, a_max = a_max, a_min
        if abs(a_max - a_min) <= 1e-12:
            return [a_min]
        out = list(np.arange(a_min, a_max + 1e-12, a_step, dtype=float))
        return [float(value) for value in (out or [a_min])]
