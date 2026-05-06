"""UI-independent structure-generation Make Dataset operations."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase.build import bulk, make_supercell
from loguru import logger

from NepTrainKit.core.alloy import best_supercell_factors_max_atoms
from NepTrainKit.core.config_type import append_config_tag
from NepTrainKit.core.config_type import stable_config_id
from NepTrainKit.core.structure import get_vibration_modes
from NepTrainKit.core.torsion_guard_pbc import TorsionGuardParams, process_single as tg_process_single

from .operation import GeneratorOperation, StructureOperation


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
    """Stack copies of positions along z."""
    num_layers = max(1, int(num_layers))
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
    dz_expr: str = "sin(x/pi) + sin(y/pi)"
    expression_params: str = ""
    apply_mode: int = 0
    elements: str = ""
    z_range: tuple[float, float] = (-1000000.0, 1000000.0)
    wrap: bool = False
    extend_cell_z: bool = True
    extra_vacuum: float = 0.0
    layers: int = 3
    distance: float = 3.0


class LayerCopyOperation(StructureOperation):
    """Warp selected atoms by dz=f(x,y,z) and copy into stacked layers."""

    def run_structure(self, structure, params: LayerCopyParams) -> list:
        if not params.dz_expr.strip():
            raise ValueError("LayerCopy: dz expression is empty.")

        base = structure.copy()
        positions = np.asarray(base.get_positions(), dtype=float)
        mask = self.apply_mask(base, params)
        if not np.any(mask):
            raise ValueError("LayerCopy: no atoms selected by apply settings.")

        expr_params = parse_dz_params(params.expression_params)
        warped_positions = positions.copy()
        dz = evaluate_dz_expression(
            params.dz_expr.strip(),
            x=positions[mask, 0],
            y=positions[mask, 1],
            z=positions[mask, 2],
            params=expr_params,
        )
        warped_positions[mask, 2] = warped_positions[mask, 2] + dz

        layers = build_layers(warped_positions, num_layers=int(params.layers), layer_distance=float(params.distance))
        combined = base.copy()
        combined.set_positions(layers[0])
        for layer_pos in layers[1:]:
            layer_struct = base.copy()
            layer_struct.set_positions(layer_pos)
            combined += layer_struct

        if params.extend_cell_z and hasattr(combined, "cell"):
            base_cell = np.asarray(base.cell.array, dtype=float)
            if base_cell.shape == (3, 3) and int(params.layers) > 1:
                dz_total = abs(float(params.distance)) * (int(params.layers) - 1) + max(0.0, float(params.extra_vacuum))
                if dz_total > 0.0:
                    base_cell = base_cell.copy()
                    base_cell[2, 2] = base_cell[2, 2] + dz_total
                    combined.set_cell(base_cell, scale_atoms=False)

        if params.wrap and hasattr(combined, "wrap"):
            combined.wrap()

        append_config_tag(combined, f"SWC(L={int(params.layers)},dz={float(params.distance):g})")
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

    def run_structure(self, structure, params: VibrationModePerturbParams) -> list:
        amplitude = float(params.amplitude)
        if amplitude <= 0.0:
            raise ValueError("VibrationModePerturb: amplitude must be positive.")

        modes_per_sample = int(params.modes_per_sample)
        if modes_per_sample <= 0:
            raise ValueError("VibrationModePerturb: modes_per_sample must be >= 1.")

        min_frequency = float(params.min_frequency) if params.exclude_near_zero else 0.0
        frequencies, modes = get_vibration_modes(structure, min_frequency=min_frequency)
        if modes.size == 0:
            logger.warning("VibrationModePerturbOperation: no vibrational modes found on structure.")
            return []

        base_seed = int(params.seed) if params.use_seed else None
        rng = np.random.default_rng(base_seed)
        freq_for_scaling = np.abs(frequencies)
        freq_for_scaling[~np.isfinite(freq_for_scaling)] = 0.0
        replace = modes_per_sample > modes.shape[0]
        orig_positions = structure.get_positions()

        generated = []
        for _ in range(int(params.max_num)):
            indices = rng.choice(modes.shape[0], size=modes_per_sample, replace=replace)
            if int(params.distribution) == 0:
                coeffs = rng.normal(loc=0.0, scale=1.0, size=modes_per_sample)
            else:
                coeffs = rng.uniform(-1.0, 1.0, size=modes_per_sample)

            if params.scale_by_frequency:
                denominators = np.sqrt(np.clip(freq_for_scaling[indices], a_min=1e-12, a_max=None))
                denominators[denominators == 0.0] = 1.0
                coeffs = coeffs / denominators

            displacement = np.sum(coeffs[:, None, None] * modes[indices], axis=0)
            new_structure = structure.copy()
            new_structure.set_positions(orig_positions + amplitude * displacement)
            if hasattr(new_structure, "wrap"):
                new_structure.wrap()
            append_config_tag(new_structure, f"Vib(a={amplitude:.3f},m={modes_per_sample})")
            generated.append(new_structure)
        return generated


@dataclass(frozen=True)
class GroupLabelParams:
    """Parameters for assigning atoms.arrays['group'] labels."""

    mode: str = "k-vector layers (recommended)"
    kvec: str = "111"
    group_a: str = "A"
    group_b: str = "B"
    overwrite: bool = True


class GroupLabelOperation(StructureOperation):
    """Attach group labels using lattice-coordinate rules."""

    def run_structure(self, structure, params: GroupLabelParams) -> list:
        if (not params.overwrite) and "group" in structure.arrays:
            return [structure]
        if structure.cell is None or np.linalg.det(structure.cell.array) == 0:
            raise ValueError("GroupLabel: structure has no valid cell.")

        atoms = structure.copy()
        if params.mode.startswith("fractional parity"):
            flags = self._label_by_parity(atoms)
            tag = "par"
        else:
            flags = self._label_by_kvec(atoms, params.kvec)
            tag = f"k{params.kvec}"

        a_label = (params.group_a or "A").strip() or "A"
        b_label = (params.group_b or "B").strip() or "B"
        atoms.arrays["group"] = np.where(flags == 0, a_label, b_label).astype(object)
        append_config_tag(atoms, f"Grp({tag},{a_label}/{b_label})")
        return [atoms]

    @staticmethod
    def _parse_kvec(text: str) -> np.ndarray:
        text = (text or "").strip()
        if text in {"100", "010", "001", "110", "111"}:
            return np.array([int(value) for value in text], dtype=float)
        return np.array([1.0, 1.0, 1.0], dtype=float)

    @classmethod
    def _label_by_kvec(cls, atoms, kvec: str) -> np.ndarray:
        k = cls._parse_kvec(kvec)
        scaled = atoms.get_scaled_positions(wrap=True)
        phase = np.floor(2.0 * (scaled @ k)).astype(int)
        return (phase % 2).astype(int)

    @staticmethod
    def _label_by_parity(atoms) -> np.ndarray:
        scaled = atoms.get_scaled_positions(wrap=True)
        ints = np.rint(2.0 * scaled).astype(int)
        return (ints.sum(axis=1) % 2).astype(int)


@dataclass(frozen=True)
class OrganicMolConfigPBCParams:
    """Parameters for torsion-guard organic conformer generation."""

    perturb_per_frame: int = 100
    torsion_range_deg: tuple[float, float] = (-180.0, 180.0)
    max_torsions_per_conf: int = 50
    gaussian_sigma: float = 0.03
    pbc_mode: str = "auto"
    local_cutoff: int = 200
    local_subtree: int = 100
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


class OrganicMolConfigPBCOperation(StructureOperation):
    """Generate torsion-driven molecular conformers using TorsionGuard PBC."""

    def run_structure(self, structure, params: OrganicMolConfigPBCParams) -> list:
        symbols = structure.get_chemical_symbols()
        coords = structure.get_positions().astype(float)
        cell_mat = self._cell_matrix(structure)
        tg_params = TorsionGuardParams(
            perturb_per_frame=int(params.perturb_per_frame),
            torsion_range_deg=tuple(map(float, params.torsion_range_deg)),
            max_torsions_per_conf=int(params.max_torsions_per_conf),
            gaussian_sigma=float(params.gaussian_sigma),
            pbc_mode=params.pbc_mode,
            local_mode_cutoff_atoms=int(params.local_cutoff),
            local_torsion_max_subtree=int(params.local_subtree),
            bond_detect_factor=float(params.bond_detect_factor),
            bond_keep_min_factor=float(params.bond_keep_min_factor),
            bond_keep_max_factor=float(params.bond_keep_max_factor) if params.bond_keep_max_enable else None,
            nonbond_min_factor=float(params.nonbond_min_factor),
            max_retries_per_frame=int(params.max_retries),
            mult_bond_factor=float(params.mult_bond_factor),
            nonpbc_box_size=float(params.nonpbc_box_size),
            bo_c_const=float(params.bo_c_const),
            bo_threshold=float(params.bo_threshold),
            seed=(int(params.seed) + stable_config_id(structure) * 1000003 if params.use_seed else None),
        )
        result_list = tg_process_single(symbols, coords, cell_mat, tg_params)

        structures_out = []
        for _symbols, new_coords, cell, pbc_active in result_list:
            new_atoms = structure.copy()
            new_atoms.set_positions(np.array(new_coords, dtype=float))
            if pbc_active and cell is not None:
                new_atoms.set_cell(np.array(cell, dtype=float))
                new_atoms.set_pbc(True)
                try:
                    new_atoms.wrap()
                except Exception:
                    pass
            else:
                box = float(params.nonpbc_box_size)
                new_atoms.set_cell(np.diag([box, box, box]))
                new_atoms.set_pbc(False)
            append_config_tag(
                new_atoms,
                f"TG(n={int(params.perturb_per_frame)},sig={float(params.gaussian_sigma)},pbc={params.pbc_mode})",
            )
            structures_out.append(new_atoms)
        return structures_out

    @staticmethod
    def _cell_matrix(structure) -> np.ndarray | None:
        try:
            cell_arr = structure.get_cell().array
            if cell_arr is not None and np.array(cell_arr).shape == (3, 3):
                return np.array(cell_arr, dtype=float)
        except Exception:
            return None
        return None


@dataclass(frozen=True)
class CrystalPrototypeBuilderParams:
    """Parameters for generating simple crystal prototypes."""

    lattice: str = "fcc"
    element: str = "Cu"
    a_range: tuple[float, float, float] = (3.6, 3.6, 0.1)
    covera: float = 1.633
    auto_supercell: bool = True
    max_atoms: int = 512
    rep: tuple[int, int, int] = (4, 4, 4)
    max_outputs: int = 200


class CrystalPrototypeBuilderOperation(GeneratorOperation):
    """Generate fcc/bcc/hcp prototype structures without input data."""

    def generate(self, params: CrystalPrototypeBuilderParams) -> list:
        element = self._canonical_element(params.element)
        lattice = params.lattice.strip().lower()
        out = []
        for a in self._a_values(params.a_range):
            if lattice == "hcp":
                base = bulk(element, "hcp", a=float(a), covera=float(params.covera))
            else:
                base = bulk(element, lattice, a=float(a), cubic=True)
            base.pbc = True
            base.wrap()

            if params.auto_supercell:
                factors = best_supercell_factors_max_atoms(base, int(params.max_atoms))
                na, nb, nc = factors.na, factors.nb, factors.nc
            else:
                na, nb, nc = [int(value) for value in params.rep]

            matrix = np.diag([max(na, 1), max(nb, 1), max(nc, 1)])
            atoms = make_supercell(base, matrix)
            atoms.wrap()
            append_config_tag(atoms, f"Proto({lattice},a={float(a):.6g},rep={int(na)}x{int(nb)}x{int(nc)})")
            out.append(atoms)
            if len(out) >= int(params.max_outputs):
                break
        return out

    @staticmethod
    def _canonical_element(element: str) -> str:
        element = element.strip() or "Cu"
        return element[0].upper() + element[1:].lower()

    @staticmethod
    def _a_values(values: tuple[float, float, float]) -> list[float]:
        a_min, a_max, a_step = [float(value) for value in values]
        if a_step <= 0:
            return [a_min]
        if a_max < a_min:
            a_min, a_max = a_max, a_min
        if abs(a_max - a_min) <= 1e-12:
            return [a_min]
        out = list(np.arange(a_min, a_max + 1e-12, a_step, dtype=float))
        return [float(value) for value in (out or [a_min])]
