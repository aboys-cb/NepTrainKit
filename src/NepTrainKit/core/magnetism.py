"""Helpers for generating collinear magnetic moment patterns."""

from __future__ import annotations

import json
import re
from typing import Iterable, Mapping

import numpy as np
from ase import Atoms


_ITEM_SPLIT_RE = re.compile(r"[,\s;]+")
_VECTOR_MAP_ITEM_RE = re.compile(
    r"(?P<element>[A-Za-z][A-Za-z]?)\s*:\s*"
    r"(?P<value>\[[^\]]*\]|[^,\s;]+)"
)


def parse_magmom_map(text: str) -> dict[str, float]:
    """Parse per-element magnetic moments from JSON or ``Fe:2.2,Cr:1.0``."""
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("{") and text.endswith("}"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError("Magmom JSON must be an object mapping element->moment.")
        return {str(k): float(v) for k, v in data.items()}
    items = [t.strip() for t in _ITEM_SPLIT_RE.split(text) if t.strip()]
    out: dict[str, float] = {}
    for item in items:
        if ":" not in item:
            continue
        key, val = item.split(":", 1)
        key = key.strip()
        if not key:
            continue
        symbol = key[0].upper() + key[1:].lower()
        out[symbol] = float(val)
    return out


def parse_element_set(text: str) -> set[str]:
    """Parse a comma/space-separated element-symbol list."""
    tokens = [t.strip() for t in _ITEM_SPLIT_RE.split((text or "").strip()) if t.strip()]
    return {token[0].upper() + token[1:].lower() for token in tokens}


def parse_magmom_map_any(text: str) -> dict[str, float | np.ndarray]:
    """Parse per-element moments allowing either scalar or 3-vector values.

    Accepts JSON like ``{"Fe": 2.2, "Cr": [0, 0, 1.0]}`` or tokens like
    ``Fe:2.2,Cr:[0,0,1.0]``.
    """
    text = (text or "").strip()
    if not text:
        return {}

    if text.startswith("{") and text.endswith("}"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise TypeError("Magmom JSON must be an object mapping element->moment.")
        out: dict[str, float | np.ndarray] = {}
        for k, v in data.items():
            sym = str(k)
            if isinstance(v, (list, tuple)) and len(v) == 3:
                out[sym] = np.array([float(x) for x in v], dtype=float)
            else:
                out[sym] = float(v)
        return out

    out: dict[str, float | np.ndarray] = {}
    for match in _VECTOR_MAP_ITEM_RE.finditer(text):
        key = match.group("element")
        val = match.group("value").strip()
        symbol = key[0].upper() + key[1:].lower()
        if val.startswith("[") and val.endswith("]"):
            vec = json.loads(val)
            if not (isinstance(vec, list) and len(vec) == 3):
                raise ValueError(f"Invalid magmom vector for {symbol}: {val}")
            out[symbol] = np.array([float(x) for x in vec], dtype=float)
        else:
            out[symbol] = float(val)
    return out


def parse_kvec(text: str) -> tuple[int, int, int]:
    """Parse one of the supported fractional-coordinate layer vectors."""
    value = str(text or "").strip()
    if value not in {"100", "010", "001", "110", "111"}:
        raise ValueError(
            f"Unsupported k-vector {value!r}; use 100, 010, 001, 110, or 111."
        )
    return tuple(int(component) for component in value)  # type: ignore[return-value]


def element_mask(symbols: Iterable[str], selected: Iterable[str] | None = None) -> np.ndarray:
    """Return a boolean mask for the selected chemical symbols."""
    symbol_list = list(symbols)
    if selected is None:
        return np.ones(len(symbol_list), dtype=bool)
    normalized = {str(sym)[0].upper() + str(sym)[1:].lower() for sym in selected if str(sym).strip()}
    if not normalized:
        return np.ones(len(symbol_list), dtype=bool)
    return np.array([sym in normalized for sym in symbol_list], dtype=bool)


def normalize_vector(vec: np.ndarray, *, default: np.ndarray | None = None) -> np.ndarray:
    """Return a unit vector; fall back to default when norm is zero."""
    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        if default is None:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        d = np.asarray(default, dtype=float).reshape(3)
        dn = float(np.linalg.norm(d))
        return d / dn if dn > 0 else np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal frame ``(e1, e2, axis_hat)`` for a propagation axis."""
    axis_hat = normalize_vector(axis)
    trial = np.array([1.0, 0.0, 0.0], dtype=float) if abs(axis_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = normalize_vector(np.cross(axis_hat, trial), default=np.array([0.0, 1.0, 0.0], dtype=float))
    e2 = normalize_vector(np.cross(axis_hat, e1), default=np.array([0.0, 0.0, 1.0], dtype=float))
    return e1, e2, axis_hat


def spiral_unit_vectors(
    positions: np.ndarray,
    *,
    axis: np.ndarray,
    period: float,
    mz: float = 0.0,
    phase_deg: float = 0.0,
    chirality: int = 1,
    origin_projection: float | None = None,
) -> np.ndarray:
    """Return unit vectors for a helical/conical spin spiral.

    The field follows

    ``m(u) = sqrt(1-mz^2) [cos(phi) e1 + sin(phi) e2] + mz axis_hat``

    with ``phi = chirality * 2π u / period + phase`` and ``u`` the projection
    of each position along ``axis``.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape (n, 3)")

    period = float(period)
    if period <= 0:
        raise ValueError("period must be positive")

    mz = float(np.clip(mz, -1.0, 1.0))
    chirality = 1 if int(chirality) >= 0 else -1

    e1, e2, axis_hat = orthonormal_frame(axis)
    projections = pos @ axis_hat
    proj0 = float(np.min(projections)) if origin_projection is None else float(origin_projection)

    phase = chirality * (2.0 * np.pi * (projections - proj0) / period) + np.deg2rad(float(phase_deg))
    radial = float(np.sqrt(max(0.0, 1.0 - mz * mz)))

    moments = (
        radial * np.cos(phase)[:, None] * e1[None, :]
        + radial * np.sin(phase)[:, None] * e2[None, :]
        + mz * axis_hat[None, :]
    )
    norms = np.linalg.norm(moments, axis=1)
    norms = np.where(norms > 0, norms, 1.0)
    return moments / norms[:, None]


def parse_axis(text: str, *, default: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Parse an axis vector from JSON/list-like text such as ``0,0,1`` or ``[0,0,1]``."""
    raw = (text or "").strip()
    if not raw:
        return normalize_vector(np.array(default, dtype=float))
    if raw.startswith("[") and raw.endswith("]"):
        vec = json.loads(raw)
        if not (isinstance(vec, list) and len(vec) == 3):
            return normalize_vector(np.array(default, dtype=float))
        return normalize_vector(np.array([float(x) for x in vec], dtype=float))
    tokens = [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
    if len(tokens) != 3:
        return normalize_vector(np.array(default, dtype=float))
    return normalize_vector(np.array([float(t) for t in tokens], dtype=float))


def existing_moments(atoms: Atoms) -> np.ndarray | None:
    """Return canonical ``spin`` values, falling back to ASE magmoms.

    NEP datasets use a three-component ``spin:R:3`` field.  ASE operations use
    ``initial_magmoms`` internally, so legacy card inputs remain accepted when
    no canonical ``spin`` array is present.
    """
    arrays = getattr(atoms, "arrays", {}) or {}
    if "spin" in arrays:
        current = np.asarray(arrays["spin"])
        if (
            current.shape != (len(atoms), 3)
            or not np.issubdtype(current.dtype, np.number)
            or not np.isfinite(current).all()
        ):
            raise ValueError("spin must be a finite numeric N x 3 array")
        return np.asarray(current, dtype=float).copy()
    if "initial_magmoms" not in arrays:
        return None
    current = np.asarray(arrays["initial_magmoms"])
    if (
        current.shape not in {(len(atoms),), (len(atoms), 3)}
        or not np.issubdtype(current.dtype, np.number)
        or not np.isfinite(current).all()
    ):
        raise ValueError(
            "initial_magmoms must be a finite numeric N or N x 3 array"
        )
    return np.asarray(current, dtype=float).copy()


def existing_moment_magnitudes(atoms: Atoms) -> np.ndarray | None:
    """Return per-atom magnitudes, preferring canonical ``spin:R:3``."""
    current = existing_moments(atoms)
    if current is None:
        return None
    if current.ndim == 1 and current.shape[0] == len(atoms):
        return np.abs(current.reshape(-1))
    if current.ndim == 2 and current.shape == (len(atoms), 3):
        return np.linalg.norm(current, axis=1)
    return None


def existing_moment_vectors(
    atoms: Atoms,
    *,
    axis: np.ndarray | None = None,
    lift_scalar: bool = True,
) -> np.ndarray | None:
    """Return vectors, preferring canonical ``spin:R:3`` when available."""
    current = existing_moments(atoms)
    if current is None:
        return None
    if current.ndim == 2 and current.shape == (len(atoms), 3):
        return np.array(current, copy=True)
    if current.ndim == 1 and current.shape[0] == len(atoms) and lift_scalar:
        axis_vec = normalize_vector(axis if axis is not None else np.array([0.0, 0.0, 1.0], dtype=float))
        return current.reshape(-1, 1) * axis_vec.reshape(1, 3)
    return None


def existing_moment_scalars(
    atoms: Atoms,
    *,
    axis: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return axis projections, preferring canonical ``spin:R:3``."""
    current = existing_moments(atoms)
    if current is None:
        return None
    if current.ndim == 1 and current.shape[0] == len(atoms):
        return np.array(current, copy=True).reshape(-1)
    if current.ndim == 2 and current.shape == (len(atoms), 3):
        axis_vec = normalize_vector(axis if axis is not None else np.array([0.0, 0.0, 1.0], dtype=float))
        return np.asarray(current, dtype=float) @ axis_vec
    return None


def mapped_moment_magnitudes(
    atoms: Atoms,
    moment_map: Mapping[str, float | np.ndarray],
    *,
    default_moment: float = 0.0,
    apply_elements: Iterable[str] | None = None,
) -> np.ndarray:
    """Return per-atom moment magnitudes from a scalar/vector element map."""
    symbols = atoms.get_chemical_symbols()
    mask = element_mask(symbols, apply_elements)
    values = np.zeros(len(symbols), dtype=float)
    for idx, sym in enumerate(symbols):
        if not mask[idx]:
            continue
        raw = moment_map.get(sym, default_moment)
        if isinstance(raw, np.ndarray):
            values[idx] = float(np.linalg.norm(raw))
        else:
            values[idx] = abs(float(raw))
    return values


def mapped_moment_vectors(
    atoms: Atoms,
    moment_map: Mapping[str, float | np.ndarray],
    *,
    default_moment: float = 0.0,
    axis: np.ndarray | None = None,
    apply_elements: Iterable[str] | None = None,
    use_element_directions: bool = False,
) -> np.ndarray:
    """Return per-atom vector magnetic moments from a scalar/vector element map."""
    symbols = atoms.get_chemical_symbols()
    mask = element_mask(symbols, apply_elements)
    axis_vec = normalize_vector(axis if axis is not None else np.array([0.0, 0.0, 1.0], dtype=float))
    moments = np.zeros((len(symbols), 3), dtype=float)
    for idx, sym in enumerate(symbols):
        if not mask[idx]:
            continue
        raw = moment_map.get(sym, default_moment)
        if isinstance(raw, np.ndarray):
            magnitude = float(np.linalg.norm(raw))
            if magnitude <= 0.0:
                continue
            if use_element_directions:
                moments[idx] = magnitude * normalize_vector(raw)
            else:
                moments[idx] = magnitude * axis_vec
        else:
            moments[idx] = abs(float(raw)) * axis_vec
    return moments


def set_initial_magmoms_safe(
    atoms: Atoms,
    magmoms: np.ndarray,
    *,
    axis: np.ndarray | None = None,
) -> None:
    """Synchronize canonical ``spin`` with ASE's internal magmom array.

    Vector input maps directly to ``spin:R:3``.  Scalar input needs an explicit
    axis before it can be represented without inventing a direction.
    """
    target = np.asarray(magmoms)
    if (
        target.shape not in {(len(atoms),), (len(atoms), 3)}
        or not np.issubdtype(target.dtype, np.number)
        or not np.isfinite(target).all()
    ):
        raise ValueError("magmoms must be a finite numeric N or N x 3 array")
    target = np.asarray(target, dtype=float)
    spin = None
    if target.ndim == 2:
        spin = target
    elif axis is not None:
        axis_vector = np.asarray(axis, dtype=float).reshape(3)
        if (
            not np.isfinite(axis_vector).all()
            or float(np.linalg.norm(axis_vector)) <= 1.0e-12
        ):
            raise ValueError(
                "scalar magmoms require a finite nonzero axis for spin:R:3"
            )
        spin = (
            target[:, np.newaxis]
            * normalize_vector(axis_vector)[np.newaxis, :]
        )
    if (
        "initial_magmoms" in atoms.arrays
        and np.asarray(atoms.arrays["initial_magmoms"]).shape != target.shape
    ):
        del atoms.arrays["initial_magmoms"]
    atoms.set_initial_magnetic_moments(target)
    if spin is None:
        atoms.arrays.pop("spin", None)
        return
    if "spin" in atoms.arrays:
        del atoms.arrays["spin"]
    atoms.set_array("spin", np.asarray(spin, dtype=float))


def prepare_magnetic_extxyz_export(atoms: Atoms) -> Atoms:
    """Return an EXTXYZ view using canonical ``spin`` without ASE aliases."""
    arrays = getattr(atoms, "arrays", {}) or {}
    spin = existing_moments(atoms) if "spin" in arrays else None
    if spin is None and "initial_magmoms" in arrays:
        legacy = existing_moments(atoms)
        if legacy is not None and legacy.ndim == 2:
            spin = legacy
    exported = atoms.copy()
    exported.info.pop("_response_manifest_record", None)
    # Tangents are derived from grouped spin(coordinate) curves.  Keeping a
    # stale per-frame tangent would turn a preprocessing quantity into an
    # apparent DFT label, so production exports always remove it.
    exported.arrays.pop("spin_tangent", None)
    if spin is None:
        return exported
    exported.arrays.pop("initial_magmoms", None)
    exported.arrays.pop("spin", None)
    exported.set_array("spin", np.asarray(spin, dtype=float))
    return exported


def per_atom_magnitudes(
    atoms: Atoms,
    moment_map: Mapping[str, float],
    *,
    default_moment: float = 0.0,
) -> np.ndarray:
    """Return per-atom magnitudes (non-negative) for collinear magnetic moments."""
    default_moment = float(default_moment)
    mags = np.array(
        [abs(float(moment_map.get(sym, default_moment))) for sym in atoms.get_chemical_symbols()],
        dtype=float,
    )
    return mags


def kvec_signs(atoms: Atoms, kvec: tuple[int, int, int]) -> np.ndarray:
    """Generate +/-1 signs using a commensurate modulation in fractional coordinates.

    The sign is ``(-1)^floor(2 * (scaled_pos · kvec))``. This yields FM for
    k=(0,0,0) and simple AFM-like layer patterns for common k vectors such as
    (1,0,0), (1,1,0), and (1,1,1).
    """
    if kvec == (0, 0, 0):
        return np.ones(len(atoms), dtype=float)
    scaled = atoms.get_scaled_positions(wrap=True)
    phase = np.floor(
        2.0 * (scaled @ np.array(kvec, dtype=float)) + 1.0e-12
    ).astype(int)
    signs = np.where((phase % 2) == 0, 1.0, -1.0)
    return signs


def random_signs(
    n: int,
    *,
    rng: np.random.Generator,
    balanced: bool = True,
) -> np.ndarray:
    """Generate random +/-1 signs (optionally balanced to near-zero net moment)."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=float)
    if not balanced:
        return rng.choice(np.array([-1.0, 1.0], dtype=float), size=n, replace=True)
    n_pos = n // 2
    n_neg = n - n_pos
    signs = np.array([1.0] * n_pos + [-1.0] * n_neg, dtype=float)
    rng.shuffle(signs)
    return signs


def random_unit_vectors_sphere(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Sample unit vectors uniformly on the sphere."""
    n = int(n)
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1)
    norms = np.where(norms > 0, norms, 1.0)
    return v / norms[:, None]


def random_unit_vectors_cone(
    n: int,
    *,
    rng: np.random.Generator,
    axis: np.ndarray,
    max_angle_deg: float,
) -> np.ndarray:
    """Sample unit vectors within a cone around ``axis`` (uniform in solid angle)."""
    n = int(n)
    axis = normalize_vector(axis)
    max_angle = float(max_angle_deg) * np.pi / 180.0
    if max_angle <= 0:
        return np.repeat(axis[None, :], n, axis=0)

    # Sample cos(theta) uniformly between [cos(max_angle), 1]
    u = rng.random(n)
    cos_t = 1.0 - u * (1.0 - np.cos(max_angle))
    sin_t = np.sqrt(np.clip(1.0 - cos_t * cos_t, 0.0, 1.0))
    phi = rng.random(n) * 2.0 * np.pi

    # Build an orthonormal basis (e1, e2, axis)
    tmp = np.array([1.0, 0.0, 0.0], dtype=float) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = normalize_vector(np.cross(axis, tmp))
    e2 = np.cross(axis, e1)

    dirs = (cos_t[:, None] * axis[None, :]) + (sin_t[:, None] * (np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :]))
    # Numeric guard
    norms = np.linalg.norm(dirs, axis=1)
    norms = np.where(norms > 0, norms, 1.0)
    return dirs / norms[:, None]


def random_unit_vectors_plane(
    n: int,
    *,
    rng: np.random.Generator,
    normal: np.ndarray,
) -> np.ndarray:
    """Sample unit vectors uniformly in the plane perpendicular to ``normal``."""
    n = int(n)
    normal = normalize_vector(normal)
    tmp = np.array([1.0, 0.0, 0.0], dtype=float) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = normalize_vector(np.cross(normal, tmp))
    e2 = np.cross(normal, e1)
    phi = rng.random(n) * 2.0 * np.pi
    return np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :]


def random_vector_moments(
    magnitudes: np.ndarray,
    *,
    rng: np.random.Generator,
    direction_mode: str = "sphere",
    axis: np.ndarray | None = None,
    max_angle_deg: float = 180.0,
    balanced: bool = False,
) -> np.ndarray:
    """Generate non-collinear magnetic moments from magnitudes and a direction distribution."""
    mags = np.asarray(magnitudes, dtype=float).reshape(-1)
    n = len(mags)
    if n == 0:
        return np.zeros((0, 3), dtype=float)

    mode = (direction_mode or "sphere").strip().lower()
    axis_vec = normalize_vector(axis if axis is not None else np.array([0.0, 0.0, 1.0], dtype=float))
    def sample_directions(count: int) -> np.ndarray:
        if mode in {"axis", "collinear"}:
            signs = rng.choice(
                np.array([-1.0, 1.0], dtype=float),
                size=count,
                replace=True,
            )
            return signs[:, None] * axis_vec[None, :]
        if mode == "cone":
            return random_unit_vectors_cone(
                count,
                rng=rng,
                axis=axis_vec,
                max_angle_deg=max_angle_deg,
            )
        if mode in {"plane", "planar"}:
            return random_unit_vectors_plane(count, rng=rng, normal=axis_vec)
        return random_unit_vectors_sphere(count, rng=rng)

    if not balanced:
        return mags[:, None] * sample_directions(n)

    # Pair opposite directions separately for each distinct magnitude.  This
    # preserves every requested magnitude and cancels exactly within complete
    # pairs; one residual direction remains for odd-sized magnitude groups.
    dirs = np.zeros((n, 3), dtype=float)
    for magnitude in np.unique(mags):
        indices = np.flatnonzero(mags == magnitude)
        rng.shuffle(indices)
        pair_count = len(indices) // 2
        if pair_count:
            base_dirs = sample_directions(pair_count)
            dirs[indices[:pair_count]] = base_dirs
            dirs[indices[pair_count : 2 * pair_count]] = -base_dirs
        if len(indices) % 2:
            dirs[indices[-1]] = sample_directions(1)[0]
    return mags[:, None] * dirs
