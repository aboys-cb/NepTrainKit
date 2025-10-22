#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Importer registry for converting simulation outputs into Structure objects."""
from __future__ import annotations
import traceback
from pathlib import Path
from typing import Iterable, Protocol, List
from loguru import logger
from NepTrainKit.paths import PathLike, as_path
from NepTrainKit.core.structure import Structure, atomic_numbers
import numpy as np
class FormatImporter(Protocol):
    """Importer interface for converting various outputs into Structure objects."""
    name: str
    def matches(self, path: PathLike) -> bool:
        """Return True if this importer can handle the given file or directory."""
        ...
    def iter_structures(self, path: PathLike, **kwargs) -> Iterable[Structure]:
        """Yield Structure objects from the given path."""
        ...
_IMPORTERS: list[FormatImporter] = []
def register_importer(importer: FormatImporter) -> FormatImporter:
    """Register a format importer in the global registry.

    Parameters
    ----------
    importer : FormatImporter
        Importer instance to expose through convenience helpers.

    Returns
    -------
    FormatImporter
        The same importer, enabling decorator usage.
    """
    _IMPORTERS.append(importer)
    return importer
def is_parseable(path: PathLike) -> bool:
    """Return ``True`` if any registered importer recognises ``path``."""
    candidate = as_path(path)
    for imp in _IMPORTERS:
        try:
            if imp.matches(candidate):
                return True
        except Exception:
            continue
    return False
def import_structures(path: PathLike, **kwargs) -> List[Structure]:
    """Try each registered importer until one yields structures."""
    candidate = as_path(path)
    for imp in _IMPORTERS:
        try:
            if imp.matches(candidate):
                return list(imp.iter_structures(candidate, **kwargs))
        except Exception:
            logger.error(
                f"Importer {imp.__class__.__name__} failed for {candidate}: {traceback.format_exc()}"
            )
            continue
    return []
# ----------- Built-in importers -----------
class ExtxyzImporter:
    """Importer for standard and extended XYZ trajectory files."""
    name = "extxyz"
    def matches(self, path: PathLike) -> bool:
        """Return ``True`` when ``path`` points to an XYZ or EXTXYZ file.

        Parameters
        ----------
        path : PathLike
            Candidate file to inspect.

        Returns
        -------
        bool
            ``True`` if the suffix matches ``.xyz`` or ``.extxyz``.
        """
        candidate = as_path(path)
        return candidate.is_file() and candidate.suffix.lower() in {".xyz", ".extxyz"}
    def iter_structures(self, path: PathLike, **kwargs):
        """Yield structures parsed from an XYZ or EXTXYZ file.

        Parameters
        ----------
        path : PathLike
            Path to the trajectory file.
        **kwargs
            Forwarded to :meth:`Structure.iter_read_multiple`.

        Yields
        ------
        Structure
            Parsed configurations in file order.
        """
        candidate = as_path(path)

        return Structure.read_multiple_fast(str(candidate), **kwargs)
register_importer(ExtxyzImporter())
# VASP XDATCAR importer
class XdatcarImporter:
    """Importer for VASP XDATCAR trajectory files."""
    name = "vasp_xdatcar"
    def matches(self, path: PathLike) -> bool:
        """Return ``True`` when ``path`` resembles a VASP XDATCAR file.

        Parameters
        ----------
        path : PathLike
            Candidate file or directory to inspect.

        Returns
        -------
        bool
            ``True`` if the filename or suffix matches XDATCAR conventions.
        """
        candidate = as_path(path)
        ext = candidate.suffix.lower()
        return candidate.is_file() and (candidate.name.lower() == "xdatcar" or ext == ".xdatcar")
    def iter_structures(self, path: PathLike, **kwargs):
        """Parse VASP XDATCAR trajectory into :class:`Structure` frames.
        Notes
        -----
        - Supports variable cell per frame (XDATCAR-style headers before each config).
        - Coordinates are converted to Cartesian and stored under ``pos``.
        - Species are taken from header; if absent, falls back to dummy ``X1``/``X2``.
        """
        candidate = as_path(path)
        cancel_event = kwargs.get("cancel_event")
        def _is_number(s: str) -> bool:
            """Return ``True`` if ``s`` can be parsed as a floating-point number."""
            try:
                float(s)
                return True
            except Exception:
                return False
        with candidate.open("r", encoding="utf8", errors="ignore") as f:
            while True:
                if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                    return
                title = f.readline()
                if not title:
                    break
                # Skip possible blank lines
                while title.strip() == "":
                    if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                        return
                    title = f.readline()
                    if not title:
                        return
                scale_line = f.readline()
                if not scale_line:
                    break
                scale_line = scale_line.strip()
                if scale_line == "":
                    # Unexpected blank; try next
                    continue
                try:
                    scale = float(scale_line.split()[0])
                except Exception:
                    # Not a valid frame start; try to continue scanning
                    continue
                # Lattice 3 lines
                latt = []
                ok = True
                for _ in range(3):
                    if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                        return
                    line = f.readline()
                    if not line:
                        ok = False
                        break
                    parts = line.split()
                    if len(parts) < 3:
                        ok = False
                        break
                    try:
                        vec = [float(parts[0]), float(parts[1]), float(parts[2])]
                    except Exception:
                        ok = False
                        break
                    latt.append(vec)
                if not ok:
                    break
                lattice = (scale * np.array(latt, dtype=np.float32)).reshape(3, 3)
                # Species line or counts line
                line = f.readline()
                if not line:
                    break
                tokens = line.split()
                # If tokens all numbers -> counts-only header (no symbols)
                if all(_is_number(t) for t in tokens):
                    counts = [int(round(float(t))) for t in tokens]
                    # Try get symbols from kwargs or fall back to X1, X2, ...
                    sym_from_kw = kwargs.get("species", None)
                    if sym_from_kw is not None:
                        if len(sym_from_kw) != len(counts):
                            raise ValueError("Provided species length does not match counts in XDATCAR")
                        symbols = list(sym_from_kw)
                    else:
                        symbols = [f"X{i+1}" for i in range(len(counts))]
                else:
                    symbols = tokens
                    # Next line is counts
                    line2 = f.readline()
                    if not line2:
                        break
                    counts = [int(round(float(x))) for x in line2.split()]
                    if len(counts) != len(symbols):
                        # Some XDATCARs repeat header; be permissive
                        counts = counts[: len(symbols)]
                n_atoms = int(sum(counts))
                # Next line indicates coordinate mode
                mode_line = f.readline()
                if not mode_line:
                    break
                mode_l = mode_line.strip().lower()
                use_direct = ("direct" in mode_l)
                # Read n_atoms coordinate lines
                coords = np.zeros((n_atoms, 3), dtype=np.float32)
                read_ok = True
                for i in range(n_atoms):
                    if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                        return
                    c_line = f.readline()
                    if not c_line:
                        read_ok = False
                        break
                    parts = c_line.split()
                    if len(parts) < 3:
                        read_ok = False
                        break
                    try:
                        coords[i, 0] = float(parts[0])
                        coords[i, 1] = float(parts[1])
                        coords[i, 2] = float(parts[2])
                    except Exception:
                        read_ok = False
                        break
                if not read_ok:
                    break
                # Expand species list in-order
                species_list = np.concatenate([
                    np.array([sym] * cnt, dtype=np.str_)
                    for sym, cnt in zip(symbols, counts)
                ])
                # Convert to Cartesian if in direct (fractional) coords
                if use_direct:
                    positions = coords @ lattice
                else:
                    positions = coords.astype(np.float32)
                properties = [
                    {"name": "species", "type": "S", "count": 1},
                    {"name": "pos", "type": "R", "count": 3},
                ]
                atomic_properties = {
                    "species": species_list,
                    "pos": positions,
                }
                additional_fields = {
                    "Config_type": title.strip(),
                    "pbc": "T T T",
                }
                yield Structure(lattice=lattice,
                                atomic_properties=atomic_properties,
                                properties=properties,
                                additional_fields=additional_fields)
register_importer(XdatcarImporter())
# VASP OUTCAR importer
class OutcarImporter:
    """Importer that streams configurations from VASP OUTCAR files."""
    name = "vasp_outcar"
    def matches(self, path: PathLike) -> bool:
        """Return ``True`` when ``path`` looks like a VASP OUTCAR file."""
        candidate = as_path(path)
        ext = candidate.suffix.lower()
        return candidate.is_file() and (candidate.name.lower() == "outcar" or ext == ".outcar")
    def iter_structures(self, path: PathLike, cancel_event=None, **kwargs):
        """Stream VASP OUTCAR configurations as :class:`Structure` objects."""
        candidate = as_path(path)
        def parse_floats(line: str) -> list[float]:
            """Return floats parsed from ``line`` while tolerating Fortran notation."""
            parts = line.replace("D", "E").split()
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    pass
            return vals
        species_by_type: list[str] | None = None
        counts_by_type: list[int] | None = None
        latest_lattice: np.ndarray | None = None  # last seen lattice (for reference)
        pending_lattice: np.ndarray | None = None  # lattice to apply to next POSITION block
        # pending tensors for the next POSITION block
        pending_stress: np.ndarray | None = None
        pending_virial: np.ndarray | None = None  # eV, 9 comps row-major
        last_force_is_ml: bool | None = None
        frames: list[dict] = []
        # helpers for species mapping
        def finalize_species_list(n_atoms: int) -> np.ndarray:
            """Expand type-wise species metadata into a per-atom array."""
            nonlocal species_by_type, counts_by_type
            if counts_by_type is None:
                # fallback: unknown composition
                return np.array(["X"] * n_atoms, dtype=np.str_)
            if species_by_type is None or len(species_by_type) < len(counts_by_type):
                # best-effort: fill missing with X
                miss = len(counts_by_type) - (len(species_by_type or []))
                base = (species_by_type or []) + ["X"] * max(miss, 0)
            else:
                base = species_by_type
            expanded: list[str] = []
            for sym, cnt in zip(base, counts_by_type):
                expanded.extend([sym] * int(cnt))
            if len(expanded) != n_atoms:
                # fall back to generic X if mismatch
                return np.array(["X"] * n_atoms, dtype=np.str_)
            return np.array(expanded, dtype=np.str_)
        # Parse file sequentially
        with candidate.open("r", encoding="utf8", errors="ignore") as f:
            for raw in f:
                if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                    break
                line = raw.rstrip("\n")
                # ions per type
                if "ions per type" in line:
                    try:
                        right = line.split("=")[-1]
                        counts_by_type = [int(x) for x in right.split()]
                    except Exception:
                        counts_by_type = None
                    continue
                # Try to collect species by type via VRHFIN or TITEL blocks
                lt = line.lstrip()
                if lt.startswith("VRHFIN") and ":" in lt and "=" in lt:
                    try:
                        sym = lt.split("=")[1].split(":")[0].strip()
                        sym = sym.replace("_sv", "").strip()
                        if species_by_type is None:
                            species_by_type = []
                        # avoid duplicates if multiple POTCAR copies
                        if sym and (len(species_by_type) == 0 or species_by_type[-1] != sym):
                            species_by_type.append(sym)
                    except Exception:
                        pass
                    continue
                if lt.startswith("TITEL") and "=" in lt:
                    # heuristic from TITEL  = PAW_PBE Fe 06Sep2000
                    try:
                        tokens = lt.split("=")[-1].split()
                        # find first token that looks like element symbol (H or He)
                        cand = None
                        for t in tokens:
                            if len(t) <= 3 and t[0].isalpha() and t[0].isupper():
                                # strip suffix like Li_sv
                                base = t[:2]
                                if base[0].isupper() and (len(base) == 1 or base[1].islower()):
                                    cand = base
                                    break
                        if cand is not None:
                            if species_by_type is None:
                                species_by_type = []
                            if len(species_by_type) == 0 or species_by_type[-1] != cand:
                                species_by_type.append(cand)
                    except Exception:
                        pass
                    continue
                # direct lattice vectors (use the three next lines)
                if "direct lattice vectors" in line and "reciprocal" in line:
                    try:
                        a = parse_floats(next(f))
                        b = parse_floats(next(f))
                        c = parse_floats(next(f))
                        latest_lattice = np.array([[a[0], a[1], a[2]],
                                                   [b[0], b[1], b[2]],
                                                   [c[0], c[1], c[2]]], dtype=np.float32)
                        pending_lattice = latest_lattice.copy()
                    except Exception:
                        latest_lattice = latest_lattice
                    continue
                # Track header indicating whether next 'in kB' belongs to ML or DFT
                if line.strip().startswith("ML FORCE on cell") and "-STRESS" in line:
                    # We currently skip ML frames; mark and continue without capturing
                    last_force_is_ml = True
                    continue
                if line.strip().startswith("FORCE on cell") and "-STRESS" in line and not line.strip().startswith("ML "):
                    last_force_is_ml = False
                    # Try to peek matrix
                    try:
                        pos = f.tell()
                        l1 = next(f, ""); l2 = next(f, ""); l3 = next(f, "")
                        a1 = parse_floats(l1); a2 = parse_floats(l2); a3 = parse_floats(l3)
                        if len(a1) >= 3 and len(a2) >= 3 and len(a3) >= 3:
                            M = np.array([[a1[0], a1[1], a1[2]],
                                          [a2[0], a2[1], a2[2]],
                                          [a3[0], a3[1], a3[2]]], dtype=np.float32)
                            pending_virial = M.reshape(-1)
                        else:
                            f.seek(pos)
                    except Exception:
                        try:
                            f.seek(pos)
                        except Exception:
                            pass
                    continue
                # Stress in kB -> assign to next frame of matching type (ML or DFT)
                if line.strip().startswith("in kB"):
                    # Ignore ML stress to avoid mismatching with DFT POSITION blocks
                    if last_force_is_ml is True:
                        continue
                    vals = parse_floats(line)
                    # format: in kB  xx yy zz xy yz zx
                    if len(vals) >= 6:
                        xx, yy, zz, xy, yz, xz = vals[-6:]
                        to_ev_a3 = 0.1 / 160.21766208
                        xx *= to_ev_a3
                        yy *= to_ev_a3
                        zz *= to_ev_a3
                        xy *= to_ev_a3
                        yz *= to_ev_a3
                        xz *= to_ev_a3
                        # Convert VASP sign convention (compression positive) ->
                        # internal convention (tension positive): multiply by -1
                        xx, yy, zz, xy, yz, xz = (-xx, -yy, -zz, -xy, -yz, -xz)
                        # Build full 3x3 with proper placement:
                        # [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]]
                        stress = np.array([[xx, xy, xz],
                                           [xy, yy, yz],
                                           [xz, yz, zz]], dtype=np.float32)
                        # assign to next POSITION block (we track ML/DFT via last_force_is_ml)
                        pending_stress = stress.reshape(-1)
                    continue
                # Energy line (free  energy   TOTEN  = ... eV)
                if "free  energy   TOTEN" in line:
                    try:
                        e = float(line.split("=")[-1].split()[0])
                        if frames:
                            frames[-1]["energy"] = e
                    except Exception:
                        pass
                    continue
                # Position + forces block
                if line.strip().startswith("POSITION") and "TOTAL-FORCE" in line:
                    is_ml_block = "(ML)" in line
                    # Optional dash separator; or immediately data lines
                    sep = next(f, "")
                    positions: list[list[float]] = []
                    forces: list[list[float]] = []
                    # If the line isn't a separator, treat it as data
                    if sep and not set(sep.strip()) == {"-"} and sep.strip() != "":
                        cand = parse_floats(sep)
                        if len(cand) >= 6:
                            positions.append(cand[0:3])
                            forces.append(cand[-3:])
                    while True:
                        if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                            break
                        l2 = next(f, "")
                        if not l2:
                            break
                        if l2.strip() == "" or set(l2.strip()) == {"-"}:
                            break
                        nums = parse_floats(l2)
                        if len(nums) < 6:
                            break
                        positions.append(nums[0:3])
                        forces.append(nums[-3:])
                    n_atoms = len(positions)
                    if n_atoms == 0:
                        continue
                    species = finalize_species_list(n_atoms)
                    if pending_lattice is not None:
                        use_lattice = pending_lattice
                    else:
                        use_lattice = latest_lattice if latest_lattice is not None else np.eye(3, dtype=np.float32)
                    # consume pending tensors (align to current block kind)
                    stress_next = pending_stress
                    virial_next = pending_virial
                    pending_stress = None
                    pending_virial = None
                    props = [
                        {"name": "species", "type": "S", "count": 1},
                        {"name": "pos", "type": "R", "count": 3},
                        {"name": "forces", "type": "R", "count": 3},
                    ]
                    atom_props = {
                        "species": species,
                        "pos": np.array(positions, dtype=np.float32),
                        "forces": np.array(forces, dtype=np.float32),
                    }
                    fields = {
                        "Config_type": "OUTCAR",
                        "pbc": "T T T",
                    }
                    # Only keep DFT frames for downstream NEP, skip ML frames
                    if is_ml_block:
                        continue
                    frames.append({
                        "lattice": use_lattice.copy(),
                        "properties": props,
                        "atomic_properties": atom_props,
                        "additional_fields": fields,
                        **({"virial": virial_next} if virial_next is not None else {}),
                        **({"stress": stress_next} if stress_next is not None else {}),
                    })
        # Emit frames as Structure objects
        for i, fr in enumerate(frames):
            add = fr["additional_fields"].copy()
            if "energy" in fr:
                add["energy"] = fr["energy"]
            if "virial" in fr or "stress" in fr:
                if "virial" in fr:
                    v = fr["virial"].reshape(3, 3)
                    virial9 = np.array([v[0,0], v[0,1], v[0,2], v[1,0], v[1,1], v[1,2], v[2,0], v[2,1], v[2,2]], dtype=np.float32)
                    add["virial"] = virial9
                    # derive stress from virial
                    try:
                        vol = float(np.abs(np.linalg.det(fr["lattice"])) )
                        s = (-v / vol)
                        stress9 = np.array([s[0,0], s[0,1], s[0,2], s[1,0], s[1,1], s[1,2], s[2,0], s[2,1], s[2,2]], dtype=np.float32)
                        add["stress"] = stress9
                    except Exception:
                        pass
                else:
                    s = fr["stress"].reshape(3, 3)
                    stress9 = np.array([s[0,0], s[0,1], s[0,2], s[1,0], s[1,1], s[1,2], s[2,0], s[2,1], s[2,2]], dtype=np.float32)
                    add["stress"] = stress9
                    try:
                        vol = float(np.abs(np.linalg.det(fr["lattice"])) )
                        v = (-s * vol)
                        virial9 = np.array([v[0,0], v[0,1], v[0,2], v[1,0], v[1,1], v[1,2], v[2,0], v[2,1], v[2,2]], dtype=np.float32)
                        add["virial"] = virial9
                    except Exception:
                        pass
            add["Config_type"] = f"OUTCAR_{i+1}"
            yield Structure(lattice=fr["lattice"],
                            atomic_properties=fr["atomic_properties"],
                            properties=fr["properties"],
                            additional_fields=add)
register_importer(OutcarImporter())
# LAMMPS dump importer
class LammpsDumpImporter:
    """Importer for LAMMPS dump trajectory files."""
    name = "lammps_dump"
    def matches(self, path: PathLike) -> bool:
        """Return ``True`` when ``path`` appears to be a LAMMPS dump file."""
        candidate = as_path(path)
        ext = candidate.suffix.lower()
        if not candidate.is_file():
            return False
        # Check dump signature/extension
        if ext in {".dump", ".lammpstrj", ".lammpstraj"} or candidate.name.lower().endswith(".dump"):
            return True
        try:
            with candidate.open("r", encoding="utf8", errors="ignore") as f:
                head = f.readline()
            return head.strip().startswith("ITEM: TIMESTEP")
        except Exception:
            return False
    def iter_structures(self, path: PathLike, **kwargs):
        """Iterate over LAMMPS dump trajectory frames."""
        candidate = as_path(path)
        cancel_event = kwargs.get("cancel_event")
        element_resolver = kwargs.get(
            "element_resolver")  # Optional callable(missing_types:list[int], context:dict)->dict[int,str]
        element_map_arg = kwargs.get("element_map")  # Optional pre-supplied {type:int -> element:str}
        def cancelled():
            """Return ``True`` if an optional cancellation event is set."""
            return cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set()
        # Build element mapping from LAMMPS in-file or referenced data file (Masses section)
        source_dir = candidate.parent
        input_files: list[Path] = []
        try:
            for child in source_dir.iterdir():
                lower = child.name.lower()
                if child.is_file() and (lower.startswith("in") or lower.endswith("in") or lower.endswith(".in")):
                    input_files.append(child)
        except Exception:
            pass
        type_to_elem: dict[int, str] = {}
        if isinstance(element_map_arg, dict):
            # seed with user-provided mapping first
            for k, v in element_map_arg.items():
                try:
                    type_to_elem[int(k)] = str(v)
                except Exception:
                    pass
        with candidate.open("r", encoding="utf8", errors="ignore") as f:
            while True:
                if cancelled():
                    return
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("ITEM: TIMESTEP"):
                    continue
                # TIMESTEP
                ts_line = f.readline()
                if not ts_line:
                    break
                try:
                    timestep = int(ts_line.strip().split()[0])
                except Exception:
                    timestep = 0
                # NUMBER OF ATOMS
                hdr = f.readline()  # ITEM: NUMBER OF ATOMS
                if not hdr or not hdr.strip().startswith("ITEM: NUMBER OF ATOMS"):
                    break
                nat_line = f.readline()
                if not nat_line:
                    break
                try:
                    n_atoms = int(nat_line.strip().split()[0])
                except Exception:
                    continue
                # BOX BOUNDS
                bounds_hdr = f.readline()
                if not bounds_hdr or not bounds_hdr.strip().startswith("ITEM: BOX BOUNDS"):
                    break
                bounds_tokens = bounds_hdr.strip().split()
                tilt_flags = {t for t in bounds_tokens if t in {"xy", "xz", "yz"}}
                def _read_bounds_line():
                    """Read a bounds line from the dump header and return floats."""
                    l = f.readline()
                    return [float(x) for x in l.strip().split()] if l else []
                b1 = _read_bounds_line(); b2 = _read_bounds_line(); b3 = _read_bounds_line()
                if not b1 or not b2 or not b3:
                    break
                if tilt_flags:
                    # triclinic: xlo xhi xy; ylo yhi xz; zlo zhi yz
                    xlo, xhi, xy = b1[0], b1[1], b1[2]
                    ylo, yhi, xz = b2[0], b2[1], b2[2]
                    zlo, zhi, yz = b3[0], b3[1], b3[2]
                else:
                    xlo, xhi = b1[0], b1[1]
                    ylo, yhi = b2[0], b2[1]
                    zlo, zhi = b3[0], b3[1]
                    xy = xz = yz = 0.0
                lx = float(xhi - xlo)
                ly = float(yhi - ylo)
                lz = float(zhi - zlo)
                a = np.array([lx, 0.0, 0.0], dtype=np.float32)
                b = np.array([xy, ly, 0.0], dtype=np.float32)
                c = np.array([xz, yz, lz], dtype=np.float32)
                lattice = np.vstack([a, b, c]).reshape(3, 3)
                # ATOMS header
                atoms_hdr = f.readline()
                if not atoms_hdr or not atoms_hdr.strip().startswith("ITEM: ATOMS"):
                    break
                cols = atoms_hdr.strip().split()[2:]
                idx = {name: i for i, name in enumerate(cols)}
                has_scaled = all(k in idx for k in ("xs", "ys", "zs"))
                has_cart = all(k in idx for k in ("x", "y", "z"))
                has_unwrapped = all(k in idx for k in ("xu", "yu", "zu"))
                has_forces = all(k in idx for k in ("fx", "fy", "fz"))
                species_col = "element" if "element" in idx else ("type" if "type" in idx else None)
                positions = np.zeros((n_atoms, 3), dtype=np.float32)
                forces = np.zeros((n_atoms, 3), dtype=np.float32) if has_forces else None
                species_list: list[str] = []
                types_buffer = np.zeros((n_atoms,), dtype=np.int32) if species_col == "type" else None
                for i in range(n_atoms):
                    if cancelled():
                        return
                    l = f.readline()
                    if not l:
                        break
                    parts = l.split()
                    # species
                    if species_col is None:
                        species_list.append("X")
                    else:
                        val = parts[idx[species_col]]
                        if species_col == "type":
                            try:
                                tnum = int(float(val))
                            except Exception:
                                tnum = -1
                            if types_buffer is not None:
                                types_buffer[i] = tnum
                            # placeholder; will resolve after reading all atoms
                            species_list.append("")
                        else:
                            species_list.append(val)
                    # fractional
                    if has_scaled:
                        fx = float(parts[idx["xs"]]); fy = float(parts[idx["ys"]]); fz = float(parts[idx["zs"]])
                    elif has_cart:
                        x = float(parts[idx["x"]]); y = float(parts[idx["y"]]); z = float(parts[idx["z"]])
                        fx = (x - xlo) / lx if lx != 0 else 0.0
                        fy = (y - ylo) / ly if ly != 0 else 0.0
                        fz = (z - zlo) / lz if lz != 0 else 0.0
                    elif has_unwrapped:
                        x = float(parts[idx["xu"]]); y = float(parts[idx["yu"]]); z = float(parts[idx["zu"]])
                        fx = (x - xlo) / lx if lx != 0 else 0.0
                        fy = (y - ylo) / ly if ly != 0 else 0.0
                        fz = (z - zlo) / lz if lz != 0 else 0.0
                    else:
                        fx = fy = fz = 0.0
                    pos = fx * a + fy * b + fz * c
                    positions[i, :] = pos
                    if has_forces and forces is not None:
                        forces[i, 0] = float(parts[idx["fx"]])
                        forces[i, 1] = float(parts[idx["fy"]])
                        forces[i, 2] = float(parts[idx["fz"]])
                # Resolve missing type->element mapping if needed
                if species_col == "type" and types_buffer is not None:
                    missing = sorted({int(t) for t in types_buffer.tolist() if int(t) >= 1 and int(t) not in type_to_elem})
                    if missing and callable(element_resolver):
                        try:
                            ctx = {
                                "path": path,
                                "n_atoms": n_atoms,
                                "timestep": timestep,
                                "present_types": missing,
                            }
                            ret = element_resolver(missing, ctx)
                            if isinstance(ret, dict):
                                for k, v in ret.items():
                                    try:
                                        type_to_elem[int(k)] = str(v)
                                    except Exception:
                                        pass
                        except Exception:
                            # ignore resolver errors, will fallback
                            pass
                    # fill species_list based on mapping (fallback to X<type>)
                    for i in range(n_atoms):
                        t = int(types_buffer[i])
                        species_list[i] = type_to_elem.get(t, f"X{t}") if t > 0 else "X"
                species_arr = np.array(species_list, dtype=np.str_)
                properties = [
                    {"name": "species", "type": "S", "count": 1},
                    {"name": "pos", "type": "R", "count": 3},
                ]
                atom_props = {
                    "species": species_arr,
                    "pos": positions,
                }
                if has_forces and forces is not None:
                    properties.append({"name": "forces", "type": "R", "count": 3})
                    atom_props["forces"] = forces
                additional_fields = {
                    "Config_type": f"LAMMPS_{timestep}",
                    "pbc": "T T T",
                }
                yield Structure(lattice=lattice,
                                atomic_properties=atom_props,
                                properties=properties,
                                additional_fields=additional_fields)
register_importer(LammpsDumpImporter())
# Skeleton for CP2K output importer (optional)
class Cp2kOutputImporter:
    """Importer for CP2K output log files."""
    name = "cp2k_output"
    def matches(self, path: PathLike) -> bool:
        """Return ``True`` when ``path`` looks like a CP2K output log."""
        candidate = as_path(path)
        if not candidate.is_file():
            return False
        base = candidate.name.lower()
        ext = candidate.suffix.lower()
        likely = base.endswith(".log") or base.endswith(".out") or ext in {".log", ".out"}
        try:
            with candidate.open('r', encoding='utf8', errors='ignore') as f:
                head = f.read(4000)
            sig = ("CP2K|" in head) or ("MODULE QUICKSTEP: ATOMIC COORDINATES" in head) or ("ENERGY| Total FORCE_EVAL" in head)
            return sig
        except Exception:
            return False
    def iter_structures(self, path: PathLike, **kwargs):
        """Parse a CP2K output into one Structure.
        Extracts:
        - Lattice from CELL| Vector a/b/c [angstrom]
        - Atomic coordinates from "MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM"
        - Forces from "ATOMIC FORCES in [a.u.]" (converted to eV/脜)
        - Total energy from "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]" (converted to eV)
        - Stress tensor from "STRESS| Analytical stress tensor [GPa]" (converted to eV/脜^3)
        """
        candidate = as_path(path)

        cancel_event = kwargs.get("cancel_event")
        def cancelled():
            """Return ``True`` if an optional cancellation event is set."""
            return cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set()
        # unit conversions
        HARTREE_TO_EV = 27.211386245988
        AU_FORCE_TO_EV_PER_ANG = 27.211386245988 / 0.52917721067
        GPA_TO_EV_PER_ANG3 = 1.0 / 160.21766208
        # accumulators
        a_vec = b_vec = c_vec = None
        positions: list[list[float]] = []
        species: list[str] = []
        forces: list[list[float]] = []
        energy_ev: float | None = None
        stress_gpa: np.ndarray | None = None
        # state flags
        in_coords = False
        coords_started = False  # started reading numeric atom lines
        in_forces = False
        read_forces_header_skipped = False
        def parse_floats_from_line(line: str) -> list[float]:
            """Return floats parsed from ``line`` with Fortran ``D`` exponents."""
            vals: list[float] = []
            for t in line.replace('D', 'E').split():
                try:
                    vals.append(float(t))
                except Exception:
                    pass
            return vals
        with candidate.open('r', encoding='utf8', errors='ignore') as f:
            for raw in f:
                if cancelled():
                    return
                line = raw.rstrip('\n')
                lstrip = line.lstrip()
                # Lattice vectors (prefer the current CELL| over *_TOP or *_REF)
                if lstrip.startswith('CELL|') and 'Vector a' in lstrip and '[angstrom' in lstrip:
                    nums = parse_floats_from_line(line)
                    if len(nums) >= 3:
                        a_vec = [nums[0], nums[1], nums[2]]
                    continue
                if lstrip.startswith('CELL|') and 'Vector b' in lstrip and '[angstrom' in lstrip:
                    nums = parse_floats_from_line(line)
                    if len(nums) >= 3:
                        b_vec = [nums[0], nums[1], nums[2]]
                    continue
                if lstrip.startswith('CELL|') and 'Vector c' in lstrip and '[angstrom' in lstrip:
                    nums = parse_floats_from_line(line)
                    if len(nums) >= 3:
                        c_vec = [nums[0], nums[1], nums[2]]
                    continue
                # Coordinates block begin
                if 'MODULE QUICKSTEP: ATOMIC COORDINATES IN ANGSTROM' in line:
                    in_coords = True
                    coords_started = False
                    continue
                if in_coords:
                    # skip blank lines until data starts
                    if line.strip() == '':
                        if coords_started:
                            # blank after data -> end of block
                            in_coords = False
                        continue
                    parts = line.split()
                    # skip section header row
                    if len(parts) >= 3 and parts[0].lower() == 'atom' and parts[1].lower() == 'kind':
                        continue
                    # Expect numeric rows: idx kind Element Z X Y Z Z(eff) Mass
                    def _is_int(s: str) -> bool:
                        """Return ``True`` when ``s`` can be interpreted as an integer."""
                        try:
                            int(float(s))
                            return True
                        except Exception:
                            return False
                    if len(parts) >= 7 and _is_int(parts[0]) and _is_int(parts[1]):
                        try:
                            elem = parts[2]
                            x = float(parts[4]); y = float(parts[5]); z = float(parts[6])
                            species.append(elem)
                            positions.append([x, y, z])
                            coords_started = True
                            continue
                        except Exception:
                            # tolerate and keep scanning within block
                            pass
                    # Non-parsable line while in block; if we already collected atoms, end block on format change
                    if coords_started:
                        in_coords = False
                    continue
                # Forces block begin
                if lstrip.startswith('ATOMIC FORCES in [a.u.]'):
                    in_forces = True
                    read_forces_header_skipped = False
                    continue
                if in_forces:
                    # skip header line that starts with '#'
                    if not read_forces_header_skipped:
                        if line.strip() == '' or line.strip().startswith('#'):
                            if line.strip().startswith('#'):
                                read_forces_header_skipped = True
                            continue
                        else:
                            read_forces_header_skipped = True
                    if line.strip() == '' or line.strip().startswith('SUM OF ATOMIC FORCES'):
                        in_forces = False
                        continue
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            fx = float(parts[-3]) * AU_FORCE_TO_EV_PER_ANG
                            fy = float(parts[-2]) * AU_FORCE_TO_EV_PER_ANG
                            fz = float(parts[-1]) * AU_FORCE_TO_EV_PER_ANG
                            forces.append([fx, fy, fz])
                        except Exception:
                            pass
                    continue
                # Energy (a.u. -> eV)
                if 'ENERGY| Total FORCE_EVAL' in line and '[a.u.]' in line and ':' in line:
                    try:
                        val = float(line.split(':')[-1].split()[0])
                        energy_ev = val * HARTREE_TO_EV
                    except Exception:
                        pass
                    continue
                # Fallback energy line
                if lstrip.startswith('Total energy:'):
                    try:
                        val = float(lstrip.split(':')[-1].split()[0])
                        energy_ev = val * HARTREE_TO_EV
                    except Exception:
                        pass
                    continue
                # Stress tensor in GPa
                if lstrip.startswith('STRESS| Analytical stress tensor'):
                    _ = next(f, '')  # header line
                    rowx = next(f, '')
                    rowy = next(f, '')
                    rowz = next(f, '')
                    try:
                        vx = parse_floats_from_line(rowx)
                        vy = parse_floats_from_line(rowy)
                        vz = parse_floats_from_line(rowz)
                        if len(vx) >= 3 and len(vy) >= 3 and len(vz) >= 3:
                            stress_gpa = np.array([[vx[-3], vx[-2], vx[-1]],
                                                   [vy[-3], vy[-2], vy[-1]],
                                                   [vz[-3], vz[-2], vz[-1]]], dtype=np.float32)
                    except Exception:
                        pass
                    continue
        # Assemble lattice
        if a_vec is None or b_vec is None or c_vec is None:
            lattice = np.eye(3, dtype=np.float32)
        else:
            lattice = np.array([a_vec, b_vec, c_vec], dtype=np.float32)
        # Compose atomic data
        properties = [
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
        ]
        atomic_properties: dict[str, np.ndarray] = {
            "species": np.array(species, dtype=np.str_),
            "pos": np.array(positions, dtype=np.float32),
        }
        if forces and len(forces) == len(positions):
            properties.append({"name": "forces", "type": "R", "count": 3})
            atomic_properties["forces"] = np.array(forces, dtype=np.float32)
        additional_fields: dict[str, object] = {
            "Config_type": "CP2K_1",
            "pbc": "T T T",
        }
        if energy_ev is not None:
            additional_fields["energy"] = float(energy_ev)
        if stress_gpa is not None:
            s = (stress_gpa * GPA_TO_EV_PER_ANG3).astype(np.float32)
            stress9 = np.array([s[0, 0], s[0, 1], s[0, 2],
                                s[1, 0], s[1, 1], s[1, 2],
                                s[2, 0], s[2, 1], s[2, 2]], dtype=np.float32)
            additional_fields["stress"] = stress9
        if len(positions) > 0:
            yield Structure(lattice=lattice,
                            atomic_properties=atomic_properties,
                            properties=properties,
                            additional_fields=additional_fields)
register_importer(Cp2kOutputImporter())
# n2p2 CFG/input.data importer
class N2p2CfgImporter:
    """Importer for n2p2 CFG datasets (input.data format)."""
    name = "n2p2_cfg"
    def matches(self, path: PathLike) -> bool:
        """Return ``True`` when ``path`` resembles an n2p2 CFG file."""
        candidate = as_path(path)
        if not candidate.is_file():
            return False
        base = candidate.name.lower()
        ext = candidate.suffix.lower()
        likely = (base.endswith('input.data') or ext in {'.data', '.cfg'})
        try:
            with candidate.open('r', encoding='utf8', errors='ignore') as f:
                head = f.read(4096)
            # Simple signature: blocks delimited by 'begin'/'end' and lines starting with atom/lattice
            sig = ("\nbegin\n" in head or head.strip().startswith("begin")) and (
                "\natom" in head or "\nlattice" in head)
            return sig or likely
        except Exception:
            return likely
    def iter_structures(self, path: PathLike, **kwargs):
        """Parse n2p2 CFG file (input.data) into Structure frames.
        Format reference: https://compphysvienna.github.io/n2p2/topics/cfg_file.html
        Block between 'begin' ... 'end'. Within a block:
          - lattice ax ay az (3 lines, optional)
          - atom x y z elem c n fx fy fz (repeat n times)
          - comment <text> (optional)
          - energy <E> (optional)
          - charge <Q> (optional)
        """
        candidate = as_path(path)

        cancel_event = kwargs.get("cancel_event")
        # Per request, input.data (n2p2 CFG) is always given in Bohr/Hartree.
        # Constants from n2p2 docs (pair_nnp):
        #   1 eV = 0.0367493254 Hartree => Hartree -> eV is 1 / 0.0367493254
        length_to_ang = 1.0 / 1.8897261328
        energy_to_ev = 1.0 / 0.0367493254
        force_to_ev_per_ang = energy_to_ev / length_to_ang
        def cancelled():
            """Return ``True`` if optional cancellation has been requested."""
            return cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set()
        # per-block accumulators
        block_idx = 0
        in_block = False
        lattice_vecs: list[list[float]] | None = None
        positions: list[list[float]] | None = None
        species: list[str] | None = None
        forces: list[list[float]] | None = None
        energy_val: float | None = None
        charge_val: float | None = None
        comment_txt: str | None = None
        def emit_if_ready():
            """Emit the current block as a Structure when all fields are populated."""
            nonlocal block_idx
            if positions is None or species is None or len(positions) == 0:
                return
            # lattice
            if lattice_vecs and len(lattice_vecs) == 3:
                lattice = (np.array(lattice_vecs, dtype=np.float32) * float(length_to_ang)).reshape(3, 3)
                pbc_txt = "T T T"
            else:
                lattice = np.eye(3, dtype=np.float32)
                pbc_txt = "F F F"
            props = [
                {"name": "species", "type": "S", "count": 1},
                {"name": "pos", "type": "R", "count": 3},
            ]
            atom_props: dict[str, np.ndarray] = {
                "species": np.array(species, dtype=np.str_),
                "pos": (np.array(positions, dtype=np.float32) * float(length_to_ang)),
            }
            if forces is not None and len(forces) == len(positions):
                props.append({"name": "forces", "type": "R", "count": 3})
                atom_props["forces"] = (np.array(forces, dtype=np.float32) * float(force_to_ev_per_ang))
            add = {
                "Config_type": (comment_txt or f"N2P2_CFG_{block_idx}"),
                "pbc": pbc_txt,
            }
            if energy_val is not None:
                add["energy"] = float(energy_val) * float(energy_to_ev)
            if charge_val is not None:
                add["charge"] = float(charge_val)
            yield Structure(lattice=lattice,
                            atomic_properties=atom_props,
                            properties=props,
                            additional_fields=add)
        # Streaming parse
        with candidate.open('r', encoding='utf8', errors='ignore') as f:
            for raw in f:
                if cancelled():
                    return
                line = raw.strip()
                if not line:
                    continue
                low = line.lower()
                if low == 'begin':
                    # start new block
                    in_block = True
                    block_idx += 1
                    lattice_vecs = []
                    positions = []
                    species = []
                    forces = []
                    energy_val = None
                    charge_val = None
                    comment_txt = None
                    continue
                if low == 'end':
                    if in_block:
                        # emit block
                        for st in emit_if_ready():
                            yield st
                    in_block = False
                    lattice_vecs = positions = species = forces = None
                    energy_val = charge_val = None
                    comment_txt = None
                    continue
                if not in_block:
                    # ignore content outside blocks
                    continue
                # Parse block lines
                if low.startswith('comment'):
                    # everything after first space
                    parts = raw.split(None, 1)
                    if len(parts) == 2:
                        comment_txt = parts[1].strip()
                    else:
                        comment_txt = ''
                    continue
                if low.startswith('lattice'):
                    toks = line.split()
                    # lattice ax ay az
                    if len(toks) >= 4:
                        try:
                            vec = [float(toks[1]), float(toks[2]), float(toks[3])]
                            if lattice_vecs is not None:
                                lattice_vecs.append(vec)
                        except Exception:
                            pass
                    continue
                if low.startswith('atom'):
                    # atom x y z elem c n fx fy fz
                    # Support possible extra whitespace in element column
                    toks = raw.split()
                    if len(toks) >= 10:
                        try:
                            x = float(toks[1]); y = float(toks[2]); z = float(toks[3])
                            elem = toks[4]
                            fx = float(toks[-3]); fy = float(toks[-2]); fz = float(toks[-1])
                            if positions is not None and species is not None and forces is not None:
                                positions.append([x, y, z])
                                species.append(elem)
                                forces.append([fx, fy, fz])
                        except Exception:
                            # Be tolerant: try minimal variant without forces (x y z elem)
                            try:
                                x = float(toks[1]); y = float(toks[2]); z = float(toks[3])
                                elem = toks[4]
                                if positions is not None and species is not None:
                                    positions.append([x, y, z])
                                    species.append(elem)
                            except Exception:
                                pass
                    continue
                if low.startswith('energy'):
                    # energy E
                    toks = line.split()
                    if len(toks) >= 2:
                        try:
                            energy_val = float(toks[1])
                        except Exception:
                            pass
                    continue
                if low.startswith('charge'):
                    toks = line.split()
                    if len(toks) >= 2:
                        try:
                            charge_val = float(toks[1])
                        except Exception:
                            pass
                    continue
        # No trailing 'end': emit last block if we were inside one
        if in_block:
            for st in emit_if_ready():
                yield st
register_importer(N2p2CfgImporter())
# ASE trajectory importer (uses ASE to read, converts to Structure)
class AseTrajectoryImporter:
    """Importer for ASE ``.traj`` trajectory files."""
    name = "ase_traj"
    def matches(self, path: PathLike) -> bool:
        """Return ``True`` when ``path`` is an ASE ``.traj`` file."""
        candidate = as_path(path)
        if not candidate.is_file():
            return False
        ext = candidate.suffix.lower()
        # Target ASE formats that are not already handled by dedicated importers
        return ext in {".traj"}
    def _ase_atoms_to_structure(self, atoms) -> Structure | None:
        """Convert an ASE ``Atoms`` object into a :class:`Structure`."""
        try:
            # Lattice/cell
            cell = getattr(atoms, 'cell', None)
            if cell is None:
                lattice = np.eye(3, dtype=np.float32)
            else:
                arr = np.array(cell.array if hasattr(cell, 'array') else cell, dtype=np.float32)
                if arr.size != 9:
                    lattice = np.eye(3, dtype=np.float32)
                else:
                    lattice = arr.reshape(3, 3)
            # Species and positions
            symbols = np.array(atoms.get_chemical_symbols(), dtype=np.str_)
            positions = np.array(atoms.get_positions(), dtype=np.float32)
            properties = [
                {"name": "species", "type": "S", "count": 1},
                {"name": "pos", "type": "R", "count": 3},
            ]
            atomic_props: dict[str, np.ndarray] = {
                "species": symbols,
                "pos": positions,
            }
            # Additional fields
            info = getattr(atoms, 'info', {}) or {}
            cfg = str(info.get('comment', info.get('Config_type', 'ASE_traj')))
            pbc_val = getattr(atoms, 'pbc', False)
            if isinstance(pbc_val, (list, tuple, np.ndarray)):
                pbc_text = " ".join(["T" if bool(x) else "F" for x in list(pbc_val)[:3]])
            else:
                pbc_text = "T T T" if bool(pbc_val) else "F F F"
            add = {
                "Config_type": cfg,
                "pbc": pbc_text,
            }
            calc_result=atoms.calc.results
            if "energy" in calc_result:
                add["energy"]=calc_result["energy"]
            # Optional forces from arrays to avoid calculator call
            forces = None
            try:
                if hasattr(atoms, 'arrays') and isinstance(atoms.arrays, dict):
                    if 'forces' in atoms.arrays:
                        forces = np.array(atoms.arrays['forces'], dtype=np.float32)
                if "forces" in calc_result:
                    forces = np.array(calc_result['forces'], dtype=np.float32)
            except Exception:
                forces = None
            if isinstance(forces, np.ndarray) and forces.shape == positions.shape:
                properties.append({"name": "forces", "type": "R", "count": 3})
                atomic_props["forces"] = forces
            # Optional stress/virial if present in info with common keys
            try:
                if 'stress' in calc_result:
                    s = np.array(calc_result['stress'], dtype=np.float32)
                    if s.size == 9:
                        add['stress'] = s.reshape(3, 3).reshape(-1)
                    elif s.size == 6:
                        # voigt -> symmetric
                        sxx, syy, szz, syz, sxz, sxy = s.tolist()
                        S = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]], dtype=np.float32)
                        add['stress'] = S.reshape(-1)
                if 'virial' in info:
                    v = np.array(info['virial'], dtype=np.float32)
                    if v.size == 9:
                        add['virial'] = v.reshape(-1)
            except Exception:
                pass
            return Structure(lattice=lattice, atomic_properties=atomic_props, properties=properties, additional_fields=add)
        except Exception:
            return None
    def iter_structures(self, path: PathLike, **kwargs):
        """Yield structures from ASE trajectory files."""
        candidate = as_path(path)
        cancel_event = kwargs.get("cancel_event")
        try:
            from ase.io import iread
        except Exception:
            return
        try:
            for atoms in iread(str(candidate), index=":"):
                if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                    return
                st = self._ase_atoms_to_structure(atoms)
                if st is not None:
                    yield st
        except Exception:
            return
register_importer(AseTrajectoryImporter())
def write_extxyz(file_path: str, structures: List[Structure]) -> str:
    """Write structures to an EXTXYZ file using Structure.write()."""
    with open(file_path, "w", encoding="utf8") as f:
        for s in structures:
            s.write(f)
    return file_path
