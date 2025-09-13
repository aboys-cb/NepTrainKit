#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Protocol, List

from loguru import logger

from NepTrainKit.core.structure import Structure
import numpy as np


class FormatImporter(Protocol):
    """Importer interface for converting various outputs into Structure objects."""

    name: str

    def matches(self, path: str) -> bool:
        """Return True if this importer can handle the given file/directory."""
        ...

    def iter_structures(self, path: str,**kwargs) -> Iterable[Structure]:
        """Yield Structure objects from the given path."""
        ...


_IMPORTERS: list[FormatImporter] = []


def register_importer(importer: FormatImporter):
    _IMPORTERS.append(importer)
    return importer

def is_parseable(path: str) -> bool:
    for imp in _IMPORTERS:
        try:
            if imp.matches(path):
                return True
        except Exception:
            pass
    return False

def import_structures(path: Path|str,**kwargs) -> List[Structure]:
    """Try all registered importers to load structures from path.

    Returns a list of Structure or an empty list if no importer matched.
    """
    if isinstance(path, Path):
        path = path.as_posix()
    for imp in _IMPORTERS:
        try:
            if imp.matches(path):
                return list(imp.iter_structures(path,**kwargs))
        except Exception:
            logger.debug(f"Importer {imp.__class__.__name__} failed on {path}")
            continue
    return []


# ----------- Built-in importers -----------



class ExtxyzImporter:
    name = "extxyz"

    def matches(self, path: str) -> bool:
        return os.path.isfile(path) and os.path.splitext(path)[1].lower() in {".xyz", ".extxyz"}

    def iter_structures(self, path: str,**kwargs):
        yield from Structure.iter_read_multiple(path,**kwargs)


register_importer(ExtxyzImporter())


# VASP XDATCAR importer


class XdatcarImporter:
    name = "vasp_xdatcar"

    def matches(self, path: str) -> bool:
        base = os.path.basename(path).lower()
        ext = os.path.splitext(path)[1].lower()
        return os.path.isfile(path) and (base == "xdatcar" or ext == ".xdatcar")

    def iter_structures(self, path: str, **kwargs):
        cancel_event = kwargs.get("cancel_event")
        """Parse VASP XDATCAR trajectory into Structure frames.

        Notes:
        - Supports variable cell per frame (XDATCAR-style headers before each config).
        - Coordinates are converted to Cartesian and stored under "pos".
        - Species are taken from header; if absent, falls back to dummy X1/X2...
        """

        def _is_number(s: str) -> bool:
            try:
                float(s)
                return True
            except Exception:
                return False

        with open(path, "r", encoding="utf8", errors="ignore") as f:
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
    name = "vasp_outcar"

    def matches(self, path: str) -> bool:
        base = os.path.basename(path).lower()
        ext = os.path.splitext(path)[1].lower()
        return os.path.isfile(path) and (base == "outcar" or ext == ".outcar")

    def iter_structures(self, path: str, cancel_event=None,**kwargs):
        def parse_floats(line: str) -> list[float]:
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
        pending_stress: np.ndarray | None = None  # eV/Å^3, 9 comps row-major
        pending_virial: np.ndarray | None = None  # eV, 9 comps row-major
        last_force_is_ml: bool | None = None
        frames: list[dict] = []

        # helpers for species mapping
        def finalize_species_list(n_atoms: int) -> np.ndarray:
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
        with open(path, "r", encoding="utf8", errors="ignore") as f:
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
                        # convert kB -> GPa -> eV/Å^3
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





# Skeleton for CP2K output importer (optional)


class Cp2kOutputImporter:
    name = "cp2k_output"

    def matches(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return os.path.isfile(path) and ext in {".out", ".log"}

    def iter_structures(self, path: str):
        # TODO: Implement CP2K output parsing
        # Typical markers:
        #  - "ATOMIC COORDINATES in angstrom" blocks
        #  - cell vectors from "CELL| Vector a/b/c"
        raise NotImplementedError("Cp2kOutputImporter.iter_structures not implemented")




def write_extxyz(file_path: str, structures: List[Structure]) -> str:
    """Write structures to an EXTXYZ file using Structure.write()."""
    with open(file_path, "w", encoding="utf8") as f:
        for s in structures:
            s.write(f)
    return file_path
