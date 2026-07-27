#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Importer registry for converting simulation outputs into Structure objects."""
from __future__ import annotations
import traceback
from pathlib import Path
from typing import Iterable, Protocol, List
from loguru import logger
from NepTrainKit.paths import PathLike, as_path
from NepTrainKit.core.precision import get_storage_float_dtype
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
def _is_blank_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                if chunk.strip():
                    return False
    except OSError:
        return False
    return True
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
    if _is_blank_file(candidate):
        return []
    matched_errors: list[str] = []
    for imp in _IMPORTERS:
        try:
            if imp.matches(candidate):
                structures = list(imp.iter_structures(candidate, **kwargs))
                if structures:
                    return structures
                matched_errors.append(f"{imp.__class__.__name__} produced no structures")
        except Exception as exc:
            detail = traceback.format_exc()
            logger.error(f"Importer {imp.__class__.__name__} failed for {candidate}: {detail}")
            matched_errors.append(f"{imp.__class__.__name__}: {exc}")
            continue
    if matched_errors:
        raise ValueError(f"Failed to import structures from {candidate}: {'; '.join(matched_errors)}")
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
        - Supports standard XDATCAR files with one header and many configs.
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

        def read_header(f, title: str):
            while title.strip() == "":
                title = f.readline()
                if not title:
                    return None
            scale_line = f.readline()
            if not scale_line:
                return None
            try:
                scale = float(scale_line.split()[0])
            except Exception:
                return None
            latt = []
            for _ in range(3):
                line = f.readline()
                if not line:
                    return None
                parts = line.split()
                if len(parts) < 3:
                    return None
                try:
                    latt.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except Exception:
                    return None
            lattice = (scale * np.array(latt, dtype=get_storage_float_dtype())).reshape(3, 3)

            line = f.readline()
            if not line:
                return None
            tokens = line.split()
            if all(_is_number(t) for t in tokens):
                counts = [int(round(float(t))) for t in tokens]
                sym_from_kw = kwargs.get("species", None)
                symbols = list(sym_from_kw) if sym_from_kw is not None else [f"X{i+1}" for i in range(len(counts))]
                if len(symbols) != len(counts):
                    raise ValueError("Provided species length does not match counts in XDATCAR")
            else:
                symbols = tokens
                line2 = f.readline()
                if not line2:
                    return None
                counts = [int(round(float(x))) for x in line2.split()]
                if len(counts) != len(symbols):
                    return None
            species_list = np.concatenate([
                np.array([sym] * cnt, dtype=np.str_)
                for sym, cnt in zip(symbols, counts)
            ])
            return lattice, int(sum(counts)), species_list

        with candidate.open("r", encoding="utf8", errors="ignore") as f:
            header = read_header(f, f.readline())
            if header is None:
                return
            frame_index = 0
            while True:
                if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                    return
                lattice, n_atoms, species_list = header
                mode_line = f.readline()
                if not mode_line:
                    return
                if not mode_line.strip():
                    continue
                mode_l = mode_line.strip().lower()
                if "direct" not in mode_l and "cart" not in mode_l:
                    header = read_header(f, mode_line)
                    if header is None:
                        return
                    continue
                use_direct = ("direct" in mode_l)
                # Read n_atoms coordinate lines
                coords = np.zeros((n_atoms, 3), dtype=get_storage_float_dtype())
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
                # Convert to Cartesian if in direct (fractional) coords
                if use_direct:
                    positions = coords @ lattice
                else:
                    positions = coords.astype(get_storage_float_dtype(), copy=False)
                properties = [
                    {"name": "species", "type": "S", "count": 1},
                    {"name": "pos", "type": "R", "count": 3},
                ]
                atomic_properties = {
                    "species": species_list,
                    "pos": positions,
                }
                frame_index += 1
                additional_fields = {
                    "Config_type": f"XDATCAR_{frame_index}",
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
        name = candidate.name.lower()
        return candidate.is_file() and (name == "outcar" or name.startswith("outcar ") or ext == ".outcar")
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
        position_only_frames: list[dict] = []
        # helpers for species mapping
        def clean_element_token(token: str) -> str | None:
            token = token.split("_", 1)[0].strip()
            letters = ""
            for ch in token:
                if not ch.isalpha():
                    break
                letters += ch
            if not letters:
                return None
            sym = letters[0].upper() + letters[1:].lower()
            return sym if sym in atomic_numbers else None

        def add_species_symbol(sym: str | None) -> None:
            nonlocal species_by_type
            if not sym:
                return
            if species_by_type is None:
                species_by_type = []
            # avoid adjacent duplicates from repeated POTCAR echoes
            if not species_by_type or species_by_type[-1] != sym:
                species_by_type.append(sym)

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

        def make_position_frame(positions: list[list[float]], lattice: np.ndarray, *, include_forces: bool = False, forces: list[list[float]] | None = None) -> dict | None:
            n_atoms = len(positions)
            if n_atoms == 0:
                return None
            species = finalize_species_list(n_atoms)
            props = [
                {"name": "species", "type": "S", "count": 1},
                {"name": "pos", "type": "R", "count": 3},
            ]
            atom_props = {
                "species": species,
                "pos": np.array(positions, dtype=get_storage_float_dtype()),
            }
            if include_forces and forces is not None:
                props.append({"name": "forces", "type": "R", "count": 3})
                atom_props["forces"] = np.array(forces, dtype=get_storage_float_dtype())
            return {
                "lattice": lattice.copy(),
                "properties": props,
                "atomic_properties": atom_props,
                "additional_fields": {
                    "Config_type": "OUTCAR",
                    "pbc": "T T T",
                },
            }
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
                if lt.startswith("POTCAR:"):
                    try:
                        for token in lt.split(":", 1)[1].split():
                            sym = clean_element_token(token)
                            if sym is not None:
                                add_species_symbol(sym)
                                break
                    except Exception:
                        pass
                    continue
                if lt.startswith("VRHFIN") and ":" in lt and "=" in lt:
                    try:
                        add_species_symbol(clean_element_token(lt.split("=")[1].split(":")[0]))
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
                            cand = clean_element_token(t)
                            if cand is not None:
                                break
                        if cand is not None:
                            add_species_symbol(cand)
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
                                                   [c[0], c[1], c[2]]], dtype=get_storage_float_dtype())
                        pending_lattice = latest_lattice.copy()
                    except Exception:
                        latest_lattice = latest_lattice
                    continue
                if line.strip().startswith("position of ions in"):
                    if counts_by_type is None:
                        continue
                    n_atoms = int(sum(counts_by_type))
                    coords: list[list[float]] = []
                    for _ in range(n_atoms):
                        l2 = next(f, "")
                        nums = parse_floats(l2)
                        if len(nums) < 3:
                            coords = []
                            break
                        coords.append(nums[:3])
                    if not coords:
                        continue
                    use_lattice = pending_lattice if pending_lattice is not None else latest_lattice
                    if use_lattice is None:
                        use_lattice = np.eye(3, dtype=get_storage_float_dtype())
                    if "fractional" in line.lower() or "direct" in line.lower():
                        positions = (np.array(coords, dtype=get_storage_float_dtype()) @ use_lattice).tolist()
                    else:
                        positions = coords
                    frame = make_position_frame(positions, use_lattice)
                    if frame is not None:
                        position_only_frames.append(frame)
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
                                          [a3[0], a3[1], a3[2]]], dtype=get_storage_float_dtype())
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
                                           [xz, yz, zz]], dtype=get_storage_float_dtype())
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
                    if pending_lattice is not None:
                        use_lattice = pending_lattice
                    else:
                        use_lattice = latest_lattice if latest_lattice is not None else np.eye(3, dtype=get_storage_float_dtype())
                    # consume pending tensors (align to current block kind)
                    stress_next = pending_stress
                    virial_next = pending_virial
                    pending_stress = None
                    pending_virial = None
                    frame = make_position_frame(positions, use_lattice, include_forces=True, forces=forces)
                    if frame is None:
                        continue
                    # Only keep DFT frames for downstream NEP, skip ML frames
                    if is_ml_block:
                        continue
                    frame.update({
                        **({"virial": virial_next} if virial_next is not None else {}),
                        **({"stress": stress_next} if stress_next is not None else {}),
                    })
                    frames.append(frame)
        # Emit frames as Structure objects
        if not frames:
            frames = position_only_frames[-1:]
        for i, fr in enumerate(frames):
            add = fr["additional_fields"].copy()
            if "energy" in fr:
                add["energy"] = fr["energy"]
            if "virial" in fr or "stress" in fr:
                if "virial" in fr:
                    v = fr["virial"].reshape(3, 3)
                    virial9 = np.array([v[0,0], v[0,1], v[0,2], v[1,0], v[1,1], v[1,2], v[2,0], v[2,1], v[2,2]], dtype=get_storage_float_dtype())
                    add["virial"] = virial9
                    # derive stress from virial
                    try:
                        vol = float(np.abs(np.linalg.det(fr["lattice"])) )
                        s = (-v / vol)
                        stress9 = np.array([s[0,0], s[0,1], s[0,2], s[1,0], s[1,1], s[1,2], s[2,0], s[2,1], s[2,2]], dtype=get_storage_float_dtype())
                        add["stress"] = stress9
                    except Exception:
                        pass
                else:
                    s = fr["stress"].reshape(3, 3)
                    stress9 = np.array([s[0,0], s[0,1], s[0,2], s[1,0], s[1,1], s[1,2], s[2,0], s[2,1], s[2,2]], dtype=get_storage_float_dtype())
                    add["stress"] = stress9
                    try:
                        vol = float(np.abs(np.linalg.det(fr["lattice"])) )
                        v = (-s * vol)
                        virial9 = np.array([v[0,0], v[0,1], v[0,2], v[1,0], v[1,1], v[1,2], v[2,0], v[2,1], v[2,2]], dtype=get_storage_float_dtype())
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
                    raise ValueError("Truncated LAMMPS dump: missing timestep value")
                try:
                    timestep = int(ts_line.strip().split()[0])
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid LAMMPS timestep value: {ts_line.strip()!r}"
                    ) from exc
                # NUMBER OF ATOMS
                hdr = f.readline()  # ITEM: NUMBER OF ATOMS
                if not hdr or not hdr.strip().startswith("ITEM: NUMBER OF ATOMS"):
                    raise ValueError(
                        f"LAMMPS frame {timestep} is missing 'ITEM: NUMBER OF ATOMS'"
                    )
                nat_line = f.readline()
                if not nat_line:
                    raise ValueError(
                        f"LAMMPS frame {timestep} is missing the atom count"
                    )
                try:
                    n_atoms = int(nat_line.strip().split()[0])
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid LAMMPS atom count in frame {timestep}: "
                        f"{nat_line.strip()!r}"
                    ) from exc
                if n_atoms <= 0:
                    raise ValueError(
                        f"LAMMPS frame {timestep} has invalid atom count {n_atoms}"
                    )
                # BOX BOUNDS
                bounds_hdr = f.readline()
                if not bounds_hdr or not bounds_hdr.strip().startswith("ITEM: BOX BOUNDS"):
                    raise ValueError(
                        f"LAMMPS frame {timestep} is missing 'ITEM: BOX BOUNDS'"
                    )
                bounds_tokens = bounds_hdr.strip().split()
                tilt_flags = {t for t in bounds_tokens if t in {"xy", "xz", "yz"}}
                boundary_tokens = [
                    token
                    for token in bounds_tokens[3:]
                    if len(token) == 2
                    and all(char in {"p", "f", "s", "m"} for char in token.lower())
                ]
                if len(boundary_tokens) != 3:
                    raise ValueError(
                        f"LAMMPS frame {timestep} does not declare three BOX BOUNDS "
                        "boundary flags"
                    )
                pbc = [token.lower() == "pp" for token in boundary_tokens]
                def _read_bounds_line():
                    """Read a bounds line from the dump header and return floats."""
                    l = f.readline()
                    return [float(x) for x in l.strip().split()] if l else []
                b1 = _read_bounds_line(); b2 = _read_bounds_line(); b3 = _read_bounds_line()
                if not b1 or not b2 or not b3:
                    raise ValueError(
                        f"LAMMPS frame {timestep} has truncated BOX BOUNDS"
                    )
                if tilt_flags:
                    # triclinic: xlo xhi xy; ylo yhi xz; zlo zhi yz
                    if len(b1) < 3 or len(b2) < 3 or len(b3) < 3:
                        raise ValueError(
                            f"LAMMPS frame {timestep} has incomplete triclinic BOX BOUNDS"
                        )
                    xlo_bound, xhi_bound, xy = b1[0], b1[1], b1[2]
                    ylo_bound, yhi_bound, xz = b2[0], b2[1], b2[2]
                    zlo, zhi, yz = b3[0], b3[1], b3[2]
                    xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
                    xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
                    ylo = ylo_bound - min(0.0, yz)
                    yhi = yhi_bound - max(0.0, yz)
                else:
                    if len(b1) < 2 or len(b2) < 2 or len(b3) < 2:
                        raise ValueError(
                            f"LAMMPS frame {timestep} has incomplete orthogonal BOX BOUNDS"
                        )
                    xlo, xhi = b1[0], b1[1]
                    ylo, yhi = b2[0], b2[1]
                    zlo, zhi = b3[0], b3[1]
                    xy = xz = yz = 0.0
                lx = float(xhi - xlo)
                ly = float(yhi - ylo)
                lz = float(zhi - zlo)
                cell_values = np.asarray(
                    [lx, ly, lz, xy, xz, yz, xlo, ylo, zlo],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(cell_values)) or min(lx, ly, lz) <= 0.0:
                    raise ValueError(
                        f"LAMMPS frame {timestep} has invalid BOX BOUNDS"
                    )
                a = np.array([lx, 0.0, 0.0], dtype=get_storage_float_dtype())
                b = np.array([xy, ly, 0.0], dtype=get_storage_float_dtype())
                c = np.array([xz, yz, lz], dtype=get_storage_float_dtype())
                lattice = np.vstack([a, b, c]).reshape(3, 3)
                # ATOMS header
                atoms_hdr = f.readline()
                if not atoms_hdr or not atoms_hdr.strip().startswith("ITEM: ATOMS"):
                    raise ValueError(
                        f"LAMMPS frame {timestep} is missing 'ITEM: ATOMS'"
                    )
                cols = atoms_hdr.strip().split()[2:]
                idx = {name: i for i, name in enumerate(cols)}
                has_scaled = all(k in idx for k in ("xs", "ys", "zs"))
                has_cart = all(k in idx for k in ("x", "y", "z"))
                has_unwrapped = all(k in idx for k in ("xu", "yu", "zu"))
                if not (has_scaled or has_cart or has_unwrapped):
                    raise ValueError(
                        f"LAMMPS frame {timestep} must contain one complete coordinate "
                        "triplet: xs/ys/zs, x/y/z, or xu/yu/zu"
                    )
                has_forces = all(k in idx for k in ("fx", "fy", "fz"))
                spin_columns = tuple(f"c_spin[{column}]" for column in range(1, 5))
                has_spin = all(column in idx for column in spin_columns)
                species_col = "element" if "element" in idx else ("type" if "type" in idx else None)
                if species_col is None:
                    raise ValueError(
                        f"LAMMPS frame {timestep} must contain an element or type column"
                    )
                positions = np.zeros((n_atoms, 3), dtype=get_storage_float_dtype())
                forces = np.zeros((n_atoms, 3), dtype=get_storage_float_dtype()) if has_forces else None
                spins = np.zeros((n_atoms, 3), dtype=get_storage_float_dtype()) if has_spin else None
                species_list: list[str] = []
                types_buffer = np.zeros((n_atoms,), dtype=np.int32) if species_col == "type" else None
                for i in range(n_atoms):
                    if cancelled():
                        return
                    l = f.readline()
                    if not l:
                        raise ValueError(
                            f"LAMMPS frame {timestep} is truncated: expected "
                            f"{n_atoms} atom rows, found {i}"
                        )
                    parts = l.split()
                    if len(parts) < len(cols):
                        raise ValueError(
                            f"LAMMPS frame {timestep}, atom row {i + 1} has "
                            f"{len(parts)} columns; expected {len(cols)}"
                        )
                    # species
                    val = parts[idx[species_col]]
                    if species_col == "type":
                        try:
                            tnum = int(val)
                        except ValueError as exc:
                            raise ValueError(
                                f"LAMMPS frame {timestep}, atom row {i + 1} has "
                                f"invalid integer type {val!r}"
                            ) from exc
                        if tnum <= 0:
                            raise ValueError(
                                f"LAMMPS frame {timestep}, atom row {i + 1} has "
                                f"invalid atom type {tnum}"
                            )
                        if types_buffer is not None:
                            types_buffer[i] = tnum
                        # Placeholder; resolve after reading all atoms.
                        species_list.append("")
                    else:
                        species_list.append(val)
                    # fractional
                    if has_scaled:
                        fx = float(parts[idx["xs"]]); fy = float(parts[idx["ys"]]); fz = float(parts[idx["zs"]])
                        pos = fx * a + fy * b + fz * c
                    elif has_cart:
                        x = float(parts[idx["x"]]); y = float(parts[idx["y"]]); z = float(parts[idx["z"]])
                        pos = np.asarray(
                            [x - xlo, y - ylo, z - zlo],
                            dtype=get_storage_float_dtype(),
                        )
                    else:
                        x = float(parts[idx["xu"]]); y = float(parts[idx["yu"]]); z = float(parts[idx["zu"]])
                        pos = np.asarray(
                            [x - xlo, y - ylo, z - zlo],
                            dtype=get_storage_float_dtype(),
                        )
                    if not np.all(np.isfinite(pos)):
                        raise ValueError(
                            f"LAMMPS frame {timestep}, atom row {i + 1} has "
                            "non-finite coordinates"
                        )
                    positions[i, :] = pos
                    if has_forces and forces is not None:
                        forces[i, 0] = float(parts[idx["fx"]])
                        forces[i, 1] = float(parts[idx["fy"]])
                        forces[i, 2] = float(parts[idx["fz"]])
                    if has_spin and spins is not None:
                        magnitude = float(parts[idx["c_spin[1]"]])
                        spins[i, 0] = magnitude * float(parts[idx["c_spin[2]"]])
                        spins[i, 1] = magnitude * float(parts[idx["c_spin[3]"]])
                        spins[i, 2] = magnitude * float(parts[idx["c_spin[4]"]])
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
                        except Exception as exc:
                            raise ValueError(
                                f"Failed to resolve LAMMPS atom types {missing}: {exc}"
                            ) from exc
                    unresolved = sorted(
                        {
                            int(t)
                            for t in types_buffer.tolist()
                            if int(t) not in type_to_elem
                        }
                    )
                    if unresolved:
                        raise ValueError(
                            "LAMMPS numeric atom types require an explicit element "
                            f"mapping; unresolved types: {unresolved}"
                        )
                    for i in range(n_atoms):
                        t = int(types_buffer[i])
                        species_list[i] = type_to_elem[t]
                invalid_species = sorted(
                    {symbol for symbol in species_list if symbol not in atomic_numbers}
                )
                if invalid_species:
                    raise ValueError(
                        f"LAMMPS frame {timestep} contains invalid element symbols: "
                        f"{invalid_species}"
                    )
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
                if has_spin and spins is not None:
                    properties.append({"name": "spin", "type": "R", "count": 3})
                    atom_props["spin"] = spins
                additional_fields = {
                    "Config_type": f"LAMMPS_{timestep}",
                    "pbc": " ".join("T" if periodic else "F" for periodic in pbc),
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
        coordinate_blocks = 0
        force_blocks = 0
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
                    coordinate_blocks += 1
                    if coordinate_blocks > 1:
                        raise ValueError(
                            "CP2K output contains multiple coordinate blocks. "
                            "Import a single-point output or convert the CP2K trajectory to EXTXYZ."
                        )
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
                    force_blocks += 1
                    if force_blocks > 1:
                        raise ValueError(
                            "CP2K output contains multiple force blocks. "
                            "Import a single-point output or convert the CP2K trajectory to EXTXYZ."
                        )
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
                                                   [vz[-3], vz[-2], vz[-1]]], dtype=get_storage_float_dtype())
                    except Exception:
                        pass
                    continue
        # Assemble lattice
        if a_vec is None or b_vec is None or c_vec is None:
            raise ValueError(
                "CP2K output is missing complete CELL vectors; "
                "NepTrainKit will not invent a unit cell."
            )
        lattice = np.array([a_vec, b_vec, c_vec], dtype=get_storage_float_dtype())
        if not np.all(np.isfinite(lattice)) or abs(float(np.linalg.det(lattice))) <= 1.0e-12:
            raise ValueError("CP2K output contains an invalid or singular cell.")
        if not positions:
            raise ValueError("CP2K output contains no atomic coordinate block.")
        if forces and len(forces) != len(positions):
            raise ValueError(
                "CP2K force count does not match the imported coordinate count."
            )
        # Compose atomic data
        properties = [
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
        ]
        atomic_properties: dict[str, np.ndarray] = {
            "species": np.array(species, dtype=np.str_),
            "pos": np.array(positions, dtype=get_storage_float_dtype()),
        }
        if forces and len(forces) == len(positions):
            properties.append({"name": "forces", "type": "R", "count": 3})
            atomic_properties["forces"] = np.array(forces, dtype=get_storage_float_dtype())
        additional_fields: dict[str, object] = {
            "Config_type": "CP2K_1",
            "pbc": "T T T",
        }
        if energy_ev is not None:
            additional_fields["energy"] = float(energy_ev)
        if stress_gpa is not None:
            s = (stress_gpa * GPA_TO_EV_PER_ANG3).astype(get_storage_float_dtype(), copy=False)
            stress9 = np.array([s[0, 0], s[0, 1], s[0, 2],
                                s[1, 0], s[1, 1], s[1, 2],
                                s[2, 0], s[2, 1], s[2, 2]], dtype=get_storage_float_dtype())
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
        atom_charges: list[float] | None = None
        atom_energies: list[float] | None = None
        energy_val: float | None = None
        charge_val: float | None = None
        comment_txt: str | None = None
        def emit_if_ready():
            """Emit the current block as a Structure when all fields are populated."""
            nonlocal block_idx
            if (
                positions is None
                or species is None
                or forces is None
                or atom_charges is None
                or atom_energies is None
                or len(positions) == 0
            ):
                raise ValueError(f"n2p2 block {block_idx} contains no complete atom rows.")
            if not (
                len(positions)
                == len(species)
                == len(forces)
                == len(atom_charges)
                == len(atom_energies)
            ):
                raise ValueError(f"n2p2 block {block_idx} has inconsistent per-atom fields.")
            # lattice
            if lattice_vecs and len(lattice_vecs) != 3:
                raise ValueError(
                    f"n2p2 block {block_idx} must contain zero or three lattice rows."
                )
            if lattice_vecs:
                lattice = (np.array(lattice_vecs, dtype=get_storage_float_dtype()) * float(length_to_ang)).reshape(3, 3)
                pbc_txt = "T T T"
            else:
                lattice = np.eye(3, dtype=get_storage_float_dtype())
                pbc_txt = "F F F"
            props = [
                {"name": "species", "type": "S", "count": 1},
                {"name": "pos", "type": "R", "count": 3},
            ]
            atom_props: dict[str, np.ndarray] = {
                "species": np.array(species, dtype=np.str_),
                "pos": (np.array(positions, dtype=get_storage_float_dtype()) * float(length_to_ang)),
            }
            if forces is not None and len(forces) == len(positions):
                props.append({"name": "forces", "type": "R", "count": 3})
                atom_props["forces"] = (np.array(forces, dtype=get_storage_float_dtype()) * float(force_to_ev_per_ang))
            props.append({"name": "charge", "type": "R", "count": 1})
            atom_props["charge"] = np.asarray(
                atom_charges,
                dtype=get_storage_float_dtype(),
            )
            props.append({"name": "atomic_energy", "type": "R", "count": 1})
            atom_props["atomic_energy"] = (
                np.asarray(atom_energies, dtype=get_storage_float_dtype())
                * float(energy_to_ev)
            )
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
        with candidate.open('r', encoding='utf8', errors='strict') as f:
            for line_number, raw in enumerate(f, start=1):
                if cancelled():
                    return
                line = raw.strip()
                if not line:
                    continue
                low = line.lower()
                if low == 'begin':
                    # start new block
                    if in_block:
                        raise ValueError(
                            f"n2p2 line {line_number} starts a new block before the previous block ended."
                        )
                    in_block = True
                    block_idx += 1
                    lattice_vecs = []
                    positions = []
                    species = []
                    forces = []
                    atom_charges = []
                    atom_energies = []
                    energy_val = None
                    charge_val = None
                    comment_txt = None
                    continue
                if low == 'end':
                    if not in_block:
                        raise ValueError(
                            f"n2p2 line {line_number} ends a block that was not started."
                        )
                    for st in emit_if_ready():
                        yield st
                    in_block = False
                    lattice_vecs = positions = species = forces = None
                    atom_charges = atom_energies = None
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
                    if len(toks) < 4:
                        raise ValueError(
                            f"n2p2 line {line_number} has an incomplete lattice row."
                        )
                    try:
                        vec = [float(toks[1]), float(toks[2]), float(toks[3])]
                    except ValueError as exc:
                        raise ValueError(
                            f"n2p2 line {line_number} has a non-numeric lattice row."
                        ) from exc
                    if not np.all(np.isfinite(vec)):
                        raise ValueError(
                            f"n2p2 line {line_number} has a non-finite lattice row."
                        )
                    if lattice_vecs is not None:
                        lattice_vecs.append(vec)
                    continue
                if low.startswith('atom'):
                    # atom x y z elem c n fx fy fz
                    toks = raw.split()
                    if len(toks) < 10:
                        raise ValueError(
                            f"n2p2 line {line_number} has {len(toks)} atom columns; expected 10."
                        )
                    try:
                        x, y, z = (float(toks[1]), float(toks[2]), float(toks[3]))
                        elem = toks[4]
                        atom_charge = float(toks[5])
                        atom_energy = float(toks[6])
                        fx, fy, fz = (
                            float(toks[-3]),
                            float(toks[-2]),
                            float(toks[-1]),
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"n2p2 line {line_number} has a non-numeric atom field."
                        ) from exc
                    numeric_values = [x, y, z, atom_charge, atom_energy, fx, fy, fz]
                    if not np.all(np.isfinite(numeric_values)):
                        raise ValueError(
                            f"n2p2 line {line_number} has a non-finite atom field."
                        )
                    if elem not in atomic_numbers:
                        raise ValueError(
                            f"n2p2 line {line_number} has unsupported element {elem!r}."
                        )
                    if (
                        positions is not None
                        and species is not None
                        and forces is not None
                        and atom_charges is not None
                        and atom_energies is not None
                    ):
                        positions.append([x, y, z])
                        species.append(elem)
                        forces.append([fx, fy, fz])
                        atom_charges.append(atom_charge)
                        atom_energies.append(atom_energy)
                    continue
                if low.startswith('energy'):
                    # energy E
                    toks = line.split()
                    if len(toks) < 2:
                        raise ValueError(f"n2p2 line {line_number} has no energy value.")
                    try:
                        energy_val = float(toks[1])
                    except ValueError as exc:
                        raise ValueError(
                            f"n2p2 line {line_number} has a non-numeric energy."
                        ) from exc
                    if not np.isfinite(energy_val):
                        raise ValueError(
                            f"n2p2 line {line_number} has a non-finite energy."
                        )
                    continue
                if low.startswith('charge'):
                    toks = line.split()
                    if len(toks) < 2:
                        raise ValueError(f"n2p2 line {line_number} has no charge value.")
                    try:
                        charge_val = float(toks[1])
                    except ValueError as exc:
                        raise ValueError(
                            f"n2p2 line {line_number} has a non-numeric charge."
                        ) from exc
                    if not np.isfinite(charge_val):
                        raise ValueError(
                            f"n2p2 line {line_number} has a non-finite charge."
                        )
                    continue
        if in_block:
            raise ValueError(f"n2p2 block {block_idx} is missing its closing 'end'.")
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
    @staticmethod
    def _ase_atoms_to_structure(atoms) -> Structure:
        """Convert an ASE ``Atoms`` object into a :class:`Structure`."""
        float_dtype = get_storage_float_dtype()
        lattice = np.asarray(atoms.cell.array, dtype=float_dtype)
        if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
            raise ValueError("ASE trajectory cell must be a finite 3 x 3 matrix.")

        symbols = np.asarray(atoms.get_chemical_symbols(), dtype=np.str_)
        positions = np.asarray(atoms.get_positions(), dtype=float_dtype)
        if positions.shape != (len(symbols), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("ASE trajectory positions must be finite N x 3 values.")

        properties = [
            {"name": "species", "type": "S", "count": 1},
            {"name": "pos", "type": "R", "count": 3},
        ]
        atomic_props: dict[str, np.ndarray] = {
            "species": symbols,
            "pos": positions,
        }

        def add_atomic_property(name: str, value) -> None:
            array = np.asarray(value)
            if array.ndim not in {1, 2} or array.shape[0] != len(symbols):
                raise ValueError(
                    f"ASE trajectory per-atom array {name!r} must have shape N or N x M."
                )
            count = 1 if array.ndim == 1 else array.shape[1]
            if count < 1:
                raise ValueError(f"ASE trajectory per-atom array {name!r} is empty.")
            if array.dtype.kind == "b":
                property_type = "L"
                converted = np.asarray(array, dtype=np.bool_)
            elif array.dtype.kind in {"i", "u"}:
                property_type = "I"
                converted = np.asarray(array, dtype=np.int32)
            elif array.dtype.kind == "f":
                property_type = "R"
                converted = np.asarray(array, dtype=float_dtype)
                if not np.all(np.isfinite(converted)):
                    raise ValueError(
                        f"ASE trajectory per-atom array {name!r} contains non-finite values."
                    )
            elif array.dtype.kind in {"S", "U"}:
                property_type = "S"
                converted = np.asarray(array, dtype=np.str_)
            else:
                raise ValueError(
                    f"ASE trajectory per-atom array {name!r} has unsupported dtype {array.dtype}."
                )
            atomic_props[name] = converted
            descriptor = {"name": name, "type": property_type, "count": count}
            for index, existing in enumerate(properties):
                if existing["name"] == name:
                    properties[index] = descriptor
                    break
            else:
                properties.append(descriptor)

        arrays = getattr(atoms, "arrays", {}) or {}
        for name, value in arrays.items():
            if name in {"numbers", "positions", "species", "pos"}:
                continue
            add_atomic_property(name, value)

        info = dict(getattr(atoms, "info", {}) or {})
        config_type = str(info.pop("Config_type", info.pop("comment", "ASE_traj")))
        info.pop("comment", None)
        pbc = np.asarray(atoms.pbc, dtype=np.bool_).reshape(-1)
        if pbc.size != 3:
            raise ValueError("ASE trajectory PBC must contain exactly three logical values.")
        additional_fields = dict(info)
        additional_fields["Config_type"] = config_type
        additional_fields["pbc"] = " ".join("T" if value else "F" for value in pbc)

        calculator = getattr(atoms, "calc", None)
        calc_results = dict(getattr(calculator, "results", {}) or {})
        if "energy" in calc_results:
            additional_fields["energy"] = calc_results["energy"]
        elif "free_energy" in calc_results and "energy" not in additional_fields:
            additional_fields["energy"] = calc_results["free_energy"]
        if "forces" in calc_results:
            add_atomic_property("forces", calc_results["forces"])
        for name in ("charges", "magmoms"):
            if name in calc_results:
                add_atomic_property(name, calc_results[name])

        for tensor_name in ("stress", "virial"):
            if tensor_name not in calc_results:
                continue
            tensor = np.asarray(calc_results[tensor_name], dtype=float_dtype)
            if tensor_name == "stress" and tensor.size == 6:
                sxx, syy, szz, syz, sxz, sxy = tensor.tolist()
                tensor = np.asarray(
                    [
                        [sxx, sxy, sxz],
                        [sxy, syy, syz],
                        [sxz, syz, szz],
                    ],
                    dtype=float_dtype,
                )
            if tensor.size != 9 or not np.all(np.isfinite(tensor)):
                raise ValueError(
                    f"ASE trajectory {tensor_name} must contain nine finite tensor values."
                )
            additional_fields[tensor_name] = tensor.reshape(-1)

        for tensor_name in ("stress", "virial"):
            if tensor_name not in additional_fields:
                continue
            tensor = np.asarray(additional_fields[tensor_name], dtype=float_dtype)
            if tensor.size != 9 or not np.all(np.isfinite(tensor)):
                raise ValueError(
                    f"ASE trajectory {tensor_name} must contain nine finite tensor values."
                )
            additional_fields[tensor_name] = tensor.reshape(-1)

        return Structure(
            lattice=lattice,
            atomic_properties=atomic_props,
            properties=properties,
            additional_fields=additional_fields,
        )
    def iter_structures(self, path: PathLike, **kwargs):
        """Yield structures from ASE trajectory files."""
        candidate = as_path(path)
        cancel_event = kwargs.get("cancel_event")
        from ase.io import iread

        try:
            for atoms in iread(str(candidate), index=":"):
                if cancel_event is not None and getattr(cancel_event, "is_set", None) and cancel_event.is_set():
                    return
                yield self._ase_atoms_to_structure(atoms)
        except Exception as exc:
            raise ValueError(f"Failed to read ASE trajectory {candidate}: {exc}") from exc
register_importer(AseTrajectoryImporter())


def ase_atoms_to_structure(atoms) -> Structure:
    """Convert an in-memory ASE ``Atoms`` object without a file round trip."""
    return AseTrajectoryImporter._ase_atoms_to_structure(atoms)


def write_extxyz(file_path: str, structures: List[Structure]) -> str:
    """Write structures to an EXTXYZ file using Structure.write()."""
    with open(file_path, "w", encoding="utf8") as f:
        for s in structures:
            s.write(f)
    return file_path
