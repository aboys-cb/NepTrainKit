#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Protocol, List

from loguru import logger

from NepTrainKit.core.structure import Structure


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


# Skeletons for VASP/CP2K importers (to be implemented by user as needed)


class VaspOutcarImporter:
    name = "vasp_outcar"

    def matches(self, path: str) -> bool:
        base = os.path.basename(path).lower()
        return os.path.isfile(path) and base == "outcar"

    def iter_structures(self, path: str):
        # TODO: Implement OUTCAR parsing (multiple frames with lattice + positions)
        # Suggested approach:
        #  - Scan for lattice blocks and POSITION/TOTAL-FORCE sections per step
        #  - Build Structure for each step with cell, positions, species (from POTCAR or POSCAR context)
        #  - Attach forces if available
        raise NotImplementedError("VaspOutcarImporter.iter_structures not implemented")


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


# To enable, uncomment registrations and implement parsers
# register_importer(VaspOutcarImporter())
# register_importer(Cp2kOutputImporter())


def write_extxyz(file_path: str, structures: List[Structure]) -> str:
    """Write structures to an EXTXYZ file using Structure.write()."""
    with open(file_path, "w", encoding="utf8") as f:
        for s in structures:
            s.write(f)
    return file_path

