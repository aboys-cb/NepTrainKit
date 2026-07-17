#!/usr/bin/env python
"""Create a small random fcc alloy as LAMMPS dump plus optional extxyz."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


FCC_BASIS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    dtype=float,
)


def make_fcc_random_alloy(elements: list[str], size: int, lattice: float, seed: int) -> Atoms:
    rng = np.random.default_rng(seed)
    scaled = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                scaled.extend((FCC_BASIS + [i, j, k]) / size)
    scaled = np.asarray(scaled, dtype=float)
    symbols = rng.choice(elements, size=len(scaled)).tolist()
    cell = np.eye(3) * lattice * size
    atoms = Atoms(symbols=symbols, scaled_positions=scaled, cell=cell, pbc=True)
    atoms.info["Config_type"] = "fcc_random_alloy"
    return atoms


def write_lammps_dump(path: Path, atoms: Atoms, elements: list[str]) -> None:
    type_by_symbol = {symbol: index + 1 for index, symbol in enumerate(elements)}
    positions = atoms.get_positions()
    cell = atoms.cell.lengths()
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ITEM: TIMESTEP\n0\n")
        handle.write(f"ITEM: NUMBER OF ATOMS\n{len(atoms)}\n")
        handle.write("ITEM: BOX BOUNDS pp pp pp\n")
        for length in cell:
            handle.write(f"0.0 {length:.16g}\n")
        handle.write("ITEM: ATOMS id type x y z\n")
        for atom_id, (symbol, pos) in enumerate(zip(atoms.get_chemical_symbols(), positions), start=1):
            atom_type = type_by_symbol[symbol]
            handle.write(f"{atom_id} {atom_type} {pos[0]:.16g} {pos[1]:.16g} {pos[2]:.16g}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-output", required=True, type=Path)
    parser.add_argument("--extxyz-output", type=Path)
    parser.add_argument("--elements", default="Ni,Co,Cr")
    parser.add_argument("--size", type=int, default=4, help="fcc conventional cells per axis")
    parser.add_argument("--lattice", type=float, default=3.6)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    elements = [element.strip() for element in args.elements.split(",") if element.strip()]
    if not elements:
        raise ValueError("At least one element is required")

    atoms = make_fcc_random_alloy(elements, size=args.size, lattice=args.lattice, seed=args.seed)
    args.dump_output.parent.mkdir(parents=True, exist_ok=True)
    write_lammps_dump(args.dump_output, atoms, elements)
    if args.extxyz_output:
        args.extxyz_output.parent.mkdir(parents=True, exist_ok=True)
        write(args.extxyz_output, atoms, format="extxyz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
