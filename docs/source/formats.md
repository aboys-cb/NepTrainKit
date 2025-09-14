# Supported Structure/Result Formats

NepTrainKit can load and convert multiple common output formats into its internal `Structure` representation for analysis and plotting. This section summarizes what is supported and any caveats.

## NEP datasets (native)

- `.xyz` / `.extxyz` (train/test)
  - Loader: native EXTXYZ importer
  - Reads per-frame lattice (if present), species and Cartesian positions
  - Optional atomic properties such as forces will be used when available

- NEP model side files
  - `nep.txt`: model descriptor (used to pick result type)
  - `energy_*.out`, `force_*.out`, `virial_*.out`, `stress_*.out` and `descriptor*.out`: results loaded by NEP loaders

## VASP

- `OUTCAR`
  - Loader: VASP OUTCAR importer (streaming)
  - Reads per-step lattice, positions, TOTAL-FORCE blocks (DFT), stress in kB (converted to eV/Å^3)
  - Computes virial from stress (or uses printed virial if present)
  - Ignores ML POSITION blocks by default to avoid mismatching with TOTEN

- `XDATCAR`
  - Loader: XDATCAR importer (streaming)
  - Reads varying cell per frame, fractional coordinates (converted to Cartesian), species from header

## LAMMPS

- `dump` / `lammpstrj`
  - Loader: LAMMPS dump importer (streaming)
  - Supports orthogonal and triclinic boxes (xy/xz/yz tilts), columns `x y z` or `xs ys zs` (xu/yu/zu), optional `fx fy fz`
  - Species: If `element` column is present, uses it; otherwise a popup window will be displayed to let the user enter the element list (corresponding to type 1..N).

## ASE trajectories

- `.traj` / `.json` (ASE)
  - Loader: ASE trajectory importer (requires ASE)
  - Uses `ase.io.iread()` to iterate frames and converts each Atoms to `Structure`
  - Captures lattice, species, positions, optional forces (`atoms.arrays['forces']`), pbc flags, and stress/virial if present in `atoms.info`

- `.traj` / `.json` / `.ndjson` (without ASE)
  - Loader: ASE JSON importer (no ASE dependency)
  - NDJSON (one JSON object per line) and single JSON array formats supported
  - Expects atoms dict (or nested under `atoms`) with `numbers` or `symbols`, `positions`, optional `cell`, `pbc`, and `forces`

## Fallback conversion

When an unrecognized format is selected, NepTrainKit attempts to detect if any registered importer can parse it. If not, no loading occurs. You can always convert your data to EXTXYZ to ensure compatibility.

## Cancelation and performance

- All stream parsers support a cooperative `cancel_event` used by the UI to stop loading.
- Rendering decimation is applied for very large datasets to keep the UI responsive; selections operate on the full data regardless of decimation.
