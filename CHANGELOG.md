# Changelog

Notable changes between v2.5.4 and v2.6.3.

See also: `docs/source/changelog.md` for the documentation version.

## v2.6.1 (2025-09-12)

- Added — GPU NEP backend:
  - Optional GPU-accelerated NEP backend with Auto/CPU/GPU selection
  - GPU batch size control
  - GPU acceleration for polarizability and dipole calculations
- Added — Data Management module (projects/versions/tags, quick search, open-folder)
- Added — Organic perturbation card; alignment and DFT‑D3 tools; batch Edit Info; export descriptors
- Changed — Rewrote NEP calculation invocation; refactored ResultData; improved imports
- Performance — Vispy rendering improvements; released GIL in native libs
- Compatibility — Older DeepMD/NPY supported; updated CUDA packaging
- Fixes — Quick entry; updated tests/interfaces

## v2.6.3 (2025-09-14)

Added
- Importers
  - VASP OUTCAR importer: reads DFT POSITION/TOTAL-FORCE, converts kB stress → eV/Å^3, derives virial; ignores ML blocks; cooperative cancel.
  - VASP XDATCAR importer: cooperative cancel support and robust frame parsing.
  - LAMMPS dump importer: orthogonal/triclinic boxes, `x/y/z` or `xs/ys/zs` (and `xu/yu/zu`), optional `fx/fy/fz`. If no `element` column, prompts user for type→element list via dialog. Full‑data polygon selection regardless of rendering decimation.
  - ASE trajectory importers: (a) With ASE (`.traj`) via `ase.io.iread()` to internal `Structure`; 
  - New “Supported Formats” docs page and links from “Data Import” sections.
- OrganicMolConfigPBCCard
  - Bond detection now uses Linus Pauling bond‑order formula `BO = exp((r0 − r)/c)`; `c` and `BO` threshold exposed in UI (defaults 0.3/0.2).
  - Multiple bonds (order ≥ 2) excluded from rotatable torsions; optional “Center molecule in non‑PBC box” switch.
- Plot Settings
  - New Settings page group to control plotting details across both PyQtGraph and VisPy canvases.
  - Adjustable scatter point size for PyQtGraph and VisPy.
  - Configurable default scatter colors: edge and face (with alpha).
  - Adjustable colors for Selected/Show/Current markers and current-marker size.

Changed
- Result loading
  - Unified `ResultData.load_structures()` to use importer pipeline; `from_path(..., structures=...)` supported to skip re‑parsing.
  - Registry refactor to lazy‑load modules (nep/deepmd/importers) and pass factory by dotted path; removed top‑level custom‑widget imports.


Fixed
- VisPy picking stability: integer HiDPI coords, larger pick patch, nearest‑valid pixel, hide overlays/diagonal/path during pick; fixes star marker misalignment.
- GetStrMessageBox layout: fixed to display text line edit properly; used lazily to avoid circular imports.

Performance
- Distance/matching
  - Replaced 27‑image allocation with minimum‑image fractional wrapping and block processing in `calculate_pairwise_distances()`.
  - Rewrote `get_bond_pairs()`/`get_bad_bond_pairs()` with ASE NeighborList to avoid O(N²) memory.
  - `get_mini_distance_info()` rewritten to block‑aggregate per‑element‑pair minima without building full NxN matrix.

Compatibility
- NEP GPU backend: self‑test and runtime fallback to CPU on CUDA driver/runtime mismatch; maintains functionality on non‑GPU systems.
