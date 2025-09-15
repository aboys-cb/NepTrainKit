# Changelog

All notable changes between v2.5.4 and v2.6.3.

## v2.6.1 (2025-09-12)

- Added — GPU NEP backend:
  - New optional GPU-accelerated NEP backend with Auto/CPU/GPU selection
  - GPU batch size control for better throughput vs memory usage
  - GPU acceleration added for polarizability and dipole calculations
- Added — Data Management module:
  - Projects, versions (models), tags, notes, quick search, and open-folder
- Added — Make Dataset cards:
  - Organic molecular rotation (TorsionGuard, PBC-aware) for conformer generation
- Added — Display tools:
  - Energy baseline alignment and DFT-D3 integration
  - Edit Info to batch edit selected structure metadata
  - Export descriptors for selected structures
- Changed — Compute & IO:
  - Rewrote NEP calculation invocation and refactored ResultData
  - Improved import: auto-detect deepmd directories; better nep.in handling
- Performance:
  - Refactored Vispy rendering for large scenes; major drawing optimizations
  - Released GIL in native libs to improve responsiveness
- Compatibility:
  - Support for older DeepMD/NPY formats
  - Updated CUDA runtime and packaging for GPU builds
- Fixes:
  - Quick-entry/open behaviors corrected
  - Test interfaces updated and failing tests fixed

Notes:
- GPU backend requires a compatible NVIDIA driver and CUDA 12 runtime. On systems without a working GPU stack or binary, the app falls back to CPU automatically.

## v2.6.3 (2025-09-14)

### Added

- Importers
  - VASP OUTCAR: DFT POSITION/TOTAL-FORCE, kB stress → eV/Å^3, virial derivation; ignores ML blocks; cancel-safe.
  - VASP XDATCAR: cancel-safe, robust per-frame parsing.
  - LAMMPS dump/lammpstrj: triclinic/orthogonal, x/y/z or xs/ys/zs (and xu/yu/zu), optional fx/fy/fz.
    - If no element column, prompts for type→element list in UI.
    - Polygon selection runs on full data even when render decimates.
  - ASE trajectories
    - With ASE (.traj): ase.io.iread() → Structure (cell/species/pos/forces/pbc/stress/virial when available).
- OrganicMolConfigPBCCard
  - Linus Pauling bond-order formula for bond detection, UI options for c (0.3) and BO threshold (0.2).
  - Exclude multiple bonds (order ≥ 2) from rotatable torsions; optional “Center molecule in non-PBC box”.
- Documentation
  - New “Supported Formats” page; cross-links from “Data Import” sections.

### Changed

- Result loading
  - ResultData.load_structures() uses importer pipeline; from_path(..., structures=...) to reuse pre-parsed structures.
  - Registry uses lazy importlib and dotted factory paths; removes top-level custom-widget imports.



### Fixed

- VisPy picking accuracy: integer HiDPI coords, 7×7 pick patch, nearest valid pixel, hide overlays/diagonal/path during pick; fixes star marker offset.
- GetStrMessageBox uses line edit correctly; lazy import avoids circular dependency.

### Performance

- Distance calculations: minimum-image + block processing; no 27-image allocation.
- Bond checks via NeighborList; memory-lean get_mini_distance_info() aggregation.

### Compatibility

- NEP GPU backend self-test and runtime fallback to CPU on CUDA driver/runtime mismatch.
