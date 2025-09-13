# Changelog

Notable changes between v2.5.4 and v2.6.1.

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
