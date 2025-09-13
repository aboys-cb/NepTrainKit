# Changelog

All notable changes between v2.5.4 and v2.6.1.

## v2.6.1 (2025-09-12)

- Added — GPU NEP backend:
  - New optional GPU-accelerated NEP backend with Auto/CPU/GPU selection
  - GPU batch size control for better throughput vs memory usage
  - GPU acceleration added for polarizability and dipole calculations
- Added — Data Management module:
  - Projects, versions (models), tags, notes, quick search, and open-folder
- Added — Make Dataset cards:
  - Organic perturbation option/card for treating organics as rigid clusters
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
