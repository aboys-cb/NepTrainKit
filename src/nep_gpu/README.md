nep_gpu notes

- Provenance: This directory adapts and trims parts of the GPUMD project
  (Graphics Processing Units Molecular Dynamics):
  https://github.com/brucefan1983/GPUMD (authors/team: Zheyong Fan and the GPUMD
  development team).
- License: GPUMD is licensed under the GNU General Public License v3, or (at your
  option) any later version. This repository includes a top‑level `LICENSE`
  compatible with the upstream license. Under the GPL, any modifications and
  redistributions must remain under the GPL and retain all copyright and
  license notices.

File origins and changes

- Files taken directly from GPUMD (unmodified or lightly adapted for build/layout):
  - Files under `main_nep/`
  - Files under `utilities/`
  - Various `.cu/.cuh` files in this directory (e.g., `dataset.*`, `structure.*`,
    `nep.*`, `nep_charge.*`, `parameters.*`, `tnep.*`, `fitness.*`). These files
    carry the original GPL headers from upstream.
- Files added/modified here to support Python bindings and descriptor-only compute:
  - `nep_gpu.cu`: GPU-side Python bindings and helper interfaces (new; header notes GPUMD origin).
  - `nep_desc.cu`, `nep_desc.cuh`: Extracted/trimmed kernels for descriptor computation (new; origin noted).
  - `nep_parameters.cu`, `nep_parameters.cuh`: Parameter loading and small wrappers (new; origin noted).

Usage and acknowledgement

- If you use this directory or cite it in academic work, please acknowledge GPUMD:
  Z. Fan and the GPUMD development team, GPUMD, https://github.com/brucefan1983/GPUMD

Legal notice

- Code in this directory is provided under GPL-3.0-or-later without any warranty.
  See the repository root `LICENSE` and the upstream project’s license for details.
