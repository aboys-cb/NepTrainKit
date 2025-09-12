Third‑Party Notices

This project incorporates or adapts code from the following upstream projects.
Each retained file preserves its original license headers where present. This
document provides a consolidated overview for convenience only.

NEP_CPU

- Repository: https://github.com/brucefan1983/NEP_CPU
- License: GNU General Public License v3.0 or later (GPL‑3.0‑or‑later)
- Notable files and derivations under `src/nep_cpu/`:
  - Upstream‑derived (carry upstream headers): `nep.cpp`, `nep.h`, `dftd3para.h`.
  - Added in this repository (binding/wrapper): `nep_cpu.cpp` (GPL‑3.0‑or‑later),
    with explicit attribution to NEP_CPU in the file header.

GPUMD (Graphics Processing Units Molecular Dynamics)

- Repository: https://github.com/brucefan1983/GPUMD
- License: GNU General Public License v3.0 or later (GPL‑3.0‑or‑later)
- Notable files and derivations under `src/nep_gpu/`:
  - Upstream‑derived or used largely as‑is: files in `main_nep/` and `utilities/`,
    and selected `.cu/.cuh` files such as `dataset.*`, `structure.*`, `nep.*`,
    `nep_charge.*`, `parameters.*`, `tnep.*`, `fitness.*` (retain original headers).
  - Added or adapted in this repository to support Python bindings and descriptor
    extraction: `nep_gpu.cu`, `nep_desc.cu`, `nep_desc.cuh`, `nep_parameters.cu`,
    `nep_parameters.cuh` (GPL‑3.0‑or‑later), with attribution to GPUMD in file headers.

License Summary

- The root of this repository includes `LICENSE` (GPL‑3.0). Consistent with the
  upstream projects, this project is distributed under GPL‑3.0‑or‑later terms.
- Per the GPL, all redistributions and modifications must remain under the GPL and
  must retain copyright and license notices.

Disclaimer

- The above summaries are provided for convenience and do not replace the full
  text of the licenses. See `LICENSE` and the upstream repositories for the
  complete license terms.

