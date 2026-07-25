Third‑Party Notices

This document lists third-party code or designs retained directly in the
NepTrainKit repository. NEP compute backends are distributed separately by
`nep-adapters` and are not vendored here.

nep-adapters

- Repository: https://github.com/MagTheoryLab/NEPAdapters
- Role: Runtime dependency supplying CPU and CUDA NEP compute backends.
- Packaging: The dependency carries the applicable NEP_CPU and GPUMD source
  notices and licenses. Those backend source trees are not part of the
  NepTrainKit distribution.

GPUMD Energy Reference Aligner

- Repository: https://github.com/brucefan1983/GPUMD
- License: GNU General Public License v3.0 or later (GPL‑3.0‑or‑later)
- The energy reference alignment workflow in
  `src/NepTrainKit/core/energy_shift.py` is conceptually adapted from GPUMD's
  `tools/Analysis_and_Processing/energy-reference-aligner`.

License Summary

- The root of this repository includes `LICENSE` (GPL‑3.0). Consistent with the
  upstream projects, this project is distributed under GPL‑3.0‑or‑later terms.
- Per the GPL, all redistributions and modifications must remain under the GPL and
  must retain copyright and license notices.

Disclaimer

- The above summaries are provided for convenience and do not replace the full
  text of the licenses. See `LICENSE` and the upstream repositories for the
  complete license terms.
