nep_cpu notes

- Provenance: Most of the core logic in this directory derives from
  the NEP_CPU project https://github.com/brucefan1983/NEP_CPU
  (authors: Zheyong Fan, Junjie Wang, Eric Lindgren, and contributors).
- License: NEP_CPU is licensed under the GNU General Public License v3, or
  (at your option) any later version. This repository includes a top‑level
  `LICENSE` compatible with the upstream license. Under the GPL, any
  modifications and redistributions must remain under the GPL and retain all
  copyright and license notices.

File origins and changes

- Files taken from NEP_CPU (carry upstream GPL headers):
  - `nep.cpp`
  - `nep.h`
  - `dftd3para.h`
- Files added/wrapped in this repository:
  - `nep_cpu.cpp`: Python binding and convenience interface. The file header
    acknowledges NEP_CPU and is published under GPLv3-or-later.

Acknowledgement and citation

- If you use code from this directory or cite it in academic work, please
  acknowledge NEP_CPU:
  Z. Fan, J. Wang, E. Lindgren, et al., NEP_CPU, https://github.com/brucefan1983/NEP_CPU

Legal notice

- Code in this directory is provided under GPL-3.0-or-later without any
  warranty. See the repository root `LICENSE` and the upstream project’s
  license for details.
