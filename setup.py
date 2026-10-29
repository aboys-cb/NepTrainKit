#!/usr/bin/env python
"""Build NepTrainKit's application-native helpers.

NEP compute backends are supplied exclusively by the ``nep-adapters``
dependency and are intentionally not built in this package.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_py import build_py


# Load the stdlib-only helper without importing NepTrainKit and its GUI runtime.
_compile_catalog = runpy.run_path(
    str(Path(__file__).parent / "src/NepTrainKit/translation_catalog.py")
)["compile_catalog"]


class BuildPyWithTranslations(build_py):
    """Ship freshly compiled catalogs in wheels, including builds from sdists."""

    def run(self):
        super().run()
        package = Path(self.get_package_dir("NepTrainKit")) if self.editable_mode else Path(self.build_lib) / "NepTrainKit"
        for ts in (package / "translations").glob("*.ts"):
            _compile_catalog(ts, force=not self.editable_mode)

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        package = Path(self.build_lib) / "NepTrainKit"
        for ts in (package / "translations").glob("*.ts"):
            qm = str(ts.with_suffix(".qm"))
            if qm not in outputs:
                outputs.append(qm)
        return outputs


extra_compile_args: list[str] = []
extra_link_args: list[str] = []
default_omp_mode = "0" if sys.platform == "darwin" else "auto"
omp_mode = os.environ.get("NEPKIT_OPENMP", default_omp_mode).strip().lower()
use_openmp = omp_mode not in {"0", "false", "off", "no"}

if sys.platform == "win32":
    if use_openmp:
        extra_compile_args.append("/openmp")
        extra_link_args.append("/openmp")
    extra_compile_args.extend(["/O2", "/std:c++11"])
    extra_link_args.extend(["/O2", "/std:c++11"])
elif sys.platform == "darwin":
    extra_compile_args.extend(["-O3", "-std=c++11"])
    extra_link_args.extend(["-O3", "-std=c++11"])
    omp_include = os.getenv("OMP_INCLUDE_PATH", "/opt/homebrew/opt/libomp/include")
    omp_lib = os.getenv("OMP_LIB_PATH", "/opt/homebrew/opt/libomp/lib")
    if use_openmp and os.path.isdir(omp_include) and os.path.isdir(omp_lib):
        extra_compile_args.extend(["-Xpreprocessor", "-fopenmp", f"-I{omp_include}"])
        extra_link_args.extend(["-lomp", f"-L{omp_lib}"])
else:
    if use_openmp:
        extra_compile_args.append("-fopenmp")
        extra_link_args.append("-fopenmp")
    # std::thread (sampling) needs pthread when OpenMP is disabled.
    extra_compile_args.extend(["-O3", "-std=c++11", "-pthread"])
    extra_link_args.append("-pthread")
    extra_link_args.extend(["-O3", "-std=c++11"])


def native_extension(
    module_name: str,
    source: str,
    *,
    depends: tuple[str, ...] = (),
) -> Extension:
    """Create one private application-native extension with shared headers."""
    return Extension(
        f"NepTrainKit._native.{module_name}",
        [source],
        include_dirs=[pybind11.get_include(), "src/native/include"],
        extra_compile_args=list(extra_compile_args),
        extra_link_args=list(extra_link_args),
        depends=list(depends),
        language="c++",
    )


neighbor_header = "src/native/include/neptrainkit/native/periodic_neighbors.hpp"
fast_float_header = "src/native/io/fast_float.h"

setup(
    cmdclass={"build_py": BuildPyWithTranslations},
    ext_modules=[
        native_extension("_io", "src/native/io/module.cpp", depends=(fast_float_header,)),
        native_extension("_dump", "src/native/dump/module.cpp", depends=(fast_float_header,)),
        native_extension("_audit", "src/native/audit/module.cpp", depends=(neighbor_header,)),
        native_extension("_phase", "src/native/phase/module.cpp", depends=(neighbor_header,)),
        native_extension("_magnetism", "src/native/magnetism/module.cpp", depends=(neighbor_header,)),
        native_extension("_sampling", "src/native/sampling/module.cpp", depends=(neighbor_header,)),
    ],
    zip_safe=False,
)
