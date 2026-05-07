#!/usr/bin/env python
"""Benchmark Make Dataset card operations without starting the Qt UI.

The benchmark is intentionally operation-level: UI widgets are parameter
binding, while the expensive work belongs in core card operations. Each case
also performs a small semantic check so timing runs cannot silently measure a
broken output path.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from ase import Atoms
from ase.build import bulk, make_supercell, molecule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from NepTrainKit.core.cards.alloy import (  # noqa: E402
    CompositionGradientOperation,
    CompositionGradientParams,
    CompositionSweepOperation,
    CompositionSweepParams,
    ConditionalReplaceOperation,
    ConditionalReplaceParams,
    RandomDopingOperation,
    RandomDopingParams,
    RandomOccupancyOperation,
    RandomOccupancyParams,
)
from NepTrainKit.core.cards.defect import (  # noqa: E402
    InsertDefectOperation,
    InsertDefectParams,
    RandomSlabOperation,
    RandomSlabParams,
    RandomVacancyOperation,
    RandomVacancyParams,
    StackingFaultOperation,
    StackingFaultParams,
    VacancyDefectOperation,
    VacancyDefectParams,
)
from NepTrainKit.core.cards.lattice import (  # noqa: E402
    CellScalingOperation,
    CellScalingParams,
    CellStrainOperation,
    CellStrainParams,
    PerturbOperation,
    PerturbParams,
    ShearAngleOperation,
    ShearAngleParams,
    ShearMatrixOperation,
    ShearMatrixParams,
    SuperCellOperation,
    SuperCellParams,
)
from NepTrainKit.core.cards.magnetism import (  # noqa: E402
    FoldedHelixOperation,
    FoldedHelixParams,
    MagneticMomentRotationOperation,
    MagneticMomentRotationParams,
    MagneticOrderOperation,
    MagneticOrderParams,
    SetMagneticMomentsOperation,
    SetMagneticMomentsParams,
    SmallAngleSpinTiltOperation,
    SmallAngleSpinTiltParams,
    SpinDisorderOperation,
    SpinDisorderParams,
    SpinSpiralOperation,
    SpinSpiralParams,
)
from NepTrainKit.core.cards.structure import (  # noqa: E402
    CrystalPrototypeBuilderOperation,
    CrystalPrototypeBuilderParams,
    GroupLabelOperation,
    GroupLabelParams,
    LayerCopyOperation,
    LayerCopyParams,
    OrganicMolConfigPBCOperation,
    OrganicMolConfigPBCParams,
    VibrationModePerturbOperation,
    VibrationModePerturbParams,
)


Validator = Callable[[list], None]
Runner = Callable[[], list]


@dataclass(frozen=True)
class BenchCase:
    group: str
    name: str
    runner: Runner
    validate: Validator
    notes: str = ""


@dataclass
class BenchResult:
    group: str
    name: str
    status: str
    repeat: int
    ms_mean: float | None = None
    ms_median: float | None = None
    ms_min: float | None = None
    ms_max: float | None = None
    outputs: int | None = None
    atoms_out: int | None = None
    notes: str = ""
    error: str = ""


def si_bulk(rep: tuple[int, int, int] = (3, 3, 3)) -> Atoms:
    atoms = make_supercell(bulk("Si", "diamond", a=5.43, cubic=True), np.diag(rep))
    atoms.info["Config_type"] = f"Si{len(atoms)}"
    atoms.wrap()
    return atoms


def cu_bulk(rep: tuple[int, int, int] = (4, 4, 4)) -> Atoms:
    atoms = make_supercell(bulk("Cu", "fcc", a=3.61, cubic=True), np.diag(rep))
    atoms.info["Config_type"] = f"Cu{len(atoms)}"
    atoms.wrap()
    return atoms


def fe_spin_bulk(rep: tuple[int, int, int] = (5, 5, 5)) -> Atoms:
    atoms = make_supercell(bulk("Fe", "bcc", a=2.86, cubic=True), np.diag(rep))
    atoms.info["Config_type"] = f"Fe{len(atoms)}"
    atoms.set_initial_magnetic_moments(np.full(len(atoms), 2.2, dtype=float))
    atoms.wrap()
    return atoms


def vib_si() -> Atoms:
    atoms = si_bulk((2, 2, 2))
    natoms = len(atoms)
    n_modes = min(3 * natoms, 24)
    mode_vectors = np.zeros((n_modes, natoms, 3), dtype=float)
    for idx in range(n_modes):
        mode_vectors[idx, idx % natoms, idx % 3] = 1.0
    freq_values = np.linspace(50.0, 300.0, n_modes)
    for mode_idx in range(n_modes):
        atoms.new_array(f"vibration_mode_{mode_idx}_x", mode_vectors[mode_idx, :, 0])
        atoms.new_array(f"vibration_mode_{mode_idx}_y", mode_vectors[mode_idx, :, 1])
        atoms.new_array(f"vibration_mode_{mode_idx}_z", mode_vectors[mode_idx, :, 2])
        atoms.new_array(f"vibration_frequency_{mode_idx}", np.full(natoms, freq_values[mode_idx], dtype=float))
    return atoms


def methane_like() -> Atoms:
    atoms = molecule("C2H6")
    atoms.center(vacuum=8.0)
    atoms.set_pbc(False)
    atoms.info["Config_type"] = "C2H6"
    return atoms


def require_outputs(outputs: list, *, min_count: int = 1, tag: str | None = None) -> None:
    if not isinstance(outputs, list):
        raise AssertionError(f"operation returned {type(outputs).__name__}, expected list")
    if len(outputs) < min_count:
        raise AssertionError(f"expected at least {min_count} outputs, got {len(outputs)}")
    if tag is not None and not any(tag in str(atoms.info.get("Config_type", "")) for atoms in outputs):
        raise AssertionError(f"no output Config_type contains {tag!r}")


def require_changed_positions(reference: Atoms) -> Validator:
    def validate(outputs: list) -> None:
        require_outputs(outputs, min_count=1)
        if not any(np.linalg.norm(atoms.positions - reference.positions) > 1e-12 for atoms in outputs if len(atoms) == len(reference)):
            raise AssertionError("expected at least one output with changed positions")

    return validate


def require_count_and_tag(min_count: int, tag: str) -> Validator:
    return lambda outputs: require_outputs(outputs, min_count=min_count, tag=tag)


def run_structure(operation, structure: Atoms, params) -> Runner:
    return lambda: operation.run_structure(structure.copy(), params)


def run_generator(operation, params) -> Runner:
    return lambda: operation.generate(params)


def build_cases() -> list[BenchCase]:
    si = si_bulk()
    cu = cu_bulk()
    spin = fe_spin_bulk()
    vib = vib_si()

    occ = cu.copy()
    occ.info["Config_type"] = "CuBase|Comp(Ni=0.5,Co=0.5)"

    group_spin = spin.copy()
    group_spin.new_array("group", np.where(np.arange(len(group_spin)) % 2 == 0, "A", "B").astype(object))

    cases = [
        BenchCase(
            "generator",
            "CrystalPrototypeBuilder",
            run_generator(
                CrystalPrototypeBuilderOperation(),
                CrystalPrototypeBuilderParams(lattice="fcc", element="Cu", a_range=(3.55, 3.65, 0.05), max_atoms=256, max_outputs=3),
            ),
            require_count_and_tag(3, "Proto("),
        ),
        BenchCase(
            "lattice",
            "SuperCell:max_atoms",
            run_structure(SuperCellOperation(), si, SuperCellParams(behavior_type=1, mode="max_atoms", max_atoms=512)),
            require_count_and_tag(1, "SC("),
        ),
        BenchCase(
            "lattice",
            "CellStrain:triaxial",
            run_structure(
                CellStrainOperation(),
                si,
                CellStrainParams(axes="triaxial", x_range=(-2, 2, 2), y_range=(-2, 2, 2), z_range=(-2, 2, 2)),
            ),
            require_count_and_tag(27, "Str("),
        ),
        BenchCase(
            "lattice",
            "CellScaling:random32",
            run_structure(CellScalingOperation(), si, CellScalingParams(max_num=32, use_seed=True, seed=1)),
            require_count_and_tag(32, "LSc("),
        ),
        BenchCase(
            "lattice",
            "Perturb:random32",
            run_structure(PerturbOperation(), si, PerturbParams(max_num=32, use_seed=True, seed=2)),
            require_changed_positions(si),
        ),
        BenchCase(
            "lattice",
            "ShearMatrix:grid27",
            run_structure(
                ShearMatrixOperation(),
                si,
                ShearMatrixParams(xy_range=(-1, 1, 1), yz_range=(-1, 1, 1), xz_range=(-1, 1, 1)),
            ),
            require_count_and_tag(27, "Shr("),
        ),
        BenchCase(
            "lattice",
            "ShearAngle:grid27",
            run_structure(
                ShearAngleOperation(),
                si,
                ShearAngleParams(alpha_range=(-1, 1, 1), beta_range=(-1, 1, 1), gamma_range=(-1, 1, 1)),
            ),
            require_count_and_tag(27, "Ang("),
        ),
        BenchCase(
            "lattice",
            "VibrationModePerturb:16",
            run_structure(VibrationModePerturbOperation(), vib, VibrationModePerturbParams(max_num=16, modes_per_sample=3, min_frequency=1.0, use_seed=True, seed=4)),
            require_changed_positions(vib),
        ),
        BenchCase(
            "structure",
            "LayerCopy",
            run_structure(LayerCopyOperation(), si, LayerCopyParams(dz_expr="0.1*sin(x)", layers=3, distance=3.0)),
            lambda outputs: require_outputs(outputs, min_count=1, tag="SWC(") or (
                len(outputs[0]) == len(si) * 3 or (_ for _ in ()).throw(AssertionError("LayerCopy atom count mismatch"))
            ),
        ),
        BenchCase(
            "structure",
            "GroupLabel",
            run_structure(GroupLabelOperation(), si, GroupLabelParams(mode="k-vector layers (recommended)", kvec="111")),
            lambda outputs: require_outputs(outputs, min_count=1, tag="Grp(") or (
                "group" in outputs[0].arrays or (_ for _ in ()).throw(AssertionError("GroupLabel did not write group array"))
            ),
        ),
        BenchCase(
            "structure",
            "OrganicMolConfigPBC:small",
            run_structure(
                OrganicMolConfigPBCOperation(),
                methane_like(),
                OrganicMolConfigPBCParams(perturb_per_frame=4, max_torsions_per_conf=2, local_cutoff=64, local_subtree=32, use_seed=True, seed=5),
            ),
            require_outputs,
            "small molecule, low perturb count",
        ),
        BenchCase(
            "alloy",
            "RandomDoping:16",
            run_structure(
                RandomDopingOperation(),
                cu,
                RandomDopingParams(
                    rules=[{"target": "Cu", "dopants": {"Ni": 0.5, "Co": 0.5}, "use": "count", "count": [32, 32], "count_mode": "fixed"}],
                    max_structures=16,
                    use_seed=True,
                    seed=6,
                ),
            ),
            require_count_and_tag(16, "Dop("),
        ),
        BenchCase(
            "alloy",
            "CompositionSweep:128",
            run_structure(
                CompositionSweepOperation(),
                cu,
                CompositionSweepParams(elements="Co,Cr,Fe,Ni,Cu", order="2,3,4,5", method="Grid", step=0.25, max_outputs=128, use_seed=True, seed=7),
            ),
            require_count_and_tag(128, "Comp("),
        ),
        BenchCase(
            "alloy",
            "CompositionGradient:8",
            run_structure(
                CompositionGradientOperation(),
                cu,
                CompositionGradientParams(elements="Ni,Co", start_composition="Ni:1,Co:0", end_composition="Ni:0,Co:1", bins=8, samples=8, use_seed=True, seed=8),
            ),
            require_count_and_tag(8, "CompGrad("),
        ),
        BenchCase(
            "alloy",
            "RandomOccupancy:16",
            run_structure(RandomOccupancyOperation(), occ, RandomOccupancyParams(samples=16, use_seed=True, seed=9)),
            require_count_and_tag(16, "Occ("),
        ),
        BenchCase(
            "alloy",
            "ConditionalReplace",
            run_structure(ConditionalReplaceOperation(), cu, ConditionalReplaceParams(target="Cu", replacements="Ni:0.5,Co:0.5", condition="x >= 0", seed=10, mode=1)),
            require_count_and_tag(1, "Repl("),
        ),
        BenchCase(
            "defect",
            "RandomVacancy:16",
            run_structure(RandomVacancyOperation(), si, RandomVacancyParams(rules=[{"element": "Si", "count": [8, 8], "count_mode": "fixed"}], max_structures=16, use_seed=True, seed=11)),
            require_count_and_tag(16, "Vac("),
        ),
        BenchCase(
            "defect",
            "VacancyDefect:16",
            run_structure(VacancyDefectOperation(), si, VacancyDefectParams(num_condition=8, max_structures=16, use_seed=True, seed=12)),
            require_count_and_tag(16, "Vac("),
        ),
        BenchCase(
            "defect",
            "StackingFault:grid5",
            run_structure(StackingFaultOperation(), si, StackingFaultParams(hkl=(1, 1, 1), step=(0.0, 0.4, 0.1), layers=2)),
            require_count_and_tag(5, "SF("),
        ),
        BenchCase(
            "defect",
            "RandomSlab:small_grid",
            run_structure(RandomSlabOperation(), si_bulk((1, 1, 1)), RandomSlabParams(h_range=(0, 1, 1), k_range=(0, 1, 1), l_range=(1, 1, 1), layer_range=(2, 3, 1), vacuum_range=(0, 4, 4))),
            require_count_and_tag(8, "Slab("),
        ),
        BenchCase(
            "defect",
            "InsertDefect:16x2",
            run_structure(InsertDefectOperation(), si, InsertDefectParams(species="H", insert_count=2, structure_count=16, min_distance=0.8, max_attempts=100, use_seed=True, seed=13)),
            require_count_and_tag(16, "Ins("),
        ),
        BenchCase(
            "magnetism",
            "SetMagneticMoments",
            run_structure(SetMagneticMomentsOperation(), spin, SetMagneticMomentsParams(magmom_map="Fe:2.2")),
            require_count_and_tag(1, "MagSet("),
        ),
        BenchCase(
            "magnetism",
            "MagneticOrder:pm16",
            run_structure(MagneticOrderOperation(), spin, MagneticOrderParams(magmom_map="Fe:2.2", gen_fm=True, gen_afm=True, gen_pm=True, pm_count=16, use_seed=True, seed=14)),
            require_outputs,
        ),
        BenchCase(
            "magnetism",
            "MagmomRotation:16",
            run_structure(MagneticMomentRotationOperation(), spin, MagneticMomentRotationParams(num_structures=16, use_seed=True, seed=15)),
            require_count_and_tag(16, "MMR("),
        ),
        BenchCase(
            "magnetism",
            "SpinSpiral:32",
            run_structure(SpinSpiralOperation(), spin, SpinSpiralParams(period_range=(8.0, 16.0, 4.0), phase_range=(0.0, 15.0, 15.0), chirality="Both", max_outputs=32)),
            require_count_and_tag(12, "Helix("),
        ),
        BenchCase(
            "magnetism",
            "FoldedHelix:32",
            run_structure(FoldedHelixOperation(), spin, FoldedHelixParams(half_period_mode="Manual", half_period_layers=(2, 6, 2), angle_step_range=(15.0, 45.0, 15.0), sequence_mode="Both", max_outputs=32)),
            require_count_and_tag(18, "FoldedHelix("),
        ),
        BenchCase(
            "magnetism",
            "SmallAngleSpinTilt:auto_pairs",
            run_structure(
                SmallAngleSpinTiltOperation(),
                group_spin,
                SmallAngleSpinTiltParams(
                    canting_mode="Atom pair canting",
                    pair_source="Auto by neighbor shell",
                    pair_shell=1,
                    angle_list="2,5",
                    tilt_signs="Both (+/- pair)",
                    include_reference=False,
                    max_outputs=64,
                ),
            ),
            require_count_and_tag(1, "SpinPair("),
            "uses pair-distance matrix",
        ),
        BenchCase(
            "magnetism",
            "SpinDisorder:32",
            run_structure(SpinDisorderOperation(), spin, SpinDisorderParams(fractions="0.1,0.3,0.5,0.7", samples_per_fraction=8, use_seed=True, seed=16, max_outputs=32)),
            require_count_and_tag(32, "SpinDis("),
        ),
    ]

    return cases


def build_count50_cases() -> list[BenchCase]:
    si = si_bulk()
    cu = cu_bulk()
    spin = fe_spin_bulk()
    vib = vib_si()

    occ = cu.copy()
    occ.info["Config_type"] = "CuBase|Comp(Ni=0.5,Co=0.5)"

    group_spin = spin.copy()
    group_spin.new_array("group", np.where(np.arange(len(group_spin)) % 2 == 0, "A", "B").astype(object))

    return [
        BenchCase(
            "generator",
            "CrystalPrototypeBuilder:50",
            run_generator(
                CrystalPrototypeBuilderOperation(),
                CrystalPrototypeBuilderParams(lattice="fcc", element="Cu", a_range=(3.50, 3.99, 0.01), max_atoms=256, max_outputs=50),
            ),
            require_count_and_tag(50, "Proto("),
        ),
        BenchCase(
            "lattice",
            "CellStrain:grid64",
            run_structure(
                CellStrainOperation(),
                si,
                CellStrainParams(axes="triaxial", x_range=(-1.5, 1.5, 1.0), y_range=(-1.5, 1.5, 1.0), z_range=(-1.5, 1.5, 1.0)),
            ),
            require_count_and_tag(50, "Str("),
            "grid card; nearest simple grid above 50",
        ),
        BenchCase(
            "lattice",
            "CellScaling:50",
            run_structure(CellScalingOperation(), si, CellScalingParams(max_num=50, use_seed=True, seed=1)),
            require_count_and_tag(50, "LSc("),
        ),
        BenchCase(
            "lattice",
            "Perturb:50",
            run_structure(PerturbOperation(), si, PerturbParams(max_num=50, use_seed=True, seed=2)),
            require_changed_positions(si),
        ),
        BenchCase(
            "lattice",
            "ShearMatrix:grid64",
            run_structure(
                ShearMatrixOperation(),
                si,
                ShearMatrixParams(xy_range=(-1.5, 1.5, 1.0), yz_range=(-1.5, 1.5, 1.0), xz_range=(-1.5, 1.5, 1.0)),
            ),
            require_count_and_tag(50, "Shr("),
            "grid card; nearest simple grid above 50",
        ),
        BenchCase(
            "lattice",
            "ShearAngle:grid64",
            run_structure(
                ShearAngleOperation(),
                si,
                ShearAngleParams(alpha_range=(-1.5, 1.5, 1.0), beta_range=(-1.5, 1.5, 1.0), gamma_range=(-1.5, 1.5, 1.0)),
            ),
            require_count_and_tag(50, "Ang("),
            "grid card; nearest simple grid above 50",
        ),
        BenchCase(
            "lattice",
            "VibrationModePerturb:50",
            run_structure(VibrationModePerturbOperation(), vib, VibrationModePerturbParams(max_num=50, modes_per_sample=3, min_frequency=1.0, use_seed=True, seed=4)),
            require_changed_positions(vib),
        ),
        BenchCase(
            "structure",
            "OrganicMolConfigPBC:50",
            run_structure(
                OrganicMolConfigPBCOperation(),
                methane_like(),
                OrganicMolConfigPBCParams(perturb_per_frame=50, max_torsions_per_conf=2, local_cutoff=64, local_subtree=32, use_seed=True, seed=5),
            ),
            lambda outputs: require_outputs(outputs, min_count=50, tag="TG("),
        ),
        BenchCase(
            "alloy",
            "RandomDoping:50",
            run_structure(
                RandomDopingOperation(),
                cu,
                RandomDopingParams(
                    rules=[{"target": "Cu", "dopants": {"Ni": 0.5, "Co": 0.5}, "use": "count", "count": [32, 32], "count_mode": "fixed"}],
                    max_structures=50,
                    use_seed=True,
                    seed=6,
                ),
            ),
            require_count_and_tag(50, "Dop("),
        ),
        BenchCase(
            "alloy",
            "CompositionSweep:50",
            run_structure(
                CompositionSweepOperation(),
                cu,
                CompositionSweepParams(elements="Co,Cr,Fe,Ni,Cu", order="2,3,4,5", method="Grid", step=0.25, max_outputs=50, use_seed=True, seed=7),
            ),
            require_count_and_tag(50, "Comp("),
        ),
        BenchCase(
            "alloy",
            "CompositionGradient:50",
            run_structure(
                CompositionGradientOperation(),
                cu,
                CompositionGradientParams(elements="Ni,Co", start_composition="Ni:1,Co:0", end_composition="Ni:0,Co:1", bins=8, samples=50, use_seed=True, seed=8),
            ),
            require_count_and_tag(50, "CompGrad("),
        ),
        BenchCase(
            "alloy",
            "RandomOccupancy:50",
            run_structure(RandomOccupancyOperation(), occ, RandomOccupancyParams(samples=50, use_seed=True, seed=9)),
            require_count_and_tag(50, "Occ("),
        ),
        BenchCase(
            "defect",
            "RandomVacancy:50",
            run_structure(RandomVacancyOperation(), si, RandomVacancyParams(rules=[{"element": "Si", "count": [8, 8], "count_mode": "fixed"}], max_structures=50, use_seed=True, seed=11)),
            require_count_and_tag(50, "Vac("),
        ),
        BenchCase(
            "defect",
            "VacancyDefect:50",
            run_structure(VacancyDefectOperation(), si, VacancyDefectParams(num_condition=8, max_structures=50, use_seed=True, seed=12)),
            require_count_and_tag(50, "Vac("),
        ),
        BenchCase(
            "defect",
            "StackingFault:50",
            run_structure(StackingFaultOperation(), si, StackingFaultParams(hkl=(1, 1, 1), step=(0.0, 4.9, 0.1), layers=2)),
            require_count_and_tag(50, "SF("),
        ),
        BenchCase(
            "defect",
            "RandomSlab:grid50",
            run_structure(RandomSlabOperation(), si_bulk((1, 1, 1)), RandomSlabParams(h_range=(0, 4, 1), k_range=(0, 0, 1), l_range=(1, 2, 1), layer_range=(2, 6, 1), vacuum_range=(4, 4, 1))),
            require_count_and_tag(50, "Slab("),
        ),
        BenchCase(
            "defect",
            "InsertDefect:50x2",
            run_structure(InsertDefectOperation(), si, InsertDefectParams(species="H", insert_count=2, structure_count=50, min_distance=0.8, max_attempts=100, use_seed=True, seed=13)),
            require_count_and_tag(50, "Ins("),
        ),
        BenchCase(
            "magnetism",
            "MagneticOrder:pm50",
            run_structure(MagneticOrderOperation(), spin, MagneticOrderParams(magmom_map="Fe:2.2", gen_fm=False, gen_afm=False, gen_pm=True, pm_count=50, use_seed=True, seed=14)),
            require_count_and_tag(50, "MagPM"),
        ),
        BenchCase(
            "magnetism",
            "MagmomRotation:50",
            run_structure(MagneticMomentRotationOperation(), spin, MagneticMomentRotationParams(num_structures=50, use_seed=True, seed=15)),
            require_count_and_tag(50, "MMR("),
        ),
        BenchCase(
            "magnetism",
            "SpinSpiral:50",
            run_structure(SpinSpiralOperation(), spin, SpinSpiralParams(period_range=(8.0, 16.0, 2.0), phase_range=(0.0, 40.0, 10.0), chirality="Both", max_outputs=50)),
            require_count_and_tag(50, "Helix("),
        ),
        BenchCase(
            "magnetism",
            "FoldedHelix:50",
            run_structure(FoldedHelixOperation(), spin, FoldedHelixParams(half_period_mode="Manual", half_period_layers=(2, 10, 2), angle_step_range=(10.0, 50.0, 10.0), sequence_mode="Both", max_outputs=50)),
            require_count_and_tag(50, "FoldedHelix("),
        ),
        BenchCase(
            "magnetism",
            "SmallAngleSpinTilt:auto_pairs50",
            run_structure(
                SmallAngleSpinTiltOperation(),
                group_spin,
                SmallAngleSpinTiltParams(
                    canting_mode="Atom pair canting",
                    pair_source="Auto by neighbor shell",
                    pair_shell=1,
                    angle_list="2,5",
                    tilt_signs="Both (+/- pair)",
                    include_reference=False,
                    max_outputs=50,
                ),
            ),
            require_count_and_tag(50, "SpinPair("),
            "uses pair-distance matrix",
        ),
        BenchCase(
            "magnetism",
            "SpinDisorder:50",
            run_structure(SpinDisorderOperation(), spin, SpinDisorderParams(fractions="0.1,0.2,0.3,0.5,0.7", samples_per_fraction=10, use_seed=True, seed=16, max_outputs=50)),
            require_count_and_tag(50, "SpinDis("),
        ),
    ]


def bench_case(case: BenchCase, repeat: int, warmup: int) -> BenchResult:
    try:
        last_outputs: list = []
        for _ in range(max(warmup, 0)):
            last_outputs = case.runner()
            case.validate(last_outputs)

        times = []
        for _ in range(max(repeat, 1)):
            gc.collect()
            t0 = time.perf_counter()
            last_outputs = case.runner()
            elapsed = (time.perf_counter() - t0) * 1000.0
            case.validate(last_outputs)
            times.append(elapsed)

        atoms_out = sum(len(atoms) for atoms in last_outputs if hasattr(atoms, "__len__"))
        return BenchResult(
            group=case.group,
            name=case.name,
            status="ok",
            repeat=max(repeat, 1),
            ms_mean=float(statistics.mean(times)),
            ms_median=float(statistics.median(times)),
            ms_min=float(min(times)),
            ms_max=float(max(times)),
            outputs=len(last_outputs),
            atoms_out=int(atoms_out),
            notes=case.notes,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report all failures
        return BenchResult(
            group=case.group,
            name=case.name,
            status="error",
            repeat=max(repeat, 1),
            notes=case.notes,
            error=f"{type(exc).__name__}: {exc}",
        )


def bench_chain_case(case: BenchCase, inputs: int) -> BenchResult:
    try:
        total_outputs = 0
        total_atoms = 0
        t0 = time.perf_counter()
        for _ in range(max(int(inputs), 1)):
            outputs = case.runner()
            case.validate(outputs)
            total_outputs += len(outputs)
            total_atoms += sum(len(atoms) for atoms in outputs if hasattr(atoms, "__len__"))
        elapsed = (time.perf_counter() - t0) * 1000.0
        return BenchResult(
            group=case.group,
            name=case.name,
            status="ok",
            repeat=max(int(inputs), 1),
            ms_mean=float(elapsed),
            ms_median=float(elapsed / max(int(inputs), 1)),
            ms_min=float(elapsed / max(total_outputs, 1)),
            ms_max=float(elapsed),
            outputs=int(total_outputs),
            atoms_out=int(total_atoms),
            notes=f"{case.notes}; chain inputs={max(int(inputs), 1)}".strip("; "),
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report all failures
        return BenchResult(
            group=case.group,
            name=case.name,
            status="error",
            repeat=max(int(inputs), 1),
            notes=f"{case.notes}; chain inputs={max(int(inputs), 1)}".strip("; "),
            error=f"{type(exc).__name__}: {exc}",
        )


def matches(case: BenchCase, filters: list[str]) -> bool:
    if not filters:
        return True
    haystack = f"{case.group}/{case.name}".lower()
    return any(token.lower() in haystack for token in filters)


def print_table(results: list[BenchResult]) -> None:
    header = f"{'status':<6} {'group':<10} {'case':<34} {'mean ms':>10} {'min ms':>10} {'max ms':>10} {'out':>6} {'atoms':>8}"
    print(header)
    print("-" * len(header))
    for result in results:
        if result.status == "ok":
            print(
                f"{result.status:<6} {result.group:<10} {result.name:<34} "
                f"{result.ms_mean:10.2f} {result.ms_min:10.2f} {result.ms_max:10.2f} "
                f"{result.outputs:6d} {result.atoms_out:8d}"
            )
        else:
            print(f"{result.status:<6} {result.group:<10} {result.name:<34} {'-':>10} {'-':>10} {'-':>10} {'-':>6} {'-':>8}")
            print(f"       {result.error}")
    print()
    print("Note: timings are operation-level and include the benchmark's fresh input copy for safety.")


def print_chain_table(results: list[BenchResult]) -> None:
    header = f"{'status':<6} {'group':<10} {'case':<34} {'total s':>10} {'ms/input':>10} {'us/output':>10} {'out':>8} {'atoms':>10}"
    print(header)
    print("-" * len(header))
    for result in results:
        if result.status == "ok":
            total_s = (result.ms_mean or 0.0) / 1000.0
            ms_input = result.ms_median or 0.0
            us_output = (result.ms_min or 0.0) * 1000.0
            print(
                f"{result.status:<6} {result.group:<10} {result.name:<34} "
                f"{total_s:10.2f} {ms_input:10.2f} {us_output:10.2f} "
                f"{result.outputs:8d} {result.atoms_out:10d}"
            )
        else:
            print(f"{result.status:<6} {result.group:<10} {result.name:<34} {'-':>10} {'-':>10} {'-':>10} {'-':>8} {'-':>10}")
            print(f"       {result.error}")
    print()
    print("Note: chain mode repeats each operation for N inputs and discards outputs after counting.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3, help="timed repeats per case")
    parser.add_argument("--warmup", type=int, default=1, help="untimed warmup runs per case")
    parser.add_argument("--only", action="append", default=[], help="substring filter, e.g. --only slab --only spin")
    parser.add_argument("--profile", choices=("default", "count50"), default="default", help="benchmark scenario set")
    parser.add_argument("--sort", choices=("slow", "order"), default="slow", help="result order")
    parser.add_argument("--json", type=Path, help="write raw results to a JSON file")
    parser.add_argument("--list", action="store_true", help="list case names without running")
    parser.add_argument("--chain-inputs", type=int, default=0, help="repeat each selected structure operation for N input structures")
    args = parser.parse_args(argv)

    all_cases = build_count50_cases() if args.profile == "count50" else build_cases()
    cases = [case for case in all_cases if matches(case, args.only)]
    if args.chain_inputs > 0:
        cases = [case for case in cases if case.group != "generator"]
    if args.list:
        for case in cases:
            print(f"{case.group}/{case.name}")
        return 0

    if args.chain_inputs > 0:
        results = [bench_chain_case(case, inputs=args.chain_inputs) for case in cases]
    else:
        results = [bench_case(case, repeat=args.repeat, warmup=args.warmup) for case in cases]
    if args.sort == "slow":
        results.sort(key=lambda item: item.ms_mean if item.ms_mean is not None else -1.0, reverse=True)

    if args.chain_inputs > 0:
        print_chain_table(results)
    else:
        print_table(results)

    if args.json:
        args.json.write_text(json.dumps([result.__dict__ for result in results], indent=2), encoding="utf-8")

    return 1 if any(result.status != "ok" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
