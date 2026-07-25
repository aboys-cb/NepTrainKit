#!/usr/bin/env python
"""Validate candidate phase refinements on independent public structures.

The database entries are not used to build the PhaseSketch or refinement
template banks. Random distortion sweeps measure geometric robustness.
Optional EMT NVT trajectories are only available for the Al-Ni L1_2 entries
and are used as a finite-temperature stress test, not as a phase-diagram
reference.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from ase import Atoms, units
from ase.calculators.emt import EMT
from ase.constraints import FixCom
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from NepTrainKit.core.audit.phase_refinement import refine_l12, refine_laves
from NepTrainKit.core.audit.phase_sketch import periodic_knn_vectors, phase_sketch
from tools.benchmark_phase_sketch import Frame, _phase_label, _ptm_prediction


@dataclass(frozen=True)
class BlindSource:
    source_id: str
    provider: str
    url: str
    data_format: str
    seed: int
    formula: str
    candidate: str
    expected_label: str
    expected_confirmed: bool
    repeats: tuple[int, int, int]
    emt_md: bool = False

SOURCES = (
    BlindSource(
        "mp-2593",
        "Materials Project OPTIMADE",
        "https://optimade.materialsproject.org/v1/structures/mp-2593",
        "optimade",
        2593,
        "Ni3Al",
        "l12",
        "l12",
        True,
        (3, 3, 3),
        True,
    ),
    BlindSource(
        "oqmd-4065738",
        "OQMD OPTIMADE",
        "https://oqmd.org/optimade/structures/4065738",
        "optimade",
        4065738,
        "Ni3Al",
        "l12",
        "l12",
        True,
        (3, 3, 3),
        True,
    ),
    BlindSource(
        "cod-1527746",
        "Crystallography Open Database",
        "https://www.crystallography.net/cod/1527746.cif",
        "cif",
        1527746,
        "Al3Zr",
        "l12",
        "l12_partial",
        False,
        (2, 2, 2),
    ),
    BlindSource(
        "cod-5910078",
        "Crystallography Open Database",
        "https://www.crystallography.net/cod/5910078.cif",
        "cif",
        5910078,
        "MgZn2",
        "laves",
        "c14",
        True,
        (2, 2, 2),
    ),
    BlindSource(
        "cod-1524810",
        "Crystallography Open Database",
        "https://www.crystallography.net/cod/1524810.cif",
        "cif",
        1524810,
        "MgCu2",
        "laves",
        "c15",
        True,
        (2, 2, 2),
    ),
)


DISTORTIONS = {
    "clean": (0.0, 0.0, 0.0),
    "low": (0.02, 0.02, 0.03),
    "moderate": (0.06, 0.05, 0.08),
    "high": (0.10, 0.08, 0.12),
    "extreme": (0.15, 0.15, 0.20),
}


def _download(source: BlindSource, cache_dir: Path) -> Path:
    suffix = ".cif" if source.data_format == "cif" else ".json"
    path = cache_dir / f"{source.source_id}{suffix}"
    if path.exists():
        return path
    request = Request(source.url, headers={"User-Agent": "NepTrainKit phase validation"})
    with urlopen(request, timeout=30) as response:
        payload = response.read()
    path.write_bytes(payload)
    return path


def _load_source(source: BlindSource, cache_dir: Path) -> Atoms:
    path = _download(source, cache_dir)
    if source.data_format == "cif":
        atoms = read(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))["data"]
        attributes = payload["attributes"]
        atoms = Atoms(
            symbols=attributes["species_at_sites"],
            positions=attributes["cartesian_site_positions"],
            cell=attributes["lattice_vectors"],
            pbc=True,
        )
    atoms = atoms.repeat(source.repeats)
    atoms.pbc = True
    return atoms


def _distort(
    atoms: Atoms,
    seed: int,
    *,
    strain: float,
    noise: float,
    shear: float,
) -> Atoms:
    distorted = atoms.copy()
    if strain == noise == shear == 0.0:
        return distorted
    rng = np.random.default_rng(seed)
    random_matrix = rng.normal(size=(3, 3))
    symmetric = 0.5 * (random_matrix + random_matrix.T)
    symmetric /= max(float(np.linalg.norm(symmetric, ord=2)), 1.0)
    deformation = np.eye(3) + strain * symmetric
    deformation += np.triu(rng.uniform(-shear, shear, size=(3, 3)), k=1)
    distorted.set_cell(distorted.cell.array @ deformation.T, scale_atoms=True)
    vectors, _, valid = periodic_knn_vectors(
        distorted.positions,
        distorted.cell.array,
        distorted.pbc,
        neighbors=1,
    )
    nearest = float(np.median(np.linalg.norm(vectors[valid], axis=1)))
    distorted.positions += rng.normal(
        scale=noise * nearest,
        size=distorted.positions.shape,
    )
    distorted.wrap()
    return distorted


def _refine(source: BlindSource, atoms: Atoms):
    refine = refine_l12 if source.candidate == "l12" else refine_laves
    return refine(atoms.positions, atoms.cell.array, atoms.pbc, atoms.numbers)


def _ptm_label(atoms: Atoms, source: BlindSource) -> str:
    prediction = _ptm_prediction(
        Frame(atoms, "unknown", "unknown", "external-blind", source.source_id)
    )
    return _phase_label(*prediction)


def _distortion_sweep(
    source: BlindSource,
    atoms: Atoms,
    seeds: int,
) -> dict[str, object]:
    conditions = {}
    for condition, (strain, noise, shear) in DISTORTIONS.items():
        results = []
        ptm_labels = Counter()
        for offset in range(1 if condition == "clean" else seeds):
            current = _distort(
                atoms,
                source.seed + 1009 * offset,
                strain=strain,
                noise=noise,
                shear=shear,
            )
            result = _refine(source, current)
            results.append(result)
            ptm_labels[_ptm_label(current, source)] += 1
        joint = np.asarray([value.joint_match_fraction for value in results])
        conditions[condition] = {
            "samples": len(results),
            "labels": dict(sorted(Counter(value.label for value in results).items())),
            "confirmed_fraction": float(np.mean([value.confirmed for value in results])),
            "expected_label_fraction": float(
                np.mean([value.label == source.expected_label for value in results])
            ),
            "correct_confirmation_fraction": float(
                np.mean(
                    [value.confirmed == source.expected_confirmed for value in results]
                )
            ),
            "joint_match_median": float(np.median(joint)),
            "joint_match_range": [float(np.min(joint)), float(np.max(joint))],
            "ptm_labels": dict(sorted(ptm_labels.items())),
        }
    return conditions


def _emt_md_sweep(
    source: BlindSource,
    atoms: Atoms,
    temperatures: tuple[int, ...],
    steps: int,
) -> list[dict[str, object]]:
    if not source.emt_md or steps <= 0:
        return []
    rows = []
    for temperature in temperatures:
        current = atoms.copy()
        current.calc = EMT()
        current.set_constraint(FixCom())
        rng = np.random.default_rng(source.seed + temperature)
        MaxwellBoltzmannDistribution(
            current,
            temperature_K=temperature,
            force_temp=True,
            rng=rng,
        )
        dynamics = Langevin(
            current,
            1.0 * units.fs,
            temperature_K=temperature,
            friction=0.01 / units.fs,
            fixcm=False,
            rng=rng,
        )
        dynamics.run(steps)
        result = _refine(source, current)
        sketch = phase_sketch(
            current.positions,
            current.cell.array,
            current.pbc,
            current.numbers,
        )
        rows.append(
            {
                "target_temperature_K": temperature,
                "instantaneous_temperature_K": float(current.get_temperature()),
                "steps": steps,
                "timestep_fs": 1.0,
                "label": result.label,
                "confirmed": result.confirmed,
                "joint_match_fraction": result.joint_match_fraction,
                "fcc_cna_fraction": float(np.mean(sketch.cna_labels == 1)),
                "translational_order_score": sketch.translational_order_score,
                "translational_order_limit": sketch.translational_order_limit,
                "ptm": _ptm_label(current, source),
            }
        )
    return rows


def validate(
    cache_dir: Path,
    *,
    seeds: int,
    emt_md_steps: int,
    temperatures: tuple[int, ...],
) -> dict[str, object]:
    sources = []
    for source in SOURCES:
        atoms = _load_source(source, cache_dir)
        sources.append(
            {
                "source_id": source.source_id,
                "provider": source.provider,
                "url": source.url,
                "formula": source.formula,
                "atoms": len(atoms),
                "candidate": source.candidate,
                "expected_label": source.expected_label,
                "expected_confirmed": source.expected_confirmed,
                "distortions": _distortion_sweep(source, atoms, seeds),
                "emt_fixed_cell_nvt": _emt_md_sweep(
                    source,
                    atoms,
                    temperatures,
                    emt_md_steps,
                ),
            }
        )
    return {
        "method": {
            "blind_source": (
                "Materials Project/OQMD relaxed OPTIMADE structures and COD experimental CIFs"
            ),
            "templates_fitted_from_blind_set": False,
            "distortion_note": (
                "random strain/shear/displacement is a robustness sweep, not finite-temperature truth"
            ),
            "emt_note": (
                "fixed-cell EMT NVT is an independent Al-Ni thermal stress test, not a phase diagram"
            ),
            "distortion_seeds": seeds,
        },
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--emt-md-steps", type=int, default=0)
    parser.add_argument(
        "--temperatures",
        type=int,
        nargs="+",
        default=(300, 600, 900, 1200, 1500),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds <= 0 or args.emt_md_steps < 0:
        parser.error("--seeds must be positive and --emt-md-steps non-negative")

    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        report = validate(
            args.cache_dir,
            seeds=args.seeds,
            emt_md_steps=args.emt_md_steps,
            temperatures=tuple(args.temperatures),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="neptrainkit-phase-blind-") as directory:
            report = validate(
                Path(directory),
                seeds=args.seeds,
                emt_md_steps=args.emt_md_steps,
                temperatures=tuple(args.temperatures),
            )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
